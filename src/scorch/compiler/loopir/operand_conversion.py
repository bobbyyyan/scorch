"""Operand conversions: one stored tensor copied into another level order.

A plan that converts an operand (``LoopPlan.conversions``) reads it in a level
order its caller did not store.  The copy is a program of this compiler, in one
of two forms.

The relabelled copy (:func:`relabelled_copy_program`) walks the operand in its
own storage order and appends every stored entry at its coordinates in the new
order, under a recorded two-pass assembly
(``lower_llir._RelabelledCopyAssemblyLowering``).  Into one compressed level,
the target counts the entries of each segment, prefix-sums the counts and
stores each entry at its slot, a counting sort linear in the entries; into two
or more, it sorts the entries' records by a radix sort of the same counting
passes and builds every level in one sweep.  It is the form every conversion
gets when the target emits its converted layout.

The sparse-workspace copy (:func:`conversion_program`) has one sparse workspace
region at the program root whose producer walks the operand the same way and
inserts every stored entry at its coordinates in the new order, and whose
consumer drains the sorted workspace into the converted tensor.  It costs a
hash insertion per entry and a comparison sort, and covers every layout; a
conversion takes it when the target refuses the relabelled copy with
``unsupported_relabelled_assembly`` (:func:`conversion_source`).

The copy is structural.  It moves every entry the operand stores, whatever its
value, so a stored zero stays stored (a dense level below a compressed one
stores every cell under each stored parent, and those cells are entries).  The
converted tensor is laid out so that it stores exactly those entries and no
others: a dense level below a compressed level would store cells the operand
does not hold, so the converted layout has its dense levels, if any, above
every compressed one (:func:`conversion_format`).  A kernel reading the
converted tensor therefore produces the stored coordinates the operand's own
entries imply.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from ...format import LevelType, stores_only_its_entries
from ...format import conversion_format  # noqa: F401  (re-exported)
from ..cin import ForAll, IndexStmt, IndexVar, TensorAssign, TensorVar
from ..cin_analysis import normalize_cin
from ..compilation_context import CompilationContext
from ..compile_options import CompileOptions
from ..identity import SymbolId
from ..loop_plan import OperandConversion
from .build import LoopIRBuilder
from .lower_cin import LoopIRLoweringError, LoopIRLoweringDefect
from .lower_cin import lower_normalized_cin_to_loopir
from .lower_llir import (
    UNSUPPORTED_RELABELLED_ASSEMBLY,
    LoopIRTargetError,
    lower_loopir_to_llir,
)
from .nodes import (
    AppendEntry,
    AssemblyStrategy,
    Block,
    LevelKind,
    LoopProgram,
    ReduceOp,
)
from .schedule_passes import _decompose_chain, _wrap_loops
from .verifier import verify_program

_LEVEL_LETTERS = {LevelType.DENSE: "d", LevelType.COMPRESSED: "s"}


def level_letters(level_types: Sequence[LevelType]) -> Optional[str]:
    """``"d"``/``"s"`` per level, or ``None`` for a level kind this copy does not read."""

    if any(level not in _LEVEL_LETTERS for level in level_types):
        return None
    return "".join(_LEVEL_LETTERS[level] for level in level_types)


def _refuse(message: str) -> None:
    raise LoopIRLoweringError(
        LoopIRLoweringDefect("unsupported_operand_conversion", message)
    )


def _in_order_copy(
    operand_format: str,
    conversion: OperandConversion,
    dtype: torch.dtype,
    source_mode_order: Optional[Sequence[int]],
    compile_options: CompileOptions,
):
    """The checked request and the in-order copy both forms are built from.

    Returns the in-order copy of the operand into an all-compressed tensor,
    lowered by the ordinary route, its loops and append leaf, a builder
    resuming it, the converted tensor's declaration fields, and ``permutation``:
    the operand's storage level each converted level holds.
    """

    rank = len(operand_format)
    if set(operand_format) - {"d", "s"}:
        _refuse("a conversion reads dense and compressed levels only")
    if sorted(conversion.mode_order) != list(range(rank)):
        _refuse("a conversion's mode order must permute the operand's modes")
    target_format = conversion.format
    if len(target_format) != rank or set(target_format) - {"d", "s"}:
        _refuse("a converted layout names one dense or compressed level per mode")
    if not stores_only_its_entries(target_format):
        _refuse(
            "a converted layout with a dense level below a compressed one "
            "would store cells the operand does not hold"
        )
    source_modes = list(range(rank) if source_mode_order is None else source_mode_order)
    if sorted(source_modes) != list(range(rank)):
        _refuse("the operand's mode order must permute its modes")
    level_of_mode = {mode: level for level, mode in enumerate(source_modes)}
    permutation = tuple(level_of_mode[mode] for mode in conversion.mode_order)

    variables = [IndexVar(f"x{level}") for level in range(rank)]
    source = TensorVar("A", fmt=operand_format, dtype=dtype)[tuple(variables)]
    copy = TensorVar("C", fmt="s" * rank, dtype=dtype)[tuple(variables)]
    statement: IndexStmt = TensorAssign(copy, source)
    for variable in reversed(variables):
        statement = ForAll(variable, statement)
    normalized = normalize_cin(statement, compile_options=compile_options)
    program = lower_normalized_cin_to_loopir(normalized).program
    loops, leaf = _decompose_chain(program)
    if type(leaf) is not AppendEntry:
        _refuse("the in-order copy of this operand layout does not end in an append")
    assert type(leaf) is AppendEntry

    builder = LoopIRBuilder.resuming(program)
    result_decl = next(decl for decl in program.tensors if decl.symbol == leaf.tensor)
    source_decl = next(decl for decl in program.tensors if decl.symbol != leaf.tensor)
    dimensions = tuple(result_decl.dimensions[level] for level in permutation)
    levels = tuple(
        builder.level(LevelKind.DENSE if kind == "d" else LevelKind.COMPRESSED, level)
        for level, kind in enumerate(target_format)
    )
    converted_decl = replace(result_decl, dimensions=dimensions, levels=levels)
    return program, loops, leaf, builder, source_decl, converted_decl, permutation


def conversion_program(
    operand_format: str,
    conversion: OperandConversion,
    dtype: torch.dtype,
    *,
    source_mode_order: Optional[Sequence[int]] = None,
    compile_options: CompileOptions,
) -> Tuple[LoopProgram, SymbolId, SymbolId]:
    """The sparse-workspace copy of an operand stored as ``operand_format``.

    The operand is read as the arrays hold it: its levels in storage order.
    ``source_mode_order`` is the logical mode each of those levels holds
    (identity when omitted), and ``conversion.mode_order`` the logical mode
    each converted level holds.  Returns the verified program, the operand's
    symbol and the converted tensor's symbol.

    The loops over the operand and the value read are those of the in-order
    copy of the operand into an all-compressed tensor, lowered by the ordinary
    route; its append leaf becomes an insertion keyed by the converted level
    order, and the drain appends into the converted layout.
    """

    program, loops, leaf, builder, source_decl, converted_decl, permutation = (
        _in_order_copy(
            operand_format, conversion, dtype, source_mode_order, compile_options
        )
    )
    workspace = builder.new_workspace_id()
    insert = builder.sparse_workspace_insert(
        workspace,
        tuple(leaf.coords[level] for level in permutation),
        ReduceOp.ADD,
        leaf.value,
    )
    drained = tuple(builder.new_index_id() for _ in permutation)
    append = builder.append_entry(
        leaf.tensor,
        tuple(builder.index_value(index) for index in drained),
        builder.sparse_workspace_value(workspace),
    )
    region = builder.sparse_workspace_region(
        builder.sparse_workspace_decl(
            workspace, "wksp", converted_decl.dtype, converted_decl.dimensions
        ),
        _wrap_loops(builder, loops, builder.block((insert,))),
        builder.block(
            (
                builder.sparse_workspace_drain_for(
                    workspace, drained, builder.block((append,))
                ),
            )
        ),
    )
    converted = builder.program(
        program.dimensions,
        (source_decl, converted_decl),
        program.inputs,
        program.outputs,
        builder.block((region,)),
    )
    verify_program(converted)
    return converted, source_decl.symbol, leaf.tensor


def relabelled_copy_program(
    operand_format: str,
    conversion: OperandConversion,
    dtype: torch.dtype,
    *,
    source_mode_order: Optional[Sequence[int]] = None,
    compile_options: CompileOptions,
    parallel: bool = False,
) -> Tuple[LoopProgram, SymbolId, SymbolId]:
    """The relabelled copy of an operand stored as ``operand_format``.

    The in-order copy of :func:`conversion_program` with its append moved to
    the converted coordinates: each stored entry is appended at its
    coordinates in the converted level order, out of the converted tensor's
    storage order, under a recorded two-pass assembly, which is what the
    verifier admits such an append under.  When the converted levels hold the
    operand's levels in the same order, the append is in order and no strategy
    is recorded: the copy needs no sort, and :func:`conversion_source` lowers
    it by the ordinary route as the in-order copy.  Same arguments and returns
    as :func:`conversion_program`.

    The copy records ``two_pass_serial``: the one-thread counting sort or
    radix sort.  ``parallel`` records ``two_pass_parallel`` instead, whose
    kernel picks its thread count when it runs and runs the one-thread kernel
    on one thread.  The parallel form is a candidate beside the serial one,
    not yet its replacement: it becomes the default only where the timing grid
    shows it neutral-or-better, behind a condition that keeps it off where it
    is slower.
    """

    program, loops, leaf, builder, source_decl, converted_decl, permutation = (
        _in_order_copy(
            operand_format, conversion, dtype, source_mode_order, compile_options
        )
    )
    append = builder.append_entry(
        leaf.tensor,
        tuple(leaf.coords[level] for level in permutation),
        leaf.value,
    )
    body: Block = _wrap_loops(builder, loops, builder.block((append,)))
    converted = builder.program(
        program.dimensions,
        (source_decl, converted_decl),
        program.inputs,
        program.outputs,
        body,
    )
    if permutation != tuple(range(len(permutation))):
        strategy = (
            AssemblyStrategy.TWO_PASS_PARALLEL
            if parallel
            else AssemblyStrategy.TWO_PASS_SERIAL
        )
        converted = replace(converted, assembly=strategy)
    verify_program(converted)
    return converted, source_decl.symbol, leaf.tensor


def converted_shape(
    shape: Sequence[int],
    conversion: OperandConversion,
    source_mode_order: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """The converted tensor's storage-order shape, from the operand's."""

    rank = len(shape)
    source_modes = list(range(rank) if source_mode_order is None else source_mode_order)
    level_of_mode = {mode: level for level, mode in enumerate(source_modes)}
    return tuple(shape[level_of_mode[mode]] for mode in conversion.mode_order)


def conversion_source(
    operand_format: str,
    conversion: OperandConversion,
    dtype: torch.dtype,
    shape: Sequence[int],
    *,
    source_mode_order: Optional[Sequence[int]] = None,
    compile_options: CompileOptions,
    forms: Sequence[str] = ("relabelled", "region"),
) -> str:
    """The C++ of the kernel that converts an operand of storage-order ``shape``.

    The conversion is a prerequisite of the scheduled operation, not the
    operation: it is lowered without the request's schedule, under its own
    compilation context.  The relabelled copy is lowered first; a layout its
    target refuses with ``unsupported_relabelled_assembly`` gets the
    sparse-workspace copy, under a fresh context.  A copy that keeps the
    operand's level order (only the layout changes) records no strategy and is
    lowered by the ordinary route as the in-order copy: one walk, appends in
    order, no sort; any target refusal of it falls back to the
    sparse-workspace copy.

    ``forms`` names the forms tried, in order: ``"relabelled"`` (on one
    thread, :func:`relabelled_copy_program`), ``"relabelled_parallel"`` (the
    same on the thread count chosen when it runs) and ``"region"`` (the
    sparse-workspace copy).  Naming one alone is for measuring the forms
    against each other: ``("region",)`` builds the sparse-workspace copy of
    any layout, and a relabelled form alone raises the target's refusal
    instead of falling back.
    """

    from ...ops import _lower_generated_llir

    builders: Dict[str, Callable[..., Tuple[LoopProgram, SymbolId, SymbolId]]] = {
        "relabelled": relabelled_copy_program,
        "relabelled_parallel": partial(relabelled_copy_program, parallel=True),
        "region": conversion_program,
    }
    if not forms or any(form not in builders for form in forms):
        raise ValueError(f"forms must name forms among {sorted(builders)}: {forms!r}")
    options = compile_options
    if options.requested_schedule is not None:
        options = replace(options, requested_schedule=None)
    result_shape = converted_shape(shape, conversion, source_mode_order)
    for position, name in enumerate(forms):
        form = builders[name]
        last = position == len(forms) - 1
        program, operand_symbol, _ = form(
            operand_format,
            conversion,
            dtype,
            source_mode_order=source_mode_order,
            compile_options=options,
        )
        relabelled = name != "region"
        # A copy that keeps the operand's level order records no strategy: its
        # appends arrive in order, and the ordinary route lowers it as one walk
        # with no sort (the in-order copy).
        in_order = relabelled and program.assembly is None
        context = CompilationContext(options)
        try:
            function = lower_loopir_to_llir(
                program,
                input_shapes={operand_symbol: tuple(shape)},
                result_shape=result_shape,
                compile_options=options,
                compilation_context=context,
            )
        except LoopIRTargetError as error:
            if not last and (
                in_order
                or (relabelled and error.defect.code == UNSUPPORTED_RELABELLED_ASSEMBLY)
            ):
                continue
            raise
        return _lower_generated_llir(function, options, context)
    raise AssertionError("the last form either returns or raises")
