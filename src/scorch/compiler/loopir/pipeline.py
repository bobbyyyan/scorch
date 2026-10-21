"""Test/debug LoopIR strangler pipeline and curated shadow comparison.

This module is the executable end of the Phase-4/5/6 vertical slices: it
carries one normalized supported-family CIN program through

``normalized CIN -> LoopPlan -> LoopIR -> scheduled LoopIR ->
structured LLIR -> C++ -> compiled kernel -> execution``

using the same JIT build, cache, and execution helpers as the legacy public
entry (:func:`scorch.ops.lower_and_exec_cin`).  When the compile options
carry a requested :class:`~scorch.compiler.scheduler.Schedule`, the shared
``Scheduler.apply_schedule`` boundary validates it and produces the
verified ``LoopPlan``; the migrated explicit families (loop reorder,
affine ``accum='direct'`` tiles, and ``accum='stack'`` workspace
accumulation) are then applied as typed LoopIR passes,
and every other schedule family fails closed with a stable code —
nothing silently ignores a requested schedule.  Kernel cache identity
remains source-derived, exactly as on the legacy path: the schedule
affects the generated source, identical source is the identical kernel
artifact, and no LoopIR- or plan-level artifact is ever cached.

It exists for dedicated LoopIR tests and curated shadow comparison only:

- production never imports it, so the legacy default path, its stage
  sequences, and its cache keys are untouched;
- kernel cache identity remains derived from the generated source exactly as
  before.  For the migrated families the LoopIR path generates byte-identical
  source, so it shares the legacy kernel artifact honestly — identical source
  is identical kernel — while LoopIR-level artifacts (programs, dumps) are
  never cached at all;
- shadow comparison (:func:`compare_generated_sources`,
  :func:`execute_shadow`) is invoked explicitly by curated differential
  tests; nothing turns it on globally, so ordinary pytest and release JIT do
  not double compile.

Stage timing: the LoopIR stages are recorded through the same
:class:`~scorch.compiler.compilation_context.CompilationContext` mechanism
under the new ``CIN_TO_LOOPIR_LOWERING`` and ``LOOPIR_TO_LLIR_LOWERING``
identities; downstream C++ generation and build-request assembly reuse the
existing canonical stage identities.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from collections import deque
from functools import reduce
from dataclasses import dataclass, replace
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from ...exceptions import (
    CompileSpecError,
    raise_if_native_coded,
    shape_capacity_error,
)
from .. import llir
from ..cin import (
    BinaryOp,
    CINVisitorAccept,
    ForAll,
    IndexStmt,
    Operation,
    TensorAccess,
    TensorAssign,
    Where,
    Workspace,
)
from ..cin_analysis import (
    _trusted_normalized_cin,
    normalize_cin,
    verify_cin_structure,
)
from ..cin_lowerer import CINLowerer
from ..compilation_context import CompilationContext, CompilerStageId
from ..compile_options import CompileOptions
from ..diagnostics import (
    InvalidSchedule,
    ShapeCapacityVerificationError,
    UnsupportedFeature,
    VerificationError,
)
from ...format import LevelType
from ...layout import SUPPORTED_VALUE_DTYPES, common_value_dtype
from ..identity import IndexId, SymbolId
from ..loop_plan import (
    LoopPlan,
    OperandConversion,
    ScheduledCIN,
    converted_cin,
    out_of_order_modes,
    storage_order_form,
    verify_loop_plan,
)
from .lower_cin import (
    LoopIRLoweringDefect,
    LoopIRLoweringError,
    LoopIRLoweringResult,
    lower_normalized_cin_to_loopir,
    refuse_where_program,
    supported_scalar_type,
)
from .assembly_selection import (
    AssemblyDecision,
    choose_unrequested_assembly,
    complete_automatic_plan,
)
from .lower_llir import assembles_sparse_result, lower_loopir_to_llir
from .operand_conversion import (
    conversion_format,
    conversion_source,
    converted_shape,
    level_letters,
)
from .nodes import LevelKind as LoopIRLevelKind, LoopProgram
from .printer import canonical_program_dump, print_program
from .schedule_passes import ScheduledLoopIR, apply_schedule_plan

# The format letter of each level kind a kernel's result is assembled in; the
# kernel returns one coordinate array per coordinate level and a position and
# a coordinate array per compressed level, and the result is wrapped as such.
_RESULT_LEVEL_LETTERS = {
    LoopIRLevelKind.DENSE: "d",
    LoopIRLevelKind.COMPRESSED: "s",
    LoopIRLevelKind.COORDINATE: "o",
}


@dataclass(frozen=True)
class LoopIRCompiledKernel:
    """One compiled supported-family kernel and its LoopIR provenance.

    For scheduled compilations ``schedule`` retains the full scheduling
    artifact (base program, exact plan, scheduled program, provenance) and
    ``program_text``/``program_dump`` describe the scheduled program — the
    artifact the target lowering consumed.  ``request_dump`` and
    ``request_identity`` are the canonical strangler request serialization
    and its SHA-256 content key, computed at the compile/shadow request
    boundary from the canonical normalized CIN, the canonical verified plan
    (or the explicit unscheduled marker), the result shape, runtime input
    bindings, and exact canonical CompileOptions state; no cache consumes
    them yet, and the release source-derived cache is untouched.

    ``conversions`` are the operand conversions the plan records: the kernel
    reads each converted operand in its converted layout.
    ``conversion_sources`` holds, in the same order, the C++ of the kernel
    that converts each one from the layout the program declares for it
    (:mod:`.operand_conversion`); it is part of the compiled program, and
    :func:`execute_cin_via_loopir` runs it before the main kernel.  An
    operand stored in another mode order is converted from that order
    instead, and the kernel execution returns holds the source that ran.

    ``result_mode_order`` is the logical mode each storage level of the
    kernel's result holds, as the program declares it: what a caller passes
    with the result arrays to the kernel-result wrapper.

    ``assembly_decision`` is the record of the assembly stage, which runs
    whenever the program records no assembly strategy and its result has a
    level that is not dense (:mod:`.assembly_selection`): the strategies the
    target could emit, the extent the rule read and the strategy the kernel is
    assembled with.  For an automatic plan ``schedule.plan`` is then the
    completed plan, which records the same strategy, and ``program_dump``
    shows it; a plan its caller wrote without a strategy and an unscheduled
    program keep recording none, and the decision is the only record.
    ``request_dump`` and ``request_identity`` are computed before the stage,
    from the plan as it was requested.  ``None`` when the stage did not run: a
    requested strategy, or an all-dense result.
    """

    lowering: LoopIRLoweringResult
    program_text: str
    program_dump: str
    llir_function: llir.Function
    cpp_source: str
    schedule: Optional[ScheduledLoopIR]
    request_dump: str
    request_identity: str
    conversions: Tuple[OperandConversion, ...] = ()
    conversion_sources: Tuple[str, ...] = ()
    result_mode_order: Tuple[int, ...] = ()
    assembly_decision: Optional[AssemblyDecision] = None

    def __post_init__(self) -> None:
        if len(self.conversion_sources) != len(self.conversions):
            raise TypeError("each conversion needs its conversion kernel source")
        if type(self.request_dump) is not str or not self.request_dump:
            raise TypeError("request_dump must be a non-empty canonical string")
        if (
            type(self.request_identity) is not str
            or len(self.request_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.request_identity
            )
        ):
            raise TypeError("request_identity must be a lowercase SHA-256 digest")
        expected = hashlib.sha256(self.request_dump.encode("utf-8")).hexdigest()
        if self.request_identity != expected:
            raise TypeError("request_identity must hash the retained request_dump")


@dataclass(frozen=True)
class ShadowSourceComparison:
    """Curated legacy-versus-LoopIR generated-source comparison."""

    loopir_cpp: str
    legacy_cpp: str

    @property
    def identical(self) -> bool:
        return self.loopir_cpp == self.legacy_cpp


def _resolve_options(
    compile_options: Optional[CompileOptions],
    compilation_context: Optional[CompilationContext] = None,
) -> CompileOptions:
    if (
        compilation_context is not None
        and type(compilation_context) is not CompilationContext
    ):
        raise TypeError("compilation_context must be a CompilationContext")
    if compile_options is None:
        options = (
            compilation_context.compile_options
            if compilation_context is not None
            else CompileOptions.from_environment()
        )
    elif type(compile_options) is not CompileOptions:
        raise TypeError("compile_options must be a CompileOptions snapshot")
    else:
        options = compile_options
    if compilation_context is not None:
        compilation_context.require_compile_options(options)
    return options


def _plan_mode_orders_to_planned_order(
    normalized: IndexStmt,
    args: Sequence[object],
    plan: LoopPlan,
    *,
    dense_by_coordinate: bool = False,
    converted_by_plan: bool = False,
):
    """The scheduled-path twin of ``ops._plan_mode_orders_to_loop_order``.

    The legacy helper reads the iteration order from the CIN's own ForAll
    chain; a scheduled compilation iterates in the plan's logical order
    instead, so runtime mode-order alignment must target that order — the
    same storage-vs-iteration consistency the reorder pass and the target
    boundary enforce.  Only the order source differs; the relayout executor
    and CIN alignment application are the shared production helpers.

    ``dense_by_coordinate`` is for a caller whose lowering reads and writes
    an all-dense tensor by its coordinates in any loop order (the LoopIR
    target): such a tensor is not copied into the planned order, so the
    kernel that runs is the kernel the same plan emits at compile time.  The
    kernel reads the operand in the mode order the CIN declares, so an
    operand the caller stored in that order is left as stored, and one stored
    in any other order is relaid out to the declared order (never the planned
    one).  A tensor with a compressed level is aligned either way, unless
    ``converted_by_plan`` is set and the plan converts it
    (``plan.conversions``): that operand is then left as stored for its own
    conversion kernel.
    """

    from ..loop_plan import entity_display_names

    index_names, _ = entity_display_names(normalized)
    loop_order_names = [index_names[index_id] for index_id in plan.loop_order]
    if not loop_order_names:
        return tuple(None for _ in args), None

    rhs_accesses = normalized.get_rhs_tensor_accesses()
    if len(rhs_accesses) != len(args):
        return tuple(None for _ in args), None

    converted_ids = {conversion.operand_id for conversion in plan.conversions}
    rhs_mode_orders: List[Optional[Tuple[int, ...]]] = []
    for arg, access in zip(args, rhs_accesses):
        tensor_var = access.get_tensor()
        access_names = [index_var.name for index_var in access.get_index_vars()]
        desired_names = [name for name in loop_order_names if name in access_names]
        desired = tuple(access_names.index(name) for name in desired_names)
        mode_order = tensor_var.mode_order
        if mode_order is None or len(desired) != len(mode_order):
            rhs_mode_orders.append(None)
            continue
        if dense_by_coordinate and _all_dense(tensor_var):
            declared = tuple(mode_order)
            rhs_mode_orders.append(
                None if _stored_mode_order(arg) == declared else declared
            )
            continue
        if converted_by_plan and tensor_var.symbol_id in converted_ids:
            rhs_mode_orders.append(None)
            continue
        rhs_mode_orders.append(desired)

    lhs_mode_order: Optional[Tuple[int, ...]] = None
    leaf: IndexStmt = normalized
    while hasattr(leaf, "stmt"):
        leaf = leaf.stmt
    lhs_access = getattr(leaf, "lhs", None)
    if lhs_access is not None:
        lhs_tensor = lhs_access.get_tensor()
        lhs_names = [index_var.name for index_var in lhs_access.get_index_vars()]
        desired_names = [name for name in loop_order_names if name in lhs_names]
        desired = tuple(lhs_names.index(name) for name in desired_names)
        lhs_modes = lhs_tensor.mode_order
        if (
            lhs_modes is not None
            and len(desired) == len(lhs_modes)
            and not (dense_by_coordinate and _all_dense(lhs_tensor))
        ):
            lhs_mode_order = desired
    return tuple(rhs_mode_orders), lhs_mode_order


def _all_dense(tensor_var) -> bool:
    """Whether every level of ``tensor_var`` is a dense level."""

    tensor_format = tensor_var.format
    return tensor_format is not None and tensor_format.is_dense()


def _stored_mode_order(arg: object) -> Tuple[int, ...]:
    """The mode order a runtime operand, or the snapshot of one, is stored in."""

    return tuple(getattr(arg, "mode_order"))


def _stored_and_declared_orders(
    snapshot, operand
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """The mode order a runtime operand is stored in and the one its program declares."""

    rank = len(snapshot.shape)
    return (
        tuple(snapshot.mode_order or range(rank)),
        tuple(operand.mode_order or range(rank)),
    )


def _declared_order_shape(snapshot, operand) -> Tuple[int, ...]:
    """A runtime operand's shape in the level order its program declares.

    The kernel is compiled for the declared order.  The alignment leaves an
    operand the plan converts in the mode order the caller stores it in, and
    its conversion kernel reads that order (:func:`_run_conversions`), so the
    stored shape of such an operand lists the same extents in another order.
    Every other operand is stored in the declared order by the time it is
    bound, which :func:`_require_converted_or_stored_as_declared` checks once
    the plan's conversions are known, and its shape is returned as it is.
    """

    stored, declared = _stored_and_declared_orders(snapshot, operand)
    if stored == declared:
        return snapshot.shape
    level_of_mode = {mode: level for level, mode in enumerate(stored)}
    return tuple(snapshot.shape[level_of_mode[mode]] for mode in declared)


def _refuse_operand(code: str, message: str) -> NoReturn:
    raise LoopIRLoweringError(LoopIRLoweringDefect(code, message))


def _require_converted_or_stored_as_declared(
    normalized: IndexStmt,
    runtime_args: Sequence[object],
    conversions: Sequence[OperandConversion],
) -> None:
    """Refuse an operand stored in an undeclared mode order that no conversion reads.

    The kernel is compiled for the declared order and reads each operand's
    arrays as they are passed.  The one operand that may still be stored in
    another order when the kernel is built is an operand the plan converts:
    its conversion kernel reads the stored order.  Any other one means the
    alignment did not run.  The kernel would read that operand's levels in the
    wrong order, and with equal extents no extent check can tell, so the entry
    checks instead of trusting the alignment.
    """

    converted = {conversion.operand_id for conversion in conversions}
    for snapshot, operand in zip(runtime_args, normalized.get_rhs_tensor_vars()):
        stored, declared = _stored_and_declared_orders(snapshot, operand)
        if stored != declared and operand.symbol_id not in converted:
            _refuse_operand(
                "unconverted_operand_order",
                f"operand {operand.name!r} is stored in mode order {stored}, its "
                f"program declares {declared}, and the plan does not convert it",
            )


def _conversion_letters(tensor) -> str:
    """The ``d``/``s`` letters of an operand a plan converts.

    A conversion copies dense and compressed levels
    (:func:`.operand_conversion.level_letters`); an operand with any other
    level kind is refused.
    """

    letters = level_letters(tensor.get_level_types())
    if letters is None:
        _refuse_operand(
            "unsupported_operand_conversion",
            f"operand {tensor.name!r} has a level that is neither dense nor "
            "compressed, which a conversion does not read",
        )
    return letters


def _require_dense_operands_in_declared_order(
    normalized: IndexStmt,
    runtime_args: Sequence[object],
) -> None:
    """Refuse an all-dense operand whose stored mode order the CIN does not declare.

    The LoopIR target reads an all-dense operand by coordinate in the declared
    order and no extent check can tell two orders of equal extents apart, so
    the entry checks the operands it is about to pass, after any relayout,
    instead of trusting that the relayout ran.
    """

    for arg, tensor_var in zip(runtime_args, normalized.get_rhs_tensor_vars()):
        declared = tensor_var.mode_order
        if declared is None or not _all_dense(tensor_var):
            continue
        if _stored_mode_order(arg) != tuple(declared):
            raise CompileSpecError(
                f"CIN tensor {tensor_var.name!r} declares mode order "
                f"{tuple(declared)} and its dense operand is stored in mode "
                f"order {_stored_mode_order(arg)}"
            )


def _execute_mode_order_alignment(
    normalized: IndexStmt,
    args: tuple,
    alignment_plan,
    options: CompileOptions,
    context: Optional[CompilationContext],
    *,
    runtime_snapshots,
) -> tuple:
    """Apply one alignment and return validated post-relayout snapshots.

    A storage relayout may compile an auxiliary kernel.  That prerequisite is
    not the scheduled operation, so it receives a schedule-free copy of the
    already-resolved options and a matching independent instrumentation owner.
    Every other build and compiler option remains identical.
    """

    from ...ops import (
        _apply_mode_order_alignment,
        _relayout_mode_order_args,
        _snapshot_runtime_tensors,
    )

    relayout_options = options
    relayout_context = context
    if options.requested_schedule is not None:
        relayout_options = replace(options, requested_schedule=None)
        relayout_context = CompilationContext(relayout_options)
    aligned = _relayout_mode_order_args(
        args,
        alignment_plan,
        relayout_options,
        relayout_context,
        snapshots=runtime_snapshots,
    )
    aligned_snapshots = _snapshot_runtime_tensors(
        aligned,
        expected_count=len(runtime_snapshots),
    )
    _apply_mode_order_alignment(normalized, alignment_plan)
    return aligned_snapshots


def _storage_order_walk(
    chains: Sequence[Sequence[IndexId]], loop_ids: Sequence[IndexId]
) -> Tuple[List[IndexId], bool]:
    """Kahn's algorithm over storage-order chains; whether it met a cycle.

    Index variables in order of first appearance (chains, then the remaining
    loops), ready ones first in, first out.  Where the chains disagree, the
    earliest-appearing remaining variable is taken and the second value is
    ``True``.
    """

    variables = list(
        dict.fromkeys(
            [index_id for chain in chains for index_id in chain] + [*loop_ids]
        )
    )
    successors: Dict[IndexId, List[IndexId]] = {index_id: [] for index_id in variables}
    in_degree = dict.fromkeys(variables, 0)
    for chain in chains:
        for parent, child in zip(chain, chain[1:]):
            if child not in successors[parent]:
                successors[parent].append(child)
                in_degree[child] += 1
    ready = deque(index_id for index_id in variables if in_degree[index_id] == 0)
    order: List[IndexId] = []
    placed = set()
    cycle = False
    while len(order) < len(variables):
        if not ready:
            cycle = True
            ready.append(next(i for i in variables if i not in placed))
        index_id = ready.popleft()
        if index_id in placed:
            continue
        placed.add(index_id)
        order.append(index_id)
        for child in successors[index_id]:
            in_degree[child] -= 1
            if in_degree[child] == 0 and child not in placed:
                ready.append(child)
    return order, cycle


def _nest_in_storage_order(cin_stmt: IndexStmt) -> IndexStmt:
    """The automatic plan's starting nest, as a function of the program alone.

    ``Scheduler.select_loop_order`` starts from the written nest and keeps its
    order wherever the cost model ties (a stable sort by sparsity), so two
    spellings of one program could get two plans, and the new compiler could
    accept one and refuse the other.  A perfect ForAll nest over one assignment
    fixes no execution order, so it is re-nested here in the order the public
    einsum writes (``utils.topo_sort_characters``): Kahn's algorithm over each
    operand's storage order, left to right, then the result's, index variables
    in order of first appearance, ready ones first in, first out.  Where the
    storage orders disagree, the earliest-appearing remaining variable is
    taken, and the scheduler's own cycle resolution then decides the plan.

    Any other statement shape, and any access whose mode order the normalized
    CIN verifier has not yet checked, is returned unchanged; so is a nest that
    is already in that order.  The caller's statement is never mutated.
    """

    loops: List[ForAll] = []
    current: IndexStmt = cin_stmt
    while type(current) is ForAll:
        loops.append(current)
        current = current.stmt
    if type(current) is not TensorAssign or not loops:
        return cin_stmt
    by_id: Dict[IndexId, ForAll] = {}
    for loop in loops:
        if loop.index_var.index_id in by_id:
            return cin_stmt
        by_id[loop.index_var.index_id] = loop
    chains: List[List[IndexId]] = []
    for access in [*current.get_rhs_tensor_accesses(), current.lhs]:
        if type(access) is not TensorAccess:
            return cin_stmt
        indices = list(access.indices)
        mode_order = access.tensor.mode_order
        modes = list(range(len(indices))) if mode_order is None else list(mode_order)
        if sorted(modes) != list(range(len(indices))):
            return cin_stmt
        chain = [indices[mode].index_id for mode in modes]
        if any(index_id not in by_id for index_id in chain):
            return cin_stmt
        chains.append(chain)

    order, _ = _storage_order_walk(chains, [loop.index_var.index_id for loop in loops])
    if order == [loop.index_var.index_id for loop in loops]:
        return cin_stmt
    nested: IndexStmt = current
    for index_id in reversed(order):
        loop = by_id[index_id]
        nested = ForAll(loop.index_var, nested, parallel=loop.parallel)
    return nested


def _storage_chain(access: TensorAccess) -> List[IndexId]:
    """The access's index variables in its tensor's storage order."""

    indices = list(access.indices)
    mode_order = access.tensor.mode_order
    modes = list(range(len(indices))) if mode_order is None else list(mode_order)
    return [indices[mode].index_id for mode in modes]


def _stored_in_order(access: TensorAccess) -> bool:
    """Whether every loop order must follow the tensor's storage order.

    A compressed level needs its parent's position before its loop starts and
    a dense level below a compressed one needs that compressed position, so a
    tensor with any non-dense level is visited in storage order.  An
    all-dense tensor is addressed by its coordinates in any order.
    """

    return any(
        level is not LevelType.DENSE for level in access.tensor.get_level_types()
    )


def _conversion_candidates(
    loops: Sequence[IndexId],
    operands: Sequence[TensorAccess],
    result: TensorAccess,
    requested: Optional[Sequence[IndexId]],
) -> List[Tuple[OperandConversion, ...]]:
    """The operand conversions that let some loop order follow every tensor.

    The loop order determines the conversions: each compressed operand the
    order does not follow is converted to that order restricted to its own
    indices.  With a requested order that is the one candidate.  Otherwise
    every set of convertible operands is tried, smallest first: the loop
    order is the storage-order walk over the tensors left unconverted (the
    rule the canonical nest uses), and the set is a candidate when that walk
    meets no cycle and converts every operand in it.  A tensor accessed more
    than once, a one-mode tensor, an all-dense tensor and a tensor with a
    level that is neither dense nor compressed are never converted; the
    result is not either (its conversion is the sparse workspace's job).
    Each converted operand takes the layout that stores exactly its entries
    (:func:`.operand_conversion.conversion_format`).
    """

    accesses_of: Dict[SymbolId, List[TensorAccess]] = {}
    for access in operands:
        accesses_of.setdefault(access.tensor.symbol_id, []).append(access)
    convertible = [
        accesses[0]
        for accesses in accesses_of.values()
        if len(accesses) == 1
        and len(accesses[0].indices) > 1
        and _stored_in_order(accesses[0])
        and level_letters(accesses[0].tensor.get_level_types()) is not None
    ]

    def conversions_for(
        order: Sequence[IndexId], chosen: Sequence[TensorAccess]
    ) -> Optional[Tuple[OperandConversion, ...]]:
        position = {index_id: n for n, index_id in enumerate(order)}
        conversions = []
        for access in chosen:
            ids = [index.index_id for index in access.indices]
            mode_order = tuple(sorted(range(len(ids)), key=lambda m: position[ids[m]]))
            current = tuple(access.tensor.mode_order or range(len(ids)))
            if mode_order == current:
                return None
            letters = _conversion_letters(access.tensor)
            conversions.append(
                OperandConversion(
                    access.tensor.symbol_id, mode_order, conversion_format(letters)
                )
            )
        return tuple(sorted(conversions, key=lambda c: c.operand_id.value))

    def follows(order: Sequence[IndexId], access: TensorAccess) -> bool:
        position = {index_id: n for n, index_id in enumerate(order)}
        chain = [position[index_id] for index_id in _storage_chain(access)]
        return chain == sorted(chain)

    fixed = [access for access in operands if _stored_in_order(access)]
    if _stored_in_order(result):
        fixed.append(result)
    if requested is not None:
        broken = [access for access in fixed if not follows(requested, access)]
        if not broken or any(
            all(access is not candidate for candidate in convertible)
            for access in broken
        ):
            return []
        found = conversions_for(requested, broken)
        return [] if found is None else [found]

    candidates: List[Tuple[OperandConversion, ...]] = []
    for size in range(1, len(convertible) + 1):
        for chosen in combinations(convertible, size):
            kept = [
                access
                for access in fixed
                if all(access is not converted for converted in chosen)
            ]
            order, cycle = _storage_order_walk(
                [_storage_chain(access) for access in kept], loops
            )
            if cycle:
                continue
            found = conversions_for(order, chosen)
            if found is not None:
                candidates.append(found)
    return candidates


def _needs_conversion(
    loops: Sequence[IndexId],
    operands: Sequence[TensorAccess],
    result: TensorAccess,
    requested: Optional[Sequence[IndexId]],
) -> bool:
    """Whether no loop order (or not the requested one) follows every tensor."""

    fixed = [access for access in operands if _stored_in_order(access)]
    if _stored_in_order(result):
        fixed.append(result)
    if requested is not None:
        position = {index_id: n for n, index_id in enumerate(requested)}
        return any(
            [position[i] for i in _storage_chain(access)]
            != sorted(position[i] for i in _storage_chain(access))
            for access in fixed
        )
    _, cycle = _storage_order_walk([_storage_chain(a) for a in fixed], loops)
    return cycle


def _schedule_with_conversions(
    cin_stmt: IndexStmt,
    options: CompileOptions,
    context: Optional[CompilationContext],
    attempt: Callable[
        [IndexStmt, Optional[CompilationContext]], Tuple[IndexStmt, LoopPlan]
    ],
) -> Optional[Tuple[IndexStmt, LoopPlan]]:
    """The cheapest plan that converts operands, or ``None`` if none verifies.

    Each candidate (:func:`_conversion_candidates`) is scheduled on the
    program with its operands converted, under a private compilation context
    (a refused candidate must not end the caller's).  Candidates are ranked
    by ``Scheduler.loop_order_cost``, the scheduler's own objective, whose
    transposition term prices each converted operand.  The winner is
    scheduled once more under the caller's context, and its plan records the
    conversions; it is verified against the unconverted program, which is
    returned with it, because execution converts the caller's operands from
    their declared layouts.
    """

    from ..scheduler import Scheduler

    schedule = options.requested_schedule
    assert schedule is not None
    loops: List[IndexId] = []
    current: IndexStmt = cin_stmt
    while type(current) is ForAll:
        loops.append(current.index_var.index_id)
        current = current.stmt
    if type(current) is not TensorAssign or len(set(loops)) != len(loops):
        return None
    operands = list(current.get_rhs_tensor_accesses())
    if any(type(access) is not TensorAccess for access in [*operands, current.lhs]):
        return None
    requested: Optional[List[IndexId]] = None
    if schedule.loop_order is not None:
        by_name: Dict[str, IndexId] = {}
        stmt: IndexStmt = cin_stmt
        while type(stmt) is ForAll:
            by_name[stmt.index_var.name] = stmt.index_var.index_id
            stmt = stmt.stmt
        if any(name not in by_name for name in schedule.loop_order):
            return None
        requested = [by_name[name] for name in schedule.loop_order]
    if not _needs_conversion(loops, operands, current.lhs, requested):
        return None

    def converted_program(conversions: Tuple[OperandConversion, ...]) -> IndexStmt:
        program = converted_cin(cin_stmt, conversions)
        if requested is None:
            program = _nest_in_storage_order(program)
        verify_cin_structure(program)
        return program

    ranked = []
    for conversions in _conversion_candidates(loops, operands, current.lhs, requested):
        program = converted_program(conversions)
        try:
            normalized, plan = attempt(program, CompilationContext(options))
        except (InvalidSchedule, UnsupportedFeature, VerificationError):
            continue
        cost = Scheduler.loop_order_cost(
            normalized,
            plan.loop_order,
            transposed=cin_stmt,
            compile_options=options,
        )
        ranked.append(
            (
                cost,
                len(conversions),
                [conversion.operand_id.value for conversion in conversions],
                conversions,
            )
        )
    if not ranked:
        return None
    _, _, _, conversions = min(ranked, key=lambda entry: entry[:3])
    _, plan = attempt(converted_program(conversions), context)
    plan = replace(plan, conversions=conversions)
    unconverted = normalize_cin(cin_stmt, compile_options=options)
    verify_loop_plan(unconverted, plan, dense_results_any_order=True)
    return unconverted, plan


def _apply_requested_schedule(
    cin_stmt: IndexStmt,
    options: CompileOptions,
    context: Optional[CompilationContext],
) -> Tuple[IndexStmt, LoopPlan]:
    """Adapt the public Schedule through the shared scheduler boundary.

    ``Scheduler.apply_schedule`` is the single owner of Schedule validation
    and Schedule-to-LoopPlan translation for both pipelines; the paths only
    diverge downstream (legacy replays CIN tree surgery, the LoopIR path
    applies typed passes to the verified base program).  It normalizes the
    CIN and records the scheduling stage itself.
    """

    from ..scheduler import Scheduler

    schedule = options.requested_schedule
    assert schedule is not None
    # This public entry owns malformed-input ordering: no scheduler hook,
    # metadata binder, or build preparation may observe a structurally invalid
    # caller graph.  The narrow trust context lets the scheduler's immediately
    # following normalization reuse this exact successful scan.
    verify_cin_structure(cin_stmt)
    # The lowering refuses every Where program, and the scheduler would first plan the
    # ForAll chain above the Where alone, a plan its own verification refuses.  So a
    # Where program meets the lowering's refusal here, as on the route without a
    # schedule.
    refuse_where_program(cin_stmt)
    if schedule.loop_order is None:
        # The scheduler chooses the order, so it must not see the spelling.
        renested = _nest_in_storage_order(cin_stmt)
        if renested is not cin_stmt:
            cin_stmt = renested
            verify_cin_structure(cin_stmt)

    def attempt(
        program: IndexStmt, attempt_context: Optional[CompilationContext]
    ) -> Tuple[IndexStmt, LoopPlan]:
        with _trusted_normalized_cin(program):
            scheduled = Scheduler.apply_schedule(
                program,
                schedule,
                compile_options=options,
                compilation_context=attempt_context,
                # This lowering requires every tensor in storage order and
                # accepts a reduction innermost into a sparse result, so an
                # automatic plan may take the order composed before the
                # scheduler's forced reorder.
                composed_order_candidate=True,
                # It also writes an all-dense result by coordinates, so the
                # plan may visit that result's levels out of storage order.
                dense_results_any_order=True,
            )
        return scheduled.normalized_cin, scheduled.verified_loop_plan

    # When no loop order (or not the requested one) follows every compressed
    # tensor, the plan converts operands; the loop order determines which.
    converting = _schedule_with_conversions(cin_stmt, options, context, attempt)
    if converting is not None:
        return converting
    return attempt(cin_stmt, context)


def _bind_runtime_metadata(
    cin_stmt: IndexStmt,
    input_bindings: Sequence[Tuple[Sequence[int], torch.dtype]],
    result_shape: Sequence[int],
    input_formats: Optional[Sequence[object]] = None,
) -> Tuple[
    Tuple[Tuple[Tuple[int, ...], torch.dtype], ...],
    Tuple[int, ...],
]:
    """Validate, snapshot, and bind public runtime metadata atomically."""

    max_int64 = (1 << 63) - 1

    def checked_shape(shape: object, owner: str) -> Tuple[int, ...]:
        if (
            type(shape) is not tuple
            and type(shape) is not list
            and type(shape) is not torch.Size
            and type(shape) is not range
        ):
            raise VerificationError(
                f"{owner} must be a tuple, list, torch.Size, or range of exact ints"
            )
        try:
            rank = len(shape)
        except OverflowError as error:
            raise VerificationError(
                f"{owner} rank exceeds the supported limit"
            ) from error
        if rank > 64:
            raise VerificationError(f"{owner} rank must not exceed 64")
        result: List[int] = []
        for mode, extent in enumerate(shape):
            if type(extent) is not int or extent < 0:
                raise VerificationError(
                    f"{owner} must contain nonnegative int64 extents"
                )
            if extent > max_int64:
                raise shape_capacity_error(
                    ShapeCapacityVerificationError,
                    f"{owner} must contain nonnegative int64 extents",
                    owner,
                    result,
                    mode,
                    extent,
                )
            result.append(extent)
        # A zero extent holds no element whatever the extents before it multiply to.
        if 0 not in result:
            count = math.prod(result)
            if count > max_int64:
                raise shape_capacity_error(
                    ShapeCapacityVerificationError,
                    f"{owner} element count exceeds signed int64",
                    owner,
                    result,
                    None,
                    count,
                )
        return tuple(result)

    if type(input_bindings) is not tuple and type(input_bindings) is not list:
        raise VerificationError("runtime input bindings must be a tuple or list")
    rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
    if len(rhs_tensor_vars) != len(input_bindings):
        raise CompileSpecError(
            f"CIN expects {len(rhs_tensor_vars)} runtime tensors, got "
            f"{len(input_bindings)}"
        )
    if input_formats is not None:
        if type(input_formats) is not tuple and type(input_formats) is not list:
            raise CompileSpecError("runtime tensor formats must be a tuple or list")
        if len(input_formats) != len(input_bindings):
            raise CompileSpecError(
                "runtime tensor formats must match the input binding count"
            )
    checked_bindings: List[Tuple[Tuple[int, ...], torch.dtype]] = []
    for position, binding in enumerate(input_bindings):
        if (type(binding) is not tuple and type(binding) is not list) or len(
            binding
        ) != 2:
            raise VerificationError(
                f"runtime input binding {position} must be a shape/dtype pair"
            )
        shape, dtype = binding
        checked_input_shape = checked_shape(
            shape,
            f"runtime input binding {position} shape",
        )
        if type(dtype) is not torch.dtype:
            raise VerificationError(
                f"runtime input binding {position} dtype must be an exact "
                "torch.dtype"
            )
        checked_bindings.append((checked_input_shape, dtype))

    checked_result_shape = checked_shape(result_shape, "runtime result shape")
    if input_formats is not None:
        for position, (tensor_var, actual_format) in enumerate(
            zip(rhs_tensor_vars, input_formats)
        ):
            if tensor_var.format != actual_format:
                raise CompileSpecError(
                    f"CIN tensor {tensor_var.name!r} expects format "
                    f"{tensor_var.format}, got {input_formats[position]}"
                )
    for position, (tensor_var, (shape, dtype)) in enumerate(
        zip(rhs_tensor_vars, checked_bindings)
    ):
        tensor_var.shape = shape
        tensor_var.dtype = dtype
        if tensor_var.mode_order is None:
            tensor_var.mode_order = list(range(len(shape)))
    # The operands' promoted type (PyTorch's rule), as ``A + B`` declares it,
    # and as ``scorch.ops.lower_and_exec_cin`` binds it.  Operands of stored
    # dtypes that promote to one Scorch does not store (int8 with uint8) are
    # refused here; an operand of an unstored dtype is left to the lowering,
    # which refuses it by name (``unsupported_dtype``).
    dtypes = [dtype for _, dtype in checked_bindings]
    if not dtypes:
        output_dtype = torch.float32
    elif all(dtype in SUPPORTED_VALUE_DTYPES for dtype in dtypes):
        output_dtype = common_value_dtype(*dtypes, operation="execute_cin_via_loopir")
    else:
        output_dtype = reduce(torch.promote_types, dtypes)
    # get_result_tensor_vars() names every left-hand side, a Where's workspace included.
    # A workspace is not the result: its rank is its own (Workspace.dim; a scalar one
    # has none) and its dtype was declared with it, so the runtime result's shape and
    # dtype are not bound onto it.
    for tensor_var in cin_stmt.get_result_tensor_vars():
        if isinstance(tensor_var, Workspace):
            continue
        tensor_var.shape = checked_result_shape
        tensor_var.dtype = output_dtype
    return tuple(checked_bindings), checked_result_shape


def _validate_runtime_formats(
    cin_stmt: IndexStmt,
    input_formats: Sequence[object],
) -> None:
    """Reject storage-format mismatches before any runtime relayout work."""

    rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
    if len(rhs_tensor_vars) != len(input_formats):
        raise CompileSpecError(
            f"CIN expects {len(rhs_tensor_vars)} runtime tensors, got "
            f"{len(input_formats)}"
        )
    for tensor_var, actual_format in zip(rhs_tensor_vars, input_formats):
        if tensor_var.format != actual_format:
            raise CompileSpecError(
                f"CIN tensor {tensor_var.name!r} expects format "
                f"{tensor_var.format}, got {actual_format}"
            )


def _compile_normalized_cin_via_loopir(
    normalized: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Sequence[int], torch.dtype]],
    *,
    options: CompileOptions,
    context: CompilationContext,
    input_formats: Optional[Sequence[object]] = None,
    plan: Optional[LoopPlan] = None,
) -> LoopIRCompiledKernel:
    """Lower one privately owned normalized CIN root through LoopIR."""

    with _trusted_normalized_cin(normalized):
        return _compile_normalized_cin_via_loopir_owned(
            normalized,
            result_shape,
            input_bindings,
            options=options,
            context=context,
            input_formats=input_formats,
            plan=plan,
        )


def _compile_normalized_cin_via_loopir_owned(
    normalized: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Sequence[int], torch.dtype]],
    *,
    options: CompileOptions,
    context: CompilationContext,
    input_formats: Optional[Sequence[object]] = None,
    plan: Optional[LoopPlan] = None,
) -> LoopIRCompiledKernel:
    """Implementation under the private normalized-CIN trust context."""

    binding_token = context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=options,
    )
    try:
        checked_input_bindings, checked_result_shape = _bind_runtime_metadata(
            normalized,
            input_bindings,
            result_shape,
            input_formats,
        )
        # A well-formed request over an element type outside the migrated
        # families is an unsupported program, not a malformed request.  The
        # request serialization below encodes only the supported element types
        # and raises an uncoded VerificationError for any other, so the
        # lowering's own classification runs first and the refusal carries its
        # ``unsupported_dtype`` code whichever stage meets the operand first.
        for bound_tensor in normalized.get_rhs_tensor_vars():
            supported_scalar_type(bound_tensor)
        # Request identity belongs to the validated request boundary.  Keep
        # malformed shape/dtype/CIN/options failures owned by this stage rather
        # than running serialization between two completed stage records.
        from .plan_identity import loopir_request_dump

        request_dump = loopir_request_dump(
            normalized,
            plan,
            checked_result_shape,
            checked_input_bindings,
            compile_options=options,
        )
        request_identity = hashlib.sha256(request_dump.encode("utf-8")).hexdigest()
    except Exception:
        context.fail_stage(binding_token)
        raise
    context.complete_stage(binding_token)

    lowering_token = context.begin_stage(
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        compile_options=options,
    )
    # A plan that converts operands is lowered on the program with those
    # operands converted; every other decision of the plan names the same
    # loops and tensors there.
    lowered_cin = normalized
    lowered_plan = plan
    conversion_sources: Tuple[str, ...] = ()
    if plan is not None and plan.conversions:
        lowered_cin = converted_cin(normalized, plan.conversions)
        lowered_plan = replace(plan, conversions=())
    # A tensor with a compressed level is lowered in the identity-order form
    # of its own storage order; the kernel's result keeps the declared order.
    declared_result_order: Optional[Tuple[int, ...]] = None
    for result_access in lowered_cin.get_result_tensor_accesses():
        if declared_result_order is None:
            declared_result_order = out_of_order_modes(result_access)
    lowered_cin = storage_order_form(lowered_cin)
    try:
        with _trusted_normalized_cin(lowered_cin):
            lowering = lower_normalized_cin_to_loopir(
                lowered_cin,
                planned_loop_order=(
                    None if lowered_plan is None else lowered_plan.loop_order
                ),
                planned_entries_of=(
                    None if lowered_plan is None else lowered_plan.entries_of
                ),
            )
        if plan is not None and plan.conversions:
            conversion_sources = _conversion_sources(normalized, plan, options)
    except Exception:
        context.fail_stage(lowering_token)
        raise
    context.complete_stage(lowering_token)

    input_shapes: Dict[SymbolId, Tuple[int, ...]] = {}
    rhs_tensor_vars = lowered_cin.get_rhs_tensor_vars()
    for tensor_var in rhs_tensor_vars:
        assert tensor_var.shape is not None
        input_shapes[tensor_var.symbol_id] = tuple(tensor_var.shape)

    schedule: Optional[ScheduledLoopIR] = None
    assembly_decision: Optional[AssemblyDecision] = None
    program: LoopProgram = lowering.program
    if lowered_plan is not None:
        schedule_token = context.begin_stage(
            CompilerStageId.LOOPIR_SCHEDULE_APPLICATION,
            compile_options=options,
        )
        try:
            schedule = apply_schedule_plan(lowering.program, lowered_plan)
        except Exception:
            context.fail_stage(schedule_token)
            raise
        context.complete_stage(schedule_token)
        program = schedule.program

    # How a sparse result is assembled when nobody requested a strategy is the
    # last decision before target lowering.  It is made on the program that is
    # about to be lowered, because the strategies the target can emit depend on
    # the loops every other decision produced.  A result with dense levels only
    # assembles nothing a strategy describes, and the stage does not run for it.
    lowered_program = program
    if program.assembly is None and assembles_sparse_result(program):
        assembly_token = context.begin_stage(
            CompilerStageId.LOOPIR_ASSEMBLY_SELECTION,
            compile_options=options,
        )
        try:
            if lowered_plan is not None and lowered_plan.provenance == "auto":
                # An automatic plan is completed: the strategy becomes part of
                # the plan the kernel keeps.  The completed plan is a new plan,
                # verified like the one it completes, conversions included,
                # before it is applied.
                assert plan is not None and schedule is not None
                completed, assembly_decision = complete_automatic_plan(
                    schedule,
                    lowered_plan,
                    input_shapes=input_shapes,
                    result_shape=checked_result_shape,
                )
                verify_loop_plan(
                    normalized,
                    replace(plan, assembly=completed.assembly),
                    dense_results_any_order=True,
                )
                schedule = apply_schedule_plan(lowering.program, completed)
                program = lowered_program = schedule.program
            else:
                # A plan its caller wrote is left as written, and an
                # unscheduled program has no plan: the strategy is recorded on
                # the copy that is lowered and in the kernel's decision record.
                lowered_program, assembly_decision = choose_unrequested_assembly(
                    program,
                    input_shapes=input_shapes,
                    result_shape=checked_result_shape,
                )
        except Exception:
            context.fail_stage(assembly_token)
            raise
        context.complete_stage(assembly_token)

    program_text = print_program(program)
    program_dump = canonical_program_dump(program)

    llir_function = lower_loopir_to_llir(
        lowered_program,
        input_shapes=input_shapes,
        result_shape=checked_result_shape,
        compile_options=options,
        compilation_context=context,
    )

    from ...ops import _lower_generated_llir

    cpp_source = _lower_generated_llir(llir_function, options, context)
    if declared_result_order is None:
        declared_result_order = tuple(
            level.mode
            for decl in lowering.program.tensors
            if decl.symbol == lowering.result_symbol
            for level in decl.levels
        )
    return LoopIRCompiledKernel(
        lowering=lowering,
        program_text=program_text,
        program_dump=program_dump,
        llir_function=llir_function,
        cpp_source=cpp_source,
        schedule=schedule,
        request_dump=request_dump,
        request_identity=request_identity,
        conversions=() if plan is None else plan.conversions,
        conversion_sources=conversion_sources,
        result_mode_order=declared_result_order,
        assembly_decision=assembly_decision,
    )


def _conversion_sources(
    normalized: IndexStmt, plan: LoopPlan, options: CompileOptions
) -> Tuple[str, ...]:
    """The conversion kernel of each operand the plan converts, in plan order.

    ``normalized`` is the unconverted program with its runtime shapes bound:
    each kernel reads its operand as the caller stores it.
    """

    operands = {tensor.symbol_id: tensor for tensor in normalized.get_rhs_tensor_vars()}
    sources = []
    for conversion in plan.conversions:
        tensor = operands[conversion.operand_id]
        letters = _conversion_letters(tensor)
        if tensor.shape is None:
            _refuse_operand(
                "unsupported_operand_conversion",
                f"operand {tensor.name!r} has no bound shape to build its "
                "conversion kernel for",
            )
        sources.append(
            conversion_source(
                letters,
                conversion,
                tensor.dtype,
                tuple(tensor.shape),
                source_mode_order=tensor.mode_order,
                compile_options=options,
            )
        )
    return tuple(sources)


def compile_cin_via_loopir(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Sequence[int], torch.dtype]],
    *,
    compile_options: Optional[CompileOptions] = None,
    compilation_context: Optional[CompilationContext] = None,
) -> LoopIRCompiledKernel:
    """Lower one supported-family CIN program to C++ through LoopIR."""

    options = _resolve_options(compile_options, compilation_context)
    context = compilation_context
    if context is None:
        context = CompilationContext(options)
    plan: Optional[LoopPlan] = None
    if options.requested_schedule is not None:
        normalized, plan = _apply_requested_schedule(cin_stmt, options, context)
    else:
        normalized = normalize_cin(
            cin_stmt,
            compile_options=options,
            compilation_context=context,
        )
    return _compile_normalized_cin_via_loopir(
        normalized,
        result_shape,
        input_bindings,
        options=options,
        context=context,
        plan=plan,
    )


def legacy_generated_cpp(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Sequence[int], torch.dtype]],
    *,
    compile_options: Optional[CompileOptions] = None,
) -> str:
    """Generate the legacy pipeline's C++ for the same program, untouched.

    When the options carry a requested schedule, the legacy side is the
    scheduled route production uses: ``Scheduler.apply_schedule`` followed
    by the legacy lowering of the verified ``ScheduledCIN`` (which replays
    the legacy tree surgery inside the lowering adapter).
    """

    options = _resolve_options(compile_options)
    context = CompilationContext(options)
    lowering_stmt: Union[IndexStmt, ScheduledCIN]
    if options.requested_schedule is not None:
        from ..scheduler import Scheduler

        scheduled = Scheduler.apply_schedule(
            cin_stmt,
            options.requested_schedule,
            compile_options=options,
            compilation_context=context,
        )
        _bind_runtime_metadata(scheduled.normalized_cin, input_bindings, result_shape)
        lowering_stmt = scheduled
    else:
        normalized = normalize_cin(cin_stmt, compile_options=options)
        _bind_runtime_metadata(normalized, input_bindings, result_shape)
        lowering_stmt = normalized
    lowerer = CINLowerer(compile_options=options, compilation_context=context)
    lowered = lowerer._lower_owned_IndexStmt(lowering_stmt)

    from ...ops import _lower_generated_llir

    return _lower_generated_llir(lowered, options, context)


def compare_generated_sources(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Sequence[int], torch.dtype]],
    *,
    compile_options: Optional[CompileOptions] = None,
) -> ShadowSourceComparison:
    """Curated shadow comparison of generated sources; never on by default.

    Both pipelines consume independent detached normalizations of the same
    semantic CIN, so neither can observe the other's working state.
    """

    options = _resolve_options(compile_options)
    loopir_kernel = compile_cin_via_loopir(
        cin_stmt,
        result_shape,
        input_bindings,
        compile_options=options,
    )
    legacy_cpp = legacy_generated_cpp(
        cin_stmt,
        result_shape,
        input_bindings,
        compile_options=options,
    )
    return ShadowSourceComparison(
        loopir_cpp=loopir_kernel.cpp_source,
        legacy_cpp=legacy_cpp,
    )


def execute_cin_via_loopir(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    *args,
    compile_options: Optional[CompileOptions] = None,
    time_dict: Optional[dict] = None,
    _compilation_context: Optional[CompilationContext] = None,
):
    """Compile through LoopIR and execute on ``args`` (STensor inputs).

    Mirrors ``scorch.ops.lower_and_exec_cin`` for the migrated families:
    the same public argument marshalling and JIT build/caching helpers,
    with result wrapping derived from the verified LoopIR declaration.
    """

    from ...stensor import _wrap_generated_result
    from ...ops import (
        _checked_runtime_shape,
        _load_validated_prepared_kernel,
        _plan_mode_orders_to_loop_order,
        _prepare_generated_kernel_build,
        _snapshot_runtime_tensors,
    )

    options = _resolve_options(compile_options, _compilation_context)
    context = _compilation_context
    if context is None:
        context = CompilationContext(options)
    plan: Optional[LoopPlan] = None
    if options.requested_schedule is not None:
        normalized, plan = _apply_requested_schedule(cin_stmt, options, context)
    else:
        normalized = normalize_cin(
            cin_stmt,
            compile_options=options,
            compilation_context=context,
        )
    planning_token = context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=options,
    )
    try:
        result_shape = _checked_runtime_shape(result_shape, "result shape")
        runtime_args = _snapshot_runtime_tensors(
            args,
            expected_count=len(normalized.get_rhs_tensor_vars()),
        )
        _validate_runtime_formats(
            normalized,
            tuple(snapshot.tensor_format for snapshot in runtime_args),
        )
        if plan is not None:
            alignment_plan = _plan_mode_orders_to_planned_order(
                normalized,
                runtime_args,
                plan,
                dense_by_coordinate=True,
                converted_by_plan=True,
            )
            # A scheduled relayout executes with an independent schedule-free
            # context, but its success or failure still belongs to this
            # compilation's frontend binding stage.  Keep the parent token
            # active so elapsed time includes the prerequisite and a child
            # failure makes the caller-owned context terminal.
            runtime_args = _execute_mode_order_alignment(
                normalized,
                tuple(snapshot.tensor for snapshot in runtime_args),
                alignment_plan,
                options,
                context,
                runtime_snapshots=runtime_args,
            )
            _require_dense_operands_in_declared_order(normalized, runtime_args)
        else:
            alignment_plan = _plan_mode_orders_to_loop_order(
                normalized,
                runtime_args,
            )
    except Exception:
        context.fail_stage(planning_token)
        raise
    context.complete_stage(planning_token)
    if plan is None:
        # The unscheduled route shares its context with nested relayout
        # compilation, so its root frontend stage must finish first.
        try:
            runtime_args = _execute_mode_order_alignment(
                normalized,
                tuple(snapshot.tensor for snapshot in runtime_args),
                alignment_plan,
                options,
                context,
                runtime_snapshots=runtime_args,
            )
        except Exception:
            if context._failed_stage_id is None:
                relayout_failure_token = context.begin_stage(
                    CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
                    compile_options=options,
                )
                context.fail_stage(relayout_failure_token)
            raise
    input_bindings = tuple(
        (_declared_order_shape(snapshot, operand), snapshot.dtype)
        for snapshot, operand in zip(runtime_args, normalized.get_rhs_tensor_vars())
    )
    kernel = _compile_normalized_cin_via_loopir(
        normalized,
        result_shape,
        input_bindings,
        options=options,
        context=context,
        input_formats=tuple(snapshot.tensor_format for snapshot in runtime_args),
        plan=plan,
    )
    _require_converted_or_stored_as_declared(
        normalized, runtime_args, kernel.conversions
    )
    # The shipped dispatch's result bound, after lowering (which says which levels the
    # kernel checks itself) and before the build: the same calls refused, with the same
    # code, before anything is allocated.
    _refuse_result_past_bound(normalized, kernel, result_shape, runtime_args)
    header_cpp_code = options.build.preamble_source
    prepared_build = _prepare_generated_kernel_build(
        header_cpp_code,
        kernel.cpp_source,
        options,
        context,
    )
    module = _load_validated_prepared_kernel(prepared_build)

    if kernel.conversions:
        start_time = time.time()
        runtime_args, sources_run = _run_conversions(
            kernel, normalized, runtime_args, options
        )
        if sources_run != kernel.conversion_sources:
            kernel = replace(kernel, conversion_sources=sources_run)
        if time_dict is not None:
            time_dict["conversion_time"] = time.time() - start_time

    module_args: List[object] = [result_shape]
    for snapshot in runtime_args:
        module_args.append(snapshot.shape)
        module_args.append(snapshot.native_mode_indices)
        module_args.append(snapshot.values)

    start_time = time.time()
    try:
        result_cpp = module.evaluate(*module_args)
    except RuntimeError as error:
        raise_if_native_coded(error)
        raise
    end_time = time.time()
    if time_dict is not None:
        time_dict["eval_time"] = end_time - start_time

    result_decl = next(
        decl
        for decl in kernel.lowering.program.tensors
        if decl.symbol == kernel.lowering.result_symbol
    )
    result_format = "".join(
        _RESULT_LEVEL_LETTERS[level.kind] for level in result_decl.levels
    )
    # The declared order: a compressed result declared in a permuted order is
    # lowered in identity-order form, and its arrays are in the declared one.
    result_mode_order = kernel.result_mode_order
    # The kernel-result wrapper every shipped dispatch path uses: the index is
    # trusted, so with kernel-result validation on it is walked and a malformed
    # one is an error.  The public constructor would instead sort and sum
    # unordered or repeated coordinates, repairing exactly the lowering defect
    # an execution check exists to catch.
    result = _wrap_generated_result(
        tuple(result_shape),
        result_format,
        result_cpp,
        mode_order=result_mode_order,
    )
    return result, kernel


_LEVEL_TYPES = {
    LoopIRLevelKind.DENSE: LevelType.DENSE,
    LoopIRLevelKind.COMPRESSED: LevelType.COMPRESSED,
    LoopIRLevelKind.COORDINATE: LevelType.COORDINATE,
}


def _sum_of_products(
    normalized: IndexStmt,
) -> Optional[Tuple[TensorAccess, List[List[TensorAccess]]]]:
    """The access the program's one assignment writes and its right-hand side as a sum
    of products of input accesses, or ``None`` when the program is not that: more or
    fewer than one assignment, a workspace, or an operation other than ``+``, ``-`` and
    ``*`` over tensor accesses.

    A product's stored coordinates lie in the join of its factors' stored entries, and a
    sum's in the union of its terms', which is what the result bound reads from them.
    """

    assignments: List[TensorAssign] = []
    wheres: List[Where] = []

    class _Collector(CINVisitorAccept):
        def visit_TensorAssign(self, node: TensorAssign) -> None:
            assignments.append(node)

        def visit_Where(self, node: Where) -> None:
            wheres.append(node)

    _Collector().visit(normalized)
    if len(assignments) != 1 or wheres:
        return None

    def expand(expression: object) -> Optional[List[List[TensorAccess]]]:
        if isinstance(expression, TensorAccess):
            if isinstance(expression.get_tensor(), Workspace):
                return None
            return [[expression]]
        if not isinstance(expression, BinaryOp):
            return None
        left, right = expand(expression.left), expand(expression.right)
        if left is None or right is None:
            return None
        if expression.op is Operation.MUL:
            return [a + b for a in left for b in right]
        if expression.op in (Operation.ADD, Operation.SUB):
            return left + right
        return None

    terms = expand(assignments[0].get_rhs())
    return None if terms is None else (assignments[0].get_lhs(), terms)


def _refuse_result_past_bound(
    normalized: IndexStmt,
    kernel: LoopIRCompiledKernel,
    result_shape: Sequence[int],
    runtime_args: Sequence[Any],
) -> None:
    """Refuse, as the shipped einsum dispatch does (scorch/result_bound.py), a result
    level the kernel assembles one entry at a time and does not check itself whose bound
    -- from the shapes and stored counts, then from the operands' coordinates -- passes
    the 32-bit position bound.  A program that is not a sum of products of its inputs
    is left to its kernel, as it was before this check."""

    from ...result_bound import entry_bound, level_totals_in_source

    found = _sum_of_products(normalized)
    if found is None:
        return
    result_access, terms = found
    accesses = normalized.get_rhs_tensor_accesses()
    position = {id(access): k for k, access in enumerate(accesses)}
    if any(id(factor) not in position for term in terms for factor in term):
        return
    result_decl = next(
        decl
        for decl in kernel.lowering.program.tensors
        if decl.symbol == kernel.lowering.result_symbol
    )
    order = kernel.result_mode_order or tuple(range(len(result_decl.levels)))
    result_names = [var.name for var in result_access.get_index_vars()]
    bound = entry_bound(
        "execute",
        [result_names[mode] for mode in order],
        [int(result_shape[mode]) for mode in order],
        [_LEVEL_TYPES[level.kind] for level in result_decl.levels],
        [
            [
                (
                    position[id(factor)],
                    [var.name for var in factor.get_index_vars()],
                )
                for factor in term
            ]
            for term in terms
        ],
        operand_names=[
            [var.name for var in access.get_index_vars()] for access in accesses
        ],
    )
    if bound is None:
        return
    bound = bound.with_checked(level_totals_in_source(kernel.cpp_source))
    bound.refuse_past_limit(
        [snapshot.values.numel() for snapshot in runtime_args],
        [getattr(snapshot, "tensor", None) for snapshot in runtime_args],
    )


def _run_conversions(
    kernel: LoopIRCompiledKernel,
    normalized: IndexStmt,
    runtime_args: tuple,
    options: CompileOptions,
) -> Tuple[tuple, Tuple[str, ...]]:
    """Convert each operand the plan converts.

    Returns the operands' snapshots and the source of each conversion kernel
    that ran.  A conversion kernel reads its operand as the caller stores it
    and returns the converted tensor, which is wrapped like any kernel
    result: with kernel-result validation on, its index arrays are walked.
    The compiled program's kernel (``kernel.conversion_sources``) reads the
    mode order the program declares; an operand stored in another mode order
    (the alignment leaves a converted operand as stored) is converted by a
    kernel built for its stored order, the same copy reading its levels in
    that order, so no relayout runs before it.  The kernels are built
    without the request's schedule, each under its own compilation context,
    as they were lowered.
    """

    from ...ops import (
        _load_validated_prepared_kernel,
        _prepare_generated_kernel_build,
        _snapshot_runtime_tensors,
    )
    from ...stensor import _wrap_generated_result

    build_options = options
    if options.requested_schedule is not None:
        build_options = replace(options, requested_schedule=None)
    operands = normalized.get_rhs_tensor_vars()
    position_of = {tensor.symbol_id: n for n, tensor in enumerate(operands)}
    tensors = [snapshot.tensor for snapshot in runtime_args]
    sources_run: List[str] = []
    for conversion, source in zip(kernel.conversions, kernel.conversion_sources):
        position = position_of[conversion.operand_id]
        snapshot = runtime_args[position]
        operand = operands[position]
        rank = len(snapshot.shape)
        stored, declared = _stored_and_declared_orders(snapshot, operand)
        if stored != declared:
            source = conversion_source(
                _conversion_letters(operand),
                conversion,
                snapshot.dtype,
                snapshot.shape,
                source_mode_order=stored,
                compile_options=build_options,
            )
        sources_run.append(source)
        module = _load_validated_prepared_kernel(
            _prepare_generated_kernel_build(
                build_options.build.preamble_source,
                source,
                build_options,
                CompilationContext(build_options),
            )
        )
        shape = converted_shape(snapshot.shape, conversion, stored)
        try:
            converted = module.evaluate(
                list(shape),
                snapshot.shape,
                snapshot.native_mode_indices,
                snapshot.values,
            )
        except RuntimeError as error:
            raise_if_native_coded(error)
            raise
        tensors[position] = _wrap_generated_result(
            shape,
            conversion.format,
            converted,
            mode_order=tuple(range(rank)),
        )
    snapshots = _snapshot_runtime_tensors(
        tuple(tensors), expected_count=len(runtime_args)
    )
    return snapshots, tuple(sources_run)


def _execute_legacy_scheduled(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    *args,
    compile_options: CompileOptions,
):
    """Execute the legacy scheduled route on ``args`` (dense outputs only).

    The untouched low-level entry rejects requested schedules, so scheduled
    shadow execution drives the same production components directly:
    ``Scheduler.apply_schedule``, the legacy lowering of the verified
    ``ScheduledCIN``, and the shared JIT build/cache/execution helpers.
    Runtime mode-order prerequisites are aligned to the verified plan, and
    the dense result wrapper is derived from the normalized result rank.
    """

    from ...stensor import _wrap_generated_result
    from ...ops import (
        _checked_runtime_shape,
        _load_validated_prepared_kernel,
        _lower_generated_llir,
        _prepare_generated_kernel_build,
        _snapshot_runtime_tensors,
    )
    from ..scheduler import Scheduler

    options = compile_options
    assert options.requested_schedule is not None
    context = CompilationContext(options)
    scheduled = Scheduler.apply_schedule(
        cin_stmt,
        options.requested_schedule,
        compile_options=options,
        compilation_context=context,
    )
    result_shape = _checked_runtime_shape(result_shape, "result shape")
    runtime_args = _snapshot_runtime_tensors(
        args,
        expected_count=len(scheduled.normalized_cin.get_rhs_tensor_vars()),
    )
    _validate_runtime_formats(
        scheduled.normalized_cin,
        tuple(snapshot.tensor_format for snapshot in runtime_args),
    )
    alignment_plan = _plan_mode_orders_to_planned_order(
        scheduled.normalized_cin,
        runtime_args,
        scheduled.verified_loop_plan,
    )
    runtime_args = _execute_mode_order_alignment(
        scheduled.normalized_cin,
        tuple(snapshot.tensor for snapshot in runtime_args),
        alignment_plan,
        options,
        context,
        runtime_snapshots=runtime_args,
    )
    _bind_runtime_metadata(
        scheduled.normalized_cin,
        tuple((snapshot.shape, snapshot.dtype) for snapshot in runtime_args),
        result_shape,
        tuple(snapshot.tensor_format for snapshot in runtime_args),
    )
    lowerer = CINLowerer(compile_options=options, compilation_context=context)
    lowered = lowerer._lower_owned_IndexStmt(scheduled)
    cpp_source = _lower_generated_llir(lowered, options, context)
    prepared_build = _prepare_generated_kernel_build(
        options.build.preamble_source,
        cpp_source,
        options,
        context,
    )
    module = _load_validated_prepared_kernel(prepared_build)

    module_args: List[object] = [result_shape]
    for snapshot in runtime_args:
        module_args.append(snapshot.shape)
        module_args.append(snapshot.native_mode_indices)
        module_args.append(snapshot.values)
    try:
        result_cpp = module.evaluate(*module_args)
    except RuntimeError as error:
        raise_if_native_coded(error)
        raise
    result_vars = scheduled.normalized_cin.get_result_tensor_vars()
    if len(result_vars) != 1 or result_vars[0].format is None:
        raise CompileSpecError(
            "legacy scheduled shadow execution requires one formatted result"
        )
    result_format = result_vars[0].format
    if not result_format.is_dense():
        raise CompileSpecError(
            "legacy scheduled shadow execution requires a dense result"
        )
    return _wrap_generated_result(
        tuple(result_shape),
        result_format,
        result_cpp,
        mode_order=(
            None
            if result_vars[0].mode_order is None
            else tuple(result_vars[0].mode_order)
        ),
    )


def execute_shadow(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    *args,
    compile_options: Optional[CompileOptions] = None,
):
    """Execute both pipelines on ``args`` for curated differential tests.

    Returns ``(loopir_result, legacy_result, source_comparison)``.  The
    legacy execution uses the untouched low-level entry (or, when the
    options request a schedule, the legacy scheduled route through
    ``Scheduler.apply_schedule``); both wrappers support dense outputs.
    Sparse-output comparisons must use source parity plus direct
    LoopIR/PyTorch/oracle execution instead.
    """

    from ...ops import (
        _checked_runtime_shape,
        _plan_mode_orders_to_loop_order,
        _snapshot_runtime_tensors,
        lower_and_exec_cin,
    )

    options = _resolve_options(compile_options)
    shadow_plan: Optional[LoopPlan] = None
    downstream_options = options
    if options.requested_schedule is not None:
        normalized, shadow_plan = _apply_requested_schedule(
            cin_stmt,
            options,
            CompilationContext(options),
        )
        # Freeze any policy-selected shorthand (notably
        # ``Schedule.loop_order=None``) to the verified plan before the
        # comparison invokes either pipeline again.  Runtime alignment below
        # may update compiler-owned mode-order metadata; it must not cause a
        # second policy selection to reinterpret the same shadow request.
        from ..scheduler import materialize_legacy_schedule

        canonical_schedule, _, _, _ = materialize_legacy_schedule(
            normalized,
            shadow_plan,
        )
        if canonical_schedule != options.requested_schedule:
            downstream_options = replace(
                options,
                requested_schedule=canonical_schedule,
            )
    else:
        normalized = normalize_cin(cin_stmt, compile_options=options)
    result_shape = _checked_runtime_shape(result_shape, "result shape")
    runtime_args = _snapshot_runtime_tensors(
        args,
        expected_count=len(normalized.get_rhs_tensor_vars()),
    )
    if any(
        tensor_var.format is None or not tensor_var.format.is_dense()
        for tensor_var in normalized.get_result_tensor_vars()
    ):
        raise CompileSpecError(
            "execute_shadow requires dense outputs because the legacy "
            "low-level result wrapper does not preserve sparse indices"
        )
    _validate_runtime_formats(
        normalized,
        tuple(snapshot.tensor_format for snapshot in runtime_args),
    )
    if shadow_plan is None:
        alignment_plan = _plan_mode_orders_to_loop_order(normalized, runtime_args)
    else:
        alignment_plan = _plan_mode_orders_to_planned_order(
            normalized,
            runtime_args,
            shadow_plan,
        )
    # Align once before source comparison so compile-only bindings describe
    # logical storage, then both execution routes independently observe the
    # same already-aligned inputs.
    runtime_args = _execute_mode_order_alignment(
        normalized,
        tuple(snapshot.tensor for snapshot in runtime_args),
        alignment_plan,
        downstream_options,
        None,
        runtime_snapshots=runtime_args,
    )

    comparison = compare_generated_sources(
        normalized,
        result_shape,
        tuple((snapshot.shape, snapshot.dtype) for snapshot in runtime_args),
        compile_options=downstream_options,
    )
    loopir_result, _ = execute_cin_via_loopir(
        copy.deepcopy(normalized),
        result_shape,
        *(snapshot.tensor for snapshot in runtime_args),
        compile_options=downstream_options,
    )
    if downstream_options.requested_schedule is not None:
        legacy_result = _execute_legacy_scheduled(
            copy.deepcopy(normalized),
            result_shape,
            *(snapshot.tensor for snapshot in runtime_args),
            compile_options=downstream_options,
        )
    else:
        legacy_result = lower_and_exec_cin(
            copy.deepcopy(normalized),
            result_shape,
            *(snapshot.tensor for snapshot in runtime_args),
            _compile_options=downstream_options,
        )
    return loopir_result, legacy_result, comparison
