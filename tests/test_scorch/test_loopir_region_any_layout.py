"""Sparse workspace regions drained into any dense/compressed result layout.

The multi-compressed assembly target lowers a :class:`SparseWorkspaceRegion` as
a leaf below prefix loops, for a result of any layout.  Two routes reach it:

* the front door: an ADD reduction whose result has a dense level below a
  compressed one, or a dense level among the levels bound below the outermost
  reduction.  The shipped scheduler's workspace fact -- ``dense=True`` for one
  dense drained level -- becomes a region, and the prefix levels are assembled
  around it with their dense blocks;
* a region at the program root whose key loops are not nested in key order: a
  transposed copy, built here by hand from the LoopIR of an in-order copy.  It
  is the program a conversion of an operand or result into another storage
  order compiles to.

Every result is judged from its own arrays against float64 ``torch.einsum``:
its values, and stored coordinates equal to what the operands' stored entries
imply (a compressed coordinate is stored exactly when an entry lies under it).
The inputs are the high-rank test's four: every entry stored, a random third,
the same with empty first and last slices, and the same with coordinates that
have nothing under them.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import List, Sequence, Tuple

import pytest
import torch

from scorch.compiler.cin import ForAll, IndexVar, TensorAssign, TensorVar
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.compilation_context import CompilationContext
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir
from scorch.compiler.loopir.lower_llir import lower_loopir_to_llir
from scorch.compiler.loopir.nodes import AppendEntry, LevelKind, ReduceOp
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.compiler.loopir.printer import print_program
from scorch.compiler.loopir.schedule_passes import _decompose_chain, _wrap_loops
from scorch.compiler.loopir.verifier import verify_program
from scorch.compiler.scheduler import Schedule
from scorch.ops import (
    _load_validated_prepared_kernel,
    _lower_generated_llir,
    _prepare_generated_kernel_build,
    _snapshot_runtime_tensors,
)
from tests.test_scorch.test_high_rank_programs import (
    DTYPE,
    Program,
    _cin,
    _inputs,
    _judge,
    _level_arrays,
    _operand,
    _options,
)

VARIANTS = ("full", "sparse", "gap", "hollow")


# -- the front door ----------------------------------------------------------------


def _front_door_programs() -> List[Program]:
    """Reductions into layouts with a dense level below a compressed one, or a
    dense level that is drained, each on a regular and a degenerate extent."""

    cells = [
        ("ijk->ik", ("sss", "dss", "sds", "ssd", "dsd"), ("sd", "ds", "ss")),
        ("ijk->jk", ("sss", "dsd", "ssd"), ("sd", "ds", "ss")),
        ("ijk->ij", ("sss", "ssd"), ("sd",)),
        (
            "ijk,kl->ijl",
            ("sss,dd", "dsd,ds", "ssd,sd"),
            ("ssd", "dsd", "sds", "sdd", "dds", "dss"),
        ),
        ("ijkl->ik", ("ssss", "dsds"), ("sd", "ds")),
        ("ijkl->jl", ("ssss", "sdss"), ("sd", "ds")),
        ("ijkl->ijl", ("ssss", "ddsd"), ("sds", "dsd", "ssd", "sdd")),
    ]
    programs = []
    for expr, operand_formats, result_formats in cells:
        letters = "".join(dict.fromkeys(expr.split("->")[0].replace(",", "")))
        for operands, result in itertools.product(operand_formats, result_formats):
            for extents in (
                tuple(zip(letters, (3, 4, 2, 3, 2)[: len(letters)])),
                tuple(zip(letters, (2, 3, 0, 3, 2)[: len(letters)])),
            ):
                programs.append(
                    Program(
                        family="region",
                        expr=expr,
                        operand_formats=tuple(operands.split(",")),
                        result_format=result,
                        extents=extents,
                    )
                )
    return programs


FRONT_DOOR = _front_door_programs()


def _run_front_door(program: Program, operands, blocking: bool):
    options = replace(_options(blocking), requested_schedule=Schedule())
    result, kernel = execute_cin_via_loopir(
        _cin(program),
        program.shape(program.output),
        *operands,
        compile_options=options,
    )
    return result, kernel


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("program", FRONT_DOOR, ids=[p.name for p in FRONT_DOOR])
def test_reductions_into_any_layout_match_pytorch(program, blocking):
    for seed, variant in enumerate(VARIANTS):
        operands, reference, structure = _inputs(program, variant, seed)
        result, _ = _run_front_door(program, operands, blocking)
        try:
            _judge(program, result, reference, structure)
        except AssertionError as error:
            raise AssertionError(f"{variant} input: {error}") from error


def test_the_front_door_reaches_the_region_leaf():
    """The table exercises the region leaf, not only scalar accumulation.

    A reduction whose key is empty takes the bound-prefix form instead, which
    other tests cover; this asserts the table keeps programs with a key.
    """

    with_region = 0
    for program in FRONT_DOOR:
        if 0 in dict(program.extents).values():
            continue
        operands, _, _ = _inputs(program, "sparse", 0)
        _, kernel = _run_front_door(program, operands, False)
        if "wksp.sort()" in kernel.cpp_source:
            with_region += 1
    assert with_region >= 20, with_region


# -- a region at the root with its key loops out of key order -----------------------


def _transposed_copy(
    operand_format: str, result_format: str, permutation: Sequence[int]
) -> Tuple[object, object, Tuple[int, ...]]:
    """``C[x_p0, x_p1, ...] = A[x_0, x_1, ...]`` as one region at the root.

    The LoopIR of the in-order copy supplies the loops over A in its storage
    order and the value read; its append leaf becomes an insertion at the
    permuted coordinates, and a drain appends into C, whose levels follow the
    permuted modes.  Returns the program, A's symbol and C's symbol.
    """

    rank = len(operand_format)
    letters = "ijkl"[:rank]
    variables = {letter: IndexVar(letter) for letter in letters}
    source = TensorVar("A", fmt=operand_format, dtype=DTYPE)[
        tuple(variables[letter] for letter in letters)
    ]
    copy = TensorVar("C", fmt="s" * rank, dtype=DTYPE)[
        tuple(variables[letter] for letter in letters)
    ]
    statement = TensorAssign(copy, source)
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    normalized = normalize_cin(
        statement, compile_options=CompileOptions.from_environment()
    )
    program = lower_normalized_cin_to_loopir(normalized).program
    loops, leaf = _decompose_chain(program)
    assert type(leaf) is AppendEntry
    builder = LoopIRBuilder.resuming(program)
    result_decl = next(d for d in program.tensors if d.symbol == leaf.tensor)
    source_decl = next(d for d in program.tensors if d.symbol != leaf.tensor)
    dims = tuple(result_decl.dimensions[mode] for mode in permutation)
    workspace = builder.new_workspace_id()
    insert = builder.sparse_workspace_insert(
        workspace,
        tuple(leaf.coords[mode] for mode in permutation),
        ReduceOp.ADD,
        leaf.value,
    )
    drained = tuple(builder.new_index_id() for _ in range(rank))
    append = builder.append_entry(
        leaf.tensor,
        tuple(builder.index_value(index) for index in drained),
        builder.sparse_workspace_value(workspace),
    )
    region = builder.sparse_workspace_region(
        builder.sparse_workspace_decl(workspace, "wksp", result_decl.dtype, dims),
        _wrap_loops(builder, loops, builder.block((insert,))),
        builder.block(
            (
                builder.sparse_workspace_drain_for(
                    workspace, drained, builder.block((append,))
                ),
            )
        ),
    )
    levels = tuple(
        builder.level(LevelKind.DENSE if kind == "d" else LevelKind.COMPRESSED, mode)
        for mode, kind in enumerate(result_format)
    )
    transposed = builder.program(
        program.dimensions,
        (source_decl, replace(result_decl, dimensions=dims, levels=levels)),
        program.inputs,
        program.outputs,
        builder.block((region,)),
    )
    verify_program(transposed)
    return transposed, source_decl.symbol, leaf.tensor


def _execute_program(program, operand, input_symbol, result_shape):
    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    function = lower_loopir_to_llir(
        program,
        input_shapes={input_symbol: tuple(operand.shape)},
        result_shape=tuple(result_shape),
        compile_options=options,
        compilation_context=context,
    )
    source = _lower_generated_llir(function, options, context)
    module = _load_validated_prepared_kernel(
        _prepare_generated_kernel_build(
            options.build.preamble_source, source, options, context
        )
    )
    (snapshot,) = _snapshot_runtime_tensors((operand,), expected_count=1)
    return (
        module.evaluate(
            list(result_shape),
            snapshot.shape,
            snapshot.native_mode_indices,
            snapshot.values,
        ),
        source,
    )


TRANSPOSES = [
    (operand, result, permutation)
    for operand in ("sss", "dsd", "sds", "ssd")
    for result in ("sss", "dss", "sds", "ssd", "dds", "dsd", "sdd")
    for permutation in ((2, 1, 0), (1, 2, 0))
] + [
    (operand, result, (3, 1, 0, 2))
    for operand in ("ssss", "dsds")
    for result in ("ssss", "sdsd", "dsds", "sddd")
]


@pytest.mark.parametrize(
    "operand_format, result_format, permutation",
    TRANSPOSES,
    ids=[f"{o}_to_{r}_{''.join(map(str, p))}" for o, r, p in TRANSPOSES],
)
def test_a_transposed_copy_at_the_root_matches_pytorch(
    operand_format, result_format, permutation
):
    program, input_symbol, _ = _transposed_copy(
        operand_format, result_format, permutation
    )
    assert "wksp" in print_program(program)
    rank = len(operand_format)
    shape = (3, 4, 2, 3)[:rank]
    result_shape = tuple(shape[mode] for mode in permutation)
    for seed, variant in enumerate(VARIANTS):
        generator = torch.Generator().manual_seed(seed)
        values = torch.rand(shape, generator=generator, dtype=DTYPE) + 0.5
        keep = (
            torch.ones(shape, dtype=torch.bool)
            if variant == "full"
            else torch.rand(shape, generator=generator) < 0.35
        )
        if variant in ("gap", "hollow"):
            keep[0] = False
            keep[-1] = False
        ghosts = None
        if variant == "hollow":
            ghosts = (torch.rand(shape, generator=generator) < 0.35) & ~keep
            ghosts[0] = True
        operand, indicator = _operand(values * keep, keep, operand_format, "A", ghosts)
        result, source = _execute_program(program, operand, input_symbol, result_shape)
        expected_leaves, expected = _level_arrays(
            indicator.permute(permutation), result_format
        )
        actual = [
            None if kind == "d" else tuple(t.reshape(-1).tolist() for t in arrays)
            for kind, arrays in zip(result_format, result.storage.index.mode_indices)
        ]
        expected = [None if e is None else (e[0], e[1]) for e in expected]
        assert actual == [
            None if e is None else tuple(e) for e in expected
        ], f"{variant} input: stored coordinates differ"
        reference = (values * keep).permute(permutation)
        expected_values = torch.tensor(
            [float(reference[leaf]) for leaf in expected_leaves], dtype=DTYPE
        )
        torch.testing.assert_close(
            result.storage.value.to(DTYPE), expected_values, rtol=0, atol=0
        )
        assert "wksp.sort()" in source
