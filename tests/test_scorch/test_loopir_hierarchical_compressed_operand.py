"""Operands with a compressed level under a compressed level, generic target.

The generic target -- the one that lowers dense results -- used to refuse any
input with more than one compressed level ("hierarchical compressed descent is
outside the migrated families"), so ``sum_j A[i, j]`` with ``A`` stored ``ss``
had no new-compiler kernel although the shipped compiler emits one.  The
refusal was a migration-scope limit, not a recorded defect: each compressed
level is descended by its own single-cursor loop whose parent is the position
of the same tensor one level up, which is the chain ``verify_program`` already
enforces, so a compressed level under a compressed level reads ``pos[parent]``
exactly as one under a dense level does.

What is locked here:

* every rank-2/3 operand with at least two compressed levels, reduced or copied
  into a dense result, emits the legacy kernel byte for byte on both automatic
  arms -- except a copy written ``C += A``, where the new compiler keeps the
  written ``+=`` and legacy writes ``=``.  That difference predates this change
  (``ds ij->ij`` has it too) and is equivalent on every accepted input, because
  the generated-kernel validator rejects a repeated coordinate inside a
  compressed level before the kernel runs;
* one declared arm difference: ``ssd ijk->k`` on the regblock arm stops at the
  dense-workspace copy-out rule (a rank-1 result has no parent level), exactly
  as the single-compressed ``dsd ijk->k`` already does there, and legacy's
  comparand fails on it too;
* every admitted cell executes correctly against a float64 reference.
"""

import difflib
from dataclasses import replace
from itertools import combinations, product

import pytest
import torch

from scorch.compiler.cin import BinaryOp as CINBinaryOp
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor

ATOL = RTOL = 1e-3
SHAPES = {2: (4, 5), 3: (3, 4, 5)}
ARMS = [False, True]

# (operand, result, operand format, arm) -> the one declared refusal code.
DECLARED_REFUSALS = {("ijk", "k", "ssd", True): "unsupported_program_shape"}


def _options(regblock_enabled, *, jit=False):
    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(regblock_enabled), requested_schedule=Schedule()
    )


def _cells():
    cells = []
    for rank in (2, 3):
        operand = "ijk"[:rank]
        for operand_fmt in ("".join(f) for f in product("ds", repeat=rank)):
            if operand_fmt.count("s") < 2:
                continue
            for size in range(1, rank + 1):
                for result in ("".join(r) for r in combinations(operand, size)):
                    ops = [None, Operation.ADD] if size == rank else [Operation.ADD]
                    for op in ops:
                        cells.append((operand, result, operand_fmt, op))
    return cells


CELLS = _cells()


def _cell_id(cell):
    operand, result, operand_fmt, op = cell
    return f"{operand_fmt} {operand}->{result}{' +=' if op else ' ='}"


def _cin(operand, result, operand_fmt, op):
    ivars = {name: IndexVar(name) for name in operand}
    source = TensorVar("A", fmt=operand_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in operand)
    ]
    target = TensorVar("C", fmt="d" * len(result), dtype=torch.float32)[
        tuple(ivars[name] for name in result)
    ]
    stmt = TensorAssign(target, source, op=op)
    for name in reversed(operand):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def _result_shape(operand, result):
    shape = SHAPES[len(operand)]
    return tuple(shape[operand.index(name)] for name in result)


@pytest.mark.parametrize("arm", ARMS, ids=["direct", "regblock"])
@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_every_cell_emits_the_legacy_kernel_or_its_declared_refusal(cell, arm):
    operand, result, operand_fmt, op = cell
    declared = DECLARED_REFUSALS.get((operand, result, operand_fmt, arm))
    try:
        comparison = compare_generated_sources(
            _cin(operand, result, operand_fmt, op),
            _result_shape(operand, result),
            ((SHAPES[len(operand)], torch.float32),),
            compile_options=_options(arm),
        )
    except (LoopIRLoweringError, LoopIRTargetError) as refusal:
        assert refusal.defect.code == declared, refusal.defect
        return
    assert declared is None, "a declared refusal now compiles; update the record"
    if comparison.identical:
        return
    # The one admitted difference: an ADD copy keeps its written operator.
    assert op is Operation.ADD and len(result) == len(operand)
    changed = [
        line
        for line in difflib.ndiff(
            comparison.legacy_cpp.splitlines(), comparison.loopir_cpp.splitlines()
        )
        if line[:2] in ("- ", "+ ")
    ]
    assert len(changed) == 2, changed
    removed, added = changed
    assert removed.startswith("- ") and added.startswith("+ ")
    assert removed[2:].replace("] = ", "] += ") == added[2:]


def _dense(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
    dense = values * (torch.rand(shape, generator=generator) < 0.4)
    dense[0] = 0.0  # an empty leading slice: a root coordinate stores nothing
    return dense.to(torch.float32)


def _sparse(dense, fmt):
    tensor = STensor.from_torch(dense.clone(), "A")
    tensor.to_sparse(fmt)
    return tensor


@pytest.mark.parametrize("arm", ARMS, ids=["direct", "regblock"])
@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_every_admitted_cell_executes_correctly(cell, arm):
    operand, result, operand_fmt, op = cell
    if (operand, result, operand_fmt, arm) in DECLARED_REFUSALS:
        pytest.skip("declared refusal; locked by the emission test")
    dense = _dense(SHAPES[len(operand)], seed=CELLS.index(cell))
    got, _ = execute_cin_via_loopir(
        _cin(operand, result, operand_fmt, op),
        _result_shape(operand, result),
        _sparse(dense, operand_fmt),
        compile_options=_options(arm, jit=True),
    )
    torch.testing.assert_close(
        got.to_torch(in_place=False).to(torch.float64),
        torch.einsum(f"{operand}->{result}", dense.to(torch.float64)),
        atol=ATOL,
        rtol=RTOL,
    )


def _spmm_cin():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="ss", dtype=torch.float32)[i, j]
    b = TensorVar("B", fmt="dd", dtype=torch.float32)[j, k]
    c = TensorVar("C", fmt="dd", dtype=torch.float32)[i, k]
    stmt = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    return ForAll(i, ForAll(j, ForAll(k, stmt)))


def _spmv_cin():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ss", dtype=torch.float32)[i, j]
    x = TensorVar("x", fmt="d", dtype=torch.float32)[j]
    y = TensorVar("y", fmt="d", dtype=torch.float32)[i]
    return ForAll(
        i, ForAll(j, TensorAssign(y, CINBinaryOp(Operation.MUL, a, x), op=Operation.ADD))
    )


@pytest.mark.parametrize("arm", ARMS, ids=["direct", "regblock"])
@pytest.mark.parametrize("width", [1, 4, 7], ids=lambda w: f"k={w}")
def test_a_dcsr_operand_multiplies_a_dense_matrix(arm, width):
    dense = _dense((6, 5), seed=width)
    other = torch.rand((5, width), generator=torch.Generator().manual_seed(3))
    comparison = compare_generated_sources(
        _spmm_cin(),
        (6, width),
        (((6, 5), torch.float32), ((5, width), torch.float32)),
        compile_options=_options(arm),
    )
    assert comparison.identical
    got, _ = execute_cin_via_loopir(
        _spmm_cin(),
        (6, width),
        _sparse(dense, "ss"),
        STensor.from_torch(other.clone(), "B"),
        compile_options=_options(arm, jit=True),
    )
    torch.testing.assert_close(
        got.to_torch(in_place=False).to(torch.float64),
        dense.to(torch.float64) @ other.to(torch.float64),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("arm", ARMS, ids=["direct", "regblock"])
def test_a_dcsr_operand_multiplies_a_dense_vector(arm):
    dense = _dense((6, 5), seed=11)
    vector = torch.rand((5,), generator=torch.Generator().manual_seed(4))
    comparison = compare_generated_sources(
        _spmv_cin(),
        (6,),
        (((6, 5), torch.float32), ((5,), torch.float32)),
        compile_options=_options(arm),
    )
    assert comparison.identical
    got, _ = execute_cin_via_loopir(
        _spmv_cin(),
        (6,),
        _sparse(dense, "ss"),
        STensor.from_torch(vector.clone(), "x"),
        compile_options=_options(arm, jit=True),
    )
    torch.testing.assert_close(
        got.to_torch(in_place=False).to(torch.float64),
        dense.to(torch.float64) @ vector.to(torch.float64),
        atol=ATOL,
        rtol=RTOL,
    )
