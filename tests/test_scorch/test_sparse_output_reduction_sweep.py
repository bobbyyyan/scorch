"""Every order-preserving reduction into a sparse result, through both compilers.

A sparse result is appended to in iteration order, so the shipped compiler must
drain each result coordinate once, after every contribution to it has been
summed, and must assemble every compressed level above the drain.  Two defects
broke that for any nest with more than one summed-over loop or more than two
compressed result levels: the workspace wrapped only the innermost reduction
loop (so an outer reduction loop drained the same coordinates again), and the
nested drain closed only the level directly above it.  The cells below are the
whole family at rank three and four, not the shapes that were found broken.

For each cell:

* the shipped compiler (public ``scorch.einsum``) must return the PyTorch
  result, in storage that passes validation (the suite validates every kernel
  result);
* the refactor's LoopIR route must return the same result, or refuse with a
  structured defect code -- never a wrong answer, never an unclassified error;
* where both return, their sparse storage must be identical.
"""

from dataclasses import replace
from itertools import combinations, product

import pytest
import torch

import scorch
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor

ATOL = RTOL = 1e-3

SHAPES = {3: (3, 4, 5), 4: (2, 3, 4, 5)}
OPERAND_FORMATS = {
    3: ["".join(levels) for levels in product("ds", repeat=3)],
    4: ["ssss", "dsss", "sdss", "ssds"],
}


def _result_formats(rank):
    formats = ["s" * rank]
    if rank > 1:
        formats.append("d" + "s" * (rank - 1))
    return formats


def _cells():
    cells = []
    for rank, operand_formats in OPERAND_FORMATS.items():
        operand = "ijkl"[:rank]
        for size in range(1, rank):
            for kept in combinations(operand, size):
                result = "".join(kept)
                for operand_fmt in operand_formats:
                    for result_fmt in _result_formats(len(result)):
                        cells.append((operand, result, operand_fmt, result_fmt))
    return cells


CELLS = _cells()


def _options():
    return replace(
        CompileOptions.from_environment().with_regblock_enabled(False),
        requested_schedule=Schedule(),
    )


def _operand(operand, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = SHAPES[len(operand)]
    values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
    keep = torch.rand(shape, generator=generator) < 0.55
    return (values * keep).to(torch.float32)


def _sparse(dense, fmt):
    tensor = STensor.from_torch(dense.clone(), "A")
    if set(fmt) != {"d"}:
        tensor.to_sparse(fmt)
    return tensor


def _reduction_cin(operand, result, operand_fmt, result_fmt):
    ivars = {name: IndexVar(name) for name in operand}
    source = TensorVar("A", fmt=operand_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in operand)
    ]
    target = TensorVar("C", fmt=result_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in result)
    ]
    statement = TensorAssign(target, source, op=Operation.ADD)
    for name in reversed(operand):
        statement = ForAll(ivars[name], statement)
    return statement


def _storage(result):
    levels = tuple(
        tuple(tuple(int(x) for x in part.tolist()) for part in mode)
        for mode in result.storage.index.mode_indices
    )
    return levels, tuple(float(v) for v in result.storage.value.tolist())


@pytest.mark.parametrize(
    "cell", CELLS, ids=[f"{c[2]} {c[0]}->{c[1]} as {c[3]}" for c in CELLS]
)
def test_sparse_result_reduction_is_correct_on_both_compilers(cell):
    operand, result, operand_fmt, result_fmt = cell
    dense = _operand(operand, seed=len(CELLS) + CELLS.index(cell))
    expected = torch.einsum(f"{operand}->{result}", dense.to(torch.float64))

    shipped = scorch.einsum(
        f"{operand}->{result}",
        _sparse(dense, operand_fmt),
        format=result_fmt,
        _compile_options=_options(),
    )
    assert str(shipped.format) == ",".join(result_fmt)
    torch.testing.assert_close(
        shipped.to_torch(in_place=False).to(torch.float64),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )

    result_shape = tuple(SHAPES[len(operand)][operand.index(c)] for c in result)
    try:
        loopir, _ = execute_cin_via_loopir(
            _reduction_cin(operand, result, operand_fmt, result_fmt),
            result_shape,
            _sparse(dense, operand_fmt),
            compile_options=_options(),
        )
    except (LoopIRLoweringError, LoopIRTargetError) as refusal:
        assert type(refusal.defect.code) is str and refusal.defect.code
        return
    torch.testing.assert_close(
        loopir.to_torch(in_place=False).to(torch.float64),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )
    assert _storage(loopir) == _storage(shipped)
