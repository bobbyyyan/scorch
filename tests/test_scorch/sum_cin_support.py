"""Sums as hand-written CIN for the developer entry: ``C[...] += A[...] + B[...]``.

No public operation builds a sum with a reduction (``einsum`` multiplies, ``A + B``
keeps every index), so these programs reach the shipped lowering only through
``lower_and_exec_cin``.  Operands are built and results read as level arrays in PyTorch
(``level_array_support``); float64 to 1e-12 of each cell, float32 to 1e-5.
"""

import torch

from scorch.compiler.cin import (
    BinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.pipeline import legacy_generated_cpp
from scorch.ops import lower_and_exec_cin
from tests.test_scorch.level_array_support import assert_matches, stored

SMALL = {"i": 5, "j": 4, "k": 6}
# 16 rows per worker: several workers wherever a loop is parallel.
LARGE = {"i": 300, "j": 9, "k": 7}

RANK2 = [("ss", "ss"), ("ss", "ds"), ("sd", "ds"), ("ss", "dd"), ("dd", "ss")]
RANK3 = [("sss", "sss"), ("sss", "dss"), ("dss", "ssd"), ("ddd", "sss")]

# (operand formats, operand indices, result indices, result format)
DENSE_REDUCTIONS = [
    (pair, "ij", result, "d") for pair in RANK2 for result in ("i", "j")
] + [
    (pair, "ijk", result, "d" * len(result))
    for pair in RANK3
    for result in ("k", "jk", "ik", "i")
]
SPARSE_REDUCTIONS = [
    (pair, "ij", result, "s") for pair in RANK2 for result in ("i", "j")
] + [
    (pair, "ijk", result, "s" * len(result))
    for pair in RANK3
    for result in ("k", "jk", "ik")
]
NO_REDUCTION = [
    (pair, "ij", "ij", result_format)
    for pair in RANK2
    for result_format in ("ss", "ds", "sd", "dd")
] + [(pair, "ijk", "ijk", "sss") for pair in RANK3]


def sum_cin(formats, indices, result_indices, result_format, dtype):
    ivars = {name: IndexVar(name) for name in indices}
    every = tuple(ivars[name] for name in indices)
    a = TensorVar("A", fmt=formats[0], dtype=dtype)[every]
    b = TensorVar("B", fmt=formats[1], dtype=dtype)[every]
    kept = tuple(ivars[name] for name in result_indices)
    result = TensorVar("C", fmt=result_format, dtype=dtype)[
        kept[0] if len(kept) == 1 else kept
    ]
    stmt = TensorAssign(result, BinaryOp(Operation.ADD, a, b), op=Operation.ADD)
    for name in reversed(indices):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def operands(formats, shape, dtype, seed):
    """Two operands; the sparse ones store nothing in about half their rows."""

    generator = torch.Generator().manual_seed(seed)
    dense = []
    for fmt in formats:
        values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
        if set(fmt) != {"d"}:
            keep = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.5
            rows = torch.rand(shape[0], generator=generator, dtype=torch.float64) < 0.5
            keep = keep & rows.reshape((-1,) + (1,) * (len(shape) - 1))
            values = values * keep
        dense.append(values.to(dtype))
    return dense


def reference(dense, indices, result_indices):
    reduced = [at for at, name in enumerate(indices) if name not in result_indices]
    total = dense[0].to(torch.float64) + dense[1].to(torch.float64)
    return total.sum(dim=reduced) if reduced else total


def emitted(formats, indices, result_indices, result_format, extents=SMALL):
    shape = tuple(extents[name] for name in indices)
    return [
        line.strip()
        for line in legacy_generated_cpp(
            sum_cin(formats, indices, result_indices, result_format, torch.float64),
            tuple(extents[name] for name in result_indices),
            ((shape, torch.float64), (shape, torch.float64)),
            compile_options=CompileOptions.from_environment(environ={}),
        ).splitlines()
    ]


def run(formats, indices, result_indices, result_format, extents, dtype, seed):
    shape = tuple(extents[name] for name in indices)
    dense = operands(formats, shape, dtype, seed)
    result = lower_and_exec_cin(
        sum_cin(formats, indices, result_indices, result_format, dtype),
        tuple(extents[name] for name in result_indices),
        stored(dense[0], formats[0], "A"),
        stored(dense[1], formats[1], "B"),
    )
    assert_matches(result, reference(dense, indices, result_indices), dtype)
