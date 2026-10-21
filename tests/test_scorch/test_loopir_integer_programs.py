"""Integer element types, on both compilers.

The shipped compiler builds kernels for int64, int32, int8 and uint8 operands
(``utils.PYTORCH_DTYPE_TO_C_DATATYPE``); the new compiler refused every one with
``unsupported_dtype`` until its scalar type gained the four integer types.  Each
cell below is one einsum over integer operands into one result layout: products
(matrix-vector, matrix-matrix, SDDMM, MTTKRP, a batched product, a reduction),
elementwise products and sums, over dense, compressed and coordinate operands,
into dense, compressed and coordinate results.

Every result is judged from its own arrays against ``torch.einsum`` on int64
copies, exactly: its element type is the operands', every value is equal, and a
sparse result stores exactly the coordinates the operands' stored entries imply.
The values are chosen so the sums leave float32's exact range: entries are drawn
from ``[low, 3 * low)`` with ``low`` set so a product of the program's operands
lies above 2**24, while every sum stays inside the type (for int32, below
2**31).  A float32 accumulator anywhere in a kernel (the shipped scalar
accumulator before B2 was one) rounds such a sum, and the exact comparison sees
it.  int8 and uint8 wrap: their reference is the int64 result cast to the type,
which is what C++'s conversion of each stored value to the element type gives,
since sums and products modulo 2**8 do not depend on where the wrap happens.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import pytest
import torch

import scorch
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.oracle import LoopIROracleError, run_program
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from tests.test_scorch import level_array_support
from tests.test_scorch.test_high_rank_programs import _level_arrays, _storage_order
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

# (expression, how terms combine, operand formats, result formats)
CELLS: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("ij,j->i", "mul", ("ds", "d"), ("d", "s")),
    ("ij,j->i", "mul", ("ss", "d"), ("d", "s")),
    ("ij,j->i", "mul", ("dd", "d"), ("d",)),
    ("ij,jk->ik", "mul", ("ds", "dd"), ("dd", "ds")),
    ("ij,jk->ik", "mul", ("ds", "ds"), ("ds", "ss")),
    ("ij,ij->ij", "mul", ("ds", "ds"), ("ds", "dd")),
    ("ijk->i", "mul", ("dss",), ("d", "s")),
    ("ij,ik,jk->ij", "mul", ("ds", "dd", "dd"), ("ds",)),
    ("ij,ik,jk->ij", "mul", ("oo", "dd", "dd"), ("oo",)),
    ("ijk,jr,kr->ir", "mul", ("dss", "dd", "dd"), ("dd",)),
    ("bij,bjk->bik", "mul", ("dds", "dds"), ("dds",)),
    ("ij,ij->ij", "add", ("ds", "ds"), ("ds",)),
    ("ij,ij->ij", "add", ("ss", "ds"), ("ss",)),
    ("ij,ij->ij", "add", ("ds", "dd"), ("dd",)),
    ("ij,ij->ij", "add", ("oo", "ss"), ("ss",)),
    # Reductions directly under the last loop of an all-coordinate result:
    # the shipped scalar accumulator (float before B2).
    ("ij->i", "mul", ("oo",), ("o",)),
    ("ijk->ij", "mul", ("ooo",), ("oo",)),
    ("ij,ij->ij", "add", ("oo", "oo"), ("oo",)),
    # Operands no loop order follows together, so the new compiler's plan
    # converts one before the main kernel (operand_conversion; the programs
    # of test_loopir_operand_conversions.CONVERTING).
    ("ij,ji->ij", "mul", ("ds", "ds"), ("ds", "dd")),
    ("ij->ji", "mul", ("ds",), ("ds", "dd")),
)
DTYPES = (torch.int64, torch.int32)
# The wrapping types run on a few cells: their kernels share every line with
# the wider types' but the element type.
NARROW_CELLS = (0, 3, 7, 11)
NARROW_DTYPES = (torch.int8, torch.uint8)
EXTENTS = (
    {"i": 4, "j": 5, "k": 3, "r": 2, "b": 2},
    {"i": 3, "j": 0, "k": 2, "r": 2, "b": 2},
)
# How far below the type's limit every sum stays: at most this many terms add
# into one cell of these extents (MTTKRP sums j * k = 15).
_TERMS = 16


def _cell_id(cell) -> str:
    expr, combine, fmts, _ = cell
    return f"{expr} {combine} {','.join(fmts)}"


def _parameters():
    parameters = []
    for number, cell in enumerate(CELLS):
        dtypes = DTYPES + (NARROW_DTYPES if number in NARROW_CELLS else ())
        for result_fmt, dtype in itertools.product(cell[3], dtypes):
            parameters.append(
                pytest.param(
                    cell,
                    result_fmt,
                    dtype,
                    id=f"{_cell_id(cell)} -> [{result_fmt}] {dtype}".replace(
                        "torch.", ""
                    ),
                )
            )
    return parameters


def _statement(expr: str, combine: str, fmts, result_fmt: str, dtype):
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    letters = "".join(dict.fromkeys("".join(terms)))
    variables = {letter: IndexVar(letter) for letter in letters}
    operation = Operation.MUL if combine == "mul" else Operation.ADD
    rhs = None
    for name, term, fmt in zip("ABDEF", terms, fmts):
        access = TensorVar(name, fmt=fmt, dtype=dtype)[
            tuple(variables[letter] for letter in term)
        ]
        rhs = access if rhs is None else CINBinaryOp(operation, rhs, access)
    lhs = TensorVar("C", fmt=result_fmt, dtype=dtype)[
        tuple(variables[letter] for letter in output)
    ]
    statement = TensorAssign(
        lhs, rhs, op=Operation.ADD if set(letters) - set(output) else None
    )
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def _low(dtype: torch.dtype, factors: int) -> int:
    """The smallest entry: ``_TERMS`` products of ``factors`` entries from
    ``[low, 3 * low)`` stay inside ``dtype``, and the largest such product is
    above 2**24.  (Each test also checks its own reference leaves 2**24.)"""

    if dtype in NARROW_DTYPES:
        return 20  # wraps on purpose
    limit = 2**31 - 1 if dtype is torch.int32 else 2**62
    low = int(((limit // _TERMS) ** (1.0 / factors)) / 3)
    while _TERMS * (3 * low) ** factors > limit:
        low -= 1
    assert (3 * low) ** factors > 2**24, (dtype, factors, low)
    return low


def _dense(shape, seed: int, low: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(low, 3 * low, shape, generator=generator, dtype=torch.int64)
    keep = torch.rand(shape, generator=generator) < 0.45
    if len(shape) and shape[0] > 1:
        keep[0] = False  # a slice the operand stores nothing under
    return (values * keep).to(dtype)


def _reference(expr: str, combine: str, denses: List[torch.Tensor], dtype):
    wide = [dense.to(torch.int64) for dense in denses]
    if combine == "mul":
        exact = torch.einsum(expr, *wide)
    else:
        terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
        exact = wide[0].permute([terms[0].index(c) for c in output])
        for term, dense in zip(terms[1:], wide[1:]):
            exact = exact + dense.permute([term.index(c) for c in output])
    return exact.to(dtype)


def _stored_cells(dense: torch.Tensor, fmt: str) -> torch.Tensor:
    """Which cells an operand built by ``level_array_support.stored`` stores:
    its nonzeros, plus every cell a dense level expands below a stored parent
    (all of them for an all-dense operand)."""

    if set(fmt) == {"o"}:
        return dense != 0
    leaves, _ = _level_arrays(dense != 0, fmt)
    cells = torch.zeros(dense.shape, dtype=torch.bool)
    for leaf in leaves:
        cells[leaf] = True
    return cells


def _structure(
    expr: str, combine: str, fmts, denses: List[torch.Tensor]
) -> torch.Tensor:
    """Which result cells the operands' stored entries imply."""

    indicators = [
        _stored_cells(dense, fmt).to(torch.int64) for dense, fmt in zip(denses, fmts)
    ]
    if combine == "mul":
        return torch.einsum(expr, *indicators) > 0
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    implied = torch.zeros_like(indicators[0], dtype=torch.bool)
    for term, indicator in zip(terms, indicators):
        implied = implied | indicator.permute([term.index(c) for c in output]).bool()
    return implied


def _judge(result, reference: torch.Tensor, structure, result_fmt: str, dtype):
    assert result.dtype is dtype, f"result dtype {result.dtype}, operands {dtype}"
    if set(result_fmt) == {"o"}:
        coords, values = level_array_support.coordinate_entries(result)
        expected = structure.nonzero()
        assert torch.equal(coords, expected), "stored entries differ from the implied"
        assert torch.equal(values.to(dtype), reference[tuple(expected.t())])
        return
    got = result.to_torch(in_place=False)
    assert got.dtype is dtype
    assert torch.equal(got, reference), (
        f"{int((got != reference).sum())} of {reference.numel()} cells differ; "
        f"max |diff| {int((got.to(torch.int64) - reference.to(torch.int64)).abs().max())}"
    )
    if "s" not in result_fmt:
        return
    # A result may be stored in another mode order (the shipped einsum returns
    # ``ij->ji`` as the operand's arrays under a permutation); the format
    # letters name its storage levels in that order.
    order = _storage_order(result, structure.dim())
    _, expected = _level_arrays(structure.permute(order), result_fmt)
    actual = []
    for level, arrays in enumerate(result.storage.index.mode_indices):
        if result_fmt[level] == "d":
            actual.append(None)
        else:
            pos, crd = (t.reshape(-1).tolist() for t in arrays)
            actual.append((pos, crd))
    assert actual == expected, "stored coordinates differ from the operands' structure"


def _inputs(cell, extents: Dict[str, int], dtype, number: int):
    expr, combine, fmts, _ = cell
    terms = expr.split("->")[0].split(",")
    low = _low(dtype, len(terms) if combine == "mul" else 1)
    denses = [
        _dense(tuple(extents[c] for c in term), 31 * number + position, low, dtype)
        for position, term in enumerate(terms)
    ]
    operands = [
        level_array_support.stored(dense, fmt, name)
        for dense, fmt, name in zip(denses, fmts, "ABDEF")
    ]
    return denses, operands


@pytest.mark.parametrize("compiler", ("shipped", "new"))
@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize(("cell", "result_fmt", "dtype"), _parameters())
def test_integer_program_matches_int64_einsum(cell, result_fmt, dtype, arm, compiler):
    expr, combine, fmts, _ = cell
    output = expr.split("->")[1]
    for number, extents in enumerate(EXTENTS):
        denses, operands = _inputs(cell, extents, dtype, number)
        reference = _reference(expr, combine, denses, dtype)
        structure = _structure(expr, combine, fmts, denses)
        if dtype in DTYPES and number == 0:
            # The premise: some value is beyond float32's exact integers.
            assert int(reference.to(torch.int64).abs().max()) > 2**24
        if compiler == "shipped":
            if combine == "add":
                # A sum has no public einsum spelling; it enters the shipped
                # compiler's generic path, as test_loopir_dense_union does.
                result = scorch.ops.lower_and_exec_cin(
                    _statement(expr, combine, fmts, result_fmt, dtype),
                    tuple(extents[c] for c in output),
                    *operands,
                    compile_options=CompileOptions.from_environment().with_regblock_enabled(
                        arm
                    ),
                )
            else:
                result = scorch.einsum(
                    expr,
                    *operands,
                    format=result_fmt,
                    _compile_options=auto_options(arm, jit=True),
                )
        else:
            result, _ = execute_cin_via_loopir(
                _statement(expr, combine, fmts, result_fmt, dtype),
                tuple(extents[c] for c in output),
                *operands,
                compile_options=auto_options(arm, jit=True),
            )
        try:
            _judge(result, reference, structure, result_fmt, dtype)
        except AssertionError as error:
            raise AssertionError(f"extents {extents}: {error}") from error


def test_the_values_stay_inside_their_types():
    for dtype in DTYPES:
        limit = 2**31 - 1 if dtype is torch.int32 else 2**63 - 1
        for factors in (1, 2, 3):
            assert _TERMS * (3 * _low(dtype, factors)) ** factors <= limit


def test_the_oracle_refuses_an_integer_program():
    cell = CELLS[0]
    expr, combine, fmts, result_fmts = cell
    denses, operands = _inputs(cell, EXTENTS[0], torch.int64, 0)
    _, kernel = execute_cin_via_loopir(
        _statement(expr, combine, fmts, result_fmts[0], torch.int64),
        (EXTENTS[0]["i"],),
        *operands,
        compile_options=auto_options(False, jit=True),
    )
    program = kernel.lowering.program
    with pytest.raises(LoopIROracleError, match="int64"):
        run_program(program, {}, {})


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_an_integer_operand_is_converted_in_its_own_type(dtype):
    """The premise of the ``ij,ji->ij`` cell: the plan converts one operand
    (``A`` wants ``i`` outer, ``B`` wants ``j`` outer), and the conversion
    kernel moves its values in the operand's own element type."""

    cell = next(c for c in CELLS if c[0] == "ij,ji->ij")
    expr, combine, fmts, result_fmts = cell
    _, operands = _inputs(cell, EXTENTS[0], dtype, 0)
    _, kernel = execute_cin_via_loopir(
        _statement(expr, combine, fmts, result_fmts[0], dtype),
        (EXTENTS[0]["i"], EXTENTS[0]["j"]),
        *operands,
        compile_options=auto_options(False, jit=True),
    )
    assert kernel.conversions, "the plan converts no operand"
    spelled = {torch.int64: "int64_t", torch.int32: "int32_t"}[dtype]
    for source in kernel.conversion_sources:
        assert spelled in source
        assert "float" not in source and "double" not in source


def test_the_value_pointer_spelling_moves_only_int32():
    """Float kernels keep their text: the one pointer spelling the new target
    now uses agrees with the one it used before for every element type but
    int32 (``int*`` before, ``int32_t*`` now, as the shared result ABI spells
    it)."""

    from scorch.compiler import llir
    from scorch.compiler.loopir.lower_llir import _value_pointer_type

    for dtype in (torch.float32, torch.float64, torch.int64, torch.int8, torch.uint8):
        assert _value_pointer_type(dtype) is llir.DataType.ptr_type(dtype)
    assert llir.DataType.ptr_type(torch.int32) is llir.DataType.PTR_INT
    assert _value_pointer_type(torch.int32) is llir.DataType.PTR_INT_32
