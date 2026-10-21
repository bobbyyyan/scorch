"""Sums of operands that hold different element types, on both compilers.

``A + B`` on two STensors declares its result with the operands' promoted type
(``layout.common_value_dtype``, PyTorch's rule): int64 plus float64 is float64,
int32 plus int64 is int64, uint8 plus int32 is int32.  The shipped compiler
computes such sums.  The new compiler refused every program whose tensors did
not share one element type (``mixed_dtype``) until 2026-10-12, when its value
expressions gained types: a read has its tensor's type, a binary operation the
promoted type of its operands (``nodes.promoted_scalar_type``), and a stored
value must have exactly the type of what it is written to.

Each cell is a sum over compressed, dense and coordinate operands into a
compressed, dense or coordinate result, under several pairs (and two triples)
of element types, on both compilers and both automatic schedule arms; the
shipped compiler is also called through ``A + B`` itself where that operator
builds the cell.  A result is judged from its own arrays against ``torch``'s
``a + b`` on the dense operands: its element type, its values (exactly, except
where a floating-point sum adds three terms or reduces, which is judged to
rounding), and stored coordinates equal to the union of the operands' stored
ones.

A sum whose result is declared with any other type is refused by the new
compiler with ``mixed_dtype``, as is int8 with uint8, which promote to int16, a
type Scorch does not store.  The verifier holds the same rule on its own.
"""

from __future__ import annotations

from typing import Dict, Tuple

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
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.loopir.lower_cin import (
    LoopIRLoweringError,
    lower_normalized_cin_to_loopir,
)
from scorch.compiler.loopir.nodes import (
    CursorValue,
    FloatConst,
    ScalarType,
    promoted_scalar_type,
)
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.loopir.verifier import LoopIRVerificationError, verify_program
from scorch.exceptions import UnsupportedDtypePromotionError
from tests.test_scorch import level_array_support
from tests.test_scorch.test_high_rank_programs import _level_arrays, _storage_order
from tests.test_scorch.test_loopir_integer_programs import _stored_cells
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

f32, f64 = torch.float32, torch.float64
i64, i32, i8, u8 = torch.int64, torch.int32, torch.int8, torch.uint8

# Every pair of distinct types whose promotion Scorch stores, in both orders
# where the order changes which operand is promoted.
ALL_PAIRS = (
    (f32, f64),
    (f64, f32),
    (i64, f64),
    (i32, i64),
    (i64, f32),
    (u8, i32),
    (i8, i64),
    (f32, i32),
)
FEW_PAIRS = ((i64, f64), (f32, f64), (u8, i32))

# (expression, operand formats, result format, operand type tuples)
CELLS: Tuple[Tuple[str, Tuple[str, ...], str, Tuple[Tuple[torch.dtype, ...], ...]], ...]
CELLS = (
    ("ij,ij->ij", ("ss", "ss"), "ss", ALL_PAIRS),
    ("ij,ij->ij", ("ds", "ds"), "ds", ALL_PAIRS),
    ("ij,ij->ij", ("oo", "oo"), "oo", ALL_PAIRS),
    ("ijk,ijk->ijk", ("sds", "sds"), "sds", ALL_PAIRS),
    ("i,i->i", ("s", "s"), "s", FEW_PAIRS),
    ("i,i->i", ("s", "d"), "d", FEW_PAIRS),
    ("ij,ij->ij", ("ss", "dd"), "ss", FEW_PAIRS),
    ("ij,ij->ij", ("sd", "ds"), "ss", FEW_PAIRS),
    ("ij,ij->ij", ("ds", "ss"), "dd", FEW_PAIRS),
    ("ij,ij->ij", ("oo", "ss"), "ss", FEW_PAIRS),
    ("ijk,ijk->ijk", ("sss", "sss"), "sss", FEW_PAIRS),
    ("ijk,ijk->ijk", ("dss", "ssd"), "sss", FEW_PAIRS),
    ("ij,ij,ij->ij", ("ss", "ds", "dd"), "ss", ((i32, i64, f32),)),
    ("ijk,ijk,ijk->ijk", ("sss", "dss", "ssd"), "sss", ((u8, i64, f64),)),
    ("ij,ij->i", ("ss", "ss"), "d", FEW_PAIRS),
    ("ij,ij->i", ("ds", "dd"), "d", FEW_PAIRS),
)
EXTENTS = ({"i": 4, "j": 5, "k": 3}, {"i": 3, "j": 0, "k": 2})


def _promoted(dtypes) -> torch.dtype:
    result = dtypes[0]
    for dtype in dtypes[1:]:
        result = torch.promote_types(result, dtype)
    return result


def _name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _parameters():
    parameters = []
    for expr, fmts, result_fmt, type_sets in CELLS:
        for dtypes in type_sets:
            compilers = ["new", "shipped"]
            terms = expr.split("->")[0].split(",")
            if (
                len(terms) == 2
                and result_fmt == fmts[0]
                and expr.split("->")[1] == terms[0]
            ):
                compilers.append("public")
            for compiler in compilers:
                parameters.append(
                    pytest.param(
                        expr,
                        fmts,
                        result_fmt,
                        dtypes,
                        compiler,
                        id=f"{expr} {','.join(fmts)} -> [{result_fmt}] "
                        f"{'+'.join(_name(d) for d in dtypes)} {compiler}",
                    )
                )
    return parameters


def _statement(expr: str, fmts, result_fmt: str, dtypes, result_dtype):
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    letters = "".join(dict.fromkeys("".join(terms)))
    variables = {letter: IndexVar(letter) for letter in letters}
    rhs = None
    for name, term, fmt, dtype in zip("ABDEF", terms, fmts, dtypes):
        access = TensorVar(name, fmt=fmt, dtype=dtype)[
            tuple(variables[letter] for letter in term)
        ]
        rhs = access if rhs is None else CINBinaryOp(Operation.ADD, rhs, access)
    lhs = TensorVar("C", fmt=result_fmt, dtype=result_dtype)[
        tuple(variables[letter] for letter in output)
    ]
    statement = TensorAssign(
        lhs, rhs, op=Operation.ADD if set(letters) - set(output) else None
    )
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def _dense(shape, seed: int, dtype: torch.dtype) -> torch.Tensor:
    """About 40% nonzero; integers in a range no sum of three leaves."""

    generator = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=generator) < 0.4
    if dtype.is_floating_point:
        values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.25
        return (values * keep).to(dtype)
    low, high = (0, 40) if dtype is u8 else (-40, 40)
    values = torch.randint(low, high, shape, generator=generator)
    values = torch.where(values == 0, torch.ones_like(values), values)
    return (values * keep).to(dtype)


def _inputs(expr: str, fmts, dtypes, extents: Dict[str, int], seed: int):
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    denses = [
        _dense(tuple(extents[c] for c in term), 17 * seed + position, dtype)
        for position, (term, dtype) in enumerate(zip(terms, dtypes))
    ]
    operands = [
        level_array_support.stored(dense, fmt, name)
        for dense, fmt, name in zip(denses, fmts, "ABDEF")
    ]
    promoted = _promoted(dtypes)
    reduced = [c for c in terms[0] if c not in output]
    total = denses[0]
    for dense in denses[1:]:
        total = total + dense  # torch's own promotion, term by term
    assert total.dtype is promoted
    if reduced:
        total = total.sum(dim=[terms[0].index(c) for c in reduced], dtype=promoted)
    structure = None
    if not reduced:
        structure = torch.zeros(total.shape, dtype=torch.bool)
        for dense, fmt in zip(denses, fmts):
            structure |= _stored_cells(dense, fmt)
    return operands, total, structure


def _judge(result, reference, structure, result_fmt: str, exact: bool) -> None:
    dtype = reference.dtype
    assert result.dtype is dtype, f"result dtype {result.dtype}, expected {dtype}"
    if set(result_fmt) == {"o"}:
        coords, values = level_array_support.coordinate_entries(result)
        expected = structure.nonzero()
        assert torch.equal(coords, expected), "stored entries differ from the implied"
        assert torch.equal(values.to(dtype), reference[tuple(expected.t())])
        return
    if "s" in result_fmt:
        # Read the arrays before ``to_torch``, which densifies the storage.
        order = _storage_order(result, structure.dim())
        _, expected = _level_arrays(structure.permute(order), result_fmt)
        actual = []
        for level, arrays in enumerate(result.storage.index.mode_indices):
            if result_fmt[level] == "d":
                actual.append(None)
            else:
                pos, crd = (t.reshape(-1).tolist() for t in arrays)
                actual.append((pos, crd))
        assert actual == expected, "stored coordinates differ from the operands'"
    got = result.to_torch(in_place=False)
    assert got.dtype is dtype
    if exact or not dtype.is_floating_point:
        assert torch.equal(
            got, reference
        ), f"{int((got != reference).sum())} of {reference.numel()} cells differ"
    else:
        tolerance = 1e-6 if dtype is f32 else 1e-12
        assert torch.allclose(got, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("expr,fmts,result_fmt,dtypes,compiler", _parameters())
def test_mixed_dtype_sum_matches_torch(expr, fmts, result_fmt, dtypes, compiler, arm):
    promoted = _promoted(dtypes)
    output = expr.split("->")[1]
    terms = expr.split("->")[0].split(",")
    exact = len(terms) == 2 and set(terms[0]) == set(output)
    for number, extents in enumerate(EXTENTS):
        operands, reference, structure = _inputs(expr, fmts, dtypes, extents, number)
        shape = tuple(extents[c] for c in output)
        statement = _statement(expr, fmts, result_fmt, dtypes, promoted)
        if compiler == "new":
            result, _ = execute_cin_via_loopir(
                statement,
                shape,
                *operands,
                compile_options=auto_options(arm, jit=True),
            )
        elif compiler == "shipped":
            result = scorch.ops.lower_and_exec_cin(
                statement,
                shape,
                *operands,
                compile_options=CompileOptions.from_environment().with_regblock_enabled(
                    arm
                ),
            )
        else:
            result = operands[0] + operands[1]
        try:
            _judge(result, reference, structure, result_fmt, exact)
        except AssertionError as error:
            raise AssertionError(f"extents {extents}: {error}") from error


def _run(compiler: str, statement, shape, operands, arm: bool):
    if compiler == "new":
        result, _ = execute_cin_via_loopir(
            statement, shape, *operands, compile_options=auto_options(arm, jit=True)
        )
        return result
    return scorch.ops.lower_and_exec_cin(
        statement,
        shape,
        *operands,
        compile_options=CompileOptions.from_environment().with_regblock_enabled(arm),
    )


@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("compiler", ("new", "shipped"))
def test_int64_values_past_two_to_the_31_are_kept_in_float64(compiler, arm):
    """A truncation to the left operand's type, or a narrowing through an
    int32 temporary, changes these sums; float64 holds them exactly."""

    generator = torch.Generator().manual_seed(5)
    keep = torch.rand((4, 5), generator=generator) < 0.5
    wide = (torch.randint(2**40, 2**41, (4, 5), generator=generator) * keep).to(i64)
    fraction = torch.rand((4, 5), generator=generator, dtype=f64) * (
        torch.rand((4, 5), generator=generator) < 0.5
    )
    operands = [
        level_array_support.stored(wide, "ss", "A"),
        level_array_support.stored(fraction, "ds", "B"),
    ]
    statement = _statement("ij,ij->ij", ("ss", "ds"), "ss", (i64, f64), f64)
    result = _run(compiler, statement, (4, 5), operands, arm)
    assert result.dtype is f64
    assert torch.equal(result.to_torch(in_place=False), wide.to(f64) + fraction)


@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("compiler", ("new", "shipped"))
def test_a_float32_by_float64_reduction_accumulates_in_float64(compiler, arm):
    """``y[i] = sum_j A[i, j] * B[i, j]`` with float32 ``A`` and float64 ``B``
    whose values float32 cannot hold (``1 + k * 1e-12``): an accumulator, a
    temporary or a hoisted factor declared at an operand's type loses them.

    A product reaches the compilers only through the CIN entries; the public
    ``matmul`` and ``einsum`` raise for mixed types, as torch does (0036)."""

    generator = torch.Generator().manual_seed(9)
    keep = torch.rand((4, 6), generator=generator) < 0.6
    narrow = (torch.randint(1, 4, (4, 6), generator=generator) * keep).to(f32)
    precise = 1 + torch.arange(24, dtype=f64).reshape(4, 6) * 1e-12
    operands = [
        level_array_support.stored(narrow, "ds", "A"),
        level_array_support.stored(precise, "dd", "B"),
    ]
    i, j = IndexVar("i"), IndexVar("j")
    statement = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("y", fmt="d", dtype=f64)[i],
                CINBinaryOp(
                    Operation.MUL,
                    TensorVar("A", fmt="ds", dtype=f32)[i, j],
                    TensorVar("B", fmt="dd", dtype=f64)[i, j],
                ),
                op=Operation.ADD,
            ),
        ),
    )
    result = _run(compiler, statement, (4,), operands, arm)
    assert result.dtype is f64
    expected = (narrow.to(f64) * precise).sum(dim=1)
    assert torch.allclose(
        result.to_torch(in_place=False), expected, rtol=1e-15, atol=0
    ), (result.to_torch(in_place=False) - expected)


def test_the_promotion_rule_is_torchs():
    to_torch = {
        ScalarType.FLOAT32: f32,
        ScalarType.FLOAT64: f64,
        ScalarType.INT64: i64,
        ScalarType.INT32: i32,
        ScalarType.INT8: i8,
        ScalarType.UINT8: u8,
    }
    for lhs in ScalarType:
        for rhs in ScalarType:
            promoted = promoted_scalar_type(lhs, rhs)
            expected = torch.promote_types(to_torch[lhs], to_torch[rhs])
            if expected not in to_torch.values():
                assert promoted is None, (lhs, rhs)
            else:
                assert to_torch[promoted] is expected, (lhs, rhs)


def _bindings(expr: str, dtypes, extents) -> Tuple:
    terms = expr.split("->")[0].split(",")
    return tuple(
        (tuple(extents[c] for c in term), dtype) for term, dtype in zip(terms, dtypes)
    )


@pytest.mark.parametrize(
    "dtypes,result_dtype",
    [
        ((i64, f64), f32),  # a narrowing store
        ((f32, f32), f64),  # a widening store
        ((i32, i64), i32),
        ((i8, u8), i8),  # int8 with uint8 promotes to int16
    ],
    ids=str,
)
def test_a_result_not_of_the_promoted_type_is_refused(dtypes, result_dtype):
    """The lowering's own rule, below the entry that binds the promoted type."""

    statement = _statement("ij,ij->ij", ("ss", "ss"), "ss", dtypes, result_dtype)
    with pytest.raises(LoopIRLoweringError) as error:
        lower_normalized_cin_to_loopir(normalize_cin(statement))
    assert error.value.defect.code == "mixed_dtype"


@pytest.mark.parametrize("entry", ("new", "shipped", "public"))
def test_int8_with_uint8_is_refused_with_its_code(entry):
    """They promote to int16, which Scorch does not store; every entry refuses
    the pair with ``unsupported_dtype_promotion`` before building a kernel."""

    extents = {"i": 4, "j": 5}
    operands, _, _ = _inputs_unpromoted("ij,ij->ij", ("ss", "ss"), (i8, u8), extents)
    statement = _statement("ij,ij->ij", ("ss", "ss"), "ss", (i8, u8), i8)
    with pytest.raises(UnsupportedDtypePromotionError) as error:
        if entry == "new":
            execute_cin_via_loopir(
                statement, (4, 5), *operands, compile_options=auto_options(False)
            )
        elif entry == "shipped":
            scorch.ops.lower_and_exec_cin(statement, (4, 5), *operands)
        else:
            operands[0] + operands[1]
    assert error.value.code == "unsupported_dtype_promotion"


def _inputs_unpromoted(expr: str, fmts, dtypes, extents: Dict[str, int]):
    """Operands only, for a pair whose promotion Scorch does not store."""

    terms = expr.split("->")[0].split(",")
    denses = [
        _dense(tuple(extents[c] for c in term), position, dtype)
        for position, (term, dtype) in enumerate(zip(terms, dtypes))
    ]
    operands = [
        level_array_support.stored(dense, fmt, name)
        for dense, fmt, name in zip(denses, fmts, "ABDEF")
    ]
    return operands, None, None


def _lowered_mixed_program():
    extents = {"i": 4, "j": 5}
    kernel = compile_cin_via_loopir(
        _statement("ij,ij->ij", ("ss", "ss"), "ss", (i64, f64), f64),
        (4, 5),
        _bindings("ij,ij->ij", (i64, f64), extents),
        compile_options=auto_options(False),
    )
    return kernel.lowering.program


def _walk(node):
    yield node
    for value in vars(node).values():
        children = value if isinstance(value, tuple) else (value,)
        for child in children:
            if hasattr(child, "node_id"):
                yield from _walk(child)


def test_the_verifier_refuses_a_value_stored_into_another_type():
    program = _lowered_mixed_program()
    verify_program(program)
    result = next(t for t in program.tensors if t.symbol in program.outputs)
    object.__setattr__(result, "dtype", ScalarType.FLOAT32)
    with pytest.raises(LoopIRVerificationError) as error:
        verify_program(program)
    assert error.value.defect.code == "mixed_dtype"


def test_a_default_has_its_cursors_type():
    """The int64 operand's union default is an integer zero, not ``0.0``."""

    program = _lowered_mixed_program()
    tensors = {t.symbol: t for t in program.tensors}
    reads = [node for node in _walk(program.body) if type(node) is CursorValue]
    int_reads = []
    for read in reads:
        cursor = next(
            c
            for node in _walk(program.body)
            for c in getattr(node, "cursors", ())
            if c.cursor == read.cursor
        )
        if tensors[cursor.tensor].dtype is ScalarType.INT64:
            int_reads.append(read)
    assert int_reads and all(type(r.default) is not FloatConst for r in int_reads)
    object.__setattr__(
        int_reads[0], "default", FloatConst(int_reads[0].default.node_id, 0.0)
    )
    with pytest.raises(LoopIRVerificationError) as error:
        verify_program(program)
    assert error.value.defect.code == "mixed_dtype"
