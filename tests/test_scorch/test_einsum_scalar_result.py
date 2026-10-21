"""einsum computes a rank-0 result (``"ij->"``) and reads a rank-0 operand (``",i->i"``).

A rank-0 tensor holds one value, at position 0: the root of its level tree, which has
no levels.  That is how TACO treats a scalar.  ``TensorVar(fmt="")`` declares one
(``parse_format("")`` is the rank-0 format), and einsum makes one for an empty term: an
empty result term sums over every index, and an empty operand term is a scalar operand.

einsum used to refuse an empty term where it checks the labels: an empty result term by
``EinsumScalarResultError`` (before that, by a plain ``CompileSpecError``), an empty
operand term by a plain ``CompileSpecError``.  Below that check each lowering refused a
rank-0 tensor by its own code, ``unsupported_scalar_tensor`` in the shipped lowering and
``unsupported_format`` in the new one.  Both lowerings, and the result assembly they
share, now follow the rule above: a read or a write of a rank-0 tensor addresses position
0 of its value array, a rank-0 dense result allocates one value (the empty product), and
every loop around a rank-0 result is a reduction, so its update is ``+=``.  An expression
with no index in any term (``"->"``) has no loop, and is still refused, by
``EinsumScalarResultError``.

Each program is checked against ``torch.einsum`` in float64 (float64 results to 1e-12,
float32 to 1e-5, integers exactly) on both automatic arms; the shipped compiler on the
first call and on the call einsum's dispatch cache serves.  The new compiler runs on the
program einsum lowers, copied after einsum's operand moves, with the operands einsum
passes to its kernel.

``scorch.einsum`` hands a call whose operands and result are all dense to
``torch.einsum`` (drop-in note, section 0.1, step 1), so the programs here whose operands
are all dense reach the shipped compiler through its own entry,
``ops._einsum_via_compiler`` (``shipped_einsum``); every other program goes through the
public einsum, as before.
"""

import copy
from dataclasses import replace

import pytest
import torch

import scorch
from scorch import ops, stensor, utils
from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.scheduler import Schedule
from scorch.exceptions import CompileSpecError, EinsumScalarResultError
from scorch.format import TensorFormat
from scorch.layout import TensorSpec
from scorch.stensor import STensor
from tests.test_scorch.level_array_support import (
    assert_matches,
    emitted_einsum_cpp,
    reference,
    stored,
)

F64 = torch.float64
EXTENT = {"i": 5, "j": 4, "k": 3}

# (expression, operand formats).  A rank-0 result over every operand layout of rank 1 to
# 3 in dense, compressed and all-coordinate levels, with one to three operands.
SCALAR_RESULTS = [
    ("i->", ["d"]),
    ("i->", ["s"]),
    ("i->", ["o"]),
    ("ij->", ["dd"]),
    ("ij->", ["ds"]),
    ("ij->", ["sd"]),
    ("ij->", ["ss"]),
    ("ij->", ["oo"]),
    ("ijk->", ["sss"]),
    ("ijk->", ["dsd"]),
    ("ijk->", ["ooo"]),
    ("i,i->", ["s", "s"]),
    ("i,i->", ["d", "s"]),
    ("ij,ij->", ["ss", "ss"]),
    ("ij,ij->", ["ds", "dd"]),
    ("ij,ij->", ["ss", "ds"]),
    ("ij,jk->", ["ds", "ds"]),
    ("ij,j->", ["ss", "d"]),
    ("ij,jk,k->", ["ds", "sd", "s"]),
]
# A rank-0 operand (format "") beside operands of rank 1 and 2, in the first, middle and
# last position, under results of rank 0 to 2.
SCALAR_OPERANDS = [
    (",i->i", ["", "s"]),
    ("i,->i", ["d", ""]),
    (",ij->ij", ["", "ds"]),
    (",ij->", ["", "ss"]),
    (",i,->", ["", "s", ""]),
    ("i,,i->", ["s", "", "s"]),
]
PROGRAMS = SCALAR_RESULTS + SCALAR_OPERANDS
DTYPES = [torch.float32, torch.float64, torch.int64, torch.int32]
ARMS = pytest.mark.parametrize(
    "regblock", (False, True), ids=("regblock-off", "regblock-on")
)
EACH_DTYPE = pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])


def program_id(program):
    expr, formats = program
    return f"{expr}[{','.join(fmt or '-' for fmt in formats)}]"


EACH_PROGRAM = pytest.mark.parametrize(
    ("expr", "formats"), PROGRAMS, ids=[program_id(p) for p in PROGRAMS]
)


def operand_values(term, dtype, seed, extent=EXTENT):
    """Values for one term, about 40% of them zero; a scalar operand is nonzero, so a
    kernel that skipped reading it could not pass."""

    generator = torch.Generator().manual_seed(seed)
    shape = tuple(extent[index] for index in term)
    kept = torch.rand(shape, generator=generator) < 0.6
    if dtype.is_floating_point:
        values = torch.rand(shape, generator=generator, dtype=F64) + 0.5
    else:
        values = torch.randint(1, 9, shape, generator=generator).to(F64)
    return (values * (kept if term else True)).to(dtype)


def operands_of(expr, formats, dtype, extent=EXTENT):
    """Dense values per term, and the same stored in ``formats`` from level arrays."""

    terms = expr.split("->")[0].split(",")
    dense = [
        operand_values(term, dtype, seed, extent)
        for seed, term in enumerate(terms, start=11)
    ]
    built = [
        stored(values, fmt, "ABC"[position])
        for position, (values, fmt) in enumerate(zip(dense, formats))
    ]
    return dense, built


def check(result, expr, dense, dtype):
    """``result`` against ``torch.einsum``; a rank-0 result also for its layout."""

    assert isinstance(result, STensor)
    assert result.dtype == dtype
    expected = reference(expr, dense)
    if expected.dim() == 0:
        assert tuple(result.shape) == ()
        assert result.format.get_order() == 0 and str(result.format) == ""
        assert list(result.mode_order) == []
        assert result.storage.index.mode_indices == []
        assert result.values.numel() == 1
    assert_matches(result, expected, dtype)


def options(regblock):
    return CompileOptions.from_environment().with_regblock_enabled(regblock)


def shipped_einsum(expr, *operands, **kwargs):
    """The shipped compiler on ``expr``: the public einsum, or, for operands that are all
    dense, which the public einsum hands to ``torch.einsum``, the compiler's entry."""

    if all(ops._is_dense_operand(operand) for operand in operands):
        return ops._einsum_via_compiler(expr, *operands, **kwargs)
    return scorch.einsum(expr, *operands, **kwargs)


@pytest.fixture
def empty_caches(monkeypatch):
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})


# -- the shipped compiler, through the public einsum --------------------------------


@ARMS
@EACH_DTYPE
@EACH_PROGRAM
def test_shipped_einsum_computes_the_program(
    empty_caches, expr, formats, dtype, regblock
):
    dense, built = operands_of(expr, formats, dtype)

    first = shipped_einsum(expr, *built, _compile_options=options(regblock))
    entries = len(ops._einsum_dispatch_cache)
    second = shipped_einsum(expr, *built, _compile_options=options(regblock))

    check(first, expr, dense, dtype)
    # The second call is served by the dispatch cache the first one filled.
    assert entries == 1 and len(ops._einsum_dispatch_cache) == 1
    check(second, expr, dense, dtype)


@ARMS
@EACH_PROGRAM
def test_shipped_einsum_computes_the_program_under_an_empty_schedule(
    empty_caches, expr, formats, regblock
):
    dense, built = operands_of(expr, formats, F64)
    scheduled = replace(options(regblock), requested_schedule=Schedule())

    result = scorch.einsum(expr, *built, _compile_options=scheduled)

    check(result, expr, dense, F64)


def test_a_scalar_result_has_its_own_cache_entry(empty_caches):
    """``"i,i->"`` and ``"i,i->i"`` over the same operands are different programs."""

    dense, built = operands_of("i,i->", ["s", "d"], F64)

    dot = scorch.einsum("i,i->", *built)
    product = scorch.einsum("i,i->i", *built)
    dot_again = scorch.einsum("i,i->", *built)

    assert len(ops._einsum_dispatch_cache) == 2
    check(dot, "i,i->", dense, F64)
    check(product, "i,i->i", dense, F64)
    check(dot_again, "i,i->", dense, F64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_a_dense_torch_operand_is_summed(empty_caches, dtype):
    """By the compiler through its entry, and by ``torch.einsum`` through the public one,
    which hands it an all-dense call unchanged."""

    square = torch.arange(1, 10, dtype=dtype).reshape(3, 3)

    compiled = ops._einsum_via_compiler("ij->", square)
    public = scorch.einsum("ij->", square)

    check(compiled, "ij->", [square], dtype)
    assert type(public) is torch.Tensor
    assert torch.equal(public, torch.einsum("ij->", square))


def test_the_kernel_writes_position_zero_once_per_stored_entry(monkeypatch):
    source = emitted_einsum_cpp(monkeypatch, "ij->", ["ss"], {"i": 3, "j": 4})

    assert "int64_t B_capacity = 1;" in source
    assert "B_values[0] += A_val[pA1];" in source
    assert "B.storage.index.mode_indices = {};" in source


# -- the new compiler, on the program einsum lowers ---------------------------------


def captured_program(monkeypatch, expr, built, compile_options):
    """The program einsum lowers and the operands it passes, after its operand moves."""

    captured = {}
    select = ops._select_and_bind_einsum_operand_mode_orders
    finish = ops._finish_einsum_operand_moves

    def remember_program(cin_stmt, *args, **kwargs):
        captured["cin"] = cin_stmt
        return select(cin_stmt, *args, **kwargs)

    def remember_moves(*args, **kwargs):
        moved = finish(*args, **kwargs)
        captured["program"] = copy.deepcopy(captured["cin"])
        captured["operands"] = list(moved[0])
        return moved

    monkeypatch.setattr(
        ops, "_select_and_bind_einsum_operand_mode_orders", remember_program
    )
    monkeypatch.setattr(ops, "_finish_einsum_operand_moves", remember_moves)
    shipped_einsum(expr, *built, _compile_options=compile_options)
    return captured["program"], captured["operands"]


def result_shape_of(program):
    stmt = program
    while isinstance(stmt, ForAll):
        stmt = stmt.stmt
    return tuple(stmt.lhs.tensor.shape)


@ARMS
@EACH_DTYPE
@EACH_PROGRAM
def test_new_compiler_computes_the_program(
    monkeypatch, empty_caches, expr, formats, dtype, regblock
):
    dense, built = operands_of(expr, formats, dtype)
    program, moved = captured_program(monkeypatch, expr, built, options(regblock))
    loopir_options = replace(options(regblock), requested_schedule=Schedule())

    result, _ = execute_cin_via_loopir(
        program, result_shape_of(program), *moved, compile_options=loopir_options
    )

    check(result, expr, dense, dtype)


# -- at extents where loops parallelize: the sum into one value must not race ---------

LARGE_EXTENT = {"i": 1031, "j": 517}
LARGE = [
    ("ij->", ["ss"]),
    ("ij->", ["ds"]),
    ("ij->", ["dd"]),
    ("ij,ij->", ["ss", "ds"]),
    ("i,i->", ["s", "s"]),
]


@ARMS
@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.float64], ids=lambda d: str(d)[6:]
)
@pytest.mark.parametrize(("expr", "formats"), LARGE, ids=[program_id(p) for p in LARGE])
def test_a_large_sum_is_exact_on_both_compilers(
    monkeypatch, empty_caches, expr, formats, dtype, regblock
):
    dense, built = operands_of(expr, formats, dtype, LARGE_EXTENT)
    program, moved = captured_program(monkeypatch, expr, built, options(regblock))
    shipped = shipped_einsum(expr, *built, _compile_options=options(regblock))
    loopir_options = replace(options(regblock), requested_schedule=Schedule())
    new, _ = execute_cin_via_loopir(
        program, result_shape_of(program), *moved, compile_options=loopir_options
    )

    check(shipped, expr, dense, dtype)
    check(new, expr, dense, dtype)


# -- matmul of two vectors, whose generated path is einsum("j,j->") ---------------------


@pytest.mark.parametrize("arm", ["0", "1"], ids=["regblock-off", "regblock-on"])
@pytest.mark.parametrize(
    "formats", [("d", "d"), ("d", "s"), ("s", "d"), ("s", "s")], ids="x".join
)
def test_matmul_of_two_vectors_is_their_dot_product(
    monkeypatch, empty_caches, formats, arm
):
    monkeypatch.setenv("SCORCH_REGBLOCK", arm)
    left, right = (operand_values("i", F64, seed) for seed in (31, 32))
    right[left != 0] += 1.0  # the vectors overlap, so the dot product is not 0
    expected = torch.dot(left, right)
    assert expected != 0

    result = scorch.matmul(
        stored(left, formats[0], "x"), stored(right, formats[1], "y")
    )

    # A rank-0 result is in the (vacuously dense) rank-0 format, so matmul returns it
    # as a torch.Tensor, as torch.matmul does.
    assert isinstance(result, torch.Tensor) and tuple(result.shape) == ()
    assert torch.allclose(result, expected, rtol=1e-12, atol=0.0), (result, expected)


# -- einsum's front end ---------------------------------------------------------------


def a_matrix():
    dense = operand_values("ij", F64, 3)
    return dense, stored(dense, "ss", "A")


@pytest.mark.parametrize(
    "requested", ["", [], TensorFormat("")], ids=("string", "list", "object")
)
def test_the_result_format_of_a_scalar_is_the_rank_zero_format(empty_caches, requested):
    dense, matrix = a_matrix()

    result = scorch.einsum("ij->", matrix, format=requested)

    check(result, "ij->", [dense], F64)


def test_a_result_format_of_another_rank_is_refused():
    _, matrix = a_matrix()

    with pytest.raises(
        CompileSpecError, match="format rank 1 does not match result rank 0"
    ):
        scorch.einsum("ij->", matrix, format="d")


def test_the_output_mode_order_of_a_scalar_is_empty(empty_caches):
    dense, matrix = a_matrix()

    check(scorch.einsum("ij->", matrix, output_mode_order=[]), "ij->", [dense], F64)
    with pytest.raises(CompileSpecError, match="permutation of the result modes"):
        scorch.einsum("ij->", matrix, output_mode_order=[0])


def test_compile_only_returns_the_scalar_specification(empty_caches):
    spec = scorch.einsum(
        "ij->", TensorSpec("ss", (5, 4), dtype=F64, name="A"), compile_only=True
    )

    assert isinstance(spec, TensorSpec)
    assert tuple(spec.shape) == ()
    assert spec.dtype == F64


def test_a_zero_dim_torch_operand_is_a_scalar_operand(empty_caches):
    scale = torch.tensor(1.75, dtype=F64)
    dense, matrix = a_matrix()

    result = scorch.einsum(",ij->ij", scale, matrix)

    check(result, ",ij->ij", [scale, dense], F64)


def test_scalar_terms_are_memoized_like_any_other():
    for expression, groups, result in (("ij->", ["ij"], ""), (",i->i", ["", "i"], "i")):
        ops._EXPR_LABELS_CACHE.pop(expression, None)
        first = ops._validated_labels(expression, groups, result)
        assert first == ([list(group) for group in groups], list(result))
        assert expression in ops._EXPR_LABELS_CACHE
        assert ops._validated_labels(expression, groups, result) == first


def test_a_result_label_that_is_not_a_letter_keeps_its_refusal():
    with pytest.raises(CompileSpecError, match="result labels must be ASCII") as e:
        ops._validated_labels("ij->i&", ["ij"], "i&")
    assert not isinstance(e.value, EinsumScalarResultError)


class _Reached(Exception):
    """Raised by a loop-order sort or a build that a refused call reached."""


def _reached(*args, **kwargs):
    raise _Reached


def nothing_past_the_label_check(monkeypatch):
    for module, name in [
        (ops, "topo_sort_characters"),
        (utils, "_prepare_jit_build"),
        (utils, "_load_validated_prepared_kernel"),
        (ops, "_prepare_jit_build"),
        (ops, "_load_validated_prepared_kernel"),
        (stensor, "_prepare_jit_build"),
        (stensor, "_load_validated_prepared_kernel"),
    ]:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _reached)


@ARMS
@pytest.mark.parametrize("terms", [1, 2, 3])
def test_an_expression_with_no_index_is_refused_before_any_build(
    monkeypatch, regblock, terms
):
    """``"->"``: every term is empty, so the product has no loop to lower."""

    expression = ",".join([""] * terms) + "->"
    scalars = [
        STensor.from_components((), "", [], torch.tensor([2.0], dtype=F64), name=name)
        for name in "ABC"[:terms]
    ]
    nothing_past_the_label_check(monkeypatch)
    for _ in range(2):
        with pytest.raises(EinsumScalarResultError) as raised:
            scorch.einsum(expression, *scalars, _compile_options=options(regblock))
        assert raised.value.code == "einsum_scalar_result_unsupported"
        assert isinstance(raised.value, CompileSpecError)
        assert repr(expression) in str(raised.value)
    assert expression not in ops._EXPR_LABELS_CACHE


# -- programs written in CIN ----------------------------------------------------------

SPELLINGS = pytest.mark.parametrize(
    "scalar_format",
    ("", [], TensorFormat()),
    ids=("empty-string", "empty-list", "object"),
)


def scalar_result(scalar_format):
    """``y[] = sum_ab A[a, b]``."""

    a, b = IndexVar("a"), IndexVar("b")
    operand = TensorVar("A", fmt="ss", dtype=F64)
    result = TensorVar("y", fmt=scalar_format, dtype=F64)
    assign = TensorAssign(result[()], operand[a, b], op=Operation.ADD)
    return ForAll(a, ForAll(b, assign)), (), (((3, 4), F64),)


def scalar_operand(scalar_format):
    """``y[a] = x[a] * s[]``."""

    a = IndexVar("a")
    vector = TensorVar("x", fmt="d", dtype=F64)
    scale = TensorVar("s", fmt=scalar_format, dtype=F64)
    result = TensorVar("y", fmt="d", dtype=F64)
    assign = TensorAssign(result[a], vector[a] * scale[()])
    return ForAll(a, assign), (3,), (((3,), F64), ((), F64))


def cin_operands(program):
    """The operands of ``program`` and the dense result it should compute."""

    matrix = operand_values("ij", F64, 5)[:3]
    vector = operand_values("k", F64, 6)
    scale = torch.tensor(1.75, dtype=F64)
    if program is scalar_result:
        return [stored(matrix, "ss", "A")], matrix.sum()
    return [stored(vector, "d", "x"), stored(scale, "", "s")], vector * scale


CIN_PROGRAMS = pytest.mark.parametrize(
    "program", (scalar_result, scalar_operand), ids=("scalar-result", "scalar-operand")
)


@SPELLINGS
def test_an_empty_format_is_the_rank_zero_format(scalar_format):
    scalar = TensorVar("y", fmt=scalar_format, dtype=F64)

    assert scalar.format is not None
    assert scalar.format.get_order() == 0
    assert scalar.mode_order == []


def test_no_format_is_still_no_format():
    unformatted = TensorVar("y", dtype=F64)

    assert unformatted.format is None
    assert unformatted.mode_order is None


@SPELLINGS
def test_the_shipped_lowering_reads_and_writes_position_zero(scalar_format):
    result_source = legacy_generated_cpp(*scalar_result(scalar_format))
    operand_source = legacy_generated_cpp(*scalar_operand(scalar_format))

    assert "int64_t y_capacity = 1;" in result_source
    assert "y_values[0] += A_val[pA1];" in result_source
    assert "s_val[0]" in operand_source


@ARMS
@SPELLINGS
@CIN_PROGRAMS
def test_both_compilers_compute_a_program_with_a_scalar_tensor(
    empty_caches, program, scalar_format, regblock
):
    statement, result_shape, _ = program(scalar_format)
    operands, expected = cin_operands(program)
    loopir_options = replace(options(regblock), requested_schedule=Schedule())

    # lower_and_exec_cin refuses a requested schedule, so the shipped side runs
    # unscheduled; the scheduled shipped route is the public einsum's, tested above.
    shipped = ops.lower_and_exec_cin(
        copy.deepcopy(statement),
        result_shape,
        *operands,
        _compile_options=options(regblock),
    )
    new, _ = execute_cin_via_loopir(
        statement, result_shape, *operands, compile_options=loopir_options
    )

    for result in (shipped, new):
        assert tuple(result.shape) == tuple(expected.shape)
        assert_matches(result, expected, F64)


def scalar_copy(scalar_format):
    """``y[] = s[]``: every tensor is rank 0, so there is no loop."""

    result = TensorVar("y", fmt=scalar_format, dtype=F64)
    scale = TensorVar("s", fmt=scalar_format, dtype=F64)
    return TensorAssign(result[()], scale[()]), (), (((), F64),)


@ARMS
@SPELLINGS
def test_a_program_without_a_loop_is_refused_by_code(scalar_format, regblock):
    statement, result_shape, bindings = scalar_copy(scalar_format)
    loopir_options = replace(options(regblock), requested_schedule=Schedule())

    with pytest.raises(UnsupportedFeature) as shipped:
        legacy_generated_cpp(
            copy.deepcopy(statement),
            result_shape,
            bindings,
            compile_options=options(regblock),
        )
    with pytest.raises(LoopIRLoweringError) as new:
        compile_cin_via_loopir(
            statement, result_shape, bindings, compile_options=loopir_options
        )

    (diagnostic,) = shipped.value.diagnostics
    assert diagnostic.code == "unsupported_scalar_tensor"
    assert diagnostic.stage == "legacy_cin_lowering"
    assert "tensor 'y'" in str(shipped.value)
    assert new.value.defect.code == "unsupported_statement"


@pytest.mark.parametrize(
    ("expr", "formats", "result_format", "marker"),
    [
        ("ab->a", ["ss"], "d", "B_values[pB0] += A_val[pA1];"),
        ("ab,bc->ac", ["ds", "ds"], "ds", "wksp.insert_unchecked({c}, "),
        ("ab,cb->ac", ["ds", "dd"], "dd", "double wksp[kTile_c] = {};"),
    ],
)
def test_programs_with_accumulators_still_lower(
    monkeypatch, expr, formats, result_format, marker
):
    """A kernel's scalar or workspace accumulator is not a rank-0 tensor, and the
    programs that use one lower as before."""

    extents = {name: 3 for name in expr if name.isalpha()}

    source = emitted_einsum_cpp(
        monkeypatch, expr, formats, extents, format=result_format
    )

    assert "Tensor evaluate(" in source
    assert marker in source
