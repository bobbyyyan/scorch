"""Phase-7 rank-1 compressed assembly: the degenerate ordered-stream family.

A rank-1 all-compressed result is the degenerate case of the
dense-prefix/multi-compressed-suffix family the ordered-assembly target
already owns: one stored stream, no dense prefix, and no parent level to
close.  Admitting it needed no new node kinds, no canonical-schema change,
and no request- or schedule-identity change — only the three places that
spelled "two or more compressed suffix levels" now also name the rank-1
degenerate case, stated through level identities alone.

Covered here: ``s`` copies, ``s+s`` ordered unions, ``s*s`` intersections,
and mixed ``s*d``/``d*s`` products, at f32 and f64, in both automatic
policy arms.

The legacy comparand is honest for this family (it generates C++ for every
cell, executes it, and produces well-formed identity-ordered storage), so
the gate is byte parity with ``legacy_generated_cpp`` in both arms plus the
LoopIR oracle and the PyTorch dense reference — the B1/B3 discipline.

Every excluded neighbour keeps its exact prior code: ``s-s`` stays at
``unsupported_sparse_subtraction`` (legacy cannot compile SUB at all),
``s+d`` at ``unsupported_union_with_dense``, and 3-ary chains at
``unsupported_program_shape``.  The rank-1 dense output ``s->d`` keeps its
existing admitted route.
"""

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.nodes import LevelKind
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.stensor import STensor
from scorch.storage import TensorIndex
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

N = 9

_OPS = {"mul": Operation.MUL, "add": Operation.ADD, "sub": Operation.SUB}


def build_rank1_cin(result_fmt, operand_fmts, op, dtype=torch.float32):
    index = IndexVar("i")
    result = TensorVar("C", fmt=result_fmt, dtype=dtype)
    value = None
    for position, fmt in enumerate(operand_fmts):
        access = TensorVar("ABD"[position], fmt=fmt, dtype=dtype)[index]
        value = access if value is None else CINBinaryOp(_OPS[op], value, access)
    return ForAll(index, TensorAssign(result[index], value))


def bindings_for(operand_fmts, dtype):
    return tuple(((N,), dtype) for _ in operand_fmts)


def fixtures(kind, dtype, seed=41):
    torch.manual_seed(seed)
    if kind == "disjoint":
        mask = torch.arange(N) % 2 == 0
        return (torch.randn(N) * mask).to(dtype), (torch.randn(N) * ~mask).to(dtype)
    if kind == "empty_a":
        return torch.zeros(N, dtype=dtype), torch.randn(N).to(dtype)
    if kind == "empty_b":
        return torch.randn(N).to(dtype), torch.zeros(N, dtype=dtype)
    if kind == "cancel":
        base = ((torch.rand(N) < 0.7) * torch.randn(N)).to(dtype)
        return base, -base.clone()
    if kind == "identical":
        base = ((torch.rand(N) < 0.5) * torch.randn(N)).to(dtype)
        return base, base.clone()
    if kind == "explicit_zero":
        return (
            torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]).to(dtype),
            torch.tensor([0.0, -1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]).to(dtype),
        )
    return (
        ((torch.rand(N) < 0.45) * torch.randn(N)).to(dtype),
        ((torch.rand(N) < 0.45) * torch.randn(N)).to(dtype),
    )


def operands_for(operand_fmts, dense_pair):
    tensors = []
    for position, fmt in enumerate(operand_fmts):
        tensor = STensor.from_torch(dense_pair[position].clone(), "ABD"[position])
        if fmt == "s":
            tensor.to_sparse("s")
        tensors.append(tensor)
    return tensors


def dense_reference(operand_fmts, dense_pair, op):
    expected = dense_pair[0].clone()
    for extra in dense_pair[1 : len(operand_fmts)]:
        expected = expected * extra if op == "mul" else expected + extra
    return expected


ADMITTED = [
    ("s copy", ("s",), "mul"),
    ("s+s union", ("s", "s"), "add"),
    ("s*s intersect", ("s", "s"), "mul"),
    ("s*d mixed", ("s", "d"), "mul"),
    ("d*s commuted", ("d", "s"), "mul"),
]

_KIND = {"d": LevelKind.DENSE, "s": LevelKind.COMPRESSED}


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("name,operand_fmts,op", ADMITTED, ids=[c[0] for c in ADMITTED])
def test_generated_source_matches_legacy_byte_for_byte(
    name, operand_fmts, op, dtype, regblock_enabled
):
    """The gate: byte parity with the legacy comparand in both arms."""

    comparison = compare_generated_sources(
        build_rank1_cin("s", operand_fmts, op, dtype),
        (N,),
        bindings_for(operand_fmts, dtype),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
@pytest.mark.parametrize("name,operand_fmts,op", ADMITTED, ids=[c[0] for c in ADMITTED])
def test_rank1_assembly_is_admitted_in_both_arms(
    name, operand_fmts, op, regblock_enabled
):
    kernel = compile_cin_via_loopir(
        build_rank1_cin("s", operand_fmts, op),
        (N,),
        bindings_for(operand_fmts, torch.float32),
        compile_options=auto_options(regblock_enabled),
    )
    assert kernel.cpp_source


@pytest.mark.parametrize(
    "kind",
    [
        "random",
        "disjoint",
        "empty_a",
        "empty_b",
        "cancel",
        "identical",
        "explicit_zero",
    ],
)
@pytest.mark.parametrize("name,operand_fmts,op", ADMITTED, ids=[c[0] for c in ADMITTED])
def test_compiled_execution_matches_the_dense_reference(name, operand_fmts, op, kind):
    """PyTorch differential over empty, disjoint, cancelling, and explicitly
    zero-valued supports."""

    dense_pair = fixtures(kind, torch.float32)
    out = execute_cin_via_loopir(
        build_rank1_cin("s", operand_fmts, op),
        (N,),
        *operands_for(operand_fmts, dense_pair),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    expected = dense_reference(operand_fmts, dense_pair, op)
    assert torch.allclose(result.to_torch(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name,operand_fmts,op", ADMITTED, ids=[c[0] for c in ADMITTED])
def test_float64_execution_matches_the_dense_reference(name, operand_fmts, op):
    dense_pair = fixtures("random", torch.float64)
    out = execute_cin_via_loopir(
        build_rank1_cin("s", operand_fmts, op, torch.float64),
        (N,),
        *operands_for(operand_fmts, dense_pair),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    expected = dense_reference(operand_fmts, dense_pair, op)
    assert torch.allclose(result.to_torch(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name,operand_fmts,op", ADMITTED, ids=[c[0] for c in ADMITTED])
def test_generated_program_matches_the_loopir_oracle(name, operand_fmts, op):
    dense_pair = fixtures("random", torch.float32)
    kernel = compile_cin_via_loopir(
        build_rank1_cin("s", operand_fmts, op),
        (N,),
        bindings_for(operand_fmts, torch.float32),
        compile_options=auto_options(False),
    )
    lowering = kernel.lowering
    inputs = {}
    for symbol, dense, fmt in zip(
        lowering.rhs_access_symbols, dense_pair, operand_fmts
    ):
        inputs[symbol] = (
            dense.tolist()
            if fmt == "d"
            else LevelTensorStorage.from_dense(
                dense.tolist(), (N,), (0,), (_KIND[fmt],)
            )
        )
    outputs = run_program(
        lowering.program,
        inputs,
        {lowering.result_symbol: (N,)},
    )
    produced = outputs[lowering.result_symbol]
    expected = dense_reference(operand_fmts, dense_pair, op)
    coordinates = produced.coordinates[0]
    assert produced.positions[0] == (0, len(coordinates))
    oracle_dense = torch.zeros(N)
    for position, coordinate in enumerate(coordinates):
        oracle_dense[coordinate] = produced.values[position]
    assert torch.allclose(oracle_dense, expected, atol=1e-6, rtol=1e-6)


def storage_pieces(result):
    positions, coordinates = result.storage.index.mode_indices[0]
    return positions.tolist(), coordinates.tolist()


def stored_vector(name, coordinates, values):
    index_dtype = torch.int32
    return STensor(
        name=name,
        shape=(N,),
        index=TensorIndex(
            "s",
            [
                [
                    torch.tensor([0, len(coordinates)], dtype=index_dtype),
                    torch.tensor(coordinates, dtype=index_dtype),
                ]
            ],
            mode_order=[0],
        ),
        value=torch.tensor(values, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "operand_fmts,op,operands,expected_coordinates,expected_values",
    [
        (("s",), "mul", ("a",), [1, 3], [0.0, 2.0]),
        (("s", "s"), "mul", ("a", "b_full"), [1, 3], [0.0, 14.0]),
        (("s", "s"), "add", ("a", "b_tail"), [1, 2, 3], [0.0, 5.0, 2.0]),
    ],
    ids=["copy", "intersection", "union"],
)
def test_hand_built_stored_zero_is_observed_and_retained(
    operand_fmts, op, operands, expected_coordinates, expected_values
):
    available = {
        "a": stored_vector("A", [1, 3], [0.0, 2.0]),
        "b_full": stored_vector("B", [1, 2, 3], [5.0, 6.0, 7.0]),
        "b_tail": stored_vector("B", [2], [5.0]),
    }
    out = execute_cin_via_loopir(
        build_rank1_cin("s", operand_fmts, op),
        (N,),
        *(available[key] for key in operands),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    _, coordinates = storage_pieces(result)
    assert coordinates == expected_coordinates
    assert result.storage.value.tolist() == expected_values


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
def test_zero_extent_union_is_byte_identical_and_canonical(regblock_enabled):
    cin = build_rank1_cin("s", ("s", "s"), "add")
    comparison = compare_generated_sources(
        cin,
        (0,),
        (((0,), torch.float32), ((0,), torch.float32)),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical

    empty_a = STensor.from_torch(torch.zeros(0), "A").to_sparse("s")
    empty_b = STensor.from_torch(torch.zeros(0), "B").to_sparse("s")
    out = execute_cin_via_loopir(
        cin,
        (0,),
        empty_a,
        empty_b,
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    assert storage_pieces(result) == ([0, 0], [])
    assert result.storage.value.numel() == 0


def test_assembled_storage_is_honest_and_identity_ordered():
    dense_pair = fixtures("random", torch.float32)
    out = execute_cin_via_loopir(
        build_rank1_cin("s", ("s", "s"), "add"),
        (N,),
        *operands_for(("s", "s"), dense_pair),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    positions, coordinates = storage_pieces(result)
    assert positions == [0, len(coordinates)]
    assert coordinates == sorted(set(coordinates))
    assert all(0 <= coordinate < N for coordinate in coordinates)
    assert result.storage.value.numel() == len(coordinates)


def test_union_stores_the_exact_ordered_support():
    dense_pair = fixtures("explicit_zero", torch.float32)
    out = execute_cin_via_loopir(
        build_rank1_cin("s", ("s", "s"), "add"),
        (N,),
        *operands_for(("s", "s"), dense_pair),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    _, coordinates = storage_pieces(result)
    expected_support = sorted(
        {index for index in range(N) if dense_pair[0][index] != 0}
        | {index for index in range(N) if dense_pair[1][index] != 0}
    )
    assert coordinates == expected_support


def test_cancellation_keeps_the_stored_explicit_zero():
    """A + (-A) unites to a stored zero, not to a dropped coordinate."""

    dense_pair = fixtures("cancel", torch.float32)
    out = execute_cin_via_loopir(
        build_rank1_cin("s", ("s", "s"), "add"),
        (N,),
        *operands_for(("s", "s"), dense_pair),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    _, coordinates = storage_pieces(result)
    support = sorted({index for index in range(N) if dense_pair[0][index] != 0})
    assert coordinates == support
    assert torch.allclose(result.storage.value, torch.zeros(len(support)))


def test_empty_operand_drains_the_other_stream():
    dense_pair = fixtures("empty_a", torch.float32)
    out = execute_cin_via_loopir(
        build_rank1_cin("s", ("s", "s"), "add"),
        (N,),
        *operands_for(("s", "s"), dense_pair),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    _, coordinates = storage_pieces(result)
    assert coordinates == sorted(
        {index for index in range(N) if dense_pair[1][index] != 0}
    )


def test_all_empty_operands_assemble_canonical_empty_storage():
    zeros = torch.zeros(N)
    out = execute_cin_via_loopir(
        build_rank1_cin("s", ("s", "s"), "add"),
        (N,),
        *operands_for(("s", "s"), (zeros.clone(), zeros.clone())),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    positions, coordinates = storage_pieces(result)
    assert positions == [0, 0]
    assert coordinates == []
    assert result.storage.value.numel() == 0


def test_repeated_compiled_execution_is_byte_stable():
    dense_pair = fixtures("random", torch.float32)
    snapshots = []
    for _ in range(3):
        out = execute_cin_via_loopir(
            build_rank1_cin("s", ("s", "s"), "add"),
            (N,),
            *operands_for(("s", "s"), dense_pair),
            compile_options=auto_options(False, jit=True),
        )
        result = out[0] if isinstance(out, tuple) else out
        snapshots.append((storage_pieces(result), result.storage.value.tolist()))
    assert snapshots[0] == snapshots[1] == snapshots[2]


NEIGHBOURS = [
    ("s-s SUB", ("s", "s"), "sub", "unsupported_sparse_subtraction"),
    ("s+d union-with-dense", ("s", "d"), "add", "unsupported_union_with_dense"),
    ("d+s union-with-dense", ("d", "s"), "add", "unsupported_union_with_dense"),
    ("3-ary s*s*s", ("s", "s", "s"), "mul", "unsupported_program_shape"),
    ("3-ary s+s+s", ("s", "s", "s"), "add", "unsupported_program_shape"),
]


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
@pytest.mark.parametrize(
    "name,operand_fmts,op,code", NEIGHBOURS, ids=[c[0] for c in NEIGHBOURS]
)
def test_excluded_neighbours_keep_their_exact_codes(
    name, operand_fmts, op, code, regblock_enabled
):
    with pytest.raises((LoopIRLoweringError, LoopIRTargetError)) as error:
        compile_cin_via_loopir(
            build_rank1_cin("s", operand_fmts, op),
            (N,),
            bindings_for(operand_fmts, torch.float32),
            compile_options=auto_options(regblock_enabled),
        )
    assert error.value.defect.code == code


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
def test_rank1_dense_output_keeps_its_existing_route(regblock_enabled):
    comparison = compare_generated_sources(
        build_rank1_cin("d", ("s",), "mul"),
        (N,),
        bindings_for(("s",), torch.float32),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical


@pytest.mark.parametrize(
    "result_fmt,operand_fmts,op",
    [
        ("ss", ("ss",), "mul"),
        ("ss", ("ss", "ss"), "add"),
        ("ss", ("ss", "dd"), "mul"),
        ("ds", ("ds", "ds"), "add"),
    ],
)
def test_rank2_families_are_byte_unchanged(result_fmt, operand_fmts, op):
    """The widening is inert on every rank>=2 family already admitted."""

    index_i, index_j = IndexVar("i"), IndexVar("j")
    result = TensorVar("C", fmt=result_fmt, dtype=torch.float32)
    value = None
    for position, fmt in enumerate(operand_fmts):
        access = TensorVar("ABD"[position], fmt=fmt, dtype=torch.float32)[
            index_i, index_j
        ]
        value = access if value is None else CINBinaryOp(_OPS[op], value, access)
    cin = ForAll(
        index_i, ForAll(index_j, TensorAssign(result[index_i, index_j], value))
    )
    comparison = compare_generated_sources(
        cin,
        (4, 5),
        tuple(((4, 5), torch.float32) for _ in operand_fmts),
        compile_options=auto_options(False),
    )
    assert comparison.identical


def test_rank1_assembly_needs_no_schedule_or_identity_change():
    """The family is admitted under the default automatic policy alone, and
    assembles the single compressed result level it declares."""

    options = CompileOptions.from_environment(environ={})
    kernel = compile_cin_via_loopir(
        build_rank1_cin("s", ("s", "s"), "add"),
        (N,),
        bindings_for(("s", "s"), torch.float32),
        compile_options=options,
    )
    source = kernel.cpp_source
    # One compressed result level is assembled; there is no second one and
    # no dense-prefix size initializer to emit.
    assert "C0_pos" in source and "C0_crd" in source
    assert "C1_pos" not in source and "C1_crd" not in source
    assert "Init result tensor level sizes" not in source
    # The ordered union is the three-case while-merge, not a dense scan.
    assert "while" in source
