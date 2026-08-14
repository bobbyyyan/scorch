"""Phase-7 parallel sparse-workspace target: the ds@ds->ds SpGEMM family.

The completed LoopIR target for the automatic dense-row CSR reduction
family must generate byte-identical C++ to the legacy two-phase OpenMP
count/fill pipeline in both automatic policy arms — the per-thread
``linked_list_workspace_1d`` pool sized by the derived SpGEMM thread
policy, borrowed per-worker ``make_view()`` lifetimes, two parallel phase
loops around the exact serial prefix-sum/Torch-allocation interlude, and
honest final assembly — execute through the shared JIT build path, and
return an honest dense-row CSR ``STensor`` whose storage matches the
production LoopIR oracle exactly (including explicit zeros) and the
PyTorch dense reference within the repository tolerance.

The sealed evidence comparand under
``~/.cache/scorch-codex/phase6-b1b2-ebb243b/phase7-comparands/`` is the
sized, sealed automatic capture this family is proven against; the tests
that consume it verify its recorded SHA-256 before use and skip when the
sealed evidence tree is absent (the in-repo legacy differential keeps
running everywhere).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

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
from scorch.compiler.compilation_context import (
    CompilationContext,
    CompilerStageId,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import (
    LoopIRTargetError,
    lower_loopir_to_llir,
)
from scorch.compiler.loopir.nodes import (
    BinaryOp as LoopIRBinaryOp,
    LevelKind,
    ReduceOp,
    ScalarType,
)
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import (
    erase_schedule,
    verify_scheduled_loopir,
)
from scorch.stensor import STensor
from tests.test_scorch.test_loopir_sparse_workspace_target import (
    _first_tensor_access_metadata,
    auto_options,
    execute_legacy_module,
)

_DS_KINDS = (LevelKind.DENSE, LevelKind.COMPRESSED)

_SEALED_COMPARAND_DIR = Path(
    "~/.cache/scorch-codex/phase6-b1b2-ebb243b/phase7-comparands"
).expanduser()
_SEALED_SPGEMM_SHA256 = (
    "fa1026bec895f9fcebf212babea4fa2fa02e610372e6ff8a494e80e303a4791c"
)


def build_spgemm_cin(
    dtype=torch.float32,
    *,
    commuted=False,
    index_names=("i", "j", "k"),
):
    i, j, k = (IndexVar(name) for name in index_names)
    a = TensorVar("A", fmt="ds", dtype=dtype)
    b = TensorVar("B", fmt="ds", dtype=dtype)
    c = TensorVar("C", fmt="ds", dtype=dtype)
    value = (
        CINBinaryOp(Operation.MUL, b[j, k], a[i, j])
        if commuted
        else CINBinaryOp(Operation.MUL, a[i, j], b[j, k])
    )
    assign = TensorAssign(c[i, k], value, op=Operation.ADD)
    return ForAll(i, ForAll(j, ForAll(k, assign)))


def sparse_ds(dense, name):
    return STensor.from_torch(dense.clone(), name).to_sparse("ds")


def validated_ds_storage(result, shape):
    """Assert honest identity-ordered ``ds`` storage; return its pieces."""

    assert str(result.index.format) == "d,s"
    assert tuple(result.index.mode_order) == (0, 1)
    assert tuple(result.shape) == tuple(shape)
    mode_indices = result.index.mode_indices
    assert len(mode_indices) == 2
    assert list(mode_indices[0]) == []
    assert len(mode_indices[1]) == 2
    pos1 = mode_indices[1][0].tolist()
    crd1 = mode_indices[1][1].tolist()
    values = result.values.tolist()
    assert len(pos1) == shape[0] + 1
    assert pos1[0] == 0
    assert pos1 == sorted(pos1), "row positions must be nondecreasing"
    assert pos1[-1] == len(crd1) == len(values)
    for row in range(shape[0]):
        columns = crd1[pos1[row] : pos1[row + 1]]
        assert columns == sorted(
            set(columns)
        ), "column coordinates must strictly increase per row segment"
        assert all(0 <= column < shape[1] for column in columns)
    return pos1, crd1, values


def dense_from_ds_storage(storage, shape, dtype):
    pos1, crd1, values = storage
    dense = torch.zeros(shape, dtype=dtype)
    for row in range(shape[0]):
        for position in range(pos1[row], pos1[row + 1]):
            dense[row, crd1[position]] = values[position]
    return dense


def oracle_result(kernel, dense_a, dense_b, result_shape):
    """Run the scheduled program through the production LoopIR oracle."""

    lowering = kernel.lowering
    assert kernel.schedule is not None
    inputs = {}
    for symbol, dense in zip(lowering.rhs_access_symbols, (dense_a, dense_b)):
        inputs[symbol] = LevelTensorStorage.from_dense(
            dense.tolist(), tuple(dense.shape), (0, 1), _DS_KINDS
        )
    outputs = run_program(
        kernel.schedule.program,
        inputs,
        {lowering.result_symbol: tuple(result_shape)},
    )
    base_outputs = run_program(
        lowering.program,
        inputs,
        {lowering.result_symbol: tuple(result_shape)},
    )
    scheduled = outputs[lowering.result_symbol]
    assert scheduled == base_outputs[lowering.result_symbol]
    return scheduled


def assert_storage_matches_oracle(storage, oracle):
    pos1, crd1, values = storage
    assert tuple(pos1) == oracle.indptr
    assert tuple(crd1) == oracle.indices
    assert len(values) == len(oracle.values)
    for got, expected in zip(values, oracle.values):
        assert got == pytest.approx(expected, abs=1e-5, rel=1e-5)


def execute_spgemm(cin, result_shape, st_a, st_b, regblock_enabled):
    return execute_cin_via_loopir(
        cin,
        result_shape,
        st_a,
        st_b,
        compile_options=auto_options(regblock_enabled, jit=True),
    )


# -- source parity ------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_source_parity_matches_legacy_in_both_arms(dtype):
    sources = {}
    for regblock_enabled in (False, True):
        comparison = compare_generated_sources(
            build_spgemm_cin(dtype),
            (4, 5),
            (((4, 6), dtype), ((6, 5), dtype)),
            compile_options=auto_options(regblock_enabled),
        )
        assert (
            comparison.identical
        ), f"regblock={regblock_enabled} LoopIR C++ diverges from legacy"
        sources[regblock_enabled] = comparison.loopir_cpp
    assert sources[False] == sources[True], "the two policy arms must agree"
    element = "float" if dtype is torch.float32 else "double"
    cpp = sources[False]
    assert f"std::vector<linked_list_workspace_1d<{element}>> wksp_pool;" in cpp
    assert "wksp_pool[(size_t)omp_get_thread_num()].make_view()" in cpp
    assert "wksp.insert_unchecked({k}, A_val[pA1] * B_val[pB1]);" in cpp
    assert cpp.count("#pragma omp parallel num_threads(scorch_nthreads(") == 2
    # Three thread-count sites (pool sizing plus both phase regions) and
    # two dynamic-chunk sites carry the SpGEMM work grain.
    assert cpp.count("SCORCH_GRAIN_CODEGEN_SPGEMM") == 5
    assert "std::vector<int> _count1((size_t)A0_size, 0);" in cpp
    assert "C1_pos_torch = torch::empty({(int64_t)(A0_size + 1)}, torch::kInt);" in (
        cpp
    )
    assert "wksp.sort();" in cpp
    assert "wksp.clear();" in cpp


def test_generated_source_matches_sealed_comparand_hash():
    """The exact sized request reproduces the sealed automatic capture."""

    for regblock_enabled in (False, True):
        kernel = compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(regblock_enabled),
        )
        digest = hashlib.sha256(kernel.cpp_source.encode("utf-8")).hexdigest()
        assert digest == _SEALED_SPGEMM_SHA256, (
            f"regblock={regblock_enabled} source digest {digest} diverges "
            "from the sealed comparand"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_commuted_rhs_source_parity(regblock_enabled, dtype):
    """The target classifies operands by cursor role, not tuple position."""

    comparison = compare_generated_sources(
        build_spgemm_cin(dtype, commuted=True),
        (4, 5),
        # RHS discovery follows the source expression: B, then A.
        (((6, 5), dtype), ((4, 6), dtype)),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_commuted_rhs_executes_against_oracle_and_pytorch(dtype):
    torch.manual_seed(20260802)
    dense_a = ((torch.rand(4, 6) < 0.5) * torch.randn(4, 6)).to(dtype)
    dense_b = ((torch.rand(6, 5) < 0.5) * torch.randn(6, 5)).to(dtype)
    result, kernel = execute_cin_via_loopir(
        build_spgemm_cin(dtype, commuted=True),
        (4, 5),
        # The module ABI follows the commuted RHS discovery order.
        sparse_ds(dense_b, "B"),
        sparse_ds(dense_a, "A"),
        compile_options=auto_options(False, jit=True),
    )
    storage = validated_ds_storage(result, (4, 5))
    dense_result = dense_from_ds_storage(storage, (4, 5), dtype)
    assert torch.allclose(dense_result, dense_a @ dense_b, atol=1e-3, rtol=1e-3)
    assert_storage_matches_oracle(
        storage,
        oracle_result(kernel, dense_b, dense_a, (4, 5)),
    )


# -- route ownership ----------------------------------------------------------


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_completed_target_owns_the_pipeline_route(regblock_enabled):
    """The family compiles end to end: schedule applied, emission recorded."""

    options = auto_options(regblock_enabled)
    context = CompilationContext(options)
    kernel = compile_cin_via_loopir(
        build_spgemm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=options,
        compilation_context=context,
    )
    stages = {record.stage_id for record in context.stage_run_records}
    assert CompilerStageId.CIN_TO_LOOPIR_LOWERING in stages
    assert CompilerStageId.LOOPIR_SCHEDULE_APPLICATION in stages
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING in stages
    assert kernel.schedule is not None
    verify_scheduled_loopir(kernel.schedule)
    assert '"kind":"sparse_workspace_region"' in kernel.program_dump
    assert '"schema":"scorch.loopir.canonical.v12"' in kernel.program_dump
    assert kernel.cpp_source == legacy_generated_cpp(
        build_spgemm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(regblock_enabled),
    )

    replay = compile_cin_via_loopir(
        build_spgemm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(regblock_enabled),
    )
    assert replay.cpp_source == kernel.cpp_source
    assert replay.request_identity == kernel.request_identity
    assert replay.program_dump == kernel.program_dump


def test_unscheduled_route_keeps_its_boundary():
    """Without the automatic plan the semantic base program stays rejected."""

    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=options,
        )
    assert error.value.defect.code == "unsupported_program_shape"


def test_canonical_dump_stability_and_erasure():
    dumps = set()
    for regblock_enabled in (False, True):
        kernel = compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(regblock_enabled),
        )
        assert kernel.schedule is not None
        dumps.add(canonical_program_dump(kernel.schedule.program))
        erased = erase_schedule(kernel.schedule.program)
        assert canonical_program_dump(erased) == canonical_program_dump(
            kernel.schedule.base_program
        )
    assert len(dumps) == 1


# -- compiled execution ---------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_execution_matches_every_reference(regblock_enabled, dtype):
    torch.manual_seed(20260730)
    dense_a = torch.zeros((5, 6), dtype=dtype)
    dense_b = torch.zeros((6, 4), dtype=dtype)
    mask_a = torch.rand(5, 6) < 0.45
    mask_b = torch.rand(6, 4) < 0.55
    dense_a[mask_a] = torch.randn(int(mask_a.sum()), dtype=dtype)
    dense_b[mask_b] = torch.randn(int(mask_b.sum()), dtype=dtype)

    result, kernel = execute_spgemm(
        build_spgemm_cin(dtype),
        (5, 4),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        regblock_enabled,
    )
    storage = validated_ds_storage(result, (5, 4))
    dense_result = dense_from_ds_storage(storage, (5, 4), dtype)
    expected = dense_a @ dense_b
    tolerance = 1e-3
    assert torch.allclose(dense_result, expected, atol=tolerance, rtol=tolerance)
    assert_storage_matches_oracle(
        storage, oracle_result(kernel, dense_a, dense_b, (5, 4))
    )

    # Safe legacy execution: independently generate the exact request's
    # legacy source, prove parity, then compile that source rather than the
    # candidate a second time.
    legacy_cpp = legacy_generated_cpp(
        build_spgemm_cin(dtype),
        (5, 4),
        (((5, 6), dtype), ((6, 4), dtype)),
        compile_options=auto_options(regblock_enabled),
    )
    assert legacy_cpp == kernel.cpp_source
    legacy_raw = execute_legacy_module(
        legacy_cpp,
        (5, 4),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        auto_options(regblock_enabled, jit=True),
    )
    legacy_modes = legacy_raw.storage.index.mode_indices
    assert list(legacy_modes[0]) == []
    assert legacy_modes[1][0].tolist() == storage[0]
    assert legacy_modes[1][1].tolist() == storage[1]
    assert legacy_raw.storage.value.tolist() == pytest.approx(storage[2])

    # Independent production execution through the public dispatch.
    import scorch

    public = scorch.matmul(sparse_ds(dense_a, "A"), sparse_ds(dense_b, "B"))
    public_dense = public.to_torch().to(dtype)
    assert torch.allclose(public_dense, expected, atol=tolerance, rtol=tolerance)


@torch.no_grad()
def test_sealed_comparand_executes_identically():
    """The sealed sized automatic comparand is a sound execution oracle."""

    comparand_path = _SEALED_COMPARAND_DIR / "spgemm_ds_ds_ds_armBASE.cpp"
    if not comparand_path.exists():
        pytest.skip("sealed Phase-7 comparand evidence tree is absent")
    sealed = comparand_path.read_bytes()
    digest = hashlib.sha256(sealed).hexdigest()
    assert digest == _SEALED_SPGEMM_SHA256, "sealed comparand digest changed"

    torch.manual_seed(20260731)
    dense_a = (torch.rand(4, 6) < 0.5) * torch.randn(4, 6)
    dense_b = (torch.rand(6, 5) < 0.6) * torch.randn(6, 5)
    result, _ = execute_spgemm(
        build_spgemm_cin(),
        (4, 5),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        False,
    )
    storage = validated_ds_storage(result, (4, 5))

    sealed_raw = execute_legacy_module(
        sealed.decode("utf-8"),
        (4, 5),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        auto_options(False, jit=True),
    )
    sealed_modes = sealed_raw.storage.index.mode_indices
    assert list(sealed_modes[0]) == []
    assert sealed_modes[1][0].tolist() == storage[0]
    assert sealed_modes[1][1].tolist() == storage[1]
    assert sealed_raw.storage.value.tolist() == pytest.approx(storage[2])


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_empty_inputs_rows_and_disjoint_supports(regblock_enabled):
    zero_a = torch.zeros(4, 6)
    zero_b = torch.zeros(6, 5)
    dense_b = (torch.rand(6, 5) < 0.7) * torch.randn(6, 5)

    result, _ = execute_spgemm(
        build_spgemm_cin(),
        (4, 5),
        sparse_ds(zero_a, "A"),
        sparse_ds(dense_b, "B"),
        regblock_enabled,
    )
    pos1, crd1, values = validated_ds_storage(result, (4, 5))
    assert pos1 == [0] * 5 and crd1 == [] and values == []

    dense_a = (torch.rand(4, 6) < 0.7) * torch.randn(4, 6)
    result, _ = execute_spgemm(
        build_spgemm_cin(),
        (4, 5),
        sparse_ds(dense_a, "A"),
        sparse_ds(zero_b, "B"),
        regblock_enabled,
    )
    pos1, crd1, values = validated_ds_storage(result, (4, 5))
    assert pos1 == [0] * 5 and crd1 == [] and values == []

    # Disjoint supports: A only reads columns whose B rows are empty.
    disjoint_a = torch.zeros(4, 6)
    disjoint_a[:, :3] = torch.randn(4, 3)
    disjoint_b = torch.zeros(6, 5)
    disjoint_b[3:, :] = torch.randn(3, 5)
    result, kernel = execute_spgemm(
        build_spgemm_cin(),
        (4, 5),
        sparse_ds(disjoint_a, "A"),
        sparse_ds(disjoint_b, "B"),
        regblock_enabled,
    )
    pos1, crd1, values = validated_ds_storage(result, (4, 5))
    assert pos1 == [0] * 5 and crd1 == [] and values == []
    assert_storage_matches_oracle(
        (pos1, crd1, values),
        oracle_result(kernel, disjoint_a, disjoint_b, (4, 5)),
    )


@torch.no_grad()
@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [((0, 6), (6, 5)), ((4, 0), (0, 5)), ((4, 6), (6, 0))],
)
def test_zero_extent_cells(a_shape, b_shape):
    dense_a = torch.zeros(a_shape)
    dense_b = torch.zeros(b_shape)
    result_shape = (a_shape[0], b_shape[1])
    result, kernel = execute_spgemm(
        build_spgemm_cin(),
        result_shape,
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        False,
    )
    pos1, crd1, values = validated_ds_storage(result, result_shape)
    assert pos1 == [0] * (result_shape[0] + 1)
    assert crd1 == [] and values == []
    assert_storage_matches_oracle(
        (pos1, crd1, values),
        oracle_result(kernel, dense_a, dense_b, result_shape),
    )


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_cancellation_retains_explicit_zeros(regblock_enabled):
    """An exact additive cancellation stays a stored explicit zero."""

    dense_a = torch.zeros(2, 3)
    dense_a[0, 0] = 1.0
    dense_a[0, 1] = 1.0
    dense_b = torch.zeros(3, 2)
    dense_b[0, 0] = 2.0
    dense_b[1, 0] = -2.0
    dense_b[1, 1] = 3.0
    result, kernel = execute_spgemm(
        build_spgemm_cin(),
        (2, 2),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        regblock_enabled,
    )
    pos1, crd1, values = validated_ds_storage(result, (2, 2))
    assert pos1 == [0, 2, 2]
    assert crd1 == [0, 1]
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(3.0)
    assert_storage_matches_oracle(
        (pos1, crd1, values),
        oracle_result(kernel, dense_a, dense_b, (2, 2)),
    )


@torch.no_grad()
def test_ragged_and_overlapping_supports():
    """Very ragged rows and heavily overlapping supports stay exact."""

    torch.manual_seed(20260801)
    dense_a = torch.zeros(6, 8)
    dense_a[0, :] = torch.randn(8)  # full row
    dense_a[2, 3] = torch.randn(1)  # singleton row
    dense_a[5, ::2] = torch.randn(4)  # strided row
    dense_b = (torch.rand(8, 7) < 0.8) * torch.randn(8, 7)
    result, kernel = execute_spgemm(
        build_spgemm_cin(),
        (6, 7),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        False,
    )
    storage = validated_ds_storage(result, (6, 7))
    dense_result = dense_from_ds_storage(storage, (6, 7), torch.float32)
    assert torch.allclose(dense_result, dense_a @ dense_b, atol=1e-3, rtol=1e-3)
    assert_storage_matches_oracle(
        storage, oracle_result(kernel, dense_a, dense_b, (6, 7))
    )


@torch.no_grad()
def test_deterministic_storage_and_repeated_execution():
    torch.manual_seed(20260803)
    dense_a = (torch.rand(5, 6) < 0.6) * torch.randn(5, 6)
    dense_b = (torch.rand(6, 4) < 0.6) * torch.randn(6, 4)
    first, _ = execute_spgemm(
        build_spgemm_cin(),
        (5, 4),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        False,
    )
    second, _ = execute_spgemm(
        build_spgemm_cin(),
        (5, 4),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        False,
    )
    assert validated_ds_storage(first, (5, 4)) == validated_ds_storage(second, (5, 4))


_THREAD_PROBE = """
import json
import torch
from dataclasses import replace
from scorch.compiler.cin import BinaryOp, ForAll, IndexVar, Operation, \
    TensorAssign, TensorVar
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.scheduler import Schedule
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.stensor import STensor

i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
a = TensorVar("A", fmt="ds")
b = TensorVar("B", fmt="ds")
c = TensorVar("C", fmt="ds")
cin = ForAll(i, ForAll(j, ForAll(k, TensorAssign(
    c[i, k], BinaryOp(Operation.MUL, a[i, j], b[j, k]), op=Operation.ADD))))
torch.manual_seed(20260804)
dense_a = (torch.rand(16, 12) < 0.5) * torch.randn(16, 12)
dense_b = (torch.rand(12, 9) < 0.5) * torch.randn(12, 9)
options = replace(
    CompileOptions.from_environment().with_regblock_enabled(False),
    requested_schedule=Schedule(),
)
result, _ = execute_cin_via_loopir(
    cin,
    (16, 9),
    STensor.from_torch(dense_a, "A").to_sparse("ds"),
    STensor.from_torch(dense_b, "B").to_sparse("ds"),
    compile_options=options,
)
mode_indices = result.index.mode_indices
print(json.dumps({
    "pos1": mode_indices[1][0].tolist(),
    "crd1": mode_indices[1][1].tolist(),
    "values": result.values.tolist(),
}))
"""


@torch.no_grad()
def test_thread_count_invariance_and_race_freedom():
    """Distinct OpenMP pools produce byte-identical deterministic storage.

    Two fresh processes run the same activating execution under different
    ``OMP_NUM_THREADS`` settings; the count/fill form must produce exactly
    the same positions, coordinates, and value bytes either way.  The
    parent process compiles the kernel first so both children share the
    session's isolated JIT cache.
    """

    torch.manual_seed(20260804)
    dense_a = (torch.rand(16, 12) < 0.5) * torch.randn(16, 12)
    dense_b = (torch.rand(12, 9) < 0.5) * torch.randn(12, 9)
    execute_spgemm(
        build_spgemm_cin(),
        (16, 9),
        sparse_ds(dense_a, "A"),
        sparse_ds(dense_b, "B"),
        False,
    )

    outputs = []
    for thread_count in ("1", "3"):
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = thread_count
        completed = subprocess.run(
            [sys.executable, "-c", _THREAD_PROBE],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        outputs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]
    torch.manual_seed(20260804)
    reference_a = (torch.rand(16, 12) < 0.5) * torch.randn(16, 12)
    reference_b = (torch.rand(12, 9) < 0.5) * torch.randn(12, 9)
    dense_result = dense_from_ds_storage(
        (outputs[0]["pos1"], outputs[0]["crd1"], outputs[0]["values"]),
        (16, 9),
        torch.float32,
    )
    assert torch.allclose(dense_result, reference_a @ reference_b, atol=1e-3, rtol=1e-3)


# -- adversarial target boundaries ---------------------------------------------


def build_parallel_workspace_program(insert_value_builder=None):
    """Hand-build the exact scheduled chain; optionally forge the insert."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    dim_k = builder.dimension("k")
    symbol_a = builder.new_symbol_id()
    symbol_b = builder.new_symbol_id()
    symbol_c = builder.new_symbol_id()

    def ds_tensor(symbol, name, dims):
        return builder.tensor(
            symbol,
            name,
            ScalarType.FLOAT32,
            dims,
            (
                builder.level(LevelKind.DENSE, 0),
                builder.level(LevelKind.COMPRESSED, 1),
            ),
        )

    decl_a = ds_tensor(symbol_a, "A", (dim_i.dimension, dim_j.dimension))
    decl_b = ds_tensor(symbol_b, "B", (dim_j.dimension, dim_k.dimension))
    decl_c = ds_tensor(symbol_c, "C", (dim_i.dimension, dim_k.dimension))

    index_i = builder.new_index_id()
    index_j = builder.new_index_id()
    index_k = builder.new_index_id()
    position_a1 = builder.new_position_id()
    position_b1 = builder.new_position_id()
    cursor_a1 = builder.sparse_cursor(
        builder.new_cursor_id(),
        symbol_a,
        1,
        builder.dense_position(
            symbol_a, 0, builder.root_position(), builder.index_value(index_i)
        ),
    )
    cursor_b1 = builder.sparse_cursor(
        builder.new_cursor_id(),
        symbol_b,
        1,
        builder.dense_position(
            symbol_b, 0, builder.root_position(), builder.index_value(index_j)
        ),
    )
    workspace = builder.new_workspace_id()
    workspace_decl = builder.sparse_workspace_decl(
        workspace, "wksp", ScalarType.FLOAT32, (dim_k.dimension,)
    )
    if insert_value_builder is None:
        insert_value = builder.binary(
            LoopIRBinaryOp.MUL,
            builder.cursor_value(cursor_a1.cursor),
            builder.cursor_value(cursor_b1.cursor),
        )
    else:
        insert_value = insert_value_builder(builder, cursor_a1, cursor_b1)
    insert = builder.sparse_workspace_insert(
        workspace,
        (builder.index_value(index_k),),
        ReduceOp.ADD,
        insert_value,
    )
    child = builder.sparse_for(
        cursor_b1, position_b1, index_k, builder.block((insert,))
    )
    producer = builder.sparse_for(
        cursor_a1, position_a1, index_j, builder.block((child,))
    )
    drain_index = builder.new_index_id()
    append = builder.append_entry(
        symbol_c,
        (builder.index_value(index_i), builder.index_value(drain_index)),
        builder.sparse_workspace_value(workspace),
    )
    drain = builder.sparse_workspace_drain_for(
        workspace, (drain_index,), builder.block((append,))
    )
    region = builder.sparse_workspace_region(
        workspace_decl,
        builder.block((producer,)),
        builder.block((drain,)),
    )
    outer = builder.dense_for(index_i, dim_i.dimension, builder.block((region,)))
    return builder.program(
        (dim_i, dim_j, dim_k),
        (decl_a, decl_b, decl_c),
        (symbol_a, symbol_b),
        (symbol_c,),
        builder.block((outer,)),
    ), (symbol_a, symbol_b, symbol_c)


def parallel_workspace_shapes(symbols):
    symbol_a, symbol_b, _ = symbols
    return {symbol_a: (4, 6), symbol_b: (6, 5)}


def test_hand_built_program_lowers_and_matches_pipeline_source():
    program, symbols = build_parallel_workspace_program()
    options = CompileOptions.from_environment(environ={})
    function = lower_loopir_to_llir(
        program,
        input_shapes=parallel_workspace_shapes(symbols),
        result_shape=(4, 5),
        compile_options=options,
    )
    from scorch.ops import _lower_generated_llir

    cpp = _lower_generated_llir(function, options, CompilationContext(options))
    kernel = compile_cin_via_loopir(
        build_spgemm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert cpp == kernel.cpp_source


def test_forged_insert_value_fails_closed():
    program, symbols = build_parallel_workspace_program(
        insert_value_builder=lambda builder, *_: builder.float_const(2.0)
    )
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes=parallel_workspace_shapes(symbols),
            result_shape=(4, 5),
            compile_options=CompileOptions.from_environment(environ={}),
        )
    assert error.value.defect.code == "unsupported_program_shape"


@pytest.mark.parametrize(
    ("reserved_name", "defect_code"),
    [
        ("__asm", "invalid_display_name"),
        ("_Implementation", "invalid_display_name"),
        ("trailing_", "invalid_display_name"),
        ("linked_list_workspace_1d", "generated_name_collision"),
        ("omp_get_thread_num", "generated_name_collision"),
        ("SCORCH_GRAIN_CODEGEN_SPGEMM", "generated_name_collision"),
        ("coo_workspace_1d", "generated_name_collision"),
        ("int64_t", "generated_name_collision"),
    ],
)
def test_runtime_and_type_names_cannot_shadow_emission(
    reserved_name,
    defect_code,
):
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(
                index_names=(reserved_name, "reduction", "column"),
            ),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert error.value.defect.code == defect_code


@pytest.mark.parametrize(
    "generated_name",
    ["_count1", "_offset1", "_total1", "_worker", "_cnt1", "_base1", "_pos1"],
)
def test_two_phase_helper_names_stay_reserved(generated_name):
    """Display names cannot shadow the two-phase assembly's own locals.

    The helper spellings begin with an underscore, so any user display
    name colliding with them is already outside the safe-identifier
    boundary; the reservation still exists for defense in depth.
    """

    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(
                index_names=(generated_name, "reduction", "column"),
            ),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert error.value.defect.code == "invalid_display_name"


def test_pass_ownership_loss_fails_closed(monkeypatch):
    """A detached no-op of the shared two-phase pass cannot degrade the
    family to an unallocated serial assembly."""

    import scorch.compiler.compressed_where_openmp_pass as two_phase

    monkeypatch.setattr(
        two_phase,
        "_find_outer_loop",
        lambda statements: (None, None),
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def _is_phase_loop(statement):
    from scorch.compiler import llir

    return type(statement) is llir.ForLoop and statement.omp_parallel_for


@pytest.mark.parametrize(
    "tamper",
    [
        "drop_count_loop",
        "duplicate_fill_loop",
        "alias_fill_loop",
        "swap_fill_drain_writes",
        "drop_phase_clear",
        "wrap_pool_init",
        "drop_validation",
        "forged_thread_policy",
        "hoisted_second_region",
    ],
)
def test_two_phase_completion_attacks_fail_closed(monkeypatch, tamper):
    """Every malformed post-pass state dies at the exact completion boundary.

    The tamper hook rides the final managed pass, so it observes the
    assembled two-phase function statements exactly as completion will.
    """

    from copy import deepcopy

    import scorch.compiler.llir_pass_manager as pass_manager
    from scorch.compiler import llir

    original = pass_manager.rewrite_dynamic_vector_accesses
    state = {"done": False}

    def tampering(value, context):
        rewritten = original(value, context)
        if state["done"] or type(rewritten) is not list:
            return rewritten
        phase_positions = [
            index
            for index, statement in enumerate(rewritten)
            if _is_phase_loop(statement)
        ]
        if len(phase_positions) != 2:
            return rewritten
        count_position, fill_position = phase_positions
        if tamper == "drop_count_loop":
            rewritten.pop(count_position)
        elif tamper == "duplicate_fill_loop":
            rewritten.insert(fill_position + 1, deepcopy(rewritten[fill_position]))
        elif tamper == "alias_fill_loop":
            rewritten.insert(fill_position + 1, rewritten[fill_position])
        elif tamper == "swap_fill_drain_writes":
            fill = rewritten[fill_position]
            drain = next(
                statement
                for statement in fill.body
                if type(statement) is llir.ForLoopAuto
            )
            writes = [
                index
                for index, statement in enumerate(drain.body)
                if type(statement) is llir.Assign
            ]
            assert len(writes) == 2
            first, second = writes
            drain.body[first], drain.body[second] = (
                drain.body[second],
                drain.body[first],
            )
        elif tamper == "drop_phase_clear":
            fill = rewritten[fill_position]
            clear_position = next(
                index
                for index, statement in enumerate(fill.body)
                if type(statement) is llir.MemberCallStmt
                and statement.member == "clear"
            )
            fill.body.pop(clear_position)
        elif tamper == "wrap_pool_init":
            pool_position = next(
                index
                for index, statement in enumerate(rewritten)
                if type(statement) is llir.VarDecl and statement.var.name == "wksp_pool"
            )
            rewritten[pool_position] = llir.IfThenElse(
                cond=llir.Literal(True),
                then_body=[rewritten[pool_position]],
            )
        elif tamper == "drop_validation":
            removed = rewritten.pop(0)
            assert type(removed) is llir.FunctionCallStmt
            assert removed.name == "scorch_native::validate_jit_result_shape"
        elif tamper == "forged_thread_policy":
            rewritten[count_position].omp_num_threads = "scorch_nthreads(-1, A0_size)"
        else:
            rewritten[fill_position].pre_parallel_body = None
        state["done"] = True
        return rewritten

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        tampering,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert state["done"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_pipeline_entry_header_mutation_fails_closed(monkeypatch):
    """In-place edits to the emitted serial header cannot become expected.

    The first managed pass receives the emitted statement objects
    themselves; a hostile in-place widening of the serial row bound flows
    into both derived phase headers, and the freshly reconstructed
    completion reference rejects them.
    """

    import scorch.compiler.llir_pass_manager as pass_manager
    from scorch.compiler import llir

    original = pass_manager.LLIRPassManager.run_compressed_where_openmp
    state = {"mutated": False}

    def hostile_entry(self, artifact, pass_spec):
        for statement in artifact.statements:
            if (
                type(statement) is llir.ForLoop
                and getattr(statement, "scorch_index_var", None) == "i"
                and not state["mutated"]
            ):
                statement.cond = llir.BinOp(
                    op="<=",
                    left=statement.cond.left,
                    right=statement.cond.right,
                )
                state["mutated"] = True
        return original(self, artifact, pass_spec)

    monkeypatch.setattr(
        pass_manager.LLIRPassManager,
        "run_compressed_where_openmp",
        hostile_entry,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert state["mutated"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_dynamic_vector_rewrite_is_idempotent_on_two_phase(monkeypatch):
    """A duplicated trailing managed pass leaves the completed form intact."""

    import scorch.compiler.llir_pass_manager as pass_manager

    original = pass_manager.rewrite_dynamic_vector_accesses

    def doubled(value, context):
        return original(original(value, context), context)

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        doubled,
    )
    kernel = compile_cin_via_loopir(
        build_spgemm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        original,
    )
    assert kernel.cpp_source == legacy_generated_cpp(
        build_spgemm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(False),
    )


def test_completion_matcher_compares_shared_metadata_by_value():
    """Frozen access provenance is value state, never censused ownership."""

    from scorch.compiler import llir
    from scorch.compiler.identity import AccessId, IndexId, SymbolId
    from scorch.compiler.loopir import lower_llir as target

    def metadata(access=7):
        return llir.TensorAccessMetadata(
            access_id=AccessId(access),
            tensor_id=SymbolId(1),
            index_ids=(IndexId(2), IndexId(3)),
            role=llir.TensorAccessRole.INPUT_READ,
        )

    def access(meta):
        return llir.ArrayAccess(
            array=llir.Var("A_val", llir.DataType.PTR_FLOAT32),
            index=llir.Var("pA1", llir.DataType.INT),
            tensor_access=meta,
        )

    shared = metadata()
    actual = [access(shared), access(shared)]
    expected = [access(metadata()), access(metadata())]
    assert target._exact_sparse_completion_matches(actual, expected)

    forged = [access(metadata()), access(metadata(access=8))]
    assert not target._exact_sparse_completion_matches(forged, expected)

    class StoredKey(str):
        pass

    malformed = metadata()
    malformed_state = vars(malformed)
    malformed_state[StoredKey("access_id")] = malformed_state.pop("access_id")
    assert not target._exact_sparse_completion_matches(
        [access(malformed)], [access(metadata())]
    )


@pytest.mark.parametrize("malformed_key", [0, "str_subclass"])
def test_malformed_metadata_keys_fail_at_completion(monkeypatch, malformed_key):
    """Untrusted metadata dictionaries cannot leak sorting/type exceptions."""

    import scorch.compiler.llir_pass_manager as pass_manager

    original = pass_manager.rewrite_dynamic_vector_accesses
    state = {"mutated": False}

    def hostile_final_pass(value, context):
        rewritten = original(value, context)
        if state["mutated"] or type(rewritten) is not list:
            return rewritten
        metadata = _first_tensor_access_metadata(rewritten)
        metadata_state = vars(metadata)
        if malformed_key == "str_subclass":

            class StoredKey(str):
                pass

            key = StoredKey("access_id")
            metadata_state[key] = metadata_state.pop("access_id")
        else:
            metadata_state[malformed_key] = "hostile"
        state["mutated"] = True
        return rewritten

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        hostile_final_pass,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert state["mutated"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_two_phase_pass_context_cannot_mutate_program_result_identity(monkeypatch):
    """The pass receives a detached result SymbolId, never the program key."""

    import scorch.compiler.llir_pass_manager as pass_manager

    original = pass_manager.LLIRPassManager.run_compressed_where_openmp
    state = {"mutated": False}

    def hostile_context(self, artifact, pass_spec):
        result_id = pass_spec.context.result_id
        object.__setattr__(result_id, "value", result_id.value + 4_000_000)
        state["mutated"] = True
        return original(self, artifact, pass_spec)

    monkeypatch.setattr(
        pass_manager.LLIRPassManager,
        "run_compressed_where_openmp",
        hostile_context,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert state["mutated"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_completion_actual_shares_nothing_with_reference(monkeypatch):
    """The compared trees own disjoint aggregates, so the match is not vacuous.

    If the reconstructed reference ever shared a mutable node, list, tuple,
    identity, or metadata aggregate with the pass-returned function, a
    hostile in-place edit could change both sides together and the exact
    comparison would accept it.  Capture both sides at the real completion
    boundary of an activating compile and prove the owner sets are disjoint.
    """

    from scorch.compiler.loopir import lower_llir as target
    from tests.test_scorch.test_loopir_sparse_workspace_target import (
        _reachable_completion_owner_ids,
    )

    captured = []
    original_matches = target._exact_sparse_completion_matches

    def capture_both(actual, expected):
        captured.append(
            (
                _reachable_completion_owner_ids(actual),
                _reachable_completion_owner_ids(expected),
            )
        )
        return original_matches(actual, expected)

    monkeypatch.setattr(
        target,
        "_exact_sparse_completion_matches",
        capture_both,
    )
    compile_cin_via_loopir(
        build_spgemm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert captured
    for actual_ids, expected_ids in captured:
        assert actual_ids and expected_ids
        assert not (actual_ids & expected_ids)


def test_pass_cannot_swap_metadata_role_singleton(monkeypatch):
    """Swapping a metadata role to the other valid singleton fails closed.

    Both role singletons carry valid import-time state, so this attack is
    invisible to enum-state pinning; only the exact actual-versus-expected
    role comparison can reject it.
    """

    import scorch.compiler.llir_pass_manager as pass_manager
    from scorch.compiler import llir

    original = pass_manager.rewrite_dynamic_vector_accesses
    state = {"mutated": False}

    def hostile_final_pass(value, context):
        rewritten = original(value, context)
        if state["mutated"] or type(rewritten) is not list:
            return rewritten
        metadata = _first_tensor_access_metadata(rewritten)
        forged_role = (
            llir.TensorAccessRole.RESULT_WRITE
            if metadata.role is llir.TensorAccessRole.INPUT_READ
            else llir.TensorAccessRole.INPUT_READ
        )
        object.__setattr__(metadata, "role", forged_role)
        state["mutated"] = True
        return rewritten

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        hostile_final_pass,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spgemm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert state["mutated"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_frozen_pass_policy_mutation_fails_completion():
    """The completion reference owns its policy spellings, not the shared
    pass-module singleton, so a frozen-policy mutation cannot drift both
    sides of the exact comparison together."""

    from scorch.compiler.compressed_where_openmp_pass import (
        COMPRESSED_WHERE_OPENMP_POLICY,
    )

    original_schedule = COMPRESSED_WHERE_OPENMP_POLICY.omp_schedule
    assert original_schedule == "dynamic, 64"
    object.__setattr__(COMPRESSED_WHERE_OPENMP_POLICY, "omp_schedule", "static")
    try:
        with pytest.raises(LoopIRTargetError) as error:
            compile_cin_via_loopir(
                build_spgemm_cin(),
                (4, 5),
                (((4, 6), torch.float32), ((6, 5), torch.float32)),
                compile_options=auto_options(False),
            )
    finally:
        object.__setattr__(
            COMPRESSED_WHERE_OPENMP_POLICY, "omp_schedule", original_schedule
        )
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_target_failure_is_a_recorded_stage_loss():
    program, symbols = build_parallel_workspace_program(
        insert_value_builder=lambda builder, *_: builder.float_const(2.0)
    )
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)
    with pytest.raises(LoopIRTargetError):
        lower_loopir_to_llir(
            program,
            input_shapes=parallel_workspace_shapes(symbols),
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=context,
        )
    assert context._failed_stage_id is CompilerStageId.LOOPIR_TO_LLIR_LOWERING
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING not in {
        record.stage_id for record in context.stage_run_records
    }


_SEALED_ROWSCOPE_SHA256 = (
    "cf1114aafa8650bd230a4c156431b7019a550fd0c47c2fd6b754d929e97b4f51"
)


@torch.no_grad()
def test_rowscope_legacy_comparand_is_failure_evidence_only():
    """The sealed ss@ss->ds comparand is sound only on full row support.

    Its ``C1_pos`` is sized by the first operand's *stored* row count, not
    the result's dense row extent, so any input whose first operand has an
    empty row returns malformed storage that silently associates later
    rows' values with earlier rows.  The sound typed route is now admitted
    under the no-parity discipline: it sizes ``C1_pos`` from the logical
    result row extent rather than reproducing this defect.  This test keeps
    the sealed legacy kernel as hermetic failure evidence and locks its
    full-row-support control separately.
    """

    def rowscope_cin():
        i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
        a = TensorVar("A", fmt="ss")
        b = TensorVar("B", fmt="ss")
        c = TensorVar("C", fmt="ds")
        assign = TensorAssign(
            c[i, k],
            CINBinaryOp(Operation.MUL, a[i, j], b[j, k]),
            op=Operation.ADD,
        )
        return ForAll(i, ForAll(j, ForAll(k, assign)))

    options = auto_options(False)
    legacy_cpp = legacy_generated_cpp(
        rowscope_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=options,
    )
    assert hashlib.sha256(legacy_cpp.encode("utf-8")).hexdigest() == (
        _SEALED_ROWSCOPE_SHA256
    )
    comparand_path = _SEALED_COMPARAND_DIR / "rowscope_ss_ss_ds_armBASE.cpp"
    if comparand_path.exists():
        assert comparand_path.read_text() == legacy_cpp

    # The sound typed route (test_loopir_rowscope_workspace_target.py) now
    # admits this family with logical-row-extent sizing; the defective
    # comparand below remains hermetic failure evidence, and the typed
    # source must never byte-match it.
    typed = compile_cin_via_loopir(
        rowscope_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert typed.cpp_source != legacy_cpp
    assert "for (; C1_pos_index < C0_size; C1_pos_index++)" in typed.cpp_source

    torch.manual_seed(20260805)
    dense_a = torch.zeros(4, 6)
    dense_a[0, 1] = 1.0
    dense_a[1, 2] = 2.0
    dense_a[3, 4] = 3.0  # row 2 stays empty: only three stored rows
    dense_b = (torch.rand(6, 5) < 0.8) * torch.randn(6, 5)

    def sparse_ss(dense, name):
        return STensor.from_torch(dense.clone(), name).to_sparse("ss")

    malformed = execute_legacy_module(
        legacy_cpp,
        (4, 5),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        auto_options(False, jit=True),
    )
    malformed_modes = malformed.storage.index.mode_indices
    malformed_pos = malformed_modes[1][0].tolist()
    malformed_crd = malformed_modes[1][1].tolist()
    malformed_values = malformed.storage.value.tolist()
    assert malformed_pos == [0, 5, 8, 11]
    assert len(malformed_pos) == 4, "honest ds storage would carry 5 positions"
    assert malformed_pos[-1] == len(malformed_crd) == len(malformed_values)

    expected = dense_a @ dense_b
    interpreted = torch.zeros_like(expected)
    for stored_row in range(len(malformed_pos) - 1):
        for position in range(malformed_pos[stored_row], malformed_pos[stored_row + 1]):
            interpreted[stored_row, malformed_crd[position]] = malformed_values[
                position
            ]
    assert torch.allclose(interpreted[0], expected[0])
    assert torch.allclose(interpreted[1], expected[1])
    assert torch.count_nonzero(expected[2]) == 0
    assert torch.allclose(interpreted[2], expected[3])
    assert not torch.allclose(interpreted, expected)

    # Full row support is the comparand's sound sub-domain: the same
    # kernel then produces complete, honest positions.
    full_a = dense_a.clone()
    full_a[2, 0] = 4.0
    sound = execute_legacy_module(
        legacy_cpp,
        (4, 5),
        sparse_ss(full_a, "A"),
        sparse_ss(dense_b, "B"),
        auto_options(False, jit=True),
    )
    sound_modes = sound.storage.index.mode_indices
    assert list(sound_modes[0]) == []
    sound_storage = (
        sound_modes[1][0].tolist(),
        sound_modes[1][1].tolist(),
        sound.storage.value.tolist(),
    )
    assert len(sound_storage[0]) == 5
    assert sound_storage[0][0] == 0
    assert sound_storage[0] == sorted(sound_storage[0])
    assert sound_storage[0][-1] == len(sound_storage[1]) == len(sound_storage[2])
    for row in range(4):
        columns = sound_storage[1][sound_storage[0][row] : sound_storage[0][row + 1]]
        assert columns == sorted(set(columns))
        assert all(0 <= column < 5 for column in columns)
    sound_dense = dense_from_ds_storage(sound_storage, (4, 5), torch.float32)
    assert torch.allclose(sound_dense, full_a @ dense_b, atol=1e-3, rtol=1e-3)


# -- adjacent seams stay fail-closed --------------------------------------------


def _expect_lowering_code(code, cin, bindings, result_shape=(4, 5), *, auto=True):
    """One seam probe; ``auto=False`` checks the unscheduled boundary.

    Cells the automatic scheduler itself rejects (before CIN-to-LoopIR
    lowering can classify them) are probed on the unscheduled route, where
    the lowering boundary owns the stable code.
    """

    options = (
        auto_options(False) if auto else CompileOptions.from_environment(environ={})
    )
    with pytest.raises((LoopIRLoweringError, LoopIRTargetError)) as error:
        compile_cin_via_loopir(
            cin,
            result_shape,
            bindings,
            compile_options=options,
        )
    assert error.value.defect.code == code


def test_adjacent_seams_stay_fail_closed():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")

    # The ss@ss->ds row-scope family left this fail-closed census when its
    # sound verified slice landed (test_loopir_rowscope_workspace_target.py):
    # the typed route sizes C1_pos from the logical row extent and never
    # byte-matches the defective legacy comparand retained above.
    a = TensorVar("A", fmt="ss")
    b = TensorVar("B", fmt="ss")
    c = TensorVar("C", fmt="ds")
    rowscope_admitted = compile_cin_via_loopir(
        ForAll(
            i,
            ForAll(
                j,
                ForAll(
                    k,
                    TensorAssign(
                        c[i, k],
                        CINBinaryOp(Operation.MUL, a[i, j], b[j, k]),
                        op=Operation.ADD,
                    ),
                ),
            ),
        ),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert (
        "for (; C1_pos_index < C0_size; C1_pos_index++)" in rowscope_admitted.cpp_source
    )

    # A dense reduction domain keeps the old seam code.  The automatic
    # scheduler rejects this cell before lowering can classify it, so the
    # lowering boundary is probed on the unscheduled route.
    i2, j2, k2 = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a2 = TensorVar("A", fmt="dd")
    b2 = TensorVar("B", fmt="ds")
    c2 = TensorVar("C", fmt="ds")
    _expect_lowering_code(
        "unsupported_sparse_output_reduction",
        ForAll(
            i2,
            ForAll(
                j2,
                ForAll(
                    k2,
                    TensorAssign(
                        c2[i2, k2],
                        CINBinaryOp(Operation.MUL, a2[i2, j2], b2[j2, k2]),
                        op=Operation.ADD,
                    ),
                ),
            ),
        ),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        auto=False,
    )

    # A trailing-level reduction below the column coordinate keeps the
    # old seam code.
    i3, j3, k3 = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a3 = TensorVar("A", fmt="dss")
    c3 = TensorVar("C", fmt="ds")
    _expect_lowering_code(
        "unsupported_sparse_output_reduction",
        ForAll(
            i3,
            ForAll(
                j3,
                ForAll(
                    k3,
                    TensorAssign(
                        c3[i3, j3],
                        a3[i3, j3, k3],
                        op=Operation.ADD,
                    ),
                ),
            ),
        ),
        (((4, 5, 6), torch.float32),),
        auto=False,
    )
