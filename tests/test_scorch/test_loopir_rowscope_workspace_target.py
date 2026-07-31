"""The sound ss@ss->ds row-scope family: dense-row CSR sizing, no parity.

The defective legacy comparand for this family sizes ``C1_pos`` by the
first operand's stored-row count, silently associating later rows'
values with earlier rows whenever a logical row is empty (the hermetic
failure lock lives in ``test_loopir_parallel_workspace_target.py``).
The typed route instead sizes and closes ``C1_pos`` from the logical
result row extent: per-row positional catch-ups close every skipped
empty row, and a final catch-up closes through ``C0_size``.  By
construction the generated source never byte-matches the legacy kernel,
so under the established no-parity discipline the family is proven by
structural activation, the production LoopIR oracle (exact CSR storage,
base and scheduled agreement), the PyTorch dense reference,
deterministic storage, and an explicit non-parity disposition.  The
generic unsized-values sparse-reduction route stays quarantined: the
unscheduled semantic form still fails closed at the general target.
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
from scorch.compiler.loopir.lower_llir import (
    LoopIRTargetError,
    lower_loopir_to_llir,
)
from scorch.compiler.loopir.nodes import LevelKind
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.stensor import STensor
from tests.test_scorch.test_loopir_parallel_workspace_target import (
    dense_from_ds_storage,
    validated_ds_storage,
)
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

_CC_KINDS = (LevelKind.COMPRESSED, LevelKind.COMPRESSED)


def build_rowscope_cin(dtype=torch.float32):
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    assign = TensorAssign(
        TensorVar("C", fmt="ds", dtype=dtype)[i, k],
        CINBinaryOp(
            Operation.MUL,
            TensorVar("A", fmt="ss", dtype=dtype)[i, j],
            TensorVar("B", fmt="ss", dtype=dtype)[j, k],
        ),
        op=Operation.ADD,
    )
    return ForAll(i, ForAll(j, ForAll(k, assign)))


def sparse_ss(dense, name):
    return STensor.from_torch(dense.clone(), name).to_sparse("ss")


def executed(cin, result_shape, st_a, st_b, regblock_enabled):
    out = execute_cin_via_loopir(
        cin,
        result_shape,
        st_a,
        st_b,
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    return out[0] if isinstance(out, tuple) else out


def oracle_csr(kernel, dense_a, dense_b, result_shape):
    """Base and scheduled oracle agreement over ss inputs; returns CSR."""

    lowering = kernel.lowering
    assert kernel.schedule is not None
    inputs = {}
    for symbol, dense in zip(lowering.rhs_access_symbols, (dense_a, dense_b)):
        inputs[symbol] = LevelTensorStorage.from_dense(
            dense.tolist(), tuple(dense.shape), (0, 1), _CC_KINDS
        )
    scheduled = run_program(
        kernel.schedule.program,
        inputs,
        {lowering.result_symbol: tuple(result_shape)},
    )[lowering.result_symbol]
    base = run_program(
        lowering.program,
        inputs,
        {lowering.result_symbol: tuple(result_shape)},
    )[lowering.result_symbol]
    assert scheduled == base
    return scheduled


def _empty_row_fixture(dtype=torch.float32):
    dense_a = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    torch.manual_seed(1)
    dense_b = (((torch.rand(6, 5) < 0.5) * torch.randn(6, 5))).to(dtype)
    return dense_a, dense_b


# -- activation and the non-parity disposition --------------------------------


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_route_activates_and_never_matches_the_defective_legacy(
    regblock_enabled, dtype
):
    options = auto_options(regblock_enabled)
    kernel = compile_cin_via_loopir(
        build_rowscope_cin(dtype),
        (4, 5),
        (((4, 6), dtype), ((6, 5), dtype)),
        compile_options=options,
    )
    source = kernel.cpp_source
    assert "for (; C1_pos_index < C0_size; C1_pos_index++)" in source
    assert "for (; C1_pos_index < i; C1_pos_index++)" in source
    assert "coo_workspace_1d" in source
    legacy = legacy_generated_cpp(
        build_rowscope_cin(dtype),
        (4, 5),
        (((4, 6), dtype), ((6, 5), dtype)),
        compile_options=options,
    )
    assert source != legacy


def test_compile_is_deterministic_and_arm_stable():
    sources = set()
    for arm in (False, True):
        for _ in range(2):
            kernel = compile_cin_via_loopir(
                build_rowscope_cin(),
                (4, 5),
                (((4, 6), torch.float32), ((6, 5), torch.float32)),
                compile_options=auto_options(arm),
            )
            sources.add(kernel.cpp_source)
    assert len(sources) == 1


# -- compiled execution -------------------------------------------------------


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_empty_rows_are_preserved_against_every_reference(regblock_enabled):
    dense_a, dense_b = _empty_row_fixture()
    result = executed(
        build_rowscope_cin(),
        (4, 5),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        regblock_enabled,
    )
    storage = validated_ds_storage(result, (4, 5))
    pos1 = storage[0]
    assert len(pos1) == 5, "C1_pos must carry the logical row extent"
    assert pos1[1] == pos1[2], "empty row 1 must close at its predecessor"
    assert pos1[3] == pos1[4], "empty row 3 must close at its predecessor"

    kernel = compile_cin_via_loopir(
        build_rowscope_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(regblock_enabled),
    )
    oracle = oracle_csr(kernel, dense_a, dense_b, (4, 5))
    assert tuple(storage[0]) == oracle.indptr
    assert tuple(storage[1]) == oracle.indices
    for got, expected in zip(storage[2], oracle.values):
        assert got == pytest.approx(expected, abs=1e-5, rel=1e-5)

    dense = dense_from_ds_storage(storage, (4, 5), torch.float32)
    assert torch.allclose(dense, dense_a @ dense_b, atol=1e-3, rtol=1e-3)


def test_random_grids_match_pytorch():
    torch.manual_seed(20260731)
    for rows, inner, cols in ((5, 4, 6), (8, 8, 3), (1, 6, 4)):
        dense_a = (torch.rand(rows, inner) < 0.4) * torch.randn(rows, inner)
        dense_b = (torch.rand(inner, cols) < 0.4) * torch.randn(inner, cols)
        result = executed(
            build_rowscope_cin(),
            (rows, cols),
            sparse_ss(dense_a, "A"),
            sparse_ss(dense_b, "B"),
            False,
        )
        storage = validated_ds_storage(result, (rows, cols))
        assert len(storage[0]) == rows + 1
        dense = dense_from_ds_storage(storage, (rows, cols), torch.float32)
        assert torch.allclose(dense, dense_a @ dense_b, atol=1e-3, rtol=1e-3)


def test_float64_and_disjoint_supports():
    dense_a = torch.zeros(3, 4, dtype=torch.float64)
    dense_b = torch.zeros(4, 3, dtype=torch.float64)
    dense_a[0, 1] = 2.0
    dense_b[2, 2] = 5.0  # A's columns never meet B's rows
    result = executed(
        build_rowscope_cin(torch.float64),
        (3, 3),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        False,
    )
    storage = validated_ds_storage(result, (3, 3))
    assert storage[0] == [0, 0, 0, 0]
    assert storage[2] == []


def test_zero_extent_rows_close_canonically():
    dense_a = torch.zeros(0, 4)
    dense_b = torch.zeros(4, 3)
    result = executed(
        build_rowscope_cin(),
        (0, 3),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        False,
    )
    storage = validated_ds_storage(result, (0, 3))
    assert storage[0] == [0]
    assert storage[2] == []


def test_deterministic_storage_across_repeated_executions():
    dense_a, dense_b = _empty_row_fixture()
    snapshots = []
    for _ in range(3):
        result = executed(
            build_rowscope_cin(),
            (4, 5),
            sparse_ss(dense_a, "A"),
            sparse_ss(dense_b, "B"),
            False,
        )
        snapshots.append(validated_ds_storage(result, (4, 5)))
    assert snapshots[0] == snapshots[1] == snapshots[2]


# -- quarantine and hostile boundaries ----------------------------------------


def test_unscheduled_semantic_form_stays_quarantined():
    """Without the workspace schedule the generic route still fails closed."""

    from scorch.compiler.compilation_context import CompilationContext
    from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir
    from scorch.compiler.cin_analysis import normalize_cin

    lowering = lower_normalized_cin_to_loopir(normalize_cin(build_rowscope_cin()))
    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            lowering.program,
            input_shapes={
                lowering.input_symbols[0]: (4, 6),
                lowering.input_symbols[1]: (6, 5),
            },
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "unsupported_program_shape"


def test_pipeline_entry_header_mutation_fails_closed(monkeypatch):
    """The row-scope completion inherits the B1 fresh-ownership discipline."""

    import scorch.compiler.llir_pass_manager as pass_manager
    from scorch.compiler import llir

    original = pass_manager.insert_sparse_prefetch
    state = {"mutated": False}

    def hostile_prefetch(statements, context):
        for statement in statements:
            if type(statement) is llir.ForLoop and statement.cond is not None:
                object.__setattr__(statement.cond, "op", "<=")
                state["mutated"] = True
                break
        return original(statements, context)

    monkeypatch.setattr(pass_manager, "insert_sparse_prefetch", hostile_prefetch)
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_rowscope_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert state["mutated"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"
