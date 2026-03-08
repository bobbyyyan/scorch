from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar, Where, Workspace
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.scheduler import Scheduler


def _lower_to_cpp(cin_stmt) -> str:
    lowered = CINLowerer().lower_IndexStmt(cin_stmt)
    return LLIRLowerer().lower_llir(lowered)


def test_spmm_codegen_emits_parallel_restrict_unroll_and_stack_workspace():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")

    c[i, j] = a[i, k] * b[k, j]
    cin_stmt = Scheduler.auto_schedule(ForAll(i, ForAll(k, ForAll(j, c._assignment))))
    cpp_code = _lower_to_cpp(cin_stmt)

    assert "#pragma omp parallel for" in cpp_code
    assert "__restrict__" in cpp_code
    assert "#pragma unroll" in cpp_code
    assert "float wksp[kTile_" in cpp_code


def test_non_tiled_dense_workspace_is_zero_initialized_and_freed():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    wksp = Workspace("wksp", dim=1, dense=True)

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        wksp[j],
                        a[i, k] * b[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    c[i, j],
                    wksp[j],
                ),
            ),
        ),
    )

    cpp_code = _lower_to_cpp(cin_stmt)
    assert "new float[" in cpp_code and "]()" in cpp_code
    assert "delete[] wksp;" in cpp_code
