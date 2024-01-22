from scorch.compiler.cin import IndexVar, ForAll, TensorAssign, TensorVar, Operation
from scorch.compiler.scheduler import Scheduler


def test_insert_dense_workspace():
    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    cin_stmt = ForAll(
        i,
        ForAll(
            j, ForAll(k, TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD))
        ),
    )

    scheduler = Scheduler()

    new_cin = scheduler.insert_workspace(cin_stmt)

    print(new_cin)


def test_add_tile():
    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    cin_stmt = ForAll(
        i,
        ForAll(
            j, ForAll(k, TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD))
        ),
    )

    scheduler = Scheduler()

    new_cin = scheduler.add_tile(cin_stmt, k, 1024)

    print(new_cin)
