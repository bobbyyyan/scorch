from src.taco_torch.cin import (
    IndexVar,
    TensorVar,
    ForAll,
    LoopOrderGetter,
    all_free_var_loops_before_reduction_loops,
)


def get_spmm_cin(loop_order=None):
    # SpMM: C[i, j] = A[i, k] * B[k, j]
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    str_to_index_var = {"i": i, "j": j, "k": k}

    C = TensorVar("C", fmt=["dense", "sparse"])
    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])

    C[i, j] = A[i, k] * B[k, j]

    # cin_stmt = ForAll(i, ForAll(j, ForAll(k, C._assignment)))
    if loop_order is None:
        loop_order = [i, j, k]

    if isinstance(loop_order, str):
        loop_order = [str_to_index_var[c] for c in loop_order]

    if isinstance(loop_order, list) and loop_order and isinstance(loop_order[0], str):
        loop_order = [str_to_index_var[c] for c in loop_order]

    cin_stmt = ForAll(
        loop_order[0],
        ForAll(
            loop_order[1],
            ForAll(
                loop_order[2],
                C._assignment,
            ),
        ),
    )

    return cin_stmt, C, A, B, i, j, k


def test_elementwise_sparse_vector_mul_cin():
    """c[i] = a[i] * b[i]"""
    i = IndexVar("i")
    a = TensorVar("a", fmt="sparse")
    b = TensorVar("b", fmt="sparse")
    c = TensorVar("c", fmt="sparse")

    c[i] = a[i] * b[i]

    cin_stmt = ForAll(i, c._assignment)

    cin_stmt_str = str(cin_stmt)

    print(cin_stmt_str)


def test_loop_order_getter():
    # SpMM: C[i, j] = A[i, k] * B[k, j]
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt=["dense", "sparse"])
    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])

    C[i, j] = A[i, k] * B[k, j]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, C._assignment)))

    loop_order_getter = LoopOrderGetter(cin_stmt)
    assert loop_order_getter.index_vars_ordered == [i, j, k]
    assert loop_order_getter.free_vars == [i, j]

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, C._assignment)))

    loop_order_getter = LoopOrderGetter(cin_stmt)
    assert loop_order_getter.index_vars_ordered == [i, k, j]
    assert loop_order_getter.free_vars == [i, j]


def test_all_free_var_loops_before_reduction_loops():
    # SpMM: C[i, j] = A[i, k] * B[k, j]
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt=["dense", "sparse"])
    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])

    C[i, j] = A[i, k] * B[k, j]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, C._assignment)))

    assert all_free_var_loops_before_reduction_loops(cin_stmt)

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, C._assignment)))

    assert not all_free_var_loops_before_reduction_loops(cin_stmt)


def test_get_result_tensors():
    cin_stmt, C, A, B, i, j, k = get_spmm_cin()

    assert cin_stmt.get_result_tensor_vars() == [C]


def test_get_rhs_tensors():
    cin_stmt, C, A, B, i, j, k = get_spmm_cin()

    assert cin_stmt.get_rhs_tensor_vars() == [A, B]
