"""
Tests for matrix multiplication operations in the CIN compiler.
Includes different formats and algorithms for SpMV and SpMM operations.
"""

import pytest

from scorch.compiler.cin import (
    IndexVar,
    TensorVar,
    ForAll,
    TensorAssign,
    Workspace,
    Where,
    Operation,
    TileSizeVar,
)
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.scheduler import Scheduler
from scorch.exceptions import defect_code

from tests.test_scorch.test_helpers import (
    lower_and_print,
    create_index_vars,
    create_tensor_vars,
    create_matrix_multiplication,
    create_workspace_operation,
)


def _assert_unscheduled_sparse_result_reduction(cin_stmt):
    with pytest.raises(UnsupportedFeature) as refusal:
        lower_and_print(cin_stmt)
    assert [diagnostic.code for diagnostic in refusal.value.diagnostics] == [
        "unscheduled_sparse_result_reduction"
    ]


def test_spmv_codegen():
    """
    Sparse matrix-vector multiplication
    y[i] = A[i, j] * x[j]
    """
    i = IndexVar("i")
    j = IndexVar("j")

    y = TensorVar("y", fmt=["dense"])
    A = TensorVar("A", fmt=["dense", "sparse"])
    x = TensorVar("x", fmt=["dense"])

    y[i] = A[i, j] * x[j]

    cin_stmt = ForAll(i, ForAll(j, y._assignment))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmv_wksp_codegen():
    """
    Sparse matrix-vector multiplication with workspace
    taco "y(i) = A(i, j) * x(j)" -f=y:d -f=A:ds -f=x:d -print-evaluate
    """
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "y": ["dense"],
        "A": ["dense", "sparse"],
        "x": ["dense"]
    })

    # Create a workspace
    workspace = Workspace(name="wksp", dim=0, dtype=tensors["y"].dtype)

    # Create the CIN statement for SpMV with workspace
    # For SpMV: y[i] = A[i,j] * x[j]
    # Producer: ForAll(j, workspace += A[i,j] * x[j])
    # Consumer: y[i] = workspace

    producer = ForAll(
        j,
        TensorAssign(
            workspace.get_default_access(),
            tensors["A"][i, j] * tensors["x"][j],
            op=Operation.ADD,
        ),
    )

    consumer = TensorAssign(
        tensors["y"][i],
        workspace.get_default_access(),
    )

    cin_stmt = ForAll(
        i,
        Where(
            producer=producer,
            consumer=consumer,
        ),
    )

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_spmm_codegen():
    """
    Sparse matrix-matrix multiplication
    taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
    """
    i, j, k = create_index_vars("i", "j", "k")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["dense", "sparse"],
        "B": ["dense", "sparse"],
        "C": ["dense", "sparse"]
    })

    # Create the CIN statement for sparse matrix multiplication
    cin_stmt = create_matrix_multiplication(tensors, (i, j, k), contract_idx_pos=2)

    # The loops are i, k, j with no workspace: the kernel appended a coordinate of
    # A for every (k, j) it met -- a column once per k that reaches it, unsorted --
    # and added into A's values at a position its vector never grew to.  The
    # unscheduled program is refused; ``scorch.matmul`` schedules a workspace.
    _assert_unscheduled_sparse_result_reduction(cin_stmt)


def test_sddmm_codegen():
    """
    Sampled dense-dense matrix multiplication (SDDMM)
    taco "A(i, j) = B(i, j) * C(i, k) * D(k, j)" -f=A:ds -f=B:ds -f=C:dd -f=D:dd -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="ds")
    C = TensorVar("C", fmt="dd")
    D = TensorVar("D", fmt="dd")

    A[i, j] = B[i, j] * C[i, k] * D[k, j]

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, A._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    # The reduced index k is the loop *around* A's compressed level j, so the
    # kernel appended A's row once per k.  Refused as written (with j around k
    # each stored cell is appended once, and that order lowers).
    _assert_unscheduled_sparse_result_reduction(cin_stmt)


def test_spmm_dd_dd_ds_ijk_gustavson():
    """
    Sparse matrix-matrix multiplication
    taco "C(i, k) = A(i, j) * B(j, k)" -f=A:dd -f=B:dd -f=C:ds -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt="dd")

    A = TensorVar("A", fmt="dd")
    B = TensorVar("B", fmt="ds")

    cin_stmt = ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    C[i, k],
                    A[i, j] * B[j, k],
                    op=Operation.ADD,
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_ds_ds_ds_kij_outer_workspace():
    """
    Sparse matrix-matrix multiplication with outer workspace pattern
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    workspace = Workspace(
        name="wksp",
        dim=2,
        dtype=A.dtype,
    )

    cin_stmt = Where(
        producer=ForAll(
            k,
            ForAll(
                i,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[i, j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
        ),
        consumer=ForAll(
            i,
            ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[i, j],
                    op=Operation.ADD,
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_ds_ds_ds_ikj_gustavson_workspace():
    """
    Sparse matrix-matrix multiplication with Gustavson workspace pattern
    taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="ds")
    C = TensorVar("C", fmt="ds")

    workspace = Workspace(
        name="wksp",
        dim=1,
        dtype=A.dtype,
    )

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_dd_ds_dd_ijk():
    """
    Sparse matrix-matrix multiplication
    taco "C(i, k) = A(i, j) * B(j, k)" -f=C:dd -f=A:ds -f=B:dd -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    C[i, k] = A[i, j] * B[j, k]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, C._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_dd_ds_dd_ijk_auto_tile():
    """
    Sparse matrix-matrix multiplication with automatic tiling
    taco "C(i, k) = A(i, j) * B(j, k)" -f=C:dd -f=A:ds -f=B:dd -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    C[i, k] = A[i, j] * B[j, k]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, C._assignment)))
    tiled_cin_stmt = Scheduler.add_tile(cin_stmt, k, 1024)

    print("\nTiled CIN statement:")
    print(tiled_cin_stmt)

    lower_and_print(tiled_cin_stmt)


def test_spmm_dd_ds_dd_tiled():
    """
    Manually tiled sparse matrix-matrix multiplication

    C[i, k] = A[i, j] * B[j, k]
    k gets tiled
    loop order: i, k_out, j, k_in
    accumulator: accum_c[k_in]
    ForAll i
      ForAll k_out
        Where(
            producer=ForAll j, ForAll k_in
                k = k_out + k_in
                (accum_c[k_in] += A[i, j] * B[j, k]),
            consumer=ForAll k_in
                k = k_out + k_in
                C[i, k] = accum_c[k_in]
        )
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k_out = IndexVar("k_out")
    k_in = IndexVar("k_in")
    k = IndexVar("k", k_out + k_in)

    k_tile_size = 1024
    k_tile_var = TileSizeVar(
        outer_index_var=k_out, inner_index_var=k_in, size=k_tile_size
    )

    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    # accum_c = TensorVar("accum_c", fmt="d")
    accum_c = Workspace("accum_c", dim=1, dtype=C.dtype, dense=True)

    cin_stmt = ForAll(
        i,
        ForAll(
            k_out,
            Where(
                producer=ForAll(
                    j,
                    ForAll(
                        k_in,
                        TensorAssign(
                            accum_c[k_in],
                            A[i, j] * B[j, k],
                            op=Operation.ADD,
                        ),
                    ),
                ),
                consumer=ForAll(
                    k_in,
                    TensorAssign(
                        C[i, k],
                        accum_c[k_in],
                    ),
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_dd_oo_dd_ikj_gustavson_workspace():
    """
    Sparse matrix-matrix multiplication with COO input format and Gustavson workspace pattern

    The sparse workspace is drained into a result that is dense in both modes.  The
    shipped compiler's drains assemble a compressed or coordinate level, and the
    kernel it used to print for this statement did not build ('pC1' was not
    declared), so it refuses it with a code.  Lowering it (a scatter into the dense
    row) is a known gap: test_known_compiler_gaps.py's
    test_known_gap_gustavson_workspace_into_a_dense_result.
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt="dd")
    B = TensorVar("B", fmt="oo")
    C = TensorVar("C", fmt="dd")

    workspace = Workspace(
        name="wksp",
        dim=1,
        dtype=A.dtype,
    )

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    with pytest.raises(UnsupportedFeature) as refusal:
        lower_and_print(cin_stmt)
    assert defect_code(refusal.value) == "unsupported_workspace_result_level"


def test_spmm_dd_ds_ds():
    """
    Sparse matrix-matrix multiplication with mixed formats
    taco "A(i, j) = B(i, k) * C(k, j)" -f=A:dd -f=B:ds -f=C:ds -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "dense"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    cin_stmt = ForAll(
        i,
        ForAll(
            k,
            ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    B[i, k] * C[k, j],
                    op=Operation.ADD,
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_sd_ds_ds():
    """
    Sparse matrix-matrix multiplication with mixed formats
    taco "A(i, j) = B(i, k) * C(k, j)" -f=A:sd -f=B:ds -f=C:ds -print-evaluate

    This test only lowers the program.  The kernel it lowers to appends a row of A
    once per iteration of the reduction loop over k, so A's compressed row level
    holds repeated rows and the result's validation rejects it: a known gap,
    test_known_compiler_gaps.py's
    test_known_gap_sd_result_with_a_reduction_below_its_compressed_level.
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["sparse", "dense"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, k] * C[k, j]

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, A._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_ikj_gustavson_workspace():
    """
    Sparse matrix-matrix multiplication using the Gustavson workspace pattern
    taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
    """
    i, j, k = create_index_vars("i", "j", "k")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["dense", "sparse"],
        "B": ["dense", "sparse"],
        "C": ["dense", "sparse"]
    })

    # Create a workspace
    workspace = Workspace(name="wksp", dim=1, dtype=tensors["A"].dtype)

    # Create the CIN statement for SpMM with Gustavson workspace pattern
    cin_stmt = create_workspace_operation(tensors, workspace, (i, j, k), operation_type="mul")

    # Lower and print the generated code
    lower_and_print(cin_stmt)
