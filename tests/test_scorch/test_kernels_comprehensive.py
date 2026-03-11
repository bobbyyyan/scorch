"""
Comprehensive correctness tests for Scorch kernels.

Every test verifies end-to-end results against a PyTorch reference using
torch.allclose (atol/rtol=1e-3).  Tests use the einsum() API (or operator+
for addition) with default mode_order [0,1] to avoid known compiler gaps.
"""

import pytest
import torch

from scorch import STensor, einsum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sparse_2d(m, n, sparsity, seed):
    """Return a (m x n) torch.Tensor with the given sparsity ratio zeroed out."""
    torch.manual_seed(seed)
    t = torch.rand(m, n)
    mask = (torch.rand(m, n) > sparsity).float()
    return t * mask


def make_sparse_3d(d0, d1, d2, sparsity, seed):
    """Return a (d0 x d1 x d2) torch.Tensor with the given sparsity."""
    torch.manual_seed(seed)
    t = torch.rand(d0, d1, d2)
    mask = (torch.rand(d0, d1, d2) > sparsity).float()
    return t * mask


def make_sparse_1d(n, sparsity, seed):
    """Return a length-n torch.Tensor with the given sparsity."""
    torch.manual_seed(seed)
    t = torch.rand(n)
    mask = (torch.rand(n) > sparsity).float()
    return t * mask


ATOL = 1e-3
RTOL = 1e-3


def assert_close(scorch_result, expected):
    """Compare an STensor result against a torch.Tensor reference."""
    if isinstance(scorch_result, STensor):
        actual = scorch_result.to_torch()
    else:
        actual = scorch_result
    assert torch.allclose(actual, expected, atol=ATOL, rtol=RTOL), (
        f"Max diff: {(actual - expected).abs().max().item()}"
    )


# ===================================================================
# 1. SpMV Format Variety
# ===================================================================

class TestSpMV:
    """SpMV: y[i] = A[i,j] * x[j]  verified against torch.mv"""

    @pytest.mark.parametrize("matrix_fmt", ["ds", "ss", "oo"])
    def test_spmv_square(self, matrix_fmt):
        torch.manual_seed(42)
        a_torch = make_sparse_2d(30, 30, 0.8, seed=42)
        x_torch = torch.rand(30)

        a_st = STensor.from_torch(a_torch).to_sparse(matrix_fmt)
        x_st = STensor.from_torch(x_torch)

        result = einsum("ij,j->i", a_st, x_st, format="d")
        expected = torch.mv(a_torch, x_torch)
        assert_close(result, expected)

    def test_spmv_dense_matrix(self):
        torch.manual_seed(43)
        a_torch = torch.rand(20, 20)
        x_torch = torch.rand(20)

        a_st = STensor.from_torch(a_torch)  # dd format
        x_st = STensor.from_torch(x_torch)

        result = einsum("ij,j->i", a_st, x_st, format="d")
        expected = torch.mv(a_torch, x_torch)
        assert_close(result, expected)

    def test_spmv_rectangular_tall(self):
        a_torch = make_sparse_2d(40, 80, 0.8, seed=44)
        x_torch = torch.rand(80)
        torch.manual_seed(44)

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        x_st = STensor.from_torch(x_torch)

        result = einsum("ij,j->i", a_st, x_st, format="d")
        expected = torch.mv(a_torch, x_torch)
        assert_close(result, expected)

    def test_spmv_rectangular_wide(self):
        a_torch = make_sparse_2d(80, 40, 0.8, seed=45)
        x_torch = torch.rand(40)
        torch.manual_seed(45)

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        x_st = STensor.from_torch(x_torch)

        result = einsum("ij,j->i", a_st, x_st, format="d")
        expected = torch.mv(a_torch, x_torch)
        assert_close(result, expected)


# ===================================================================
# 2. Element-wise Mul with Random Data
# ===================================================================

class TestElemwiseMulRandom:
    """Element-wise mul: C[i,j] = A[i,j] * B[i,j]"""

    @pytest.mark.parametrize(
        "a_fmt,b_fmt,out_fmt",
        [
            ("ds", "ds", "ds"),
            ("oo", "oo", "oo"),
            ("ss", "ss", "ss"),
            ("ds", "oo", "oo"),
            ("ds", "dd", "ds"),
        ],
    )
    def test_elemwise_mul_2d_random(self, a_fmt, b_fmt, out_fmt):
        a_torch = make_sparse_2d(25, 25, 0.7, seed=100)
        b_torch = make_sparse_2d(25, 25, 0.7, seed=101)

        a_st = STensor.from_torch(a_torch).to_sparse(a_fmt)
        b_st = STensor.from_torch(b_torch).to_sparse(b_fmt)

        result = einsum("ij,ij->ij", a_st, b_st, format=out_fmt)
        expected = a_torch * b_torch
        assert_close(result, expected)


# ===================================================================
# 3. Element-wise Add with Random Data
# ===================================================================

class TestElemwiseAddRandom:
    """Element-wise add: C = A + B  (operator+)"""

    def test_add_ds_ds(self):
        a_torch = make_sparse_2d(20, 20, 0.7, seed=200)
        b_torch = make_sparse_2d(20, 20, 0.7, seed=201)

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        b_st = STensor.from_torch(b_torch).to_sparse("ds")

        result = a_st + b_st
        expected = a_torch + b_torch
        assert_close(result, expected)

    def test_add_ss_ss(self):
        a_torch = make_sparse_2d(20, 20, 0.7, seed=202)
        b_torch = make_sparse_2d(20, 20, 0.7, seed=203)

        a_st = STensor.from_torch(a_torch).to_sparse("ss")
        b_st = STensor.from_torch(b_torch).to_sparse("ss")

        result = a_st + b_st
        expected = a_torch + b_torch
        assert_close(result, expected)

    def test_add_oo_oo(self):
        a_torch = make_sparse_2d(20, 20, 0.7, seed=204)
        b_torch = make_sparse_2d(20, 20, 0.7, seed=205)

        a_st = STensor.from_torch(a_torch).to_sparse("oo")
        b_st = STensor.from_torch(b_torch).to_sparse("oo")

        result = a_st + b_st
        expected = a_torch + b_torch
        assert_close(result, expected)

    @pytest.mark.parametrize(
        "a_fmt,b_fmt",
        [
            ("ds", "oo"),
            ("oo", "ds"),
            ("ss", "ds"),
            ("ds", "ss"),
        ],
    )
    def test_add_mixed_formats(self, a_fmt, b_fmt):
        a_torch = make_sparse_2d(20, 20, 0.7, seed=206)
        b_torch = make_sparse_2d(20, 20, 0.7, seed=207)

        a_st = STensor.from_torch(a_torch).to_sparse(a_fmt)
        b_st = STensor.from_torch(b_torch).to_sparse(b_fmt)

        result = a_st + b_st
        expected = a_torch + b_torch
        assert_close(result, expected)


# ===================================================================
# 4. 3D Element-wise Mul Correctness
# ===================================================================

class TestElemwiseMul3D:
    """3D element-wise mul: C[i,j,k] = A[i,j,k] * B[i,j,k]"""

    @pytest.mark.parametrize(
        "a_fmt,b_fmt,out_fmt",
        [
            ("dss", "dss", "dss"),
            ("sss", "sss", "sss"),
            ("dss", "ddd", "ddd"),
        ],
    )
    def test_elemwise_mul_3d_random(self, a_fmt, b_fmt, out_fmt):
        a_torch = make_sparse_3d(4, 5, 6, 0.7, seed=300)
        b_torch = make_sparse_3d(4, 5, 6, 0.7, seed=301)

        a_st = STensor.from_torch(a_torch).to_sparse(a_fmt)
        b_st = STensor.from_torch(b_torch).to_sparse(b_fmt)

        result = einsum("ijk,ijk->ijk", a_st, b_st, format=out_fmt)
        expected = a_torch * b_torch
        assert_close(result, expected)


# ===================================================================
# 5. Outer Product Variety
# ===================================================================

class TestOuterProduct:
    """Outer product: C[i,j] = a[i] * b[j]"""

    @pytest.mark.parametrize("out_fmt", ["ds", "oo", "dd"])
    def test_outer_random(self, out_fmt):
        torch.manual_seed(400)
        a_torch = make_sparse_1d(10, 0.5, seed=400)
        b_torch = make_sparse_1d(15, 0.5, seed=401)

        a_st = STensor.from_torch(a_torch).to_sparse("s")
        b_st = STensor.from_torch(b_torch).to_sparse("s")

        result = einsum("i,j->ij", a_st, b_st, format=out_fmt)
        expected = torch.outer(a_torch, b_torch)
        assert_close(result, expected)


# ===================================================================
# 6. Rectangular SpMM/SpGEMM
# ===================================================================

class TestRectangularSpMM:
    """Rectangular matrix multiply: C[i,j] = A[i,k] * B[k,j]"""

    def test_spmm_ds_dd_rect(self):
        """ds(30x50) * dd(50x20) -> dd(30x20)"""
        a_torch = make_sparse_2d(30, 50, 0.8, seed=500)
        b_torch = torch.rand(50, 20)
        torch.manual_seed(500)

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        b_st = STensor.from_torch(b_torch)

        result = einsum("ik,kj->ij", a_st, b_st, format="dd")
        expected = torch.matmul(a_torch, b_torch)
        assert_close(result, expected)

    def test_spgemm_ds_ds_rect(self):
        """ds(25x40) * ds(40x35) -> ds(25x35)"""
        a_torch = make_sparse_2d(25, 40, 0.8, seed=501)
        b_torch = make_sparse_2d(40, 35, 0.8, seed=502)

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        b_st = STensor.from_torch(b_torch).to_sparse("ds")

        result = einsum("ik,kj->ij", a_st, b_st, format="ds")
        expected = torch.matmul(a_torch, b_torch)
        assert_close(result, expected)

    def test_spmm_oo_dd_rect(self):
        """oo(20x60) * dd(60x15) -> dd(20x15)"""
        a_torch = make_sparse_2d(20, 60, 0.8, seed=503)
        b_torch = torch.rand(60, 15)
        torch.manual_seed(503)

        a_st = STensor.from_torch(a_torch).to_sparse("oo")
        b_st = STensor.from_torch(b_torch)

        result = einsum("ik,kj->ij", a_st, b_st, format="dd")
        expected = torch.matmul(a_torch, b_torch)
        assert_close(result, expected)


# ===================================================================
# 7. SpGEMM with Random Data
# ===================================================================

class TestSpGEMMRandom:
    """SpGEMM: C[i,j] = A[i,k] * B[k,j]  (both sparse)"""

    @pytest.mark.parametrize(
        "a_fmt,b_fmt,out_fmt",
        [
            ("ds", "ds", "ds"),
            ("oo", "oo", "oo"),
            ("ss", "ss", "ss"),
        ],
    )
    def test_spgemm_random(self, a_fmt, b_fmt, out_fmt):
        a_torch = make_sparse_2d(30, 30, 0.8, seed=600)
        b_torch = make_sparse_2d(30, 30, 0.8, seed=601)

        a_st = STensor.from_torch(a_torch).to_sparse(a_fmt)
        b_st = STensor.from_torch(b_torch).to_sparse(b_fmt)

        result = einsum("ik,kj->ij", a_st, b_st, format=out_fmt)
        expected = torch.matmul(a_torch, b_torch)
        assert_close(result, expected)


# ===================================================================
# 8. SDDMM Format Variety
# ===================================================================

class TestSDDMM:
    """SDDMM: A[i,j] = B[i,j] * C[i,k] * D[k,j]"""

    @pytest.mark.parametrize("mask_fmt", ["ds", "oo", "ss"])
    def test_sddmm_format_variety(self, mask_fmt):
        torch.manual_seed(700)
        b_torch = make_sparse_2d(30, 50, 0.85, seed=700)
        c_torch = torch.rand(30, 20)
        d_torch = torch.rand(20, 50)

        b_st = STensor.from_torch(b_torch).to_sparse(mask_fmt)
        c_st = STensor.from_torch(c_torch)
        d_st = STensor.from_torch(d_torch)

        result = einsum("ij,ik,kj->ij", b_st, c_st, d_st, format=mask_fmt)
        expected = torch.einsum("ij,ik,kj->ij", b_torch, c_torch, d_torch)
        assert_close(result, expected)

    def test_sddmm_auto_format_uses_coo_and_scalar_accum(self):
        """When no format is specified, SDDMM should auto-infer COO output
        to enable the scalar-accum codegen path (i,j,k loop order, no
        workspace, SIMD-vectorized reduction)."""
        from scorch.format import LevelType

        torch.manual_seed(701)
        b_torch = make_sparse_2d(30, 50, 0.85, seed=701)
        c_torch = torch.rand(30, 20)
        d_torch = torch.rand(20, 50)

        b_st = STensor.from_torch(b_torch).to_sparse("ds")
        c_st = STensor.from_torch(c_torch)
        d_st = STensor.from_torch(d_torch)

        # No explicit format — should auto-infer COO for SDDMM
        result = einsum("ij,ik,kj->ij", b_st, c_st, d_st)

        # Check format BEFORE assert_close, because to_torch() densifies
        # the STensor in-place.
        level_types = result.index.format.get_level_types()
        assert all(lt == LevelType.COORDINATE for lt in level_types), (
            f"Expected all-COO output for SDDMM auto-format, got {level_types}"
        )

        expected = torch.einsum("ij,ik,kj->ij", b_torch, c_torch, d_torch)
        assert_close(result, expected)

    def test_sddmm_auto_format_alternative_subscripts(self):
        """SDDMM detection should work for the ij,ik,jk->ij variant too."""
        torch.manual_seed(702)
        mask = make_sparse_2d(40, 40, 0.9, seed=702)
        Q = torch.rand(40, 16)
        K = torch.rand(40, 16)

        mask_st = STensor.from_torch(mask).to_sparse("ds")
        Q_st = STensor.from_torch(Q)
        K_st = STensor.from_torch(K)

        result = einsum("ij,ik,jk->ij", mask_st, Q_st, K_st)
        expected = (Q @ K.T) * mask
        assert_close(result, expected)

    @pytest.mark.parametrize("n,d,sparsity", [
        (16, 8, 0.5),
        (64, 32, 0.9),
        (128, 16, 0.95),
        (256, 64, 0.99),
    ])
    def test_sddmm_auto_format_various_sizes(self, n, d, sparsity):
        """SDDMM auto-format correctness across various matrix sizes,
        head dimensions, and sparsity levels."""
        torch.manual_seed(n + d)
        mask = make_sparse_2d(n, n, sparsity, seed=n + d)
        Q = torch.rand(n, d)
        K = torch.rand(n, d)

        mask_st = STensor.from_torch(mask).to_sparse("ds")
        Q_st = STensor.from_torch(Q)
        K_st = STensor.from_torch(K)

        result = einsum("ij,ik,jk->ij", mask_st, Q_st, K_st)
        expected = (Q @ K.T) * mask
        assert_close(result, expected)


# ===================================================================
# 9. TTM Correctness
# ===================================================================

class TestTTM:
    """TTM: C[i,j,m] = A[i,j,k] * B[k,m]"""

    def test_ttm_small_fixed(self):
        torch.manual_seed(800)
        a_torch = torch.rand(3, 3, 3)
        b_torch = torch.rand(3, 4)

        a_st = STensor.from_torch(a_torch)
        b_st = STensor.from_torch(b_torch)

        result = einsum("ijk,km->ijm", a_st, b_st, format="ddd")
        expected = torch.einsum("ijk,km->ijm", a_torch, b_torch)
        assert_close(result, expected)

    def test_ttm_random(self):
        torch.manual_seed(801)
        a_torch = torch.rand(8, 8, 8)
        b_torch = torch.rand(8, 6)

        a_st = STensor.from_torch(a_torch)
        b_st = STensor.from_torch(b_torch)

        result = einsum("ijk,km->ijm", a_st, b_st, format="ddd")
        expected = torch.einsum("ijk,km->ijm", a_torch, b_torch)
        assert_close(result, expected)


# ===================================================================
# 10. MTTKRP Correctness
# ===================================================================

class TestMTTKRP:
    """MTTKRP: D[i,m] = A[i,j,k] * B[j,m] * C[k,m]"""

    def test_mttkrp_small_fixed(self):
        torch.manual_seed(900)
        a_torch = torch.rand(3, 3, 3)
        b_torch = torch.rand(3, 2)
        c_torch = torch.rand(3, 2)

        a_st = STensor.from_torch(a_torch)
        b_st = STensor.from_torch(b_torch)
        c_st = STensor.from_torch(c_torch)

        result = einsum("ijk,jm,km->im", a_st, b_st, c_st, format="dd")
        expected = torch.einsum("ijk,jm,km->im", a_torch, b_torch, c_torch)
        assert_close(result, expected)

    def test_mttkrp_random(self):
        torch.manual_seed(901)
        a_torch = torch.rand(6, 6, 6)
        b_torch = torch.rand(6, 4)
        c_torch = torch.rand(6, 4)

        a_st = STensor.from_torch(a_torch)
        b_st = STensor.from_torch(b_torch)
        c_st = STensor.from_torch(c_torch)

        result = einsum("ijk,jm,km->im", a_st, b_st, c_st, format="dd")
        expected = torch.einsum("ijk,jm,km->im", a_torch, b_torch, c_torch)
        assert_close(result, expected)


# ===================================================================
# 11. Edge Cases
# ===================================================================

class TestEdgeCases:
    """Edge cases: zeros, disjoint sparsity, very high sparsity."""

    def test_all_zero_spmv(self):
        """All-zero sparse matrix SpMV -> zero result."""
        a_torch = torch.zeros(10, 10)
        x_torch = torch.rand(10)
        torch.manual_seed(1100)

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        x_st = STensor.from_torch(x_torch)

        result = einsum("ij,j->i", a_st, x_st, format="d")
        expected = torch.zeros(10)
        assert_close(result, expected)

    def test_disjoint_sparsity_elemwise_mul(self):
        """Disjoint sparsity patterns -> zero result."""
        a_torch = torch.zeros(10, 10)
        b_torch = torch.zeros(10, 10)
        # Place non-zeros in disjoint positions
        a_torch[0, 0] = 5.0
        a_torch[1, 1] = 3.0
        b_torch[2, 2] = 7.0
        b_torch[3, 3] = 9.0

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        b_st = STensor.from_torch(b_torch).to_sparse("ds")

        result = einsum("ij,ij->ij", a_st, b_st, format="ds")
        expected = a_torch * b_torch  # all zeros
        assert_close(result, expected)

    def test_very_high_sparsity_spmm(self):
        """99% sparsity SpMM still correct."""
        a_torch = make_sparse_2d(30, 30, 0.99, seed=1102)
        b_torch = make_sparse_2d(30, 30, 0.99, seed=1103)

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        b_st = STensor.from_torch(b_torch).to_sparse("ds")

        result = einsum("ik,kj->ij", a_st, b_st, format="ds")
        expected = torch.matmul(a_torch, b_torch)
        assert_close(result, expected)

    def test_single_nonzero_elemwise_mul(self):
        """Single overlapping non-zero in element-wise mul."""
        a_torch = torch.zeros(8, 8)
        b_torch = torch.zeros(8, 8)
        a_torch[3, 5] = 2.0
        b_torch[3, 5] = 4.0
        # Also add non-overlapping entries
        a_torch[0, 0] = 1.0
        b_torch[7, 7] = 1.0

        a_st = STensor.from_torch(a_torch).to_sparse("ds")
        b_st = STensor.from_torch(b_torch).to_sparse("ds")

        result = einsum("ij,ij->ij", a_st, b_st, format="ds")
        expected = a_torch * b_torch
        assert_close(result, expected)
