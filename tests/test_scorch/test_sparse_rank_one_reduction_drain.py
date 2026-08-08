"""A one-dimensional sparse workspace drain reads a scalar key.

``coo_workspace_1d`` (``src/scorch/csrc/header.h``) dereferences to
``std::pair<int64_t, T>``: its key IS the coordinate, not an indexable
container.  Every other workspace is ``coo_workspace<T, N>``, whose key is a
``std::vector<int64_t>`` and is subscripted per level.

The result drain subscripted the key unconditionally, so every reduction into
a sparse rank-1 result emitted C++ that did not compile --
``scorch_vector_set(T0_crd_vec, pT, it.first[0])`` -> *subscripted value is
not an array, pointer, or vector*.  Public ``scorch.einsum('ij->i')`` over a
compressed matrix failed at build time.  A third reader in the same file
already spelled the rule correctly; the drain now agrees with it.
"""

import pytest
import torch

import scorch
from scorch.stensor import STensor


def sparse(dense, name, fmt):
    tensor = STensor.from_torch(dense.clone(), name)
    if not all(character == "d" for character in fmt):
        tensor.to_sparse(fmt)
    return tensor


@pytest.mark.parametrize("fmt", ["ss", "ds"])
def test_matrix_row_reduction_into_a_sparse_vector_compiles_and_matches(fmt):
    torch.manual_seed(3)
    dense = (torch.rand(4, 5) < 0.4) * torch.randn(4, 5)
    result = scorch.einsum("ij->i", sparse(dense, "A", fmt))
    assert torch.allclose(result.to_torch(), dense.sum(dim=1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fmt", ["ss", "ds"])
def test_column_reduction_into_a_sparse_vector_matches(fmt):
    torch.manual_seed(4)
    dense = (torch.rand(4, 5) < 0.4) * torch.randn(4, 5)
    result = scorch.einsum("ij->j", sparse(dense, "A", fmt))
    assert torch.allclose(result.to_torch(), dense.sum(dim=0), atol=1e-5, rtol=1e-5)


def test_two_contraction_reduction_into_a_vector_is_still_defective():
    """CHARACTERIZATION -- a SECOND, distinct pre-existing defect, now reachable.

    This fix closes the *compile* failure, so a reduction with ONE contracted
    index (``ij->i``, ``ij->j``) now works.  Reducing TWO indices into a rank-1
    sparse vector (``ijk->i``, ``ijk->j``) still produces malformed storage:
    the workspace is rebuilt inside the second contraction loop, so the drain
    emits one entry per (surviving coordinate, outer contracted coordinate)
    pair instead of one per surviving coordinate.  Storage validation catches
    it -- the result is rejected, never silently wrong.

    This is a workspace *placement* defect in the legacy lowering, not the
    key-arity defect this change fixes.  Update this test when it is closed.
    """

    torch.manual_seed(5)
    dense = (torch.rand(3, 4, 5) < 0.4) * torch.randn(3, 4, 5)
    for subscripts in ("ijk->i", "ijk->j"):
        with pytest.raises(Exception) as error:
            scorch.einsum(subscripts, sparse(dense, "A", "sss"))
        assert "strictly increasing" in str(error.value)


def test_float64_reduction_into_a_sparse_vector_matches():
    torch.manual_seed(6)
    dense = ((torch.rand(4, 5) < 0.4) * torch.randn(4, 5)).to(torch.float64)
    result = scorch.einsum("ij->i", sparse(dense, "A", "ss"))
    assert torch.allclose(result.to_torch(), dense.sum(dim=1), atol=1e-12, rtol=1e-12)


def test_all_empty_reduction_into_a_sparse_vector_is_zero():
    dense = torch.zeros(4, 5)
    result = scorch.einsum("ij->i", sparse(dense, "A", "ss"))
    assert torch.allclose(result.to_torch(), torch.zeros(4), atol=1e-6)


def test_ragged_reduction_into_a_sparse_vector_matches():
    dense = torch.zeros(4, 5)
    dense[1] = torch.arange(1.0, 6.0)
    dense[3, 4] = 7.0
    result = scorch.einsum("ij->i", sparse(dense, "A", "ss"))
    assert torch.allclose(result.to_torch(), dense.sum(dim=1), atol=1e-6)


def test_repeated_reduction_is_stable():
    torch.manual_seed(7)
    dense = (torch.rand(4, 5) < 0.4) * torch.randn(4, 5)
    first = scorch.einsum("ij->i", sparse(dense, "A", "ss")).to_torch()
    second = scorch.einsum("ij->i", sparse(dense, "A", "ss")).to_torch()
    assert torch.equal(first, second)


def test_multi_level_workspace_keys_stay_subscripted():
    """The N>1 workspace key is a vector and must keep its per-level index.

    ``ijk->ij`` drains a two-dimensional workspace, so the generated reader
    must still address ``it.first[0]`` and ``it.first[1]``.
    """

    torch.manual_seed(8)
    dense = (torch.rand(3, 4, 5) < 0.4) * torch.randn(3, 4, 5)
    result = scorch.einsum("ijk->ij", sparse(dense, "A", "sss"))
    assert torch.allclose(result.to_torch(), dense.sum(dim=2), atol=1e-5, rtol=1e-5)
