"""Index sharing is a mechanism, so it is tested by pointer identity, not by values.

``STensor.from_torch`` / ``from_csr`` / ``from_coo`` adopt a torch sparse tensor's index
arrays instead of copying them (``TensorIndex(..., _adopt=True)``), and a generated
kernel's result keeps the arrays the kernel wrote (``_trusted_index``). Every numeric test
passes whether or not the arrays are shared -- a copy holds the same values -- so the
only assertion that can see the mechanism go away is that the storage's arrays are the
caller's own buffers. The write witnesses (native_abi.h) and the plans keyed on index
arrays rely on that sharing.
"""

import torch

import scorch
from scorch import STensor


def _stored(st: STensor):
    return st.storage.index._mode_indices


def _csr(dtype=torch.int64):
    dense = torch.zeros(6, 5)
    dense[0, 1] = 1.0
    dense[2, 0] = 2.0
    dense[2, 4] = 3.0
    dense[5, 3] = 4.0
    s = dense.to_sparse_csr()
    crow = s.crow_indices().to(dtype).clone()
    col = s.col_indices().to(dtype).clone()
    return torch.sparse_csr_tensor(crow, col, s.values().clone(), (6, 5)), crow, col


def test_from_torch_csr_shares_the_callers_index_arrays():
    for dtype in (torch.int64, torch.int32):
        a, crow, col = _csr(dtype)
        st = STensor.from_torch(a)
        level = _stored(st)[1]
        assert level[0].data_ptr() == crow.data_ptr(), dtype
        assert level[1].data_ptr() == col.data_ptr(), dtype


def test_from_torch_csr_shares_through_the_sparse_tensors_members():
    a, _, _ = _csr()
    st = STensor.from_torch(a)
    level = _stored(st)[1]
    assert level[0].data_ptr() == a.crow_indices().data_ptr()
    assert level[1].data_ptr() == a.col_indices().data_ptr()


def test_storage_built_from_an_index_does_not_copy_it():
    a, _, _ = _csr()
    st = STensor.from_torch(a)
    before = [t.data_ptr() for arrays in _stored(st) for t in arrays]
    rebuilt = scorch.storage.SparseStorage(
        st.storage.layout, st.storage.value, index=st.storage.index
    )
    after = [t.data_ptr() for arrays in rebuilt._mode_indices for t in arrays]
    assert after == before


def test_the_storage_audit_sees_exactly_its_three_fields():
    a, _, _ = _csr()
    st = STensor.from_torch(a)
    storage = st.storage
    storage.validate_unless_already_checked()  # sets the stamp
    assert set(vars(storage)) == {"_layout", "_mode_indices", "_value"}
    # And a storage handed back in through the public constructor is accepted.
    again = STensor(shape=st.shape, storage=storage)
    assert torch.equal(again.to_torch(), st.to_torch())
