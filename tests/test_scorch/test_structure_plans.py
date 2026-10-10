"""A torch sparse CSR operand's plans belong to its index arrays, not to the tensor object.

Every tensor built over the same two index arrays and shape shares one entry
(csrc/torch_structures.h): its plans, sightings, declines and x @ A records. A program that
builds a new CSR tensor for every call over fixed arrays -- learned edge weights, attention
scores -- is therefore planned from its second call, where keyed on the object it never was.

The entry must live exactly as long as the arrays: nothing in it may hold them (plans borrow
them), and it is dropped when they die. Arrays that differ in any way -- storage, buffer,
length, dtype, or the operand's shape -- never share an entry. A write to the arrays clears
what the entry holds on the next call it does not serve.
"""

import gc
import weakref

import pytest
import torch

import scorch
import scorch_ops
from scorch import plan as plan_mod

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]


def _structure(rows=40, cols=30, density=0.15, index_dtype=torch.int64, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(rows, cols, generator=g) < density
    mask[0, 0] = True  # never empty
    csr = mask.double().to_sparse_csr()
    crow = csr.crow_indices().to(index_dtype).clone()
    col = csr.col_indices().to(index_dtype).clone()
    return crow, col


def _values(n, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g, dtype=torch.float64).to(dtype) + 0.5


def _csr(crow, col, values, shape):
    return torch.sparse_csr_tensor(crow, col, values, size=shape)


def _dense(crow, col, values, shape):
    return torch.sparse_csr_tensor(
        crow.long(), col.long(), values, size=shape
    ).to_dense()


def _tol(dtype):
    return (
        {"rtol": 1e-5, "atol": 1e-5}
        if dtype == torch.float32
        else {"rtol": 0, "atol": 1e-12}
    )


def _plan_of(a):
    plans = plan_mod.plans_of(a)
    assert len(plans) == 1, plans
    return next(iter(plans.values()))


@pytest.fixture(autouse=True)
def _plans_on():
    before = plan_mod.enabled()
    plan_mod.set_enabled(True)
    yield
    plan_mod.set_enabled(before)


# --------------------------------------------------------------------------- #
# Sharing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_a_new_tensor_over_the_same_arrays_every_call_is_planned(index_dtype, dtype):
    crow, col = _structure(index_dtype=index_dtype)
    shape = (40, 30)
    b = torch.rand(30, 8, dtype=dtype)
    for step in range(6):
        values = _values(col.numel(), dtype, step)
        a = _csr(crow, col, values, shape)
        out = scorch.matmul(a, b)
        torch.testing.assert_close(
            out, _dense(crow, col, values, shape) @ b, **_tol(dtype)
        )
        if step == 0:
            assert not plan_mod.plans_of(a), "a first sighting installs nothing"
    plan = _plan_of(a)
    assert plan.borrows
    assert (
        plan.served == 4
    ), "sighted on call 1, installed on call 2, served on calls 3-6"


def test_tensors_over_the_same_arrays_share_one_state():
    crow, col = _structure()
    a1 = _csr(crow, col, _values(col.numel(), torch.float64, 1), (40, 30))
    a2 = _csr(crow, col, _values(col.numel(), torch.float64, 2), (40, 30))
    state = plan_mod.torch_operand_state(a1, create=True)
    assert state is not None
    assert plan_mod.torch_operand_state(a2) is state


def _distinct_structures():
    crow, col = _structure()
    copy_crow, copy_col = crow.clone(), col.clone()  # equal contents, other storages
    wider = (40, 31)  # the same arrays under another shape
    narrowed = (crow.to(torch.int32), col.to(torch.int32))
    return (
        crow,
        col,
        [
            ("copies", copy_crow, copy_col, (40, 30)),
            ("shape", crow, col, wider),
            ("dtype", narrowed[0], narrowed[1], (40, 30)),
        ],
    )


@pytest.mark.parametrize("which", ["copies", "shape", "dtype"])
def test_tensors_over_different_arrays_never_share_a_state(which):
    crow, col, others = _distinct_structures()
    _, ocrow, ocol, oshape = next(o for o in others if o[0] == which)
    a = _csr(crow, col, _values(col.numel(), torch.float64, 1), (40, 30))
    other = _csr(ocrow, ocol, _values(ocol.numel(), torch.float64, 2), oshape)
    state = plan_mod.torch_operand_state(a, create=True)
    assert plan_mod.torch_operand_state(other) is None
    assert plan_mod.torch_operand_state(other, create=True) is not state


def test_a_slice_of_the_same_buffer_is_another_structure():
    """Two arrays in one storage at different offsets: same storage, different buffers."""
    crow, col = _structure(rows=10, cols=10, density=0.3)
    nnz = col.numel()
    doubled = torch.cat([col, col])
    a = _csr(crow, doubled[:nnz], torch.ones(nnz, dtype=torch.float64), (10, 10))
    b = _csr(crow, doubled[nnz:], torch.ones(nnz, dtype=torch.float64), (10, 10))
    assert plan_mod.torch_operand_state(
        a, create=True
    ) is not plan_mod.torch_operand_state(b, create=True)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_different_values_over_one_structure_give_their_own_products(index_dtype):
    crow, col = _structure(index_dtype=index_dtype)
    shape = (40, 30)
    b = torch.rand(30, 5, dtype=torch.float64)
    tensors = [
        _csr(crow, col, _values(col.numel(), torch.float64, s), shape) for s in range(4)
    ]
    for _ in range(3):  # interleaved, so one shared plan serves all four
        for a in tensors:
            torch.testing.assert_close(
                scorch.matmul(a, b), a.to_dense() @ b, **_tol(torch.float64)
            )
    assert _plan_of(tensors[0]) is _plan_of(tensors[3])
    assert _plan_of(tensors[0]).served >= 8


def test_float32_and_float64_values_over_one_structure_get_separate_plans():
    crow, col = _structure()
    shape = (40, 30)
    a32 = _csr(crow, col, _values(col.numel(), torch.float32, 1), shape)
    a64 = _csr(crow, col, _values(col.numel(), torch.float64, 1), shape)
    b32, b64 = torch.rand(30, 4), torch.rand(30, 4, dtype=torch.float64)
    for _ in range(3):
        torch.testing.assert_close(
            scorch.matmul(a32, b32), a32.to_dense() @ b32, **_tol(torch.float32)
        )
        torch.testing.assert_close(
            scorch.matmul(a64, b64), a64.to_dense() @ b64, **_tol(torch.float64)
        )
    plans = plan_mod.plans_of(a32)
    assert len(plans) == 2 and all(p.served == 1 for p in plans.values())


def test_a_borrowing_plan_serves_only_through_run_csr():
    crow, col = _structure()
    a = _csr(crow, col, _values(col.numel(), torch.float32, 0), (40, 30))
    b = torch.rand(30, 4)
    for _ in range(3):
        scorch.matmul(a, b)
    plan = _plan_of(a)
    assert plan.borrows
    assert plan.run(a.values(), b, -1, False) is None
    members = scorch_ops.torch_sparse_index_members(a)
    out = plan.run_csr(*members, a.values(), b, -1, False)
    torch.testing.assert_close(out, a.to_dense() @ b, **_tol(torch.float32))


# --------------------------------------------------------------------------- #
# Lifetime
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_the_entry_dies_with_the_arrays_not_before(index_dtype):
    crow, col = _structure(index_dtype=index_dtype)
    b = torch.rand(30, 4, dtype=torch.float64)
    before = scorch_ops.torch_structure_count()
    for step in range(3):
        a = _csr(crow, col, _values(col.numel(), torch.float64, step), (40, 30))
        scorch.matmul(a, b)
    assert _plan_of(a).served == 1
    del a
    gc.collect()
    assert scorch_ops.torch_structure_count() == before + 1, "the arrays still live"
    died = []
    weakref.finalize(col.untyped_storage(), died.append, True)
    del crow, col
    gc.collect()
    assert died, "nothing scorch keeps holds the arrays"
    assert scorch_ops.torch_structure_count() == before


@pytest.mark.parametrize("calls", [1, 3])
def test_new_arrays_where_dead_ones_were_are_a_new_structure(calls):
    """An entry keeps its storages' StorageImpls from being freed until it is dropped -- at once
    for a planned structure (calls=3), at the next sweep for one only sighted (calls=1) -- so
    arrays allocated afterwards, often at the same buffer addresses, never find it."""
    shape = (40, 30)
    b = torch.rand(30, 4, dtype=torch.float64)
    crow, col = _structure(seed=1)
    for step in range(calls):
        scorch.matmul(
            _csr(crow, col, _values(col.numel(), torch.float64, step), shape), b
        )
    addresses = (crow.data_ptr(), col.data_ptr())
    del crow, col
    gc.collect()
    hits = 0
    for seed in range(20):
        crow, col = _structure(seed=100 + seed)
        a = _csr(crow, col, _values(col.numel(), torch.float64, seed), shape)
        hits += (crow.data_ptr(), col.data_ptr()) == addresses
        assert plan_mod.torch_operand_state(a) is None
        torch.testing.assert_close(
            scorch.matmul(a, b), a.to_dense() @ b, **_tol(torch.float64)
        )
        del a, crow, col
    # Not asserted: whether the allocator reused the addresses is up to it.
    print("address reuse in", hits, "of 20")


def test_structures_used_in_turn_keep_their_own_plans():
    """The registry remembers the entry it found last (torch_structures.h, last_entry): two
    structures used in turn must each find their own entry every time, not the other's."""
    shape = (40, 30)
    b = torch.rand(30, 8, dtype=torch.float64)
    first = _structure(seed=3)
    second = _structure(seed=4)
    operands = []
    for step in range(8):
        crow, col = first if step % 2 == 0 else second
        values = _values(col.numel(), torch.float64, step)
        a = _csr(crow, col, values, shape)
        torch.testing.assert_close(
            scorch.matmul(a, b), _dense(crow, col, values, shape) @ b, **_tol(torch.float64)
        )
        operands.append(a)
    plan_first, plan_second = _plan_of(operands[-2]), _plan_of(operands[-1])
    assert plan_first is not plan_second
    assert plan_first.served == 2 and plan_second.served == 2, (
        "each sighted on its first call, installed on its second, served on its third and fourth"
    )


def _storage_key(crow, col):
    return (
        crow.untyped_storage()._cdata,
        col.untyped_storage()._cdata,
        crow.data_ptr(),
        col.data_ptr(),
    )


def test_the_entry_found_last_is_forgotten_when_it_is_dropped():
    """A planned structure's entry is dropped the moment its arrays die (its weak-reference
    callback), and it is the entry the registry found last. Arrays allocated next often reuse the
    same StorageImpls and buffers -- the whole key -- and must then find no entry, not the
    dropped one."""
    shape = (40, 30)
    b = torch.rand(30, 4, dtype=torch.float64)
    reused = 0
    for trial in range(20):
        crow, col = _structure(seed=200 + trial)
        for step in range(3):
            scorch.matmul(
                _csr(crow, col, _values(col.numel(), torch.float64, step), shape), b
            )
        old = _storage_key(crow, col)
        # Found last, through the same lookup a product makes.
        assert plan_mod.torch_operand_state(
            _csr(crow, col, _values(col.numel(), torch.float64, 9), shape)
        ) is not None
        n = col.numel()
        del crow, col
        gc.collect()
        crow, col = _structure(seed=200 + trial)  # the same lengths, so the same allocations
        assert col.numel() == n
        reused += _storage_key(crow, col) == old
        a = _csr(crow, col, _values(n, torch.float64, 5), shape)
        assert plan_mod.torch_operand_state(a) is None
        torch.testing.assert_close(
            scorch.matmul(a, b), a.to_dense() @ b, **_tol(torch.float64)
        )
        del a, crow, col
        gc.collect()
    # Not asserted: whether the allocator reused the whole key is up to it.
    print("whole key reused in", reused, "of 20")


def test_first_sightings_of_arrays_that_die_stay_bounded():
    """A first sighting adds an entry without the weak references that drop it (they would be
    most of what a call over new arrays costs); a sweep drops such entries once the registry has
    doubled, so a program making new arrays for every call keeps a bounded registry."""
    b = torch.rand(30, 4, dtype=torch.float64)
    gc.collect()
    start = scorch_ops.torch_structure_count()
    peak = start
    for seed in range(2000):
        crow, col = _structure(seed=seed)
        scorch.matmul(
            _csr(crow, col, _values(col.numel(), torch.float64, seed), (40, 30)), b
        )
        peak = max(peak, scorch_ops.torch_structure_count())
    # Swept once the registry doubles past what the last sweep left (at least 64 entries): a
    # registry that kept them would reach start + 2000.
    assert peak - start < 500


def test_planned_structures_that_die_leave_nothing():
    b = torch.rand(30, 4, dtype=torch.float64)
    gc.collect()
    # From a swept registry: an earlier test's first sightings whose arrays have died are dropped
    # by a sweep, which may run during this one.
    start = scorch_ops.torch_structure_sweep()
    for seed in range(300):
        crow, col = _structure(seed=seed)
        a = _csr(crow, col, _values(col.numel(), torch.float64, seed), (40, 30))
        for _ in range(3):
            scorch.matmul(a, b)
        assert plan_mod.plans_of(a)
        del a, crow, col
    gc.collect()
    assert scorch_ops.torch_structure_count() == start


def test_ten_thousand_planned_structures_leave_memory_flat():
    psutil = pytest.importorskip("psutil")
    process = psutil.Process()
    b = torch.rand(300, 8)

    def churn(count, offset):
        for seed in range(count):
            crow, col = _structure(rows=400, cols=300, density=0.02, seed=offset + seed)
            a = _csr(crow, col, _values(col.numel(), torch.float32, seed), (400, 300))
            for _ in range(3):
                scorch.matmul(a, b)
            del a, crow, col

    churn(1000, 0)  # warm every allocator pool first
    gc.collect()
    before = process.memory_info().rss
    churn(10000, 1000)
    gc.collect()
    grown = process.memory_info().rss - before
    # A leaked structure here holds its arrays (~2.4k int64 nonzeros, ~20 KB) and an int32
    # copy: 10k of them is ~300 MB. Allow allocator noise well below that.
    assert grown < 40 * 2**20, f"RSS grew {grown / 2**20:.1f} MB over 10k structures"


def test_a_thousand_tensors_over_one_structure_keep_the_witness_list_short():
    crow, col = _structure()
    b = torch.rand(30, 4, dtype=torch.float64)
    for step in range(1000):
        a = _csr(crow, col, _values(col.numel(), torch.float64, step), (40, 30))
        scorch.matmul(a, b)
        del a
    assert scorch_ops.abi_witness_count(col) <= 4
    assert scorch_ops.abi_witness_count(crow) <= 4


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #


def _rows_one_each(index_dtype=torch.int64):
    crow = torch.arange(7, dtype=index_dtype)
    col = torch.tensor([0, 1, 2, 3, 4, 0], dtype=index_dtype)
    return crow, col


NEW_COL = [4, 3, 2, 1, 0, 4]


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize(
    "route", ["caller_array", "old_tensor_accessor", "new_tensor_accessor"]
)
def test_a_write_reaches_every_tensor_over_the_arrays(index_dtype, route):
    crow, col = _rows_one_each(index_dtype)
    shape = (6, 5)
    b = torch.rand(5, 3, dtype=torch.float64)
    old = None
    for step in range(4):
        old = _csr(crow, col, _values(6, torch.float64, step), shape)
        scorch.matmul(old, b)
    assert _plan_of(old).served >= 1
    new = _csr(crow, col, _values(6, torch.float64, 9), shape)
    replacement = torch.tensor(NEW_COL, dtype=index_dtype)
    with torch.no_grad():
        if route == "caller_array":
            col.copy_(replacement)
        elif route == "old_tensor_accessor":
            old.col_indices().copy_(replacement)
        else:
            new.col_indices().copy_(replacement)
    fresh = _csr(crow, col, _values(6, torch.float64, 10), shape)
    for a in (new, fresh, old):
        torch.testing.assert_close(
            scorch.matmul(a, b), a.to_dense() @ b, **_tol(torch.float64)
        )


@pytest.mark.parametrize("write", ["values_accessor", "col_accessor"])
def test_a_structure_written_once_is_planned_again(write):
    """A write clears the entry on the next call no plan serves; the second sighting after it
    installs a plan over the arrays as they are now."""
    crow, col = _rows_one_each()
    shape = (6, 5)
    b = torch.rand(5, 3, dtype=torch.float64)
    for step in range(3):
        scorch.matmul(_csr(crow, col, _values(6, torch.float64, step), shape), b)
    a = _csr(crow, col, _values(6, torch.float64, 5), shape)
    first = _plan_of(a)
    with torch.no_grad():
        if write == "values_accessor":
            a.values().mul_(2.0)
        else:
            a.col_indices().copy_(torch.tensor(NEW_COL))
    for step in range(4):
        x = _csr(crow, col, _values(6, torch.float64, 20 + step), shape)
        torch.testing.assert_close(
            scorch.matmul(x, b), x.to_dense() @ b, **_tol(torch.float64)
        )
    second = _plan_of(x)
    assert second is not first
    assert (
        second.served == 2
    ), "cleared on call 1, sighted on call 1, installed on call 2"


def test_values_written_before_every_call_are_never_planned(monkeypatch):
    """torch moves one counter for a write through values() and one through col_indices(), so
    either reads as a possible write to the arrays. A structure written before every call is
    cleared on every call and so never sighted twice in one state: nothing is built for it.
    """
    crow, col = _rows_one_each()
    a = _csr(crow, col, _values(6, torch.float64, 0), (6, 5))
    b = torch.rand(5, 3, dtype=torch.float64)
    for step in range(3):
        scorch.matmul(a, b)
    assert _plan_of(a).served == 1
    built = []
    real = scorch_ops.make_spmm_csr_plan

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scorch_ops, "make_spmm_csr_plan", counting)
    for step in range(10):
        with torch.no_grad():
            a.values().add_(1.0)
        torch.testing.assert_close(
            scorch.matmul(a, b), a.to_dense() @ b, **_tol(torch.float64)
        )
    assert (
        built == []
    ), f"{len(built)} plans built for a structure written before every call"


def test_forget_drops_the_structure_s_plans_for_every_tensor_over_it():
    crow, col = _structure()
    b = torch.rand(30, 4, dtype=torch.float64)
    a1 = _csr(crow, col, _values(col.numel(), torch.float64, 1), (40, 30))
    a2 = _csr(crow, col, _values(col.numel(), torch.float64, 2), (40, 30))
    for _ in range(3):
        scorch.matmul(a1, b)
    assert plan_mod.plans_of(a2)
    plan_mod.forget(a1)
    assert not plan_mod.plans_of(a2)


def test_the_state_of_a_tensor_that_is_not_a_2d_csr_is_none():
    assert scorch_ops.torch_structure_state(torch.eye(3).to_sparse()) is None
    assert scorch_ops.torch_structure_state(torch.eye(3)) is None
    batched = torch.stack([torch.eye(3), torch.eye(3)]).to_sparse_csr()
    assert scorch_ops.torch_structure_state(batched, create=True) is None
    with pytest.raises(TypeError):
        scorch_ops.torch_structure_state(object())


# --------------------------------------------------------------------------- #
# x @ A
# --------------------------------------------------------------------------- #


def _right_served(a):
    return sum(tp.served for tp in plan_mod.transposed_plans_of(a).values())


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_x_times_a_s_record_is_shared_by_new_tensors_over_the_arrays(dtype):
    crow, col = _structure()
    shape = (40, 30)
    x = torch.rand(3, 40, dtype=dtype)
    for step in range(4):
        a = _csr(crow, col, _values(col.numel(), dtype, step), shape)
        with torch.no_grad():
            torch.testing.assert_close(
                scorch.matmul(x, a), x @ a.to_dense(), **_tol(dtype)
            )
    assert plan_mod.transposed_plans_of(
        a
    ), "precondition: the structure records its x @ A"
    before = _right_served(a)
    fresh = _csr(crow, col, _values(col.numel(), dtype, 99), shape)
    with torch.no_grad():
        torch.testing.assert_close(
            scorch.matmul(x, fresh), x @ fresh.to_dense(), **_tol(dtype)
        )
    assert _right_served(fresh) == before + 1


def test_x_times_a_s_record_does_not_keep_the_arrays_alive():
    crow, col = _structure()
    x = torch.rand(3, 40, dtype=torch.float64)
    for step in range(4):
        a = _csr(crow, col, _values(col.numel(), torch.float64, step), (40, 30))
        with torch.no_grad():
            scorch.matmul(x, a)
    assert plan_mod.transposed_plans_of(a)
    died = []
    weakref.finalize(col.untyped_storage(), died.append, True)
    del a, crow, col
    gc.collect()
    assert died


def test_first_sightings_keep_no_python_containers_alive():
    """A first sighting through the C entry is kept as plain data in its structure's entry, not
    as plan.py's SEEN set in a dict: two containers the garbage collector would track for as long
    as the arrays live (torch_structures.h, TorchStructureEntry::sighted)."""
    b = torch.rand(30, 4)
    kept = []
    for seed in range(200):
        crow, col = _structure(seed=seed)
        kept.append(
            _csr(crow, col, _values(col.numel(), torch.float32, seed), (40, 30))
        )
    scorch.matmul(kept[0], b)  # anything made once per process is made here
    gc.collect()
    before = len(gc.get_objects())
    structures = scorch_ops.torch_structure_sweep()
    for a in kept[1:]:
        scorch.matmul(a, b)
    gc.collect()
    assert (
        scorch_ops.torch_structure_count() >= structures + 150
    ), "the sightings were not kept"
    assert len(gc.get_objects()) - before < 20, "first sightings kept Python containers"
    # Python reading a structure gets what C sighted, as if it had been kept in the dict.
    key = (b.shape, b.dtype, plan_mod.GENERATION[0])
    assert all(plan_mod.torch_operand_state(a)[plan_mod.SEEN_ATTR] == {key} for a in kept)
