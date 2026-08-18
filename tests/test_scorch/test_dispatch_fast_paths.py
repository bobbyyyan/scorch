"""The per-call fast paths must be indistinguishable from the paths they replace.

Four shortcuts remove work that a small ``scorch.matmul`` or ``scorch.einsum`` was
repeating on every call: ``parse_format`` hands back a shared ``TensorFormat``,
``TensorLayout.from_physical_shape`` hands back a shared layout, ``STensor.from_torch``
assembles a dense operand from cached immutable parts instead of re-deriving them, and the
einsum dispatch key holds the frozen layout instead of a JSON rendering of it. Each is
only legitimate if what comes out is what the ordinary path would have produced -- for the
caches, that a shared value object cannot be mutated through one holder and observed
through another; for the key, that it separates exactly the tensors the string separated.
"""

import pytest
import torch

import scorch
from scorch import STensor, TensorFormat, TensorLayout, parse_format
from scorch.format import LevelFormat
import scorch.ops as ops_module
import scorch.stensor as stensor_module
from scorch.exceptions import CompileSpecError, TensorTypeError
from scorch.ops import _einsum_cache_key, _logical_index_sizes, _validated_labels
from scorch.stensor import _DENSE_PARTS_CACHE


def _fields(t):
    """Everything observable about an STensor's structure, for a field-by-field diff."""
    storage = t.storage
    return dict(
        shape=tuple(t.shape),
        fmt=str(t.format),
        dtype=t.values.dtype,
        device=t.values.device,
        mode_order=tuple(storage.index.mode_order),
        index_dtype=storage.layout.index_dtype,
        logical=tuple(storage.layout.logical_shape),
        physical=tuple(storage.layout.physical_shape),
        n_levels=len(storage.index.mode_indices),
        name=t.name,
    )


def test_dense_from_torch_cached_matches_first_build():
    """The cached assembly agrees with the build that populated the cache."""
    _DENSE_PARTS_CACHE.clear()
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cold = STensor.from_torch(x, "A")  # populates
    warm = STensor.from_torch(x, "A")  # assembles from the cache
    assert _fields(cold) == _fields(warm)
    assert torch.equal(cold.values, warm.values)
    assert torch.equal(cold.to_torch(in_place=False), warm.to_torch(in_place=False))


def test_dense_from_torch_cache_key_separates_shape_dtype_name_and_order():
    """Anything the cached parts depend on has to be part of the key."""
    _DENSE_PARTS_CACHE.clear()
    base = torch.zeros((2, 3), dtype=torch.float32)
    variants = [
        STensor.from_torch(base, "A"),
        STensor.from_torch(torch.zeros((3, 2), dtype=torch.float32), "A"),
        STensor.from_torch(torch.zeros((2, 3), dtype=torch.float64), "A"),
        STensor.from_torch(base, "B"),
        STensor.from_torch(base, "A", mode_order=[1, 0]),
    ]
    seen = {tuple(sorted(_fields(v).items(), key=lambda kv: kv[0])) for v in variants}
    assert len(seen) == len(variants), "a key component is not distinguishing entries"


def test_dense_from_torch_values_are_not_shared_between_tensors():
    """Only the immutable parts are shared; two operands must not alias each other."""
    _DENSE_PARTS_CACHE.clear()
    first = STensor.from_torch(torch.ones((2, 2), dtype=torch.float32))
    second = STensor.from_torch(torch.full((2, 2), 5.0, dtype=torch.float32))
    assert first.values.data_ptr() != second.values.data_ptr()
    assert torch.equal(first.values, torch.ones(4))
    assert torch.equal(second.values, torch.full((4,), 5.0))


def test_dense_from_torch_reflects_later_writes_like_the_ordinary_path():
    """Values alias the source tensor exactly as the uncached construction did."""
    _DENSE_PARTS_CACHE.clear()
    x = torch.zeros((2, 2), dtype=torch.float32)
    STensor.from_torch(x)  # populate, so the next call takes the cached branch
    y = torch.zeros((2, 2), dtype=torch.float32)
    wrapped = STensor.from_torch(y)
    y[0, 0] = 7.0
    assert wrapped.values[0].item() == 7.0


def test_parse_format_returns_equal_formats_for_every_spelling():
    assert parse_format("ds") == parse_format("d,s") == TensorFormat("ds")
    assert parse_format(["dense", "compressed"]) == parse_format("ds")
    already = TensorFormat("oo")
    assert parse_format(already) is already


def test_layout_cache_returns_equal_layouts_and_rejects_bad_input():
    a = TensorLayout.from_physical_shape((4, 5), "ds")
    b = TensorLayout.from_physical_shape([4, 5], "ds")
    assert a == b
    assert a.logical_shape == (4, 5)
    # A permutation is part of the identity, not something the cache may ignore.
    p = TensorLayout.from_physical_shape((4, 5), "ds", permutation=[1, 0])
    assert p.logical_shape == (5, 4)
    assert p != a


def test_shared_layout_cannot_be_mutated_through_a_holder():
    """The cached parts are frozen; an attempt to write one must still raise."""
    layout = TensorLayout.from_physical_shape((2, 2), "dd")
    for field, value in (("index_dtype", torch.int64), ("permutation", (1, 0))):
        try:
            setattr(layout, field, value)
        except Exception:
            continue
        raise AssertionError(f"{field} was mutable on a shared TensorLayout")


def test_matmul_result_unchanged_across_repeated_calls():
    """The caches must not make the second call of a repeated matmul differ."""
    torch.manual_seed(0)
    dense = (torch.rand(24, 24) < 0.3).float()
    A = STensor.from_torch(dense.to_sparse_csr())
    B = torch.rand(24, 6)
    reference = torch.matmul(dense.double(), B.double()).float()
    outs = [scorch.matmul(A, B) for _ in range(3)]
    for out in outs:
        got = out if isinstance(out, torch.Tensor) else out.to_torch(in_place=False)
        torch.testing.assert_close(got, reference, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(got, outs[0], atol=0, rtol=0)


def test_raw_values_matches_values():
    """_raw_values skips a defensive copy, not a transformation."""
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    t = STensor.from_torch(x)
    assert torch.equal(t._raw_values, t.values)
    assert t._raw_values.dtype == t.values.dtype
    assert not t._raw_values.requires_grad


def _layout_variants():
    """One layout per field of TensorLayout, each differing from `base` in that field."""
    base = TensorLayout.from_physical_shape((4, 5), "ds", None, torch.int32)
    return base, {
        "physical_shape": TensorLayout.from_physical_shape(
            (4, 6), "ds", None, torch.int32
        ),
        "format": TensorLayout.from_physical_shape((4, 5), "dd", None, torch.int32),
        # A permutation moves logical_shape too, which is the fifth field.
        "permutation": TensorLayout.from_physical_shape(
            (4, 5), "ds", (1, 0), torch.int32
        ),
        "index_dtype": TensorLayout.from_physical_shape(
            (4, 5), "ds", None, torch.int64
        ),
        "level bit_width": TensorLayout(
            (4, 5),
            (4, 5),
            TensorFormat([LevelFormat("d"), LevelFormat("s", bit_width=64)]),
            (0, 1),
            torch.int32,
        ),
    }


def test_layout_is_hashable_and_discriminates_every_field_like_serialize():
    """The dispatch key holds the layout itself, so its __eq__ must match serialize().

    ``_einsum_cache_key`` used to call ``layout.serialize()`` per operand -- 2.3 us of
    ``json.dumps`` to obtain a key for an in-process dict. It now keys on the frozen
    layout. That is only sound if the generated ``__eq__``/``__hash__`` separate exactly
    what the JSON separated: every field, including a level's bit_width, which lives two
    value objects down. ``_fill_value`` is a ClassVar rather than a field, so it is the
    one thing in ``to_dict()`` the key omits -- and it cannot vary between instances.
    """
    base, variants = _layout_variants()
    assert hash(base) == hash(TensorLayout.from_physical_shape((4, 5), "ds"))
    for field, other in variants.items():
        assert (other == base) is (other.serialize() == base.serialize()), field
        assert other != base, field
        assert hash(other) != hash(base), field

    # Same question asked the way the cache asks it: identical bucketing.
    by_object: dict = {}
    by_string: dict = {}
    for field, layout in (("base", base), *variants.items()):
        by_object.setdefault(layout, []).append(field)
        by_string.setdefault(layout.serialize(), []).append(field)
    assert sorted(by_object.values()) == sorted(by_string.values())


def test_einsum_dispatch_key_discriminates_every_layout_field():
    """The same field-by-field guarantee, through the real key builder."""
    base, variants = _layout_variants()

    class FakeTensor:
        def __init__(self, layout):
            self.layout = layout
            self.dtype = torch.float32
            self.device = torch.device("cpu")

    def key(layout):
        return _einsum_cache_key(
            "ij,jk->ik", (FakeTensor(layout), FakeTensor(base)), "dd", None, None
        )

    assert key(base) == key(TensorLayout.from_physical_shape((4, 5), "ds"))
    keys = {field: key(layout) for field, layout in variants.items()}
    assert len(set(keys.values())) == len(keys)
    for field, k in keys.items():
        assert k != key(base), field
    # And the key must still be usable as a dict key, which is the whole point.
    assert len({key(base): 0, **{k: 0 for k in keys.values()}}) == len(keys) + 1


def test_layout_without_serialize_falls_back_without_colliding():
    """A tensor-like that is not one of ours still gets a discriminating contract."""

    class Foreign:
        dtype = torch.float32
        device = torch.device("cpu")

        def __init__(self, fmt, shape):
            self.format = fmt
            self.shape = shape
            self.mode_order = (0, 1)
            self.index_dtype = torch.int32

    def key(t):
        return _einsum_cache_key("ij,jk->ik", (t, t), "dd", None, None)

    assert key(Foreign("d,s", (4, 5))) == key(Foreign("d,s", (4, 5)))
    assert key(Foreign("d,s", (4, 5))) != key(Foreign("d,d", (4, 5)))
    assert key(Foreign("d,s", (4, 5))) != key(Foreign("d,s", (4, 6)))

    # A foreign layout that offers its own canonical string keeps being keyed on it.
    class Serializing:
        dtype = torch.float32
        device = torch.device("cpu")

        class layout:  # noqa: N801 - an attribute that happens to be a class
            @staticmethod
            def serialize():
                return "foreign-canonical-form"

    assert "foreign-canonical-form" in repr(key(Serializing()))


def test_validated_labels_memo_returns_fresh_lists():
    """Callers downstream sort and rewrite these, so the memo must not hand out its own."""
    ops_module._EXPR_LABELS_CACHE.pop("ik,kj->ij", None)
    first_in, first_out = _validated_labels("ik,kj->ij", ["ik", "kj"], "ij")
    first_in[0][0] = "MUTATED"
    first_out[0] = "MUTATED"
    second_in, second_out = _validated_labels("ik,kj->ij", ["ik", "kj"], "ij")
    assert second_in == [["i", "k"], ["k", "j"]]
    assert second_out == ["i", "j"]
    assert second_in is not first_in and second_out is not first_out


@pytest.mark.parametrize(
    ("expression", "groups", "result", "message"),
    [
        ("i&,kj->ij", ["i&", "kj"], "ij", "input labels must be non-empty"),
        (",kj->ij", ["", "kj"], "ij", "input labels must be non-empty"),
        ("ik,kj->i&", ["ik", "kj"], "i&", "result labels must be non-empty"),
        ("ik,kj->ii", ["ik", "kj"], "ii", "result labels must be unique"),
        ("ik,kj->iz", ["ik", "kj"], "iz", "must appear in an input"),
    ],
)
def test_validated_labels_rejects_the_same_expressions_cold_and_warm(
    expression, groups, result, message
):
    """A malformed expression is never memoized, so it must raise every time."""
    for _ in range(2):
        with pytest.raises(CompileSpecError, match=message):
            _validated_labels(expression, groups, result)
    assert expression not in ops_module._EXPR_LABELS_CACHE


def test_validated_labels_memo_respects_its_bound(monkeypatch):
    """Past the bound the checks simply run, which is what happened before the memo."""
    monkeypatch.setattr(ops_module, "_EXPR_LABELS_CACHE_MAX", 0)
    monkeypatch.setattr(ops_module, "_EXPR_LABELS_CACHE", {})
    assert _validated_labels("ab,bc->ac", ["ab", "bc"], "ac") == (
        [["a", "b"], ["b", "c"]],
        ["a", "c"],
    )
    assert ops_module._EXPR_LABELS_CACHE == {}


def test_a_malformed_expression_still_reports_the_operand_problem_first():
    """The memo is consulted where the checks ran, so error precedence is unchanged.

    Both of these expressions have an illegal label. Neither error is the one a caller
    sees, because the operand count and the None check come first -- and they still do.
    """
    B = torch.rand(4, 4)
    with pytest.raises(CompileSpecError, match="expects 2 operands"):
        scorch.einsum("i&,kj->ij", B)
    with pytest.raises(TensorTypeError, match="operands cannot be None"):
        scorch.einsum("i&,kj->ij", None, B)


def test_logical_index_sizes_survive_relayout():
    """Why the cached dispatch path may reuse the map the validating call built.

    It only reuses it when no operand was relayouted, so this is not load-bearing -- but
    the comment there claims the invariance, and a claim in a comment should be a test.
    """
    torch.manual_seed(0)
    A = STensor.from_torch((torch.rand(5, 7) < 0.4).float().to_sparse_csr())
    before = _logical_index_sizes([["i", "k"]], (A,))
    assert before == {"i": 5, "k": 7}
    transposed = A.copy()
    transposed.change_mode_order([1, 0])
    assert tuple(transposed.shape) == (7, 5)
    assert _logical_index_sizes([["i", "k"]], (transposed,)) == before


# --------------------------------------------------------------------------- #
# Materializing a non-contiguous dense operand
# --------------------------------------------------------------------------- #


@pytest.fixture
def copy_arms(monkeypatch):
    """Set the two levers explicitly per test, and start from an empty memo."""

    def select(*, memo, native):
        monkeypatch.setattr(stensor_module, "_MEMO_OPERAND_COPY", [memo])
        monkeypatch.setattr(
            stensor_module,
            "_NATIVE_TRANSPOSE",
            stensor_module._NATIVE_TRANSPOSE if native else None,
        )
        monkeypatch.setattr(stensor_module, "_OPERAND_COPY_CACHE", {})

    return select


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (8, 8), (5, 37), (64, 3)])
def test_transposed_operand_materializes_bit_identically(shape):
    """The cache-blocked transpose has to be `.contiguous()`, not an approximation of it."""
    torch.manual_seed(0)
    base = torch.rand(*shape)
    operand = base.T
    assert not operand.is_contiguous() or shape[0] == 1 or shape[1] == 1
    got = stensor_module._contiguous_copy(operand)
    torch.testing.assert_close(got, operand.contiguous(), atol=0, rtol=0)
    assert got.is_contiguous()
    assert tuple(got.shape) == tuple(operand.shape)


def test_every_other_layout_materializes_exactly_too():
    """Whichever branch a layout takes, the copy has to equal `.contiguous()`.

    The first four fall to `.contiguous()` -- wrong dtype, wrong rank, and two views whose
    transpose is not itself contiguous. The last two do take the kernel, which is worth
    pinning separately: both are column-major at a non-zero storage offset, and the kernel
    reads through `data_ptr()`, so the offset has to land where it belongs.

    Which of these lands where is not obvious from reading them. A column slice of a
    transpose keeps transpose-contiguity and a row slice of one does not, so the two are
    asserted below rather than assumed.
    """
    torch.manual_seed(0)
    fallback = [
        torch.rand(8, 8).double().T,  # not float32
        torch.rand(4, 5, 6).permute(2, 0, 1),  # not 2-D
        torch.rand(16, 8)[::2],  # strided rows, so the transpose is not contiguous
        torch.rand(12, 6).T[2:5],  # a *row* slice of a transpose: also not contiguous
    ]
    kernel = [
        torch.rand(8, 8).T[:, 1:5],  # a column slice of a transpose: offset 8
        torch.rand(16, 6)[3:9].T,  # the transpose of a row slice: offset 18
    ]
    for operand in fallback:
        assert not (
            operand.dim() == 2
            and operand.dtype == torch.float32
            and operand.t().is_contiguous()
        )
    for operand in kernel:
        assert operand.t().is_contiguous() and operand.storage_offset() > 0
    for operand in fallback + kernel:
        got = stensor_module._contiguous_copy(operand)
        torch.testing.assert_close(got, operand.contiguous(), atol=0, rtol=0)


def test_memo_serves_one_copy_for_a_repeated_operand(copy_arms):
    """The point of it: the second call does not copy again."""
    copy_arms(memo=True, native=True)
    torch.manual_seed(0)
    base = torch.rand(32, 8)
    first = STensor.from_torch(base.T)
    second = STensor.from_torch(base.T)
    assert first.values.data_ptr() == second.values.data_ptr()
    torch.testing.assert_close(first.values, base.T.reshape(-1), atol=0, rtol=0)


def test_memo_is_not_consulted_when_it_is_off(copy_arms):
    """The control arm has to be a real control: no sharing at all."""
    copy_arms(memo=False, native=True)
    torch.manual_seed(0)
    base = torch.rand(32, 8)
    first = STensor.from_torch(base.T)
    second = STensor.from_torch(base.T)
    assert first.values.data_ptr() != second.values.data_ptr()
    assert stensor_module._OPERAND_COPY_CACHE == {}


def test_memo_misses_after_an_in_place_write_to_the_base(copy_arms):
    """A torch write through any view of the base bumps the counter they share."""
    copy_arms(memo=True, native=True)
    torch.manual_seed(0)
    base = torch.rand(16, 4)
    before = STensor.from_torch(base.T)
    base[0, 0] = 42.0
    after = STensor.from_torch(base.T)
    assert after.values.data_ptr() != before.values.data_ptr()
    torch.testing.assert_close(after.values, base.T.reshape(-1), atol=0, rtol=0)
    assert after.values[0].item() == 42.0


def test_memo_misses_after_a_write_through_the_values_it_handed_out(copy_arms):
    """The remembered copy *is* the STensor's values, so a write through those is a miss.

    Without this the next caller would be served a buffer the previous one had scribbled
    on. `detach` shares the version counter, which is what makes it visible here.
    """
    copy_arms(memo=True, native=True)
    torch.manual_seed(0)
    base = torch.rand(16, 4)
    handed_out = STensor.from_torch(base.T)
    handed_out.values[0] = -1.0
    again = STensor.from_torch(base.T)
    assert again.values[0].item() != -1.0
    torch.testing.assert_close(again.values, base.T.reshape(-1), atol=0, rtol=0)


def test_memo_does_not_keep_a_dead_operand_alive(copy_arms):
    """It holds a weak reference, so a base that goes out of scope is collectable."""
    import gc
    import weakref

    copy_arms(memo=True, native=True)
    base = torch.rand(16, 4)
    watch = weakref.ref(base)
    STensor.from_torch(base.T)
    assert len(stensor_module._OPERAND_COPY_CACHE) == 1
    del base
    gc.collect()
    assert watch() is None
    # The entry survives until an insert sweeps it, and it can never be a hit again.
    entry = next(iter(stensor_module._OPERAND_COPY_CACHE.values()))
    assert entry[0]() is None
    for _ in range(stensor_module._OPERAND_COPY_CACHE_MAX + 2):
        STensor.from_torch(torch.rand(16, 4).T)
    assert len(stensor_module._OPERAND_COPY_CACHE) <= (
        stensor_module._OPERAND_COPY_CACHE_MAX
    )


def test_memo_separates_views_that_share_a_base(copy_arms):
    """Same base, different geometry: different entries, and each one correct."""
    copy_arms(memo=True, native=True)
    torch.manual_seed(0)
    base = torch.rand(8, 10)
    rows = STensor.from_torch(base[0:4].T)
    other = STensor.from_torch(base[4:8].T)
    assert rows.values.data_ptr() != other.values.data_ptr()
    torch.testing.assert_close(rows.values, base[0:4].T.reshape(-1), atol=0, rtol=0)
    torch.testing.assert_close(other.values, base[4:8].T.reshape(-1), atol=0, rtol=0)


def test_contiguous_operands_are_untouched_by_any_of_this(copy_arms):
    """A contiguous operand still shares the caller's buffer and is never remembered."""
    copy_arms(memo=True, native=True)
    torch.manual_seed(0)
    dense = torch.rand(8, 8)
    wrapped = STensor.from_torch(dense)
    assert wrapped.values.data_ptr() == dense.data_ptr()
    assert stensor_module._OPERAND_COPY_CACHE == {}


@pytest.mark.parametrize("memo", [True, False])
@pytest.mark.parametrize("native", [True, False])
def test_matmul_against_a_transposed_operand_agrees_across_arms(
    copy_arms, memo, native
):
    """Whatever the levers say, the product is the product."""
    copy_arms(memo=memo, native=native)
    torch.manual_seed(0)
    dense = (torch.rand(40, 40) < 0.25).float()
    A = STensor.from_torch(dense.to_sparse_csr())
    base = torch.rand(12, 40)
    reference = dense.double() @ base.T.double()
    for _ in range(3):
        out = scorch.matmul(A, base.T)
        got = out if isinstance(out, torch.Tensor) else out.to_torch(in_place=False)
        torch.testing.assert_close(got, reference.float(), atol=1e-3, rtol=1e-3)


def _reset_memo(monkeypatch, **overrides):
    monkeypatch.setattr(stensor_module, "_MEMO_OPERAND_COPY", [True])
    monkeypatch.setattr(stensor_module, "_OPERAND_COPY_CACHE", {})
    monkeypatch.setattr(
        stensor_module,
        "_OPERAND_COPY_STATE",
        [0, stensor_module._OPERAND_COPY_CACHE_MAX],
    )
    for name, value in overrides.items():
        monkeypatch.setattr(stensor_module, name, value)


def test_memo_withdraws_after_repeated_stale_misses(monkeypatch):
    """An operand refilled in place every call must stop being looked up.

    Otherwise it pays for a memo that can never serve it -- measured at 1.11x of the
    smallest cell before this existed. The retained copies go with the withdrawal, because
    holding blocks the allocator cannot recycle was itself worth a further few percent.
    """
    _reset_memo(monkeypatch, _OPERAND_COPY_GIVE_UP=4)
    base = torch.rand(16, 4)
    for _ in range(2 + 4 * 2):
        base[0, 0] += 0.0  # a torch in-place write, so the version counter moves
        wrapped = STensor.from_torch(base.T)
        torch.testing.assert_close(wrapped.values, base.T.reshape(-1), atol=0, rtol=0)
    assert stensor_module._OPERAND_COPY_STATE[0] >= 4
    assert stensor_module._OPERAND_COPY_CACHE == {}


def test_a_hit_resets_the_withdrawal_counter(monkeypatch):
    """One stable operand among changing ones has to keep the memo alive."""
    _reset_memo(monkeypatch, _OPERAND_COPY_GIVE_UP=4)
    stable = torch.rand(16, 4)
    churning = torch.rand(16, 4)
    STensor.from_torch(stable.T)  # cold miss, installs
    for _ in range(20):
        churning[0, 0] += 0.0
        STensor.from_torch(churning.T)
        hit = STensor.from_torch(stable.T)
        torch.testing.assert_close(hit.values, stable.T.reshape(-1), atol=0, rtol=0)
    assert stensor_module._OPERAND_COPY_STATE[0] < 4
    assert stensor_module._OPERAND_COPY_CACHE != {}


def test_cold_misses_do_not_count_toward_withdrawal(monkeypatch):
    """A deep model's first forward is all cold misses, and it must keep the memo.

    Counting every miss alike would withdraw here -- from exactly the workload the memo
    exists for, one that will reuse each of these operands on the next call.
    """
    _reset_memo(monkeypatch, _OPERAND_COPY_GIVE_UP=4)
    operands = [torch.rand(8, 4) for _ in range(20)]
    for operand in operands:
        STensor.from_torch(operand.T)
    assert stensor_module._OPERAND_COPY_STATE[0] == 0
    # And the second pass over the ones that fit is served from the memo.
    served = 0
    for operand in operands:
        before = STensor.from_torch(operand.T).values.data_ptr()
        after = STensor.from_torch(operand.T).values.data_ptr()
        served += before == after
    assert served >= 1
