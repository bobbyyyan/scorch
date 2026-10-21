"""A COO tensor that stores one coordinate twice (Bobby's decision of 2026-10-12).

Scorch's kernels read every stored coordinate once and never sum two of them. So:

- raw COO arrays that repeat a coordinate are refused where Scorch storage is built from
  them -- ``from_coo(indices=...)``, ``from_components``, a hand-built ``TensorIndex`` --
  with :class:`RepeatedCoordinateError` (code ``coo_repeated_coordinate``), which names
  the repeated tuple and its position and says "call coalesce() to sum repeated
  coordinates";
- a torch COO tensor's repeats are summed on import, as torch defines them --
  ``from_torch``, ``from_coo(coo_matrix=...)``, a torch COO operand of ``scorch.matmul``
  -- and every result equals torch's;
- a repeat written into a COO operand's index arrays in place, after construction, is
  refused by the kernel's own index check (csrc/native_abi.h), and leaves Scorch as the
  same structured error through every Python call site of a kernel: ``scorch.einsum``,
  ``ops.spmv``, ``lower_and_exec_cin``, ``execute_cin_via_loopir``, a conversion, the
  prebuilt COO SpMM and SDDMM -- including after an earlier call recorded the arrays as
  valid while they still ascended;
- ``STensor.coalesce()`` repairs such a tensor exactly as torch's ``coalesce`` would.

Every result is checked in float64 to 1e-12 (float32 to 1e-5 for the float32-only prebuilt
kernels), at small and larger extents.
"""

import math
import pickle
import re
from pathlib import Path

import pytest
import torch

import scorch
import scorch_ops
from scorch import ops
from scorch import storage as storage_module
from scorch.compiler.cin import (
    BinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.exceptions import (
    COALESCE_ADVICE,
    RepeatedCoordinateError,
    TensorIndexError,
    raise_if_native_coded,
)
from scorch.prebuilt_kernels import execute_prebuilt_binary_kernel
from scorch.stensor import STensor
from scorch.storage import TensorIndex
from scorch.utils import native_resource_text
from tests.test_scorch.level_array_support import logical_dense

F64 = torch.float64
SHAPES = {1: (9,), 2: (6, 7), 3: (4, 5, 3)}
LARGER = {1: (5000,), 2: (300, 40), 3: (30, 20, 12)}
EXTENTS = [pytest.param(SHAPES, id="small"), pytest.param(LARGER, id="larger")]


def close(got: torch.Tensor, want: torch.Tensor, dtype=F64) -> None:
    tolerance = 1e-12 if dtype == F64 else 1e-5
    assert tuple(got.shape) == tuple(want.shape)
    assert torch.allclose(got.to(F64), want.to(F64), rtol=tolerance, atol=tolerance)


def distinct_entries(shape, count, seed, dtype=F64):
    """``count`` distinct coordinates of ``shape`` in lexicographic order, ``[rank, count]``
    int64, and a nonzero value for each."""

    generator = torch.Generator().manual_seed(seed)
    cells, _ = torch.sort(torch.randperm(math.prod(shape), generator=generator)[:count])
    indices = torch.stack(torch.unravel_index(cells, shape))
    values = (torch.rand(count, generator=generator, dtype=F64) + 0.5).to(dtype)
    return indices, values


def with_repeat(indices, at=2):
    """The same coordinates with the one at ``at`` replaced by its predecessor's: still
    sorted, and the predecessor's tuple is stored at ``at - 1`` and ``at``."""

    repeated = indices.clone()
    repeated[:, at] = indices[:, at - 1]
    return repeated, tuple(int(c) for c in indices[:, at - 1]), at


def coo_operand(shape, seed, dtype=F64, count=None, name="A"):
    count = count or max(3, math.prod(shape) // 4)
    indices, values = distinct_entries(shape, count, seed, dtype)
    tensor = STensor.from_components(
        shape, "o" * len(shape), [[level] for level in indices], values, name=name
    )
    return tensor, indices, values


def write_repeat_in_place(tensor, at=2):
    """Make entry ``at`` of ``tensor``'s own index arrays repeat entry ``at - 1``, in
    place, as a caller holding the arrays could: the version counters move."""

    coordinate = []
    for (level,) in tensor.storage._mode_indices:
        level[at] = level[at - 1]
        coordinate.append(int(level[at]))
    return tuple(coordinate), at


def dense_of_entries(indices, values, shape):
    """What torch makes of possibly repeated entries: their sum at each coordinate."""

    return torch.sparse_coo_tensor(indices, values.to(F64), shape).to_dense()


def assert_refusal(error, coordinate, position):
    assert isinstance(error, RepeatedCoordinateError), repr(error)
    assert isinstance(error, TensorIndexError)
    assert error.code == "coo_repeated_coordinate"
    assert error.coordinate == coordinate
    assert error.position == position
    assert COALESCE_ADVICE in str(error)
    assert "call coalesce() to sum repeated coordinates" in str(error)


# -- raw arrays: refused at construction ------------------------------------------------


def _from_coo(shape, indices, values):
    return STensor.from_coo(indices=indices, values=values, shape=shape)


def _from_components(shape, indices, values):
    return STensor.from_components(
        shape, "o" * len(shape), [[level] for level in indices], values
    )


def _hand_built_index(shape, indices, values):
    index = TensorIndex(
        tensor_format="o" * len(shape), mode_indices=[[level] for level in indices]
    )
    return STensor(shape=shape, index=index, value=values)


RAW_CONSTRUCTORS = [_from_coo, _from_components, _hand_built_index]


@pytest.mark.parametrize("extents", EXTENTS)
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("construct", RAW_CONSTRUCTORS, ids=lambda f: f.__name__)
def test_raw_arrays_that_repeat_a_coordinate_are_refused(construct, rank, extents):
    shape = extents[rank]
    indices, values = distinct_entries(shape, max(3, math.prod(shape) // 5), seed=rank)
    repeated, coordinate, position = with_repeat(indices)
    with pytest.raises(RepeatedCoordinateError) as caught:
        construct(shape, repeated, values)
    assert_refusal(caught.value, coordinate, position)
    # The arrays without the repeat build, and hold exactly the given entries.
    close(logical_dense(construct(shape, indices, values)),
          dense_of_entries(indices, values, shape))


def test_raw_arrays_out_of_order_name_the_first_two_positions_of_the_repeat():
    """from_coo sorts raw arrays, so it reports positions in the caller's order."""

    indices = torch.tensor([[2, 0, 1, 2], [1, 3, 0, 1]])
    with pytest.raises(RepeatedCoordinateError) as caught:
        STensor.from_coo(indices=indices, values=torch.ones(4, dtype=F64), shape=(3, 4))
    assert_refusal(caught.value, (2, 1), 3)
    assert "positions 0 and 3" in str(caught.value)


def test_raw_arrays_without_a_repeat_are_sorted_not_refused():
    indices = torch.tensor([[2, 0, 1], [1, 3, 0]])
    values = torch.tensor([1.5, 2.5, 3.5], dtype=F64)
    tensor = STensor.from_coo(indices=indices, values=values, shape=(3, 4))
    close(logical_dense(tensor), dense_of_entries(indices, values, (3, 4)))


# -- torch COO tensors: their repeats are summed, as torch defines them ---------------


def torch_coo_with_repeats(shape, seed, dtype=F64):
    """An uncoalesced torch COO tensor that stores several coordinates two or three times,
    in no particular order."""

    wanted = max(3, math.prod(shape) // 5)
    indices, values = distinct_entries(shape, wanted, seed, dtype)
    generator = torch.Generator().manual_seed(seed + 100)
    count = indices.shape[1]
    again = torch.randint(0, count, (max(2, count // 3),), generator=generator)
    indices = torch.cat([indices, indices[:, again], indices[:, again[:1]]], dim=1)
    values = torch.cat([values, values[again] * 0.25, values[again[:1]] * 2.0])
    order = torch.randperm(indices.shape[1], generator=generator)
    tensor = torch.sparse_coo_tensor(indices[:, order], values[order], shape)
    assert not tensor.is_coalesced() and tensor.coalesce()._nnz() < tensor._nnz()
    return tensor


@pytest.mark.parametrize("extents", EXTENTS)
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("entry", ["from_torch", "from_coo"])
def test_a_torch_coo_tensor_with_repeats_is_summed_as_torch_does(entry, rank, extents):
    tensor = torch_coo_with_repeats(extents[rank], seed=10 + rank)
    converted = (
        STensor.from_torch(tensor) if entry == "from_torch"
        else STensor.from_coo(coo_matrix=tensor)
    )
    assert converted.values.numel() == tensor.coalesce()._nnz()
    close(logical_dense(converted), tensor.to_dense())


@pytest.mark.parametrize("extents", EXTENTS)
def test_scorch_matmul_sums_a_torch_coo_operand_as_torch_matmul_does(extents):
    rows, cols = extents[2]
    a = torch_coo_with_repeats((rows, cols), seed=21)
    generator = torch.Generator().manual_seed(22)
    b = torch.rand(cols, 5, generator=generator, dtype=F64)
    x = torch.rand(cols, generator=generator, dtype=F64)
    # logical_dense: the matrix-vector product comes back as a dense STensor, the
    # matrix-matrix product as a torch.Tensor.
    close(logical_dense(scorch.matmul(a, b)), torch.matmul(a, b))
    close(logical_dense(scorch.matmul(a, x)), torch.matmul(a, x))
    # Two torch COO operands (both converted to CSR by torch, which sums).
    c = torch_coo_with_repeats((cols, 6), seed=23)
    close(logical_dense(scorch.matmul(a, c)),
          torch.matmul(a.to_dense(), c.to_dense()))


def test_scorch_matmul_sums_a_torch_coo_operand_under_autograd():
    """The torch COO operand path of the autograd planner (autograd.py)."""

    a = torch_coo_with_repeats((40, 30), seed=31)
    generator = torch.Generator().manual_seed(32)
    b = torch.rand(30, 4, generator=generator, dtype=F64, requires_grad=True)
    b_ref = b.detach().clone().requires_grad_()
    got, want = scorch.matmul(a, b), torch.matmul(a, b_ref)
    close(got.detach(), want.detach())
    got.sum().backward()
    want.sum().backward()
    close(b.grad, b_ref.grad)


# -- a repeat written in place: refused by the kernel's own check ----------------------


def _sum_cin(fmts, result_fmt):
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt=fmts[0], dtype=F64)[i, j]
    b = TensorVar("B", fmt=fmts[1], dtype=F64)[i, j]
    c = TensorVar("C", fmt=result_fmt, dtype=F64)[i, j]
    return ForAll(i, ForAll(j, TensorAssign(c, BinaryOp(Operation.ADD, a, b))))


# Each route takes the operand and its dense value AS BUILT, and returns (result, reference).
# The reference is computed from the dense value, never by decoding ``a``: the second call
# of a test runs after ``a``'s index arrays were written in place, and decoding a tensor
# that repeats a coordinate fails in the decoder (level_array_support.coordinate_entries
# asserts strict order) before the route under test is reached.


def _run_einsum(a, dense_a):
    x = torch.rand(a.shape[1], generator=torch.Generator().manual_seed(3), dtype=F64)
    got = scorch.einsum("ij,j->i", a, STensor.from_torch(x), format="d")
    return got, dense_a @ x


def _run_spmv(a, dense_a):
    x = torch.rand(a.shape[1], generator=torch.Generator().manual_seed(4), dtype=F64)
    return ops.spmv(a, STensor.from_torch(x), "d"), dense_a @ x


def _run_lower_and_exec_cin(a, dense_a):
    b, _, _ = coo_operand(tuple(a.shape), seed=5, name="B")
    want = dense_a + logical_dense(b)
    got = ops.lower_and_exec_cin(_sum_cin(("oo", "oo"), "dd"), tuple(a.shape), a, b)
    return got, want


def _run_loopir(a, dense_a):
    b, _, _ = coo_operand(tuple(a.shape), seed=6, name="B")
    want = dense_a + logical_dense(b)
    result, _ = execute_cin_via_loopir(
        _sum_cin(("oo", "oo"), "dd"), tuple(a.shape), a, b
    )
    return result, want


def _run_conversion(a, dense_a):
    # Out of place, so the conversion kernel reads ``a``'s own arrays on every call.
    # (``to_sparse`` converts in place: after one call ``a`` is no longer a COO tensor.)
    return a.to_dense(in_place=False), dense_a


def _run_transpose(a, dense_a):
    # The other mode order each time: asking for the order a tensor already has returns
    # it untouched, without reading its index arrays.
    stored_order = [int(mode) for mode in a.mode_order]
    return a.change_mode_order(stored_order[::-1]), dense_a


ROUTES = [_run_einsum, _run_spmv, _run_lower_and_exec_cin, _run_loopir, _run_conversion,
          _run_transpose]


@pytest.mark.parametrize("extents", EXTENTS)
@pytest.mark.parametrize("route", ROUTES, ids=lambda f: f.__name__[5:])
def test_a_repeat_written_after_a_valid_call_is_refused(route, extents):
    """The first call validates the ascending arrays (and records the verdict where the
    route keeps one); the in-place write then makes a repeat, and the next call refuses
    it as a RepeatedCoordinateError, never an unstructured RuntimeError."""

    a, indices, values = coo_operand(extents[2], seed=7)
    dense_a = dense_of_entries(indices, values, extents[2])
    got, want = route(a, dense_a)
    close(logical_dense(got), want)
    coordinate, position = write_repeat_in_place(a)
    with pytest.raises(RepeatedCoordinateError) as caught:
        route(a, dense_a)
    assert_refusal(caught.value, coordinate, position)


@pytest.mark.parametrize("at", [256, 257, 512, 513, 2816, 2817, 2999])
@pytest.mark.parametrize(
    "route",
    [_run_lower_and_exec_cin, _run_loopir, _run_conversion],
    ids=lambda f: f.__name__[5:],
)
def test_a_generated_kernel_refuses_a_repeat_wherever_it_sits_in_a_block(route, at):
    """A generated kernel does not call the extension's index check: ``native_abi.h`` is
    its preamble, so it carries its own copy, built by the kernel's compiler.  That copy
    screens the coordinates 256 positions at a time (``kLexBlock``), the first block
    starting at position 1.  On 3,000 entries, a repeat at the last position of a block
    (256, 512, 2816), at the first of the next (257, 513, 2817) and at the last position
    of the partial block that ends the array is refused with its own position, by a
    kernel of each compiler and by a conversion kernel."""

    a, indices, values = coo_operand(LARGER[2], seed=7)
    assert a.values.numel() == 3000
    dense_a = dense_of_entries(indices, values, LARGER[2])
    got, want = route(a, dense_a)
    close(logical_dense(got), want)
    coordinate, position = write_repeat_in_place(a, at=at)
    assert position == at
    with pytest.raises(RepeatedCoordinateError) as caught:
        route(a, dense_a)
    assert_refusal(caught.value, coordinate, position)


@pytest.mark.parametrize("at", [256, 257, 512, 513, 2816, 2817, 2999])
@pytest.mark.parametrize(
    "route", [_run_lower_and_exec_cin, _run_loopir], ids=lambda f: f.__name__[5:]
)
def test_a_generated_kernel_refuses_a_descent_wherever_it_sits_in_a_block(route, at):
    """The same positions with the other defect: entries ``at - 1`` and ``at`` exchanged
    in place, so the order decreases at ``at`` and nowhere else, and nothing repeats.  It
    is refused as a descent, as it was before repeats were, and never as a repeat."""

    a, indices, values = coo_operand(LARGER[2], seed=7)
    dense_a = dense_of_entries(indices, values, LARGER[2])
    got, want = route(a, dense_a)
    close(logical_dense(got), want)
    for (level,) in a.storage._mode_indices:
        earlier, later = int(level[at - 1]), int(level[at])
        level[at - 1], level[at] = later, earlier
    with pytest.raises((RuntimeError, TensorIndexError)) as caught:
        route(a, dense_a)
    assert not isinstance(caught.value, RepeatedCoordinateError)
    text = str(caught.value)
    assert "lexicographically ordered" in text
    if "order decreases at position" in text:
        assert re.search(rf"order decreases at position {at}\b", text), text


@pytest.mark.parametrize("extents", EXTENTS)
def test_the_prebuilt_coo_spmm_refuses_a_repeat_written_after_a_valid_call(extents):
    """Straight at the native kernel, so nothing in Python looks at the arrays first: the
    refusal is the native check's, mapped by execute_prebuilt_binary_kernel."""

    rows, cols = extents[2]
    a, _, _ = coo_operand((rows, cols), seed=8, dtype=torch.float32)
    dense = torch.rand(cols, 5, generator=torch.Generator().manual_seed(9))
    b = STensor.from_torch(dense)
    result_cpp, _ = execute_prebuilt_binary_kernel(scorch_ops.spmm_coo_float, a, b)
    # the [rows, 5] product itself: the kernel allocates it in that shape (spmm.h)
    product = result_cpp.storage.value
    assert tuple(product.shape) == (rows, 5) and product._base is None
    close(product, logical_dense(a) @ dense.to(F64), torch.float32)
    coordinate, position = write_repeat_in_place(a)
    with pytest.raises(RepeatedCoordinateError) as caught:
        execute_prebuilt_binary_kernel(scorch_ops.spmm_coo_float, a, b)
    assert_refusal(caught.value, coordinate, position)


def test_the_prebuilt_sddmm_refuses_a_repeat_written_after_a_valid_call():
    s, _, _ = coo_operand((30, 20), seed=11, dtype=torch.float32, name="S")
    generator = torch.Generator().manual_seed(12)
    a = STensor.from_torch(torch.rand(30, 8, generator=generator))
    b = STensor.from_torch(torch.rand(20, 8, generator=generator))
    want = logical_dense(s) * (logical_dense(a) @ logical_dense(b).t())
    close(logical_dense(scorch.einsum("ij,ik,jk->ij", s, a, b, format="oo")), want,
          torch.float32)
    coordinate, position = write_repeat_in_place(s)
    with pytest.raises(RepeatedCoordinateError) as caught:
        scorch.einsum("ij,ik,jk->ij", s, a, b, format="oo")
    assert_refusal(caught.value, coordinate, position)


# -- the memo: keyed by the strict rule --------------------------------------------------


def test_a_bounds_verdict_does_not_satisfy_the_order_check():
    """One COO level, a repeat inside it. Its bounds verdict is recorded first; the order
    check over the same array must still run and flag the repeat."""

    if storage_module._NATIVE_LEX is None or storage_module._NATIVE_BOUNDS is None:
        pytest.skip("extension has no screens")
    coordinate = torch.tensor([1, 3, 3, 5], dtype=torch.int32)
    scorch_ops.abi_memo_clear()
    assert storage_module._NATIVE_BOUNDS(coordinate, 8, 1) is False
    held = scorch_ops.abi_memo_size()
    assert storage_module._NATIVE_LEX([coordinate], 4, 1) is True
    assert scorch_ops.abi_memo_size() == held, "a flagged order verdict was recorded"


def test_the_order_check_records_only_under_the_strict_rule():
    """The retired kind 3 (the order check that admitted a repeat) is recorded nowhere, so
    no verdict made under it can satisfy the strict check (kind 5)."""

    header = native_resource_text("native_abi.h")
    assert re.search(r"\bAbiCheckLexStrict = 5\b", header)
    assert not re.search(r"\bAbiCheckLex\b", header)
    assert not re.search(r"^\s*AbiCheck\w+ = 3,", header, flags=re.MULTILINE)
    # scorch_ops's own screen (abi_screen_lex_levels) records into the same memo.
    ops_cpp = Path(scorch.__file__).parent / "csrc" / "ops.cpp"
    if not ops_cpp.exists():
        pytest.skip("ops.cpp is not shipped with this installation")
    source = ops_cpp.read_text()
    assert "AbiCheckLexStrict" in source
    assert not re.search(r"\bAbiCheckLex\b", source)


# -- the native refusal's text and its structured form -----------------------------------


def test_the_native_refusal_text_becomes_the_structured_error():
    native = RuntimeError(
        "evaluate: A COO coordinate (4, 0, 2) is stored at positions 6 and 7; call "
        "coalesce() to sum repeated coordinates [coo_repeated_coordinate]\n"
        "Exception raised from check_coo_strictly_ascending at native_abi.h:1790"
    )
    with pytest.raises(RepeatedCoordinateError) as caught:
        raise_if_native_coded(native)
    assert_refusal(caught.value, (4, 0, 2), 7)
    assert caught.value.__cause__ is native
    assert str(caught.value).startswith("evaluate: A COO coordinate (4, 0, 2)")
    single = RuntimeError(
        "op: x COO coordinate (3) is stored at positions 0 and 1; call coalesce() to sum "
        "repeated coordinates [coo_repeated_coordinate]"
    )
    with pytest.raises(RepeatedCoordinateError) as caught:
        raise_if_native_coded(single)
    assert_refusal(caught.value, (3,), 1)


@pytest.mark.parametrize("text", [
    "evaluate: A COO coordinates must be lexicographically ordered; order decreases at "
    "position 3",
    "evaluate: A coordinates must be strictly increasing at compressed level 1 position 2",
    "something else entirely",
])
def test_every_other_native_error_passes_through_unchanged(text):
    assert raise_if_native_coded(RuntimeError(text)) is None


def test_the_structured_error_survives_pickling():
    error = RepeatedCoordinateError("message", 3, (1, 2))
    copy = pickle.loads(pickle.dumps(error))
    assert (type(copy), str(copy), copy.position, copy.coordinate, copy.code) == (
        RepeatedCoordinateError, "message", 3, (1, 2), "coo_repeated_coordinate"
    )


# -- coalesce() ------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_coalesce_returns_the_tensor_itself_when_nothing_repeats(rank):
    tensor, _, _ = coo_operand(SHAPES[rank], seed=40 + rank)
    assert tensor.coalesce() is tensor
    empty = STensor.from_components(
        SHAPES[rank], "o" * rank, [[torch.zeros(0, dtype=torch.int64)]] * rank,
        torch.zeros(0, dtype=F64),
    )
    assert empty.coalesce() is empty


@pytest.mark.parametrize("extents", EXTENTS)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_coalesce_sums_a_repeat_written_in_place_as_torch_does(rank, extents):
    shape = extents[rank]
    tensor, _, _ = coo_operand(shape, seed=50 + rank)
    for at in range(2, tensor.values.numel(), 7):  # many repeats, some out of order
        write_repeat_in_place(tensor, at)
    if tensor.values.numel() > 4:  # and one descent: coalesce() sorts too
        for (level,) in tensor.storage._mode_indices:
            level[[0, -1]] = level[[-1, 0]]
    stored = torch.stack([level for (level,) in tensor.storage._mode_indices]).long()
    expected = torch.sparse_coo_tensor(stored, tensor.values, shape).coalesce()
    coalesced = tensor.coalesce()
    assert coalesced is not tensor
    got = torch.stack([level for (level,) in coalesced.storage._mode_indices]).long()
    assert torch.equal(got, expected.indices())
    close(coalesced.values, expected.values())
    # And it computes: a product equals torch's over the summed entries.
    if rank == 2:
        x = torch.rand(shape[1], generator=torch.Generator().manual_seed(51), dtype=F64)
        close(logical_dense(scorch.einsum("ij,j->i", coalesced, STensor.from_torch(x),
                                          format="d")),
              expected.to_dense() @ x)


def test_coalesce_keeps_an_explicitly_stored_zero():
    """Like torch's: a coordinate whose entries sum to zero, and an entry stored as zero,
    both stay stored."""

    tensor = STensor.from_components(
        (3, 4), "oo",
        [[torch.tensor([0, 1, 1, 2])], [torch.tensor([1, 0, 2, 3])]],
        torch.tensor([2.0, -1.5, 1.5, 0.0], dtype=F64),
    )
    (rows,), (cols,) = tensor.storage._mode_indices
    cols[2] = 0  # (1, 0) is now stored twice, as -1.5 and 1.5
    coalesced = tensor.coalesce()
    expected = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), tensor.values, (3, 4)
    ).coalesce()
    assert coalesced.values.tolist() == expected.values().tolist() == [2.0, 0.0, 0.0]
    assert [level.tolist() for (level,) in coalesced.storage._mode_indices] == [
        [0, 1, 2], [1, 0, 3]
    ]


def test_coalesce_repairs_a_compressed_level_written_in_place():
    """A CSR whose row stores one column twice after an in-place write: summed, as the
    construction of a caller's CSR arrays already does."""

    dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]], dtype=F64)
    tensor = STensor.from_torch(dense.to_sparse_csr())
    positions, columns = tensor.storage._mode_indices[1]
    columns[3] = 1  # row 1 now stores column 1 twice: 3.0 and 4.0
    coalesced = tensor.coalesce()
    close(logical_dense(coalesced), torch.tensor([[1.0, 0.0, 2.0], [0.0, 7.0, 0.0]],
                                                 dtype=F64))
