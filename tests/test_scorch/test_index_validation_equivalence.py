"""The vectorized index validation must be the old row loop, exactly.

``_validate_index_storage`` runs on every ``STensor`` built over a compressed level --
``from_torch``, ``to_sparse``, a relayout, and every generated kernel's result. It checked
that each parent's coordinates are sorted with a *Python loop over parents*, each
iteration slicing a tensor, launching a comparison kernel and syncing on ``.item()``:
3.7 us per row, so 74 ms to wrap a 20,000-row CSR, and it ran twice per construction.

Replacing that with two whole-array kernels is only legitimate if the replacement is
indistinguishable -- same verdict, same exception type, same message, same *precedence*
when several things are wrong at once. So this file holds the old implementation
verbatim, as ``reference_validate``, and compares the two over a grid of well-formed and
malformed storage. When they disagree the test prints both outcomes.

The reference is a frozen copy on purpose: it is what shipped, and it must not be
refactored alongside the production code it exists to check.
"""

import copy
import math
import pickle

import pytest
import torch

from scorch import TensorLayout
from scorch.exceptions import (
    TensorDeviceError,
    TensorIndexError,
    TensorStorageError,
)
from scorch.format import LevelType
from scorch import storage as storage_module
from scorch.stensor import STensor
from scorch.storage import (
    IndexModes,
    SparseStorage,
    _check_coordinate_bounds,
    _validate_index_storage,
)


@pytest.fixture(autouse=True, params=["native_screen", "no_screen"])
def screen_mode(request, monkeypatch):
    """Run this whole file twice: with the native screen, and without it.

    `_validate_index_storage` consults a native screen (csrc/native_abi.h) that does
    the bounds and sortedness passes in one fused loop and answers only "no violation
    exists" or "go and look". If that is true, then every case below -- 59 of which
    compare production against the frozen old implementation -- must reach an identical
    verdict either way. Parametrizing the whole module is the cheap way to assert it:
    any message, precedence or acceptance difference the screen introduces fails here.
    """
    if request.param == "no_screen":
        monkeypatch.setattr(storage_module, "_NATIVE_SCREEN", None)
        monkeypatch.setattr(storage_module, "_NATIVE_BOUNDS", None)
        monkeypatch.setattr(storage_module, "_NATIVE_LEX", None)
    return request.param


def reference_validate(
    layout: TensorLayout, mode_indices: IndexModes, values: torch.Tensor
) -> None:
    nnz = values.numel()
    level_types = layout.format.get_level_types()

    for mode, arrays in enumerate(mode_indices):
        for slot, index in enumerate(arrays):
            if index.dtype != layout.index_dtype:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] has dtype {index.dtype}, "
                    f"expected layout index dtype {layout.index_dtype}"
                )
            if index.device != values.device:
                raise TensorDeviceError(
                    f"mode_indices[{mode}][{slot}] and values must share a device"
                )

    if all(level_type == LevelType.DENSE for level_type in level_types):
        expected = math.prod(layout.physical_shape)
        if nnz != expected:
            raise TensorStorageError(
                f"dense values contain {nnz} elements, expected {expected} for shape "
                f"{layout.physical_shape}"
            )
        return

    if level_types and all(
        level_type == LevelType.COORDINATE for level_type in level_types
    ):
        coordinate_values = []
        for mode, arrays in enumerate(mode_indices):
            coordinate = arrays[0]
            if coordinate.numel() != nnz:
                raise TensorStorageError(
                    f"COO mode {mode} has {coordinate.numel()} coordinates, "
                    f"but values has {nnz} elements"
                )
            _check_coordinate_bounds(
                coordinate, layout.physical_shape[mode], f"COO mode {mode}"
            )
            coordinate_values.append(coordinate.tolist())
        for position in range(1, nnz):
            previous = tuple(
                coordinate_values[mode][position - 1]
                for mode in range(len(coordinate_values))
            )
            current = tuple(
                coordinate_values[mode][position]
                for mode in range(len(coordinate_values))
            )
            if current < previous:
                raise TensorIndexError(
                    "COO coordinates must be lexicographically ordered"
                )
        return

    parent_positions = 1
    for mode, (level_type, arrays, extent) in enumerate(
        zip(level_types, mode_indices, layout.physical_shape)
    ):
        if level_type == LevelType.DENSE:
            parent_positions *= extent
            continue
        if level_type == LevelType.COMPRESSED:
            positions, coordinates = arrays
            expected_positions = parent_positions + 1
            if positions.numel() != expected_positions:
                raise TensorIndexError(
                    f"compressed mode {mode} position array has {positions.numel()} "
                    f"elements, expected {expected_positions}"
                )
            if positions.numel() == 0 or int(positions[0].item()) != 0:
                raise TensorIndexError(
                    f"compressed mode {mode} position array must start at zero"
                )
            if positions.numel() > 1 and bool(
                torch.any(positions[1:] < positions[:-1]).item()
            ):
                raise TensorIndexError(
                    f"compressed mode {mode} position array must be nondecreasing"
                )
            terminal = int(positions[-1].item())
            if terminal != coordinates.numel():
                raise TensorIndexError(
                    f"compressed mode {mode} terminal position {terminal} does not "
                    f"match coordinate count {coordinates.numel()}"
                )
            _check_coordinate_bounds(coordinates, extent, f"compressed mode {mode}")
            position_values = positions.tolist()
            for parent, (start, end) in enumerate(
                zip(position_values[:-1], position_values[1:])
            ):
                segment = coordinates[start:end]
                if segment.numel() > 1 and bool(
                    torch.any(segment[1:] < segment[:-1]).item()
                ):
                    raise TensorIndexError(
                        f"compressed mode {mode} coordinates must be sorted "
                        f"within parent {parent}"
                    )
            parent_positions = coordinates.numel()
            continue
        coordinate = arrays[0]
        _check_coordinate_bounds(coordinate, extent, f"mode {mode}")
        if level_type == LevelType.SINGLETON and coordinate.numel() != parent_positions:
            raise TensorIndexError(
                f"singleton mode {mode} must contain one coordinate per parent"
            )
        if level_type == LevelType.COORDINATE and coordinate.numel() != nnz:
            raise TensorStorageError(
                f"coordinate mode {mode} has {coordinate.numel()} entries, "
                f"but values has {nnz} elements"
            )
        parent_positions = coordinate.numel()

    if parent_positions != nnz:
        raise TensorStorageError(
            f"layout describes {parent_positions} stored positions, but values has {nnz} elements"
        )


def outcome(fn, layout, mode_indices, values):
    """(exception type name, message) or ("ok", "") -- what the caller would see."""
    try:
        fn(layout, mode_indices, values)
    except Exception as exc:  # noqa: BLE001 - the type is part of what we compare
        return type(exc).__name__, str(exc)
    return "ok", ""


def check_same(layout, mode_indices, values, label):
    expected = outcome(reference_validate, layout, mode_indices, values)
    actual = outcome(_validate_index_storage, layout, mode_indices, values)
    assert actual == expected, (
        f"{label}: vectorized validation disagrees with the row loop\n"
        f"  row loop  : {expected}\n"
        f"  vectorized: {actual}"
    )
    return expected


def layout_for(fmt, shape, index_dtype=torch.int32):
    return TensorLayout.from_physical_shape(shape, fmt, index_dtype=index_dtype)


def csr_parts(rows, cols, degree, dtype=torch.int32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    per_row = []
    for _ in range(rows):
        chosen = torch.randperm(cols, generator=generator)[:degree]
        per_row.append(torch.sort(chosen).values)
    coordinates = (torch.cat(per_row) if per_row
                   else torch.zeros(0, dtype=torch.int64)).to(dtype)
    positions = (torch.arange(rows + 1, dtype=torch.int64) * degree).to(dtype)
    values = torch.rand(rows * degree, generator=generator)
    return positions, coordinates, values


# --------------------------------------------------------------------------- #
# Well-formed storage, over the formats and dtypes that reach this code
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rows,cols,degree", [(1, 1, 1), (3, 4, 2), (64, 32, 3),
                                              (128, 128, 4), (7, 5, 0)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_well_formed_csr_agrees(rows, cols, degree, index_dtype):
    positions, coordinates, values = csr_parts(rows, cols, degree, index_dtype)
    layout = layout_for("ds", (rows, cols), index_dtype)
    verdict = check_same(layout, ((), (positions, coordinates)), values,
                         f"csr {rows}x{cols} deg={degree}")
    assert verdict == ("ok", ""), f"a well-formed CSR was rejected: {verdict}"


@pytest.mark.parametrize("fmt,shape", [("dd", (3, 4)), ("d", (5,)), ("ddd", (2, 3, 2))])
def test_well_formed_dense_agrees(fmt, shape):
    layout = layout_for(fmt, shape)
    values = torch.rand(math.prod(shape))
    verdict = check_same(layout, tuple(() for _ in shape), values, f"dense {fmt}")
    assert verdict == ("ok", "")


def test_well_formed_coo_agrees():
    layout = layout_for("oo", (4, 4))
    rows = torch.tensor([0, 0, 1, 3], dtype=torch.int32)
    cols = torch.tensor([1, 3, 0, 2], dtype=torch.int32)
    values = torch.rand(4)
    verdict = check_same(layout, ((rows,), (cols,)), values, "coo")
    assert verdict == ("ok", "")


def test_well_formed_dcsr_agrees():
    """Two compressed levels: the parent count of the second comes from the first."""
    layout = layout_for("ss", (4, 4))
    outer_pos = torch.tensor([0, 2], dtype=torch.int32)
    outer_crd = torch.tensor([1, 3], dtype=torch.int32)
    inner_pos = torch.tensor([0, 2, 3], dtype=torch.int32)
    inner_crd = torch.tensor([0, 2, 1], dtype=torch.int32)
    values = torch.rand(3)
    verdict = check_same(layout, ((outer_pos, outer_crd), (inner_pos, inner_crd)),
                         values, "dcsr")
    assert verdict == ("ok", "")


# --------------------------------------------------------------------------- #
# Malformed storage -- every rejection path, and the precedence between them
# --------------------------------------------------------------------------- #


def test_unsorted_within_a_parent_is_rejected_identically():
    """The message names the offending parent, so the vectorized version has to
    recover that index and not merely report that *some* parent is unsorted."""
    rows, cols, degree = 16, 8, 4
    positions, coordinates, values = csr_parts(rows, cols, degree)
    layout = layout_for("ds", (rows, cols))
    for parent in range(rows):
        broken = coordinates.clone()
        start = parent * degree
        broken[start], broken[start + 1] = coordinates[start + 1], coordinates[start]
        verdict = check_same(layout, ((), (positions, broken)), values,
                             f"descent in parent {parent}")
        assert verdict[0] == "TensorIndexError"
        assert f"within parent {parent}" in verdict[1], verdict


def test_first_offending_parent_wins_when_several_are_unsorted():
    """The row loop reports the first parent it reaches; so must the replacement."""
    rows, cols, degree = 12, 8, 3
    positions, coordinates, values = csr_parts(rows, cols, degree)
    layout = layout_for("ds", (rows, cols))
    broken = coordinates.clone()
    for parent in (7, 3, 9):
        start = parent * degree
        broken[start], broken[start + 1] = coordinates[start + 1], coordinates[start]
    verdict = check_same(layout, ((), (positions, broken)), values, "three descents")
    assert "within parent 3" in verdict[1], verdict


def test_descent_across_a_parent_boundary_is_allowed():
    """Coordinates restart at each row, so a descent at a boundary is normal and
    must not be reported -- the failure mode a naive whole-array check would have."""
    layout = layout_for("ds", (3, 8))
    positions = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    coordinates = torch.tensor([5, 7, 0, 3, 1, 6], dtype=torch.int32)
    values = torch.rand(6)
    verdict = check_same(layout, ((), (positions, coordinates)), values, "boundaries")
    assert verdict == ("ok", ""), f"boundary descents rejected: {verdict}"


def test_empty_parents_between_populated_ones():
    """Empty rows make start == end, which the loop skipped and a mask must too."""
    layout = layout_for("ds", (5, 8))
    positions = torch.tensor([0, 2, 2, 2, 4, 4], dtype=torch.int32)
    coordinates = torch.tensor([1, 6, 0, 7], dtype=torch.int32)
    values = torch.rand(4)
    verdict = check_same(layout, ((), (positions, coordinates)), values, "empty rows")
    assert verdict == ("ok", "")


def test_single_entry_parents():
    layout = layout_for("ds", (4, 8))
    positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    coordinates = torch.tensor([3, 0, 7, 2], dtype=torch.int32)
    values = torch.rand(4)
    assert check_same(layout, ((), (positions, coordinates)), values, "singletons") == (
        "ok", "")


def test_duplicate_coordinates_in_a_parent_are_allowed():
    """Sortedness is nondecreasing, not strictly increasing."""
    layout = layout_for("ds", (2, 8))
    positions = torch.tensor([0, 3, 4], dtype=torch.int32)
    coordinates = torch.tensor([1, 1, 4, 2], dtype=torch.int32)
    values = torch.rand(4)
    assert check_same(layout, ((), (positions, coordinates)), values, "dups") == ("ok", "")


@pytest.mark.parametrize(
    "positions,label",
    [
        (torch.tensor([1, 2, 3], dtype=torch.int32), "does not start at zero"),
        (torch.tensor([0, 3, 2], dtype=torch.int32), "decreasing positions"),
        (torch.tensor([0, 2], dtype=torch.int32), "too few positions"),
        (torch.tensor([0, 1, 2, 3], dtype=torch.int32), "too many positions"),
        (torch.tensor([0, 1, 9], dtype=torch.int32), "terminal past the coordinates"),
    ],
)
def test_malformed_position_arrays_agree(positions, label):
    layout = layout_for("ds", (2, 8))
    coordinates = torch.tensor([1, 4, 6], dtype=torch.int32)
    values = torch.rand(3)
    verdict = check_same(layout, ((), (positions, coordinates)), values, label)
    assert verdict[0] != "ok", f"{label} was accepted"


@pytest.mark.parametrize("bad", [-1, 8, 99])
def test_out_of_range_coordinates_agree(bad):
    layout = layout_for("ds", (2, 8))
    positions = torch.tensor([0, 2, 3], dtype=torch.int32)
    coordinates = torch.tensor([1, 4, 6], dtype=torch.int32)
    coordinates[2] = bad
    values = torch.rand(3)
    verdict = check_same(layout, ((), (positions, coordinates)), values, f"coord {bad}")
    assert verdict[0] == "TensorIndexError"


def test_wrong_index_dtype_agrees():
    layout = layout_for("ds", (2, 8), torch.int32)
    positions = torch.tensor([0, 2, 3], dtype=torch.int64)
    coordinates = torch.tensor([1, 4, 6], dtype=torch.int32)
    check_same(layout, ((), (positions, coordinates)), torch.rand(3), "dtype")


def test_values_length_disagreeing_with_the_layout_agrees():
    layout = layout_for("ds", (2, 8))
    positions = torch.tensor([0, 2, 3], dtype=torch.int32)
    coordinates = torch.tensor([1, 4, 6], dtype=torch.int32)
    check_same(layout, ((), (positions, coordinates)), torch.rand(5), "nnz mismatch")


def test_dense_values_length_agrees():
    layout = layout_for("dd", (3, 4))
    check_same(layout, ((), ()), torch.rand(11), "dense short")


@pytest.mark.parametrize("rows,cols", [(0, 4), (4, 0)])
def test_degenerate_extents_agree(rows, cols):
    positions = torch.zeros(rows + 1, dtype=torch.int32)
    coordinates = torch.zeros(0, dtype=torch.int32)
    layout = layout_for("ds", (rows, cols))
    check_same(layout, ((), (positions, coordinates)), torch.zeros(0), "degenerate")


def test_unordered_coo_agrees():
    layout = layout_for("oo", (4, 4))
    rows = torch.tensor([0, 2, 1, 3], dtype=torch.int32)
    cols = torch.tensor([1, 3, 0, 2], dtype=torch.int32)
    check_same(layout, ((rows,), (cols,)), torch.rand(4), "coo out of order")


# --------------------------------------------------------------------------- #
# A fuzz sweep: many small random storages, valid and corrupted the same way
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(24))
def test_fuzzed_storage_agrees(seed):
    generator = torch.Generator().manual_seed(seed)
    rows = int(torch.randint(1, 9, (1,), generator=generator).item())
    cols = int(torch.randint(1, 9, (1,), generator=generator).item())
    degree = int(torch.randint(0, min(cols, 4) + 1, (1,), generator=generator).item())
    positions, coordinates, values = csr_parts(rows, cols, degree, seed=seed)
    layout = layout_for("ds", (rows, cols))
    check_same(layout, ((), (positions, coordinates)), values, f"fuzz {seed} clean")
    if coordinates.numel() >= 2:
        for corrupt in ("swap", "negate", "overflow", "shuffle"):
            broken = coordinates.clone()
            if corrupt == "swap":
                broken[0], broken[1] = coordinates[1], coordinates[0]
            elif corrupt == "negate":
                broken[0] = -1
            elif corrupt == "overflow":
                broken[-1] = cols
            else:
                broken = coordinates[torch.randperm(coordinates.numel(),
                                                    generator=generator)]
            check_same(layout, ((), (positions, broken)), values,
                       f"fuzz {seed} {corrupt}")


# --------------------------------------------------------------------------- #
# Validating once instead of twice
#
# `SparseStorage.__init__` validates, and then `STensor._set_state` -- which every
# constructor and every in-place structural change funnels through immediately
# afterwards -- validated the same arrays again. The second walk is skipped when a
# stamp of what was checked (`_validation_stamp`: data_ptr, version counter and numel
# per array) still matches. These tests pin both halves of that: it really is skipped
# when nothing changed, and it really is not skipped otherwise.
# --------------------------------------------------------------------------- #


@pytest.fixture
def validation_counter(monkeypatch):
    """Counts calls to the real validator, which stays in force."""
    calls = []
    real = storage_module._validate_index_storage

    def counting(layout, mode_indices, values):
        calls.append(1)
        return real(layout, mode_indices, values)

    monkeypatch.setattr(storage_module, "_validate_index_storage", counting)
    return calls


def components_csr(rows=4, cols=4, degree=2, seed=1):
    positions, coordinates, values = csr_parts(rows, cols, degree, seed=seed)
    return (rows, cols), "ds", [[], [positions, coordinates]], values


def test_construction_validates_exactly_once(validation_counter):
    shape, fmt, indices, values = components_csr()
    STensor.from_components(shape, fmt, indices, values)
    assert len(validation_counter) == 1


@pytest.mark.parametrize("rows,degree", [(1, 1), (4, 2), (64, 3)])
def test_from_torch_validates_exactly_once(validation_counter, rows, degree):
    positions, coordinates, values = csr_parts(rows, rows, degree, torch.int64)
    csr = torch.sparse_csr_tensor(
        positions.long(), coordinates.long(), values, size=(rows, rows)
    )
    STensor.from_torch(csr)
    assert len(validation_counter) == 1


def test_explicit_validate_always_re_runs_the_full_check(validation_counter):
    shape, fmt, indices, values = components_csr()
    tensor = STensor.from_components(shape, fmt, indices, values)
    before = len(validation_counter)
    tensor.validate()
    tensor.validate()
    assert len(validation_counter) == before + 2


def test_storage_validate_always_re_runs_the_full_check(validation_counter):
    positions, coordinates, values = csr_parts(4, 4, 2)
    store = SparseStorage(
        layout_for("ds", (4, 4)), values, mode_indices=[[], [positions, coordinates]]
    )
    before = len(validation_counter)
    store.validate()
    assert len(validation_counter) == before + 1


def test_in_place_corruption_of_the_internal_arrays_is_still_caught():
    positions, coordinates, values = csr_parts(4, 4, 2, seed=3)
    tensor = STensor.from_components(
        (4, 4), "ds", [[], [positions, coordinates]], values
    )
    internal = tensor._storage._mode_indices[1][1]
    assert internal[0] < internal[1], "need an ascending pair to break"
    internal[1] = internal[0] - 1 if internal[0] > 0 else 0
    internal[0] = 3
    # The stamp's version counter moved, so the skip does not apply.
    with pytest.raises(TensorIndexError, match="sorted within parent"):
        tensor._storage.validate_unless_already_checked()
    with pytest.raises(TensorIndexError, match="sorted within parent"):
        tensor.validate()


def test_in_place_truncation_of_the_values_is_still_caught():
    positions, coordinates, values = csr_parts(4, 4, 2, seed=4)
    tensor = STensor.from_components(
        (4, 4), "ds", [[], [positions, coordinates]], values
    )
    tensor._storage._value.resize_(3)
    with pytest.raises(Exception) as excinfo:
        tensor._storage.validate_unless_already_checked()
    assert "nnz" in str(excinfo.value) or "value" in str(excinfo.value).lower()


def test_unstamped_storage_is_validated_from_scratch(validation_counter):
    """A storage assembled without going through ``__init__`` carries no verdict."""
    positions, coordinates, values = csr_parts(4, 4, 2, seed=5)
    broken = coordinates.clone()
    broken[0], broken[1] = coordinates[1].clone(), coordinates[0].clone()
    store = object.__new__(SparseStorage)
    object.__setattr__(store, "_layout", layout_for("ds", (4, 4)))
    object.__setattr__(store, "_mode_indices", ((), (positions, broken)))
    object.__setattr__(store, "_value", values)
    assert not hasattr(store, "_checked")
    with pytest.raises(TensorIndexError):
        store.validate_unless_already_checked()
    assert len(validation_counter) == 1


def test_a_passing_check_stamps_so_the_next_one_is_free(validation_counter):
    positions, coordinates, values = csr_parts(4, 4, 2, seed=6)
    store = object.__new__(SparseStorage)
    object.__setattr__(store, "_layout", layout_for("ds", (4, 4)))
    object.__setattr__(store, "_mode_indices", ((), (positions, coordinates)))
    object.__setattr__(store, "_value", values)
    store.validate_unless_already_checked()
    store.validate_unless_already_checked()
    assert len(validation_counter) == 1


def test_structural_change_validates_its_new_storage_once(validation_counter):
    shape, fmt, indices, values = components_csr(seed=7)
    tensor = STensor.from_components(shape, fmt, indices, values)
    before = len(validation_counter)
    tensor.change_mode_order([1, 0])
    # The relayout builds one new storage, which is validated when it is built and
    # not again when the tensor adopts it.
    assert len(validation_counter) - before == 1
    tensor.validate()


def test_the_stamp_covers_every_index_array():
    """Not just the first: a two-array level and a two-level format both matter."""
    coordinates = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
    values = torch.tensor([4.0, 5.0])
    tensor = STensor.from_coo(indices=coordinates, values=values, shape=(3, 3))
    stamped = tensor._storage._checked
    arrays = [a for level in tensor._storage._mode_indices for a in level]
    assert len(stamped[0]) == len(arrays) and arrays
    # Reading an array, however thoroughly, must not move the stamp.
    for array in arrays:
        _ = array + 0, array.sum(), array.tolist(), array.clone()
    assert (
        storage_module._validation_stamp(
            tensor._storage._mode_indices, tensor._storage._value
        )
        == stamped
    )
    # Writing to any one of them must, even when the value written is the one that
    # was already there -- the version counter cannot tell, and over-reporting a
    # change only costs a validation that was not needed.
    for index, array in enumerate(arrays):
        original = array.clone()
        array[0] = array[0]
        moved = storage_module._validation_stamp(
            tensor._storage._mode_indices, tensor._storage._value
        )
        assert moved != stamped, f"array {index} is not covered by the stamp"
        array.copy_(original)


@pytest.mark.parametrize("clone", ["pickle", "deepcopy", "copy"])
def test_a_stamp_does_not_survive_being_copied(validation_counter, clone):
    """Addresses and version counters are meaningless outside the process that took
    them, so a copy must carry no verdict and must validate itself on first use."""
    positions, coordinates, values = csr_parts(4, 4, 2, seed=8)
    store = SparseStorage(
        layout_for("ds", (4, 4)), values, mode_indices=[[], [positions, coordinates]]
    )
    assert hasattr(store, "_checked")
    if clone == "pickle":
        revived = pickle.loads(pickle.dumps(store))
    elif clone == "deepcopy":
        revived = copy.deepcopy(store)
    else:
        revived = copy.copy(store)
    assert not hasattr(revived, "_checked")
    before = len(validation_counter)
    revived.validate_unless_already_checked()
    assert len(validation_counter) == before + 1
    assert hasattr(revived, "_checked")


def test_a_copied_tensor_still_rejects_broken_storage(validation_counter):
    """The copy validates for real, not just nominally."""
    shape, fmt, indices, values = components_csr(seed=9)
    tensor = STensor.from_components(shape, fmt, indices, values)
    revived = copy.deepcopy(tensor)
    internal = revived._storage._mode_indices[1][1]
    internal[0], internal[1] = internal[1].clone(), internal[0].clone()
    with pytest.raises(TensorIndexError, match="sorted within parent"):
        revived._storage.validate_unless_already_checked()


# --------------------------------------------------------------------------- #
# The native screen's one-directional promise
#
# The screen may say "go and look" about perfectly good storage -- that only costs a
# walk. What it must never do is clear something malformed, because the Python checks
# it stands in for are then skipped. These tests only assert that direction, over the
# same corruption shapes the differential corpus uses.
# --------------------------------------------------------------------------- #


def screen(positions, coordinates, extent):
    if storage_module._NATIVE_SCREEN is None:
        pytest.skip("extension has no screen entry point")
    return storage_module._NATIVE_SCREEN(positions, coordinates, int(extent), True)


def test_screen_clears_well_formed_storage():
    """Otherwise it is not an optimization at all -- everything would take the slow
    path and the tests below would pass vacuously."""
    positions, coordinates, _ = csr_parts(64, 32, 3)
    assert screen(positions, coordinates, 32) is False


@pytest.mark.parametrize("width", [torch.int32, torch.int64])
def test_screen_never_clears_an_unsorted_parent(width):
    positions, coordinates, _ = csr_parts(16, 8, 4, width)
    for parent in range(16):
        broken = coordinates.clone()
        start = parent * 4
        broken[start], broken[start + 1] = coordinates[start + 1], coordinates[start]
        assert screen(positions, broken, 8) is True, f"cleared parent {parent}"


@pytest.mark.parametrize("width", [torch.int32, torch.int64])
@pytest.mark.parametrize("bad", [-1, 8, 9, 1000])
def test_screen_never_clears_an_out_of_range_coordinate(width, bad):
    positions, coordinates, _ = csr_parts(16, 8, 4, width)
    for where in (0, 1, coordinates.numel() // 2, coordinates.numel() - 1):
        broken = coordinates.clone()
        broken[where] = bad
        assert screen(positions, broken, 8) is True, f"cleared {bad} at {where}"


@pytest.mark.parametrize("width", [torch.int32, torch.int64])
def test_screen_never_clears_broken_positions(width):
    positions, coordinates, _ = csr_parts(16, 8, 4, width)
    for where in range(1, positions.numel()):
        broken = positions.clone()
        broken[where] = 0  # a decrease, unless it already was zero
        if bool((broken[1:] >= broken[:-1]).all()):
            continue
        assert screen(broken, coordinates, 8) is True, f"cleared decrease at {where}"
    beyond = positions.clone()
    beyond[-1] = coordinates.numel() + 1
    assert screen(beyond, coordinates, 8) is True


@pytest.mark.parametrize("seed", range(12))
def test_screen_never_clears_fuzzed_corruption(seed):
    generator = torch.Generator().manual_seed(seed)
    rows = int(torch.randint(1, 9, (1,), generator=generator).item())
    cols = int(torch.randint(2, 9, (1,), generator=generator).item())
    degree = int(torch.randint(2, min(cols, 4) + 1, (1,), generator=generator).item())
    positions, coordinates, _ = csr_parts(rows, cols, degree, seed=seed)
    if coordinates.numel() < 2:
        pytest.skip("nothing to corrupt")
    for mode in ("swap", "negate", "overflow", "shuffle"):
        broken = coordinates.clone()
        if mode == "swap":
            broken[0], broken[1] = coordinates[1].clone(), coordinates[0].clone()
        elif mode == "negate":
            broken[0] = -1
        elif mode == "overflow":
            broken[-1] = cols
        else:
            broken = coordinates[
                torch.randperm(coordinates.numel(), generator=generator)
            ]
        if torch.equal(broken, coordinates):
            continue  # the corruption happened to be a no-op
        # Only assert about corruption the Python validator itself rejects: a shuffle
        # can land back on a legal arrangement, and the screen is not obliged to flag
        # storage that is in fact well formed.
        rejected = True
        try:
            _validate_index_storage(
                layout_for("ds", (rows, cols)), ((), (positions, broken)),
                torch.rand(coordinates.numel()),
            )
            rejected = False
        except Exception:
            pass
        if rejected:
            assert screen(positions, broken, cols) is True, f"cleared {mode} @{seed}"


def test_the_screen_is_actually_being_consulted(monkeypatch):
    """Pin the wiring: if the screen clears a level, the Python walks do not run."""
    calls = []
    real = storage_module._check_sorted_within_parents

    def counting(positions, coordinates, mode):
        calls.append(1)
        return real(positions, coordinates, mode)

    monkeypatch.setattr(storage_module, "_check_sorted_within_parents", counting)
    positions, coordinates, values = csr_parts(32, 16, 3)
    STensor.from_components((32, 16), "ds", [[], [positions, coordinates]], values)
    if storage_module._NATIVE_SCREEN is None:
        assert len(calls) == 1  # no screen: the Python walk is the only check
    else:
        assert calls == []  # screened clean: the Python walk was not needed


# --------------------------------------------------------------------------- #
# COO: the lexicographic screen
#
# The Python loop this stands in for compared a tuple per NONZERO, so the screen has
# to agree with it on the one thing tuple comparison does that a per-level test does
# not: the first level that differs decides, and every deeper level is then irrelevant.
# A screen that only looked at the last level, or only at the first, would still pass
# a naive test -- so these cases are built to separate those.
# --------------------------------------------------------------------------- #


def lex(levels, count=None):
    if storage_module._NATIVE_LEX is None:
        pytest.skip("extension has no lexicographic screen")
    n = levels[0].numel() if count is None else count
    return storage_module._NATIVE_LEX(list(levels), int(n))


def i32(values):
    return torch.tensor(values, dtype=torch.int32)


@pytest.mark.parametrize("width", [torch.int32, torch.int64])
def test_lex_screen_sees_a_descent_only_the_deeper_level_can_see(width):
    """mode 0 ties, so only mode 1 reveals the descent."""
    rows = torch.tensor([0, 0], dtype=width)
    assert lex([rows, torch.tensor([3, 0], dtype=width)]) is True
    assert lex([rows, torch.tensor([0, 3], dtype=width)]) is False


def test_lex_screen_ignores_deeper_levels_once_a_level_decides():
    """(1, 1) -> (2, 0) ascends: mode 0 decided, mode 1 must not veto it."""
    assert lex([i32([1, 2]), i32([1, 0])]) is False
    assert lex([i32([0, 0, 1, 2]), i32([0, 3, 1, 0])]) is False


def test_lex_screen_treats_duplicates_as_ordered():
    """Equal coordinates are not a descent, and the Python loop agrees (`<` not `<=`)."""
    assert lex([i32([0, 0]), i32([1, 1])]) is False
    assert lex([i32([0, 0, 0]), i32([2, 2, 2])]) is False


def test_lex_screen_on_a_single_level():
    assert lex([i32([0, 1, 5])]) is False
    assert lex([i32([0, 5, 1])]) is True


def test_lex_screen_three_levels_deep():
    assert lex([i32([0, 0]), i32([1, 1]), i32([5, 2])]) is True
    assert lex([i32([0, 0]), i32([1, 1]), i32([2, 5])]) is False


@pytest.mark.parametrize("count", [0, 1])
def test_lex_screen_has_nothing_to_say_about_short_input(count):
    assert lex([i32([5, 0]), i32([5, 0])], count) is False


def test_lex_screen_declines_what_it_cannot_serve():
    """Every decline is "go and look", so validation still happens."""
    rows, cols = i32([0, 0]), i32([3, 0])
    assert lex([rows.long(), cols]) is True  # mismatched widths
    assert lex([rows, cols], 99) is True  # count past the arrays
    assert lex([rows.float(), cols.float()]) is True  # unsupported dtype
    assert lex([rows, cols.repeat(2)[::2]]) is True  # non-contiguous


@pytest.mark.parametrize("seed", range(16))
def test_coo_validation_agrees_with_the_oracle_over_fuzzed_input(seed):
    """The differential comparison, on COO specifically, ordered and not."""
    generator = torch.Generator().manual_seed(seed)
    nnz = int(torch.randint(2, 40, (1,), generator=generator).item())
    extent = int(torch.randint(2, 12, (1,), generator=generator).item())
    rows = torch.randint(0, extent, (nnz,), generator=generator, dtype=torch.int32)
    cols = torch.randint(0, extent, (nnz,), generator=generator, dtype=torch.int32)
    order = sorted(range(nnz), key=lambda i: (int(rows[i]), int(cols[i])))
    srows, scols = rows[order], cols[order]
    values = torch.rand(nnz, generator=generator)
    layout = layout_for("oo", (extent, extent))
    check_same(layout, ((srows,), (scols,)), values, f"coo sorted {seed}")
    check_same(layout, ((rows,), (cols,)), values, f"coo as-generated {seed}")
    # A real descent at every position, in each level in turn. Level 1 is the case
    # that matters most: it needs the level above it to tie, so a screen that stopped
    # after the first level would miss it.
    for spot in range(1, nnz):
        if srows[spot - 1] > 0:
            broken_rows = srows.clone()
            broken_rows[spot] = srows[spot - 1] - 1  # decides at level 0
            check_same(layout, ((broken_rows,), (scols,)), values,
                       f"coo level-0 descent {seed} at {spot}")
        if scols[spot - 1] > 0:
            broken_rows, broken_cols = srows.clone(), scols.clone()
            broken_rows[spot] = srows[spot - 1]  # tie at level 0 ...
            broken_cols[spot] = scols[spot - 1] - 1  # ... descent at level 1
            check_same(layout, ((broken_rows,), (broken_cols,)), values,
                       f"coo level-1 descent {seed} at {spot}")


def test_the_coo_screen_is_actually_being_consulted(monkeypatch):
    """A cleared COO tensor must not materialize its index arrays at all."""
    calls = []
    real = torch.Tensor.tolist

    def counting(self):
        calls.append(1)
        return real(self)

    nnz, extent = 64, 16
    generator = torch.Generator().manual_seed(5)
    rows = torch.randint(0, extent, (nnz,), generator=generator, dtype=torch.int32)
    cols = torch.randint(0, extent, (nnz,), generator=generator, dtype=torch.int32)
    order = sorted(range(nnz), key=lambda i: (int(rows[i]), int(cols[i])))
    rows, cols = rows[order], cols[order]
    values = torch.rand(nnz, generator=generator)
    monkeypatch.setattr(torch.Tensor, "tolist", counting)
    _validate_index_storage(
        layout_for("oo", (extent, extent)), ((rows,), (cols,)), values
    )
    if storage_module._NATIVE_LEX is None:
        assert len(calls) >= 2  # no screen: both modes materialized
    else:
        assert calls == []  # screened clean: nothing materialized
