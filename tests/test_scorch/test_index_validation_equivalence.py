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

import math

import pytest
import torch

from scorch import TensorLayout
from scorch.exceptions import (
    TensorDeviceError,
    TensorIndexError,
    TensorStorageError,
)
from scorch.format import LevelType
from scorch.storage import (
    IndexModes,
    _check_coordinate_bounds,
    _validate_index_storage,
)


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
