"""Operands and results as level arrays, without the library's own conversions.

``STensor.to_sparse``, ``to_torch`` on a sparse tensor and ``change_mode_order`` are
compiled kernels themselves.  A test of one generated kernel that builds its operand or
reads its result through them is a test of two or three generated kernels at once, and
cannot say which one is wrong.  The helpers here build an operand's level arrays and
decode a result's level arrays in PyTorch, and return the C++ a public ``einsum`` call
emits without building it.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from scorch import ops
from scorch.layout import TensorSpec
from scorch.stensor import STensor
from tests.test_scorch.kernel_cache_support import empty_kernel_caches


def level_arrays(
    dense: torch.Tensor, fmt: str
) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
    """The level arrays and stored values holding ``dense`` in ``fmt``.

    ``fmt`` has one letter per storage level, ``d`` (dense) or ``s`` (compressed), or is
    ``o`` (coordinate) in every level.  A compressed level stores a coordinate exactly when
    some entry under it is nonzero.  A coordinate tensor stores each nonzero once, as one
    coordinate array per level, in lexicographic order.
    """

    shape = tuple(dense.shape)
    coords = dense.nonzero()
    if fmt and set(fmt) == {"o"}:
        assert len(fmt) == dense.dim(), fmt
        crd = [[coords[:, level].clone()] for level in range(dense.dim())]
        return crd, dense[tuple(coords.t())]
    position = torch.zeros(coords.shape[0], dtype=torch.int64)
    count = 1
    mode_indices: List[List[torch.Tensor]] = []
    for level, kind in enumerate(fmt):
        extent = shape[level]
        key = position * extent + coords[:, level]
        if kind == "d":
            mode_indices.append([])
            position, count = key, count * extent
            continue
        assert kind == "s", fmt
        stored, position = torch.unique(key, sorted=True, return_inverse=True)
        parent = torch.div(stored, extent, rounding_mode="floor") if extent else stored
        pos = torch.zeros(count + 1, dtype=torch.int64)
        pos[1:] = torch.cumsum(torch.bincount(parent, minlength=count), 0)
        mode_indices.append([pos, stored - parent * extent])
        count = stored.numel()
    values = torch.zeros(count, dtype=dense.dtype)
    values[position] = dense[tuple(coords.t())]
    return mode_indices, values


def stored(
    dense: torch.Tensor,
    fmt: str,
    name: str,
    mode_order: Optional[Sequence[int]] = None,
) -> STensor:
    """``dense`` (logical mode order) as an ``STensor`` in ``fmt``.

    ``mode_order[level]`` is the logical mode stored at ``level``; the format's letters
    describe storage levels.
    """

    order = list(range(dense.dim())) if mode_order is None else list(mode_order)
    physical = dense.permute(order).contiguous() if dense.dim() else dense
    mode_indices, values = level_arrays(physical, fmt)
    return STensor.from_components(
        tuple(physical.shape),
        fmt,
        mode_indices,
        values,
        name=name,
        mode_order=None if order == sorted(order) else order,
    )


def logical_dense(result) -> torch.Tensor:
    """A result as a dense float64 tensor in logical mode order.

    Read from the result's own format, level arrays and mode order.  Each compressed
    level is checked on the way (position array length, start, monotonicity, coordinate
    count, range and strict order inside a parent), so a malformed result fails here
    with the level named instead of decoding to something.
    """

    if isinstance(result, torch.Tensor):
        return result.to(torch.float64)
    kinds = [level_type.name for level_type in result.format.level_types]
    shape = tuple(result.physical_shape)
    order = [int(mode) for mode in result.mode_order]
    if kinds and set(kinds) == {"COORDINATE"}:
        coords, values = coordinate_entries(result)
        physical = torch.zeros(shape, dtype=torch.float64)
        physical[tuple(coords.t())] = values.to(torch.float64)
        return physical.permute([order.index(mode) for mode in range(len(order))])
    arrays = result.storage.index.mode_indices
    coords = torch.zeros((1, 0), dtype=torch.int64)
    for level, kind in enumerate(kinds):
        parents, extent = coords.shape[0], shape[level]
        if kind == "DENSE":
            parent = torch.arange(parents).repeat_interleave(extent)
            crd = torch.arange(extent).repeat(parents)
        else:
            assert kind == "COMPRESSED", f"level {level} is {kind}"
            assert (
                len(arrays[level]) == 2
            ), f"level {level}: {len(arrays[level])} arrays"
            pos, crd = (array.reshape(-1).to(torch.int64) for array in arrays[level])
            assert (
                pos.numel() == parents + 1
            ), f"level {level}: {pos.numel()} positions for {parents} parents"
            assert int(pos[0]) == 0 and not bool(
                (pos[1:] < pos[:-1]).any()
            ), f"level {level}: positions do not start at 0 or decrease"
            assert crd.numel() == int(pos[-1]), (
                f"level {level}: {crd.numel()} coordinates, positions end at "
                f"{int(pos[-1])}"
            )
            assert not crd.numel() or (
                int(crd.min()) >= 0 and int(crd.max()) < extent
            ), f"level {level}: a coordinate outside 0..{extent - 1}"
            parent = torch.arange(parents).repeat_interleave(pos[1:] - pos[:-1])
            assert not bool(
                ((parent[1:] == parent[:-1]) & (crd[1:] <= crd[:-1])).any()
            ), f"level {level}: coordinates not strictly increasing inside a parent"
        coords = torch.cat([coords[parent], crd.reshape(-1, 1)], dim=1)
    values = result.values.reshape(-1)
    assert (
        values.numel() == coords.shape[0]
    ), f"{values.numel()} values for {coords.shape[0]} stored positions"
    physical = torch.zeros(shape, dtype=torch.float64)
    physical[tuple(coords.t())] = values.to(torch.float64)
    return physical.permute([order.index(mode) for mode in range(len(order))])


def coordinate_entries(result) -> Tuple[torch.Tensor, torch.Tensor]:
    """An all-coordinate result's stored coordinates and values, in storage order.

    One row of coordinates per stored entry.  Checked on the way: one coordinate array
    per level, each as long as the values; every coordinate inside its extent; entries in
    strictly increasing lexicographic order.  That order is what the library's own input
    validation requires of a coordinate tensor, so a result that breaks it cannot be
    passed to the next kernel, and it also rules out storing a coordinate twice -- which a
    dense reading would hide by adding the copies.
    """

    shape = tuple(result.physical_shape)
    arrays = result.storage.index.mode_indices
    values = result.values.reshape(-1)
    columns = []
    for level, level_arrays_ in enumerate(arrays):
        assert (
            len(level_arrays_) == 1
        ), f"level {level}: {len(level_arrays_)} arrays, a coordinate level has one"
        crd = level_arrays_[0].reshape(-1).to(torch.int64)
        assert (
            crd.numel() == values.numel()
        ), f"level {level}: {crd.numel()} coordinates for {values.numel()} stored values"
        assert not crd.numel() or (
            int(crd.min()) >= 0 and int(crd.max()) < shape[level]
        ), f"level {level}: a coordinate outside 0..{shape[level] - 1}"
        columns.append(crd)
    coords = torch.stack(columns, dim=1) if columns else torch.zeros((0, 0))
    if coords.shape[0] > 1:
        key = torch.zeros(coords.shape[0], dtype=torch.int64)
        for level, extent in enumerate(shape):
            key = key * extent + coords[:, level]
        assert bool(
            (key[1:] > key[:-1]).all()
        ), "entries not in strictly increasing lexicographic order (or one stored twice)"
    return coords, values


def reference(expr: str, operands: Sequence[torch.Tensor]) -> torch.Tensor:
    """``torch.einsum`` on float64 copies (exact for the integers the tests use)."""

    return torch.einsum(expr, *[operand.to(torch.float64) for operand in operands])


def assert_matches(result, expected: torch.Tensor, dtype: torch.dtype) -> None:
    """float64 to 1e-12 of each cell, float32 to 1e-5, integers exactly."""

    got = logical_dense(result)
    assert tuple(got.shape) == tuple(expected.shape)
    if not dtype.is_floating_point:
        assert torch.equal(got, expected)
        return
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    assert torch.allclose(got, expected, rtol=tolerance, atol=0.0)


def emitted_einsum_cpp(
    monkeypatch,
    expr: str,
    formats: Sequence[str],
    extents: Dict[str, int],
    *,
    dtype: torch.dtype = torch.float64,
    **kwargs,
) -> str:
    """The kernel body public ``einsum`` emits for ``expr``; nothing is built.

    Operands are payload-free ``TensorSpec`` descriptions, so no operand conversion
    runs either.  The kernel caches are emptied for the call (empty_kernel_caches).
    """

    captured: List[str] = []

    def stop_before_native(prepared):
        captured.append(prepared.request.cpp_sources[-1])
        return object()

    empty_kernel_caches(monkeypatch)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", stop_before_native)
    terms = expr.split("->")[0].split(",")
    specs = [
        TensorSpec(
            fmt,
            tuple(extents[index] for index in term),
            dtype=dtype,
            name="ABDEFGH"[position],
        )
        for position, (term, fmt) in enumerate(zip(terms, formats))
    ]
    ops.einsum(expr, *specs, compile_only=True, **kwargs)
    assert len(captured) == 1, f"{len(captured)} kernels prepared"
    return captured[0]
