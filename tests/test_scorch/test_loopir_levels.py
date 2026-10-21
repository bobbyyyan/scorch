"""Format-neutral level-storage coverage for the production LoopIR oracle.

The execution core consumes only ``segment`` / ``next_position`` /
``coordinate_at`` / ``leaf_value`` on a validated
:class:`LevelTensorStorage`; CSR is exactly one adapter.  These tests lock
the canonical-storage contract, the CSR container/adapter boundaries, the
order-checked output builder, and COO storage (every level a coordinate
level).
"""

import itertools
import random

import pytest

from scorch.compiler.loopir.levels import (
    CompressedLevel,
    CoordinateLevel,
    CsrFormatError,
    CsrMatrix,
    CsrOutputBuilder,
    DenseLevel,
    LevelOutputBuilder,
    LevelStorageError,
    LevelTensor,
    LevelTensorStorage,
    MAX_LEVEL_STORAGE_RANK,
    from_csr,
)
from scorch.compiler.loopir.nodes import LevelKind


def csr_fixture() -> CsrMatrix:
    return CsrMatrix(
        n_rows=3,
        n_cols=4,
        indptr=(0, 2, 2, 3),
        indices=(0, 3, 1),
        values=(1.0, 2.0, 3.0),
    )


def test_from_csr_exposes_level_contracts():
    storage = from_csr(csr_fixture())
    assert storage.kinds == (LevelKind.DENSE, LevelKind.COMPRESSED)
    assert storage.segment(1, 0) == (0, 2)
    assert storage.segment(1, 1) == (2, 2)
    assert storage.segment(1, 2) == (2, 3)
    assert storage.coordinate_at(1, 1) == 3
    assert storage.leaf_value(2) == 3.0


def test_from_dense_round_trips_csr_and_permuted_layouts():
    dense = [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
    for modes, kinds in (
        ((0, 1), (LevelKind.DENSE, LevelKind.COMPRESSED)),
        ((1, 0), (LevelKind.DENSE, LevelKind.COMPRESSED)),
        ((0, 1), (LevelKind.COMPRESSED, LevelKind.COMPRESSED)),
        ((0, 1), (LevelKind.DENSE, LevelKind.DENSE)),
    ):
        storage = LevelTensorStorage.from_dense(dense, (2, 3), modes, kinds)
        assert storage.to_dense() == dense


def test_positions_differ_from_coordinates_in_dcsr():
    """The stored row position of a DCSR matrix is not its row coordinate."""

    dense = [
        [0.0, 0.0],
        [0.0, 0.0],
        [5.0, 0.0],
    ]
    storage = LevelTensorStorage.from_dense(
        dense, (3, 2), (0, 1), (LevelKind.COMPRESSED, LevelKind.COMPRESSED)
    )
    start, end = storage.segment(0, 0)
    assert (start, end) == (0, 1)
    assert storage.coordinate_at(0, 0) == 2  # position 0 stores coordinate 2
    inner = storage.segment(1, 0)
    assert inner == (0, 1)
    assert storage.coordinate_at(1, 0) == 0
    assert storage.leaf_value(0) == 5.0


def test_storage_snapshot_detaches_from_caller_mutation():
    storage = from_csr(csr_fixture())
    snapshot = storage.snapshot()
    object.__setattr__(storage, "values", (9.0, 9.0, 9.0))
    assert snapshot.leaf_value(0) == 1.0


def test_storage_rejects_inconsistent_offsets():
    with pytest.raises(LevelStorageError):
        LevelTensorStorage(
            shape=(2, 2),
            modes=(0, 1),
            levels=(DenseLevel(2), CompressedLevel((0, 1), (0, 1))),
            values=(1.0, 2.0),
        )


def test_storage_rejects_decreasing_coordinates():
    with pytest.raises(LevelStorageError):
        LevelTensorStorage(
            shape=(1, 3),
            modes=(0, 1),
            levels=(DenseLevel(1), CompressedLevel((0, 2), (2, 1))),
            values=(1.0, 2.0),
        )


def test_storage_rejects_wrong_value_count():
    with pytest.raises(LevelStorageError):
        LevelTensorStorage(
            shape=(1, 3),
            modes=(0, 1),
            levels=(DenseLevel(1), CompressedLevel((0, 2), (0, 1))),
            values=(1.0,),
        )


def test_storage_rejects_forged_state_on_validate():
    storage = from_csr(csr_fixture())
    object.__setattr__(storage, "values", (1.0, 2.0))
    with pytest.raises(LevelStorageError):
        storage.validate()


def test_segment_rejects_out_of_range_parents():
    storage = from_csr(csr_fixture())
    with pytest.raises(LevelStorageError):
        storage.segment(1, 3)
    with pytest.raises(LevelStorageError):
        storage.segment(0, 0)


def test_csr_container_rejects_malformed_state():
    with pytest.raises(CsrFormatError):
        CsrMatrix(n_rows=2, n_cols=2, indptr=(0, 1), indices=(0,), values=(1.0,))
    with pytest.raises(CsrFormatError):
        CsrMatrix(
            n_rows=1,
            n_cols=2,
            indptr=(0, 2),
            indices=(1, 0),
            values=(1.0, 2.0),
        )
    with pytest.raises(CsrFormatError):
        CsrMatrix(n_rows=1, n_cols=1, indptr=(0, 1), indices=(0,), values=(1,))


def test_csr_from_dense_prunes_zeros_and_round_trips():
    dense = [[0.0, 1.5], [0.0, 0.0]]
    matrix = CsrMatrix.from_dense(dense)
    assert matrix.indptr == (0, 1, 1)
    assert matrix.indices == (1,)
    assert matrix.to_dense() == dense


def test_csr_from_dense_wraps_unrepresentable_numeric_values():
    with pytest.raises(CsrFormatError, match="representable numeric"):
        CsrMatrix.from_dense([[10**10000]])


def test_level_storage_rejects_rank_before_recursive_conversion():
    rank = MAX_LEVEL_STORAGE_RANK + 1
    dense = 1.0
    for _ in range(rank):
        dense = [dense]
    with pytest.raises(LevelStorageError, match="exceeds the level-storage limit"):
        LevelTensorStorage.from_dense(
            dense,
            (1,) * rank,
            tuple(range(rank)),
            (LevelKind.DENSE,) * rank,
        )


def test_output_builder_orders_and_finishes():
    builder = CsrOutputBuilder("C", (2, 3))
    builder.append((0, 1), 1.0)
    builder.append((1, 0), 2.0)
    builder.append((1, 2), 0.0)  # explicit zero from UNION cancellation stays
    matrix = builder.finish()
    assert matrix.indptr == (0, 1, 3)
    assert matrix.indices == (1, 0, 2)
    assert matrix.values == (1.0, 2.0, 0.0)


def test_output_builder_rejects_unordered_appends():
    builder = CsrOutputBuilder("C", (2, 3))
    builder.append((1, 1), 1.0)
    with pytest.raises(LevelStorageError):
        builder.append((1, 1), 2.0)
    with pytest.raises(LevelStorageError):
        builder.append((0, 2), 2.0)


def test_output_builder_rejects_escaping_coordinates():
    builder = CsrOutputBuilder("C", (2, 3))
    with pytest.raises(LevelStorageError):
        builder.append((2, 0), 1.0)
    with pytest.raises(LevelStorageError):
        builder.append((0, 3), 1.0)


def test_level_output_builder_handles_zero_trailing_dense_extents():
    compressed_dense = LevelOutputBuilder(
        "C",
        (2, 0),
        (LevelKind.COMPRESSED, LevelKind.DENSE),
    ).finish()
    assert compressed_dense.positions == ((0, 0), None)
    assert compressed_dense.coordinates == ((), None)
    assert compressed_dense.values == ()

    dense_compressed_dense = LevelOutputBuilder(
        "D",
        (2, 3, 0),
        (LevelKind.DENSE, LevelKind.COMPRESSED, LevelKind.DENSE),
    ).finish()
    assert dense_compressed_dense.positions == (None, (0, 0, 0), None)
    assert dense_compressed_dense.coordinates == (None, (), None)
    assert dense_compressed_dense.values == ()


def test_level_output_builder_does_not_materialize_empty_dense_suffixes():
    result = LevelOutputBuilder(
        "C",
        (1, 10**12),
        (LevelKind.COMPRESSED, LevelKind.DENSE),
    ).finish()
    assert result.values == ()


@pytest.mark.parametrize("kind", (LevelKind.COORDINATE, LevelKind.SINGLETON))
def test_level_output_builder_rejects_unsupported_level_kinds(kind):
    with pytest.raises(LevelStorageError, match="DENSE and COMPRESSED"):
        LevelOutputBuilder(
            "C",
            (2, 2),
            (LevelKind.COMPRESSED, kind),
        )


def test_level_output_builder_bounds_rank_before_assembly():
    with pytest.raises(LevelStorageError, match="exceeds the level-storage limit"):
        LevelOutputBuilder(
            "C",
            (1,) * (MAX_LEVEL_STORAGE_RANK + 1),
            (LevelKind.COMPRESSED,) * (MAX_LEVEL_STORAGE_RANK + 1),
        )


def test_level_output_builder_zero_fills_cells_no_append_addressed():
    # A stored coordinate stream appends only what its operand stored.  The
    # block it opens under row 1 still has both cells, and the one it never
    # addressed holds zero.  Row 0 has nothing under it and is not stored.
    builder = LevelOutputBuilder(
        "C",
        (2, 2),
        (LevelKind.COMPRESSED, LevelKind.DENSE),
    )
    builder.append((1, 0), 2.0)
    result = builder.finish()
    assert result.positions == ((0, 1), None)
    assert result.coordinates == ((1,), None)
    assert result.values == (2.0, 0.0)

    last_cell_only = LevelOutputBuilder(
        "C",
        (2, 2),
        (LevelKind.COMPRESSED, LevelKind.DENSE),
    )
    last_cell_only.append((0, 1), 5.0)
    last_cell_only.append((1, 1), 7.0)
    result = last_cell_only.finish()
    assert result.positions == ((0, 2), None)
    assert result.coordinates == ((0, 1), None)
    assert result.values == (0.0, 5.0, 0.0, 7.0)

    complete = LevelOutputBuilder(
        "C",
        (2, 2),
        (LevelKind.COMPRESSED, LevelKind.DENSE),
    )
    complete.append((1, 0), 2.0)
    complete.append((1, 1), 3.0)
    result = complete.finish()
    assert result.positions == ((0, 1), None)
    assert result.coordinates == ((1,), None)
    assert result.values == (2.0, 3.0)


def test_level_output_builder_keeps_an_appended_zero_as_structure():
    # Structure follows what was appended, never the value appended: a block
    # holding only an explicit zero is stored, and so is its coordinate.
    builder = LevelOutputBuilder(
        "C",
        (3, 2),
        (LevelKind.COMPRESSED, LevelKind.DENSE),
    )
    builder.append((2, 1), 0.0)
    result = builder.finish()
    assert result.positions == ((0, 1), None)
    assert result.coordinates == ((2,), None)
    assert result.values == (0.0, 0.0)


def _storage_by_definition(shape, level_kinds, entries):
    """Level storage of ``entries`` read off the storage rule, top down.

    Written as a recursion over stored positions rather than as passes over
    the append stream, so it shares no code path with the builder: a
    compressed level stores the coordinates that have an entry under them, a
    dense level stores every coordinate of its extent, and the last level's
    positions each hold the entry at that coordinate tuple or zero.
    """

    rank = len(shape)
    positions = [None if kind is LevelKind.DENSE else [0] for kind in level_kinds]
    coordinates = [None if kind is LevelKind.DENSE else [] for kind in level_kinds]
    values = []

    def descend(level, prefix):
        if level == rank:
            values.append(entries.get(prefix, 0.0))
            return
        if level_kinds[level] is LevelKind.DENSE:
            stored = range(shape[level])
        else:
            stored = sorted(
                {coords[level] for coords in entries if coords[:level] == prefix}
            )
        for coordinate in stored:
            if level_kinds[level] is LevelKind.COMPRESSED:
                coordinates[level].append(coordinate)
            descend(level + 1, prefix + (coordinate,))
        if level_kinds[level] is LevelKind.COMPRESSED:
            positions[level].append(len(coordinates[level]))

    descend(0, ())
    return (
        tuple(None if p is None else tuple(p) for p in positions),
        tuple(None if c is None else tuple(c) for c in coordinates),
        tuple(values),
    )


def _layouts(rank):
    for kinds in itertools.product(
        (LevelKind.DENSE, LevelKind.COMPRESSED), repeat=rank
    ):
        if LevelKind.COMPRESSED in kinds:
            yield kinds


@pytest.mark.parametrize(
    "level_kinds",
    [kinds for rank in (1, 2, 3, 4) for kinds in _layouts(rank)],
    ids=lambda kinds: "".join("d" if k is LevelKind.DENSE else "s" for k in kinds),
)
def test_level_output_builder_follows_the_storage_rule_in_every_layout(level_kinds):
    # Every DENSE/COMPRESSED layout through rank 4, over every subset of
    # cells for small shapes and random subsets for larger ones, with unit
    # and zero extents among them.  The dense level may sit anywhere: above
    # every compressed level, between two, or below the last.
    rank = len(level_kinds)
    rng = random.Random(20261011 + rank)
    shapes = [
        tuple([2, 3, 2, 2][:rank]),
        tuple([1, 1, 1, 1][:rank]),
        tuple([3, 1, 2, 1][:rank]),
        tuple([2, 0, 3, 2][:rank]),
        tuple([0, 2, 2, 2][:rank]),
        tuple([2, 2, 2, 0][:rank]),
    ]
    checked = 0
    for shape in dict.fromkeys(shapes):
        cells = list(itertools.product(*(range(extent) for extent in shape)))
        if len(cells) <= 6:
            subsets = [
                [cell for bit, cell in enumerate(cells) if mask >> bit & 1]
                for mask in range(1 << len(cells))
            ]
        else:
            subsets = [[], cells] + [
                [cell for cell in cells if rng.random() < density]
                for density in (0.1, 0.3, 0.5, 0.8)
                for _ in range(6)
            ]
        for subset in subsets:
            entries = {cell: float(rng.randint(-3, 3)) for cell in subset}
            builder = LevelOutputBuilder("C", shape, level_kinds)
            for cell in sorted(entries):
                builder.append(cell, entries[cell])
            result = builder.finish()
            expected = _storage_by_definition(shape, level_kinds, entries)
            assert (result.positions, result.coordinates, result.values) == expected
            checked += 1
    assert checked >= 6


def test_level_tensor_validates_its_canonical_storage():
    with pytest.raises(LevelStorageError, match="one owned value"):
        LevelTensor(
            shape=(2,),
            level_kinds=(LevelKind.COMPRESSED,),
            positions=((0, 1),),
            coordinates=((1,),),
            values=(),
        )
    with pytest.raises(LevelStorageError, match="strictly increasing"):
        LevelTensor(
            shape=(3,),
            level_kinds=(LevelKind.COMPRESSED,),
            positions=((0, 2),),
            coordinates=((2, 1),),
            values=(1.0, 2.0),
        )


def test_level_diagnostics_bound_huge_exact_integers():
    huge = 10**5000
    with pytest.raises(LevelStorageError, match="integer too large to render"):
        LevelTensor(
            shape=(huge,),
            level_kinds=(LevelKind.COMPRESSED,),
            positions=((0, 1),),
            coordinates=((huge,),),
            values=(1.0,),
        )

    builder = LevelOutputBuilder(
        "C",
        (huge,),
        (LevelKind.COMPRESSED,),
    )
    with pytest.raises(LevelStorageError, match="integer too large to render"):
        builder.append((huge,), 1.0)


# -- COO: tensors whose levels are all coordinate levels ----------------------

COORDINATE = LevelKind.COORDINATE
COMPRESSED = LevelKind.COMPRESSED
DENSE = LevelKind.DENSE
_COORDINATE_ONLY = "requires every level of the tensor to be a coordinate level"


def coo_storage(shape, modes, entries, values=None):
    """COO storage of ``entries``, each a coordinate tuple in level order."""

    if values is None:
        values = tuple(float(position + 1) for position in range(len(entries)))
    return LevelTensorStorage(
        shape=shape,
        modes=modes,
        levels=tuple(
            CoordinateLevel(tuple(entry[level] for entry in entries))
            for level in range(len(shape))
        ),
        values=values,
    )


def stored_entries(storage):
    """Every stored entry, read through the level interface alone.

    One cursor per level walks ``segment`` with ``next_position``, the way
    the oracle's nested sparse loops do, and returns ``(coordinates in level
    order, value)`` in visiting order.
    """

    rank = len(storage.shape)
    found = []

    def descend(level, parent_position, prefix):
        position, end = storage.segment(level, parent_position)
        while position < end:
            coords = prefix + (storage.coordinate_at(level, position),)
            if level == rank - 1:
                found.append((coords, storage.leaf_value(position)))
            else:
                descend(level + 1, position, coords)
            position = storage.next_position(level, position, end)

    descend(0, 0, ())
    return found


def random_dense(rng, shape, density):
    """Nested lists of ``shape`` with nonzero integer-valued floats."""

    if len(shape) == 1:
        return [
            float(rng.randint(1, 9)) if rng.random() < density else 0.0
            for _ in range(shape[0])
        ]
    return [random_dense(rng, shape[1:], density) for _ in range(shape[0])]


def nonzero_cells(dense, shape, modes):
    """``(coordinates in level order, value)`` of the nonzero cells, sorted."""

    cells = []
    for logical in itertools.product(*(range(extent) for extent in shape)):
        value = dense
        for coordinate in logical:
            value = value[coordinate]
        if value != 0.0:
            cells.append((tuple(logical[mode] for mode in modes), value))
    return sorted(cells)


@pytest.mark.parametrize(
    "shape,modes,entries",
    [
        ((5,), (0,), [(0,), (3,), (4,)]),
        ((3, 4), (0, 1), [(0, 1), (0, 3), (2, 0)]),
        ((2, 3, 4), (0, 1, 2), [(0, 0, 1), (0, 2, 0), (1, 1, 3)]),
        ((3, 4), (0, 1), []),
        ((0, 4), (0, 1), []),
        # Level 0 stores mode 1 (extent 4) and level 1 stores mode 0 (extent
        # 3): coordinate 3 on level 0 is inside its extent only through modes.
        ((3, 4), (1, 0), [(0, 2), (3, 0), (3, 1)]),
        ((2, 3, 4), (2, 0, 1), [(0, 1, 2), (3, 0, 0), (3, 1, 2)]),
    ],
    ids=["rank1", "rank2", "rank3", "empty", "zero-extent", "permuted", "permuted3"],
)
def test_coordinate_storage_accepts_canonical_tensors(shape, modes, entries):
    storage = coo_storage(shape, modes, entries)
    storage.validate()
    assert storage.kinds == (COORDINATE,) * len(shape)
    assert storage.snapshot() == storage
    assert stored_entries(storage) == [
        (entry, float(position + 1)) for position, entry in enumerate(entries)
    ]
    dense = storage.to_dense()
    for position, entry in enumerate(entries):
        cell = dense
        for mode in range(len(shape)):
            cell = cell[entry[modes.index(mode)]]
        assert cell == float(position + 1)


@pytest.mark.parametrize(
    "entries,match",
    [
        ([(0, 1), (0, 1)], "lexicographically increasing"),
        ([(1, 0), (0, 3)], "lexicographically increasing"),
        ([(0, 3), (0, 1)], "lexicographically increasing"),
        ([(0, 1), (1, 0), (0, 2)], "lexicographically increasing"),
        ([(0, 4)], r"levels\[1\] coordinate 4 outside \[0, 4\)"),
        ([(3, 0)], r"levels\[0\] coordinate 3 outside \[0, 3\)"),
        ([(-1, 0)], r"levels\[0\] coordinate -1 outside \[0, 3\)"),
        ([(True, 0)], "exact int"),
        ([(0, 1.0)], "exact int"),
    ],
    ids=[
        "repeated",
        "decreasing-level0",
        "decreasing-level1",
        "unsorted",
        "past-extent-level1",
        "past-extent-level0",
        "negative",
        "bool",
        "float",
    ],
)
def test_coordinate_storage_rejects_noncanonical_entries(entries, match):
    with pytest.raises(LevelStorageError, match=match):
        coo_storage((3, 4), (0, 1), entries)


def test_coordinate_storage_rejects_mismatched_stream_lengths():
    with pytest.raises(LevelStorageError, match="1 coordinates for 2 stored entries"):
        LevelTensorStorage(
            shape=(3, 4),
            modes=(0, 1),
            levels=(CoordinateLevel((0, 1)), CoordinateLevel((0,))),
            values=(1.0, 2.0),
        )
    with pytest.raises(LevelStorageError, match="3 coordinates for 2 stored entries"):
        LevelTensorStorage(
            shape=(3, 4),
            modes=(0, 1),
            levels=(CoordinateLevel((0, 1)), CoordinateLevel((0, 1, 2))),
            values=(1.0, 2.0),
        )
    for values in ((1.0,), (1.0, 2.0, 3.0), ()):
        with pytest.raises(LevelStorageError, match="values for 2 leaf positions"):
            coo_storage((3, 4), (0, 1), [(0, 1), (2, 0)], values=values)
    with pytest.raises(LevelStorageError, match="exact float"):
        coo_storage((3, 4), (0, 1), [(0, 1)], values=(1,))
    with pytest.raises(LevelStorageError, match="owned tuple"):
        LevelTensorStorage(
            shape=(3,), modes=(0,), levels=(CoordinateLevel([0, 1]),), values=(1.0, 2.0)
        )


@pytest.mark.parametrize(
    "levels,values",
    [
        ((DenseLevel(2), CoordinateLevel((0, 1))), (1.0, 2.0)),
        ((CoordinateLevel((0, 1)), DenseLevel(2)), (1.0, 2.0, 3.0, 4.0)),
        ((CompressedLevel((0, 2), (0, 1)), CoordinateLevel((0, 1))), (1.0, 2.0)),
        ((CoordinateLevel((0, 1)), CompressedLevel((0, 1, 2), (0, 1))), (1.0, 2.0)),
    ],
    ids=["dense-coo", "coo-dense", "compressed-coo", "coo-compressed"],
)
def test_coordinate_level_cannot_mix_with_other_level_kinds(levels, values):
    with pytest.raises(LevelStorageError, match=_COORDINATE_ONLY):
        LevelTensorStorage(shape=(2, 2), modes=(0, 1), levels=levels, values=values)


def test_coordinate_storage_snapshot_detaches_and_validate_rechecks():
    storage = coo_storage((3, 4), (0, 1), [(0, 1), (0, 3), (2, 0)])
    snapshot = storage.snapshot()
    object.__setattr__(storage.levels[1], "coords", (1, 1, 0))
    with pytest.raises(LevelStorageError, match="lexicographically increasing"):
        storage.validate()
    with pytest.raises(LevelStorageError, match="lexicographically increasing"):
        storage.snapshot()
    with pytest.raises(LevelStorageError, match="lexicographically increasing"):
        storage.to_dense()
    snapshot.validate()
    assert snapshot.levels[1].coords == (1, 3, 0)


# Level 0 has runs of lengths 3, 1 and 2.  Level 1 has runs inside them, and
# positions 2 and 3 repeat a level-1 coordinate across a level-0 boundary.
_RUN_ENTRIES = [
    (0, 0, 1),
    (0, 0, 4),
    (0, 1, 0),
    (1, 1, 2),
    (3, 0, 0),
    (3, 0, 3),
]


def run_storage():
    return coo_storage((4, 3, 5), (0, 1, 2), _RUN_ENTRIES)


def test_coordinate_segments_are_the_runs_of_the_levels_above():
    storage = run_storage()
    assert storage.segment(0, 0) == (0, 6)
    assert [storage.segment(1, parent) for parent in (0, 3, 4)] == [
        (0, 3),
        (3, 4),
        (4, 6),
    ]
    # Entries 2 and 3 agree on level 1 but not on level 0, so the level-2
    # segment of entry 2 stops at the level-0 boundary.
    assert [storage.segment(2, parent) for parent in (0, 2, 3, 4)] == [
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 6),
    ]


@pytest.mark.parametrize(
    "level,parent_position,match",
    [
        (0, 1, r"outside \[0, 1\) at level 0"),
        (0, -1, r"outside \[0, 1\) at level 0"),
        (1, 1, "does not start a run"),
        (1, 2, "does not start a run"),
        (1, 5, "does not start a run"),
        (2, 1, "does not start a run"),
        (2, 5, "does not start a run"),
        (1, 6, r"outside \[0, 6\) at level 1"),
        (2, -1, r"outside \[0, 6\) at level 2"),
        (3, 0, "no level 3"),
        (1, True, "exact int"),
    ],
)
def test_coordinate_segment_rejects_a_parent_that_starts_no_run(
    level, parent_position, match
):
    with pytest.raises(LevelStorageError, match=match):
        run_storage().segment(level, parent_position)


def test_next_position_steps_over_runs_of_one_coordinate():
    storage = run_storage()
    # Level 0, inside the root segment: one step per run.
    assert storage.next_position(0, 0, 6) == 3
    assert storage.next_position(0, 3, 6) == 4
    assert storage.next_position(0, 4, 6) == 6
    # A position inside a run steps to the end of that run as well.
    assert storage.next_position(0, 1, 6) == 3
    assert storage.next_position(0, 2, 6) == 3
    assert storage.next_position(0, 5, 6) == 6
    # Level 1, inside each level-0 run.
    assert storage.next_position(1, 0, 3) == 2
    assert storage.next_position(1, 2, 3) == 3
    assert storage.next_position(1, 3, 4) == 4
    assert storage.next_position(1, 4, 6) == 6
    # The step never leaves the segment: entry 3 repeats entry 2's level-1
    # coordinate, and only the segment end separates them.
    assert storage.coordinate_at(1, 2) == storage.coordinate_at(1, 3)
    assert storage.next_position(1, 2, 3) == 3
    # The last level stores every tuple once, so it steps one position.
    for start, end in ((0, 2), (2, 3), (3, 4), (4, 6)):
        for position in range(start, end):
            assert storage.next_position(2, position, end) == position + 1


@pytest.mark.parametrize(
    "level,position,end,match",
    [
        (0, -1, 6, r"position -1 outside \[0, 6\)"),
        (0, 6, 6, r"position 6 outside \[0, 6\)"),
        (0, 2, 2, r"segment end 2 outside \(2, 6\]"),
        (0, 2, 1, r"segment end 1 outside \(2, 6\]"),
        (0, 2, 7, r"segment end 7 outside \(2, 6\]"),
        (3, 0, 1, "no level 3"),
        (-1, 0, 1, "no level -1"),
        (0, 0.0, 6, "exact int"),
        (0, 0, True, "exact int"),
    ],
)
def test_next_position_rejects_arguments_outside_the_storage(
    level, position, end, match
):
    with pytest.raises(LevelStorageError, match=match):
        run_storage().next_position(level, position, end)


def test_next_position_on_a_compressed_level_is_the_next_position():
    csr = from_csr(csr_fixture())
    assert csr.next_position(1, 0, 2) == 1
    assert csr.next_position(1, 1, 2) == 2
    assert csr.next_position(1, 2, 3) == 3
    with pytest.raises(LevelStorageError, match="no stored coordinates"):
        csr.next_position(0, 0, 1)
    with pytest.raises(LevelStorageError, match=r"segment end 4 outside \(2, 3\]"):
        csr.next_position(1, 2, 4)
    dcsr = LevelTensorStorage.from_dense(
        [[0.0, 0.0], [0.0, 0.0], [5.0, 6.0], [7.0, 0.0]],
        (4, 2),
        (0, 1),
        (COMPRESSED, COMPRESSED),
    )
    assert dcsr.next_position(0, 0, 2) == 1
    assert dcsr.next_position(0, 1, 2) == 2
    assert dcsr.next_position(1, 0, 2) == 1


def test_coordinate_at_reads_every_coordinate_level():
    storage = run_storage()
    for level in range(3):
        assert [storage.coordinate_at(level, position) for position in range(6)] == [
            entry[level] for entry in _RUN_ENTRIES
        ]
    assert [storage.leaf_value(position) for position in range(6)] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]
    with pytest.raises(LevelStorageError, match=r"position 6 outside \[0, 6\)"):
        storage.coordinate_at(1, 6)
    with pytest.raises(LevelStorageError, match=r"position -1 outside \[0, 6\)"):
        storage.coordinate_at(2, -1)
    with pytest.raises(LevelStorageError, match="no level 3"):
        storage.coordinate_at(3, 0)
    with pytest.raises(LevelStorageError, match=r"leaf position 6 outside \[0, 6\)"):
        storage.leaf_value(6)


def test_coordinate_access_fails_closed_on_forged_state():
    # Access after construction rereads the stored fields, so state forged
    # behind the validated instance cannot leak an unstructured exception.
    storage = run_storage()
    object.__setattr__(storage.levels[0], "coords", (0, 0, 0))
    with pytest.raises(LevelStorageError, match="6 coordinates for 3 stored entries"):
        storage.segment(1, 0)

    storage = run_storage()
    for level in storage.levels:
        object.__setattr__(level, "coords", level.coords[:2])
    with pytest.raises(LevelStorageError, match="no entry at parent position 4"):
        storage.segment(1, 4)
    with pytest.raises(LevelStorageError, match=r"position 4 outside \[0, 2\)"):
        storage.next_position(0, 4, 6)

    storage = run_storage()
    object.__setattr__(storage.levels[0], "coords", [0, 0, 0, 1, 3, 3])
    for access in (
        lambda: storage.segment(0, 0),
        lambda: storage.segment(1, 0),
        lambda: storage.next_position(0, 0, 6),
        lambda: storage.coordinate_at(0, 0),
    ):
        with pytest.raises(LevelStorageError, match="owned tuple"):
            access()

    storage = run_storage()
    object.__setattr__(storage.levels[0], "coords", (0, 0.0, 0, 1, 3, 3))
    with pytest.raises(LevelStorageError, match="exact int"):
        storage.segment(1, 0)
    with pytest.raises(LevelStorageError, match="exact int"):
        storage.next_position(0, 0, 6)

    storage = run_storage()
    object.__setattr__(
        storage,
        "levels",
        (DenseLevel(4),) + storage.levels[1:],
    )
    with pytest.raises(LevelStorageError, match=_COORDINATE_ONLY):
        storage.segment(1, 0)


def _layout_cases():
    for shape in ((4,), (3, 4), (2, 3, 4), (1, 1), (0, 3), (3, 0), (2, 0, 2)):
        for modes in itertools.permutations(range(len(shape))):
            yield shape, modes


@pytest.mark.parametrize(
    "shape,modes",
    list(_layout_cases()),
    ids=lambda value: "x".join(str(item) for item in value),
)
def test_coordinate_from_dense_round_trips_and_stores_the_nonzero_cells(shape, modes):
    rng = random.Random(20261012 + 31 * len(shape) + sum(modes))
    rank = len(shape)
    for density in (0.0, 0.2, 0.5, 1.0):
        dense = random_dense(rng, shape, density)
        coo = LevelTensorStorage.from_dense(dense, shape, modes, (COORDINATE,) * rank)
        assert coo.kinds == (COORDINATE,) * rank
        assert coo.modes == modes
        assert coo.to_dense() == dense
        # The stored entries are the nonzero cells in lexicographic order of
        # the storage levels, and the level interface reads back exactly
        # them, each once.
        expected = nonzero_cells(dense, shape, modes)
        assert stored_entries(coo) == expected
        assert coo.values == tuple(value for _, value in expected)
        # The all-compressed layout of the same data stores the same entries
        # and yields them in the same order through the same interface.
        compressed = LevelTensorStorage.from_dense(
            dense, shape, modes, (COMPRESSED,) * rank
        )
        assert stored_entries(compressed) == expected


def test_coordinate_from_dense_examples():
    dense = [[0.0, 2.0], [3.0, 0.0]]
    by_rows = LevelTensorStorage.from_dense(dense, (2, 2), (0, 1), (COORDINATE,) * 2)
    assert by_rows.levels == (CoordinateLevel((0, 1)), CoordinateLevel((1, 0)))
    assert by_rows.values == (2.0, 3.0)
    by_columns = LevelTensorStorage.from_dense(dense, (2, 2), (1, 0), (COORDINATE,) * 2)
    assert by_columns.levels == (CoordinateLevel((0, 1)), CoordinateLevel((1, 0)))
    assert by_columns.values == (3.0, 2.0)
    all_zero = LevelTensorStorage.from_dense(
        [[0.0, 0.0], [0.0, 0.0]], (2, 2), (0, 1), (COORDINATE,) * 2
    )
    assert all_zero.levels == (CoordinateLevel(()), CoordinateLevel(()))
    assert all_zero.values == ()
    assert all_zero.segment(0, 0) == (0, 0)
    assert all_zero.to_dense() == [[0.0, 0.0], [0.0, 0.0]]
    vector = LevelTensorStorage.from_dense(
        [0.0, 7.0, 0.0, 8.0], (4,), (0,), (COORDINATE,)
    )
    assert vector.levels == (CoordinateLevel((1, 3)),)
    assert vector.values == (7.0, 8.0)


@pytest.mark.parametrize(
    "kinds",
    [
        (COORDINATE, DENSE),
        (DENSE, COORDINATE),
        (COORDINATE, COMPRESSED),
        (COMPRESSED, COORDINATE),
    ],
)
def test_from_dense_rejects_coordinate_mixed_with_other_kinds(kinds):
    with pytest.raises(LevelStorageError, match=_COORDINATE_ONLY):
        LevelTensorStorage.from_dense([[1.0, 0.0]], (1, 2), (0, 1), kinds)


def test_from_dense_still_rejects_singleton_levels():
    for kinds in ((LevelKind.SINGLETON,), (COORDINATE, LevelKind.SINGLETON)):
        with pytest.raises(LevelStorageError, match="cannot build singleton"):
            LevelTensorStorage.from_dense(
                [[1.0]] if len(kinds) == 2 else [1.0],
                (1,) * len(kinds),
                tuple(range(len(kinds))),
                kinds,
            )


def test_level_output_builder_assembles_coordinate_levels():
    builder = LevelOutputBuilder("C", (4, 3, 5), (COORDINATE,) * 3)
    for position, entry in enumerate(_RUN_ENTRIES):
        builder.append(entry, float(position + 1))
    result = builder.finish()
    assert type(result) is LevelTensor
    assert result.shape == (4, 3, 5)
    assert result.level_kinds == (COORDINATE,) * 3
    assert result.positions == (None, None, None)
    assert result.coordinates == (
        (0, 0, 0, 1, 3, 3),
        (0, 0, 1, 1, 0, 0),
        (1, 4, 0, 2, 0, 3),
    )
    assert result.values == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    # The assembled levels are canonical COO storage of the same entries.
    storage = LevelTensorStorage(
        shape=result.shape,
        modes=(0, 1, 2),
        levels=tuple(CoordinateLevel(coords) for coords in result.coordinates),
        values=result.values,
    )
    assert storage == run_storage()

    vector = LevelOutputBuilder("v", (5,), (COORDINATE,))
    vector.append((1,), 2.0)
    # Structure follows what was appended: an explicit zero is stored.
    vector.append((4,), 0.0)
    result = vector.finish()
    assert result.positions == (None,)
    assert result.coordinates == ((1, 4),)
    assert result.values == (2.0, 0.0)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 0, 3), (0, 0)])
def test_level_output_builder_finishes_an_empty_coordinate_result(shape):
    rank = len(shape)
    result = LevelOutputBuilder("C", shape, (COORDINATE,) * rank).finish()
    assert result.positions == (None,) * rank
    assert result.coordinates == ((),) * rank
    assert result.values == ()


def test_level_output_builder_orders_coordinate_appends():
    builder = LevelOutputBuilder("C", (3, 4), (COORDINATE, COORDINATE))
    builder.append((1, 2), 1.0)
    with pytest.raises(LevelStorageError, match="lexicographically increasing"):
        builder.append((1, 2), 2.0)
    with pytest.raises(LevelStorageError, match="lexicographically increasing"):
        builder.append((1, 1), 2.0)
    with pytest.raises(LevelStorageError, match="lexicographically increasing"):
        builder.append((0, 3), 2.0)
    with pytest.raises(LevelStorageError, match="escapes extent 4 at level 1"):
        builder.append((2, 4), 2.0)
    with pytest.raises(LevelStorageError, match="2 exact ints"):
        builder.append((2,), 2.0)
    builder.append((1, 3), 3.0)
    builder.append((2, 0), 4.0)
    result = builder.finish()
    assert result.coordinates == ((1, 1, 2), (2, 3, 0))
    assert result.values == (1.0, 3.0, 4.0)


@pytest.mark.parametrize(
    "level_kinds,match",
    [
        ((COORDINATE, DENSE), "DENSE and COMPRESSED"),
        ((DENSE, COORDINATE), "DENSE and COMPRESSED"),
        ((COORDINATE, COMPRESSED), "DENSE and COMPRESSED"),
        ((COORDINATE, LevelKind.SINGLETON), "DENSE and COMPRESSED"),
        ((LevelKind.SINGLETON, LevelKind.SINGLETON), "DENSE and COMPRESSED"),
        ((DENSE, DENSE), "at least one compressed level"),
    ],
)
def test_level_output_builder_rejects_mixed_coordinate_kinds(level_kinds, match):
    with pytest.raises(LevelStorageError, match=match):
        LevelOutputBuilder("C", (2, 2), level_kinds)


def test_level_tensor_validates_coordinate_levels():
    def tensor(**changes):
        fields = dict(
            shape=(3, 4),
            level_kinds=(COORDINATE, COORDINATE),
            positions=(None, None),
            coordinates=((0, 0, 2), (1, 3, 0)),
            values=(1.0, 2.0, 3.0),
        )
        fields.update(changes)
        return LevelTensor(**fields)

    assert tensor().values == (1.0, 2.0, 3.0)
    empty = tensor(coordinates=((), ()), values=())
    assert empty.coordinates == ((), ())
    with pytest.raises(LevelStorageError, match="no positions"):
        tensor(positions=((0, 3), None))
    with pytest.raises(LevelStorageError, match="owned coordinate tuple"):
        tensor(coordinates=((0, 0, 2), None))
    with pytest.raises(LevelStorageError, match="2 coordinates for 3 stored entries"):
        tensor(coordinates=((0, 0, 2), (1, 3)))
    with pytest.raises(LevelStorageError, match="escapes level 1 extent 4"):
        tensor(coordinates=((0, 0, 2), (1, 4, 0)))
    with pytest.raises(LevelStorageError, match="exact int"):
        tensor(coordinates=((0, 0, 2.0), (1, 3, 0)))
    with pytest.raises(LevelStorageError, match="lexicographically increasing"):
        tensor(coordinates=((0, 0, 2), (3, 1, 0)))
    with pytest.raises(LevelStorageError, match="lexicographically increasing"):
        tensor(coordinates=((0, 0, 2), (1, 1, 0)))
    with pytest.raises(LevelStorageError, match="one owned value"):
        tensor(values=(1.0, 2.0))
    with pytest.raises(LevelStorageError, match=_COORDINATE_ONLY):
        tensor(
            level_kinds=(COMPRESSED, COORDINATE),
            positions=((0, 2), None),
            coordinates=((0, 2), (1, 0)),
            values=(1.0, 2.0),
        )
    with pytest.raises(LevelStorageError, match="COORDINATE kind on every level"):
        tensor(level_kinds=(COORDINATE, LevelKind.SINGLETON))
