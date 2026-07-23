"""Format-neutral level-storage coverage for the production LoopIR oracle.

The execution core consumes only ``segment`` / ``coordinate_at`` /
``leaf_value`` on a validated :class:`LevelTensorStorage`; CSR is exactly
one adapter.  These tests lock the canonical-storage contract, the
CSR container/adapter boundaries, and the order-checked output builder.
"""

import pytest

from scorch.compiler.loopir.levels import (
    CompressedLevel,
    CsrFormatError,
    CsrMatrix,
    CsrOutputBuilder,
    DenseLevel,
    LevelStorageError,
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
