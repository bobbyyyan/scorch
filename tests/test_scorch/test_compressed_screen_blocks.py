"""The compressed-level screen (csrc/native_abi.h, abi_screen_compressed_typed) against a
reference written from its contract, on structures that put descents, empty spans, coordinates
out of range and decreasing positions at the edges of its blocks (16 KiB of coordinates) and of
its workers' ranges, under several thread counts.

The screen checks positions with one split scan, then each worker of the coordinate scan finds
its first span boundary by binary search and walks the boundaries block by block. Every way that
can go wrong -- a boundary counted twice or not at all, a range that starts on a run of empty
spans, a descent on the last element of a block or a range -- shows up as a verdict that differs
from the reference at one of these positions.

The flat loop counts a block's non-increases as its length less its increases, one packed
compare on AVX2 rather than two (patch 0077); a coordinate equal to its predecessor is the case
that separates the two counts, so it is placed inside spans (a defect) and across a boundary
(legal) at every edge, and a population of random levels with random defects is compared with
the reference as well.
"""

import numpy as np
import pytest
import torch

scorch_ops = pytest.importorskip("scorch_ops")

# Wide enough that the longest span (a worker range in the empty-span test, 12289 coordinates at
# int32) can hold strictly increasing coordinates.
LIMIT = 1 << 15
BLOCK_BYTES = 16384


def reference(pos, crd, limit, require_sorted):
    """True when the level is malformed: a span outside [0, nnz] or running backwards, a
    coordinate outside [0, limit), or (when required) a coordinate inside a span that does not
    exceed the one before it: within a span coordinates must be strictly increasing, as torch's
    sparse CSR invariant requires ("sorted and distinct")."""
    nnz = len(crd)
    starts, stops = pos[:-1], pos[1:]
    if ((starts < 0) | (stops < starts) | (stops > nnz)).any():
        return True
    if nnz == 0:
        return False
    if ((crd < 0) | (crd >= limit)).any():
        return True
    if not require_sorted:
        return False
    descents = np.nonzero(crd[1:] <= crd[:-1])[0] + 1
    return bool(np.setdiff1d(descents, pos[1:-1]).size)


def strictly_increasing(rng, n):
    """n coordinates in [0, LIMIT), strictly increasing: cumulative gaps of 1..LIMIT // n."""
    return np.cumsum(rng.integers(1, LIMIT // n + 1, n)) - 1


def structure(rng, nnz, empty_fraction):
    """Positions and per-span strictly increasing coordinates: spans of 1-6, empty ones, and a
    few long enough to cross several blocks."""
    lengths, total = [], 0
    while total < nnz:
        u = rng.random()
        if u < empty_fraction:
            n = 0
        elif u < empty_fraction + 0.03:
            n = int(rng.integers(500, 9000))
        else:
            n = int(rng.integers(1, 7))
        n = min(n, nnz - total)
        lengths.append(n)
        total += n
        if rng.random() < 0.02:  # runs of empty spans
            lengths.extend([0] * int(rng.integers(2, 6)))
    pos = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    crd = np.concatenate(
        [strictly_increasing(rng, n) for n in lengths if n] or [np.zeros(0)]
    ).astype(np.int64)
    return pos, crd


def edges(nnz, parents, threads, itemsize):
    """Coordinate positions next to a worker range's start or a block's start, and parent
    indices next to a worker range's start in the positions scan."""
    block = BLOCK_BYTES // itemsize
    chunk = -(-(nnz - 1) // threads)
    at = set()
    for t in range(threads):
        b = 1 + t * chunk
        for j in range(0, chunk // block + 2):
            for d in (-1, 0, 1):
                at.add(b + j * block + d)
    coords = sorted(p for p in at if 1 <= p < nnz)
    pchunk = -(-parents // threads)
    parent_edges = sorted(
        i
        for t in range(threads)
        for d in (-1, 0, 1)
        if 1 <= (i := t * pchunk + d) < parents
    )
    return coords, parent_edges


def boundaries(pos, nnz):
    """Distinct span boundaries p with 0 < p < nnz whose spans on both sides are non-empty."""
    out = []
    for p in np.unique(pos[1:-1]):
        if 0 < p < nnz:
            out.append(int(p))
    return np.array(out, dtype=np.int64)


def inside(p, bset, nnz):
    """The nearest coordinate position at or after p that does not start a span."""
    while p < nnz and p in bset:
        p += 1
    return p if p < nnz else None


def inject(kind, pos, crd, p, rng):
    pos, crd = pos.copy(), crd.copy()
    nnz = len(crd)
    if kind == "descent_inside":
        q = inside(p, set(pos[1:-1].tolist()), nnz)
        if q is None:
            return None
        if crd[q - 1] > 0:
            crd[q] = crd[q - 1] - 1
        else:
            crd[q - 1], crd[q] = LIMIT - 1, 0
        return pos, crd
    if (
        kind == "repeat_inside"
    ):  # a coordinate equal to the one before it, inside a span
        q = inside(p, set(pos[1:-1].tolist()), nnz)
        if q is None:
            return None
        crd[q] = crd[q - 1]
        return pos, crd
    if (
        kind == "repeat_at_boundary"
    ):  # legal: a span may start at the last value before it
        starts = set(pos[:-1].tolist())
        for q in sorted(boundaries(pos, nnz).tolist(), key=lambda x: abs(x - p)):
            # Raise the earlier span's last value to the later span's first, or lower the
            # later span's first to the earlier span's last, whichever keeps both spans
            # strictly increasing.
            if q - 1 in starts or crd[q - 2] < crd[q]:
                crd[q - 1] = crd[q]
                return pos, crd
            if q + 1 == nnz or q + 1 in starts or crd[q + 1] > crd[q - 1]:
                crd[q] = crd[q - 1]
                return pos, crd
        return None
    if kind == "descent_at_boundary":  # legal
        b = boundaries(pos, nnz)
        if not b.size:
            return None
        q = int(b[np.argmin(np.abs(b - p))])
        crd[q - 1], crd[q] = LIMIT - 1, 0
        # keep both spans sorted: the earlier span ends at its largest, the later starts at 0
        return pos, crd
    if kind == "above_limit":
        crd[p] = LIMIT
        return pos, crd
    if kind == "negative":
        crd[p] = -1
        return pos, crd
    raise AssertionError(kind)


def screen(pos, crd, require_sorted, grain, dtype):
    scorch_ops.abi_memo_clear()
    return scorch_ops.abi_screen_compressed_level(
        torch.from_numpy(pos.copy()).to(dtype),
        torch.from_numpy(crd.copy()).to(dtype),
        LIMIT,
        require_sorted,
        grain,
    )


@pytest.fixture
def threads():
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("nthreads", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "kind",
    [
        "clean",
        "descent_inside",
        "repeat_inside",
        "repeat_at_boundary",
        "descent_at_boundary",
        "above_limit",
        "negative",
    ],
)
@pytest.mark.parametrize("empty_fraction", [0.0, 0.4])
def test_coordinate_defects_at_block_and_range_edges(
    threads, dtype, nthreads, kind, empty_fraction
):
    torch.set_num_threads(nthreads)
    itemsize = torch.empty(0, dtype=dtype).element_size()
    rng = np.random.default_rng(
        [nthreads, itemsize, int(empty_fraction * 10), len(kind)]
    )
    nnz = 2 * nthreads * (BLOCK_BYTES // itemsize) + 777
    pos, crd = structure(rng, nnz, empty_fraction)
    coords, _ = edges(nnz, len(pos) - 1, nthreads, itemsize)
    targets = [None] if kind == "clean" else coords
    checked = 0
    for p in targets:
        case = (pos, crd) if p is None else inject(kind, pos, crd, p, rng)
        if case is None:
            continue
        cp, cc = case
        for require_sorted in (True, False):
            want = reference(cp, cc, LIMIT, require_sorted)
            if kind in ("above_limit", "negative") or (
                kind in ("descent_inside", "repeat_inside") and require_sorted
            ):
                assert want, (kind, p)  # the injection made a defect
            if kind in ("clean", "descent_at_boundary", "repeat_at_boundary"):
                assert not want, (kind, p)  # and these made none
            for grain in (1, 1 << 40):
                got = screen(cp, cc, require_sorted, grain, dtype)
                assert got == want, (kind, p, require_sorted, grain)
        checked += 1
    assert checked >= (1 if kind == "clean" else 6)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("nthreads", [2, 3, 5])
def test_decreasing_positions_at_range_edges(threads, dtype, nthreads):
    torch.set_num_threads(nthreads)
    rng = np.random.default_rng(nthreads)
    pos, crd = structure(rng, 20000, 0.3)
    _, parent_edges = edges(len(crd), len(pos) - 1, nthreads, 4)
    assert len(parent_edges) >= 3
    for i in parent_edges:
        for value in (pos[i - 1] - 1, len(crd) + 1):
            if value < 0:
                continue
            bad = pos.copy()
            bad[i] = value
            assert reference(bad, crd, LIMIT, True)
            for grain in (1, 1 << 40):
                assert screen(bad, crd, True, grain, dtype) is True, (i, value, grain)
                assert screen(bad, crd, False, grain, dtype) is True, (i, value, grain)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("nthreads", [2, 4])
def test_a_range_that_starts_on_a_run_of_empty_spans(threads, dtype, nthreads):
    """Every worker range begins at a boundary repeated by empty spans, with a legal descent
    there: counted once it is clean, and a descent one past it is reported."""
    torch.set_num_threads(nthreads)
    itemsize = torch.empty(0, dtype=dtype).element_size()
    nnz = 3 * nthreads * (BLOCK_BYTES // itemsize) + 1
    chunk = -(-(nnz - 1) // nthreads)
    cuts = [1 + t * chunk for t in range(1, nthreads)]
    pos = [0]
    for c in cuts + [nnz]:
        pos.extend([c, c, c])  # the span ending at c, then two empty spans
    pos = np.array(pos[:-2], dtype=np.int64)  # the last span ends at nnz once
    crd = np.zeros(nnz, dtype=np.int64)
    starts = [0] + cuts
    stops = cuts + [nnz]
    for s, e in zip(starts, stops):
        crd[s:e] = strictly_increasing(np.random.default_rng(s), e - s)
    for c in cuts:
        crd[c - 1], crd[c] = LIMIT - 1, 0
    assert not reference(pos, crd, LIMIT, True)
    assert screen(pos, crd, True, 1, dtype) is False
    for c in cuts:
        bad = crd.copy()
        bad[c + 1] = LIMIT - 1
        bad[c + 2] = 0
        assert reference(pos, bad, LIMIT, True)
        assert screen(pos, bad, True, 1, dtype) is True, c


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("nthreads", [1, 3])
def test_random_levels_match_the_reference(threads, dtype, nthreads):
    """Random levels (spans of every length, empty spans, sizes around a block and a worker
    range) with up to three random defects or legal boundary values each, at random positions
    and at block edges: the verdict is the reference's, sorted or not, for both grains.
    """
    torch.set_num_threads(nthreads)
    itemsize = torch.empty(0, dtype=dtype).element_size()
    block = BLOCK_BYTES // itemsize
    rng = np.random.default_rng([2077, nthreads, itemsize])
    kinds = [
        "descent_inside",
        "repeat_inside",
        "repeat_at_boundary",
        "descent_at_boundary",
        "above_limit",
        "negative",
    ]
    verdicts = {True: 0, False: 0}
    for case in range(160):
        nnz = int(
            rng.choice([1, 2, 7, block - 1, block, block + 1, 2 * block + 3, 3000])
        )
        pos, crd = structure(rng, nnz, float(rng.choice([0.0, 0.2, 0.6])))
        for _ in range(int(rng.integers(0, 4))):
            at = [b * block + d for b in range(1, nnz // block + 1) for d in (-1, 0, 1)]
            p = (
                int(rng.choice(at))
                if at and rng.random() < 0.5
                else int(rng.integers(0, nnz))
            )
            p = min(max(p, 1 if nnz > 1 else 0), nnz - 1)
            changed = inject(str(rng.choice(kinds)), pos, crd, p, rng)
            if changed is not None:
                pos, crd = changed
        for require_sorted in (True, False):
            want = reference(pos, crd, LIMIT, require_sorted)
            verdicts[want] += 1
            for grain in (1, 1 << 40):
                assert screen(pos, crd, require_sorted, grain, dtype) == want, (
                    case,
                    require_sorted,
                    grain,
                )
    assert verdicts[True] >= 40 and verdicts[False] >= 40, verdicts
