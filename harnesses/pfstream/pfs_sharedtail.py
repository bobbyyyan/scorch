"""Find cells where the DENOMINATOR is the outlier, so they are not attributed to an arm.

WHY. On the f64 screen's round 1, higgs-twitter_retweet at k=10 showed every arm about 21x
slower than ship. That is not three arms regressing: ship read 0.2132 ms against 4.49 / 4.30 /
4.53 ms for the other three, so the shared factor is ship's own cell. A median is immune, but
a TAIL statistic is not -- the per-matrix p99 of the byte-identical floor at k=16 fell from
12.280% to 9.195% once such cells were removed, a 25% overstatement of the floor's tail.

THE DETECTOR MUST BE TWO-SIDED, and the first version was not. `min(arm/ship) > thr` can only
see ship-too-FAST. The largest deviation on that screen was the opposite case: DIMACS10/nc2010
at k=16, where all three arms read 0.10x because ship was anomalously SLOW. A one-sided test
on a two-sided scale is blind to exactly one tail, knowably the one below parity, and here that
tail held the biggest cell. Measured: at thr=1.05 the two-sided form finds 19 cells and the
one-sided form 9.

WHAT IT IS NOT. Agreement among arms does not prove the denominator is at fault -- it is
consistent with a real effect shared by every arm. What makes the denominator reading the
strong one here is that one of the arms is BYTE-IDENTICAL to the denominator, so a shared
deviation of 21x cannot be code. A caller without a byte-identical arm gets a weaker claim and
should say so.
"""


def shared_tail(by_arm, denom, thr=1.20):
    """-> {(key, k): (direction, ratios)} for cells where EVERY arm deviates the same way.

    `thr` is a ratio, applied as min(r) > thr for the denominator-too-fast side and
    max(r) < 1/thr for the denominator-too-slow side, so the two sides are symmetric in the
    multiplicative scale the ratios live on.
    """
    if thr <= 1.0:
        raise ValueError("thr must exceed 1.0; it is a ratio, not a fraction")
    arms = [a for a in by_arm if a != denom]
    if not arms:
        raise ValueError("no arms besides the denominator %r, so there is no agreement to "
                         "find and this would return empty on any input" % denom)
    out = {}
    for cell, d in by_arm[denom].items():
        if d <= 0:
            continue
        rs = []
        for a in arms:
            v = by_arm[a].get(cell)
            if v is None or v <= 0:
                rs = None
                break
            rs.append(v / d)
        if not rs:
            continue
        if min(rs) > thr:
            out[cell] = ("denominator-too-fast", tuple(rs))
        elif max(rs) < 1.0 / thr:
            out[cell] = ("denominator-too-slow", tuple(rs))
    return out


def sensitivity(by_arm, denom, thrs=(1.05, 1.10, 1.20, 1.50, 2.0)):
    """Counts across thresholds, two-sided beside one-sided.

    A bare "1 of 2926 cells" is uninformative if the threshold was unreachable elsewhere, so
    the count is partly a property of the threshold. Reporting the sweep is what makes a small
    count mean something, and pairing it with the one-sided count keeps the blind tail visible.

    HOW FAR THE THRESHOLD REACHES IS `binding`, NOT A MAXIMUM ARM RATIO. An earlier version of
    this docstring said "every width except one had a maximum below thr=2.0". That was computed
    with the one-sided form, which cannot see a cell whose binding arm is BELOW 1, and it is
    also the wrong quantity -- a maximum arm ratio says nothing about whether the cell is shared.
    Re-derived with `binding` on the f64 screen's round 1 (418 SuiteSparse keys, seven widths):
    TWO widths have a cell binding above 2.0, k=10 at 20.178x and k=16 at 8.960x, and every
    other width's largest binding is at most 1.122x.
    """
    rows = []
    for thr in thrs:
        two = shared_tail(by_arm, denom, thr)
        one = sum(1 for cell, v in two.items() if v[0] == "denominator-too-fast")
        rows.append((thr, len(two), one, sorted({k for _key, k in two})))
    return rows


def binding(rs):
    """The LARGEST thr at which a cell with these arm ratios is still excluded.

    `shared_tail` asks a yes/no question at one threshold; this reports the threshold at which
    the answer flips, which is what a sweep's shape is made of. The two are tied by a property
    the selftest checks mechanically rather than restating: for thr > 1 -- the domain
    `shared_tail` accepts -- a cell is in `shared_tail(thr)` if and only if
    `binding(rs) > thr`.

    `min(rs) > thr` needs thr < min(rs); `max(rs) < 1/thr` needs thr < 1/max(rs). A cell whose
    arms straddle 1 is shared at no thr > 1, so its binding threshold is 1 -- NOT its largest
    arm ratio. Taking a max arm ratio instead is the error this function exists to stop: it
    ignores whether the cell is shared at all, so it overstates, and it was published twice.
    """
    if not rs:
        raise ValueError("binding() needs at least one arm ratio; got none")
    if min(rs) > 1.0:
        return min(rs)
    if max(rs) < 1.0:
        return 1.0 / max(rs)
    return 1.0


def binding_thresholds(by_arm, denom):
    """{cell: binding} over every cell the denominator and all arms share."""
    cells = set(by_arm[denom])
    for a in by_arm:
        cells &= set(by_arm[a])
    arms = [a for a in by_arm if a != denom]
    out = {}
    for cell in cells:
        d = by_arm[denom][cell]
        if d <= 0:
            continue
        out[cell] = binding([by_arm[a][cell] / d for a in arms])
    return out
