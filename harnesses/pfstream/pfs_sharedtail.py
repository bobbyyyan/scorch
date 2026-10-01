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

    A bare "1 of 2926 cells" is uninformative if the threshold was unreachable elsewhere, and
    on that screen every width except one had a maximum below thr=2.0 -- so the count was
    partly a property of the threshold. Reporting the sweep is what makes a small count mean
    something, and pairing it with the one-sided count keeps the blind tail visible.
    """
    rows = []
    for thr in thrs:
        two = shared_tail(by_arm, denom, thr)
        one = sum(1 for cell, v in two.items() if v[0] == "denominator-too-fast")
        rows.append((thr, len(two), one, sorted({k for _key, k in two})))
    return rows
