"""Refuse a per-pass CSV that is still being written. ONE implementation, imported by readers.

WHY THIS EXISTS. Reading the f64 screen's round 1 the moment its fourth build appeared, the
`nopf` file held 700 of 2926 cells -- 100 matrices per width instead of 418 -- because the
pass was mid-write. The file existed, parsed cleanly, and produced plausible medians on a
quarter of the corpus against the other arms' full one. Nothing in the read said so; the
cell counts had to be printed beside the medians to notice, and a reader comparing arms
would have been comparing two different populations.

That is the partial-write case of a result indistinguishable from a complete one. An
existence check does not catch it, a parse check does not catch it, and a per-arm median is
exactly as well-formed on a quarter of the data.

THE GUARD IS A COMPARISON, NOT A THRESHOLD. A hardcoded expected row count is a constant that
rots and that cannot be written before the corpus is known. The arms of one pass are meant to
cover the same keys at the same widths, so the population is compared ACROSS arms and any arm
that differs is refused by name. That also fails in the safe direction: if every arm is
equally truncated the counts agree and this says nothing, so `require_complete` takes the
expected count when a caller knows it.
"""


class Incomplete(Exception):
    """A pass that cannot be read because its arms do not cover the same cells."""


def census(by_arm):
    """by_arm: {arm: {(key, k): value}} -> {arm: {k: n}} plus the total per arm."""
    out = {}
    for arm, cells in by_arm.items():
        per = {}
        for (_key, k) in cells:
            per[k] = per.get(k, 0) + 1
        out[arm] = per
    return out


def check(by_arm, expected_total=None):
    """Refuse unless every arm covers the same (width -> count) census.

    -> the agreed census. Raises Incomplete naming the offending arm and the shortfall, so the
    message identifies the file to wait for rather than saying the pass is bad.
    """
    if len(by_arm) < 2 and expected_total is None:
        raise Incomplete(
            "a single arm cannot be checked against its siblings and no expected total was "
            "given, so this guard would pass on any file including an empty one")
    cen = census(by_arm)
    totals = {a: sum(p.values()) for a, p in cen.items()}
    if expected_total is not None:
        short = {a: n for a, n in totals.items() if n != expected_total}
        if short:
            raise Incomplete(
                "arm(s) %s hold %s cells against an expected %d -- still being written, or "
                "written against a different corpus"
                % (", ".join(sorted(short)), sorted(short.values()), expected_total))
    if len(by_arm) < 2:
        return cen[next(iter(cen))]
    biggest = max(totals.values())
    short = {a: n for a, n in totals.items() if n != biggest}
    if short:
        raise Incomplete(
            "arm(s) %s hold %s cells against %d for the fullest arm. A median over a subset "
            "of the corpus is as well formed as one over all of it, so this would have read "
            "as a result. WAIT for those file(s) rather than reading them."
            % (", ".join("%s=%d" % (a, n) for a, n in sorted(short.items())),
               sorted(short.values()), biggest))
    ref_arm = sorted(cen)[0]
    ref = cen[ref_arm]
    for arm in sorted(cen):
        if cen[arm] != ref:
            diff = sorted(set(ref) | set(cen[arm]))
            raise Incomplete(
                "arm %s and arm %s have equal totals but different per-width censuses, so "
                "they cover different cells: %s"
                % (arm, ref_arm,
                   ", ".join("k=%d %s/%s" % (k, cen[arm].get(k, 0), ref.get(k, 0))
                             for k in diff if cen[arm].get(k, 0) != ref.get(k, 0))))
    return ref
