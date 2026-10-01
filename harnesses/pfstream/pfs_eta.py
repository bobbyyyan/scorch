"""Project a screen job's finish time from its own pass log, PRINTING ITS INPUTS.

WHY THIS FILE EXISTS. Two hand-computed completion estimates in one hour failed to
reproduce from the agreed form (one by 2 s on 13435 s, one by 3 s). Neither mattered
numerically and both mattered procedurally: a published figure that does not follow
from the stated estimator cannot be checked without a round trip, and the round trip
cannot tell arithmetic slop from a silently changed estimator. So the projection is
computed here and the report carries the per-build means and the remaining counts
that produced it. A reader can then redo the multiplication.

THE ESTIMATOR, stated once. Passes run strictly serially with no measured gap, so
  finish = (last pass start) + sum over builds of  mean(landed durations) * remaining
with `remaining = rounds - landed` per build. It is remaining-by-build rather than a
pooled mean because the builds differ systematically in duration (the arms are
different kernels), and the schedule is a balanced rotation, so which builds are
still owed is known exactly.

It REFUSES rather than extrapolating when a build has no landed pass, because then
its mean does not exist and a pooled substitute would silently change the estimator.
"""
import re
import sys

LINE = re.compile(r"---\s+p(\d+)\s+(\S+)\s+(start|done)\s+(\d\d):(\d\d):(\d\d)")


def _secs(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse(path):
    """-> (events, rounds) where events is [(pass, build, kind, seconds-of-day)]."""
    ev = []
    for ln in open(path):
        m = LINE.search(ln)
        if m:
            ev.append((int(m.group(1)), m.group(2), m.group(3),
                       _secs(m.group(4), m.group(5), m.group(6))))
    return ev


def durations(ev):
    """Landed durations per build, plus the open pass. Pairs a done with its start."""
    open_start = {}
    per_build = {}
    running = None
    for p, b, kind, t in ev:
        if kind == "start":
            open_start[(p, b)] = t
            running = (p, b, t)
        else:
            st = open_start.pop((p, b), None)
            if st is None:
                continue
            d = t - st
            if d < 0:
                d += 86400          # the log carries no date; a run can cross midnight
            per_build.setdefault(b, []).append(d)
            if running == (p, b, st):
                running = None
    return per_build, running


def project(path, rounds, now=None):
    ev = parse(path)
    if not ev:
        return ["REFUSE: no pass lines matched in %s" % path]
    per, running = durations(ev)
    builds = sorted(set(b for _p, b, _k, _t in ev))
    missing = [b for b in builds if not per.get(b)]
    if missing:
        return ["REFUSE: no landed pass for build(s) %s, so their mean does not exist; "
                "a pooled substitute would be a different estimator" % ",".join(missing)]
    landed = sum(len(v) for v in per.values())
    total = rounds * len(builds)
    out = ["  estimator: finish = last-start + SUM_builds mean(landed) * remaining",
           "  landed %d of %d passes over %d builds, %d rounds"
           % (landed, total, len(builds), rounds)]
    rem_s = 0.0
    for b in builds:
        v = per[b]
        mean = sum(v) / len(v)
        rem = rounds - len(v)
        if rem < 0:
            return ["REFUSE: build %s has %d landed passes but only %d rounds were "
                    "declared -- the rotation or the round count is wrong" % (b, len(v), rounds)]
        rem_s += mean * rem
        out.append("    %-5s landed %d  durations %-22s mean %8.3f  remaining %d"
                   % (b, len(v), ",".join(str(x) for x in v), mean, rem))
    out.append("  remaining seconds = %.3f over %d passes" % (rem_s, total - landed))
    if running is None:
        base = max(t for _p, _b, _k, t in ev)
        out.append("  NOTE: no pass is open; projecting from the last event %s" % hms(base))
    else:
        base = running[2]
        out.append("  open pass p%d %s started %s" % (running[0], running[1], hms(base)))
    fin = base + rem_s
    out.append("  PROJECTED FINISH %s  (+%s from the open pass's start)"
               % (hms(fin), hms(rem_s)))
    if now is not None:
        out.append("  now %s, so %s to go" % (hms(now), hms(max(0.0, fin - now))))
    return out


def hms(t):
    t = int(round(t)) % 86400
    return "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: pfs_eta.py <pass-log> <rounds> [HH:MM:SS now]")
    nw = None
    if len(sys.argv) > 3:
        hh, mm, ss = sys.argv[3].split(":")
        nw = _secs(hh, mm, ss)
    print("\n".join(project(sys.argv[1], int(sys.argv[2]), nw)))
