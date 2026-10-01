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

TWO INPUT SOURCES, because the two live runs record their passes differently and the
difference is not cosmetic. A pass LOG carries each pass's start and done, so a
duration is measured directly and an inter-pass gap is visible. Per-pass output file
MTIMES carry only COMPLETIONS, so a duration is a difference between consecutive
completions -- which silently assumes the gap is zero, gives the first pass no
measurable duration at all (it has no predecessor), and makes the open pass's "start"
the previous pass's completion. Those are different quantities that coincide only
while the gap is zero, so the mtime mode says which one it used and never labels a
completion as a start.

The mtime mode's own trap: the open pass's file EXISTS and its mtime is moving, so
including it would attribute a near-zero duration to a pass that has not finished.
It is identified by being both newest and smaller than every landed file, and if that
test does not separate it the mode refuses rather than guessing.
"""
import os
import re
import sys
import time

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
    if now is not None and rem_s >= 86400:
        return ["REFUSE: %.0f s remain, which is a day or more. `now` is a bare clock "
                "with no date, so the elapsed time on the open pass cannot be "
                "distinguished from itself plus a multiple of 24 h" % rem_s]
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
        out.append(_to_go_line(rem_s, _elapsed_since(base, now), now))
    return out


def hms(t):
    t = int(round(t)) % 86400
    return "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)


def _elapsed_since(base_sod, now_sod):
    """Seconds from `base` to `now`, both seconds-of-day, assuming under 24 h apart.

    Neither input carries a date: the log's timestamps are bare wall clocks and `now` is
    given as HH:MM:SS. A negative difference is therefore a midnight crossing, not a
    `now` in the past -- justified because the quantity measured is the age of the OPEN
    pass, which is minutes to hours, never days.
    """
    d = now_sod - base_sod
    if d < 0:
        d += 86400
    return d


def _to_go_line(rem_s, elapsed, now_sod):
    """The remainder as a SIGNED quantity, because a deficit must not read as zero.

    Both modes previously computed this from wrapped clock values and clamped at zero:
      * the mtimes mode wrapped `finish` to seconds-of-day BEFORE subtracting `now`, so a
        projection past midnight produced a large negative that clamped -- a two-hour
        remainder printed as `00:00:00 to go`;
      * both modes clamped an OVERDUE run to `00:00:00`, so a run an hour late read as
        about to land rather than as possibly dead. That is the reading that matters when
        a watcher is the primary wake signal.
    The fix is to stop deriving it from clocks at all. `remaining - elapsed` is wrap-free
    because `elapsed` is, and its sign is the information the clamp was destroying.
    """
    left = rem_s - elapsed
    if left >= 0:
        return "  now %s, %s elapsed on the open pass, so %s to go" % (
            hms(now_sod), hms(elapsed), hms(left))
    return ("  now %s, %s elapsed on the open pass -- OVERDUE by %s against a %s "
            "projection. The run may have stalled or died; the projection is not a "
            "deadline" % (hms(now_sod), hms(elapsed), hms(-left), hms(rem_s)))


def read_mtimes(d, pattern):
    """-> [(mtime, size, build, pass)] from filenames matching `pattern`.

    `pattern` must name groups `build` and `pass`. A per-pass output file's mtime is
    the pass's COMPLETION, never its start -- the caller must not conflate them.
    """
    rx = re.compile(pattern)
    out = []
    for f in sorted(os.listdir(d)):
        m = rx.match(f)
        if not m:
            continue
        st = os.stat(os.path.join(d, f))
        out.append((st.st_mtime, st.st_size, m.group("build"), int(m.group("pass")), f))
    out.sort()
    return out


def project_mtimes(d, pattern, rounds, now=None):
    """Projection from per-pass completion mtimes. Prints the same inputs as the log mode."""
    fs = read_mtimes(d, pattern)
    if len(fs) < 3:
        return ["REFUSE: %d file(s) matched in %s -- a difference-of-completions "
                "estimator needs at least two landed passes plus an open one" % (len(fs), d)]
    sizes = [x[1] for x in fs]
    openf = None
    if sizes[-1] < min(sizes[:-1]):
        openf = fs[-1]
        landed = fs[:-1]
    else:
        return ["REFUSE: the newest file %s is %d bytes and not smaller than every "
                "landed file (min %d), so the open pass cannot be identified by size; "
                "its mtime is a moving target and including it would attribute a "
                "near-zero duration to an unfinished pass"
                % (fs[-1][4], sizes[-1], min(sizes[:-1]))]
    builds = sorted(set(x[2] for x in fs))
    per = {}
    for i in range(1, len(landed)):
        per.setdefault(landed[i][2], []).append(landed[i][0] - landed[i - 1][0])
    nolast = [b for b in builds if not per.get(b)]
    if nolast:
        return ["REFUSE: build(s) %s have no pass with a measurable duration. A "
                "completion-difference needs a predecessor, so the earliest file "
                "contributes none" % ",".join(nolast)]
    n_landed = len(landed) + 1          # the open pass counts against its build's rounds
    out = ["  estimator: finish = open pass's start + SUM_builds mean(landed) * remaining",
           "  input: per-pass COMPLETION mtimes, so each duration is a difference "
           "between consecutive completions and assumes a zero inter-pass gap",
           "  landed %d of %d passes over %d builds, %d rounds; %s is open (%d bytes "
           "against a landed min of %d)"
           % (len(landed), rounds * len(builds), len(builds), rounds, openf[4],
              openf[1], min(sizes[:-1])),
           "  %s has no measurable duration: it is the earliest file and has no "
           "predecessor" % landed[0][4]]
    rem_s = 0.0
    rem_n = 0
    for b in builds:
        v = per[b]
        mean = sum(v) / len(v)
        # `done` counts this build's FILES; the open pass has a file but is not landed,
        # so it is still owed -- and only by the build that owns it. Adding 1 for every
        # build (the first version of this) overstated the remainder by one pass per
        # build, which the per-build/total cross-check below now catches.
        done = sum(1 for x in fs if x[2] == b)
        rem = rounds - done + (1 if b == openf[2] else 0)
        if rem < 0:
            return ["REFUSE: build %s already has %d files but only %d rounds were "
                    "declared" % (b, done, rounds)]
        rem_s += mean * rem
        rem_n += rem
        out.append("    %-5s durations %-34s mean %9.3f  remaining %d"
                   % (b, ",".join("%.1f" % x for x in v), mean, rem))
    # NOT a cross-check against the global count. sum_b (rounds - done_b) plus one for
    # the open build is rounds*B - len(fs) + 1 = rounds*B - len(landed) ALGEBRAICALLY, so
    # that equality holds whatever the per-build attribution is and would have passed the
    # off-by-one this file shipped with. It is the partition-identity trap: making the
    # total an identity lets a misrouted item still sum correctly.
    #
    # What IS evidence is a property of the SCHEDULE rather than of the arithmetic: the
    # rotation is balanced, so at any instant two builds' file counts differ by at most
    # one. A build running ahead means the rotation broke, and then every per-build mean
    # is conditioned on a different stretch of the run.
    counts = {b: sum(1 for x in fs if x[2] == b) for b in builds}
    if max(counts.values()) - min(counts.values()) > 1:
        return ["REFUSE: the rotation is not balanced -- file counts %s differ by %d. "
                "The schedule is meant to hold them within one, so each build's mean "
                "would be conditioned on a different stretch of the run"
                % (dict(sorted(counts.items())),
                   max(counts.values()) - min(counts.values()))]
    base = landed[-1][0]
    fin = base + rem_s
    if now is not None and rem_s >= 86400:
        return ["REFUSE: %.0f s remain, which is a day or more. `now` is a bare clock "
                "with no date, so the elapsed time on the open pass cannot be "
                "distinguished from itself plus a multiple of 24 h" % rem_s]
    out.append("  remaining seconds = %.3f over %d passes" % (rem_s, rem_n))
    # LOCAL time, not `epoch % 86400`. st_mtime is epoch UTC and the log mode reads a
    # local wall clock out of the log text, so a UTC rendering here put the two modes
    # 7 hours apart on the same kind of run -- a disagreement no plant of mine covered
    # because every plant built its own mtimes and compared them to each other.
    out.append("  the open pass's start is taken as %s's COMPLETION, %s local"
               % (landed[-1][4], local_hms(base)))
    out.append("  PROJECTED FINISH %s local  (+%s)" % (local_hms(fin), hms(rem_s)))
    if now is not None:
        out.append(_to_go_line(rem_s, _elapsed_since(_secs_of_day(base), now), now))
    return out


def _secs_of_day(epoch):
    t = time.localtime(epoch)
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec


def local_hms(epoch):
    return time.strftime("%H:%M:%S", time.localtime(epoch))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--mtimes":
        if len(a) < 4:
            sys.exit("usage: pfs_eta.py --mtimes <dir> <regex with (?P<build>)(?P<pass>)> "
                     "<rounds> [HH:MM:SS now]")
        # `a` is ["--mtimes", dir, regex, rounds, now], so `now` is a[4] and len(a) is 5.
        # This read a[5] behind a `len(a) > 5` gate -- both off by one in the SAME
        # direction, so the gate that would have raised the IndexError was itself never
        # true and the documented 4-argument form silently dropped `now`. Repairing only
        # one half converts a silent no-op into a crash or leaves it silent.
        nw = None
        if len(a) > 4:
            hh, mm, ss = a[4].split(":")
            nw = _secs(hh, mm, ss)
        print("\n".join(project_mtimes(a[1], a[2], int(a[3]), nw)))
    else:
        if len(a) < 2:
            sys.exit("usage: pfs_eta.py <pass-log> <rounds> [HH:MM:SS now]\n"
                     "       pfs_eta.py --mtimes <dir> <regex> <rounds> [HH:MM:SS now]")
        nw = None
        if len(a) > 2:
            hh, mm, ss = a[2].split(":")
            nw = _secs(hh, mm, ss)
        print("\n".join(project(a[0], int(a[1]), nw)))
