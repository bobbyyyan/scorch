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
# A quarantined artefact (chain194's hv_ex4b_float64_p2.contaminated.csv) is sealed
# evidence that this file's glob admits. LC refuses rather than letting it join a
# recount silently, or be dropped silently, which are both decisions for a reader.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ledger_csvs as LC

LINE = re.compile(r"---\s+p(\d+)\s+(\S+)\s+(start|done)\s+(\d\d):(\d\d):(\d\d)")


class _Refuse(Exception):
    """A refusal, RAISED rather than returned.

    WHY A RAISE AND NOT A RETURN VALUE. Six findings in a row were all the same
    question: "is this return an error exit?" -- and it is not statically answerable for
    a returned value, because an error message and a report line have the same type. The
    predicates kept missing paths for different reasons (wrong wording, wrong expression
    kind, wrong wrapper, a helper returning a bare string with no marker at all), and
    each fix narrowed the premise instead of removing it.

    Raising removes the question. An abnormal termination is a raise; a returned string
    is a line of the report; there is no third category to classify. So the coverage
    population -- the call sites of `_refuse`, which is the only thing that raises this --
    is complete over abnormal terminations BY CONSTRUCTION rather than by a convention
    somebody has to keep. A helper that returns "refusing to project" is then simply
    emitting a badly worded LINE, which is visible in the report and not a silent exit.
    """


def _report(lines):
    """Guard the projectors' return type at the boundary where confusing it is silent.

    `_refuse` once had two call conventions by context -- projectors returning
    `[_refuse(...)]` and helpers returning `_refuse(...)` -- and the helper form used
    inside a projector produced 43 lines of ONE CHARACTER EACH at exit status 0, because
    `"\n".join(a_string)` iterates its characters. Every static check passed it: the site
    was a genuine `_refuse` call, the literal was inside a call, and nothing looked at the
    wrapper. `_refuse` now RAISES, so that class of confusion cannot recur -- this guard
    stays because the type is still the thing that goes wrong silently, and a guard is
    cheaper than the argument that it can no longer happen.
    """
    if isinstance(lines, str):
        raise TypeError(
            "a projector returned a bare string, not a list of lines: %r. Joining it "
            "would emit one line per character." % (lines[:60],))
    # A REPORT INVARIANT, because a projector could still return a list whose content is
    # an error message -- `return ["cannot project against 99 rounds"]` is a structurally
    # normal return and would have printed as a one-line report at exit status 0. Every
    # real report states its estimator and then either projects or says why it cannot.
    body = "\n".join(lines)
    if "estimator:" not in body:
        raise TypeError(
            "a projector returned a report with no `estimator:` line: %r. Either the "
            "report is malformed, or this is an error exit returning a value instead of "
            "raising -- refusals must call _refuse(), which raises _Refuse." % (body[:120],))
    if "PROJECTED FINISH" not in body and "NOTE: no pass is open" not in body:
        raise TypeError(
            "a projector returned a report that neither projects nor says why it cannot: "
            "%r" % (body[:160],))
    return body


def _refuse(msg, *args):
    """The ONE producer of a refusal string. Every error exit in this module calls it.

    WHY A SINK RATHER THAN A SHAPE. Two coverage predicates in a row had a premise this
    file could fall through, each found by a planted exit that the predicate could not
    see. A LEXICAL one -- a string constant beginning "REFUSE:" -- missed an exit spelled
    differently, and missed this module's own helper refusal, whose literal began with
    two spaces. A STRUCTURAL one -- a `return` of a list display whose elements are
    Constant or BinOp -- missed an f-string element, and would miss `"...".format(x)` (a
    Call) or a message bound to a name first. Lexical to structural narrowed the premise
    rather than removing it, and an enumeration of expression kinds cannot be completed.

    A named sink has no premise. The coverage population is the set of CALL SITES of this
    function: exact regardless of how the message is built, what the caller returns, or
    how the literal is spelled. And the convention is ENFORCED rather than assumed -- the
    self-test fails if any "REFUSE" literal appears outside a call to this function,
    which is the check that was missing when the lexical predicate silently dropped a
    live path out of its own denominator.
    """
    raise _Refuse(_refuse_text(msg, *args))


def _refuse_text(msg, *args):
    """The ONE place the "REFUSE: " prefix is produced.

    Split out from `_refuse` because the CLI's handler also has to RENDER a refusal --
    specifically a message-less one -- and a second literal carrying the prefix would be
    a second producer, which is exactly what the enforcement check is for. With one
    producer the check needs no exclusion list: the prefix may appear only inside this
    function, and nothing else has to be remembered.
    """
    return "REFUSE: " + (msg % args if args else msg)


def _surface(exc):
    """The text a raised refusal should show the user. Separated from the handler so it
    can be exercised directly -- the handler's own branch was otherwise reachable only by
    planting a defect in production code."""
    m = str(exc)
    if m.strip():
        return m
    # `raise _Refuse` with no arguments bypasses _refuse and carries nothing. Fail-closed
    # either way (status 1), so this is a DIAGNOSIS gap rather than a wrong answer -- but
    # relaying an empty line tells the user nothing, so name the defect instead.
    return _refuse_text("a refusal was raised with no message -- some path raised "
                        "_Refuse directly instead of calling _refuse(). Fail-closed, "
                        "but the reason is lost.")


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
        _refuse("no pass lines matched in %s", path)
    per, running = durations(ev)
    builds = sorted(set(b for _p, b, _k, _t in ev))
    missing = [b for b in builds if not per.get(b)]
    if missing:
        _refuse("no landed pass for build(s) %s, so their mean does not exist; "
                "a pooled substitute would be a different estimator", ",".join(missing))
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
            _refuse("build %s has %d landed passes but only %d rounds were "
                    "declared -- the rotation or the round count is wrong", b, len(v), rounds)
        rem_s += mean * rem
        out.append("    %-5s landed %d  durations %-22s mean %8.3f  remaining %d"
                   % (b, len(v), ",".join(str(x) for x in v), mean, rem))
    if now is not None and rem_s >= 86400:
        _refuse("%.0f s remain, which is a day or more. The log's timestamps "
                "are bare wall clocks, so the elapsed time on the open pass cannot be "
                "distinguished from itself plus a multiple of 24 h", rem_s)
    out.append("  remaining seconds = %.3f over %d passes" % (rem_s, total - landed))
    if running is None:
        base = max(t for _p, _b, _k, t in ev)
        out.append("  NOTE: no pass is open; projecting from the last event %s" % hms(base))
    else:
        base = running[2]
        out.append("  open pass p%d %s started %s" % (running[0], running[1], hms(base)))
    fin = base + rem_s
    out.append("  PROJECTED FINISH %s  (+%s from the open pass's start)"
               % (hms(fin), dur(rem_s)))
    if now is not None:
        # Always a lower bound here, with or without a real clock: the LOG's timestamps
        # are bare wall clocks, so `base` has no date to anchor against. That is a
        # property of the input, unlike the mtime mode where the date was discarded.
        out.append(_to_go_line(rem_s, _elapsed_since(base, now), now, exact=False))
    return out


def hms(t):
    """A TIME OF DAY. Wraps at 24 h, which is correct for a clock and wrong for a span."""
    t = int(round(t)) % 86400
    return "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)


def dur(t):
    """A SPAN. Never wraps, and names the days when there are any.

    This exists because `hms` took the modulus internally and every duration in the
    report went through it. The arithmetic for a 25-hour-old stalled pass was already
    exact once the mtime's date stopped being discarded -- and the formatter still
    printed "01:00:00 elapsed", because 90000 mod 86400 is 3600. A correct measurement
    rendered through a wrapping formatter is indistinguishable from the wrong one, so
    fixing the wrap in `_elapsed_since` alone changed nothing the reader could see.
    """
    neg = "-" if t < 0 else ""
    t = int(round(abs(t)))
    d, t = divmod(t, 86400)
    base = "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)
    return "%s%s%s" % (neg, ("%dd " % d) if d else "", base)


def _elapsed_since(base_sod, now_sod):
    """Seconds from `base` to `now`, both seconds-of-day. Known only MODULO 24 h.

    Neither input carries a date, so this cannot tell a 1-hour-old pass from a
    25-hour-old one. The earlier version of this docstring justified the wrap by saying
    the open pass is "minutes to hours, never days" -- which is exactly backwards for
    the branch that needs it: the OVERDUE report exists for the run that has stopped
    advancing, and a stalled run's age is unbounded while its projected remainder stays
    at whatever it was when it stopped. So the guard on `rem_s` (a day or more remaining)
    is on the operand that stays small and is absent from the one that grows. Callers
    must treat this as a LOWER BOUND; `_elapsed_exact` is the unambiguous route.
    """
    d = now_sod - base_sod
    if d < 0:
        d += 86400
    return d


def _elapsed_exact(base_epoch, now_epoch):
    """Seconds from base to now when BOTH carry a date. No wrap, no assumption."""
    return now_epoch - base_epoch


def _to_go_line(rem_s, elapsed, now_sod, exact=True):
    """The remainder as a SIGNED quantity, because a deficit must not read as zero.

    Both modes previously computed this from wrapped clock values and clamped at zero:
      * the mtimes mode wrapped `finish` to seconds-of-day BEFORE subtracting `now`, so a
        projection past midnight produced a large negative that clamped -- a two-hour
        remainder printed as `00:00:00 to go`;
      * both modes clamped an OVERDUE run to `00:00:00`, so a run an hour late read as
        about to land rather than as possibly dead. That is the reading that matters when
        a watcher is the primary wake signal.
    The fix is to stop deriving it from clocks at all: `remaining - elapsed` carries the
    sign the clamp was destroying.

    A NEGATIVE `elapsed` is refused rather than rendered. It means the open pass's start
    is in the future relative to `now`, which no live run can produce -- clock skew
    between hosts, a stale `now`, or a fixture built ahead of the real clock. It was
    found that last way, printing "-09:47:47 elapsed ... so 14:17:47 to go": a plausible
    remainder derived from an impossible state, which is worse than an error.

    `exact` says whether `elapsed` is a measurement or a lower bound. When both endpoints
    carry a date it is exact and a stall of any length reads correctly. When either is a
    bare wall clock it is known only modulo 24 h, and the line says "at least" -- because
    a 25-hour stall and a 1-hour stall produced IDENTICAL output here, to the second, and
    nothing in the figure revealed which one it was.
    """
    if elapsed < 0:
        _refuse("the open pass's start is %s AFTER the given `now` of %s. No "
                "live run can be in that state -- check for clock skew between hosts, a "
                "stale `now`, or a fixture built ahead of the real clock", dur(-elapsed), hms(now_sod))
    left = rem_s - elapsed
    at = "" if exact else "at least "
    note = "" if exact else (" (`elapsed` is known only modulo 24 h: both endpoints "
                             "would need a date to rule out a stall a whole day longer)")
    if left >= 0:
        return "  now %s, %s%s elapsed on the open pass, so %s to go%s" % (
            hms(now_sod), at, dur(elapsed), dur(left), note)
    return ("  now %s, %s%s elapsed on the open pass -- OVERDUE by %s%s against a %s "
            "projection. The run may have stalled or died; the projection is not a "
            "deadline%s" % (hms(now_sod), at, dur(elapsed), at, dur(-left), dur(rem_s), note))


def read_mtimes(d, pattern):
    """-> [(mtime, size, build, pass)] from filenames matching `pattern`.

    `pattern` must name groups `build` and `pass`. A per-pass output file's mtime is
    the pass's COMPLETION, never its start -- the caller must not conflate them.
    """
    try:
        rx = re.compile(pattern)
    except re.error as e:
        _refuse("the pattern %r does not compile: %s", pattern, e)
    # Checked here rather than at the first match, where a missing group raised a bare
    # IndexError from inside the loop -- an unstructured exception for a caller error the
    # tool can name exactly, and one that cannot be distinguished from a corrupt file.
    want = set(rx.groupindex) & {"build", "pass"}
    if want != {"build", "pass"}:
        _refuse("the pattern %r must name groups (?P<build>...) and (?P<pass>...); it "
                "names %s. Without both, a per-build round count cannot be formed and "
                "every pass would be attributed to one arm", pattern,
                sorted(rx.groupindex) or "no groups at all")
    # ANCHORING IS CHECKED LAST OF THE THREE, and the order is the point. re.match anchors only
    # at the start, so an unanchored pattern accepts a renamed artefact (x.csv.bad) as a clean
    # pass file. But placing this check ahead of the compile made a truncated pattern --
    # `dl_(?P<build>[a-z` -- refuse for missing a `$` when its actual fault is that it does not
    # compile, and SystemExit does not pass through `except re.error`, so the better diagnosis
    # became unreachable. A new guard in front of existing validation can silence it; the
    # narrowest failure is diagnosed last.
    LC.require_end_anchored(pattern)
    out = []
    for f in sorted(os.listdir(d)):
        m = rx.match(f)
        if not m:
            continue
        st = os.stat(os.path.join(d, f))
        out.append((st.st_mtime, st.st_size, m.group("build"), int(m.group("pass")), f))
    out.sort()
    return out


def project_mtimes(d, pattern, rounds, now=None, now_epoch=None):
    """Projection from per-pass completion mtimes. Prints the same inputs as the log mode.

    `now_epoch` is the unambiguous route and the DEFAULT: an mtime is a real epoch, so
    when `now` is also one the open pass's age is measured rather than assumed, and a
    stall of any length reports its true size. `now` as a bare HH:MM:SS is kept because
    it answers a different question -- what the projection looked like at an arbitrary
    instant T -- and it is marked as a lower bound rather than refused. Discarding the
    mtime's date to match the log mode's bare-clock convention was a choice, not a
    constraint, and it threw away the only field that could resolve the ambiguity.
    """
    fs = read_mtimes(d, pattern)
    if len(fs) < 3:
        _refuse("%d file(s) matched in %s -- a difference-of-completions "
                "estimator needs at least two landed passes plus an open one", len(fs), d)
    sizes = [x[1] for x in fs]
    openf = None
    if sizes[-1] < min(sizes[:-1]):
        openf = fs[-1]
        landed = fs[:-1]
    else:
        _refuse("the newest file %s is %d bytes and not smaller than every "
                "landed file (min %d), so the open pass cannot be identified by size; "
                "its mtime is a moving target and including it would attribute a "
                "near-zero duration to an unfinished pass", fs[-1][4], sizes[-1], min(sizes[:-1]))
    builds = sorted(set(x[2] for x in fs))
    per = {}
    for i in range(1, len(landed)):
        per.setdefault(landed[i][2], []).append(landed[i][0] - landed[i - 1][0])
    nolast = [b for b in builds if not per.get(b)]
    if nolast:
        _refuse("build(s) %s have no pass with a measurable duration. A "
                "completion-difference needs a predecessor, so the earliest file "
                "contributes none", ",".join(nolast))
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
            _refuse("build %s already has %d files but only %d rounds were "
                    "declared", b, done, rounds)
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
        _refuse("the rotation is not balanced -- file counts %s differ by %d. "
                "The schedule is meant to hold them within one, so each build's mean "
                "would be conditioned on a different stretch of the run",
                dict(sorted(counts.items())),
                max(counts.values()) - min(counts.values()))
    base = landed[-1][0]
    fin = base + rem_s
    # Scoped to the BARE-CLOCK route only. With a real clock both endpoints carry a
    # date, so a remainder over a day is no longer ambiguous and refusing would be
    # refusing a question the inputs can answer.
    if now is not None and now_epoch is None and rem_s >= 86400:
        _refuse("%.0f s remain, which is a day or more, and `now` was given "
                "as a bare clock. Pass a real clock instead -- an mtime carries a "
                "date, so that route has no ambiguity to refuse", rem_s)
    out.append("  remaining seconds = %.3f over %d passes" % (rem_s, rem_n))
    # LOCAL time, not `epoch % 86400`. st_mtime is epoch UTC and the log mode reads a
    # local wall clock out of the log text, so a UTC rendering here put the two modes
    # 7 hours apart on the same kind of run -- a disagreement no plant of mine covered
    # because every plant built its own mtimes and compared them to each other.
    ref = now_epoch if now_epoch is not None else time.time()
    out.append("  the open pass's start is taken as %s's COMPLETION, %s local"
               % (landed[-1][4], local_hms(base, ref)))
    out.append("  PROJECTED FINISH %s local  (+%s)" % (local_hms(fin, ref), dur(rem_s)))
    if now_epoch is not None:
        out.append(_to_go_line(rem_s, _elapsed_exact(base, now_epoch),
                               _secs_of_day(now_epoch), exact=True))
    elif now is not None:
        out.append(_to_go_line(rem_s, _elapsed_since(_secs_of_day(base), now), now,
                               exact=False))
    return out


def _secs_of_day(epoch):
    t = time.localtime(epoch)
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec


def local_hms(epoch, ref=None):
    """HH:MM:SS, with the DATE when it falls on a different local day from `ref`.

    A 25-hour-old stalled run printed a finish of "10:27:14 local" for an instant that
    was YESTERDAY. HH:MM:SS alone cannot say which day, and the projection's whole job
    is to be compared against a clock.
    """
    t = time.localtime(epoch)
    if ref is not None and time.localtime(ref)[:3] != t[:3]:
        return time.strftime("%Y-%m-%d %H:%M:%S", t)
    return time.strftime("%H:%M:%S", t)


def _main(a):
    if a and a[0] == "--mtimes":
        if len(a) < 4:
            sys.exit("usage: pfs_eta.py --mtimes <dir> <regex with (?P<build>)(?P<pass>)> "
                     "<rounds> [HH:MM:SS now]")
        # `a` is ["--mtimes", dir, regex, rounds, now], so `now` is a[4] and len(a) is 5.
        # This read a[5] behind a `len(a) > 5` gate -- both off by one in the SAME
        # direction, so the gate that would have raised the IndexError was itself never
        # true and the documented 4-argument form silently dropped `now`. Repairing only
        # one half converts a silent no-op into a crash or leaves it silent.
        # Omitting the clock used to print no time-to-go line at all, which made the
        # OVERDUE report reachable ONLY through the ambiguous bare-clock argument. The
        # default is now the real clock, which in this mode is fully unambiguous, so the
        # bare clock is no longer the only route and the capability it exists for
        # (asking what the projection looked like at an arbitrary T) is unaffected.
        nw, nwe = None, None
        if len(a) > 4:
            nw = _secs(*a[4].split(":"))
        else:
            nwe = time.time()
        print(_report(project_mtimes(a[1], a[2], int(a[3]), nw, nwe)))
        return 0
    else:
        if len(a) < 2:
            sys.exit("usage: pfs_eta.py <pass-log> <rounds> [HH:MM:SS now]\n"
                     "       pfs_eta.py --mtimes <dir> <regex> <rounds> [HH:MM:SS now]")
        # Same default; here it stays a lower bound because the log carries no date.
        nw = _secs(*a[2].split(":")) if len(a) > 2 else _secs_of_day(time.time())
        print(_report(project(a[0], int(a[1]), nw)))
        return 0


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv[1:]))
    except _Refuse as _e:
        print(_surface(_e))
        sys.exit(1)
