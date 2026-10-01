"""Plant corpus for pfs_eta.py. Covers every refusal site, and proves the coverage.

WHY THIS FILE EXISTS. pfs_eta.py shipped with ten refusal paths and one optional output
line, and the repo supplied an input to none of them. Three defects were then found by
reading its output against a directory listing rather than by any check: an argument
index off by one that silently dropped the documented `now` argument, a time-to-go that
wrapped past midnight and clamped a two-hour remainder to zero, and the same clamp
turning an hour-late run into "00:00:00 to go". All three sat on the one path a plant
would have had to construct an argument list AND a clock to reach.

WHY THE FIXTURES ARE SYNTHESIZED AND NOT COMMITTED FILES. The mtime mode's entire input
is file modification times, and **git does not record mtimes** -- a checked-out fixture
carries the checkout's clock, not the one the test needs. So the corpus is built by
`os.utime` into a temporary directory on each run. That is forced by the input's nature,
not a convenience, and it is also why these plants cannot live in the scanned tree the
way fmtscan's can.

HOW COVERAGE IS ESTABLISHED, and it is not a typed list. The refusal sites are extracted
from pfs_eta.py's own AST -- every string constant beginning "REFUSE:" -- and the plants
run under a line tracer that records which lines of that file actually executed. The two
sets are then differenced. A refusal added later with no plant fails this file without
anyone remembering to update a list, which a list of expected messages cannot do:
a list can only check what someone thought to write down.
"""
import ast
import contextlib
import datetime
import io
import os
import runpy
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "pfs_eta.py")
sys.path.insert(0, HERE)
import pfs_eta                                                   # noqa: E402

RX = r"dl_(?P<build>[a-z_]+)_float32_p(?P<pass>[0-9]+)\.csv"


def early_exits():
    """Line numbers of every EARLY ERROR EXIT: the CALL SITES of pfs_eta._refuse, plus
    every sys.exit. A named sink, not a shape and not a spelling.

    THREE PREDICATES, TWO OF WHICH HAD A PREMISE THE FILE FELL THROUGH. Each was found
    by a planted exit the predicate could not see:

      * LEXICAL -- a string constant beginning "REFUSE:". Missed an exit worded
        differently, and missed the module's own helper refusal, whose literal began
        with two spaces so `.startswith("REFUSE:")` was False. The unplanted path left
        the DENOMINATOR rather than the numerator, so the ratio read 10 of 10 and OK.
      * STRUCTURAL -- a `return` of a list display with a Constant or BinOp element.
        Missed an f-string element (an ast.JoinedStr), and would miss `.format()` (a
        Call) or a message bound to a name first. It also could not see a refusal in a
        HELPER, which returns a string rather than a list -- and the justification it
        rested on, "every normal return returns a NAME", holds exactly inside the two
        projectors and NOT module-wide: all eight helper returns are BinOps. So the
        reasoning was right about the projectors and overstated about the module, and
        the gap is exactly where the newest refusal landed.
      * THIS ONE -- call sites of a single named function. Exact regardless of how the
        message is built, how the literal is spelled, or what the caller returns.

    Enumerating expression kinds cannot be completed; naming the sink removes the need
    to. The remaining assumption -- that every refusal goes through the sink -- is
    CHECKED by t_refuse_sink_is_the_only_producer rather than assumed, which is the step
    missing from both earlier predicates.
    """
    t = ast.parse(open(TARGET).read())
    out = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id == "_refuse":
                out.add(n.lineno)
            if isinstance(n.func, ast.Attribute) and n.func.attr == "exit":
                out.add(n.lineno)
    return out


HIT = set()


def _tracer(frame, event, arg):
    if frame.f_code.co_filename == TARGET:
        if event == "line":
            HIT.add(frame.f_lineno)
        return _tracer
    return None


def _clock(hh, mm, ss=0):
    return hh * 3600 + mm * 60 + ss


def mkdir_mtimes(spec):
    """spec: [(name, size, (H, M))] -> a tempdir with those mtimes on 2026-10-01."""
    d = tempfile.mkdtemp(prefix="pfseta_")
    for name, size, (hh, mm) in spec:
        p = os.path.join(d, name)
        with open(p, "wb") as fh:
            fh.write(b"x" * size)
        t = datetime.datetime(2026, 10, 1, hh, mm, 0).timestamp()
        os.utime(p, (t, t))
    return d


def run_cli(argv):
    """Execute pfs_eta.py's __main__ block IN-PROCESS with a real argv -> (out, err, rc).

    A subprocess would be correct for the argument indices and invisible to the line
    tracer, so the two usage exits could never be covered and the denominator would
    permanently read 10 of 12. runpy re-executes the module under __main__ in this
    process, so argv is genuinely argv AND every line executed is traced.
    """
    old, out, err = sys.argv, io.StringIO(), io.StringIO()
    sys.argv = [TARGET] + argv
    rc = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(TARGET, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if not isinstance(e.code, (int, type(None))):
            err.write(str(e.code))
    finally:
        sys.argv = old
    return out.getvalue(), err.getvalue(), rc


def mkdir_relative(spec, now):
    """spec: [(name, size, seconds_before_now)] -> a tempdir with real-clock mtimes."""
    d = tempfile.mkdtemp(prefix="pfseta_")
    for name, size, age in spec:
        q = os.path.join(d, name)
        with open(q, "wb") as fh:
            fh.write(b"x" * size)
        os.utime(q, (now - age, now - age))
    return d


def mklog(lines):
    d = tempfile.mkdtemp(prefix="pfseta_")
    p = os.path.join(d, "run.log")
    with open(p, "w") as fh:
        fh.write("".join(lines))
    return p


def passes(spec):
    """[(pass, build, startHMS, doneHMS or None)] -> log lines."""
    out = []
    for pn, b, st, dn in spec:
        out.append("--- p%d %s start %s\n" % (pn, b, st))
        if dn:
            out.append("--- p%d %s done  %s  rc=0\n" % (pn, b, dn))
    return out


FAIL = []
CHECKS = 0


def expect(label, got, want_sub, want_absent=()):
    """`got` is the projector's line list. Assert a substring present and others absent."""
    global CHECKS
    CHECKS += 1
    body = "\n".join(got)
    if want_sub not in body:
        FAIL.append("%s: expected %r in output, got:\n%s" % (label, want_sub, body))
    for w in want_absent:
        if w in body:
            FAIL.append("%s: did NOT expect %r in output, got:\n%s" % (label, w, body))


# ---------------------------------------------------------------- log-mode refusals
def t_log_no_lines():
    expect("log/no-lines", pfs_eta.project(mklog(["nothing\n"]), 4),
           "REFUSE: no pass lines matched")


def t_log_no_landed_build():
    p = mklog(passes([(1, "a", "00:00:00", "00:10:00"), (1, "b", "00:10:00", None)]))
    expect("log/no-landed-build", pfs_eta.project(p, 2),
           "REFUSE: no landed pass for build(s) b")


def t_log_over_rounds():
    p = mklog(passes([(1, "a", "00:00:00", "00:10:00"),
                      (2, "a", "00:10:00", "00:20:00")]))
    expect("log/over-rounds", pfs_eta.project(p, 1), "REFUSE: build a has 2 landed")


def t_log_day_or_more():
    """A remainder of a day makes `elapsed` undecidable from a bare clock."""
    p = mklog(passes([(1, "a", "00:00:00", "08:00:00"), (1, "b", "08:00:00", "16:00:00"),
                      (2, "a", "16:00:00", None)]))
    expect("log/day-or-more", pfs_eta.project(p, 4, _clock(17, 0)),
           "REFUSE: ")
    # and WITHOUT a clock it must still project -- the refusal is about `now`, not the run
    expect("log/day-or-more-no-clock", pfs_eta.project(p, 4),
           "PROJECTED FINISH", want_absent=("REFUSE",))


# ------------------------------------------------------------- mtime-mode refusals
def t_mt_too_few():
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (1, 0)),
                      ("dl_b_float32_p1.csv", 400, (2, 0))])
    expect("mt/too-few", pfs_eta.project_mtimes(d, RX, 2), "REFUSE: 2 file(s) matched")


def t_mt_open_not_smaller():
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (1, 0)),
                      ("dl_b_float32_p1.csv", 400, (2, 0)),
                      ("dl_a_float32_p2.csv", 400, (3, 0))])
    expect("mt/open-not-smaller", pfs_eta.project_mtimes(d, RX, 2),
           "REFUSE: the newest file")


def t_mt_no_predecessor():
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (1, 0)),
                      ("dl_b_float32_p1.csv", 400, (2, 0)),
                      ("dl_b_float32_p2.csv", 10, (3, 0))])
    expect("mt/no-predecessor", pfs_eta.project_mtimes(d, RX, 2),
           "REFUSE: build(s) a have no pass with a measurable duration")


def t_mt_over_rounds():
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (1, 0)),
                      ("dl_b_float32_p1.csv", 400, (2, 0)),
                      ("dl_a_float32_p2.csv", 400, (3, 0)),
                      ("dl_b_float32_p2.csv", 400, (4, 0)),
                      ("dl_a_float32_p3.csv", 10, (5, 0))])
    expect("mt/over-rounds", pfs_eta.project_mtimes(d, RX, 1),
           "REFUSE: build a already has 3 files but only 1 rounds")


def t_mt_unbalanced():
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (1, 0)),
                      ("dl_b_float32_p1.csv", 400, (2, 0)),
                      ("dl_c_float32_p1.csv", 400, (3, 0)),
                      ("dl_a_float32_p2.csv", 400, (4, 0)),
                      ("dl_a_float32_p3.csv", 400, (5, 0)),
                      ("dl_c_float32_p2.csv", 10, (6, 0))])
    expect("mt/unbalanced", pfs_eta.project_mtimes(d, RX, 4),
           "REFUSE: the rotation is not balanced")


def t_mt_day_or_more():
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (0, 0)),
                      ("dl_b_float32_p1.csv", 400, (8, 0)),
                      ("dl_a_float32_p2.csv", 400, (16, 0)),
                      ("dl_b_float32_p2.csv", 10, (23, 0))])
    expect("mt/day-or-more", pfs_eta.project_mtimes(d, RX, 4, _clock(23, 30)), "REFUSE: ")


# ----------------------------------------------------- the arithmetic and the clock
def t_log_arithmetic():
    """Hand-computable: means 100 and 200, one round owed each, open pass at 00:05:00."""
    p = mklog(passes([(1, "a", "00:00:00", "00:01:40"), (1, "b", "00:01:40", "00:05:00"),
                      (2, "a", "00:05:00", None)]))
    expect("log/arithmetic", pfs_eta.project(p, 2), "PROJECTED FINISH 00:10:00")


def t_log_no_open_pass():
    """Every pass closed: there is no open pass to project from, and it must say so."""
    p = mklog(passes([(1, "a", "00:00:00", "00:10:00"), (1, "b", "00:10:00", "00:20:00")]))
    expect("log/no-open-pass", pfs_eta.project(p, 2), "NOTE: no pass is open")


def t_mt_local_clock():
    """st_mtime is epoch UTC; the printed instant must be the LOCAL wall clock."""
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (10, 29)),
                      ("dl_b_float32_p1.csv", 400, (10, 31)),
                      ("dl_a_float32_p2.csv", 400, (10, 33)),
                      ("dl_b_float32_p2.csv", 10, (10, 34))])
    expect("mt/local-clock", pfs_eta.project_mtimes(d, RX, 2), "COMPLETION, 10:33:00 local")


def t_mt_across_midnight():
    """THE CLAMP DEFECT. 5400 s per pass, open at 21:00, 3 owed -> finish 01:30 next day.
    Read at 23:30 the truth is 02:00:00 remaining, and the shipped code printed 00:00:00
    because it wrapped `finish` to seconds-of-day before subtracting `now`."""
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (18, 0)),
                      ("dl_b_float32_p1.csv", 400, (19, 30)),
                      ("dl_a_float32_p2.csv", 400, (21, 0)),
                      ("dl_b_float32_p2.csv", 120, (22, 30))])
    got = pfs_eta.project_mtimes(d, RX, 3, _clock(23, 30))
    expect("mt/across-midnight", got, "so 02:00:00 to go", want_absent=("so 00:00:00",))
    # The finish now carries a DATE when it is not on the reference day. A bare
    # HH:MM:SS printed a next-day instant as though it were today's.
    expect("mt/across-midnight-finish", got, "PROJECTED FINISH 2026-10-02 01:30:00")


def t_overdue_both_modes():
    """A late run must report its DEFICIT. Both modes clamped it to 00:00:00, which reads
    as about-to-land for a run that may be dead -- the dangerous direction when a watcher
    is the wake signal."""
    p = mklog(passes([(1, "a", "09:40:00", "09:50:00"), (1, "b", "09:50:00", "10:00:00"),
                      (2, "a", "10:00:00", None)]))
    # "at least", because a pass log's timestamps carry no date: this route cannot rule
    # out a stall a whole day longer. The log mode has no unambiguous alternative.
    expect("log/overdue", pfs_eta.project(p, 2, _clock(11, 20)),
           "OVERDUE by at least 01:00:00", want_absent=("so 00:00:00",))
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (9, 40)),
                      ("dl_b_float32_p1.csv", 400, (9, 50)),
                      ("dl_a_float32_p2.csv", 400, (10, 0)),
                      ("dl_b_float32_p2.csv", 10, (10, 5))])
    expect("mt/overdue", pfs_eta.project_mtimes(d, RX, 2, _clock(11, 20)),
           "OVERDUE by", want_absent=("so 00:00:00",))


def t_cli_now_is_honoured():
    """THE ARGUMENT-INDEX DEFECT, tested through argv because that is where it lived.
    `--mtimes dir regex rounds now` read a[5] behind a `len(a) > 5` gate -- both off by
    one in the same direction, so the gate that would have raised the IndexError was
    never true and the documented form silently dropped `now`. A call into
    project_mtimes() cannot see this; only argv can."""
    global CHECKS
    import time as _t
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (18, 0)),
                      ("dl_b_float32_p1.csv", 400, (19, 30)),
                      ("dl_a_float32_p2.csv", 400, (21, 0)),
                      ("dl_b_float32_p2.csv", 120, (22, 30))])
    # The default route needs a fixture behind the REAL clock, not a fixed wall time:
    # the absolute fixture above sits in the future for most of the day, and the
    # real-clock route then correctly refuses rather than reporting.
    dr = mkdir_relative([("dl_a_float32_p1.csv", 400, 7200),
                         ("dl_b_float32_p1.csv", 400, 5400),
                         ("dl_a_float32_p2.csv", 400, 3600),
                         ("dl_b_float32_p2.csv", 120, 60)], _t.time())
    # Both forms must now report a clock line: the bare-clock one marked "at least",
    # the default one (no argument, real clock) exact. Omitting the argument used to
    # print nothing, which made OVERDUE reachable only through the ambiguous route.
    for args, want in ((["--mtimes", d, RX, "3", "23:30:00"], "at least"),
                       (["--mtimes", dr, RX, "3"], "elapsed on the open pass")):
        CHECKS += 1
        o, _e, _rc = run_cli(args)
        if want not in o:
            FAIL.append("cli/now: %r missing from `%s`:\n%s" % (want, " ".join(args), o))
    # And the default route must NOT carry the modulo-24h caveat, since an mtime has a date.
    CHECKS += 1
    o, _e, _rc = run_cli(["--mtimes", dr, RX, "3"])
    if "modulo 24 h" in o:
        FAIL.append("cli/default-exact: the real-clock route claimed modulo-24h "
                    "ambiguity, but both endpoints carry a date:\n%s" % o)
    # The log mode's default must also report, and must stay marked as a lower bound.
    CHECKS += 1
    lp = mklog(passes([(1, "a", "00:00:00", "00:10:00"), (1, "b", "00:10:00", "00:20:00"),
                       (2, "a", "00:20:00", None)]))
    o, _e, _rc = run_cli([lp, "2"])
    if "elapsed on the open pass" not in o or "modulo 24 h" not in o:
        FAIL.append("cli/log-default: the log mode's default clock line is missing or "
                    "is not marked as a lower bound:\n%s" % o)


def t_mt_exact_beats_the_wrap():
    """A 25-HOUR STALL. The bare-clock route cannot tell it from a 1-hour stall -- both
    printed an identical line to the second -- and the real-clock route must, because an
    mtime carries a date. The day-or-more refusal cannot catch this: it guards `rem_s`,
    which for a dead run is whatever the projection was when it stopped, i.e. small."""
    import time as _t
    now = _t.time()
    d = tempfile.mkdtemp(prefix="pfseta_")
    for name, size, age in (("dl_a_float32_p1.csv", 400, 90000 + 2400),
                            ("dl_b_float32_p1.csv", 400, 90000 + 1200),
                            ("dl_a_float32_p2.csv", 400, 90000),
                            ("dl_b_float32_p2.csv", 10, 0)):
        q = os.path.join(d, name)
        with open(q, "wb") as fh:
            fh.write(b"x" * size)
        os.utime(q, (now - age, now - age))
    lt = _t.localtime(now)
    sod = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    expect("mt/25h-exact", pfs_eta.project_mtimes(d, RX, 2, None, now),
           "1d 01:00:00 elapsed", want_absent=("at least",))
    expect("mt/25h-bare", pfs_eta.project_mtimes(d, RX, 2, sod),
           "at least 01:00:00 elapsed")


def t_dur_does_not_wrap():
    """The formatter was the residue. `hms` takes a modulus, so an exactly-measured
    90000 s printed as 01:00:00 and the arithmetic fix was invisible to the reader."""
    global CHECKS
    CHECKS += 1
    if pfs_eta.dur(90000) != "1d 01:00:00" or pfs_eta.hms(90000) != "01:00:00":
        FAIL.append("dur/wrap: dur(90000)=%r hms(90000)=%r -- a span must not wrap and a "
                    "clock must" % (pfs_eta.dur(90000), pfs_eta.hms(90000)))


def t_now_before_open_pass():
    """A `now` EARLIER than the open pass's start rendered a plausible remainder from an
    impossible state (-09:47:47 elapsed, "so 14:17:47 to go"). Found by a fixture whose
    mtimes sat ahead of the real clock, which is also what host clock skew looks like."""
    import time as _t
    now = _t.time()
    d = tempfile.mkdtemp(prefix="pfseta_")
    for name, size, ahead in (("dl_a_float32_p1.csv", 400, 3600),
                              ("dl_b_float32_p1.csv", 400, 5400),
                              ("dl_a_float32_p2.csv", 400, 7200),
                              ("dl_b_float32_p2.csv", 10, 7300)):
        q = os.path.join(d, name)
        with open(q, "wb") as fh:
            fh.write(b"x" * size)
        os.utime(q, (now + ahead, now + ahead))
    expect("mt/now-before-open", pfs_eta.project_mtimes(d, RX, 2, None, now),
           "REFUSE: the open pass's start is", want_absent=("to go",))


def t_usage_exits():
    """The two usage exits. The structural coverage predicate demands these; the
    prefix-based one could not see them, since `sys.exit("usage: ...")` carries no
    REFUSE. Run in-process so the tracer records them."""
    global CHECKS
    for args, want in ((["--mtimes", "d", "rx"], "usage: pfs_eta.py --mtimes"),
                       ([], "usage: pfs_eta.py <pass-log>")):
        CHECKS += 1
        _o, e, rc = run_cli(args)
        if want not in e or rc == 0:
            FAIL.append("cli/usage %r: rc=%d stderr=%r" % (args, rc, e))


def t_refuse_sink_is_the_only_producer():
    """ENFORCE the convention the coverage predicate rests on, rather than assume it.

    Both earlier predicates assumed something about refusals and neither checked it, so a
    non-conforming exit silently left the population. The sink's population is exact only
    while every refusal goes through the sink -- so this fails if any "REFUSE" literal in
    pfs_eta.py sits outside a `_refuse(...)` call. Docstrings and the sink's own body are
    excluded by SCOPE, computed from the AST, not by a list of allowed line numbers.
    """
    global CHECKS
    CHECKS += 1
    src = open(TARGET).read()
    t = ast.parse(src)
    ok = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_refuse":
            ok.update(id(c) for c in ast.walk(n))
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            d = ast.get_docstring(n, clean=False)
            if d is not None and n.body:
                ok.update(id(c) for c in ast.walk(n.body[0]))
        if isinstance(n, ast.FunctionDef) and n.name == "_refuse":
            ok.update(id(c) for c in ast.walk(n))
    stray = [(n.lineno, n.value[:60]) for n in ast.walk(t)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and "REFUSE" in n.value and id(n) not in ok]
    if stray:
        FAIL.append("refuse-sink: %d REFUSE literal(s) outside a _refuse() call, so the "
                    "coverage population is not exact: %s" % (len(stray), stray))


def t_coverage_population_is_not_vacuous():
    """The population must be non-trivial and must include the HELPER refusal.

    A sink predicate that found zero call sites would print "0 of 0" and read as full
    coverage -- the same vacuity as the ratio it replaced. And the specific path the
    structural predicate could not see was the one in _to_go_line, so its presence is the
    regression test for finding 7 at the population level rather than the verdict level.
    """
    global CHECKS
    CHECKS += 1
    t = ast.parse(open(TARGET).read())
    in_helper = []
    for f in ast.walk(t):
        if isinstance(f, ast.FunctionDef) and f.name == "_to_go_line":
            in_helper = [n.lineno for n in ast.walk(f) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "_refuse"]
    n = len(early_exits())
    if n < 8:
        FAIL.append("population: only %d early exits found -- a ratio over a near-empty "
                    "population reads as coverage" % n)
    if not in_helper:
        FAIL.append("population: _to_go_line has no _refuse call site, so the helper "
                    "refusal is outside the coverage population again (finding 7)")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def main():
    sys.settrace(_tracer)
    try:
        for fn in TESTS:
            fn()
    finally:
        sys.settrace(None)
    want = early_exits()
    missed = sorted(want - HIT)
    src = open(TARGET).read().split("\n")
    print("pfs_eta selftest: %d plants, %d assertions, %d early exits in the AST "
          "(call sites of the _refuse sink, plus sys.exit)" % (len(TESTS), CHECKS, len(want)))
    print("  early exits reached: %d of %d" % (len(want & HIT), len(want)))
    rc = 0
    if missed:
        rc = 1
        print("  UNCOVERED early exits -- a plant is missing for each:")
        for ln in missed:
            print("    pfs_eta.py:%d  %s" % (ln, src[ln - 1].strip()[:80]))
    for f in FAIL:
        rc = 1
        print("  FAIL " + f)
    # The coverage number is only evidence if the tracer actually saw the file.
    if len(HIT) < 50:
        rc = 1
        print("  REFUSE: the tracer recorded only %d lines of pfs_eta.py, so the coverage "
              "figure above is vacuous" % len(HIT))
    print("  " + ("OK" if rc == 0 else "FAILED"))
    for d in [t for t in os.listdir(tempfile.gettempdir()) if t.startswith("pfseta_")]:
        shutil.rmtree(os.path.join(tempfile.gettempdir(), d), ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
