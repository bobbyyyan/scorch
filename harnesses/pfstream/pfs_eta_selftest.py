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
import datetime
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "pfs_eta.py")
sys.path.insert(0, HERE)
import pfs_eta                                                   # noqa: E402

RX = r"dl_(?P<build>[a-z_]+)_float32_p(?P<pass>[0-9]+)\.csv"


def refusal_lines():
    """Line numbers of every REFUSE literal in pfs_eta.py, from its AST."""
    t = ast.parse(open(TARGET).read())
    return set(n.lineno for n in ast.walk(t)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value.startswith("REFUSE:"))


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
    expect("mt/across-midnight-finish", got, "PROJECTED FINISH 01:30:00")


def t_overdue_both_modes():
    """A late run must report its DEFICIT. Both modes clamped it to 00:00:00, which reads
    as about-to-land for a run that may be dead -- the dangerous direction when a watcher
    is the wake signal."""
    p = mklog(passes([(1, "a", "09:40:00", "09:50:00"), (1, "b", "09:50:00", "10:00:00"),
                      (2, "a", "10:00:00", None)]))
    expect("log/overdue", pfs_eta.project(p, 2, _clock(11, 20)),
           "OVERDUE by 01:00:00", want_absent=("so 00:00:00",))
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (9, 40)),
                      ("dl_b_float32_p1.csv", 400, (9, 50)),
                      ("dl_a_float32_p2.csv", 400, (10, 0)),
                      ("dl_b_float32_p2.csv", 10, (10, 5))])
    expect("mt/overdue", pfs_eta.project_mtimes(d, RX, 2, _clock(11, 20)),
           "OVERDUE by", want_absent=("so 00:00:00",))


def t_cli_now_is_honoured():
    """THE ARGUMENT-INDEX DEFECT, tested through the CLI because that is where it lived.
    `--mtimes dir regex rounds now` read a[5] behind a `len(a) > 5` gate -- both off by
    one in the same direction, so the gate that would have raised the IndexError was
    never true and the documented form silently dropped `now`. A call into
    project_mtimes() cannot see this; only argv can."""
    global CHECKS
    import subprocess
    d = mkdir_mtimes([("dl_a_float32_p1.csv", 400, (18, 0)),
                      ("dl_b_float32_p1.csv", 400, (19, 30)),
                      ("dl_a_float32_p2.csv", 400, (21, 0)),
                      ("dl_b_float32_p2.csv", 120, (22, 30))])
    for args, want in (([TARGET, "--mtimes", d, RX, "3", "23:30:00"], "to go"),
                       ([TARGET, "--mtimes", d, RX, "3"], None)):
        CHECKS += 1
        o = subprocess.run([sys.executable] + args, capture_output=True, text=True).stdout
        if want and want not in o:
            FAIL.append("cli/now: %r missing from `%s`:\n%s" % (want, " ".join(args[1:]), o))
        if want is None and ("to go" in o or "OVERDUE" in o):
            FAIL.append("cli/no-now: a clock line appeared with no clock argument:\n%s" % o)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def main():
    sys.settrace(_tracer)
    try:
        for fn in TESTS:
            fn()
    finally:
        sys.settrace(None)
    want = refusal_lines()
    missed = sorted(want - HIT)
    src = open(TARGET).read().split("\n")
    print("pfs_eta selftest: %d plants, %d assertions, %d refusal sites in the AST"
          % (len(TESTS), CHECKS, len(want)))
    print("  refusal sites reached: %d of %d" % (len(want & HIT), len(want)))
    rc = 0
    if missed:
        rc = 1
        print("  UNCOVERED refusal sites -- a plant is missing for each:")
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
