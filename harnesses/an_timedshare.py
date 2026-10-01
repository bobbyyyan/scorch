#!/usr/bin/env python3
"""What fraction of a DLMC pass is spent inside kernel calls -- and the retraction of 21-22%.

THE ERROR BEING CORRECTED. Two sessions independently computed the "timed-call share" of a pass as
sum(ms * batch) over all rows, divided by the pass wall time, and got 21.3-22.3%. Both then drew the
same conclusion: at most 0.223 * 24 = 5.35 core-equivalents can come from timed calls against an
observed 8.651 busy cores, so at least 3.30 cores -- 38% of busy time -- must be outside them.

That conclusion is wrong, because sum(ms * batch) is not the time the calls took. From mrgate.py's
timeit(), verbatim:

    b = float("inf")
    for _ in range(A.reps):
        t0 = time.perf_counter()
        for _ in range(batch): fn()
        b = min(b, (time.perf_counter() - t0) / batch)
    return b * 1e3, batch

The returned ms is a PER-CALL time (elapsed divided by batch, minimised over reps), so ms * batch is
the duration of ONE batch. The loop runs `reps` batches. chain207 and chain210 both ran --reps 4, so
the calls actually occupied about four times the quoted figure: 85-89% of the pass, not 21-22%.

THREE WAYS THE FACTOR IS ESTABLISHED, none of them relying on a carried memory of a flag:
  1. UNITS. calibrate() picks batch so that one batch lasts about --target-ms. Therefore
     mean(ms * batch) over rows must come out near target_ms if and only if ms is per-call. At
     target 6 ms it reads 5.2-5.3 ms. Had ms been a whole-batch time, mean(ms * batch) would be
     target_ms * batch -- two orders of magnitude out. This is a units check, so it also refutes
     any reading in which the CSV's ms means something else.
  2. ARITHMETIC CEILING. reps batches cannot outlast the pass, so
     reps <= floor(wall / sum(ms * batch)). On these passes that is 4, which by itself refutes
     mrgate.py's --reps DEFAULT of 13 without consulting any log: 13 batches would need 3963 s
     inside a 1370 s object-pass.
  3. THE PROCESS ITSELF. /proc/<pid>/cmdline of the live twelfth pass reads
     "--reps 4 --target-ms 6", which is the running command rather than a script that may not be
     the one that launched it.
Agreement of (2)'s bound with (3)'s value is the guard: this script refuses if the reps it is told
to use exceeds what the wall clock allows.

WHAT IS STILL NOT TIMED. calibrate() makes 5 untimed calls and is called twice per cell (once for
the shared ours/oursb batch, once for mkl), so 10 untimed kernel calls per cell. Priced at the
per-call times in the same row, they are 0.3% of a pass -- the reason "outside the timed calls" and
"outside the kernel" differ by almost nothing here, which is the opposite of what I expected when I
opened this question.

ESTIMAND. Per object-pass, with its own wall time from the stage log's start/done pair -- never a
pass-averaged denominator:
    one_batch   = sum over rows of ms * batch / 1000                      [seconds]
    timed       = reps * one_batch                 (a LOWER bound: ms is a min over reps, and
                                                    every non-minimal rep ran longer)
    calibration = 5 * sum over cells of (ms_ours + ms_mkl) / 1000
    share       = (timed + calibration) / wall
Occupancy is then bracketed rather than divided, because the non-kernel remainder is not idle:
busy core-seconds outside the kernel lie between 1 * remainder and NCPU * remainder, so in-kernel
occupancy is bounded on both sides and reported as an interval.
"""
import collections, csv, glob, os, sys

CAL_CALLS_PER_CALIBRATE = 5      # calibrate(): `for _ in range(5)`
CALIBRATES_PER_CELL = 2          # batch_ours (shared by ours+oursb) and batch_mkl


def hms(s):
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def pass_walls(stage):
    """start/done seconds-of-day pairs per (pass, build), wrap-corrected. Refuses, never guesses."""
    st, dn = {}, {}
    for ln in open(stage):
        t = ln.split()
        if len(t) >= 5 and t[0] == "---" and t[3] == "start":
            st[(t[1], t[2])] = hms(t[4])
        elif len(t) >= 5 and t[0] == "---" and t[3] == "done":
            dn[(t[1], t[2])] = hms(t[4])
    out = {}
    for k, a in st.items():
        if k not in dn:
            continue                                  # still running; not an error
        b = dn[k]
        if b < a:
            b += 86400                                # one midnight crossing
        d = b - a
        if not (0 < d < 7200):
            sys.exit("REFUSE: %s %s spans %d s, which no object-pass does; the stage log's clock "
                     "cannot be read as seconds-of-day here." % (k[0], k[1], d))
        out[k] = d
    return out


def read_csv(path):
    one_batch_ms = 0.0
    percall_ms = collections.defaultdict(float)
    cells = set()
    for r in csv.DictReader(open(path)):
        try:
            ms, b = float(r["ms"]), float(r["batch"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (ms > 0 and b >= 1):
            continue
        one_batch_ms += ms * b
        percall_ms[r["arm"]] += ms
        cells.add((r["key"], r["k"]))
    return one_batch_ms, dict(percall_ms), len(cells)


def report(d, reps, target_ms, busy=None, ncpu=24, label=""):
    stages = glob.glob(os.path.join(d, "*_stage.log"))
    if len(stages) != 1:
        sys.exit("REFUSE: %d files match *_stage.log in %s; the wall-time denominator has no "
                 "unambiguous source." % (len(stages), d))
    walls = pass_walls(stages[0])
    csvs = sorted(glob.glob(os.path.join(d, "dl_*_float32_p*.csv")))
    if not csvs:
        sys.exit("REFUSE: no dl_*_float32_p*.csv under %s -- an empty sweep reads as a 0%% share, "
                 "which is the one answer a broken glob always gives." % d)

    print("%s  %d CSVs, stage log %s, reps=%d, target_ms=%g"
          % (label, len(csvs), os.path.basename(stages[0]), reps, target_ms))
    print()
    print("  %-26s %7s %9s %9s %8s %8s %7s" %
          ("object-pass", "wall s", "1 batch s", "timed s", "calib s", "kernel s", "share"))
    print("  " + "-" * 82)

    tot = collections.Counter()
    rows_seen = 0
    repcap = None
    for f in csvs:
        base = os.path.basename(f)
        build = base.split("_")[1]
        p = "p" + base.rsplit("_p", 1)[1].split(".")[0]
        if (p, build) not in walls:
            print("  %-26s   (no completed start/done pair in the stage log -- skipped)"
                  % (p + " " + build))
            continue
        w = walls[(p, build)]
        one_ms, percall, ncells = read_csv(f)
        if ncells == 0:
            sys.exit("REFUSE: %s parsed 0 cells." % base)
        one = one_ms / 1000.0
        timed = reps * one
        cal = CAL_CALLS_PER_CALIBRATE * (percall.get("ours", 0.0) + percall.get("mkl", 0.0)) / 1000.0
        ker = timed + cal
        cap = int(w / one) if one > 0 else 0
        repcap = cap if repcap is None else min(repcap, cap)
        rows_seen += ncells
        tot.update({"wall": w, "one": one, "timed": timed, "cal": cal, "ker": ker, "n": 1})
        print("  %-26s %7d %9.1f %9.1f %8.1f %8.1f %6.1f%%"
              % (p + " " + build, w, one, timed, cal, ker, 100.0 * ker / w))

    if not tot["n"]:
        sys.exit("REFUSE: no CSV paired with a completed stage-log pass; nothing was measured.")
    print("  " + "-" * 82)
    print("  %-26s %7d %9.1f %9.1f %8.1f %8.1f %6.1f%%"
          % ("TOTAL over %d object-passes" % tot["n"], tot["wall"], tot["one"], tot["timed"],
             tot["cal"], tot["ker"], 100.0 * tot["ker"] / tot["wall"]))
    print()

    # ---- GUARD 1: units. ms must be per-call, or every figure above is off by `batch`.
    per_row = tot["one"] * 1000.0 / (rows_seen * 3)      # 3 arms share each cell
    print("  UNITS  mean ms*batch per row = %.3f ms against --target-ms %g" % (per_row, target_ms))
    if not (0.25 * target_ms <= per_row <= 1.10 * target_ms):
        sys.exit("REFUSE: one batch reads %.3f ms where calibration aims at %g ms. Either ms is not "
                 "a per-call time or batch is not the count calibration chose, and in both cases "
                 "the share above is wrong by a factor of batch." % (per_row, target_ms))
    print("         within [0.25, 1.10] x target -> ms is a per-call time and batch is its count")

    # ---- GUARD 2: reps cannot exceed what the wall clock allows.
    print("  REPS   wall/one_batch allows at most reps=%d; told to use reps=%d" % (repcap, reps))
    if reps > repcap:
        sys.exit("REFUSE: reps=%d needs %.0f s of timed calls inside the mean %d s object-pass. "
                 "mrgate.py's DEFAULT is 13; this run was not launched with it."
                 % (reps, reps * tot["one"] / max(tot["n"], 1), int(tot["wall"] / max(tot["n"], 1))))
    print("         consistent: the timed loops fit inside the measured wall time")
    print()
    print("  RETRACTION  sum(ms*batch)/wall = %.1f%% is the share of ONE batch per arm, not of the"
          % (100.0 * tot["one"] / tot["wall"]))
    print("              calls. With reps=%d counted the kernel calls are %.1f%% of the pass, so"
          % (reps, 100.0 * tot["ker"] / tot["wall"]))
    print("              'at least 38% of busy time is outside the timed calls' does not follow.")
    print("              Untimed calibration is %.1f%% of the pass: 'outside the timed calls' and"
          % (100.0 * tot["cal"] / tot["wall"]))
    print("              'outside the kernel' are nearly the same set here.")

    if busy is None:
        return
    print()
    rem = tot["wall"] - tot["ker"]
    bcs = busy * tot["wall"]
    hi = (bcs - rem * 1.0) / tot["ker"]
    lo = (bcs - rem * ncpu) / tot["ker"]
    print("  OCCUPANCY, given %.3f busy cores measured over a whole pass (NCPU=%d):" % (busy, ncpu))
    print("    busy core-seconds   %9.1f   over %.0f s of wall" % (bcs, tot["wall"]))
    print("    non-kernel remainder %8.1f s (%.1f%%); its busy share is unknown, so bracket it"
          % (rem, 100.0 * rem / tot["wall"]))
    print("    in-kernel occupancy  %.3f to %.3f cores of %d  = %.1f%% to %.1f%%"
          % (lo, hi, ncpu, 100.0 * lo / ncpu, 100.0 * hi / ncpu))
    print("      lower end assumes all %d cores busy while loading (implausible), upper end 1" % ncpu)
    print("    ceiling on the KERNEL's own time at perfect %d-way parallelism: %.2fx to %.2fx"
          % (ncpu, ncpu / hi, ncpu / lo))
    print("    -> the occupancy deficit sits INSIDE the kernel calls, which is the reverse of what")
    print("       the 21-22% figure implied.")
    print()
    print("  CROSS-CHAIN CAVEAT: a busy-core figure measured on one chain's pass paired with")
    print("  another chain's CSVs is a cross-chain pairing. The matched version pairs a chain's")
    print("  own CSVs with its own /proc/stat snapshots, which is what the seal makes possible.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: an_timedshare.py <dir with dl_*.csv and *_stage.log> <reps> <target_ms> "
                 "[busy_cores] [ncpu]\n"
                 "  reps is REQUIRED and is checked against the wall clock: mrgate.py's default is "
                 "13 and these runs used 4, so a default here would reproduce the original error.")
    report(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]),
           float(sys.argv[4]) if len(sys.argv) > 4 else None,
           int(sys.argv[5]) if len(sys.argv) > 5 else 24,
           label="TIMED SHARE  %s" % sys.argv[1])
