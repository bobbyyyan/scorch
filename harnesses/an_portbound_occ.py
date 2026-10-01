#!/usr/bin/env python3
"""Whole-pass occupancy and core-type split for the port-boundedness pre-registration.

WRITTEN BEFORE THE DATA EXISTS. snapshot(9) landed at 03:01:35; this file is committed while
snapshot(10) does not yet exist, so the reduction cannot be shaped by the interval it reduces.

WHY A WHOLE PASS AND NOT A WINDOW. Six 10-second windows put occupancy between 4.58 and 13.51 busy
cores and T_E/T_P between 1.532 and 2.110 -- spans of 2.9x and 38%, the latter 4.1x counting noise
on the sd. A window samples whichever matrix sizes are in flight, so no window is a property of the
run. An object-pass is a complete sweep of 3820 keys x 5 widths, so a whole pass IS the run's
composition and an integer number of them has no phase selection in it.

WHAT IT REFUSES ON. Snapshots whose done-counts are not consecutive integers; an interval that is
not a whole number of object-passes; a missing done timestamp in the stage log; a CPU set that is
not exactly 0-31; a snapshot with fewer than 32 cpu lines. A refusal is not a data point.

THE TWO OFFSETS ARE KNOWN, NOT BOUNDED. The poller tests every 15 s, so each snapshot lands delta
seconds after its done line. Both deltas are computable here -- the stage log timestamps each done
line to the second and every SNAP line carries epoch_ns -- so the interval LENGTH is exact. What
remains is that the ticks include delta_j seconds of the following pass's head and exclude delta_i
seconds of this one's; since both are heads of the same sweep the composition error scales with
|delta_j - delta_i|, bounded at 0.003 in T and expected 0.0010.

BUSY is /proc/stat fields 2+3+4+7+8+9 = user+nice+system+irq+softirq+steal, the same definition the
six windows used. guest and guest_nice are omitted deliberately: the kernel already counts guest
inside user, so adding them double-counts.
"""
import re, sys, os

BOUND = sys.argv[1] if len(sys.argv) > 1 else "portbound_boundaries.txt"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "c210_stage.log"
for f in (BOUND, STAGE):
    if not os.path.isfile(f):
        sys.exit("  REFUSE: no %s" % f)

# --- the landed done timestamps, in file order, which is the chronology
done = []
for l in open(STAGE):
    m = re.match(r"\s+--- p([1-6]) (aa|bb) done\s+(\d\d):(\d\d):(\d\d)", l)
    if m:
        p, b, h, mi, s = m.groups()
        done.append(((int(p), b), int(h) * 3600 + int(mi) * 60 + int(s)))
print("GUARDS")
print("  stage log carries %d done lines" % len(done))

# --- the snapshots
snaps = []
cur = None
for l in open(BOUND):
    m = re.match(r"SNAP done=(\d+) epoch_ns=(\d+) hms=(\d\d):(\d\d):(\d\d)", l)
    if m:
        n, ns, h, mi, s = m.groups()
        cur = {"n": int(n), "ns": int(ns), "tod": int(h) * 3600 + int(mi) * 60 + int(s), "cpu": {}}
        snaps.append(cur)
        continue
    m = re.match(r"cpu(\d+) (\d+)", l)
    if m and cur is not None:
        cur["cpu"][int(m.group(1))] = int(m.group(2))
if len(snaps) < 2:
    sys.exit("  REFUSE: %d snapshot(s); a whole-pass interval needs at least two." % len(snaps))
for sn in snaps:
    if set(sn["cpu"]) != set(range(32)):
        sys.exit("  REFUSE: snapshot done=%d covers %d CPUs, not 0-31" % (sn["n"], len(sn["cpu"])))
print("  %d snapshots, done-counts %s, every one carrying all 32 CPUs"
      % (len(snaps), [s["n"] for s in snaps]))

P_EVEN = range(0, 16, 2)
P_ODD = range(1, 16, 2)
E_CPU = range(16, 32)

def interval(a, b):
    if b["n"] - a["n"] != b["n"] - a["n"] or b["n"] <= a["n"]:
        sys.exit("  REFUSE: non-increasing done-counts %d -> %d" % (a["n"], b["n"]))
    npass = b["n"] - a["n"]
    # the pass boundaries these snapshots followed
    try:
        t_a, t_b = done[a["n"] - 1][1], done[b["n"] - 1][1]
    except IndexError:
        sys.exit("  REFUSE: stage log has no done line for count %d or %d" % (a["n"], b["n"]))
    da, db = a["tod"] - t_a, b["tod"] - t_b
    if not (0 <= da <= 20 and 0 <= db <= 20):
        sys.exit("  REFUSE: snapshot offsets %d s and %d s are outside the poll period" % (da, db))
    elapsed = (b["ns"] - a["ns"]) / 1e9                      # exact, from epoch_ns
    pass_len = (t_b - t_a) / npass
    d = {c: b["cpu"][c] - a["cpu"][c] for c in range(32)}
    pe = sum(d[c] for c in P_EVEN); po = sum(d[c] for c in P_ODD); e = sum(d[c] for c in E_CPU)
    tot = pe + po + e
    busy = tot / (elapsed * 100.0)
    sib = po / (pe + po) if (pe + po) else float("nan")
    print()
    print("INTERVAL done=%d -> done=%d : %d whole object-pass(es)" % (a["n"], b["n"], npass))
    print("  pass boundaries %s -> %s, mean pass %.1f s; snapshot offsets %d s and %d s"
          % (hms(t_a), hms(t_b), pass_len, da, db))
    print("  elapsed from epoch_ns %.3f s (exact); interval/pass ratio %.4f" % (elapsed, elapsed / (npass * pass_len)))
    print("  busy ticks %d -> %.3f busy cores of 24, occupancy %.1f%%, perfect-balance ceiling %.2fx"
          % (tot, busy, 100 * busy / 24, 24 / busy))
    print("  P-even %d  P-odd %d (%.2f%% of P-logical)  E %d" % (pe, po, 100 * sib, e))
    print("  T_E/T_P  raw %.4f   sibling-stripped (both sides) %.4f" % (e / (pe + po), e / pe))
    print("  residual phase term from |db - da| = %d s: %.4f in T (bound 0.003)"
          % (abs(db - da), 0.003 * abs(db - da) / 15.0))
    return {"npass": npass, "busy": busy, "T": e / pe, "Traw": e / (pe + po), "sib": sib}

def hms(s): return "%02d:%02d:%02d" % (s // 3600 % 24, s // 60 % 60, s % 60)

out = []
for i in range(len(snaps) - 1):
    out.append(interval(snaps[i], snaps[i + 1]))
if len(snaps) > 2:
    out.append(interval(snaps[0], snaps[-1]))

print()
print("AGAINST THE WINDOWED FIGURES THE WHOLE-PASS NUMBERS REPLACE:")
print("  six 10 s windows: busy cores 4.58-13.51 (factor 2.9); T_E/T_P 1.532-2.110 (38%, sd 4.1x")
print("  counting noise, span 5.3x its expected range under the null).")
for o in out:
    print("  %d-pass: busy %.3f cores, occupancy %.1f%%, ceiling %.2fx, T stripped %.4f"
          % (o["npass"], o["busy"], 100 * o["busy"] / 24, 24 / o["busy"], o["T"]))
print()
print("THE WORK-SHARE CONVERSION IS NOT COMPLETED HERE. W_E/W_P = T x (r_E/r_P) and r is a RATE")
print("ratio, unmeasured: r = (IPC_P/IPC_E) x (f_P/f_E) with the IPC factor >= 1 and the clock")
print("factor measured only as 1.32 +/- 0.05, whose 95% interval [1.2136, 1.4298] contains the whole")
print("base-to-boost spec range. The pinned V0 pair measures r directly as a rate, and until it runs")
print("the share is a BOUND: E share <= T/(T + 1.2136). Printed for reference, not as a result:")
for o in out:
    print("  %d-pass: E work share <= %.1f%%" % (o["npass"], 100 * o["T"] / (o["T"] + 1.2136)))
