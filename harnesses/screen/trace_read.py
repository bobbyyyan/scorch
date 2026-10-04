#!/usr/bin/env python3
"""Summarise probe_steal.py traces: per cell, regime and arm, medians over calls of where the
region's time goes. All times in microseconds from the moment the launch began.
  wall      launch to return                 last_in   latest worker entry
  skew      latest minus earliest entry      own_max   longest own-range time (entry -> own done)
  own_mean  mean own-range time               after     mean time from own done to exit (search + stolen rows)
  last_out  latest exit                       join      wall minus last_out (barrier + return)
  stolen%   rows processed by a thief         steals, scanned  per call, all workers
usage: trace_read.py <trace.csv> [<ab.csv>]"""
import csv, sys, statistics as st, collections
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(sys.argv[1])):
    rows[(r["corpus"], r["key"], int(r["k"]), r["regime"], int(r["steal"]))][int(r["call"])].append(r)
ab = {}
if len(sys.argv) > 2:
    for r in csv.DictReader(open(sys.argv[2])):
        ab[(r["corpus"], r["key"], int(r["k"]), r["arm"])] = float(r["us"])
med = lambda xs: st.median(xs) if xs else float("nan")
hdr = "%-5s %-5s %4s %7s %7s %6s %7s %8s %7s %7s %6s %7s %6s %7s" % ("regim", "steal", "nt", "wall", "last_in", "skew", "own_max", "own_mean", "after", "last_out", "join", "stolen%", "steals", "scanned")
last = None
for key in sorted(rows):
    c, k_, k, regime, steal = key
    if (c, k_, k) != last:
        last = (c, k_, k)
        extra = ""
        if ab:
            extra = "  A/B us: on_warm %.1f off_warm %.1f | on_cold %.1f off_cold %.1f" % tuple(ab.get((c, k_, k, a), float("nan")) for a in ("on_warm", "off_warm", "on_cold", "off_cold"))
        print("\n== %s %s k=%d%s" % (c, k_[-50:], k, extra)); print(hdr)
    per = collections.defaultdict(list)
    for call, rs in rows[key].items():
        wall = [float(r["t_entry"]) for r in rs if int(r["worker"]) == -1]
        ws = [r for r in rs if int(r["worker"]) >= 0]
        if not ws or not wall: continue
        e = [float(r["t_entry"]) for r in ws]; o = [float(r["t_own"]) for r in ws]; x = [float(r["t_exit"]) for r in ws]
        total = sum(int(r["rows_own"]) + int(r["rows_stolen"]) for r in ws)
        per["wall"].append(wall[0]); per["last_in"].append(max(e)); per["skew"].append(max(e) - min(e))
        per["own_max"].append(max(oo - ee for oo, ee in zip(o, e))); per["own_mean"].append(st.mean(oo - ee for oo, ee in zip(o, e)))
        per["after"].append(st.mean(xx - oo for xx, oo in zip(x, o))); per["last_out"].append(max(x)); per["join"].append(wall[0] - max(x))
        per["stolen"].append(100.0 * sum(int(r["rows_stolen"]) for r in ws) / max(total, 1))
        per["steals"].append(sum(int(r["steals"]) for r in ws)); per["scanned"].append(sum(int(r["scanned"]) for r in ws)); per["nt"].append(len(ws))
    u = lambda n: med(per[n]) / 1e3
    print("%-5s %-5s %4d %7.2f %7.2f %6.2f %7.2f %8.2f %7.2f %7.2f %6.2f %7.1f %6.0f %7.0f" % (regime, "on" if steal else "off", med(per["nt"]), u("wall"), u("last_in"), u("skew"), u("own_max"), u("own_mean"), u("after"), u("last_out"), u("join"), med(per["stolen"]), med(per["steals"]), med(per["scanned"])))
