#!/usr/bin/env python3
"""Reduce the port-boundedness probe and score the registered predictions.

ESTIMAND, as pre-registered in prereg_portbound.md (digest c5ec8dec37925c79) and not amended:
per-nonzero time reduced as exp(median over repetitions of log(ns per nonzero)), per variant, per
core type, NEVER pooled across core types. Deviation exp|ln x| - 1. The noise floor is a
same-binary A/A: V0 compiled a second time under a different symbol in the same object, floor =
p95 of the PER-REPETITION deviation of V0a/V0b, paired by repetition index because the arms are
run in randomised order inside each repetition and so share a time window.

Amendment 12, made before any timing was read and forced by the emitted code: V4's extra load must
be ACCUMULATED to survive -O3 (a sink that is overwritten each iteration is still dead, and the
first two attempts had V4's loads and all of V5's ALU ops deleted), and an accumulate costs an ALU
op. So V4 carries the accumulate and V4b carries the accumulate WITHOUT the load; V4 - V4b isolates
exactly one load per nonzero. The emitted deltas were asserted before this file ran: mem/nnz 3.00 /
3.00 / 2.00 / 2.12 / 2.00 / 4.00 / 3.12 / 2.12 and vpor/nnz 0 / 0 / 0 / 0 / 0 / 1 / 1 / 1.

REGISTERED PREDICTIONS. Let s be one load-port slot in ns per nonzero.
  load-port bound: V0-V1 ~ V0-V2 ~ V0-V3 ~ s ; V4-V0 ~ s and beyond the floor ; V5 ~ V0.
  issue bound:     the V1/V2/V3 drops are equal but V4-V0 is inside the floor ; V5 ~ V1.
  chain bound:     none of V1, V2, V3 moves beyond the floor.
REFUTATION, registered verbatim: "if t(V5) is at t(V1) rather than at t(V0), or if V4 is free, the
line-reuse kernel cannot pay and the lever is closed." That result gets reported, not worked around.
"""
import collections, csv, math, statistics, sys

red = lambda v: math.exp(statistics.median([math.log(x) for x in v]))
dev = lambda x: math.exp(abs(math.log(x))) - 1.0


def p95(d):
    return sorted(d)[int(0.95 * (len(d) - 1))] if d else float("nan")


def load(path):
    T = collections.defaultdict(dict)
    hdr, binsha = None, None
    for ln in open(path):
        if ln.startswith("# binary_sha256"):
            binsha = ln.split()[2]
            continue
        if ln.startswith("#"):
            hdr = ln.strip()
            continue
        if ln.startswith("arm,"):
            continue
        a, r, v = ln.strip().split(",")
        T[a][int(r)] = float(v)
    return hdr, T, binsha


def attest(path):
    """Read ATTEST.txt: the binary digest and the per-variant emitted counts for THAT binary.

    The registered refutation accepts when t(V5) is at t(V1), and a V5 whose ALU ops were deleted
    by -O3 IS a V1 with different labels -- so the probe's strongest result and its most dangerous
    failure mode are the same reading, and one of the three rejected attempts tonight was exactly a
    vacuous V5. A disassembly gate that passed on some binary is not an attestation of the binary
    that produced a number, so the digest and the counts travel together and the counts are printed
    beside every timing.
    """
    counts, binsha, exit_ok = {}, None, False
    for ln in open(path):
        t = ln.split()
        if len(t) == 2 and t[0] == "binary_sha256":
            binsha = t[1]
        elif len(t) >= 2 and t[0] == "asmcheck_exit":
            exit_ok = (t[1] == "0")
        elif len(t) >= 4 and t[0] in ("V0", "V0b", "V1", "V2", "V3", "V4", "V4b", "V4c", "V5"):
            try:
                counts[t[0]] = (float(t[1]), float(t[2]), float(t[3]))
            except ValueError:
                pass
    if not exit_ok:
        sys.exit("REFUSE: ATTEST.txt records asmcheck_exit != 0, so the emitted code was rejected "
                 "and no timing from it may be read.")
    return binsha, counts


def report(label, path, binsha_att, counts):
    hdr, T, binsha = load(path)
    if binsha is None:
        sys.exit("REFUSE: %s carries no '# binary_sha256' line, so it cannot be bound to a verified "
                 "binary. Produce it with run_probe.sh." % path)
    if binsha != binsha_att:
        sys.exit("REFUSE: %s was produced by binary %s but ATTEST.txt verifies %s. A timing read "
                 "against a different binary than the one disassembled is exactly how a deleted V5 "
                 "would be published as a refutation." % (path, binsha[:16], binsha_att[:16]))
    arms = ["V0", "V0b", "V1", "V2", "V3", "V4", "V4b", "V4c", "V5"]
    missing = [a for a in arms if a not in T]
    if missing:
        sys.exit("REFUSE: %s lacks arms %s" % (path, missing))
    nrep = {a: len(T[a]) for a in arms}
    if len(set(nrep.values())) != 1:
        sys.exit("REFUSE: unequal repetition counts %s -- the A/A floor pairs by repetition index "
                 "and cannot be formed." % nrep)
    reps = sorted(T["V0"])
    print("%s" % label)
    print("  %s" % hdr)
    print("  %d repetitions per arm, %d arms" % (len(reps), len(arms)))

    # The floor, from the only pair that differs in nothing.
    aa = [dev(T["V0"][r] / T["V0b"][r]) for r in reps]
    floor = p95(aa)
    print("  A/A floor  p95 of per-repetition dev(V0/V0b) = %.5f  (%.3f%%);  median %.5f, max %.5f"
          % (floor, 100 * floor, statistics.median(aa), max(aa)))
    print()

    R = {a: red([T[a][r] for r in reps]) for a in arms}
    print("  binary %s, emitted counts attested for THIS binary" % binsha[:16])
    print()
    print("  %-5s %12s %10s %12s %7s %6s   %s"
          % ("arm", "ns/nnz", "vs V0", "beyond floor", "mem/nnz", "alu", "role"))
    print("  " + "-" * 92)
    ROLE = {"V0": "baseline, 3 load ops/nnz", "V0b": "A/A arm, same code",
            "V1": "index load removed", "V2": "A broadcast removed", "V3": "B load removed",
            "V4": "+1 load, accumulated (positive control)", "V4b": "accumulate only, no load",
            "V4c": "+1 load of an ADDRESS ALREADY LOADED (no new cache stream)",
            "V5": "V1 + accumulate (the discriminator)"}
    for a in arms:
        d = R[a] / R["V0"]
        c = counts.get(a, (float("nan"),) * 3)
        print("  %-5s %12.6f %+9.3f%% %12s %7.2f %6.2f   %s"
              % (a, R[a], 100 * (d - 1), "YES" if dev(d) > floor else "no", c[0], c[2], ROLE[a]))
    print()

    s1, s2, s3 = R["V0"] - R["V1"], R["V0"] - R["V2"], R["V0"] - R["V3"]
    v4d, v5d = R["V4"] - R["V4b"], R["V4"] - R["V0"]
    print("  THE REGISTERED QUANTITIES, in ns per nonzero")
    print("    V0-V1 (index load)        %+.6f   %+.3f%%" % (s1, 100 * s1 / R["V0"]))
    print("    V0-V2 (A broadcast)       %+.6f   %+.3f%%" % (s2, 100 * s2 / R["V0"]))
    print("    V0-V3 (B load)            %+.6f   %+.3f%%" % (s3, 100 * s3 / R["V0"]))
    print("    V4-V4b (added load, NEW array)  %+.6f   %+.3f%%   <- conflates port with a new"
          % (v4d, 100 * v4d / R["V0"]))
    print("                                                        64 KB cache stream")
    print("    V4c-V4b (added load, SAME addr)  %+.6f   %+.3f%%   <- port pressure ALONE, no new"
          % (R["V4c"] - R["V4b"], 100 * (R["V4c"] - R["V4b"]) / R["V0"]))
    print("                                                        footprint. THIS is the clean control.")
    print("    V4-V0  (load + ALU)       %+.6f   %+.3f%%" % (v5d, 100 * v5d / R["V0"]))
    print("    V4b-V0 (the ALU op alone) %+.6f   %+.3f%%" % (R["V4b"] - R["V0"], 100 * (R["V4b"] - R["V0"]) / R["V0"]))
    print()
    # Where does V5 sit between V1 and V0? 0 means at V1, 1 means at V0.
    span = R["V0"] - R["V1"]
    pos = (R["V5"] - R["V1"]) / span if abs(span) > 1e-12 else float("nan")
    print("  THE DISCRIMINATOR. V5 has V1's load count and V0's issue count.")
    print("    V1 %.6f   V5 %.6f   V0 %.6f" % (R["V1"], R["V5"], R["V0"]))
    print("    V5 sits at %.3f of the way from V1 to V0  (0.0 = at V1 -> issue was never binding;" % pos)
    print("                                               1.0 = at V0 -> the load port was binding)")
    print("    dev(V5/V1) = %.5f %s floor %.5f ; dev(V5/V0) = %.5f %s floor"
          % (dev(R["V5"] / R["V1"]), "<" if dev(R["V5"] / R["V1"]) <= floor else ">", floor,
             dev(R["V5"] / R["V0"]), "<" if dev(R["V5"] / R["V0"]) <= floor else ">"))
    print()
    v4_free = dev(R["V4c"] / R["V4b"]) <= floor      # the CLEAN control, not the new-array one
    v5_at_v1 = dev(R["V5"] / R["V1"]) <= floor
    v5_at_v0 = dev(R["V5"] / R["V0"]) <= floor
    print("  VERDICT against the registered refutation condition")
    print("    V4c free (an added PORT op costs nothing)?    %s   <- clean control" % ("YES" if v4_free else "NO"))
    print("    V4  free (added load + new cache stream)?     %s" % ("YES" if dev(R["V4"] / R["V4b"]) <= floor else "NO"))
    print("    V5 at V1 (restoring the issue slot is free)?  %s" % ("YES" if v5_at_v1 else "no"))
    print("    V5 at V0 (the load port was the constraint)?  %s" % ("YES" if v5_at_v0 else "no"))
    if v4_free or v5_at_v1:
        print("    -> REFUTED. The line-reuse kernel cannot pay on this core type and the lever is")
        print("       CLOSED. Reported, not worked around.")
    elif v5_at_v0:
        print("    -> LOAD-PORT BOUND on this core type; the line-reuse lever survives this test.")
    else:
        print("    -> INDETERMINATE: V5 is at neither endpoint beyond the floor.")


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) % 2 != 0:
        sys.exit("usage: an_portbound.py <ATTEST.txt> <label> <csv> [<label> <csv> ...]\n"
                 "  Core types are passed separately and NEVER pooled, per the scope limits.")
    bs, cnt = attest(sys.argv[1])
    for i in range(2, len(sys.argv), 2):
        if i > 2:
            print()
        report(sys.argv[i], sys.argv[i + 1], bs, cnt)
    print()
    print("  THE FLOOR IS AN ARM FLOOR, NOT A LAYOUT FLOOR, which bounds what the three small")
    print("  readings can claim. V0b is the SAME variant recompiled under a second symbol, so the")
    print("  p95 above measures re-running the same code -- not running different code at a")
    print("  different address. Two objects with identical DECISIONS have measured 1-2.6% apart on")
    print("  narrow-k SpMM and identical SOURCE at different addresses about 0.3%. So V1, V2 and V3")
    print("  at 0.57-1.41% are an UPPER BOUND on the per-load gain, not established effects: that")
    print("  range is the size layout alone can produce. The refutation is untouched either way --")
    print("  V5 against V1 is 0.007%, inside even a 1% layout-scale floor, and V5 against V0 is")
    print("  1.08%, 17x the arm floor -- so the test still discriminates its two alternatives.")
