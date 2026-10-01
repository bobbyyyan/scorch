#!/usr/bin/env python3
"""Verify every probe variant's HOT LOOP differs from V0's in the intended way and ONLY that way.

The pre-registration makes this binding: "A variant whose assembly does not match its intended
delta is a refusal, not a data point." The plant here is the object code and the planter is the
optimiser -- a dead load and a dead ALU op are both removable at -O3, and a vacuous V5 reads
identical to V1, which is the signature that would be MISREAD as confirming the load-port lever.
"""
import re, sys, collections

ASM = sys.argv[1]
FUNCS = {"_Z2v0RK4Args": "V0", "_Z3v0bRK4Args": "V0b", "_Z2v1RK4Args": "V1",
         "_Z2v2RK4Args": "V2", "_Z2v3RK4Args": "V3", "_Z2v4RK4Args": "V4",
         "_Z3v4bRK4Args": "V4b", "_Z3v4cRK4Args": "V4c", "_Z2v5RK4Args": "V5",
         "_Z2v6RK4Args": "V6", "_Z2v7RK4Args": "V7",
         "_Z2v8RK4Args": "V8", "_Z2v9RK4Args": "V9"}

# Parse into functions.
body, cur = collections.defaultdict(list), None
for ln in open(ASM):
    m = re.match(r"^([0-9a-f]+) <(\S+)>:", ln)
    if m:
        cur = FUNCS.get(m.group(2))
        continue
    if cur is None:
        continue
    m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", ln.rstrip())
    if m:
        body[cur].append((int(m.group(1), 16), m.group(2), m.group(3).split("#")[0].strip()))

MEMOP = re.compile(r"-?(0x[0-9a-f]+)?\(%[a-z0-9]+(,%[a-z0-9]+(,[0-9])?)?\)")
# `lea` and the multi-byte alignment NOPs use the SAME (%reg,%reg,scale) syntax as a memory
# operand but touch no memory: lea is address arithmetic, nop is padding. Counting them inflated
# mem/nnz and is why the table did not compose -- V4b read 3.12 where V0 + an ALU accumulate must
# be 3.00, and V6/V7 read 3.38 where they emit exactly V0's three operations. Their excess was
# 2 lea + 1 nopl = 3 per 8 FMAs = 0.375/nnz, which is the whole discrepancy.
NOTMEM = ("lea", "nop")


EPILOGUE = ("ret", "pop", "push", "call")


def spans(ins):
    """Every backward-jump span in the function, as (target, jump) address pairs."""
    addrs = {a for a, _, _ in ins}
    out = set()
    for a, mn, ops in ins:
        if not mn.startswith("j"):
            continue
        m = re.match(r"^([0-9a-f]+)", ops)
        if not m:
            continue
        t = int(m.group(1), 16)
        if t in addrs and t < a:
            out.add((t, a))
    return sorted(out, key=lambda s: s[1] - s[0])


def hot_loop(ins):
    """The INNERMOST loop that contains every FMA -- not the largest backward jump.

    THE DEFECT THIS REPLACES, and it was live. The previous version took the body of the
    LARGEST backward jump. For V0/V1/V5 that is the vector loop and the answer was right. For
    V6 and V7 -- the 4- and 8-chain candidates -- the largest backward jump spans almost the
    whole function, so the "hot loop" came back as 302 and 315 bytes containing `pop`, `ret`,
    the horizontal accumulator fold-down (`vshufps`/`vunpckhps`/`vaddps`/`vaddss`) and the
    scalar remainder: 72 and 73 instructions where the real loops are 37 and 36. A span
    containing `ret` is not a loop.

    Why the registered checks survived it anyway, which is luck and not design: the extra span
    contributes zero memory operands, zero broadcasts and zero `vpor`, so mem/nnz, bcast/nnz
    and vpor/nnz all read correctly at 3.00/1.00/0.00. Only the instruction count was
    contaminated -- and that is the axis a reader would use to ask whether a candidate kernel
    got bigger. 9.000 insn/nnz on a 6-wide core is 1.50 cycles/nnz, which is close enough to
    the measured 1.57 to have been adopted as a mechanism. The real figure is 4.625, giving
    0.77, which rules that mechanism out. A guard whose guarantee is parameterised by the
    control-flow shape of its input degraded silently on exactly the two variants it had never
    seen.

    So: enumerate the backward-jump spans, DISCARD any that contains a function epilogue or a
    call, and among the rest take the SMALLEST that attains the maximum FMA count. Smallest
    rather than largest is what makes it the innermost loop; the FMA-count tie-break is what
    stops a peeled or remainder loop with fewer FMAs from being chosen over the main one.
    """
    cand = [(t, a) for t, a in spans(ins)
            if not any(mn.startswith(EPILOGUE) for x, mn, _ in ins if t <= x <= a)]
    if not cand:
        return []
    def body(s):
        return [x for x in ins if s[0] <= x[0] <= s[1]]
    nf = {s: sum(1 for _, mn, _ in body(s) if mn.startswith("vfmadd")) for s in cand}
    best = max(nf.values())
    if best == 0:
        return []
    # `spans` is already sorted by size, so the first match is the smallest.
    return body(next(s for s in cand if nf[s] == best))


def profile(ins):
    loop = hot_loop(ins)
    p = collections.Counter()
    for _, mn, ops in loop:
        has_mem = bool(MEMOP.search(ops)) and not mn.startswith(NOTMEM)
        p["insn"] += 1
        if has_mem:
            p["mem"] += 1
        if mn.startswith("vfmadd"):
            p["fma"] += 1
            if has_mem:
                p["fma_mem"] += 1
        elif mn.startswith("vbroadcast") or mn == "vpbroadcastd":
            p["bcast"] += 1
        elif mn.startswith("vor") or mn.startswith("vpor"):
            p["alu_or"] += 1
        elif mn.startswith("v") and not mn.startswith("vmov"):
            p["valu_other"] += 1
        elif mn.startswith("vmov"):
            p["vmov"] += 1
            if has_mem:
                p["vmov_mem"] += 1
    p["len"] = len(loop)
    return p


P = {k: profile(v) for k, v in body.items()}
print("HOT-LOOP PROFILE (innermost loop containing every FMA; spans with ret/pop/call refused)")
print("  %-5s %6s %6s %5s %7s %7s %7s %7s %8s" %
      ("arm", "insn", "mem", "fma", "fma_mem", "bcast", "vpor", "vmov_m", "unroll"))
print("  " + "-" * 70)
for a in ("V0", "V0b", "V1", "V2", "V3", "V4", "V4b", "V4c", "V5", "V6", "V7", "V8", "V9"):
    if a not in P:
        print("  %-5s MISSING FROM THE DISASSEMBLY" % a); continue
    p = P[a]
    print("  %-5s %6d %6d %5d %7d %7d %7d %7d %8d" %
          (a, p["insn"], p["mem"], p["fma"], p["fma_mem"], p["bcast"],
           p["alu_or"], p["vmov_mem"], p["fma"]))
print()
print("  'unroll' is the FMA count, which equals the nonzeros per loop iteration: the source is")
print("  2-way unrolled and -funroll-loops may multiply it, so every delta below is PER FMA.")
print()

# ---- MACHINE BODY WIDTH IS A CHECKED AXIS, because the per-nonzero counts cannot see it.
#
# WHY IT WAS MISSED. mem/nnz, bcast/nnz and vpor/nnz are precisely the quantities INVARIANT to
# unroll width, so no amount of care with them can detect a compiler-changed body. V8 was written
# as "V6 with the index load removed" -- a delta of one memory operation -- and gcc unrolled its
# `p += 4` FOURFOLD to a sixteen-nonzero body where V6's is eight, because dropping the B pointers
# freed the registers to do it. V8 passed the gate at 2.00/1.00/0.00, exactly as registered, while
# carrying 2.250 insn/nnz against V6's 4.625. Half that reduction is the removed load; the rest is
# loop control de-amortised over sixteen nonzeros instead of eight. The same defect is in V4 and
# V4c, which are four-wide against V4b's eight, and it silently confounded the +15.253% figure
# published as the clean port control.
#
# The gate's own principle already demanded this axis: "a variant whose assembly does not match its
# intended delta is a refusal, not a data point." A doubled unroll is not the intended delta.
#
# RULE: every arm's body width must equal V0's, unless it is listed in BODY_DECLARED with a reason.
# A declared mismatch is printed as a warning on every run rather than silently tolerated, so a
# contrast drawn across one cannot be quoted without the caveat travelling with it.
# RETIRED BY THE PRAGMA, kept as the record of what it was for. V4 and V4c used to run 4-nonzero
# bodies against V4b's 8, because the extra sink's register pressure halved the compiler's chosen
# unroll -- so the +15.253% published as "an added load-port operation with the footprint fixed"
# was that operation PLUS a doubled per-nonzero loop control. Now that every main loop carries
# `#pragma GCC unroll`, all thirteen arms run 8 and the dict is empty. The payoff is visible in the
# table below: V4c - V4b is exactly +1.00 insn/nnz and +1.00 mem/nnz, one instruction and one
# memory operation, which is the clean contrast the arm was built for and never delivered until
# the width was pinned. An entry here would mean an arm whose body width legitimately differs; if
# one is ever added, the warning prints on every run so a contrast drawn across it cannot be
# quoted without the caveat.
BODY_DECLARED = {}

bw0 = P["V0"]["fma"] if "V0" in P else 0
print("MACHINE BODY WIDTH (nonzeros consumed per loop body = FMA count), reference V0 = %d" % bw0)
width_fail, width_warn = [], []
for a in ("V0", "V0b", "V1", "V2", "V3", "V4", "V4b", "V4c", "V5", "V6", "V7", "V8", "V9"):
    if a not in P:
        continue
    w = P[a]["fma"]
    if a in BODY_DECLARED:
        exp, why = BODY_DECLARED[a]
        if w != exp:
            width_fail.append("%s body %d, declared %d" % (a, w, exp))
        else:
            width_warn.append("  %-4s body %d != V0's %d -- DECLARED: %s" % (a, w, bw0, why))
    elif w != bw0:
        width_fail.append("%s body %d != V0's %d, and it is not declared. An unroll change is not "
                          "the intended delta." % (a, w, bw0))
for ln in width_warn:
    print(ln)
if not width_warn:
    print("  every arm matches V0's body width and none is declared different")
if width_fail:
    print("  REFUSE on body width:")
    for f in width_fail:
        print("    %s" % f)
print()

v0 = P.get("V0")
if not v0 or not v0["fma"]:
    sys.exit("REFUSE: V0's hot loop has no FMA; the loop finder did not find the kernel.")


def per(p, key):
    return p[key] / float(p["fma"])


# nnz/body and insn/nnz join the ATTESTED table, not just the diagnostic one above. The three
# original metrics are per-nonzero RATIOS and are therefore invariant to unroll width -- V8 read
# 2.00 mem/nnz at a 16-wide body and V1 read 2.00 at 8-wide. insn/nnz is the one that is NOT
# invariant, because loop control amortises: V4b 5.625 against V4c 7.250 is how the first
# width confound surfaced. So both travel with the timings, and an A/B whose nnz/body differs is
# refused above rather than quietly read.
print("PER-NONZERO COUNTS, and the registered intent of each delta")
print("  %-5s %8s %10s %10s %10s %9s   %s" % ("arm", "nnz/body", "insn/nnz", "mem/nnz",
                                              "bcast/nnz", "vpor/nnz", "verdict"))
print("  " + "-" * 76)
fails = []
# mem/nnz is the load-port figure the design is about. V0 ships at 3.
EXPECT = {"V0":  (3.0, 1.0, 0.0, "baseline: index load + A broadcast + folded B load"),
          "V0b": (3.0, 1.0, 0.0, "A/A floor arm: must equal V0 exactly"),
          "V1":  (2.0, 1.0, 0.0, "index load removed"),
          "V2":  (2.0, 0.0, 0.0, "A broadcast removed (hoisted)"),
          "V3":  (2.0, 1.0, 0.0, "B load removed (register-resident)"),
          "V4":  (4.0, 1.0, 1.0, "positive control: V0 + one extra load, accumulated"),
          "V4b": (3.0, 1.0, 1.0, "V0 + the accumulate ONLY; V4-V4b isolates the load"),
          "V4c": (4.0, 1.0, 1.0, "clean control: +1 load of an already-loaded address"),
          "V5":  (2.0, 1.0, 1.0, "discriminator: V1 + the same accumulate"),
          "V6":  (3.0, 1.0, 0.0, "CANDIDATE: 4 accumulator chains, V0's 3 load ops/nnz"),
          "V7":  (3.0, 1.0, 0.0, "CANDIDATE: 8 accumulator chains, V0's 3 load ops/nnz"),
          "V8":  (2.0, 1.0, 0.0, "4 chains, index load removed -- the port discriminator"),
          "V9":  (2.0, 1.0, 0.0, "4 chains, B register-resident -- dependency vs MLP")}
for a in ("V0", "V0b", "V1", "V2", "V3", "V4", "V4b", "V4c", "V5", "V6", "V7", "V8", "V9"):
    if a not in P:
        fails.append("%s missing" % a); continue
    p = P[a]
    em, eb, eo, why = EXPECT[a]
    gm, gb, go = per(p, "mem"), per(p, "bcast"), per(p, "alu_or")
    bad = []
    if abs(gm - em) > 0.26: bad.append("mem/nnz %.2f != %.1f" % (gm, em))
    if abs(gb - eb) > 0.26: bad.append("bcast/nnz %.2f != %.1f" % (gb, eb))
    if abs(go - eo) > 0.26: bad.append("vpor/nnz %.2f != %.1f" % (go, eo))
    print("  %-5s %8d %10.3f %10.2f %10.2f %9.2f   %s"
          % (a, p["fma"], p["insn"] / float(p["fma"]), gm, gb, go, why))
    if bad:
        fails.append("%s: %s" % (a, "; ".join(bad)))

print()
# V0b must be instruction-identical to V0, or the A/A floor arm is not the same code.
same = (P["V0"]["insn"] == P.get("V0b", {}).get("insn") and
        P["V0"]["mem"] == P.get("V0b", {}).get("mem") and
        P["V0"]["fma"] == P.get("V0b", {}).get("fma"))
print("  V0b hot loop identical to V0 in insn/mem/fma counts: %s" % ("YES" if same else "NO"))
if not same:
    fails.append("V0b is not the same code as V0, so it is not an A/A arm")

# The two refutation-relevant deltas, stated as the design states them.
print()
print("  THE TWO DELTAS THE DESIGN TURNS ON:")
print("    V4 - V0 load-port ops per nonzero = %+.2f   (must be about +1, or the dead load was deleted)"
      % (per(P["V4"], "mem") - per(P["V0"], "mem")) if "V4" in P else "")
print("    V5 - V1 vector ALU ops per nonzero = %+.2f   (must be about +1, or V5 is vacuous and"
      % (per(P["V5"], "alu_or") - per(P["V1"], "alu_or")) if "V5" in P and "V1" in P else "")
print("                                                  reads identical to V1 -- the failure that")
print("                                                  LOOKS like confirming the lever)")
print()
fails = fails + width_fail
if fails:
    print("REFUSE: the emitted code does not match the registered design:")
    for f in fails:
        print("  %s" % f)
    sys.exit(1)
print("ALL VARIANTS MATCH THEIR REGISTERED DELTA. Timing may be read.")
