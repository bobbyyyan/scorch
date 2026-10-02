"""Plants and ablations for floor_design.

A plant shows a check is REACHED on input of the shape it will see. An ablation shows it is
NECESSARY by feeding it the defect and requiring a refusal that names the defect. The ablation
that matters here is the cyclic step-4 schedule: that is the design a live run was executing while
an earlier version of this check printed "BALANCED", so it is the one input the guard must reject.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floor_design as FD

N_PLANTS = 0
N_ASSERT = 0
FAIL = []


def plant(label):
    global N_PLANTS
    N_PLANTS += 1
    return label


def eq(label, got, want):
    global N_ASSERT
    N_ASSERT += 1
    if got != want:
        FAIL.append("%s: got %r, want %r" % (label, got, want))


def istrue(label, got):
    eq(label, bool(got), True)


def mentions(label, text, needle):
    global N_ASSERT
    N_ASSERT += 1
    if needle not in text:
        FAIL.append("%s: %r not found in %r" % (label, needle, text[:400]))


def cyclic(M, S, K):
    return {m: [((m - K * s) % M) for s in range(S)] for m in range(M)}


# ---------------------------------------------------------------- the real schedule
plant("the 16/4 schedule this run uses is balanced on both axes")
pos, how = FD.schedule(16, 4)
a = FD.audit(pos, 16, 4)
eq("real: failures", FD.failures(a), [])
eq("real: distinct position sums", a["distinct_sums"], [30])
eq("real: worst within-class range", a["worst_within"], 0.0)
eq("real: between-class range", a["between"], 0.0)
istrue("real: every slice a permutation", a["every_slice_a_permutation"])
istrue("real: every member visits every block", a["every_member_visits_every_block"])
eq("real: leader classes", a["leader_classes"], [0, 16, 32, 48])
mentions("real: how names the construction", how, "Latin-square")

# ---------------------------------------- ABLATION: the design that was actually running
plant("ABLATION cyclic step 4 -- the schedule a live run executed under a 'BALANCED' banner")
a4 = FD.audit(cyclic(16, 4, 4), 16, 4)
eq("step4: worst within-class range is 0, which is why the one-axis check passed",
   a4["worst_within"], 0.0)
eq("step4: between-class range", a4["between"], 3.0)
eq("step4: class means", {c: round(v, 2) for c, v in sorted(a4["class_mean"].items())},
   {0: 9.0, 16: 6.0, 32: 7.0, 48: 8.0})
eq("step4: distinct position sums", a4["distinct_sums"], [24, 28, 32, 36])
istrue("step4: every slice is still a permutation", a4["every_slice_a_permutation"])
istrue("step4: every member still visits every block", a4["every_member_visits_every_block"])
f4 = FD.failures(a4)
eq("step4: two independent named failures", len(f4), 2)
mentions("step4: names unequal sums", " ".join(f4), "position sums are not all equal")
mentions("step4: carries the between-class size", " ".join(f4), "between-class range 3.00")
# The second defect is the one no mean-position check could see: the leader of slice s is member
# 4s mod 16, so {0,4,8,12} -- all class 16 -- take every cold-cache first position.
eq("step4: every leader is the same class", a4["leader_classes"], [16])
eq("step4: and they are the four members of that class", a4["leaders"], [0, 4, 8, 12])
mentions("step4: names the lead collapse", " ".join(f4), "running first")

plant("ABLATION cyclic step 1 -- unbalanced on BOTH axes")
a1 = FD.audit(cyclic(16, 4, 1), 16, 4)
eq("step1: worst within-class range", a1["worst_within"], 12.0)
istrue("step1: it is refused", len(FD.failures(a1)) >= 1)

plant("ABLATION a schedule whose slices are not permutations")
bad = {m: [0] * 4 for m in range(16)}
ab = FD.audit(bad, 16, 4)
eq("notperm: sums are equal, so the sum check alone would pass it", ab["distinct_sums"], [0])
istrue("notperm: the permutation flag is False", not ab["every_slice_a_permutation"])
fb = FD.failures(ab)
mentions("notperm: refusal names the duplication", " ".join(fb), "runs twice in a slice")

plant("ABLATION equal sums but a member never leaves two blocks")
# Member 0 sits at 0,1 and member 1 at 14,15 twice over; sums stay 30 by pairing 0+1+14+15.
eqsum = {m: list(FD.schedule(16, 4)[0][m]) for m in range(16)}
eqsum[0] = [0, 1, 14, 15]
eqsum[1] = [15, 14, 1, 0]
ae = FD.audit(eqsum, 16, 4)
eq("blocks: sums still include 30 for the altered members", sum(eqsum[0]), 30)
istrue("blocks: the block check is False", not ae["every_member_visits_every_block"])
mentions("blocks: refusal names the span", " ".join(FD.failures(ae)), "never runs early")

plant("ABLATION leaders confined to fewer classes")
lead = {m: list(FD.schedule(16, 4)[0][m]) for m in range(16)}
al = FD.audit(lead, 16, 4)
eq("leaders: the real schedule covers all 4", len(al["leader_classes"]), 4)
# Force every slice's position 0 onto class 16 by rotating within-class members into the lead.
forced = {m: [(p + 0) for p in lead[m]] for m in range(16)}
for s in range(4):
    cur = [m for m in range(16) if forced[m][s] == 0][0]
    tgt = [0, 4, 8, 12][s]
    if cur != tgt:
        pc, pt = forced[cur][s], forced[tgt][s]
        forced[cur][s], forced[tgt][s] = pt, pc
af = FD.audit(forced, 16, 4)
eq("leaders: forced leader classes", af["leader_classes"], [16])
mentions("leaders: refusal names the lead", " ".join(FD.failures(af)), "leading position is taken")

# ---------------------------------------------------------------- generality, not just 16/4
plant("a different square case, 9 members in 3 slices, is balanced too -- and 9/3 is the case "
      "that forced the Latin square's sign, since the other sign collapses its leaders to one class")
p9, how9 = FD.schedule(9, 3)
a9 = FD.audit(p9, 9, 3)
eq("9/3: failures", FD.failures(a9), [])
eq("9/3: one distinct sum", len(a9["distinct_sums"]), 1)

plant("25 members in 5 slices has equal sums yet is still refused, so the leader axis is not "
      "decoration -- it rejects a schedule the sum check passes")
p25, _h25 = FD.schedule(25, 5)
a25 = FD.audit(p25, 25, 5)
eq("25/5: sums are equal", len(a25["distinct_sums"]), 1)
istrue("25/5: yet refused", len(FD.failures(a25)) >= 1)
mentions("25/5: on the leader axis", " ".join(FD.failures(a25)), "leading position")

plant("the other Latin-square sign is what the leader check was added to catch")
alt = {}
for m in range(9):
    a_, b_ = divmod(m, 3)
    alt[m] = [3 * ((a_ + s) % 3) + ((b_ + s) % 3) for s in range(3)]
aalt = FD.audit(alt, 9, 3)
eq("altsign: equal sums, so the sum check passes it", len(aalt["distinct_sums"]), 1)
eq("altsign: leaders collapse to one class", aalt["leader_classes"], [16])
istrue("altsign: refused", len(FD.failures(aalt)) >= 1)

plant("a non-square divisor count is refused, not silently run")
p8, how8 = FD.schedule(8, 4)
a8 = FD.audit(p8, 8, 4)
istrue("8/4: refused", len(FD.failures(a8)) >= 1)
mentions("8/4: the reason names the missing condition", how8, "M = S*S does not hold")

plant("a slice count that does not divide the members has no schedule at all")
p5, how5 = FD.schedule(16, 5)
eq("16/5: no schedule", p5, None)
mentions("16/5: the reason names divisibility", how5, "does not divide")

plant("zero members is refused rather than producing an empty, vacuously-passing audit")
p0, how0 = FD.schedule(0, 0)
eq("0/0: no schedule", p0, None)

# ---------------------------------------------------------------- the emitted order IS the audited one
plant("--emit writes the order bash runs, and it reconstructs the audited positions")
d = tempfile.mkdtemp(prefix="floordesign")
path = os.path.join(d, "sched.txt")
rc = FD.main(["floor_design.py", "16", "4", "--emit", path])
eq("emit: main returned success", rc, 0)
lines = [ln.split() for ln in open(path).read().strip().split("\n")]
eq("emit: one line per slice", len(lines), 4)
back = {}
for ln in lines:
    s = int(ln[0]) - 1
    order = [int(x) for x in ln[1:]]
    eq("emit: slice %d lists every member once" % (s + 1), sorted(order), list(range(16)))
    for position, m in enumerate(order):
        back.setdefault(m, [None] * 4)[s] = position
eq("emit: reconstructed positions equal the audited schedule", back, FD.schedule(16, 4)[0])
eq("emit: reconstructed audit has no failures", FD.failures(FD.audit(back, 16, 4)), [])

plant("the command line refuses the unbuildable case with a nonzero exit")
out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "floor_design.py"), "16", "5"],
                     capture_output=True, text=True)
istrue("cli: nonzero exit", out.returncode != 0)
mentions("cli: refusal on the exit message", out.stdout + out.stderr, "REFUSE")

plant("the report prints BOTH axes, which the earlier one-axis version did not")
import io
buf = io.StringIO()
FD.report(16, 4, cyclic(16, 4, 4), "cyclic step 4", a4, stream=buf)
text = buf.getvalue()
mentions("report: names the within axis", text, "WITHIN-class range")
mentions("report: names the between axis", text, "BETWEEN-class range")
mentions("report: prints the between value that the old check missed", text, "BETWEEN-class range 3.00")

plant("plant-count floor, so plants appended after the entry point cannot go undefined")
if N_PLANTS < 16:
    FAIL.append("only %d plants ran; the file is meant to hold at least 16" % N_PLANTS)
N_ASSERT += 1

print("%d plants, %d assertions" % (N_PLANTS, N_ASSERT))
if FAIL:
    print("FAILURES (%d):" % len(FAIL))
    for f in FAIL:
        print("  " + f)
    sys.exit(1)
print("OK")
