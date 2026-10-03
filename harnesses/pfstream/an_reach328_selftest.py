"""Counted self-test for an_reach328.py, run BEFORE it scores anything.

Every check is appended to RAN, and the count is printed, so a check deleted or never reached
cannot pass silently -- which is how a planted mutation once survived a suite that reported
"14 PASSED" while only 13 had executed.
"""
import importlib.util as iu
import os
import random
import sys

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "an_reach328.py")
_s = iu.spec_from_file_location("reach", _p)
R = iu.module_from_spec(_s)
_s.loader.exec_module(R)

RAN, FAIL = [], []


def ck(name, cond, detail=""):
    RAN.append(name)
    print("  %-62s %s%s" % (name, "PASS" if cond else "FAIL", ("   " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def synth(n, share_le2, live_frac, seed):
    """n keys with a target live-and-<=2 share, built so the registered numerator is exactly it."""
    rnd = random.Random(seed)
    keys = ["k%04d" % i for i in range(n)]
    res, live = {}, set()
    want = int(round(share_le2 * n))
    for i, k in enumerate(keys):
        if i < want:
            res[k] = 1 + (i % 2)   # <=2
            live.add(k)            # and live: counts in the registered numerator
        else:
            res[k] = 8 + (i % 4)   # >=8
            if rnd.random() < live_frac:
                live.add(k)
    return keys, res, live


print("an_reach328.py self-test")

# 1-2. the registered numerator is what it says it is, and the alternatives differ from it
keys, res, live = synth(100, 0.30, 1.0, 1)
# add 20 NON-LIVE <=2 keys: in the all-<=2 numerator, out of the registered one
for i in range(20):
    k = "nl%02d" % i
    keys.append(k)
    res[k] = 2            # <=2
    # deliberately NOT added to `live`
d = R.shares(keys, res, live)
ck("registered numerator excludes non-live <=2 cells",
   d["n_live_le2"] == 30, "n_live_le2=%d, expected 30" % d["n_live_le2"])
ck("all-<=2 numerator INCLUDES them and is larger",
   d["n_all_le2"] == 50 and d["all_le2_over_all"] > d["registered"],
   "all=%d reg=%.4f alt=%.4f" % (d["n_all_le2"], d["registered"], d["all_le2_over_all"]))

# 3. the denominator is ALL keys of the half, not the live subset
ck("registered denominator is all keys, not the live subset",
   abs(d["registered"] - 30 / 120) < 1e-12,
   "%.6f vs %.6f" % (d["registered"], 30 / 120))

# 4. the inflation factor is the thing amendment 26 measured: here 50/30
ck("the unpinned reading inflates by n_all/n_live_le2",
   abs(d["all_le2_over_all"] / d["registered"] - 50 / 30) < 1e-9,
   "%.4fx" % (d["all_le2_over_all"] / d["registered"]))

# 5. absent keys are counted, not silently dropped
d2 = R.shares(keys + ["missing1", "missing2"], res, live)
ck("absent keys are reported and leave the denominator at the PRESENT count",
   d2["absent"] == 2 and d2["n"] == d["n"], "absent=%d n=%d" % (d2["absent"], d2["n"]))

# 6-7. the bootstrap finds a real difference and its CI excludes zero
ka, ra, la = synth(300, 0.60, 0.9, 11)
kb, rb, lb = synth(300, 0.20, 0.9, 12)
kb = ["b" + x for x in kb]
rb = {"b" + k: v for k, v in rb.items()}
lb = {"b" + k for k in lb}
res2 = dict(ra); res2.update(rb)
live2 = set(la) | lb
b = R.boot_diff(ka, kb, res2, live2, nboot=600, seed=7)
ck("bootstrap CI excludes zero on an injected +0.40 difference",
   b is not None and b[0] > 0, "CI [%+.4f, %+.4f]" % (b[0], b[1]) if b else "None")
ck("bootstrap CI covers the injected difference",
   b is not None and b[0] <= 0.40 <= b[1], "CI [%+.4f, %+.4f]" % (b[0], b[1]) if b else "None")

# 8. no-effect control: equal shares must span zero
kc, rc, lc = synth(300, 0.60, 0.9, 13)
kc = ["c" + x for x in kc]
rc = {"c" + k: v for k, v in rc.items()}
lc = {"c" + k for k in lc}
res3 = dict(ra); res3.update(rc)
live3 = set(la) | lc
b0 = R.boot_diff(ka, kc, res3, live3, nboot=600, seed=7)
ck("no-effect control spans zero", b0 is not None and b0[0] <= 0 <= b0[1],
   "CI [%+.4f, %+.4f]" % (b0[0], b0[1]) if b0 else "None")

# 9-10. digests: the VALUE digest must move with the data, the INDEX digest must not
ck("same n, different data -> DIFFERENT value digest", b[2] != b0[2], "%s vs %s" % (b[2], b0[2]))
ck("same n, same seed -> SAME index digest (so the value digest is the one that binds)",
   b[3] == b0[3], "%s vs %s" % (b[3], b0[3]))

# 11. too-small halves refuse rather than return a narrow interval
ck("a half under 8 present keys returns None",
   R.boot_diff(ka[:5], kc, res3, live3, nboot=50, seed=7) is None)

# 12. keys are the sampling unit: a half of n keys resamples n keys, not n*widths
ck("resampling draws exactly len(half) keys per replicate",
   R.boot_diff(ka, kc, res3, live3, nboot=20, seed=7)[4] == len(ka),
   "nA=%d len=%d" % (R.boot_diff(ka, kc, res3, live3, nboot=20, seed=7)[4], len(ka)))

# 13. an empty half is None, not a zero share
ck("an empty half returns None rather than a 0.0 share", R.shares([], res, live) is None)

# 14. a half with NO live <=2 key reads exactly zero, and says so through n_live_le2
kz, rz, lz = synth(50, 0.0, 1.0, 21)
dz = R.shares(kz, rz, lz)
ck("a half with no live <=2 key reads 0.0 with n_live_le2 = 0",
   dz["registered"] == 0.0 and dz["n_live_le2"] == 0)

print()
print("%d checks ran, %d failed" % (len(RAN), len(FAIL)))
if len(RAN) != len(set(RAN)):
    print("FAIL: duplicate check names, so the count is not a count of distinct checks")
    sys.exit(1)
sys.exit(1 if FAIL else 0)
