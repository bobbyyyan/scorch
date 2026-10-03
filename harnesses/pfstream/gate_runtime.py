"""Runtime cost of the SpMM work gate (scorch_policy.h:2618), arm-vs-arm on ONE object.

THE CLAIM UNDER TEST is the shipped comment at scorch_policy.h:1575-1579, which says the
gate exists because the row partition's bookkeeping is not amortised on short kernels:
"the cells where it falls more than 10% behind the shared counter sit at 19-30
microseconds where the cells it wins sit at 31-110". The constant is 2 grains. Every
earlier probe on this concern measured DECISIONS -- which cells the predicate selects --
and none measured time. This measures time.

WHY ONE PROCESS AND NOT TWO BUILDS. The getenv at :2532 and :2599 sits inside the
function, so the value is read per call and an arm is an env var, not a binary. Flipping
os.environ between two timings of the same cell holds the object, the page placement, the
thread binding and the CPU's frequency state fixed in a way two processes cannot. The
object's digest is recorded because it is the one thing that cannot be recovered later.

WHY THE PRIMARY READOUT IS ABSOLUTE MICROSECONDS AND NOT A RATIO. This must be a hooks
object for the env vars to be read at all, and a hooks object charges ~17-26 us of getenv
per call under Slurm on kernels the comment above puts at 19-110 us. An additive per-call
tax T cancels exactly in a difference -- (t_a + T) - (t_b + T) = t_a - t_b -- and biases a
ratio toward 1, since (t_a + T)/(t_b + T) is closer to unity than t_a/t_b. So the
difference is unbiased by the tax and the ratio is attenuated by it. Both are printed, the
ratio is labelled, and the tax is estimated from the intercept so the attenuation has a
size rather than a hand-wave. A comparison against MKL would be invalid here for the same
reason and is not made: both arms are ours.

THE FLOOR IS MEASURED, NOT ASSUMED, AND THE GEOMETRY SUPPLIES IT FOR FREE. The gate is
`nnz * max(k,16) < 2 * 150000`. A matrix with nnz in [9375, 18750) is INSIDE the region at
k=16 (nnz*16 < 300000) and OUTSIDE it at k=32 (nnz*32 >= 300000). So the same matrix is a
treatment cell at one width and an A/A cell at the other: same object, same env-var count,
and -- checked below rather than asserted -- the same resolved decision in both arms. A
second control holds width fixed and moves nnz just above the threshold instead. Two
controls matched on different axes bound the noise better than either alone.
"""
import argparse
import csv
import hashlib
import math
import os
import random
import statistics
import sys
import time
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("root", help="cache root holding the .npz operands")
ap.add_argument("--csv", required=True)
ap.add_argument("--pool", type=int, default=8)
ap.add_argument("--k", type=int, default=16, help="treatment width; the gate's proxy is "
                "nnz*max(k,16), so membership is identical for every k <= 16 while the "
                "kernel selected is not -- which is why this is swept")
ap.add_argument("--nproc-cap", type=int, default=0,
                help="restrict this process's CPU affinity to N of the allocated CPUs "
                     "BEFORE importing torch. Slurm's -c is a request and it rounds up to "
                     "a whole core, so -c 15 on an SMT-2 host yields 16 and lands exactly "
                     "ON 2*pool instead of below it; setting the affinity here is exact.")
ap.add_argument("--allow-adoption", action="store_true",
                help="permit omp_get_num_procs() >= 2*pool. The isolated configuration "
                     "forbids it so the gate is the only thing moving, but a user at pool "
                     "8 on a 128-CPU host HAS it live, so the representative reading needs "
                     "this flag and gets labelled with it.")
ap.add_argument("--ncell", type=int, default=220, help="matrices sampled per population")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--target-ms", type=float, default=12.0)
ap.add_argument("--seed", type=int, default=20251002)
A = ap.parse_args()

# BEFORE the heavy imports: omp_get_num_procs() reads the cpuset, and libgomp latches
# parts of its view at load time, so the cap has to be in place first.
if A.nproc_cap:
    have = sorted(os.sched_getaffinity(0))
    if A.nproc_cap > len(have):
        sys.exit("REFUSE: asked to cap at %d CPUs but the allocation holds %d"
                 % (A.nproc_cap, len(have)))
    os.sched_setaffinity(0, set(have[:A.nproc_cap]))
    print("affinity capped to %d of %d allocated CPUs"
          % (len(os.sched_getaffinity(0)), len(have)))

import ctypes as _ct
import glob
import zipfile
import numpy as np
import torch
import scorch_ops as so

# IS THE CACHING getenv SHIM LIVE, AND DID IT ACTUALLY CACHE. A hooks object performs 99
# getenv per SpMM call, measured by interposition, which costs ~20 us per call end-to-end
# on this host -- comparable to the whole kernel on the cells this gate selects. That tax
# cancels in an A-minus-B difference and does NOT cancel in the cell scale, so comparing a
# cell against the microsecond bands in the comment at :1577 requires it gone. The shim
# returns the same pointer the real getenv would and flushes on any write to the
# environment, so decisions are unchanged and an arm flip is still seen; a release object
# performs no getenv at all, so a shimmed hooks object is CLOSER to what ships.
#
# The counters are read rather than the shim's presence assumed: LD_PRELOAD silently does
# nothing if the path is wrong, and a tax-free claim resting on an inactive shim would read
# as a plausible number.
SHIM = None
try:
    _c = _ct.CDLL("libfastenv.so")
    for _f in ("scorch_fastenv_real", "scorch_fastenv_hits", "scorch_fastenv_flushes"):
        getattr(_c, _f).restype = _ct.c_long
    SHIM = _c
except Exception:                                                      # noqa: BLE001
    pass
# The shim's digest, for the same reason the object's is recorded: a run that mutates a
# shared .so while other runs read it leaves no trace in the output. One grid was discarded
# for exactly that -- the library and the harness were overwritten mid-flight, so early
# widths used one version and later widths another, with nothing in the data saying which.
SHIM_SHA = "none"
if SHIM:
    import glob as _g
    for _c in (os.environ.get("LD_PRELOAD", "").split(":") + ["libfastenv.so"]):
        if _c and os.path.exists(_c):
            SHIM_SHA = hashlib.sha256(open(_c, "rb").read()).hexdigest()[:16]
            break
    print("shim object sha %s" % SHIM_SHA)
print("getenv shim: %s" % ("LIVE, so readings are tax-free and the cell scale is "
      "comparable to the comment's bands" if SHIM else "ABSENT, so every reading carries "
      "~20 us of getenv; A-B is still valid, the cell SCALE is not"), flush=True)

GRAIN = 150000
THRESH = 2 * GRAIN                      # the shipped constant, pinned to exactly this
SO_SHA = hashlib.sha256(open(so.__file__, "rb").read()).hexdigest()[:16]
print("object: %s  sha %s" % (so.__file__, SO_SHA), flush=True)

# ---------------------------------------------------------------- premise 1, the object
if not so.scorch_tune_hooks():
    sys.exit("REFUSE: not a hooks object. The getenv blocks at scorch_policy.h:2532 and "
             ":2599 are inside #ifdef SCORCH_TUNE_HOOKS, so on this object both arms are "
             "the compiled default and the experiment would read a flat zero that looks "
             "exactly like a null result.")
print("hooks object: env arms are live", flush=True)

torch.set_num_threads(A.pool)
POOL = torch.get_num_threads()
if POOL != A.pool:
    sys.exit("REFUSE: asked for pool %d, torch resolved %d" % (A.pool, POOL))

# ------------------------------------------------- premise 2, the other two predicates
# The pool ceiling at :2614 zeroes mingrains when pool > gate_maxthreads, which would
# switch the gate off in BOTH arms and make the treatment population empty-by-construction.
# And concern #1's adoption path needs omp_get_num_procs() >= 2*pool, so the allocation
# must be narrow enough that it cannot fire alongside what is being measured. Both are
# arithmetic on numbers this process can read, so both are checked rather than assumed.
try:
    import ctypes
    NPROC = ctypes.CDLL(None).omp_get_num_procs()
except Exception:                                                      # noqa: BLE001
    NPROC = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else -1
    print("NOTE omp_get_num_procs() unavailable; using affinity %d" % NPROC)
print("pool %d   omp_get_num_procs() %d   2*pool %d" % (POOL, NPROC, 2 * POOL), flush=True)
ADOPTION = NPROC >= 2 * POOL
if ADOPTION and not A.allow_adoption:
    sys.exit("REFUSE: omp_get_num_procs() %d >= 2*pool %d, so the 2x-pool adoption path "
             "(concern #1, scorch_policy.h:2313) is REACHABLE in this allocation and would "
             "move threads underneath the gate being measured. Allocate fewer CPUs than "
             "twice the pool, or pass --allow-adoption to measure the representative "
             "configuration instead of the isolated one." % (NPROC, 2 * POOL))
print("CONFIGURATION: %s" % ("REPRESENTATIVE -- the 2x-pool adoption path IS reachable "
      "here, so an effect is the gate AND adoption together and is not attributable to "
      "the gate alone" if ADOPTION else "ISOLATED -- adoption cannot fire, so an effect "
      "is attributable to the gate; this is not a user's configuration on a wide host"),
      flush=True)
GATE_MAXTHREADS = 16
CEILING = POOL > GATE_MAXTHREADS
if CEILING:
    # Not a refusal. Above the ceiling :2614 zeroes mingrains, so R1's two arms set the same
    # variable to values that resolve identically -- a DECLINED ARGUMENT, which is the one
    # floor construction that sits on the treatment's OWN CELLS at the same nproc instead of
    # beside them on a different population. R2 forces the mode directly and is unaffected,
    # so the pool axis stays measurable above the ceiling while R1 turns into its own
    # control. An earlier version exited here and gave up both.
    print("pool %d is ABOVE the ceiling %d, so :2614 zeroes mingrains in both of R1's arms. "
          "R1 is therefore a DECLINED-ARGUMENT FLOOR on the treatment's own cells here, not "
          "a treatment; R2 forces the mode directly and remains live."
          % (POOL, GATE_MAXTHREADS), flush=True)
else:
    print("pool %d is at or below the ceiling %d, so the work gate is ACTIVE and R1 is a "
          "treatment" % (POOL, GATE_MAXTHREADS), flush=True)


def dims(path):
    """(nnz, rows, shape) from the .npy headers -- exact, and without decompressing."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "indices.npy" not in names or "indptr.npy" not in names:
                return None
            h = {}
            for m in ("indices.npy", "indptr.npy"):
                with z.open(m) as fh:
                    if np.lib.format.read_magic(fh) != (1, 0):
                        return None
                    h[m] = np.lib.format.read_array_header_1_0(fh)[0]
            if "shape.npy" not in names:
                return None
            with z.open("shape.npy") as fh:
                shape = tuple(int(x) for x in np.lib.format.read_array(fh))
    except Exception:                                                  # noqa: BLE001
        return None
    if len(shape) != 2:
        return None
    return int(h["indices.npy"][0]), int(h["indptr.npy"][0]) - 1, shape


files = sorted(glob.glob(os.path.join(A.root, "**", "*.npz"), recursive=True))
print("cache %s: %d .npz files" % (A.root, len(files)), flush=True)
scan = []
for p in files:
    d = dims(p)
    if d is None:
        continue
    nnz, rows, shape = d
    if min(shape) == 1:          # non-operator: a right-hand side or a single-input map
        continue
    scan.append((p, os.path.basename(p)[:-4], nnz, rows, shape))
print("readable operands: %d" % len(scan), flush=True)
if len(scan) < 0.9 * len(files):
    sys.exit("REFUSE: %d operands from %d files. A silently shorter corpus is a different "
             "experiment." % (len(scan), len(files)))

# --------------------------------------------------------------------- the populations
MAXOUT = 2 * so.scorch_llc_bytes()


def inregion(nnz, rows, k):
    return nnz * max(k, 16) < THRESH and rows * k * 4 < MAXOUT


KT, KC = A.k, 32
if KT >= KC:
    sys.exit("REFUSE: treatment width %d must be below the control width %d, or C1 is not "
             "out of region and is not a floor." % (KT, KC))
T = [s for s in scan if inregion(s[2], s[3], KT) and not inregion(s[2], s[3], KC)]
C2 = [s for s in scan if not inregion(s[2], s[3], KT) and s[2] * max(KT, 16) < 2 * THRESH]
print("\nPOPULATIONS at pool %d, treatment width %d" % (POOL, KT))
print("  T  treatment, k=%d, in-region, nnz in [%d, %d): %d matrices"
      % (KT, math.ceil(THRESH / KC), THRESH // max(KT, 16), len(T)))
print("  C1 control, the SAME matrices at k=%d, out of region:       %d matrices"
      % (KC, len(T)))
print("  C2 control, k=%d, nnz just above the threshold:             %d matrices"
      % (KT, len(C2)))
if not T or not C2:
    sys.exit("REFUSE: a population is empty (T %d, C2 %d)" % (len(T), len(C2)))
rnd = random.Random(A.seed)
Ts = sorted(rnd.sample(T, min(A.ncell, len(T))), key=lambda s: s[2])
C2s = sorted(rnd.sample(C2, min(A.ncell, len(C2))), key=lambda s: s[2])
print("  sampled %d of T and %d of C2 at seed %d" % (len(Ts), len(C2s), A.seed), flush=True)

# -------------------------------------------------------------------------- the arms
# Each route's two arms set the SAME NUMBER of env vars, because in a hooks object every
# var an arm sets costs a getenv and an unequal count is a measured bias of about 1% on
# short kernels -- large enough here to be mistaken for the effect.
#
# Each route also DECLARES which populations it should move, and the premise check tests
# that declaration rather than a rule hardcoded in the checker. The first version of this
# file assumed "treatment moves, controls do not" for every route, which is true of R1 and
# false of R2: R2 forces the mode directly, so it moves every cell at every width and C1/C2
# are treatment cells under it, not controls. The check caught it; the fix is below.
# R1's declaration is COMPUTED from the pool rather than written down, because whether its
# arms differ at all is decided by the ceiling at :2614. Hardcoding "T moves" would have the
# premise check fail at every pool above 16 and report it as a broken apparatus rather than
# as the declined argument it is.
R1 = ("R1-gate", {"SCORCH_SPMM_PARTITION_MINGRAINS": "2"},
                 {"SCORCH_SPMM_PARTITION_MINGRAINS": "0"},
      {"T": "none", "C1": "none", "C2": "none"} if CEILING
      else {"T": "all", "C1": "none", "C2": "none"})
# R2 asks the mechanism the gate is a proxy for -- mode 0 against mode 3 -- at three work
# scales, with the gate itself off in both arms so it cannot overwrite the override at
# :2618. All three populations move, which is the point: if the partition's sign really
# flips with work, T and the two controls should disagree in SIGN and not just in size.
R2 = ("R2-mode", {"SCORCH_SPMM_PARTITION_MINGRAINS": "0", "SCORCH_SPMM_PARTITION": "0"},
                 {"SCORCH_SPMM_PARTITION_MINGRAINS": "0", "SCORCH_SPMM_PARTITION": "3"},
      {"T": "all", "C1": "all", "C2": "all"})
# R2's FLOOR, and it is a declined argument rather than a repeated measurement. :2534
# accepts the override only for 0 <= v <= 3, so 7 is parsed, rejected, and leaves the
# default 3 -- while 3 sets it to 3. Both arms set two vars, both pay the same getenv and
# atol, and both land on mode 3 at every width. A floor built by timing one arm twice would
# not exercise the env path at all, and the env path is the thing that could bias an arm.
R2F = ("R2-floor", {"SCORCH_SPMM_PARTITION_MINGRAINS": "0", "SCORCH_SPMM_PARTITION": "3"},
                   {"SCORCH_SPMM_PARTITION_MINGRAINS": "0", "SCORCH_SPMM_PARTITION": "7"},
       {"T": "none", "C1": "none", "C2": "none"})
ROUTES = (R1, R2, R2F)
ALLVARS = ("SCORCH_SPMM_PARTITION_MINGRAINS", "SCORCH_SPMM_PARTITION")


N_SWITCH = [0]


def setarm(env):
    """Set exactly one arm's variables, and COUNT the switch.

    The count exists to check the shim's flush. If a flush were missed across an arm
    switch, both arms would read the first arm's cached values, the measured difference
    would collapse to the floor, and the readout would say the gate does not matter --
    the most flattering way for an apparatus to fail, and indistinguishable from a real
    null. The invariant asserted below is flushes >= arm switches."""
    N_SWITCH[0] += 1
    for v in ALLVARS:
        os.environ.pop(v, None)
    os.environ.update(env)


def decide(s, k, env):
    setarm(env)
    nnz, rows = s[2], s[3]
    nt = so.scorch_spmm_nthreads(nnz * max(k, 16), rows, POOL, nnz * k, nnz)
    return so.scorch_spmm_partition_mode(rows, nnz, k, k, 4, nt, POOL)


# ------------------------------------------- premise 3, the arms actually move, in-process
# Asking the object is the only way to know the env flip reaches the predicate: a value
# cached at load time, or a var spelled differently from the one the source reads, both
# look identical to a null result. The treatment cells must ALL move and the control cells
# must NONE move -- the second half is what makes C1 and C2 an A/A rather than a hope.
print("\nPREMISE: the env flip reaches the predicate, asked of the object per cell")
rc = 0
for route, ea, eb, want in ROUTES:
    for popname, pop, k in (("T", Ts, KT), ("C1", Ts, KC), ("C2", C2s, KT)):
        da = [decide(s, k, ea) for s in pop]
        db = [decide(s, k, eb) for s in pop]
        moved = sum(1 for x, y in zip(da, db) if x != y)
        exp = want[popname]
        ok = (moved == len(pop)) if exp == "all" else (moved == 0)
        print("   %-9s %-3s k=%-3d %4d of %4d move (declared %-4s)  modes %s -> %s   %s"
              % (route, popname, k, moved, len(pop), exp,
                 sorted(set(da)), sorted(set(db)), "OK" if ok else "FAIL"))
        if not ok:
            rc += 1
if rc:
    sys.exit("EXIT: %d premise check(s) failed. Either the treatment population does not "
             "respond to the arm (so there is nothing to measure) or a control does (so it "
             "is not an A/A and its spread is not a floor). Timing either one would read "
             "as a plausible number." % rc)
print("   every route moves exactly the populations it declared", flush=True)

# ------------------------------------------------------------------------ the timing
NPDT, TDT = np.float32, torch.float32
SYM = so.spmm_csr_float_v2
out = open(A.csv, "w", newline="")
w = csv.DictWriter(out, fieldnames=["route", "pop", "key", "k", "kt", "rows", "cols",
                                    "nnz", "arm", "us", "batch", "mode", "nthreads",
                                    "pool", "nproc", "adoption", "shim", "shim_sha",
                                    "so_sha"])
w.writeheader()


def load(path):
    with np.load(path) as z:
        return (np.ascontiguousarray(z["indptr"], dtype=np.int32),
                np.ascontiguousarray(z["indices"], dtype=np.int32))


def calibrate(fn):
    fn(); fn(); fn()
    c = sorted(time_once(fn) for _ in range(5))
    return max(1, int((A.target_ms * 1e-3) / max(c[2], 1e-9)))


def time_once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def timeit(fn, batch):
    """ALL reps in microseconds, not just the minimum.

    Min-of-n is the right estimator for one arm's cost -- it is the reading least
    contaminated by preemption and interrupts. But the quantity reported here is a
    DIFFERENCE of two arms, and the minimum of n draws sits further below the centre the
    wider that arm's own spread, so unequal arm noise leaves a bias in the paired
    difference that pairing does not remove. Its direction depends on which arm is noisier.
    Returning every rep lets the readout print the min-based and median-based differences
    side by side and the per-arm spread that decides whether they can disagree."""
    reps = []
    for _ in range(A.reps):
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        reps.append((time.perf_counter() - t0) / batch * 1e6)
    return reps


CELLS = []
for route, ea, eb, want in ROUTES:
    for popname, pop, k in (("T", Ts, KT), ("C1", Ts, KC), ("C2", C2s, KT)):
        CELLS.append((route, ea, eb, popname, pop, k))
print("\nTIMING %d route-population blocks, %d reps, target %.0f ms per reading"
      % (len(CELLS), A.reps, A.target_ms), flush=True)
res = {}
t_start = time.time()
for route, ea, eb, popname, pop, k in CELLS:
    per = []
    for i, s in enumerate(pop):
        path, key, nnz, rows, shape = s
        try:
            indptr, indices = load(path)
        except Exception as exc:                                       # noqa: BLE001
            print("   skip %s: %s" % (key, type(exc).__name__))
            continue
        if int(indptr[-1]) != nnz or len(indptr) - 1 != rows:
            print("   skip %s: header and arrays disagree" % key)
            continue
        g0 = torch.Generator().manual_seed(zlib.crc32(key.encode()) % (2 ** 31))
        vt = (torch.rand(nnz, generator=g0) - 0.5).to(TDT)
        ipt, ixt = torch.from_numpy(indptr), torch.from_numpy(indices)
        J = shape[1]
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2 ** 31))
        B = torch.rand((J, k), generator=g).to(TDT)
        native = dict(result_shape=[rows, k], A_shape=[rows, J],
                      A_mode_indices=[[], [ipt, ixt]], A_values=vt,
                      B_shape=[J, k], B_mode_indices=[[], []],
                      B_values=B.reshape(-1).contiguous())

        def run():
            return SYM(nthreads_override=POOL, atparallel=False, **native)

        setarm(ea)
        batch = calibrate(run)
        # ABBA within the cell. One A then one B leaves a linear drift in the pair
        # difference; A,B,B,A cancels it to first order and the two halves are averaged.
        setarm(ea); a1 = timeit(run, batch)
        setarm(eb); b1 = timeit(run, batch)
        setarm(eb); b2 = timeit(run, batch)
        setarm(ea); a2 = timeit(run, batch)
        ra, rb = a1 + a2, b1 + b2
        ua, ub = 0.5 * (min(a1) + min(a2)), 0.5 * (min(b1) + min(b2))
        ma = 0.5 * (statistics.median(a1) + statistics.median(a2))
        mb = 0.5 * (statistics.median(b1) + statistics.median(b2))
        # Within-cell spread per arm, measured WHERE THE ESTIMATOR REDUCES. Both estimators
        # reduce inside each ABBA half and then average the halves, so they never see the
        # drift between halves -- which is the drift ABBA exists to cancel. Pooling the two
        # halves would fold that drift into a diagnostic about a quantity the estimator does
        # not use. The larger of the two halves is taken, so the figure is an upper bound on
        # the spread either estimator actually faces.
        def spread(v):
            return (max(v) - min(v)) / max(statistics.median(v), 1e-9)
        sa = max(spread(a1), spread(a2))
        sb = max(spread(b1), spread(b2))
        nt = so.scorch_spmm_nthreads(nnz * max(k, 16), rows, POOL, nnz * k, nnz)
        for arm, us in (("A", ua), ("B", ub)):
            w.writerow(dict(route=route, pop=popname, key=key, k=k, kt=KT, rows=rows,
                            cols=J, nnz=nnz, arm=arm, us="%.4f" % us, batch=batch,
                            nproc=NPROC, adoption=int(ADOPTION),
                            shim=int(SHIM is not None), shim_sha=SHIM_SHA, mode=decide(s, k, ea if arm == "A" else eb), nthreads=nt,
                            pool=POOL, so_sha=SO_SHA))
        per.append((key, nnz, rows, ua, ub, ma, mb, sa, sb))
        if i % 50 == 0:
            print("   %s %s %d/%d  %s nnz %d  A %.1f us  B %.1f us  batch %d"
                  % (route, popname, i, len(pop), key, nnz, ua, ub, batch), flush=True)
    res[(route, popname)] = per
    out.flush()
    print("   %s %s done, %d cells, %.0f s elapsed"
          % (route, popname, len(per), time.time() - t_start), flush=True)
out.close()

# ------------------------------------------------------------------------ the readout
def boot_med(v, B=4000, seed=7):
    r = random.Random(seed)
    n = len(v)
    s = sorted(statistics.median([v[r.randrange(n)] for _ in range(n)]) for _ in range(B))
    return s[int(0.025 * (B - 1))], s[int(0.975 * (B - 1))]


if SHIM:
    fl, sw = SHIM.scorch_fastenv_flushes(), N_SWITCH[0]
    hp, hs = SHIM.scorch_fastenv_hits_ptr(), SHIM.scorch_fastenv_hits_str()
    ev = SHIM.scorch_fastenv_evictions()
    print("\nSHIM COUNTERS, and two of them are premises rather than trivia.")
    print("   real getenv %d   cache hits %d   evictions %d"
          % (SHIM.scorch_fastenv_real(), hp + hs, ev))
    print("   flushes %d against %d arm switches  -- %s"
          % (fl, sw, "OK, every switch reached the environment" if fl >= sw else
             "FAIL: a flush was missed across a switch, so both arms may have read one "
             "arm's cached values and the difference would collapse toward the floor"))
    print("   hits by pointer identity %d, by string compare %d  -- %s"
          % (hp, hs, "every caller passed a stable literal, which is the cache's premise"
             if hs == 0 else "some caller passed a non-literal; the lookup is still correct "
             "because keys are owned copies, but the premise is not universal"))
    if ev:
        print("   NOTE %d evictions: the table filled, so those calls were uncached and "
              "paid full cost. Correct, just slower." % ev)
    if fl < sw:
        sys.exit("EXIT: the shim did not flush on every arm switch. Readings under it are "
                 "not trustworthy and the failure direction is toward a null.")
print("\n" + "=" * 78)
print("PRIMARY: per-cell A-minus-B in MICROSECONDS. Arm A has the partition OFF on the")
print("treatment cells (mode 0) and arm B has it ON (mode 3), so a NEGATIVE difference")
print("means the gate's choice is the faster one and the shipped gate is earning its")
print("constant. An additive per-call tax cancels here, so this column is tax-free.")
print("=" * 78)
def zscore(npos, n):
    """Standard deviations from the exchangeability null, where a sign is a fair coin."""
    return (npos - n / 2.0) / math.sqrt(n / 4.0) if n else 0.0


for route, ea, eb, want in ROUTES:
    print("\n%s" % route)
    # Two passes. The treatment row's null can be stated two ways -- the theoretical n/2
    # of exchangeable arms, and the EMPIRICAL rate the floor populations actually show --
    # and an earlier version of this block printed "null is n/2, i.e. the floor rows' own
    # sign counts", asserting with "i.e." that they coincide. The same run falsified it:
    # C1 read 79 of 120 where n/2 is 60, which is 3.47 sd away. They are different
    # quantities and both are reported, because the honest scope statement is that the
    # treatment survives the WORSE of the two.
    emp = {}
    for popname in ("T", "C1", "C2"):
        per = res.get((route, popname)) or []
        if per and want[popname] != "all":
            d0 = [c[3] - c[4] for c in per]
            emp[popname] = (sum(1 for x in d0 if x > 0), len(d0))
    for popname in ("T", "C1", "C2"):
        per = res.get((route, popname)) or []
        if not per:
            print("   %-3s no cells" % popname)
            continue
        d = [c[3] - c[4] for c in per]
        dmed = [c[5] - c[6] for c in per]
        lo, hi = boot_med(d)
        scale = statistics.median([0.5 * (c[3] + c[4]) for c in per])
        rat = statistics.median([c[3] / c[4] for c in per if c[4] > 0])
        # The sign count has no reduction step, is invariant to min-versus-median, and is
        # invariant to the additive per-call tax. It is the statistic that transfers.
        npos = sum(1 for x in d if x > 0)
        nneg = sum(1 for x in d if x < 0)
        spa = statistics.median([c[7] for c in per])
        spb = statistics.median([c[8] for c in per])
        # The label comes from the route's own declaration, so a population cannot be
        # printed as a floor in a route that moves it.
        tag = "TREATMENT" if want[popname] == "all" else "A/A FLOOR"
        print("   %-3s %-9s n %3d   median A-B %+8.3f us  CI [%+7.3f, %+7.3f]  %s"
              % (popname, tag, len(d), statistics.median(d), lo, hi,
                 "spans 0" if lo <= 0 <= hi else "excludes 0"))
        print("       SIGN COUNT %3d of %3d positive, %3d negative   (no reduction step, "
              "invariant to min-vs-median and to the tax)" % (npos, len(d), nneg))
        if tag == "A/A FLOOR":
            print("         this row is an EMPIRICAL null for the treatment above, matched "
                  "on n but NOT on population; z = %+.2f against the theoretical %d"
                  % (zscore(npos, len(d)), len(d) // 2))
        else:
            # Each null carries its RATE, its statistic, and -- when the count is
            # unanimous -- an exact tail. The normal approximation is weakest at npos == n,
            # where the treatment arm has zero sample variance, which is also the most
            # clear-cut case, so the exact binomial tail is reported there instead.
            nulls = []
            zt = zscore(npos, len(d))
            # 0.5 by DEFINITION, not len(d)//2/len(d), which at odd n reads 0.496 and would
            # route a true coincidence into a "differ" branch.
            nulls.append(("theoretical", 0.5, len(d) // 2, len(d), zt))
            for pn, (fp, fn_) in sorted(emp.items()):
                # Pooled two-proportion test. Treating the floor's rate as KNOWN understates
                # the spread, because that rate is itself estimated on the floor's own n,
                # and the understatement always favours the treatment.
                n1, n2 = len(d), fn_
                pp = (npos + fp) / float(n1 + n2)
                se = math.sqrt(pp * (1 - pp) * (1.0 / n1 + 1.0 / n2))
                z = ((npos / float(n1)) - (fp / float(n2))) / se if se > 1e-12 else float("inf")
                nulls.append(("%s's rate" % pn, fp / float(n2), fp, fn_, z))
            unanimous = (npos == len(d))

            def tail(rate):
                return len(d) * math.log10(rate) if rate > 0 else float("-inf")

            print("         nulls, %d of them:" % len(nulls))
            for nm, rate, cnt, nn, z in nulls:
                if unanimous:
                    print("           %-16s rate %.4f (%3d of %3d)   exact P(all %d "
                          "positive) = 10^%.1f" % (nm, rate, cnt, nn, len(d), tail(rate)))
                else:
                    print("           %-16s rate %.4f (%3d of %3d)   z = %+6.2f%s"
                          % (nm, rate, cnt, nn, z,
                             "" if nm == "theoretical" else "  (two-proportion)"))
            # NO RELATION IS ASSERTED. Three earlier versions of this line each claimed one
            # -- that the nulls coincide, that they differ, that coinciding rates make the
            # choice immaterial -- and each was falsified by the run that printed it. The
            # last failed because equal RATES do not imply equal STATISTICS: a null rate
            # treated as known and one estimated from a floor's own sample give different
            # spreads, measured at 2.04 apart in z on cells whose rates were identical. So
            # the only thing said here is which null BINDS, chosen on the statistic that
            # decides rather than on the rate beside it.
            if unanimous:
                b = max(nulls, key=lambda x: tail(x[1]))
                print("         the BINDING null is %s: least extreme exact tail, 10^%.1f"
                      % (b[0], tail(b[1])))
            else:
                b = min(nulls, key=lambda x: x[4])
                print("         the BINDING null is %s: smallest statistic, z = %+.2f"
                      % (b[0], b[4]))
        print("       median-of-reps A-B %+8.3f us  vs min-of-reps %+8.3f  gap %+.3f "
              "CI [%+.3f, %+.3f] %s"
              % (statistics.median(dmed), statistics.median(d),
                 statistics.median([x - y for x, y in zip(dmed, d)]), glo, ghi,
                 "AGREE" if glo <= 0 <= ghi else "DIFFER, so the estimator choice matters"))
        print("       within-HALF spread, arm A %.4f of level, arm B %.4f (worse half, over "
              "%d reps)  -- comparable spreads are what let the two differences agree; this "
              "is a between-arm comparison at matched n, not a noise level, since max-min "
              "grows with n" % (spa, spb, A.reps))
        print("       scale %8.2f us   median ratio A/B %.4f%s"
              % (scale, rat, "" if SHIM else " (attenuated by the per-call tax)"))
if SHIM:
    print("\nTHE `scale` COLUMN IS COMPARABLE TO THE SHIPPED COMMENT'S BANDS, AND READING IT")
    print("DIRECTLY IS WHY THE SHIM EXISTS. That comment puts the cells where the partition")
    print("loses at 19-30 us and the ones it wins at 31-110.")
    print("")
    print("WHAT THE SHIM REMOVES IS THE MISS COST, NOT ALL OF IT. The counters above show a")
    print("few hundred real getenv calls across the whole run -- tens of microseconds in")
    print("total, which really is amortised to nothing per call. The HIT cost is not: the")
    print("kernel still performs 99 lookups every call and each costs a hash and a strcmp.")
    print("So a residual of order 99 x (a few ns) remains inside `scale`.")
    print("")
    print("WHICH MAKES `scale` AN UPPER BOUND ON THE TAX-FREE LEVEL, AND A BOUND IS STRONGER")
    print("THAN AN ESTIMATE HERE. The shim only ever ADDS work along the getenv path, so the")
    print("true release level K satisfies K < scale. A scale below 30 therefore places the")
    print("cell inside the 19-30 band with no subtracted constant and no calibration, and the")
    print("residual's size cannot overturn it because the residual has a known sign. The one")
    print("premise is that the shim perturbs nothing but the getenv path; if it also moved")
    print("cache or branch behaviour the bound would be premised rather than proven.")
    print("")
    print("AND THE RATIO COLUMN IS LIKEWISE UNATTENUATED, which matters because the")
    print("comment's criterion is itself a ratio -- 'falls more than 10% behind' -- so an")
    print("additive tax shrinks the very quantity the threshold is stated in.")
else:
    print("\nTHE DIFFERENCE IS TAX-FREE AND THE SCALE IS NOT, SO THE `scale` COLUMN ABOVE MUST")
    print("NOT BE COMPARED WITH THE MICROSECOND BANDS IN THE SHIPPED COMMENT. That comment")
    print("puts the cells where the partition loses at 19-30 us and the ones where it wins at")
    print("31-110. This object charges ~20 us of getenv per call, measured, so a cell printed")
    print("here at 50 us is a ~30 us cell without it -- the other side of the comment's")
    print("crossover. The sign and size of A-B stand; their position relative to the")
    print("comment's threshold does not. Re-run under the shim to read the band directly.")
print("\nR1's two controls are matched on different axes: C1 is the SAME MATRICES as T at")
print("another width, C2 is the same width as T at another nnz. An R1 effect has to clear")
print("both to be a property of the gate rather than of the population.")
print("R2 has no control population -- it moves every cell by construction -- so its floor")
print("is R2-floor, a declined override that pays the same getenv and lands on the same")
print("mode. Read R2's three rows as a dose-response in work, not as effect against floor:")
print("the shipped comment claims the partition's sign FLIPS with work, so T (below the")
print("threshold) and C1/C2 (above it) should differ in sign and not merely in magnitude.")
