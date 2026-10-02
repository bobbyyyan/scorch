#!/usr/bin/env python3
"""Price SCORCH_SPMM_ADOPT_GRAIN: does sizing the ADOPTED team to real arithmetic beat MKL?

WHAT IS BEING TESTED AND WHY IT IS A BUILD. The drop-in SpMM adopts the caller's pool so a
pipeline does not reshape its OpenMP team at every op boundary. That adoption has no work term:
it takes min(pool, rows/16) whatever the arithmetic is. scorch_policy.h ships
SCORCH_SPMM_ADOPT_GRAIN at 0, which is off; set to a grain it caps the adopted count so each
worker gets a grain of REAL arithmetic (nnz*k), never below one, so it narrows an adopted team
without declining the adoption. The constant is compiled in, so the arms are BUILDS -- and they
have to be: the hook form is readable only under SCORCH_TUNE_HOOKS, and a hooks object charges
~19 microseconds of getenv a call, which is enough to invert a sub-30-microsecond cell against
MKL.

THE OVERRIDE IS NOT A HARNESS CHOICE. This calls with nthreads_override=POOL because that is
what production does: scorch.matmul routes through the drop-in path, and timed three ways on
this corpus it matches the override reading and not the pure-policy one (2.135x apart at k=2).
A harness that called with pure policy would be measuring a configuration the library does not
use by default, and the lever under test lives entirely inside the adoption branch.

WHAT THIS IS NOT INERT ON. The cap binds wherever work_true/grain < pool, which includes WIDE k
on small matrices -- 32530 nonzeros at k=16 goes from 6 workers to 3. Those are the widths we
already beat MKL at, so the corpus deliberately carries the wide-k small-nonzero corner. k>=16
is an exposed surface here, not a negative control.

`oursb` is `ours` again -- identical code, identical argument -- and bounds the timing only. The
PLACEMENT floor is the adg0/adgb build pair, which this file cannot see; it is two objects of one
configuration and the analysis reads it across CSVs.
"""
import argparse, csv, math, os, sys, time, zlib
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="DLMC .smtx tree, or a cache of .npz")
ap.add_argument("--keys", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--dtype", default="float32")
ap.add_argument("--ks", default="2,4,8,12,16,32")
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--target-ms", type=float, default=15.0)
ap.add_argument("--pass-tag", default="p1")
ap.add_argument("--build", required=True)
A = ap.parse_args()
import hashlib
import numpy as np, torch, scorch_ops as so
if so.scorch_tune_hooks():
    sys.exit("REFUSE: this is a hooks object. Against MKL that charges ~19 microseconds of getenv "
             "a call and manufactures a deficit on short kernels; the arms here are builds.")
NPDT = np.float32 if A.dtype == "float32" else np.float64
TDT = torch.float32 if A.dtype == "float32" else torch.float64
SYM = so.spmm_csr_float_v2 if A.dtype == "float32" else so.spmm_csr_double_v2
KS = [int(x) for x in A.ks.split(",")]
POOL = torch.get_num_threads()
# The reference arm is named for WHAT IT IS, not for what we wish it were. torch.sparse.mm
# dispatches to MKL sparse only where torch was built with it; on this project's Mac it falls
# back to PyTorch's native CSR path and reads 11.7x slower than our kernel, which would make an
# "ours/mkl" column in a CSV look like an enormous win and mean nothing. The backend is recorded
# in every row so a reader cannot have to infer it from the host, which is exactly how the
# chain212 filenames went wrong.
REF_BACKEND = "mkl" if torch.backends.mkl.is_available() else "torch-native"
ARMS = ["ours", "oursb", "ref"]

# THE OBJECT THAT ANSWERED, DIGESTED AND RECORDED IN EVERY ROW.
#
# The `scorch` conda env carries an editable install whose __editable__ .pth puts
# /Users/bobby/scorch-beat-mkl/src on sys.path for the WHOLE environment. So every
# `import scorch_ops` in that env loads whatever in-place .so sits in that worktree -- from any
# directory, with no PYTHONPATH, including a plain `import scorch as torch`. For about seventy
# minutes an experimental object with SCORCH_SPMM_BASE_WORK_TRUE=1 sat there, left by a two-build
# script that built its second arm in place, and two sessions spent an hour reconciling readings
# of one policy function that differed only because of it. A PYTHONPATH naming a staged tree wins
# over that path, which is why the staged runs were clean -- but nothing in the OUTPUT said so.
#
# Recording the digest is the only part of the three-part rule that cannot be recovered
# afterwards: a figure's pool and probe argument can be re-derived from the script, and the
# object cannot be re-derived from anything once it is overwritten.
SO_SHA = hashlib.sha256(open(so.__file__, "rb").read()).hexdigest()[:16]
print("object: %s  sha %s" % (so.__file__, SO_SHA), flush=True)


def read_smtx(path):
    """A DLMC .smtx: a `rows, cols, nnz` header, then indptr, then indices. No values."""
    with open(path) as fh:
        head = fh.readline(); ipt = fh.readline(); idx = fh.readline()
    rows, cols, nnz = (int(x) for x in head.replace(",", " ").split())
    indptr = np.array(ipt.split(), dtype=np.int64)
    indices = np.array(idx.split(), dtype=np.int64)
    if len(indptr) != rows + 1 or int(indptr[-1]) != nnz or len(indices) != nnz:
        raise ValueError("%s: header nnz %d, indptr[-1] %d, len(indices) %d -- a truncated file "
                         "parses cleanly and reads as a sparser matrix"
                         % (path, nnz, int(indptr[-1]), len(indices)))
    return indptr, indices, rows, cols


def load(key):
    """Resolve a key as .smtx under --root, else as .npz. Both corpora, one key list."""
    p = os.path.join(A.root, key + ".smtx")
    if os.path.exists(p):
        return read_smtx(p)
    p = os.path.join(A.root, key + ".npz")
    if not os.path.exists(p):
        base = os.path.basename(key)
        hits = [q for q in _npz_index.get(base, [])]
        if len(hits) != 1:
            raise IOError("%s: %d resolutions" % (key, len(hits)))
        p = hits[0]
    z = np.load(p)
    s = z["shape"]
    return z["indptr"], z["indices"], int(s[0]), int(s[1])


_npz_index = {}
for dirpath, _, files in os.walk(A.root):
    for f in files:
        if f.endswith(".npz"):
            _npz_index.setdefault(f[:-4], []).append(os.path.join(dirpath, f))

keys = [l.strip() for l in open(A.keys) if l.strip() and not l.startswith("#")]
bad = []
loaded = {}
for key in keys:
    try:
        loaded[key] = load(key)
    except Exception as exc:
        bad.append((key, type(exc).__name__))
print("corpus %d keys, %d resolved, %d unresolved" % (len(keys), len(loaded), len(bad)), flush=True)
if bad:
    sys.exit("REFUSE: %d of %d keys unresolved under %s (first %s). An unresolved key means the "
             "corpus and the tree disagree, and a silently shorter corpus is a different "
             "experiment." % (len(bad), len(keys), A.root, bad[0][0]))


def calibrate(fn):
    fn(); fn(); fn()
    cals = []
    for _ in range(5):
        t0 = time.perf_counter(); fn(); cals.append(time.perf_counter() - t0)
    cals.sort()
    return max(1, int((A.target_ms * 1e-3) / max(cals[len(cals) // 2], 1e-9)))


def timeit(fn, batch):
    """Min over `reps` batched means, at a batch the CALLER fixes -- so an arm and its own floor
    cannot differ in the one parameter that biases the reading."""
    best = float("inf")
    for _ in range(A.reps):
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        dt = (time.perf_counter() - t0) / batch
        if dt < best:
            best = dt
    return best * 1e3, batch


out = open(A.csv, "w", newline="")
w = csv.DictWriter(out, fieldnames=["build", "key", "k", "rows", "cols", "nnz", "deg", "arm",
                                    "ms", "batch", "nthreads", "refbackend", "so_sha"])
w.writeheader()
print("build %s dtype %s pool %d  %d matrices  %d widths  reference backend %s"
      % (A.build, A.dtype, POOL, len(keys), len(KS), REF_BACKEND), flush=True)
if REF_BACKEND != "mkl":
    print("NOTE: torch has no MKL here, so the reference arm is PyTorch's native CSR path and is "
          "NOT an MKL comparison. Every row records this; the analysis refuses to label it MKL.",
          flush=True)
nrow = 0
for key in keys:
    indptr, indices, M, J = loaded[key]
    nnz = int(indptr[-1])
    ipt = torch.from_numpy(np.ascontiguousarray(indptr, dtype=np.int32))
    ixt = torch.from_numpy(np.ascontiguousarray(indices, dtype=np.int32))
    g0 = torch.Generator().manual_seed(zlib.crc32(key.encode()) % (2 ** 31))
    vt = (torch.rand(nnz, generator=g0) - 0.5).to(TDT)
    Acsr = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
    for k in KS:
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2 ** 31))
        B = torch.rand((J, k), generator=g).to(TDT)
        native = dict(result_shape=[M, k], A_shape=[M, J], A_mode_indices=[[], [ipt, ixt]],
                      A_values=vt, B_shape=[J, k], B_mode_indices=[[], []],
                      B_values=B.reshape(-1).contiguous())

        def ours():
            return SYM(nthreads_override=POOL, atparallel=False, **native)

        fns = {"ours": ours, "oursb": ours, "ref": lambda: torch.sparse.mm(Acsr, B)}
        # The thread count the OBJECT resolves to, asked of the object rather than restated here.
        # A harness that recomputes a production policy is wrong silently and flatteringly.
        nt = so.scorch_spmm_nthreads(nnz * max(k, 16), M, POOL, nnz * k, nnz)
        try:
            batch_ours = calibrate(ours)
            batch_ref = calibrate(fns["ref"])
        except Exception as exc:
            print("  SKIP %s k=%d calibration: %s" % (key, k, type(exc).__name__), flush=True)
            continue
        order = list(ARMS)
        np.random.default_rng(zlib.crc32(("%s|%d|%s|%s" % (key, k, A.pass_tag, A.build))
                                         .encode())).shuffle(order)
        for arm in order:
            try:
                ms, batch = timeit(fns[arm], batch_ref if arm == "ref" else batch_ours)
            except Exception as exc:
                print("  SKIP %s k=%d %s: %s" % (key, k, arm, type(exc).__name__), flush=True)
                continue
            w.writerow(dict(build=A.build, key=key, k=k, rows=M, cols=J, nnz=nnz,
                            deg=round(nnz / max(M, 1), 2), arm=arm, ms="%.6f" % ms,
                            batch=batch, nthreads=nt, refbackend=REF_BACKEND,
                            so_sha=SO_SHA))
            nrow += 1
        out.flush()
print("wrote %d rows to %s" % (nrow, A.csv), flush=True)
