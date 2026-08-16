#!/usr/bin/env python3
"""Does the fast ABI screen still reject everything the serial loop rejected?

The screens in native_abi.h fold every violation into an OR / min / max accumulator
and hand off to the original serial loop for the message. These cases drive both the
serial path (small nnz) and the PARALLEL path (nnz >= SCORCH_ABI_VALIDATE_GRAIN),
since the screen only splits across threads above that grain — a screen that were
wrong only when parallel would otherwise pass every small test.

Calls scorch_ops directly so the native validator is what is under test, not the
Python-side narrowing in prebuilt_kernels.
"""
import os
import sys

import numpy as np
import torch

import scorch_ops as ops


def csr_args(M, J, N, pos, crd, val, idtype=np.int32):
    B = torch.zeros((J, N), dtype=torch.float32)
    return dict(
        result_shape=[M, N],
        A_shape=[M, J],
        A_mode_indices=[[], [torch.from_numpy(pos.astype(idtype)),
                              torch.from_numpy(crd.astype(idtype))]],
        A_values=torch.from_numpy(val.astype(np.float32)),
        B_shape=[J, N],
        B_mode_indices=[[], []],
        B_values=B.reshape(-1),
    )


def banded(M, deg):
    """A valid CSR with `deg` sorted columns per row."""
    pos = np.arange(M + 1, dtype=np.int64) * deg
    crd = np.concatenate([np.arange(deg) + (i % max(1, M - deg)) for i in range(M)])
    val = np.ones(M * deg)
    return pos, crd.astype(np.int64), val


FAILS = []
PASSES = 0


def expect_raise(name, fragment, fn):
    global PASSES
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the point is what the C++ layer raised
        if fragment in str(exc):
            PASSES += 1
            print(f"  ok    {name}: raised, message contains {fragment!r}")
        else:
            FAILS.append(name)
            print(f"  FAIL  {name}: raised but message lacks {fragment!r}: {exc}")
        return
    FAILS.append(name)
    print(f"  FAIL  {name}: no exception raised")


def expect_ok(name, fn):
    global PASSES
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        FAILS.append(name)
        print(f"  FAIL  {name}: unexpected raise: {exc}")
        return
    PASSES += 1
    print(f"  ok    {name}: accepted")


def run_suite(M, deg, label):
    """Same battery at a size that is below and above the parallel-screen grain."""
    N = 8
    J = M
    print(f"\n[{label}] M={M} deg={deg} nnz={M*deg} "
          f"({'parallel' if M*deg >= 262144 else 'serial'} screen)")

    pos, crd, val = banded(M, deg)
    expect_ok("valid int32", lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, crd, val)))
    expect_ok("valid int64", lambda: ops.spmm_csr_float_v2(
        **csr_args(M, J, N, pos, crd, val, idtype=np.int64)))

    bad = crd.copy()
    bad[len(bad) // 2] = J          # one past the last legal column
    expect_raise("column out of range", "outside [0,",
                 lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, bad, val)))

    bad = crd.copy()
    bad[len(bad) // 2] = -1
    expect_raise("negative column", "outside [0,",
                 lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, bad, val)))

    # Swap two coordinates inside one row so that row alone is unsorted.
    bad = crd.copy()
    mid = M // 2
    lo = int(pos[mid])
    if deg >= 2 and bad[lo] != bad[lo + 1]:
        bad[lo], bad[lo + 1] = bad[lo + 1], bad[lo]
        expect_raise("unsorted within a row", "must be sorted",
                     lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, bad, val)))

    badpos = pos.copy()
    badpos[0] = 1
    expect_raise("positions[0] != 0", "positions[0] must be 0",
                 lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, badpos, crd, val)))

    # A negative position: the span check must catch it. (Making a span merely go
    # backwards instead trips the sortedness check first, because the preceding row's
    # range then runs into the next row's coordinates -- true of the original serial
    # validator too, so it is not a difference this change introduces.)
    badpos = pos.copy()
    badpos[M // 2] = -1
    expect_raise("negative span", "invalid CSR span",
                 lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, badpos, crd, val)))

    badpos = pos.copy()
    badpos[M] = int(pos[M]) + 5                    # terminal position past nnz
    expect_raise("terminal past nnz", "invalid CSR span",
                 lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, badpos, crd, val)))

    # int64 element that cannot be narrowed. Put it in the coordinates, where the
    # representability screen (min/max reduction) is the thing being tested.
    bad64 = crd.astype(np.int64).copy()
    bad64[len(bad64) // 2] = 2 ** 31 + 7
    expect_raise("int64 not representable", "cannot be represented as int32",
                 lambda: ops.spmm_csr_float_v2(
                     **csr_args(M, J, N, pos, bad64, val, idtype=np.int64)))

    bad64 = crd.astype(np.int64).copy()
    bad64[len(bad64) // 2] = -(2 ** 31) - 7
    expect_raise("int64 below int32 min", "cannot be represented as int32",
                 lambda: ops.spmm_csr_float_v2(
                     **csr_args(M, J, N, pos, bad64, val, idtype=np.int64)))


def run_sortedness_suite(M, deg, label):
    """The sortedness screen counts descents flat and subtracts the ones sitting on a
    row boundary, so the cases that could break it are all about boundaries: empty
    rows (a boundary position shared by several rows), and legal descents at every
    boundary (the normal case for a banded matrix)."""
    N = 8
    J = M
    nnz = M * deg
    print(f"\n[{label}] M={M} deg={deg} nnz={nnz} "
          f"({'parallel' if nnz >= 65536 else 'serial'} screen)")

    # Every row restarts at column 0, so EVERY row boundary is a descent. All legal.
    pos = np.arange(M + 1, dtype=np.int64) * deg
    crd = np.tile(np.arange(deg, dtype=np.int64), M)
    val = np.ones(nnz)
    expect_ok("descent at every boundary",
              lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, crd, val)))

    # Same, with an internal descent inside one row -> must be rejected even though
    # every boundary also descends (the count must not absorb it).
    bad = crd.copy()
    lo = int(pos[M // 2])
    bad[lo], bad[lo + 1] = bad[lo + 1], bad[lo]
    expect_raise("internal descent among boundary descents", "must be sorted",
                 lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, bad, val)))

    # Empty rows: every other row is empty, so a boundary position is shared by two
    # rows and must not be double-counted.
    epos = np.zeros(M + 1, dtype=np.int64)
    k = 0
    for i in range(M):
        epos[i] = k
        if i % 2 == 0:
            k += deg
    epos[M] = k
    ecrd = np.tile(np.arange(deg, dtype=np.int64), (M + 1) // 2)[:k]
    eval_ = np.ones(k)
    expect_ok("empty rows, valid",
              lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, epos, ecrd, eval_)))

    ebad = ecrd.copy()
    mid = (k // deg // 2) * deg
    if mid + 1 < k:
        ebad[mid], ebad[mid + 1] = ebad[mid + 1], ebad[mid]
        expect_raise("empty rows, internal descent", "must be sorted",
                     lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, epos, ebad, eval_)))

    # A descent in the very LAST row, and in the very first, are the positions a
    # boundary-based argument is most likely to mishandle.
    for where, name in ((0, "first row"), (M - 1, "last row")):
        bad = crd.copy()
        lo = int(pos[where])
        bad[lo], bad[lo + 1] = bad[lo + 1], bad[lo]
        expect_raise(f"descent in {name}", "must be sorted",
                     lambda b=bad: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, b, val)))


def run_memo_suite():
    """The validation memo must never turn a rejected input into an accepted one, and
    its one documented blind spot must actually be documented.

    Mutations here are all order violations, never out-of-range columns: an accepted
    out-of-range column would read past B and take the process down, which is exactly
    the risk the memo trades away and not something to demonstrate inside a test run.
    """
    print("\n[memo] fresh tensors must always be validated")
    M, deg, J, N = 2000, 8, 2000, 8
    pos = np.arange(M + 1, dtype=np.int64) * deg
    crd = np.tile(np.arange(deg, dtype=np.int64), M)
    val = np.ones(M * deg)

    # Distinct tensor objects with the same *contents* must each be validated: storage
    # addresses get recycled constantly, and a cached verdict must not ride along.
    for i in range(3):
        expect_ok(f"fresh valid tensor #{i}",
                  lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, crd, val)))
    bad = crd.copy()
    bad[deg], bad[deg + 1] = bad[deg + 1], bad[deg]
    for i in range(3):
        expect_raise(f"fresh invalid tensor #{i}", "must be sorted",
                     lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, bad, val)))

    # Same tensors reused: second and later calls take the memo. Still correct.
    args = csr_args(M, J, N, pos, crd, val)
    for i in range(3):
        expect_ok(f"reused valid tensor, call {i}", lambda: ops.spmm_csr_float_v2(**args))

    bad_args = csr_args(M, J, N, pos, bad, val)
    for i in range(3):
        expect_raise(f"reused invalid tensor, call {i}", "must be sorted",
                     lambda: ops.spmm_csr_float_v2(**bad_args))

    # The documented blind spot: a write straight through the numpy buffer the tensor
    # aliases does not bump torch's version counter, so after a successful call the
    # memo keeps the old verdict.
    shared = crd.astype(np.int32).copy()
    args2 = dict(
        result_shape=[M, N], A_shape=[M, J],
        A_mode_indices=[[], [torch.from_numpy(pos.astype(np.int32)),
                             torch.from_numpy(shared)]],
        A_values=torch.from_numpy(val.astype(np.float32)),
        B_shape=[J, N], B_mode_indices=[[], []],
        B_values=torch.zeros((J, N), dtype=torch.float32).reshape(-1),
    )
    ops.spmm_csr_float_v2(**args2)                 # validated and memoized
    shared[deg], shared[deg + 1] = shared[deg + 1], shared[deg]   # behind torch's back
    global PASSES
    try:
        ops.spmm_csr_float_v2(**args2)
        PASSES += 1
        print("  ok    documented blind spot: memo keeps the earlier verdict")
    except Exception as exc:  # noqa: BLE001
        PASSES += 1
        print(f"  note  blind spot did not trigger here (still rejected): {exc}")

    # Two operands whose coordinate tensors are VIEWS of one buffer map to the same
    # memo key, since the key is the StorageImpl address. The recorded data_ptr is what
    # keeps them apart; without it, validating one would vouch for the other.
    nnz = M * deg
    buf = np.zeros(2 * nnz, dtype=np.int32)
    ok_crd = np.tile(np.arange(deg, dtype=np.int32), M)
    bad_crd = ok_crd.copy()
    bad_crd[deg], bad_crd[deg + 1] = bad_crd[deg + 1], bad_crd[deg]
    buf[:nnz] = ok_crd
    buf[nnz:] = bad_crd
    shared_t = torch.from_numpy(buf)
    pos32 = torch.from_numpy(np.arange(M + 1, dtype=np.int32) * deg)
    val32 = torch.from_numpy(np.ones(nnz, dtype=np.float32))
    Bz = torch.zeros((J, N), dtype=torch.float32).reshape(-1)

    def call_view(view):
        return ops.spmm_csr_float_v2(
            result_shape=[M, N], A_shape=[M, J],
            A_mode_indices=[[], [pos32, view]], A_values=val32,
            B_shape=[J, N], B_mode_indices=[[], []], B_values=Bz)

    v_ok, v_bad = shared_t[:nnz], shared_t[nnz:]
    for rnd in range(3):
        expect_ok(f"shared-storage valid view, round {rnd}", lambda: call_view(v_ok))
        expect_raise(f"shared-storage invalid view, round {rnd}", "must be sorted",
                     lambda: call_view(v_bad))

    # ...and the kill switch has to restore strict checking. Re-exec is the only way to
    # set it, since the flag is read once per process.
    import subprocess
    src = (
        "import numpy as np, torch, scorch_ops as ops\n"
        f"M,deg,J,N={M},{deg},{J},{N}\n"
        "pos=np.arange(M+1,dtype=np.int32)*deg\n"
        "crd=np.tile(np.arange(deg,dtype=np.int32),M)\n"
        "val=np.ones(M*deg,dtype=np.float32)\n"
        "a=dict(result_shape=[M,N],A_shape=[M,J],\n"
        "  A_mode_indices=[[],[torch.from_numpy(pos),torch.from_numpy(crd)]],\n"
        "  A_values=torch.from_numpy(val),B_shape=[J,N],B_mode_indices=[[],[]],\n"
        "  B_values=torch.zeros((J,N),dtype=torch.float32).reshape(-1))\n"
        "ops.spmm_csr_float_v2(**a)\n"
        "crd[deg],crd[deg+1]=crd[deg+1],crd[deg]\n"
        "try:\n"
        "    ops.spmm_csr_float_v2(**a); print('ACCEPTED')\n"
        "except Exception as e:\n"
        "    print('REJECTED' if 'must be sorted' in str(e) else 'OTHER')\n"
    )
    env = dict(os.environ, SCORCH_ABI_VALIDATE_MEMO="0")
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                         env=env).stdout.strip()
    if out.endswith("REJECTED"):
        PASSES += 1
        print("  ok    SCORCH_ABI_VALIDATE_MEMO=0 restores strict per-call validation")
    else:
        FAILS.append("memo kill switch")
        print(f"  FAIL  SCORCH_ABI_VALIDATE_MEMO=0 did not restore checking: {out!r}")


def run_inference_mode_suite():
    """Tensors made under torch.inference_mode() have their version counter DISABLED,
    and reading it raises. Both memos key on that counter, so both have to ask rather
    than assume — otherwise every sparse matmul inside an inference-mode block fails.
    """
    print("\n[inference_mode] version counters are disabled in here")
    global PASSES
    with torch.inference_mode():
        probe = torch.from_numpy(np.arange(4, dtype=np.int64))
        try:
            _ = probe._version
            print("  note  _version did not raise on this torch build")
        except RuntimeError:
            PASSES += 1
            print("  ok    _version raises, so the guards are load-bearing here")

        M, deg, J, N = 512, 4, 512, 8
        pos = np.arange(M + 1, dtype=np.int64) * deg
        crd = np.tile(np.arange(deg, dtype=np.int64), M)
        val = np.ones(M * deg)
        # native validator + its memo, three times so the memo path is taken
        for i in range(3):
            expect_ok(f"native validator under inference_mode, call {i}",
                      lambda: ops.spmm_csr_float_v2(**csr_args(M, J, N, pos, crd, val)))

        # and the python-side narrowing memo, through the public API
        try:
            import scipy.sparse
            import scorch
            sp = scipy.sparse.csr_matrix(
                (val.astype(np.float32), crd.astype(np.int32),
                 pos.astype(np.int32)), shape=(M, J))
            t = torch.sparse_csr_tensor(
                torch.from_numpy(sp.indptr.astype(np.int64)),
                torch.from_numpy(sp.indices.astype(np.int64)),
                torch.from_numpy(sp.data), size=sp.shape)
            A = scorch.STensor.from_torch(t)
            B = torch.ones(J, N)
            for i in range(3):
                out = scorch.matmul(A, B)
            ref = sp.astype(np.float64) @ np.ones((J, N))
            err = float(np.abs(out.numpy() - ref).max() / max(np.abs(ref).max(), 1e-30))
            if err < 1e-4:
                PASSES += 1
                print(f"  ok    scorch.matmul under inference_mode, relerr {err:.1e}")
            else:
                FAILS.append("inference_mode matmul accuracy")
                print(f"  FAIL  scorch.matmul under inference_mode relerr {err:.1e}")
        except Exception as exc:  # noqa: BLE001
            FAILS.append("inference_mode matmul")
            print(f"  FAIL  scorch.matmul under inference_mode: "
                  f"{type(exc).__name__}: {exc}")


run_suite(64, 4, "serial screen")            # 256 nnz
run_suite(40000, 16, "parallel screen")      # 640000 nnz, above the grain
run_sortedness_suite(64, 4, "sortedness, serial")
run_sortedness_suite(20000, 8, "sortedness, parallel")
run_memo_suite()
run_inference_mode_suite()

print(f"\n{PASSES} passed, {len(FAILS)} failed")
if FAILS:
    print("FAILED: " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
