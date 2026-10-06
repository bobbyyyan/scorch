"""Bring a fresh process to its steady state before it times anything.

Every harness in this directory starts a process, loads a matrix single-threaded, and then times
arms. Measured on mkt1 (24 pinned threads, pre2 at float32 width 1 and debr_G_18 at float64 width
1, 2026-10-05), the first parallel work in a process runs slow for about 20 ms, whichever library
does it:

  scorch.matmul first:   calls 2-24 at 514-536 us, then 230-235 (pre2)
  MKL first:             calls 2-24 at 895-942 us, then 367-370; scorch.matmul after it at 233-253
  the plan's own run first:  392-428 us, then scorch.matmul at 237-249

It happens once per process. It does not come back after a later matrix is loaded, after 1 s of
idling, or after 150 ms of single-threaded work, and every call in it took zero page faults. So it
is not an allocator or idle-clock effect but the process's first use of the thread team and of the
cores it is pinned to. A harness that always times one library first charges that library alone:
full_api.py timed scorch first, and its first cell in a process read 0.84 of MKL on a cell where
scorch is 1.59x faster (pre2 float32 k=1: 445 against 234 us).

`settle()` runs an all-thread, bandwidth-bound torch op -- the same OpenMP team both libraries run
on -- for SECONDS before the first measurement. After 300 ms of it every arm read its steady state
in 4 of 4 processes (probe4, warm_first). Call it once per process, after imports and before the
first timed call.
"""
import time

import torch

SECONDS = 0.3


def settle(seconds: float = SECONDS) -> None:
    # 64 MB, past every last-level cache on the hosts this runs on, so the op streams memory.
    w = torch.ones(1 << 24)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        torch.mul(w, 1.0, out=w)
