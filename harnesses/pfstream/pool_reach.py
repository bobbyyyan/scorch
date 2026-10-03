#!/usr/bin/env python3
"""Reachability of two pool-dependent SpMM predicates on x86, at the DECISION level.

Both predicates carry a source comment asserting they are inert on x86. Each comment
reasons from the DEFAULT pool, and neither says so:

  scorch_policy.h:2313   if ((long)nthreads >= 2 * pool) nt_cap = 0;
      "On an all-physical-cores pool no recruit is EVER at stake (x86: pool 24, gate at
       48, count at most 32), so this condition cannot fire there"

  scorch_policy.h:2612   if (pool > gate_maxthreads) mingrains = 0;
      "Above the ceiling this whole block is unreachable, which is how the x86 reading
       is preserved."

In both, `pool` is the caller's thread count, which a caller sets with
torch.set_num_threads. So the question is not whether they fire at the default pool --
the comments establish that -- but whether an ordinary caller can reach them.

This asks production rather than restating it. scorch_spmm_nthreads and
scorch_spmm_partition_mode are exported for exactly this reason, and a harness that
recomputed either predicate would be wrong silently and flatteringly.

Detection, for each predicate, from the exported value alone:

  #1 the recruit outcome IS `resolved >= 2 * pool` -- that is the condition the launch
     site tests to fork a private team, so observing the resolved count reach it is
     observing the recruit become available. No internal state needed.
  #2 the gate's effect is to force partition_mode to 0 on cells below the work proxy.
     So compare the SAME cell at a small pool and a large one: a cell that resolves
     nonzero above the ceiling and 0 below it is a cell where the gate fired.

Beside every "never fired" the probe prints the quantity that shows the grid was not
empty -- the maximum resolved count and the ratio it reached against the threshold.
"""
import os
import sys

import torch
import scorch  # noqa: F401

# The extension is a TOP-LEVEL module sitting next to the package in src/, not a
# submodule of it, so `from scorch import scorch_ops` fails. Try both and say which.
try:
    import scorch_ops as SO
    _how = "top-level scorch_ops"
except ImportError:
    from scorch import scorch_ops as SO
    _how = "scorch.scorch_ops"

GRAIN = None
# Capped at 32 so the largest gate (2x32 = 64) is reachable inside a 64-CPU
# allocation. mkt1 would not grant 128, and a pool whose gate exceeds the
# allocation reports the allocation rather than the policy.
POOLS = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32]
# rows x nnz x k, spanning short kernels to large ones in both axes
ROWS = [64, 512, 4096, 65536, 1000000]
NNZ = [3_000, 50_000, 1_000_000, 20_000_000, 200_000_000]
KS = [1, 2, 4, 8, 16, 32, 128]


def main():
    # The probe's own premise, checked before anything is read from it. The resolved
    # thread count is bounded by the CPUs the process can actually run on, so inside a
    # one-CPU Slurm allocation every pool resolves to the same tiny number and the sweep
    # measures the allocation rather than the policy. The first run of this probe did
    # exactly that: it reported predicate 1 "reachable at pool 1" because the allocation
    # had 2 CPUs, which made 2 >= 2*1 true for reasons having nothing to do with x86.
    avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print("host %s  nproc %d  CPUs this process may use %d  torch default pool %d"
          % (os.uname().nodename, os.cpu_count(), avail, torch.get_num_threads()))
    need = 2 * max(POOLS)
    print("PREMISE: predicate 1's largest gate is 2 x %d = %d, so a verdict of 'never"
          " reached'" % (max(POOLS), need))
    print("         is only informative where the process may use at least that many CPUs.")
    if avail < need:
        print("  !! this allocation offers %d. Pools above %d cannot even in principle"
              % (avail, avail // 2))
        print("  !! reach their gate here, so their rows report the ALLOCATION, not the policy.")
        print("  !! Rerun with at least %d CPUs before reading predicate 1." % need)
    print("extension imported as %s from %s" % (_how, SO.__file__))
    print("exports present: scorch_spmm_nthreads=%s scorch_spmm_partition_mode=%s"
          % (hasattr(SO, "scorch_spmm_nthreads"), hasattr(SO, "scorch_spmm_partition_mode")))
    if not (hasattr(SO, "scorch_spmm_nthreads") and hasattr(SO, "scorch_spmm_partition_mode")):
        print("EXIT: the policy is not exported in this build; nothing below could be read")
        return 1

    cells = [(r, n, k) for r in ROWS for n in NNZ for k in KS]
    print("grid: %d pools x %d cells (rows x nnz x k) = %d decisions per predicate\n"
          % (len(POOLS), len(cells), len(POOLS) * len(cells)))

    # ---------------------------------------------------------------- predicate 1
    print("=" * 78)
    print("PREDICATE 1  scorch_policy.h:2313   resolved_nthreads >= 2 * pool")
    print("  the recruit becomes AVAILABLE exactly when this holds, so reaching it is the")
    print("  observable. Comment asserts it cannot hold on x86.")
    print("=" * 78)
    print("  %-6s %-9s %-9s %-9s %-7s %s"
          % ("pool", "gate=2p", "max resol", "max/gate", "n fire", "firing example rows/nnz/k"))
    any_fire = False
    for pool in POOLS:
        best, nfire, ex = 0, 0, None
        for (r, n, k) in cells:
            work = n * max(k, 16)
            got = SO.scorch_spmm_nthreads(work, r, pool, n * k, n)
            if got > best:
                best = got
            if got >= 2 * pool:
                nfire += 1
                if ex is None:
                    ex = (r, n, k, got)
        any_fire = any_fire or nfire > 0
        print("  %-6d %-9d %-9d %-9.2f %-7d %s"
              % (pool, 2 * pool, best, best / (2.0 * pool), nfire,
                 ("%d/%d/%d -> %d threads" % ex) if ex else "-"))
    print("\n  VERDICT predicate 1: %s"
          % ("REACHABLE on this host at the pools marked above"
             if any_fire else "never reached over this grid"))

    # ---------------------------------------------------------------- predicate 2
    print()
    print("=" * 78)
    print("PREDICATE 2  scorch_policy.h:2612   pool > gate_maxthreads disables the gate")
    print("  so the gate is ACTIVE at small pools. A cell whose mode is nonzero at a large")
    print("  pool and 0 at a small one is a cell the ARM-derived work gate forced to 0.")
    print("=" * 78)
    REF = 32
    print("  %-6s %-10s %-10s %-10s %s"
          % ("pool", "cells n>0", "forced to 0", "share", "example rows/nnz/k"))
    for pool in POOLS:
        forced, live, ex = 0, 0, None
        for (r, n, k) in cells:
            work = n * max(k, 16)
            nt_s = SO.scorch_spmm_nthreads(work, r, pool, n * k, n)
            nt_r = SO.scorch_spmm_nthreads(work, r, REF, n * k, n)
            m_s = SO.scorch_spmm_partition_mode(r, n, k, k, 4, nt_s, pool)
            m_r = SO.scorch_spmm_partition_mode(r, n, k, k, 4, nt_r, REF)
            if m_r != 0:
                live += 1
                if m_s == 0:
                    forced += 1
                    if ex is None:
                        ex = (r, n, k)
        print("  %-6d %-10d %-10d %-10s %s"
              % (pool, live, forced,
                 ("%.1f%%" % (100.0 * forced / live)) if live else "n/a",
                 ("%d/%d/%d" % ex) if ex else "-"))
    print("\n  the 'cells n>0' column is the non-vacuity check: a zero there would make the")
    print("  'forced to 0' column unable to report anything, and the two must be read together.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
