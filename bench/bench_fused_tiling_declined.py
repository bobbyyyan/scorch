"""What does the fusion+tiling composition cost the shapes it cannot help?

The neutrality half of `bench_fused_tiling.py`. Every GCN-small layer, every
autoencoder layer, and anything whose dense operand fits in the last-level cache is
declined by the tiling gate, and on those the composition must not tax the fused call
it used to make directly.

Timed by isolation rather than end to end, deliberately. A declined consultation never
launches a kernel, so it can be timed on its own to a few nanoseconds; the end-to-end
difference on a 25-500 us fused call is a fraction of a percent and drowns in the
OpenMP contamination that six interleaved arms in one process produce (measured: an
A/A floor of 0.85-1.06 on exactly these cells, which resolves nothing).

The comparison that matters is not "is it free" -- it is not -- but "is it more than
`ops.matmul` already pays". The gate is the same mechanism, asked by a second caller.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import scorch  # noqa: E402
from scorch import ops, tiling  # noqa: E402
from scorch.prebuilt_kernels import (  # noqa: E402
    resolve_prebuilt_fused,
    resolve_prebuilt_matmul,
)
from scorch.stensor import STensor  # noqa: E402
from scorch.trace import (  # noqa: E402
    _TILED_TAILS,
    _fused_kernel_args,
    _try_tiled_fused,
)

def old_gate(a, b, resolved, _tls=tiling._tls):
    """The gate exactly as it stood before this work, replicated so the two can be
    timed in one process against one binary.

    Three differences, all in the shared layer: the thread-local level was read with
    `getattr(..., default)` (which raises and catches an AttributeError when no
    override is active, the common case), nnz was read off the index arrays before the
    cache test that rejects it, and the operand-over-cache boundary lived inside
    `_eligible` rather than being callable on its own.
    """
    if resolved.symbol_name != "spmm_csr_float_v2":
        return None, False
    lvl = getattr(_tls, "level", None)
    level = lvl if lvl is not None else tiling._global_level
    if not tiling._HAS_TILEJ or level == "off" or b.dim() != 2:
        return level, False
    J = int(a.shape[1])
    N = int(b.shape[1])
    nnz = int(a.storage._mode_indices[1][1].numel())
    C = tiling.query_llc()
    if level == "learned" and tiling._LEARNED_WIDEN and tiling._load_learned_model():
        return level, (J * 4 * N) > C
    operand = J * 4 * N
    if operand <= C:
        return level, False
    deg = nnz / max(1, J)
    return level, deg > max(tiling._DEG_FLOOR, 2.0 * operand / C)


SHAPES = [
    ("cora-class", 2708, 4, 16),
    ("citeseer-class", 3327, 3, 16),
    ("pubmed-class", 19717, 5, 16),
    ("arxiv-class", 169343, 7, 128),
    ("ae-class", 4096, 40, 128),
    ("ae-wide", 8192, 20, 256),
]


def synthetic_csr(rows, degree, seed=7):
    """Fixed-degree CSR. Columns are sampled with replacement, so a row may repeat a
    column -- legal for these kernels, and irrelevant here because the gate reads only
    (rows, cols, nnz, N, C) and the degree."""
    generator = torch.Generator().manual_seed(seed)
    indptr = torch.arange(rows + 1, dtype=torch.int64) * degree
    indices = torch.randint(0, rows, (rows * degree,), generator=generator)
    indices = indices.view(rows, degree).sort(dim=1).values.reshape(-1)
    values = torch.randn(rows * degree, generator=generator)
    return STensor.from_torch(
        torch.sparse_csr_tensor(indptr, indices, values, size=(rows, rows))
    )


def best(fn, reps):
    """Minimum mean-per-call over 5 batches of `reps`."""
    fn()
    lowest = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        lowest = min(lowest, (time.perf_counter() - t0) / reps)
    return lowest * 1e6


def main():
    scorch.set_autotune("balanced")
    print(f"host threads: {torch.get_num_threads()}   "
          f"LLC: {tiling.query_llc() / 2**20:.1f} MiB   level: balanced", flush=True)
    header = (
        f"{'shape':<16} {'nnz':>10} {'N':>5} {'old_gate':>9} {'gate_us':>9} "
        f"{'added_us':>9} {'fused_us':>9} {'gate%':>7} {'added%':>7} {'old/new':>8}"
    )
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)

    for label, rows, degree, n in SHAPES:
        a = synthetic_csr(rows, degree)
        nnz = int(a.storage._mode_indices[1][1].numel())
        b_dense = torch.randn(rows, n)
        b = STensor.from_torch(b_dense)
        bias = torch.randn(n)
        if tiling.is_candidate(a, b):
            print(f"{label:<16} SKIPPED: the gate opens on this shape", flush=True)
            continue

        spmm_resolved = resolve_prebuilt_matmul(a, b, output_format="dd")
        fused = resolve_prebuilt_fused("d,s", "d,d", ("add", "relu"), torch.float32)
        tail = _TILED_TAILS[("add", "relu")]
        result_shape = [rows, n]

        # (1) the gate on its own -- the whole added cost on a declined shape, and the
        #     same call ops.matmul makes on every prebuilt CSR@dense product.
        # Interleaved, so neither ordering favours either arm.
        gate_us = float("inf")
        old_us = float("inf")
        for _ in range(3):
            gate_us = min(gate_us, best(lambda: ops.tiling_gate(a, b, spmm_resolved), 50000))
            old_us = min(old_us, best(lambda: old_gate(a, b, spmm_resolved), 50000))
        assert old_gate(a, b, spmm_resolved)[1] == ops.tiling_gate(a, b, spmm_resolved)[1]

        # (2) what the runner adds around it: asking the gate and branching on it. If
        #     the gate says no, nothing else in the composition runs.
        def added():
            gate = ops.tiling_gate(a, b, spmm_resolved)
            if gate[1]:
                return _try_tiled_fused(
                    a, b, bias, spmm_resolved, fused.symbol_name, fused.fn, tail, gate
                )
            return None

        added_us = best(added, 50000)

        # (3) the fused kernel this composition sits in front of.
        # The ordinary fused call, marshalled exactly as the runner marshals it.
        fused_us = best(
            lambda: fused.fn(*_fused_kernel_args(result_shape, a, b, bias)), 2000
        )

        print(
            f"{label:<16} {nnz:>10,} {n:>5} {old_us:>9.3f} {gate_us:>9.3f} "
            f"{added_us:>9.3f} {fused_us:>9.1f} {gate_us / fused_us * 100:>6.2f}% "
            f"{added_us / fused_us * 100:>6.2f}% {old_us / gate_us:>8.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
