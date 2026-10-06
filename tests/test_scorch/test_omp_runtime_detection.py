"""The OpenMP runtime the SpMM launch decisions use is the one the process loaded, not the header's.

spmm_csr_v2_core launches a multi-worker call differently under GNU libgomp, which releases and
re-creates pool threads whenever a team is narrower than the last, and under LLVM's libomp, which
does not. That used to be decided by the OpenMP header the build found first. A conda environment
holding MKL ships intel-openmp's omp.h in $PREFIX/include, ahead of GCC's, so a GCC build there
compiled the LLVM branch while running libgomp -- and every product under pool * 16 rows paid a
thread re-creation on every call beside a torch op (redwood: the next torch op took 186 us
instead of 16.6). The decision is now asked of the loaded process.

What these pin down:

* the runtime the module names is a library this process has actually loaded;
* the libgomp answer agrees with that library's file name;
* on Linux, it is the same OpenMP library torch's own parallel regions run on, so the SpMM and the
  ops around it share one pool;
* behaviourally, a few-row product called alternately with a full-width torch op replaces no
  threads under libgomp (the defect re-created 16 per call).
"""
import os
import sys

import pytest
import torch

scorch_ops = pytest.importorskip("scorch_ops")

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "scorch_omp_runtime_path"),
    reason="scorch_ops predates the run-time OpenMP runtime check",
)


def _mapped_libraries():
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc/self/maps")
    with open("/proc/self/maps") as f:
        return {line.split()[-1] for line in f if "/" in line.split()[-1]}


def test_the_named_runtime_is_loaded():
    path = scorch_ops.scorch_omp_runtime_path()
    assert path, "the OpenMP runtime could not be named"
    if sys.platform.startswith("linux"):
        assert os.path.realpath(path) in {os.path.realpath(p) for p in _mapped_libraries()}


def test_the_answer_matches_the_runtime_file_name():
    path = scorch_ops.scorch_omp_runtime_path()
    assert scorch_ops.scorch_omp_is_libgomp() == ("libgomp" in os.path.basename(path))


def test_it_is_the_runtime_torch_runs_on():
    # One OpenMP library in the process: the SpMM's regions and torch's share its pool. Two would
    # be a different defect (two pools oversubscribing the CPUs); either way the answer must name
    # one torch has loaded.
    torch.ones(1 << 16).sum()
    omp_libs = {os.path.realpath(p) for p in _mapped_libraries()
                if any(n in os.path.basename(p) for n in ("libgomp", "libomp", "libiomp"))}
    assert os.path.realpath(scorch_ops.scorch_omp_runtime_path()) in omp_libs
    assert len(omp_libs) == 1, omp_libs


def _threads_replaced(fn, iters=200):
    """Thread ids of this process that were not there before `iters` calls of fn. Ids come from
    the system-wide pid space and are not reused until it wraps, so a thread destroyed and
    re-created shows up as a new id -- provided the last region before the check re-grew the
    team, which a full-width torch op does."""
    def tids():
        return set(os.listdir("/proc/self/task"))
    for _ in range(20):
        fn()
    before = tids()
    for _ in range(iters):
        fn()
    return tids() - before


def test_a_few_row_product_beside_a_torch_op_creates_no_threads():
    if not sys.platform.startswith("linux") or not scorch_ops.scorch_omp_is_libgomp():
        pytest.skip("the resize exists under libgomp only")
    pool = torch.get_num_threads()
    if pool < 4:
        pytest.skip("needs a pool wide enough for a narrower team to exist")
    import scorch
    from scorch import STensor

    g = torch.Generator().manual_seed(0)
    rows, cols, k = 8 * pool, 4096, 4          # under pool * 16 rows: the defect's population
    dense = torch.rand(rows, cols, generator=g) * (torch.rand(rows, cols, generator=g) < 0.05)
    A = STensor.from_csr(dense.to_sparse_csr(), "A")
    B = torch.rand(cols, k, generator=g)
    X = torch.rand(1 << 18)
    Y = torch.empty_like(X)

    def step():
        scorch.matmul(A, B)
        torch.mul(X, 2.0, out=Y)              # a full-width torch region after every call

    got = scorch.matmul(A, B)
    got = got if isinstance(got, torch.Tensor) else got.to_torch()
    assert torch.allclose(got, dense @ B, atol=1e-3, rtol=1e-3)
    assert not _threads_replaced(step)
