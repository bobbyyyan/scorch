"""Every parallel region scorch's native code opens, and who sizes its team.

The thread ceiling (scorch_policy.h, "The caller's thread setting") promises two bounds:
  1. no team is wider than the caller's thread count, when the caller has set one;
  2. a team scorch sizes itself is also no wider than the processors this process may use.
A region whose team the runtime sizes -- a ``#pragma omp parallel`` with no num_threads clause,
whose team is the OpenMP count torch.set_num_threads and OMP_NUM_THREADS set, or torch's own
at::parallel_for / at::parallel_reduce pool -- takes the caller's count, as torch's own ops do, and
so keeps bound 1; above the processors it oversubscribes exactly as torch does when asked to.

This test lists every such site in src/scorch/csrc, one line of source each (whitespace collapsed,
comments and string literals skipped), with how many times it occurs and which of the kinds below
sizes its team. A site added, removed or changed fails here until it is classified, so no new
region can escape the ceiling unseen. test_thread_ceiling counts the teams the runtime actually
runs; generated kernels are covered there (the ``jit`` rows), since their regions are emitted text.
"""

import collections
import pathlib
import re

CEILING = "scorch-sized: scorch_nthreads, the SpMM thread decision or the zero-fill, each capped by scorch_thread_ceiling()"
RUNTIME = "runtime-sized: no num_threads clause, so the OpenMP count -- the caller's"
POOL = "torch's intra-op pool (at::parallel_for / at::parallel_reduce): the caller's count"
TORCH_COUNT = "at::get_num_threads(), or 4 when fewer suit the input: the caller's count"
PASSED = "the count its generated caller passes, the region's own team count"

SITES = {
    ("header.h", "#pragma omp parallel for num_threads(threads) schedule(dynamic)"): (2, PASSED),
    ("header.h", "#pragma omp parallel num_threads(thread_count)"): (1, CEILING),
    ("kernels.h", "#pragma omp parallel"): (1, RUNTIME),
    ("kernels.h", "#pragma omp parallel for schedule(static)"): (1, RUNTIME),
    ("kernels.h", "#pragma omp parallel num_threads(nthreads)"): (5, CEILING),
    ("native_abi.h", "const int bad = at::parallel_reduce("): (2, POOL),
    ("native_abi.h", "const std::pair<int64_t, int64_t> range = at::parallel_reduce("): (1, POOL),
    ("native_abi.h", "return at::parallel_reduce(begin, end, split, ident, body, combine);"): (1, POOL),
    ("sparse_grad.h", "#pragma omp parallel num_threads(nthreads)"): (2, CEILING),
    ("sparse_grad.h", "at::parallel_for(0, (int64_t)nthreads, 1, [&](int64_t wbeg, int64_t wend) {"): (1, POOL),
    ("sparse_grad.h", "at::parallel_for(0, n, (n + pool - 1) / pool,"): (1, POOL),
    ("spmm.h", "#pragma omp parallel"): (3, RUNTIME),
    ("spmm.h", "#pragma omp parallel for"): (3, RUNTIME),
    ("spmm.h", "#pragma omp parallel for schedule(dynamic)"): (1, RUNTIME),
    ("spmm.h", "#pragma omp parallel for schedule(dynamic, 16)"): (6, RUNTIME),
    ("spmm.h", "#pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads)"): (1, CEILING),
    ("spmm.h", "#pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)"): (2, CEILING),
    ("spmm.h", "#pragma omp parallel for schedule(static)"): (2, RUNTIME),
    ("spmm.h", "#pragma omp parallel for schedule(static) num_threads(nthreads)"): (3, CEILING),
    ("spmm.h", "#pragma omp parallel num_threads(nthreads)"): (4, CEILING),
    ("spmm.h", "#pragma omp parallel num_threads(num_threads)"): (2, TORCH_COUNT),
    ("spmm.h", "at::parallel_for(0, (int64_t)C1_size, fold_grain, [&](int64_t j0, int64_t j1) {"): (1, POOL),
    ("spmm.h", "at::parallel_for(0, (int64_t)nthreads, 1, [&](int64_t wbeg, int64_t wend) {"): (2, POOL),
    ("spmm.h", "at::parallel_for(0, S, 8, [&](int64_t r0, int64_t r1) {"): (1, POOL),
    ("spmm.h", "at::parallel_for(0, ncblk * nbands, 1, [&](int64_t t0, int64_t t1) {"): (1, POOL),
    ("spmm.h", "at::parallel_for(0, ncblk, grain_blocks,"): (1, POOL),
    ("spmm.h", "at::parallel_for(0, nrows, 128,"): (1, POOL),
}

_PATTERN = re.compile(r"pragma omp parallel|at::parallel_for|at::parallel_reduce|omp_set_num_threads")
_CSRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "scorch" / "csrc"


def _sites():
    found = collections.Counter()
    for path in sorted(_CSRC.iterdir()):
        if path.suffix not in (".h", ".cpp"):
            continue
        for line in path.read_text().splitlines():
            text = " ".join(line.split())
            if not _PATTERN.search(text) or text.startswith(("//", "/*", "*", '"', "#include")):
                continue
            found[(path.name, text)] += 1
    return found


def test_every_parallel_site_is_classified():
    found = _sites()
    expected = {key: count for key, (count, _) in SITES.items()}
    unclassified = {key: n for key, n in found.items() if key not in expected}
    gone = {key: n for key, n in expected.items() if key not in found}
    recounted = {key: (expected[key], n) for key, n in found.items() if key in expected and expected[key] != n}
    assert not (unclassified or gone or recounted), (
        "classify every parallel site (test_thread_sites.SITES):\n"
        "  new or changed: %r\n  no longer present: %r\n  count changed (was, now): %r"
        % (unclassified, gone, recounted)
    )


def test_no_site_changes_the_process_count():
    """omp_set_num_threads would change the caller's count -- torch's too, the same runtime value --
    for the rest of the process (the experimental kernels did, test_thread_ceiling)."""
    assert not [key for key in _sites() if "omp_set_num_threads" in key[1]]
