"""The caller's thread setting caps every team scorch launches (scorch_policy.h, "The caller's
thread setting").

A setting is read when scorch_ops loads and on every call, so each case runs in a fresh Python
process: the route (how the count is set, and whether before or after ``import scorch``) is applied
there, one kernel runs, and the process reports what the policy answered and how many threads the
kernel created. An OpenMP runtime creates a team's workers on the first region of that width and
keeps them, so after a warm-up torch op (which builds torch's own pool) the number of threads a
kernel adds is how much wider than torch's pool its widest team was. Counting threads, rather than
reading the policy's return value, checks the team the runtime actually ran.

Routes:
  none              nothing set
  torch_before_1    torch.set_num_threads(1) BEFORE ``import scorch``. The documented gap: the
                    count at import is the baseline, so this reads as no setting. It is also the
                    positive control -- torch's pool is 1, so an uncapped team shows as added threads.
                    The gap arises only when ``torch`` itself is imported and set first; the
                    drop-in spelling ``import scorch as torch; torch.set_num_threads(n)`` reaches
                    scorch.set_num_threads and has no gap.
  torch_after_1     torch.set_num_threads(1) after ``import scorch``
  scorch_set_1      scorch.set_num_threads(1)
  alias_set_1       ``import scorch as torch; torch.set_num_threads(1)`` (the drop-in spelling)
  env_omp_1         OMP_NUM_THREADS=1 in the environment at start
"""

import json
import os
import subprocess
import sys

import pytest

_CHILD = r'''
import json, os, subprocess, sys
route, kernel = sys.argv[1], sys.argv[2]


def tasks():
    """This process's threads by kernel name (Linux), or None where /proc has no task list."""
    if not os.path.isdir("/proc/self/task"):
        return None
    names = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open("/proc/self/task/%s/comm" % tid) as f:
                names[tid] = f.read().strip()
        except OSError:
            pass  # the thread ended between the listing and the read
    return names


def main_name(names):
    """The main thread's name as it is now: a torch that names its main thread at import (some name
    it "pt_main_thread") does so before any team exists, so every worker inherits the new name."""
    return names.get(str(os.getpid()))


def threads():
    """The threads an OpenMP runtime can have made. On Linux, those that carry the main thread's
    current name: libgomp never names its workers, so each inherits the name of the thread that
    created it, while a library that starts threads of its own usually names them -- on a GPU host
    the first generated kernel call reaches the CUDA driver's cuInit, which starts threads named
    "cuda..." (sapling 84423) that are no team's. Those are listed in the report (other_threads) and
    not counted. On macOS (ps -M, no names) every thread counts."""
    names = tasks()
    if names is not None:
        main = main_name(names)
        return sum(1 for name in names.values() if name == main)
    out = subprocess.run(["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True,
                         check=True).stdout
    return len(out.strip().splitlines()) - 1


def other_threads():
    names = tasks()
    if names is None:
        return []
    main = main_name(names)
    return sorted(n for n in names.values() if n != main)


def procs():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


import torch
t_torch = threads()
if route.startswith("torch_before_1"):
    torch.set_num_threads(1)
load_count = torch.get_num_threads()
import scorch
import scorch_ops
if route.endswith("_nocapsule"):
    del scorch_ops._thread_policy  # no kernel module can find scorch_ops' state from here on
# Threads that exist before the route applies. If importing scorch had run a parallel region, its
# workers would already be in the pool and a kernel's team could reuse them without adding any; the
# parent refuses such a reading instead of passing it.
report = {"route": route, "kernel": kernel, "procs": procs(), "load_count": load_count,
          "uncapped_at_load": scorch_ops.thread_policy_uncapped_at(), "t_torch": t_torch,
          "t_scorch": threads()}
if route == "torch_after_1":
    torch.set_num_threads(1)
elif route == "scorch_set_1":
    scorch.set_num_threads(1)
elif route == "alias_set_1" or route == "alias_set_2":
    import scorch as torch_alias
    n = int(route.rsplit("_", 1)[1])
    report["alias_is_scorch"] = torch_alias.set_num_threads is scorch.set_num_threads
    torch_alias.set_num_threads(n)
elif route == "torch_after_2":
    torch.set_num_threads(2)
elif route == "torch_reset":
    torch.set_num_threads(2 if load_count != 2 else 3)
    torch.set_num_threads(load_count)
elif route == "scorch_set_load":
    scorch.set_num_threads(load_count)
elif route == "scorch_set_above":
    scorch.set_num_threads(2 * procs())
elif route == "scorch_set_invalid":
    try:
        scorch.set_num_threads(0)
        report["raised"] = None
    except Exception as error:
        report["raised"] = type(error).__name__
report["torch_count"] = torch.get_num_threads()
report["ceiling"] = scorch_ops.thread_ceiling()
report["uncapped_at"] = scorch_ops.thread_policy_uncapped_at()

if kernel != "policy":
    import numpy as np
    rng = np.random.default_rng(20261014)
    others_before = other_threads()
    if kernel == "dense":
        before = threads()
        torch.ones(1 << 24).mul_(2.0).sum()
        report["added"] = threads() - before
    else:
        import scipy.sparse
        # torch's own pool exists, at torch's whole count, before anything is counted: an elementwise
        # op takes min(count, elements / 32768) workers, so the op is at least 2 x 32768 per thread.
        torch.ones(max(1 << 22, 65536 * torch.get_num_threads())).mul_(2.0).sum()
        n, per_row = 4000, 32
        cols = np.concatenate([np.sort(rng.choice(n, per_row, replace=False)) for _ in range(n)])
        crow = torch.arange(0, n * per_row + 1, per_row, dtype=torch.int64)
        col = torch.from_numpy(cols.astype(np.int64))
        dtype = torch.float32 if kernel.startswith("spmm") else torch.float64
        values = rng.uniform(-1, 1, n * per_row)
        a = torch.sparse_csr_tensor(crow, col, torch.from_numpy(values).to(dtype), size=(n, n))
        a64 = scipy.sparse.csr_matrix((values, cols, crow.numpy()), shape=(n, n))
        if kernel == "spmspm":
            run = lambda: scorch.matmul(a, a)
            ref = torch.from_numpy((a64 @ a64).toarray())
        elif kernel in ("spmm_k64", "spmm_k128", "jit"):
            k = 128 if kernel == "spmm_k128" else 64
            x = torch.from_numpy(rng.uniform(-1, 1, (n, k))).to(dtype)
            if kernel == "jit":
                run = lambda: scorch.matmul(a, x, use_cache=False)
            else:
                run = lambda: scorch.matmul(a, x)
            ref = torch.from_numpy(a64 @ x.double().numpy())
        before = threads()
        got = run()
        report["added"] = threads() - before
        if not isinstance(got, torch.Tensor):
            got = got.to_torch()
        if got.layout != torch.strided:
            got = got.to_dense()
        err = float((got.double() - ref).abs().max() / ref.abs().max())
        report["error"] = err
        tol = 1e-5 if dtype == torch.float32 else 1e-12
        assert err <= tol, report
    others = other_threads()
    for name in others_before:
        others.remove(name)
    report["other_threads"] = others  # started meanwhile, not counted (see threads)
    report["ceiling_after"] = scorch_ops.thread_ceiling()
    report["torch_count_after"] = torch.get_num_threads()
print("REPORT " + json.dumps(report))
'''

_ULTRA_CHILD = r'''
import json, sys
import torch
import scorch
import scorch_ops
from scorch import STensor

load_count = torch.get_num_threads()
fn = getattr(scorch_ops, sys.argv[1])


def call(rows, density):
    g = torch.Generator().manual_seed(rows)
    a = torch.rand((rows, 512), generator=g) * (torch.rand((rows, 512), generator=g) < density)
    b = torch.rand((512, 64), generator=g)
    sa = STensor.from_torch(a, "A").to_sparse("ds")
    sb = STensor.from_torch(b, "B")
    out = fn([rows, 64], list(sa.shape), sa.index.mode_indices, sa.values, list(sb.shape),
             sb.index.mode_indices, sb.values)
    got = out.storage.value.view(rows, 64)
    assert torch.allclose(got, a @ b, atol=1e-3, rtol=1e-3)


call(500, 0.05)  # fewer than 1000 rows: the kernel takes min(count, 4) workers
after_small = torch.get_num_threads()
call(12000, 0.05)  # 12000 rows of about 26 nonzeros: the kernel takes the whole count
print("REPORT " + json.dumps({"load_count": load_count, "after_small": after_small,
                              "after_large": torch.get_num_threads()}))
'''


def _run(source, *args, env_extra=None, timeout=900):
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env.update(env_extra or {})
    done = subprocess.run([sys.executable, "-c", source, *args], env=env, capture_output=True,
                          text=True, timeout=timeout)
    assert done.returncode == 0, done.stderr[-4000:]
    lines = [line for line in done.stdout.splitlines() if line.startswith("REPORT ")]
    assert len(lines) == 1, done.stdout[-4000:]
    return json.loads(lines[0][len("REPORT "):])


def _child(route, kernel="policy"):
    env = {}
    if route.startswith("env_omp_"):
        env["OMP_NUM_THREADS"] = route.rsplit("_", 1)[1]
    elif route.startswith("env_mkl_"):
        env["MKL_NUM_THREADS"] = route.rsplit("_", 1)[1]
    return _run(_CHILD, route, kernel, env_extra=env)


def _team(route, kernel):
    """A kernel's report, refused when importing scorch had already grown the thread pool."""
    r = _child(route, kernel)
    assert r["t_scorch"] == r["t_torch"], (
        "importing scorch ran a parallel region, so a kernel's team could reuse its workers "
        "without adding a thread; this reading cannot see the team: %r" % (r,))
    return r


_COUNTABLE = os.path.isdir("/proc/self/task") or sys.platform == "darwin"


def _wide_enough():
    procs = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return procs >= 4


# --- what the policy answers -------------------------------------------------------------------


def test_nothing_set_keeps_todays_ceiling():
    r = _child("none")
    assert r["uncapped_at_load"] == r["load_count"]
    assert r["ceiling"] == r["procs"]


def test_torch_set_after_import_caps():
    r = _child("torch_after_2")
    assert r["torch_count"] == 2
    assert r["ceiling"] == min(2, r["procs"])


def test_scorch_set_caps_even_at_the_load_count():
    r = _child("scorch_set_load")
    assert r["uncapped_at"] == -1
    assert r["ceiling"] == min(r["load_count"], r["procs"])


def test_drop_in_spelling_reaches_scorch_and_sets_torch():
    r = _child("alias_set_2")
    assert r["alias_is_scorch"]
    assert r["torch_count"] == 2
    assert r["uncapped_at"] == -1
    assert r["ceiling"] == min(2, r["procs"])


def test_count_above_the_processors_gives_the_processors():
    r = _child("scorch_set_above")
    assert r["torch_count"] == 2 * r["procs"]
    assert r["ceiling"] == r["procs"]


def test_rejected_count_records_nothing():
    r = _child("scorch_set_invalid")
    assert r["raised"] is not None
    assert r["uncapped_at"] == r["load_count"]


@pytest.mark.parametrize("route", ["env_omp_1", "env_omp_4", "env_mkl_1", "env_mkl_4"])
def test_environment_at_start_caps(route):
    r = _child(route)
    assert r["uncapped_at_load"] == -1
    assert r["ceiling"] == min(r["torch_count"], r["procs"])
    if route.startswith("env_omp_"):
        assert r["torch_count"] == int(route.rsplit("_", 1)[1])


def test_gap_set_before_import_reads_as_no_setting():
    """The first documented gap: torch.set_num_threads before ``import scorch`` is the baseline.

    Only when ``torch`` itself is imported and set first; scorch.set_num_threads, or the drop-in
    spelling ``import scorch as torch; torch.set_num_threads(n)``, has no gap.
    """
    r = _child("torch_before_1")
    assert r["load_count"] == 1 and r["uncapped_at_load"] == 1
    assert r["ceiling"] == r["procs"]


def test_gap_resetting_the_load_count_reads_as_no_setting():
    """The second documented gap: torch.set_num_threads back to the count at import."""
    r = _child("torch_reset")
    assert r["torch_count"] == r["load_count"]
    assert r["ceiling"] == r["procs"]


# --- the teams the runtime runs -----------------------------------------------------------------

_KERNELS = ["spmspm", "spmm_k64", "spmm_k128", "jit"]
# Kernels whose team, with no setting in force, is wider than a pool of 1: the prebuilt SpGEMM, the
# drop-in SpMM's recruit decline (nnz * k = 16.4M, over its 10M threshold) and a generated kernel
# with its zero-fill. The drop-in SpMM below the threshold already keeps to torch's pool.
_WIDE_UNCAPPED = {"spmspm", "spmm_k128", "jit"}


@pytest.mark.skipif(not _COUNTABLE, reason="threads are counted from /proc or ps -M")
@pytest.mark.parametrize("kernel", _KERNELS)
def test_positive_control_uncapped_team_is_seen(kernel):
    """With torch's pool at 1 and no setting in force, the wide kernels add threads.

    This is what makes the zero readings below evidence: the count can see a team wider than
    torch's pool, and for the generated kernel, a first-ever generated call that runs a threaded
    region in a fresh process finds scorch_ops' state through the capsule (a module that could not
    would cap at torch's count of 1 and add nothing).
    """
    if not _wide_enough():
        pytest.skip("fewer than 4 processors: a capped team cannot be told from an uncapped one")
    r = _team("torch_before_1", kernel)
    if kernel in _WIDE_UNCAPPED:
        assert r["added"] >= 1, r
    else:
        assert r["added"] == 0, r


@pytest.mark.skipif(not _COUNTABLE, reason="threads are counted from /proc or ps -M")
def test_kernel_module_without_the_capsule_stays_within_torchs_count():
    """A generated kernel module that cannot import scorch_ops._thread_policy caps at torch's count.

    The same process as the positive control -- torch's pool 1 before import, so no setting is in
    force and a module that finds the state runs a wide team -- with the capsule removed before the
    first generated call. The module must then take min(torch's count, processors) = 1, add no
    thread, and still return the right values.
    """
    if not _wide_enough():
        pytest.skip("fewer than 4 processors: a capped team cannot be told from an uncapped one")
    r = _team("torch_before_1_nocapsule", "jit")
    assert r["error"] <= 1e-12
    assert r["added"] == 0, r


@pytest.mark.skipif(not _COUNTABLE, reason="threads are counted from /proc or ps -M")
@pytest.mark.parametrize("route", ["torch_after_1", "scorch_set_1", "alias_set_1", "env_omp_1"])
@pytest.mark.parametrize("kernel", _KERNELS)
def test_one_thread_means_one_thread(route, kernel):
    if not _wide_enough():
        pytest.skip("fewer than 4 processors: a capped team cannot be told from an uncapped one")
    r = _team(route, kernel)
    assert r["ceiling"] == 1
    assert r["added"] == 0, r


@pytest.mark.skipif(not _COUNTABLE, reason="threads are counted from /proc or ps -M")
@pytest.mark.parametrize("kernel", _KERNELS)
def test_a_count_above_the_processors_adds_no_thread(kernel):
    """scorch.set_num_threads(2 x processors): torch's pool grows to that count on the warm-up op,
    and the kernel adds no thread to it. The teams scorch sizes itself stop at the processors (the
    ceiling, test_count_above_the_processors_gives_the_processors); a region sized by the runtime --
    no num_threads clause, or torch's at::parallel_for -- takes the caller's count, as torch's own
    ops do (test_thread_sites lists every such region).

    The count can fall: when a team narrower than the pool starts, libgomp lets the surplus pool
    threads exit. So this sees the kernel's last team, not every team; a team wider than the count
    followed by a narrower one would pass. Every team's width is read by the ceiling's diagnostic
    job, which wraps each GOMP_parallel entry; here the claim is only that no thread is added."""
    r = _team("scorch_set_above", kernel)
    assert r["torch_count"] == 2 * r["procs"]
    assert r["added"] <= 0, r


@pytest.mark.skipif(not _COUNTABLE, reason="threads are counted from /proc or ps -M")
def test_drop_in_spelling_gives_torch_ops_the_count():
    """``import scorch as torch; torch.set_num_threads(2)`` sets torch's own count: a dense torch
    op then runs a team of 2, one thread beside the caller's."""
    if not _wide_enough():
        pytest.skip("fewer than 4 processors")
    r = _team("alias_set_2", "dense")
    assert r["torch_count"] == 2
    assert r["added"] == 1, r


@pytest.mark.parametrize("name", ["spmm_csr_float_ultra", "spmm_csr_float_apex"])
def test_experimental_kernels_leave_the_count_alone(name):
    """They took min(count, 4) workers on a small input by setting the process's count, and their
    "restore" read back the value just set, so one small call left torch's count at 4."""
    r = _run(_ULTRA_CHILD, name)
    if r["load_count"] <= 4:
        pytest.skip("torch's count is 4 or fewer, so a fall to 4 cannot show")
    assert r["after_small"] == r["load_count"]
    assert r["after_large"] == r["load_count"]
