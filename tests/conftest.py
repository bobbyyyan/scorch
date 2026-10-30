import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def _set_torch_extensions_dir(tmp_path_factory):
    old_value = os.environ.get("TORCH_EXTENSIONS_DIR")
    ext_dir = tmp_path_factory.mktemp("torch_extensions")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = old_value


@pytest.fixture(scope="session", autouse=True)
def _private_autotune_files(tmp_path_factory):
    """Point the tiling selector's two files into the session's temporary directory.

    The "max" autotune level keeps its verdicts in $XDG_CACHE_HOME/scorch/autotune.json
    unless SCORCH_AUTOTUNE_CACHE names another file, and the "learned" level reads its
    model from the same directory or from the tree's autotune-levels/models/ unless
    SCORCH_AUTOTUNE_MODEL names one. Without this the suite read and wrote the
    developer's own files: test_every_autotune_level_composes[max] stored its verdicts in
    the real autotune.json, and the runs after it were served them instead of probing.
    The model variable names a file that does not exist, which is what a machine with no
    model has. Tests of either file point the variable at a file of their own.
    """
    directory = tmp_path_factory.mktemp("autotune")
    paths = {
        "SCORCH_AUTOTUNE_CACHE": str(directory / "autotune.json"),
        "SCORCH_AUTOTUNE_MODEL": str(directory / "no_model.json"),
    }
    old_values = {name: os.environ.get(name) for name in paths}
    os.environ.update(paths)
    if "scorch.tiling" in sys.modules:
        # Read either file again, from the new paths, if anything read it already. The
        # module is loaded, so the import only binds it, with its types for a checker.
        from scorch import tiling

        tiling._cache_loaded = False
        tiling._persist_cache = None
        tiling._reset_learned_model_cache()
    try:
        yield
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="session", autouse=True)
def _validate_kernel_results():
    """Walk the index arrays of every generated result, for the whole suite.

    Release skips that walk: a sparse result's index arrays come out of our own codegen,
    which allocates each output level with `torch::empty` sized from a counted extent, so
    validating them re-derives what the compiler already established -- 35-41% of a wrap.
    The same trade PyTorch makes with `torch.sparse.check_sparse_tensor_invariants`, which
    also defaults to off.

    That trade is only honest if the fact is still checked somewhere, and this is where.
    Turning it on here means a bug in lowering, in the scheduler, or in a new codegen path
    shows up as a structured `TensorIndexError` naming the mode, on whichever test first
    produces a malformed result -- rather than as unchecked pointer arithmetic in a kernel
    on a user's machine. It costs the suite a walk per generated result and is worth it.

    Set at the cell, because `scorch.storage` reads the variable once, at import, and
    pytest may already have imported it -- AND in the environment, because the cell is
    per-process. A test that runs scorch in a subprocess (a disposable worker, an
    import-isolation probe) imports it fresh, reads `SCORCH_VALIDATE_KERNEL_RESULTS`, and
    without the variable runs with the walk off. That hole let the characterization of the
    defective legacy routes in test_loopir_ordered_key_workspace_target.py, whose worker
    is a subprocess, read a malformed result as "ok".
    """
    from scorch import storage as storage_module

    old_value = storage_module._VALIDATE_KERNEL_RESULTS[0]
    old_env = os.environ.get("SCORCH_VALIDATE_KERNEL_RESULTS")
    storage_module._VALIDATE_KERNEL_RESULTS[0] = True
    os.environ["SCORCH_VALIDATE_KERNEL_RESULTS"] = "1"
    try:
        yield
    finally:
        storage_module._VALIDATE_KERNEL_RESULTS[0] = old_value
        if old_env is None:
            os.environ.pop("SCORCH_VALIDATE_KERNEL_RESULTS", None)
        else:
            os.environ["SCORCH_VALIDATE_KERNEL_RESULTS"] = old_env


def processors():
    """The processors this process may run on: the affinity mask where the platform has one.

    That is the set libgomp's omp_get_num_procs() counts on Linux (and so scorch_num_procs(), the
    ceiling of every team scorch launches with no thread setting in force). macOS has no affinity
    mask, and LLVM's libomp reports every processor, which is os.cpu_count().
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@pytest.fixture
def torch_threads():
    """Run a test at a torch thread count it names, and restore the count afterwards.

    Once a thread setting is in force (OMP_NUM_THREADS or MKL_NUM_THREADS at load, or torch's count
    changed since), scorch caps every team it launches at torch's count (scorch_thread_ceiling in
    scorch_policy.h). So a test that passes a worker override above 1, asserts a team above 1, or
    runs a kernel at a shape chosen for several workers gets that team only if torch's count allows
    it, and a process started with MKL_NUM_THREADS=1 runs it on one worker. The test calls
    ``torch_threads(n)`` with the largest count it simulates; the count is capped at the processors.

    ``needs_team`` (the default) skips the test where only one processor is available, because what
    it asserts needs a second worker. A test whose assertions hold on one worker as well passes
    ``needs_team=False`` and runs anyway.

    torch.set_num_threads, not scorch.set_num_threads: scorch's records a setting for the rest of
    the process. Restoring torch's count restores the state exactly.
    """
    import scorch_ops
    import torch

    before = torch.get_num_threads()

    def use(n, needs_team=True):
        procs = processors()
        if needs_team and procs < 2:
            pytest.skip("one processor: every team is one worker")
        count = max(1, min(n, procs))
        torch.set_num_threads(count)
        # The ceiling is now `count`, or scorch_num_procs() when `count` is torch's count at load
        # and nothing was set; never less than `count` unless scorch counts fewer processors than
        # processors() does, which would make the test's premise false. Say so here, not in the
        # test's own assertion.
        ceiling = scorch_ops.thread_ceiling()
        assert ceiling >= count, (
            "asked for %d threads (processors %d) but scorch's ceiling is %d" % (count, procs, ceiling)
        )
        return count

    yield use
    torch.set_num_threads(before)


@pytest.fixture
def fresh_kernel_caches(monkeypatch):
    """Run a test with empty kernel caches, so its calls choose, lower and build again.

    The caches are tests/test_scorch/kernel_cache_support.py's ``KERNEL_CACHES``: the
    einsum dispatch cache, the kernel cache and the generated-kernel store, which let a
    call skip the compiler, and the caches beside them that let it skip a check, a
    schedule's validation, the choice of a prebuilt kernel or of a tiling, or resolving
    its build options. They live for the process, and their keys cannot see a replaced
    function. So a test that replaces a scheduler, lowering, LLIR-pass, codegen, tiling or
    prebuilt-resolution function and then calls a public entry is served what an earlier
    test with the same key kept, and its replacement never runs. Such a test must request
    this fixture, and a test that needs fresh caches for any other reason asks the same
    way, so that every one of them empties the same list.

    A cache held in one dict is replaced by an empty dict for the test and restored after
    it; a cache its module empties with a function is emptied before the test and again
    after it (``_finish_kernel_caches``). Either way the test's own entries are not kept
    for later tests. A test that must empty them again partway through calls
    ``kernel_cache_support.empty_kernel_caches``.
    """
    from tests.test_scorch.kernel_cache_support import empty_kernel_caches

    empty_kernel_caches(monkeypatch)


@pytest.fixture(autouse=True)
def _finish_kernel_caches():
    """After a test that emptied the kernel caches, empty again the ones a function empties.

    A dict ``empty_kernel_caches`` replaced is restored by ``monkeypatch``; a cache its
    module empties with a function cannot be restored, so it is emptied again instead,
    and what the test put in it is not kept for later tests. Nothing runs after a test
    that did not empty the caches.
    """
    yield
    from tests.test_scorch.kernel_cache_support import finish_test

    finish_test()
