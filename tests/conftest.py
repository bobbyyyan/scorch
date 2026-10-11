import os

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
