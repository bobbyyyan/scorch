import contextlib

import pytest
import torch

import scorch
import scorch.ops as ops
from scorch import STensor
from scorch.compiler.scheduler import Scheduler

# The forced orders name their loop order with ``schedule=``, which the einsum
# dispatch cache and the kernel caches key, so a test that runs after another in the
# same process still builds its own order's kernel.  These tests used to replace
# ``Scheduler.select_loop_order``, which those caches cannot see: in one process the
# first test of each group decided the kernel the whole group ran, and a forced
# order served that kernel passed without its own kernel being compiled.
# ``use_cache=False`` is not passed with a schedule: a schedule already skips the
# prebuilt kernels, and the keyword never made a call compile again.
#
# The SpMM canonical order still replaces ``select_loop_order``, with
# ``use_cache=False`` to skip the prebuilt kernels, and runs with empty kernel caches
# (``fresh_kernel_caches``) so that the replacement runs.  matmul builds its default
# SpMM kernel, which branches at run time between a register-blocked k tile for a
# narrow dense operand and the plain loop, only when no schedule is given
# (``ops._build_regblock_dual_path``).  A schedule naming (i, j, k) builds the plain
# loop alone, so this test would no longer build that kernel.


@contextlib.contextmanager
def _force_loop_order(loop_order_names):
    original_select_loop_order = Scheduler.select_loop_order

    def _forced_select_loop_order(
        cin,
        costs=Scheduler._DEFAULT_COSTS,
    ):
        all_index_vars = Scheduler.get_index_variables(cin)
        index_var_by_name = {index_var.name: index_var for index_var in all_index_vars}
        selected = []
        for name in loop_order_names:
            index_var = index_var_by_name.get(name)
            if index_var is not None and index_var not in selected:
                selected.append(index_var)
        for index_var in all_index_vars:
            if index_var not in selected:
                selected.append(index_var)
        return selected

    Scheduler.select_loop_order = staticmethod(_forced_select_loop_order)
    try:
        yield
    finally:
        Scheduler.select_loop_order = original_select_loop_order


def _make_sparse_matrix(n: int, keep_prob: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    dense = torch.rand((n, n), generator=generator)
    mask = torch.rand((n, n), generator=generator) < keep_prob
    return dense * mask


def _run_forced_loop_matmul(
    fmt_a: str, fmt_b: str, forced_loop_order, replace_select_loop_order: bool = False
):
    n = 32
    a_torch = _make_sparse_matrix(n, keep_prob=0.08, seed=0)
    if fmt_b == "dd":
        b_torch = torch.rand((n, n), generator=torch.Generator().manual_seed(1))
    else:
        b_torch = _make_sparse_matrix(n, keep_prob=0.08, seed=1)

    a = STensor.from_torch(a_torch, "A", mode_order=[0, 1])
    b = STensor.from_torch(b_torch, "B", mode_order=[0, 1])

    if fmt_a != "dd":
        a = a.to_sparse(fmt_a)
    if fmt_b != "dd":
        b = b.to_sparse(fmt_b)

    if replace_select_loop_order:
        # The replacement runs only on a call that compiles: the caller's test must
        # request fresh_kernel_caches.
        with _force_loop_order(forced_loop_order):
            result = scorch.matmul(a, b, format="dd", use_cache=False)
    else:
        result = scorch.matmul(
            a, b, format="dd", schedule=scorch.Schedule(loop_order=forced_loop_order)
        )
    result_torch = result if isinstance(result, torch.Tensor) else result.to_torch()

    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(result_torch, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.usefixtures("fresh_kernel_caches")
def test_spmm_forced_loop_order_canonical_succeeds():
    _run_forced_loop_matmul(
        fmt_a="ds",
        fmt_b="dd",
        forced_loop_order=("i", "j", "k"),
        replace_select_loop_order=True,
    )


def test_spgemm_forced_loop_order_canonical_succeeds():
    _run_forced_loop_matmul(fmt_a="ds", fmt_b="ds", forced_loop_order=("i", "j", "k"))


def test_known_gap_spmm_forced_loop_order_ikj():
    _run_forced_loop_matmul(fmt_a="ds", fmt_b="dd", forced_loop_order=("i", "k", "j"))


def test_known_gap_spmm_forced_loop_order_jik():
    _run_forced_loop_matmul(fmt_a="ds", fmt_b="dd", forced_loop_order=("j", "i", "k"))


def test_known_gap_spgemm_forced_loop_order_ikj():
    _run_forced_loop_matmul(fmt_a="ds", fmt_b="ds", forced_loop_order=("i", "k", "j"))


def test_known_gap_spgemm_forced_loop_order_jik():
    _run_forced_loop_matmul(fmt_a="ds", fmt_b="ds", forced_loop_order=("j", "i", "k"))


# (format of A, format of B, loop order, whether the call replaces select_loop_order),
# each as its test above calls it.
_PRODUCTS = [
    ("ds", "dd", ("i", "j", "k"), True),
    ("ds", "dd", ("i", "k", "j"), False),
    ("ds", "dd", ("j", "i", "k"), False),
    ("ds", "ds", ("i", "j", "k"), False),
    ("ds", "ds", ("i", "k", "j"), False),
    ("ds", "ds", ("j", "i", "k"), False),
]


@pytest.mark.usefixtures("fresh_kernel_caches")
def test_each_loop_order_runs_its_own_kernel_in_one_process():
    """The six products above, one after another in one process, run six kernels.

    The canonical order of each group runs first, as it does in the suite, and must
    not serve the group's forced orders: each call leaves its own entry in the einsum
    dispatch cache, holding its own module.  The SpMM canonical order replaces
    select_loop_order, which runs because the caches start empty.
    """

    for fmt_a, fmt_b, order, replace_select_loop_order in _PRODUCTS:
        _run_forced_loop_matmul(fmt_a, fmt_b, order, replace_select_loop_order)
    modules = [entry[0] for entry in ops._einsum_dispatch_cache.values()]
    assert len(modules) == len(_PRODUCTS)
    assert len({id(module) for module in modules}) == len(_PRODUCTS)
