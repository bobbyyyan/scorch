import contextlib
import torch

import scorch
from scorch import STensor
from scorch.compiler.scheduler import Scheduler


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


def _run_forced_loop_matmul(fmt_a: str, fmt_b: str, forced_loop_order):
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

    with _force_loop_order(forced_loop_order):
        result = scorch.matmul(a, b, format="dd", use_cache=False)
    result_torch = result if isinstance(result, torch.Tensor) else result.to_torch()

    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(result_torch, expected, atol=1e-3, rtol=1e-3)


def test_spmm_forced_loop_order_canonical_succeeds():
    _run_forced_loop_matmul(fmt_a="ds", fmt_b="dd", forced_loop_order=("i", "j", "k"))


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
