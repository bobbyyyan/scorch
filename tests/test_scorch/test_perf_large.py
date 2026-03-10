import statistics
import time
from typing import Callable, Dict, List, Tuple

import pytest
import torch

import scorch
import scorch_ops as ops
from scorch import STensor




def _make_spmm_inputs(
    n: int = 1024, sparsity: float = 0.95, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    a_dense = torch.rand((n, n), generator=generator, dtype=torch.float32)
    b_dense = torch.rand((n, n), generator=generator, dtype=torch.float32)
    mask = torch.rand((n, n), generator=generator) > sparsity
    a_dense = a_dense * mask
    return a_dense, b_dense


def _to_kernel_args(a_dense: torch.Tensor, b_dense: torch.Tensor) -> Tuple[STensor, STensor, List]:
    a_scorch = STensor.from_torch(a_dense, "A").to_sparse("ds")
    b_scorch = STensor.from_torch(b_dense, "B")
    args = [[a_dense.shape[0], b_dense.shape[1]]]
    for tensor in (a_scorch, b_scorch):
        args.append(list(tensor.shape))
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)
    return a_scorch, b_scorch, args


def _median_time(
    fn: Callable[[], torch.Tensor], warmup: int = 1, repeats: int = 3
) -> Tuple[torch.Tensor, float]:
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    assert result is not None
    return result, statistics.median(times)


@pytest.mark.perf
def test_spmm_variants_and_compiler_large_tensor():
    a_dense, b_dense = _make_spmm_inputs(n=1024, sparsity=0.95, seed=7)
    reference = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
    a_scorch, b_scorch, kernel_args = _to_kernel_args(a_dense, b_dense)

    variant_fns: Dict[str, Callable] = {
        "spmm_csr_float_untiled": ops.spmm_csr_float_untiled,
        "spmm_csr_float": ops.spmm_csr_float,
        "spmm_csr_float_optimized": ops.spmm_csr_float_optimized,
        "spmm_csr_float_turbo": ops.spmm_csr_float_turbo,
        "spmm_csr_float_ultra": ops.spmm_csr_float_ultra,
        "spmm_csr_float_apex": ops.spmm_csr_float_apex,
        "spmm_csr_float_tiled_i_k": ops.spmm_csr_float_tiled_i_k,
    }

    variant_medians: Dict[str, float] = {}

    for name, fn in variant_fns.items():
        result, median_time = _median_time(lambda f=fn: f(*kernel_args).storage.value)
        assert torch.allclose(result.view_as(reference), reference, atol=1e-3, rtol=1e-3), name
        variant_medians[name] = median_time

    compiler_eval_times: List[float] = []
    compiler_result = None
    for _ in range(2):
        _ = scorch.matmul(
            a_scorch,
            b_scorch,
            format="dd",
            use_cache=False,
            output_mode_order=[0, 1],
        )
    for _ in range(3):
        time_dict = {}
        compiler_result = scorch.matmul(
            a_scorch,
            b_scorch,
            format="dd",
            use_cache=False,
            output_mode_order=[0, 1],
            time_dict=time_dict,
        )
        compiler_eval_times.append(time_dict["eval_time"])

    assert isinstance(compiler_result, torch.Tensor)
    assert torch.allclose(compiler_result, reference, atol=1e-3, rtol=1e-3)

    compiler_median = statistics.median(compiler_eval_times)
    untiled_median = variant_medians["spmm_csr_float_untiled"]
    best_handwritten = min(variant_medians.values())

    assert best_handwritten < untiled_median
    assert compiler_median <= untiled_median * 2.0


@pytest.mark.perf
@pytest.mark.parametrize(
    "n,sparsity",
    [
        (768, 0.90),
        (1024, 0.95),
    ],
)
def test_compiler_spmm_large_correct_and_nontrivial_runtime(n: int, sparsity: float):
    a_dense, b_dense = _make_spmm_inputs(n=n, sparsity=sparsity, seed=11)
    reference = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
    a_scorch = STensor.from_torch(a_dense, "A").to_sparse("ds")
    b_scorch = STensor.from_torch(b_dense, "B")

    eval_times = []
    result = None
    for _ in range(2):
        _ = scorch.matmul(
            a_scorch, b_scorch, format="dd", use_cache=False, output_mode_order=[0, 1]
        )
    for _ in range(3):
        time_dict = {}
        result = scorch.matmul(
            a_scorch,
            b_scorch,
            format="dd",
            use_cache=False,
            output_mode_order=[0, 1],
            time_dict=time_dict,
        )
        eval_times.append(time_dict["eval_time"])

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3)
    assert all(t > 0.0 for t in eval_times)
