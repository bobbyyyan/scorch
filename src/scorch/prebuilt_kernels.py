from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union, List
import weakref

import torch
import scorch_ops as native_ops

from .format import TensorFormat
from .utils import parse_format

if TYPE_CHECKING:
    from .stensor import STensor


KernelFn = Callable[..., Any]


# --------------------------------------------------------------------------- #
# Narrowed-index cache
#
# The prebuilt kernels index with int32. An STensor built from the int64 arrays
# scipy and torch.sparse_csr_tensor hand a user therefore pays an O(nnz) cast on the
# native side of EVERY call (native_abi.h checked_index_tensor). Measured on redwood
# that cast plus the representability scan it guards is ~0.6-0.7 ns per nonzero — on
# reddit at N=16, 80 ms against a 30 ms kernel.
#
# A tensor's indices do not change between calls, so narrow once and keep the result
# for as long as the original lives. Keyed on the index tensor's identity and
# invalidated by its `_version`, so an in-place mutation is not served stale. The
# representability test below is exactly the one checked_index_tensor performs; when
# it fails we hand the int64 tensor to the native validator unchanged so the error
# still names the offending element.
#
# Cost: an int32 copy of each index array, i.e. +50% on top of the int64 arrays the
# caller already holds. It replaces a same-size allocation that was happening on
# every call, so peak memory does not grow, but steady-state does. Set
# SCORCH_NARROW_INDEX_CACHE=0 to keep the old per-call cast.
# --------------------------------------------------------------------------- #
_INT32_MIN = -(2 ** 31)
_INT32_MAX = 2 ** 31 - 1
_NARROW_CACHE_ON = os.environ.get("SCORCH_NARROW_INDEX_CACHE", "1") != "0"
# id(tensor) -> (version, narrowed). weakref.finalize evicts the entry when the key
# tensor dies, so a recycled id can never serve another tensor's narrowed copy.
_narrowed: dict = {}


def _evict_narrowed(key: int) -> None:
    _narrowed.pop(key, None)


def _narrow_index(index: torch.Tensor) -> torch.Tensor:
    """int64 index tensor -> memoized int32 copy (or the original if not int64)."""
    if index.dtype != torch.int64:
        return index
    if not _NARROW_CACHE_ON:
        return index
    key = id(index)
    hit = _narrowed.get(key)
    if hit is not None and hit[0] == index._version:
        return hit[1]
    if index.numel():
        # One pass, not two: this runs once per tensor but on the full int64 array,
        # and on a 115M-nonzero graph a second pass is another 0.9 GB of reads.
        lo, hi = torch.aminmax(index)
        if int(lo) < _INT32_MIN or int(hi) > _INT32_MAX:
            return index  # let the native validator report which element
    narrowed = index.to(torch.int32)
    if hit is None:
        try:
            weakref.finalize(index, _evict_narrowed, key)
        except TypeError:  # not weak-referenceable; skip memoization entirely
            return narrowed
    _narrowed[key] = (index._version, narrowed)
    return narrowed


def _narrowed_mode_indices(tensor: "STensor") -> List[List[torch.Tensor]]:
    return [
        [_narrow_index(index) for index in level]
        for level in tensor._native_mode_indices()
    ]


def clear_narrowed_index_cache() -> None:
    """Drop every memoized int32 index copy (tests, and memory-pressure escapes)."""
    _narrowed.clear()


@dataclass(frozen=True)
class PrebuiltMatmulSpec:
    lhs_rank: int
    rhs_rank: int
    lhs_format: str
    rhs_format: str
    output_format: str
    symbol_by_dtype: Mapping[torch.dtype, Sequence[str]]


@dataclass(frozen=True)
class ResolvedPrebuiltKernel:
    fn: KernelFn
    output_format: TensorFormat
    symbol_name: str


_MATMUL_PREBUILT_SPECS: List[PrebuiltMatmulSpec] = [
    PrebuiltMatmulSpec(
        lhs_rank=2,
        rhs_rank=2,
        lhs_format="d,s",
        rhs_format="d,d",
        output_format="dd",
        symbol_by_dtype={
            torch.float32: ("spmm_csr_float_v2", "prebuilt_spmm_csr_f32", "spmm_csr_float"),
            torch.float64: ("prebuilt_spmm_csr_f64", "spmm_csr_double"),
            torch.int32: ("prebuilt_spmm_csr_i32",),
            torch.int64: ("prebuilt_spmm_csr_i64",),
        },
    ),
    PrebuiltMatmulSpec(
        lhs_rank=2,
        rhs_rank=2,
        lhs_format="d,s",
        rhs_format="d,s",
        output_format="ds",
        symbol_by_dtype={
            torch.float32: ("prebuilt_spmspm_csr_f32", "spmspm_csr_float"),
            torch.float64: ("prebuilt_spmspm_csr_f64",),
            torch.int32: ("prebuilt_spmspm_csr_i32",),
            torch.int64: ("prebuilt_spmspm_csr_i64",),
        },
    ),
    PrebuiltMatmulSpec(
        lhs_rank=2,
        rhs_rank=2,
        lhs_format="o,o",
        rhs_format="o,o",
        output_format="oo",
        symbol_by_dtype={torch.float32: ("spmspm_coo_float",)},
    ),
    PrebuiltMatmulSpec(
        lhs_rank=2,
        rhs_rank=2,
        lhs_format="o,o",
        rhs_format="d,d",
        output_format="dd",
        symbol_by_dtype={torch.float32: ("spmm_coo_float",)},
    ),
    PrebuiltMatmulSpec(
        lhs_rank=2,
        rhs_rank=1,
        lhs_format="d,s",
        rhs_format="d",
        output_format="d",
        symbol_by_dtype={
            torch.float32: ("prebuilt_spmv_csr_f32",),
            torch.float64: ("prebuilt_spmv_csr_f64",),
            torch.int32: ("prebuilt_spmv_csr_i32",),
            torch.int64: ("prebuilt_spmv_csr_i64",),
        },
    ),
]


def _resolve_symbol(candidates: Sequence[str]) -> Tuple[Optional[KernelFn], Optional[str]]:
    for symbol_name in candidates:
        fn = getattr(native_ops, symbol_name, None)
        if fn is not None:
            return fn, symbol_name
    return None, None


# Resolution depends only on (ranks, format strings, requested output format,
# dtypes) — all static for a given kernel shape — so memoize it. On a warm call
# this collapses the whole spec scan + repeated parse_format() calls into a
# single dict lookup, eliminating a large slice of the per-call Python overhead
# that dominates SpMM latency on small matrices. The key space is tiny and
# bounded (a handful of formats/ranks/dtypes), so no eviction is needed.
_RESOLVE_MISS = object()
_resolve_matmul_cache: dict = {}


def resolve_prebuilt_matmul(
    a: "STensor",
    b: "STensor",
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
) -> Optional[ResolvedPrebuiltKernel]:
    of_key = tuple(output_format) if isinstance(output_format, list) else output_format
    if of_key is not None and not isinstance(of_key, (str, tuple)):
        of_key = str(of_key)
    cache_key = (
        a.dim(), b.dim(), str(a.format), str(b.format),
        of_key, a.values.dtype, b.values.dtype,
    )
    cached = _resolve_matmul_cache.get(cache_key, _RESOLVE_MISS)
    if cached is not _RESOLVE_MISS:
        return cached
    resolved = _resolve_prebuilt_matmul_uncached(a, b, output_format)
    _resolve_matmul_cache[cache_key] = resolved
    return resolved


def _resolve_prebuilt_matmul_uncached(
    a: "STensor",
    b: "STensor",
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
) -> Optional[ResolvedPrebuiltKernel]:
    if a.values.dtype != b.values.dtype:
        return None

    requested_format = str(parse_format(output_format)) if output_format is not None else None
    a_format = str(a.format)
    b_format = str(b.format)
    a_rank = a.dim()
    b_rank = b.dim()

    for spec in _MATMUL_PREBUILT_SPECS:
        if a_rank != spec.lhs_rank or b_rank != spec.rhs_rank:
            continue
        if a_format != spec.lhs_format or b_format != spec.rhs_format:
            continue
        if requested_format is not None and requested_format != str(parse_format(spec.output_format)):
            continue
        symbols = spec.symbol_by_dtype.get(a.values.dtype)
        if symbols is None:
            continue
        fn, symbol_name = _resolve_symbol(symbols)
        if fn is None or symbol_name is None:
            continue
        return ResolvedPrebuiltKernel(
            fn=fn,
            output_format=parse_format(spec.output_format),
            symbol_name=symbol_name,
        )

    return None


# ---------------------------------------------------------------------------
# Fused prebuilt kernel specs (SpMM + postops)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrebuiltFusedSpec:
    lhs_format: str
    rhs_format: str
    post_op_kinds: Tuple[str, ...]
    symbol_by_dtype: Mapping[torch.dtype, Sequence[str]]


@dataclass(frozen=True)
class ResolvedPrebuiltFusedKernel:
    fn: KernelFn
    symbol_name: str


_FUSED_PREBUILT_SPECS: List[PrebuiltFusedSpec] = [
    # CSR x dense + bias + relu
    PrebuiltFusedSpec(
        lhs_format="d,s",
        rhs_format="d,d",
        post_op_kinds=("add", "relu"),
        symbol_by_dtype={torch.float32: ("spmm_csr_bias_relu_float",)},
    ),
    # CSR x dense + bias only
    PrebuiltFusedSpec(
        lhs_format="d,s",
        rhs_format="d,d",
        post_op_kinds=("add",),
        symbol_by_dtype={torch.float32: ("spmm_csr_bias_float",)},
    ),
]


def resolve_prebuilt_fused(
    a_format: str,
    b_format: str,
    post_op_kinds: Tuple[str, ...],
    dtype: torch.dtype,
) -> Optional[ResolvedPrebuiltFusedKernel]:
    """Match against prebuilt fused kernels. Returns kernel or None."""
    for spec in _FUSED_PREBUILT_SPECS:
        if a_format != spec.lhs_format or b_format != spec.rhs_format:
            continue
        if post_op_kinds != spec.post_op_kinds:
            continue
        symbols = spec.symbol_by_dtype.get(dtype)
        if symbols is None:
            continue
        fn, symbol_name = _resolve_symbol(symbols)
        if fn is None or symbol_name is None:
            continue
        return ResolvedPrebuiltFusedKernel(fn=fn, symbol_name=symbol_name)
    return None


def execute_prebuilt_binary_kernel(
    kernel_fn: KernelFn,
    a: "STensor",
    b: "STensor",
    time_dict: Optional[dict] = None,
    nthreads: Optional[int] = None,
    atparallel: bool = False,
) -> Tuple[Any, Tuple[int, ...]]:
    if b.dim() == 2:
        result_shape: Tuple[int, ...] = (a.shape[0], b.shape[1])
    elif b.dim() == 1:
        result_shape = (a.shape[0],)
    else:
        raise ValueError(f"Unsupported RHS rank for prebuilt matmul kernel: {b.dim()}")

    args = [result_shape]
    for tensor in [a, b]:
        args.append(tensor.shape)  # type: ignore[arg-type]
        args.append(_narrowed_mode_indices(tensor))  # type: ignore[arg-type]
        args.append(tensor.values)  # type: ignore[arg-type]

    start_time = time.time()
    # nthreads/atparallel are only supplied for the drop-in SpMM (spmm_csr_float_v2,
    # the only kernel accepting nthreads_override/atparallel); the caller gates on
    # symbol_name. atparallel launches the SpMM on torch's intra-op pool so it
    # shares one warm team with the pipeline's torch epilogue.
    if nthreads is not None:
        result_cpp = kernel_fn(*args, nthreads_override=nthreads, atparallel=atparallel)
    else:
        result_cpp = kernel_fn(*args)
    end_time = time.time()

    if time_dict is not None:
        time_dict["eval_time"] = end_time - start_time

    return result_cpp, result_shape
