from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union, List

import torch
import scorch_ops as native_ops

from .format import TensorFormat
from .utils import parse_format

if TYPE_CHECKING:
    from .stensor import STensor


KernelFn = Callable[..., Any]


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
            torch.float32: ("prebuilt_spmm_csr_f32", "spmm_csr_float"),
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


def resolve_prebuilt_matmul(
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


def execute_prebuilt_binary_kernel(
    kernel_fn: KernelFn,
    a: "STensor",
    b: "STensor",
    time_dict: Optional[dict] = None,
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
        args.append(tensor.index.mode_indices)  # type: ignore[arg-type]
        args.append(tensor.values)  # type: ignore[arg-type]

    start_time = time.time()
    result_cpp = kernel_fn(*args)
    end_time = time.time()

    if time_dict is not None:
        time_dict["eval_time"] = end_time - start_time

    return result_cpp, result_shape
