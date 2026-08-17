import copy
import os
import time
from typing import Any, Union, Sequence, Optional, List, Tuple

import torch
from torch.fx import Proxy

from .compiler import llir
from .compiler.cin import (
    IndexVar,
    TensorVar,
    ForAll,
    Workspace,
    Where,
    TensorAssign,
    Operation,
    IndexStmt,
)
from .compiler.cin_lowerer import CINLowerer
from .compiler.codegen import LLIRLowerer
from .compiler.scheduler import (
    Schedule,
    Scheduler,
    _regblock_enabled,
    _regblock_max_n,
    get_forced_schedule,
    regblock_force,
)
from .exceptions import CompileSpecError, TensorTypeError, TensorValidationError
from .format import TensorFormat, LevelFormat, LevelType
from .layout import TensorSpec
from .plan import ENABLED_CELL as _PLAN_ENABLED
from .plan import GENERATION as _PLAN_GENERATION
from .plan import PLANS_ATTR as _PLANS_ATTR
from .plan import install as _plan_install
from .prebuilt_kernels import execute_prebuilt_binary_kernel, resolve_prebuilt_matmul
from .tiling import maybe_dispatch as _tiling_maybe_dispatch
from .tiling import is_candidate as _tiling_is_candidate
from .tiling import decided as _tiling_decided
from .tiling import _current_level as _tiling_current_level
from .storage import TensorIndex
from .stensor import STensor, _finalize_generated_mode_indices
from .utils import (
    parse_format,
    topo_sort_characters,
    get_extra_cflags,
    get_extra_ldflags,
    jit_preamble_text,
    _kernel_name,
    _load_kernel,
)

_kernel_cache = {}
_einsum_dispatch_cache = {}


def _effective_schedule(kwargs: dict) -> Optional[Schedule]:
    """Resolve an explicit or context-forced compiler schedule."""
    public = kwargs.get("schedule")
    internal = kwargs.get("_schedule")
    if public is not None and internal is not None and public != internal:
        raise ValueError("schedule and _schedule specify different schedules")
    schedule = public if public is not None else internal
    if schedule is None:
        schedule = get_forced_schedule()
    if schedule is not None and not isinstance(schedule, Schedule):
        raise TypeError("schedule must be a scorch.compiler.scheduler.Schedule")
    return schedule


def _schedule_cache_key(schedule: Optional[Schedule]) -> Optional[str]:
    return schedule.cache_key if schedule is not None else None


def _einsum_cache_key(
    expression: str,
    tensors: Sequence[Any],
    output_format: Any,
    output_mode_order: Any,
    schedule: Optional[Schedule],
) -> tuple:
    """Build the early dispatch key, including every scheduling decision."""

    def layout_contract(tensor: Any) -> Any:
        layout = getattr(tensor, "layout", None)
        serialize = getattr(layout, "serialize", None)
        if callable(serialize):
            return serialize()
        # Scheduling tools may use lightweight tensor-like objects. Preserve a
        # deterministic fallback without weakening real tensors' canonical key.
        shape = getattr(tensor, "shape", ()) or ()
        mode_order = getattr(tensor, "mode_order", ()) or ()
        return (
            "tensor_like",
            str(getattr(tensor, "format", None)),
            tuple(shape),
            tuple(mode_order),
            str(getattr(tensor, "index_dtype", None)),
        )

    return (
        expression,
        tuple(layout_contract(t) for t in tensors),
        tuple(
            (str(getattr(t, "dtype", None)), str(getattr(t, "device", "cpu")))
            for t in tensors
        ),
        str(output_format) if output_format is not None else None,
        tuple(output_mode_order) if output_mode_order else None,
        _schedule_cache_key(schedule),
    )


def _codegen_kernel_cache_key(
    cin: IndexStmt,
    post_ops: Any,
    schedule: Optional[Schedule],
) -> str:
    """Cache a lowered CIN without allowing dtype/schedule fields to alias."""
    tensor_contracts = tuple(
        sorted(
            {
                (
                    access.tensor.name,
                    str(access.tensor.dtype),
                    access.tensor.shape,
                    str(access.tensor.format),
                    tuple(access.tensor.mode_order or ()),
                )
                for access in cin.tensor_accesses
            }
        )
    )
    key = str(cin) + f"|contracts:{tensor_contracts!r}"
    if post_ops:
        key += f"|post_ops:{post_ops}"
    if schedule is not None:
        key += f"|schedule:{schedule.cache_key}"
    return key


def _logical_index_sizes(
    input_index_strs: Sequence[Sequence[str]],
    tensors: Sequence[Any],
) -> dict:
    """Map logical einsum indices to sizes after physical mode reordering."""
    index_to_size = {}
    for operand, (index_strs, tensor) in enumerate(zip(input_index_strs, tensors)):
        mode_order = list(tensor.mode_order)
        if len(index_strs) != len(mode_order):
            raise CompileSpecError(
                f"einsum operand {operand} has {len(index_strs)} indices but "
                f"tensor rank {len(mode_order)}"
            )
        for logical_axis, index_str in enumerate(index_strs):
            physical_axis = mode_order.index(logical_axis)
            size = tensor.shape[physical_axis]
            previous = index_to_size.get(index_str)
            if previous is not None and previous != size:
                raise CompileSpecError(
                    f"einsum index {index_str!r} has incompatible dimensions "
                    f"{previous} and {size}"
                )
            index_to_size[index_str] = size
    return index_to_size


def _with_compiler_mode_order(
    tensor: Union[STensor, TensorSpec], mode_order: Sequence[int]
) -> Union[STensor, TensorSpec]:
    """Relayout a runtime tensor or functionally update a compile-only spec."""
    if isinstance(tensor, TensorSpec):
        return tensor.with_mode_order(mode_order)
    result = tensor.copy()
    result.change_mode_order(list(mode_order))
    return result


# Composition thread-count matching for the drop-in CSR-SpMM (spmm_csr_float_v2).
# When a scorch SpMM is used as a torch op inside a host pipeline (e.g. a GCN
# layer: F.linear -> scorch.matmul -> ...), the SpMM's throttled policy thread
# count differs from the host's (torch) thread count, forcing a libgomp team
# reshape at every op boundary (~15% of a sub-ms GCN forward). Passing
# torch.get_num_threads() lets the kernel keep one warm team spanning the
# pipeline; the kernel only adopts it above the fork/join floor and caps it by
# row-parallelism, so tiny/standalone products (the SuiteSparse panel) are
# unaffected. On by default (non-fused GCN forward: pubmed 0.78x -> 1.15x vs
# PyTorch, cora/citeseer +5-6%, big graphs neutral, panel safe). Set
# SCORCH_MATCH_HOST_THREADS=0 to fall back to the pure standalone policy.
_MATCH_HOST_THREADS = os.environ.get("SCORCH_MATCH_HOST_THREADS", "1") == "1"

# Composition, part two: for the same drop-in pipeline SpMM, launch the kernel's
# workers on torch's own intra-op pool (at::parallel_for) instead of a private
# libgomp team, so the SpMM shares one warm pool with the surrounding torch
# epilogue (bias/activation) — a private omp team's threads sleep between ops
# under the default passive OMP policy and re-wake per call, a pool-transition tax
# that dominated the sparse-AE @0.99 mid-size gap (cur/mkl 1.2-1.3 -> ~0.9, 6/7
# apples-to-apples wins vs MKL). Fires together with _MATCH_HOST_THREADS; the
# kernel gates it to products that can feed the full host pool, so row-starved
# tiny products (the SuiteSparse panel's 130-row cell) keep omp. On by default;
# set SCORCH_ATPARALLEL_PIPELINE=0 to force the private-team launch.
_ATPARALLEL_PIPELINE = os.environ.get("SCORCH_ATPARALLEL_PIPELINE", "1") == "1"

# The same two composition hints, resolved once, for the planned dispatch: a plan
# call passes them positionally, so binding them here keeps that path branchless.
# `-1` is the kernel ABI's "no override", which is what the ordinary path passes
# when host-thread matching is off.
_PLAN_NTHREADS = torch.get_num_threads if _MATCH_HOST_THREADS else (lambda: -1)
_PLAN_ATPARALLEL = _ATPARALLEL_PIPELINE and _MATCH_HOST_THREADS

# `torch.Tensor`, bound once. The plan probe tests it on every matmul, and reading
# it through the module is a global lookup plus an attribute lookup each time.
_TENSOR = torch.Tensor

# start_time = time.time()
# # Register custom classes
# load(
#     name="pybind",
#     sources=[...],
# )
# end_time = time.time()
# compile_time = end_time - start_time
# print(f"Pybind load time: {compile_time:.5f} seconds")
#
# load_to_kernel_cache("spmm_csr", _kernel_cache, "spmm-csr.cpp")
# load_to_kernel_cache("spmm_csr_ones", _kernel_cache, "spmm-csr-ones.cpp")


def spmv(
    a: STensor,
    b: STensor,
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
    **kwargs,
) -> STensor:
    """Sparse matrix-vector product (internal; reached through ``matmul``).

    Computes ``y[i] = sum_j a[i, j] * b[j]`` by lowering a per-row ``Where`` with a
    scalar accumulating ``Workspace`` through the CIN compiler (CIN -> LLIR -> C++),
    JIT-compiling, and executing. This is the generic-path SpMV used when the
    prebuilt ``prebuilt_spmv_csr_*`` kernel does not match.

    This function is **not exported**: ``scorch.spmv`` is not defined (it would fall
    through ``__getattr__`` to ``torch``, which has no ``spmv``). The user-facing
    way to reach it is ``matmul(a, x)`` with a 2-D sparse ``a`` and a 1-D ``x`` —
    ``matmul`` tries the prebuilt CSR-SpMV kernel first and falls back to this
    function on a miss. Reachable directly as ``scorch.ops.spmv`` for advanced use.

    Parameters
    ----------
    a : STensor
        2-D sparse matrix operand of shape ``(m, n)``.
    b : STensor
        1-D vector operand of shape ``(n,)``.
    output_format : TensorFormat or str or list of str, optional
        Requested output format. Defaults to ``"d"`` (a dense 1-D vector). Accepts
        anything :func:`scorch.utils.parse_format` understands.
    **kwargs
        ``time_dict`` : dict, optional -- if given, ``time_dict["eval_time"]`` is
        set to the kernel wall-clock time in seconds.

    Returns
    -------
    STensor
        Result of shape ``(a.shape[0],)`` in ``output_format``.

    Notes
    -----
    ``spmv`` itself keeps no module cache; each call recompiles unless the
    persistent ``.so`` cache in ``_load_kernel`` (keyed under
    ``TORCH_EXTENSIONS_DIR``) hits.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> A = scorch.from_torch((torch.rand(128, 128) < 0.1).float(), "A").to_sparse("ds")
    >>> x = torch.rand(128)
    >>> y = scorch.matmul(A, x)   # 2-D x 1-D -> dense vector via prebuilt/spmv
    >>> torch.allclose(y, A.to_torch() @ x, atol=1e-3, rtol=1e-3)
    True
    """
    if output_format is None:
        output_format = parse_format("d")
    elif not isinstance(output_format, TensorFormat):
        output_format = parse_format(output_format)

    result_shape = (a.shape[0],)
    y = TensorVar("y", shape=result_shape, fmt=output_format, dtype=a.dtype)
    A = TensorVar("A", shape=a.shape, fmt=a.format, dtype=a.dtype)
    x = TensorVar("x", shape=b.shape, fmt=b.format, dtype=b.dtype)

    i = IndexVar("i")
    j = IndexVar("j")

    workspace = Workspace(
        name="wksp",
        dim=0,
    )

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                j,
                TensorAssign(
                    workspace.get_default_access(), A[i, j] * x[j], op=Operation.ADD
                ),
            ),
            consumer=TensorAssign(
                y[i],
                workspace.get_default_access(),
            ),
        ),
    )

    lowerer = CINLowerer()
    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
    llir_lowerer = LLIRLowerer()
    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    header_cpp_code = jit_preamble_text()

    # start_time = time.time()
    module = _load_kernel(
        name=_kernel_name(header_cpp_code, cpp_code),
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=get_extra_cflags(),
        extra_ldflags=get_extra_ldflags(),
    )
    # end_time = time.time()

    # compile_time = end_time - start_time
    #  Print kernel compile time to 5 decimal places
    # print(f"Kernel compile time: {compile_time:.5f} seconds")

    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._native_mode_indices())  # type: ignore
        args.append(tensor.values)  # type: ignore

    start_time = time.time()
    result_cpp = module.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time
    # m

    result = STensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=_finalize_generated_mode_indices(
                output_format, result_cpp.storage.index.mode_indices
            ),
            tensor_format=output_format,
        ),
        value=result_cpp.storage.value,
    )

    return result


def matmul_wksp(
    a: Union[torch.Tensor, STensor],
    b: Union[torch.Tensor, STensor],
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
    **kwargs,
) -> STensor:
    """Workspace-based SpMM / SpGEMM through the CIN compiler.

    Explicit workspace-lowering variant of :func:`matmul` that **always** goes
    through the generic CIN compiler pipeline (never the prebuilt hand-written
    kernels or the adaptive tiling selector). It builds a ``Where`` with an
    accumulating ``Workspace`` — dense when the output format is dense (avoids the
    COO hash-map overhead), otherwise COO-hashed — reducing over ``k`` then ``j``,
    and JIT-compiles once per ``(a.format, b.format, output_format)`` (memoized on
    the function attribute ``matmul_wksp._module_cache``).

    Use this when you specifically want to exercise the workspace codegen path;
    :func:`matmul` is the tuned entry point for everyday products.

    Parameters
    ----------
    a, b : torch.Tensor or STensor
        2-D operands. ``torch.Tensor`` inputs are **always** converted to sparse
        via ``STensor.from_torch(...).to_sparse()`` (note the unconditional
        ``.to_sparse()``).
    output_format : TensorFormat or str or list of str, optional
        Requested output format. Defaults to ``"ds"`` (CSR). Accepts anything
        :func:`scorch.utils.parse_format` understands.
    **kwargs
        ``time_dict`` : dict, optional -- if given, ``time_dict["eval_time"]`` is
        set to the kernel wall-clock time in seconds.

    Returns
    -------
    STensor
        Result of shape ``(a.shape[0], b.shape[1])`` in ``output_format``. Unlike
        :func:`matmul`, a dense-format result is **not** converted back to a
        ``torch.Tensor`` — an ``STensor`` is always returned.

    See Also
    --------
    matmul : Tuned entry point (prebuilt kernels + tiling + thread policy).

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> A = scorch.from_torch((torch.rand(64, 64) < 0.1).float(), "A").to_sparse("ds")
    >>> B = scorch.from_torch((torch.rand(64, 64) < 0.1).float(), "B").to_sparse("ds")
    >>> C = scorch.matmul_wksp(A, B, output_format="ds")   # STensor (CSR)
    >>> torch.allclose(C.to_torch(), A.to_torch() @ B.to_torch(), atol=1e-3, rtol=1e-3)
    True
    """
    if isinstance(a, torch.Tensor):
        a = STensor.from_torch(a).to_sparse()
    if isinstance(b, torch.Tensor):
        b = STensor.from_torch(b).to_sparse()

    if output_format is None:
        output_format = parse_format("ds")
    elif not isinstance(output_format, TensorFormat):
        output_format = parse_format(output_format)

    # ── Module cache: skip CIN→LLIR→codegen on repeat calls ──────────
    result_shape = (a.shape[0], b.shape[1])
    _cache_key = (
        a.layout.serialize(),
        b.layout.serialize(),
        str(a.dtype),
        str(b.dtype),
        str(output_format),
        result_shape,
    )
    if not hasattr(matmul_wksp, "_module_cache"):
        matmul_wksp._module_cache = {}

    module = matmul_wksp._module_cache.get(_cache_key)
    if module is None:
        C = TensorVar("C", shape=result_shape, fmt=output_format, dtype=a.dtype)
        A = TensorVar("A", shape=a.shape, fmt=a.format, dtype=a.dtype)
        B = TensorVar("B", shape=b.shape, fmt=b.format, dtype=b.dtype)

        # Use a dense workspace when the output is dense (avoids COO hash-map overhead).
        wksp_dense = output_format.is_dense()
        workspace = Workspace(
            name="wksp",
            dim=1,
            dense=wksp_dense,
        )

        i = IndexVar("i")
        j = IndexVar("j")
        k = IndexVar("k")

        cin_stmt = ForAll(
            i,
            Where(
                producer=ForAll(
                    k,
                    ForAll(
                        j,
                        TensorAssign(
                            workspace[j],
                            A[i, k] * B[k, j],
                            op=Operation.ADD,
                        ),
                    ),
                ),
                consumer=ForAll(
                    j,
                    TensorAssign(
                        C[i, j],
                        workspace[j],
                    ),
                ),
            ),
        )

        lowerer = CINLowerer()
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        header_cpp_code = jit_preamble_text()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )
        matmul_wksp._module_cache[_cache_key] = module

    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._native_mode_indices())  # type: ignore
        args.append(tensor.values)  # type: ignore

    start_time = time.time()
    result_cpp = module.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time
    # print("Time taken for evaluate:", eval_time)

    result = STensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=_finalize_generated_mode_indices(
                output_format, result_cpp.storage.index.mode_indices
            ),
            tensor_format=output_format,
        ),
        value=result_cpp.storage.value,
    )

    return result


def _dispatch_tiled(a, b, resolved, nthreads, atparallel, time_dict):
    """Run the adaptive tiling selector's choice, or report that v2 should run.

    Returns ``(result_cpp, result_shape, plan_kind, plan_param)``, with the first two
    ``None`` when the caller must run its ordinary v2 dispatch, and the last two
    naming the kernel that *did* serve the call so a plan can reproduce it.

    On the drop-in CSR@dense SpMM this routes the high-degree operand-over-LLC thrash
    regime (reddit/products-class) to the column-panel kernel
    ``spmm_csr_float_tilej``; v2 serves everything else byte-unchanged. Provably
    no-regression: v2 is always the probe baseline, so the memoized choice is never
    slower than v2 (see ``tiling.maybe_dispatch``). It only engages when the cheap
    O(1) pre-filter says a shape can even benefit — no overhead on
    GCN-small/AE/panel.
    """
    if resolved.symbol_name != "spmm_csr_float_v2":
        return None, None, "v2", None
    # Resolve the autotune level ONCE so is_candidate and maybe_dispatch see a
    # consistent value (a context manager could exit between two lookups). Only
    # touched on the v2 symbol -> other prebuilt kernels (bias_act/fused) stay
    # byte-identical.
    level = _tiling_current_level()
    if not _tiling_is_candidate(a, b, level=level):
        return None, None, "v2", None
    result_shape = [a.shape[0], b.shape[1]]

    def _v2_fn(nt):
        rc, _ = execute_prebuilt_binary_kernel(
            resolved.fn, a, b, nthreads=nt, atparallel=atparallel
        )
        return rc

    disp = _tiling_maybe_dispatch(
        a, b, result_shape, _v2_fn, nthreads, time_dict=time_dict, level=level
    )
    if disp is None:
        return None, None, "v2", None
    result_cpp, _ = disp
    # A tiled kernel served this call. Read which one HERE and nowhere else: the
    # memo is only authoritative where the selector actually ran. Reading it
    # wherever an entry merely exists let a plan carry tile-j for a shape the gate
    # had rejected, so the plan ran a different kernel than the ordinary path --
    # caught by a test comparing the two bitwise.
    verdict = _tiling_decided(a, b.shape[1], level=level)
    if verdict is None:
        return result_cpp, tuple(result_shape), "v2", None
    return result_cpp, tuple(result_shape), verdict[0], verdict[1]


def matmul(
    a: Union[torch.Tensor, STensor],
    b: Union[torch.Tensor, STensor],
    **kwargs: Any,
) -> Union[torch.Tensor, STensor]:
    """Matrix multiplication for any mix of dense and sparse operands.

    The tuned public entry point for two-operand products (SpMV, SpMM, SpGEMM, and
    the dense-dense passthrough). Dispatch proceeds through three tiers, tried in
    order:

    1. **Both-dense torch delegate.** If both operands are dense (dense
       ``torch.Tensor`` x dense ``torch.Tensor``, or dense ``STensor`` x dense
       ``STensor``), the product is handed to ``torch.matmul`` directly.
    2. **Prebuilt hand-written C++ kernel.** ``resolve_prebuilt_matmul`` matches the
       operand ranks/formats/dtype against the native ``scorch_ops`` extension
       (CSR x dense -> dense, CSR x CSR -> CSR, COO x COO -> COO, COO x dense ->
       dense, CSR x vector -> dense vector). On the drop-in ``spmm_csr_float_v2``
       symbol it additionally applies the adaptive tiling selector
       (``tiling.maybe_dispatch``, provably no-regression because v2 is always the
       probe baseline) and the host-thread composition hints.
    3. **Generic JIT compiler pipeline.** On a prebuilt miss the product is emitted
       as ``einsum("ij,jk->ik", a, b, ...)`` (2-D x 1-D goes to :func:`spmv`),
       compiling a C++ kernel through the CIN -> LLIR -> codegen pipeline.

    Sparse 2-D operands are normalized to canonical mode order ``[0, 1]`` before
    dispatch so transposed / non-default layouts stay correct on the fast paths.

    Parameters
    ----------
    a, b : torch.Tensor or STensor
        Operands. ``torch.Tensor`` inputs are auto-converted: dense-dense returns
        immediately via ``torch.matmul``; two ``sparse_coo`` tensors are promoted
        to CSR; any other sparse input triggers ``STensor.from_torch``.
    **kwargs
        format : str or list or TensorFormat, optional
            Requested output format. Read as
            ``kwargs.get("format", kwargs.get("output_format", None))``. For the
            dense-dense path only ``format`` is consulted. When omitted for a
            compiled product the output format is inferred (see :func:`einsum`).
        output_format : str or list or TensorFormat, optional
            Alias for ``format`` (see above).
        use_cache : bool, default True
            When ``True``, allow the prebuilt-kernel fast path and the adaptive
            tiling selector. ``False`` forces the generic ``einsum`` path (used by
            the compiler gap tests).
        schedule : Schedule, optional
            Explicit JIT loop order and affine tiling decisions. Supplying one
            forces the generic compiler path even when a prebuilt kernel matches.
        time_dict : dict, optional
            If given, ``time_dict["eval_time"]`` is set to the kernel wall-clock
            time in seconds.

    Returns
    -------
    torch.Tensor or STensor
        A dense ``torch.Tensor`` when the result format is dense (the common SpMM
        case, e.g. ``ds @ dd -> dd``, and every dense-dense product), otherwise an
        ``STensor`` (e.g. the SpGEMM ``ds @ ds -> ds``).

    Other Parameters
    ----------------
    SCORCH_MATCH_HOST_THREADS : env, default "1"
        Set to ``"0"`` to stop passing ``torch.get_num_threads()`` to the drop-in
        ``spmm_csr_float_v2`` kernel (thread-team matching for GCN pipelines).
    SCORCH_ATPARALLEL_PIPELINE : env, default "1"
        Set to ``"0"`` to launch the SpMM on a private libgomp team instead of
        torch's intra-op pool.

    See Also
    --------
    einsum : The generic compiler front door matmul falls through to.
    matmul_wksp : Explicit workspace-lowering SpMM/SpGEMM (always compiled).

    Notes
    -----
    Compiled kernels are memoized aggressively; a stale extensions build directory
    can mask a codegen edit, so clear ``TORCH_EXTENSIONS_DIR`` to force a recompile.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> A_dense = (torch.rand(256, 256) * (torch.rand(256, 256) < 0.1)).float()
    >>> x = torch.rand(256, 128)
    >>> A = scorch.from_torch(A_dense, "A").to_sparse("ds")   # CSR STensor
    >>> Y = scorch.matmul(A, x)                                # -> dense torch.Tensor
    >>> torch.allclose(Y, A_dense @ x, atol=1e-3, rtol=1e-3)
    True

    Sparse x sparse (SpGEMM) producing a sparse ``STensor``:

    >>> B = scorch.from_torch(A_dense, "B").to_sparse("ds")
    >>> C = scorch.matmul(A, B, format="ds")                   # -> STensor (CSR)
    >>> torch.allclose(C.to_torch(), A_dense @ A_dense, atol=1e-3, rtol=1e-3)
    True
    """

    # A repeated CSR x dense product is served by a plan: the resolution, the
    # selector's verdict, the validated index structure and the kernel arguments
    # were all settled on an earlier call and live in the native extension (see
    # plan.py). What is left is a dict lookup and one Python->C++ hop.
    #
    # Deliberately the first thing this function does, and written as inline
    # dict/attribute reads rather than a helper call, because everything below is
    # what a plan exists to skip. The exact-type tests keep it off every other
    # shape of call at the cost of two pointer comparisons: a torch.fx Proxy, an
    # STensor right-hand operand, a dense left-hand operand and a sparse-sparse
    # product all fail them and fall through unchanged. Keyword arguments are
    # excluded wholesale -- output_format, use_cache, schedule and time_dict all
    # change what the call means or what it must report.
    # The same test decides whether a plan may serve this call and whether one may
    # be installed for it at the end of the prebuilt branch, so it is made once.
    # plan_a/plan_b are the caller's own objects: a plan is keyed on what the
    # caller will pass again, never on an operand this function wrapped or relaid
    # out, which is why the installer below also checks `a is plan_a`.
    #
    # `_PLAN_ENABLED[0]` comes first so that switching plans off leaves this whole
    # paragraph -- and the installation below, which is what `plan_b is not None`
    # tests for -- costing one list index. That is what makes the off state a
    # control arm for measuring the on state rather than a different code path with
    # its own overhead. Measured: with the probe ungated, the off state ran 2-4%
    # slower than the tree before plans on the small cells of both hosts.
    plan_a = plan_b = plan_key = None
    if _PLAN_ENABLED[0] and not kwargs and type(a) is STensor and type(b) is _TENSOR:
        plan_a, plan_b = a, b
        # Built once and handed to the installer below, because every call that gets
        # here needs it and it used to be built twice: hashing it means hashing a
        # torch.Size and a dtype, which is most of what the probe costs on a call that
        # has no plan to find.
        plan_key = (b.shape, b.dtype, _PLAN_GENERATION[0])
        plans = a.__dict__.get(_PLANS_ATTR)
        if plans is not None:
            plan = plans.get(plan_key)
            # A miss means no plan for this free dimension; None back from `run`
            # means a plan that declined (see plan.py). Both fall through.
            if plan is not None:
                served = plan.run(
                    a._storage._value, b, _PLAN_NTHREADS(), _PLAN_ATPARALLEL
                )
                if served is not None:
                    return served

    # ``scorch.compile`` traces user functions with torch.fx.  Keep the leaf
    # behavior on the function itself so tracing never has to replace the
    # public ``scorch.matmul`` binding (or aliases in user globals).  A Proxy
    # carries its tracer, which can record this exact call without descending
    # into the eager dispatch below.
    proxy = a if isinstance(a, Proxy) else b if isinstance(b, Proxy) else None
    if proxy is not None:
        return proxy.tracer.create_proxy(
            "call_function", matmul, (a, b), kwargs
        )  # type: ignore[return-value]

    effective_schedule = _effective_schedule(kwargs)

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if (
            a.is_sparse
            and b.is_sparse
            and a.layout == torch.sparse_coo
            and b.layout == torch.sparse_coo
        ):
            a = a.to_sparse_csr()
            b = b.to_sparse_csr()
        if a.is_sparse or a.is_sparse_csr or b.is_sparse or b.is_sparse_csr:
            a = STensor.from_torch(a)
            b = STensor.from_torch(b)
        else:
            if effective_schedule is not None:
                raise NotImplementedError(
                    "Explicit compiler schedules currently require a sparse operand"
                )
            return torch.matmul(a, b)

    if isinstance(a, torch.Tensor):
        a = STensor.from_torch(a)
    if isinstance(b, torch.Tensor):
        b = STensor.from_torch(b)

    # Dense STensor x dense STensor: use torch dense matmul directly.
    # This is both faster and more reliable than lowering through sparse
    # scheduling paths for fully dense operands.
    if a.format.is_dense() and b.format.is_dense():
        if effective_schedule is not None:
            raise NotImplementedError(
                "Explicit compiler schedules currently require a sparse operand"
            )
        start_time = time.time()
        result_torch = torch.matmul(
            a.to_torch(in_place=False),
            b.to_torch(in_place=False),
        )
        end_time = time.time()
        eval_time = end_time - start_time
        if "time_dict" in kwargs:
            time_dict = kwargs["time_dict"]
            time_dict["eval_time"] = eval_time

        output_format_kw = kwargs.get("format", None)
        if output_format_kw is None:
            return result_torch

        output_format = parse_format(output_format_kw)
        if output_format.is_dense():
            return result_torch

        return STensor.from_torch(result_torch).to_sparse(output_format)

    # Never silently swallow a codegen schedule in a prebuilt dispatch.
    use_cache = kwargs.get("use_cache", True) and effective_schedule is None
    time_dict = kwargs.get("time_dict", None)
    requested_output_format = kwargs.get("format", kwargs.get("output_format", None))
    einsum_kwargs = dict(kwargs)
    if "output_format" in einsum_kwargs and "format" not in einsum_kwargs:
        einsum_kwargs["format"] = einsum_kwargs.pop("output_format")

    if a.dim() == 2 and b.dim() == 1:
        if effective_schedule is not None:
            raise NotImplementedError(
                "Explicit schedules are not yet threaded through the SpMV compiler"
            )
        default_mode_order = [0, 1]
        if (
            not a.format.is_dense()
        ) and a.storage.index.mode_order != default_mode_order:
            a = a.copy()
            a.change_mode_order(default_mode_order)

        if use_cache:
            resolved = resolve_prebuilt_matmul(
                a, b, output_format=requested_output_format
            )
            if resolved is not None:
                result_cpp, result_shape = execute_prebuilt_binary_kernel(
                    resolved.fn, a, b, time_dict=time_dict
                )
                result = STensor(
                    shape=result_shape,
                    index=TensorIndex(
                        mode_indices=_finalize_generated_mode_indices(
                            resolved.output_format,
                            result_cpp.storage.index.mode_indices,
                        ),
                        tensor_format=resolved.output_format,
                    ),
                    value=result_cpp.storage.value,
                )
                if result.format.is_dense():
                    return result.to_torch()
                return result

        spmv_kwargs = dict(kwargs)
        if "format" in spmv_kwargs and "output_format" not in spmv_kwargs:
            spmv_kwargs["output_format"] = spmv_kwargs.pop("format")
        return spmv(a, b, **spmv_kwargs)

    # Normalize sparse 2D operands to canonical mode order before dispatch.
    # This keeps fast kernels correct and avoids known non-default mode-order
    # issues in the generic path.
    if a.dim() == 2 and b.dim() == 2:
        default_mode_order = [0, 1]
        has_non_default_mode_order = (
            a.storage.index.mode_order != default_mode_order
            or b.storage.index.mode_order != default_mode_order
        )
        has_sparse_input = (not a.format.is_dense()) or (not b.format.is_dense())
        if has_non_default_mode_order and has_sparse_input:
            if a.storage.index.mode_order != default_mode_order:
                a = a.copy()
                a.change_mode_order(default_mode_order)
            if b.storage.index.mode_order != default_mode_order:
                b = b.copy()
                b.change_mode_order(default_mode_order)

    if use_cache:
        resolved = resolve_prebuilt_matmul(a, b, output_format=requested_output_format)
        if resolved is not None:
            nthreads = None
            atparallel = False
            if _MATCH_HOST_THREADS and resolved.symbol_name == "spmm_csr_float_v2":
                nthreads = torch.get_num_threads()
                atparallel = _ATPARALLEL_PIPELINE
            result_cpp, result_shape, _plan_kind, _plan_param = _dispatch_tiled(
                a, b, resolved, nthreads, atparallel, time_dict
            )
            if result_cpp is None:
                result_cpp, result_shape = execute_prebuilt_binary_kernel(
                    resolved.fn,
                    a,
                    b,
                    time_dict=time_dict,
                    nthreads=nthreads,
                    atparallel=atparallel,
                )
            # Fast path: a dense output kernel already produced a contiguous
            # row-major value buffer. Return it reshaped directly, skipping the
            # STensor/TensorIndex construction and the to_dense()/to_torch()
            # round-trip — all pure per-call Python overhead here, since the
            # result is default mode order with matching dtype (exactly what
            # to_torch() would have reconstructed).
            if resolved.output_format.is_dense():
                out = result_cpp.storage.value.reshape(result_shape)
                # Everything above is a function of the operands' structure and
                # the free dimension, so a second call with the same pair need not
                # repeat it: hand the resolution and the kernel that actually ran to
                # a plan the next call can invoke directly (plan.py). `plan_b` is
                # None unless plans are on and the caller passed a sparse STensor and
                # a dense torch.Tensor with no keywords; `a is plan_a` fails when the
                # operand was relaid out into a copy above, and a plan must describe
                # the operand the caller holds, not one this function made.
                if plan_b is not None and a is plan_a:
                    _plan_install(
                        a,
                        plan_b,
                        plan_key,
                        resolved.symbol_name,
                        _plan_kind,
                        _plan_param,
                    )
                return out
            result = STensor(
                shape=result_shape,
                index=TensorIndex(
                    mode_indices=_finalize_generated_mode_indices(
                        resolved.output_format,
                        result_cpp.storage.index.mode_indices,
                    ),
                    tensor_format=resolved.output_format,
                ),
                value=result_cpp.storage.value,
            )
        else:
            result = einsum("ij,jk->ik", a, b, **einsum_kwargs)
    else:
        result = einsum("ij,jk->ik", a, b, **einsum_kwargs)

    if isinstance(result, STensor) and result.format.is_dense():
        result = result.to_torch()

    return result


# Activation codes for the fused Linear kernel (must match spmm.h
# scorch_act_scalar / scorch_apply_row_bias_act).
_ACT_CODES = {None: 0, "none": 0, "identity": 0, "relu": 1, "sigmoid": 2}
# Empty mode-indices for a dense operand (reused so the chain allocates no lists).
_EMPTY_MODE_INDICES = [[], []]


def fast_transpose(x: torch.Tensor) -> torch.Tensor:
    """Materialize the transpose of a 2D float32 tensor, fast.

    Returns a contiguous ``[C, R]`` tensor equal (bit-identical) to
    ``x.T.contiguous()`` for a ``[R, C]`` input, using the prebuilt cache-blocked
    ``scorch_transpose_2d_float`` (AVX2 8x8 / NEON 4x4 / scalar micro-tiles). This
    exists because ``torch.Tensor.T.contiguous()`` is a naive element-scatter that
    runs ~5x below memory bandwidth; once the fused ``sparse_linear`` epilogue is
    folded away, that input transpose is 40-66% of the whole forward.

    Falls back to ``x.T.contiguous()`` if the kernel is unavailable or the input
    is not a 2D float32 tensor (exactness is preserved either way).

    Parameters
    ----------
    x : torch.Tensor
        2-D ``float32`` tensor ``[R, C]`` (fast path); any other rank/dtype takes
        the exact ``x.T.contiguous()`` fallback.

    Returns
    -------
    torch.Tensor
        Contiguous ``[C, R]`` tensor, bit-identical to ``x.T.contiguous()``.

    Notes
    -----
    Uses ``torch.get_num_threads()`` when ``SCORCH_MATCH_HOST_THREADS`` is on, else
    ``-1``. Backs the input transpose inside :func:`sparse_linear`.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> x = torch.rand(1024, 768)
    >>> xt = scorch.fast_transpose(x)              # [768, 1024]
    >>> torch.equal(xt, x.T.contiguous())          # bit-identical
    True
    """
    import scorch_ops as _native

    fn = getattr(_native, "scorch_transpose_2d_float", None)
    if fn is None or x.dim() != 2 or x.dtype != torch.float32:
        return x.T.contiguous()
    nthreads = torch.get_num_threads() if _MATCH_HOST_THREADS else -1
    return fn(x.contiguous(), nthreads)


def sparse_softmax_csr(
    crow_indices: torch.Tensor,
    values: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Row-wise softmax over the nonzeros of a CSR value array, ``scale`` folded in.

    Returns ``softmax(scale * values)`` computed per CSR row span
    ``[crow[i], crow[i+1])``, matching a torch scatter softmax up to float
    rounding. Backs the sparse-attention chain (SDDMM -> softmax -> SpMM): the
    prebuilt ``scorch_sparse_softmax_csr_float`` walks each row's contiguous span
    sequentially (max, exp+sum, normalize) with no scatter and no nnz-sized
    intermediates, parallel over rows on torch's warm intra-op pool.

    Falls back to a torch segment-softmax if the kernel is unavailable or the
    values are not float32 (exactness preserved either way).

    Parameters
    ----------
    crow_indices : torch.Tensor
        CSR row-pointer tensor of length ``nrows + 1``.
    values : torch.Tensor
        CSR nonzero values (``float32`` for the fast path); softmax is computed
        within each row's span.
    scale : float, default 1.0
        Multiplier folded into the logits before softmax (e.g. the attention
        scaling ``1 / sqrt(d)``).

    Returns
    -------
    torch.Tensor
        Dense tensor the same length as ``values``, normalized per CSR row span.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> M = (torch.rand(8, 8) < 0.4).float()
    >>> csr = M.to_sparse_csr()
    >>> w = scorch.sparse_softmax_csr(csr.crow_indices(), csr.values().float())
    >>> w.shape[0] == csr.values().numel()       # sums to 1 within each row span
    True
    """
    import scorch_ops as _native

    fn = getattr(_native, "scorch_sparse_softmax_csr_float", None)
    if fn is None or values.dtype != torch.float32:
        crow = crow_indices.long()
        nrows = crow.numel() - 1
        row_ids = torch.repeat_interleave(
            torch.arange(nrows, device=values.device), crow[1:] - crow[:-1]
        )
        logits = values * scale
        row_max = torch.full((nrows,), float("-inf"), device=values.device)
        row_max.scatter_reduce_(0, row_ids, logits, reduce="amax", include_self=False)
        exp_vals = torch.exp(logits - row_max[row_ids])
        row_sum = torch.zeros(nrows, device=values.device)
        row_sum.scatter_add_(0, row_ids, exp_vals)
        return exp_vals / row_sum[row_ids]
    nthreads = torch.get_num_threads() if _MATCH_HOST_THREADS else -1
    return fn(crow_indices, values.contiguous(), float(scale), nthreads)


def _sparse_attention_fallback(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Pure-torch reference for ``sparse_attention`` (kernel unavailable/dtype).

    Reproduces the fused kernel's math head-by-head with a gather SDDMM, the
    CSR-native row softmax (``sparse_softmax_csr``, itself torch-fallback safe),
    and a torch sparse SpMM — so the API always returns the right ``[S,H,D]``.
    """
    S, H, D = int(Q.shape[0]), int(Q.shape[1]), int(Q.shape[2])
    crow = crow_indices.long()
    col = col_indices.long()
    counts = crow[1:] - crow[:-1]
    rows = torch.repeat_interleave(torch.arange(S, device=Q.device), counts)
    out = torch.empty_like(Q)
    for h in range(H):
        Qh, Kh, Vh = Q[:, h, :], K[:, h, :], V[:, h, :]
        score = (Qh[rows] * Kh[col]).sum(dim=-1)  # (nnz,)
        attn = sparse_softmax_csr(crow_indices, score.contiguous(), scale)
        attn_csr = torch.sparse_csr_tensor(crow, col, attn, (S, S))
        out[:, h, :] = torch.sparse.mm(attn_csr, Vh.contiguous())
    return out


def sparse_attention(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused sparse (masked) multi-head attention over a shared CSR mask.

    Computes, for each query row ``i`` and head ``h``,
    ``out[i,h] = sum_j softmax_j(scale * Q[i,h]·K[j,h]) * V[j,h]`` over the row's
    attended columns ``j`` (the nonzeros of the CSR mask ``crow``/``col``) — i.e.
    the same masked-attention math the dense path does with ``-inf`` fills, over
    exactly the mask's structural nonzeros.

    ``Q``/``K``/``V`` are dense ``[S, H, D]`` float32 tensors (as produced by
    ``q_proj(x).view(S, H, D)``); the CSR mask structure is passed once. Returns a
    dense ``[S, H, D]`` tensor. The prebuilt ``scorch_sparse_attention_csr_float``
    does the whole thing in ONE parallel pass over rows (inline SDDMM + two-pass
    row softmax in registers + weighted-V accumulation), batched over heads — no
    per-head kernel dispatch, no nnz-sized intermediates, no CSR round-trip. This
    is "lever 2" for the sparse-attention bench: it removes the residual per-head
    Python-dispatch/materialization overhead left after the CSR-native softmax
    (``sparse_softmax_csr``, "lever 1").

    Falls back to a pure-torch per-head reference if the kernel is unavailable or
    the tensors are not float32 (result identical up to float rounding).

    Parameters
    ----------
    crow_indices, col_indices : torch.Tensor
        Shared CSR attention **mask** structure over the ``[S, S]`` query x key
        grid (passed once, shared across heads).
    Q, K, V : torch.Tensor
        Dense ``[S, H, D]`` ``float32`` tensors (``S`` = sequence length,
        ``H`` = heads, ``D`` = head dim).
    scale : float, default 1.0
        Attention scale, e.g. ``1 / sqrt(D)``, folded into the logits.

    Returns
    -------
    torch.Tensor
        Dense ``[S, H, D]`` attention output.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> S, H, D = 128, 4, 32
    >>> csr = (torch.rand(S, S) < 0.1).float().to_sparse_csr()
    >>> Q, K, V = torch.rand(S, H, D), torch.rand(S, H, D), torch.rand(S, H, D)
    >>> out = scorch.sparse_attention(
    ...     csr.crow_indices(), csr.col_indices(), Q, K, V, scale=1.0 / (D ** 0.5),
    ... )
    >>> tuple(out.shape)
    (128, 4, 32)
    """
    import scorch_ops as _native

    fn = getattr(_native, "scorch_sparse_attention_csr_float", None)
    if (
        fn is None
        or Q.dtype != torch.float32
        or K.dtype != torch.float32
        or V.dtype != torch.float32
    ):
        return _sparse_attention_fallback(crow_indices, col_indices, Q, K, V, scale)
    nthreads = torch.get_num_threads() if _MATCH_HOST_THREADS else -1
    return fn(
        crow_indices,
        col_indices,
        Q.contiguous(),
        K.contiguous(),
        V.contiguous(),
        float(scale),
        nthreads,
    )


def sparse_linear_fm(
    x_fm: Union[torch.Tensor, STensor],
    weight: STensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Fused feature-major sparse Linear layer.

    Computes ``Y = act(weight @ x_fm + bias[:, None])`` in ONE prebuilt parallel
    region (SpMM + per-output-channel bias + activation), returning a dense,
    FEATURE-MAJOR result — so an autoencoder forward stays feature-major
    throughout (transpose the input once in, transpose the output once out) with
    NO per-layer transpose and NO separate torch bias/act epilogue.

    Parameters
    ----------
    x_fm : torch.Tensor or STensor
        Dense activation ``[in, batch]`` in FEATURE-MAJOR (row = input feature)
        layout. The output of a previous ``sparse_linear_fm`` (already
        ``[out, batch]``) feeds straight in.
    weight : STensor
        Sparse CSR weight ``[out, in]`` (as in ``F.linear``: ``y = x @ weight.T``).
    bias : torch.Tensor, optional
        Dense ``[out]`` bias, added per output channel.
    activation : str, optional
        ``None`` / ``"relu"`` / ``"sigmoid"``.

    Returns
    -------
    torch.Tensor
        Dense ``[out, batch]`` result (feature-major).

    Raises
    ------
    ValueError
        If ``activation`` is not one of ``None`` / ``"none"`` / ``"identity"`` /
        ``"relu"`` / ``"sigmoid"``.

    Notes
    -----
    Routes to the prebuilt ``spmm_csr_linear_fused_float`` kernel (v2's fast
    regtile/regblock inner loops + the fused epilogue), falling back to a dense
    torch reference when the kernel is unavailable or the dtype is not float32.
    Carries the same pipeline composition hints as the drop-in ``matmul`` SpMM
    (``SCORCH_MATCH_HOST_THREADS`` / ``SCORCH_ATPARALLEL_PIPELINE``). This is a
    DISTINCT entry point from :func:`matmul`, so the FEM/GCN SpMM path
    (``scorch.matmul`` -> ``spmm_csr_float_v2`` / ``spmm_csr_bias_act``) never
    reaches this kernel — the guardrail is the API boundary.

    See Also
    --------
    sparse_linear : Natural-layout ``F.linear`` drop-in built on this kernel.
    """
    import scorch_ops as _native

    act = _ACT_CODES.get(activation, None)
    if act is None:
        raise ValueError(f"unsupported activation {activation!r}")

    # Extract the dense feature-major operand [in, batch] WITHOUT wrapping it in an
    # STensor — a dense operand needs only its shape, empty mode-indices, and flat
    # values, so the full STensor.from_torch construction (TensorFormat/TensorIndex
    # objects) is pure per-layer overhead in a chain. A dense STensor input is
    # unwrapped the same cheap way.
    if isinstance(x_fm, torch.Tensor):
        x_vals = x_fm.reshape(-1)
        in_dim = int(x_fm.shape[0])
        batch = int(x_fm.shape[1])
    else:
        x_vals = x_fm.values
        in_dim = int(x_fm.shape[0])
        batch = int(x_fm.shape[1])
    out_dim = int(weight.shape[0])
    result_shape = [out_dim, batch]

    if bias is None:
        bias_t = torch.zeros(out_dim, dtype=torch.float32)
    else:
        bias_t = bias if bias.dtype == torch.float32 else bias.to(torch.float32)
        bias_t = bias_t.contiguous()

    fn = getattr(_native, "spmm_csr_linear_fused_float", None)
    if (
        fn is None
        or weight.values.dtype != torch.float32
        or x_vals.dtype != torch.float32
    ):
        # Fallback (kernel unavailable / unsupported dtype): dense reference in
        # feature-major layout so the op still produces the right result.
        w_dense = weight.to_torch()
        x_dense = x_vals.reshape(in_dim, batch)
        y = torch.matmul(w_dense, x_dense) + bias_t.view(-1, 1)
        if act == 1:
            return torch.relu(y)
        if act == 2:
            return torch.sigmoid(y)
        return y

    nthreads = -1
    atparallel = False
    if _MATCH_HOST_THREADS:
        nthreads = torch.get_num_threads()
        atparallel = _ATPARALLEL_PIPELINE

    result_cpp = fn(
        result_shape,
        list(weight.shape),
        weight._native_mode_indices(),
        weight.values,
        [in_dim, batch],
        _EMPTY_MODE_INDICES,
        x_vals,
        bias_t,
        act,
        256,
        nthreads,
        atparallel,
    )
    return result_cpp.storage.value.reshape(result_shape)


def sparse_linear(
    x: torch.Tensor,
    weight: Union[STensor, torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Drop-in ``F.linear`` with a sparse weight, fused with bias + activation.

    Computes ``act(x @ weight.T + bias)`` for a NATURAL-layout activation
    ``x`` ``[batch, in]``, returning ``[batch, out]`` — the exact signature of
    ``torch.nn.functional.linear`` (plus an optional fused activation). This is
    the transparent per-call entry point: replace ``act(F.linear(x, W, b))`` with
    ``scorch.sparse_linear(x, W, b, "relu")`` and the SpMM + bias + activation run
    in ONE prebuilt parallel region.

    ``weight`` may be a sparse CSR ``STensor`` ``[out, in]`` (preferred — build it
    once and reuse) or a dense ``torch.Tensor`` ``[out, in]`` (converted to CSR per
    call; pass a pre-built STensor in a hot loop to avoid the conversion).
    ``activation`` -- ``None`` / ``"relu"`` / ``"sigmoid"``.

    Internally transposes ``x`` into feature-major ``[in, batch]`` (via the fast
    cache-blocked ``fast_transpose`` — NOT torch's cache-hostile
    ``x.T.contiguous()``, which would dominate the fused forward), runs the fused
    feature-major kernel (``sparse_linear_fm``), and returns the result's
    transpose as a lazy view — the SAME transpose pattern
    ``torch.sparse.mm(W, x.T).T`` MKL already pays, minus MKL's separate bias/act
    epilogue. For a multi-layer chain, ``sparse_linear_fm`` (staying feature-major
    end to end) avoids the intermediate transposes.

    Parameters
    ----------
    x : torch.Tensor
        Dense ``[batch, in]`` ``float32`` activation in natural layout.
    weight : STensor or torch.Tensor
        Sparse CSR ``STensor`` ``[out, in]`` (**preferred** — build it once and
        reuse), or a dense ``torch.Tensor`` ``[out, in]`` which is converted to CSR
        *per call*; pass a pre-built STensor in a hot loop to avoid the conversion.
    bias : torch.Tensor, optional
        Dense ``[out]`` bias.
    activation : str, optional
        ``None`` / ``"relu"`` / ``"sigmoid"``.

    Returns
    -------
    torch.Tensor
        Dense ``[batch, out]`` result.

    See Also
    --------
    sparse_linear_fm : Feature-major variant that avoids per-layer transposes.
    fast_transpose : The cache-blocked transpose used on the input.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> W_dense = (torch.rand(256, 512) < 0.1).float()
    >>> W = scorch.from_csr(W_dense.to_sparse_csr(), "W")   # build CSR STensor once
    >>> b = torch.rand(256)
    >>> x = torch.rand(64, 512)
    >>> y = scorch.sparse_linear(x, W, b, activation="relu")   # [64, 256]
    >>> torch.allclose(y, torch.relu(x @ W_dense.T + b), atol=1e-3, rtol=1e-3)
    True
    """
    if isinstance(weight, torch.Tensor) and not isinstance(weight, STensor):
        weight = STensor.from_csr(weight.to_sparse_csr(), "weight")
    x_fm = fast_transpose(x)
    y_fm = sparse_linear_fm(x_fm, weight, bias, activation)
    return y_fm.T


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2b (codegen-parity): register-block dual-path kernel emission.
#
# The free-dim register-block transform (holding the output tile in a stack-local
# accumulator across a row's nonzeros) WINS for narrow N (<=~3) but REGRESSES for
# wide N (re-traversal of the sparse row per free-dim tile). Because ONE
# format-keyed kernel serves all N, the fix is a RUNTIME branch inside the kernel on
# the free-dim size: register-block nest for small N (single tile, no re-traversal),
# the byte-identical baseline memory-destination nest for large N. This preserves
# wide-k parity by construction (the regblock arm cannot fire for large N) — the same
# discipline as the spmm.h nfloor / E-core-recruit gates.
#
# We build it by lowering the SAME (unscheduled) CIN twice — regblock forced OFF and
# ON — and splicing the two top-level compute loops under one llir.IfThenElse. See
# codegen-parity/04-phase2-results.md for the route evaluation (LLIR if-stitch chosen
# over an N-bucketed cache key).
#
# ROLLOUT: dual-path emission now defaults ON via `_regblock_dual_active()` after
# validation on M5 (2026-07-09) and x86/redwood (2026-07-10) showed it is
# neutral-or-better across the whole grid — narrow-k register-block WIN (x86 rb/base
# geomean 0.89), wide-k byte-identical else-arm PARITY. This is DECOUPLED from the
# scheduler's `SCORCH_REGBLOCK` env (which stays default OFF, so a direct
# `Scheduler.auto_schedule` caller — and the two schedule-shape tests — is unchanged):
# `_build_regblock_dual_path` forces the tiling LOCALLY via `regblock_force`, so it
# needs nothing from the global env. Escape hatch: `SCORCH_REGBLOCK_DUAL=0` restores
# the pre-flip single baseline path (also byte-identical to before the whole feature).
# ─────────────────────────────────────────────────────────────────────────────


def _regblock_dual_active() -> bool:
    """Whether `einsum` emits the register-block dual-path kernel for the qualifying
    pattern (dense output, sparse contraction, a tileable free dim).

    Default TRUE post-validation (see the module comment above). Set
    ``SCORCH_REGBLOCK_DUAL=0`` to restore the pre-flip single baseline path. This is
    intentionally SEPARATE from the scheduler's ``SCORCH_REGBLOCK`` / ``_regblock_enabled()``
    (default OFF): the dual builder forces its two lowerings locally, so flipping the
    per-op default here leaves every direct ``auto_schedule`` caller untouched. When the
    dual-path does not apply (non-qualifying pattern → `_build_regblock_dual_path`
    returns None), einsum falls back to the byte-identical baseline single path.
    """
    return os.environ.get("SCORCH_REGBLOCK_DUAL", "1") != "0"


def _find_top_compute_loop(
    body: List[llir.Stmt],
) -> Tuple[Optional[int], Optional[llir.ForLoop]]:
    """Locate the single top-level compute loop in a lowered function body.

    Prefer the omp-parallel-for row loop; fall back to a lone top-level ForLoop.
    Returns (None, None) when the shape is ambiguous (0 or >1 candidates) so the
    caller can safely decline the dual-path stitch.
    """
    for_loops = [(i, s) for i, s in enumerate(body) if isinstance(s, llir.ForLoop)]
    if not for_loops:
        return None, None
    parallel = [(i, s) for i, s in for_loops if getattr(s, "omp_parallel_for", False)]
    if len(parallel) == 1:
        return parallel[0]
    if len(for_loops) == 1:
        return for_loops[0]
    return None, None


def _regblock_free_size_expr(row_loop: llir.ForLoop) -> Optional[llir.Expr]:
    """Extract the free-dim size expression from the register-block row loop.

    The tiled nest contains a `for (k_out = 0; k_out < <free_size>; k_out += kTile)`
    loop; its bound (`cond.right`) IS the free-dim extent as the lowerer computed it
    (e.g. `B1_size`). Using it avoids hard-coding a variable name. Returns None if no
    `*_out` tile loop is found.
    """

    def _walk(stmts: List[llir.Stmt]) -> Optional[llir.Expr]:
        for s in stmts:
            if isinstance(s, llir.ForLoop):
                init = getattr(s, "init", None)
                var = getattr(init, "var", None) if init is not None else None
                name = getattr(var, "name", "") or ""
                if name.endswith("_out"):
                    cond = getattr(s, "cond", None)
                    right = getattr(cond, "right", None) if cond is not None else None
                    if right is not None:
                        return right
                found = _walk(getattr(s, "body", None) or [])
                if found is not None:
                    return found
        return None

    return _walk(getattr(row_loop, "body", None) or [])


def _stitch_regblock_dual_path(
    fn_rb: llir.Function, fn_base: llir.Function, cutoff_n: int
) -> Optional[llir.Function]:
    """Splice the register-block and baseline compute loops under one runtime branch.

    Keeps `fn_rb`'s prologue (a strict superset of the baseline's — it only adds the
    `kTile` decl) and epilogue, replacing its single top-level compute loop with
    `if (free_size <= cutoff_n) { <regblock loop> } else { <baseline loop> }`.
    Returns None (decline) if either function's compute region is ambiguous or the
    free-dim size expression can't be found.
    """
    idx_rb, loop_rb = _find_top_compute_loop(fn_rb.body)
    _, loop_base = _find_top_compute_loop(fn_base.body)
    if loop_rb is None or loop_base is None or idx_rb is None:
        return None
    free_expr = _regblock_free_size_expr(loop_rb)
    if free_expr is None:
        return None
    cond = llir.BinOp(
        op="<=",
        left=free_expr,
        right=llir.Literal(value=cutoff_n, data_type=llir.DataType.INT64),
    )
    branch = llir.IfThenElse(cond=cond, then_body=[loop_rb], else_body=[loop_base])
    new_body = list(fn_rb.body)
    new_body[idx_rb] = branch
    fn_rb.body = new_body
    return fn_rb


def _build_regblock_dual_path(
    cin_unscheduled: IndexStmt, post_ops: Any
) -> Optional[Tuple[llir.Function, str]]:
    """Build the Phase 2b dual-path kernel from an unscheduled CIN.

    Schedules + lowers the CIN twice (regblock forced OFF -> baseline nest, ON ->
    tiled nest) and stitches a runtime free-dim branch. Returns (Function, cache_key)
    or None when the register-block path doesn't apply (schedules identical) or the
    stitch can't be formed safely — the caller then falls back to the baseline path.
    """
    if not Scheduler._has_dense_output(cin_unscheduled):
        return None

    # auto_schedule mutates the CIN in place, so schedule each arm from a pristine copy.
    with regblock_force(False):
        cin_base = Scheduler.auto_schedule(copy.deepcopy(cin_unscheduled))
    with regblock_force(True):
        cin_rb = Scheduler.auto_schedule(copy.deepcopy(cin_unscheduled))
    if str(cin_base) == str(cin_rb):
        # Register-block didn't change the schedule (pattern doesn't qualify) ->
        # nothing to branch on; let the caller use the plain baseline path.
        return None

    with regblock_force(False):
        fn_base = CINLowerer(post_ops=post_ops).lower_IndexStmt(cin_base)
    with regblock_force(True):
        fn_rb = CINLowerer(post_ops=post_ops).lower_IndexStmt(cin_rb)
    if not (isinstance(fn_base, llir.Function) and isinstance(fn_rb, llir.Function)):
        return None

    stitched = _stitch_regblock_dual_path(fn_rb, fn_base, _regblock_max_n())
    if stitched is None:
        return None
    return stitched, _codegen_kernel_cache_key(cin_rb, post_ops, None) + "|rbdual"


def einsum(
    expression: str,
    *tensors: Optional[Union[torch.Tensor, STensor, TensorSpec]],
    compile_only: Optional[bool] = False,
    **kwargs: Any,
) -> Union[STensor, TensorSpec]:
    """Compile and evaluate a numpy-style einsum over sparse/dense operands.

    The generic front door to the sparse-tensor compiler. Given an index-notation
    expression it builds Compiler Index Notation (CIN), schedules and lowers it
    through LLIR to a C++ kernel, JIT-compiles (memoized), and executes — covering
    the binary contraction / elementwise / SDDMM patterns the library supports.
    :func:`matmul` falls through to this for anything the prebuilt kernels miss.

    Parameters
    ----------
    expression : str
        A numpy-style einsum string that **must contain an explicit** ``"->"``
        output, e.g. ``"ik,kj->ij"`` (matmul / SpMM / SpGEMM), ``"ij,ij->ij"``
        (elementwise multiply), or ``"ij,ik,jk->ij"`` (SDDMM). One character per
        index. An implicit-output form (no ``"->"``) is **not** supported and
        raises ``CompileSpecError``.
    *tensors : torch.Tensor, STensor, or TensorSpec
        The operands, one per comma-separated input group. ``torch.Tensor`` inputs
        are auto-converted with ``STensor.from_torch``. ``TensorSpec`` operands
        are accepted only for compile-only calls.
    compile_only : bool, default False
        When ``True``, compile and cache the kernel without executing and return
        an immutable output ``TensorSpec`` (used by :func:`precompile_kernels`).
    **kwargs
        format : str or list or TensorFormat, optional
            Requested output format. If omitted, the output format is inferred
            per-level (see Notes).
        output_mode_order : list of int, optional
            Permutation applied to the output modes.
        schedule : Schedule, optional
            Exact loop order and affine tiling plan for the JIT scheduler. The
            complete plan participates in both compiler cache keys.
        time_dict : dict, optional
            If given, ``time_dict["eval_time"]`` is set to the kernel wall-clock
            time in seconds.
        _post_ops, _post_ops_tensors : advanced/internal
            Fused post-op chain (bias/scale/activation) and its extra operand
            tensors, used by ``scorch.compile``. When ``_post_ops`` is set the fast
            dispatch cache and the register-block dual-path are bypassed.

    Returns
    -------
    STensor
        The result (dense or sparse per the output format). Callers such as
        :func:`matmul` convert a dense-format result back to a ``torch.Tensor``.

    Raises
    ------
    CompileSpecError
        If the expression or requested output contract is malformed.

    Notes
    -----
    **Output-format inference** (when ``format=`` is omitted) is decided per level:
    *sparse * anything = sparse; dense + anything = dense; otherwise compressed*,
    with ties preferring coordinate over compressed. A sparse level may not precede
    a dense level (a preceding sparse level is forced dense). As a special case,
    an SDDMM pattern (sparse output with reduction vars plus an input mirroring the
    output sparsity) yields an all-COO output to enable the scalar-accum codegen
    path — and there is a dedicated prebuilt fast path inside ``einsum`` for exactly
    ``"ij,ik,jk->ij"`` with ``S`` (COO), ``A`` (dense), ``B`` (dense) float32.

    For the qualifying pattern (dense output, sparse contraction, tileable free dim,
    ``_post_ops is None``) ``einsum`` emits ONE kernel that branches at runtime on
    the free-dim size: a register-block nest for small ``N`` and the byte-identical
    baseline nest for large ``N``. Set ``SCORCH_REGBLOCK_DUAL=0`` to restore the
    single baseline path (distinct from the scheduler's ``SCORCH_REGBLOCK``, default
    off).

    Historically fragile format / mode-order / loop-order combinations (transposed
    operands, dense-STensor products, broadcast vectors) are covered by the green
    regression tests in ``tests/test_scorch/test_known_compiler_gaps*.py`` — despite
    the name, those are *supported* cases. Unsupported patterns surface as compiler
    exceptions rather than documented errors.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> A = scorch.from_torch((torch.rand(64, 32) < 0.2).float(), "A").to_sparse("ds")
    >>> B = torch.rand(32, 48)
    >>> C = scorch.einsum("ik,kj->ij", A, B, format="dd")   # STensor, dense format
    >>> torch.allclose(C.to_torch(), A.to_torch() @ B, atol=1e-3, rtol=1e-3)
    True

    SDDMM (sampled ``A @ B.T`` on the nonzeros of a COO mask ``S``):

    >>> S = scorch.from_coo(torch.rand(50, 50).to_sparse_coo(), "S")
    >>> Ad = torch.rand(50, 16)
    >>> Bd = torch.rand(50, 16)
    >>> Sd = scorch.einsum("ij,ik,jk->ij", S, Ad, Bd)       # STensor (COO)
    """
    # e.g. expression might be e.g. "i,i->i" and "ij,ij->ij" for
    # elementwise multiplication or "ik,kj->ij" for matrix multiplication

    # # If any of the tensors have the same name, rename them
    # tensor_names = [tensor.name for tensor in tensors]
    # tensor_name_counts = {name: tensor_names.count(name) for name in tensor_names}
    # for i, tensor in enumerate(tensors):
    #     if tensor_name_counts[tensor.name] > 1:
    #         tensor.name = tensor.name + str(i)

    # Convert all torch.Tensor inputs to STensor early
    tensors = tuple(
        STensor.from_torch(t) if isinstance(t, torch.Tensor) else t for t in tensors
    )
    if not isinstance(expression, str) or expression.count("->") != 1:
        raise CompileSpecError("einsum expression must contain an explicit '->'")
    input_expression, result_expression = expression.split("->")
    input_groups = input_expression.split(",")
    if len(input_groups) != len(tensors):
        raise CompileSpecError(
            f"einsum expression expects {len(input_groups)} operands, got {len(tensors)}"
        )
    if any(tensor is None for tensor in tensors):
        raise TensorTypeError("einsum operands cannot be None")
    invalid = [
        type(tensor).__name__
        for tensor in tensors
        if not isinstance(tensor, (STensor, TensorSpec))
    ]
    if invalid:
        raise TensorTypeError(
            "einsum operands must be torch.Tensor, STensor, or TensorSpec; got "
            + ", ".join(invalid)
        )
    if any(
        not group or not all(label.isascii() and label.isalpha() for label in group)
        for group in input_groups
    ):
        raise CompileSpecError("einsum input labels must be non-empty ASCII letters")
    if not result_expression or not all(
        label.isascii() and label.isalpha() for label in result_expression
    ):
        raise CompileSpecError("einsum result labels must be non-empty ASCII letters")
    if len(set(result_expression)) != len(result_expression):
        raise CompileSpecError("einsum result labels must be unique")
    input_labels = set("".join(input_groups))
    unknown_result_labels = [
        label for label in result_expression if label not in input_labels
    ]
    if unknown_result_labels:
        raise CompileSpecError(
            "einsum result labels must appear in an input; unknown labels: "
            + ", ".join(unknown_result_labels)
        )
    input_index_strs = [list(group) for group in input_groups]
    result_index_strs = list(result_expression)
    requested_output_format = kwargs.get("format")
    if requested_output_format is not None:
        requested_output_format = parse_format(requested_output_format)
        if requested_output_format.get_order() != len(result_index_strs):
            raise CompileSpecError(
                f"einsum output format rank {requested_output_format.get_order()} "
                f"does not match result rank {len(result_index_strs)}"
            )
        if any(
            level_type == LevelType.SINGLETON
            for level_type in requested_output_format.get_level_types()
        ):
            raise CompileSpecError(
                "singleton output levels are not supported by the compiler"
            )
    for operand, (labels, tensor) in enumerate(zip(input_index_strs, tensors)):
        if len(labels) != tensor.dim():
            raise CompileSpecError(
                f"einsum operand {operand} has {len(labels)} labels but tensor "
                f"rank {tensor.dim()}"
            )
    output_mode_order = kwargs.get("output_mode_order")
    if output_mode_order is not None:
        if isinstance(output_mode_order, (str, bytes)) or not isinstance(
            output_mode_order, Sequence
        ):
            raise CompileSpecError("output_mode_order must be a sequence of integers")
        if any(
            isinstance(mode, bool) or not isinstance(mode, int)
            for mode in output_mode_order
        ):
            raise CompileSpecError("output_mode_order entries must be integers")
        if len(output_mode_order) != len(result_index_strs) or sorted(
            output_mode_order
        ) != list(range(len(result_index_strs))):
            raise CompileSpecError(
                "output_mode_order must be a permutation of the result modes"
            )
        output_mode_order = list(output_mode_order)
    _logical_index_sizes(input_index_strs, tensors)
    if not compile_only and any(isinstance(tensor, TensorSpec) for tensor in tensors):
        raise CompileSpecError(
            "TensorSpec has no runtime payload; pass compile_only=True or use STensor"
        )
    effective_schedule = _effective_schedule(kwargs)

    # ── Prebuilt SDDMM dispatch ────────────────────────────────────────
    # Pattern: 'ij,ik,jk->ij' with S(COO), A(dense), B(dense)
    if (
        effective_schedule is None
        and not compile_only
        and expression == "ij,ik,jk->ij"
        and len(tensors) == 3
        and tensors[0].values.dtype == torch.float32
        and str(tensors[0].format) == "o,o"
        and str(tensors[1].format) == "d,d"
        and str(tensors[2].format) == "d,d"
        and all(
            tuple(tensor.mode_order) == tuple(range(tensor.dim())) for tensor in tensors
        )
    ):
        import scorch_ops as _ops

        _sddmm_fn = getattr(_ops, "sddmm_coo_float_prebuilt", None)
        if _sddmm_fn is not None:
            S, A, B = tensors
            result_shape = S.shape
            result_cpp = _sddmm_fn(
                result_shape,
                S.shape,
                S._native_mode_indices(),
                S.values,
                A.shape,
                A._native_mode_indices(),
                A.values,
                B.shape,
                B._native_mode_indices(),
                B.values,
            )
            return STensor(
                shape=result_shape,
                index=TensorIndex(
                    mode_indices=_finalize_generated_mode_indices(
                        S.format, result_cpp.storage.index.mode_indices
                    ),
                    tensor_format=S.format,
                    mode_order=S.mode_order,
                ),
                value=result_cpp.storage.value,
            )

    # ── Fast dispatch cache ─────────────────────────────────────────────
    # On a cache hit, skip the entire scheduling pipeline (select_loop_order
    # + auto_schedule) which dominates wall-clock time for cached kernels.
    _dispatch_key = None
    if not compile_only and "_post_ops" not in kwargs:
        _dispatch_key = _einsum_cache_key(
            expression,
            tensors,
            kwargs.get("format", None),
            output_mode_order,
            effective_schedule,
        )
        _cached = _einsum_dispatch_cache.get(_dispatch_key)
        if _cached is not None:
            _module = _cached[0]
            _output_fmt = _cached[1]
            _temp_mo = _cached[2]
            _final_mo = _cached[3]
            _input_mos = _cached[4]
            _input_idx_strs = _cached[5]
            _result_idx_strs = _cached[6]

            # Set correct mode orders on input tensors
            cached_tensors = list(tensors)
            for _index, (_t, _mo) in enumerate(zip(cached_tensors, _input_mos)):
                if list(_t.mode_order) != _mo:
                    cached_tensors[_index] = _with_compiler_mode_order(_t, _mo)
            tensors = tuple(cached_tensors)

            # Compute result shape from expression + current tensor shapes
            _idx_to_size = _logical_index_sizes(_input_idx_strs, tensors)
            _logical_result_shape = tuple(_idx_to_size[_s] for _s in _result_idx_strs)
            _result_mode_order = _temp_mo or list(range(len(_logical_result_shape)))
            _physical_result_shape = tuple(
                _logical_result_shape[logical_mode]
                for logical_mode in _result_mode_order
            )

            # Build args and evaluate
            _args: List[Any] = [_physical_result_shape]
            for _t in tensors:
                _args.append(_t.shape)
                _args.append(_t._native_mode_indices())
                _args.append(_t.values)

            _t0 = time.time()
            _result_cpp = _module.evaluate(*_args)
            _eval_time = time.time() - _t0

            _result = STensor(
                shape=_physical_result_shape,
                index=TensorIndex(
                    mode_indices=_finalize_generated_mode_indices(
                        _output_fmt, _result_cpp.storage.index.mode_indices
                    ),
                    tensor_format=_output_fmt,
                    mode_order=_temp_mo if _temp_mo else _final_mo,
                ),
                value=_result_cpp.storage.value,
            )

            if "time_dict" in kwargs:
                kwargs["time_dict"]["eval_time"] = _eval_time

            if _temp_mo:
                _result.change_mode_order(_final_mo)

            return _result
    # ── End fast dispatch ────────────────────────────────────────────────

    # unique_index_strs should be a list of unique index strings
    # e.g. ["i", "j", "k"]
    unique_index_strs = list("".join(input_groups) + result_expression)
    # Make sure the index strings are unique, keeping the order
    unique_index_strs = list(dict.fromkeys(unique_index_strs))
    # Reorder input index strings by each tensor's mode_order
    input_index_strs_sorted = [
        [input_index_strs[i][idx] for idx in tensors[i].mode_order]
        for i in range(len(tensors))
    ]

    # Reorder result index strings by output_mode_order if provided
    result_index_strs_sorted = (
        [result_index_strs[i] for i in output_mode_order]
        if output_mode_order
        else result_index_strs
    )

    # Build concatenated substrings for topo_sort_characters
    index_strs_concat = ["".join(s) for s in input_index_strs_sorted] + [
        "".join(result_index_strs_sorted)
    ]
    index_strs_by_schedule = topo_sort_characters(index_strs_concat, tensors)

    # Create a list of IndexVar objects, and a dict mapping index strings
    # to IndexVar objects
    index_vars = [IndexVar(index_str) for index_str in unique_index_strs]
    index_var_dict = {index_var.name: index_var for index_var in index_vars}

    # Compute temp_mode_order: the scheduler-recommended mode order for result
    index_str_to_mode_index = {s: i for i, s in enumerate(result_index_strs)}
    temp_mode_order = [
        index_str_to_mode_index[s]
        for s in index_strs_by_schedule
        if s in result_index_strs
    ]
    final_mode_order = output_mode_order if output_mode_order else temp_mode_order

    # Change input tensor mode orders to match schedule
    tensors = list(tensors)
    for tensor_index, input_index_str in enumerate(input_index_strs):
        new_mode_order = []
        str_to_mode = {s: i for i, s in enumerate(input_index_str)}
        for s in index_strs_by_schedule:
            if s in str_to_mode:
                new_mode_order.append(str_to_mode[s])
        tensors[tensor_index] = _with_compiler_mode_order(
            tensors[tensor_index], new_mode_order
        )
    tensors = tuple(tensors)

    # Create a mapping from each index string to the list of LevelFormats
    # of the levels it indexes into each input tensor
    index_str_to_level_formats = {}
    tensors_new = []
    for sorted_index_strs, tensor in zip(input_index_strs_sorted, tensors):
        if not isinstance(tensor, (STensor, TensorSpec)):
            raise TensorTypeError("input tensor is not a Scorch tensor or spec")
        tensors_new.append(tensor)

        for i, index_str in enumerate(sorted_index_strs):
            if index_str not in index_str_to_level_formats:
                index_str_to_level_formats[index_str] = []
            index_str_to_level_formats[index_str].append(
                tensor.format.get_level_formats()[i]
            )

    tensors = tensors_new

    # Create TensorVar's for each tensor
    tensor_vars = []
    tensor_names_available = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    output_tensor_dtype = None
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, (STensor, TensorSpec)):
            tensor_name = tensor_names_available.pop(0)
            tensor_vars.append(
                TensorVar(
                    name=tensor_name,
                    fmt=tensor.format,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    mode_order=list(tensor.mode_order),
                )
            )
            if output_tensor_dtype is None:
                output_tensor_dtype = tensor.dtype
            else:
                if output_tensor_dtype != tensor.dtype:
                    raise TensorValidationError(
                        "all einsum operands must have the same value dtype"
                    )

    # Get output format from kwargs
    output_format = requested_output_format

    # If output format is not specified, do sparse for all levels first
    if output_format is None:
        # Use format inference rules to infer the optimal format of the output
        # tensor
        # The format inference rules are decided on a per-level basis:
        # 1. Let the index variable indexing into the level be called i
        # 2. If the index variable is used to index into any input tensor's sparse
        #    dimension and multiplied with any other tensor, then the level is
        #    sparse
        # 3. If the index variable is used to index into any input tensor's dense
        #    dimension and added with any other tensor, then the level is dense
        # 4. Otherwise, the level is compressed
        # i.e sparse * anything = sparse
        #     dense + anything = dense
        # Note that LevelType.COMPRESSED and LevelType.COORDINATE are both "sparse"
        # levels
        # To break ties, we use the following priority: we always prefer coordinate
        # over compressed

        # Create a list of LevelFormat objects
        output_level_formats = []
        for index_str in result_index_strs:
            level_format = LevelFormat(LevelType.DENSE)
            # Use the index_str_to_level_formats to get the list of LevelFormats
            # of the levels it indexes into each input tensor
            level_formats: List[LevelFormat] = index_str_to_level_formats[index_str]
            # If any of them is sparse, then the output level is sparse
            if any(
                level_format.get_level_type() == LevelType.COMPRESSED
                for level_format in level_formats
            ):
                level_format = LevelFormat(LevelType.COMPRESSED)
            # If any of them is coordinate, then the output level is coordinate format
            elif any(
                level_format.get_level_type() == LevelType.COORDINATE
                for level_format in level_formats
            ):
                level_format = LevelFormat(LevelType.COORDINATE)

            output_level_formats.append(level_format)

        # Make sure that the output format doesn't have a sparse level preceding
        # a dense level
        # If it does, then we need to make the preceding level dense as well
        # e.g. if the output level formats are [sparse, dense, dense], then we
        # need to make it [dense, dense, dense]
        # TODO: unless we are dealing with block tensors
        for i in range(len(output_level_formats) - 1, 0, -1):
            if (
                output_level_formats[i].get_level_type() == LevelType.DENSE
                and output_level_formats[i - 1].get_level_type() != LevelType.DENSE
            ):
                output_level_formats[i - 1] = LevelFormat(LevelType.DENSE)

        # For SDDMM-like patterns (sparse output with reduction variables
        # where an input tensor mirrors the output sparsity), use all-COO
        # output to enable the scalar-accum codegen path.  This gives
        # optimal loop order (reduction innermost), no workspace, and SIMD
        # vectorization of the dense reduction.
        _has_sparse_output = any(
            lf.get_level_type() in (LevelType.COMPRESSED, LevelType.COORDINATE)
            for lf in output_level_formats
        )
        if _has_sparse_output:
            _result_set = set(result_index_strs)
            _reduction_strs = [s for s in unique_index_strs if s not in _result_set]
            if _reduction_strs:
                # Check if any input tensor has the same index variables as
                # the output and contains sparse levels (SDDMM pattern).
                for inp_strs, tensor in zip(input_index_strs, tensors):
                    if set(inp_strs) == _result_set:
                        _inp_level_types = tensor.format.get_level_types()
                        if any(
                            lt in (LevelType.COMPRESSED, LevelType.COORDINATE)
                            for lt in _inp_level_types
                        ):
                            output_level_formats = [
                                LevelFormat(LevelType.COORDINATE)
                                for _ in output_level_formats
                            ]
                            break

        output_format = TensorFormat(output_level_formats)
        # print(f"\nUnspecified output format, using inferred {output_format}")
    result_index_sizes = _logical_index_sizes(input_index_strs, tensors)
    logical_result_shape = tuple(
        result_index_sizes[index_str] for index_str in result_index_strs
    )
    result_mode_order = temp_mode_order or list(range(len(logical_result_shape)))
    physical_result_shape = tuple(
        logical_result_shape[logical_mode] for logical_mode in result_mode_order
    )

    # Create the result TensorVar
    if output_tensor_dtype is None:
        raise CompileSpecError("einsum requires at least one typed operand")
    result_tensor_var = TensorVar(
        name=tensor_names_available.pop(0),
        fmt=output_format,
        shape=physical_result_shape,
        dtype=output_tensor_dtype,
        mode_order=temp_mode_order,
    )

    # Build RHS expression: product of all tensor accesses
    if not index_var_dict:
        raise CompileSpecError("einsum expression does not define any indices")
    rhs_expr = None
    for i, tensor_var in enumerate(tensor_vars):
        indices = [index_var_dict[s] for s in input_index_strs[i]]
        access = (
            tensor_var[indices[0]] if len(indices) == 1 else tensor_var[tuple(indices)]
        )
        rhs_expr = access if rhs_expr is None else rhs_expr * access

    # Build LHS access and create assignment
    lhs_indices = [index_var_dict[s] for s in result_index_strs]
    lhs_key = lhs_indices[0] if len(lhs_indices) == 1 else tuple(lhs_indices)
    result_tensor_var[lhs_key] = rhs_expr

    # Wrap in nested ForAll loops (outermost first in schedule, built inside-out)
    cin_stmt = result_tensor_var._assignment
    for index_str in reversed(index_strs_by_schedule):
        cin_stmt = ForAll(index_var_dict[index_str], cin_stmt)

    # print("CIN:\n", cin_stmt)

    # Align input tensor mode orders with the selected loop order to keep
    # parent-child level traversal valid during lowering for non-canonical
    # schedules.
    if effective_schedule is not None and effective_schedule.loop_order is not None:
        selected_loop_order = Scheduler.resolve_loop_order(
            cin_stmt, effective_schedule.loop_order
        )
    else:
        selected_loop_order = Scheduler.select_loop_order(cin_stmt)
    selected_loop_order_names = [index_var.name for index_var in selected_loop_order]
    for tensor_index, input_index_str in enumerate(input_index_strs):
        desired_index_strs = [
            index_str
            for index_str in selected_loop_order_names
            if index_str in input_index_str
        ]
        desired_mode_order = [
            input_index_str.index(index_str) for index_str in desired_index_strs
        ]
        if list(tensors[tensor_index].mode_order) != desired_mode_order:
            tensors[tensor_index] = _with_compiler_mode_order(
                tensors[tensor_index], desired_mode_order
            )
        tensor_vars[tensor_index].shape = tensors[tensor_index].shape
        if tensor_vars[tensor_index].mode_order != desired_mode_order:
            tensor_vars[tensor_index].mode_order = desired_mode_order
    tensors = tuple(tensors)

    # Extract PostOps for fused kernel compilation
    _post_ops = kwargs.get("_post_ops", None)
    _post_ops_tensors = kwargs.get("_post_ops_tensors", None)
    _cache_key_suffix = str(_post_ops) if _post_ops else ""

    # Phase 2b (codegen-parity): emit ONE kernel that branches at runtime on the
    # free-dim size — register-block for small N (single tile, no re-traversal), the
    # byte-identical baseline nest for large N. Default ON via `_regblock_dual_active()`
    # (post M5+x86 validation); `SCORCH_REGBLOCK_DUAL=0` restores the single path.
    # When the pattern doesn't qualify, `_build_regblock_dual_path` returns None and
    # the else-branch below emits the byte-identical baseline (unchanged from before).
    _dual_llir: Optional[llir.Function] = None
    if effective_schedule is None and _regblock_dual_active() and _post_ops is None:
        _dual = _build_regblock_dual_path(cin_stmt, _post_ops)
        if _dual is not None:
            _dual_llir, _kernel_cache_key = _dual
            _kernel_cache_key = _kernel_cache_key + _cache_key_suffix

    if _dual_llir is None:
        if effective_schedule is not None:
            cin_stmt = Scheduler.apply_schedule(cin_stmt, effective_schedule)
        # Default single path. When regblock is on but the dual-path wasn't
        # applicable, force it OFF here so we never ship the wide-k-regressing
        # single-path tiled kernel as a silent fallback.
        elif _regblock_enabled():
            with regblock_force(False):
                cin_stmt = Scheduler.auto_schedule(cin_stmt)
        else:
            cin_stmt = Scheduler.auto_schedule(cin_stmt)
        _kernel_cache_key = _codegen_kernel_cache_key(
            cin_stmt, _post_ops, effective_schedule
        )

    # print("Auto-scheduled CIN:\n", cin_stmt)

    if _kernel_cache_key in _kernel_cache:
        # print(f"Using cached kernel for {cin_stmt}")
        module = _kernel_cache[_kernel_cache_key]
    else:
        if _dual_llir is not None:
            lowered_llir: Union[llir.Stmt, List[llir.Stmt]] = _dual_llir
        else:
            lowerer = CINLowerer(post_ops=_post_ops)
            lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

        llir_lowerer = LLIRLowerer()

        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        # print("\n\n", cpp_code)

        header_cpp_code = jit_preamble_text()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )

        _kernel_cache[_kernel_cache_key] = module

    # Populate the dispatch cache so future calls skip scheduling entirely.
    if _dispatch_key is not None:
        _einsum_dispatch_cache[_dispatch_key] = (
            module,
            output_format,
            temp_mode_order,
            final_mode_order,
            [list(t.mode_order) for t in tensors],
            input_index_strs,
            result_index_strs,
        )

    if compile_only:
        result_mode_order = final_mode_order or list(range(len(result_index_strs)))
        return TensorSpec(
            output_format,
            logical_result_shape,
            dtype=output_tensor_dtype,
            mode_order=result_mode_order,
            index_dtype=torch.int32,
            name="einsum_result",
        )

    # Call module.evaluate with the output shape,and the mode indices and values of each tensor
    args: Sequence[Any] = [physical_result_shape]
    for tensor in tensors:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._native_mode_indices())  # type: ignore
        args.append(tensor.values)  # type: ignore

    # Append extra tensors for PostOps (bias, scale, etc.)
    if _post_ops_tensors:
        for extra_t in _post_ops_tensors:
            args.append(extra_t)  # type: ignore

    start_time = time.time()
    result_cpp = module.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    # print("Time taken for evaluate:", eval_time)

    result = STensor(
        shape=physical_result_shape,
        index=TensorIndex(
            mode_indices=_finalize_generated_mode_indices(
                output_format, result_cpp.storage.index.mode_indices
            ),
            tensor_format=output_format,
            mode_order=temp_mode_order if temp_mode_order else final_mode_order,
        ),
        value=result_cpp.storage.value,
    )

    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time

    # Convert to final mode order if it differs from temporary mode order
    if temp_mode_order:
        result.change_mode_order(final_mode_order)

    return result


def _align_mode_orders_to_loop_order(cin_stmt: IndexStmt, args: tuple) -> None:
    """Align input tensor mode orders to the CIN loop order.

    The lowerer requires parent physical levels to be iterated before child
    levels.  When a tensor's mode_order doesn't match the loop nesting, the
    generated code references coordinate variables before they are defined.
    This mirrors the alignment that ``einsum`` performs (ops.py L581-591).

    Mutates TensorVar.mode_order in *cin_stmt* and calls
    ``STensor.change_mode_order`` on the corresponding *args* entries.
    """
    # 1. Extract loop order
    loop_order_names: List[str] = []
    curr: IndexStmt = cin_stmt
    while isinstance(curr, ForAll):
        loop_order_names.append(curr.index_var.name)
        curr = curr.stmt

    if not loop_order_names:
        return

    # 2. Get RHS tensor accesses (left-to-right order matches *args*)
    rhs_accesses = cin_stmt.get_rhs_tensor_accesses()
    if len(rhs_accesses) != len(args):
        return  # can't align if we don't have a 1:1 mapping

    for ta, stensor in zip(rhs_accesses, args):
        tv = ta.get_tensor()
        index_var_names = [iv.name for iv in ta.get_index_vars()]
        # Filter loop order to vars present in this tensor
        desired_names = [n for n in loop_order_names if n in index_var_names]
        desired_mode_order = [index_var_names.index(n) for n in desired_names]

        # Skip when tiling/broadcasting causes a rank mismatch
        if len(desired_mode_order) != len(tv.mode_order):
            continue
        if list(tv.mode_order) != desired_mode_order:
            tv.mode_order = desired_mode_order
            if stensor.has_index and stensor.shape is not None:
                stensor.change_mode_order(desired_mode_order)

    # 3. Also align the output tensor
    if isinstance(curr, TensorAssign):
        lhs_tv = curr.lhs.get_tensor()
        lhs_names = [iv.name for iv in curr.lhs.get_index_vars()]
        desired_names = [n for n in loop_order_names if n in lhs_names]
        desired_mode_order = [lhs_names.index(n) for n in desired_names]
        if (
            len(desired_mode_order) == len(lhs_tv.mode_order)
            and list(lhs_tv.mode_order) != desired_mode_order
        ):
            lhs_tv.mode_order = desired_mode_order


def lower_and_exec_cin(
    cin_stmt: IndexStmt, result_shape: Sequence[int], *args: STensor, **kwargs
) -> STensor:
    """Lower a CIN statement to LLIR then codegen and call on the input tensors.

    Low-level entry to the generic compiler path: aligns the operand mode orders to
    the CIN loop order (``_align_mode_orders_to_loop_order``), lowers CIN -> LLIR ->
    C++, JIT-loads the kernel, executes it on ``args``, and wraps the result as an
    ``STensor``. This is the primitive the compiler gap tests use to drive the
    pipeline directly; it is **not exported** — advanced users reach it as
    ``scorch.ops.lower_and_exec_cin``. Most code should use :func:`matmul` or
    :func:`einsum` instead.

    Parameters
    ----------
    cin_stmt : IndexStmt
        CIN statement (index-notation AST) to lower.
    result_shape : Sequence[int]
        Shape of the result tensor.
    *args : STensor
        Input tensors, matched left-to-right against the RHS tensor accesses.
    **kwargs
        ``time_dict`` : dict, optional -- if given, ``time_dict["eval_time"]`` is
        set to the kernel wall-clock time in seconds.

    Returns
    -------
    STensor
        Output tensor. The output format is hard-coded to ``"dd"`` (dense).
    """
    _align_mode_orders_to_loop_order(cin_stmt, args)

    rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
    if len(rhs_tensor_vars) != len(args):
        raise CompileSpecError(
            f"CIN expects {len(rhs_tensor_vars)} runtime tensors, got {len(args)}"
        )
    for tensor_var, arg in zip(rhs_tensor_vars, args):
        if tensor_var.format != arg.format:
            raise CompileSpecError(
                f"CIN tensor {tensor_var.name!r} expects format "
                f"{tensor_var.format}, got {arg.format}"
            )
        tensor_var.shape = arg.shape
        tensor_var.dtype = arg.dtype
        tensor_var.mode_order = list(arg.mode_order)

    output_dtype = args[0].dtype if args else torch.float32
    for tensor_var in cin_stmt.get_result_tensor_vars():
        if not isinstance(tensor_var, Workspace):
            tensor_var.shape = tuple(result_shape)
            tensor_var.dtype = output_dtype

    # Lower to LLIR
    lowerer = CINLowerer()
    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
    llir_lowerer = LLIRLowerer()
    cpp_code = llir_lowerer.lower_llir(lowered_llir)
    # print(cpp_code)
    header_cpp_code = jit_preamble_text()

    module = _load_kernel(
        name=_kernel_name(header_cpp_code, cpp_code),
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=get_extra_cflags(),
        extra_ldflags=get_extra_ldflags(),
    )

    module_args: List[Any] = [result_shape]

    for arg in args:
        module_args.append(arg.shape)
        module_args.append(arg._native_mode_indices())
        module_args.append(arg.values)

    start_time = time.time()
    result_cpp = module.evaluate(*module_args)
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        kwargs["time_dict"]["eval_time"] = eval_time

    result = STensor(
        shape=tuple(result_shape),
        index=TensorIndex(
            mode_indices=_finalize_generated_mode_indices(
                parse_format("dd"), result_cpp.storage.index.mode_indices
            ),
            tensor_format="dd",
        ),
        value=result_cpp.storage.value,
    )

    return result


def precompile_kernels():
    """Warm the JIT cache by compiling the common SpMM / SpGEMM kernels.

    Calls :func:`einsum` with ``compile_only=True`` for the frequently used
    contraction/format combinations so the first real product does not pay the
    schedule + codegen + C++ compile cost:

    - ``ds @ dd -> dd``  (CSR x dense -> dense)
    - ``oo @ dd -> dd``  (COO x dense -> dense)
    - ``oo @ ds -> dd``  (COO x CSR -> dense)
    - ``ds @ ds -> dd``  (CSR x CSR -> dense)
    - ``ds @ ds -> ds``  (CSR x CSR -> CSR / SpGEMM)

    Prints ``"Precompiled kernels."`` when done. The call at module import in
    ``__init__.py`` is **commented out**, so this does not run automatically —
    invoke ``scorch.precompile_kernels()`` explicitly to warm up.

    Returns
    -------
    None

    Examples
    --------
    >>> import scorch
    >>> scorch.precompile_kernels()   # doctest: +SKIP
    Precompiled kernels.
    """
    # Extents are irrelevant to generated code; a real, immutable rank-2 spec
    # replaces the former format-only (and invalid) STensor placeholders.
    DS = TensorSpec("ds", (1, 1), name="csr_spec")
    DD = TensorSpec("dd", (1, 1), name="dense_spec")
    OO = TensorSpec("oo", (1, 1), name="coo_spec")

    einsum("ik,kj->ij", DS, DD, compile_only=True, format="dd")
    einsum("ik,kj->ij", OO, DD, compile_only=True, format="dd")
    einsum("ik,kj->ij", OO, DS, compile_only=True, format="dd")
    einsum("ik,kj->ij", DS, DS, compile_only=True, format="dd")
    einsum("ik,kj->ij", DS, DS, compile_only=True, format="ds")

    print("Precompiled kernels.")
