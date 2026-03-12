"""@scorch.compile — torch.fx-based kernel fusion decorator.

Traces a user function with torch.fx, identifies fusible contraction +
elementwise chains, and dispatches to prebuilt C++ kernels or JIT-compiled
fused kernels.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx

from . import ops
from .compiler.cin import PostOp, PostOps
from .stensor import STensor


# ---------------------------------------------------------------------------
# Custom torch.fx Tracer that treats scorch.matmul as a leaf
# ---------------------------------------------------------------------------

class _ScorchTracer(torch.fx.Tracer):
    """Tracer that records scorch.matmul as an opaque call_function node."""

    _LEAF_FUNCTIONS = frozenset()

    def __init__(self, leaf_functions=None):
        super().__init__()
        if leaf_functions:
            self._LEAF_FUNCTIONS = frozenset(leaf_functions)

    def create_arg(self, a: Any) -> torch.fx.Node:
        # Let string/int/float kwargs pass through as-is
        return super().create_arg(a)


def _symbolic_trace(fn: Callable) -> torch.fx.GraphModule:
    """Trace *fn* with scorch.matmul as a leaf node.

    We temporarily replace scorch.matmul (in all dicts that the function
    might access) with a thin proxy-aware wrapper.  When torch.fx calls the
    function with Proxy arguments, the wrapper creates a ``call_function``
    node targeting the *original* ``ops.matmul`` instead of tracing into it.
    """
    import scorch as _scorch_mod

    _original_matmul = ops.matmul
    _sentinel = object()

    # Dicts where the function might look up ``matmul``
    _dicts_to_patch: List[Tuple[dict, str, Any]] = []

    # 1. scorch package namespace  (scorch.matmul)
    _prev = _scorch_mod.__dict__.get("matmul", _sentinel)
    _dicts_to_patch.append((_scorch_mod.__dict__, "matmul", _prev))

    # 2. ops module namespace  (scorch.ops.matmul)
    _prev_ops = ops.__dict__.get("matmul", _sentinel)
    _dicts_to_patch.append((ops.__dict__, "matmul", _prev_ops))

    # 3. The traced function's own globals  (from scorch import matmul)
    fn_globals = getattr(fn, "__globals__", {})
    if "matmul" in fn_globals and fn_globals["matmul"] is _original_matmul:
        _dicts_to_patch.append((fn_globals, "matmul", fn_globals["matmul"]))

    # Build a thin wrapper that creates proxy nodes
    def _proxy_matmul(*args, **kwargs):
        # During tracing, at least one arg will be a torch.fx.Proxy
        from torch.fx import Proxy
        proxy = None
        for a in args:
            if isinstance(a, Proxy):
                proxy = a
                break

        if proxy is not None:
            # Record a call_function node targeting the ORIGINAL ops.matmul
            return proxy.tracer.create_proxy(
                "call_function", _original_matmul, args, kwargs
            )
        # Outside tracing, call the real function
        return _original_matmul(*args, **kwargs)

    # Preserve identity for later analysis
    _proxy_matmul._scorch_original = _original_matmul

    # Patch
    for d, key, _ in _dicts_to_patch:
        d[key] = _proxy_matmul

    try:
        tracer = _ScorchTracer()
        graph = tracer.trace(fn)
        graph_module = torch.fx.GraphModule(tracer.root, graph)
    finally:
        # Restore originals
        for d, key, prev in _dicts_to_patch:
            if prev is _sentinel:
                d.pop(key, None)
            else:
                d[key] = prev

    return graph_module


# ---------------------------------------------------------------------------
# Elementwise op mapping: torch.fx node target -> PostOp kind
# ---------------------------------------------------------------------------
_ELEMENTWISE_OPS: Dict[Any, str] = {
    operator.add: "add",
    operator.mul: "mul",
    torch.relu: "relu",
    torch.nn.functional.relu: "relu",
    torch.sigmoid: "sigmoid",
    torch.tanh: "tanh",
}


# ---------------------------------------------------------------------------
# FusionSpec: result of FX graph analysis
# ---------------------------------------------------------------------------
@dataclass
class FusionSpec:
    matmul_node: torch.fx.Node
    matmul_arg_indices: List[int]       # which function args feed the matmul
    matmul_kwargs: dict                 # e.g. {"format": "dd"}
    post_ops: PostOps
    extra_arg_indices: List[int]        # function arg indices for postop operands
    output_format: Optional[str] = None


# ---------------------------------------------------------------------------
# FX graph analysis
# ---------------------------------------------------------------------------

def _trace_to_placeholder(node: torch.fx.Node, placeholders: List[torch.fx.Node]) -> Optional[int]:
    """If node is a placeholder, return its index. Otherwise None."""
    if node in placeholders:
        return placeholders.index(node)
    return None


def analyze_fx_graph(graph: torch.fx.Graph, real_args: tuple) -> FusionSpec:
    """Walk FX graph: find matmul node, collect elementwise postop chain."""
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]

    # 1. Find the scorch.matmul call_function node
    matmul_node = None
    for node in graph.nodes:
        if node.op == "call_function" and node.target is ops.matmul:
            matmul_node = node
            break

    if matmul_node is None:
        raise ValueError("No scorch.matmul found in traced function")

    # Map matmul positional args back to placeholder indices
    matmul_arg_indices = []
    for arg in matmul_node.args:
        if isinstance(arg, torch.fx.Node):
            idx = _trace_to_placeholder(arg, placeholders)
            if idx is not None:
                matmul_arg_indices.append(idx)

    matmul_kwargs = dict(matmul_node.kwargs)

    # 2. Follow the consumer chain through elementwise ops
    post_op_list: List[PostOp] = []
    extra_arg_indices: List[int] = []
    extra_tensor_names: List[str] = []

    current = matmul_node
    while True:
        users = list(current.users.keys())
        if len(users) != 1:
            break
        next_node = users[0]

        if next_node.op == "output":
            break

        if next_node.op == "call_function" and next_node.target in _ELEMENTWISE_OPS:
            kind = _ELEMENTWISE_OPS[next_node.target]

            if kind in ("add", "mul"):
                # Find which arg is the chain and which is the extra operand
                args = next_node.args
                if len(args) == 2:
                    if args[0] is current:
                        extra_node = args[1]
                    else:
                        extra_node = args[0]

                    extra_idx = _trace_to_placeholder(extra_node, placeholders)
                    if extra_idx is not None:
                        tensor_name = f"postop_{len(extra_tensor_names)}"
                        extra_arg_indices.append(extra_idx)
                        extra_tensor_names.append(tensor_name)
                        post_op_list.append(PostOp(kind=kind, tensor_name=tensor_name))
                    else:
                        break
                else:
                    break
            else:
                # Unary ops: relu, sigmoid, tanh
                post_op_list.append(PostOp(kind=kind))

            current = next_node
        else:
            break

    post_ops = PostOps(ops=post_op_list, extra_tensors=extra_tensor_names)
    output_format = matmul_kwargs.get("format", None)

    return FusionSpec(
        matmul_node=matmul_node,
        matmul_arg_indices=matmul_arg_indices,
        matmul_kwargs=matmul_kwargs,
        post_ops=post_ops,
        extra_arg_indices=extra_arg_indices,
        output_format=output_format,
    )


# ---------------------------------------------------------------------------
# Compilation dispatch
# ---------------------------------------------------------------------------

def _try_prebuilt_fused(spec: FusionSpec, args: tuple) -> Optional[Callable]:
    """Try to match against prebuilt fused C++ kernels."""
    from .prebuilt_kernels import resolve_prebuilt_fused

    if len(spec.matmul_arg_indices) != 2:
        return None

    a = args[spec.matmul_arg_indices[0]]
    b = args[spec.matmul_arg_indices[1]]

    # Get format strings
    if isinstance(a, STensor):
        a_format = str(a.format)
    elif isinstance(a, torch.Tensor):
        return None  # prebuilt only handles STensor inputs
    else:
        return None

    if isinstance(b, STensor):
        b_format = str(b.format)
    elif isinstance(b, torch.Tensor):
        # Dense torch.Tensor
        b_format = ",".join(["d"] * b.dim())
    else:
        return None

    post_op_kinds = tuple(op.kind for op in spec.post_ops.ops)
    dtype = a.values.dtype if isinstance(a, STensor) else a.dtype

    resolved = resolve_prebuilt_fused(a_format, b_format, post_op_kinds, dtype)
    if resolved is None:
        return None

    kernel_fn = resolved.fn
    matmul_arg_indices = spec.matmul_arg_indices
    extra_arg_indices = spec.extra_arg_indices

    def _prebuilt_runner(call_args: tuple) -> torch.Tensor:
        a_arg = call_args[matmul_arg_indices[0]]
        b_arg = call_args[matmul_arg_indices[1]]

        # Convert torch.Tensor to STensor if needed
        if isinstance(a_arg, torch.Tensor) and not isinstance(a_arg, STensor):
            a_arg = STensor.from_torch(a_arg)
        if isinstance(b_arg, torch.Tensor) and not isinstance(b_arg, STensor):
            b_arg = STensor.from_torch(b_arg)

        # For CSR SpMM: extract CSR structure
        N = a_arg.shape[0]
        K = b_arg.shape[1] if b_arg.dim() == 2 else 1
        result_shape = [N, K]

        # Collect extra tensors (bias, scale, etc.)
        extra_tensors = [call_args[i] for i in extra_arg_indices]
        assert len(extra_tensors) >= 1, "Prebuilt fused kernel requires at least one extra tensor (e.g. bias)"
        bias = extra_tensors[0]

        result_cpp = kernel_fn(
            result_shape,
            list(a_arg.shape),
            a_arg.index.mode_indices,
            a_arg.values,
            list(b_arg.shape),
            b_arg.index.mode_indices if b_arg.has_index else [[], []],
            b_arg.values if isinstance(b_arg, STensor) else b_arg.contiguous().view(-1),
            bias,
        )
        return result_cpp.storage.value.view(N, K)

    return _prebuilt_runner


def _jit_compile_fused(spec: FusionSpec, args: tuple) -> Callable:
    """JIT compile a fused kernel via CIN + PostOps.

    Runs the matmul via einsum, then applies post-ops as separate torch
    operations.  In-kernel fusion (eliminating extra memory passes) is
    handled by the prebuilt path for CSR; the JIT path provides
    correctness for all other format combinations.
    """
    matmul_arg_indices = spec.matmul_arg_indices
    extra_arg_indices = spec.extra_arg_indices
    post_ops = spec.post_ops
    matmul_kwargs = {k: v for k, v in spec.matmul_kwargs.items()
                     if not k.startswith("_")}

    _extra_indices = list(extra_arg_indices)

    # Map PostOp kinds to torch ops
    _UNARY_OPS = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }

    def _jit_runner(call_args: tuple) -> torch.Tensor:
        a = call_args[matmul_arg_indices[0]]
        b = call_args[matmul_arg_indices[1]]

        if isinstance(a, torch.Tensor) and not isinstance(a, STensor):
            a = STensor.from_torch(a)
        if isinstance(b, torch.Tensor) and not isinstance(b, STensor):
            b = STensor.from_torch(b)

        extra_tensors = [call_args[i] for i in _extra_indices]

        # Step 1: Run the matmul
        result = ops.einsum("ij,jk->ik", a, b, **matmul_kwargs)

        if isinstance(result, STensor) and result.format.is_dense():
            result = result.to_torch()

        # Step 2: Apply post-ops as torch operations
        extra_idx = 0
        for op in post_ops.ops:
            if op.kind == "add":
                result = result + extra_tensors[extra_idx]
                extra_idx += 1
            elif op.kind == "mul":
                result = result * extra_tensors[extra_idx]
                extra_idx += 1
            elif op.kind in _UNARY_OPS:
                result = _UNARY_OPS[op.kind](result)

        return result

    return _jit_runner


def compile_fused(spec: FusionSpec, args: tuple) -> Callable:
    """Try prebuilt dispatch, fall back to JIT."""
    kernel = _try_prebuilt_fused(spec, args)
    if kernel is not None:
        return kernel
    return _jit_compile_fused(spec, args)


# ---------------------------------------------------------------------------
# @scorch.compile decorator
# ---------------------------------------------------------------------------

class compile:
    """Decorator that traces a function with torch.fx and compiles fused kernels."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self._fx_graph: Optional[torch.fx.GraphModule] = None
        self._cache: Dict[tuple, Callable] = {}

    def __call__(self, *args: Any) -> Any:
        # Trace once
        if self._fx_graph is None:
            self._fx_graph = _symbolic_trace(self.fn)

        # Build cache key from input signatures
        cache_key = self._build_cache_key(args)
        if cache_key not in self._cache:
            spec = analyze_fx_graph(self._fx_graph.graph, args)
            self._cache[cache_key] = compile_fused(spec, args)

        return self._cache[cache_key](args)

    @staticmethod
    def _build_cache_key(args: tuple) -> tuple:
        parts = []
        for a in args:
            if isinstance(a, STensor):
                parts.append((str(a.format), a.dtype))
            elif isinstance(a, torch.Tensor):
                parts.append((f"dense_{a.dim()}d", a.dtype))
            else:
                parts.append(type(a))
        return tuple(parts)
