"""Semantics-preserving torch.fx subgraph fusion for ``scorch.compile``."""

from __future__ import annotations

import copy
import operator
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.fx

from . import ops
from .compiler.cin import PostOp, PostOps
from .stensor import STensor
from .utils import parse_format

_TRACE_LOCK = threading.RLock()


def _symbolic_trace(fn: Callable) -> torch.fx.GraphModule:
    """Trace *fn* with scorch.matmul as a leaf node.

    ``ops.matmul`` handles FX proxies directly, so tracing does not mutate the
    ``scorch`` package, ``scorch.ops``, or the user's globals.  Serializing FX
    traces also avoids overlapping torch.fx's own temporary tracer hooks when
    multiple compiled functions initialize concurrently.
    """
    with _TRACE_LOCK:
        return torch.fx.symbolic_trace(fn)


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
class FusionStep:
    """One proven elementwise step in a fusible chain."""

    kind: str
    target: Callable
    chain_arg_index: int
    extra_arg_index: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionSpec:
    matmul_node: torch.fx.Node
    matmul_arg_indices: List[int]  # which function args feed the matmul
    matmul_kwargs: dict  # e.g. {"format": "dd"}
    post_ops: PostOps
    extra_arg_indices: List[int]  # function arg indices for postop operands
    output_format: Optional[Any] = None
    output_node: Optional[torch.fx.Node] = None
    fused_nodes: List[torch.fx.Node] = field(default_factory=list)
    steps: List[FusionStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FX graph analysis
# ---------------------------------------------------------------------------


def _trace_to_placeholder(
    node: Any, placeholders: List[torch.fx.Node]
) -> Optional[int]:
    """If node is a placeholder, return its index. Otherwise None."""
    if node in placeholders:
        return placeholders.index(node)
    return None


def _contains_fx_node(value: Any) -> bool:
    """Return whether a nested FX argument contains a graph node."""
    if isinstance(value, torch.fx.Node):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_fx_node(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_fx_node(item) for item in value.values())
    return False


def analyze_fx_graph(graph: torch.fx.Graph, real_args: tuple) -> FusionSpec:
    """Find one structurally proven matmul/post-op fusion subgraph.

    The proof is deliberately conservative: the graph must contain exactly one
    ``scorch.matmul``; both operands and every binary post-op operand must be
    direct placeholders; and accepted nodes must be contiguous.  Anything that
    does not satisfy those conditions is left to eager execution.
    """
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    ordered_nodes = list(graph.nodes)
    node_positions = {node: index for index, node in enumerate(ordered_nodes)}

    matmul_nodes = [
        node
        for node in ordered_nodes
        if node.op == "call_function" and node.target is ops.matmul
    ]
    if len(matmul_nodes) != 1:
        raise ValueError("Expected exactly one scorch.matmul in a fusible FX subgraph")
    matmul_node = matmul_nodes[0]

    # The replacement runner accepts the original function arguments.  Do not
    # attempt to reproduce computed operands or dynamic keyword arguments.
    matmul_kwargs = dict(matmul_node.kwargs)
    if len(matmul_node.args) != 2 or _contains_fx_node(matmul_kwargs):
        raise ValueError("scorch.matmul operands must be direct function arguments")
    if set(matmul_kwargs) - {"format", "output_format"} or any(
        value is not None and not isinstance(value, str)
        for value in matmul_kwargs.values()
    ):
        raise ValueError("scorch.matmul kwargs are not proven fusible")

    matmul_arg_indices: List[int] = []
    for arg in matmul_node.args:
        idx = _trace_to_placeholder(arg, placeholders)
        if idx is None or idx >= len(real_args):
            raise ValueError("scorch.matmul operands must be direct function arguments")
        matmul_arg_indices.append(idx)

    post_op_list: List[PostOp] = []
    extra_arg_indices: List[int] = []
    extra_tensor_names: List[str] = []
    fused_nodes = [matmul_node]
    steps: List[FusionStep] = []

    current = matmul_node
    while True:
        users = list(current.users.keys())
        if len(users) != 1:
            break
        next_node = users[0]

        if next_node.op == "output":
            break

        # Without alias/effect analysis, moving a contraction across an
        # interleaved node could observe mutated inputs.  Fuse only adjacent
        # nodes so the rewrite cannot cross any other graph operation.
        if node_positions[next_node] != node_positions[current] + 1:
            break
        if next_node.op != "call_function" or next_node.target not in _ELEMENTWISE_OPS:
            break
        if _contains_fx_node(next_node.kwargs):
            break

        kind = _ELEMENTWISE_OPS[next_node.target]
        step_kwargs = dict(next_node.kwargs)
        if kind in ("add", "mul"):
            node_args = next_node.args
            matching_positions = [
                index for index, arg in enumerate(node_args) if arg is current
            ]
            if len(node_args) != 2 or len(matching_positions) != 1:
                break

            chain_arg_index = matching_positions[0]
            extra_node = node_args[1 - chain_arg_index]
            extra_idx = _trace_to_placeholder(extra_node, placeholders)
            if extra_idx is None or extra_idx >= len(real_args):
                break

            tensor_name = f"postop_{len(extra_tensor_names)}"
            extra_arg_indices.append(extra_idx)
            extra_tensor_names.append(tensor_name)
            post_op_list.append(PostOp(kind=kind, tensor_name=tensor_name))
            steps.append(
                FusionStep(
                    kind=kind,
                    target=next_node.target,
                    chain_arg_index=chain_arg_index,
                    extra_arg_index=extra_idx,
                    kwargs=step_kwargs,
                )
            )
        else:
            if len(next_node.args) != 1 or next_node.args[0] is not current:
                break
            post_op_list.append(PostOp(kind=kind))
            steps.append(
                FusionStep(
                    kind=kind,
                    target=next_node.target,
                    chain_arg_index=0,
                    kwargs=step_kwargs,
                )
            )

        fused_nodes.append(next_node)
        current = next_node

    # Replacing a bare matmul has no fusion benefit.  More importantly, this
    # makes unsupported first consumers (sin, branches, etc.) take the untouched
    # eager path instead of treating an analyzed prefix as the whole function.
    if not steps:
        raise ValueError("No proven fusible post-op chain found")

    post_ops = PostOps(ops=post_op_list, extra_tensors=extra_tensor_names)
    output_format = matmul_kwargs.get(
        "format", matmul_kwargs.get("output_format", None)
    )

    return FusionSpec(
        matmul_node=matmul_node,
        matmul_arg_indices=matmul_arg_indices,
        matmul_kwargs=matmul_kwargs,
        post_ops=post_ops,
        extra_arg_indices=extra_arg_indices,
        output_format=output_format,
        output_node=current,
        fused_nodes=fused_nodes,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Compilation dispatch
# ---------------------------------------------------------------------------


def _prebuilt_inputs(
    spec: FusionSpec, args: tuple
) -> Optional[tuple[STensor, STensor, torch.Tensor]]:
    """Validate every assumption made by the native fused SpMM kernels."""
    if len(spec.matmul_arg_indices) != 2 or len(spec.extra_arg_indices) != 1:
        return None

    # The native wrapper does not implement schedules, timing side effects, or
    # alternate output layouts.  Only the default/dense format call is exact.
    if set(spec.matmul_kwargs) - {"format"}:
        return None
    requested_format = spec.matmul_kwargs.get("format")
    if requested_format is not None:
        try:
            if str(parse_format(requested_format)) != "d,d":
                return None
        except (AssertionError, TypeError, ValueError):
            return None

    if any(step.chain_arg_index != 0 or step.kwargs for step in spec.steps):
        return None

    a = args[spec.matmul_arg_indices[0]]
    b = args[spec.matmul_arg_indices[1]]
    bias = args[spec.extra_arg_indices[0]]

    if type(a) is not STensor or a.dim() != 2 or str(a.format) != "d,s":
        return None
    if a.storage.index.mode_order != [0, 1]:
        return None
    if a.values.device.type != "cpu" or not a.values.is_contiguous():
        return None

    if type(b) is torch.Tensor:
        if (
            b.layout != torch.strided
            or b.dim() != 2
            or b.device.type != "cpu"
            or not b.is_contiguous()
        ):
            return None
        b_dtype = b.dtype
        b_requires_grad = b.requires_grad
    elif type(b) is STensor:
        if (
            b.dim() != 2
            or str(b.format) != "d,d"
            or b.storage.index.mode_order != [0, 1]
            or b.values.device.type != "cpu"
            or not b.values.is_contiguous()
        ):
            return None
        b_dtype = b.values.dtype
        b_requires_grad = b.values.requires_grad
    else:
        return None

    if (
        type(bias) is not torch.Tensor
        or bias.layout != torch.strided
        or bias.device.type != "cpu"
        or bias.dim() != 1
        or not bias.is_contiguous()
    ):
        return None
    if a.shape[1] != b.shape[0] or bias.shape[0] != b.shape[1]:
        return None
    if not (a.values.dtype == b_dtype == bias.dtype == torch.float32):
        return None
    if torch.is_grad_enabled() and (
        a.values.requires_grad or b_requires_grad or bias.requires_grad
    ):
        return None

    b_values = b.values if isinstance(b, STensor) else b
    if any(
        tensor.is_neg() or tensor.is_conj() for tensor in (a.values, b_values, bias)
    ):
        return None
    if torch.autograd.forward_ad._current_level >= 0:
        if any(
            torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
            for tensor in (a.values, b_values, bias)
        ):
            return None
    b_stensor = b if isinstance(b, STensor) else STensor.from_torch(b)
    return a, b_stensor, bias


def _jit_compile_fused(spec: FusionSpec, args: tuple) -> Callable:
    """Build an exact eager equivalent for a proven fusion subgraph.

    This is the correctness fallback when no native fused kernel applies.  It
    deliberately calls the original matmul and exact FX post-op targets instead
    of changing ranks, formats, autograd behavior, operand order, or kwargs.
    """
    del args
    matmul_arg_indices = list(spec.matmul_arg_indices)
    matmul_kwargs = dict(spec.matmul_kwargs)
    steps = list(spec.steps)

    def _jit_runner(call_args: tuple) -> Any:
        a = call_args[matmul_arg_indices[0]]
        b = call_args[matmul_arg_indices[1]]
        result = ops.matmul(a, b, **matmul_kwargs)

        for step in steps:
            if step.extra_arg_index is None:
                result = step.target(result, **step.kwargs)
                continue

            extra = call_args[step.extra_arg_index]
            step_args = (
                (result, extra) if step.chain_arg_index == 0 else (extra, result)
            )
            result = step.target(*step_args, **step.kwargs)

        return result

    return _jit_runner


def _try_prebuilt_fused(spec: FusionSpec, args: tuple) -> Optional[Callable]:
    """Try a native fused kernel, guarded by an eager-equivalent fallback."""
    from .prebuilt_kernels import resolve_prebuilt_fused

    prepared = _prebuilt_inputs(spec, args)
    if prepared is None:
        return None

    a, b, _ = prepared
    post_op_kinds = tuple(op.kind for op in spec.post_ops.ops)
    resolved = resolve_prebuilt_fused(
        str(a.format), str(b.format), post_op_kinds, a.values.dtype
    )
    if resolved is None:
        return None

    fallback = _jit_compile_fused(spec, args)
    kernel_fn = resolved.fn

    def _prebuilt_runner(call_args: tuple) -> Any:
        try:
            runtime_inputs = _prebuilt_inputs(spec, call_args)
        except Exception:
            return fallback(call_args)
        if runtime_inputs is None:
            return fallback(call_args)

        a_arg, b_arg, bias = runtime_inputs
        n = a_arg.shape[0]
        k = b_arg.shape[1]
        result_shape = [n, k]

        try:
            result_cpp = kernel_fn(
                result_shape,
                list(a_arg.shape),
                a_arg._native_mode_indices(),
                a_arg.values,
                list(b_arg.shape),
                b_arg._native_mode_indices(),
                b_arg.values,
                bias,
            )
            return result_cpp.storage.value.view(n, k)
        except Exception:
            # A failed optimization must not turn a supported eager graph into
            # a compile-only failure.
            return fallback(call_args)

    return _prebuilt_runner


def compile_fused(spec: FusionSpec, args: tuple) -> Callable:
    """Try guarded native fusion, then use the exact eager equivalent."""
    step_kinds = tuple(step.kind for step in spec.steps)
    postop_kinds = tuple(op.kind for op in spec.post_ops.ops)
    step_extras = [
        step.extra_arg_index for step in spec.steps if step.extra_arg_index is not None
    ]
    if (
        not spec.steps
        or step_kinds != postop_kinds
        or step_extras != spec.extra_arg_indices
        or any(_ELEMENTWISE_OPS.get(step.target) != step.kind for step in spec.steps)
    ):
        raise ValueError("FusionSpec does not describe one proven post-op chain")
    kernel = _try_prebuilt_fused(spec, args)
    if kernel is not None:
        return kernel
    return _jit_compile_fused(spec, args)


class _CompiledSubgraph(torch.nn.Module):
    """FX-callable adapter around a signature-specific fused runner."""

    def __init__(self, runner: Callable):
        super().__init__()
        self.runner = runner

    def forward(self, *args: Any) -> Any:
        return self.runner(args)


def _rewrite_fused_subgraph(
    graph_module: torch.fx.GraphModule,
    spec: FusionSpec,
    runner: Callable,
) -> torch.fx.GraphModule:
    """Replace only ``spec`` inside a complete copied FX graph."""
    output_node = spec.output_node
    if output_node is None or not spec.fused_nodes:
        raise ValueError("FusionSpec does not identify a complete subgraph")

    module_name = "_scorch_fused"
    suffix = 0
    while hasattr(graph_module, module_name):
        suffix += 1
        module_name = f"_scorch_fused_{suffix}"
    graph_module.add_module(module_name, _CompiledSubgraph(runner))

    graph = graph_module.graph
    placeholders = tuple(node for node in graph.nodes if node.op == "placeholder")
    with graph.inserting_after(output_node):
        fused_node = graph.call_module(module_name, args=placeholders)
    fused_node.meta = dict(output_node.meta)
    output_node.replace_all_uses_with(fused_node)

    for node in reversed(spec.fused_nodes):
        graph.erase_node(node)

    graph.lint()
    graph_module.recompile()
    return graph_module


# ---------------------------------------------------------------------------
# @scorch.compile decorator
# ---------------------------------------------------------------------------


class compile:
    """Trace a function with torch.fx and fuse its contraction + elementwise chain.

    ``@scorch.compile`` is a *function-level* fusion decorator, distinct from
    scorch's per-op sparse compiler. It uses ``torch.fx`` to symbolically trace
    the decorated function, locates a single ``scorch.matmul`` contraction
    followed by a linear chain of supported elementwise "post-ops" (bias add,
    scale, activation), and replaces only that proven subgraph in a copy of the
    complete FX graph. The complete graph always executes, so unsupported
    suffixes, branches, reused values, and structured outputs are preserved. If
    tracing or the fusion proof fails, the original function runs eagerly.

    Usable bare (``@scorch.compile``) or parameterized with a per-call autotune
    level escape hatch (``@scorch.compile(autotune="max")``); the traced function
    then runs inside that autotune scope (thread-local, like ``with
    scorch.autotune(level)``). The autotune level tunes the SpMM tiling selector
    (see ``scorch.set_autotune``); it does not change the fused kernel's numerics.

    Parameters
    ----------
    fn : Callable, optional
        The function to wrap. Supplied positionally in the bare form
        ``@scorch.compile``. When ``None`` (the parameterized form
        ``@scorch.compile(autotune=...)``), the instance is left pending and
        binds the function on the next call.
    autotune : str, optional, keyword-only
        Autotune level forwarded to ``scorch.tiling.autotune(...)`` as a
        thread-local scope around every execution. Accepts the same level
        strings as the autotune ladder (e.g. ``"off"``, ``"analytic"``,
        ``"balanced"``, ``"max"``, ``"learned"``; see ``scorch.set_autotune``).
        When ``None``, no autotune scope is entered. Numerics-neutral: it only
        tunes the SpMM tiling selector, never the fused output values.

    Returns
    -------
    compile
        A callable instance that preserves the wrapped function's return value.

    Raises
    ------
    TypeError
        If the parameterized form ``@scorch.compile(autotune=...)`` is applied
        to something that is not a single callable.

    Notes
    -----
    **What it can fuse.** Exactly one ``scorch.matmul(...)`` call (spelled as
    ``scorch.matmul``, ``scorch.ops.matmul``, or an imported alias) followed by
    a contiguous, single-consumer chain of these elementwise ops:

    - bias/scale written with the Python ``+`` / ``*`` operators (they trace to
      ``operator.add`` / ``operator.mul`` — ``torch.add`` / ``torch.mul`` are
      *not* recognized);
    - activations ``torch.relu`` / ``torch.nn.functional.relu``,
      ``torch.sigmoid``, ``torch.tanh``.

    Both matmul operands and the other operand of a fused ``+`` / ``*`` must be
    direct function arguments (FX placeholders). The chain stops at fan-out,
    computed operands, interleaved nodes, or unsupported operations. Those nodes
    are not dropped: the rewritten full graph executes them normally. If no
    non-empty chain is proven, the original function runs eagerly. Calls using
    keyword arguments also use the eager path.

    **Prebuilt vs eager-equivalent dispatch.** Two strictly guarded patterns hit
    a hand-written C++ kernel: canonical CPU float32 CSR (``"d,s"``) LHS × dense
    RHS with a vector bias add, optionally followed by ``relu``. Everything else
    replays the exact ``ops.matmul`` and FX post-op targets inside the rewritten
    graph; this correctness path does not emit a fused C++ kernel.

    **Caching.** The base FX graph is traced once. Rewritten complete graphs (or
    eager fallbacks) are memoized on each argument's ``(format, dtype)`` for an
    ``STensor`` and ``(rank, dtype)`` for a dense tensor. Shapes are not part of
    the key; native eligibility is rechecked on every execution.

    As with ``torch.fx.symbolic_trace``, decorated functions should express
    side-effect-free tensor computation. Python side effects performed while FX
    traces the function are not graph nodes and therefore are not replayed.

    See Also
    --------
    scorch.matmul : The contraction traced as an opaque leaf.
    scorch.set_autotune : Defines the accepted ``autotune`` level strings.

    Examples
    --------
    A GCN-style layer that hits the prebuilt CSR + bias + relu kernel:

    >>> import torch
    >>> import scorch
    >>> from scorch import STensor
    >>> torch.manual_seed(42)
    >>> mask = torch.rand(64, 64) < 0.1
    >>> vals = torch.rand(64, 64) * mask.float()
    >>> adj = STensor.from_csr(vals.to_sparse_csr(), "A")   # format "d,s" (CSR)
    >>> x = torch.rand(64, 16)                              # dense torch.Tensor
    >>> bias = torch.rand(16)
    >>> @scorch.compile
    ... def gcn_layer(adj, x, bias):
    ...     h = scorch.matmul(adj, x, format="dd")  # contraction (traced leaf)
    ...     h = h + bias                            # operator.add -> fused "add"
    ...     return torch.relu(h)                    # torch.relu   -> fused "relu"
    >>> out = gcn_layer(adj, x, bias)               # torch.Tensor [64, 16]
    >>> expected = torch.relu(adj.to_torch(in_place=False) @ x + bias)
    >>> torch.allclose(out, expected, atol=1e-4)
    True

    The parameterized form scopes SpMM autotuning around each call:

    >>> @scorch.compile(autotune="max")
    ... def fused(adj, x, bias):
    ...     h = scorch.matmul(adj, x, format="dd")
    ...     return torch.relu(h + bias)
    """

    def __init__(
        self, fn: Optional[Callable] = None, *, autotune: Optional[str] = None
    ):
        self._autotune = autotune
        # Parameterized form @compile(autotune=...) is called with no fn: defer
        # until the decorated function arrives via the first __call__.
        self._pending = fn is None
        self.fn = fn
        self._fx_graph: Optional[torch.fx.GraphModule] = None
        self._cache: Dict[tuple, Callable] = {}
        self._trace_failed = False
        self._lock = threading.RLock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Parameterized-decorator form: @compile(autotune=...) returns this
        # instance, which is then called with the function to wrap.
        if self._pending:
            if kwargs or len(args) != 1 or not callable(args[0]):
                raise TypeError("scorch.compile(autotune=...) must decorate a callable")
            return compile(args[0], autotune=self._autotune)

        if self._autotune is not None:
            from .tiling import autotune as _autotune_scope

            with _autotune_scope(self._autotune):
                return self._run(args, kwargs)
        return self._run(args, kwargs)

    def _run(self, args: tuple, kwargs: Dict[str, Any]) -> Any:
        # _run is only reached in execution mode, where fn was provided.
        assert self.fn is not None

        # Placeholder indexing is positional. Until keyword/default binding is
        # part of the fusion proof, forward keyword calls eagerly and exactly.
        if kwargs:
            return self.fn(*args, **kwargs)

        # Keep lock acquisition ordered global -> instance. FX tracing invokes
        # user code, which may call another compiled function; taking the
        # instance lock first can deadlock two concurrent first calls.
        if not self._trace_failed and self._fx_graph is None:
            with _TRACE_LOCK:
                with self._lock:
                    if not self._trace_failed and self._fx_graph is None:
                        try:
                            self._fx_graph = torch.fx.symbolic_trace(self.fn)
                        except Exception:
                            self._trace_failed = True

        trace_failed = self._trace_failed

        if trace_failed:
            return self.fn(*args)

        cache_key = self._build_cache_key(args)
        executor = self._cache.get(cache_key)
        if executor is None:
            with self._lock:
                executor = self._cache.get(cache_key)
                if executor is None:
                    assert self._fx_graph is not None
                    try:
                        # Copy graph structure, but retain get_attr/call_module
                        # objects by identity so in-place state mutations remain
                        # visible on later calls.
                        rewritten = torch.fx.GraphModule(
                            self._fx_graph,
                            copy.deepcopy(self._fx_graph.graph),
                        )
                        spec = analyze_fx_graph(rewritten.graph, args)
                        runner = compile_fused(spec, args)
                        executor = _rewrite_fused_subgraph(rewritten, spec, runner)
                    except Exception:
                        # Analysis and rewriting are optimizations. Any graph we
                        # cannot prove and construct safely retains eager
                        # semantics instead of returning a partial result.
                        executor = self.fn
                    self._cache[cache_key] = executor

        return executor(*args)

    @staticmethod
    def _build_cache_key(args: tuple) -> tuple:
        parts: List[Any] = []
        for a in args:
            if isinstance(a, STensor):
                parts.append((str(a.format), a.dtype))
            elif isinstance(a, torch.Tensor):
                parts.append((f"dense_{a.dim()}d", a.dtype))
            else:
                parts.append(type(a))
        return tuple(parts)
