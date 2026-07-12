# Tracing with `@scorch.compile`

`scorch.compile` is a function-level decorator that traces a small Python
function with `torch.fx`, finds a sparse contraction followed by a chain of
elementwise operations (bias, scale, activation), and dispatches that whole
subgraph to a single fused kernel. Where the per-op compiler optimizes *one*
`scorch.matmul` call at a time, `scorch.compile` fuses the matmul **and** its
epilogue — the pattern at the heart of a GCN or MLP layer — so the intermediate
result never round-trips through memory.

This page explains what it fuses, how to use it, and — just as importantly —
what it deliberately refuses to trace.

## A motivating example

A graph-convolution layer is a sparse matmul, a bias add, and a ReLU. Written
eagerly, each step materializes a full dense tensor. Wrapped in
`@scorch.compile`, the three steps collapse into one kernel call:

```python
import torch
import scorch
from scorch import STensor

torch.manual_seed(42)

# A sparse CSR adjacency (STensor) and a dense feature matrix.
mask = torch.rand(64, 64) < 0.1
vals = torch.rand(64, 64) * mask.float()
adj = STensor.from_csr(vals.to_sparse_csr(), "A")   # format "ds" (CSR)
x = torch.rand(64, 16)                              # dense torch.Tensor
bias = torch.rand(16)


@scorch.compile
def gcn_layer(adj, x, bias):
    h = scorch.matmul(adj, x, format="dd")   # contraction (traced as a leaf)
    h = h + bias                             # operator.add  -> fused "add"
    return torch.relu(h)                     # torch.relu    -> fused "relu"


out = gcn_layer(adj, x, bias)   # dense torch.Tensor of shape [64, 16]

# Verify against an eager PyTorch reference.
expected = torch.relu(adj.to_torch(in_place=False) @ x + bias)
assert torch.allclose(out, expected, atol=1e-4)
```

The decorated function is callable exactly like the original. On the first call
it is traced and compiled; every later call with the same input signature reuses
the compiled kernel.

:::{note}
The result is a dense `torch.Tensor`, not an {class}`~scorch.STensor` — both the
fused kernel and the fallback return a dense output when the matmul is written
with `format="dd"`.
:::

## How it differs from the per-op compiler

Scorch has two compilers, and they operate at different granularities. The
per-op compiler (described in {doc}`the compiler pipeline </compiler/pipeline>`)
fires implicitly on *every* `scorch.matmul` / `scorch.einsum` call and lowers
that single operation through CIN → LLIR → codegen. `scorch.compile` sits one
level up: it traces a *whole function* and fuses the contraction with its
elementwise epilogue.

| Aspect       | `@scorch.compile`                              | Per-op compiler                         |
|--------------|------------------------------------------------|-----------------------------------------|
| Granularity  | Whole function: matmul + postop chain          | A single op (`matmul`, `einsum`, …)     |
| Front end    | `torch.fx` symbolic trace of Python            | Index-notation CIN AST                   |
| Goal         | **Fuse** contraction + bias/activation         | Lower one op to an optimized kernel      |
| Dispatch     | Prebuilt fused C++ **or** JIT fallback         | Fast path → prebuilt → CIN/LLIR pipeline |
| Trigger      | Explicit `@scorch.compile` decorator           | Implicit on every matmul/einsum call     |
| Cache key    | Per-arg `(format, dtype)` / rank               | `(format_a, format_b, format_out)`       |

The two compose. Inside a decorated function the `scorch.matmul` node is treated
as an opaque leaf; when the fused subgraph does not match a prebuilt kernel, that
leaf still routes through the per-op compiler on the fallback path (below).

## Usage and options

`scorch.compile` is a class whose instances are callable, so it works both bare
and parameterized.

**Bare** — the common case:

```python
@scorch.compile
def layer(adj, x, bias):
    return torch.relu(scorch.matmul(adj, x, format="dd") + bias)
```

**Parameterized** with an autotune escape hatch — runs the traced function inside
a thread-local autotune scope, equivalent to wrapping every call in
`with scorch.autotune(level)`:

```python
@scorch.compile(autotune="max")
def layer(adj, x, bias):
    return torch.relu(scorch.matmul(adj, x, format="dd") + bias)
```

The `autotune` level tunes the SpMM tiling selector (see
{func}`~scorch.set_autotune` and {doc}`autotuning </user_guide/autotuning>` for
the accepted levels such as `"off"`, `"analytic"`, `"balanced"`, `"max"`,
`"learned"`). It changes *how fast* the contraction runs, never the fused
kernel's numerics.

:::{warning}
The wrapper forwards **positional arguments only**. Call the decorated function
positionally — keyword arguments to the wrapped function are not supported by the
`__call__` signature.
:::

### Trace-once, compile-per-signature

The FX graph is built on the first call and reused forever after. Compilation,
however, is keyed by an input signature: `(format, dtype)` for each
{class}`~scorch.STensor`, `(rank, dtype)` for each dense tensor. **Shapes are not
part of the key** — two calls that differ only in matrix dimensions reuse the
same compiled kernel, while a change in format, dtype, or rank triggers a fresh
compile.

## What gets fused

The tracer looks for exactly one contraction node and then walks the
single-consumer chain of elementwise operations immediately downstream of it.
The recognized post-ops are:

| In your code                       | Fused op   |
|------------------------------------|------------|
| `h + bias`  (`operator.add`)       | `add`      |
| `h * scale` (`operator.mul`)       | `mul`      |
| `torch.relu(h)` / `F.relu(h)`      | `relu`     |
| `torch.sigmoid(h)`                 | `sigmoid`  |
| `torch.tanh(h)`                    | `tanh`     |

For a binary op (`add` / `mul`), the operand that is *not* the running chain must
be a direct function argument (a placeholder). Unary activations take no extra
operand.

:::{warning}
The recognized spellings are exact:

- Bias and scale must use the `+` and `*` **operators**. `torch.add` and
  `torch.mul` are not recognized and stop fusion.
- The contraction must be `scorch.matmul`. `scorch.einsum` is not patched as a
  leaf, so a function whose contraction is written with `einsum` produces no
  matmul node and fails to trace.
- **`gelu` is not fusible.** Although Scorch's internal post-op enum lists
  `gelu`, it is absent from the tracer's recognized set — a `gelu` in the chain
  simply ends fusion at that point.
:::

## Dispatch: prebuilt vs. JIT fallback

Once the fused subgraph is identified, `scorch.compile` first tries to match a
hand-written prebuilt C++ kernel and otherwise falls back to a correctness path.

**Prebuilt (in-kernel fused).** Reached only for a CSR left-hand side times a
dense right-hand side, in float32, with one of exactly two epilogues:

| LHS format | RHS format | Epilogue         | dtype   |
|------------|------------|------------------|---------|
| `ds` (CSR) | dense      | `+ bias`         | float32 |
| `ds` (CSR) | dense      | `+ bias` + relu  | float32 |

These run the SpMM, the bias add, and (optionally) the ReLU in a single pass with
no intermediate materialization — the fast path exercised by the GCN example
above. A bias operand is required.

**JIT fallback.** Every other combination — a dense or COO left-hand side, a
non-float32 dtype, a `sigmoid`/`tanh`/`mul` epilogue, and so on — takes the
fallback. It runs the contraction through the per-op compiler (via
`ops.einsum`) and then applies the post-ops as ordinary torch operations, in
order.

```python
# COO left-hand side bypasses the prebuilt table — correct via the fallback.
adj_coo = STensor.from_torch(vals, "A").to_sparse("oo")


@scorch.compile
def fused(adj, x, bias):
    h = scorch.matmul(adj, x, format="dd")
    h = h + bias
    return torch.relu(h)


out = fused(adj_coo, x, bias)
expected = torch.relu(adj_coo.to_torch(in_place=False) @ x + bias)
assert torch.allclose(out, expected, atol=1e-4)
```

:::{note}
The fallback is a **correctness** path, not a code-generating one: it computes
matmul + torch post-ops separately rather than emitting a single fused C++
kernel. True in-kernel fusion currently exists only in the prebuilt CSR path
above. The fallback still guarantees the fused function returns exactly what the
unfused sequence would.
:::

## Limitations

`scorch.compile` is a targeted pattern matcher, not a general graph compiler.
Be explicit with yourself about what it will and won't handle:

- **Exactly one `scorch.matmul`.** A function with no matmul raises
  `ValueError("No scorch.matmul found in traced function")`; matmul-of-matmul is
  not fused.
- **A linear, single-consumer chain.** If any intermediate value is used more
  than once (branches to multiple consumers), the walk stops there and only the
  ops before the branch are fused.
- **Post-op operands must be function arguments.** A bias or scale computed
  *inside* the function (rather than passed in) breaks the chain at that op.
- **Only the six recognized elementwise ops** in the exact spellings above.
  Anything else — `gelu`, `torch.add`/`torch.mul`, reshapes, indexing — ends
  fusion at that node.
- **Data-dependent control flow** that `torch.fx` cannot symbolically trace
  (Python `if` on a tensor value, data-dependent loops) is a standard FX
  limitation and will fail during tracing.
- **Positional arguments only** to the decorated function.

When a function falls outside these bounds, prefer writing the ops directly and
letting the per-op compiler optimize each `scorch.matmul` on its own.

## See also

- {doc}`The compiler pipeline </compiler/pipeline>` — how a single
  `scorch.matmul` is lowered through CIN, LLIR, and codegen, i.e. the path the
  traced leaf and the JIT fallback route through.
- {doc}`Autotuning </user_guide/autotuning>` — the levels accepted by the
  `autotune=` escape hatch and what they tune.
- {doc}`compile API reference </api/compile>` — the `scorch.compile` object and
  its signature.
