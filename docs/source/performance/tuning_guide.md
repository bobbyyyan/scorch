# Tuning guide

This page collects the practical levers that make Scorch fast in a real
application: warming the compile cache, building sparse operands once, reaching
for the fused neural-network kernels, and choosing an autotune level. Most of the
performance is already automatic — Scorch's headline result is
**1.05–5.80× over PyTorch Sparse (CGO 2026)** with no tuning at all — but a few
habits keep you on the fast path and out of the slow ones.

The short version:

1. **Warm the cache** — the first call to an operation compiles a kernel; every
   call after that is fast. Call {func}`~scorch.precompile_kernels` at startup.
2. **Reuse STensors** — build a CSR weight once and keep it; don't reconvert a
   dense tensor on every forward.
3. **Use the fused NN kernels** — {func}`~scorch.sparse_linear` /
   {func}`~scorch.sparse_linear_fm` (with {func}`~scorch.fast_transpose`) instead
   of `matmul` + bias + activation as three separate ops.
4. **Pick an autotune level** that matches how long-lived your process is
   (see {doc}`/user_guide/autotuning`).
5. **Let tiling stay out of your way** — it only engages for high-degree graphs
   whose dense operand overflows the last-level cache, and is provably inert
   everywhere else.

---

## 1. Warm the compile cache

Scorch is a JIT compiler. The *first* call to a given operation-and-format
combination generates a specialized C++ kernel with OpenMP parallelization and
compiles it (roughly seven seconds). Every subsequent call — **including calls in
later processes** — loads the cached shared library from disk and runs at full
speed. Compilation cost is paid once, not per call.

That means a naive "time one call" microbenchmark measures the compiler, not the
kernel. Warm up first:

```python
import torch
import scorch

# Compile the common SpMM / SpGEMM format combinations up front, so the first
# real call in your training or inference loop is already fast.
scorch.precompile_kernels()        # prints "Precompiled kernels."
```

{func}`~scorch.precompile_kernels` compiles the frequently used
`CSR @ dense -> dense`, `COO @ dense -> dense`, `COO @ CSR -> dense`, and
`CSR @ CSR -> CSR`/`dense` kernels ahead of time. It is *not* run automatically
at import — you call it explicitly.

:::{tip}
The on-disk `.so` cache lives under `TORCH_EXTENSIONS_DIR`. Because it survives
process restarts, a warm machine skips compilation entirely on the second run of
your script. If you edit codegen or C++ templates while developing, a stale cache
can mask your change — clear the torch extensions build directory to force a
recompile.
:::

If you only touch a couple of shapes, you can also warm them directly by running
one representative call of each before you start timing:

```python
A = scorch.from_torch((torch.rand(2048, 2048) < 0.05).float(), "A").to_sparse("ds")
B = torch.randn(2048, 128)
_ = scorch.matmul(A, B)            # first call compiles; discard the timing
# ... now measure.
```

---

## 2. Reuse STensors — convert once

Converting a dense tensor to a sparse {class}`~scorch.STensor` (building the CSR
index structure) costs real work. In a hot loop — an inference server, a training
epoch — do it **once**, outside the loop, and pass the same STensor every call.

```python
import torch
import scorch

A_dense = (torch.rand(2048, 2048) < 0.05).float()   # a sparse adjacency / weight
A = scorch.from_torch(A_dense, "A").to_sparse("ds")  # build the CSR STensor ONCE

for B in batches:                                    # B: [2048, 128] dense
    Y = scorch.matmul(A, B)                          # reuse A every iteration
```

The anti-pattern is rebuilding the sparse operand inside the loop — calling
`from_torch(...).to_sparse("ds")` on the same matrix every iteration re-runs the
CSR construction for no reason. The fused NN kernels make the choice explicit:
{func}`~scorch.sparse_linear` accepts either a CSR STensor (**preferred, reused**)
or a dense weight (**converted to CSR per call**). In a hot loop, always pass the
pre-built STensor.

:::{note}
`scorch.from_csr` is usable even though it is omitted from the package's
`__all__` list — it is bound as a top-level convenience constructor alongside
{func}`~scorch.from_torch` and {func}`~scorch.from_coo`.
:::

---

## 3. Use the fused neural-network kernels

A sparse `Linear` layer is not just an SpMM. It is SpMM **plus** a bias add
**plus** an activation — and, for a natural-layout activation, a transpose of the
input so the contraction is contiguous. Done as three or four separate PyTorch
ops, the epilogue (bias/activation) and the input transpose can dominate the whole
forward pass. Scorch folds them into one prebuilt parallel region.

Replace this:

```python
import torch.nn.functional as F
y = F.relu(F.linear(x, W_dense, b))        # SpMM + bias + relu, three passes
```

with this:

```python
import torch
import scorch

batch, in_f, out_f = 64, 512, 256
W_dense = (torch.rand(out_f, in_f) < 0.1).float()
W = scorch.from_csr(W_dense.to_sparse_csr(), "W")   # build the CSR weight once
b = torch.rand(out_f)
x = torch.rand(batch, in_f)

y = scorch.sparse_linear(x, W, b, activation="relu")     # ONE fused kernel
ref = torch.relu(x @ W_dense.T + b)
assert torch.allclose(y, ref, atol=1e-3, rtol=1e-3)
```

`activation` accepts `None`, `"relu"`, or `"sigmoid"`. {func}`~scorch.sparse_linear`
has the exact signature of `torch.nn.functional.linear` (plus the fused
activation), so it is a drop-in replacement.

### Stay feature-major across a stack

{func}`~scorch.sparse_linear` computes `act(x @ W.T + b)` for a natural-layout
`x` of shape `[batch, in]`. Internally it transposes the input once with
{func}`~scorch.fast_transpose`, runs the fused feature-major kernel, and transposes
the result back. If you chain several sparse layers (an autoencoder, an MLP),
transpose **once in and once out** and keep the intermediate activations
feature-major with {func}`~scorch.sparse_linear_fm`, which returns a feature-major
`[out, batch]` result that feeds straight into the next layer:

```python
x_fm = scorch.fast_transpose(x)                     # [in, batch], transpose once
h1_fm = scorch.sparse_linear_fm(x_fm, W1, b1, "relu")   # [hidden, batch]
h2_fm = scorch.sparse_linear_fm(h1_fm, W2, b2, "relu")  # [out, batch]
y = scorch.fast_transpose(h2_fm)                    # back to [batch, out] once
```

Why {func}`~scorch.fast_transpose` and not `x.T.contiguous()`? PyTorch's
transpose is a naive element scatter that runs well below memory bandwidth; once
the fused epilogue is folded away, that input transpose becomes a large fraction
of the whole forward. `fast_transpose` uses a cache-blocked AVX2 / NEON kernel and
is **bit-identical** to `x.T.contiguous()`:

```python
x = torch.rand(1024, 768)
xt = scorch.fast_transpose(x)                       # [768, 1024]
assert torch.equal(xt, x.T.contiguous())            # exactly equal
```

Both fused kernels fall back to a pure-torch reference when the input is not
float32 (or the kernel is unavailable), so they are always safe to use. See
{doc}`/user_guide/neural_network_ops` for the full API and the sparse-attention
kernels ({func}`~scorch.sparse_attention`, {func}`~scorch.sparse_softmax_csr`).

---

## 4. Pick an autotune level

Autotuning controls **one thing**: how the CSR-sparse × dense SpMM path inside
{func}`~scorch.matmul` is dispatched. It does not touch einsum, other operations,
or the general JIT compiler. Its whole job is to decide, for an *eligible* shape,
whether a cache-blocking kernel beats the default — and it is designed so it can
never lose (the default kernel `v2` is always a candidate).

The level is a compiler-style `-O` ladder. Default is `analytic`.

| Level | When to prefer it |
|-------|-------------------|
| `off` | You want the pure baseline with zero dispatch logic — debugging, or A/B measurement. |
| `analytic` | **Default.** A cost-model pick with no kernel timing. Zero probe stall; recovers ~97% of the tiling win on graphs that need it. Good for most workloads. |
| `balanced` | A long-lived process where a one-time first-call micro-probe (a few candidate timings, memoized) is worth paying for a better pick. |
| `max` | A recurring workload on a fixed machine: same probe as `balanced`, plus a persistent on-disk cache so the search is paid **once ever**, across process restarts. |
| `learned` | Experimental. A trained cost model; **falls back to `analytic`** unless a per-machine model file is installed. |

Set it globally, scope it, or use it as a decorator — the API mirrors
`torch.no_grad`:

```python
import scorch

scorch.set_autotune("balanced")            # process-global default
print(scorch.get_autotune())               # -> "balanced"

with scorch.autotune("max"):               # scoped override, restored on exit
    Y = scorch.matmul(A, B)

@scorch.autotune("analytic")               # per-call decorator
def forward(A, B):
    return scorch.matmul(A, B)
```

You can also set the level at process start with `SCORCH_AUTOTUNE=max python ...`,
and wipe the `max` cache with `scorch.clear_autotune_cache()`. The full API,
the level semantics, and the eligibility gate are documented in
{doc}`/user_guide/autotuning`.

:::{note}
On small or low-degree matrices, autotuning is a **no-op by design**: the pre-gate
routes them straight to the byte-identical default kernel, so the result is the
same at every level. A visible difference requires a high-degree operand large
enough to overflow the last-level cache (see the next section).
:::

---

## 5. When tiling actually helps

The SpMM `C[i,k] = Σ_j A[i,j] · B[j,k]` streams each sparse row of `A` against the
dense operand `B`. As long as `B` (size `J · 4 · N` bytes) fits in the last-level
cache, the default kernel is already optimal — and that covers **essentially every
GCN and autoencoder shape** (hidden dimensions 16–256, batch 256). For those,
tiling does nothing and should do nothing.

Tiling — column/cache blocking of the contraction axis — only pays off when two
things are true at once:

- The dense operand `B` **thrashes** the last-level cache (`J · 4 · N > C`), and
- the sparse matrix has **high degree** (enough nonzeros per column to reuse each
  reloaded slice of `B`), and its nonzeros are **scattered** (not banded).

That describes high-degree power-law graphs — social / web-scale adjacency
matrices — at moderate-to-wide free dimensions. For those, blocking `B` into
column panels that fit the cache recovers cross-row reuse and wins substantially.
Well-ordered banded matrices (FEM meshes) fail the scatter test and stay on the
default path even at high degree, because the default kernel already streams them
from cache.

| Your workload | What tiling does |
|---------------|------------------|
| GCN with small hidden dims, sparse autoencoder, attention, FEM, arXiv-scale graphs | Nothing — routes to the default kernel. |
| High-degree scattered graph (social/web), operand overflows LLC, moderate `N` | Contraction-axis cache blocking (`tile-j`). |
| Same, but very wide `N` (≥ 512) | B width-panel relayout (`tile-ijk`), which keeps output traffic linear. |

You do not select these kernels by hand — the autotune selector picks them, and
only inside the gate above. The practical takeaway: **if your hidden dimension is
small, tiling is irrelevant to you, and leaving autotune at its `analytic` default
costs you nothing.**

---

## 6. The no-regression philosophy

Every optimization in the tiling selector — and in Scorch's kernels generally —
ships under one rule: **it must generalize, and it must never regress anything.**
A kernel, threshold, tile size, or schedule is never tuned to a single benchmark,
matrix, or model. A change ships only if it is neutral-or-better across the whole
space Scorch cares about: narrow-`N` *and* wide-`N` SpMM, small *and* large row
counts, sparse *and* near-dense, and the real workload families (GCN, autoencoder,
attention).

When an optimization only helps a sub-regime, it is **gated behind a runtime
condition that provably cannot fire** on the shapes it would hurt. The tiling
selector is the textbook example:

- An O(1) "thrash-and-tile" pre-filter routes the vast majority of shapes — every
  GCN-small, autoencoder, FEM, and arXiv shape — straight to the default kernel at
  integer-comparison cost, before any dispatch closure is built.
- The default kernel `v2` is **always a candidate** in every measured decision, so
  a probe can only ever match or beat it — never lose.
- The wide-`N` relayout kernel joins the candidate set only above a free dimension
  wider than any current Scorch workload, so it is provably inert on all of them.

The upshot for you as a user: **you can leave the defaults on.** Autotuning at
`analytic`, the fused NN kernels, the fast transpose — all of them are designed to
be free wins where they help and invisible where they don't. You never have to
hand-guard against a Scorch optimization firing on the wrong shape.

---

## 7. Environment knobs (quick reference)

The Python API is primary; these environment variables exist for overrides and CI.
All are optional — the defaults are the recommended settings.

| Variable | Effect |
|----------|--------|
| `SCORCH_AUTOTUNE` | Initial global autotune level (`off`/`analytic`/`balanced`/`max`/`learned`). |
| `SCORCH_AUTOTUNE_CACHE=<path>` | Location of the persistent `max`-level cache; `=0` disables it. |
| `SCORCH_LLC_BYTES=<n>` | Override the detected last-level-cache size used by the tiling gate. |
| `SCORCH_MATCH_HOST_THREADS=0` | Stop matching `torch.get_num_threads()` in the drop-in SpMM (thread-team matching for GCN pipelines is on by default). |
| `SCORCH_ATPARALLEL_PIPELINE=0` | Run the SpMM on a private OpenMP team instead of torch's intra-op pool. |
| `SCORCH_REGBLOCK_DUAL=0` | Disable the register-block dual-path in the JIT SpMM (restores the single baseline nest). |
| `TORCH_EXTENSIONS_DIR` | Where compiled `.so` kernels are cached across runs. |

For the full set of autotune-specific knobs (legacy `SCORCH_TILING*` vars, the
learned-model overrides, and the gate thresholds), see
{doc}`/user_guide/autotuning`.

---

## Next steps

- {doc}`/performance/benchmarks` — the measured results and how to reproduce them.
- {doc}`/user_guide/autotuning` — the full autotune API and gate mechanics.
- {doc}`/user_guide/neural_network_ops` — the fused Linear, transpose, and
  attention kernels in depth.
