# Neural-network operations

Scorch ships a small family of **fused, prebuilt C++ kernels** for the layers
that dominate sparse neural networks: a sparse-weight linear layer, its
feature-major twin for chaining, a fast transpose, a CSR row-softmax, and a
fully fused masked attention. Each one collapses what would otherwise be several
PyTorch calls (SpMM, a bias add, an activation, a scatter, a softmax) into a
single parallel region, and each one is a **drop-in replacement** for the
equivalent `torch.nn.functional` expression — with a safe torch fallback so the
result is always correct even when the native kernel is unavailable.

These are hand-written kernels in the native `scorch_ops` extension, *not* the
JIT compiler pipeline behind {func}`~scorch.matmul` / {func}`~scorch.einsum`
(see {doc}`operations </user_guide/operations>`). They are reached through
dedicated Python entry points, and that API boundary is deliberate — the general
SpMM path never accidentally lands in a fused-Linear kernel and vice versa.

:::{admonition} Every kernel here has a safe torch fallback
:class: tip
Each function checks its inputs (2-D, `float32`, kernel present in the build). If
anything doesn't qualify, it transparently falls back to the pure-PyTorch
reference — `x.T.contiguous()`, a segment softmax, a per-head attention loop,
a dense `x @ W.T`. The numeric result is identical up to float rounding either
way; you never lose correctness, only the speedup.
:::

All examples use the standard preamble — `import torch` then `import scorch`
(the two are separate; Scorch is not aliased over torch here):

```python
import torch
import scorch
```

---

## `sparse_linear` — drop-in `F.linear` with a sparse weight

```python
def sparse_linear(
    x: torch.Tensor,
    weight: Union[STensor, torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor
```

Computes `act(x @ weight.T + bias)` for a **natural-layout** activation
`x` of shape `[batch, in]`, returning `[batch, out]`. This is exactly the
signature of `torch.nn.functional.linear`, plus one optional fused activation —
so anywhere you write `act(F.linear(x, W, b))` you can write
`scorch.sparse_linear(x, W, b, "relu")` and the SpMM, the bias add, and the
activation all run in one prebuilt parallel region instead of three separate
PyTorch passes over the output.

### Parameters

`x`
: Dense `[batch, in]` `float32` activation, natural (row-major) layout.

`weight`
: The `[out, in]` weight. Pass a **CSR {class}`~scorch.STensor` built once and
  reused** (preferred — see the note below). A dense `torch.Tensor` is also
  accepted but is converted to CSR *on every call*, which is wasteful in a
  training loop.

`bias`
: Optional dense `[out]` tensor, fused into the epilogue.

`activation`
: One of `None`, `"relu"`, or `"sigmoid"`. `None` (also `"none"` / `"identity"`)
  applies no activation. Any other string raises `ValueError`.

**Returns** a dense `[batch, out]` `torch.Tensor`.

:::{tip}
Build the CSR weight `STensor` **once**, outside your hot loop, and reuse it:

```python
W = scorch.from_csr(W_dense.to_sparse_csr(), "W")   # build once
for x in batches:
    y = scorch.sparse_linear(x, W, b, "relu")       # reuse
```

Passing a dense `weight` re-runs `to_sparse_csr()` every call.
:::

### Example (verified against `F.linear`)

```python
import torch
import scorch

batch, in_f, out_f = 64, 512, 256
W_dense = (torch.rand(out_f, in_f) < 0.1).float()      # 10%-dense weight
W = scorch.from_csr(W_dense.to_sparse_csr(), "W")      # CSR STensor, built once
b = torch.rand(out_f)
x = torch.rand(batch, in_f)

y = scorch.sparse_linear(x, W, b, activation="relu")   # [batch, out]

ref = torch.relu(x @ W_dense.T + b)                    # torch reference
assert torch.allclose(y, ref, atol=1e-3, rtol=1e-3)
```

`activation=None` and `activation="sigmoid"` match
`x @ W_dense.T + b` and `torch.sigmoid(x @ W_dense.T + b)` respectively under the
same tolerance.

Internally `sparse_linear` transposes `x` with {func}`~scorch.fast_transpose`,
calls {func}`~scorch.sparse_linear_fm`, and returns a lazy `.T` view — so if you
are chaining several sparse layers, working feature-major directly (next section)
avoids the round-trip transposes entirely.

---

## `sparse_linear_fm` — the feature-major variant

```python
def sparse_linear_fm(
    x_fm: Union[torch.Tensor, STensor],
    weight: STensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor
```

Computes `Y = act(weight @ x_fm + bias[:, None])` in one prebuilt parallel
region (SpMM + per-output-channel bias + activation) and returns a **dense,
feature-major** result of shape `[out, batch]`. "Feature-major" means each *row*
is one feature and each *column* is one example — the transpose of the usual
`[batch, features]` layout.

Why bother with the layout? Because it lets a multi-layer forward stay
feature-major from start to finish. The `[out, batch]` output of one
`sparse_linear_fm` is exactly the `[in, batch]` input of the next, so you
**transpose once on the way in and once on the way out** and pay no per-layer
transpose and no separate torch bias/activation epilogue in between. That is the
natural shape for stacking encoder/decoder layers in a
{doc}`sparse autoencoder </tutorials/sparse_autoencoder>`.

### Parameters

`x_fm`
: Dense `[in, batch]` **feature-major** input (`float32`). A `torch.Tensor` or a
  dense {class}`~scorch.STensor`. The `[out, batch]` output of a previous
  `sparse_linear_fm` feeds straight in with no reshaping.

`weight`
: CSR {class}`~scorch.STensor` of shape `[out, in]` — the same convention as
  `F.linear` (`y = x @ weight.T`).

`bias`
: Optional dense `[out]`, added per output channel.

`activation`
: `None`, `"relu"`, or `"sigmoid"` (as above; other values raise `ValueError`).

**Returns** a dense `[out, batch]` `torch.Tensor`.

### Example (verified, and shown to equal the natural-layout result)

```python
import torch
import scorch

batch, in_f, out_f = 64, 512, 256
W_dense = (torch.rand(out_f, in_f) < 0.1).float()
W = scorch.from_csr(W_dense.to_sparse_csr(), "W")
b = torch.rand(out_f)
x = torch.rand(batch, in_f)

x_fm = scorch.fast_transpose(x)                        # [in, batch]  (once in)
h_fm = scorch.sparse_linear_fm(x_fm, W, b, "relu")     # [out, batch]

ref = torch.relu(W_dense @ x_fm + b[:, None])          # torch reference
assert torch.allclose(h_fm, ref, atol=1e-3, rtol=1e-3)

# It is exactly the transpose of the natural-layout sparse_linear result:
y = scorch.sparse_linear(x, W, b, "relu")              # [batch, out]
assert torch.allclose(h_fm, y.T, atol=1e-3, rtol=1e-3)
```

Chaining two layers stays feature-major throughout — one transpose in, one out:

```python
x_fm = scorch.fast_transpose(x)                        # transpose once, in
h1 = scorch.sparse_linear_fm(x_fm, W1, b1, "relu")     # [hidden, batch]
h2 = scorch.sparse_linear_fm(h1,   W2, b2, None)       # [out,    batch]
out = scorch.fast_transpose(h2)                        # transpose once, out -> [batch, out]
```

`sparse_linear_fm` routes to the prebuilt `spmm_csr_linear_fused_float` kernel
and falls back to a dense torch reference if that kernel is missing or the
tensors are not `float32`.

---

## `fast_transpose` — cache-blocked materialized transpose

```python
def fast_transpose(x: torch.Tensor) -> torch.Tensor
```

Materializes the transpose of a 2-D `float32` tensor as a contiguous `[C, R]`
tensor **bit-identical to `x.T.contiguous()`** for a `[R, C]` input.

Why does this exist as its own kernel? Because `x.T.contiguous()` in PyTorch is a
naive element-by-element scatter that runs well below memory bandwidth. Once the
fused `sparse_linear` epilogue removes the bias/activation cost, that single
input transpose becomes a large fraction of the whole forward pass — so
`fast_transpose` replaces it with a cache-blocked kernel (AVX2 8×8 / NEON 4×4 /
scalar micro-tiles) that streams the data close to bandwidth. It is what
`sparse_linear` uses internally to get from natural to feature-major layout.

### Behavior

- **Input:** a 2-D `float32` `torch.Tensor` `[R, C]`. **Returns** a contiguous
  `[C, R]` tensor.
- **Fallback:** if the input is not 2-D `float32`, or the kernel is unavailable,
  it returns `x.T.contiguous()`. Exactness is preserved in every case.

### Example (bit-identical)

```python
import torch
import scorch

x = torch.rand(1024, 768)
xt = scorch.fast_transpose(x)               # [768, 1024]

assert torch.equal(xt, x.T.contiguous())    # not allclose — exactly equal
```

:::{note}
`fast_transpose` **materializes** a new contiguous buffer; it is not a lazy view
like `x.T`. Use it when you specifically need a contiguous transposed copy (for
example, to feed a feature-major kernel), not merely to reinterpret strides.
:::

---

## `sparse_softmax_csr` — row-wise softmax over CSR values

```python
def sparse_softmax_csr(
    crow_indices: torch.Tensor,
    values: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor
```

Computes `softmax(scale * values)` independently within each CSR row span
`[crow[i], crow[i+1])` and returns a dense value array the same length as
`values`. This is the softmax stage of a sparse-attention chain (SDDMM → softmax
→ SpMM): it walks each row's contiguous span sequentially (max, then exp-and-sum,
then normalize) with **no scatter and no `nnz`-sized intermediate tensors**,
parallel over rows on torch's warm intra-op thread pool. That makes it a direct
replacement for a `torch.scatter`-based segment softmax over CSR values.

### Parameters

`crow_indices`
: CSR row-pointer tensor of length `nrows + 1`.

`values`
: The CSR nonzero values (`float32` for the fast path). The softmax is normalized
  within each row's span.

`scale`
: Multiplier folded into the logits before the softmax — the attention scale
  `1/sqrt(d)`. Defaults to `1.0`.

**Returns** a dense `torch.Tensor` the same length as `values`, per-row
normalized. Falls back to a torch segment softmax if the kernel is unavailable or
`values` is not `float32`.

### Example (verified against `F.softmax` per row)

```python
import torch
import torch.nn.functional as F
import scorch

M = (torch.rand(8, 8) < 0.4).float()
csr = M.to_sparse_csr()
crow = csr.crow_indices()
vals = csr.values().float()

scale = 0.5
w = scorch.sparse_softmax_csr(crow, vals, scale=scale)

# Each row span is a torch softmax of the scaled logits:
for i in range(8):
    a, b = crow[i].item(), crow[i + 1].item()
    if b > a:
        ref = F.softmax(scale * vals[a:b], dim=0)
        assert torch.allclose(w[a:b], ref, atol=1e-4)
```

Within each CSR row the returned weights sum to 1 (an empty row contributes
nothing).

---

## `sparse_attention` — fully fused masked multi-head attention

```python
def sparse_attention(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor
```

Computes masked multi-head attention over a **shared CSR mask**. For each query
row `i` and head `h`,

$$
\text{out}[i,h] \;=\; \sum_{j} \operatorname*{softmax}_{j}\!\big(\text{scale}\cdot Q[i,h]\cdot K[j,h]\big)\, V[j,h]
$$

where `j` ranges over the columns attended by row `i` — the structural nonzeros
of the CSR mask (`crow_indices` / `col_indices`). This is the same math the dense
path performs with `-inf` fills on the masked positions, but evaluated over
*only* the mask's nonzeros.

The prebuilt `scorch_sparse_attention_csr_float` kernel does the whole thing in
one parallel pass over rows — inline SDDMM, a two-pass row softmax in registers,
and the weighted-V accumulation — batched over all heads. There is no per-head
kernel dispatch, no `nnz`-sized intermediate, and no CSR round-trip between the
stages, which is what makes it substantially cheaper than assembling the same
computation from separate SDDMM / softmax / SpMM calls. See the
{doc}`sparse transformer tutorial </tutorials/sparse_transformer>` for the full
attention layer built around it.

### Parameters

`crow_indices`, `col_indices`
: The shared CSR attention **mask** structure over the `[S, S]` query×key grid,
  passed once and shared across all heads.

`Q`, `K`, `V`
: Dense `[S, H, D]` `float32` tensors (`S` = sequence length, `H` = heads,
  `D` = head dimension), as produced by `q_proj(x).view(S, H, D)`.

`scale`
: Attention scale; fold in `1/sqrt(D)`. Defaults to `1.0`.

**Returns** a dense `[S, H, D]` `torch.Tensor`. Falls back to a pure-torch
per-head reference when the kernel is unavailable or the tensors are not
`float32`.

### Example (verified against dense masked attention)

```python
import torch
import scorch

S, H, D = 128, 4, 32
mask = (torch.rand(S, S) < 0.1).float()      # structural attention mask
csr = mask.to_sparse_csr()
Q = torch.rand(S, H, D)
K = torch.rand(S, H, D)
V = torch.rand(S, H, D)
scale = 1.0 / (D ** 0.5)

out = scorch.sparse_attention(
    csr.crow_indices(), csr.col_indices(), Q, K, V, scale=scale,
)                                            # [S, H, D]

# Dense masked-attention reference (-inf on masked positions):
ref = torch.zeros(S, H, D)
for h in range(H):
    scores = (Q[:, h] @ K[:, h].T) * scale
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=1).nan_to_num(0.0)   # rows w/ no cols -> 0
    ref[:, h] = attn @ V[:, h]

assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
```

:::{note}
The mask is **structural**: only its nonzero positions can ever attend, exactly
like a dense score matrix filled with `-inf` off the mask. A query row with no
attended columns produces a zero output row (matching the `nan_to_num` in the
reference above).
:::

---

## Choosing between these and the general ops

| You want to… | Use |
|---|---|
| A linear layer with a sparse weight, `[batch, in]` in / `[batch, out]` out | {func}`~scorch.sparse_linear` |
| A chain of sparse linear layers, staying feature-major | {func}`~scorch.sparse_linear_fm` (+ two {func}`~scorch.fast_transpose`) |
| A contiguous transposed copy of a `float32` matrix | {func}`~scorch.fast_transpose` |
| A row-softmax over a CSR value array | {func}`~scorch.sparse_softmax_csr` |
| Masked multi-head attention over a fixed sparsity mask | {func}`~scorch.sparse_attention` |
| A general sparse `A @ B` / einsum contraction | {func}`~scorch.matmul` / {func}`~scorch.einsum` (see {doc}`/user_guide/operations`) |

These kernels are what let a full sparse model — a
{doc}`GCN </tutorials/gcn>`, a
{doc}`sparse autoencoder </tutorials/sparse_autoencoder>`, or a
{doc}`sparse transformer </tutorials/sparse_transformer>` — run without falling
back to per-stage PyTorch ops. Scorch as a whole reports **1.05–5.80× over
PyTorch Sparse** (CGO 2026) across its benchmark suite.

## See also

- {doc}`/tutorials/sparse_autoencoder` — a full autoencoder built on the
  feature-major `sparse_linear_fm` chain.
- {doc}`/tutorials/sparse_transformer` — sparse attention end to end with
  `sparse_attention` and `sparse_softmax_csr`.
- {doc}`/api/operations` — the complete operations API reference.
