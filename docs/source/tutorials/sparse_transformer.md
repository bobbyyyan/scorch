# Sparse-attention transformer

Dense self-attention costs $O(S^2)$ in both compute and memory: every query
attends to every key. For long sequences that quadratic term dominates. This
tutorial builds the attention core of a **BigBird-style** sparse transformer —
each query attends only to a structured subset of keys (block-local and
sliding-window neighbours) — and runs the whole score → softmax → value pipeline
over that sparse pattern with Scorch.

The teaching payload is one line:

```python
context = torch.einsum("bhij,bhjd->bhid", probs, value).to_torch()
```

a **batched 4-D sparse × dense contraction** — sparse attention weights `probs`
(`[B, H, S, S]`, most entries structurally absent) times dense values `value`
(`[B, H, S, D]`), producing the dense context `[B, H, S, D]`. Everything before
it builds the sparse `probs`; everything after reshapes the result back to a
sequence of token embeddings.

Like the other application tutorials, this page uses the drop-in shim idiom
`import scorch as torch`: every `torch.*` call is a Scorch call, and anything
Scorch doesn't override (`torch.sparse.softmax`, `torch.sparse_coo_tensor`,
`torch.rand`, …) falls through to real PyTorch. That fall-through is exactly what
lets a BigBird attention block mix Scorch's sparse einsum with stock PyTorch ops.

## The idea: mask, softmax, contract

Standard scaled-dot-product attention is

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V.$$

The BigBird variant keeps the same math but zeroes out most of the $S \times S$
score matrix *before* the softmax, so both the softmax and the value contraction
touch only the allowed positions. The pipeline is four steps:

1. **Scores.** Compute `QKᵀ / √d` — a dense `[B, H, S, S]` tensor (this part is
   still dense; BigBird's savings come from what happens next).
2. **Mask to COO.** Keep only the allowed `(b, h, i, j)` positions — block-local
   and sliding-window neighbours — and pack their scores into a sparse
   `torch.sparse_coo_tensor`.
3. **Sparse softmax.** `torch.sparse.softmax(..., dim=-1)` normalizes each query
   row over just its attended keys. This is numerically identical to a dense
   softmax with `-inf` filled into the disallowed positions.
4. **Contract with V.** `scorch.einsum("bhij,bhjd->bhid", probs, value)` sums the
   attended values, weighted by the sparse probabilities.

:::{note}
Scorch's {func}`~scorch.einsum` **requires an explicit `->` output spec** — it
does not infer the output subscripts. Write the full
`"bhij,bhjd->bhid"`, never the implicit `"bhij,bhjd"` form.
:::

## The attention core (runnable)

This is the distilled `BigBirdSparseAttention.forward` — no dataset, no model
weights, no `torchtext`, so it runs in isolation. It uses a sliding-window mask
(each token attends to itself and its immediate neighbours, `|i − j| ≤ 1`) and
verifies the sparse result against a dense masked-attention reference.

```python
import scorch as torch

torch.manual_seed(0)
B, H, S, D = 1, 2, 6, 4            # batch, heads, seq len, head dim

q = torch.rand(B, H, S, D)
k = torch.rand(B, H, S, D)
v = torch.rand(B, H, S, D)

scale = D**0.5 + 1e-6
scores = torch.matmul(q, k.transpose(-1, -2)) / scale     # dense [B, H, S, S]

# --- BigBird sparsity pattern: allow only |i - j| <= 1 (sliding window) ---
idx = [(b, h, i, j)
       for b in range(B) for h in range(H)
       for i in range(S) for j in range(max(0, i - 1), min(S, i + 2))]
indices = torch.tensor(idx, dtype=torch.long).t()         # [4, nnz]

# gather the kept scores by flattening the (b, h, i, j) coordinate
flat = (indices[0] * H * S * S + indices[1] * S * S
        + indices[2] * S + indices[3])
values = scores.reshape(-1)[flat]

# --- sparse score matrix -> sparse softmax -> sparse x dense contraction ---
sparse_scores = torch.sparse_coo_tensor(indices, values, scores.size())
probs = torch.sparse.softmax(sparse_scores, dim=-1)       # falls through to torch
context = torch.einsum("bhij,bhjd->bhid", probs, v).to_torch()   # Scorch einsum

print("sparse attention context:", context.shape)         # [1, 2, 6, 4]

# --- dense masked-attention reference: -inf fill outside the window ---
allow = (torch.arange(S)[:, None] - torch.arange(S)[None, :]).abs() <= 1
masked = scores.masked_fill(~allow, float("-inf"))
ref = torch.matmul(torch.softmax(masked, dim=-1), v)

assert torch.allclose(context, ref, atol=1e-3, rtol=1e-3)
print("sparse attention matches dense masked reference")
```

A few things worth pointing at:

- **`.to_torch()` is required.** {func}`~scorch.einsum` returns an
  {class}`~scorch.STensor`; the `.to_torch()` brings it back to a dense
  `torch.Tensor` so the subsequent `.view(...)` / `.transpose(...)` /
  projection layers (which are plain PyTorch) work on a normal tensor.
- **The mask is data, not a code path.** Swapping the sliding window for
  BigBird's full block-local + random-block + global pattern is purely a change
  to how `idx` is built — the softmax and einsum lines are unchanged.
- **`torch.sparse.softmax` and `torch.sparse_coo_tensor` fall through** to real
  PyTorch. Only the value contraction is a Scorch kernel here; that is the shim
  doing its job.

In the full model this core lives inside a multi-head attention module, and the
context is reshaped back to `[batch, seq, embed_dim]` before the output
projection:

```python
context = context.contiguous().view(batch_size, seq_length, self.embed_dim)
```

The rest of the transformer block — output projection, residual add, layer norm,
the feed-forward network — is ordinary PyTorch and needs no Scorch-specific code.

:::{tip}
The score step (`QKᵀ` sampled only at the mask positions) is a **Sampled
Dense-Dense Matrix Multiplication (SDDMM)**. The runnable core above computes the
full dense `scores` and then gathers, which is fine for teaching; to compute
*only* the masked scores, see the {doc}`SDDMM tutorial </tutorials/sddmm>`.
:::

## Making it fast

The COO-mask + `torch.sparse.softmax` + `einsum` pipeline above is the clearest
way to *understand* sparse attention, but it materializes several intermediates:
a dense score matrix, an nnz-sized COO value array, and a separate softmax pass
that scatters over rows. Scorch ships two **CSR-native fused successors** that
collapse those passes. Both are exported at the top level and both have exact
pure-torch fallbacks, so they are safe to call unconditionally.

### Lever 1 — fused row softmax

{func}`~scorch.sparse_softmax_csr` replaces the scatter softmax with a single
sequential walk over each CSR row's contiguous value span:

```python
# crow_indices: CSR row pointers of the mask; values: the nnz attention scores
probs_values = scorch.sparse_softmax_csr(crow_indices, values, scale)
```

It computes `softmax(scale * values)` per row span `[crow[i], crow[i+1])` — max,
exp-and-sum, normalize — with no scatter and no nnz-sized intermediates,
parallel over rows on PyTorch's warm intra-op thread pool. The `scale` (your
`1/√d`) is folded in, so you pass the raw dot-product scores.

### Lever 2 — fully fused sparse attention

{func}`~scorch.sparse_attention` fuses the whole chain — SDDMM, row softmax, and
the weighted-value sum — into **one parallel pass over query rows**, batched
across heads:

```python
# Q, K, V: dense [S, H, D] float32; the CSR mask (crow/col) is passed once.
context = scorch.sparse_attention(crow_indices, col_indices, Q, K, V, scale)
```

For each query row `i` and head `h` it computes

$$\text{out}[i, h] = \sum_{j} \operatorname{softmax}_j\!\big(\text{scale} \cdot Q[i,h]\!\cdot\!K[j,h]\big)\, V[j,h]$$

over that row's attended columns `j` (the CSR mask's structural nonzeros) —
exactly the masked-attention math the dense path expresses with `-inf` fills.
The inline SDDMM, a two-pass in-register row softmax, and the value accumulation
happen without a per-head kernel dispatch, without nnz-sized intermediates, and
without a CSR round-trip. It returns a dense `[S, H, D]` context.

:::{note}
`sparse_attention` takes `Q`/`K`/`V` as **`[S, H, D]`** (sequence-major, as
produced by `q_proj(x).view(S, H, D)`) and a single shared CSR mask, not the
`[B, H, S, S]` probability tensor the pedagogical einsum consumes. It is the
one-call replacement for the entire *scores → mask → softmax → contract*
sequence, per batch element.
:::

Both kernels are float32 CSR paths that fall back to a bit-compatible torch
reference when the native extension is unavailable or the inputs are not float32,
so switching to them never changes results — only speed. Across the sparse
operations Scorch targets, the shipped figure is **1.05–5.80× over PyTorch
Sparse (CGO 2026)**. See {doc}`/user_guide/neural_network_ops` for the full
sparse-attention API and how it composes with the fused sparse-Linear layers.

## Running the full example

The example model in `examples/sparse_transformer/` is a two-layer BigBird
classifier trained on text. Inference loads pretrained weights and reports
accuracy and timing against the PyTorch reference:

```bash
# train the reference model, then run Scorch inference
python torch_sparse_transformer.py  --mode train --dataset imdb
python scorch_sparse_transformer.py --mode test  --dataset imdb
```

Model configuration (from the example's `main`): `embed_dim=128`, `num_heads=4`,
`num_layers=2`, `block_size=16`, `num_random_blocks=2`, `num_sliding_blocks=2`,
`intermediate_size=256`. Datasets: `imdb` (2 classes), `ag_news` (4 classes),
`yahoo_answers` (10 classes; expects `data/yahoo_answers_csv/{train,test}.csv`).
Flags: `--batch_size` (default 64), `--epochs` (default 1), `--lr` (default
0.001), `--model_path`.

:::{warning}
The full example depends on **`torchtext`** (`IMDB`, `AG_NEWS`, `get_tokenizer`,
`build_vocab_from_iterator`), which is **deprecated and hard to install on recent
PyTorch** — the dataset imports may fail outright on a modern environment. If you
want to run it end to end, pin a compatible `torch` + `torchtext` pair, or swap
in your own tokenizer and vocabulary. The **attention core snippet above has no
`torchtext` dependency** and is the portable piece to learn from; the sparse
kernels it exercises are identical to the ones the full model uses.
:::

## See also

- {doc}`/user_guide/neural_network_ops` — the fused
  {func}`~scorch.sparse_attention` / {func}`~scorch.sparse_softmax_csr` API and
  the sparse-Linear layers, in full.
- {doc}`/tutorials/sddmm` — the sampled dense-dense product that computes masked
  attention scores at only the allowed positions.
- {doc}`/user_guide/operations` — {func}`~scorch.einsum`, its explicit-`->`
  requirement, and the other sparse contraction paths.
