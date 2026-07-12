# Sparse autoencoder

An autoencoder squeezes an input through a narrow bottleneck and reconstructs it:
an **encoder** `Linear → ReLU` maps the input into a low-dimensional code, and a
**decoder** `Linear → sigmoid` maps that code back to the input space. Trained with
MSE reconstruction loss, it learns a compact representation of the data.

The Scorch angle is entirely about **inference**. Real inputs — MNIST digits,
grayscaled CIFAR frames, CelebA faces — are mostly zeros. If we hand the encoder a
*sparse* batch, the very first thing it does, `input @ W.T`, becomes a **sparse ×
dense matmul (SpMM)** instead of a dense GEMM. Scorch's SpMM skips the zeros, so
the higher the input sparsity the more work disappears.

This tutorial is written as a drop-in shim: `import scorch as torch` replaces
PyTorch, and everything Scorch doesn't override (`torch.relu`, `torch.sigmoid`,
`nn.Module`, …) falls through to real PyTorch unchanged. We build the model, run
the encoder on a ~90%-sparse batch, verify it against a dense reference, and then
show the production-tuned **fused** successors that remove the last overheads.

## The model

Only one layer is special. The encoder's Linear is written out as a `SparseLinear`
module so we can control exactly what happens to a sparse input; the decoder is an
ordinary `nn.Linear`.

```python
import scorch as torch
import torch.nn as nn


class SparseLinear(nn.Module):
    """A Linear layer whose forward is an explicit ``input @ W.T + bias``.

    With a *dense* input this is identical to ``nn.Linear``. With a *sparse*
    input, ``torch.matmul`` dispatches an SpMM instead of a dense GEMM.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def forward(self, input):
        # sparse input @ dense W.T  ->  SpMM;  + bias is a broadcast add
        return torch.matmul(input, self.weight.t()) + self.bias


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim=256):
        super().__init__()
        self.encoder = SparseLinear(input_size, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_size)

    def forward(self, x):
        x = torch.relu(self.encoder(x))     # encode  -> [batch, 256]
        x = torch.sigmoid(self.decoder(x))  # decode  -> [batch, input_size]
        return x
```

The weights stay dense — an autoencoder's parameters are fully populated. The
sparsity lives in the **input**, and we introduce it deliberately at inference
time by converting the batch to CSR:

```python
sparse_data = data.to_sparse_csr()   # dense batch [B, input_size] -> sparse CSR
output = model(sparse_data)          # encoder's matmul is now sparse x dense
```

Because `to_sparse_csr()` produces a PyTorch sparse-CSR tensor, `torch.matmul`
inside `SparseLinear.forward` sees a 2-D sparse left operand and a dense right
operand and routes to Scorch's SpMM. The bias add and the `relu`/`sigmoid` are
plain elementwise ops.

:::{note}
`.to_sparse_csr()` triggers a one-time *"Sparse CSR tensor support is in beta
state"* `UserWarning` from PyTorch. It is harmless — the example scripts filter it
out with `warnings.filterwarnings`.
:::

## Running the encoder

Here is the smallest complete program that exercises the same path as the full
example: build a `SparseLinear`, feed it a ~90%-sparse batch as CSR, and check the
result against a dense reference with the project's `atol=rtol=1e-3` convention.

```python
import scorch as torch
import torch.nn as nn

torch.manual_seed(0)

input_size, enc = 784, 256          # MNIST: 28*28 pixels, 256-dim code
batch = torch.rand(16, input_size)
batch[batch < 0.9] = 0.0            # ~90% sparse input


class SparseLinear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(cout, cin))
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


enc_layer = SparseLinear(input_size, enc)

sparse_batch = batch.to_sparse_csr()             # dense -> sparse CSR
out = torch.relu(enc_layer(sparse_batch))        # sparse x dense encoder

# dense reference: the exact same math, computed densely
ref = torch.relu(batch @ enc_layer.weight.t() + enc_layer.bias)
out_t = out.to_torch() if hasattr(out, "to_torch") else out
assert torch.allclose(out_t, ref, atol=1e-3, rtol=1e-3)
print("Sparse-AE encoder OK:", tuple(out_t.shape))   # (16, 256)
```

A Scorch matmul may hand back an {class}`~scorch.STensor`; the
`hasattr(out, "to_torch")` guard brings it back to a plain tensor before the
comparison. With a dense output format the result is already a tensor, so the
guard is a no-op — but it makes the snippet robust either way.

:::{tip}
The win grows with sparsity, not batch size. A batch that is 60% zeros barely
beats dense torch; at 90–99% zeros the SpMM is doing a small fraction of the dense
FLOPs. Measure on *your* input distribution before assuming a speedup.
:::

## Making it fast

The `SparseLinear` above is deliberately transparent, and that transparency costs
a little. `torch.matmul(x, W.t())` runs the SpMM, then `+ bias` is a second pass
over the output, then `torch.relu` is a third — an **epilogue** of separate
memory-bound passes. And `W.t()` (or, in the feature-major layouts these kernels
prefer, transposing the *input*) materializes a transpose with PyTorch's
element-scatter, which runs well below memory bandwidth.

Scorch ships fused kernels that fold all of that away. See
{doc}`/user_guide/neural_network_ops` for the full API; the two you want here are
{func}`~scorch.sparse_linear` and {func}`~scorch.sparse_linear_fm`.

### `sparse_linear` — a fused drop-in for `F.linear`

{func}`~scorch.sparse_linear` has the exact signature of
`torch.nn.functional.linear`, plus an optional fused activation, and runs the SpMM
**+ bias + activation in one parallel region**:

```python
scorch.sparse_linear(x, weight, bias, "relu")   # == relu(x @ weight.T + bias)
```

:::{important}
`sparse_linear`'s sparse operand is the **weight**, not the input — it is the
`F.linear` form `x @ weight.T`, where `weight` is a sparse CSR
{class}`~scorch.STensor` `[out, in]` (a *pruned* Linear) and `x` is a dense
`[batch, in]` activation. That is the complement of the `SparseLinear` module
above, whose *input* is sparse. Use the plain `torch.matmul` path when the batch
is sparse and a dense-weight fused Linear; use `sparse_linear` when the weights
themselves are sparse.
:::

```python
import scorch as torch
import scorch
import torch.nn.functional as F

torch.manual_seed(0)

batch, cin, cout = 16, 784, 256
x = torch.rand(batch, cin)               # dense activations [batch, in]
W_dense = torch.rand(cout, cin)
W_dense[W_dense < 0.9] = 0.0             # ~90% sparse (pruned) weight
b = torch.rand(cout)

# Build the sparse weight ONCE and reuse it every forward.
W = scorch.from_csr(W_dense.to_sparse_csr(), "weight")

y = scorch.sparse_linear(x, W, b, "relu")           # fused SpMM + bias + relu
ref = F.relu(F.linear(x, W_dense, b))               # torch reference
assert torch.allclose(y, ref, atol=1e-3, rtol=1e-3)
print("sparse_linear OK:", tuple(y.shape))          # (16, 256)
```

Pass a dense `torch.Tensor` weight and `sparse_linear` will convert it to CSR for
you, but that conversion happens *per call* — in a hot loop, build the
{class}`~scorch.STensor` weight once (as above) and reuse it.

### Feature-major chaining with `fast_transpose`

Internally `sparse_linear` works in **feature-major** layout (`[in, batch]`,
one row per input feature) because that is where the fused SpMM kernel is fastest.
It transposes the input in, computes, and transposes the result back out. For a
*single* layer that is fine, but a multi-layer stack would pay a transpose at every
layer boundary.

{func}`~scorch.sparse_linear_fm` stays feature-major end to end. You transpose the
input **once** on the way in with {func}`~scorch.fast_transpose` — a cache-blocked
transpose (AVX2 / NEON micro-tiles) that is bit-identical to `x.T.contiguous()` but
far faster — feed the `[out, batch]` output of one layer straight into the next,
and transpose **once** on the way out:

```python
import scorch as torch
import scorch

torch.manual_seed(0)
batch, cin, cout = 16, 784, 256
x = torch.rand(batch, cin)
W = scorch.from_csr((torch.rand(cout, cin) * (torch.rand(cout, cin) > 0.9))
                    .to_sparse_csr(), "weight")
b = torch.rand(cout)

x_fm = scorch.fast_transpose(x)                       # [batch, in] -> [in, batch]
assert torch.allclose(x_fm, x.T.contiguous(), atol=1e-3, rtol=1e-3)

# Fused, feature-major:  relu(W @ x_fm + bias[:, None])  ->  [out, batch]
y_fm = scorch.sparse_linear_fm(x_fm, W, b, "relu")

# ... feed y_fm straight into the next sparse_linear_fm (already [out, batch]),
# and transpose ONCE at the very end:
y = y_fm.T                                             # back to [batch, out]
print("feature-major chain OK:", tuple(y.shape))      # (16, 256)
```

Across a chain, this removes every per-layer transpose and every separate
bias/activation pass, leaving only the one transpose in and one transpose out that
even a well-tuned dense/sparse BLAS path already pays.

:::{note}
Both fused entry points require **float32** operands and fall back to an exact
dense reference if the prebuilt kernel is unavailable or the dtype is unsupported
— so correctness is preserved regardless, and you only lose the fused speedup.
:::

## Running the full example

The distilled snippets above are the load-bearing lines; the shipped example in
`examples/sparse_autoencoder/` is a full train/test harness over real image
datasets.

**Extra dependency.** `torchvision` (for `datasets` and `transforms`). The
supported `--dataset` values and their flattened input sizes:

| Dataset | `input_size` | Notes |
|---|---|---|
| `mnist` | 784 | 28×28 grayscale |
| `cifar10` | 1024 | grayscaled + flattened |
| `cifar100` | 1024 | grayscaled + flattened |
| `celeba` | 4096 | 64×64 grayscale |

Datasets are auto-downloaded to `./data`. The bottleneck is `encoding_dim=256` for
all of them.

**Train, then test.** Training runs in plain PyTorch and saves the weights;
Scorch is used for the sparse inference pass, which loads those weights and
converts each batch to CSR:

```bash
# 1. Train (dense PyTorch) — saves models/<dataset>_sparse_autoencoder.pt
python torch_sparse_autoencoder.py --mode train --dataset mnist

# 2. Test (Scorch sparse inference) — loads the saved weights
python scorch_sparse_autoencoder.py --mode test --dataset mnist
```

Useful flags: `--batch-size` (train, default 64), `--test-batch-size`
(default 1000), `--epochs` (default 10), `--lr` (default 0.01). The example is
CPU-only.

:::{tip}
`--test-batch-size` matters for the sparse path: SpMM amortizes its fixed costs
over the rows of a batch, so a larger test batch of high-sparsity inputs is where
Scorch pulls ahead of a dense forward. Across the sparse-tensor workloads in the
paper, Scorch runs **1.05–5.80× over PyTorch Sparse (CGO 2026)**.
:::

## See also

- {doc}`/user_guide/neural_network_ops` — the full reference for
  {func}`~scorch.sparse_linear`, {func}`~scorch.sparse_linear_fm`, and
  {func}`~scorch.fast_transpose`.
- {doc}`/tutorials/spmm` — the sparse × dense matmul that powers the encoder.
- {doc}`/tutorials/gcn` — another end-to-end model built on the same SpMM, using
  `torch.matmul(adj, x, format="dd")`.
