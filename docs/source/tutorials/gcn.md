# Graph convolutional network (GCN)

A graph convolutional network classifies the nodes of a graph by repeatedly
mixing each node's features with those of its neighbours. The neighbour-mixing
step is a **sparse matrix × dense matrix product** (SpMM): the graph adjacency
`A` is sparse, the node features `X` are dense. This tutorial builds a two-layer
GCN with Scorch, shows the one line that makes the sparse product fast, and runs
a self-contained toy graph you can paste and execute — no PyTorch Geometric or
OGB required.

Each graph-convolution layer computes

$$H' = A \, X \, W + b,$$

where `A` is the `[N, N]` sparse adjacency, `X` is the `[N, F]` dense node-feature
matrix, and `W`/`b` are an ordinary linear layer. A full node classifier stacks
two of these: `conv1 → relu → dropout → conv2 → log_softmax`.

## The drop-in idiom

The GCN example is written as a **drop-in replacement** for PyTorch:

```python
import scorch as torch
import torch.nn as nn
import torch.nn.functional as F
```

After `import scorch as torch`, every `torch.*` call goes through Scorch first.
Operations Scorch specializes — `matmul`, `einsum` — get sparse implementations;
everything else (`torch.nn`, `torch.rand`, `F.relu`, `torch.zeros`, …) falls
through to real PyTorch unchanged. So the model reads like ordinary PyTorch, and
only the calls that *can* benefit from sparsity are rerouted.

:::{note}
This `import scorch as torch` idiom is specific to the end-to-end model
tutorials, where treating Scorch as a shim is the teaching point. In library code
the recommended style is `import torch` **and** `import scorch` as separate
names — see {doc}`/getting_started/quickstart`.
:::

## The graph-convolution layer

Here is the layer. The single load-bearing line is the SpMM:

```python
class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, x, adjacency):
        out = torch.matmul(adjacency, x, format="dd")  # sparse A @ dense X
        out = self.lin(out)
        out += self.bias
        return out
```

### `format="dd"` — forcing a dense output

`torch.matmul(adjacency, x, format="dd")` multiplies the **sparse** adjacency by
the **dense** feature matrix. The `format=` keyword names the *output* layout as a
{doc}`format string </user_guide/format_system>`: `"dd"` means "dense rows, dense
columns", i.e. a fully dense result.

Why spell it out? The product of a sparse matrix and a dense matrix is dense in
general — every row of `A` that has at least one neighbour produces a fully
populated row of features. Passing `format="dd"` tells Scorch to allocate a dense
output buffer and run the SpMM kernel that writes directly into it, instead of
inferring a sparse output format and paying to build compressed index structures
it would immediately have to densify. For the `A · X` step of a GCN this is
exactly the right choice: the result feeds straight into `self.lin`, which
expects a dense tensor.

:::{tip}
`format=` (equivalently `output_format=`) is how you pin the result layout for any
Scorch matmul. Reach for `"dd"` whenever the sparse product is dense downstream —
GCN feature propagation, sparse-input linear layers, and similar cases. See
{doc}`SpMM </tutorials/spmm>` for the kernel behind this call.
:::

The rest of the layer is plain PyTorch: `self.lin` applies the learnable weight
`W` (a bias-free `nn.Linear`), and `+= self.bias` adds `b`.

## Building the sparse adjacency — three idioms

`A` must be a Scorch sparse tensor ({class}`~scorch.STensor`). Depending on where
your graph comes from, you construct it one of three ways. All three produce the
same operand; pick whichever matches the data you already have.

**1. From an edge list (COO).** The most common case — you have a `[2, E]` tensor
of `(source, destination)` pairs:

```python
adjacency = torch.from_coo(
    indices=edge_index,                      # [2, E] long tensor
    values=torch.ones(edge_index.shape[1]),  # edge weights (1.0 = unweighted)
    shape=(N, N),
)
```

**2. From a PyG CSR adjacency.** When a PyG `data` object already carries a
transposed CSR adjacency (e.g. after `ToSparseTensor`), hand it straight to
{func}`~scorch.from_torch`'s CSR sibling:

```python
adjacency = torch.from_csr(data.adj_t)       # data.adj_t is a torch sparse CSR
```

**3. From a dense adjacency.** If you built `A` densely, convert it and choose the
sparse level types explicitly. `"ds"` = dense rows + compressed columns, i.e. CSR:

```python
adjacency = torch.from_torch(dense_adj).to_sparse("ds")
```

:::{note}
`"ds"` is the CSR layout (dense-outer / compressed-inner) and is the natural
choice for a graph adjacency: one compressed list of neighbours per node. The
{doc}`format system </user_guide/format_system>` page covers the other level types
(`"oo"` = COO, `"ss"` = DCSR).
:::

## Runnable: one conv layer on a toy graph

This is the complete, self-contained program — a five-node toy graph, one
`GraphConvolution` layer, and a check against a dense reference. It needs only
`scorch` (no PyG, no OGB, no dataset download):

```python
import scorch as torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

N, F_in, H = 5, 8, 4                      # 5 nodes, 8 features, 4 hidden units

# Toy undirected edges as COO, with self-loops.
edge_index = torch.tensor([[0, 1, 2, 3, 4, 0, 1],
                           [0, 1, 2, 3, 4, 1, 0]])
adjacency = torch.from_coo(
    indices=edge_index,
    values=torch.ones(edge_index.shape[1]),
    shape=(N, N),
)
x = torch.rand(N, F_in)


class GraphConvolution(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.lin = nn.Linear(cin, cout, bias=False)
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x, adj):
        out = torch.matmul(adj, x, format="dd")   # SpMM, dense output
        return self.lin(out) + self.bias


conv = GraphConvolution(F_in, H)
h = F.relu(conv(x, adjacency))
print("GCN layer output:", h.shape)               # [5, 4]

# Reference: dense A @ X, then the same linear + bias + activation.
A_dense = torch.zeros(N, N)
A_dense[edge_index[0], edge_index[1]] = 1.0
ref = F.relu(conv.lin(A_dense @ x) + conv.bias)

assert torch.allclose(h, ref, atol=1e-3, rtol=1e-3)
print("GCN layer matches dense reference")
```

Running it prints:

```console
GCN layer output: torch.Size([5, 4])
GCN layer matches dense reference
```

The `assert torch.allclose(..., atol=1e-3, rtol=1e-3)` follows Scorch's
correctness convention: every sparse result is checked against a dense PyTorch
reference at `atol = rtol = 1e-3`. Because `import scorch as torch` shims the
namespace, the reference path (`A_dense @ x`, `conv.lin`, `F.relu`) runs on real
PyTorch, giving a genuine cross-check rather than Scorch-vs-Scorch.

:::{admonition} Stacking two layers
:class: tip
A full node classifier chains two `GraphConvolution` layers, reusing the same
`adjacency` for both:

```python
class GCN(nn.Module):
    def __init__(self, in_channels, hidden, num_classes):
        super().__init__()
        self.conv1 = GraphConvolution(in_channels, hidden)
        self.conv2 = GraphConvolution(hidden, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)
```

The adjacency is built once and shared, so its sparse structure is amortized
across both propagation steps.
:::

## Running the full example

The shipped example (`examples/gcn/`) trains a 128-hidden-unit GCN with PyTorch
Geometric and then runs Scorch for inference against a PyG/DGL baseline.

**Extra dependencies** (beyond Scorch and PyTorch):

```bash
pip install torch_geometric ogb tqdm
```

`torch_geometric` supplies the datasets and `ToSparseTensor`; `ogb` supplies
`ogbn-arxiv`. The DGL/PyG baseline scripts (`dgl_gcn.py`, `pyg_gcn.py`) may need
`dgl` as well.

**Datasets** (downloaded automatically into `./data/<dataset>`):

| Dataset | Source | Notes |
|---|---|---|
| `cora`, `citeseer`, `pubmed` | PyG `Planetoid` | small citation graphs, full-batch |
| `ogbn-arxiv` | OGB `PygNodePropPredDataset` | larger citation graph |
| `reddit` | PyG `Reddit` | mini-batch inference (`--batch-size`, default 2) |

**Train, then test.** Training runs through PyG; Scorch loads the trained weights
(`weights/gcn_<dataset>_weights.pth`) for inference:

```bash
# Train with PyTorch Geometric (produces the weight file).
python pyg_gcn.py --mode train --dataset cora

# Run Scorch inference against the PyG reference.
python scorch_gcn.py --mode test --dataset cora
```

:::{note}
`hidden_channels` is fixed at 128 and must match the trained PyG model, since
Scorch loads its weights. The example runs CPU-only (`device = "cpu"`). Larger
graphs such as `reddit` switch to mini-batch inference; the citation graphs run
full-batch.
:::

Across sparse workloads like this, Scorch delivers **1.05–5.80× over PyTorch
Sparse** (CGO 2026). The win comes entirely from the `format="dd"` SpMM: it is
Scorch's most heavily tuned kernel, and the GCN forward pass is dominated by the
two `A · X` propagation steps.

## See also

- {doc}`SpMM — sparse × dense matrix product </tutorials/spmm>` — the kernel
  behind `torch.matmul(adjacency, x, format="dd")`, in depth.
- {doc}`Neural-network operations </user_guide/neural_network_ops>` — fused sparse
  building blocks ({func}`~scorch.sparse_linear`, {func}`~scorch.sparse_attention`)
  for the layers around your GCN.
- {doc}`The format system </user_guide/format_system>` — what `"dd"`, `"ds"`, and
  the other format strings mean.
