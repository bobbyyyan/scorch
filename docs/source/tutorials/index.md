# Tutorials

Hands-on, runnable walkthroughs. The first four cover Scorch's core sparse
kernels one at a time; the last three build complete models — a graph
convolutional network, a sparse autoencoder, and a sparse-attention transformer.
Every tutorial verifies its result against a PyTorch reference, following the
project's correctness convention (`atol = rtol = 1e-3`).

## Core kernels

::::{grid} 1 2 4 4
:gutter: 2

:::{grid-item-card} SpMV
:link: spmv
:link-type: doc
Sparse matrix × dense vector.
:::

:::{grid-item-card} SpMM
:link: spmm
:link-type: doc
Sparse matrix × dense matrix.
:::

:::{grid-item-card} SDDMM
:link: sddmm
:link-type: doc
Sampled dense-dense product.
:::

:::{grid-item-card} SpGEMM
:link: spgemm
:link-type: doc
Sparse × sparse matmul.
:::

::::

## End-to-end models

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} {octicon}`share-android;1.5em;sd-text-primary` GCN
:link: gcn
:link-type: doc
A graph convolutional network for node classification.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-text-primary` Sparse autoencoder
:link: sparse_autoencoder
:link-type: doc
Exploit sparse inputs in a reconstruction model.
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-text-primary` Sparse transformer
:link: sparse_transformer
:link-type: doc
BigBird-style sparse attention over long sequences.
:::

::::

```{toctree}
:hidden:
:caption: Core kernels

spmv
spmm
sddmm
spgemm
```

```{toctree}
:hidden:
:caption: End-to-end models

gcn
sparse_autoencoder
sparse_transformer
```
