# User guide

The user guide is the conceptual heart of the documentation. It explains how to
build and convert sparse tensors, the format notation that names every layout,
the core operations, the fused neural-network kernels, tracing with
`@scorch.compile`, and the autotuning ladder.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-text-primary` Sparse tensors
:link: sparse_tensors
:link-type: doc
Create, convert, and inspect `STensor`s; interop with PyTorch.
:::

:::{grid-item-card} {octicon}`typography;1.5em;sd-text-primary` The format system
:link: format_system
:link-type: doc
Level types, format strings, and how CSR / COO / DCSR map onto them.
:::

:::{grid-item-card} {octicon}`x;1.5em;sd-text-primary` Operations
:link: operations
:link-type: doc
`matmul`, `einsum`, SpMV, SpMM, SDDMM, and SpGEMM.
:::

:::{grid-item-card} {octicon}`hubot;1.5em;sd-text-primary` Neural-network ops
:link: neural_network_ops
:link-type: doc
Fused `sparse_linear`, `sparse_attention`, `sparse_softmax_csr`, `fast_transpose`.
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-text-primary` `@scorch.compile`
:link: scorch_compile
:link-type: doc
Trace a function and fuse contraction + elementwise chains.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-text-primary` Autotuning
:link: autotuning
:link-type: doc
The `-O` level ladder and the no-regression tiling selector.
:::

::::

```{toctree}
:hidden:

sparse_tensors
format_system
operations
neural_network_ops
scorch_compile
autotuning
```
