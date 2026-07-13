---
sd_hide_title: true
---

# Scorch

<div class="scorch-hero">
  <div class="scorch-hero-flame">🔥</div>
  <h1><span class="scorch-gradient-text">Scorch</span></h1>
  <p class="scorch-tagline">A compiler-based sparse tensor library for PyTorch — declare a format, and Scorch generates the kernel.</p>
  <div class="scorch-cta">
    <a class="scorch-btn scorch-btn-primary" href="getting_started/index.html">Get started</a>
    <a class="scorch-btn scorch-btn-secondary" href="https://github.com/bobbyyyan/scorch">GitHub</a>
    <a class="scorch-btn scorch-btn-secondary" href="https://ieeexplore.ieee.org/abstract/document/11394842">Read the paper</a>
  </div>
</div>

```{code-block} bash
pip install -e .   # from a clone; see the install guide
```

Scorch is a **sparse tensor compiler** wearing a PyTorch shim. You `import scorch`,
describe each tensor's layout with a compact **format notation** (`"ds"` is CSR,
`"oo"` is COO), and call familiar operations like {func}`~scorch.matmul` and
{func}`~scorch.einsum`. The first call **JIT-compiles a specialized C++ kernel**
with OpenMP parallelism; every call after that reuses the cached shared library
and runs at full speed. In the [CGO 2026 paper](https://ieeexplore.ieee.org/abstract/document/11394842),
Scorch reached **1.05–5.80× speedups over PyTorch Sparse** across sparse-matrix
and graph-neural-network workloads.

---

## Why Scorch

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1.5em;sd-text-primary` Compiler, not a kernel zoo
Instead of hand-writing a kernel for every format × operation pair, Scorch
lowers your expression through **Compiler Index Notation → LLIR → C++** and
compiles it on demand. New formats need *no* new kernel code.
:::

:::{grid-item-card} {octicon}`typography;1.5em;sd-text-primary` One notation for every layout
Every dimension is `d` (dense), `s`/`c` (compressed), or `o` (coordinate). CSR
is `"ds"`, COO is `"oo"`, DCSR is `"ss"`. The same notation drives storage *and*
code generation.
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-text-primary` Drop-in for PyTorch
`STensor`s wrap ordinary `torch.Tensor`s. Anything Scorch doesn't define falls
through to real PyTorch, so `import scorch as torch` turns a model sparse with
almost no rewrites.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-text-primary` Autotuned, never regressed
An optional `-O` ladder (`off` → `analytic` → `balanced` → `max` → `learned`)
picks the best SpMM schedule per matrix — behind gates that *provably* cannot
fire on the shapes they would hurt.
:::

::::

---

## Quick example

```python
import torch
import scorch

# Build STensors from PyTorch tensors.
A = scorch.from_torch(
    torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]), "A"
).to_sparse("ds")            # CSR
B = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])

# Sparse × dense matrix multiply → dense torch.Tensor.
C = scorch.matmul(A, B)
print(C)

# The same result with Einstein summation.
D = scorch.einsum("ij,jk->ik", A, B)
assert torch.allclose(C, D.to_torch(), atol=1e-3)
```

```{note}
The **first** call to a new operation/format combination compiles a C++ kernel
(a few seconds). Subsequent calls — even across process restarts — load the
cached `.so` and run at full speed.
```

---

## Where to go next

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`play;1.5em;sd-text-primary` Get started
:link: getting_started/index
:link-type: doc
:class-card: scorch-nav-card
Install Scorch and run your first sparse kernel in five minutes.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-text-primary` User guide
:link: user_guide/index
:link-type: doc
:class-card: scorch-nav-card
Formats, tensors, operations, fused neural-net ops, and autotuning.
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-text-primary` Tutorials
:link: tutorials/index
:link-type: doc
:class-card: scorch-nav-card
SpMV, SpMM, SDDMM, SpGEMM, and full GCN / autoencoder / transformer models.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-text-primary` Compiler internals
:link: compiler/index
:link-type: doc
:class-card: scorch-nav-card
How CIN lowers to LLIR and then to JIT-compiled C++.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-text-primary` Performance
:link: performance/index
:link-type: doc
:class-card: scorch-nav-card
Benchmarks, the no-regression philosophy, and a tuning guide.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-text-primary` API reference
:link: api/index
:link-type: doc
:class-card: scorch-nav-card
Every public function and class, generated from the source.
:::

::::

```{toctree}
:hidden:
:caption: Get started

getting_started/index
```

```{toctree}
:hidden:
:caption: User guide

user_guide/index
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:caption: Compiler

compiler/index
```

```{toctree}
:hidden:
:caption: Performance

performance/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
```

```{toctree}
:hidden:
:caption: Develop

development/index
```

```{toctree}
:hidden:
:caption: About

faq
glossary
citation
changelog
```
