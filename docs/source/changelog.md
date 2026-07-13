# Changelog

Scorch is research software. Versions follow the source `__version__` in
`src/scorch/__init__.py`.

```{admonition} Research software
:class: warning
Scorch is at version **0.0.1**. The public API may change between releases as the
compiler and kernel set evolve.
```

## 0.0.1

Initial research release accompanying the [CGO 2026 paper](https://ieeexplore.ieee.org/abstract/document/11394842).

Highlights:

- **Sparse tensor compiler** — expressions lower through Compiler Index Notation
  (CIN) → LLIR → C++, which is JIT-compiled and cached as a shared library.
- **Format notation** — declare any layout as a per-mode sequence of level types
  (`d`, `s`/`c`, `o`); CSR is `"ds"`, COO is `"oo"`, DCSR is `"ss"`.
- **PyTorch shim** — `import scorch as torch` with fall-through to real PyTorch
  for anything Scorch does not define.
- **Core operations** — {func}`~scorch.matmul`, {func}`~scorch.einsum`, and SpMV
  / SpMM / SDDMM / SpGEMM via the compiler and hand-written prebuilt kernels.
- **Fused neural-network kernels** — {func}`~scorch.sparse_linear`,
  {func}`~scorch.sparse_linear_fm`, {func}`~scorch.sparse_attention`,
  {func}`~scorch.sparse_softmax_csr`, and {func}`~scorch.fast_transpose`.
- **Autotuning** — the `off` / `analytic` / `balanced` / `max` / `learned`
  optimization-level ladder over a no-regression SpMM tiling selector.
- **`@scorch.compile`** — an FX-based decorator that fuses contraction +
  elementwise chains.
```
