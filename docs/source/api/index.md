# API reference

The complete public API, generated from the source docstrings. Everything listed
here is importable directly from the top-level `scorch` namespace (for example
`scorch.matmul`, `scorch.STensor`). Anything Scorch does not define falls through
to PyTorch via `scorch.__getattr__`, so `scorch.relu`, `scorch.nn`, and
`scorch.tensor` are simply the corresponding `torch` objects.

## Public surface at a glance

| Area | Objects |
| --- | --- |
| {doc}`Operations <operations>` | {func}`~scorch.matmul`, {func}`~scorch.matmul_wksp`, {func}`~scorch.einsum`, {func}`~scorch.sparse_linear`, {func}`~scorch.sparse_linear_fm`, {func}`~scorch.sparse_attention`, {func}`~scorch.sparse_softmax_csr`, {func}`~scorch.fast_transpose`, {func}`~scorch.precompile_kernels` |
| {doc}`Tensors <tensors>` | {class}`~scorch.STensor`, {func}`~scorch.from_torch`, {func}`~scorch.from_coo`, `from_csr` |
| {doc}`Formats <formats>` | {class}`~scorch.TensorFormat`, `LevelType`, `LevelFormat` |
| {doc}`Autotuning <autotune>` | {class}`~scorch.autotune`, {func}`~scorch.set_autotune`, {func}`~scorch.get_autotune`, {func}`~scorch.clear_autotune_cache`, {func}`~scorch.compiler_schedule_search_space` |
| {doc}`Compile <compile>` | {func}`~scorch.compile` |

```{toctree}
:hidden:

operations
tensors
formats
autotune
compile
```
