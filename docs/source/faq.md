# FAQ

Short answers to the questions that come up most often when adopting Scorch.
Each points to the page with the full story.

## Is Scorch a drop-in replacement for PyTorch?

Mostly, and deliberately so. Scorch re-exports a curated set of names
(`matmul`, `einsum`, `from_torch`, `STensor`, and the neural-network helpers),
and its module `__getattr__` forwards **any name it does not define** straight
to real PyTorch:

```python
def __getattr__(name):
    return getattr(torch, name)
```

That means `import scorch as torch` gives you sparse implementations where
Scorch has them and vanilla PyTorch everywhere else. In these docs we write the
explicit two-import form — `import torch` then `import scorch` — because it is
clearer about which library each call comes from; the application tutorials
(GCN, autoencoder, transformer) keep the `import scorch as torch` shim because
the drop-in idiom is their teaching point. See
{doc}`Key concepts </getting_started/key_concepts>` and the
{doc}`operations guide </user_guide/operations>`.

## Why is the first call to an operation slow?

The first time you hit a given operation-and-format combination, Scorch
JIT-compiles a specialized C++ kernel (OpenMP-parallel, `-O3 -march=native
-ffast-math -funroll-loops`), which takes a few seconds. That kernel is cached
as a shared library on disk under `TORCH_EXTENSIONS_DIR`, so every subsequent
call — **including after a process restart** — loads the `.so` directly and runs
at full speed. To pay the compile cost up front, call
{func}`~scorch.precompile_kernels` to warm the common SpMM/SpGEMM combinations.
The prebuilt hand-written kernels (the common CSR paths) are compiled at install
time and have no such first-call cost. See the
{doc}`tuning guide </performance/tuning_guide>` and
{doc}`building </development/building>`.

## Does Scorch use the GPU?

No. Scorch is a **CPU** library: it generates and compiles OpenMP-parallel C++
kernels and scales across CPU cores. There is no CUDA backend. Because it falls
through to PyTorch for anything it doesn't implement, unrelated PyTorch code can
still use the GPU — but Scorch's own sparse kernels run on the host. See the
{doc}`architecture overview </development/architecture>`.

## Which sparse formats are supported?

Formats are described per-mode with one level type each: `d` (dense), `s` or `c`
(compressed — the two letters are synonyms), and `o` (coordinate). The familiar
matrix layouts fall out of this notation:

| Format string | Name | Meaning |
|---|---|---|
| `"dd"` | Dense | dense rows + dense columns |
| `"ds"` | CSR | dense rows + compressed columns |
| `"oo"` | COO | two coordinate lists |
| `"ss"` | DCSR | both modes compressed |

A fourth level type, `singleton`, exists in the type system but is **reserved**:
it has no single-letter alias, it is not accepted as an op output format, and
there is no lowering path for it yet. There is no block/BCSR level type. For the
full model — including the gotcha that a bare format string is split one
character per mode — see the {doc}`format system </user_guide/format_system>`.

## Why does `scorch.spmv` raise `AttributeError`?

Because `spmv` is not part of the public surface. It is a real function in
`scorch.ops`, but it is not re-exported, so `scorch.spmv` falls through
`__getattr__` to `getattr(torch, "spmv")` — and PyTorch has no `spmv` either,
which is the `AttributeError` you see. The supported way to do a sparse
matrix-vector product is {func}`~scorch.matmul` with a 1-D operand, which
dispatches to the prebuilt CSR SpMV kernel (or {func}`~scorch.ops.spmv`
internally on a miss):

```python
import torch
import scorch

A = scorch.from_torch((torch.rand(128, 128) < 0.1).float(), "A").to_sparse("ds")
x = torch.rand(128)
y = scorch.matmul(A, x)          # 2-D x 1-D -> dense vector
assert torch.allclose(y, A.to_torch() @ x, atol=1e-3, rtol=1e-3)
```

See the {doc}`SpMV tutorial </tutorials/spmv>`.

## Does `einsum` accept an implicit output (no `->`)?

No. {func}`~scorch.einsum` requires an explicit `->` in the expression string.
The parser splits on `->` and then on `,`, so an implicit form like
`"ik,kj"` raises `IndexError` rather than inferring the output indices. Always
write the output group:

```python
import torch
import scorch

A = scorch.from_torch((torch.rand(64, 32) < 0.2).float(), "A").to_sparse("ds")
B = torch.rand(32, 48)
C = scorch.einsum("ik,kj->ij", A, B, format="dd")   # explicit '->'
assert torch.allclose(C.to_torch(), A.to_torch() @ B, atol=1e-3, rtol=1e-3)
```

The output *format* is still inferred when you omit `format=`; it's only the
index notation that must be explicit. See the
{doc}`operations guide </user_guide/operations>`.

## How do I make Scorch faster?

Three levers, in rough order of effort:

1. **Reuse STensors.** Build a sparse operand once (e.g. a pruned weight matrix)
   and pass the same {class}`~scorch.STensor` on every call rather than
   re-converting a dense tensor each time — conversion is not free.
2. **Use the fused neural-network kernels** where they fit:
   {func}`~scorch.sparse_linear` / {func}`~scorch.sparse_linear_fm` fuse
   SpMM + bias + activation into one parallel region, and
   {func}`~scorch.sparse_attention` / {func}`~scorch.sparse_softmax_csr` do the
   same for masked attention. See the
   {doc}`neural-network ops guide </user_guide/neural_network_ops>`.
3. **Pick an autotune level** with {func}`~scorch.set_autotune` or the
   {class}`~scorch.autotune` scope (`off` / `analytic` / `balanced` / `max` /
   `learned`). This tunes the CSR-sparse × dense SpMM path and is no-regression
   by construction. See the {doc}`autotuning guide </user_guide/autotuning>`
   and the {doc}`tuning guide </performance/tuning_guide>`.

Across the CGO 2026 paper benchmarks, Scorch achieved 1.05–5.80× speedups over
PyTorch Sparse on sparse-matrix and GNN workloads.

## How do I clear the compiled-kernel cache?

There are two independent caches. The JIT-compiled `.so` kernels live under
`TORCH_EXTENSIONS_DIR` (default `~/.cache/torch_extensions/…`); delete that
directory — or point `TORCH_EXTENSIONS_DIR` at a fresh path — to force a
recompile, which you'll want after changing codegen or C++ templates so a stale
cache doesn't mask your edit. The autotune probe cache is separate: call
{func}`~scorch.clear_autotune_cache` to wipe the persistent on-disk JSON written
by the `max` level. See the {doc}`building page </development/building>`.

## Can I train models with Scorch?

Scorch targets **sparse inference**, not training — there is no autograd
integration for its sparse kernels. The application benchmarks and tutorials run
in evaluation mode (`--mode test`) with pretrained weights that are pruned to a
sparse format for the forward pass. Train with dense PyTorch (or your framework
of choice), prune, then run inference through Scorch. See the
{doc}`GCN </tutorials/gcn>`,
{doc}`sparse autoencoder </tutorials/sparse_autoencoder>`, and
{doc}`sparse transformer </tutorials/sparse_transformer>` tutorials.

## See also

- {doc}`Quickstart </getting_started/quickstart>` — install and run your first op.
- {doc}`The format system </user_guide/format_system>` — the level-type model in full.
- {doc}`Tuning guide </performance/tuning_guide>` — caching, autotuning, and fused kernels.
