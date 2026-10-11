# Workspaces

A **workspace** is a temporary accumulation buffer the compiler inserts to carry
the partial results of a *reduction* — the sum over a contracted index — while it
produces an output with a different shape or format. Workspaces are the mechanism
that makes SpMM row accumulation and SpGEMM (sparse × sparse) work: they let a
loop reduce over an inner variable (say `j`) into scratch storage, then emit the
finished output row from that scratch. This page explains what a workspace is,
the two variants (dense and sparse/COO-hashed), how it surfaces in the CIN as a
{doc}`Where node </compiler/index_notation>`, and how ordinary generated
{func}`~scorch.matmul` and {func}`~scorch.einsum` lowering reaches the scheduler's
workspace-insertion policy.

For where workspaces sit in the overall lowering stack, see
{doc}`/compiler/pipeline`; for the index-notation IR they live in, see
{doc}`/compiler/index_notation`.

## Why a reduction needs a buffer

Take the canonical matmul, $C_{ik} = \sum_j A_{ij} B_{jk}$. The index `j` is a
**reduction** variable — it appears in the inputs but not in the result `C`, so
the compiler must sum over it. The indices `i` and `k` are **free** — they index
the result. The naive nest is:

```text
ForAll(i, ForAll(j, ForAll(k, C[i,k] += A[i,j] * B[j,k])))
```

When `A` is sparse (CSR) the `j` loop only visits the stored columns of row `i`,
and every one of those contributes to *many* output positions `C[i, :]`. Rather
than scatter each `A[i,j] * B[j,k]` term straight into the final output — which,
for a *sparse* output, means you don't even know yet which `C[i,k]` are nonzero —
the compiler accumulates the whole row into a small buffer first, then writes the
finished row out once. That buffer is the workspace.

After the scheduler rewrites the nest, the computation becomes a producer that
fills the workspace and a consumer that drains it:

```text
ForAll(i,
    Where(
        producer = ForAll(j, ForAll(k, accum[k] += A[i,j] * B[j,k])),
        consumer = ForAll(k, C[i,k] = accum[k])))
```

Here `accum[k]` is a length-`K` scratch row (a workspace). The `Where` node is
what splits the loop nest into these two halves.

## The `Where` node: producer and consumer

A workspace never appears alone — it is always introduced together with a
{doc}`Where </compiler/index_notation>` node. `Where` expresses a
producer/consumer split:

Producer
: the sub-nest that *fills* the workspace, running the reduction. In the SpMM
  example this is `ForAll(j, ForAll(k, accum[k] += A[i,j] * B[j,k]))` — it sweeps
  the contracted index `j` and accumulates into `accum`.

Consumer
: the sub-nest that *reads* the workspace and writes the real output. For a dense
  output that is a plain copy, `ForAll(k, C[i,k] = accum[k])`; for a sparse output
  it sorts the accumulated coordinates and emits the compressed result.

The `Where` sits *inside* the outer free loop (`ForAll(i, …)`), so the buffer is
filled and drained once per row `i` and can be reused across rows.

:::{note}
`ForAll` in the CIN does **not** fix an execution order — loop ordering is chosen
by the scheduler. The `Where`/workspace pair is inserted only *after* the loop
order is selected, because whether a buffer is needed depends on which free
variables trail the last reduction. See {doc}`/compiler/lowering`.
:::

## Dense vs. sparse (COO-hashed) workspaces

A `Workspace` is really a specialized tensor operand with its own format, and it
comes in two flavors. The compiler picks the flavor from the *output* format.

### Dense workspace

Chosen when the output level being accumulated is **dense**. The workspace format
becomes `"d…d"` — a flat array indexed directly by the trailing free variable(s).
In the SpMM-into-a-dense-matrix case, `accum[k]` is a contiguous length-`K` row of
`float`s; the producer does `accum[k] += …` and the consumer copies it into the
dense output. Dense workspaces are also what the scheduler uses to support tiling
of a dense output.

This is the common path for **SpMM** (sparse `A` × dense `B` → dense `C`): you
know the full output row is dense, so a flat scratch row is exactly the right
buffer.

### Sparse / COO-hashed workspace

Chosen (the default) when the output is **sparse** and you *don't know in advance*
which output coordinates are nonzero — the situation in **SpGEMM / SpMSpM**
(sparse × sparse). The workspace format becomes `"o…o"` (coordinate), backed at
the C++ level by a `coo_workspace<T, dim>` template: a hash-of-coordinates
accumulator. The producer inserts or accumulates by coordinate as it discovers
nonzeros; the consumer then sorts the accumulated coordinates and emits the
compressed (e.g. CSR/DCSR) result row.

| | Dense workspace | Sparse (COO-hashed) workspace |
|---|---|---|
| Chosen for | dense output (SpMM) | sparse output (SpGEMM) |
| Format | `"d…d"` | `"o…o"` (coordinate) |
| C++ backing | flat scratch array | `coo_workspace<T, dim>` hash |
| Consumer does | copy into output | sort coordinates, emit compressed |
| Known nonzeros? | yes (full dense row) | no (discovered during reduction) |

## When the compiler inserts a workspace

Workspace insertion is a scheduling decision (see {doc}`/compiler/lowering`).
After the loop order is chosen, the scheduler:

1. Finds the **last reduction variable** in the loop order.
2. Collects the free variables that come *after* it — these are the modes the
   buffer must span.
3. Makes a workspace whose dimension equals the number of those trailing free
   variables, and wraps the body in a `Where(producer, consumer)`.

If **no** free variable follows the last reduction (the trailing-free count is
zero), no workspace is inserted — the reduction can accumulate straight into a
scalar in place. This is exactly the SpMV case: $y_i = \sum_j A_{ij} x_j$ has no
free variable after `j`, so the per-row reduction uses a scalar accumulator rather
than a buffer.

There is also a refinement for dense outputs: a dense-output SpMM only needs a
workspace if the schedule is actually going to **tile**; otherwise it can write
directly to the dense result. So not every SpMM materializes a buffer — it
depends on the chosen schedule.

## Reaching workspace lowering

Workspaces are scheduler-owned compiler artifacts, not a separate public
matrix-multiplication mode. Most calls to {func}`~scorch.matmul` use a prebuilt
kernel when one matches. Passing `use_cache=False` bypasses that prebuilt
dispatch and reaches the generic `einsum` compiler path; the scheduler then
decides whether the selected loop order and output format require a workspace.
The generated kernel still participates in the normal JIT kernel and persistent
`.so` caches.

### SpGEMM — the COO-hashed workspace

Sparse × sparse into a sparse (CSR) output exercises the coordinate workspace:
the compiler cannot know the output sparsity ahead of time, so it accumulates by
coordinate and emits the compressed result.

```python
import torch
import scorch

torch.manual_seed(0)
A_dense = (torch.rand(64, 48) < 0.15).float() * torch.rand(64, 48)
B_dense = (torch.rand(48, 32) < 0.15).float() * torch.rand(48, 32)

A = scorch.from_torch(A_dense, "A").to_sparse("ds")   # CSR STensor
B = scorch.from_torch(B_dense, "B").to_sparse("ds")   # CSR STensor

C = scorch.matmul(A, B, format="ds", use_cache=False)  # -> STensor (CSR)

ref = A_dense @ B_dense
assert torch.allclose(C.to_torch(), ref, atol=1e-3, rtol=1e-3)
```

For a dense output, reaching the generic compiler does not by itself force a
workspace. As described above, the scheduler inserts dense workspace storage
only when the chosen tiling plan needs it. Compiler tests that require a
particular `Where`/`Workspace` shape construct or schedule that CIN explicitly;
the public operation remains policy-driven.

## Where workspaces show up

- **SpMM row accumulation** — sparse × dense → dense. A dense workspace holds one
  output row; the producer accumulates all of a row's contributions, the consumer
  copies the row out. This is the workhorse behind GCN and sparse-autoencoder
  layers.
- **SpGEMM / SpMSpM** — sparse × sparse → sparse. A COO-hashed workspace
  accumulates unknown output coordinates, then the consumer sorts and emits the
  compressed result.
- **Reductions with trailing free modes generally** — any contraction where a free
  index follows the last reduced index in the schedule triggers workspace
  insertion.

Workspaces are also the seam the compiler uses to fuse epilogues (bias /
activation) onto the consumer, and to split producer and consumer across threads
for parallel compressed-output generation — see {doc}`/compiler/lowering` and
{doc}`/compiler/codegen`.

## See also

- {doc}`/compiler/index_notation` — the CIN AST, including `Where`, `ForAll`, and
  `Workspace` nodes.
- {doc}`/compiler/lowering` — the scheduler's workspace-insertion pass and the
  producer/consumer lowering that fills and drains the buffer.
- {doc}`/user_guide/operations` — {func}`~scorch.matmul`,
  {func}`~scorch.einsum`, and the SpMM / SpGEMM operations end to end.
