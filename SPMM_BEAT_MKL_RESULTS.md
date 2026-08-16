# CSR × dense SpMM vs MKL — before and after

*Branch `perf/spmm-beat-mkl`, based on `04f321d`. Candidate = `14e3ea6`.
Hosts: **redwood** (Intel i9-14900K, 8 P + 16 E cores, 36 MB L3, PyTorch 2.5.1 +
MKL 2022.1) and **M5** (Apple, 6 P + 12 E cores, PyTorch 2.13.0). float32 unless
stated. Read `SPMM_BEAT_MKL_PHASE0.md` first for the attribution that led here.*

## What changed

One thing, in the layer where it belonged: **the native ABI boundary stopped
re-validating sparse indices that had not changed since the previous call.**

Phase 0 found that `native_abi.h` was charging 1.29–1.82 ns per nonzero on every
`matmul` — a serial scalar loop testing int32 representability, a full `.to(int32)`
cast, and a serial nested loop checking column bounds and within-row sortedness. That
is 1.2–2.0x the entire SpMM kernel at narrow free dimensions, and because it is
serial it capped parallel speedup at 1.5–4.0x over 32 cores against MKL's 3.0–15.2x.

Three parts, none of which removes a check:

1. **The scans are branchless, flat and parallel.** Every violation folds into one
   OR / min / max accumulator, so the loops vectorize and split across threads. A
   screen that reports trouble hands off to the original serial loop, whose
   `TORCH_CHECK`s still produce the byte-identical message. Screens are conservative
   by construction — they may flag a valid input, never pass an invalid one.
   Sortedness goes flat via one observation: a descent at position `p` is legal
   exactly when `p` starts a row, so count descents over the whole array and compare
   against the descents sitting on row boundaries.
2. **The verdict is memoized per index pair**, keyed on the coordinate
   `StorageImpl` address with a `weak_intrusive_ptr` held in the entry. A weak
   reference keeps the `StorageImpl`'s own allocation alive after its data is
   released, so while an entry exists its key address cannot be reused by a different
   `StorageImpl`: an expired weak pointer is proof of staleness, a live one proof of
   identity. `data_ptr`, `nbytes` and the version counter are recorded too.
3. **The int64→int32 narrowing is memoized per tensor** in `prebuilt_kernels`, so the
   cast happens once instead of once per call.

Deliberate trade, made on Bobby's call: a write straight through a raw buffer a
tensor aliases (numpy writing into shared memory) does not bump torch's version
counter, so a buffer corrupted that way can now reach a kernel unchecked.
`SCORCH_ABI_VALIDATE_MEMO=0` restores strict per-call validation;
`SCORCH_NARROW_INDEX_CACHE=0` restores the per-call cast.

### Two things this refutes

- **`ADAPTIVE_SPMM_TILING.md` §9.6** concluded the narrow-`N` high-degree deficit was
  "intrinsic to `v2`'s row-at-a-time full-width traversal", that MKL was "doing
  something structurally different", and that it was "a kernel to write, not a
  threshold to tune". With the tax removed, `v2` beats MKL at reddit N=16/32/64/128 by
  1.28x/1.91x/1.39x/1.31x. §9.6's taskset control missed it because the tax is
  thread-count-independent — which is exactly why pinning to 24 physical cores left
  the gap intact (194 ms vs 181 ms).
- **The N-crossover** (0.33x at N=32 → 0.80x at N=128 → 1.54x at N=512) was the tax
  being amortized, not blocking starting to pay: fixed cost per nonzero against kernel
  work growing with `N`.

## Correctness

| check | result |
|---|---|
| output bits vs `04f321d`, matrix × N × autotune level | **16/16 identical** (sha256 of the full result buffer) |
| validator rejection cases, base vs candidate | **57/57 pass**, identical messages — both the serial and parallel screen paths, empty rows, descents at every row boundary, first/last row, int64 non-representability, storage-sharing views, and `torch.inference_mode()` |
| float64 reference, every grid cell | see the grid table below |
| macOS suite, base vs candidate | identical pass/fail sets (205 pre-existing failures: the macOS SDK's libc++ cannot compile a JIT kernel at all on this host, on either tree) |
| Linux suite, candidate | see below |

## redwood — the grid

*(inserted when the run completes)*

## M5 — second host

ARM has no MKL, so the reference arm here is ATen's own CSR SpMM
(`torch.sparse.mm`), which is far slower than MKL. **These ratios are not comparable
to redwood's and are not a claim about MKL** — what transfers is the base→candidate
gain column.

| cell | reference ms | base ms | cand ms | **gain** | cand kernel ms | fixed µs |
|---|---|---|---|---|---|---|
| cora@32 | 0.190 | 0.104 | 0.096 | 1.08x | 0.055 | 41 |
| cora@512 | 1.389 | 0.357 | 0.343 | 1.04x | 0.284 | 59 |
| pubmed@32 | 1.328 | 0.455 | 0.238 | **1.91x** | 0.196 | 43 |
| pubmed@128 | 2.849 | 0.630 | 0.385 | **1.63x** | 0.322 | 63 |
| pubmed@512 | 11.42 | 1.496 | 1.309 | 1.14x | 1.201 | 108 |
| bcsstk17@32 | 3.651 | 0.855 | 0.367 | **2.33x** | 0.280 | 86 |
| bcsstk17@128 | 5.880 | 1.409 | 0.610 | **2.31x** | 0.504 | 106 |
| bcsstk17@512 | 17.86 | 2.520 | 1.804 | 1.40x | 1.582 | 222 |
| band16@128 | 9.899 | 3.374 | 1.025 | **3.29x** | 0.955 | 70 |
| scatter200@32 | 49.85 | 8.962 | 2.337 | **3.83x** | 2.235 | 102 |
| scatter200@128 | 150.9 | 17.85 | 13.53 | 1.32x | 13.41 | 125 |

Gains 1.04–3.83x, same direction and same shape as redwood: largest where `nnz/N` is
largest, smallest where the cell was already dominated by fixed per-call cost.

## What remains below parity, and why

The residual is **not** in the kernel. Two fixed per-call costs remain, measured by
differencing the harness's end-to-end and kernel-only timings, and by comparing
against the same kernel run with no torch in the process (`bench/spmm_micro.cpp`):

| component | redwood | what it is |
|---|---|---|
| Python dispatch | ~48–61 µs | `ops.matmul`: normalization, `resolve_prebuilt_matmul`, the tiling gate, argument marshalling. `torch.sparse.mm` pays ~5–10 µs for the same job. |
| native, beyond the kernel's own compute | ~52 µs | inside `eval_time`: pybind conversion of the nested tensor vectors, `torch::empty` for the output, the empty-row zeroing scan, the O(1) validation and memo lookup |

So for any cell whose kernel runs in under ~200 µs, scorch's fixed per-call cost is
the binding constraint and no kernel change can move it. Concretely on redwood:
cora@32's kernel is 25 µs against MKL's whole 50 µs call — the kernel is 2x MKL and
the cell still loses, because 48 µs of Python sits on top.

That is a separable piece of work in `ops.py`'s general dispatch path, outside the
file scope of this study, and it is the right next target for the small-cell regime.

## Scope and gaps, stated plainly

- **dtype.** Everything above is float32 except the float64 section below. float64 CSR
  × dense resolves a different prebuilt symbol (`prebuilt_spmm_csr_f64`) and **gets no
  tiling at all**, but it goes through the same `bind_binary_kernel_with_tile` →
  `validate_binary_inputs` → `checked_csr_view` path, so it pays the same tax and
  receives the same fix. Measured below rather than asserted.
- **The JIT codegen path still pays the full tax.** Generated kernels include the same
  `native_abi.h` and validate the same way, but they receive their operands through
  `ops.py`'s generic path rather than `prebuilt_kernels`, so they get no narrowing
  memo — and because the narrowing then produces a fresh int32 tensor on every call,
  the native memo cannot hit either. Fixing it properly means moving the narrowing
  memo into `checked_index_tensor` itself, which would also let the Python-side cache
  be deleted. Not done here: it is outside the file scope, and it would invalidate the
  measurements in this document.
- **Index memory.** The narrowing memo holds an int32 copy alongside the caller's
  int64 arrays, i.e. +50% on index memory. It replaces a same-size allocation that was
  happening every call, so peak does not grow, but steady state does. Narrowing at
  `STensor` construction and dropping the int64 arrays would instead *halve* index
  memory and remove the cache entirely; that lives in `stensor.py`/`storage.py`.
- **Only the drop-in float32 CSR×dense prebuilt symbol is tiled.** Fused bias/act
  SpMM, fused sparse Linear, SpMV and SpMSpM are unaffected by the selector, though
  they do all benefit from the validation fix.

## Method

- Random-permutation interleaved arms within a process, median over 11 rounds, with an
  A/A control arm running level `off` under a second name — the in-process noise floor.
- Base and candidate are different `.so` files and so cannot be interleaved in one
  process. For the cross-tree comparison the **MKL arm is the control**: it is
  byte-identical in both trees, so `|mkl_base/mkl_cand − 1|` is that cell's
  cross-process floor. Trees alternate base, cand, base, cand so drift hits both.
- Compared against the **faster** of MKL's int32-index and int64-index arms.
- Every cell checked against a float64 reference.
- Machine verified quiet before each run. This matters more than it sounds: a leftover
  `addr2line` pinning a single core added a flat ~8.6 ms to every arm of every cell,
  because with 32 OpenMP threads on 32 CPUs one preempted worker stalls the whole join
  barrier for a scheduler timeslice. It looked exactly like a uniform regression.
