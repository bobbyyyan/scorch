# CSR × dense SpMM vs MKL — before and after

*Branch `perf/spmm-beat-mkl`, based on `04f321d`. The grid below was measured at
`14e3ea6`; `6eec90f` fixed the selector, `d9450ca` extended the same fix to the JIT
codegen path, and `ba59040` cut the Python dispatch cost the grid was still paying —
all three land after the grid was taken. Hosts: **redwood** (Intel i9-14900K, 8 P + 16
E cores, 36 MB L3, PyTorch 2.5.1 + MKL 2022.1) and **M5** (Apple, 6 P + 12 E cores,
PyTorch 2.13.0).
**Every headline number is float32 through the prebuilt route**; float64 and the JIT
codegen path have their own sections and their own, weaker, conclusions. Read
`SPMM_BEAT_MKL_PHASE0.md` first for the attribution that led here.*

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
3. **The int64→int32 narrowing is memoized per tensor**, so the cast happens once
   instead of once per call. This now lives in `checked_index_tensor`, the one place
   every caller reaches — it started as a Python cache in `prebuilt_kernels`, which
   covered only the prebuilt route and, by handing the validator a fresh tensor each
   call, kept the structural memo from ever hitting on the JIT route. That cache is
   deleted; see the JIT section.

Deliberate trade, made on Bobby's call: a write straight through a raw buffer a
tensor aliases (numpy writing into shared memory) does not bump torch's version
counter, so a buffer corrupted that way can now reach a kernel unchecked.
`SCORCH_ABI_VALIDATE_MEMO=0` restores strict per-call validation and per-call
narrowing both — but read the JIT section before reaching for it, because on large
operands it is slower than the code it replaced.

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
| validator rejection cases, base vs candidate | **73/73 pass**, identical messages — both the serial and parallel screen paths, empty rows, descents at every row boundary, first/last row, int64 non-representability, storage-sharing views, `torch.inference_mode()`, the narrowing memo (repeat calls, in-place edits, unrepresentable arrays rejected on *every* call, caller arrays returned unmutated), COO bounds and lexicographic order, and dead-entry reclamation |
| float64 reference, every grid cell | see the grid table below |
| macOS suite, base vs candidate | **identical failure sets** — 208 failures on each, the same 208 test IDs, `comm` empty in both directions. Candidate passes 374 against base's 365, i.e. exactly the 9 new tests. The 208 are pre-existing and unrelated — every error is a `ninja` failure inside `is_trivially_copyable.h` / `strong_order.h`. **Since fixed**: they were one hardcoded SDK path in `get_extra_cflags`, not a macOS limitation; see the toolchain note below |
| Linux suite, candidate with the JIT change *and* the dispatch levers | **582 passed, 14 skipped, 0 failed** (full suite, perf tests included). This is the only host whose toolchain compiles a generated kernel, so it is the only real test of the JIT validator. Collection is 596 against base's 587 — the 9 new tests and nothing dropped — and the skip count is unchanged, so no test silently became a skip |
| Linux suite, base and candidate before the JIT change | **567 passed**, 14 skipped, 0 failed on each (`-m "not perf"`, measured on an earlier tree with fewer tests collected) |
| Linux suite, with the call plan (lever 5) | **652 passed, 14 skipped, 0 failed** — 582 before, so exactly the 70 new plan tests and nothing dropped or turned into a skip |
| macOS suite, with the call plan | **208 failed, 442 passed, 16 skipped**, and the failure set is still the same 208 test IDs as base (`comm` empty in both directions). 442 against base's 374: the 68 plan tests that run on this host |
| plan vs every legacy symbol it stands in for, bitwise | `spmm_csr_float_v2` (at 3 thread counts), `spmm_csr_float_tilej` (3 panel widths), `spmm_csr_float_tileijk` (3 panel pairs) and the float64 reference kernel, plus a shape × density × dtype grid through `matmul` — **exact equality, atol=rtol=0**, not `assert_close` |
| plan tests under both thread-policy arms | 4 combinations of `SCORCH_MATCH_HOST_THREADS` × `SCORCH_ATPARALLEL_PIPELINE`, all pass |

## redwood — the grid

Four matrix sets, the same ones the 195-cell tiling study used, plus `N`=16 and 64 on
`main` where that study had no coverage. 236 cells, 9 arms each, random-permutation
interleaved, median of 11 rounds, per-cell A/A control.

### Aggregate, vs the faster MKL arm

| group | cells | `off` | `analytic` | `balanced` | `max` | `learned` | **best** |
|---|---|---|---|---|---|---|---|
| main **base** | 108 | 0.492 | 0.521 | 0.534 | 0.532 | 0.529 | 0.541 |
| main **cand** | 108 | 1.470 | 1.623 | 1.703 | 1.686 | 1.667 | **1.729** |
| ss-tiling **base** | 57 | 0.698 | 0.686 | 0.713 | 0.713 | 0.695 | 0.719 |
| ss-tiling **cand** | 57 | 1.977 | 1.900 | 2.108 | 2.089 | 1.997 | **2.123** |
| ss-quick **base** | 63 | 0.682 | 0.686 | 0.707 | 0.708 | 0.699 | 0.719 |
| ss-quick **cand** | 63 | 1.541 | 1.556 | 1.643 | 1.635 | 1.616 | **1.676** |
| wide **base** | 8 | 1.155 | 4.677 | 5.160 | 5.160 | 4.561 | 5.182 |
| wide **cand** | 8 | 1.187 | 5.333 | 5.841 | 5.875 | 5.130 | **5.877** |

**Pooled over 236 cells: 0.675x → 1.878x.** Scorch was losing to MKL by a third on
average; it now beats it by 1.88x.

The `off` row is the honest one for judging the kernel: with no selector at all,
untiled scorch goes from 0.49–0.70x to 1.47–1.98x of MKL. The selector adds
1.06–1.18x on top of that, and 4.9x on the wide-`B` grid where it fires on every cell.

### Per-cell: the largest gains

| cell | MKL ms | base ms | base/MKL | cand ms | cand/MKL | **gain** |
|---|---|---|---|---|---|---|
| ct20stif@16 | 0.710 | 4.364 | 0.163x | 0.323 | 2.196x | **13.49x** |
| pdb1HYS@16 | 0.852 | 7.080 | 0.120x | 0.539 | 1.582x | **13.15x** |
| consph@16 | 1.834 | 10.64 | 0.172x | 0.873 | 2.100x | **12.18x** |
| pdb1HYS@32 | 1.537 | 7.653 | 0.201x | 0.731 | 2.104x | **10.47x** |
| scatter200@16 | 2.075 | 9.795 | 0.212x | 0.945 | 2.196x | **10.37x** |
| nd24k@16 | 6.014 | 48.53 | 0.124x | 4.724 | 1.273x | **10.27x** |
| mouse_gene@16 | 6.382 | 48.80 | 0.131x | 4.886 | 1.306x | **9.99x** |
| rma10@32 | 0.899 | 3.920 | 0.229x | 0.418 | 2.152x | **9.39x** |
| cant@32 | 1.421 | 7.030 | 0.202x | 0.764 | 1.860x | **9.20x** |

### Nothing regressed

Gain distribution over all 236 cells: **geomean 2.782x, min 0.760x, max 13.49x.**
Exactly one cell came out below 0.98x — `ash292@128`, whose own A/A noise floor in the
candidate run is **24.7%**: a 2208-nonzero matrix at 43 µs is not measurable at this
resolution, and the same matrix gains 1.21x at N=32 and 1.19x at N=512. Apart from
that one unmeasurable cell, no cell is slower than base.

The gain has a clean monotone shape in cell size, which is what a per-nonzero tax
being removed should look like:

| base `off` time | cells | gain geomean | min | max |
|---|---|---|---|---|
| < 50 µs | 11 | 1.127x | 0.574 | 1.451 |
| 50–200 µs | 21 | 1.547x | 0.939 | 3.998 |
| 0.2–2 ms | 42 | **3.628x** | 1.343 | 9.022 |
| 2–50 ms | 115 | 3.039x | 1.073 | 10.83 |
| > 50 ms | 47 | 2.184x | 1.007 | 9.559 |

Small cells gain least because fixed per-call cost, not the tax, dominates them; very
large cells gain less than the middle because the kernel's own work has taken over.

### Correctness on the grid

236 cells against a float64 reference: **max relative error 1.04e-06** (`gupta2@128`),
median 1.31e-07.

## The selector, re-measured on the fixed build

Removing the tax made the selector's own defect *worse*, because the tax had been
inflating both arms and masking it. Over the 236 cells, at the tiled routes:

| level | tiled-route regressions | worst |
|---|---|---|
| `analytic` (was default) | **6** | 0.373x (audikw_1@128, floor 1.9%) |
| `learned` | 4 | 0.385x (inline_1@512, floor 5.1%) |
| `balanced` | **0** | — |
| `max` | 0 | — |

`balanced` picked `v2` on exactly the cells `analytic` lost on. The gate cannot be
tightened out of this with the features on hand: the span proxy reads 0.823 on
`crankseg_1` (loses) and 0.823 on `mouse_gene` (wins); degree is ~201 on `crankseg_1`
(loses) and ~199 on `scatter200` (wins).

Fixed in `6eec90f`: both non-probing levels now confirm their cost-model pick against
`v2` once per shape before memoizing it — 6 kernel invocations against `balanced`'s 18.
Every non-`off` level is now no-regression-vs-`v2` by construction rather than by the
gate happening to be right.

The table above is the **pre-fix** measurement. The argument that the fix closes it is
structural — a level that times `v2` cannot ship a route slower than `v2` by more than
the timing's own error — but structural arguments are what produced the defect in the
first place, so it is not settled until re-measured. The re-run of `ss-tiling`, the
group that held all six `analytic` regressions and `learned`'s worst, is queued behind
the Linux suite; the number to check is zero tiled-route regressions at `analytic` and
`learned`. Until that lands, treat `analytic`'s row as open.

## float64

float64 CSR × dense resolves a different prebuilt symbol (`prebuilt_spmm_csr_f64`) and
**gets no tiling at all**, but goes through the same
`bind_binary_kernel_with_tile` → `validate_binary_inputs` → `checked_csr_view` path, so
it pays the same tax and gets the same fix. Measured on the M5, 3 rounds:

| cell | reference ms | base ms | cand ms | **gain** | floor |
|---|---|---|---|---|---|
| pubmed@32 | 2.171 | 0.850 | 0.548 | **1.55x** | 0.4% |
| bcsstk17@32 | 4.144 | 1.185 | 0.625 | **1.90x** | 1.5% |
| bcsstk17@128 | 8.466 | 1.993 | 1.430 | **1.39x** | 1.7% |
| band16@128 | 16.43 | 5.464 | 3.779 | **1.45x** | 0.5% |
| scatter200@32 | 101.8 | 19.04 | 12.56 | **1.52x** | 1.5% |

redwood, 3 rounds, against the faster MKL arm — and this is the part worth reading:

| cell | MKL ms | base ms | cand ms | **gain** | cand vs MKL | was | floor |
|---|---|---|---|---|---|---|---|
| bcsstk17@32 | 0.131 | 0.697 | 0.220 | **3.17x** | 0.596x | 0.188x | 6.6% |
| band16@128 | 7.840 | 13.07 | 10.59 | **1.23x** | 0.741x | 0.602x | 3.2% |
| bcsstk17@128 | 1.160 | 1.822 | 1.328 | **1.40x** | 0.873x | 0.637x | 13.9% |
| inline_1@32 | 42.93 | 104.9 | 45.13 | **2.33x** | 0.951x | 0.435x | 1.5% |
| pubmed@32 | 0.211 | 0.539 | 0.217 | **2.48x** | 0.972x | 0.394x | 33.7% |
| scatter200@32 | 4.524 | 12.35 | 4.169 | **2.96x** | 1.085x | 0.379x | 23.6% |

Pooled vs MKL: **0.409x → 0.853x**. The gain is real and large (1.23–3.17x, and the two
tightest floors — inline_1 at 1.5% and bcsstk17@32 at 6.6% — carry the biggest gains),
but **float64 is still below MKL parity on 5 of 6 cells**. That is not the validation
tax; it is that float64 resolves `prebuilt_spmm_csr_f64`, which has no register-blocked
row kernel and no tiling route at all, so removing the tax exposes a plain kernel-quality
gap that was previously hidden underneath it. float32 beats MKL on the same cells.

Stated plainly: float32 CSR × dense is done, float64 is not. Two of these cells
(pubmed@32 at a 33.7% floor, scatter200@32 at 23.6%) are too small to measure at 3
rounds and their ratios should not be quoted without a longer run. The M5 table above
is a comparison against torch's own float64 path, not MKL — ARM has no MKL — so it says
nothing about parity.

## The kernel hypotheses, all measured

The brief ranked five kernel hypotheses. With the tax gone, a 64-cell torch-free
variant grid (11 matrices × N=8…512, at 8 and 32 threads, each cell carrying its own
A/A control — which came out at geomean 1.0000, median floor 0.13%) settles them:

| hypothesis | verdict |
|---|---|
| **1. launch / thread policy for small work** | not the cause. Single-thread was slow too, and the poor scaling was Amdahl on the serial validator. What remains on small cells is dispatch, not launch. |
| **2a. deeper ILP** | **refuted.** 4- and 8-nonzero ILP measure 0.960–0.970 geomean, losing on 37–45 of 64 cells. |
| **2b. multi-row register blocking** | **refuted.** A two-pointer merge over consecutive rows is correct (relerr 1e-7) but 0.56–0.89x on every `N` tested. The reuse is real — adjacent-row overlap is 0.81–0.89 on exactly the matrices that lose — but a runtime merge cannot collect it: the data-dependent 3-way branch costs more than the saved B load, and it halves the FMA ILP the base kernel gets from 2-nonzero unrolling. Exploiting that overlap needs a *format* change (pre-merged / blocked columns), i.e. hypothesis 4. |
| **3. prefetch / NTA hints** | the traffic amplification this targeted turned out to be the tax, not B re-fetching. The shipped prefetch (2 nonzeros ahead, `PREFETCHT2`) is mildly miscalibrated: 16 ahead into L1 plus dropping the redundant mask when `N%8==0` is +2.7% at 32 threads and +4.2% at 8. **Not shipped** — it regresses 3–6 of 64 cells beyond their floors (worst 0.949x), and which cells regress changes with thread count, so any static choice trades one regime for another. Reported, not tuned around. |
| **4. cached preprocessing (CSB / reordering)** | untested, and now the *only* remaining route to the measured adjacent-row overlap, since 2b showed a runtime merge cannot pay for itself. |
| **5. full-`N`-in-registers for `N` ≤ 64** | **already implemented.** The narrow path holds the whole output row in YMM accumulators for `N` ≤ 32, and at `N`=64 the wide path is a single 64-wide tile with 8 accumulators. Nothing to add. |


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
| Python dispatch | ~48–61 µs | `ops.matmul`: normalization, `resolve_prebuilt_matmul`, the tiling gate, argument marshalling. For scale: `torch.sparse.mm` completes its *entire* call — dispatch and kernel — in 5–17 µs on cells this size. |
| native, beyond the kernel's own compute | ~52 µs | inside `eval_time`: pybind conversion of the nested tensor vectors, `torch::empty` for the output, the empty-row zeroing scan, the O(1) validation and memo lookup |

So for any cell whose kernel runs in under ~200 µs, scorch's fixed per-call cost is
the binding constraint and no kernel change can move it. Concretely on redwood:
cora@32's kernel is 25 µs against MKL's whole 50 µs call — the kernel is 2x MKL and
the cell still loses, because 48 µs of Python sits on top.

The Python half of that has since been fixed — 43 µs down to 9.7 µs, measured below.
The native half (~52 µs inside `eval_time`) has not been touched and is the next
target; the grid above and every ratio in it predates the dispatch fix, so the
small-cell numbers there are pessimistic by roughly 33 µs per call.

### Where the Python dispatch cost actually is

Measured on the M5 by differencing `scorch.matmul(A, B)` against the same
`scorch_ops.spmm_csr_float_v2(*args)` call with the argument list built once up front,
best of 3 batches of 2000–4000 calls:

| cell | kernel µs | `matmul` µs | overhead | share | `torch.sparse.mm`, whole call µs |
|---|---|---|---|---|---|
| M=500 deg=4 N=8 | 5.8 | 27.0 | **21.2** | 78.5% | 11.6 |
| M=2000 deg=8 N=32 | 31.6 | 57.9 | **26.3** | 45.4% | 146.0 |
| M=2000 deg=8 N=128 | 65.1 | 94.4 | **29.3** | 31.0% | 236.7 |
| M=20000 deg=24 N=32 | 267.6 | 297.4 | **29.8** | 10.0% | 3078.0 |
| M=20000 deg=24 N=128 | 353.8 | 390.0 | **36.1** | 9.3% | 5569.5 |

The overhead is nearly flat in problem size — 21 to 36 µs across a 60x range of kernel
time — which is what makes it fatal only at the small end. The smallest cell is the
one to look at: scorch's kernel is **half** MKL-free torch's whole call (5.8 vs 11.6 µs)
and scorch still loses 27.0 to 11.6, entirely on dispatch.

`cProfile` on that smallest cell, ranked by cumulative time, says where it goes:

| cost | share of the call | what it is |
|---|---|---|
| `STensor.from_torch` | **60%** | wrapping the dense `B` operand in an `STensor` — building a `Layout`, a storage object, normalized mode-indices, and running `_validate_index_storage` — every single call, then discarding it |
| `typing.__instancecheck__` | ~4% | 10 calls per matmul. `layout.py` and `storage.py` import `Mapping`/`Sequence` from `typing`, whose `isinstance` costs 153 ns against 73 ns for `collections.abc` and 33 ns for a concrete `(list, tuple)` |
| `layout.from_physical_shape` + `__post_init__` | ~11% | re-deriving and re-validating a `Layout` for the result on every call |
| `parse_format` | ~3% | called 3x per matmul on constant strings |
| the kernel itself | **8%** | |

### The dispatch fix, measured

Four levers were implemented, all in the Python object layer, none in the kernel:

1. **Assemble a dense operand from cached immutable parts.** Everything about a dense
   `STensor` except its values is a function of `(shape, dtype, device, name,
   mode_order)`: the format, the empty per-mode index arrays, the layout, the metadata.
   The first call builds them with the ordinary constructor and keeps them; later calls
   with the same key reuse them and attach the new values buffer. Sharing is sound
   because each part is a frozen dataclass and a dense tensor's mode-index arrays are
   empty tuples. A test compares the cached and ordinary paths field by field.
2. **`Mapping`/`Sequence` from `collections.abc`, not `typing`** — the same isinstance
   check for 73 ns instead of 153 ns, ten times per matmul.
3. **Memoize `parse_format` and `TensorLayout.from_physical_shape`**, both pure
   functions of their arguments that were re-deriving and re-validating a value object
   per call, plus `TensorFormat.__str__` and `.is_dense()` on the format itself.
4. **Stop making copies nothing reads.** `STensor.values` returns `self._value.detach()`
   and `_value` is already detached, so four of those allocations per matmul bought
   nothing; internal callers now read `_raw_values`. And `execute_prebuilt_binary_kernel`
   read the clock twice per call to fill a `time_dict` that nobody passed.

Measured on redwood, base = the tree at `b4f8985`, candidate = the same tree plus the
four levers. Both are the same compiled extension; only the Python differs. Arms
alternate base, cand, base, cand for four rounds; the harness's `kernel` arm calls
`spmm_csr_float_v2` directly with the argument list built once up front, and since that
arm is identical C++ in both trees its agreement across trees is the validity check.

| cell | kernel µs | `matmul` before | after | gain | dispatch before | after | `torch.sparse.mm` |
|---|---|---|---|---|---|---|---|
| M=64 deg=2 N=1 | 2.4 | 45.6 | 12.0 | **3.79x** | 43.1 | **9.7** | 8.6 |
| M=256 deg=2 N=4 | 3.1 | 45.6 | 12.4 | **3.67x** | 42.4 | **9.4** | 8.8 |
| M=500 deg=4 N=8 | 4.1 | 46.6 | 13.8 | **3.39x** | 42.4 | **9.6** | 10.4 |
| M=500 deg=4 N=32 | 7.3 | 50.4 | 17.9 | **2.82x** | 43.5 | **10.1** | 14.4 |
| M=2000 deg=8 N=8 | 14.7 | 56.4 | 21.3 | **2.65x** | 41.6 | **6.6** | 16.6 |
| M=2000 deg=8 N=32 | 18.4 | 61.3 | 26.7 | **2.30x** | 43.6 | **8.2** | 41.3 |
| M=2000 deg=8 N=128 | 41.7 | 85.2 | 55.9 | **1.52x** | 46.6 | **14.2** | 103.2 |
| M=20000 deg=24 N=32 | 129.6 | 188.1 | 144.5 | **1.30x** | 64.4 | **16.6** | 346.1 |
| M=20000 deg=24 N=128 | 508.9 | 432.6 | 530.2 | 0.82x | 62.3 | **27.4** | 1767.4 |

Geomean 2.22x on `matmul`, and the dispatch cost itself drops from a flat 42–46 µs to
6.6–10.1 µs on every cell whose kernel is small. It repeats: the smallest cell reads
43.6 / 42.0 / 42.6 / 44.6 µs across the four base rounds against 9.6 / 9.9 / 9.5 / 9.7
for the candidate. `cProfile` on that cell agrees with the wall clock — `matmul`
cumulative time over 4000 calls falls 0.480 s to 0.128 s (3.75x against the measured
3.79x), and `from_torch` falls from 65% of the call to 17%.

**The last row is drift, not a regression.** Its shared `kernel` arm disagrees between
the two trees by 37.7% — the largest mismatch in the run, against under 5% on seven of
the nine cells — so that cell's `matmul` medians are dominated by kernel noise. Its
dispatch component still falls 62.3 to 27.4 µs, in line with the rest.

**How far that leaves scorch from `torch.sparse.mm`, end to end.** The `torch` column
above is torch's *whole* call — its dispatch and its kernel together — so it is not
comparable to scorch's dispatch component. Comparing whole call against whole call:

| cell | scorch total before | after | `torch.sparse.mm` total | before | after |
|---|---|---|---|---|---|
| M=64 deg=2 N=1 | 45.6 | 12.0 | 8.6 | 5.33x slower | **1.41x slower** |
| M=256 deg=2 N=4 | 45.6 | 12.4 | 8.8 | 5.19x slower | **1.41x slower** |
| M=500 deg=4 N=8 | 46.6 | 13.8 | 10.4 | 4.48x slower | **1.32x slower** |
| M=500 deg=4 N=32 | 50.4 | 17.9 | 14.4 | 3.50x slower | **1.24x slower** |
| M=2000 deg=8 N=8 | 56.4 | 21.3 | 16.6 | 3.40x slower | **1.28x slower** |
| M=2000 deg=8 N=32 | 61.3 | 26.7 | 41.3 | 1.48x slower | **1.55x faster** |
| M=2000 deg=8 N=128 | 85.2 | 55.9 | 103.2 | 1.21x faster | **1.85x faster** |
| M=20000 deg=24 N=32 | 188.1 | 144.5 | 346.1 | 1.84x faster | **2.39x faster** |
| M=20000 deg=24 N=128 | 432.6 | 530.2 | 1767.4 | 4.09x faster | **3.33x faster** |

So the gap is closed in the sense that mattered — the fixed cost is no longer several
times the entire call — but **the smallest cells are still 1.24–1.41x slower than
`torch.sparse.mm` end to end**, down from 3.4–5.3x. The crossover moved from N=128 to
N=32 at 2000 rows: scorch now wins from `2000x8@32` upward and loses below it. (The last
row's before/after is the drift cell noted above; read its 4.09x and 3.33x as one number,
not a change.)

The arithmetic on the smallest cell says what is left to do: 12.0 µs is 2.4 µs of kernel
plus 9.6 µs of Python, and torch does its *whole* job in 8.6. Scorch's Python alone still
exceeds torch's entire call, so parity at the small end needs the Python under ~6 µs,
which the levers cannot reach — they removed the redundant work, and what remains is the
structure of the path itself. From the profile that is property-chain traffic: one matmul
does 18 `layout` reads, 14 `shape` reads and 8 `dim()` calls, each an attribute hop
through `_metadata`/`_storage`, with the kernel call 9% of the total and `from_torch`
17%. **Lever 5 is therefore still the endgame for the small end** — one pybind entry
doing resolution and marshalling in C++, so a warm call is a single Python→C++ hop, which
is where `torch.sparse.mm` gets its number from. What has changed is that it is no longer
worth 33 µs; it is worth the last ~4 µs on cells under ~20 µs.

### Lever 5: the call plan

Built, and it closes the gap. A repeated CSR × dense product now runs through a
**native call plan** — a `SpmmCsrPlan` (`src/scorch/csrc/plan.h`) holding everything the
dispatch used to re-derive: the resolved kernel, the tiling selector's memoized verdict
and its panel widths, the validated and narrowed index arrays, the tile clamp, the
shapes. `plan.run(values, B, nthreads, atparallel)` runs O(1) screens and calls the
kernel straight through, returning the 2-D result. A warm `scorch.matmul` is then a dict
probe and one Python→C++ hop.

Three pieces:

- **The kernel entries are split.** `spmm_csr_float_v2`, `spmm_csr_float_tilej`,
  `spmm_csr_float_tileijk` and the typed reference kernel each become a thin wrapper
  over a pointer-based core, and the wrapper keeps the exact signature and behaviour it
  had. The legacy entry unpacks the nested `vector<vector<Tensor>>` the pybind ABI hands
  over and calls the core; a plan calls the core with pointers it already holds. The
  kernel bodies are untouched — the diff moves the first fourteen and last four lines of
  each function.
- **`scorch/plan.py` decides when a plan is worth having.** A plan is installed on the
  *second* sighting of a `(sparse operand, B shape, B dtype)`, so a program that wraps a
  fresh `STensor` per call never pays for one it cannot reuse. It is built from what the
  ordinary dispatch just decided, so first-call semantics — the selector's probe, its
  measurements, its structured errors — are unchanged. At most 8 per operand.
- **A plan may always decline.** `run` returns `None` — never an error — when anything
  about the call is outside what it was built for: a non-contiguous or wrongly-strided
  operand, another device, a non-strided layout, a lazy `conj`/`neg` view, a values array
  of the wrong length, or index arrays written since (it records the source arrays' data
  pointers and version counters, the same evidence the ABI memo uses). The caller then
  takes the ordinary path, which produces the canonical result or the canonical error.
  Each screen is *required* for correctness — drop the contiguity test and a planned call
  returns a wrong answer — while being over-conservative only ever costs speed, and that
  asymmetry is why the design leans on declining. `STensor._set_state` — the funnel every
  in-place structural change goes through — drops plans outright; any autotune policy
  change retires them by moving a generation counter that is part of every key; and a
  plan that refuses everything withdraws itself (see the refusal cost below).

Measured on **both hosts**, **one arm per process** (see the method notes below), base =
the tree at `e72217b`, three alternating rounds of 9 reps × 300 calls. Both hosts ran the
committed code; the M5's base tree is `git archive HEAD` built in place, so the two trees
differ by lever 5 and nothing else.

| cell | redwood before | after | gain | M5 before | after | gain |
|---|---|---|---|---|---|---|
| M=64 deg=2 N=1 | 12.3 | **2.1** | **6.01x** | 8.1 | **1.2** | **6.69x** |
| M=256 deg=2 N=4 | 12.4 | **2.7** | **4.64x** | 9.4 | **2.5** | **3.76x** |
| M=500 deg=4 N=8 | 13.5 | **3.8** | **3.57x** | 13.0 | **5.8** | **2.21x** |
| M=500 deg=4 N=32 | 16.5 | **6.7** | **2.45x** | 20.5 | **12.5** | **1.64x** |
| M=2000 deg=8 N=8 | 21.3 | **10.8** | **1.97x** | 32.4 | **22.4** | **1.45x** |
| M=2000 deg=8 N=32 | 26.1 | **15.0** | **1.73x** | 47.4 | **35.9** | **1.32x** |
| M=2000 deg=8 N=128 | 49.8 | **37.3** | **1.34x** | 100.4 | **92.6** | 1.08x |
| M=20000 deg=24 N=32 | 142.1 | **124.7** | 1.14x | 283.5 | **275.3** | 1.03x |
| M=20000 deg=24 N=128 | 403.6 | **364.4** | 1.11x | 965.0 | 1026.3 | 0.94x |
| M=20000 deg=64 N=256 | 1966.6 | **1867.0** | 1.05x | 5985.9 | **5830.5** | 1.03x |
| M=50000 deg=32 N=128 | 1599.8 | **1499.9** | 1.07x | 4377.0 | **4349.4** | 1.01x |
| **geomean** | | | **1.95x** | | | **1.61x** |

**The four cells above ~500 µs say nothing either way, and the control arm is how we know.**
A plan removes ~10 µs of Python; those cells' kernels run for 1–6 ms, so the effect is ~1%
and the cross-process spread at that size is ±5% on the M5. The M5's one sub-1.0 cell is
the clearest case: the **plans-off** arm, where the machinery is a single list index and
cannot touch the kernel, moves with the plans-on arm in every round (973/991, 1014/1004,
1010/1026 µs). That is a tree-to-tree offset on that cell, not something a plan did. The
measurable effect is on cells under ~100 µs, and there it is 1.3–6.7x on both hosts.

**What remains of the dispatch once a plan serves** is `matmul` minus a direct `plan.run`
on the same operands, differenced *within* one process: **0.4–0.7 µs** on every cell whose
kernel is small, against 43 µs before any of this work and 9.6 µs after levers 1–4. A warm
planned call is also **faster than calling the prebuilt kernel directly through its own
pybind entry** — 2.1 µs against 2.5 on redwood, 1.2 against 1.7 on the M5, consistently
across every round on both hosts — because the plan skips the nested
`vector<vector<Tensor>>` unpacking and the validation that entry performs.

**Against `torch.sparse.mm`, end to end, scorch is now faster on every cell of both
hosts** — both arms in one process, same arm set in both trees:

| cell | redwood before | now | M5 before | now |
|---|---|---|---|---|
| M=64 deg=2 N=1 | 1.64x slower | **3.53x faster** | 5.00x slower | **1.20x faster** |
| M=256 deg=2 N=4 | 1.64x slower | **2.75x faster** | 2.56x slower | **1.50x faster** |
| M=500 deg=4 N=8 | 1.64x slower | **2.25x faster** | 1.14x slower | **2.09x faster** |
| M=500 deg=4 N=32 | 1.03x slower | **1.40x faster** | 1.12x slower | **1.41x faster** |
| M=2000 deg=8 N=8 | 1.49x slower | **1.29x faster** | 2.95x faster | **4.34x faster** |
| M=2000 deg=8 N=32 | 1.36x faster | **1.78x faster** | 3.35x faster | **4.43x faster** |
| M=2000 deg=8 N=128 | 1.50x faster | **3.10x faster** | 2.87x faster | **3.17x faster** |
| M=20000 deg=24 N=32 | 1.92x faster | **3.37x faster** | 12.49x faster | **12.50x faster** |
| M=20000 deg=24 N=128 | 3.96x faster | **5.45x faster** | 8.79x faster | **8.99x faster** |
| M=20000 deg=64 N=256 | 3.67x faster | **4.01x faster** | 9.93x faster | **9.87x faster** |
| M=50000 deg=32 N=128 | 4.57x faster | **5.15x faster** | 6.34x faster | **6.64x faster** |

The smallest cell went from **5.0x slower to 1.2x faster** on the M5 and from 1.64x slower
to 3.53x faster on redwood. Read the ratios only within a host and within a process: the
torch arm's own cross-tree drift is 0.2–1.4% on ten of the eleven M5 cells but 31–54% on
four redwood cells, so those four redwood ratios are approximate. And read *absolute*
microseconds only from the single-arm runs — on `2000x8@32`, `matmul` reads 15.0 µs alone
and 31.7 µs in the process it shares with the torch arm, which is the same arm-interference
effect the method notes describe.

**Nothing regressed, and here is what that rests on.** Two neutrality questions, each
measured as its own single-arm A/B on both hosts:

- *Did splitting the kernel entry cost the kernel anything?* The legacy pybind entry,
  called directly with a pre-built argument list, plans off in both trees: **redwood
  geomean 1.016** (0.956–1.069), **M5 geomean 1.049** (1.006–1.123). Both sit inside the
  cross-process spread these cells show on a byte-identical control arm, and the redwood
  figure straddles 1.0.
- *Did the plan machinery slow the path it cannot help?* With installation disabled — the
  same binary, `SCORCH_DISPATCH_PLAN=0`, which costs one list index — **redwood geomean
  1.008** (0.966–1.046) and **M5 geomean 1.028** (0.977–1.068). Gating the probe on that
  flag is what earned this: with the probe ungated the M5 read 1.019 against a ~1% floor,
  and passing the lookup key from the probe to the installer took redwood from 1.020 to
  1.008.
- *And the site a plan can never serve?* Its own single-arm A/B, with the withdrawal doing
  the work: **redwood geomean 1.012**, **M5 geomean 1.029** over the cells with N > 1.

**What a refused call costs, and why that needed a different measurement.** A plan
returns nothing when the call is outside what it was built for, and the ordinary path
then runs. The commonest cause by far is a right-hand operand with the planned shape and
dtype but the wrong memory: `B.T`, a column slice, any strided view. Two facts make the
cross-tree A/B useless for pricing that:

- Such a call is **1.2–25x more expensive than the servable one at baseline, with no
  plans in the picture at all** — the path calls `.contiguous()` (170 µs for a 1 MB
  operand on redwood) and the kernel then reads a freshly allocated cold buffer instead
  of a cache-warm one. On `2000x8@128` that is 30 µs contiguous against 315 µs sliced.
  Pre-existing, unrelated to this work, and worth its own look.
- The refusal itself is a few hundred nanoseconds, so it sits inside a much larger
  number that differs between two builds for reasons of its own.

So this one is measured **in one process**, flipping `scorch.plan.set_enabled` between
arms — the binary never changes, and `ops.matmul` reads that flag through a list index
that everything else hangs off, so the arms differ by the refusal and nothing else.
Interleaved, median of 11 rounds × 300 calls, on both hosts. Cost of a refused call
above the same call with the machinery inert:

| | redwood | M5 |
|---|---|---|
| plan withdrawn (a site a plan can never serve) | **+0.4–0.7 µs**, 1.01–1.04x | **+0.2–0.4 µs**, 1.01–1.04x |
| plan kept alive by serving another operand | **+1.2–1.4 µs**, 1.03–1.09x | **+0.4–0.9 µs**, 1.02–1.07x |

Read against the 14–37 µs those calls cost at baseline. This is the one path where plans
cost more than they save, it is bounded, and it is stated rather than averaged away. Two
cells above 20000 rows are dominated by per-call allocation of a 2.5 MB operand and read
±300 µs run to run; only the small cells mean anything here.

**Four defects this created, all found and fixed before it shipped.** One was caught by a
test, one by review, and two only by measuring — which is the argument for measuring.

*A correctness one*, caught by one of the tests above. The installer read the tiling
selector's verdict from the wrong place. The selector's memo holds an entry for any
`(operand, free dimension)` it has ever probed, and that entry outlives the conditions that
produced it: the same operand at a narrower free dimension, or on a host whose LLC swallows
B, fails the O(1) eligibility gate and is served by v2 without the memo being consulted at
all. Reading the memo from the installer — which sees every call — therefore built a plan
that ran tile-j while the ordinary path ran v2. Both answers are correct arithmetic; they
differ in the last bits (4.8e-06 on a 400×400 float32 product), which is exactly what a
path advertised as indistinguishable may not do. The fix reads the verdict at the dispatch
site, inside the branch where a tiled kernel actually served the call, and defaults to v2
everywhere else. Two tests now pin it: one that a tiled verdict *is* carried into the plan
on a shape that passes the gate, and one that a memo entry the gate rejects leaves the plan
on v2 and bit-identical to the ordinary path.

*A performance one.* The installer also read that verdict on *every* call rather than on
the call that installs. Reading it hashes a signature over the index arrays — `.item()`
calls, microseconds — and that made the ordinary path **1.53x slower** (12.4 → 19.4 µs on
the smallest cell) whenever a plan was not in play: a single-use operand, a declined call,
or plans switched off. Moving the read to the dispatch site fixed both defects at once.

*A permanent tax on call sites a plan cannot help.* A plan that refuses every call still
charged for the refusal, for the life of the operand. Now a plan that has served nothing
after `MAX_FRUITLESS_DECLINES` refusals withdraws itself, which halved the figure in the
table above (redwood 0.85 → 0.5 µs), and the detection is free: `matmul` returns
immediately when a plan serves, so reaching the installer with the key still present *is*
a decline. A plan that has ever served is never withdrawn, so a site mixing servable and
unservable operands at one shape keeps its plan. Passing the lookup key from the probe to
the installer, rather than building it twice, took the declining case down with it
(1.44–1.80 → 1.18–1.40 µs).

*Two things plans broke that had nothing to do with them.* `copy.deepcopy` and `pickle`
on an `STensor` raised `TypeError: cannot pickle 'scorch_ops.SpmmCsrPlan' object` as soon
as the operand had been multiplied twice; both work on the tree before plans.
`STensor.__getstate__` now omits the plan cache, which covers `pickle`, `deepcopy` and
`copy.copy` in one method, since Python routes all three through `__reduce_ex__`. A plan
is memoized work rather than state, so the duplicate builds its own on its second use.

**Three method notes worth keeping**, each earned by getting it wrong first.

*One arm per process.* The first version of this measurement ran all arms in one process
and reported the shared kernel arm 1.26–1.53x slower in the new tree and the plans-off path
50% slower. Both were artefacts: the new tree has an extra arm, so the arm *sets* differed,
and which arm precedes a cell decides the thread team and allocator state it inherits. One
arm per process removed the first entirely and left the second at 1.02. Interleaving arms
defends against drift over time; it does not defend against two trees having different arm
sets.

That "inherits the thread team it was handed" clause was inference when it was written. It
is now measured directly: a screen that builds its own OpenMP team made a tensor copy
standing next to it 9.5x slower, and made `torch.matmul` -- identical code, same process --
1.12-1.33x slower, both documented below under the index validation. So an interleaved
design does not merely fail to defend against this; each arm actively inherits the previous
arm's damage, which inflates every arm and makes the ordering between them noise rather
than biasing one of them.

*Prefer one process to two trees when the effect is sub-microsecond.* Everything the
cross-tree A/B could say about the refusal cost was noise, because a refused call is
dominated by an operand copy that has nothing to do with plans. Flipping a runtime flag
between arms in one process resolved 0.2 µs cleanly. The cross-tree design is only forced
where the *binary* differs; where a flag can switch the behaviour, one process is strictly
better evidence.

*Check that a benchmark arm is measuring the case it is named after.* The decline arm
originally shared its sparse operand with the `matmul` arm, so the same key was being
served thousands of times by one arm while refusing the other — which means it measured the
mixed case and could never have exercised the withdrawal it was supposed to test. Giving it
its own operand split the two cases apart, and both are now reported. An arm that shares
mutable per-operand state with another arm is measuring their interaction, not itself.

Everything above is the prebuilt route. Generated kernels reach the same header by a
different door — CINLowerer emits one `scorch_native::validate_jit_tensor` per
right-hand-side operand at the top of every `evaluate()` — and that door had none of
the fix applied to it. `validate_jit_tensor` walked the whole index structure in
serial nested loops with a `TORCH_CHECK` per element, for coordinate levels, for
compressed levels (spans, bounds, and sortedness), and for COO lexicographic order.
Nothing was screened and nothing was memoized, so a generated kernel paid the full
O(nnz) walk on every call.

What changed:

- **The narrowing is memoized in `checked_index_tensor`**, which is where every caller
  reaches it. This is what unblocks the rest: the structural memo keys on array
  identity, and a fresh int32 tensor per call could never match. The Python-side cache
  in `prebuilt_kernels.py` is deleted — it only ever covered the prebuilt route, and it
  was the thing handing the validator a new tensor each call.
- **The structural checks are screened and memoized**, using the same branchless
  accumulator screens as the CSR path, with every original serial loop left in place
  verbatim as the diagnostic. `checked_csr_view`'s inline screen is factored out and
  shared, so the CSR view and the JIT compressed level now run the same code.
- **`checked_coo_view`, `validate_csr_segments` and `validate_attention_inputs`** got
  the same treatment. They are not the JIT path, but they are the same defect in the
  same layer: the COO view walked every level's coordinates and then the lexicographic
  order serially, and the sparse-softmax and sparse-attention entry points walked their
  position and coordinate arrays per call.

One risk this created and the check that clears it: deleting the Python cache moves the
one-time narrowing from argument-building into the *first* kernel call, so if the tiling
selector timed a first call it would charge the narrowing to whichever candidate ran
first and pick the other. It does not — `_confirm_vs_v2` and the `balanced`/`max` ladder
probe both invoke each candidate once as a warmup before timing it, and both candidates
come from the same module and so share one memo, so the narrowing lands in a warmup and
never inside a measurement. `time_dict["eval_time"]` still includes the narrowing on a
tensor's first call and not after, exactly as it did when the Python cache did the work.

Two details worth knowing:

- **Verdicts are memoized per check family, not per array.** A cached "these
  coordinates are in range" must not satisfy "these coordinates ascend" for a
  one-level COO tensor, where both would otherwise record the same array and the same
  two parameters. The family tag is `params[0]`.
- **Each loaded module has its own memo.** The maps are inline function-local statics
  in a header, and Python dlopens extension modules with `RTLD_LOCAL`, so the prebuilt
  extension and every JIT-compiled kernel carry their own. Each amortizes over its own
  calls, which is all the fix needs; the cost is that a tensor used on both paths is
  narrowed once per module. `scorch_ops.abi_memo_clear()` clears the prebuilt
  extension's only.

**Why not compile the kernel for int64 and skip narrowing entirely?** It is a real
option — kernels are already compiled per format and dtype, so index dtype could join
the cache key — and it was measured, not argued. A 12-byte-per-nonzero A stream instead
of 8 costs **1.19–1.51x on DRAM-bound cells** (reddit@16 1.19x, reddit@32 1.21x,
inline_1@32 1.27x, scatter200@32 1.51x) and nothing on cache-resident ones. So it
trades a one-time O(nnz) cast for a permanent per-call bandwidth tax; with the cast
memoized, narrowing once wins for any tensor used more than once, which is every
training loop. Where a dtype-parameterized kernel family IS the right answer is tensors
that genuinely exceed int32 — today those fail closed with "cannot be represented as
int32" — dispatched on measured magnitude rather than on the caller's declared dtype.
Note also that scorch's own format conversions already emit int32: `to_sparse("ss")`
does, and only operands handed in by a user through `from_torch`/scipy arrive as int64.
That is the argument for narrowing at construction rather than at the boundary.

### What the JIT fix is worth

`bench/bench_codegen_abi.py` is the harness — two routes (`matmul_wksp` with int64
operands and a sparse result, which charges narrowing plus scans; DCSR × dense through
`matmul` with a dense result, which charges scans only), interleaved arms, an in-process
A/A floor, and `torch.sparse.mm` as the cross-tree control. Base is the tree at
`b4f8985`. redwood only: at the time this ran, the laptop could not compile a generated
kernel at all, so the numbers below are single-machine and should be read as such. That
was a toolchain defect, since fixed (see the macOS note below), so a second host for
this table is now possible and has not yet been run.

| route | matrix | M | nnz | N | before ms | after ms | gain | A/A floor | control |
|---|---|---|---|---|---|---|---|---|---|
| dcsr_dd | band | 100000 | 2.4M | 8 | 3.486 | 0.949 | **3.67x** | 42.3% | 1.9% |
| dcsr_dd | band | 100000 | 2.4M | 32 | 3.865 | 1.799 | **2.15x** | 13.7% | 21.9% |
| dcsr_dd | band | 100000 | 2.4M | 128 | 11.728 | 8.163 | **1.44x** | 15.6% | 13.4% |
| dcsr_dd | band | 20000 | 480k | 8 | 0.761 | 0.335 | **2.27x** | 9.8% | 24.2% |
| dcsr_dd | scatter | 100000 | 2.4M | 8 | 2.334 | 1.044 | **2.23x** | 25.3% | 39.1% |
| dcsr_dd | scatter | 100000 | 2.4M | 32 | 3.818 | 1.782 | **2.14x** | 14.8% | 69.4% |
| dcsr_dd | scatter | 20000 | 480k | 8 | 0.677 | 0.323 | **2.10x** | 10.1% | 0.5% |
| dcsr_dd | scatter | 20000 | 480k | 32 | 3.130 | 1.300 | **2.41x** | 16.6% | 36.6% |
| wksp_ds | (all 6 cells) | 20000 | 160k | 8–128 | 137.9–148.4 | 139.5–150.7 | 0.98–0.99x | 0.5–1.7% | 0.2–4.0% |

Geomean 1.527x over the 18 cells; **9 of 18 clear both their own A/A floor and the
cross-tree control spread**, and the ones that do not are dominated by noise rather than
by a small effect — the `dcsr_dd` route's floors run 9–42% because the kernel it emits
is itself variable at these sizes. Restricted to `dcsr_dd`, geomean is 1.899x
(1.04–3.67x). Max relative error against the float64 reference across every cell and
arm: 2.7e-07.

**The workspace route gains nothing, and that is structural, not a failure.** All six
`wksp_ds` cells land at 0.98–0.99x, consistently just below 1. Their calls take 138–151
ms, essentially all of it sparse-output assembly, against ~0.1 ms of index validation
for 160k nonzeros — so the fix is removing 0.07% of the call and the residual 1–2% is
inside the 0.2–4.0% cross-tree control spread on five of the six. Nothing about this
route is bandwidth- or validation-bound; it is bound by building a sparse result.

**`SCORCH_ABI_VALIDATE_MEMO=0` is not a safe fallback.** The escape hatch runs the
branchless screens with memoization disabled, and a third arm measured it: at 480k
nonzeros screens-only is 1.2–1.5x slower than the memo, and at 2.4M nonzeros it is
**4.3–6.4x slower** — slower even than the original serial validation it replaced (6.06
ms against 3.49 ms on band@8). A private thread team standing next to the kernel costs
more than the walk it saves. The variable is for diagnosing a suspected memo bug on small
operands, not for running production with the memo off.

Two caveats on that paragraph, both added after the fact.

*The mechanism was stated here as a team reshape — "a parallel screen spawns a team
immediately before the kernel wants a differently shaped one" — and that was never
measured.* What is measured, later in this document, is that the workers a private team
spawns keep costing time after the scan returns: a tensor copy standing next to a screen
went 121 → 1157 us, and `torch.matmul` on identical code in the same process ran
1.12–1.33x slower in the build with private teams. That is a spin-after-return cost, not
a construction cost, and nothing here separates the two. The numbers above do not depend
on which it is, so they stand; the explanation should not have been asserted.

*And that prediction was wrong.* This paragraph first said the memo-off ratios were
"expected to be smaller" once the screens moved onto torch's pool. Re-measured on the same
grid with the ATen screens, memo-off is still **1.76-3.18x** slower at N=8 (band 0.98 ->
3.11 ms, scatter 0.87 -> 1.68, pubmed 0.86 -> 1.52, bcsstk17 0.91 -> 1.74), 1.19-2.73x at
N=32, and within noise at N=128. The reason has nothing to do with threads, which is why
reaching for one was a mistake here: without the memo a full O(nnz) scan runs on *every
call* instead of once, and at N=8 the kernel it precedes is only about a millisecond. That
is simply what removing a memo costs. The A/A floor on the memo-off arm reaches 20%, so
treat the individual ratios as approximate; the direction is not in doubt at N=8 or N=32.

## Wrapping a matrix: the index validation

The call plan above ends the prebuilt route's Python tax. Asking the same question of
the generated route — is a warm `scorch.einsum` call also mostly Python? — found
something bigger and simpler, and not on the dispatch path at all.

`_validate_index_storage` runs on **every** `STensor` built over a compressed level:
`from_torch`, `from_csr`, `to_sparse`, a relayout, and the result of every generated
kernel. Its sortedness check — each parent's coordinates must ascend — was a Python
loop over parents, and each iteration sliced a tensor, launched a comparison kernel and
synced on `.item()`. That is **3.7 µs per row**, so 74 ms to wrap a 20,000-row CSR;
`torch.sparse.mm` on the same matrix takes 3 ms. And it ran **twice** per construction,
because `SparseStorage.__init__` validates and then `STensor._set_state` — which every
constructor and every in-place structural change funnels through immediately afterwards
— validated the same arrays again.

Two changes, both in `src/scorch/storage.py`:

1. **Vectorize the predicate.** Descents are allowed exactly at parent boundaries, so
   the whole check is two whole-array kernels: `coordinates[1:] < coordinates[:-1]`,
   then clear the entries that sit on a boundary (`positions[1:-1]`). Where it fails,
   the offending parent is recovered with one `searchsorted` — so the exception still
   names the same parent the loop named.
2. **Do not walk twice.** `SparseStorage.__init__` records a stamp of what it validated
   — `(data_ptr, _version, numel)` per index array and for the values — and
   `_set_state` skips the second walk when the stamp still matches. Where it does not
   match, the full check runs. The public `validate()` methods always re-run in full:
   an explicit call asks for the work.

### Both hosts, three arms, one process, one binary

Neither change needs a separate build, so both are switchable at runtime and the
switches are thrown outside the timed region. `loop2` is what shipped, `vec2` is the
vectorized predicate still run twice, `vec1` is the version that ships now. Arms are
visited in a fresh random order every round, the figure is the median of 9, and every
arm's result is compared against the others before any of them is timed.

| case | host | loop2 µs | vec2 µs | vec1 µs | total gain | of which the second walk |
|---|---|---|---|---|---|---|
| `from_torch` 128×4 | redwood | 920.3 | 79.5 | 53.2 | **17.3x** | 1.49x |
| `from_torch` 128×4 | M5 | 555.9 | 45.1 | 28.4 | **19.6x** | 1.59x |
| `from_torch` 1000×8 | redwood | 6,806.9 | 98.9 | 63.2 | **108x** | 1.56x |
| `from_torch` 1000×8 | M5 | 4,171.1 | 64.5 | 39.2 | **106x** | 1.65x |
| `from_torch` 20000×24 | redwood | 135,289 | 557.0 | 363.6 | **372x** | 1.53x |
| `from_torch` 20000×24 | M5 | 83,040 | 631.1 | 352.0 | **236x** | 1.79x |
| `from_torch` 100000×16 | redwood | 677,329 | 1,696.6 | 1,151.0 | **588x** | 1.47x |
| `from_torch` 100000×16 | M5 | 424,822 | 1,620.4 | 961.0 | **442x** | 1.69x |
| `to_sparse` "ds" 2000×2000 @1% | redwood | 23,833 | 3,218.1 | 2,924.7 | **8.1x** | 1.10x |
| `to_sparse` "ds" 2000×2000 @1% | M5 | 15,018 | 2,474.0 | 2,263.2 | **6.6x** | 1.09x |

The gain scales with rows because the cost it removes was per row: wrapping a
100,000-row CSR went from **0.68 s to 1.2 ms** on redwood.

**Where this does and does not show up.** It is a per-wrap cost, so it moves a workload
bar only when the wrap is inside the timed region. In the two standing harnesses it is
not: `bench_gcn.py` converts the adjacency to an `STensor` before `benchmark_fn` (see
the comment at line 507), and `bench_sparse_autoencoder.py` builds its `STensor` dict
before its timed loops. For those, this is a setup-cost win and must be reported
separately rather than folded into a workload comparison. It is steady-state only for
code that wraps inside its loop — a model that calls `from_torch`, `from_csr` or
`to_sparse` per forward pass, or any interactive use where wrapping a large matrix at
all was the thing that felt slow. `to_sparse` gains only
6.6–8.1x because most of its call is a generated kernel building a sparse result, not
validation — the same structural reason the workspace route gained nothing from the ABI
fix. Deduplicating the walk is worth a flat 1.47–1.79x on top of the vectorization on
`from_torch`, and 1.09–1.10x on `to_sparse`, on both hosts.

### A third rung: one native pass instead of five torch passes

Even vectorized, the check is five whole-array torch operations plus a bool temporary --
about 15 MB of traffic to validate 2.56 MB of indices, which at 640,000 nonzeros measured
**304 us of the 385 us** it took to wrap a generated kernel's sparse result. That is
bandwidth, not overhead, so the only way to get it back is to stop making the passes.

The pass already existed. `csrc/native_abi.h` has held a fused screen since the ABI
boundary was fixed on this branch: positions-monotonicity, coordinate bounds and
per-parent sortedness in **one loop**, thread-split, no allocation, using the same
descents-minus-boundaries observation the Python version uses. It was only reachable
from C++. Exposing it as `abi_screen_compressed_level` lets Scorch's own validator use
it, and the screen's existing contract is exactly what makes that safe: it answers "no
violation exists" or "go and look", never "this is fine" about something that is not.

So the Python checks remain the only thing that ever reports, and the three O(1) checks
around them -- position-array length, starts-at-zero, terminal-equals-nnz -- stay where
they were, which keeps message *precedence* identical too. On a clean level the three
whole-array walks are skipped because the screen has established they cannot fail; on a
suspect level they run exactly as before.

| case | host | loop2 | vec2 | vec1 | screen | this rung | cumulative |
|---|---|---|---|---|---|---|---|
| `from_torch` 128x4 | M5 | 486.9 us | 42.7 | 27.3 | **12.9** | 2.12x | **37.8x** |
| `from_torch` 128x4 | redwood | 943.6 | 78.7 | 51.9 | **26.6** | 1.95x | **35.4x** |
| `from_torch` 1000x8 | M5 | 3,678 | 61.1 | 36.8 | **15.9** | 2.31x | **231x** |
| `from_torch` 1000x8 | redwood | 7,034 | 98.2 | 62.1 | **29.1** | 2.13x | **242x** |
| `from_torch` 20000x24 | M5 | 73,500 | 627.2 | 367.9 | **152.2** | 2.42x | **483x** |
| `from_torch` 20000x24 | redwood | 139,398 | 504.3 | 272.6 | **187.8** | 1.45x | **742x** |
| `from_torch` 100000x16 | M5 | 377,457 | 1,771 | 985.4 | **391.5** | 2.52x | **964x** |
| `from_torch` 100000x16 | redwood | 694,491 | 1,208 | 1,070 | **376.3** | 2.84x | **1,846x** |
| `to_sparse` "ds" 2000x2000 @1% | M5 | 13,558 | 2,441 | 2,228 | **2,166** | 1.03x | 6.3x |
| `to_sparse` "ds" 2000x2000 @1% | redwood | 24,314 | 3,121 | 2,914 | **2,747** | 1.06x | 8.8x |

Four arms in one process, random order each round, median of 9, arms compared for
agreement before timing. `to_sparse` gains little for the same structural reason as
before: its call is mostly a generated kernel building a sparse result.

These are the *final* figures. As first measured the last rung was weaker and wildly
asymmetric between hosts -- 1.62x on the M5 against 1.09x on redwood, and a 100k-row wrap
that still cost 869.4 us there -- and chasing that asymmetry is what turned up the defect
below. Fixing it removed the asymmetry along with the cost: the same wrap is now 376.3 us
on redwood, the last rung is 2.5-2.8x on both hosts, and x86 has gone from the worse host
to the better one. The pre-fix numbers are not reproducible against this tree, which is
why they are quoted here rather than tabulated.

**The asymmetry above is what led here.** The screens sized their own worker count from
`SCORCH_ABI_VALIDATE_GRAIN`, one million nonzeros per worker, so at 1.6M nonzeros the scan
ran on one thread of a 32-core machine. Lowering that constant looked like the whole story.
It was not.

### A private thread team is not free, and the bill lands on the neighbours

Lowering the grain on redwood made the wrap *slower*: 5.6x at 480k nonzeros and 6.6x at
1.6M against leaving the scan serial. Two rounds of the sweep then disagreed with each
other by 5.5x on identical settings, which is the signature of a harness measuring
something other than what it names. Timing the scan alone -- on index arrays that already
exist, so no clone and no 13-25 MB allocation in the loop -- separated the two effects:

| redwood, 32 torch threads, 480k nonzeros | scan | the tensor copy next to it | total |
|---|---|---|---|
| screen serial | 138.5 us | ~121 us | 259.9 us |
| screen parallel, private team | **72.6 us** | **~1157 us** | **1229.4 us** |

The scan got 1.9x faster and the call got 4.7x slower. The screens were splitting their
work with `#pragma omp parallel for num_threads(nt)`, which builds a team that is neither
torch's team nor torch's width, and the cost of that lands on the code standing next to
it: the victim here is a copy that our scan never touches. What the measurements pin down
is *where* the time goes, not which of two mechanisms puts it there -- a team libgomp has
to reshape because we asked for a width torch never asks for, or workers that keep
spinning after the scan returns with nothing to do but compete for cores. Both are
consistent with every number here and nothing below separates them, so the claim is the
cost, not the cause. Because the sweep interleaved its arms, each arm inherited the
previous arm's spinners, so every arm was inflated and the ordering between them was
noise.

The fix is not a better constant. It is to stop building a private team: every screen now
splits with `at::parallel_reduce`, so the scan runs on torch's pool at torch's width.
PyTorch's own source documents this hazard from the other side --
`ATen/ParallelOpenMP.h` carries the comment *"can't use num_threads clause due to bugs in
GOMP's thread pool"* -- and reading it settles what `grain` now means: a scan shorter than
one grain stays serial, and past that the region opens at full width with work handed to
`min(team, ceil(nnz / grain))` of those threads. So the constant is a threshold, not a
worker count.

Removing the private team is worth more than any grain ever was, on the same cell:

| redwood, 4 torch threads, 1.6M nonzeros | private team | torch's pool |
|---|---|---|
| scan, serial grain | 974.8 us | **170.4 us** |
| scan, split | 578.4 us | **132.8 us** |

**The same defect was taxing the already-shipped ABI path, and the evidence for that is a
column I did not put there for this purpose.** `bench/bench_codegen_abi.py` reports
`torch_ms` -- the reference `torch.matmul`, identical code in both builds, timed in the
same process. Two rounds, alternating builds, at N=128 where the A/A floor is tightest
(0.04-1.5%):

| matrix | `torch_ms`, private team | `torch_ms`, torch's pool | torch slower by |
|---|---|---|---|
| band | 0.8642 / 0.8940 | 0.6687 / 0.6744 | 1.29x / 1.33x |
| scatter | 1.4715 / 1.5594 | 1.1490 / 1.1875 | 1.28x / 1.31x |
| pubmed | 1.4939 / 1.5290 | 1.1755 / 1.1799 | 1.27x / 1.30x |
| bcsstk17 | 1.5419 / 1.4665 | 1.1879 / 1.3052 | 1.30x / 1.12x |

Our validators were making *PyTorch's own kernels* 12-33% slower in the same process. The
generated-kernel time on that route improves 1.12-1.28x with the fix, but the `torch_ms`
column is the cleaner evidence, because nothing about it is ours. Elsewhere in this
repository the same mechanism has been paid for once already: matching
`torch.get_num_threads()` in the drop-in SpMM is what fixed pubmed (0.78 -> 1.15x,
commit e795127). That fix and this one are the same finding in two layers.

### Choosing the threshold, and a second thing the first attempt got wrong

With the teams gone, the remaining question is how long a scan has to be before splitting
it is worth anything. The sweep through `from_torch` cannot answer it -- that call clones
13-25 MB of index arrays at the sizes where the answer changes, and the same setting read
141.5, 784.0 and 418.7 us across three runs, a 5.5x spread against an effect of a few
percent to 3x. So the sweep moved onto `_validate_index_storage` over arrays that already
exist: no clone, no allocation, nothing in the loop but the thing being swept
(`bench/bench_index_validation.py --what scan`). The small cells held to 2% across all
three of the earlier runs, which is what placed the variance in the allocations.

That harness immediately showed the first attempt was leaving most of the win behind, for
a reason worth stating precisely, because it is a property of ATen and not of this code.
`at::parallel_reduce` splits when `n > grain_size`; it then opens `#pragma omp parallel`
over **the whole thread pool** and gives work to `min(team, ceil(n / grain_size))` of
those threads. The region is opened at full width whether one thread or all of them get
work. So a grain used as a worker limit buys nothing and costs the difference:

| M5, 4 torch threads, 98,751 nonzeros | grain as worker limit | grain as threshold only |
|---|---|---|
| workers given work | 2 of 4 | 4 of 4 |
| scan | 106.0 us | **77.2 us** |

Same region, same pool, 1.37x. The constant was doing two jobs and doing the second one
badly, so it now does one: below it the scan stays serial, above it every thread in the
region that was opened anyway gets a share (`abi_split` in `csrc/native_abi.h`).

**65536 nonzeros is the threshold, and it is the largest-win value that regresses
nothing.** Nine cells from 8k to 1.6M nonzeros, CSR and COO, at two torch thread counts
per host, each with a torch operation interposed so the measurement sits in the context a
wrap really runs in:

| M5 (4 / 8 torch threads) | nnz | serial | threshold 65536 | ratio |
|---|---|---|---|---|
| csr 1000x8 | 8,000 | 4.85 / 4.82 | 4.79 / 4.83 | 1.01x / 1.00x |
| csr 2000x10 | 20,000 | 8.22 / 7.95 | 8.28 / 7.97 | 0.99x / 1.00x |
| csr 4000x8 | 32,000 | 12.43 / 11.99 | 12.51 / 12.19 | 0.99x / 0.98x |
| csr 8000x8 | 64,000 | 31.15 / 46.64 | 32.34 / 47.01 | 0.96x / 0.99x |
| csr 20000x24 | 480,000 | 133.99 / 137.48 | **84.07 / 108.56** | 1.59x / 1.27x |
| csr 100000x16 | 1,600,000 | 221.95 / 233.42 | 218.88 / 232.76 | 1.01x / 1.00x |
| coo | 19,250 | 23.15 / 23.01 | 22.99 / 22.93 | 1.01x / 1.00x |
| coo | 98,751 | 115.74 / 127.44 | **77.17 / 119.37** | 1.50x / 1.07x |
| coo | 998,775 | 1028.96 / 1045.79 | **330.13 / 348.31** | 3.12x / 3.00x |

| redwood (4 / 32 torch threads) | nnz | serial | threshold 65536 | ratio |
|---|---|---|---|---|
| csr 1000x8 | 8,000 | 6.45 / 6.50 | 6.53 / 6.35 | 0.99x / 1.02x |
| csr 2000x10 | 20,000 | 9.04 / 8.86 | 8.83 / 8.95 | 1.02x / 0.99x |
| csr 4000x8 | 32,000 | 11.68 / 12.00 | 11.72 / 12.04 | 1.00x / 1.00x |
| csr 8000x8 | 64,000 | 20.46 / 31.91 | 20.47 / 32.43 | 1.00x / 0.98x |
| csr 20000x24 | 480,000 | 82.35 / 95.35 | **56.83 / 68.02** | 1.45x / 1.40x |
| csr 100000x16 | 1,600,000 | 184.38 / 155.58 | 183.78 / 156.86 | 1.00x / 0.99x |
| coo | 19,250 | 33.61 / 35.21 | 33.37 / 35.08 | 1.01x / 1.00x |
| coo | 98,751 | 161.82 / 182.87 | **55.36 / 77.68** | 2.92x / 2.35x |
| coo | 998,775 | 1640.95 / 1735.33 | **555.60 / 203.26** | 2.95x / **8.54x** |

Read those two tables carefully, because most of their rows are not comparisons at all --
and that is the strongest thing they say. **Any cell under 65,536 nonzeros runs identical
code in both columns**, since the threshold is what decides whether to split and neither
column splits below it. So the five sub-threshold rows -- 8,000 / 20,000 / 32,000 / 64,000
and the 19,250-nonzero COO -- are a same-code control, and their spread *is* this harness's
noise floor: 0.2-3.8% on the M5, 0.3-2.4% on redwood. The 3.8% at 64,000 nonzeros is the
largest number in either table that looks like a regression and it cannot be one.

That leaves four rows where the columns genuinely differ, and every one of them is a win:
1.45-1.59x at 480k nonzeros, 1.07-2.92x at 98,751 COO, and 2.95-8.54x at 998,775 COO. The
`csr 100000x16` row is the exception that proves the rule -- it reads flat because at 1.6M
nonzeros the "serial" column exceeds the *ABI* threshold and splits too, so it is
full-width against full-width.

**So the threshold cannot regress anything by construction**, and the only open question was
whether it is high enough. A lower one is not: 16384 costs 2.5-3.6x at 20,000 nonzeros and
4096 costs 3.0-5.7x at 8,000, because below the threshold the scan is a few microseconds
while opening a region is 3-13 us depending on pool width. Those are real regressions -- the
code genuinely differs there -- which is what fixes 65536 from below.

**One measured gap, left open rather than tuned away.** A single threshold in *nonzeros* is
a compromise across screens whose per-element work differs: the COO lexicographic screen
compares up to one coordinate per level per element and runs alongside a bounds screen per
level, so its scan costs several times what a CSR coordinate scan costs at the same length.
That moves its crossover down. At 19,250 nonzeros on redwood with four threads, a 4096
threshold reads 21.28 us against 33.37 us at 65536 -- 1.58x that a per-screen threshold
would capture and this one does not. It is one cell at one thread count, and a global 4096
would cost 3.0-5.7x at 8,000 nonzeros to buy it, so the right fix is a threshold scaled by
each screen's work per element, measured on its own grid. Not attempted here.

Two incidental notes. The screens are now templated over index width, because Scorch
keeps int64 indices when a caller hands them in that way and torch's CSR does exactly
that; the int32 entry points are preserved as delegating wrappers, so all eight existing
ABI call sites are unchanged. And an inline check in the *emitted* kernel would be
cheaper still -- the previous coordinate is already in a register there -- but the
assembly comes from the CIN lowerer, so it would change the emitted C++ for every
sparse-output kernel and invalidate the byte-identical-emission gate, forcing a
re-capture of every pinned corpus. That is not worth 80 us.

### COO was worse than any of it

The compressed path above sat next to something larger. The COO branch of the same
validator called `.tolist()` on every mode's index array and then iterated over every
**nonzero** in Python, building and comparing two tuples per iteration. The compressed
loop that started all this was per *row*; this was per *nonzero*, and it cost a flat
**0.365 us each**:

| nnz | host | Python loop | native screen | speedup | us per nonzero |
|---|---|---|---|---|---|
| 997 | M5 | 366.8 us | **14.5 us** | 25.3x | 0.3679 -> 0.0145 |
| 997 | redwood | 429.8 | **30.6** | 14.0x | 0.4311 -> 0.0307 |
| 9,988 | M5 | 3,639.9 | **32.0** | 113.7x | 0.3644 -> 0.0032 |
| 9,988 | redwood | 4,049.2 | **57.9** | 69.9x | 0.4054 -> 0.0058 |
| 99,812 | M5 | 37,174.5 | **165.7** | 224.3x | 0.3724 -> 0.0017 |
| 99,812 | redwood | 41,652.6 | **557.7** | 74.7x | 0.4173 -> 0.0056 |
| 399,222 | M5 | 148,450.3 | **406.8** | 364.9x | 0.3718 -> 0.0010 |
| 399,222 | redwood | 168,578.6 | **807.0** | 208.9x | 0.4223 -> 0.0020 |
| 998,775 | M5 | **371,594.3** | **841.7** | **441.5x** | 0.3720 -> 0.0008 |
| 998,775 | redwood | **426,684.6** | **3,041.0** | **140.3x** | 0.4272 -> 0.0030 |

Wrapping a million-nonzero COO tensor took **372 ms** on the M5 and **427 ms** on
redwood; it now takes 0.84 ms and 3.04 ms.

redwood's smaller factor is no longer the single-threaded-scan effect -- that is fixed --
and it is worth being clear about what it is instead, because it changes what the number
means. At these sizes the wrap is not scan-bound at all: `--what scan` times the same
validation at 55-78 us for 98,751 nonzeros and 190-556 us for 998,775, against the 558 us
and 3,041 us the whole `from_torch` costs. So four fifths of what is left is cloning and
allocating 8-16 MB of index arrays, and redwood is simply slower at that than the M5. The
validation is no longer the thing to optimize on this path; the copy is, which is what the
adoption of kernel-owned arrays below is about.

`abi_screen_lex` was already in `native_abi.h` for the same check, unexposed, and it makes
the same comparison the Python loop makes -- the first level that differs decides, every
deeper level then irrelevant. That is the part worth testing rather than assuming, because
a screen that consulted only the first level, or only the last, would pass a naive test:
so the suite pins a descent that only the second level can see, a case where level 0
decides and level 1 must not veto it, three-level ties, duplicates (not a descent, since
the Python comparison is `<` and not `<=`), and single-level COO. The coordinate-bounds
check went the same way, replacing a min reduction, a max reduction and two syncs.

All three screens release the interpreter lock around the scan, since it is O(nnz),
thread-split, and touches only tensor metadata and raw data.

### Why this is not a correctness risk

A faster validator that accepts different storage is not a faster validator. The old
implementation is kept **verbatim** as `reference_validate` in
`tests/test_scorch/test_index_validation_equivalence.py`, and 59 differential tests
compare `(exception type, message)` between the two over well-formed CSR/dense/COO/DCSR,
per-parent descent naming, first-offending-parent precedence, descents that straddle a
boundary, empty and single-entry parents, duplicate coordinates, malformed position
arrays, out-of-range coordinates, wrong dtypes, nnz mismatches, degenerate extents, and
a 24-seed fuzz sweep with four corruption modes. Three deliberately wrong versions of
the vectorized check were each caught by that file before this shipped.

The screen is held to the same standard, and by the same file: every one of those 59
comparisons now runs **twice**, once with the screen and once without, so any acceptance,
message or precedence difference it introduced would fail there. Further tests assert
only the direction that matters -- it never clears an unsorted parent (every parent, both
index widths), never clears an out-of-range coordinate (four bad values at four
positions), never clears broken positions, and never clears fuzzed corruption that the
Python validator itself rejects. One more pins the wiring, so this cannot quietly degrade
into "always take the slow path" and pass vacuously.

Twelve more tests cover the stamp: construction validates exactly **once** (it was
twice), an in-place write to any index array or a `resize_` of the values is still
caught, storage assembled without going through the constructor is validated from
scratch, and both public `validate()` methods still re-run the full check every time.
The stamp deliberately over-reports: writing a value back over itself moves the version
counter and buys a validation nobody needed, which costs speed and never correctness.

What the stamp cannot see is a write through a raw pointer or a numpy view, which bumps
no version counter. Neither could anything before it: validation happens when a tensor
is built or reassembled, and such a write happens in between without telling anyone.

### Why validate at all, and where to stop

Making the check cheap raised the prior question: why does Scorch validate index arrays
when PyTorch appears not to? It does appear not to, and it isn't so.
`torch.sparse.check_sparse_tensor_invariants` exists and is **disabled by default**, and
PyTorch says what disabling costs, in a warning that shows up in this repository's own test
output: *"Memory errors (e.g. SEGFAULT) will occur when operating on a sparse tensor which
violates the invariants, but checks incur performance overhead."* So PyTorch's position is
not that the checks are unnecessary; it is that they cost, that off is the default, and
that off means memory corruption.

The reasoning transfers directly. Our kernels take `data_ptr<int>()` and do unchecked
pointer arithmetic, so a coordinate past its extent is an out-of-bounds read or write in
C++, not an exception. Without the walk, a typo'd `indptr` handed to `from_torch` is a
segfault rather than a `TensorIndexError` naming the mode.

**What the checks were doing wrong was not existing -- it was not distinguishing who built
the array.** For a caller's arrays the walk is the only thing between a mistake and
corruption. For the index arrays of a *generated result* it re-derives what our own codegen
just established: every sparse output level comes from a `torch::empty` sized from a counted
extent and is filled by the kernel. That re-derivation was 35-41% of a `from_torch`-shaped
wrap, and on a sparse-result `einsum` it is worth this much of the **whole call**, kernel
included (M5, four torch threads, three arms in one process, random order, median of 9):

| case | host | nnz | copy + walk | adopt + walk | adopt (ships) | total | walk alone |
|---|---|---|---|---|---|---|---|
| ds 200x4 | M5 | 800 | 52.83 us | 51.39 | **46.25** | 1.142x | 1.111x |
| ds 200x4 | redwood | 800 | 92.01 | 87.57 | **81.62** | 1.127x | 1.073x |
| ds 2000x8 | M5 | 16,000 | 329.00 | 315.50 | **305.60** | 1.077x | 1.032x |
| ds 2000x8 | redwood | 16,000 | 152.37 | 143.39 | **133.09** | 1.145x | 1.077x |
| ds 20000x16 | M5 | 320,000 | 810.44 | 787.87 | **711.70** | 1.139x | 1.107x |
| ds 20000x16 | redwood | 320,000 | 1,276.69 | 1,242.17 | **1,225.85** | 1.041x | 1.013x |

Both hosts, 1.04-1.15x off the whole call. Skipping the walk is the larger half of it
everywhere except redwood's largest cell, where the kernel is 1.2 ms and both effects are
small against it.

So the line is drawn at provenance, following PyTorch's design rather than inventing one:

* **Caller-supplied arrays are always walked.** `from_torch`, `from_csr`, `TensorIndex`,
  `SparseStorage` with `mode_indices` -- no flag involved, in any build.
* **Generated results are walked only when `storage._VALIDATE_KERNEL_RESULTS` is on**,
  which `tests/conftest.py` turns on for the entire suite. Release skips it.
* **The cheap per-array checks always run**, trusted or not: dtype, rank, contiguity,
  device, one array per level. They are O(1) each, and they are what makes adopting a
  kernel's arrays safe at all -- a strided or wrongly-typed array would be misread by
  `data_ptr` arithmetic no matter who produced it.

The honest risk is that a bug in lowering, in the scheduler, or in a new codegen path emits
a malformed index that now reaches a kernel as raw pointers. The LoopIR migration is
actively changing that layer, which is exactly why the flag is on in CI rather than
removed: a codegen bug is then a structured error on whichever test first produces a bad
result, instead of a segfault on someone's machine. Four tests pin the arrangement in both
directions -- a malformed generated result raises with the flag on, constructs with it off
(otherwise the flag saves nothing), still fails the cheap checks with it off, and a
malformed *caller* array raises regardless.

### The macOS toolchain, which is why there are two hosts here

Every measurement in the previous section is redwood-only, because this laptop could not
compile a generated kernel **at all** — 208 tests failed on it, and `to_sparse` was
unavailable. The cause was one line in `get_extra_cflags`: it pointed clang at the
CommandLineTools SDK's libc++ headers by absolute path, while `xcode-select -p` on this
host is Xcode. The compiler torch invokes is then Xcode's clang, and handing it another
toolchain's libc++ fails inside the headers themselves — `reference to unresolved using
declaration` in `<__type_traits/is_trivially_copyable.h>`. A flag-by-flag bisect
isolated it: baseline, `-march=native`, `-ffast-math`, `-funroll-loops`, the OpenMP
flags and the Homebrew libomp include all compile; the CommandLineTools include fails
and the `xcrun --show-sdk-path` include succeeds.

The prebuilt extension was never affected, because `scorch_build.py` does not add this
flag — which is why the breakage looked for a long time like a codegen defect rather
than a toolchain one. The fix resolves the SDK once per process via
`xcrun --show-sdk-path`, falls back to the CommandLineTools path, and adds nothing at
all if neither has the headers (a consistent toolchain finds its own libc++ without
help; a wrong `-isystem` is worse than none). `SCORCH_MACOS_SDK` overrides it.

## A warm einsum, and the four constants it re-derived per call

Everything above is `scorch.matmul` and the prebuilt kernels. `scorch.einsum` is the other
front door -- the generic one, and the only one for SpGEMM, elementwise products and any
format combination the prebuilt table misses. On a hit in its dispatch cache the whole
scheduling pipeline is skipped, so the call should be four steps: look up the module, bind
the operands, run the kernel, wrap the result.

Four things in that sequence were being re-derived on every call and are constants of it.
Measured on the M5, on a 64x64 CSR times a 64x4 dense with a dense result -- a call whose
kernel runs in 1.6 us and whose Python was 25.8:

| what | cost | why it was there |
|---|---|---|
| dispatch key | 2.4 us | the key rendered each operand's layout with `json.dumps` |
| result wrap | 7.3 us | a dense result's index, layout and metadata rebuilt per call |
| label validation | ~0.8 us | `isascii()`/`isalpha()` per label of the expression, per call |
| index sizes | 0.7 us | the logical index -> size map built twice, to validate and to shape |
| declined relayout | 1.0 us | a relayout requested into the order the result already had |

**The dispatch key.** `_einsum_cache_key` called `layout.serialize()` per operand: 2.32 us
of `json.dumps` to obtain a key for an in-process dict that only ever compares keys for
equality. `TensorLayout` is a frozen dataclass over logical shape, physical shape, format,
permutation and index dtype, so it hashes and it discriminates on exactly those fields --
the JSON was never buying anything. The key now holds the layout itself, and a test pins
the equivalence field by field, including a level's `bit_width` two value objects down.
`_fill_value` is the one thing `to_dict()` carries that the key omits, and it is a
`ClassVar`, so it cannot differ between instances. `matmul_wksp`'s own module cache was
keyed the same way and is fixed the same way. 3.20 us -> 0.82 us for the whole key.

**The result wrap.** A generated *dense* result is described completely by its physical
shape, format, mode order, name, dtype and device: the index (every level dense, so every
per-mode array tuple is empty), the layout, and the metadata. Only the values tensor
differs between calls. Those three are now built once per key and shared, which is sound
for the reason `_DENSE_PARTS_CACHE` is sound -- each is a frozen value object whose only
`object.__setattr__` calls are inside its own constructor, and there are no index arrays to
write through. 7.26 us -> 3.54 us. A *sparse* result keeps the ordinary path: its index
arrays are different arrays every call, so there is nothing constant to hold but the
layout, which `TensorLayout.from_physical_shape` already caches.

The shortcut declines unless the format is all-dense **and** the kernel handed back no
index arrays. The second condition is not redundant. An all-dense format requires zero
arrays per level, and it is `_normalize_mode_indices` inside the ordinary path that
enforces that, so a shortcut inferring it from the format would turn a fail-closed check
into a silent one. A test drives a stray array into a dense level and requires the error.

**The labels and the sizes.** Validating an expression's labels is a pure function of the
expression string, so it is memoized -- but consulted in exactly the position the checks
ran in, after the operand count/None/type checks, so a malformed call still raises the same
exception it raised before. Two tests pin that precedence: `einsum("i&,kj->ij", B)` still
reports the operand count and `einsum("i&,kj->ij", None, B)` still reports the None. The
index-size map was built twice, once to validate and once to shape the result; the cached
path now reuses the first unless an operand was relayouted in between, in which case it
recomputes. A relayout permutes an operand's physical shape and its mode order together
and so cannot change a logical index's size either -- but there is no reason to depend on
that, and the test that says so exists because the comment claims it.

**The declined relayout.** The cached path ended with `if _temp_mo: change_mode_order(...)`.
Both orders are constants of the cache entry, and on the common path they are equal, so
this called a method that validated the permutation, took a defensive copy of the result's
own mode order and sorted it, all to discover there was nothing to do. Comparing the two
constants instead costs nothing.

### Both hosts, two trees, one binary

None of these can be swapped at runtime -- three of them are the absence of code -- so the
arms are two *trees* sharing one native binary and one JIT cache, run as subprocesses in a
fresh random order every round, with `base` twice as the A/A control. The estimator is the
minimum rather than the median because each arm is a fresh process and a median would
measure how each one warmed. `bench/bench_einsum_dispatch.py`.

M5, 4 threads, 5 rounds:

| cell | base us | cand us | cand/base | A/A |
|---|---|---|---|---|
| spmm 64x4 N=4 -> dd | 25.78 | 13.71 | **0.532** | 0.999 |
| spmm 64x4 N=64 -> dd | 27.25 | 15.08 | **0.554** | 1.009 |
| spmm 256x8 N=8 -> dd | 51.21 | 36.26 | 0.708 | 0.956 |
| spmm 256x8 N=64 -> dd | 51.63 | 37.04 | 0.717 | 0.977 |
| spmm 2000x8 N=8 -> dd | 101.17 | 84.83 | 0.839 | 1.006 |
| spmm 2000x8 N=64 -> dd | 163.07 | 147.76 | 0.906 | 1.002 |
| spmm 20000x16 N=32 -> dd | 310.40 | 293.21 | 0.945 | 1.006 |
| spmm 64x4 N=4 -> ds | 30.54 | 22.69 | 0.743 | 1.021 |
| spmm 256x8 N=8 -> ds | 88.95 | 79.38 | 0.892 | 1.012 |
| spmm 2000x8 N=64 -> ds | 445.07 | 437.00 | 0.982 | 1.002 |
| spgemm 64x4 | 34.16 | 25.93 | 0.759 | 0.997 |
| spgemm 512x8 | 316.47 | 305.94 | 0.967 | 1.001 |
| mul 64x4 | 26.77 | 18.91 | 0.707 | 1.000 |
| mul 2000x8 | 68.56 | 59.67 | 0.870 | 1.006 |
| **geomean** | | | **0.781** | 0.999 |

redwood, 4 threads, 5 rounds:

| cell | base us | cand us | cand/base | A/A |
|---|---|---|---|---|
| spmm 64x4 N=4 -> dd | 54.25 | 28.73 | **0.530** | 0.997 |
| spmm 64x4 N=64 -> dd | 58.56 | 32.26 | **0.551** | 0.997 |
| spmm 256x8 N=8 -> dd | 62.13 | 34.52 | 0.556 | 0.998 |
| spmm 256x8 N=64 -> dd | 72.45 | 44.89 | 0.620 | 0.984 |
| spmm 2000x8 N=8 -> dd | 76.63 | 48.57 | 0.634 | 1.003 |
| spmm 2000x8 N=64 -> dd | 107.21 | 76.93 | 0.718 | 0.996 |
| spmm 20000x16 N=32 -> dd | 298.36 | 264.05 | 0.885 | 1.003 |
| spmm 64x4 N=4 -> ds | 63.78 | 47.15 | 0.739 | 0.998 |
| spmm 256x8 N=8 -> ds | 85.81 | 68.40 | 0.797 | 0.997 |
| spmm 2000x8 N=64 -> ds | 249.74 | 232.07 | 0.929 | 1.029 |
| spgemm 64x4 | 64.20 | 48.03 | 0.748 | 1.000 |
| spgemm 512x8 | 154.84 | 134.71 | 0.870 | 1.030 |
| mul 64x4 | 55.86 | 38.93 | 0.697 | 0.985 |
| mul 2000x8 | 160.64 | 138.77 | 0.864 | 0.991 |
| **geomean** | | | **0.712** | 1.000 |

Nothing regressed on either host, and no cell sits inside its own A/A floor except the
largest sparse-result one on the M5 (0.982 against a 1.002 control -- call that flat). The
gain is largest where the fixed cost is most of the call, which is what a fixed-cost fix
should look like, and it does not vanish at the kernel-bound end: 0.945 on the M5's
20000x16 cell and 0.885 on redwood's.

### What is left, measured and not claimed

- A **sparse** result's wrap is still 9.4 us on the M5, of which 2.4 us is `TensorIndex`
  and 1.2 us `_finalize_generated_mode_indices`. Both are genuine per-call work on arrays
  that are new every call. The 0.8 us of `TensorMetadata` and 0.35 us of layout inside it
  are constants and could be shared the same way; that is ~5% of a small sparse-result
  call and is not done.
- Wrapping the dense right-hand operand costs 1.4-1.6 us per call. It is already served
  from `_DENSE_PARTS_CACHE`; what is left is the `STensor` assembly around the cached
  parts. Not building an STensor for a dense operand at all is a larger change to
  `einsum`'s interior.
- `einsum`'s own remaining bytecode -- kwargs handling, the schedule resolution, the
  operand type and rank checks -- is roughly 3 us of the M5's 13.6.

## A transposed dense operand, and the two ways not to pay for it

`SparseStorage` holds a flat contiguous values array, so `scorch.matmul(A, B)` with a
non-contiguous `B` -- `W.T`, `x.permute(1, 0)`, a strided slice -- has to materialize a
copy. For a contiguous operand `.reshape(-1)` is a view and the STensor shares the
caller's buffer, which is what it has always done. For a non-contiguous one it is a full
copy, paid again on every call even when the operand has not changed since the last one.

Two independent things reduce that bill, and they answer different questions, so both
shipped and both were measured crossed: **materialize it faster**, and **remember it**.

### Materializing it faster, and a defect that turned up on the way

A transposed 2-D float32 operand is column-major, and transposing the row-major view of a
column-major matrix *is* its contiguous copy -- the same floats in the same order, bit for
bit. Scorch already ships a cache-blocked transpose for that shape (AVX2 8x8 / NEON 4x4,
public as `scorch.fast_transpose`), so `.contiguous()` can be replaced by it exactly.

Doing that exposed a real defect in the shipped kernel. It launched with
`at::parallel_for(0, ncblk, 1, ...)`, and a grain of 1 is ATen for "split whenever there
is more than one iteration", so an 8 KiB transpose opened a thread team:

| operand (col-major) | elements | `.contiguous()` | kernel serial | kernel threaded |
|---|---|---|---|---|
| 256x8 | 2 048 | 0.74 us | **0.32** | 9.95 |
| 1024x32 | 32 768 | 10.12 | **3.84** | 11.91 |
| 4096x64 | 262 144 | 84.54 | 66.08 | **32.40** |
| 20000x784 | 15 680 000 | 9858 | 5665 | **2328** |

Opening the team costs ~10 us at four torch threads and ~22 us at eight on an M5, and
~2-3 us at four and ~18-20 us at thirty-two on the 32-core x86 -- against a serial
transpose of that same data in 0.3-1.3 us. Nothing was bought with it, because below the
threshold the serial kernel is not merely adequate: it is 2.0-7.8x faster than the
`.contiguous()` scatter it replaces.

So the kernel got a real grain, expressed in elements and converted to blocks because
ATen's grain is in units of the iteration space.

Choosing its value has one hard constraint. The kernel is **already shipped**, and
`fast_transpose` / `sparse_linear` call it with the host thread count, so before this every
shape with more than one column block ran threaded. Raising the grain can only move a shape
from threaded to serial, so any shape it moves the wrong way is a regression in a path that
has already been measured. Candidate rules were therefore replayed against both modes'
times over the 40-shape grid (`R` in {8, 32, 64, 256, 784}, `C` in {64...20000}) on two
hosts at two thread counts each -- 160 cells -- and scored first on how many shapes they
made slower than always-threaded, and only then on what they bought.

| rule | regressed, of 160 | worst | improved, per grid | geomean gain, per grid | max |
|---|---|---|---|---|---|
| fixed 65536 | 2 | 0.75x | 18, 18, 16, 18 | 2.55, 3.43, 1.35, 2.59 | 80x |
| fixed 131072 | 5 | 0.55x | 19, 21, 16, 21 | 2.59, 3.56, 1.31, 2.68 | 80x |
| 8192 x threads | 1 | 0.69x | 15, 18, 15, 25 | 2.42, 3.43, 1.37, 2.72 | 80x |
| **max(32768, 4096 x threads)** | **0** | **1.00x** | 15, 15, 15, 21 | 2.42, 3.13, 1.37, 2.68 | 80x |

(Grids in the order M5 at 4 and 8 threads, x86 at 4 and 32.)

The floor is the work below which no pool size is worth a launch; above it the threshold
grows with the pool, because the launch cost does. The rules that score better on average
do buy a little more -- three or four extra shapes per grid, and a slightly higher geomean --
and they pay for it by making two to five shapes up to 1.8x slower than what already
ships. That is not a trade this gets to make, so the zero-regression rule wins and the
difference is recorded rather than argued away.

Three things about the choice are worth stating rather than hiding:

* **Element count cannot separate every case.** `784x256` and `256x784` are the same
  200 704 elements and want opposite modes, because the block geometry differs. So the rule
  does not pick the better mode everywhere. Three M5 shapes stay threaded where serial
  would have been 2-6x faster: at four threads they land within a few percent of
  `.contiguous()` (0.92-0.98x), and at eight they fall to 0.61-0.89x of it. Each of them was
  *worse* before this change, so that is forgone upside rather than a regression, and it is
  left on the table rather than bought with one.
* **The two hosts disagree, for a reason.** Per thread, opening the team costs ~2.7 us on
  the M5 and ~0.6 us on the x86 -- a 4.5x difference -- so the ideal threshold differs by
  host and no portable constant serves both. Closing that would mean calibrating the launch
  cost once per process, which is a piece of work rather than a line, and is not done here.
* **The zero-regression claim rests on a same-run replay**, where both modes are timed in
  one interleaved process and the rule is then applied to those numbers. Comparing a fresh
  run of the new build against the old grid instead shows nine "regressions" of up to 0.77x
  -- every one of them at a shape where the rule picks the *same mode as before*, i.e. runs
  the same code. Sub-10 us cells drift up to 31% between runs on that host. The controlled
  comparison is the one to believe.

The grid deliberately includes `C = 784`, the orientation `sparse_linear` hands the kernel
(`fast_transpose` on an `[batch, 784]` input), and that path is where inertness matters most
because its numbers are already recorded. Against the always-threaded kernel it shipped
with, at eight threads on the M5:

| batch | always-threaded | now | |
|---|---|---|---|
| 8 | 21.91 us | 0.62 | **35x faster** |
| 32 | 22.38 | 2.07 | **10.8x faster** |
| 64 | 23.22 | 24.00 | same mode, so same code |
| 256 | 26.61 | 29.40 | same mode, so same code |
| 784 | 82.77 | 86.21 | same mode, so same code |

Two batch sizes get an order of magnitude and the other three run exactly the code they ran
before -- the differences on those rows are run-to-run drift, not the change. So the AE path
is improved or inert, measured rather than assumed. What it is *not* is optimal: at batch 64
the kernel stays threaded at 24 us where serial would take 4.1 and `.contiguous()` takes
14.7. That was true before this change too, and fixing it is what the per-host calibration
above would buy.

### And then the copy has to be serial, for a reason that is not about the copy

Inside `matmul` the threaded kernel is a disaster, and not because it is slow. On the
x86 host, with a `2000x256` operand:

| | |
|---|---|
| threaded transpose, alone | 19.07 us |
| scorch SpMM on an already-flat B, alone | 47.38 us |
| torch `.contiguous()` + scorch SpMM | 321.18 us |
| **threaded transpose + scorch SpMM** | **2681.01 us** |
| serial transpose + scorch SpMM | **181.57 us** |

The pair costs forty times the sum of its parts. It is not the transpose: two threaded
transposes back to back cost 65.60 us. It is not threading as such: a threaded transpose
followed by `torch.sparse.mm` costs 423 us against 361 for the pieces. What is expensive
is an **ATen parallel region opening immediately before scorch's own team** -- the same
neighbour effect the validation screens ran into (see the private-team section above), an
order of magnitude larger.

The serial path leaves no region behind, and it still beats `.contiguous()` on 39 of the 40
grid shapes -- 2.0-7.8x on the small ones, never worse than 0.95x -- so that is what this
call site uses. What it gives up is the threaded win on very large operands, where the
neighbour cost is a smaller share of a much bigger copy and threading might well pay. That
crossover is left open: closing it means understanding the interaction, not picking a
second threshold against it. `fast_transpose` and `sparse_linear` are untouched and still
pass the host thread count.

### Remembering the copy

The second lever holds the copy, keyed on the identity of the *base* tensor rather than the
view -- the view is a fresh object per call (`W.T` inside a loop) while its base is the
parameter that persists. Three things must hold for a hit:

* the weak reference still resolves to the same base object, which is why a reused
  allocator address cannot fool it,
* the base's version counter is unchanged -- torch shares one counter between a tensor and
  every view of it, so any in-place torch write through any of them is a miss,
* the copy's own counter is unchanged, because the copy is handed out as the STensor's
  values and a write through those would otherwise be served to the next caller.

What none of that sees is a write through a raw pointer or a numpy view, which bumps no
counter. Nothing in Scorch sees those and neither does autograd, but here the consequence
is a stale *value* rather than a skipped check, so it is said plainly rather than buried:
an operand mutated behind torch's back and then multiplied again reads as unchanged.
`SCORCH_MEMO_OPERAND_COPY=0` turns it off in the same binary, which is also how it is
measured.

A memo that cannot hit is a tax on every call, and `plan.py` met this problem first. An
operand refilled in place every call -- a dataloader's buffer -- paid **1.11x** on the
smallest cell for a lookup that could never pay. Two fixes, both borrowed from there:
after eight consecutive *stale* misses the memo stops being consulted, and the retained
copies are released with it, because blocks the caching allocator cannot recycle were worth
a further 1.6-5% on the small cells. What is counted is the stale miss -- key present,
version moved -- and not the cold miss, because a deep model's first forward is all cold
misses and it is exactly the workload the memo exists for. Any hit resets the streak, so a
weight updated once per step and multiplied ten times within it never withdraws.

### Both levers, both hosts, one process, one binary

Both are runtime cells, so every arm here is the same binary and the same tree, visited in
a fresh random order every round, with `plain` under a second name as the A/A control. Two
scenarios, because they separate what the levers do: `stable` reuses one base (the memo
hits), `changing` writes the base in place before each use (the memo always misses, and
only the faster materialization can help). `bench/bench_operand_copy.py`.

M5, 4 threads, median of 9:

| scenario | cell | plain | A/A | xpose | memo | **both** |
|---|---|---|---|---|---|---|
| stable | 256x64 | 40.35 | 40.25 | 38.11 | 34.82 | **34.71 (1.16x)** |
| stable | 2000x64 | 139.38 | 139.88 | 106.99 | 88.13 | **87.72 (1.59x)** |
| stable | 2000x256 | 426.68 | 429.11 | 309.29 | 171.85 | **173.58 (2.46x)** |
| stable | 20000x32 | 636.32 | 626.14 | 429.75 | 261.14 | **254.06 (2.50x)** |
| changing | 256x64 | 47.48 | 46.79 | 44.25 | 46.60 | **44.46 (1.07x)** |
| changing | 2000x64 | 177.91 | 179.45 | 142.67 | 176.25 | **149.57 (1.19x)** |
| changing | 2000x256 | 476.02 | 482.42 | 382.18 | 481.05 | **382.04 (1.25x)** |
| changing | 20000x32 | 683.43 | 684.10 | 496.75 | 694.64 | **497.34 (1.37x)** |

redwood, 4 threads, median of 9:

| scenario | cell | plain | A/A | xpose | memo | **both** |
|---|---|---|---|---|---|---|
| stable | 256x64 | 32.23 | 32.28 | 28.18 | 19.01 | **19.06 (1.69x)** |
| stable | 2000x64 | 101.60 | 102.40 | 70.36 | 35.76 | **35.71 (2.85x)** |
| stable | 2000x256 | 335.55 | 336.47 | 204.90 | 61.41 | **61.28 (5.48x)** |
| stable | 20000x32 | 459.40 | 458.72 | 259.75 | 128.44 | **128.92 (3.56x)** |
| changing | 256x64 | 38.64 | 38.71 | 34.78 | 38.69 | **34.81 (1.11x)** |
| changing | 2000x64 | 113.23 | 111.56 | 81.47 | 111.66 | **83.25 (1.36x)** |
| changing | 2000x256 | 429.52 | 429.47 | 264.91 | 435.41 | **269.97 (1.59x)** |
| changing | 20000x32 | 532.90 | 531.87 | 289.69 | 535.26 | **289.86 (1.84x)** |

Nothing regressed on either host. The A/A control sits within 0.0-1.5%. The memo-only arm
lands within a few percent of `plain` in the `changing` scenario, which is the withdrawal
working: it is what a lever that cannot pay is supposed to cost.

One thing the first version of this bench got wrong, recorded because it is an easy
mistake: the `changing` scenario originally cycled a pool of 32 operands against a memo
bound of 16, on the theory that a pool larger than the bound would always miss. It does
not -- the first sixteen keys stay resident, so exactly half the calls hit and the arm
read as a partial win. Writing the base in place is what actually makes it miss.

### What is left on the table here, measured and not claimed

- **The threaded transpose for very large operands.** Serial gives up 2-3x of the copy
  itself at 15M elements. Whether threading wins there once the neighbour cost is included
  is unmeasured; it was measured at one shape only (`2000x256`), where it loses badly.
- **A per-process launch-cost calibration** would let the grain suit both hosts instead of
  taking the intersection. The per-thread cost differs 4.5x between them, which is the whole
  reason a portable constant leaves 2-4x unclaimed at three M5 shapes.
- **The memo's first few dozen calls** on a call site it cannot serve still pay for the
  lookup before the withdrawal fires. Bounded and one-off, but not zero.
- **`fast_transpose` is not uniformly better than `.contiguous()`** after the grain fix --
  three shapes on the M5 sit within a few percent of it rather than above. They were far
  worse before, so this is unfinished rather than regressed.

## Scope and gaps, stated plainly

- **dtype.** Everything above is float32 except the float64 section below. float64 CSR
  × dense resolves a different prebuilt symbol (`prebuilt_spmm_csr_f64`) and **gets no
  tiling at all**, but it goes through the same `bind_binary_kernel_with_tile` →
  `validate_binary_inputs` → `checked_csr_view` path, so it pays the same tax and
  receives the same fix. Measured, not asserted — and the answer is that the fix helps
  float64 by 1.23–3.17x but leaves it **below MKL on 5 of 6 redwood cells** (0.853x
  pooled). The headline claim in this document is float32 only.
- **Index memory.** The narrowing memo holds an int32 copy alongside the caller's
  int64 arrays, i.e. +50% on index memory. It replaces a same-size allocation that was
  happening every call, so peak does not grow, but steady state does. An entry is
  dropped as soon as its source array dies — the sweep runs before every insert, not
  only when the map fills, because otherwise 4096 dead graph-scale index arrays would
  accumulate. Narrowing at `STensor` construction and dropping the int64 arrays would
  instead *halve* index memory and remove the memo entirely; that lives in
  `stensor.py`/`storage.py`. See the JIT section for why narrowing beats compiling the
  kernel for int64.
- **Only the drop-in float32 CSR×dense prebuilt symbol is tiled.** Fused bias/act
  SpMM, fused sparse Linear, SpMV and SpMSpM are unaffected by the selector, though
  they do all benefit from the validation fix.
- **The dispatch levers are measured on redwood only.** The M5 numbers that led to them
  are in the diagnosis above, and a local A/B was attempted and thrown away: a container
  VM at ~50% of a core doubled the kernel arm. The levers are pure Python with no
  platform-dependent behaviour, but "confirmed on both hosts" is not something this
  document can claim for them.
- **The dense-operand cache is bounded at 512 entries and never evicts.** The key is
  `(shape, dtype, device, name, mode_order)`, so a program that calls `matmul` with a
  dense operand of a new shape every time — a variable-length batch dimension, say —
  fills it and then stops caching, falling back to the ordinary construction rather than
  degrading. What it holds is a layout, a metadata record and an empty index tuple per
  key, no values, so the ceiling is kilobytes.
- **The grid, the selector re-measurement and the float64 table all predate the
  dispatch levers.** They were measured at `14e3ea6`/`6eec90f`; nothing in them is
  invalidated, but every small-cell ratio in them now understates scorch by ~33 µs of
  per-call Python.
- **Call plans cover one operation.** `SpmmCsrPlan` serves CSR × dense with a dense
  output — the drop-in float32 SpMM and the float64 reference kernel, plus the two tiled
  kernels the selector can choose. SpMV, SpGEMM, COO operands, sparse outputs, the fused
  bias/act and Linear kernels and everything on the generated-kernel route get no plan and
  are byte-unchanged. The plan machinery costs them a dict probe on a `__dict__` that has
  no plans in it.
- **Lever 5 is measured on both hosts**, unlike levers 1–4: its base tree is small enough
  to reproduce locally (`git archive HEAD` built in place), which the earlier levers'
  container-VM problem had made impossible. Its tests run on both too — 70 on redwood, 68
  + 2 skipped on the M5, the skips needing a generated kernel that toolchain cannot build
  — including the bitwise comparison against every legacy symbol a plan can stand in for.
  What is *not* two-host is everything above lever 5 in this document.
- **A defect found along the way and left alone, because it is not this work's.**
  `scorch.matmul` raises on a CSR operand whose mode order has been permuted — the shape
  says (48, 32) while the stored indices are still the original row-major CSR, and prebuilt
  resolution keys on the format string and the rank, never on the mode order, so it picks
  the row-major kernel and that kernel's ABI guard rejects the shapes it is handed. It
  fails closed, but with a kernel-level `RuntimeError` rather than a structured scorch
  error, and it does so identically on both hosts and on the tree before any of this work.
  Fixing it means teaching prebuilt resolution about mode order, which is a change to
  release dispatch; it is recorded here rather than folded in. A test pins the current
  behaviour so plans cannot quietly change it.

## Method

- Random-permutation interleaved arms within a process, median over 11 rounds, with an
  A/A control arm running level `off` under a second name — the in-process noise floor.
- Base and candidate are different `.so` files and so cannot be interleaved in one
  process. For the cross-tree comparison the **MKL arm is the control**: it is
  byte-identical in both trees, so `|mkl_base/mkl_cand − 1|` is that cell's
  cross-process floor. Trees alternate base, cand, base, cand so drift hits both.
- Compared against the **faster** of MKL's int32-index and int64-index arms.
- For the dispatch A/B, where MKL is not in the picture, the control is the harness's
  own **direct `spmm_csr_float_v2` call** with the argument list built once up front:
  identical C++ in both trees, so its cross-tree agreement per cell says whether that
  cell is trustworthy. Seven of nine agreed within 5%; the one that disagreed by 37.7%
  is called out rather than averaged in. `torch.sparse.mm` runs in the same process as
  the external reference point.
- Every cell checked against a float64 reference.
- Machine verified quiet before each run. This matters more than it sounds: a leftover
  `addr2line` pinning a single core added a flat ~8.6 ms to every arm of every cell,
  because with 32 OpenMP threads on 32 CPUs one preempted worker stalls the whole join
  barrier for a scheduler timeslice. It looked exactly like a uniform regression.
