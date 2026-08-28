# CSR × dense SpMM vs MKL — before and after

*Branch `perf/spmm-beat-mkl`, based on `04f321d`. The grid below was measured at
`14e3ea6`; `6eec90f` fixed the selector, `d9450ca` extended the same fix to the JIT
codegen path, and `ba59040` cut the Python dispatch cost the grid was still paying —
all three land after the grid was taken. Hosts: **redwood** (Intel i9-14900K, 8 P + 16
E cores, 36 MB L3, PyTorch 2.5.1 + MKL 2022.1) and **M5** (Apple, 6 P + 12 E cores,
PyTorch 2.13.0).
**Every headline number is float32 through the prebuilt route.** float64 now has its
own section and its own, equally strong, x86 conclusion — a register-resident kernel
above MKL on every cell — but it is a separate measurement on a separate kernel
instantiation, and on ARM it is deliberately unchanged. The JIT codegen path has its own
section and its own, weaker, conclusions. Read `SPMM_BEAT_MKL_PHASE0.md` first for the
attribution that led here.*

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

float64 CSR × dense used to resolve `prebuilt_spmm_csr_f64` → `spmm_csr_double` →
`spmm_csr_typed_core<double>`: the plain reference kernel, with no register-blocked row
kernel and no tiling route. Removing the validation tax sped it up, but only exposed
what the tax had been hiding — a kernel-quality gap. float32 beat MKL on the same cells
because float32 resolved a different, much better kernel.

The fix is the obvious one done properly: **make the good kernel generic over the scalar
type** rather than write a second one. `spmm_csr_float_v2_core` became
`spmm_csr_v2_core<scalar_t>`, and the three AVX2 row kernels
(`scorch_spmm_row_regblock`, `scorch_spmm_row_regtile_partial`,
`scorch_spmm_row_regtile`) became templates over a `scorch_simd<T>` traits struct
supplying `zero / splat / load / maskload / store / maskstore / fma / add / lane_mask`
and a `lanes` count — 8 for `__m256`, 4 for `__m256d`. Every lane count in those kernels
had been the literal `8`; each became `V::lanes`, so the register-blocking threshold and
the wide tile now scale with the type instead of being pinned to float. Narrow-k
register blocking is k ≤ 32 for float and k ≤ 16 for double, and the wide tile is 64
float or 32 double elements — in both cases the same register budget, which is the point
of expressing it in lanes. float32 reaches the template through a non-template
forwarder, so `spmm_csr_float_v2` and every caller of it are textually untouched.

x86, **129 cells** — the whole pinned corpus, 26 matrices (GCN, SuiteSparse and
synthetic) × N = 8…128, minus the cells a 3 GB working-set cap skips — 9 rounds, three
arms plus an A/A control drawn in a fresh random permutation every round, one draw per
arm, all in one process against one binary. The new kernel is called with the thread
count and launch mode dispatch actually passes: the harness asks
`ops._composition_hints` (`nthreads_override=24, atparallel=True`) rather than naming a
value, and refuses to run if production resolves a symbol other than the one it times.

| | geomean vs MKL | min | max | cells below MKL parity |
|---|---|---|---|---|
| old kernel (`spmm_csr_double`) | 1.296 | 0.549 | 3.099 | **45 of 129** |
| new kernel (`spmm_csr_double_v2`) | **2.273** | **1.086** | 7.647 | **0 of 129** |

The old kernel was up to **1.8x slower than MKL** (`syn__aeshape@16`, 0.549) and lost on
a third of the corpus. The new one is above MKL on every cell measured.

New against the kernel it replaces: geomean **1.754x**, min 0.998, max 3.848. Two of the
129 cells read a hair *below* 1.000 — `ss__mouse_gene@8` at 0.998 against its own A/A of
1.022, and `gcn__reddit@8` at 0.999 against 1.008 — so the honest claim is not "faster
everywhere" but **"faster everywhere outside the noise floor, and tied on two cells"**.
Both are inside their own controls by a wide margin, and both are in the same regime as
the other tight cells.

The A/A control ran 1.000–1.317, with three cells above 1.10. Screening the grid by each
cell's own control barely moves the result, which is why it is computed: 2.273 over all
129 cells, 2.265 over the 126 with A/A ≤ 1.10, 2.248 over the 120 with A/A ≤ 1.05 — and
zero cells below MKL parity in every subset. The noisy cells (`gcn__pubmed@64` at 1.317,
`ss__consph@32` at 1.124, `gcn__citeseer@32` at 1.108) should not have their individual
ratios quoted, but they are not what the headline rests on.

The weakest cell is worth naming because it is the one to push on: `gcn__reddit@128` at
**1.086x** of MKL on a control of 1.004 — a real 1.086, not a floor artefact. It sits
with the other tight cells (`gcn__reddit@8/@16/@64`, `ss__mouse_gene@8`,
`ss__nd24k@8`, `syn__scatter16@8`) in a coherent regime: 114.8M and 29.0M nonzeros at
narrow k, where the reference kernel is already bandwidth-bound on the sparse operand
and there is almost no output traffic for a register-resident row to save. New/ref on
those cells is 0.998–1.018 — the register kernel neither helps nor hurts when the thing
it removes is not the bottleneck. Everywhere the output round-trip *does* dominate, it is
worth 1.4–3.8x.

### What the first version of this table got wrong

Recorded because the error was invisible and the number it produced looked fine. The
first grid reported geomean 2.573 over 60 cells. It passed `atparallel=False` to the new
kernel while dispatch ships `atparallel=True` — two different launch paths, `at::parallel_for`
on torch's intra-op pool versus a private OpenMP team, with different core counts on a
hybrid P+E part. So it measured a configuration nobody runs, and being a *harness*
constant rather than a production one, nothing could disagree with it. The fix is that
the harness no longer states the policy: it calls the same `_composition_hints` dispatch
calls, and it hard-fails if resolution names a symbol it is not timing. The corrected
number is lower — 2.273 against 2.573 — over more than twice the cells, and the count of
cells the *old* kernel lost on went from 19 of 60 to 45 of 129.

### float32 is untouched, and that is proved statically

The templating rewrites the source of the kernel that carries every float32 headline in
this document, so the first thing to establish is that float32's machine code did not
move. Both `.so` files were disassembled and compared symbol by symbol, with addresses,
immediates and branch/call targets normalized — a renamed callee is not a code change —
and with the set of callee *names* reported separately on each side, so a genuine
retarget could not hide behind that normalization. Baseline is the last build before this work
(`7f88c18`; the commit after it, `8a8e6af`, is docs-only, so the C++ is the same).
Rather than infer that, every file under `src/scorch/csrc/` was compared between the two
trees: exactly three differ — `spmm.h`, `plan.h`, `ops.cpp` — and they are the three this
work touches.

**13 of 14 symbol pairs are instruction-identical**, including the one that matters
most: the per-row work, which the compiler inlines into the chunk lambda, is identical
at **996 instructions**. The register row kernels were templated and their float32 code
did not move a single instruction. Also identical: `spmm_csr_linear_fused_float` (539),
`spmm_csr_float_tilej_core` (434 body, 422 in its OpenMP region), and
`spmm_csr_float_tileijk_core` (325 body, and 262/153/423/262 across its four regions).
The callee-name differences are precisely the template renames —
`scorch_spmm_row_regblock<4, true>` becoming `<float, 4, true>` — matching one to one.

The one pair that differs is the once-per-call **setup body**, 581 instructions against
594. It diverges at instruction 264, where the register allocator picks `%edx` instead
of `%r8d`, and the substitution then cascades: 99 lines read differently for 13 more
instructions in total. This is the block that validates arguments, allocates the output
tensor and computes the thread count and chunk width, so thirteen instructions there are
not a cost anything can measure — but it is the only thing that changed, so it is the
only thing that needs measuring, and the small-cell dispatch grid is where it would show
first.

Measured, on the grid where per-call Python is a large share of the call:
`bench_dispatch_overhead.py --dtype float32`, one identical harness copied into both
trees so the harness itself is not a variable, the same arm set on both sides, and
**base, cand, cand, base** passes — position-balanced, because cross-run drift on this
host is far larger than the effect being bounded. Median dispatch overhead
(`matmul − kernel`) came out **−0.4 and −0.5 µs on base against −0.5 and −0.5 µs on
cand**: indistinguishable at the resolution of the measurement.

Do not read the per-cell numbers in that run. `20000x24@128` moved from 365.8 to 583.1 µs
*within the same tree* across passes — a 1.6x swing on identical code — which is the
redwood cross-run drift this document measures elsewhere (p95 1.398, max 4.13x). The
per-cell A/A floors in that run span 0.22–11.85% and bracket every tree-to-tree
difference in the table. The median over cells is the only statistic that survives, and
it is flat.

### An estimator that gave one arm two draws

Third finding in this section, and the one most likely to recur, so it is written down
rather than just fixed. The harness times `[mkl, ref, new, new]` — `new` twice, because
the second entry is the A/A control — and it reported `new` as `min` of the two. That
gives `new` twice as many draws as `mkl` and `ref`, and a minimum over more draws is
biased low. The bias flatters exactly the arm the document is arguing for.

It was caught by the ARM run, which is a control in a stronger sense than intended: off
AVX2 the `new` and `ref` arms are the *same machine code*, so the true ratio there is
exactly 1.000 and any departure is the harness. What the M5 showed, over 69 cells:

| estimator | geomean new/ref | median | cells >5% fast | cells >5% slow |
|---|---|---|---|---|
| `min` of two draws | 1.0068 | 1.0000 | **10** | 2 |
| one draw per arm | 0.9917 | 0.9970 | 5 | 8 |

The 10-against-2 lopsidedness is the tell: identical code cannot produce it, and a
doubled draw count can. With one draw per arm it becomes 5 against 8, which is a
coin-flip.

The bias was also bounded independently of that host, since each cell's A/A pair *is*
the two draws: a min-of-two beats a single draw by at most `A/A − 1`, on average about
half of it, which on redwood is an expected **1.18%** against an upper bound of
**2.35%**. Deflating every redwood cell by a flat 3% — more than the bound — left
geomean 2.218, minimum 1.054 and **zero cells below MKL parity**, so no verdict ever
depended on it. Both grids above were nonetheless re-taken with the corrected estimator.

The same ARM run then gives something worth keeping: the **harness's own accuracy**, on
cells where the answer is known to be exactly 1.000. At 9 rounds the deviation is
**±1.4% at the median, ±7.4% at p90 and ±25.9% at worst**. So a single M5 cell's ratio
means nothing finer than about ±7%, while the geomean over 69 cells is good to roughly
±1%. That is the right prior to read every per-cell number in this section with.

### ARM float64 keeps the reference kernel here, and one sentence in this section was wrong

`spmm_csr_double_v2` is guarded on `__AVX2__ && __FMA__` and falls back to
`spmm_csr_typed<double>` — the same reference kernel float64 already resolved —
everywhere else, so this change leaves ARM's float64 exactly as it was. The M5 confirm
reads new/ref **0.992** over 69 cells (A/A control 1.000–1.200): unchanged, as intended,
and the 0.8% shortfall from 1.000 is the harness's accuracy rather than a difference in
the code, which off AVX2 is the same code by construction.

An earlier version of this section gave the wrong reason for that fallback. It said NEON
has no masked load or store, so a register-resident row kernel would need a different
strategy for the ragged tail before double could be ported to it. The premise is true and
the conclusion does not follow. This file already contained a NEON register kernel —
`scorch_spmm_row_neon_regtile`, written for the fused Linear kernel — and it handles the
ragged tail with scalar accumulators updated in the same pass over the row: no masks, no
overread, nothing to invent. The real reason every ARM row went through the workspace loop
is duller. `spmm_csr_v2_core`'s register path was written under `#if defined(__AVX2__)`
and nobody had put a NEON arm next to it. That is a wiring gap, not a portability barrier,
and it is being fixed separately from this change.

Two figures from an earlier attempt at ARM float64 are worth keeping, because they say
what not to do. Routing float64 through the generic non-AVX2 path on the M5 gave geomean
**1.642x** over the reference across the M5 grid, with one regression: `gcn__pubmed@k=8`
at **0.689**, i.e. 1.45x *slower*. Two attempts to remove it both failed:

- single-tile direct accumulation with an assigning first nonzero — 0.681, no change;
- bulk zeroing on the non-AVX2 path plus pure in-place accumulation — **worse overall**,
  geomean 1.642 → 1.267.

Both were reverted. Shipping a 1.45x regression on a real GCN shape to collect a 1.64x
geomean is the trade this project's performance convention exists to refuse. Those two
figures also carry the same caveat as the x86 table above: they were taken with
`atparallel=False`, and on the M5 that is the launch mode that gets all 18 cores rather
than the six P-cores `at::parallel_for` hands out. They are not re-derived, because they
describe a path this tree does not ship.

One fact about the hosts, because it bounds how much confirmation any ARM claim in this
document can get: **MKT is x86_64, not ARM.** Slurm's `sinfo` reports `Arch=x86_64` for
the allocation, so of the machines this project measures on, exactly one — the M5 — can
execute a NEON instruction. Two-host confirmation is available for every x86 claim here
and for no ARM one. An ARM number has to get its discipline from repetition and per-cell
controls on a single host instead, and that host is a laptop that sleeps: one run in this
session spent three hours cycling between Clamshell Sleep, a Thermal Emergency Sleep, and
DarkWake, produced a full set of plausible ratios, and said nothing about it. Any ARM pass
reported below therefore carries its own sleep check, taken from the gap between
`time.time()` and `time.monotonic()` across the pass.

### The call plan the new symbol nearly lost

Worth recording because it failed silently and in the direction this whole branch is
about. Dispatch installs a native call plan (`plan.py` → `csrc/plan.h`) so a repeated
product skips resolution, validation and argument marshalling; which plan to build is
looked up by **symbol name** in `plan._SYMBOL_KINDS`. float64 was in that table twice,
as `prebuilt_spmm_csr_f64` and `spmm_csr_double`, both mapping to the `reference` kind.
Putting `spmm_csr_double_v2` in front of them in the prebuilt registry moved resolution
to a symbol the table did not know, `.get` returned `None`, and **float64 stopped
getting a plan at all** — paying back, on every call, exactly the per-call dispatch cost
this branch exists to remove, while the kernel measurements above stayed true. An absent
entry looks identical to "this kernel deliberately has no plan", which is why nothing
raised.

The fix is a fifth plan kind, `v2_double`, and one guard test —
`test_every_resolvable_csr_dense_symbol_has_a_plan_kind` — which resolves the CSR ×
dense symbol for each float dtype the way dispatch does and fails if the result is not
plannable. That is the general statement the table itself cannot make.

Two details the plan kind had to get right. It calls `spmm_csr_double_v2_core`, not
`spmm_csr_v2_core<double>`: the AVX2-or-reference choice described above is a policy, and
reaching past the one function that makes it would have had the plan take the generic
path on ARM — the single thing the ARM decision was made to avoid. And a tiled verdict
is still refused for a float64 plan, which is belt-and-braces given `tiling_gate` never
offers float64 to the selector. `test_plan_matches_the_f64_drop_in_symbol_bitwise` pins
the plan to the pybind entry with `atol=rtol=0`, so a plan that reached past the policy
fails on ARM rather than shipping.

### What float64 still does not get

A tiling route. `spmm_csr_float_tilej` and `spmm_csr_float_tileijk` have no float64
instantiation, so `tiling_gate` in `ops.py` is deliberately float32-only, with a comment
recording why. float64 therefore never reaches the selector at all. That is a gap
against a hypothetical tiled float64 on the wide-N, high-degree shapes where tile-j and
tile-ijk pay — reddit at N ≥ 512 — and not a gap against MKL, which the drop-in beats on
every cell measured.

### Correctness

`tests/test_scorch/test_spmm_float64.py`, 60 tests, plus four in
`test_dispatch_plans.py` for the plan kind: that float64 resolves the new symbol; agreement with a dense reference at rel 1e-12 for every N in 1…40 plus
47, 48, 63, 64, 65, 96, 127, 128 — which covers every register-block width, every mask
remainder, and the first widths past the wide tile; exact agreement with the reference
kernel it replaces; an all-empty matrix; more output rows than sparse rows; and a row
wider than one tile. A 450-cell shape × density sweep through `matmul` is clean.

## A register kernel for ARM, and a three-hour measurement of nothing

Every row of the drop-in SpMM on Apple silicon went through the workspace loop: memset
a tile, load-modify-store into it once per nonzero, memcpy it out. The register-resident
path in `spmm_csr_v2_core` was written under `#if defined(__AVX2__)` and no NEON arm had
ever been put next to it. An earlier section of this file said NEON couldn't have one
because it has no masked load or store; that was wrong, and it is corrected there. The
file already contained `scorch_spmm_row_neon_regtile` for the fused Linear kernel, whose
ragged tail is scalar accumulators updated in the same pass over the row. No masks
needed.

What now exists is one strip kernel, `scorch_spmm_row_neon_strip<T, NV, TAIL, ALLOW_DUAL>`,
templated on element type, on the number of vector accumulators, on how many scalar
accumulators carry the ragged tail, and on whether two nonzeros are in flight at once.
`scorch_neon<T>` supplies the lanes (4 float / 2 double) and the strip width. A row
narrower than one strip is a single dispatch — one pass over the nonzeros — and a wider
row is cut into strips. `neon_single`, `neon_nv` and `neon_tail` are hoisted above the
row loop because they are functions of k alone, so each row runs one straight-line
instantiation, the same way the AVX2 arm switches on a loop-invariant `nvec`.

Two things about that were wrong on the first attempt and are worth stating, because
both were mechanism errors rather than tuning:

- **The strip was sized in registers, not elements.** Eight vectors is 32 floats but
  only 16 doubles, and every extra strip re-walks the row's nonzeros — a degree-3
  float64 row did 24 index walks where it needed 3. `syn__aeshape@128` read 0.969.
  Sizing the strip by element count (`strip_vecs` 8 for float, 16 for double, 32
  elements either way) fixed it.
- **The hoisted switch topped out below the reachable value.** `nv` reaches 16 for
  float64 at k=32, and a `case 15:` ceiling with a literal default wrote NV-1 vectors
  and left the last lanes of the row unwritten — a silent wrong answer, not a slow one.
  Caught by the dense-k sweep in `tests/test_scorch/test_spmm_float64.py`, which walks
  every k from 1 to 40; that test exists because of exactly this class of bug.

### What it is worth

Both arms are the same binary, built with `-DSCORCH_TUNE_HOOKS`, selected per call by
`SCORCH_SPMM_WORKSPACE`, which `spmm_csr_v2_core` reads on every call. Interleaved in
one process, because cross-run drift on this laptop is larger than the effect. The hook
build carries per-row hook overhead on **both** arms, so these are lower bounds on the
shipped kernel. 12 passes, 173 distinct cells, 447 cell-readings, 14 matrices plus three
synthetic degree sweeps, k from 4 to 128, both dtypes:

| | readings | geomean | min | max | below 1.0 |
|---|---|---|---|---|---|
| float32 | 259 | **1.393** | 1.014 | 2.208 | 0 |
| float64 | 188 | **1.339** | 1.007 | 2.046 | 0 |
| all | 447 | **1.370** | 1.007 | 2.208 | **0** |

The same-code control over those readings: median 1.011, p90 1.035, worst 1.114. The
smallest single reading, 1.007, is inside that control; nothing else is close to it.

By k, and by row length, which is where the shape of the win lives:

| | k=4 | k=8 | k=12 | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|---|---|---|
| float32 | 1.345 | 1.451 | 1.428 | 1.502 | 1.310 | 1.352 | 1.300 |
| float64 | 1.319 | 1.461 | 1.292 | 1.342 | 1.326 | 1.293 | 1.253 |

| mean row length | <4 | 4–8 | 8–25 | ≥25 |
|---|---|---|---|---|
| float32 | 1.248 | 1.397 | 1.505 | 1.659 |
| float64 | 1.217 | 1.322 | 1.403 | 1.562 |

The win grows with row length, which is the mechanism: the workspace loop pays a
load-modify-store per nonzero and the register kernel pays nothing per nonzero beyond
the FMA, so the longer the row the more there is to save. It does not vanish at wide k,
because a strip is 32 elements and a wide row is simply more strips. Largest single
cells are `ss__ct20stif@16` (degree 52) at 2.168 and `gcn__ogbn-arxiv@32` at 2.109;
smallest are the near-empty `ss__webbase-1M` (degree 3.1) at 1.007–1.063 and
`syn__aeshape@128` (degree 3.0) at 1.060.

### The 2-nonzero unroll, priced on its own

`ALLOW_DUAL` gives the kernel-before-the-unroll its own arm in the same binary, so the
unroll is priced against its own predecessor rather than across two builds. Geomean
1.031 on float32 and 1.014 on float64; 13% of float32 readings and 29% of float64 ones
below 1.0, worst 0.928; the four cells that lose in a majority of their passes lose
1.1–2.2%, inside the p90 of the control. It stays.

That comparison carries a second control worth more than the first. The threshold
`2*NV <= 16` turns the unroll off for float64 at k ≥ 32, where a strip is 16 vectors and
a doubled set would want the whole register file. Those 85 readings are therefore
identical machine code on both arms, and they measure **1.0066 / 1.0002 / 1.0052** at
k = 32 / 64 / 128. That is the floor the float32 numbers above sit on. (Note the arithmetic
that is easy to get wrong: a wide row is cut into strips of `strip_vecs`, so NV per strip
is 8 for float32 at every k, not k/lanes. The unroll is never off for float32.)

### The fused Linear path had its own copy of this kernel, and its tail was wrong

`spmm_csr_linear_fused_float` carried `scorch_spmm_row_neon_regtile`, written beside its
own row loop rather than shared with the drop-in SpMM. The 32-wide strip body was fine.
The tail was not: it walked the row's nonzeros **once per remaining column**, outside the
nonzero loop. A free dimension under 32 read the row that many times instead of once, and
a ragged width paid an extra full walk per leftover column. Here the free dimension is the
**batch** — the sparse operand is the weight — so the widths that hit it are small batches
and every batch that is not a multiple of 32, which includes a dataset's last incomplete
one.

The fused path now calls `scorch_spmm_row_neon`. In-binary A/B against the kernel it
replaced via `SCORCH_FUSED_LEGACY_TAIL`, 4 passes, 5 weight shapes (64×256 through
4096×1024, densities 0.01–0.10) × 15 batch widths, both relu and identity epilogues,
300 readings: geomean **1.164**, max **2.658**, A/A median 1.010 / p90 1.042.

Two changes ride together and the batch width separates them cleanly, which is the whole
reason to sweep widths that are and are not multiples of 32:

| | n | geomean |
|---|---|---|
| batch % 32 == 0 — no remainder, so the 2-nonzero unroll alone | 30 | 1.021–1.029 |
| batch % 32 != 0 — unroll plus the tail | 45 | 1.265–1.271 |
| batch < 32 — the row was re-walked once per column | 30 | 1.362–1.368 |

The peak is at batch 31, the widest single-pass width: 2.51x, 2.29x, 2.22x on three of
the five shapes. At batch 32 it drops to 1.03 — same kernel, one fewer leftover column.

Nine of 75 cells lose in a majority of their four passes, none by more than 5.1%, and
their mechanisms are known: batch=1 (0.949, 0.955), where the whole width is a single
scalar accumulator; batch=33 (0.967–0.980), one full strip plus a 1-wide remainder, so
two passes over the row; and the smallest weight (64×256) at batch 96–128 (0.982–0.987),
where the whole op is a few microseconds and mostly fixed cost. They are reported rather
than tuned around, because both attempts to fix them made things worse.

### Two fixes for those nine cells, both measured, both rejected

Worth writing down because each was a plausible mechanism argument that lost to a
measurement.

**Don't run the unroll when there are no vector accumulators.** At a free dimension of 1
there is one FMA per nonzero and the loop's own pointer arithmetic and prefetch obviously
dominate, so a second scalar chain looked like pure overhead. Gating `DUAL` on `NV >= 1`
made that case **worse**, 0.968 → 0.922. Two scalar chains still pay. The argument was
clean and wrong.

**Let a single dispatch take a ragged tail at full strip width.** A dispatch covers NV
vectors *and* TAIL scalars, so it could serve 35 elements for float32, not 32 —
`strip<float, 8, 1>` writes exactly the 33 columns that currently cost two passes. Widening
`neon_single` to that bound did not help the widths it was for (33: 0.986 → 0.981) and it
**cost their neighbours**: batch 24 went 1.688 → 1.623 and batch 31 went 1.997 → 1.942,
neither of which the change touches. Adding `NV=strip_vecs`-with-a-tail instantiations to
the hoisted switch changed register allocation for the cases around them. That is the
lesson worth keeping: in a switch over template instantiations, a new arm is not free to
the arms beside it, so "this cannot affect that shape" is a claim about source and not
about code.

Both reverted. The revert was re-measured and reproduces the prior build to within 1% at
every batch width (0.981–1.015), which is also what licenses comparing across these
builds at all: the legacy arm is present in both binaries as an anchor, and rebuilding the
same source lands in the same place.

### tile-ijk kept the scalar inner loop too, and this one only pays on one host

`spmm_csr_float_tileijk` relayouts a strip of B, then for each contraction panel and
each row accumulates that panel's nonzeros into the row's `Nc`-wide slice of a
cache-resident output panel `Cp`:

    for (; pb < pe; ++pb) { a = A_val[pb]; ...; for (k = 0; k < w; ++k) C_row[k] += a * B_row[k]; }

That is `w` L1 loads and `w` L1 stores **per nonzero** into a row that is already hot.
`Nc` exists so that `Cp` fits the *cache* — and at the widths production's own cost model
picks it also fits *registers*, so the row can be loaded once, accumulated over the
panel's nonzeros, and stored once. The harness asks `tiling._ijk_params` and
`tiling.query_llc()` for `Nc` and `Jc` rather than restating the formula, because a
harness that restates a production policy drifts from it silently, which has already
happened three times here.

On the M5 this is worth a uniform 22%. Three passes, 5 matrices (uniform, power-law and
banded, degrees 8–64) × N ∈ {512, 1024, 2048}, in-binary A/B via
`SCORCH_TILEIJK_SCALAR`:

| pass | geomean | min | max | below 1.0 | A/A median |
|---|---|---|---|---|---|
| 1 | 1.217 | 1.124 | 1.288 | 0/15 | 1.008 |
| 2 | 1.220 | 1.117 | 1.288 | 0/15 | 1.006 |
| 3 | 1.217 | 1.123 | 1.271 | 0/15 | 1.010 |

`Nc <= 32`, where the slice is one register strip and the row is walked once, reads
1.196–1.204. `Nc > 32`, where it is walked once per strip, reads 1.226–1.237 — *higher*,
not lower, because a wider slice saves more stores per nonzero and the extra walk is a
cache-hot index stream.

**The same change on x86 was built, measured, and rejected.** Cutting the slice into
64-lane tiles and running `scorch_spmm_row_regtile_partial` with an accumulating seed
gives, on redwood over the identical grid and three passes: geomean 1.033, min 0.971, and
**4 to 5 of 15 cells below 1.0 in every pass**. The losers are every `Nc=112` cell on the
short-row matrices (degree 16 and 33), where 112 lanes is two register tiles and so two
walks of the row per panel, and where a panel holds only 8–16 nonzeros to amortize them.

The asymmetry is the hosts, not the code. What the change removes is **L1** traffic to the
output row. redwood achieves ~56 GB/s and is DRAM-bound streaming the relaid B, so
removing L1 operations buys almost nothing; the M5 achieves ~412 GB/s, is core-bound, and
it buys 22%. A 3.3% mean does not pay for a 2.4% regression on real shapes, so the AVX2
arm is not taken and the path is `#if defined(__ARM_NEON)`. The `ACCUM` template parameter
stays on `scorch_spmm_row_regtile_partial`, and the measured option if the x86 3% is ever
wanted is to gate on the slice fitting **one** tile — every `Nc` of 16 or 32 won
(1.027–1.058) and every `Nc=112` lost.

One thing the change buys beyond speed: tile-ijk and the drop-in SpMM now run literally
the same row kernel on ARM, and the existing bit-exactness test between the tiled fused
route and `scorch.matmul` still passes.

**tile-j was examined and is not the same opportunity**, which is worth stating because
the two look alike. tile-j panels the *contraction* index, so its panel loop sits outside
the row loop and every nonzero updates the full N-wide output row — N is exactly what
tile-j is for, so the row cannot live in registers across panels. It is a streaming axpy,
and the comment on that loop already records that a manual 16-wide unroll there measured
**2x slower on ARM**. Left alone.

### Three ways this measurement lied before it worked

None of the numbers above come from the first three attempts, and each failure was
silent in a different way.

**The machine was asleep.** A run started at 17:41 and finished at 20:48 with a full set
of ratios — geomean 1.44 to 1.48, tight-looking controls, five sections. Four minutes in,
the lid closed: Clamshell Sleep at 17:46:13, a Thermal Emergency Sleep at 17:46:59, then
DarkWake→Sleep every fifteen minutes until 20:41. The same work now takes **five
minutes**, so that run was about 99% sleep and throttle. Its only visible symptom was
that two sections logged a starting load average of 12.52 and 33.51, which is
indistinguishable from a busy desktop. Every pass now measures its own sleep directly:
on Darwin `time.monotonic()` does not advance across a sleep and `time.time()` does, so
their difference across the pass **is** the sleep duration. Every pass above reports
0.0s.

**The hook was not in the binary.** The next attempt pointed `PYTHONPATH` at a tree built
without `-DSCORCH_TUNE_HOOKS`. Both arms took the same path. It produced 400-odd cells at
geomean 0.98–1.03 with controls of 1.005–1.045 — a clean, tight, entirely fictional null,
shaped exactly like "this change does nothing." The harness carried a comment promising
to refuse such a build and did not implement the check. It does now, two ways: the
extension publishes `spmm_tune_hooks` under the same `#ifdef`, and failing that the
harness looks for the string literal `SCORCH_SPMM_WORKSPACE` in the shared object, since
a getenv that was compiled out cannot name it. The tell that was there all along:
relative error between arms read exactly 0.00e+00, where the unrolled kernel sums in two
chains and must differ in the last bits.

**The cells that mattered most were too fast to measure.** `gcn__cora@8` is a
20-microsecond call, and its time does not move with k at all — 0.0250 ms at k=4 and
0.0253 ms at k=32 — so its runtime is per-call fixed cost, not this kernel. Timed one
call at a time, a 20µs parallel region is shorter than the OpenMP thread-wake noise it
sits in; min-over-21-rounds never finds a clean draw and the control blows out to
1.4–1.7. That cell is the entire reason the 2-nonzero unroll was written: it read below
1.0 in three runs of four. Eight readings of it scatter from 0.875 to 1.440 with no
consistent sign. Nothing was ever established there. The fix is to time a batch of
back-to-back calls and divide, sized per cell to clear 3 ms; the control tightened from
1.001–1.463 to 1.005–1.045 on the same busy machine, and became insensitive to load —
the twelve passes above ran at starting loads from 3.25 to 13.64 and their geomeans
agree to within 1%. Batching does change what is measured, from a cold first touch to
the steady state, and both arms get it equally.

The unroll turned out to be worth keeping anyway. The reason first given for it was not
a reason.

### One host, and that is all there is

**MKT is x86_64.** `sinfo` reports `Arch=x86_64` for the allocation, so of the machines
this project measures on, exactly one — the M5 — can execute a NEON instruction. There is
no two-host confirmation available for anything in this section and there will not be
one. What stands in for it: 12 interleaved passes rather than one, a per-cell same-code
control on every cell, a second control from 85 readings where the compared code is
provably identical, per-pass sleep detection, and a per-cell majority test before any
single cell is called a regression.

x86 is unaffected, and that is now measured rather than argued. Building the commit
before this work and the tip on redwood (x86_64, 32 cores, gcc/torch 2.5.1) produces two
shared objects of **exactly the same size**, 939968 bytes, differing in **126 bytes of
the whole file**. Normalizing addresses, the disassembly is identical. Keeping the
immediates, 11 instructions differ, all of the form `mov $IMM,%edx`, and every one of the
11 shifts by exactly **+371** — which is exactly how many lines `spmm.h` grew
(3914 → 4285). Those immediates are `TORCH_CHECK`'s `__LINE__` argument, and the line
numbers they carry land on `TORCH_CHECK` calls in the new file. The rest of the 126 bytes
is the GNU build-id, which cannot match across two builds.

So the x86 machine code is identical apart from eleven assertion line numbers. Every line
added here is inside `#if defined(__ARM_NEON)`, and the compiler agrees.

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

## Fusion locked the selector out, and what it cost to let it back in

`scorch.compile` traces `relu(scorch.matmul(a, b) + bias)` into one call on a native
fused kernel — `spmm_csr_bias_relu_float`, which folds the bias add and the clamp into
the SpMM's row epilogue and so beats the drop-in SpMM plus two torch passes. The
adaptive tiling selector, meanwhile, is gated on the symbol name
`spmm_csr_float_v2`. A fused graph resolves to a different symbol, so it never reached
the gate: **`scorch.compile` silently opted every user out of tiling.** On a
high-degree operand that overflows the last-level cache that is not a small thing —
it is the difference between the column-panel kernel and a kernel thrashing on B.

Measured on reddit (232,965 × 232,965, 114,848,857 nonzeros), the fused path was
**1.4–2.4x slower than not fusing at all** on the M5 and **1.8–5.5x slower** on
redwood, purely because fusing cost you the selector.

### Two ways to compose them, one of them wrong

The naive composition mirrors `ops._dispatch_tiled` inside the fused runner: same
symbol check, same candidate gate, same untiled baseline handed to
`tiling.maybe_dispatch`. It has two defects, and neither is visible from reading the
mirror against the original.

**The thread hints are not part of "the same baseline".** `ops.matmul` derives
`nthreads = torch.get_num_threads()` and `atparallel = _ATPARALLEL_PIPELINE` on the v2
symbol, and hands them both to the baseline *and* to the tiled kernels the selector
picks. A mirror that passes neither compares `tilej(nthreads=None)` against
`v2(atparallel=False)` while the ordinary path compares against `v2(atparallel=True)`.
Same kernel, different configuration — and since the memo was keyed on
`(signature, level)` and shared, whichever caller ran first wrote a verdict the other
could not reproduce. It also forfeits the host-thread match on the tiled kernels
themselves, the lever worth pubmed 0.78 → 1.15x in `e795127`.

**"Is tile-j faster than v2" is the wrong question for a fused graph.** This is the
worse defect. `_confirm_vs_baseline` — `_confirm_vs_v2` before this
change — keeps the tiled pick iff it is *strictly* faster than the baseline, with no
margin. Fusion saves a whole M×N
read-modify-write pass: at N=16 on reddit, where the gate declines and both sides run
untiled, fusing is 11% faster than not fusing (39043 µs against 43969). So a shape where tile-j
beats v2 by less than that is a shape where routing the fused call to tile-j plus a
separate tail is a **regression against the fused kernel you already had**. Routing on
the v2 verdict gets reddit right by luck, because reddit's margin is ~2.4x; it does not
get the gate boundary right.

### The fix: the selector takes the caller's own baseline

`tiling.maybe_dispatch` now accepts `baseline_fn`, an `epilogue`, and a
`baseline_tag`:

- **`baseline_fn`** is the caller's own alternative. `ops.matmul` passes the drop-in
  SpMM; the fused path passes the fused kernel. The no-regression-by-construction
  property — the baseline is always a probe candidate — now holds against whatever the
  caller would otherwise have run, instead of against v2 specifically.
- **`epilogue`** is applied to every tiled candidate *inside the timed region* of the
  probe and the one-shot confirm, and on the memoized dispatch. The tiled kernels have
  no fused epilogue, so the fused caller's tail runs out of line; timing a bare tiled
  kernel against a fused baseline would credit it with work it did not do.
- **`baseline_tag`** namespaces the decision memo and the `max` level's on-disk cache.
  Two baselines are two questions and can legitimately disagree; sharing one entry
  would let either caller run a kernel that lost its own comparison. The default
  baseline keeps its historical unprefixed persistent-cache key, so caches written
  before this change stay readable.

Both callers then share `ops._dispatch_tiled` and one `ops._composition_hints`, so the
first defect cannot recur by construction rather than by review. `ops.tiling_gate` is
split out of `_dispatch_tiled` so a caller that must build closures to consult the
selector can ask first and skip building them on the shapes the gate declines.

### reddit, both hosts

`bench/bench_fused_tiling.py`. Six arms interleaved in a fresh random order per round,
minimum estimator, 3 rounds × 2 repeats. The "before" arm is `set_autotune("off")`,
which short-circuits `is_candidate` so the fused path runs exactly the fused kernel it
ran before this composition existed — same binary, same process, nothing to keep in
sync. `off/on` is what the composition buys; `fus/unf` is whether fusing is still worth
it once both sides can tile; `A/A` is the same arm twice, and nothing inside it counts.

M5 (6 threads, 16 MiB LLC), level `balanced`:

| nnz | N | fused_on µs | fused_off µs | unfused_on µs | off/on | fus/unf | A/A | verdict |
|---|---|---|---|---|---|---|---|---|
| 114.8M | 32 | 47412 | 68923 | 48181 | **1.454** | 0.984 | 1.007 | tilej@65536 |
| 114.8M | 64 | 87944 | 146238 | 90788 | **1.663** | 0.969 | 0.998 | tilej@32768 |
| 114.8M | 128 | 166577 | 338492 | 171686 | **2.032** | 0.970 | 0.999 | tilej@32768 |
| 114.8M | 256 | 333166 | 810096 | 341818 | **2.432** | 0.975 | 1.000 | tilej@8192 |
| 57.5M | 32 | 26718 | 35599 | 26587 | **1.332** | 1.005 | 1.005 | tilej@65536 |
| 57.5M | 64 | 50797 | 76428 | 54985 | **1.505** | 0.924 | 1.005 | tilej@32768 |
| 57.5M | 128 | 92745 | 171651 | 96524 | **1.851** | 0.961 | 0.993 | tilej@16384 |
| 57.5M | 256 | 183302 | 408467 | 195594 | **2.228** | 0.937 | 0.998 | tilej@16384 |

redwood (24 threads, 36 MiB L3), level `balanced`:

| nnz | N | fused_on µs | fused_off µs | unfused_on µs | off/on | fus/unf | A/A | verdict |
|---|---|---|---|---|---|---|---|---|
| 114.8M | 32 | 38956 | 39170 | 49360 | 1.006 | 0.789 | 1.007 | *declined* |
| 114.8M | 64 | 65398 | 114636 | 73607 | **1.753** | 0.888 | 0.993 | tilej@73728 |
| 114.8M | 128 | 113925 | 373451 | 127849 | **3.278** | 0.891 | 1.003 | tilej@36864 |
| 114.8M | 256 | 235926 | 1301570 | 258839 | **5.517** | 0.911 | 1.000 | tilej@18432 |
| 57.5M | 32 | 19973 | 20178 | 27477 | 1.010 | 0.727 | 0.997 | *declined* |
| 57.5M | 64 | 42820 | 60207 | 50807 | **1.406** | 0.843 | 1.000 | tilej@73728 |
| 57.5M | 128 | 83499 | 189074 | 96125 | **2.264** | 0.869 | 1.001 | tilej@36864 |
| 57.5M | 256 | 179690 | 640775 | 204650 | **3.566** | 0.878 | 1.004 | tilej@36864 |

On cross-run reproducibility: a second M5 run on the final build read `off/on` of 1.669
and 2.286 at N = 64 and 256, against 1.663 and 2.432 in the table. The within-run A/A
floor is ±1%, so a 6% cross-run move on a 350 ms cell is the honest reproducibility of
these cells and the third digit should not be read. The verdicts and the direction are
stable; the magnitudes are "1.7x" and "2.3–2.4x", not 2.432.

redwood's win is much larger — up to 5.5x — and its N=32 row is the neutrality control
that fell out for free: reddit's B is 29.8 MB against a 36 MiB L3, so the gate declines
and `off/on` reads 1.006 against an A/A floor of 1.007. Exactly neutral where the
selector does not engage, on the second host.

At N=16 the gate correctly declines on the M5 — reddit's B is 15 MB against a 16 MiB
LLC — and `off/on` reads 1.004 against an A/A floor of 0.995, which is the neutrality
result at the one cell where the same matrix both qualifies and does not.

`fus/unf` is the number worth reading twice. Before this change it was `fused_off /
unfused_on`: 1.43 at N=32 to 2.37 at N=256, i.e. **fusing made the call up to 2.4x
slower**. After, it is 0.92–1.01 — fusing is now slightly *better* than not fusing,
which is what it should always have been.

### Every autotune level composes

The tail and the baseline have to be threaded through each level's decision strategy,
not only the ladder probe: `off` short-circuits at the gate, `analytic` and `learned`
reach the one-shot confirm, `balanced` and `max` run the probe, and `max` additionally
reads and writes the on-disk cache. reddit, `off/on`, both hosts:

| level | M5 N=64 | M5 N=256 | M5 verdict @256 | redwood N=64 | redwood N=256 | redwood verdict @256 |
|---|---|---|---|---|---|---|
| analytic | 1.604 | 2.524 | tilej@16384 | 1.530 | 4.485 | tilej@36864 |
| balanced | 1.641 | 2.514 | tilej@16384 | 1.844 | 5.501 | tilej@18432 |
| max | 1.645 | 2.501 | tilej@16384 | 1.985 | 5.308 | tilej@18432 |
| learned | 1.690 | 2.542 | tilej@8192 | 1.572 | 4.988 | tilej@36864 |

Every level composes and every level wins, on both hosts. The panel widths differ the
way they should: `analytic` and `learned` take the byte model's base width while
`balanced` and `max` search the ladder down from it, on both machines.

**What this table does not support is a ranking between levels.** The redwood cells'
own A/A controls span 0.915–1.067 — a floor of up to 8% — and each level ran in its own
process, so comparing 1.530 against 1.985 is a cross-run comparison across a floor
wider than the gap. The M5 cells are tighter (0.993–1.023) and their four levels sit
within 5% of each other, which is also inside their floor. The claim here is "all four
compose and deliver the win", not "max beats analytic".

### The out-of-line tail, and why one host is not enough to price it

On the tiled route the bias and the clamp run as a separate pass over the M×N output,
because the tiled kernels have no fused epilogue. The question was whether that
residual is worth closing with a fused tiled kernel in `spmm.h`, and the prediction
attached to it was "~5% of the call".

**The mechanism is exactly as predicted, on both hosts.** The tail is linear in the free
dimension (M5: 341 µs at N=32 to 2315 µs at N=256 — 8x N for 6.8x the cost) and does
not move when nnz is halved at fixed M and N (M5: 341 → 384 at N=32, 2315 → 2396 at
N=256; redwood: 8586 → 8524 at N=128). A per-output-element pass on both axes.

**The magnitude is host-dependent, by a factor of ten.** As a share of the fused call:

| N | M5 | redwood |
|---|---|---|
| 32 | 0.7% | 1.8% |
| 64 | 0.7% | 3.0% |
| 128 | 0.8% | 7.5% |
| 256 | 0.7% | 7.2% |
| 128, nnz÷2 | 1.2% | 10.2% |
| 256, nnz÷2 | 1.3% | 9.5% |

This is not an artifact and the reason is bandwidth, not scheduling. The tail moves
~954 MB for reddit at N=256 — `add_` and `relu_` are two torch passes, each reading and
writing the 238 MB output. 17.0 ms on redwood is ~56 GB/s, about 63% of a 14900K's
dual-channel DDR5 peak; 2.3 ms on the M5 is ~412 GB/s. Both are what their machine can
do. A bandwidth-bound output pass costs 7% of the call on x86 and 0.7% on a machine with
seven times the bandwidth per unit of compute.

Those are the tail timed standalone, on a cold buffer of its own, which is a proxy.
The direct measurement is `fused_on` minus the winning tiled kernel run with no tail —
the tail where it actually runs, on an output in whatever cache state the tiled kernel
just left it. On redwood the two agree within ~10% in both directions: 2095 / 8101 /
15318 µs in route at N = 64 / 128 / 256 against 1811 / 8222 / 17012 standalone, i.e.
**3.2% / 7.2% / 6.5%** of the call. reddit's output is 238 MB at N=256, far past any
cache, which is why the cold proxy is not optimistic here; on a shape whose output fits
in cache it would be.

**The direct arm does not work on the M5, and that is a property of the estimator, not
of the machine.** It is a difference of two ~350 ms arms being used to measure a ~2.5 ms
quantity, so a 1% error on either arm is ±40% of the answer — and it duly returned 139 µs
at N=64 and 8370 µs at N=256 against a proxy of 636 and 2549. Those two numbers are not
reported as results anywhere. The M5 figure rests on the standalone proxy, which *can*
resolve it because it times the tail alone. The direct arm becomes the better estimator
exactly when the tail is a large enough share to survive the subtraction, which on this
grid means x86.

**The precondition, which the level sweep then demonstrated by violating it.** The
redwood in-route numbers above come from a run whose per-cell A/A controls were
0.998–1.005. Re-running the same arm in the level sweep, where those controls were
0.915–1.067, produced 38770 µs against a 17264 µs proxy in one cell, 3755 against 19828
in another, and **−7050 µs** in a third — a negative tail, i.e. the bare tiled kernel
timing slower than the same kernel plus a pass over its output. That is not a
measurement of anything; it is the subtraction failing out loud, and it is the useful
form of failure. So the rule the arm needs is explicit: **a difference-of-two-arms
estimator is only readable when each arm's own A/A control is tight**, and a negative
result is the check that catches it when it is not.

So: **a fused tiled kernel is worth building, for x86.** Up to ~7% of a reddit-class
fused call, well outside the A/A floor of 0.993–1.007. It is not in this change — it is
C++ in `spmm.h`, both tiled kernels would need a notion of "last panel" to apply the
epilogue while a row is still hot rather than in a final pass, and it needs its own
grid — but it is a real opportunity and not noise. Recording the trap plainly, because
it caught this document once: measured only on the M5, the tail reads as 0.7% and the
conclusion is "not worth writing", which is wrong for every x86 host we ship on.

Separately, what accounts for `fus/unf` landing below 1.00 — 0.92–1.01 on the M5,
0.73–0.91 on redwood — is the opposite direction: the *unfused* arm materializes two
temporaries (`out + bias` allocates, `relu` allocates) where the tiled fused route does
one in-place `add_().relu_()` pass.

### The lesson that outlives both numbers

Neither the "~5%" prediction nor the "0.7%, not worth writing" refutation named a host.
The tail is a bandwidth-bound quantity, so it never had a machine-independent value —
and this repository's own convention says to name the host for every number and never
average across them. The two failure shapes are worth separating, because they are
caught by different disciplines: **reasoning from a mechanism without sizing it** is
caught by measuring once, and **generalizing a size from one host** is caught only by
measuring twice. The convention asks for both, which is why it asks for both.

### What it costs the shapes it cannot help

Every GCN-small layer, every autoencoder layer, and anything whose dense operand fits
in cache is declined at the gate. Those must not be taxed.

This is measured by isolation, not end to end, and deliberately: a declined
consultation never launches a kernel, so it can be timed to a few nanoseconds, whereas
the end-to-end difference on a 15–500 µs fused call is a fraction of a percent and
drowns in the noise six interleaved OpenMP teams in one process produce. The first
attempt did it end to end and returned an A/A floor of 0.85–1.06 on exactly these
cells, which resolves nothing; that number is not reported anywhere as a result.

`bench/bench_fused_tiling_declined.py`. `gate_us` is `ops.tiling_gate` alone — the same
call `ops.matmul` makes on every prebuilt CSR×dense product; `added_us` is everything
the composition adds to a fused call; `old_gate` is the gate as it stood before this
work, replicated in the harness so both can be timed in one process against one binary.

| shape | N | old_gate µs | gate µs | added µs | fused µs | added % | old/new |
|---|---|---|---|---|---|---|---|
| **M5** | | | | | | | |
| cora-class | 16 | 0.484 | 0.348 | 0.358 | 27.4 | 1.31% | 1.391 |
| citeseer-class | 16 | 0.484 | 0.347 | 0.354 | 28.5 | 1.24% | 1.395 |
| pubmed-class | 16 | 0.483 | 0.346 | 0.354 | 82.2 | 0.43% | 1.396 |
| arxiv-class | 128 | 0.576 | 0.529 | 0.550 | 5638.4 | 0.01% | 1.088 |
| ae-class | 128 | 0.537 | 0.387 | 0.399 | 282.1 | 0.14% | 1.386 |
| ae-wide | 256 | 0.528 | 0.387 | 0.397 | 563.8 | 0.07% | 1.362 |
| **redwood** | | | | | | | |
| cora-class | 16 | 0.595 | 0.357 | 0.358 | 17.2 | 2.08% | 1.667 |
| citeseer-class | 16 | 0.601 | 0.352 | 0.364 | 19.6 | 1.86% | 1.707 |
| pubmed-class | 16 | 0.606 | 0.355 | 0.360 | 51.8 | 0.69% | 1.710 |
| arxiv-class | 128 | 0.731 | 0.638 | 0.651 | 11578.3 | 0.01% | 1.145 |
| ae-class | 128 | 0.613 | 0.355 | 0.366 | 96.0 | 0.38% | 1.726 |
| ae-wide | 256 | 0.610 | 0.356 | 0.367 | 283.1 | 0.13% | 1.712 |

Two things to read off this. First, `added − gate` is 0.001–0.021 µs on every cell:
after `ops.tiling_gate` was split out so the runner can ask before it builds anything,
**the composition costs a declined fused call the gate and nothing more.** Second, the
gate itself is now **1.09–1.73x cheaper than before**, on both hosts and on every cell,
which is a saving on the dispatch path of every prebuilt CSR×dense product in the
library, not only the fused ones. Three changes did that, all in the shared layer:

- `_current_level()` read the thread-local override with `getattr(_tls, "level", None)`.
  With no override active — the common case, since one only exists inside a
  `set_autotune` context manager — that raises and catches an AttributeError
  internally: **0.132 µs against 0.029 µs** for `_tls.__dict__.get("level")`, which
  returns the identical value in every thread, set or unset.
- `is_candidate` read nnz off the index arrays (an attribute chain plus a pybind
  `numel()`) *before* the cache test that rejects 99% of shapes without needing it.
  `_operand_over_cache` is now callable on its own, so the 99% answer on two int
  operations.
- `_eligible_learned` took an `nnz` argument it never used, which was the only reason
  the widened gate needed it either.

So the honest statement of the cost is: **1.2–2.1% of the fused SpMM call on the two
smallest GCN graphs, 0.01–0.7% everywhere else**, all of it the selector's pre-filter,
which is the same mechanism `ops.matmul` has paid on every prebuilt CSR×dense call
since the selector shipped — and which is now 1.09–1.73x cheaper for both callers than
it was this morning. Removing it entirely would need a memo keyed on operand identity,
and the safe form of that (weak references plus version counters, as in the
transposed-operand copy) costs more than the 0.35 µs it would save.

### Correctness

27 tests in `tests/test_scorch/test_fused_tiling_composition.py`.

The load-bearing one is not the numeric comparison. On the tiled route a fused call
must equal `matmul + bias + act` **bitwise**, because both sides run the same tiled
SpMM and only a wrong tail can differ — but that assertion alone is vacuous, since if
the fused graph fell back to its eager equivalent, that equivalent is `scorch.matmul`
routing through the *same* tiled kernel plus the same tail, and produces identical
bits. So every test that means to exercise a route says so out of band, through a spy
on `ops.tiling_gate` and `ops.dispatch_tiled_fused` that records how far each call got.
With the route disabled, **15 of the 27 fail**. With the spy's route assertions
neutralized as well, only **10** do — so 5 of those 15 are caught by the spy alone, and
would otherwise have passed against a fused call that never reached a tiled kernel.

The rest: tile-ijk as well as tile-j; the bias-only kernel as well as bias+relu; that
the in-place tail does not write through to an operand the caller still owns; that a
declined shape returns bit-for-bit what a direct fused-kernel call returns and writes
no verdict; that a verdict measured against the drop-in SpMM does **not** route a fused
call (the second defect above, pinned by making the two tags disagree and checking
which kernel ran); that a fused call going *first* writes only its own entry and leaves
`matmul` to measure its own — the ordering no benchmark exercises, since `bench_gcn`'s
`FRAMEWORK_ORDER` always runs the unfused arm first; that the tail is timed with every
tiled candidate (3 invocations per candidate, none for the baseline); that every fused
prebuilt kernel has exactly one out-of-line tail and no tail exists without a kernel
behind it; that the gate shuts when the drop-in SpMM symbol is absent from the build,
since `resolve_prebuilt_matmul` falls back through two other symbols and only the first
is tiled; that a `max`-level fused verdict round-trips through a real cache file under
the prefixed key and is *not* readable as the drop-in SpMM's; that one compiled graph
re-called with different shapes stays correct and still reaches the selector, which is
what makes resolving the untiled SpMM once at trace time sound — the resolution keys on
ranks, formats and dtypes and nothing else, all of which are re-checked per call; that a
dtype change on that same graph is caught by that re-check and falls back rather than
running against a kernel resolved for a dtype it no longer has; and that SpMV-shaped
(N=1), COO and float64 fused calls all decline without touching the selector.

## The probe that decides whether a tiled kernel ships, and the two things it never measured

`tiling.maybe_dispatch` is the routine that decides whether an SpMM runs on a tiled
kernel or on the drop-in one. Its docstring claimed no-regression "by construction",
on the grounds that the caller's baseline is always one of the candidates. Two things
made that claim weaker than it reads.

**Candidate order.** Each candidate was timed to completion before the next one
started, and the baseline is always `cands[0]`. So the baseline was the one arm that
never ran on a machine an earlier candidate had warmed — clocks still ramping, OpenMP
team not yet settled. The one-shot confirm used by the default level had the same bug
pointing the other way: it timed the tiled candidate first and the baseline second,
biasing *towards* the baseline. Two routines, opposite biases, neither measured.

**No floor.** Nothing measured the noise. A min of two timings per candidate decided a
verdict that is then memoized for the life of the process, and at the `max` level
written to a file for the life of the machine. A cell whose true margin sits inside the
run-to-run spread got a permanent answer from a coin flip.

### The instrument

The baseline goes into the candidate list twice, first and last — the same function
with the same arguments. Under a position-free scheme the two entries measure one
number, so whatever gap opens between them is position, with no model of turbo ramp or
thread settling in between. Three schemes run on the same list: what shipped, the same
reversed, and interleaved with a rotating start.

That the effect is real and not variance shows up in the reversal. redwood, `nd24k` at
N=1024, ratio of the two identical baseline arms:

| scheme | last/first |
|---|---|
| sequential, shipped order | 1.240 |
| sequential, reversed | 0.924 |
| interleaved | 1.112 |

Reverse the order and the sign reverses. That is position. What survives interleaving —
1.112 on that cell — is not; interleaving removes the bias and cannot remove the
variance, which is why the fix needs both halves.

### One correction to make first

The first version of this harness spelled out the kernel's argument list by hand. That
list takes `tile_size` before the thread count, so the call passed the thread count as
a tile width and left `nthreads_override` at 0, selecting a different threading policy
from the one `matmul` uses. On Apple silicon, where the workspace path is live because
there is no AVX2, that crippled the baseline arm and inflated every tiled margin the
harness reported: `nd3k` at N=512 read 3.27x and is really 1.026x. The corrected
harness calls `execute_prebuilt_binary_kernel`, deriving the call from production
rather than restating it, and the numbers below are all from after that fix. The
uncorrected run said every gate-admitted cell wins by 2.1–4.2x and the defect was
therefore unreachable. That was wrong, and wrong in the direction that would have
closed the investigation.

### Both hosts

The rule that shipped accepts a tiled candidate if it beats the baseline as timed. The
rule now shipping requires the win to exceed the gap between the two identical baseline
arms. Over the matrices whose gate actually opens:

| host | cells | rules disagree | margin min / median / max | A/A interleaved |
|---|---|---|---|---|
| M5, N=512, rounds=2 | 21 | **5** | 0.850 / 1.038 / 1.771 | 0.912–1.179 |
| M5, N=512, rounds=8 | 21 | **4** | 0.820 / 1.003 / 1.744 | 0.939–1.088 |
| redwood, N=128/512/1024, rounds=2 | 14 | **2** | 0.395 / 2.552 / 5.625 | 0.907–1.034 |

The exposure is host-dependent, and the reason is the gate. redwood's last-level cache
is 36 MiB against the M5's 16, so the eligibility test admits only products that
overflow a much larger cache, and what it admits wins by a lot — median 2.55x. The M5's
gate admits cells whose median margin is 1.038, which is inside the floor. Same code,
same rule, and on one host it is adjudicating landslides while on the other it is
adjudicating coin flips.

### What the disagreements actually are

Every one of them is inside its own cell's floor. Not one is a proven regression:

| cell | floor | margin | proven-loss threshold | verdict |
|---|---|---|---|---|
| M5 r8 `ship_001` | 1.065 | 0.941 | 0.939 | inside, by 0.2pp |
| M5 r2 `TSOPF_FS_b300` | 1.045 | 1.042 | 0.957 | inside, by 0.3pp |
| redwood r2 `crankseg_1`/512 | 1.059 | 0.952 | 0.944 | inside, by 0.8pp |
| M5 r8 `nd3k` | 1.055 | 0.999 | 0.948 | inside, by 5.1pp |
| redwood r2 `crankseg_1`/1024 | 1.102 | 1.022 | 0.907 | inside, by 8.0pp |
| M5 r2 `mixtank_new` | 1.179 | 1.037 | 0.848 | inside, by 14.2pp |

So the claim is not that the old rule shipped measurable regressions. It is that in
these cells the comparison cannot be resolved at all, the point estimate is often
unfavourable — `ship_001` sits 6.3% slower on the point estimate and 0.2 percentage
points from being provably so — and the old rule resolved them anyway from a single
baseline sample, then memoized the answer. The new rule declines them.

Nothing that is measurable moves. Every cell whose margin clears its floor keeps its
verdict on both hosts, in both directions: the nine real M5 wins (1.05–1.74x), the
eight real M5 losses (0.82–0.95), redwood's `inline_1` (0.54) and `audikw_1` (0.395)
declines, and `reddit` at 3.671 / 5.625 / 5.248 for N=128 / 512 / 1024. The tile-j and
tile-ijk ledgers elsewhere in this document rest on `reddit` and were produced through
`scorch.matmul` rather than by naming the kernel, so they are unaffected by either the
defect or the harness bug.

### Cost

The probe goes from 18 kernel invocations to 22: one more candidate, plus re-running
the winner for its output instead of retaining every candidate's, because one dense
output per candidate is 950 MB for `reddit` at N=1024. The confirm goes from 6 to 9.
Paid once per shape.

The floor requirement gets *less* conservative as the measurement improves, which is
the right shape for it: at rounds=8 the M5's disagreement count falls from 5 to 4 and
its A/A band narrows from 0.912–1.179 to 0.939–1.088. It is a tax on having measured
badly, not a fixed tax.

### Tests

22 in `tests/test_scorch/test_probe_noise_floor.py`. Negative controls: all 22 fail
against the pre-fix module, but 13 of those fail only because the new helpers do not
exist, so the sharper control keeps every helper and the interleaving and weakens only
the floor requirement back to "faster than the baseline, full stop" — 4 tests fail,
and those 4 are what the floor itself buys.

## Four kernel levers: two ship, one is retired by its own null, one needed a gate

Four changes to the SpMM row kernels and their work-stealing policy. They are
numbered separately from the dispatch levers earlier in this document. Each is measured
separately, because a lever measured only in combination cannot be attributed. The
mask lever lives inside the AVX2/FMA guard and is x86-only by construction: there is
nothing to measure on Apple silicon and nothing there to regress.

All four are timed by flipping a `SCORCH_TUNE_HOOKS` switch **within one process
against one binary**, following the existing `SCORCH_REGTILE_BASE` precedent. That
removes build-to-build and process-to-process variance entirely. A
`scorch_tune_hooks()` binding lets each harness refuse to run against a build where
its hook is inert — otherwise both arms are the same code and the harness reports a
difference of zero, which reads as "the change did nothing" rather than as "this
measured nothing".

### The control group, and why it matters more than any of the levers

The narrow-k mask change applies only where `k % 8 == 0`. Ragged `k` keeps the mask
and its code path is untouched, so **ragged `k` is a null**: whatever the instrument
reports there is what it reports when the true effect is exactly zero.

| group | n | geomean | 95% CI | worst A/A |
|---|---|---|---|---|
| `k % 8 == 0`, the mask is dropped | 32 | 1.0387 | 1.017–1.061 | 1.566 |
| ragged `k`, code unchanged (**null**) | 24 | 1.0057 | 0.998–1.014 | 1.088 |

The null behaves: 1.0057, essentially one. But applying a per-cell significance test
— is this cell's ratio outside its own two-sample A/A control — to the null group
returns **11 of 24 "significant" cells, a 46% false-positive rate**. At a few
percent, per-cell verdicts from a single grid are not evidence, whichever direction
they point.

That retired two findings of my own:

* **The mask lever's "3 proven per-cell regressions" are not findings.** They sit
  inside the same band the null group populates.
* **The chunk rule's blocker dissolved.** `ogbn-arxiv` at k=64 read 0.967 against an
  A/A of 1.020 — a 3.4% regression on a real GCN graph at a real hidden width, which
  is exactly what the performance convention refuses to ship. Re-measured at 61
  rounds instead of 15 it reads **1.020 with an A/A of 1.007**. The sign flipped. One
  grid at a few percent was not enough to condemn a cell, and it very nearly cost a
  lever worth 1.35x on that host.

What survives is the group comparison against the null: difference of log-ratios
between the applying group and the null, **+0.0323 with a standard error of 0.0117**,
so **+3.3% at 2.8 standard errors**.

One cell in the applying group has an A/A of 1.566 — the instrument declaring, in
its own words, that two identical arms came out 57% apart. It should not be averaged
with cells whose A/A is 1.01. Dropping the single cell whose own A/A exceeds 1.10
(`ct20stif` at k=24, which reported 1.404):

| group | n | geomean | 95% CI |
|---|---|---|---|
| `k % 8 == 0`, A/A ≤ 1.10 | 31 | 1.0287 | 1.019–1.039 |
| ragged `k` (**null**) | 24 | 1.0057 | 0.998–1.014 |

**+2.3% at z = 3.4** — smaller than the headline and better resolved, because the
discarded cell contributed almost all of the scatter. 2.3% is the number to quote.
Screening on a cell's own A/A before pooling is not the same mistake as judging a
cell by its A/A: it discards cells the instrument disclaims, rather than promoting
cells whose ratio happens to exceed a badly calibrated threshold.

### Kernel lever 1 — the cache size, queried instead of assumed

The kernels had no notion of the last-level cache. `scorch_llc_bytes()` reads the
same sources as `tiling.query_llc` — P-cluster L2 via sysctl on macOS, largest sysfs
cache level on Linux, `SCORCH_LLC_BYTES` overriding both.

As ported, its fallback was 8 MiB unconditionally against Python's 16 on Darwin and
36 on Linux, so the comment claiming the two layers can never disagree about the
machine was false in exactly the case that matters — a failed query is when the
fallback is all they have to agree on. Fallbacks now match, and the agreement is a
test rather than a comment, because a disagreement is otherwise invisible in both
layers.

With kernel lever 3 retired, no C++ kernel gates on this. It stays because the selector's
number has to be inspectable from a harness without the harness restating how it is
derived — the failure mode recorded under "Scope and gaps" below.

### Kernel lever 2 — no mask where the last vector is full (x86 only) — **ships**

`vmaskmovps` is 2 uops on the load side and cannot fold into the FMA as a memory
operand; its store form is worse. At k=16 half of every row's B loads were masked
for no reason. The row kernel is now templated on whether the last vector is full,
so the shipped build has no branch — the predicate depends only on k, so it hoists.
Worth **+2.3%** as above.

Equivalence is by construction: with all eight lanes enabled a masked load is a load
and a masked store is a store. But the widths where that argument applies were
exactly the widths nothing was checking, so every k from 1 to 40 is now compared
against a dense reference — both sides of all four instantiation boundaries and the
crossing into the wide path above 32 — plus a check that a full final vector does
not write past its own row, which with a contiguous row-major output would corrupt
the next row silently.

The ported prefetch-distance template parameter is dropped. It had been measured as
a wash and left in place defaulted to the original distance, so it was an unused
template dimension.

### Kernel lever 3 — non-temporal stores on the wide path — **retired, by its own null**

An ordinary store to a line about to be overwritten in full still pays a
read-for-ownership. Skipping it should be worth something when C cannot stay in
cache, so the gate was `k >= 64` (only the 64-wide tile loop streams), `k % 16 == 0`
(a row is a whole number of 64-byte lines, so no line is shared with the empty-row
pre-zeroing), C larger than the last-level cache, and — the condition the ported gate
was missing — a 32-byte-aligned base, because `vmovntps` **faults** on a misaligned
address and has no unaligned form to fall back on.

Correctness was never the problem. Output was bit-identical in all 12 cells checked,
and the gate fired exactly where predicted. Performance was the problem. The 40-cell
grid splits by whether the gate can fire at all, and the cells it cannot touch are a
null:

| group | n | geomean | range |
|---|---|---|---|
| gate fires, streaming actually used | 19 | **0.9972** | 0.975–1.028 |
| gate provably shut (**null**) | 21 | 1.0315 | 0.966–1.620 |

The null reads *higher* than the effect, and the largest number in the whole table —
1.620, `cop20k_A` at k=72 — is a cell where `72 % 16 != 0` and no streaming store is
ever issued. Its own A/A is 1.106, the worst in the grid.

The premise is refuted rather than merely unsupported: the biggest outputs, where
skipping the read-for-ownership should matter most, are exactly the cells at or below
1.0 — `thermal2` writing 1199 MiB against a 36 MiB L3 reads 0.978, and `ogbn-arxiv`
reads 0.975–0.982 across three widths. Whatever this part does for a full-line write
already costs what the read-for-ownership would have, so there was nothing to skip.

Removed: the template parameter, the alignment gate, the store fence, the hook, and
the harness. A retired lever leaves behind a comment on `scorch_spmm_row_regtile`
saying ordinary stores are deliberate and what the measurement was, so the next
reader does not re-derive the idea and re-measure it. This is the only one of the
four that would have shipped on a plausible mechanism plus a correctness check.

One near-miss worth recording. Placing the `_mm_sfence()` by matching surrounding
context put it in `spmm_csr_linear_fused_float`, where `stream_c` is not declared.
The M5 build was clean, because the whole block is inside the AVX2 guard — so **the
M5 is a false negative for this entire class of error**, and only an x86 compile
catches it. Anything keyed on surrounding context in `spmm.h` needs an explicit
function check; the anchor text recurs in both kernels.

### Kernel lever 4 — the work-stealing chunk width — **ships, behind a gate**

The drop-in SpMM hands rows to workers in fixed chunks through one atomic counter.
The generic width is a load-balance rule that knows only the row count and the total
work, so it returns 64 for almost everything. Two costs actually trade off:

    steal stream   (rows / chunk) * c        c = one contended atomic
    tail           chunk * deg * k * w      w = one nonzero of work on one core

Minimising the sum gives `chunk* = rows * sqrt(K * KREF / (nnz * k))`, where `K` is
the ratio of a contended atomic to a nonzero of work — a property of the machine, not
of the matrix, since everything matrix-specific is already in rows, nnz and k.

`K` arrived as a literal 16 fitted on one host, which the performance convention does
not allow shipping. It is also a **weak** parameter: `chunk*` moves as `sqrt(K)`, so
being 4x wrong in K is 2x wrong in the width. Rather than sweep K — which compresses
exactly the axis under calibration — the response surface was mapped directly by
sweeping the width itself, and K back-solved from the winner. Across cells and both
hosts the implied K spans roughly **3 to 23000** against the 16 written down.

That span is the reason for a gate. A model whose parameter is uncertain by three
orders of magnitude is not entitled to act on a recommendation 6% away from the
status quo. `SCORCH_SPMM_CHUNK_MINRATIO 2` departs from the generic width only when
the rule asks for at least twice it; below that, the rule returns the generic width
and the emitted schedule is unchanged. The grid rules out a threshold of 1 and cannot
separate 1.5 from 3 (whole-grid geomean 1.138 to 1.114, inside a ±3.6% null band), so
2 is the middle of the range the data admits rather than the value that maximises
this grid.

Ten matrices — GCN graphs, SuiteSparse, and synthetic scatter and banded — crossed
with k in 8, 16, 32, 64, on both hosts. Shuffled arm order, every compared quantity
entered twice, and the override installed outside the timing window:

| host | fires | firing geomean | firing range | no-op cells (**null**) | mechanism null |
|---|---|---|---|---|---|
| M5 Max (18 threads) | 19/40 | **1.273** | 0.951–2.309 | 0.997 (0.949–1.073) | 1.007 |
| redwood (32 threads) | 15/40 | **1.346** | 0.993–2.154 | 1.003 (0.977–1.067) | 1.000 |

Three nulls, not one, and they are what make the table readable:

* **The mechanism null.** The rule's own chosen width, re-requested through the
  override — identical width, identical kernel — over all 40 cells. 1.000 on redwood,
  1.007 on the M5. Without it, "the override is the confound" stays a live hypothesis
  for every number in the table.
* **The no-op cells.** Where the gate returns the generic width the two arms run
  provably identical code, so their spread is the floor. 25 such cells on redwood, 21
  on the M5, reading 1.003 and 0.997.
* **Zero firing cells below either floor.** The worst firing cell on each host —
  0.993 on redwood, 0.951 on the M5 — sits inside the band its own no-op cells
  populate. The M5's is `pubmed` at k=8, whose mechanism null reads 1.152, the worst
  in that grid; that cell is not resolvable, in either direction.

The two hosts fire on different cells, and that is the rule working rather than
noise: the load-balance cap is `rows / (threads * 16)`, so the M5's 18 threads leave
more headroom than redwood's 32 and admit smaller matrices.

Inertness where it matters is provable by asking the rule instead of timing it. At
the shapes existing published ledgers depend on, the rule returns the generic width
for **reddit at every k from 8 to 256**, and likewise for `cora`, `mouse_gene`,
`crankseg_1` and `nd24k` — so the tile-j and tile-ijk ledgers cannot move. It is
*not* inert on `ogbn-products`, `ogbn-arxiv` up to k=128, or `pubmed`/`citeseer` at
small k. I had reported the gate as making the rule a no-op on the sparse autoencoder
grid too; that was wrong, and checking it against the rule rather than restating it
is what caught it. Two things follow. The AE @0.99 ledger is safe for a
different reason than I gave: `spmm_csr_linear_fused_float` deliberately keeps the
generic chunk, so everything routed through `sparse_linear` is untouched by
construction. And the two regimes no grid cell covered — many rows at degree 3, and
many rows at degree 25 — were then measured directly on the M5:

| regime | rows | degree | k | M5 Max | redwood |
|---|---|---|---|---|---|
| AE-shaped | 60,000 | 3 | 8–128 | 1.126–1.399, fires at every k | **1.558 / 1.431** at k=8/16, gated off above |
| products-shaped | 2,449,029 | 25 | 8–128 | 0.998–1.044 | 0.998–1.001, net geomean **0.9996** |

AE-shaped wins on both, and the hosts split for a reason worth stating: the
load-balance cap is `rows / (threads * 16)`, which is 208 rows on the M5's 18 threads
and 117 on redwood's 32. At k ≥ 32 that cap drops redwood's recommendation below twice
the generic width, so the gate returns generic and the cell becomes a provable no-op —
reading 1.002–1.004. The rule is served on one host and declines on the other, and
where it declines it changes nothing, which is the safe direction for a gate to fail.

Products-shaped fires at every width on both hosts and buys nothing on x86: netting
each cell against its own mechanism null gives a geometric mean of 0.9996 over five
widths, worst cell 0.998. That is neutral, not a regression, and it is the honest
result for the regime — the rule's benefit is concentrated in the middle of the range
and at this scale the steal stream it saves is already negligible against the work.

Reading that took the mechanism null rather than the no-op null, and the harness said
the wrong thing first. With the rule firing at every products-shaped width, the no-op
group came out as three AE-shaped cells spanning 1.002–1.004, and against a floor that
tight five neutral cells were reported as "below the null". A group of three does not
set a floor. The harness now refuses to draw that conclusion below eight cells and
prints each firing cell against its own mechanism null instead — which is the only
control that exists on *every* cell, firing ones included, and which tracked `vs_gen`
here to within 0.002 cell for cell.

The fused Linear kernel's chunk is left generic, stated rather than defaulted: its
workload is the autoencoder grid and it deserves its own measurement before adopting
a rule fitted on the drop-in path's.

### A measurement that started too soon, and the control that caught it

The first redwood chunk grid reported a whole-grid geomean of 1.571 against the
generic width — inflated by about 40% against the 1.120 the corrected instrument
gives. It is void, and its own control said so before I read the table: cells where
the rule provably returns the generic width, so both arms run identical code, read
**2.263, 2.034, 2.858 and 3.896**, and the A/A control between two identical arms
reached **1.332**. An instrument reporting a 3.9x difference between two runs of the
same code has disclaimed its own output. I read 1.332 as "small enough" and went on
to interpret 1.05 ratios as findings.

Re-measured on a settled machine the same four cells read 1.004, 1.001, 1.051 and
1.001. The cause is not what I first blamed. I attributed it to the rotating arm
order — `j = (i + r) % n` moves each arm's absolute position but never its
predecessor, so with a chunk ladder whose arms differ 10x in cost the neighbour
effect is a fixed per-arm offset, not variance. That is a real property of a rotation
and worth knowing, but it is not what happened here: under rotation, with the same
ladder present, the corrected harness reads 0.979–1.013 on those cells. Nor is it the
other two defects the rewrite fixed. Mutating `os.environ` inside the timing window
costs 0.3–0.4 µs against a 60 µs kernel, and `pop` is the dearer of the two, which is
the wrong direction. Comparing min-over-two-arms against min-over-one is worth 4–8%,
measured by computing both estimators from one run.

What is left is the machine. That grid started within seconds of a 51-minute,
24-thread test suite finishing, and the same harness code on a settled machine gives
1.023–1.041 on the cells that read 2.26–3.90. So: **do not start a measurement
immediately after a long saturating job**, and the queued re-measurement now waits
for the suite and then sleeps five minutes. The general lesson is cheaper than the
diagnosis, though: the A/A control had already reported the run unusable, and no
amount of care about arm order substitutes for reading it.

## The published 236-cell grid, candidate against the branch point

The levers above were each measured on their own grid. That does not answer the
question the performance convention actually asks, which is whether the shipped tree
regressed anything on the corpus this document's headline numbers come from. So:
`600cba2`, the `perf/spmm-beat-mkl` tip and the branch point, against the candidate
tip, over all 236 cells at all five autotune levels — 1180 points.

Two trees built from one bundle, differing in exactly four runtime files. Passes run
**base, cand, cand, base** per group so both trees have the same mean position and a
monotonic drift over the run cancels in the contrast rather than favouring whichever
ran later. The two same-tree passes are the floor, and they need to be: the cross-run
spread between `base_r1` and `base_r2` — same binary, two processes — has a median of
1.035, a p95 of 1.398 and a **maximum of 4.13x**. A single base-run against a single
cand-run on this corpus can invent almost any regression it likes.

The split that makes the table readable is mechanism, not size. A cell is **live** only
if a shipped change can execute in it:

* the chunk rule fires only where `scorch_spmm_chunk` differs from the generic width —
  **37 of 236 cells**, all through `spmm_csr_float_v2`, since the tiled kernels take no
  chunk at all;
* the mask lever runs only where the register-block path owns the row (N ≤ 32) or the
  register-tile path has a ragged last tile (N % 64 ≠ 0). At N = 64, 128, 512, 1024 and
  2048 the width is a whole number of 64-wide tiles and the masked instantiation never
  executes.

Everything else is running identical machine code in both trees.

| group | n | geomean (base/cand) | 95% CI | worst point |
|---|---|---|---|---|
| **null** — no change can execute | 680 | 0.9947 | [0.989, 1.001] | **2.214x slower** |
| **live** — a change can execute | 500 | **1.0470** | [1.038, 1.056] | 1.124x slower |

**live − null: +5.25%, z = 9.55. Zero live points fall below the null's floor.** The
worst live point — `syn:band16` at N=32, 1.124x slower across four levels — sits at the
**4.9th percentile of the null distribution**: 33 of 680 points running unchanged code
are at least that slow. It is not separable from noise, and four levels agreeing is not
four observations, because all four route to `v2` and re-time the same kernel.

Per group, and the wide-B tail is the cleanest case of all — at N = 1024 and 2048 the
mask is inert and the chunk rule fires on none of those four matrices, so *every* wide
cell is a null cell, and it reads 0.9995:

| group | live n | live geomean | null n | null geomean |
|---|---|---|---|---|
| main | 255 | 1.0543 | 285 | 0.9984 |
| ss-tiling | 130 | 1.0427 | 155 | 1.0020 |
| ss-quick | 115 | 1.0357 | 200 | 0.9830 |
| wide | 0 | — | 40 | 0.9995 |

Decomposed by which lever can run, the two separate cleanly:

| mechanism | n | geomean | range |
|---|---|---|---|
| mask only | 315 | 1.0133 | 0.890–1.469 |
| chunk only | 75 | 1.0303 | 0.894–1.191 |
| both | 110 | **1.1622** | 0.976–1.805 |

`both` is superadditive against the other two because it is exactly the N = 16 and 32
cells on the many-row matrices — `webbase-1M`, `thermal2`, `scircuit`, `ogbn-arxiv` —
which is where the chunk rule's largest wins live. The mask-only figure, 1.0133 over
315 points, is also the third independent estimate of that lever, against +2.3% from
its own grid.

**Route changed on 0 of 1180 points.** The probe's A/A control and fail-closed margin
did not move a single dispatch decision on this corpus, which is what "behaviour-neutral
on today's frontier" should mean when checked against a real grid rather than a
frontier sample.

Two things this grid says that the per-lever grids could not. The per-cell significance
test flagged **93 regressions, 69 of them in the null group** — including the entire top
of the list, `bcsstk17@512` at 2.21x and `cop20k_A@N=128` at 1.24x on all five levels,
both of them cells where no shipped line can execute. That is the same false-positive
mode measured at 46% earlier, reproduced on an unrelated corpus, and it is the reason
this section reports groups rather than cells. And the null group is the only check made
anywhere on the `scorch_spmm_nthreads` extraction, which touches the thread count of
*every* call: it was factored out of two kernels that each computed it inline, one of
them carrying a comment that merely claimed to match the other. 0.9947 over 680 points
sharing no other change is the evidence that the arithmetic did not move.

## The shared row counter, and what it was costing

The narrow-k deficit against MKL was not the kernel. It was that we threw away A's
cache residency between calls.

`spmm_csr_v2_core` handed rows out from a single `std::atomic<int> next_row`. Which
worker got which rows was therefore decided by arrival order at that counter, and
arrival order differs on every call, so a core that held rows 400-450 in its L2 last
call gets rows 1200-1250 this call and re-fetches its whole slice from L3. MKL assigns
statically and keeps its slice.

Measured with `perf stat` on redwood, float32, k=1, over four L3-resident matrices:

| matrix | our L2 misses/nnz | MKL's | ratio |
|---|---|---|---|
| ts-palko | 0.1238 | 0.0062 | 15.3x |
| nemeth09 | 0.0693 | 0.0134 | 5.2x |

`0.125` is exactly what four bytes of value plus four bytes of index per nonzero
predicts. We were re-reading all of A, every call. On ts-palko that is 138404 misses
per call, 8.86 MB at a 64-byte line, against MKL's 579 KB -- MKL kept 93% of A in the
cores' L2 and we kept none of it.

### Home ranges, and why they need stealing

The fix is a contiguous home range per worker. The split is balanced on
`A1_pos[i] + i`, not `A1_pos[i]`: a row costs its nonzeros *plus* a fixed amount --
the row-pointer pair, the accumulator reduction, the output store -- and adding one
unit per row both prices that and makes the prefix strictly increasing, which stops a
power-law matrix collapsing every boundary onto the same row.

Ranges alone are not shippable. Three variants, same binary, interleaved:

| mode | L2 misses, ts-palko k=1 | x86 f32 cells >10% slower than base |
|---|---|---|
| 0 global counter (ships today) | 133363 | -- |
| 1 home ranges, no stealing | **5324** (below MKL's own 6637) | 21.4% |
| 2 home + front-stealing | 37039 | 1.1% |
| 3 home + back-stealing | -- | 1.5% |

Mode 1 has the fewest misses of anything measured, including MKL, and is rejected on
both hosts and three separate corpora: 21.4% of x86 float32 cells and 26.7% of float64
cells more than 10% slower, and on the large-A corpus it is worse than what ships
today (85 cells below MKL against 45). A range balanced in *work* is not balanced in
*time* on 8P+16E or 6P+12E, and nothing absorbs the straggler.

Front-stealing gives most of the locality back because it takes exactly the rows the
owner would have reached next, so that work migrates between cores from call to call.
Back-stealing takes from the far end instead. It claims through a compare-exchange on
a packed `(head, tail)` uint64 -- one word deliberately, because two separate counters
let an owner and a thief each read a stale view and hand out the same rows; with one
word head only rises and tail only falls, so there is no ABA either.

### What back-stealing measures

Kernel timer, interleaved random-order arms, per-cell same-code A/A control:

| host | dtype | mode 2 | mode 3 | >10% slower | A/A floor | below MKL, base -> mode 3 |
|---|---|---|---|---|---|---|
| x86 | f32 | 1.207x | **1.267x** | 1.5% | 2.3% | 472 -> **101** of 2172 |
| x86 | f64 | 1.172x | **1.213x** | 2.1% | 2.8% | 437 -> **84** of 2172 |
| ARM | f32 | 1.016x | 1.039x | 3.8% | 1.8% | (M5 has no MKL) |
| ARM | f64 | -- | 1.045x | 4.5% | 2.5% | -- |

Across both x86 dtypes: 909 of 4344 cells below MKL becomes 185. The vs-MKL geomean
goes 1.79 -> 2.26 (float32) and 1.91 -> 2.32 (float64). Positive in every k, A-size,
degree and duration band on both hosts, and the harmed tail is at or below the
same-code floor everywhere. `tsteal/steal` is 1.0495, faster on 1807 of 2172 cells.
One mode wins on both architectures, so there is no arm-variance to declare.

Part of the gain is not residency at all. On a corpus of 56 matrices with A between
16.8 and 225.5 MB the partition is still worth 1.042x *above* the machine's aggregate
32 MB of L2, where no assignment can keep A resident: contiguous ranges narrow the
band of B columns a worker touches, and 24-32 threads stop serialising on one atomic
line.

### The chunk width is not implicated

`scorch_spmm_chunk`'s ceiling `rows / (nthreads * chunks_min)` was calibrated against
the mechanism back-stealing replaces, so it owed a crossed grid. Over 2172 x86 float32
cells, with the partition on, against a 0.9988 A/A:

| CHUNKS_MIN | 2 (shipped) | 2 forced | 8 | 16 |
|---|---|---|---|---|
| kernel speedup over base | **1.1964** | 1.1881 | 1.1920 | 1.1884 |

Within 0.7% end to end, and the shipped width is the best of the four. No change.

### Cold, as a guard rather than a target

A real caller is in a reuse loop or is partially evicted by the rest of its pipeline,
never cold in the synthetic sense, so cold is asked only as "was anything traded
away". Flushing 256 MB between calls and taking the median of 21 single calls, ARM
float32, 372 cells: back-stealing is 1.035x on the whole-call timer cold and 1.080x
warm, and every cell is above the ATen reference in both regimes. It gains less cold
than warm, which is what a residency mechanism should do, and it does not go negative.
The first cold run carried no same-code arm, which made "slower on a third of the
cells" uninterpretable -- a median over flushed single calls is a far noisier
estimator than a min over a warm batch. The probe now carries one.

### The thread-count gate, a separate defect found on the way

`scorch_spmm_nthreads` gates its composition-adoption override on
`work = nnz * max(k, 16)`. The cache-line floor on k is right for throttling a
bandwidth-bound product and overstates a k=1 product sixteenfold, so a product with
12625 nonzero-units of real arithmetic reads 202000 against a 150000 grain and takes
the whole host team. This is the third site with that defect; the raise gate directly
above it already reads the unfloored measure for exactly this reason.

Crossed against the partition on ARM, 1650 cells per dtype:

| arm | f32 kernel | f64 kernel | >10% slower (f32/f64) |
|---|---|---|---|
| gate alone | 1.070x | 1.065x | 1.8% / 1.6% |
| partition alone | 1.045x | 1.053x | 3.2% / 3.3% |
| both | **1.115x** | **1.111x** | 1.8% / 2.5% |

against A/A floors of 3.5% and 2.7% -- both arms and the combination sit at or under
the floor. They are additive and they own different regions: the gate is worth
1.12-1.17x at k<=8 and A under 256 KB and is inert to within 0.3% elsewhere, the
partition is worth 1.21-1.27x on A between 4 and 16 MB and at k>=64. Each keeps
essentially all of its value on top of the other (gate on top of the partition 1.067,
partition on top of the gate 1.043).

The gate is the one change here with real downside risk, because the adoption it
prices exists to stop a GCN forward reshaping its team at every op boundary -- worth
pubmed 0.78 -> 1.15x when it landed. It does not ship on the ARM numbers alone.

### What the residual is now

101 of 2172 x86 float32 cells, and the shape of it has changed completely. Crossing
the surviving deficit against mean row degree and k:

| degree | k=1 | k=2 | k=8 | k=64 | k=256 | k=512 |
|---|---|---|---|---|---|---|
| 0-8 | 3 | 5 | 0 | 0 | 0 | 0 |
| 8-64 | 0 | 2 | 0 | 0 | 0 | 1 |
| 64-256 | 22 | 18 | 6 | 1 | 2 | 2 |
| 256+ | 9 | 14 | 8 | 1 | 5 | 2 |

(cells below MKL, of 362 per k). It is one class: high degree at narrow k. And 32 of
the 101 are held below the parallelism they could use by the policy rather than by
the structure -- `rows / SCORCH_ROWS_PER_THREAD` gives **four** threads for
`Meszaros/kl02`, which has 71 rows holding 212536 nonzeros, on a 1.7 MB A that is
L3-resident. Per thread we are already faster than MKL there; MKL wins on thread
count. "16 rows per worker" is a proxy for "enough work to amortise fork/join" and for
a row of 3000 nonzeros one row is plenty, so the fix is to express that ceiling in
work rather than in rows.

**This is not a decomposition limit, which is what it first looked like.** Row-parallel
work-splitting cannot exceed one worker per row, so a matrix with fewer rows than the
host has threads needs a nonzero-axis (segmented or merge-based) decomposition
instead. No cell in the residual is that matrix: the smallest row count among the 101
float32 cells and the 84 float64 cells is **64**, against 32 logical processors, and
all 32 of the ceiling-limited cells have at least 64 rows. Row-parallel can reach the
full width on every one of them. The limit is the policy alone.

The other 69 are not ceiling-limited. `lp_osa_14` has 2337 rows and already gets all
32 logical processors for a 78 us kernel at k=2, and reads 0.665; that is a
*too-many-threads* question, not too few, and the same sweep answers it.

Two candidate expressions of the ceiling, both implemented behind hooks and neither
routed, taking `kl02` at k=2 from four workers:

| candidate | what it changes | kl02 |
|---|---|---|
| `SCORCH_SPMM_NNZ_PER_THREAD=256` | states the requirement in nonzeros per worker, `max()`ed with the row proxy so it can only widen, capped at one worker per row | 4 -> 22 |
| `SCORCH_SPMM_RAISE_ON_FLOORED=1` | the existing raise's bound reads `nnz*max(k,16)` rather than `nnz*k`, on the grounds that how many workers a product can *feed* is a question about time | 4 -> 11 |

The first is inert on the low-degree shapes by construction: a 100000-row matrix of
degree 2 already resolves to the full width, and `max()` cannot lower anything.

### The nonzero ceiling fails on its own, and why

Measured on ARM over 1650 cells against back-stealing alone:

| arm | kernel speedup over ship | >10% slower | A/A floor |
|---|---|---|---|
| back-stealing | 1.0590x | 1.7% | 1.4% |
| + nonzero ceiling | **1.0319x** | **6.4%** | 1.4% |
| + raise on the floored measure | 1.0505x | 2.1% | 1.4% |

The ceiling change is a net loss and its harmed tail is four times the floor. The
mechanism is in the resolved thread counts, not in the timings: the policy resolves a
different count on only 108 of the 1650 cells, and 81 of the 107 harmed cells are in
that 108. The harmed shapes are 64-row pruned ResNet-50 layers of degree 44 to 288,
going from 4 workers to 6 and running 1.5 to 2x slower, and `kl02` itself at k=1,
going from 4 to 18 and running at 0.511.

The route is the **composition adoption**, which has no work bound at all. A 64-row
layer with 18432 nonzeros at k=1 gets `by_work = 1` from the policy, so the policy
alone would run it on one worker; the adoption then hands it `min(host, rows/16)` = 4
today and `min(host, rows_axis)` = 6 with the wider ceiling. The ceiling is not
choosing 6 workers for a 20-microsecond kernel -- the adoption is, and the ceiling
only removed the accident that was holding it back.

So the two changes are not independent and the ceiling cannot ship without the graded
adoption. The grain that grades it is bracketed rather than guessed: `kl02` needs
`work_true / G >= 22` to reach the width it wants and the 64-row layer needs
`work_true / G <= 4` to keep the width it has, which is **G in [4608, 19321]**. G near
5000 gives `kl02` its 24 and the ResNet layer 3 or 4, and 5000 nonzero-units is about
two microseconds of single-thread work -- a defensible "worth waking a worker" bar,
where `SCORCH_GRAIN_SPMM`'s 150000 is about sixty-five.

### What the k=8 story turned out to be

Single-threaded counters (one thread, so cycles and instructions are attributable --
with a team both runtimes spin-wait idle workers and only cache-miss deltas can be
read from a whole-process `perf stat`) said we lost 0.53-0.88x at k=8 with our
instructions per nonzero *rising* to 13.5-18.1 while MKL's *fell* to 7.9-8.9, and
that at k=1 we beat MKL 1.38-1.71x on 3.2-7.6 against its 13.7-14.8. The conclusion
drawn from that -- that k=1 was a parallel-efficiency problem and k=8 was the inner
loop -- was half right. With back-stealing, k=8 is below MKL on 14 of 362 float32
cells and 6 of 362 float64 cells, all of them at degree >= 64. The inner loop is worth
looking at, and the arm that does it (`regblock_deep`, which resolves a group of
addresses at once and drops the prefetch, about 5.75 instructions per nonzero against
the shipped loop's ~10.5) has never been run at k >= 2, because the hook was only ever
discussed at k=1 where the shipped path is the *gather* kernel and enabling it swaps
kernel families. But it is no longer where the residual is.

## The shipping decision, and the two arms it rejects

The crossed grid on the host that has MKL, 2172 cells per dtype, kernel timer,
interleaved arms, per-cell same-code control:

| arm | f32 speedup | f32 >10% slower | f32 below MKL | f64 speedup | f64 >10% slower | f64 below MKL |
|---|---|---|---|---|---|---|
| ships today | 1.0000 | -- | 463 | 1.0000 | -- | 431 |
| thread gate alone | 0.9955 | 9.6% | 446 | 0.9678 | 13.5% | 411 |
| **back-stealing alone** | **1.2416** | **1.7%** | **99** | **1.2013** | **1.5%** | **78** |
| gate + front-stealing | 1.1325 | 6.5% | 327 | 1.0717 | 12.0% | 292 |
| gate + back-stealing | 1.1650 | 6.4% | 286 | 1.0869 | 11.4% | 276 |

A/A floors 2.1% and 2.2%. **Back-stealing alone is the answer** and it is not close:
894 of 4344 cells below MKL becomes 177, and its harmed tail is under the floor on
both dtypes. On the narrow-k-weighted grid (k = 1, 2, 4, 8, 16, 32) it reads 1.3286x
and 1.2867x with harmed tails of 0.7% and 0.8% against 0.8% and 1.3% floors, taking
705 and 556 cells below MKL down to 137 and 102.

### The thread-count gate is rejected, and it is arm-variance

Pricing the composition adoption on real arithmetic instead of `nnz*max(k,16)` read
**1.070x float32 and 1.065x float64 on ARM**, with harmed tails *below* the noise
floor, which is what made it look shippable. On x86 the same change is **0.996 and
0.968** with harmed tails of 9.6% and 13.5% against 2.1% floors, and crossed with the
partition it costs six to nine percent and multiplies the harmed tail by four.

The cause is visible in the resolved thread counts, not in the timings. The adoption
is all-or-nothing: clear the grain and the whole host team is taken, miss it and the
count falls back to the policy's, which under one grain is **one**. A Cora-shaped
output layer -- 2708 rows, 13264 nonzeros, k=7, so 92848 multiply-adds -- goes from 24
workers to 1. On the 24-thread host that cliff is 24x; on the 6-thread host it is 6x,
and there it lands inside the E-core noise. Same change, opposite sign, because the
cliff height is the host's thread count.

So it does not ship, and the graded form -- cap the adopted count so each worker gets
a grain of *real* arithmetic, never below one, so the pipeline still shares one team
-- is what gets measured instead.

### The deep-unroll register kernel is rejected at narrow k

`regblock_deep` runs UNROLL independent nonzero streams instead of two and drops the
prefetch. It had never been run at k >= 2, because the hook was only ever discussed at
k=1 where the shipped path is the gather kernel. Run at k = 1..32 against
back-stealing:

| arm | f32 | f64 | f32 >10% slower | below MKL (f32) |
|---|---|---|---|---|
| back-stealing | 1.3286 | 1.2867 | 0.7% | 137 |
| + unroll 4 | 1.2274 | 1.1893 | 14.5% | 310 |
| + unroll 8 | 1.2131 | 1.1668 | 13.8% | 317 |

A ten percent loss with a harmed tail eighteen times the floor. By k it is worst
exactly where it was supposed to help -- k=2 reads 1.0757 against back-stealing's
1.2839, k=4 1.0908 against 1.2967 -- and its only positive region is float32 k=16 and
k=32 (1.3970 and 1.4116 against 1.3680 and 1.3760), where the mask is nearly full
anyway. So stream depth is not what narrow k is short of, which leaves the mask and
the address arithmetic, and those are what the exact-width kernel removes.

## The exact-width kernel, and the two policy arms that survive

### Exact widths are worth ten percent where the mask covered the whole row

x86 float32, 1448 cells, k = 1, 2, 4, 8, against back-stealing, per-cell same-code
control at 0.9987 with 0.7% harmed:

| arm | speedup over ship | >10% slower | vs MKL | below MKL |
|---|---|---|---|---|
| ships today | 1.0000 | -- | 1.4173 | 599 |
| back-stealing | 1.3064 | 0.4% | 1.8515 | 121 |
| + exact width, unroll 2 | 1.3529 | 0.3% | 1.9174 | 92 |
| **+ exact width, unroll 4** | **1.3620** | **0.2%** | **1.9304** | **74** |
| + exact width, unroll 8 | 1.3614 | 0.3% | 1.9295 | 68 |

By k, against back-stealing alone, with the per-k floor beside it:

| k | floor | unroll 4 | unroll 8 |
|---|---|---|---|
| 1 | 0.9994 | 0.9925 | 0.9896 |
| **2** | 0.9975 | **1.1042** | **1.1039** |
| **4** | 0.9981 | **1.0885** | **1.0920** |
| 8 | 0.9996 | 0.9906 | 0.9888 |

Ten percent at k=2 and nine at k=4, and the two widths where the kernel cannot fire
read the floor -- k=1 goes to the nonzero-axis gather, k=8 fills the vector so
regblock is already mask-free. The residual below MKL nearly halves again, 121 to 74.

The 0.7 to 1.0% those inert columns sit below 1.0 is the hook build's own cost: with
the hook, `if (narrowk_exact)` is a runtime branch taken once per ROW, and a row of
five nonzeros at k=8 is about fifty instructions. In a build without hooks the
dispatch is unconditional and only the loop-invariant `switch (B1_size)` remains,
which hoists like the `nvec` switch beside it -- but that is an argument, not a
measurement, and it is what the two-build comparison is for.

### The nonzero-expressed ceiling failed, and the failure was one line of mine

It failed alone on ARM (6.4% of cells more than 10% slower against a 2.6% floor), and
grading the adoption did not rescue it. ARM float32, each arm against back-stealing
alone:

| arm | vs back-stealing | >10% slower |
|---|---|---|
| same-code floor | 0.9949 | 2.6% |
| binary gate | 1.0469 | 2.1% |
| graded adoption, G=15000 | 1.0330 | 2.1% |
| graded adoption, G=5000 | 1.0062 | 2.3% |
| graded adoption, G=2500 | 0.9999 | 1.8% |
| + nonzero ceiling, G=15000 | 1.0111 | **6.4%** |
| + nonzero ceiling, G=5000 | 0.9832 | **6.7%** |

The diagnosis was that the route was the composition adoption, which has no work
bound. That diagnosis was right, and the code was wrong in one line: the widened
ceiling was shared with the adoption as well as with the base path. The base path
pairs its ceiling with `min(work / grain)`, so widening it there cannot over-thread a
small product -- the 64-row layer's `by_work` is 1 either way. The adoption has no such
term, and it was the adoption handing those layers six workers.

With `rows / SCORCH_ROWS_PER_THREAD` restored in the adoption, the resolved counts for
a 24-thread host are:

| shape | ships | + ceiling | + graded G=5000 | both |
|---|---|---|---|---|
| kl02 k=2 (71 rows, 212536 nnz) | 4 | **22** | 4 | **22** |
| nw14 k=4 (73 rows, 904910 nnz) | 12 | **32** | 12 | **32** |
| bibd_17_8 k=8 (136 rows) | 18 | **32** | 18 | **32** |
| rn50 256-row k=1 | 16 | **31** | 16 | **31** |
| rn50 64-row k=1 (the harmed shape) | 4 | 4 | 3 | 3 |
| rn50 64-row k=64 (harmed) | 4 | 4 | 4 | 4 |
| cora output k=7 | 24 | 24 | 18 | 18 |
| cora hidden k=16 | 24 | 24 | 24 | 24 |
| pubmed k=16 | 24 | 24 | 24 | 24 |
| reddit k=64 | 32 | 32 | 32 | 32 |

Every shape the ARM run harmed is now untouched by the ceiling, and every shape the
residual needs still widens. So **the table above measured a different rule from the
one that now exists**, and it is recorded as the reason for the fix rather than as a
verdict on the fix. Both hosts owe it a fresh grid; the arm that had already started
on the older binary was killed a minute in rather than spend an hour confirming a
superseded rule.

## Every thread-count change on this branch is arm-variant

Three separate ways of changing how many workers the SpMM resolves were built and
measured on both hosts. All three flip sign between the two.

| change | ARM | x86 |
|---|---|---|
| price the adoption gate on real arithmetic | **1.070x / 1.065x**, tails under the floor | **0.996 / 0.968**, tails 9.6% / 13.5% |
| grade the adopted count by real arithmetic (G=15000) | **+3.3%** over back-stealing, tail at the floor | **0.9122**, tail 18.7% against a 1.4% floor |
| grade at G=5000 | +0.6% | 0.9831, tail 10.0% |
| state the row ceiling in nonzeros (corrected) | (owed) | 0.9889, tail 1.5% -- inside the floor |

The mechanism is not mysterious and it is the same one each time. On the 24-thread
host the fallback when adoption is declined or capped is the policy count, which under
one grain is **one**, so every cap is a cliff twenty-four workers deep; on the
six-thread host the same cliff is six deep and lands inside the E-core noise. The
cliff height *is* the host's thread count. Anything that narrows when the host team is
adopted therefore has to be worth more than a 24x drop on the cells it catches, and on
x86 nothing measured here is.

So **no thread-count change ships**. Back-stealing does not touch the policy at all --
it changes which rows a worker gets, not how many workers there are -- which is why it
is the one change that reads the same sign on both hosts.

The corrected row ceiling is the one arm still open, and it is open for a narrow
reason: it is inside the noise floor on the general corpus (0.9889 against 0.9960 with
the same harmed tail), because it can only change an answer on 32 of 2172 cells. Its
decision rests entirely on whether those 32 -- `kl02` and its class, held to four
workers on a 1.7 MB L3-resident matrix -- are actually faster with more, which is what
the forced-thread-count sweep over those 52 matrices asks and nothing else can.

## Scope and gaps, stated plainly

- **dtype.** Everything above is float32. float64 CSR × dense has its own section and
  its own kernel: the register-blocked drop-in was made generic over the scalar type, so
  on x86 float64 now resolves `spmm_csr_double_v2` and is **above MKL on 129 of 129
  grid cells (geomean 2.273x)**, where the reference kernel it replaced lost on 45. It is
  still true that float64 **gets no tiling at all** — the tiled kernels have no float64
  instantiation and `tiling_gate` is float32-only on purpose.
- **float64 on ARM is deliberately unchanged.** NEON has no masked load/store, so the
  SIMD traits layer has no ARM specialization and `spmm_csr_double_v2` falls back to the
  reference kernel off AVX2. Routing ARM through the generic path measured 1.642x
  geomean on the M5 but regressed `gcn__pubmed@k=8` to 0.689; two fixes failed and the
  cause is unexplained, so it was not shipped. A NEON float64 register kernel is open
  work, and its ragged tail needs a mechanism the AVX2 mask supplied for free.
- **Index memory.** The narrowing memo holds an int32 copy alongside the caller's
  int64 arrays, i.e. +50% on index memory. It replaces a same-size allocation that was
  happening every call, so peak does not grow, but steady state does. An entry is
  dropped as soon as its source array dies — the sweep runs before every insert, not
  only when the map fills, because otherwise 4096 dead graph-scale index arrays would
  accumulate. Narrowing at `STensor` construction and dropping the int64 arrays would
  instead *halve* index memory and remove the memo entirely; that lives in
  `stensor.py`/`storage.py`. See the JIT section for why narrowing beats compiling the
  kernel for int64.
- **Which paths reach the selector.** The drop-in float32 CSR×dense symbol, and — as
  of the composition section above — `scorch.compile`'s traced fused SpMM+bias+act.
  **`sparse_linear` / `sparse_linear_fm` still do not**, and that is a real remaining
  instance of the same lockout: they are a separate prebuilt dispatch that never
  consults the gate. Wiring them is not a copy of the traced-path fix. Their bias is
  per output channel (`bias[:, None]`, broadcast along rows, not along the free
  dimension), the activation includes `sigmoid`, the fused symbol is the same for every
  activation so the memo tag must carry the activation rather than the symbol, and
  `sparse_linear_fm` deliberately never builds an `STensor` for its dense operand —
  which is what `resolve_prebuilt_matmul` needs. A survey of the 56 layer
  configurations in the autoencoder figure (7 models × 4 layers × 2 sparsities) found
  exactly one that the gate would admit: stl10 at 0.95 sparsity, layer 1, a 28 MB dense
  operand against the M5's 16 MiB LLC. That survey used uniformly random column indices
  at the right density rather than trained checkpoints, so treat it as a pointer, not a
  result. Separate change, its own tail family, its own grid. SpMV and SpMSpM are
  unaffected by the selector by shape, not by omission.
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
- **The grid and the selector re-measurement predate the dispatch levers.** They were
  measured at `14e3ea6`/`6eec90f`; nothing in them is invalidated, but every small-cell
  ratio in them now understates scorch by ~33 µs of per-call Python. The float64 grid is
  the exception: it was re-taken on the current tree, in one process, and so carries no
  such correction.
- **Call plans cover one operation.** `SpmmCsrPlan` serves CSR × dense with a dense
  output — the drop-in SpMM at float32 and float64 and the float64 reference kernel,
  plus the two tiled kernels the selector can choose. SpMV, SpGEMM, COO operands, sparse outputs, the fused
  bias/act and Linear kernels and everything on the generated-kernel route get no plan and
  are byte-unchanged. The plan machinery costs them a dict probe on a `__dict__` that has
  no plans in it.
- **Lever 5 is measured on both hosts**, unlike levers 1–4: its base tree is small enough
  to reproduce locally (`git archive HEAD` built in place), which the earlier levers'
  container-VM problem had made impossible. Its tests run on both too — 70 on redwood, 68
  + 2 skipped on the M5, the skips needing a generated kernel that toolchain cannot build
  — including the bitwise comparison against every legacy symbol a plan can stand in for.
  What is *not* two-host is everything above lever 5 in this document.
- **The chunk grid was measured in the instrumented build, not the one that ships.**
  `SCORCH_TUNE_HOOKS` is what makes the override possible, and it also allocates the
  workspace pool, adds a `force_workspace` branch per row, and turns two compile-time
  predicates into runtime ones. Every ratio in that grid is between two arms of the
  *same* instrumented binary, so the overhead is on both sides — and a constant per-row
  cost added to both sides of a ratio moves it toward 1, which means the shipped build's
  response is at least what is reported and not less. That is an argument, not a
  measurement: no arm of the chunk grid ran in the binary that ships. Confirming it
  would need two shipped builds differing only in `SCORCH_SPMM_CHUNK_MINRATIO`, which
  reintroduces exactly the build-to-build variance the hook exists to remove.
- **`K` is still one constant for both hosts.** The gate is what makes that defensible —
  below twice the generic width the rule does not act, so a wrong `K` cannot do harm
  there — but it is not the same thing as deriving `K` on the host. The implied-`K`
  distributions do overlap across hosts, which is the condition originally set for one
  constant; they also each span three orders of magnitude, so the overlap is weak
  evidence. Measuring `K` at startup from a microbenchmark of a contended atomic against
  a nonzero of work is the principled version and is not done.
- **The two uncovered regimes are synthetic.** `syn__aeshape` and `syn__prodshape` place
  uniform-random column indices at a fixed degree, so they have neither a real degree
  distribution nor real locality, and duplicate columns within a row are left in (they
  feed `spmm_csr_float_v2` directly, which validates no indices, and a repeated column
  only re-reads a B row). They exist to put a measurement at the `(rows, nnz, k)` points
  where the rule departs from generic and no real matrix in the grid does. Treat the
  decisions as exact and the magnitudes as a pointer.
- **The fused Linear kernel's chunk is unmeasured.** It keeps the generic width, stated
  in the code rather than defaulted into. Adopting the rule there needs the autoencoder
  grid, which is its workload, and would have to re-check the @0.99 ledger.

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

## The corrected row ceiling is a null, and grading the adoption is a loss

The ceiling rule ("let nonzeros, not just rows, express how many workers a matrix can
feed") was re-measured after the one-line fix that had leaked the widened row axis into
the composition adoption. The earlier ARM rejection had therefore measured a different
rule. This is the corrected one, on redwood, 2172 float32 cells (ks 1, 2, 8, 64, 256,
512), interleaved arms, `p0` duplicated as the same-code control:

| arm | kernel speedup | >10% slower | vs MKL | below MKL |
|---|---|---|---|---|
| ships today (`p0`) | 1.0000 | — | 1.8095 | 477/2172 |
| **back-stealing (`p3`)** | **1.2670** | **1.2%** | **2.2927** | **104/2172** |
| + row ceiling | 1.2579 | 1.2% | 2.2762 | 107/2172 |
| + graded adoption G=5000 | 1.2424 | 1.8% | 2.2481 | 247/2172 |
| + graded adoption G=15000 | 1.1817 | 4.9% | 2.1384 | 303/2172 |
| + ceiling + G=5000 | 1.2344 | 1.7% | 2.2336 | 250/2172 |
| + ceiling + G=15000 | 1.1752 | 5.3% | 2.1266 | 293/2172 |

A/A control: 0.9979 geomean, 1.7% harmed.

**The ceiling is a null.** 1.2579 against 1.2670 is 0.7% down on a floor of 0.9979, and
it costs three cells (104 → 107) rather than recovering any. It does not reach the
residual it was designed for and it does not harm anything either; there is nothing to
ship.

**Grading the adoption is a loss, and the loss is exactly where we need the wins.** By k,
G=5000 reads 1.1706 at k=1 and 1.2268 at k=2 against back-stealing's 1.2848 and 1.2901,
while k=8 and above are unchanged (1.3524 vs 1.3422). So the grain that was bracketed to
rescue 64-row ResNet layers on ARM pays for them by throttling narrow-k SpMM on x86 —
the regime the whole branch exists to win. G=15000, the other end of the bracket, is
worse still and quadruples the harmed tail.

That closes the pair. The ceiling needed graded adoption to be safe, graded adoption
costs more than the ceiling was ever going to return, and the ceiling returns nothing.
Back-stealing alone remains the change.

The float64 half of the same grid (2172 cells, A/A 0.9979 / 1.5% harmed):

| arm | kernel speedup | >10% slower | vs MKL | below MKL |
|---|---|---|---|---|
| ships today (`p0`) | 1.0000 | — | 1.9127 | 439/2172 |
| **back-stealing (`p3`)** | **1.2374** | **1.1%** | **2.3667** | **79/2172** |
| + row ceiling | 1.2362 | 1.1% | 2.3645 | 67/2172 |
| + graded adoption G=5000 | 1.1963 | 2.0% | 2.2881 | 188/2172 |
| + graded adoption G=15000 | 1.1197 | 6.6% | 2.1416 | 292/2172 |

Grading loses on both dtypes and for the same reason — 1.1635 at k=1 and 1.2530 at k=2
against back-stealing's 1.3161 and 1.3077.

The ceiling deserves a more careful word than "null". Its geomean is unchanged on both
dtypes (0.1% and 0.7% down, against a 0.2% floor), but the count below MKL moves in
*opposite directions*: 104 → 107 on float32 and 79 → **67** on float64. Twelve float64
cells recovered for free is not nothing, and it says the mechanism does reach some of the
few-row high-degree residual rather than missing it entirely. What it does not do is
reach it *reliably enough to ship as an unconditional policy change*, and the arm that
was supposed to make it safe costs an order of magnitude more than it returns. So the
rule stays out of the default, and the open question narrows to a sharper one the forced
thread sweep can answer: on the specific few-row matrices, does more width actually help,
and is the region it helps separable by a runtime condition that provably cannot fire on
the shapes the ARM run showed it harms.

## Where the row ceiling actually pays, and the analysis error that nearly hid it

Two corrections to the section above, one of method and one of substance.

**The method error, because it is the interesting one.** Decomposing the ceiling's
effect, I divided `p3_kms` by `aa_kms` and called the result a same-code noise floor. It
read 0.887 inside the high-degree band and 0.681 over the widest row band, which looked
like an alarming positional bias on a harness that randomises arm order every repetition
and reports min-of-reps. It is not a bias. `aa` duplicates `ARMS[0]`, and for this grid
`ARMS[0]` is **`p0`** — the baseline — so `p3/aa` is the partition's own speedup wearing
the label of a floor. The shared analyzer had been hardened against exactly this after it
reported a 1.4189 floor once; the ad-hoc script I wrote alongside it had not. The floor is
`p0/aa`, and it is 0.985–1.014 inside every band examined below.

**The substance.** Aggregated over 2172 cells the ceiling is a null, but that null is a
sum of a real gain and a real loss which the aggregate hides. Scored against `p0/aa`
inside each region, with a z on the difference of the two means:

| region | n | p3n/p3 (f32) | z | p3n/p3 (f64) | z |
|---|---|---|---|---|---|
| rows ≤ 128 and mean degree ≥ 192 | 42 | **1.1109** | 3.38 | **1.1542** | 3.18 |
| rows ≤ 128 and mean degree < 256 | 276 | 0.9837 | −2.63 | 1.0071 | 1.85 |
| rows ≤ 128, any degree | 318 | 0.9996 | 0.09 | 1.0254 | 3.35 |

The third row is the first two cancelling. Below MKL inside the gated region goes 24 → 14
on float64 and is unchanged at 26 on float32, whose cells there are too far behind (kl02
sits at 0.52) for 11% to close.

Both conditions are necessary and neither is a tuned constant. The complement is
*negative* on float32 with z = −2.63, so dropping the degree condition costs 1.6% on 276
cells to buy 11% on 42. And the region is a plateau rather than an edge: rows ∈ {96, 128,
192} crossed with degree ∈ {192, 256} all give 42–48 cells at 1.108–1.164 with z of
3.2–3.9, while rows ≤ 256 admits the 256-row ResNet bottlenecks and the float32 harmed
tail jumps from 4.8% to 12.2%.

The mechanism is the one the rule was designed for, now localised: `rows / 16` caps the
worker count, and it is the wrong cap exactly when it would leave most of the machine idle
*and* each row still has thousands of nonzeros to chew. kl02 (71 rows, degree 2993) and
nw14 (73 rows, degree 12396) are the shape; a 64-row ResNet layer at degree 288 with
18432 nonzeros total is the shape that must not be caught, and the degree condition alone
does not exclude it — its own measured gain is what keeps it in.

So the ceiling ships gated, not unconditionally, and the gate is measured on one host and
one corpus so far. Next: confirm on the M5, then on the large-A corpus, which is held out
from the grid the thresholds were read off.

## The output-size gate was set two steps too loose

The gate that switches the row partition off on large outputs shipped at four times the
last-level cache, a value picked from the band table rather than swept. Swept now on the
large-A corpus (56 matrices; 204 float32 and 183 float64 cells; same-code floor 0.9964 and
0.9978), against what ships today:

| threshold | f32 | >10% slower | f64 | >10% slower |
|---|---|---|---|---|
| 1x LLC (36 MB) | 1.0917 | 0.0% | 1.0604 | 0.5% |
| **2x LLC (72 MB)** | **1.0944** | **0.0%** | **1.0649** | **0.5%** |
| 4x LLC (144 MB, shipped) | 1.0918 | 1.5% | 1.0636 | 1.6% |
| 8x LLC (288 MB) | 1.0916 | 2.5% | 1.0589 | 3.8% |
| no gate | 1.0866 | 2.9% | 1.0544 | 4.9% |

Below MKL falls 29 → 4 (float32) and 18 → 2 (float64) at every threshold, so the gate is
not what wins the corpus — the partition is. What the threshold buys is the tail, and it
buys it monotonically: 0.0% harmed at 2x against 2.9% ungated.

By output size the mechanism is exactly the predicted one. The partition gains
1.13–1.14x below 16 MB and 1.04–1.07x from 16 to 64 MB, then flattens, and every cell it
harms sits in the 144–256 MB band, where 8x reads 0.9213 (f32) and 0.9536 (f64) and the
ungated arm 0.9153 and 0.9498, while 1x and 2x hold 1.00. 1x is not better than 2x
because it also switches the partition off through the 16–64 MB band that still pays
(1.0573 against 1.0730 on float32).

Changed to 2x. The constant is only read inside `if (partition_mode != 0)`, and the
partition still defaults off, so this is inert in the binary that ships today — argued,
not yet measured, and the two-build check is owed.

## No fixed thread width beats what the policy already resolves

`SCORCH_SPMM_NT_FORCE` overrides the final count after both the policy and the
composition adoption, so it answers directly whether the residual is short of workers.
Run on the 52 matrices that were still below MKL, 208 cells per dtype, against
back-stealing with the policy's own choice:

| forced width | f32 | >10% slower | below MKL | f64 | >10% slower | below MKL |
|---|---|---|---|---|---|---|
| the policy's choice | 1.0000 | — | 78/208 | 1.0000 | — | 64/208 |
| 4 | 0.6649 | 81.2% | 155 | 0.6451 | 84.6% | 161 |
| 8 | 0.7927 | 81.2% | 141 | 0.8197 | 80.3% | 124 |
| 16 | 0.9400 | 42.8% | 100 | 0.9495 | 42.8% | 95 |
| 24 (the host's torch count) | 0.9822 | 20.2% | 78 | 1.0236 | 13.0% | 53 |
| 32 (every logical processor) | 0.8515 | 37.5% | 127 | 0.8729 | 39.9% | 109 |
| row ceiling | 1.0058 | 6.7% | 78 | 1.0170 | 3.8% | 65 |

Floors 0.9988 / 1.0075. **Nothing beats the policy**, including forcing every logical
processor, which costs 13–15%. Forcing the host's torch count is the only arm that comes
close, and on float64 it does recover eleven cells below MKL — at a 13.0% harmed tail
against a 3.8% floor, so it is not a shape that can ship.

Per matrix the aggregate hides two populations pulling opposite ways, which is exactly
why every fixed rule loses. Nineteen of the 52 have *some* fixed width beating the policy
by more than 5%, and they split cleanly: the ultra-sparse many-row matrices want more
workers than the base grain allows (Pd_b, 8081 rows holding 6323 nonzeros, reads 1.828 at
16 workers; Pd_rhs 1.622; bips07_3078_iv 1.404), while the 64-row pruned ResNet layers
want *fewer* and collapse if pushed (0.608 at 24, 0.352 at 32). One rule cannot serve
both, and the policy already sits between them.

So the thread-width line is closed. Three separate mechanisms — the binary gate, the
nonzero ceiling ungated, and graded adoption — are all rejected, and a direct sweep now
shows there was no fixed width to find. What remains of the residual is per-thread
efficiency and, for the few-row high-degree cases, the decomposition itself: kl02 is 71
rows and 212536 nonzeros, already faster than MKL per thread, and row-parallel gives it
at most 71 units of work no matter how the policy is tuned. That is a kernel question,
not a policy one.

## Cold: the kernel wins by 1.71x and the call loses by 0.87x

The cold probe flushes 256 MB between calls and times exactly one, taking the median
over 21 repetitions because the minimum over cold repetitions picks whichever flush
evicted least. MKL is in the same interleave. 372 float32 cells:

| regime | arm | vs MKL, kernel only | vs MKL, whole call | non-kernel | below MKL |
|---|---|---|---|---|---|
| cold | ships today | 1.611 | 0.837 | 88.3 µs (49.7%) | 284/372 |
| cold | back-stealing | **1.714** | **0.866** | 88.2 µs (52.5%) | 249/372 |
| warm | ships today | 0.986 | 0.806 | 7.6 µs (18.7%) | 247/372 |
| warm | back-stealing | 1.429 | 1.096 | 7.5 µs (24.5%) | 177/372 |

The same-code control reads 1.0129 cold and 1.0007 warm, so the cold estimator is about
ten times noisier than the warm one, as expected, and the arm differences still clear it.

**Cold, our kernel is 71% faster than MKL's entire call and we still lose the call by
13%.** The reason is the fourth column: 88 µs of non-kernel cost cold against 7.5 µs
warm, and it is *flat* — 87.2, 87.6, 95.1, 134.7 µs across kernel-duration bands from
20 µs to over a millisecond. A fixed cost, not a proportional one, and it is not the
partition: base and back-stealing carry the same 88 µs.

Two things go into it and they need separating rather than lumping. Some is irreducible:
a Python dispatch faulted in from a cold instruction cache will always cost more than one
C++ dispatcher call, and `torch.sparse.mm` is the latter. But some is the harness. The
plan cache in `ops.matmul` serves a repeat call directly, and it is gated on
`not kwargs and type(b) is torch.Tensor` — so asking for `time_dict` to get a kernel
timer, or passing an `STensor` for B, disables it. **Every grid in this branch does both.**
Arm against arm that cancels exactly; against MKL's whole call it is a handicap a real
caller does not carry.

Measured directly on the M5 warm, the plan path saves 5–33 µs (typically 10–20) on the
same shapes, so it is not the whole 88 µs — the cold figure really is mostly cold-cache
cost on the dispatch path. The cold run is repeating with a `plan` arm that reaches
back-stealing the way a caller reaches it, with a per-cell check that the two paths agree,
so the user-facing cold number stops being measured through a handicap the user would not
have.

## Which vs-MKL number is the honest one

Every grid here reports `mkl_ms / ours_kms` — MKL's whole call over our kernel only. That
looks like it flatters us, and the obvious correction is `mkl_ms / ours_ms`. On float32
the two readings are far apart: back-stealing is 2.2927x with 104 of 2172 cells below MKL
on the first and 1.5440x with **512** below MKL on the second. Worth knowing which is
right before quoting either.

It is the first, and the reason is the plan cache. `ops.matmul` serves a repeat call from
a per-tensor plan, gated on `not kwargs and type(b) is torch.Tensor`; the harness passes
`time_dict` to get a kernel timer *and* an `STensor` for B, so it disables the fast path
twice over. Measured on the M5, median of 201 calls each:

| shape, k | kernel | general path − kernel | plan path − kernel | plan saves |
|---|---|---|---|---|
| 2708 rows, 13264 nnz, k=8 | 35.1 µs | +11.9 | **−0.3** | 12.3 |
| 2708 rows, 13264 nnz, k=1 | 28.1 µs | +5.3 | **−1.6** | 6.8 |
| 8081 rows, 6323 nnz, k=1 | 22.2 µs | +7.8 | **+2.3** | 5.5 |
| 9506 rows, 395506 nnz, k=8 | 101.1 µs | +14.8 | **+7.0** | 7.8 |

**On the caller's path the whole call is the kernel**, to within the measurement, on every
shape tried. So `mkl_ms / ours_kms` is not a flattering approximation of the user-facing
number, it *is* the user-facing number, and the 512-cells-below-MKL reading is an artifact
of a handicap the harness imposes on itself and a caller never sees.

Two consequences. The first is that the whole-call column in every grid above should be
read as "what a caller would get if they also asked for a kernel timer", not as the
user-facing figure. The second is that this needs confirming on x86, where the dispatch
constant is a different size — the cold rerun carries a `plan` arm reaching back-stealing
the way a caller reaches it, with a per-cell agreement check, and it reports both regimes.

It also reframes the cold result. Cold, non-kernel cost was 88 µs against 7.5 µs warm; if
the plan path removes as much of the cold constant as it removes of the warm one, the cold
whole-call number moves a long way from 0.866x. That is the measurement the rerun exists
to make, and it is the last thing standing between the two halves of the target.

## Stating the target as a distribution instead of a count

"How many cells are below MKL" turns out to be a bad summary, and the reason is worth
recording. Two harnesses measured the *same* 372 cells and reported 52 and 110 cells below
MKL. They agree: our kernel reads 7.8% slower in the cold probe than in the corpus grid,
because the cold probe's warm batch follows a 256 MB flush and gets one warm-up call,
where the corpus grid settles and then runs a full batch. MKL only slows 1.7% under the
same treatment, which is itself the residency effect back-stealing exists to fix. An 8%
shift moved 58 cells across the line, which means the line is where the mass is, and a
count near it is mostly reporting its own sensitivity.

The distribution says it properly. 2172 cells, caller-facing basis:

| dtype | arm | p5 | p25 | median | p75 | p95 | below 1.0 | below 0.9 |
|---|---|---|---|---|---|---|---|---|
| f32 | ships today | 0.61 | 1.16 | 2.06 | 2.93 | 4.19 | 22.0% | 19.4% |
| f32 | **back-stealing** | **1.00** | 1.42 | 2.62 | 3.52 | 4.76 | **4.8%** | **1.6%** |
| f64 | ships today | 0.65 | 1.22 | 2.24 | 3.08 | 4.40 | 20.2% | 17.5% |
| f64 | **back-stealing** | **1.05** | 1.46 | 2.68 | 3.61 | 4.81 | **3.6%** | **1.0%** |

**The fifth percentile moves from 0.61 to 1.00 on float32 and from 0.65 to 1.05 on
float64.** Cells more than 10% behind MKL go from 19.4% to 1.6% and from 17.5% to 1.0%.
Most of what remains "below MKL" is within 10% of parity — 11.3% of float32 cells sit in
that band — so it is a tie, not a loss, and it is the band an 8% measurement shift moves
wholesale.

Cold, on the same basis, is our *strongest* regime rather than our weakest: back-stealing
reads 1.714x with 13 of 372 cells below MKL on float32 and 1.685x with 10 of 372 on
float64, against warm's 1.429x/110 and 1.507x/72. That is the expected direction once
stated plainly — MKL's advantage is keeping 93% of A in the cores' L2 between calls, and a
flush takes that away from MKL too.

## The partition code is free when it is off, measured against the tree without it

Mode 0 is not byte-identical to the pre-partition source: acquiring the row range moved
from a const initialiser to mutable locals behind a runtime branch. That owed a
measurement rather than an assertion. Three builds, float32, 495 cells of the tb corpus:

| comparison | ratio | >10% slower |
|---|---|---|
| old2 / old1 — two runs of the *same* pre-partition build (the floor) | 0.9950 | 7.7% |
| **new with the partition off / old1 — the code present but inert** | **1.0058** | **5.7%** |
| same, whole call | 1.0073 | 5.5% |
| old1's own in-process same-code arm (for contrast) | 1.0022 | 2.6% |
| new with the partition on / old1 | **1.2917** | 2.6% |

Float64 says the same on 333 cells: floor 1.0178 with 3.3% harmed, present-but-off
1.0000 with 3.3% harmed, z = **-2.33**, partition on 1.2824.

z of "present but off" against the two-run floor is **0.85** on float32 and **-2.33** on
float64 — neither separable. The code costs nothing when it is off, and the partition switched on is worth 1.2917x against the
tree that predates it, which is the honest end-to-end number rather than a mode-0-versus-
mode-3 comparison inside one binary.

Worth noting for later cross-process work: the same-code floor is 1.0022 with 2.6% harmed
*within* a process and 0.9950 with 7.7% harmed *across* two, so a cross-process comparison
carries about three times the tail. That is the floor the three-build shipped-shape stage
has to clear, and it is why that stage runs each build twice in opposite orders.

## What is actually left, and what limits each part of it

With back-stealing, 34 of 2172 float32 cells and 22 of 2172 float64 cells are more than
10% behind MKL. Small enough now to name every one, and they sort into four mechanisms.
`P` below is the workers the row axis can actually feed with the gate in place, and
`bound` is `max_row / (nnz/P)` — how much worse than perfect balance the longest row makes
the best possible row-parallel schedule.

| what limits it | f32 | f64 | worst | the mechanism | status |
|---|---|---|---|---|---|
| narrow-k per-thread efficiency | 10 | 7 | 0.605 | masked loads and an imul per nonzero at k=1,2 | exact-width kernel, grid running |
| longest row | 8 | 2 | 0.558 | `bound` 2.0–3.9: one row is a tenth of the matrix | **nothing built** |
| ultra-sparse, grain-bound | 6 | 4 | 0.538 | mean degree < 1, held to one worker by the base grain | grain sweep queued |
| few rows, very high degree | 6 | 9 | 0.609 | `rows/16` caps the width | the gate, validation queued |
| other | 4 | 0 | 0.673 | 64-row layers at k=256 | — |

Three of the four already have a measurement in flight. The fourth is new and it is the
one place row-parallelism itself is the wrong shape: lp_osa_14 is 2337 rows and one row
holding 38336 of its 317097 nonzeros, lp_osa_30 is 4350 rows with one holding 72555 of
604488, and nw14 is 73 rows with one holding 90951 of 904910. Once the width is widened to
32 workers those give bounds of 3.87, 3.84 and 3.22 — so **no scheduling of whole rows can
get within a factor of three of balanced**, however many workers are available, and every
policy arm measured in this document was arguing about the wrong axis for these cells. The
mechanism that fits is splitting a long row's nonzero range across workers and reducing
the partials, which is a kernel change rather than a policy one and is worth ten cells of
2172, so it goes behind the three measurements already running rather than ahead of them.

Note also what is *not* in the table: nothing is limited by decomposition width any more,
which was the diagnosis three sections ago. The forced-width sweep closed that, and this
is what the residual looks like once it is.

## Checking the gate resolves what it claims, without a machine

A policy change is checkable without timing anything: compile the policy header into a
standalone binary with the hooks in and ask it what it resolves. That needs no rebuild of
the extension and cannot disturb a run in progress, which matters when both hosts are
mid-grid. (One trap: torch's bundled `libomp.dylib` carries the install name
`/opt/llvm-openmp/lib/libomp.dylib`, so a standalone link needs
`-Wl,-headerpad_max_install_names` and then `install_name_tool -change`, the same rewrite
`scorch_build.py` does for the extension.)

Resolved worker counts, `omp_get_num_procs()` = 18:

| shape | ceiling off | gated | ungated | capped at pool 6 |
|---|---|---|---|---|
| kl02, 71 rows, degree 2993 | 4 | **18** | 18 | **6** |
| nw14, 73 rows, degree 12396 | 6 | **18** | 18 | **6** |
| bibd_17_8, 136 rows, degree 5005 | 8 | 8 | 18 | 8 |
| rn50 256 rows, degree 1152 | 16 | 16 | 18 | 16 |
| rn50 64 rows, degree 288 | 4 | 4 | 4 | 4 |
| cora hidden layer | 18 | 18 | 18 | 18 |
| Pd_b, degree < 1 | 1 | 1 | 1 | 1 |

Three things fall out. The gate excludes the 256-row ResNet bottlenecks that the ungated
rule was harming, which is what it is for. It also excludes bibd_17_8 at 136 rows — and
the residual measurement says the ungated rule was worth 1.142x there, so the `rows <= 128`
edge is leaving a real gain on the table; that is exactly what the `rows <= 192` arm in the
validation exists to price. And the 64-row layer at degree 288 is *inside* the gate yet its
count does not move, because the work term binds before the row axis does — which is why
it was never the shape the ceiling harmed.

The pool cap is confirmed x86-inert by construction rather than by hope. On redwood the
gated count is 22, not 32, because `work / grain` = 3400576 / 150000 binds first; capping
at torch's 24 therefore changes nothing. On the M5 `omp_get_num_procs()` = 18 binds first,
and capping at torch's 6 takes kl02 from 18 workers to 6. So the arm is a no-op on the host
where the ceiling helps and a real change on the host where it hurts, which is the
prediction the grid now has to confirm.

## Design for the one unaddressed mechanism: splitting a long row

Recorded now because the analysis is done and the measurements that come first may change
whether it is worth building.

**The problem.** Ten cells are limited by a single row. lp_osa_14 holds 38336 of its 317097
nonzeros in one row of 2337; lp_osa_30 holds 72555 of 604488 in one of 4350; nw14 holds
90951 of 904910 in one of 73. With the width widened to 32 workers the balanced share is
9909, 18890 and 28278 nonzeros, so those rows are 3.87x, 3.84x and 3.22x the share. **No
assignment of whole rows to workers can beat that**, so every policy arm in this document
was arguing about the wrong axis for these cells.

**The shape of the fix.** Split only rows longer than a threshold, and only those:

1. The normal row-parallel pass runs over all rows as today, except that a worker reaching
   a row longer than the threshold writes its output zeros and skips its arithmetic.
2. A second pass handles the long rows. For each, the workers divide its nonzero range,
   each accumulating `k` partials into its own slot of a `P x k` scratch buffer, and the
   slots are then summed into that output row.

Boundary handling is the whole difficulty in the general merge-based formulation — a
worker owning an arbitrary nonzero range has two partial rows and empty rows can make the
row-from-nonzero mapping ambiguous. Restricting the split to rows chosen *by index* avoids
all of it: ownership of every row stays exactly as it is today, so zeroing stays correct
for empty rows, and the only shared state is one scratch buffer per long row.

**The cost when it does not fire, which is the part that decides it.** Finding the long
rows needs `max` over `A1_pos` differences, which is O(rows) per call. On Pd_b that is 8081
subtractions against a 25 µs kernel — around a tenth of it, which is far too much to pay on
every call for a mechanism that fires on three matrices. So the statistic cannot be
computed in the kernel. It belongs with the tensor, computed once: the plan cache in
`ops.matmul` already stores per-tensor state keyed by `(b.shape, b.dtype, generation)`, and
the longest row is a property of A alone, so it can be computed on the first call and
passed to the kernel as an argument thereafter. That also means the mechanism is reachable
only on the path that has a plan, which is the caller's path and not the harness's — so the
grid harness would need the same statistic threaded through, or the arm cannot be measured
at all.

That last point is the reason this is written down rather than started: it needs an ABI
change to the native symbol, and the ABI is what the previous round of work spent its time
removing per-call costs from. Ten cells of 2172 does not obviously justify that, and three
measurements now in flight will move which cells are left.

## The shipped shape on ARM, and the regression the average hid

Three hookless builds of one source tree, differing only in `-D` flags: `ship` (what ships
today), `cand` (`PARTITION_DEFAULT=3`, `NARROWK_EXACT_UNROLL=4`), and `ctrl`, a second build
of `ship` with identical flags. Three separate `.so` files cannot be interleaved inside one
process, so each build ran twice — once in the order ship, ctrl, cand and once reversed —
and the two passes are averaged per cell, which cancels a monotonic drift instead of
letting it land on whichever build went last. `ship` and `ctrl` both came out at 987168
bytes, `cand` at 987360.

float32, 1650 cells, two passes averaged:

| comparison | kernel | whole call | >10% slower |
|---|---|---|---|
| ctrl / ship — the cross-process floor | 1.0081 | 1.0072 | 0.7% |
| **cand / ship — the change** | **1.0352** | **1.0328** | 3.9% |
| cand / ctrl — the same from the other side | 1.0269 | 1.0254 | 6.1% |

Positive at every width, and rising with it: 1.0248 at k=1, 1.0233 at k=2, 1.0216 at k=4,
1.0299 at k=8, 1.0537 at k=64, 1.0588 at k=256, against a floor of 1.0035–1.0107. ATen's
own agreement across the three processes has a geomean spread of 1.0260 and a worst case of
1.577x, which is the honest scale of cross-process noise on this host.

**The tail is where the result is.** 64 of 1650 cells are more than 10% slower under `cand`
and only **one** of them is also slow in the same-code control, so 63 are attributable.
They are not spread out:

| group | cells | shape | k |
|---|---|---|---|
| ss:as-735 | 35 | 7716 rows, mean degree 1 | 2, 4 |
| dlmc:rn50 | 15 | mostly 64 rows, degree 288 | 1, 4, 8, 64 |
| dlmc:transformer | 9 | 512–2048 rows | 1, 4 |
| others | 4 | — | — |

By width the tail is {1: 13, 2: 14, 4: 21, 8: 9, 64: 5, 256: 2}. The exact-width kernel
fires only at float k=2..7 and float k=1 is not routed to it by default, so **the 35
as-735 cells and the k=2/k=4 tail are the kernel's, and the 29 cells at k=1, 8, 64 and 256
cannot be — they belong to the partition and are a separate question** the exact-width grid
has the arms to settle.

The kernel's part has a mechanism, and it is in the code rather than in the schedule. The
prologue is `UNROLL*K` zero stores and the epilogue `(UNROLL-1)*K` adds, both paid once per
row whatever that row holds. At K=4 and UNROLL=4 a row of a single nonzero does sixteen
zero stores, twelve reduction adds, four multiply-adds and four stores — about thirty-two
operations of overhead for four of work. as-735 is 7716 rows of mean degree 1, so that
overhead *is* the regression. Fixed by choosing the unroll from the row's own length: one
halving loop of at most three steps, landing on an instantiation that already exists, and a
row long enough to fill the configured depth compiles to exactly what it did before. Off by
default, because both hosts had grids in flight measuring the unadapted kernel and an arm
is worth more than a silent change.

## The exact-width kernel does not ship as it stands, and the widened grid is why

The earlier reading — 1.3064 to 1.3620 on float32, below MKL 121 to 74 — came from a grid
at k = 1, 2, 4, 8. Widening it to every width where the kernel fires (float 1..7, double
1..3) plus k = 8 and 16 as controls reverses the verdict. 3258 cells per dtype, same-code
floor 0.9998 and 1.0001:

| arm | f32 | >10% slower | below MKL | f64 | >10% slower | below MKL |
|---|---|---|---|---|---|---|
| back-stealing | 1.0000 | — | 191/3258 | 1.0000 | — | 132/3258 |
| + exact, unroll 4 | **0.9873** | 10.9% | 296 | **1.0039** | 2.1% | **95** |
| + exact, unroll 8 | 0.9965 | 9.8% | 283 | 0.9919 | 4.3% | 123 |
| + exact at k=1 too | 0.9771 | 14.6% | 294 | 0.9942 | 2.0% | 98 |

By width, float32 unroll 4: 0.9723 (k=1), **1.0666** (k=2), **1.0506** (k=3), 0.9942,
0.9692, **0.9132** (k=6), 0.9611 (k=7), then 0.9832 and 0.9843 at k=8 and 16 where it
cannot fire. Float64 unroll 4: 0.9764, **1.0699**, **1.0611**, then 0.9867–0.9908 for k>=4
where it cannot fire.

Two separate defects, both visible in those numbers.

**The per-row dispatch is a tax on every width the kernel does not serve.** The control
widths read 0.983–0.987 on float32 and 0.988–0.991 on float64 — a consistent 1.2–1.7%.
`B1_size` is loop-invariant, so testing `narrowk_exact` and then switching on it once per
row is work that belongs outside the row loop entirely. That tax is paid by the majority of
widths in exchange for helping two.

**The accumulator spills at the wider widths.** `T acc[UNROLL][K]` is UNROLL*K scalars
live across the row. At K=2, UNROLL=4 that is 8 — fine. At K=6, UNROLL=4 it is 24, and
float32 k=6 is the worst cell in the grid at 0.9132. The unroll has to be bounded by a
register budget divided by K, not chosen independently of it.

So the honest verdict is that the kernel's *idea* is right — it wins 6.1–7.0% at k=2 and
3.4–6.1% at k=3 on both dtypes, and on float64 it recovers 37 of the 132 cells below MKL
even carrying both defects — and its *implementation* is not shippable yet. Three fixes are
identified and two are already written: bounding the unroll by the row's length (committed,
off), bounding it by a register budget divided by K, and hoisting the width decision out of
the row loop.

## The base grain is rejected, globally and by degree band

150000 nonzero-units is about sixty-five microseconds of single-thread work, which is a
conservative bar for waking a second worker, and it is what holds the ultra-sparse matrices
to one thread. Swept alone on 1810 cells per dtype against a 1.0% and 1.5% floor:

| grain | f32 | >10% slower | below MKL | f64 | >10% slower | below MKL |
|---|---|---|---|---|---|---|
| 150000 (shipped) | 1.0000 | — | 102/1810 | 1.0000 | — | 74/1810 |
| 75000 | 0.9661 | 11.6% | 110 | 0.9748 | 10.9% | 90 |
| 50000 | 0.9414 | 19.6% | 116 | 0.9480 | 18.7% | 103 |
| 30000 | 0.9174 | 26.2% | 137 | 0.9319 | 24.8% | 107 |

Monotonic in the wrong direction on every measure. It is also rejected *within the band it
was aimed at*: on the 495 cells of mean degree below 1, grain 50000 reads 0.9706 on float32
and 0.9875 on float64 — still worse than the baseline.

One thing it does do: in that band, below MKL goes 7 to 0 on float32 and 5 to 0 on float64.
So a lower grain fixes precisely the handful of ultra-sparse cells that are behind MKL while
costing the several hundred in the same band that were already ahead. Mean degree is
therefore the wrong discriminator, and twelve cells do not justify hunting for the right one
ahead of the other work. **A global grain change is settled: no.**

## The gate is inert off its own region, measured on held-out data

The gate's thresholds were read off `final_groups`, so `big_groups` — the large-A corpus,
56 matrices — is the first honest test. All 255 float32 cells fall *outside* the gated
region, which makes it a test of inertness rather than of gain, and it passes cleanly:
0.9993 for the gate, 0.9999 ungated, 0.9995 with the pool cap, all within z of ±0.6 of a
0.9987 floor, and **0.0% of cells harmed by any arm**. Below MKL is 20 for back-stealing and
18–19 for every gated arm.

The liveness check on the same rebuild also confirms the pool cap is x86-inert by
construction rather than by hope: uncapped 22, capped 22, pool 24 — exactly what the
standalone policy driver predicted before the run.

## There is no ARM exact-width kernel, so the ARM grid for it could never have measured anything

`scorch_spmm_row_narrow_exact`, its dispatch, and every one of its hooks sit inside
`#if defined(__AVX2__) && defined(__FMA__)`. An ARM binary therefore has no exact-width
kernel to switch on and does not even contain the string `SCORCH_NARROWK_EXACT_SHORT`,
which is why the ARM stage refused to run: the check was right and the premise behind the
stage was wrong. Its header had reasoned about what ARM's *shipped* narrow-k path is —
`scorch_spmm_row_neon_regtile`, a 4×4 kernel over 128-bit vectors — and concluded ARM
"should still gain" at float k=1..3, where a q-register load produces 4 of 16 useful
bytes. That conclusion is about an implementation nobody wrote.

Two consequences worth stating. The ARM shipped-shape candidate was built with
`-DSCORCH_NARROWK_EXACT_UNROLL=4` and that flag was **inert**, so the ARM number below is
a clean measurement of the partition alone and not a two-change candidate. And the ARM
stage was replaced with the measurement the ARM data actually asks for, which is the next
section.

## The partition on ARM: 1.035 and 1.092 above the floor, with a tail the average hides

Three hookless builds of one tree — `ship`, a second build of `ship` as the cross-process
floor, and `cand` with `-DSCORCH_SPMM_PARTITION_DEFAULT=3` — each run twice in opposite
orders and the two passes averaged per cell, 1650 cells per dtype:

| comparison | float32 kernel | float64 kernel | float32 >10% slower | float64 >10% slower |
|---|---|---|---|---|
| ctrl / ship (the floor) | 1.0081 | 1.0191 | 0.7% | 1.2% |
| **cand / ship (the change)** | **1.0352** | **1.0918** | 3.9% | 3.4% |
| cand / ctrl (the other side) | 1.0269 | 1.0713 | 6.1% | 4.1% |

MKL's own agreement across the three processes is 1.0260 (float32) and 1.0496 (float64),
which is the independent floor no scorch flag can move. By k, the float32 gain rises with
width (1.025 at k=1 to 1.059 at k=256) and float64 is flat at 1.09–1.12 until k=256 drops
to 1.043.

The tail is the finding. 64 of 1650 float32 cells and 56 of 1650 float64 cells are more
than 10% slower, against floor tails of 0.7% and 1.2%, and only one float32 cell has the
floor below 0.9 as well — so it is the change, not the process. **63 of the 64 have an
output under one megabyte.** The worst are rn50 bottleneck layers (64 rows of degree 288,
whose entire A is 147 KB) at 0.70–0.80, a 512-row transformer layer at 0.76, and as-735
(7716 rows of degree 1, A = 117 KB) at 0.81.

## A minimum-A gate, argued from the mechanism rather than fitted to the tail

The partition's whole benefit is that a worker touches the *same* slice of A on every
call, so the slice stays in that core's private cache; the shared counter hands out a
different slice each call, which is why it re-fetched all 8.6 MB of ts-palko's A every
call against MKL's 579 KB. **When the whole of A already fits in one core's private
cache, there is nothing to keep resident** — the counter's rotating assignment is
resident too — and all that is left to pay is claiming chunks and, at the end of the
call, every exhausted worker scanning every cursor for something to steal. A 147 KB A is
exactly that case, and so is a 117 KB one.

`SCORCH_SPMM_PARTITION_MINA_DIV` switches the partition off when A's bytes times the
divisor are still under the queried LLC. It is a divisor of a queried quantity and not a
byte constant: a sixteenth of redwood's 37.7 MB is 2.4 MB, which is an i9-14900K P-core's
L2, and the same divisor means the analogous thing on the M5's reported 16.8 MB. Default
0, so nothing changes until the ladder has a plateau on both hosts. The ladder runs `p0`
as an arm on purpose — a gate that switches the partition off has to land on `p0`'s
number for the cells it fires on, and if it does not then it is not the fallback it
claims to be.

## On ARM the row ceiling is harmful inside its own gate, and the pool cap is why

The ARM gate grid, 1375 cells per dtype, 35 of them inside the gated region:

| arm | float32 in-gate | harmed | float64 in-gate | harmed |
|---|---|---|---|---|
| floor | 0.9959 | 2.9% | 0.9989 | 2.9% |
| ungated rule | 0.9117 (z −3.2) | 28.6% | 0.9536 | 20.0% |
| the gated rule | 0.9321 (z −2.5) | 25.7% | 0.9690 | 14.3% |
| gate one step wider | 0.9215 | 25.7% | 0.9542 | 14.3% |
| **gated + capped at the pool** | **0.9871** | **2.9%** | **0.9965** | 8.6% |

So the rule that reads 1.11 and 1.15 inside the gate on redwood reads 0.93 and 0.97 on
the M5, and capping the widened count at the pool the caller manages recovers almost all
of it — 0.9871 and 0.9965, both inside the floor. That is the mechanism the design
predicted: `omp_get_num_procs()` is 32 against torch's 24 on redwood but 18 against
torch's 6 on the M5, so the same rule widens kl02 to 22 workers inside a 24-thread pool
on one host and to 18 — three times the pool, pulling in twelve efficiency cores — on the
other. **The pool cap is the version of this rule that could ship**, and it is x86-inert
by construction (uncapped 22, capped 22, pool 24).

## Two runners collided, and what was discarded

A chain had been parked waiting for the gate grid to finish. Interrupting that grid — so
that the hookless three-build number, which had never run, could go first — released the
parked chain, and for about two minutes redwood was doing two things at once: the parked
chain rebuilt `scorch_ops` at 20:25:47 while the environment-cost ladder began timing on
that same `.so` at 20:25:49. Discarded: `envcost_float32.csv` and
`short8_nk2_float32.csv`, both renamed `.aborted` rather than deleted.

The second file would have been worthless anyway, and for a reason worth stating: it was
the exact-width grid, and the binary it was running on predates
`SCORCH_NARROWK_EXACT_ACCUM`, so **three of its seven arms were silent duplicates of
`p3e4`** — a hook that does not exist in the binary reads as absent, and the arm then
measures the arm above it. That is the same failure the ARM stage refused to commit
earlier, caught by a `strings` check on the `.so`; the x86 stage had no such check because
its rebuild normally precedes it in the same script.

Redwood now runs from a single script that refuses to start if anything else is timing,
rebuilds once from the synced headers so every hooked stage shares one source revision,
and orders the stages by decision value rather than by when they were written.

## Setting an environment variable costs about a percent, and it has been distorting the arm comparisons

The ARM gate grid had an anomaly that could not be a real effect. Four arms are *provably*
inert outside the gated region — `SCORCH_SPMM_NNZ_PER_THREAD` has exactly one consumer and
it sits inside the gate's condition — and yet on the 1335 cells outside it they read
0.9913, 0.9770, 0.9753 and 0.9647 against a 0.9974 same-code floor, at z between −5 and
−21. Ordered by how many variables each arm sets — one, two, two, three — the deficits are
monotone in the count.

So a ladder of six arms was run, all the **same configuration**, differing only in how many
names nothing in the binary reads are also set. One of them, `p3b`, has an *identical*
environment to the reference, which separates "how long the environment is" from "which
slot in the sequence the arm occupies". Both hosts, both dtypes, no rebuild:

| extra variables | x86 f32 | x86 f64 | ARM f32 | ARM f64 |
|---|---|---|---|---|
| 0, identical environment | 1.0003 | 1.0006 | 0.9997 | 0.9999 |
| 1 | 0.9888 | 0.9920 | 0.9964 | 0.9953 |
| 2 | 0.9805 | 0.9785 | 0.9899 | 0.9894 |
| 4 | 0.9656 | 0.9633 | 0.9807 | 0.9805 |
| 8 | 0.9348 | 0.9373 | 0.9618 | 0.9632 |

Two arms with identical environments agree to within 0.06% on all four grids, so the arm's
position is not the cause. **One extra variable costs 1.1% on x86 and 0.4–0.5% on ARM**,
and by kernel duration the whole effect is in the short cells: on x86 float32 the
eight-variable arm is 0.9294 below 30 µs, 0.9833 between 30 and 100 µs, and 0.9897 above
100 µs. That is the signature of a fixed per-call cost, and the mechanism is not subtle: an
instrumented build calls `getenv` about thirty times per SpMM, most of those are misses,
and a miss scans the whole environment.

It predicts the anomaly it was built to explain. Against the x86 slope the four gate arms
should read 0.989, 0.980, 0.980 and 0.967; they read 0.9913, 0.9770, 0.9753 and 0.9647.
**Every one within 0.3%.**

Three earlier readings change, all in the same direction:

- **The exact-width kernel's rejection was mostly the instrument.** Its control columns at
  k=8 and 16, where the kernel cannot fire, read 0.983–0.987, and that is what the "per-row
  dispatch tax" was inferred from. Priced for one extra variable they are 0.994–0.998. The
  kernel itself goes 0.9873 → ≈0.9985 on float32 and 1.0039 → ≈1.0153 on float64. The
  hoist is still right as code — a loop-invariant test does not belong in a row loop — but
  the number attached to it in `ef7a770`'s message was misattributed.
- **The gated row ceiling's x86 rejection flips sign.** "1.2579 against back-stealing's
  1.2670" becomes ≈1.2721 against 1.2670.
- **On ARM the pool-capped ceiling becomes positive inside its gate**, 0.9871 → ≈1.007 at
  the ARM slope for its two extra variables.

The fix is structural rather than a correction factor: `kprobe --pad-env` pads every arm up
to the widest arm's variable count with names nothing reads, so the environments are the
same length and `getenv` costs every arm the same. It is off by default so it cannot change
the meaning of a grid that was already half-run, and it is now on for every stage whose
arms differ in count — including the GCN guardrail, whose four arms set 1, 1, 2 and 2 and
whose cora and citeseer layers are exactly the sub-30 µs cells this hits hardest. The cold
probe needed no change: it sets one variable for every arm and only varies its value.

A second consequence is worth stating even though nothing acts on it yet. That per-call
`getenv` constant is present in **both** arms of every instrumented comparison, so it
dilutes every ratio on a short kernel toward one. The hookless three-build comparison is
the only measurement here that is free of it, and it is the one that found the ARM
small-output tail at all.

## The partition's min-A gate is refuted, and duration is what separates the two populations

The min-A gate was argued from a mechanism: when the whole of A fits in one core's private
cache there is nothing for home ranges to keep resident, so the partition can only cost.
The ARM ladder says no. Over 1650 cells, against a 0.9953 floor, the divisor ladder reads
0.968 (÷8) to 0.980 (÷64) pooled — and the reason is in the split. On the 537 cells where
the partition **wins** more than 5%, the same arms read 0.909 to 0.948. Even at ÷64, which
only fires below about 260 KB of A, the gate takes 5% off the wins. It is giving back
exactly what the partition is for, so A's size does not separate the two populations —
home ranges help small matrices too, because contiguous ranges narrow each worker's B
column band and the workers stop contending on one counter line.

The gate does work on the cells it was built for: on the 51 float32 cells where the
partition is more than 10% behind the counter, ÷8 reads 1.1294 and lands within 2.6% of
`p0` there. The problem is only that it fires far too widely.

What separates the two populations is **duration**. min / q1 / median / q3 kernel time:

| | min | q1 | median | q3 |
|---|---|---|---|---|
| partition ≥10% behind the counter | 19.2 µs | 27.3 | 28.5 | 29.5 |
| partition ≥5% ahead | 15.6 µs | 30.9 | 62.1 | 110 |

Searching every feature the grid carries, in both directions and in every two-feature
conjunction, the cleanest single separator is **`nnz*max(k,16)`** — the work proxy the
thread policy already uses — at 3.15e5 on float32 and 3.43e5 on float64, catching 74% and
77% of the regressed cells against 14.5% and 13.7% of the winning ones. The two
conjunctions that score higher both contain `rows <= 7716`, which is one matrix's row
count, and are rejected as fitted to the corpus rather than to a mechanism. Both thresholds
are about **two `SCORCH_GRAIN_SPMM`**, so the gate is spelled in grains:
`SCORCH_SPMM_PARTITION_MINGRAINS`, default 0, ladder 1–4 running now on both hosts. The
min-A hook is removed rather than kept behind a flag, because a rejected hook still costs a
`getenv` per call in every instrumented build, and the section above is what that costs.

## The x86 shipping number was a three-way A/A, and how it announced itself

The first completed run of the hookless three-build comparison on x86 read **cand/ship
1.0022 against a 1.0007 floor**, with 4.0% of cells more than 10% slower against a 4.2%
floor tail, and by k it was 0.9955, 1.0054, 1.0015, 0.9936, 1.0088, 1.0086 — flat
everywhere. The interleaved grid on the *same* corpus (`final_groups`, 362 matrices) and
the same k values reads **1.31 uniformly**, k=1 through k=16, against a 1.000 floor. Both
cannot be true, and the flatness is what gives it away: a real mechanism does not read
1.000 ± 0.006 in every band.

`kprobe.py` did `sys.path.insert(0, <its own directory>/src)`. On this host kprobe lives
in the measured tree, so that path holds a built `scorch_ops`, and inserting it at position
zero **overrode the `PYTHONPATH` each arm was given**. All three arms loaded the
instrumented tune build with no variables set, i.e. `SCORCH_SPMM_PARTITION_DEFAULT=0` — one
binary, three times. Confirmed directly rather than inferred:

    scorch_ops -> /scratch/bobbyy/mklcheck/tune/src/scorch_ops...so

with `PYTHONPATH` pointing at `cand`. Every arm's `-D` flag was present in its own build log
and every build produced a distinct `.so` (1012288 bytes for ship and ctrl, 1016528 for
cand), so nothing upstream of the run was wrong — the binaries were built correctly and
then not used.

The ARM three-build number is unaffected and was checked, not assumed: there kprobe lives
in the scratchpad, `<scratchpad>/src` does not exist, so the insert was a miss and
`PYTHONPATH` won. 1.0352 and 1.0918 stand.

Three changes, in the layer each belongs to:

- `kprobe` falls back to its own tree **only when the caller has not set `PYTHONPATH`**.
- `kprobe` prints the resolved `scorch` and `scorch_ops` paths in every log, so which
  binary ran is on the record whether or not anyone thought to check.
- `rw_stage6` **refuses** any run whose printed `.so` is not under that arm's own tree.
  This is the check that would have caught it, and it is the third time this session that
  the thing which found a silent null was a guard on the *output* rather than on the last
  line a script reached.

The six quarantined files are kept as `.threeway_aa`. The shipping number is unmeasured
again, and it is first in the queue.

## The exact-width kernel replicates on ARM, at the same widths and nearly the same size

The kernel had never run on ARM because it had never existed there. Its first ARM grid,
padded, 1650 cells per dtype, against back-stealing:

| k | 1 | **2** | **3** | 4 | 8 | 64 |
|---|---|---|---|---|---|---|
| ARM float32 | 0.9943 | **1.0623** | **1.0638** | 0.9962 | 0.9964 | 0.9961 |
| ARM float64 | 0.9908 | **1.0443** | **1.0591** | 0.9952 | 0.9981 | 0.9992 |
| x86 float32, corrected | 0.9833 | **1.0787** | **1.0625** | 1.0054 | 0.9944 | 0.9954 |
| x86 float64, corrected | 0.9843 | **1.0785** | **1.0697** | 0.9960 | 0.9988 | 0.9981 |

Same two widths, same sign at every other width, and 4.4–7.9% where it fires against floors
of 0.9982–0.9984. **Two hosts, two dtypes.** The widths it does not serve read 0.991–1.005,
so the family is inert off its own region on both architectures.

## No steal mode recovers the partition's short-kernel regression, so there is nothing to fix in the steal path

The code says the steal scan cannot be the cost: a failed claim returns after one relaxed
load — it only compare-exchanges a range that has work — and the victim offset never
rewinds, so the wasted atomics per worker per call are bounded by `nsplit`, about six loads.
The grid agrees. On the cells where back-stealing falls more than 10% behind the shared
counter (118 float32, 93 float64):

| arm | f32 on those cells | still behind `p0` | f64 on those cells | still behind `p0` |
|---|---|---|---|---|
| home ranges, **no** stealing | 1.0543 | 0.9124 | 0.9766 | 0.8396 |
| front-stealing | 0.9611 | 0.8317 | 0.9685 | 0.8327 |
| minimum-work gate | 1.0927 | 0.9456 | 1.1009 | 0.9464 |

Removing stealing entirely leaves 8.8% (float32) and 16% (float64) on the table, and
front-stealing is *worse* than back-stealing. Only switching the partition off recovers
those cells. So the cost is inherent to static home ranges on a 20–30 µs kernel — imbalance,
or a range nobody reaches until the tail — and no change to how stealing works addresses it.

## ARM, against what ships today

1650 cells per dtype, kernel time, padded, `p0` = the shared counter that ships:

| arm | f32 pooled | >10% slower | >10% faster | f64 pooled | >10% slower | >10% faster |
|---|---|---|---|---|---|---|
| back-stealing | 1.0137 | 118 | 203 | 1.0183 | 93 | 219 |
| **+ exact-width kernel** | **1.0316** | 119 | **315** | **1.0326** | 91 | **309** |
| + minimum-work gate | 1.0117 | **38** | 171 | 1.0159 | **36** | 190 |

The `aa` duplicate of back-stealing reports 114 and 98 cells more than 10% slower than `p0`,
against back-stealing's own 118 and 93 — so that set is reproducible rather than noise, and
it is not something the partition does to *some* cells but a stable property of those cells.

The exact-width kernel adds 1.8% and about 100 more decisively-faster cells on both dtypes
**without adding a single slower one**. The gate cuts the more-than-10%-slower count by
about two thirds for 0.2% pooled and roughly 30 fewer decisive wins. Nothing measured yet
carries both, which is what the combination ladder is for.

## Cold and warm, on the path a caller actually takes: 1.20x and 1.65x above MKL

The question was whether scorch beats MKL cold and warm. Everything before this measured
the harness's path, not a caller's, and the two differ: asking the probe for a `time_dict`
and handing it an `STensor` B both disable the per-tensor plan cache that
`matmul(A, B)` gets for free. `cold_probe.py`'s `plan` arm calls it the way a caller does.
It has no kernel timer by construction — the whole point is that nothing extra is asked
of it. 372 cells, the general corpus, caches flushed with 256 MB between cold readings,
back-stealing compiled in. Every number is MKL's time over ours.

| | k=1 | k=8 | k=64 | all |
|---|---|---|---|---|
| **cold, caller path (float32)** | 1.0539 | 1.2058 | 1.3658 | **1.2018** |
| cold, harness path | 0.7570 | 0.8817 | 1.0745 | 0.8951 |
| **warm, caller path (float32)** | 1.1954 | 1.6100 | 2.3217 | **1.6471** |
| warm, harness path | 0.8530 | 1.1789 | 1.8594 | 1.2320 |
| **cold, caller path (float64)** | 1.0857 | 1.2464 | 1.4644 | **1.2560** |
| **warm, caller path (float64)** | 1.1681 | 1.7799 | 2.2526 | **1.6731** |

Cells below MKL, whole call: cold 81 of 372 (float32) and 53 (float64); warm 37 and 31.
On the harness path those counts are 245 and 168.

The non-kernel cost is where the two paths separate. Median over all cells, float32:
88.0 µs cold and 7.1 µs warm on the harness path, **42.4 µs cold and −0.4 µs warm** on the
caller's. A warm caller pays nothing measurable for dispatch — the whole call is the
kernel — and the cold penalty halves.

**Where the remaining losses are.** Warm, they are one shape family: 24 of the 37 sit at
degree ≥ 64 with ≤ 1024 rows, and the worst are 256-row pruned-ResNet layers at degree
691–1152 (0.638 and 0.647 at k=1), then `lp_osa_14`, `connectus`, `nw14`, `kl02`. By
output width, k=64 loses 1 cell of 124 and k=1 loses 25. Cold, they are the short calls:
the band where MKL itself takes under 10 µs reads 0.9885 with 63.8% of cells below parity,
while the 100 µs band reads 1.2396 with 14.1% — cold, a fixed cost of tens of microseconds
is the whole measurement on a 10 µs kernel.

## The row ceiling was rejected by the environment cost, and on padded held-out data it is a win

The nonzero-expressed row ceiling — where a product's width was limited by the row count,
state the requirement in nonzeros instead — was recorded as a null: 1.2579 against
back-stealing's 1.2670. That comparison charged the arm about a percent for naming two
extra variables (`## Setting an environment variable costs about a percent`). Re-measured
with `--pad-env`, so every arm sets the same number of names, on the three corpora the
thresholds were *not* read off:

| corpus | region | n | floor | the rule (`p3ng`) | harmed | below MKL |
|---|---|---|---|---|---|---|
| nk2 | in gate | 10 | 1.1129 | 1.2865 (f32) / 1.3973 (f64) | 0.0% / 0.0% | 8→4 / 7→2 |
| fewrow | in gate | 15 | 0.9214 / 0.9629 | 1.1763 / 1.2071 | 6.7% / 6.7% | 6→3 / 8→4 |
| final | in gate | 35 | 1.0269 / 1.0575 | 1.0900 / 1.2124 (z=3.12) | 5.7% / 0.0% | 14→10 / 15→4 |
| all three | outside | 575+240+1770 | ~0.998 | 0.9966–1.0019 | ≤2.1% | unchanged |

Inert outside its gate on 2585 cells, positive inside on all three corpora in both dtypes,
and the cells below MKL inside the gate fall by more than half. The pool-capped variant
(`p3ngc`) reads the same on x86 (1.2239 / 1.0781 / 1.2009 in-gate, inert outside), so if it
is what ARM needs, taking it costs x86 nothing.

The ungated form is a different matter and stays rejected: `p3nu` costs 0.7% on 1770
outside-gate cells with z = −4.71 (f32) and −4.98 (f64). Both conditions are load-bearing.

**What is still open is the row condition, and 128 rows is not it.** The rule exists
because the row proxy, not the arithmetic, held the width down — and that is a comparison
against the available width, `rows/16 < pool`: rows < 384 on redwood's 24-thread pool and
rows < 96 on the M5's 6-thread pool, where the constant says 128 on both hosts. It matters
here: the largest group of cells still below MKL on the caller path is 256-row
pruned-ResNet layers at degree 691–1152, which a 128-row threshold excludes and the
row-bind statement includes, while the 64-row layers the ARM run regressed on are inside
either form, so this cannot make that tail worse. `SCORCH_SPMM_CEIL_ROWBIND` states it that
way; default off, both forms queued against each other on both hosts.

## The ARM three-build measured a laptop, not a change, and its own controls said so

The sliced ARM three-build exists because the unsliced one is unreadable. It ran each
build over the whole 275-matrix corpus before starting the next, so the three readings of a
cell were minutes to hours apart, and the run spanned fifteen hours. Its controls:

| | float32 | float64 |
|---|---|---|
| ctrl / ship (two identical builds) | 0.9469, 50.9% of cells >10% apart | 0.9580, 37.1% |
| reference column across processes (identical code) | spread 1.3163, worst 2.502x | 1.5604, worst 2.635x |
| cand / ship (what was being asked) | 1.0018 | 1.2453 |

The float64 reading would be a large win and it cannot be claimed: two builds of the same
source disagree by more than 10% on 37% of cells, and code no flag touches moved 1.56x.
The x86 run of the same comparison passes the same test (floor 0.9800 / 1.0186 with 4.8% /
4.1% harmed, reference spread 1.079), so this is the host, not the method.

`an_ship3.py` now refuses instead of reporting when a control is looser than the effect —
reference spread above 1.10, or two identical builds more than 10% apart on more than 15%
of cells. It fires on the run that motivated it and passes the x86 run. The fix to the
measurement is locality rather than repetition: slice the corpus into 25-matrix pieces and
rotate the three builds *within* each slice, so a cell's three readings are about four
minutes apart, with both orders per slice so no build keeps the late position.

## The GCN guardrail, finally with a control tight enough to read

Three earlier attempts produced no verdict. The third measured nothing at all: it passed
five dataset names comma-joined to `--dataset`, which takes one name or the literal
`"all"`, so all six passes printed `Unknown dataset: cora,citeseer,...`. (`"all"` is not
usable either — `load_dataset()` runs *before* the weights check, so a dataset with no
weights still triggers a multi-gigabyte download.) One invocation per dataset, three
rotations, per-dataset minimum across passes, equal environment-variable counts:

| dataset | ships today | back-stealing | + ceiling | + gate | same-code control |
|---|---|---|---|---|---|
| cora | 0.279 | 0.264 | 0.248 | 0.266 | 1.053x |
| citeseer | 0.542 | 0.523 | 0.532 | 0.546 | 1.004x |
| pubmed | 0.727 | 0.594 | 0.596 | 0.593 | 1.022x |
| ogbn-arxiv | 82.334 | 82.843 | 81.522 | 82.841 | 1.015x |
| reddit | 425.895 | 427.923 | 425.333 | 428.460 | 1.008x |

Milliseconds, non-fused Scorch, minimum over three passes. The control is the bench's own
PyTorch column, which reads none of the scorch environment. It spread 1.004–1.053x here
against the 3.6x that made attempt 2 unreadable.

**No arm regresses on any of the five graphs.** pubmed goes 0.727 → 0.594 with the
partition on, which agrees with the earlier finding that pubmed's problem was the thread
reshape rather than the kernel; the analyzer declines to call it because its rule asks for
1.5x the control spread, which is too crude a test when the control is this tight — a
1.226x arm spread against a 1.022x control is not noise. The fused path is flat everywhere
(spreads 1.003–1.074x), as expected: it has its own partition knob, defaulted off.

## The autoencoder guardrail passes its control, and still cannot resolve three percent

At three rotations the controls read 0.9348 and 0.9554 (PyTorch Dense), 0.9936 and 0.9452
(PyTorch Sparse) — inside the 15% tolerance, so the comparison is admitted, but the effect
being asked about is 3% (Scorch 0.9684 for back-stealing, 0.9948 for back-stealing plus the
fused partition, worst cell 0.8625). **The effect is smaller than the control's own
movement, so the honest statement is only that nothing regresses by more than about 14%.**
Each AE arm is roughly four seconds of work, so the variance is between processes, not
within them; the fix is more rotations, not more repeats, and it is re-queued at eight.

## Which earlier numbers the second environment charge moves, and in which direction

`--pad-env` equalises how many names each arm puts in the environment. It does not
equalise how many of those names the code looks up: an arm that sets a variable
`scorch_policy.h` reads pays a successful `getenv` plus an `atol` on every call, and an arm
that leaves it unset pays a miss and no conversion. Over 2485 cells where the row ceiling
provably cannot fire, the four ceiling arms read 0.9863–0.9962 ordered by exactly that
count (one read-variable → 0.9962 / 0.9934, two → 0.9959 / 0.9871, three → 0.9950 /
0.9863). So the charge is roughly 0.3–0.6% per variable an arm sets *and* the code reads,
on top of the ~1.1% per variable it merely sets.

Everything this moves, it moves in the same direction — **the arm with more knobs was
undercharged for its effect, so every padded number here understates the candidate**:

- the exact-width kernel's padded x86 grid (1.0257 / 1.0199, harmed 0.2% / 0.1%) sets one
  read variable more than its reference, so the true effect is nearer 1.029 / 1.023;
- the ARM replication (1.0157 / 1.0143) likewise, at the ARM rate;
- the ceiling's in-gate readings (1.11 / 1.19) are understated by the same amount, which is
  immaterial at that size.

What it does **not** let us conclude is inertness. "Reads 0.996 where it cannot fire" is
indistinguishable from "costs 0.4% because it names a knob", and no arm ordering fixes
that. Both remaining questions of that kind — is the ceiling inert outside its gate, is the
exact-width kernel inert at widths it does not serve — are settled by compiled-in
three-builds (`rw_stage16.sh` for the ceiling, the k-band table of `rw_stage6.sh` for the
kernel), where no `getenv` runs at all.

## Chunk-aligned partition boundaries: a null on four corpus/dtype combinations, and removed

The partition splits the row range on nonzeros and `scorch_spmm_chunk` picks a chunk width
for the output store, so a worker's range generally starts mid-chunk: every range ends in a
partial chunk and there are `nsplit` of those where the shared counter has exactly one. On
the shape where the partition is worst — a 64-row rn50 bottleneck at chunk 4 — six workers
each end in a two- or three-row claim. Snapping each interior boundary to the nearest chunk
multiple removes that by construction. It also moves nonzeros between workers, and the
nonzero-balanced split is what put them there.

Measured against the candidate, with the same-code A/A control alongside:

| corpus | dtype | + alignment | A/A floor | >10% slower than today |
|---|---|---|---|---|
| general (1810 cells) | float32 | 0.9960 | 1.0002 | 19 vs 14 |
| general (1810 cells) | float64 | 0.9925 | 0.9996 | 19 vs 11 |
| large-A (255 cells) | float32 | 1.0004 | 0.9999 | 1 vs 2 |
| large-A (164 cells) | float64 | 0.9973 | 0.9972 | 2 vs 2 |

Null every time, and on the general corpus the harmed tail rises. Part of the ~0.4–0.7%
deficit is the arm being charged for naming one more knob the code reads, so the honest
reading is "nothing, with a slightly worse tail" rather than "a small loss". **Removed**
— the code, the default, and the hook — with the measurement recorded at the site where
the boundaries are built. The same disposition the non-temporal store lever got.

## What the cold penalty is made of, and the one class where it is five times the fit

The cold-minus-warm **kernel** penalty over the 372-cell cold grid, banded by the bytes
the call must touch (A values and indices, B, C), float32:

| A+B+C | n | penalty, median | per MB | warm kernel | MKL's own cold penalty (whole call) |
|---|---|---|---|---|---|
| 0.12–0.25 MB | 32 | 36.9 µs | 158.6 | 13.0 µs | 80.9 µs |
| 0.5–1 MB | 77 | 43.3 | 58.3 | 16.4 | 95.1 |
| 2–4 MB | 59 | 78.0 | 27.4 | 28.5 | 135.0 |
| 8–16 MB | 26 | 170.7 | 17.4 | 74.0 | 230.6 |
| 16–32 MB | 24 | 284.6 | 14.8 | 96.8 | 390.2 |

The fit is **35 µs + 14 µs/MB**. 14 µs/MB is about 71 GB/s — compulsory DRAM traffic,
which MKL pays too, and its own cold penalty is larger in every band (that column is a
whole call against our kernel, so it is a scale check, not a comparison). This is why the
cold *ratio* against MKL is 1.20x where warm is 1.65x rather than something being wrong
with the cold path: both runtimes pay a fixed ~35–80 µs, and on a 13 µs kernel that term
is the measurement.

**The four worst cold cells do not fit.** rn50 `bottleneck_2` at k=64: 256 rows, 2304
columns, 176947 nonzeros — 2 MB of operands, so the fit predicts a 63 µs penalty, and the
measured kernel is 342–369 µs cold against 51 µs warm. The shared row counter is the same
(315–387 µs), so it is not the partition. The candidate mechanism is **per-thread
replication of B**: at k=64 with 2304 columns B is 590 KB, which fits in one core's 2 MB
L2, so warm every worker holds its own resident copy and cold every worker must fetch one.
24 copies of 590 KB is 14 MB, and at 56 GB/s that is 250 µs — the missing term. MKL, which
wakes fewer workers for a 256-row product, fetches fewer copies.

If that is right, cold kernel time on these cells falls roughly linearly with a forced
thread count while warm time rises, and cells whose B is too large for any worker to hold
show no such fall. That is a prediction, and `cold_threads.py` (arms
`SCORCH_SPMM_NT_FORCE` = 1/4/8/16/24/policy, tail group and a control group from the middle
of the same distribution) is queued to test it. Note what it cannot become: capping workers
would trade warm time for cold time on the same shape, and warm is the claim.

## Counted per matrix instead of per cell, the row-bind form is a null — and the correction matters

The regions the ceiling's two row conditions differ in are small, and small regions are
where treating cells as independent goes wrong: a corpus contributes each matrix at five
widths, and the five readings of one matrix move together. Aggregating each matrix to one
number first, then testing across matrices, float64:

| region | cells | **matrices** | count form | + pool cap | row-bind form | + pool cap | floor |
|---|---|---|---|---|---|---|---|
| both forms fire (rows ≤ 128, deg ≥ 192) | 60 | **7** | 1.1549 (z=4.53) | 1.1483 (z=3.42) | 1.1622 (z=4.03) | 1.1412 (z=3.16) | 0.9991 |
| only the row-bind form (129–383 rows) | 130 | **9** | 1.0027 (z=0.16) | 0.9853 (z=−1.72) | 1.0053 (z=0.15) | 1.0370 (z=1.27) | 1.0015 |

The gated region survives the correction and gets *larger* (1.14–1.16 against the
cell-level 1.19). **The row-bind extension does not survive it.** Its pool-capped variant
read 1.0438 with z = 3.15 at cell level and reads 1.0370 with z = 1.27 across the nine
matrices that produced those 130 cells — the significance was five widths of nine matrices
counted as 130 independent observations.

So the row condition stays a row count, `SCORCH_SPMM_CEIL_ROWBIND` is removed, and the
measurement stays in the comment at the condition. The motivation for it was sound — 128
rows is a constant where the mechanism is a comparison against the available width, and
the 256-row pruned-ResNet layers that dominate the warm caller-path residual sit outside
128 and inside 384 — but the widening does not pay on those cells, so the residual is not
a thread-count problem. What is still open on the ceiling is only whether the *count* form
is inert outside its gate, which needs the compiled-in three-build.

### and with the separability test fixed, two of the five graphs are wins, not just non-regressions

The GCN analyzer declared an arm separable only when its spread exceeded 1.5x the
same-code control's spread. That is the wrong shape of test: it scales the *threshold* with
the control instead of comparing *excesses*, so a 1.226x arm spread against a 1.004x
control was labelled "inside the control". Replaced by "the effect must exceed three times
the control's own excess over one, with a 2% floor" — so a quiet control cannot make half a
percent a result, and a tight control is allowed to resolve a large effect. Re-scored:

| dataset | arm spread | control | verdict |
|---|---|---|---|
| pubmed | 1.226x | 1.022x | **separable** — 0.727 → 0.594 ms with the partition |
| citeseer | 1.044x | 1.004x | **separable** — 0.542 → 0.523 ms |
| cora | 1.125x | 1.053x | inside the control |
| ogbn-arxiv | 1.016x | 1.015x | inside the control |
| reddit | 1.007x | 1.008x | inside the control |

So on real GCN inference the candidate is a win on two of five graphs and indistinguishable
on the other three, with nothing regressed.

## The ARM shipping number, sliced: 1.079 and 1.059 against floors of 1.017 and 1.004

Same comparison as the x86 one — three hookless builds of one tree differing only in -D
flags, `cand` carrying back-stealing plus the exact-width kernel at unroll 4 — but run in
25-matrix slices with the three builds rotated inside each slice, and with the two slices a
machine stall damaged re-measured afterwards. 1925 cells per dtype.

| | kernel | whole call | floor (ctrl/ship) | harmed | floor harmed | reference spread |
|---|---|---|---|---|---|---|
| float32 | **1.0791** | 1.0728 | 1.0171 | 5.0% | 1.9% | 1.0626 |
| float64 | **1.0593** | 1.0457 | 1.0043 | 4.3% | 0.6% | 1.0215 |

Read the other way — against `ctrl`, the second build of the baseline — 1.0609 and 1.0548.
Net of the floor, **the candidate is 5.5–6% above what ships today on ARM**.

By free dimension, float32 against a floor of ~1.018: k=1 1.0456, **k=2 1.1036, k=3
1.1299**, k=4 1.0510, k=8 1.0609, k=64 1.0866, k=256 1.0785. The k=2 and k=3 peaks are the
exact-width kernel, which serves exactly those two widths; the k≥64 gains are the
partition. float64 has the same shape (k=2 1.0848, k=3 1.0880).

**The tail is real and it is not the floor.** 90 of 1750 float32 cells and 83 of 1925
float64 cells are more than 10% behind, against floors of 1.9% and 0.6%, and only 4 of
each have the floor below 0.9 as well. Every one is in the 0–1 MB output band, and they
concentrate at k=1–4 — 24 of the 90 at k=2 and 15 at k=3, which is where the exact-width
kernel fires, so *which* part of the candidate owns the tail is now a question with a
suspect. Three shape families:

| family | example | cand/ship |
|---|---|---|
| very low degree | as-735, 7716 rows, degree 1.6, k=2 | 0.695–0.793 |
| mid-row transformer layers | 512 rows, degree 172, k=3 | 0.702 |
| few-row pruned ResNet | 64 rows, degree 288, k=4 | 0.702–0.767 |

`m5_stage18.sh` splits the candidate on that corpus — the partition without the kernel, the
kernel with its short-row unroll clamp, the kernel at unroll 1 — because a degree-1.6 graph
is nothing but short rows and the shipped configuration does not clamp the unroll by row
length.

## The x86 shipping number for the full candidate: kernel 1.33 and 1.29, below-MKL 803 → 629 and 688 → 449

The first three-build ran before the exact-width grid existed, so its chooser had no data
and fell back to the partition alone. Re-run with the chooser reading the finished grid —
it picked `p3e4`, unroll 4, at 1.0198 of its floor — the candidate is back-stealing plus
the exact-width kernel at k ∈ {2,3}. 2172 cells per dtype, two passes per build in
opposite orders.

| | kernel | whole call | floor (ctrl/ship) | harmed | floor harmed | vs MKL, before → after | below MKL |
|---|---|---|---|---|---|---|---|
| float32 | **1.3305** | 1.2559 | 1.0128 | **2.5%** | 3.8% | 1.1223 → **1.4011** | **803 → 629** |
| float64 | **1.2851** | 1.2347 | 0.9877 | **3.6%** | 4.4% | 1.2713 → **1.5545** | **688 → 449** |

**The harmed tail is at or below the floor on both dtypes** — 2.5% against 3.8%, 3.6%
against 4.4% — which is the strongest form the "no regressions" requirement takes: two
builds of identical source disagree on more cells than the change does. Reference spread
1.0755 and 1.0796, inside the limit.

By free dimension, float32: k=1 1.3298, k=2 1.3842, k=4 1.3252, k=8 1.3810, k=64 1.3895,
k=256 1.1850, against floors of 1.004–1.030. float64 the same shape (1.2953 / 1.3477 /
1.3137 / 1.3648 / 1.2745 / 1.1294). Uniform, not a single-width effect, and k=256 is the
only band under 1.2 — the width where the register-block kernel already streams well.

Against the partition-only run the float32 candidate is a touch better (1.3305 vs 1.3250)
and float64 a touch worse (1.2851 vs 1.3106), but the floor moved 3% between the two runs
(0.9877 here against 1.0186 there), so the exact-width kernel's ~2% is inside the
run-to-run movement of the baseline. Net of each run's own floor both say the same thing:
**about 1.30x the kernel that ships today, on x86, in both dtypes**.

One caution on the "vs MKL" column: it is the harness path, which carries the plan-cache
handicap. The caller-path answer is the cold/warm grid — 1.20x cold, 1.65x warm.

## The ARM tail is the partition, it is not the thread count, and it stops at k=8

A corpus of exactly the matrices the ARM three-build put more than 10% behind — 70 of them,
grouped by what makes them odd: 32 very low degree (as-735, degree 1.6–1.9), 24 mid-row
(512-row transformer layers at degree ~175), 14 few-row (64-row pruned ResNet at degree
288). 420 cells per dtype, 8 arms, equal environment-variable counts, A/A floor 0.9964.
Ratios are against the candidate, so above 1 means *the arm is faster than what we propose
to ship*:

| arm | float32 | float64 | lowdeg | midrow | fewrow |
|---|---|---|---|---|---|
| candidate (partition + kernel) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| partition, no kernel | 0.9878 | 0.9900 | 0.9966 | 0.9854 | 0.9719 |
| kernel at unroll 1 | 0.9855 | 0.9872 | 0.9925 | 0.9850 | 0.9705 |
| + per-row short clamp | 0.9970 | 0.9928 | 0.9970 | 0.9975 | 0.9962 |
| + per-call degree unroll | 0.9987 | 0.9989 | 1.0010 | 0.9953 | 0.9992 |
| **no partition (ships today)** | **1.0294** | **1.0206** | **1.0557** | 1.0050 | 1.0125 |
| forced 6 threads | 0.8265 | 0.8326 | 0.8868 | 0.9774 | **0.5279** |
| forced 2 threads | 0.9264 | 0.8867 | 0.9761 | 0.8987 | 0.8661 |
| A/A floor | 0.9964 | 0.9965 | 0.9942 | 0.9959 | 1.0023 |

**It is the partition.** Removing the kernel makes these cells *worse* (0.9878); removing
the partition makes them better (1.0294, per-matrix z of +5.7 at k=1, +6.7 at k=4, +4.6 at
k=8 — and these are 70 distinct matrices, so cells and matrices are the same count at each
width).

**It is not the thread count.** The mechanism I expected was heterogeneity: this host
launches twice its six-thread pool so the twelve efficiency cores join, and a home range
handed to an E-core running at a third of a P-core's speed has nothing left to steal at
chunk granularity. Forcing the count *down* is far worse, not better — 0.8265 at six
threads and 0.5279 on the few-row group. These shapes want every worker they can get; the
static decomposition is the problem, not the width.

**It stops at k=8.** At k=64 the partition wins even on this corpus (0.9752, z = −2.7). So
the regressed region is low degree *and* narrow, which is where the kernel is 15–30 µs and
a fixed per-call cost is the measurement.

The per-call degree unroll is neutral here (0.9987, inside the 0.9964 floor) with a small
real gain at the widths it serves (k=2 1.0100, per-matrix z = +2.6). It is not the fix for
this tail, and it needs an x86 reading before it could ship.

## No home-range scheme wins on those shapes: it is the decomposition, and the reason is worker count

Same 70-matrix corpus, four arms, A/A floor 0.9951 (float32) and 1.0006 (float64). Ratios
against the candidate, so above 1 means the arm beats what we propose to ship:

| arm | float32 | float64 | k=1 | k=2 | k=4 | k=8 | k=64 |
|---|---|---|---|---|---|---|---|
| back-stealing (candidate) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| home ranges, **no stealing** | 0.9826 | 0.9622 | 1.0138 | 1.0239 | 0.9782 | 0.9504 | 0.9542 |
| front-stealing | 0.9682 | 0.9592 | 0.9560 | 0.9795 | 0.9604 | 0.9699 | 0.9823 |
| **no partition** | **1.0303** | **1.0222** | 1.0581 | 1.0372 | 1.0719 | 1.0187 | 0.9678 |

The probe hypothesis is dead: mode 1 does no cursor probing at all and it is *worse* than
back-stealing, not better (0.9826 / 0.9622). Front-stealing is worse again. **Every
home-range scheme loses on these shapes and the shared counter wins** — so the cost is the
static decomposition, and there is nothing inside the partition's parameter space to tune.

Which makes the mechanism legible, because on x86 the same comparison goes the other way by
a mile (p0/p3 = 0.785 on the general corpus, 65% of cells more than 10% behind). The
partition buys inter-call L2 residency for A and pays a fixed setup plus a claim per chunk;
the shared counter buys perfect load balance and pays contention on one atomic line. **A is
100 KB on as-735 — it was never leaving L2, so there is no residency to win** — and the
counter's contention scales with the number of workers: 24–32 of them on redwood make it
the binding cost, 12 on the M5 do not. Same code, opposite sign, and the quantity that
flips it is the worker count, which is why this is arm-variance rather than a bug.

That is also why a work gate is not simply "the ARM fix": on x86 a min-work gate costs 4–9%
because the partition wins there even on tiny products. The condition has to be the
mechanism — apply the gate only where the pool is small enough that the counter is not the
bottleneck — so that it *provably cannot fire* on a 24-thread host.

## The row ceiling, compiled in on both hosts: x86 wants it, ARM does not, so it stays off

The hooked grid could not answer whether the rule is inert where it cannot fire, because an
arm pays a `getenv` hit and an `atol` per call for every variable it sets that the code
reads. Compiled in — `cand` = the candidate plus the rule, `ctrl` = a second build of the
candidate — there is no `getenv` at all. Both hosts, both dtypes, per-cell and per-matrix
(the gated region is 35 cells from **7** matrices, so the second number is the one to read):

| host / dtype | region | n / matrices | cand / ship | z per cell | **z per matrix** | floor | harmed vs floor |
|---|---|---|---|---|---|---|---|
| x86 float32 | in gate | 35 / 7 | **1.3066** | 3.47 | 1.61 | 1.0533 | 11.4% vs 31.4% |
| x86 float64 | in gate | 35 / 7 | **1.4011** | 3.58 | **2.95** | 1.0665 | 8.6% vs 22.9% |
| x86 float32 | outside | 1730 / 346 | 1.0047 | 1.13 | 0.75 | 1.0016 | 2.9% vs 3.8% |
| x86 float64 | outside | 1730 / 346 | 0.9777 | −1.08 | −0.64 | 0.9807 | 11.6% vs 6.5% |
| **ARM float32** | in gate | 35 / 7 | **0.9887** | −0.69 | −0.58 | 0.9971 | 5.7% vs 0.0% |
| **ARM float64** | in gate | 35 / 7 | **0.9753** | 1.31 | 0.49 | 0.9465 | 14.3% vs 14.3% |
| ARM float32 | outside | 1300 / 260 | 0.9981 | −2.38 | −1.70 | 1.0001 | 0.5% vs 0.2% |
| ARM float64 | outside | 1300 / 260 | 0.9793 | −5.43 | **−2.15** | 0.9990 | 4.7% vs 4.5% |

Two things settled. **Outside its gate the rule is inert on x86** — 1.0047 against a 1.0016
floor on float32, and 0.9777 against a 0.9807 floor on float64, i.e. 0.997 of the floor,
neither significant. The hooked grid's 0.9863–0.9962 with z from −6 to −19 was the
instrument, exactly as predicted. **And the rule does not carry to ARM**: no gain in its own
gate (0.9887 and 0.9753) and a ~2% loss outside it on float64 that survives a compiled-in
build (per-matrix z = −2.15 over 260 matrices).

**So the ceiling stays off.** It is the third thread-count lever in this branch whose sign
is set by the pool size, and all three point the same way: the rule widens a few-row
product's worker count, and on x86 that means 4 → 22 workers while on ARM, capped at the
caller's pool, it means 4 → 6. There are only six to have. The gain is not a property of the
shape, it is a property of how much machine the shape was failing to use, so a shared
default cannot express it — and a 7-matrix, one-host, one-dtype effect is not enough to
justify an ISA-conditional one.

## A work gate recovers half the ARM tail, and its own consistency check caught the wrong condition

The gate turns the partition off below N grains of `nnz·max(k,16)`. On the 70-matrix ARM
tail corpus, against the candidate (A/A floor 1.0008 / 0.9987):

| arm | float32 | float64 | >10% behind today, float32 | float64 |
|---|---|---|---|---|
| candidate | 1.0000 | 1.0000 | 81 | 55 |
| **+ gate at 2 grains** | **1.0154** | **1.0089** | **45** | **22** |
| + gate at 2, ceiling 16 | 1.0087 | 1.0031 | 46 | 31 |
| + gate at 4, ceiling 16 | 1.0097 | 1.0048 | 48 | 23 |
| no partition (ships today) | 1.0270 | 1.0150 | 0 | 0 |

The gate recovers about 57% of the gap on float32 and 59% on float64, and roughly halves the
cells behind today. It leaves k=64 alone, where the partition wins even here (0.9972 against
the candidate, and no-partition reads 0.9734).

**The consistency check failed, and that is what it was for.** The ceiling is meant to stop
the gate firing on a large pool; on a six-thread host it cannot bind, so the gated and
gated-with-ceiling arms had to read *identically*. They differ by 0.7% (1.0154 against
1.0087) — because the first version of the condition compared the **resolved worker count**,
which is raised per shape and capped by `omp_get_num_procs()` = 18 on this host, so it
straddles a ceiling of 16 and the ceiling blocked the gate on exactly the shapes whose count
had been raised. Written down as a prediction before the run, it took one table to see.

The condition now reads the **caller's pool** — 6 here, 24 on redwood, shape-independent,
and the quantity the contention argument is actually about. Both hosts re-run against it,
and the same check applies again: on ARM the two arms must now agree, on x86 they must
differ. If ARM still shows them apart, the pool is not reaching this call site and the
condition has to come from `at::get_num_threads()` instead.

## The transformer guardrail: neutral on the path that ships, and its one bad cell is noise

Sparse-attention transformer inference, three configs × two sequence lengths, three
rotations, per-cell minimum over passes. The bench's own Dense and Sparse PyTorch columns
are the same code in every arm and moved 1.0034 / 0.9872 and 0.9833 / 0.9837 — so the arms
are comparable to about 1.7%.

| framework | + back-stealing | + back-stealing and the ceiling | worst cell |
|---|---|---|---|
| **Scorch (fused)** — the path that ships | **0.9924** | 0.9991 | 0.9617 |
| Scorch (unfused) | 0.9701 | 0.9772 | 0.8454 |

The fused path — the single fused sparse-attention kernel — is neutral. The unfused chain
(per-head SDDMM → CSR softmax → SpMM) shows one cell at 0.8454: base-w128 at sequence 1024,
164.8 → 194.9 ms.

**That cell cannot be attributed, and the run says so itself.** The row ceiling requires
rows ≤ 128 and every transformer config here has rows = sequence length ≥ 1024, so the
ceiling *cannot fire* — which makes the third arm a second copy of the second one for this
workload, an A/A pair by accident. On that cell they read 194.913 and 172.238: **13% apart on
behaviourally identical code.** Pooled they agree to 0.7%, so the pooled numbers stand and
the per-cell floor on the unfused path at this shape is 13%, not 1.7%. The Dense and Sparse
PyTorch controls are tight only because their kernels are ten to forty times longer.

So: no regression on the transformer at the pooled level, nothing separable per cell on the
unfused path, and the fused path — which is what a user gets — is flat.

## The work gate on the ARM tail, with the ceiling now read from the caller's pool

Re-run after moving the ceiling's condition from the *resolved* worker count to the caller's
pool. The resolved count is raised per shape and capped by `omp_get_num_procs()` = 18 on the
M5, so a ceiling of 16 straddled it and the gate fired on some shapes and not others; the
pool is 6 there, so a ceiling of 16 provably cannot bind and the capped arm must read the
same as the uncapped one. That pair is the run's own consistency check.

Hooked, `--pad-env`, 6 k values, ARM. Two corpora: the tail (the 8 matrices where the
partition loses) and the general 275-cell corpus where the gate must be inert.

| arm | tail f32 | tail f64 | general f32 | general f64 |
|---|---|---|---|---|
| back-stealing + exact width (reference) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| + work gate at 2 grains | **1.0135** | **1.0111** | 1.0005 | 1.0000 |
| + work gate at 2 grains, ceiling 16 | 1.0098 | 1.0067 | 0.9956 | 0.9948 |
| + work gate at 4 grains, ceiling 16 | 1.0128 | 1.0091 | 0.9936 | 0.9927 |
| the shared counter | 1.0347 | 1.0236 | 0.9740 | 0.9743 |
| A/A | 1.0010 | 0.9987 | 0.9988 | 0.9986 |

Tail cells more than 10% behind the shared counter: **96 → 55** (f32) and **76 → 38** (f64).

**The consistency check passes.** The capped arm reads 0.37–0.52% below the uncapped one in
all four columns, and split by the reference's own duration on the 1650-cell corpus that
deficit is 1.21% / 1.23% below 10 µs, 0.55% / 0.66% at 10–30 µs, 0.21% / 0.31% at 30–100 µs
and 0.08% / −0.49% above 100 µs — monotone to zero with kernel length, z = 26.7 at the short
end. A fixed per-call cost, which is what one more variable *the code looks up* buys
(`--pad-env` equalises names, not lookups). A gate that bound would not order itself by
duration. The earlier version's 0.7% did not have this shape.

Four grains buys nothing on the tail (1.0128 against 1.0098 at the same variable count) and
costs 0.2% more on the general corpus, so **two grains is the setting**.

The gate recovers about a third of the ARM tail (the shared counter is 3.47% / 2.36% ahead
there) and halves the count of cells behind it, while staying inside the A/A floor
everywhere else. Whether it ships depends on x86 inertness, which is a separate measurement:
redwood's pool is 24, above the ceiling, so the gate's whole block should be unreachable
there.

## The sparse-autoencoder guardrail at eight rotations, and the limit of what it can resolve

Eight rotations of three arms over two datasets × three sparsities, per-key minimum. Arms:
what ships today, back-stealing, and back-stealing extended to the fused Linear path (a
separate knob, default off).

| framework | back-stealing | + on the fused path | worst cell |
|---|---|---|---|
| **Scorch** | **1.0210** | 1.0140 | 0.9872 / 0.9362 |
| PyG (not our code) | 1.1021 | 1.1288 | 0.9684 / 0.9498 |
| PyTorch Dense (control) | 1.0005 | 0.9741 | — |
| PyTorch Sparse (control) | 0.9488 | 0.9701 | — |

**The per-cell numbers say this bench cannot resolve a 2% effect, and the aggregate hid it.**
The control gate passed because it scores geomeans, but per cell the columns that *cannot be
affected by the change* move like this: PyG mnist at 0.9 sparsity reads 1.5481 and 1.4036,
PyG fashion at 0.8 reads 1.5065, PyTorch Sparse fashion at 0.9 reads 1.1685, PyTorch Sparse
mnist at 0.8 reads 1.1187, PyTorch Dense mnist at 0.9 reads 0.9376. A 55% swing on a
third-party framework is the bench's resolution at these sizes — every measured forward here
is 0.5–4 ms.

Scorch's own column stays in 0.9872–1.0779 (back-stealing) and 0.9362–1.0649 (fused path).
So the honest statement is: **the autoencoder shows no regression, and this harness cannot
support a claim finer than about 6% per cell.** Same conclusion as the transformer, where two
behaviourally identical arms differed 13% on one cell. Whole-model benches on millisecond
forwards are guardrails against breakage, not instruments for single-digit percentages — the
kernel grids are the instruments.

One thing the run does say: extending the partition to the fused Linear path is not positive
(1.0140 against 1.0210, worst cell 0.9362), so that knob **stays off**, which is its default.

## The cold penalty is not per-thread replication: cold wants MORE workers than warm

The cold grid's tail cells sit 5× above the 35 µs + 14 µs/MB fit, and the mechanism written
down before this ran was per-thread replication of B — at k = 64 over 2304 columns B is
590 KB, one core's L2, so 24 copies is 14 MB, about 250 µs at redwood's 56 GB/s. The
prediction was that **reducing** the forced thread count would cut cold time while raising
warm time, and that the mid group (B too large for any worker to hold) would not show it.

Forced thread ladder, 23 matrices × k ∈ {1, 8, 64}, 256 MB cache flush before each cold
call, median of 21, MKL timed in the same interleaved slot. Kernel columns, against the
policy count each shape resolves today:

| | mid cold | mid warm | tail cold | tail warm |
|---|---|---|---|---|
| 1 thread | 0.5922 / 0.5161 | 0.5126 / 0.4076 | 0.5700 / 0.4321 | 0.3396 / 0.2428 |
| 4 | 0.7915 / 0.7333 | 0.7877 / 0.7042 | 0.8400 / 0.7481 | 0.7680 / 0.6512 |
| 8 | 0.9233 / 0.8922 | 0.9034 / 0.8721 | 1.0041 / 0.9300 | 0.9693 / 0.9314 |
| 16 | 1.0157 / 0.9941 | 1.0029 / 0.9984 | 1.1588 / 1.0783 | 0.9710 / 0.9555 |
| **24** | 1.0213 / 1.0207 | 1.0142 / 1.0212 | **1.1804 / 1.1979** | **0.8759 / 0.9163** |

(f32 / f64. Above 1.0 is faster than the policy count.)

**The prediction is refuted.** Fewer workers makes cold *worse*, monotonically, in both
groups and both dtypes — 1 thread costs 1.7× (mid) to 2.3× (tail) cold. Cold time falls all
the way to 24 workers. So the cold penalty is compulsory DRAM traffic, and more workers buy
memory-level parallelism to cover it; that is consistent with the fit's 71 GB/s and with MKL
paying it too. Replication would have shown the opposite sign.

**What it does find is a lever, and it is one I had already rejected.** On the tail group —
few rows, high degree, exactly where `rows/SCORCH_ROWS_PER_THREAD` holds the team narrow —
24 workers is **1.18× (f32) / 1.20× (f64) faster cold** and **0.88× / 0.92× slower warm**.
The mid group is flat both ways (1.02 either direction). So the policy count is right warm
and 18–20% short cold, on one identifiable family of shapes.

That is the nonzero-expressed row ceiling, which I rejected as a shared default because warm
it cost 0.7% on x86 and 1–2% on ARM. Cold it is worth 18–20% on the family it was built for.
The two readings are not in conflict — they are different calls. **The ceiling is a cold-only
mechanism:** raise the team on a plan-cache miss, where the tensor was just built and A is
not resident, and leave every repeat call exactly as it is. Not implemented; the measurement
is recorded so the option is costed. Warm is the claim, so this stays behind the settled work.

## What is actually left below MKL, and why most of it is the measurement path

Offline, from the shipping three-build's own cells (2172 per dtype, candidate build, best of
two passes). Comparing our whole call against MKL's whole call:

| | float32 | float64 |
|---|---|---|
| cells below MKL | 615 / 2172 | 423 / 2172 |
| their median ratio to MKL | 0.795 | 0.829 |
| **share of total corpus time at stake** | **2.8%** | **0.7%** |
| below-MKL cells under 30 µs | 564 of 615 | 361 of 423 |
| below-MKL cells above 100 µs | 4 | 11 |
| below-MKL cells at 100k+ rows | 0 / 12 | 0 / 12 |

**87% (f32) / 82% (f64) of them have a kernel that already beats MKL's whole call.** They
lose on a per-call cost outside the kernel, and that cost is flat: q1 6.9, median 7.0, q3
7.1 µs on f32 (6.4 / 6.5 / 6.7 on f64), against a median gap to MKL of 4.5 / 3.7 µs. It is
1.6–1.8× the gap. It is also present, at the same size, on the 1557 cells we *win*
(median 6.9 µs) — so it is not a property of the losing shapes, it is a constant.

**And it is a constant this corpus pays for asking.** These cells are timed through
`scorch.matmul(..., time_dict=td)`, and passing `time_dict` is a keyword, which disables the
per-tensor plan cache in `ops.matmul`. The separately measured caller path — the one a user
gets, no keywords — has a non-kernel cost of **−0.4 µs warm**, and on it 37 of 372 f32 cells
and 31 f64 cells are below MKL, about 10%, against 28% here. So most of the 615 is the
instrument, and the honest count of the remaining deficit is the caller-path one.

**The residue that is genuinely the kernel is 82 cells (f32) / 77 (f64), and it is two
families.**

1. **Few rows, enormous degree — and the mechanism for it is already built.** kl02 (71 rows,
   degree 2993, k=2): 32 µs against MKL's 25. nw14 (73 rows, degree 12396, k=2): 100 against
   62. bibd_17_8 (136 rows, degree 5005, k=4): 63 against 41. rn50 magnitude-pruned (256 rows,
   degree 1152, k=1): 29 against 22.

   My first reading of this was that row-parallel decomposition cannot reach these and that
   within-row parallelism — the CSR-Vector / segmented-reduction shape — was the missing
   mechanism. **That is wrong, and the row counts say so directly:** the smallest is 64
   against a 24-thread pool, so every one of these has more rows than workers and a row split
   reaches full width on all of them. What holds the team narrow is not the decomposition, it
   is `rows/SCORCH_ROWS_PER_THREAD`, which gives a 71-row product four workers.

   That is the nonzero-expressed row ceiling, and compiled in it reads **1.3066 (f32) /
   1.4011 (f64)** inside its own gate on x86 with the harmed tail *below* the A/A floor. It
   is off today because it does not carry to ARM — no gain in gate (0.9887 / 0.9753) and
   about 2% off outside it on float64, per-matrix z = −2.15. But the doc's own account of why
   is that the sign is set by **how much machine the shape was failing to use**: on x86 the
   rule takes a few-row product from 4 workers to 22, on ARM from 4 to 6, and there are only
   six to have. A rule conditioned on the caller's pool expresses exactly that, is not
   ISA-conditional, and on a 6-thread host becomes unreachable code — which removes the ARM
   cost by construction rather than by tuning. It is the mirror image of the work gate above,
   which fires only *below* a pool ceiling. Both fail closed to what ships today.

   The evidence for the x86 side is 7 matrices (per-matrix z 1.61 on f32, 2.95 on f64), which
   is thin — and enumerating the corpus says it supplies **no more in-gate matrices at all**:
   the gate is rows ≤ 128, and the same 7 are the only ones that qualify. The extra
   family-(i) cells are 256-row DLMC blocks (degree 555–1152) and bibd_17_8 (136 rows), 8
   matrices that sit *just outside* the gate — at 256 rows the cap gives 16 workers of 24,
   under-threaded but less so.

   Which means fixing this family needs the row limit stated as the mechanism rather than as
   128: the row proxy is below the width available, `rows/SCORCH_ROWS_PER_THREAD < pool`,
   i.e. rows < 384 on a 24-thread host. That form was built and measured and rejected as
   paying nothing — but the rejection was 9 matrices, and its pool-capped variant read
   **1.0370** there with a per-matrix z of 1.27. That is a positive point estimate at low
   power, not a refutation, and the way to settle it is more matrices in the band. DLMC has
   thousands of layers of exactly this shape; this corpus samples 8 of them.
2. **Degree 1 with many rows.** Pd_rhs and Pd_b (8081 rows, 6323 nonzeros, k = 1–2): 27 µs
   against MKL's 17. I first wrote this up as fixed per-row cost, and `scorch_policy.h`'s own
   comment on the grain says something more specific: at k = 1 this shape's floored work is
   101168 against a grain of 150000, so **it runs single-threaded**. So the deficit is 8081
   iterations of the row loop at about 3.3 ns each against MKL's 2.1, and it is either the
   grain being too conservative here or the row loop being too fat — forcing the thread count
   separates the two, and neither has been measured on this matrix. The grain is not free to
   lower: the same comment records that raising thread counts off this measure by another
   route took the 20–50 µs cells to 0.920.

Neither is a tuning question and neither is addressed by the change in flight. They are the
next campaign, and they are now specified: a corpus, a mechanism, and a number to beat.

### The same corpus by density, which is the cleanest statement of where we stand

Our kernel against MKL's whole call, geomean, float32, 2172 cells:

| density of A | cells | MKL / ours | cells below 1.0 |
|---|---|---|---|
| < 0.1% | 1050 | **2.794** | 6 |
| 0.1–1% | 330 | **2.126** | 5 |
| 1–5% | 258 | **2.600** | 0 |
| 5–20% | 210 | **1.679** | 15 |
| ≥ 20% | 324 | **1.525** | 56 |

We are 1.5–2.8× ahead of MKL in every density band, and 71 of the 82 cells where the kernel
genuinely loses sit at 5% density or above. Splitting those 82 by shape:

- **fewer rows than the team can use, high degree** — 28 cells, 11 matrices, all DLMC
  ResNet-50 pruned blocks (64–256 rows, degree 288–2300). Needs within-row parallelism.
- **near-dense blocks** — 47 cells, 37 of them at ≥20% density, mostly k = 4, kernel around
  19 µs, median ratio 0.955 and worst 0.775. A few percent behind in the regime where MKL's
  kernel starts to look like a blocked dense one. The smallest of the three gaps.
- **degree below 4** — 7 cells, 3 matrices (Pd_b, Pd_rhs, bips07_3078_iv). Fixed per-row cost
  charged once per nonzero.

So the goal reduces to three named mechanisms on 3.8% of cells, not to a broad deficit.

### The kernel-losing count, tested against each cell's own pass spread

The 82 / 77 figures above came from a best-of-two number per cell. Each build ran two passes,
so every cell carries two of our readings and two of MKL's, and the deficit can be required
to exceed the cell's own pass-to-pass spread:

| | float32 | float64 |
|---|---|---|
| kernel slower than MKL on the best pass | 86 | 82 |
| **and by more than that cell's pass spread** | **60** | **41** |
| their median pass spread | 2.3% | 1.7% |
| their median deficit | 13.7% | 7.8% |
| corpus-wide median pass spread | 3.1% | 3.6% |

So the real kernel deficit is **60 of 2172 cells (2.8%) on float32 and 41 (1.9%) on
float64** — the survivors are not marginal, their deficits are 5–6× their own noise. Split:
32 near-dense, 22 few-rows-high-degree, 6 degree-below-4 on float32; 24 / 15 / 2 on float64.
The near-dense family is the largest and the least understood; the few-row family has a
mechanism waiting for power; degree-below-4 is three matrices.

### Did the change in flight cause the near-dense residual? No — it improves that family

22 of the 32 solid near-dense deficits are at k = 4, so the obvious worry is that the thing
about to ship is what put them there. It is not.

(I first wrote that k = 4 is an exact width the new narrow-k kernel claims. **It is not** —
`SCORCH_NARROWK_EXACT_HI` is 3, so the exact-width kernel takes k ∈ {2, 3} on both dtypes and
k = 4 runs the register-block kernel. The ARM degree-floor grid is the direct evidence: the
floor gates only `exact_width`, and it moved k = 2 by 7.8% and k = 3 by 10.7% while leaving
k = 4 at 0.9925.) Over all 246 cells at ≥20% density, candidate against what ships
today, kernel time:

| | cand / ship | A/A floor | vs MKL: ship → cand |
|---|---|---|---|
| float32, all ≥20% density | **1.1081** | 0.9855 | 1.462 → 1.620 |
| float32, k = 4 only | 1.0699 | 0.9683 | 1.340 → 1.434 |
| float64, all ≥20% density | **1.1490** | 1.0027 | 1.407 → 1.617 |
| float64, k = 4 only | 1.2187 | 1.0473 | 1.297 → 1.580 |

The candidate is 7–22% faster on the near-dense family and takes it from 1.30–1.46× ahead of
MKL to 1.43–1.62× ahead. Three of 41 k=4 cells are more than 5% harmed on float32, none on
float64, against a 3.2% float32 floor at that width. So the near-dense residual is a
pre-existing minority inside a family we win pooled, not something this change introduced.

## The work gate's pool ceiling, measured on x86: the ceiling is what makes it shippable

redwood, pool 24, hooked with `--pad-env`. Two corpora: the 312-cell few-row corpus and the
full 2172-cell general one. Reference is the candidate (back-stealing + exact width); `p0` is
what ships today.

| arm | general f32 | general f64 | few-row f32 | few-row f64 |
|---|---|---|---|---|
| candidate (reference) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| + work gate, **no** ceiling | **0.9303** | **0.9153** | 0.9916 | 0.9915 |
| + work gate, ceiling at 16 | 0.9991 | 0.9858 | 0.9955 | 1.0049 |
| + gate at 4 grains, ceiling 16 | 0.9913 | 0.9972 | 0.9964 | 1.0043 |
| A/A | 0.9998 | 0.9999 | 1.0024 | 1.0062 |
| what ships today | 0.7551 | 0.7609 | 0.7940 | 0.8181 |

**The ungated gate costs 7.0% (f32) and 8.5% (f64) on the general corpus** — uniform across
every k, 0.90–0.94 at each — and it costs almost nothing on the few-row corpus, because what
it turns off is the partition on the many small products the general corpus carries and the
few-row one does not. That is the x86 cost the ceiling exists to remove, reproduced.

**With the ceiling the gate is inert, to the resolution this grid actually has.** At pool 24
against a ceiling of 16 the gate's whole block is unreachable, so the two capped arms are an
A/A pair by construction — and they read 0.9991 / 0.9913 on f32 and 0.9858 / 0.9972 on f64,
i.e. **they disagree with each other by 0.8–1.1%**. That is this grid's arm-to-arm floor, and
both capped arms sit inside it. Anything finer than about 1% is not decidable here; the 7–8.5%
of the ungated arm is nowhere near it.

Also the headline for the change itself, on the same 2172 cells: **the candidate is 1.3242
(f32) / 1.3142 (f64) faster than what ships today**, with 1161 / 1155 cells more than 10%
faster and 7 / 6 more than 10% slower.

**So the shipping configuration is decided:** back-stealing, the exact-width narrow-k kernel
at unroll 4, the work gate at two grains, and the gate's pool ceiling at 16.

## The scoreboard on the path a user actually gets

Everything above times the kernel, or times a call that passed `time_dict` and so bypassed
the per-tensor plan cache. The production path is `ops.matmul(A, B)` with no keywords, served
from the plan cache on every repeat call. Scored against MKL's whole call, 372 cells:

| | float32 | float64 |
|---|---|---|
| **warm, whole call vs MKL** | **1.6471** | **1.6731** |
| cells below MKL | **37 / 372** | **31 / 372** |
| worst cell | 0.638 | 0.723 |
| cold, whole call vs MKL | 1.2018 | 1.2560 |
| cells below MKL, cold | 81 / 372 | 53 / 372 |
| non-kernel cost, production path | **−2.4 µs** | **−2.0 µs** |
| non-kernel cost, `time_dict` path | +7.3 µs | +7.0 µs |

The kernel-only column reads 1.4978 / 1.5459 with 117 / 73 cells below MKL — *worse* than the
whole-call column, because comparing our kernel against MKL's whole call charges MKL nothing
for its own per-call cost. Whole call against whole call is both the fairer comparison and
the one a user experiences.

**And the remaining deficit is small in absolute terms.** Of the 37 below-MKL cells on float32,
32 are the kernel's own (5 are overhead-only), and their gap is a **median 0.7 µs on an 18.6 µs
call — 3.9%**. For the 22 of them under 20 µs the gap is 0.6 µs. Float64 is the same at 1.1 µs
on 19.9 µs, except for 7 cells above 40 µs where the gap is real: **12.3 µs, 11.1%**.

By family, the below-MKL kernel-fault cells are **29 near-dense and 3 few-row on float32, 25
and 6 on float64**. So on the production path the near-dense family is essentially the whole
remaining gap, the few-row family is three cells, and degree-below-4 does not appear at all.

That reorders the next campaign. It is not a missing kernel: it is a **sub-microsecond fixed
cost on ~19 µs calls** whose arithmetic is about 2 µs — output zero-fill, team fork and join,
policy resolution — plus a separate group of seven large float64 cells that is worth its own
look.

### Count says near-dense; magnitude says few-row

The two ways of ranking the production-path residual disagree, and the disagreement is the
useful part. Every cell above 40 µs that is the kernel's own fault:

| matrix | k | rows | degree | ours | MKL | ratio |
|---|---|---|---|---|---|---|
| kl02 | 8 | 71 | 2993 | 45.6 | 33.3 | **0.731** |
| nw14 | 1 | 73 | 12396 | 119.7 | 88.5 | **0.740** |
| rn50 bottleneck_2 (f32) | 1 | 256 | 1152 | 42.0 | 26.8 | **0.638** |
| transformer body_decoder | 1 | 2048 | 300 | 45.6 | 37.6 | 0.823 |
| kl02 | 64 | 71 | 2993 | 538.0 | 478.1 | 0.889 |
| connectus | 8 | 512 | 2202 | 287.5 | 272.0 | 0.946 |
| lp_osa_14 | 8 | 2337 | 136 | 86.5 | 85.8 | 0.992 |

The near-dense family is 29 of the 32 kernel-fault cells but its median gap is 0.7 µs. The
few-row family is 3 cells and it owns the three worst ratios in the whole corpus. Read the
row counts against `rows/SCORCH_ROWS_PER_THREAD` on a 24-thread pool: kl02 and nw14 get **4
workers of 24**, rn50's 256 rows get **16 of 24**. A 4-of-24 team predicts about 0.7 and the
measurement is 0.731; a 16-of-24 team predicts about 0.67 and the measurement is 0.638.

**So the pool-conditioned row ceiling is the highest-value next step**, not because it fixes
many cells but because it fixes the worst ones, and the mechanism is already measured at
1.3066 / 1.4011 inside its gate on this host. connectus (512 rows → the full pool) and the
transformer block (2048 rows) are *not* explained by under-threading and stay open.

## Asking the compiled rule where the work gate fires, instead of restating it

`scorch_spmm_partition_mode` is now exported from the instrumented build, so the firing set
can be read rather than modelled. Splitting the ARM grid by what the rule actually decides:

| corpus | group | cells | shared counter | the gate | A/A | median µs |
|---|---|---|---|---|---|---|
| tail f32 | gate fires | 253 | 1.0615 | 1.0282 | 1.0023 | 27 |
| tail f32 | gate inert | 167 | 0.9954 | 0.9918 | 0.9991 | 30 |
| general f32 | gate fires | 798 | 1.0219 | 1.0052 | 0.9981 | 15 |
| general f32 | gate inert | 851 | **0.9311** | 0.9961 | 0.9995 | 47 |

**The rule is doing exactly what it is for.** On the 851 general-corpus cells where it does
not fire the partition is 6.9% ahead of the counter and the gate leaves them alone; on the 798
where it does fire the counter is ahead and the gate switches to it.

**And my earlier explanation of why it captures only half was wrong.** I had guessed that the
single-worker gate already forced many cells to mode 0, so an offline threshold sweep was
fitting noise between two identical arms. The exported rule says that group is **0 to 3 cells**
— the hypothesis is dead. What actually differs is that the counter arm sets
`SCORCH_SPMM_PARTITION` alone, so **the exact-width narrow-k kernel is off in it**: it is
today's code, not today's handout with the new kernel. On the firing cells the gate arm also
pays two more successful `getenv` lookups, a per-call charge concentrated below 30 µs, and
those cells are 15–27 µs. Both confounds push the same way and neither is separable from this
arm set, so a fourth arm — today's handout *with* the new kernel — is measuring now.

## The exact-width kernel's degree floor: rejected on both hosts

The floor refuses the exact-width kernel on matrices holding fewer nonzeros than rows. ARM,
44 matrices in three groups, per-matrix z (the corpus contributes each matrix at five widths):

| group | floor off (discriminator) | floor at 1 | floor at 2 | kernel off entirely |
|---|---|---|---|---|
| degree < 1 — the floor fires | 0.9971 | **0.9601** (z −5.5) | 0.9602 | 0.9615 |
| degree 1–2 | 0.9976 | 0.9989 | 0.9909 | 0.9961 |
| degree 2–8 | 0.9957 | 0.9936 | 0.9897 | **0.9532** (z −8.3) |

**On ARM the floor costs 4%** on exactly the shapes where on x86 it recovers 5–17% — by
width, k=2 reads 0.9221 and k=3 reads 0.8933 with the floor on. And the bottom-right cell is
the same kernel's own case for existing: turning it off at degree 2–8 costs 4.7%, per-matrix
z −8.3.

So the NEON exact-width kernel is cheaper per row than the NEON general path even on
mostly-empty matrices. The x86 arm of the same grid has now reported and says the same thing,
which retracts the arm-variance reading above: 390 cells per dtype, 26 matrices per group.

The arm to read against is not the reference but `p3ez`, which sets the floor to 0 — the value
it already has — and so is behaviourally identical while paying for one more variable the code
looks up. It reads **1.0094–1.0101 with per-matrix z near 8**, a systematic ~1% offset that has
to come out before any floor arm is read. The widths where the floor cannot change behaviour
(k ∈ {1, 4, 8}, outside the exact-width kernel's k ∈ {2, 3}) supply the null directly: 0.991 on
float32, 0.999 on float64. Against that null, with the floor ON:

| | k = 2 | k = 3 |
|---|---|---|
| float32 | 0.8% slower | **4.5% slower** |
| float64 | **6.6% slower** | **5.0% slower** |

Per cell the mechanism is visible and it sorts on *width*, not degree. On `Pd_b` (8081 rows,
0.78 nonzeros per row) the exact-width kernel is worth 26.1 µs against 31.4 without it at
k = 3 — 1.20x — and costs 3.8% at k = 2. Every one of the six low-degree matrices I pulled by
hand favours keeping it at k = 3 (0.831, 0.831, 0.954, 0.958, 0.943, 0.932).

**So the floor does not ship, on either host, and `SCORCH_NARROWK_EXACT_MINDEG` stays 0.** The
offline x86 evidence that motivated it was wrong twice over: it conflated the floor with the
row partition, and its counter arm had the exact-width kernel off *and* a different getenv
count. What survives is a real but narrow x86 float32 residual — four tall auxiliary matrices
(`Pd_b`, `Pd_rhs`, `bips07_3078_iv`, `sts4098_b`) lose 3.8–7.6% at k = 2 only, which is
1.2–2.3 µs on ~30 µs calls that already beat MKL. A discriminator that caught those four
without also refusing the k = 3 win would be a per-matrix tune, so it is reported here rather
than built.

## The decision refactor is not byte-identical, and the per-symbol reading is why that is fine

Moving the row-handout rule into `scorch_spmm_partition_mode()` changed the emitted binary:
70125 → 70193 instructions across every symbol whose name contains "spmm", 838 differing
lines. A flattened diff cannot say whether that is the inner loop or the call setup, because
a register renumbering in a prologue propagates through the whole function — here the
differing window is 91% of the entry symbol even though the streams re-sync in opcode terms.

Split by symbol, on both ISAs (x86 by cross-compiling the pinned pre- and post-refactor
headers for `x86_64` with the shipping defines and disassembling the objects):

| | ARM | x86 |
|---|---|---|
| spmm symbols | 220 → 221 | 78 → 78 |
| **identical** | **218** | **76** |
| differing | 2 | 2 |
| new | 1 (`scorch_spmm_partition_mode`, 74 insn) | 0 |

**The arithmetic loops are byte-identical.** On x86 the row lambda is 7325 instructions with
966 fused-multiply/multiply ops (float) and 3653 with 550 (double); the NEON register-tile
kernel is 1672 with 315; the fused Linear kernel is 1256. Every one of them is unchanged. On
ARM the outlined loop bodies are 8680 and 12937 instructions with 1463 and 2266 float ops —
unchanged, including all 1499 and 2166 memory operations.

**The only changed symbol is the once-per-call entry function**, which carries two
multiply/add instructions in total, i.e. no arithmetic. It got *shorter*: 944 → 936 (f32) and
942 → 934 (f64) on x86 with memory-operand instructions 321 → 313 and 318 → 311, and 947 →
943 and 944 → 940 on ARM with 283 → 280 memory ops. It trades a load, a store, two sign
extensions and two branches for a compare and two conditional selects.

The new ARM-only symbol is an out-of-line copy of the decision, which exists **because it is
exported** — the x86 object, which does not export it, has no such symbol and no extra
instructions, which confirms the attribution.

So the waiver is not "trust me, it should inline": the loops are the same bytes, the setup is
strictly smaller, and no memory operation was added anywhere. No runtime A/B is owed.

## The change is in: both hosts correct, the flip proven to be the measured binary

| | ARM (M5) | x86 (redwood) |
|---|---|---|
| correctness, shipping configuration compiled in | **1099 passed, 48 skipped** | **1097 passed, 48 skipped** |
| wall clock | 33 min | 48 min |
| exit | 0 | 0 |

(The two-test difference is collection, not failure: 1150 against 1148 items, the ARM host
carrying two NEON-specific tests.)

Flipped: `SCORCH_SPMM_PARTITION_DEFAULT` 0 → 3, `SCORCH_NARROWK_EXACT_UNROLL` 0 → 4,
`SCORCH_SPMM_PARTITION_MINGRAINS` 0 → 2, `SCORCH_SPMM_PARTITION_GATE_MAXTHREADS` 0 → 16.

The flip was verified rather than assumed: built once with the old defaults plus those four
`-D` flags — the shape every grid measured — then again with the new defaults and no flags, and
**all 221 spmm symbols disassemble identically**. The `-D` grid results therefore describe the
shipped binary, not a near relative of it.

`pre-commit.sh` fails on this repository at `main` too (262 flake8 findings, 186 mypy errors,
14 files black would reformat, all from black-version drift and missing stubs in compiler
files). This branch is cleaner on all three (241, 184, 3) and changes no Python at all.

## What the residual actually is: a per-ROW cost, not a per-call one

I expected the remaining below-MKL cells to be fixed per-call cost — 29 near-dense blocks with
a 0.7 µs median gap on 19 µs calls whose arithmetic is about 2 µs has exactly that shape. So I
measured it directly: sweep the nonzero count at a fixed shape, fit `time = a + b·nnz` for both
libraries, compare intercepts. An empty matrix was deliberately not used as the baseline,
since an empty CSR may take a different path in either library.

**Our per-call cost is lower than MKL's on seven of eight shapes** — intercepts of 2.9–16.6 µs
against 8.9–36.9 µs, and at 512×512 k=64 we are 17–20 µs cheaper. That kills the hypothesis.

But the 50000-row shape read a 63.6 µs "intercept", which is the tell: a single-shape fit puts
everything that scales with **rows** into the constant. Fitting `time = a + c·rows + b·nnz`
jointly across five row counts at k = 4, 48 readings per library:

| float32, k=4 | per call | **per row** | per nonzero |
|---|---|---|---|
| ours | 5.83 µs | **1.162 ns** | 0.173 ns |
| MKL | 6.53 µs | **0.086 ns** | 0.163 ns |

**Per call we are cheaper, per nonzero we are level, and per row we cost 13.5× what MKL
costs.** About 1.08 ns of excess per row — four to five cycles.

That single coefficient predicts the residual on cells it was not fitted to. Over the 32
float32 cells still behind MKL on the production path, `1.076 ns × rows` gives a median 0.55 µs
against an observed median gap of 0.72 µs. It also explains why the deficits cluster below
30 µs, why the degree-below-4 matrices (8081 rows, 6323 nonzeros) lose, and why the near-dense
DLMC blocks at 512–2048 rows lose by a microsecond or two. On the 335 cells we *win*, the same
per-row excess is 7.9 µs against a 14.7 µs margin — we are paying it there too and winning
anyway, so removing it widens those margins as well.

Float64's joint fit is not trustworthy — dropping the 50000-row shape flips the sign of the
per-row difference (1.373 vs 1.512 ns) — and it under-predicts the observed gap fivefold. Only
the float32 coefficient is established.

**Retracted: the per-row coefficient is not identifiable from this design.** See the
section at the end of this file. The rest of this section is kept for the record.

**The fit was at k = 4, so this is the register-block kernel's per-row cost, not the
exact-width kernel's** — the exact-width kernel takes only k ∈ {2, 3}, and the degree-floor
grid confirms it by moving k = 2 and k = 3 while leaving k = 4 at 0.9925. That kills the first
mechanism I reached for (the exact-width kernel's per-row accumulator array) and redirects the
search to whatever the register-block kernel does once per row regardless of row length. The
queued fit at k ∈ {1, 4, 64} is the right grid for that after all, since k = 4 is a
register-block width.

### Counting the per-row work in the source, which agrees with the measured coefficient

`scorch_spmm_row_narrow_exact<T, K, UNROLL>` does this per row, whatever the row's length:

```cpp
T acc[UNROLL][K];
for (u) for (j) acc[u][j] = T(0);          // UNROLL*K zeroing ops
... unrolled main loop over nonzeros ...
for (u = 1..UNROLL-1) for (j) acc[0][j] += acc[u][j];   // (UNROLL-1)*K adds
... remainder loop ...
for (j) C_row[j] = acc[0][j];              // K stores
```

At the shipped `UNROLL = 4` and `K = 4` that is **16 zeroing operations, 12 cross-accumulator
adds and 4 stores — 32 operations per row that do not depend on the row at all.** At four
operations per cycle on a 4 GHz part that is about 8 cycles, 2 ns; the measured excess over
MKL is 1.08 ns. Same order, and the fit was over synthetic matrices while this count is from
the source, so they are independent.

The separate accumulator sets exist to break the FMA dependency chain, so they cannot simply
be collapsed. The three levers, in increasing order of how much they change:

1. **Initialise the accumulators from the first `UNROLL` nonzeros** instead of zeroing them.
   Removes 16 of the 32 operations and costs nothing — the values are loaded anyway.
2. **Size the accumulator set to the latency, not to the unroll.** Four independent chains
   cover a 4-cycle FMA; at K = 4 on float32 that is 16 scalar accumulators for what fits in
   two 256-bit registers, so the reduction is over half-empty vectors.
3. **Two output rows per register** at K = 4 float32, where four floats occupy half a YMM.
   Halves both the zeroing and the reduction per row of output.

None of this is implemented. The discriminating measurement — the same three-term fit with the
exact-width kernel switched off — is queued, and if the per-row coefficient falls to MKL's with
the kernel off, this is confirmed as its cost rather than the row loop's in general.

## The gated row ceiling on ARM: nil, and the control arm is the whole reading

The row ceiling refuses the row partition above a row count, gated so it can only fire where a
pool is large enough to matter. The ARM grid ran it at full power: 165 matrices split into four
groups by whether the gate can fire (in-gate 42, band 41, low-degree 40, wide 42), five widths,
both dtypes, 815 cells per dtype.

`p3ec0` sets the ceiling to 0 — off, the value it already has — so it is behaviourally
identical to the reference and measures nothing but the cost of naming a variable the code
looks up. Reading the real ceiling arms against *it* rather than against the reference:

| group | float32 p3ec / p3ecb | float64 p3ec / p3ecb |
|---|---|---|
| in-gate — where it fires | 0.9974 / 0.9979 | 1.0070 / 1.0070 |
| band | 0.9958 / 0.9987 | 0.9997 / 0.9982 |
| low degree | 0.9987 / 0.9964 | 0.9992 / 0.9978 |
| wide | 1.0011 / 0.9985 | 1.0015 / 1.0003 |

Everything is inside ±0.7%, and the two dtypes disagree in sign in the one group built to
isolate the effect. **The ceiling does not ship on ARM.**

That is a statement about ARM only, and the sentence that stood here — that the x86 verdict
"closes the ceiling on both hosts" — was wrong. The x86 arm of this same grid reopened it: in
its float64 in-gate group the rule reads 1.0651 against a 0.9887 control, and 1.3351 at k = 64.
See the two sections below. On ARM the mechanism cannot exist, because a 128-row matrix already
asks for 8 workers of a 6-thread pool, so there is no headroom to redistribute; nil there is
what the gate failing closed looks like, not evidence against the rule.

The control arm is the more useful number here. `p3ec0` changes no behaviour and reads 0.9652
in the low-degree group, 0.9782–0.9795 in-gate, 0.9887–0.9911 in the wide group — a clean
monotone ordering by how long the kernel runs, which is what a fixed per-call charge looks
like. So the instrumented build's resolution floor is not one number: it is ~3.5% on the
shortest kernels in this corpus and ~1% on the longest. Any effect smaller than that is not
decidable in a hooked build, which is why three of this session's four knobs came back nil.


## Retraction: the residual is per-NONZERO, not per-row

The per-row reading above does not survive being asked twice. Refitting
`time = fixed + per_row*rows + per_nnz*nnz` at k = 4 over the same 48 cells, the only thing
that decides the sign of the per-row difference is whether the 50000-row shape is in the fit:

| float32, k=4 | fixed (µs) | ns/row | ns/nnz | R² |
|---|---|---|---|---|
| all shapes — scorch | 6.19 | **−0.167** | 0.1750 | 0.996 |
| all shapes — MKL | 6.63 | 0.215 | 0.1615 | 0.992 |
| without the 50000-row shape — scorch | 8.54 | **+1.156** | 0.0605 | 0.637 |
| without the 50000-row shape — MKL | 10.47 | 0.015 | 0.0490 | 0.498 |

The per-row difference is −0.382 ns one way and +1.140 ns the other, and R² falls from 0.996
to 0.637 when the leverage point comes out — so within the small shapes the linear model
explains almost nothing and the big shape is supplying all of the apparent fit. Float64 does
the same thing (−0.666 against +0.158). The earlier note had already recorded this for float64
and treated float32's coefficient as established; it is not. The 1.162 ns figure was a choice
of shapes, and its "out-of-sample" prediction of 0.55 µs against a 0.72 µs observed gap sits
inside the model's own ±0.6 µs spread, so it was never a confirmation.

**What is robust to the shape selection is the per-nonzero term.** On float32 we cost more per
nonzero than MKL in both fits — 0.0605 against 0.0490 and 0.1750 against 0.1615, 8–23% — while
our *fixed* cost is lower in both (8.54 against 10.47, 6.19 against 6.63). That agrees with the
production scoreboard, where the non-kernel cost is −2.4 µs, i.e. already in our favour.

And a per-nonzero deficit at k = 4 float32 has a mechanism in the source rather than in a
regression. The width lands on `scorch_spmm_row_regblock<float, NVEC=1, FULL_LAST=false>`, whose
inner loop issues `_mm256_maskload_ps` for **every nonzero**, using 4 lanes of 8. So each
nonzero pays a masked 256-bit load instead of a plain one and throws away half the FMA width.
`scorch_simd<T>` has no 128-bit form, so there is currently no way to express "four floats,
full width, no mask" — which is exactly what k = 4 is.

That also explains the per-width shape of the exact-width kernel's own result without appealing
to the aligned load: it wins 6–8% at k = 2 and k = 3 because a scalar loop has no mask either,
and it loses from k = 5 up because there the mask is over half full and `UNROLL*K` accumulators
begin to spill. At k = 4 neither kernel is the right one — the scalar loop has no vector width
left to exploit and the masked 256-bit loop wastes half of its own.

## The exact-width bound does not want k=4, and float64 calibrates the whole design

The per-width sweep that set `SCORCH_NARROWK_EXACT_HI` to 3 measured k = 1, 2, 3, 5, 6 and 7
and asserted the boundary at 4 without a number for it — while 22 of the 32 near-dense cells
behind MKL sit at k = 4. Widening the bound is a runtime hook, so the missing number was cheap
to get: 302 matrices (the 180 the original sweep used, plus 122 grouped by density), five
widths, both dtypes, arms differing only in the value of one variable so the getenv charge
cancels exactly.

**It is nil.** At k = 4 on float32 the widened arm reads 1.0044 against a 1.0030 slot floor,
and nothing anywhere in the grid clears |z| = 2.5. The unroll-capping variant, which exists
because `UNROLL*K` accumulators start spilling above k = 4, changes nothing either.

The float64 half of this grid is worth more than its verdict, because on float64 the exact-width
kernel caps at 3 and **all five arms are provably the same code path** — the whole 1510-cell
grid is a null. It reads within ±0.6% with z running up to **±3.1**. So |z| ≤ 3 is the floor in
this design, which retires every float32 reading in the table above, and is the number to hold
other hooked grids to.

### The width where we actually lose, and why float64 is the control for it

Kernel time alone against MKL's call, on this corpus, for the arm that ships:

| | k=2 | k=3 | k=4 | k=5 | k=8 |
|---|---|---|---|---|---|
| float32 | 0.9654 | 0.9871 | **0.9321** | 1.0706 | 1.0390 |
| float64 | 1.0380 | 1.0776 | **1.0629** | 1.1208 | 1.2402 |

float32 k = 4 is the worst cell in the grid — 200 of 302 matrices behind MKL — and float32 is
behind float64 at every width. That is the shape the mask mechanism predicts, and k = 4 makes
it a controlled comparison rather than an argument: **float32 has 8 lanes, so k = 4 is 4 of 8
and every nonzero pays `_mm256_maskload_ps`; float64 has 4 lanes, so k = 4 is FULL_LAST and
pays a plain load.** Same width, same kernel, same corpus, one masked and one not — 0.9321
against 1.0629.

Stated before the measurement: if the mask is the mechanism, giving float32 k = 4 a 128-bit
register block (`SCORCH_SPMM_HALFVEC=1`, four floats, no mask, full FMA width) should move it
from 0.9321 toward its unmasked twin's 1.0629, about 14%, and should do nothing at k = 2, 3 and
8, where the hook cannot fire. If it moves the widths it cannot reach, the reading is the
instrument.

One caveat on the whole-call column that is not in the table: on this harness path the same
cells read 0.6848 with 248 of 302 behind MKL, because passing `time_dict` disables the
per-tensor plan cache and adds about 7 µs to a kernel that is only a few microseconds long.
That is the measurement path, not the caller path, and the kernel column is the signal.

## Sizing the row axis off nonzeros, and a pool sweep that beats a second host

The row axis is `rows / SCORCH_ROWS_PER_THREAD`, so a 128-row matrix asks for 8 workers of a
24-thread pool however much work each row carries. `SCORCH_SPMM_NNZ_PER_THREAD` sizes it off
nonzeros instead and raises the count to `nnz / v`, capped at the caller's pool. 103 matrices
with more work per row than 128 nonzeros and fewer than `16 × pool` rows, in three groups by
how starved they are, five widths, both dtypes, 515 cells per dtype per pool. Every arm sets
the same five variables; only `v` differs.

**It is real at k ≥ 32 on both dtypes, and the value of `v` does not matter.**

| pool 24, best arm | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|---|
| float32 | 1.0199 z+2.9 | — | 1.0557 z+3.3 | **1.0898 z+3.8** | 1.0300 z+2.8 |
| float64 | 1.0228 z+3.7 | — | 1.0796 z+3.1 | **1.1273 z+6.5** | 1.0580 z+2.6 |

All four values of `v` from 128 to 2048 — sixteen-fold apart — read within a percent of each
other. That is not four thresholds agreeing; it is the **pool cap** doing the work, because for
every matrix in this corpus `nnz/128` and `nnz/2048` both exceed the pool. So the rule is not
"one worker per 256 nonzeros", it is "use the whole pool rather than `rows/16` of it", and there
is no constant in it to tune. k=16 is a dead zone on both dtypes and both pools, unexplained.

And it moves cells across MKL parity, which is the point: at float64 k=64 the group goes from
0.9715 with 56 of 103 behind MKL to **1.0725 with 34 behind**.

### The pool sweep is the mechanism test, and it is better evidence than a second host

This rule cannot be confirmed on the M5, and not for want of trying: at a 6-thread pool a
128-row matrix already asks for 8 workers, so the headroom the rule exploits does not exist and
the ARM grid's nil is the gate failing closed. Halving redwood's pool to 12 is the test that a
second host would only have imitated, because it changes the one quantity the mechanism names.

Both predictions hold. The effect **shrinks with the headroom** — float64 k=64 goes 1.1273 at
pool 24 to 1.0792 at pool 12, float32 k=32 goes 1.0557 to 1.0260 — and the group the gate must
*exclude* at pool 12 goes flat while its neighbour does not:

| pool 12, float64, arm n2048 | k=32 | k=64 |
|---|---|---|
| `starve12` — gate still fires | 1.077 z+3.0 | **1.156 z+5.4** |
| `starve24` — `rows/16 ≥ 12`, gate refuses | 1.001 z−0.6 | 0.997 z−0.1 |

`starve24` reads 0.995–1.005 with every |z| under 1.4 across all five widths and both dtypes at
pool 12, having read 1.004–1.057 at pool 24. The gate self-limits on the quantity it is written
in terms of, measured rather than argued.

### Why this is shippable rather than tuned

Two properties, both readable in `scorch_policy.h` rather than inferred from the grid:

- The rule is **monotone**: `if (by_nnz > rows_axis) rows_axis = by_nnz;`. It can only raise the
  worker count, never lower it, so it cannot take parallelism away from a shape it misjudges.
- It is **provably inert above the gate**: with `CEIL_ROWBIND`, `ceil_pool_ok` returns false
  once `rows / SCORCH_ROWS_PER_THREAD >= pool`, i.e. for every matrix with at least `16 × pool`
  rows — 384 on redwood, 96 on the M5. Most of the collection is above that line.

Nothing in the grid is significantly negative: the worst reading anywhere is `starve4` at
float32 k=16, 0.970 with z −2.0, inside the ±3 floor the float64 null established. The
remaining work before this ships is the no-regression pass on a general corpus, where the gate
should read flat because it cannot fire, and confirmation on the caller path — both of the
parity numbers above are from the harness path, which adds about 7 µs by disabling the plan
cache and so understates a 30 µs call by a fifth.

## The k=4 story was a pooling artifact: the variable is matrix SIZE, not width

The table above reads float32 k=4 as the worst cell in the grid and k=8 as a win, and I built a
128-bit kernel on that reading. Splitting the same 1510 cells by size instead of pooling them
shows the width was carrying an artifact. Kernel time against MKL's call, cells behind MKL
beside each:

| float32, nnz | k=2 | k=3 | k=4 | k=5 | k=8 |
|---|---|---|---|---|---|
| < 20k | 1.789 6/108 | 1.797 7/108 | 1.723 7/108 | 1.798 7/108 | 1.802 10/108 |
| 20k–200k | 0.698 98/101 | 0.731 100/101 | 0.728 100/101 | 0.834 91/101 | 0.823 98/101 |
| 200k–1M | 0.657 55/56 | 0.668 56/56 | 0.599 56/56 | 0.748 50/56 | 0.683 54/56 |
| > 1M | 0.692 37/37 | 0.704 37/37 | 0.594 37/37 | 0.801 35/37 | 0.743 36/37 |

Every width behaves the same way. k=8's pooled 1.0390 was carried entirely by the 108 small
matrices; at 200k–1M nonzeros k=8 reads 0.683 with 54 of 56 cells behind, no better than k=4's
0.599. So "k=4 is the width where we lose" was the size distribution of the corpus showing
through a width average.

**And the per-library normalisation says the k=4 dip is MKL's curve, not ours.** Dividing each
library by its own k=8 time on the same matrix:

| float32, k=4 vs own k=8 | < 20k | 20k–200k | 200k–1M | > 1M |
|---|---|---|---|---|
| scorch(8)/scorch(4) | 1.054 | 0.977 | 1.068 | 1.053 |
| mkl(8)/mkl(4) | 1.103 | 1.105 | 1.217 | 1.316 |

Ours is flat: our k=4 costs the same fraction of our k=8 at every size. MKL's grows. So nothing
in our kernel singles out k=4, and the earlier "float32 k=4 is 0.9321 where its unmasked
float64 twin is 1.0629" comparison is not evidence about the mask — both numbers are pooled
over a size distribution, and float64's corpus-weighted average simply sits on the other side
of the crossover.

### What the size split does establish, and the one thing it cannot

Two readings survive, and they are not the same claim:

- **Above ~20k nonzeros our kernel is 0.60–0.85 of MKL's call at every width from 2 to 8**, on
  91–100% of cells. That is trustworthy: at 220 µs the ~7 µs harness cost is 3% and `_kms`
  excludes it anyway. It is also not new — it is the narrow-k B-size band (L2-miss, L3-hit)
  that the `scorch_spmm_chunk` guard already addresses part of, re-found from a different
  direction, and the size dependence here is the same mechanism stated as a corpus split.
- **Below ~20k nonzeros the 1.7–1.8x is partly an artifact of the columns**, because `e3_kms`
  is our kernel and `mkl_ms` is MKL's whole call. On a 5 µs call MKL's dispatch is a large
  fraction of what it is charged. The fixed-cost fits put our per-call cost at 8.54 µs against
  MKL's 10.47, a 1.23x edge — not 1.8. So the small-matrix win is real in direction and
  overstated in size by this measurement.

Both halves want the same fix in the harness rather than in the analysis: compare whole call to
whole call on the path a caller reaches. That is what `rw_chain39.sh` does, from a hookless
build, at the widths the published scoreboard skipped.

**The half-vector kernel is still worth measuring, but its motivation is now narrower.** The
mask is still a real per-nonzero cost and 128-bit still removes it; what the size split
withdraws is the claim that k=4 is anomalous and that 14% is the prize. It should be judged on
what it does at k=4 within each size band, not on the pooled number.

### The size deficit is not a selection artifact, but it is family-dependent

180 of the 302 matrices above come from `narrowk_groups.csv`, which was built around the
narrow-k deficit, so "91–100% of cells behind" partly measures how the corpus was chosen. The
other 122 were sampled by density from DLMC with no reference to MKL. Splitting on provenance,
float32 kernel time against MKL's call:

| float32, nnz | density-sampled, 122 matrices | narrowk groups, 180 matrices |
|---|---|---|
| < 20k | 1.71–1.79, 5–7 of 43 behind | 1.73–1.84, 0–3 of 65 behind |
| 20k–200k | **0.733–0.862**, 59–68 of 68 | 0.631–0.817, 32–33 of 33 |
| > 200k | **0.693–0.938**, 6–11 of 11 | 0.585–0.749, 79–82 of 82 |

The unselected half shows the same phenomenon at nearly the same size, so the crossover is
real. Selection makes it about 0.1 worse, which is what selection should do.

**But a third corpus disagrees.** The 165-matrix ceiling grid, also unselected on parity and
also mostly DLMC, reads 0.975–1.206 at 20k–200k and 0.805–1.257 at 200k–1M on float32, and wins
at every size at k=64. Its matrices are rn50 blocks — dense-ish, high degree, few rows — where
the density-sampled groups spread across sparsity levels and shapes.

So the crossover is family-dependent, and none of these three corpora is the one the published
scoreboard used. Quoting a single "above 20k nonzeros we are 0.7x of MKL" would be picking a
corpus. `rw_chain39.sh` measures `final_groups.csv` — the scoreboard's own 362 matrices —
whole call against whole call from a hookless build, across k = 2, 4, 8, 16, 32. That is the
number to quote, and it is the one that does not yet exist.

## The no-regression pass, and the degree floor it forced into the shipping configuration

302 matrices, four widths, both dtypes, arms differing only in the value of one variable. Split
on whether the gate can fire at all — `rows / SCORCH_ROWS_PER_THREAD < pool`, i.e. rows < 384
on a 24-thread pool.

**Outside the gate the rule is inert, measured rather than argued.** Over 224 matrices the arm
reads 0.9957–1.0006 against a floor of 0.9985–1.0046, every |z| ≤ 2.0, and the harmed tail is
0.9% against the floor's 0.6% on float32 and 0.3% against 0.0% on float64 — 3 and 8 cells of
896. That is the `ceil_pool_ok` refusal doing what the source says it does.

**Inside the gate, on a general corpus, the distribution widens both ways.** float64 goes from
8.3% of cells more than 10% slower and 10.3% more than 10% faster, to 11.9% and 14.1%. The mean
is still positive (1.0059–1.0134), so it is not a systematic loss — it is a thread-count change
redistributing outcomes on kernels short enough that the same-code floor itself puts 8–11% of
cells outside ±10%. But "the mean is fine and the tail got wider" is not the no-regression
standard.

Splitting those in-gate cells by degree says exactly where the harm is:

| in-gate subset | float32, floor → rule | float64, floor → rule |
|---|---|---|
| degree ≥ 128 — 20 matrices | 0.9783 → **1.0053**, harmed 15 → 13 | 1.0254 → **1.0338**, helped 12 → 21 |
| degree < 128 — 58 matrices | 1.0069 → 1.0111, neutral | 1.0039 → **0.9886**, harmed 18 → **27** |

All of it is the float64 low-degree subset, and on the high-degree subset the rule *reduces* the
harmed count on float32 while adding to the helped count on float64.

**Two different knobs are both called a degree floor in this file, and one is rejected while
the other ships.** `SCORCH_NARROWK_EXACT_MINDEG` gates the exact-width *kernel* and is rejected
on both hosts (it costs 4.5–6.6%, see the section above). `SCORCH_SPMM_CEIL_MINDEG` gates the
row-axis *thread count* and ships at 128. They share a word and nothing else: one decides which
arithmetic kernel runs, the other decides how many workers get rows. A reader who conflates them
will read this file as contradicting itself.

This is not a threshold discovered by sweeping. `nnz >= 128 * rows` is the filter chain31's
corpus already applied, because the mechanism needs enough work per row to be worth
redistributing — a matrix with four nonzeros per row has nothing to give a twenty-fourth worker.
`SCORCH_SPMM_CEIL_MINDEG` states that precondition and already exists. Setting it to 128 makes
the shipped rule fire on the family it was measured on and provably not on the family that
reads 0.9886.

**So the shipping configuration is four values, not three**, and because it is a different
configuration from the one measured above it gets its own confirm on both corpora
(`rw_chain41.sh`) rather than being inferred from this table:

```
SCORCH_SPMM_NNZ_PER_THREAD = 256    // insensitive: 128..2048 agree within 1%, the pool cap binds
SCORCH_SPMM_CEIL_CAP_POOL  = 1      // cap at the caller's pool, not at omp_get_num_procs()
SCORCH_SPMM_CEIL_ROWBIND   = 1      // the row condition as the mechanism: rows/16 < pool
SCORCH_SPMM_CEIL_MINDEG    = 128    // enough work per row to redistribute
```

Two guardrail families need no re-measurement, because the rule cannot reach them. There is one
call site, `spmm_csr_v2_core`. Every GCN graph is outside the gate by row count — the smallest,
cora at 2708 rows, asks for 169 workers against a pool of 24 — and the autoencoder's shipped
path is `sparse_linear`, which routes to the fused `spmm_csr_linear_fused_float` and never calls
the rule. Inertness by construction beats inertness by measurement.

## What the size deficit actually is: ~19 cycles a nonzero, which is L3 latency over two loads

The crossover is not a cache-residency cliff and not a bandwidth wall. Computing the working
set of every cell — `A = nnz*8 + rows*4`, `B = cols*k*4`, `C = rows*k*4` — **exactly one of 302
cells exceeds redwood's 36 MB L3**, and the median at >1M nonzeros is 14.5 MB. So in these warm
measurements both libraries run entirely out of cache, and nothing here is DRAM-bound.

What the deficit is, is a per-nonzero throughput gap that is flat in size once fixed cost stops
dominating. float32 k=4, median cell in each band:

| band | median ours | median MKL | ours ns/nnz | MKL ns/nnz | ratio |
|---|---|---|---|---|---|
| < 20k | 7.8 µs | 13.7 µs | 1.49 | 2.61 | we win, on fixed cost |
| 20k–200k | 22.1 µs | 16.3 µs | 0.25 | 0.19 | 1.32 |
| 200k–1M | 125.0 µs | 73.3 µs | 0.22 | 0.127 | 1.73 |
| > 1M | 220.5 µs | 129.7 µs | 0.18 | 0.108 | 1.67 |

So the crossover at ~20k nonzeros is not a mechanism — it is just where a per-nonzero cost we
lose overtakes a fixed cost we win. Our fixed cost is genuinely lower (7.8 µs against 13.7 at
5242 nonzeros); our per-nonzero cost is genuinely worse, by about 1.7x, and it does not improve
with size.

**0.18 ns per nonzero over 24 threads is about 19 cycles per nonzero per thread.** For one
sequential index load, one sequential value load, an integer multiply and one FMA, the floor is
two to four cycles with the operand in L1. Nineteen is what you get when the *random* load —
the gather into B — is not covered: L3 latency is around 40 cycles and the register-block kernel
keeps exactly **two** B loads in flight. 40/2 = 20. MKL's 0.108 ns/nnz is about 11 cycles, which
is what four or more in flight would give.

That is a quantitative case for the thing the source comment on `regblock_deep` already argued
and the thing its measurement was thought to have refuted. It did not refute it: that kernel
carries no prefetch while the shipped 2-deep kernel does, so the arm that lost by 14.5% changed
depth and prefetch together. `rw_chain42.sh` separates them.

It is also a case for `rw_chain43.sh`, from the other side: a wider chunk keeps a worker on
consecutive rows for longer, which is the only thing that makes B's lines worth keeping, and the
chunk-width rule's own ceiling is what currently discards the width the locality model asks for.

### What this rules out

- **DRAM bandwidth.** One cell of 302 exceeds L3.
- **A per-call cost.** The 20k–200k band's median call is 22.1 µs, well under the ~52 µs
  native-side figure recorded earlier, so that overhead is not what this band is made of.
- **A width effect.** The per-nonzero ratio is the same at k=2, 3, 4, 5 and 8.
- **Lane masking as the main term.** A masked load costs the same per nonzero at every size,
  and would not leave the ratio flat while the absolute per-nonzero cost stays at 19 cycles.
  The half-vector kernel remains worth its measurement, but it cannot be worth 1.7x.

## Two grids on this host disagree by 1.5x about our own kernel, and agree about MKL to 1%

The section above diagnoses a size deficit from the `exact4` grid. Before building anything on
it I checked the diagnosis against another grid, and the two do not agree.

`exact4` and `split_final` share 17 matrices and two widths (k=2 and k=8). Both were produced by
the same harness on this host. On MKL they agree: as-caida_G_055 at k=2 reads 0.0475 ms in one
and 0.0479 in the other, and the median MKL spread over all shared cells is under 1%. On **our**
kernel the same cell reads 0.0759 ms in the newer grid against 0.0223 in the older — 3.4x — and
over all 34 shared pairs the older grid reads **1.545x (float32) / 1.516x (float64)** better
parity. Both arms of the newer grid are equally slow (its automatic A/A duplicate reads 0.0752
against 0.0759), so this is not an arm; rows, nnz, mean_row, max_row and Bmb all match to the
digit, so it is not the matrix.

### The one hypothesis with a number attached, predicted and then refuted

A `SCORCH_TUNE_HOOKS` build makes 44 `std::getenv` calls on the per-call path — 24 in `spmm.h`,
20 in `scorch_policy.h` — all inside the region `time_dict["eval_time"]` measures, and a miss
scans the whole environment. MKL is timed through `torch.sparse.mm` and enters none of our code,
so unlike the arm-to-arm charge already recorded here, this one would land on one side of the
ratio only. It is the right shape: a fixed per-call cost is 3.4x of a 22 µs kernel and 1.3x of a
200 µs one, which is why the deficit had no signature in width, degree, B footprint or nonzero
count.

Measured before reading any verdict, so the prediction could fail: 44 misses against this
host's environment cost **0.34 µs per call**, 7.7 ns a lookup. The hypothesis is wrong by 160x
and the hooked getenv path is not worth optimising.

### What the difference actually looks like

Taking the difference in microseconds rather than as a ratio separates an additive charge from a
slower kernel, and it is neither — it is three different things:

| family | rows | delta (µs) | reading |
|---|---|---|---|
| 6 DLMC pruned-ResNet layers | 64 | −0.6 to +0.7 | the two builds agree exactly |
| 5 as-caida graphs | 31379 | +47 to +56, the same at k=2 and k=8 | a fixed charge that scales with nothing in k |
| connectus, bibd_17_8, cari | 136–512 | +3 to +34 | small next to a 100–150 µs kernel |
| nw14 at k=8 | 73 | 81 → 832 | **10x** |
| lp_osa_14 at k=8 | 2337 | 38 → 623 | **16x** |

Regressing the delta on nonzeros gives R² 0.11, on rows 0.00, and on the kernel's own time 0.07.
No single mechanism fits, which is why no more of this is worth doing by inspection.

The last two rows are the interesting ones, and they are the shape the row-axis rule in
`scorch_spmm_nthreads` names in its own comment: few rows, very high degree. nw14 is 73 rows of
mean degree 12396, lp_osa_14 is 2337 rows, and kl02 (71 rows, degree 2993) is the cell that
comment is written about.

But the nonzero-expressed ceiling does not explain the 10x, and the arithmetic says so without
a measurement. With `RAISE_GRAINS` at 2, nw14 at k=8 has `work_true/300000 = 24`, so it already
resolves to 24 workers today; turning the ceiling on moves it to 32, which is a widening and not
a 10x. Where the ceiling is the fix is the same family at NARROW k, because that is where the
raise's real-arithmetic bound is what binds: nw14 at k=2 goes from 6 workers to 32, and kl02 at
k=2 from 4 to 22. Both are inside the gate (73 and 71 rows against `CEIL_MAXROWS` 128; degree
12396 and 2993 against `CEIL_MINDEG` 128). So nw14's k=8 reading has a different cause, and
locating it is a question about which build, not about the thread rule.

### What this does to the section above, and to the published number

The ~19-cycles-a-nonzero figure, the "0.60–0.94 above 20k nonzeros" family, and the case it
makes for the multi-row and depth-versus-prefetch kernels all come from the grid that reads
1.5x worse. On the other grid, every degree band at every width from k=1 to k=64 is at or above
parity (0.95 to 3.40), and the only cells below 1.0 are degree ≥ 64 at k=1: 0.954 float32 and
0.960 float64.

So the honest position is that this session does not yet know its own parity to better than
1.5x, and no kernel should be shipped against a number in that range. `rw_chain37.sh` settles
it the only way that can: the same corpus, the same harness process, alternating between the
hooked build and the hookless build that has the candidate shipping configuration compiled in,
three rotations with the order flipped in the middle one, and eight matrices where the two
original grids agreed carried along as the control group.

## What is actually below MKL on the better grid: few rows, very high degree, and half of it at k=1

Banding that grid's parity by degree and reading the geomeans says every band from k=1 to k=64
is at or above parity. That reading is wrong, and it is wrong the way pooled numbers usually
are: the deg 8–16 band reads 1.377 to 2.438 while containing a cell at 0.815. Counting cells
instead of averaging them:

| | float32 | float64 |
|---|---|---|
| cells below MKL, nnz ≥ 20k | 64 of 620 | 53 of 620 |
| below by more than that cell's own A/A spread | 55 | 50 |
| by width | k=1: 33, k=2: 7, k=8: 18, k=64: 2, k=256: 4 | k=1: 34, k=2: 8, k=8: 7, k=64: 2, k=256: 2 |

**The shape is few rows at very high degree, and k=1 is half the count.** The worst cells:

| parity | k | rows | degree | ours | MKL |
|---|---|---|---|---|---|
| 0.482 (f32) | 8 | 71 | 2993 | 81.6 µs | 39.4 µs |
| 0.621 (f64) | 8 | 71 | 2993 | 54.9 | 34.1 |
| 0.675 (f64) | 1 | 73 | 12396 | 115.8 | 78.1 |
| 0.755 (f32) | 1 | 256 | 1152 | 30.0 | 22.6 |
| 0.801–0.831 (f64) | 1 | 2048 | 154–300 | 33.0–46.2 | 26.4–38.4 |

The 37 few-row losers are mostly **DLMC transformer and pruned-ResNet layers** — 256 to 2048
rows at degree 100 to 2300, `body_encoder_layer_*_self_attention`, `bottleneck_*_block_group*`,
`body_decoder_layer_*_ffn_conv1`. This is a family real callers have, which the size band the
previous sections chased is not.

### The lever this points at was built, instantiated, and never measured

k=1 is where the nonzero-axis gather kernel ships, and the comment on
`scorch_spmm_row_gather_f32_ms` already states the mechanism: the profile there is memory
latency, MKL's SpMV-shaped loop keeps eight or more loads in flight, and one `VGATHERDPS` is a
single outstanding memory operation. That kernel runs S independent gather+FMA chains for
exactly this reason. It is instantiated for (K,S) = (1,2) (1,4) (1,8) (2,2) (2,4) (4,2), gated
behind `SCORCH_NARROWK_GATHER_STREAMS`, defaulted to 1, and there is no measurement of it
anywhere in this file. The `SCORCH_NARROWK_UNROLL` result that concluded "stream depth is not
what narrow k is short of" deepened the *regblock* kernel, which is not the kernel k=1 runs —
the gather kernel's own comment says so.

There is a second candidate for the same family. `scorch_spmm_row_narrow_exact<T, K, UNROLL>`
keeps UNROLL independent **scalar** accumulator chains, and `SCORCH_NARROWK_EXACT_K1` lowers the
exact band's floor to 1 so it claims the width (`exact_width` is tested before
`narrowk_gather`). Its comment says the gather measures better at k=1, but that predates the
unroll now compiled in, and on Intel eight scalar loads issue two a cycle while one gather is
microcoded — the same loads-in-flight argument by a different route. It is also templated on the
scalar type and its dispatch has a double branch, so unlike the streams kernel it is a candidate
for the 34 float64 k=1 cells as well.

`rw_chain42.sh` measures both as one six-arm ladder in which **every arm sets exactly one
variable this code looks up** — equal name counts and equal lookup counts, which is what the
second environment charge needs and `--pad-env` cannot give. k=2, 4 and 8 come along as free
structural nulls, since `narrowk_gather` is 1 only at k=1 by default and the exact band already
contains 2: three widths of pure null on the same matrices in the same run, which is a better
resolution floor than an A/A arm alone. float64's streams arms are a fourth null.

## A user-facing op with a provable ceiling that no grid in this session has measured: SpMV

Every parity number in this file is SpMM. `scorch.matmul(A, x)` with a 1-D `x` is a different
code path — matmul's 2-D × 1-D branch into `ops.spmv`, which resolves to
`prebuilt_spmv_csr_*` (verified: `resolve_prebuilt_matmul` returns `prebuilt_spmv_csr_f32`).
That kernel has never appeared in a grid, and reading its disassembly says why it should have.

Its row loop is one accumulator. The compiler does not widen it. The extension is built with
`-O3 -march=native -ffast-math -funroll-loops`, and in the float and double bodies of
`spmv_csr` on this host GCC unrolls eight to sixteen deep and then targets the **same register**
with every FMA:

```
vmovss     (%rsi,%rbp,4),%xmm0
vfmadd132ss (%rdi,%rdx,4),%xmm1,%xmm0
vmovss     (%rsi,%rdx,4),%xmm6
vfmadd231ss (%rdi,%rbp,4),%xmm6,%xmm0      <- and every one after it, into xmm0
```

So a row costs one **FMA latency** — four cycles — per nonzero, whatever the memory system is
doing, and there is no gather anywhere: the B load is indirect and GCC declines to vectorise a
reduction over one. `-ffast-math` permits the reassociation that would fix this and the compiler
did not take it. That is a hard ~4 cycles/nonzero/thread ceiling, roughly 4x off what the
arithmetic allows, and it is what a caller doing `A @ x` gets today.

`scorch_spmv_row<T, ACC>` (in `kernels.h`) runs ACC independent chains instead. Nothing routes
to it: `SCORCH_SPMV_ACCUM` is 1, and in a build without hooks it is `constexpr`, so the switch
folds and the other three instantiations are never emitted.

**Getting the default to byte-identity took two attempts, and both failures are worth recording.**
Written the obvious way — `nb = n - n % ACC` with a remainder loop — `ACC == 1` makes both
provably dead and the compiler emitted them anyway: a full normalised disassembly of `ops.o`
moved **3040 lines**, including a vector zero and a block-count guard the function never had.
Given its own `if constexpr (ACC == 1)` branch containing the original loop, it still moved **55
lines** at an identical instruction count, because reaching the same loop through a call changes
when `A1_pos[i + 1]` is loaded — the bound becomes an argument evaluated before the call rather
than a condition inside it. Only with the one-chain case written out **at its original call
site** is the object unchanged across all 155951 instruction lines. The duplication is what
byte-identity costs here, and it is cheaper than a waiver.

Correctness: 4000 rows over eight (dtype, ACC) combinations against a double-precision scalar
reference, lengths 0 to 40, so empty rows and lengths that are not multiples of ACC are covered.

`rw_chain47.sh` measures it — the 362-matrix corpus, `ref`/`refb`/`a2`/`a4`/`a8`, both dtypes,
against MKL's own sparse mat-vec (`torch.mv`), banded by degree because ACC chains cannot pay on
a row shorter than ACC. The probe is `vprobe.py`, a marked copy of `kprobe.py` rather than a mode
inside it: nine queued chains invoke kprobe and an edit there to add an axis it does not have
would risk all of them for nothing. The five changes are all about the operand being a vector;
everything that decides the measurement — per-arm batching, interleaved random arm order, the
automatic A/A duplicate, environment padding, min-of-reps — is untouched, so the two probes'
numbers are comparable. Its refusal counts **ok rows, not lines**, because a CSV of 362 error
rows would pass a line count.

### A prediction, written before the verdicts

Recorded ahead of `rw_chain42.sh` and `rw_chain46.sh` so it can fail, which is what made the
environment-lookup hypothesis cheap to kill.

Take the worst float32 k=1 cell: `bottleneck_2_block_group3_1_1`, 294912 nonzeros over 256 rows
at degree 1152, ours 30.0 µs against MKL's 22.6. B at k=1 is about 4.6 KB, so it is L1-resident
and no gather can be missing. The thread rule resolves this shape to **16 workers**:
`rows / SCORCH_ROWS_PER_THREAD = 16`, the work term `nnz * max(k,16) / 150000 = 31` does not
bind, and the raise needs `work_true / (2 * 150000) = 294912 / 300000 = 0` grains, so it never
fires. The row loop at k=1 is one `VGATHERDPS` plus one FMA per eight nonzeros — under a cycle
per nonzero out of L1, and 4 cycles per eight even if the single accumulator chain is fully
exposed.

30.0 µs over 16 workers is **7.2 cycles per nonzero per thread**, seven to fourteen times any of
those bounds. `eval_time` brackets the whole native call, so it contains the OpenMP fork/join and
the output allocation. Sixteen workers launched for roughly 4 µs of arithmetic is a fork/join
cost, not a kernel cost, and MKL's 22.6 µs on the same shape is mostly fixed cost too.

So the prediction is:

- **`rw_chain42.sh` reads near its floor at k=1.** The streams ladder and the scalar-unroll arm
  are both changing the row loop, and the row loop is not what these cells are made of. If the
  ladder does move, the mechanism above is wrong and the gather is more exposed than the
  arithmetic says.
- **`rw_chain46.sh` finds a rung below the default that wins**, with the effect concentrated in
  the `rows<600` bands, because that is where `rows/16` hands out workers a k=1 product cannot
  feed.
- A third lever follows if both hold and neither is enough: the per-call floor itself — the
  allocation and the team launch — which no arm in either chain touches.

This also means the two chains are not redundant with each other, and that the k=1 half of the
below-MKL count may not be a kernel problem at all.

### Correcting that prediction before it was tested: the floor is 4 µs, not 26

The prediction above assumed the 30 µs was mostly fork/join and allocation. Measuring the floor
instead of assuming it says otherwise, and the data was already on disk.

The grid contains products with **one nonzero**, which is as close to pure per-call cost as a
measurement gets. Ours, on 20k–23k rows:

| k | nnz | rows | ours | MKL | parity |
|---|---|---|---|---|---|
| 1 | 1 | 23412 | 8.9 µs | 18.9 µs | 2.12 |
| 2 | 1 | 23412 | 9.6 | 46.1 | 4.80 |
| 8 | 1 | 23412 | 17.8 | 55.0 | 3.10 |

and the **lowest kernel time anywhere in the grid is 4.21 µs (float32) / 4.11 µs (float64),
against MKL's 12.84 / 12.69**. So our per-call floor is about a third of MKL's, and what floor
there is scales with the OUTPUT (one nonzero over 23412 rows costs 8.9 µs at k=1 and 17.8 µs at
k=8 — that is zeroing 749 KB, not launching a team).

The worst k=1 cell has 256 rows, so its output is 1 KB and its floor is ~4.5 µs, not 26. Removing
it leaves ~25.5 µs over 16 workers = **6.1 cycles a nonzero per thread**. A `VGATHERDPS` covers
eight nonzeros at roughly 20–25 cycles of latency, so even fully exposed with one chain in flight
that is ~2.8 cycles/nonzero. We are still about 2x above the latency-bound estimate and far above
the throughput one.

**So the k=1 deficit is kernel-side after all, and `rw_chain42.sh` is the right experiment.** The
fork/join reading was wrong; the corrected prediction is that the streams ladder *does* move at
k=1, and that `rw_chain46.sh`'s thread ladder finds much less than the earlier reasoning implied.

One mechanism falls out of the same reading, and it is specific. `narrowk_gather` is gated on
`std::is_same<scalar_t, float>`, so there is **no gather kernel at all for float64 k=1**: those
cells run the register-block kernel with `nvec == 1` and a one-lane mask, doing one lane of
useful work in four. That is a mechanical explanation for float64 k=1 being the worst band
(34 cells, minimum 0.675), and it predicts the `ex1` arm — the scalar exact-width kernel, which
*is* dtype-generic and instantiates `case 1` in both branches — should help float64 k=1 most of
anything in either chain.

## The accumulator lever, measured on the row kernel directly: it needs L1 and it needs degree

Before spending a grid on it, the row kernel was swept standalone on the ARM host —
single-threaded, the kernel text extracted verbatim from `kernels.h`, four B footprints × eight
degrees × ACC ∈ {1,2,4,8} × both dtypes. Single-threaded on purpose: threading is a separate axis
and would hide this one.

Speedup of ACC over one chain (ARM, ns/nonzero at ACC=1 in the first column):

| B footprint | deg 4 | 16 | 64 | 128 | 512 | 2048 | ns/nnz @1 (deg 2048) |
|---|---|---|---|---|---|---|---|
| **4.5 KB (L1)** f32 | 1.05 | 1.12 | 1.15 | **1.30** | **1.86** | **2.31** | 0.380 |
| 256 KB (L2) f32 | 0.95 | 1.05 | 1.05 | 1.05 | 1.04 | 1.09 | 0.402 |
| 4 MB (SLC) f32 | 0.97 | 1.01 | 0.97 | 1.01 | 1.02 | 1.00 | 0.552 |
| 16 MB f32 | 0.94 | 1.08 | 1.06 | 1.04 | 1.04 | 1.04 | 0.662 |
| **9 KB (L1)** f64 | 0.85 | 1.02 | 1.04 | **1.22** | **1.81** | **2.21** | 0.386 |
| 256 KB (L2) f64 | 0.95 | 1.01 | 1.00 | 1.03 | 0.94 | 0.98 | 0.426 |

(columns are the best of ACC=4 and ACC=8)

Three things, and the dtypes agree on all of them:

1. **The gain is confined to an L1-resident B.** At 256 KB it is already gone — 1.01 to 1.10,
   inside the run-to-run spread — and past L2 there is nothing. So this is not a general SpMV
   improvement; it is a fix for one regime.
2. **It needs degree.** At degree ≤ 64 it is 1.02–1.15 even in L1, and at degree 4 with eight
   chains it is a *regression* (0.85–1.05): the cross-chain sum and the remainder loop cost more
   than the chains save on a four-element row.
3. **ACC=2 buys nothing anywhere** (0.94–1.06). Four chains capture most of it and eight add a
   little at the top. Whatever the dependency structure is, two chains do not break it.

So the rule this implies is `B fits L1 AND degree ≥ 128`, which is exactly the
"gate it behind a condition that provably cannot fire on the shapes it would hurt" shape — and
the shapes it would hurt (degree ≤ 8) are excluded by the same condition.

**It matches the losing family.** The worst k=1 cell is a DLMC layer with 256 rows, degree 1152,
and a column count that puts B at about 4.6 KB — L1-resident, degree well over 128, so squarely
in the gate, with 1.86x predicted. That would take its 30.0 µs to about 16 µs against MKL's 22.6,
i.e. 0.755 → ~1.4.

**A sharper prediction for `rw_chain42.sh`, which measures a different kernel by the same
mechanism.** Those k=1 cells run the nonzero-axis *gather* kernel, not this scalar loop, and its
single accumulator is the same defect. So the streams arms should move by roughly this much —
1.3x to 1.9x — on the high-degree, few-column matrices, and by nothing on the rest. `s2` should
be flat, since two chains buy nothing here either.

**The first version of this sweep measured nothing and printed a full table of `0.000` ns with
`inf` ratios.** Every rep wrote the same values to the same output array, so the compiler kept
only the last one. The fix is a barrier per rep plus a refusal to print any cell whose fastest
arm ran under 5 ms — a table of zeros reads like a result, and "measured nothing" has to say so.

## The half-vector kernel: a float32 win at k=4, and a float64 loss at k=2

`rw_chain36.sh`, 1510 cells per dtype over the 302-matrix corpus, arms `ref`/`refb` (both
`HALFVEC=0, EXACT_HI=3`), `h1` (`HALFVEC=1`), `h2` (`HALFVEC=2`), `s0` (`HALFVEC=0, EXACT_HI=1`),
`hv` (`HALFVEC=2, EXACT_HI=1`). The kernel's exact width is k=4 for float and k=2 for double, so
each dtype has one firing width and four structural nulls.

**float32, at k=4 where it fires** — `h1` **1.1008 (z +14.6)** against a `refb` floor of 0.9945,
and `h2` adds nothing over `h1` (1.0997), so the simpler gate — take only the exact half-vector
width, not the widths below it — is the whole effect. By group:

| group | h1 | | group | h1 |
|---|---|---|---|---|
| nk_l2band (80) | **1.2143** (z +40.6) | | d_sp (40) | 1.0684 |
| d_ss (4) | 1.1463 | | d_mid (39) | 1.0564 |
| nk_shortrow (40) | 1.1305 | | d_nd (39) | 1.0446 |
| nk_inert (30) | 1.0408 | | nk_winning (30) | 1.0113 |

Against MKL at k=4 float32: kernel parity **1.2880 → 1.4179**, cells below MKL **125/302 → 70**;
whole call 0.9202 → 0.9940, below 192 → 148. At the four widths where it cannot fire it reads
0.9913–0.9955, which is the arm being charged for naming a variable, not a kernel effect.

**float64, at k=2 where it fires** — `h1` **0.9646 (z −10.7)**. Every group is negative except
nk_shortrow (1.0353); nk_l2band is 0.9528 and d_ss 0.8451. Against MKL it is worse than that
sounds: `ref` at k=2 float64 has kernel parity 1.6631 with **0 of 302 cells below MKL**, and `h1`
takes that to 1.6042 with **21 below**. It creates below-MKL cells where there were none.

**So this ships for float32 only.** 128-bit registers are the right shape for a four-lane float
row and the wrong shape for a two-lane double row, where the halved register width costs more
than the halved mask waste saves. The gate has to be stated per dtype.

**Two things the control arms gave away for free.** `s0` turns the exact-width kernel off
(`EXACT_HI=1` empties the band): at k=4, 5 and 8 it reads 1.0000 / 0.9987 / 1.0003, which is a
clean structural null confirming the arm machinery; at k=2 and 3, where the exact kernel does
ship, it reads 0.9191 and 0.9084 on float32 and **0.8647** and 0.9606 on float64 — so the
exact-width kernel already in the default is worth 8–9% at float32 k=2/3 and 13.5% at float64
k=2. And `hv` (half-vector on, exact off) reads 1.0944 at k=4 float32 against `h1`'s 1.1008, so
the half-vector win does not depend on the exact kernel being there.

## The accumulator lever through the real call path, on the second host

`arm_spmv_ladder.py` on the ARM host — `scorch.matmul(A, x)` → `ops.spmv` →
`prebuilt_spmv_csr_*`, so threading, allocation and dispatch are all inside the measurement,
unlike the standalone sweep. Synthetic shapes so B's footprint and the degree move independently,
which a corpus cannot do. Best of ACC=4 and ACC=8, against a `refb` floor that runs ±2%:

| B | deg 8 | 64 | 256 | 1152 |
|---|---|---|---|---|
| 9 KB f32 | 1.04 | 1.02 | 1.19 | **1.47** |
| 32 KB f32 | 1.01 | 1.03 | 1.19 | **1.45** |
| 256 KB f32 | 1.05 | 1.01 | 1.01 | 1.03 |
| 4 MB f32 | 1.00 | 1.00 | 1.00 | 1.00 |
| 18 KB f64 | 1.04 | 1.03 | 1.12 | **1.29** |
| 64 KB f64 | 1.00 | 1.02 | 1.12 | **1.27** |
| 512 KB f64 | 1.00 | 1.01 | 1.00 | 0.99 |
| 8 MB f64 | 1.02 | 1.00 | 1.01 | 1.01 |

Same gate as the standalone sweep, both dtypes, through the real call path: **small B and high
degree, nothing otherwise**, and ACC=2 buys nothing anywhere (0.97–1.02). The magnitudes are
lower than standalone (1.47 against 2.31) because the call now includes the parts ACC cannot
touch, and the footprint threshold is looser (64 KB rather than 4.5 KB) because six workers each
have their own L2.

## chain37 was void on its first run, and its own cross-check is what said so

The build-attribution chain carried MKL as a control on the grounds that it enters none of our
code and must therefore read the same in both builds. It did not: **median spread 1.185, p90
4.14, max 7.83** over 136 cells, with the within-build A/A at ±4–5% against the ~0.03% kprobe
normally reaches. So the run could not attribute a 3.4x effect and nothing in its table meant
anything — a 17-matrix corpus with a single arm makes `per_rep` tiny, so every cell got a tiny
batch, and twelve short-lived processes never let the host settle.

Re-running with `--reps 25 --target-ms 400 --batch-ms 20 --settle 4`, and the analyzer now prints
a **VOID** banner and says nothing below it can be read whenever the median MKL spread exceeds
1.05, rather than leaving that inference to whoever reads the table.

### The two threads are the same lever, and that sharpens the prediction

`rw_chain42.sh`'s `ex1` arm routes k=1 to `scorch_spmm_row_narrow_exact<T, K=1, UNROLL>`. With
`NARROWK_EXACT_UNROLL` at 4 and `DEGUNROLL`, `SHORT` and `ACCUM` all defaulting to 0 — so nothing
clamps the unroll — that instantiation is **four independent scalar accumulator chains over a
k=1 row**. Which is precisely the `ACC=4` configuration the SpMV sweep measured, on precisely the
shape family the sweep says it pays on.

So the SpMV work and the k=1 SpMM work are one mechanism reached by two paths, and the ARM
numbers become a quantitative prediction for an x86 grid that has not run yet:

- `ex1` wins at k=1 by roughly **1.2x to 1.5x** on the matrices with a small B and a high degree,
  and by nothing on the rest. The corpus is banded by degree for exactly this reason.
- It wins on **both dtypes**, because the exact-width kernel is templated on the scalar type and
  instantiates `case 1` in both dispatch branches. This is the only arm in either chain that can
  reach the 34 float64 k=1 cells, since `narrowk_gather` is gated on
  `is_same<scalar_t, float>` and those cells currently run the register-block kernel at one
  useful lane in four.
- `s2` and the two-chain rungs stay flat, because two chains bought nothing anywhere in either
  sweep (0.94–1.06 across 60 cells).

If `ex1` moves and the streams arms do not, the mechanism is the accumulator chain and not the
gather; if the streams arms move and `ex1` does not, it is the gather's outstanding-load count
and the two are separable after all. Either way the answer is attributable, which is the point of
giving them separate arms in one interleaved grid.

### What the half-vector flip still owes, and why it is not staged yet

The flip is committed but deliberately **not** staged to the measurement host. Three queued runs
read the width it changes, and one of them would refuse:

- **chain39** copies the instrumented tree and then *checks* that eight shipped defaults are what
  it expects, `SCORCH_SPMM_HALFVEC 0` among them. The flip deletes that definition, so chain39
  would refuse — the guard working exactly as intended. Its own comment says the run measures
  "what ships today, which is the baseline the two candidates will be judged against", so it
  should stay the pre-flip baseline and the post-flip caller-path number is a separate run.
- **chain40** is the emission gate for the kernel's *addition* with the default off. Flipping the
  default under it turns a "this is neutral" gate into a "this changes emission, as intended"
  gate, which is a different question.
- **chain42** reads k=4, so the flip would move its baseline mid-grid.

Two things still owed after those clear:

1. **x86 emission attribution per dtype.** ARM is settled — the object moves 10 instructions, all
   `__LINE__` constants shifted by exactly the net lines added, nothing else. On x86 the float32
   instantiation is *supposed* to change and the float64 one is not, and that needs a per-symbol
   diff of the two instantiations rather than a whole-object one. It needs a compile window on a
   host that is not timing.
2. **The post-flip caller-path scoreboard**, on chain39's corpus, widths and probe, so the
   difference is a difference and not two different measurements.

### The row partition reverses sign between the two hosts, and it is not a width effect

The ARM ladder was built to decide which part of the back-stealing partition costs the ARM
tail: mode 1 is home ranges with no stealing and therefore no cursor probing at all, mode 2
steals from the front, mode 3 is the candidate that steals from the back and whose workers
probe every other cursor once their own range is done. Twelve workers probing twelve cursors
on shared lines is of the order of a coherence miss each on a kernel fifteen to thirty
microseconds long, which made the probe the obvious suspect. The grid pre-registered the
fork: if mode 1 recovers the tail the probe is the cost and can be bounded, and if mode 1 is
as bad then the static decomposition is, which tuning cannot fix.

**Mode 1 did not recover the tail.** ARM, 420 cells per dtype, arm-to-arm speedups against
the candidate, with the released atomic counter (`p0`) as the thing a release has to beat:

| arm | what it is | f32 | f64 |
|---|---|---|---|
| p3e | back-stealing, the candidate | 1.0000 | 1.0000 |
| p1e | home ranges, no stealing, no probing | 0.9826 | 0.9622 |
| p2e | steals from the front | 0.9682 | 0.9592 |
| **p0** | **released global atomic counter** | **1.0303** | **1.0222** |
| aa | same code as p3e | 0.9951 | 1.0006 |

Removing the probe makes it *worse*, not better. So the cost is the static decomposition, and
by the rule set before the run it is not fixable by tuning the stealing policy. On ARM the
released counter wins at every width measured from k=1 to k=8 — 1.0581, 1.0372, 1.0313,
1.0719, 1.0187 in float32 — and the candidate only wins at k=64 (1.0330). 95 of 420 float32
cells have the candidate more than 10% slower than what ships; the same-code arm reports 99,
so that count is the code and not a slot.

**The obvious inference from that is wrong.** A k≤8 reversal invites a width gate, so I
resolved the already-collected x86 grid by width before writing any. There is no crossover
on x86: the candidate is faster than the counter at *every* width, in both dtypes.

| k | f32 candidate over counter | f64 | f32 same-code floor |
|---|---|---|---|
| 1 | 1.2666 | 1.2975 | 0.9998 |
| 2 | 1.3868 | 1.3781 | (1.0882 — see below) |
| 3 | 1.3501 | 1.3375 | (1.0617) |
| 4 | 1.2807 | 1.3000 | 0.9977 |
| 8 | 1.3243 | 1.3298 | 0.9999 |
| 64 | 1.3122 | 1.2389 | 1.0005 |
| 256 | 1.1833 | 1.1243 | 0.9982 |

A width gate would therefore hand back a 1.27x x86 win at exactly k=1, the width that holds
the most below-MKL cells. The reversal is host-specific, not width-specific, and the
mechanism fits: the partition's whole value is keeping A resident across calls, which is
worth 1.3x on a host achieving ~56 GB/s and negative on one achieving ~412 GB/s. A
bandwidth-scarcity benefit has no single value across hosts, so the honest home for this
decision is the host-calibrated selector, not a compile-time default — and until it has one,
compiling `SCORCH_SPMM_PARTITION_DEFAULT 3` in for both hosts is a 2-3% ARM regression that
should be reported rather than shipped quietly.

**The aa column is two different things in the same table, which is what makes it useful.**
It duplicates the first arm, `p3`, not `p3e`, and those differ by the exact-width narrow-k
kernel. The dispatch serves widths 2 and 3 only, so at k=1, 4, 8, 64 and 256 the kernel
cannot fire and the column is a genuine same-code floor: 0.9977 to 1.0005 across five
widths, on the same matrices in the same run. At k=2 and k=3 it stops being a floor and
becomes a measurement — 1.0882 and 1.0617 — which independently reproduces the 6-8% at k=2
and k=3 that the policy header already records for that kernel. The partition effect is two
orders of magnitude above that floor.

### Which cells are actually below MKL: float32, narrow k

Same grid, 362 cells per width, counting rather than averaging:

| k | f32 cells below MKL | f32 pooled | f64 cells below | f64 pooled |
|---|---|---|---|---|
| 1 | **189 / 362** | **0.9379** | 57 | 1.2948 |
| 2 | 172 | 1.0329 | 23 | 1.4737 |
| 3 | 165 | 1.0852 | 16 | 1.5576 |
| 4 | 165 | 1.0891 | 17 | 1.5877 |
| 8 | 105 | 1.3212 | 8 | 1.8287 |
| 64 | 36 | 1.8240 | 6 | 2.2726 |
| 256 | 21 | 2.1469 | 7 | 2.4040 |

float32 k=1 is the only width that pools below parity anywhere in the grid, and float64 never
pools below 1.29. The absolute level here is a hooked build and therefore is exactly what
chain37 is settling, but the *ordering* by width is a within-build comparison and survives
whatever chain37 says. The target is float32 at k≤4, worst at k=1 — which is the width the
gather kernel claims (`narrowk_gather = (B1_size == 1)`) and which the exact-width dispatch
clamps itself out of, since it serves 2..3. chain42's `ex1` arm forces exactly that width
onto the exact-width kernel, so it is aimed at this cell and not at a shape I picked after
seeing it.

### chain37: the instrument costs 1.10x, not 1.5x, and the proof is a paired null

The session had two grids from the same tree on the same host disagreeing 1.545x about our
own kernel while agreeing on MKL to 1%. chain37 settles it: one corpus, one harness process,
alternating between the instrumented build and a hookless build with the candidate shipping
configuration compiled in, three rotations with the order flipped in the middle one.

**The charge is additive.** Regressing the hooked-minus-hookless delta on hookless kernel
time gives a slope of **-0.022**, where 0 is purely additive and 1 purely proportional.
Median delta 1.2 us, IQR 0.8-2.7, over kernels spanning 4.1 to 916 us. So it is a fixed
per-call cost: 15-17% of an 8 us kernel, and inside the noise of a 100 us one.

**The first gate I wrote for it was the wrong statistic, and the guard I thought I had
was not there.** The analyzer prints MKL's spread as `max/min` over every reading for a
cell, which with two builds times three rotations is a max-over-six range — a far wider
statistic than the comparison the parity table actually makes, which is median(build A)
against median(build B). It read 1.0712 median, above the 1.05 threshold I had recorded as
enforced; grepping the file shows no such check was ever added, so the run analyzed itself
anyway and nothing announced that.

The right control is MKL's own **build-to-build median ratio**, computed with the identical
estimator used on our kernel. MKL is timed through `torch.sparse.mm` and enters neither
build, so that ratio is a pure null.

| group | median hookless kernel | MKL null | our ratio | null-corrected charge |
|---|---|---|---|---|
| agree f32 / f64 | 8.4 / 8.3 us | 0.9959 / 0.9914 | 1.1326 / 1.1263 | **1.1372 / 1.1361** |
| dis_big f32 / f64 | 23.9 / 23.0 us | 0.9984 / 1.0023 | 1.0963 / 1.1243 | **1.0982 / 1.1218** |
| dis_mid f32 / f64 | 72.8 / 118.2 us | 1.0180 / 1.0092 | 1.0164 / 0.9753 | 0.9984 / 0.9664 |
| pooled, 136 cells | | **0.9995** | 1.0979 | **1.0985** |

The paired null is 0.9995 pooled and 1.7% median per cell — not the 7.1% the max/min
statistic claimed — and it is 189x smaller than the effect. So: **the instrumented grids
understate our parity by 1.10x on kernels below about 25 microseconds and by nothing above
it.** The two-grid 1.545x disagreement is therefore not the build. The build is 1.10x of it.
The rest is corpus composition, and this one grid makes that obvious on its own: parity
ranges from 0.97 to 3.47 *across groups inside it*.

### The scoreboard, corrected: float32 is the whole remaining problem

Applying that measured 1.0985x to our time only, per cell, and only where the kernel is
below the 25 us knee where chain37 measured it — then counting rather than averaging:

| k | f32 raw / corrected parity | f32 cells <MKL raw / corr / <0.95 | f64 corrected parity | f64 <MKL corr / <0.95 |
|---|---|---|---|---|
| 1 | 0.9379 / **1.0210** | 189 / 175 / **164** | 1.4040 | 19 / 10 |
| 2 | 1.0329 / 1.1261 | 172 / 150 / 139 | 1.6038 | 8 / 2 |
| 3 | 1.0852 / 1.1801 | 165 / 150 / 136 | 1.6942 | 5 / **0** |
| 4 | 1.0891 / 1.1831 | 165 / 145 / 100 | 1.7252 | 7 / **0** |
| 8 | 1.3212 / 1.4360 | 105 / 92 / 81 | 1.9819 | 1 / 1 |
| 64 | 1.8240 / 1.9605 | 36 / 24 / 12 | 2.3832 | 2 / 1 |
| 256 | 2.1469 / 2.2182 | 21 / 13 / 9 | 2.4506 | 5 / 3 |
| all | 1.2894 / 1.3900 | 853 / 749 / **641** | **1.8571** | **47 / 17** |

**float64 is essentially done**: 47 of 2534 cells below MKL, 17 below 0.95, and two widths
with none at all. The correction cleared 38 of its 57 raw losses, which is what a marginal
deficit looks like.

**float32 is not**: 749 below, 641 below 0.95, and at k=1 the correction cleared only 14 of
189 — so those are not marginal cells being flattered by the instrument, they are losing by
more than ten percent. This is the first statement of the target that survives knowing what
the instrument costs, and it is narrower than any earlier one: not "few rows and high
degree", but **float32, worst at k=1 and material through k=8**.

That width in float32 is served by `scorch_spmm_row_gather_f32<1>` with `narrowk_streams = 1`
— a single accumulator — and the kernel's own comment already records both that it sits at
0.762 of MKL there and why: "the profile is memory LATENCY, not FMA throughput, and MKL's
SpMV-shaped loop keeps eight or more loads in flight". chain42 is aimed exactly at that, with
`--pad-env` and a duplicate-environment `refb` arm so the environment charge cannot be
mistaken for the effect.

### The float32 deficit has one mechanism, and dtype scaling proves it without any parity number

Corrected parity is 1.02-2.22 in float32 and 1.40-2.45 in float64, so the float64 kernel is
about 1.4x further ahead of MKL at narrow k. Both dtypes run structurally the same kernel and
float32 moves half the bytes, so that has exactly two explanations pointing opposite ways:
MKL's float32 kernel exploits the halved footprint better than its float64 one, or ours
exploits it worse than ours does. The discriminator is the absolute dtype speedup each
implementation gets on the *same* cell, time(f64)/time(f32), where 2.0 is perfect scaling
with the bytes and 1.0 is float32 buying nothing.

This is a within-build, within-run, same-matrix comparison, so none of the session's
disputed absolute parity numbers enter it and the instrument question is irrelevant to it.
Our side uses kernel time, because the ~4.2 us per-call floor is dtype-independent and wall
time would dilute the ratio toward 1.0 — the very direction of the answer. MKL has no
kernel-time column and never will; it keeps wall time, and since its own floor is the larger
one (~12.8 us) that dilution biases *against* the reading below rather than for it.

| k | ours (kernel) | MKL (wall) | MKL, floor-subtracted | gap |
|---|---|---|---|---|
| 1 | **1.0669** | 1.4059 | 2.0498 (n=39) | 1.32 |
| 2 | **1.0087** | 1.3988 | 1.7466 (n=63) | 1.39 |
| 3 | **1.0256** | 1.4244 | 1.6682 (n=83) | 1.39 |
| 4 | **0.9984** | 1.4239 | 1.6763 (n=89) | 1.43 |
| 8 | **1.0647** | 1.4187 | 1.5853 (n=173) | 1.33 |
| 64 | 1.4591 | 1.6846 | 1.9920 (n=275) | 1.15 |
| 256 | **2.1326** | 2.2353 | 2.5398 (n=322) | 1.05 |

**Our narrow-k float32 kernel takes the same time as our float64 one.** Halving the element
size buys between 0.998x and 1.067x at every width through k=8. MKL gets 1.41x on wall time
there and 1.59-2.05x once its floor is subtracted. At k=256 ours scales 2.13x, so the kernel
*can* be throughput-bound — at narrow k it simply is not.

A loop that gets nothing from halving its bytes is not bandwidth-bound; it is latency-bound,
because a load costs the same cycles whether it moves four bytes or eight. So this
independently confirms, and much more sharply, what the multi-stream kernel's own comment
already asserted from a profile: "the profile is memory LATENCY, not FMA throughput, and
MKL's SpMV-shaped loop keeps eight or more loads in flight."

It also reframes the float64 result. We are not 1.86x ahead of MKL at float64 because our
float64 kernel is good; we are dtype-blind, and MKL's float64 pays for its bytes while ours
does not. The float64 lead is MKL's float64 disadvantage. Fixing the latency bound therefore
widens both dtypes, and the honest statement of the remaining work is one mechanism, not two
dtype problems.

**The prediction, registered before chain42 reports.** The fix is memory-level parallelism —
more independent loads in flight — not vectorization and not blocking. If our narrow-k
float32 reaches MKL's degree of throughput-boundness it gets up to 1.4x faster, taking
corrected float32 k=1 parity from 1.021 to about 1.43 and clearing most of the 164 cells
below 0.95. chain42's `s2`/`s4`/`s8` arms are exactly that intervention.

**And a discriminator for chain42's analysis, so the two candidate sub-mechanisms do not get
conflated.** The multi-stream kernel's comment predicts its win is largest where B is *not*
L1-resident, because what it covers is L3 latency. My standalone scalar accumulator ladder
found its 2.31x where B *was* L1-resident, because what it covered was the FMA dependency
chain. Those are different causes with different gates, so chain42 must be binned on B
footprint rather than pooled: a win concentrated in the large-B cells is the latency story,
a win concentrated in the small-B cells is the dependency-chain story, and a flat win across
both means neither explanation is complete.

### The analyzer for the key run would have crashed on its primary case

Before chain42 could depend on it, I ran `an_streams.py` on a synthetic file with the right
columns and arm names — a smoke test for crashes, not for numbers. It raised on line 75:

    if a != "ex1" and (dt == "float64" or (ki in INST and int(a[1:]) not in INST[ki])):

`ARMS[1:]` includes `refb`, the floor arm, which is not a (K,S) pair, so `int(a[1:])`
evaluates `int("efb")`. On float64 the `dt == "float64"` disjunct short-circuits before
reaching it, which is why nothing had caught it; on float32 it raises at every k in `INST`,
which is every width where the arms are live. chain42's float32 verdict — the run that tests
the fix for the one identified mechanism — would have been a traceback.

Two things were added to the same file while it was open, both required by the finding above:
a banding of k=1 on **B footprint** rather than degree, because the losing family is
few-rows-and-high-degree and therefore has high degree and a small B simultaneously, so a
degree banding alone cannot separate the L3-latency story from the dependency-chain story;
and a **dtype-scaling** pass across the two dtype files, printing whether each arm moves
f64/f32 off 1.0 toward MKL's 1.41. Since the multi-stream dispatch sits inside an
`if constexpr (is_same<scalar_t, float>)`, float64 is a null for `s2`/`s4`/`s8` and any
movement in that ratio comes from the float32 side alone — which is exactly what a fix for a
latency bound should produce, and it is measurable without any absolute parity number.

The synthetic file was deleted rather than left in the scratch directory, so it cannot later
be mistaken for a measurement.
