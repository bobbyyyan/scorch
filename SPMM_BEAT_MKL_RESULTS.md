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
