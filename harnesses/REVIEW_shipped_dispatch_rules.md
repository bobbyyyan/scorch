# Review response: the four shipped dispatch-rule concerns

A reviewing agent raised five concerns about shipped code. #5 (`train_autotune_model.py`
exporting an in-sample figure as `heldout_geomean`) is confirmed by reading and is a labelling
defect; it is tracked separately. This file is the measurement log for #1–#4, which are claims
about dispatch rules I did not change and had not measured.

**Standing rule for this file: a concern is not a defect until it has a size and a null.** Each
entry records what was measured, the null it was measured against, and whether the concern's
*stated consequence* survived — which is not the same question as whether it pointed at real code.

---

## #4 — the tiling plan cache omits thread count (`tiling.py:428`, `:1517`)

**The concern.** `maybe_dispatch`'s memo key is `(signature, level, baseline_tag)` where
`_signature` is `(M, J, nnz, N, 4 sampled index entries)`. `nthreads` reaches the probe
(`tiling.py:1396`) and the dispatch (`:1528`) but is not in the key, so a plan probed at one
thread count is replayed at every other thread count for the same matrix and free dim. The
reviewer paired this with the "never slower by construction" claim at `ops.py:781`.

**The code reads as described.** Verified: `memo_key = (sig, level, baseline_tag)` at
`tiling.py:1523`; `nt = nthreads if nthreads is not None else -1` at `:1517` is passed to
`_dispatch_decision` and never enters the key. `torch.set_num_threads` between calls reaches this
path (the drop-in SpMM follows `torch.get_num_threads()`), so two thread counts in one process is
reachable rather than hypothetical.

**First measurement, and why it was not the answer.** Probing the same cell at nthreads ∈
{2,4,6,8} with the memo cleared between, on three DLMC matrices × N ∈ {1024, 2048} (M5, 16 MB LLC;
at N ≤ 512 **no** DLMC matrix is eligible — `J·4·N > C` fails — so the instrument refused a verdict
on the first run rather than printing six reassuring "STABLE" rows):

| cell | nt=2 | nt=4 | nt=6 | nt=8 | |
|---|---|---|---|---|---|
| `bottleneck_2_block_group4_1_1`, N=1024 | tilej/1024 | tilej/1024 | **tilej/512** | tilej/1024 | moves |
| same, N=2048 | tilej/512 | tilej/512 | **tilej/1024** | tilej/512 | moves |
| `bottleneck_2_block_group4_2_1`, N=1024 | tilej/1024 | tilej/1024 | tilej/1024 | tilej/1024 | stable |
| same, N=2048 | tilej/512 | tilej/512 | tilej/512 | tilej/512 | stable |
| `..._projection_block_group3`, N=2048 | v2 | v2 | v2 | v2 | stable |

2 of 5 eligible cells moved, and the **kind** never moved — only the panel width.

**The null, which overturns that reading — and a second run that overturns my reading of the
null.** A pick that moves across thread counts is evidence of thread dependence only if it is
stable at a *fixed* thread count. Repeating the probe 8 times at each of nt=2 and nt=6:

| run | cell | nt=2 | nt=6 | majority agrees? |
|---|---|---|---|---|
| first | N=1024 | tilej/1024 ×8 — **0/7 transitions** | 1024 ×6, 2048 ×2 — 2/7 | yes (1024) |
| first | N=2048 | tilej/512 ×8 — **0/7 transitions** | 512 ×7, 256 ×1 — 2/7 | yes (512) |
| second | N=1024 | 1024,1024,512,512,1024,512,1024,512 — **5/7 = 0.71** | 3/7 = 0.43 | yes (1024) |
| second | N=2048 | 256,512,256,256,256,512,512,256 — **4/7 = 0.57** | 4/7 = 0.57 | yes (**256**) |

**Three corrections to what the first run let me write, two of them structural.**

- The first version of this entry compared raw **counts** (`within >= across`) across different
  denominators — 14 adjacent pairs against 8 matched-index pairs — and printed "does not exceed"
  for the N=1024 cell where the *rates* are 0.25 against 0.14. The claim was false on that cell and
  the published range "0.12–0.25" hid it at the top end.
- **The two rates were never the same statistic.** Within-condition counts adjacent transitions;
  across-condition counts matched-index disagreements. For a minority set of size *m* in *r*
  contiguous interior runs the numerators are 2*r* and *m*, so when both describe the same events
  their ratio is the denominator ratio and nothing else — on N=1024, 0.25/0.14 is exactly 14/8.
  **The ratio is withdrawn**; it could not have carried information either way.
- **The first run's pooled null averaged a perfectly stable condition with the condition under
  test.** nt=2 was 0/7 on both cells and every switch was at nt=6, so the pooled 0.14 was the mean
  of 0.00 and 0.29 — half its mass was the quantity being explained away, which makes the
  comparison against nt=6 circular and leaves the comparison against nt=2 with no null at all.
  A within-condition null is reported **per condition** when the conditions are the arms of the
  contrast.

**And the second run kills the explanation I had built on the first.** nt=2 is not stable: it is
*more* unstable than nt=6 on the N=1024 cell (0.71 against 0.43) and equal on N=2048. So the
"instability appears at nt=6, which is the M5's fast-core count" reading — the hybrid-core
alternative I had flagged as live — **is dead**, because the first run's nt=2 stability was a draw
and not a property. The N=2048 majority pick also moved between runs, 512 to 256, so even the
surviving leg agrees *within* a run without the majority itself being reproducible *across* runs.

**What is left is simpler than the concern and more actionable.** The pick is substantially
irreproducible at fixed configuration — 0.43 to 0.71 adjacent-transition rate over 8 repeats, two
distinct widths in every condition of every run — and the memo freezes whichever width won the
first call for the process. **Thread count is not the variable.** Adding `nthreads` to the memo key
would partition the cache without changing a decision, because there is no thread-dependent
decision to protect; the defect is the probe's width resolution.

**The structural form of that claim, which holds on every host rather than on the four thread
counts swept.** The sweep establishes equivalence over the levels and the one host it covered. A
census of which decision signatures *accept* a thread count settles it everywhere:

    _signature(a, N)            _panel_width(N, C)
    _eligible(J, nnz, N, C)     _eligible_learned(J, N, C)

None takes a thread argument, and parsing their bodies gives **zero `global` statements and zero
`os.*` reads in all four** -- so the dependence cannot enter through a module global or an
environment read either, which is the residual a signature census alone would leave open. `C` comes
from `query_llc()` (`:358-405`): it reads `SCORCH_LLC_BYTES`, `platform.system()`, sysctl and
sysfs, caches into `_llc_bytes`, and is the **whole** LLC, never divided by a worker count. `nt` is
`nthreads if nthreads is not None else -1` (`:1517`) and reaches `_dispatch_decision`,
`_tilej_args`/`_tileijk_args` and `baseline_fn(nthreads)` -- the executor and the caller's
baseline, never a decision.

**The one thread-related read in the file, named rather than left implicit.** `os.cpu_count()`
occurs exactly once, at `:883` inside `_machine_id()`. That is a host-identity hash selecting
*which* learned spec loads, not a per-call thread count entering a per-call decision, so it leaves
eligibility and panel width thread-independent. It is the only hit in the file for
`get_num_threads|cpu_count|OMP_NUM_THREADS|sched_getaffinity`.

The census is the stronger half, and the sweep is still worth having: neither a signature list nor
a body parse can see a dependence entering below the pybind boundary in C++, which only measurement
reaches. The structural claim is the peer reviewer's; the body parse, the `query_llc` reading and
the `cpu_count` exception are mine.

### The census's own residual: the memo key is coarser than the functions it protects

The statement above is about the thread count, which turns out not to be a decision input at all.
The hazard concern #4 actually audits is more general: **a decision input that can differ between
two calls in one process without changing the memo key.** `memo_key = (sig, level, baseline_tag)`
(`:1523`). Taking the transitive closure of calls from `is_candidate` and `maybe_dispatch` -- 40
functions, not the 4 either census listed, which is why the closure was computed rather than
enumerated -- every decision input classifies as:

| decision input | can differ between two calls in one process? | captured in the key? |
|---|---|---|
| M, J, nnz, N | yes | yes -- `_signature` returns `(M, J, nnz, N, 4 samples)` (`:428-453`) |
| level | yes -- `_current_level` reads `_global_level` / `_tls` | yes, keyed directly |
| baseline_tag | yes | yes, keyed directly |
| `C` = `query_llc()` | no -- memoised into `_llc_bytes` on first call | n/a |
| 8 module-level env reads (`:342,348,456,1038,1043,1046,1047,1061`) | no -- evaluated at import | n/a |
| `os.cpu_count()` via `_machine_id()` (`:883`) | no -- host property, cached in `_machine_id_val` | n/a |
| matrix **content** via `_locality_ratio` / `_degree_cv` | yes | **only through 4 sampled entries** |

Every row but the last closes by reading. The last does not: **the key samples the matrix at 4
points (`pos[M/3]`, `pos[2M/3]`, `crd[nnz/3]`, `crd[2nnz/3]`) while the decisions sample it at 64
(`_LOC_NSAMP`, reaching `_scattered`) and 4096 (`_CV_NSAMP`, reaching `_degree_cv`).** The key is
coarser than the functions it protects, so two matrices can share a key and decide differently.
Both samplers use a fixed seed (`manual_seed(0)` / `manual_seed(1)`), so equal M draws the same
rows and only the values read differ.

**Measured on the real collection, locally, with neither host touched.** Reproducing `_signature`
over all 3846 local DLMC matrices, then calling the *production* `_locality_ratio` / `_scattered` /
`_degree_cv` on every surviving collision group (import resolved to
`/Users/bobby/scorch-beat-mkl/src/scorch`, `_LOC_MIN=0.3`, `_LOC_NSAMP=64`, `_CV_NSAMP=4096`):

| | keys | distinct | groups > 1 | keys in a collision group |
|---|---|---|---|---|
| `(M, J, nnz)` alone -- the bound | 3846 | 1860 | 202 | **2188 (56.89%)** |
| full production signature | 3846 | 3823 | 3 | **26 (0.68%)** |

The 4 sampled entries separate 2162 of the 2188, so **the dimension-only bound overstates the
collision population by 84x** -- reported beside the exact figure rather than instead of it.

DLMC's largest bound-group is **144 keys at `(512, 512, 26214)`** -- larger than SuiteSparse's 94,
and on the corpus where the exact figure is available, so both extrema are named rather than
whichever one reads smaller.

**And the 26 are not a defect.** All three groups carry `nnz = 0`: they are exactly the 26 DLMC
keys with no nonzeros (`chain209-dlmc-dims/dlmc_dims.csv`, sealed digest `ff167f71d50abf74`,
manifest 9 of 9 OK) -- variational_dropout 0.98 layers pruned away entirely. Being all-zero they
are *identical* as CSR operands: `_locality_ratio` 0.0, `_scattered` False and `_degree_cv` 0.0 for
all 26, ranges exactly 0.000000. **Replaying one's memo entry for another therefore returns the
right answer rather than a misroute, and that is the reason the verdict rests on.**

**The floor of this result is forced by the sealed census; the ceiling is the measurement's alone.**
`chain209-dlmc-dims` carries only `key, rows, cols, nnz`, and within it the 26 `nnz == 0` keys fall
in exactly 3 `(M, J, nnz)` groups of sizes 5, 5 and 16. An all-zero CSR has an all-zero `indptr`
and an empty `indices`, so the 4 sampled entries are identical within a group and the dimensions
differ across groups -- those 3 groups and 26 keys **must** collide at the full signature, with no
access to the arrays. What needs the run is only that *nothing else* collides. Stating the
decomposition matters because an error in the separation would appear as **more** than 26, and the
sealed floor would still hold. Peer reviewer's observation.

**A correction to the second reason, which is narrower than first committed.** "They never reach
the memo" is true for the analytic levels and **false for the learned one.** `_degree_pays`
(`:483-490`, the expression at **`:490`**) is
`nnz / max(1, J) > max(_DEG_FLOOR, 2.0 * (J * 4 * N) / C)`; at `nnz = 0` the left side is exactly
0.0 and the right is `max(64.0, a term)`, so it is False **for every N, J and every C != 0** rather
than at the five widths I sampled -- a theorem, not a sweep, and the peer reviewer's point.

**The quantifier has one reachable exception, and it is not a False.** At `C = 0` the right side
divides by zero and `_degree_pays` **raises `ZeroDivisionError`**. `C = 0` is reachable without
touching the floor: `query_llc`'s env guard is `if env:` (`:366-369`) and the *string* `"0"` is
truthy, so `SCORCH_LLC_BYTES=0` memoises `C = 0`; `_operand_over_cache` is then `J*4*N > 0` =
True, so the call reaches the division. Verified: `query_llc() -> 0`,
`_operand_over_cache(512, 16384, 0) = True`, `_degree_pays -> ZeroDivisionError: float division by
zero`. A negative floor with negative `C` does return True, as earlier stated; `C = 0` reaches
neither True nor False. **A quantifier must range only over values where the expression is
defined**, and both the peer's "the second term is non-negative" and my transcription of it had the
same hole.

**And the learned conjunct is not structurally False.** `_eligible_learned(J, N, C)` is
`_operand_over_cache(J, N, C)`, which **reads no nnz at all**; my five N values simply all sat at
or below the cache boundary. The deciding quantity is the **product `J * 4 * N`**, so a sweep over
J and N separately reads as more evidence than it is -- of the three rows I first published, the
two True ones are the *same* product:

| J | N | `J * 4 * N` | | `_operand_over_cache` |
|---|---|---|---|---|
| 512 | 8192 | 16777216 | = **1.0000 C** | False -- by *equality*, which pins the boundary |
| 512 | 16384 | 33554432 | = 2.0000 C | True |
| 2048 | 4096 | 33554432 | = 2.0000 C | True -- **same product, not a second confirmation** |

Reduced to the quantity itself, at the tightest granularity the expression allows (`J = 1`, step 4):
product `C + 0` is False and `C + 4` is True, which is what shows the comparison is strict `>`
rather than something coarser. One premise under that pair: `J * 4 * N` over integer J and N is
always a multiple of 4, so step 4 *is* the finest granularity, **and the equality row exists only
when `C` is divisible by 4**. It holds for both C values on record (16777216 and the Linux fallback
37748736, both `% 4 == 0`), so the pair is valid as published; on a C not divisible by 4 there is no
equality row and the straddling pair is a different one. One thing `C = 16777216` does **not**
certify, so that no later entry quotes it that way: `query_llc()` has **four** macOS routes and
**three** of them return that exact value -- `hw.perflevel0.l2cachesize` succeeding; the Darwin
fallback `16 * 1024 * 1024` at `:402` after a raise or two empty reads; and a first non-empty key
returning `"0"`, which the inner `if out:` accepts as truthy bytes and the outer `val or` then
discards as a falsy int. Only `hw.l2cachesize` is distinguishable, at 8388608 -- which is also
`hw.perflevel1.l2cachesize`, the E-cluster L2, not the P-cluster cache the docstring calls binding.
Benign here precisely *because* three agree; the value is right either way, and it is quoted here
as a value and not as evidence the query ran. Worth one further clause, because it is the reason
neither reviewer predicted it from the `SCORCH_LLC_BYTES=0` case: **this function treats zero as
significant at `:368` (the string `"0"` is truthy, so C = 0 is accepted and crashes downstream) and
as absent at `:402` (the int `0` is falsy, so a measured zero is discarded)** -- same literal,
opposite handling, 34 lines apart. Peer reviewer's findings; both sysctl keys and both falsy tests
read here.

**So an all-zero operand does become a learned candidate once `J * 4 * N > C`, and that is the
DEFAULT, not an opt-in.** `_LEARNED_WIDEN = os.environ.get("SCORCH_AUTOTUNE_WIDEN", "1") == "1"`
(`:1061`), so the gate at `:555` --
`if level == "learned" and _LEARNED_WIDEN and _load_learned_model() is not None` -- needs no
configuration. My measured `is_candidate(level="learned") == True` also *proves a spec loads on
this host*, since the third conjunct had to be non-None for it to fire. **Which** spec is a
separate question the conjunct cannot answer -- `:1234` forces the loaded spec to be *for* this
machine, not to be one of the two tracked files, and the user-cache route is checked first -- so it
was read rather than inferred: `_machine_id()` is `74c1226957b8a5ce` and `_learned_model_path()`
returns `autotune-levels/models/model_74c1226957b8a5ce.json`, the first tracked spec, via the
second route. A conjunct firing identifies its population as non-empty, not which member.

**The conjunct that actually bounds it is the model, and that bound is width-independent.**
`_load_learned_model` rejects on `spec["machine_id"] != _machine_id()` (`:1234`) and `_machine_id()`
hashes `platform.system()` among its five components (`:883`), so no Darwin-built spec can ever
match a Linux host's id.

**But "both tracked specs are Darwin" is a statement about the git index, and the loader stats the
disk.** `_learned_model_path()` (`:1202-1216`) tries two routes in order:
`$XDG_CACHE_HOME`-or-`~/.cache` + `scorch/autotune_model_<mid>.json`, then
`<repo>/autotune-levels/models/model_<mid>.json` -- and `bench/train_autotune_model.py:307` writes
`model_{machine_id}.json` into `MODELS_DIR = REPO/autotune-levels/models` (`:42`), **the loader's
second route, same directory**. One trainer run on either Linux host would leave a resolving spec
that `git ls-files` would never show, and the error runs in the dangerous direction: the widened
gate would be live on a measurement host at every N, the opposite of what the sentence asserts.
Peer reviewer's finding, against their own earlier point and mine.

**So it was censused per host, on disk.** Only one thing writes route 1's filename pattern -- and
it is a *read*, at `tiling.py:1209`; a tree-wide grep for `autotune_model_` returns that one hit,
so route 1 requires hand placement.

| host | route 1 (`~/.cache/scorch/autotune_model_<mid>.json`) | route 2 (`<repo>/autotune-levels/models/`) |
|---|---|---|
| this Mac | absent (`~/.cache/scorch/` holds `autotune.json`, a different name) | the 2 tracked specs; `model_74c1226957b8a5ce.json` **resolves** |
| redwood | absent; `~/.cache/scorch/` holds only `autotune.json`; `XDG_CACHE_HOME` unset | **no `autotune-levels/` directory exists** in `/scratch/bobbyy/scorch`, `/scratch/bobbyy/mrhv/tree` or `/scratch/bobbyy/raisefloor/tree` |
| MKT | `~` is AFS and `~/.cache/scorch/` **does not exist**; `XDG_CACHE_HOME` unset | `/scr/bobbyy/scorch` holds **the same two tracked Darwin filenames**; the four `pfs/{ship,shpb,strm,nopf}` trees the measurement imports from have **no `autotune-levels/` at all** |

MKT's repo tree is therefore *populated* and still cannot resolve, because the two ids on disk are
Darwin ids and `_machine_id()` hashes `platform.system()`. No `model_<linux-id>.json` exists in any
tree on either host, so the trainer-run hazard has not occurred. The running job's own environment
(pid 4110182 on mkt1) carries no `SCORCH_AUTOTUNE*`, `XDG_CACHE*` or `SCORCH_LLC*` entry, so no
override is in play either, and `SCORCH_AUTOTUNE_MODEL` (`:1206-1207`) opens no hole in any case --
an explicit path still passes the machine_id check.

**On redwood and MKT, then, no spec resolves at any N**, `is_candidate` falls through to
`_degree_pays`, and the theorem makes it False. This sits *beside* the width caveat rather than
replacing it, because the width caveat is what covers a Darwin host that does have its own model.
The census is a property of two directories per host, not of the repo, so it is re-checkable with
`ls autotune-levels/models/` and `ls ~/.cache/scorch/` and should be re-run rather than carried.

**The width caveat is DLMC-specific, and SuiteSparse is ruled out by something stronger.** At
N = 32 the gate needs `cols > C/128` = 131072 at 16 MB or 294912 at the Linux 36 MB fallback. From
`chain209-dlmc-dims`: DLMC's maximum `cols` is **4608**
(`dlmc/rn50/extended_magnitude_pruning/0.8/bottleneck_2_block_group4_1_1`), so **0 of 3846** clear
it under either C, and the 26 `nnz == 0` keys top out at 2048 cols -- product 262144 at N = 32,
a factor of **64 below C**. That half needs no run either, with the margin named. On SuiteSparse,
though, **292 of 4584 keys clear `cols * 4 * 32 > C`** at 16 MB (165 at 36 MB; max cols 4690002 at
`ss/rajat31/rajat31`), so `N <= 32` is **not** a bound there. What rules the hazard out on
SuiteSparse is width-independent: the sealed census has **zero `nnz == 0` keys** on that corpus.

**Two shipped comments assert the opposite default, and they are not mine to amend.** `:552-554`
reads "DEFAULT: learned uses the analytic gate (no widening)" and `:1506-1509` reads "by default it
uses the analytic gate + the `_scattered` pre-gate below". The first sentence of each is true (the
condition *is* `_LEARNED_WIDEN`); the default sentence is false against `:1061`. The second
misdescribes two behaviours, because `:1555` is `if not _LEARNED_WIDEN and not _scattered(a, J)` --
under the real default the `_scattered` pre-exclusion is **skipped**, not applied. Both sit between
two correct statements of the same fact: the justification block at `:1052-1061` ends
"SCORCH_AUTOTUNE_WIDEN=0 reverts to the analytic gate", and `docs/source/user_guide/autotuning.md:177`
says "default is widened". **Queued for Bobby as the ninth item** -- this is production source, and
the choice between rewording the comments and flipping the default is his. Found by the peer
reviewer; verified here at all five sites.

**So the hazard has zero reachable instances on DLMC, for two independent reasons, and that is a
measurement rather than a reading.** Scope, exactly: exact on DLMC because all 3846 matrices are
local. On SuiteSparse it stays a **bound** -- 1274 of 4584 keys (27.8%) share `(M, J, nnz)` with
another key, largest group 94 at `(9801, 9801, 87025)` -- because the 4 sampled entries need the
matrices, which live on the two hosts, and one of them is mid-measurement.

**The cost, with the base named, because the base inverts the ranking.** Timing each pick by
seeding the memo and calling the production entry, two interleaved passes per pick, M5 at nt=6:

| cell | widths in play | span of runtime between them | same-pick pass spread | v2 over best tiled | span as share of runtime | span as share of the gain over v2 |
|---|---|---|---|---|---|---|
| N=1024 | 512 / 1024 / 2048 | 4.66% | 1.15% | +54.87% | **4.66%** | **8.5%** |
| N=2048 | 256 / 512 / 1024 | 6.12% | 2.74% | +94.99% | **6.12%** | **6.4%** |

The span is relative to the **best tiled width**. The first version of this entry called it
"4.7–6.1% of the available tiling gain", which is the span wearing the gain's label: the gain over
v2 is 0.5487 and 0.9499 of the best width's time, so the span is 0.0466/0.5487 = 8.5% and
0.0612/0.9499 = 6.4% of it. Both bases are defensible and the pairing was not. **The choice inverts
which cell is worse** — by share of runtime the second (6.12 > 4.66), by share of the gain the
first (8.5 > 6.4) — so neither number travels without its base.

**The never-slower-than-baseline guarantee is untouched.** v2 is 54.9% and 95.0% slower than the
best tiled width in these cells, so every width the draw chooses between is far above the baseline
the guarantee is about. This costs headroom, not correctness.

**Still one host, and no code change proposed.** The hybrid-core alternative is dead but the
single-host limit is not: everything above is the M5. What a second host should run, in order:
(a) the same per-condition repetition at 2, 8, 16 and 32 workers; (b) whether adjacent candidate
widths are inside the probe's floor **by construction**, which would make the draw a property of
the search rather than of the machine, and is now the leading hypothesis; (c) whether increasing
the probe's repetition count resolves the width, which is the fix if (b) holds.

**Harnesses** (scratchpad; to be filed with the measurement if a second host confirms):
`tile_thread_dep.py` (refuses when no cell is eligible), `tile_thread_null.py` (reports its own
null and exits non-zero only when the signal exceeds it), `tile_width_cost.py` (seeds the memo and
asserts it was not overwritten, so a non-replay cannot read as a replay).

---

## #1 `nthreads >= 2 * pool` (`scorch_policy.h:2277`), #2 `pool <= 16` (`:1583`), #3 mean-degree-16 unroll (`:1420`)

Not measured. All three need an x86 host — #1 and #2 are about pool size and worker counts that the
M5 cannot express, and #3 is an AVX2 threshold on a NEON-only machine. Recorded now so the file
does not read as though the absence of an entry were a verdict.

My position from reading alone, which is not evidence: #2 is the strongest of the three, because a
boundary chosen *between* the two tested machines has no measurement on either side of it. #3's
evidence base (one i9-14900K, 12 matrices, a bin boundary standing in for an optimum) is below the
two-host standard the rest of this branch is held to. #1's 64-CPU/8-worker case is a real
counterexample to the gate's stated intent, which was byte-neutrality on all-physical-core pools.

### #4 continued — the probe measures its own floor on every call and never applies it to the width pick

A peer supplied the structural route statically. **Verified at source rather than taken on report**,
and it holds at every load-bearing point:

- The baseline is entered **twice**, as `cands = [("v2", 0, 0)]` (`tiling.py:1285`) and
  `cands.append(("v2", None, _baseline))` (`:1681`) — two identical arms at opposite ends of the
  candidate list, which is this cell's own A/A control, measured on every call.
- `_clears_noise` (`:1370`) derives `spread` from those two arms and demands the winner beat the
  *faster* of them by more than the spread. Its docstring is explicit about why: *"the probe
  memoizes its verdict permanently, so a coin flip decided by noise is not a one-call mistake, it
  is every subsequent call on that shape."*
- It has **exactly two call sites**, `:1437` and `:1694`. Both compare a winner against the
  **baseline**. Neither compares two tiled candidates to each other.
- The width is chosen by a bare argmin at `:1689-1692`, strict `<`, first wins ties, with no margin.
- Each `times[i]` is a **minimum** over `rounds=2` (`:1311`, `:1357-1365`), so the pick is an argmin
  of a downward-biased statistic over two draws — the candidate that catches one lucky round wins.

**So the quantity needed to detect that two widths are indistinguishable is computed on the same
call, by the same function, and discarded for the pick.** The file states the hazard precisely and
applies the guard to one of the two decisions it governs.

**Measured, because a structural route is not a rate.** Logging the probe's own `times` and
`spread` on each call, 24 calls per cell, M5 at nt=6:

| cell | top-two gap (median) | baseline spread (median) | calls whose top-two gap is INSIDE the measured spread |
|---|---|---|---|
| N=1024 | 1.0093 | 1.0380 | **21/24 = 0.88** |
| N=2048 | 1.0184 | 1.0276 | **15/24 = 0.62** |

The typical gap is well inside the typical floor. **The probe is choosing between candidates it
cannot distinguish, and it holds the number that says so.**

**This changes the fix.** My earlier note suggested a higher repetition count. That is wrong on its
own: more rounds shrink a floor the rule never consults. The matched change is to apply the existing
measured spread to the candidate-versus-candidate comparison and break the resulting tie
deterministically — the analytic `_panel_width` prediction, or the incumbent — rather than by
argmin. The same gap exists one level up: `_LEARNED_MARGIN` (`:1303-1305`) guards the tiled winner
against the predicted v2 and nothing guards it against the other tiled candidates. **Neither level
has a margin between two tiled candidates.**

**A documented claim that the rerun refutes.** `_interleaved_times`' docstring (`:1330-1341`)
defends rotating the start position over drawing a fresh permutation partly on the grounds that *"a
shuffle would also make a dispatch decision nondeterministic across runs, and that decision is
memoized; a verdict that varies run to run is worse than one carrying a bias the margin absorbs."*
The decision **is** nondeterministic across runs — measured above — and the cause is upstream of
ordering, so it survives any ordering discipline. "A bias the margin absorbs" is true only of the
baseline comparison, which is the one comparison that has a margin. The docstring cites real
measurement elsewhere (0.979–1.013 rotated against 1.001–1.051 shuffled) and rests this one clause
on an untested claim about its own output.

### The learned level's determinism prediction is VACUOUS, and declining to tile is the larger finding

The peer offered a cheap falsifiable prediction: `level="learned"` picks from a model prediction
rather than a timing, so its transition rate should be 0.00 where the probed levels flip. Measured,
8 repeats per condition:

| level | N=1024 nt=2 | N=1024 nt=6 | N=2048 nt=2 | N=2048 nt=6 |
|---|---|---|---|---|
| balanced | 0.29 | 0.14 | 0.71 | 0.43 |
| learned | **0.00** | **0.00** | **0.00** | **0.00** |

**The 0.00 is vacuous and the prediction is untested.** The learned level returned `('v2', None)` on
all 32 calls — it never chose a width, so there was nothing for it to be stable about. My first
version of this check verified only that a model was *present*, which is the wrong guard: the
instrument has to confirm the decision under test was actually *made*. Corrected to print the whole
pick rather than the parameter.

**And the reason it is vacuous is a bigger finding than the one it was testing.** The learned model
routes to the baseline on two cells where the probe measures tiling **54.9% and 95.0% faster**. This
is in-domain: `_load_learned_model` (`:1232-1237`) rejects on `machine_id` mismatch, version bump
and feature-schema drift, and it loaded, so this model was trained for this machine. It reached
`('v2', None)` through the prediction path rather than the eligibility gate (`:1510-1514` would have
returned `None` and memoized nothing).

**This is adjacent to concern #5 and should be read with it.** The only quality figure
`train_autotune_model.py` exports is `heldout_geomean`, and that figure is in-sample — `lreg` is
computed over all training rows and the comment above it says "optimistic". A model whose sole
published score is optimistic, leaving a 1.5–1.9× measured win on the table on the first two cells
probed, is the failure mode that mislabel conceals. **Not yet a verdict**: two cells, one host, and
I have not checked whether these shapes are inside the model's training distribution. What would
settle it: the grouped-CV figure that `group_split` already computes at `:215` and never exports,
plus the model's predicted-versus-measured ratio on these two cells.

### #5 — confirmed at source, and the learned level's prediction is wrong by 3–4× in scale and inverted in rank on the two cells probed

**The mislabel, verified.** `bench/train_autotune_model.py:298` reads
`# in-sample cell agreement (sanity, optimistic)`; `:299` computes `lreg` with
`eval_cells(rows, …)` over **all** training rows; `:305` writes exactly that into the shipped spec
as `heldout_geomean`. Two things the concern did not say, both from a peer and both confirmed:

- `heldout_geomean=round(geomean([r for r in lreg]), 4)` — `[r for r in lreg]` is an **identity
  comprehension**, a filter shape with no predicate. That is the construction that lets an unsplit
  list read as a selected one at a glance.
- **The honest figure is not missing. It is computed, printed, and discarded.** `group_split`
  (`:215`) splits by matrix so a matrix's rows never straddle train/test, and `:252-266` runs it
  over multiple seeds and prints `learned held-out geomean = …` under the correct name. The script
  holds both numbers and persists the wrong one. The fix is to write down the figure already
  running, not to add a computation.

**Scope, which bounds it.** `tiling.py`'s loader reads only `version`, `machine_id` and
`feature_names` (`:1233-1235`), so `heldout_geomean` is not consumed by dispatch and the mislabel
cannot misroute anything. It misleads a reader. **Two specs ship the field, not one** —
`autotune-levels/models/model_74c1226957b8a5ce.json` carries `heldout_geomean` 0.9896
(`n_train_rows` 1572) and `model_83841bfe8e6c0539.json` carries 0.9861 (`n_train_rows` 1504). Both
are tracked and both sit in `MODELS_DIR` (`train_autotune_model.py:42`), so the mislabel ships
twice; an earlier revision of this paragraph named only the first and understated it by one
artefact.

**Both shipped specs are `platform=Darwin`, which bounds this whole section to two Mac
machine_ids.** `_load_learned_model` (`tiling.py:1219-1241`) rejects a spec whose `machine_id` is
not this machine's and returns `None`; its own docstring names the consequence, `-> analytic
fallback`, and the `except` arm repeats it as "silent analytic fallback". So on redwood and MKT the
learned level runs the analytic selector, and every finding below about the learned path is a
finding about those two Darwin hosts. The module docstring's redwood figure
(`train_autotune_model.py:16-18`) therefore has no shipped artefact to attach to — that
parenthetical is ambiguous between naming the analytic baselines and the learned held-out results,
and I am not reading it further than that.

**There is no domain check anywhere on the learned path.** `_eligible_learned` (`:1269-1276`) has a
one-line body: `return _operand_over_cache(J, N, C)`. Its docstring states the delegation —
*"Relax the degree floor + locality pre-exclusion; the model + v2 floor route the newly-admitted
products/arxiv/FEM/banded shapes"* — so the two gates that would have bounded `f_log_degree` and
`f_locality` are the two the widening removed **by name**. The feature vector is **17** wide; one
derived combination of three of them is all that is constrained.

**And the shipped artefact could not support a domain check even if the code wanted one.** The spec
carries `feature_names`, `trees`, `init`, `learning_rate`, `machine_id`, `platform`,
`n_train_rows` (1572 and 1504, one per shipped spec) and `heldout_geomean` — and **no feature ranges, quantiles or bounds**. A
consumer cannot tell whether an input is inside the training domain from what ships.

**The measurement that discriminates, run because a structure is not a rate.** Capturing the
model's own predictions and timing the same candidates, M5 at nt=6:

| cell | model's predicted times | measured | |
|---|---|---|---|
| N=1024 | all six candidates in **3.04–3.95 ms** | v2 17.772 ms, best tilej(1024) 11.734 ms | v2 **+51.5%** slower |
| N=2048 | all six in **7.8–10.8 ms** | v2 43.567 ms, best tilej(512) 22.814 ms | v2 **+91.0%** slower |

Two distinct failures:

- **Scale.** Predictions are compressed into a narrow band roughly **3–4× below** the measured
  times. A gradient-boosted tree is piecewise constant and cannot extrapolate: an out-of-domain
  input is clamped into the outermost leaf region and returns whatever the nearest training region
  said, with **no signal in the output** that the input was outside. Compression far from the truth
  is that signature.
- **Rank, and the margin cannot catch it.** At N=1024 the predicted spread between the best
  candidate and v2 is `exp(−5.824 + 5.794) = 0.970` — a **3.0%** predicted improvement, which
  `_LEARNED_MARGIN` correctly rejects, against a **51.5%** measured one. At N=2048 the model's own
  argmin **is v2**, where v2 is measured 91.0% slower: the predicted ranking is inverted.

**This is the structural point made concrete.** At the probed levels the floor is a *measurement* —
two identical baseline arms bracketing the candidates. At the learned level the "v2 floor" is
`_LEARNED_MARGIN` comparing `pred[best]` against `v2_pred` (`:1303-1305`): **two predictions from
the same model.** When the model is wrong it is wrong in both terms, and the margin between two
wrong numbers carries no information about either. The floor cited to justify removing the input
gates is the model grading its own homework — and that is specific to the path the widening opened.

**What is still not established.** Whether these two cells' features lie outside the training rows'
ranges. Neither shipped spec carries ranges, and neither one's training rows (1572 and 1504) are in the repo, so
saturation remains a candidate structure rather than the demonstrated mechanism. Two cells, one
host. **But the consequence does not depend on the mechanism**: on these cells the learned level
routes to the arm it would measure as 1.5–1.9× slower, and nothing in its path can detect that.

## Concerns #1 and #2 — both predicates are REACHABLE on x86; the inertness argument is a statement about one host's core count

Measured 2026-10-02 on mkt1 (AMD EPYC 9334 Zen4, 128 CPUs on the machine; this run saw
64 via its cpuset, and `omp_get_num_procs()` in-process read 64), inside a 64-CPU Slurm
allocation, against `/scr/bobbyy/adg/adg0` — verified to carry the shipped constants
(`SCORCH_SPMM_PARTITION_GATE_MAXTHREADS 16`, `SCORCH_SPMM_NT_CAP -2`,
`SCORCH_SPMM_RECRUIT_MIN_WORK 1e7`). Probe: `harnesses/pfstream/pool_reach.py`.

Line numbers: the review named `:2277` and `:1583`; the predicates are at `:2313` and
`:2612` in the tree as measured. Both were read at both sites to confirm they are the same
two predicates, so this is traceability drift and not a wrong target. The inertness claim
for #1 appears **twice** -- `:333` (the constant's doc comment, which carries the word
"provably") and `:2298` (which does not) -- plus `:2283` asserting the same of the cap, so
a comment fix has three sites.

**This is a reachability result and not a performance result.** No runtime was measured.
What follows says the predicates can fire on x86, not that firing helps or hurts.

The probe asks production rather than restating it: `scorch_spmm_nthreads` and
`scorch_spmm_partition_mode` are both exported to Python for exactly this purpose, and a
harness that recomputed either predicate would be wrong silently and flatteringly.

### Concern #1 — `scorch_policy.h:2313`, `nthreads >= 2 * pool`

The source comment at `:333` reads: *"On an all-physical-cores pool no recruit is EVER at
stake (x86: pool 24, gate at 48, count at most 32), so this condition provably cannot fire
there."* Sweeping the caller's pool over a 175-cell grid of rows x nnz x k:

| pool | gate = 2 x pool | max resolved count | cells reaching the gate, of 175 |
|---|---|---|---|
| 8 | 16 | 64 | 85 |
| 16 | 32 | 64 | 85 |
| 24 | 48 | 64 | **85** |
| 32 | 64 | 64 | 83 |

**At pool 24 -- the exact pool the comment names -- the gate is reachable on 85 of 175
cells.** The comment's premise is that the resolved count is "at most 32". That bound is
`omp_get_num_procs()`, read at `scorch_policy.h:2118` and `:2167` and nowhere else.

**Corrected**: the first version of this section said that bound "is 32 on redwood and 128
on mkt1". 128 is `os.cpu_count()`. libgomp honours the cpuset, so inside the 64-CPU
allocation the run actually used, `omp_get_num_procs()` returned **64** -- which the
`max resolved` column was already showing by reading 64 at pool 8, 16, 24 and 32 alike, a
hard ceiling rather than four coincidences. Three quantities were going by one name:

| quantity | mkt1, this run | what it is |
|---|---|---|
| `os.cpu_count()` | 128 | the machine; what the sentence quoted |
| `len(sched_getaffinity(0))` | 64 | what the process may use |
| `omp_get_num_procs()` in-process | **64** | the predicate's actual bound |

The conclusion was stated in the one quantity the probe did not read, and it named a value
no process in the run ever saw. The probe now prints all three. This **strengthens** the
result rather than weakening it: the predicate fires at an effective processor count of 64,
so an unrestricted node is not needed for it, and 85 of 175 cells at pool 24 against a
ceiling of 64 is a stronger statement than the same count attributed to 128.

It also removes a contradiction between the rule and the probe's own guard: at pool 40,
`128 >= 80` says reachable while `64 >= 80` says not, and `pool_reach.py` already warns
that a pool whose gate exceeds the allocation reports the allocation. With the right
variable the two agree.

So the inertness is real and the reasoning is sound, but what it depends on is the ratio
`omp_get_num_procs() / pool`, not the instruction set:

> the predicate is reachable exactly when `omp_get_num_procs() >= 2 * pool`.

On redwood at its 24-thread pool that is 32 >= 48, false, inert -- which is what the x86
grid measured. On mkt1 at the same 24-thread pool it is 128 >= 48, true. The comment
states a one-host fact as a property of "x86" and of "an all-physical-cores pool", and
mkt1 is both an x86 host and an all-physical-cores pool where it does not hold.

### Concern #2 — `scorch_policy.h:2612`, `pool > gate_maxthreads` disables the work gate

The comment reads: *"Above the ceiling this whole block is unreachable, which is how the
x86 reading is preserved."* That is correct and the boundary reproduces exactly at the
shipped 16. But the converse is what matters: at or below 16 the gate is **active on
x86**, and its sign was established on ARM.

| pool | cells with mode != 0 at the reference pool | of those, forced to mode 0 | share |
|---|---|---|---|
| 1 .. 16 | 165 | 29 | 17.6% |
| 17, 20, 24 | 165 | 0 | 0.0% |
| 32 | 165 | 0 | 0.0% -- IDENTITY, the negative control |

Two corrections to the first version of this table.

**The boundary is located at 16, not merely bracketed.** The original sweep stepped
16 -> 20, which produces an identical table for a ceiling of 16, 17, 18 or 19. A
geometric pool sweep is right for reachability and wrong for localization, and one grid
cannot do both. Pool 17 was added and reads 0 forced, so the transition is between 16 and
17 and matches the shipped constant exactly.

**The pool-32 row is an identity and is now labelled as such.** The reference pool is 32
and 32 was in the sweep, so at that row the two calls being compared are the same call and
`forced` could only ever be 0. The first version banded it with two genuine zeros,
inflating the count of real zeros by one. It is kept rather than deleted, because it is the
only row whose answer is known in advance: a nonzero there would mean the detection is
broken rather than that the gate is inert. The general check is to intersect the sweep list
with every constant the comparison holds fixed, which finds both the vacuous row and the
free negative control in one pass.

The `cells with mode != 0` column is the non-vacuity check: it is 165 at every pool, so the
remaining zero rows are genuine.

Reachable by ordinary means. `torch.set_num_threads(16)` or less is a normal thing for a
caller to do -- it is what a process sharing a node, or running inside a dataloader
worker, will have -- and it puts an ARM-derived gate in charge of 17.6% of the cells in
this grid on an x86 host.

### A defect in my own probe, caught by its own premise check

The first run of this probe reported predicate 1 as "reachable at pool 1 and nowhere
else". It had run inside a default `srun -n 1`, which grants 2 CPUs, so the resolved count
was 2 at every pool and `2 >= 2 x 1` held for reasons having nothing to do with x86 or
with the policy. Every other row read "never fired" because the allocation could not
supply the threads, not because the predicate refused.

A thread-policy probe is bounded by the CPUs its process may use, so it has to state that
number and compare it against the largest threshold it intends to test. The probe now
prints `len(os.sched_getaffinity(0))` beside `2 * max(POOLS)` and says, before any table,
that a "never reached" verdict is uninformative when the allocation is smaller than the
gate. This is the same shape as the harnesses that report DONE having measured nothing:
the output was well-formed, plausible, and about the allocation rather than the subject.

### What is still open

Concern #3 (the mean-degree-16 width-1 unroll at `:1420`) is untouched. Its comment
already carries a redwood grid with confidence intervals by degree band, so the question
there is a second host rather than a first measurement, and the bands are n=22 and n=12.

Neither #1 nor #2 has a runtime measurement on mkt1 yet. The next step for each is
arm-vs-arm at a fixed small pool -- not against MKL, because this build reads env vars
per call and that tax manufactures deficits on short kernels.

### Concern #2, continued — the 17.6% was not attributable, the isolated figure is 16.7%, and the gate is toggleable on MKT

Measured 2026-10-02, same host and allocation. Three corrections and one enabling result.

**WITHDRAWN: the earlier design was not confounded, and this paragraph originally said it
was.** The first sweep varied the caller's pool and recomputed the resolved thread count
from it, so one swept variable reached both arguments of `scorch_spmm_partition_mode`. I
concluded from that structure that the number was not attributable, and named the
single-worker gate as the second consumer.

That gate is `partition_solo_off && nthreads <= 1` and
`SCORCH_SPMM_PARTITION_SOLO_OFF` is **0** at `:531`, so it cannot fire in a shipped build
-- and it is the only one of the four predicates that reads `nthreads` at all. Measured
over 288 cells at a fixed pool, `partition_mode` is identical at nthreads of 1, 2, 8, 32
and 128, at both pools: **zero cells differ**. With the solo gate switched on by its hook,
nthreads=8 against nthreads=1 moves **215** cells, which is the control showing the
comparison could have detected a difference had one existed.

So the second argument the swept variable reached has no effect at shipped settings, the
old figure was not confounded, and the 29-against-35 difference was entirely the grid
(175 cells against 216). I reasoned from the shape of the call rather than measuring
whether the shape mattered, which is the same error as inventing a mechanism for a
misattributed failure -- a structural argument standing in for a measurement.

What survives: holding the count fixed is still the right construction, and running it is
how the independence above is now known rather than assumed. It is a result, not a repair.

Instrument: `harnesses/pfstream/gate_isolate.py`, grid 6 rows x 8 nnz x 6 k = 288 cells,
committed so every figure below is regenerable. The first version of this entry quoted
figures from a script that existed only in a scratchpad, in a commit that touched no code,
so its "grid used" was not the committed probe's 175-cell grid and no number in it could be
reproduced from the repository. That is the defect `scratchpad is not storage` names.

**Isolated, by holding the resolved count fixed and varying only the pool:**

| held nthreads | gate-attributable | live cells | share |
|---|---|---|---|
| 2 | 65 | 280 | 23.2% |
| 4 | 65 | 280 | 23.2% |
| 8 | 65 | 280 | 23.2% |
| 16 | 65 | 280 | 23.2% |
| 32 | 65 | 280 | 23.2% |

Identical at every held count, and every row now reads **exact** against the composed
prediction. The share moved from 16.7% to 23.2% when two cells were added to the grid,
which is the plainest possible demonstration that a share of a synthetic grid is a property
of the grid -- the prevalence section below computes it on a real collection instead.

Identical at every held count, which is the expected result rather than a suspicious one:
the gate's test is `work_proxy = nnz * max(k,16)` against `mingrains * SCORCH_GRAIN_SPMM`,
and neither side contains a thread term. The conclusion of the earlier entry is unchanged
-- the gate is active on x86 at pool <= 16 and its sign was established on ARM -- but the
share is now attributable where before it was not.

**A third leg: the closed form, which the two env routes cannot supply.** `work_proxy`
contains no thread term and no row term, so the fired set is fixed by `(nnz, k)` and the
two committed constants -- `SCORCH_SPMM_PARTITION_MINGRAINS 2L` at `:1581` and
`SCORCH_GRAIN_SPMM 150000L` at `:82`, giving a threshold of 300000. The prediction is 36
cells, from 6 `(nnz,k)` pairs (all at nnz=3000) times 6 row values.

**All four predicates composed, and both columns are predicted from source.** At the large
reference pool the ceiling fires, so the work gate is skipped and the only exclusion is
maxout: 280 live of 288 predicted, 280 measured. At the small pool the ceiling does not
fire, so the work gate applies on top: 65 predicted, 65 measured, at every held thread
count including 1. Previously the ceiling was implicit in the choice of reference pool
rather than part of the prediction.

**The prediction's only host constant, and how much it matters.** Sweeping `llc_bytes`:
16,777,216 gives 272 live / 63 attributable; 33,554,432, 50,000,000 and 64,000,000 all give
280 / 65; 67,108,864 gives 288 / 66. So the prediction is constant exactly across
(32,000,000, 64,000,000] -- the same interval the `elem_size` probe bounds -- and changes
outside it. The result therefore needs `llc_bytes` *bounded* rather than known, which the
probe supplies independently. It is not host-free, and the coincidence of the two intervals
is not luck: both derive from the same two products.

**Composed, the prediction is exact.** A closed form for one predicate in a chain is an
upper bound, not a prediction, and an uncomposed one turns a disagreement into a narrative
instead of a defect -- which is what the first version of this section did, explaining its
one leftover cell in prose afterwards. `scorch_llc_bytes()` is exported, reads 33554432
(32.0 MiB) on mkt1, and `MAXOUT_LLC` is 2L, so maxout is 67108864 exactly. Composing the
work gate with maxout predicts **65** of the 66 the work gate alone predicts, removing
exactly `(rows=1000000, nnz=3000, k=32)` at `1000000 x 32 x 4 = 128000000 >= 67108864`.
Production returns 65. The cell is predicted rather than accounted for.

The measured bound of (64M, 128M] from the elem_size probe contains the exported 67108864,
so the two routes to the constant agree.

**Corrected: as first run, leg 3 BOUNDED the constant rather than confirming it.** The
original grid was geometric, with a 16x gap between nnz=3000 and nnz=20000. Its largest
firing work proxy was 96000 and its smallest non-firing one 320000, so *any* threshold in
(96000, 320000] -- a factor of 3.33, covering `MINGRAINS` of 1 or 2 against `GRAIN_SPMM` of
50000 or 150000 -- predicts the identical 36-cell set. A geometric grid cannot resolve
candidates 2x apart. No cell had work exactly equal to the threshold either, so the
comparison's strictness was untested as well.

Fixed with two bracket points rather than a finer grid. At k=16, nnz=18749 gives a proxy of
299984 and nnz=18750 gives exactly 300000, so under a strict `<` exactly one fires -- and no
other candidate threshold produces that split (100000 and 150000 give zero firing, 450000
and 600000 give two). Re-run with them in: the prediction is 66 cells, production attributes
**65 with zero extra**, so every nnz=18749 cell fires and every nnz=18750 cell does not.
**The product is exactly 300000 and the comparison is strict.** The constant is now
confirmed rather than bracketed.

This leg matters because the two env routes only establish that two readings of the source
agree with each other; both go through the same exported function, so neither can tell
whether that function reads the constants the header declares. A linked object whose
`MINGRAINS` or `GRAIN_SPMM` differs from the header -- which is what a defines-only rebuild
relinking stale objects produces -- would leave the two routes agreeing and both
disagreeing with the closed form.

**The row-dependent exclusion is identified, and confirmed from a second direction.**
`scorch_spmm_partition_mode` holds five predicates, and exactly one of them has a `rows`
term: `rows * out_cols * elem_size >= maxout`. So a row-dependent exclusion can only be
that gate -- an enumeration with a one-member candidate set, not a proposed mechanism.

It predicts a movable boundary, which is what makes it testable rather than merely
plausible. The probe passes `elem_size`, so at 8 bytes the product doubles and the
exclusion must move from k=32 to k=16. Measured at rows=1e6, nnz=3000:

| elem_size | k=8 | k=16 | k=32 |
|---|---|---|---|
| 4 | product 32M, mode 3 | product 64M, mode 3 | product 128M, **mode 0** |
| 8 | product 64M, mode 3 | product 128M, **mode 0** | product 256M, mode 0 |

The exclusion moved exactly as predicted, and the boundary sits at the same product both
times. With `SCORCH_SPMM_PARTITION_MAXOUT_LLC 2L` at `:534` that bounds the host's
`scorch_llc_bytes()`: live at 64,000,000 bytes and excluded at 128,000,000 gives

    maxout in (64,000,000, 128,000,000]   so   llc_bytes in (32,000,000, 64,000,000]

**Corrected, and the first version of this line excluded its own measured value.** It read
"maxout in (64M, 128M], hence an LLC of (32 MiB, 64 MiB]". The products that produced the
bound are decimal, so halving them gives decimal bytes; writing the halves in binary units
makes the interval (33,554,432, 67,108,864], whose lower endpoint is **open at exactly the
33,554,432 that `scorch_llc_bytes()` returns** -- so the published bound excluded the
measured value, and its upper end implied a maxout of 134,217,728 which the probe had
already ruled out. The slip is invisible because 32 and 64 appear on both sides of the
conversion. Stated in bytes from here.

**Two invariances the predicate's own form demands, both run.** Over the held thread count:
identical sets, as required. Over rows: the harness reports **DIFFERS**, and correctly --
6 `(nnz,k)` cells at each of rows 64, 256, 512, 4096, 65536 and only 5 at rows=1000000.
That is the same single cell as above, so the check fired on a real asymmetry and the
asymmetry has an explanation outside the gate. A check that passed silently here would have
been the weaker outcome.

**A defect in my own toggle, which read as a missing feature.** To test whether the build
could switch the gate off I set `SCORCH_SPMM_PARTITION_GATE_MAXTHREADS=0`. That is
backwards. The source is

    if (gate_maxthreads > 0) { if (pool > gate_maxthreads) mingrains = 0; }
    if (mingrains > 0) { if (work_proxy < mingrains * SCORCH_GRAIN_SPMM) partition_mode = 0; }

so 0 skips the *pool ceiling*, which leaves `mingrains` at its default and the work gate
**active at every pool** -- the opposite of off. Zero means "unset this guard" in one
constant and would have to mean "disable this mechanism" in the other, and the knob's name
discloses neither. The probe reported "hooks are NOT live" on a build where they are.

**The gate is toggleable on MKT, so no build is needed.** Ten hooks builds already exist
there; `adg/*` are not among them (0 of 4 knob strings present in any, which is why the
first probe read nothing). `/scr/bobbyy/raisefloor/tree` carries 49 `SCORCH_SPMM_*`
environment strings. With the resolved count held at 8 and the pool at 8 (control row first, because it is the
one that separates a live hook from a live hook whose object carries a stale default):

| arm | cells with mode != 0 | moved vs shipped |
|---|---|---|
| `MAXTHREADS=16` (shipped) | 175 | - |
| `MAXTHREADS=4` (ceiling fires, so the gate is skipped) | 210 | 35 |
| `MINGRAINS=0` (the gate's own documented "leaves it out") | 210 | 35 |
| `MINGRAINS=2` (shipped value, as a control) | 175 | **0** |

The two independent routes to "gate off" agree on all 35 cells and disagree on none, so
the two readings of the source are consistent. The last row is the control that matters:
setting a knob to its compiled default must move nothing, and it moved nothing -- without
it, a live hook and a hook whose default differs from the header would look the same.

Each arm sets exactly one environment variable, so the hooks build's per-call `getenv` tax
is equal on both sides. That tax is why the runtime contrast must be arm-against-arm and
never against MKL.

**What this enables, and what is still not measured.** The runtime question -- does the
ARM-derived gate help or hurt on x86 at pool <= 16 -- is now a single-binary A/B on 35
identified cells, with no second build and so no layout artefact between the arms. Still
unmeasured: any runtime at all, for either concern.

### Concern #2, prevalence on a real collection — 30.2% of DLMC, and it needed no host

Measured 2026-10-02 locally. The grid figures above are on synthetic `(rows, nnz, k)`
tuples, and a share of a synthetic grid is a property of the grid. The decisive question is
how much of a real collection the gate governs, and the closed form answers it from
dimensions alone: the gate forces `partition_mode = 0` when
`nnz * max(k,16) < 300000`, so it needs `nnz < 18750` at k <= 16 and `nnz < 9375` at k=32.

Over the 3820 DLMC matrices **this project measures** -- the screen's corpus and
chain207's, which is the right population here but is not all of DLMC; the same predicate
over chain209's 3846 keys gives 1180 = 30.68%:

| k | matrices in the gate region | of 3820 | share |
|---|---|---|---|
| 1, 2, 4, 8, 16 | 1154 | 3820 | **30.2%** |
| 32 | 724 | 3820 | 19.0% |

The flat share across k <= 16 is the `max(k,16)` in the work proxy, not a coincidence.
The collection's nnz runs min 6, median 52428, max 8.5M, so the region is the lower third
rather than an edge case.

And over the 4594 SuiteSparse matrices cached on mkt1, with **both** predicates applied:

| k | gate alone | + output-bytes | + operands only | share of 4446 operands |
|---|---|---|---|---|
| 1, 2, 4, 8 | 2181 | 2181 | 2069 | **46.5%** |
| 16 | 2181 | 2180 | 2069 | **46.5%** |
| 32 | 1379 | 1378 | 1294 | **29.1%** |

**The operand predicate is symmetric and I verified one half where it cannot fire.** The
first version excluded `shape[1] == 1` on SuiteSparse and reported 0 one-row entries on
DLMC -- the corpus where *both* halves are vacuous. On SuiteSparse, `shape[0] == 1` catches
**23** more entries, and all 23 sit inside the region at every width: a one-row CSR has
`len(indptr)-1 = 1`, so its nnz is tiny and its `rows * k * 4` is `4k`, which never trips
the output-bytes predicate. They inflate the region exactly as column vectors do.

With both halves, non-operators are 125 + 23 = 148, the operand denominator is 4446, and
the shares are 46.5% and 29.1% -- 0.93 and 0.92 pp below the uncomposed 47.47% and 30.02%.

**The shape predicate is better than the filename one and is still not complete.**
Showing a filename match misses 80% of the non-operators does not show that
`min(shape) == 1` catches all of them, and the complement was untested: an entry can be a
non-operator and still have `min(shape) > 1`. A multi-input state-space `B` is n x m, a
multi-output `C` is m x n. Measured on the cache, the complement is **not empty**:

| min(shape) of the 1066 non-square entries | count |
|---|---|
| 1 (the shipped predicate's catch) | 148 |
| 2-4 | 41 |
| 5-16 | 56 |
| 17-64 | 57 |
| 65-256 | 105 |
| >= 257 | 659 |

`boneS01_C` is 3 x 127224, `inlet_B` is 11730 x 2, `rail_1357_B` is 1357 x 7. And the
families confirm the role: 26 base names hold both a degenerate and a thin member, e.g.
`bips07_1693_iv` at (13275, 1) beside `bips07_1693_B` at (13275, 4) and `bips07_1693_C` at
(4, 13275) -- a four-input power-system model, where the single-input members are the n x 1
the shipped predicate already catches.

**No cutoff completes it, which is why the fix is a bracket and not a wider predicate.**
The distribution above is continuous from 2 upward with no gap, so any `min(shape) >= c`
is a chosen constant of exactly the kind this project does not ship, and `bibd_12_4` at
(66, 495) is a combinatorial design matrix -- genuinely sparse, genuinely an operator,
and thin. The share's sensitivity to c is therefore the honest statement:

**And the bracket must vary only the incompleteness.** The first version of this table
swept `min(shape) >= c` over the whole corpus, which also discards every SQUARE matrix
below c x c -- operators this very section argues are legitimate, and small ones, so they
sit preferentially *inside* the small-nnz region being measured. At c = 17 that sweep
removed 120 entries where only 97 thin candidates exist below 17; the excess 23 were
square. The span it produced, 6.14 pp, measures the share and the operator definition
moving together. Restricting the cutoff to non-square entries gives the sensitivity to
the incompleteness alone, with the square count the naive sweep would have taken shown
beside it:

| c | operands | k<=16 region | share | k=32 region | share | squares spared |
|---|---|---|---|---|---|---|
| 2 (shipped) | 4446 | 2069 | **46.536%** | 1294 | **29.105%** | 0 |
| 3 | 4439 | 2062 | 46.452% | 1287 | 28.993% | 1 |
| 5 | 4405 | 2028 | 46.039% | 1253 | 28.445% | 1 |
| 17 | 4349 | 1972 | 45.344% | 1197 | 27.524% | 23 |
| 65 | 4292 | 1918 | 44.688% | 1144 | 26.654% | 102 |
| 257 | 4187 | 1831 | 43.731% | 1060 | 25.316% | 234 |

Over a 128-fold range of c the share moves **2.81 pp** at k<=16 and 3.79 pp at k=32,
against the 6.14 pp the contaminated sweep reported. So the headline does not hinge on the
incompleteness, and the margin is twice what the first version claimed. The divergence
between the two sweeps begins at c = 17 and not before, because nothing square in this
cache has min(shape) below 5. The denominator moves with c in both (an excluded entry
leaves numerator and denominator together), which is stated because the two conventions
differ by over a point at the widest cutoff and a share alone does not say which was used.
Reported at c = 2: the only cutoff needing no chosen constant.

**A suffix list over-counts, which is why it is not the test.** Of the 158 entries matching
`_(b|B|C|E|iv|x|rhs)` with `min(shape) > 1`, **70 are square** -- `LF10_E` at 18 x 18 and
`ted_B` at 10605 x 10605 are descriptor-system mass and input matrices that are perfectly
good SpMM operators. A square matrix is an operator whatever its suffix. The two tests that
carry the finding above use no list: the `min(shape)` histogram and the mixed-family count.

**This does not change the region list, and that is a different question from the share.**
A 4 x 13275 CSR times a 13275 x 16 dense is a valid SpMM and a valid kernel cell; whether a
user would ever call it is workload realism. The denominator matters for the prevalence
claim, where counting a right-hand side as an operand inflates it. The emitted list feeds a
runtime A/B that needs valid shapes, not realistic ones, so it stays at 2069 members.

**The one-row count rests on a property that was implied and is now printed.** All 23
one-row entries leaving the k=32 region requires nnz < 9375 for every one of them. That is
not free: `LF10000_C` is 1 x 19998, and a dense row would give nnz*32 = 639936, outside the
region, so a dense-row `_C` would break the count. Measured max nnz among the 23 is **64**
(`inlet_C`, shape (1, 11730), nnz 64), so the arithmetic holds -- these are sparse output
maps picking out a handful of states, not dense ones.

**The pp shifts are rounded from the difference, not differenced from rounded inputs.**
47.475 - 46.536 = 0.939 and 30.017 - 29.105 = 0.912, so **0.94 and 0.91**; the previously
published 0.93 and 0.92 came from differencing 47.47 - 46.54 and 30.02 - 29.10, off by one
hundredth in *opposite* directions. Pre-rounding an input can move a difference either way,
so it is not a conservative simplification. And the 0.939 pp bundles two corrections of very
different size: composition is **0.022 pp** and operand exclusion is **0.917 pp**.

**The denominator counted vectors as matrices.** SuiteSparse ships the right-hand side of
each linear system alongside the operator, and a one-column entry stored as CSR has
`len(indptr)-1 = rows` with very few nonzeros -- so a predicate selecting on *small nnz*
collects them preferentially. 125 of the 4594 cached entries have `shape[1] == 1`, and 88
of them were inside the k=16 region. Excluding them shrinks numerator and denominator
together, so the published 47.5% and 30.0% become 46.8% and 29.5%: a small move, but the
earlier figures were a share of files rather than a share of operands.

The suffixes say what these are. Of the 125 one-column entries: 34 end `_B`, 29 end `_b`,
11 end `_iv`, and the rest are numbered or bare; the one-row entries are `_C` members like
`LF10_C` at shape (1, 18) and `LF10000_C` at (1, 19998). These are the components of linear
systems and state-space models -- right-hand sides, input matrices, initial values, output
matrices -- shipped alongside the operators. `shape` is the predicate and the filename is
not: a convention matching `_b` alone would have found 29 of 148. It
also supersedes an nnz floor I had noted as a design need after seeing region members with
nnz = 1 over hundreds of thousands of rows -- those are these, and "not an operator" needs
no chosen constant.

The single entry the output-bytes predicate removes is one of them: `Hamrle3_b.npz`, shape
`(1447360, 1)`, nnz 440 -- the right-hand side of Hamrle3, whose operator is
1447360 x 1447360. So that removal was never about a matrix.

DLMC carries no such contamination: over all 3820 keys the minimum column count is well
above 1 and there are no one-column or one-row entries, which is what the collections'
contents would predict -- pruned weight matrices rather than linear systems.

The denominator is the readable count, which here equals the file count -- all 4594 parse,
so the two coincide; an earlier version published the file count while computing the
readable one, which is only correct when they do.

**The composition matters structurally even though it removes one matrix.** The gate is one
predicate of four, and a share computed from it alone is an upper bound -- the same omission
the synthetic-grid section had already corrected one commit earlier, repeated here on a real
corpus. The omitted predicate excludes when `rows * k * 4 >= 2 * llc`, needing rows >=
1048576 at k=16, and SuiteSparse's largest row count is 9,888,048, so it is **live** on this
corpus.

On DLMC it is **vacuous**: the sealed dims ledger gives a maximum of 33288 rows over all
3820 keys, so the largest `rows * 32 * 4` in the collection is 4,260,864 -- a factor of
15.75 under the threshold -- and the predicate excludes 0 of 3820 at every k in
{1,2,4,8,16,32}. So the 30.21% and 18.95% above **are** the composed shares exactly, and
this finding is isolated to SuiteSparse, whose row counts are unbounded.

**So at any caller pool of 16 or fewer, an ARM-derived gate decides the row-handout mode
for about a third of DLMC and nearly half of SuiteSparse on an x86 host.** That is the
reason the runtime question is worth measuring rather than leaving at reachability.

Instrument: `harnesses/pfstream/npz_gate_region.py`. It now reads nnz and rows **exactly**,
from each member's `.npy` header, which states the shape and the dtype -- so there is
nothing to infer. Index dtype over all 4594 readable files is `int32` uniformly.

The first version inferred nnz by dividing the ZIP directory's byte count by a guessed
element size, inside a `for isz in (4, 8)` loop that **could only ever return 4**: dividing
a fixed byte count by a larger element gives a smaller count, so if the first pair passed
the positivity test the others were unreachable. It then discarded the element size before
the filter. Its docstring described a selection the code did not make, and its twelve-matrix
cross-check could not cover a dtype the loop could not reach.

**The row axis is now checked rather than taken from a directory name.** `len(indptr) - 1`
is the row count for CSR and the **column** count for CSC, and the output-bytes predicate
reads rows -- so the composed column rested on the cache being called `csrcache`. The
archives carry `shape`, and `len(indptr) - 1 == shape[0]` holds for all 4594 with zero
disagreements. The number that makes that non-vacuous: **1066 of the 4594 are non-square**,
and only those can discriminate CSR from CSC, since a square matrix satisfies the check
under either layout.

Corpus-wide non-squareness does not certify the axis for the matrices a *figure* is about,
so it is also reported per region: **611 of the 2180** at k=16 are non-square (28.0%) and
478 of 1378 at k=32 (34.7%).

One further confound, and it is bounded rather than removed. `scipy.sparse.csr_matrix`
infers `shape = (len(indptr)-1, max(indices)+1)` when none is given, and an archive written
from such an object satisfies the check *by construction*. The discriminator is
`shape[1] > max(indices)+1`, which can only hold if `shape` carries information the arrays
do not -- a matrix with an empty trailing column is not reconstructible from `indices`
alone. On a 60-entry sample, **16 are strictly greater** and 44 are equal.

**And one witness settles it, not sixteen** -- a correction to how the first version of this
paragraph read the result. Whether `shape` was inferred from `indices` is a property of the
code that *wrote* the archives, not of any individual matrix, so a single entry where
`shape[1] > max(indices)+1` proves the writer did not infer, for every archive that writer
produced. The 44 equal cases are resolved by the same argument rather than left open; the
26.7% rate answers a different question (how often a matrix happens to carry an empty
trailing column), which is a fact about the matrices and says nothing about the check.

The premise is a single writer for the whole cache, and two uniformity counts collected for
other reasons support it: the member set is identical on 4594 of 4594, and the index dtype
is `int32` on 4594 of 4594. A second writer would have had to match both. What would retract
it is evidence of incremental or multi-script construction, and those two counts are the
evidence against.

A `format == b'csr'` check was written first and is not used, because **no file in this
cache carries a `format` member** -- they are not scipy `save_npz` archives, and a check
that exempts a missing key passes on all 4594 while proving nothing. The check I expected
to be weak is the strong one and the one I expected to be decisive is absent.

**The one matrix composition removes is the same matrix at both widths**: `Hamrle3_b.npz`,
nnz=440 over 1,447,360 rows. Its nnz is below the k=32 region's own bound too, so the
counts of 1 and 1 are one matrix and not two -- which the counts alone left open.

**A guard that could not run.** The unreadable-files branch had `%d` on one string and its
argument on another, so it raised `TypeError` before reaching its `return 1` -- and it
executes only when files are unreadable, which is exactly the condition it exists to
report. It fails closed by accident, a traceback also being a non-zero exit, but the named
count and clean exit it claims never happened. Third guard on this branch that could not
fire on its own input.

Two things about that, in opposite directions. It was **one-signed**: a true int64 index
read as int32 doubles the inferred nnz against an upper-bound predicate, so matrices with
real nnz in [9375, 18750) were dropped and none was added -- the published figure was a
lower bound presented as exact. And it turns out to have been **accidentally correct here**,
because the corpus is uniformly int32. That is luck, not design, and the fix removes the
dependence on it.

One design note for the runtime work: the region's smallest members have nnz = 1, some with
hundreds of thousands of rows. Those are not meaningful SpMM subjects and must be excluded
by a stated floor rather than left to dominate a geometric mean.

Provenance: **three** independent records of rows and nnz agree on all 3820 common keys
with zero disagreements -- the sealed `verifier-dlmc-dims` ledger (digest verified),
`dlmc_rowdisp.py`'s own read of the `.smtx` row pointers, and `chain209-dlmc-dims`, sealed
in a different milestone before this thread existed. A prevalence figure computed from one
record of the corpus would be a claim about that record.

The two dims ledgers share the filename `dlmc_dims.csv` and look unrelated on inspection:
different row counts (3846 against 3820), different byte sizes, different digests, `cmp`
reports DIFFER and `diff` reports every line changed. The causes are a `dlmc/` key prefix
on one side and a two-line `#` preamble on the other. Normalized, they are identical. **A
digest mismatch between two records of the same quantity is evidence of nothing until the
comparison is normalized** -- this nearly read as a provenance alarm.

## Concern #2, runtime: the work gate's threshold was calibrated against a different mode

Every earlier probe on this predicate measured which cells it selects. This section is
about time, and about where the gate's original evidence does and does not reach.

### The measurement

One object, arms are env vars read per call at `:2532` and `:2599`, interleaved inside one
process. Pool 8, width 16, float32, mkt1, 120 cells, isolated at `omp_get_num_procs()` = 14
so concern #1's adoption path cannot fire. Readings are tax-free: a caching `getenv`
returns the same pointers and flushes on every write to the environment, so decisions are
unchanged while the hooks build's measured 20.13 us per-call `getenv` cost stops being
per-call.

| route | population | median A-B | sign count | scale | ratio A/B |
|---|---|---|---|---|---|
| R1 gate as shipped | T in-region | **+14.224 us** | **120 of 120** | 29.86 us | **1.6175** |
| R1 | C1 floor, same matrices at k=32 | +0.008 us | 79 of 120 | 26.03 us | 1.0003 |
| R1 | C2 floor, same width, larger nnz | -0.000 us | 60 of 120 | 23.35 us | 1.0000 |

Arm A sets `MINGRAINS=2`, the shipped value, which forces mode 0. So **the gate selects the
slower mode on its entire treatment population**, by 61.75%, on every cell. Under the
theoretical null P(all 120 positive) is 10^-36.1; under C1's own observed rate, 10^-21.8.
Min-of-reps and median-of-reps agree to 0.005 us with comparable arm spreads, so the
estimator choice is not carrying it.

### A proposed explanation, WITHDRAWN, and the error is worth more than the claim was

An earlier version of this section argued that the threshold had been calibrated against a
mode the gate does not gate. The reasoning: `dcf2e1f` states its separator as "the cells
where the partition falls more than 10% behind **the shared counter**", and at that commit
`SCORCH_SPMM_PARTITION_DEFAULT` was **0**, so -- the argument went -- back-stealing was not
what "the partition" meant, and `8ccd104` introduced it when it flipped the default to 3.

**That is wrong, and the policy header at `dcf2e1f` says so in terms.** Its comment on the
same macro reads: "0 = one global atomic counter, which is what ships today ...; 3 =
contiguous home ranges with stealing from the back of a victim's range." Same number, same
semantics, at the calibrating commit. `steal` appears 41 times in that commit's `spmm.h`.
And its `SOLO_OFF` comment eleven lines above describes "what the `p3` arm means on the tiny
cells partway through a study whose other arms are already measured" -- so that study had a
p3 arm and had already measured others. `8ccd104` changed which mode SHIPS. It did not
introduce the mode, and the contrast the gate decides was 0 against 3 before and after.

The error is the generalisable part: **a default flip tells you what shipped, not what was
swept.** A calibration grid studying a feature that is off by default must set it
explicitly, so `PARTITION_DEFAULT 0` carries no information about which arms the study ran.
And the inverted prior is in the same comment, in the past tense: that study is described
as one "whose other arms are already measured". A study with measured arms, at a commit
whose header documents mode 3 by number, makes it likely that mode 3 *was* among them --
the opposite of what the withdrawn argument assumed. (The following sentence there, "priced
as its own arm first, then flipped", is `SOLO_OFF`'s own prospective plan for itself, not a
declared project method, so it is the weaker of the two citations and not the one this
rests on.)

What remains open is narrower and empirical: `dcf2e1f`'s separator column is labelled only
"the partition", so whether it is mode 3, a pooled aggregate over variants, or the
non-stealing partition with p3 priced separately is a question about that grid's arm labels.
Answering it also answers the base-rate question below, since the same grid holds both. The
grid is not readable from this tree.

### And x86 at pool <= 16 was never measured, by that commit's own account

`8ccd104` says, of the pool ceiling: "ungated the work gate costs 7.0% and 8.5% across the
whole x86 corpus, uniformly at every k. Capped at 16 it is **unreachable on a 24-thread
pool**". Its x86 grid is redwood's, at a 24-thread pool -- so the ceiling made the gate
inert for that entire grid, and the headline 1.3242x never exercised it. The gate's only
positive evidence is the next sentence: "On the **6-thread host** the gate recovers a third
of the row partition's only regression", and that host is the ARM one (the same commit
reports "1099 passed, 48 skipped on the ARM host").

So the evidence partitions cleanly:

| configuration | gate fires? | evidence |
|---|---|---|
| x86, pool > 16 | no, ceiling blocks | 1.3242x grid, gate inert |
| x86, pool > 16, ceiling removed | yes | **costs 7.0% / 8.5%** |
| ARM, 6 threads | yes | recovers a third of one regression |
| **x86, pool <= 16** | **yes** | **none** |

The last row is the gate's active range on x86 and it is empty. The 7.0%/8.5% cost figure
already has the sign I measure; the measurement above is the same sign, larger, inside the
range the ceiling deliberately leaves open.

### What this does NOT establish

**The duration band is not evidence either way, and I withdraw leaning on it.** The header
reads "the cells where it falls more than 10% behind the shared counter sit at 19-30
microseconds where the cells it wins sit at 31-110". `dcf2e1f` gives the quartiles those
came from: losing cells at **19 / 27 / 28 / 30** and winning cells at **16 / 31 / 62 / 110**
at min / q1 / median / q3. So "19-30" is min-to-q3 of one population and "31-110" is
q1-to-q3 of the other -- different functionals in one sentence -- and the winning cells'
minimum of 16 is *below* the losing cells' minimum. The two distributions overlap across
the whole of 19-30 and at least a quarter of winning cells sit under 31. A cell measuring
29.86 us is therefore in a region both populations occupy. What survives is only that 29.86
is inside the losing cells' IQR (27-30) and below the winning cells' q1 (31).

**And those microseconds carry their own object.** `dcf2e1f` swept the LLC-divisor ladder
through a hook and closes by noting "a hook that is measured and rejected still costs a
getenv on every call in every instrumented build" -- so its durations were very likely
taken on an instrumented build and carry that build's `getenv` tax, as this concern's own
readings did before the shim. The comparison would need both sides on the same footing.
Not reading, inference: `dcf2e1f` does not name its object, and if the ladder was built
per-value with `-D` rather than swept by env var its durations are tax-free and the bands
stand as written.

**The band is not the criterion in the first place.** `dcf2e1f` scores the shipped
predicate as "catching 74% and 77% of the regressed cells against 14.5% and 13.7% of the
winning ones" -- those are P(in region | outcome). The 120-of-120 above is
P(slower | in region). Converting between them needs the base rates, how many regressed and
how many winning cells that grid held, which the commit never states. So 14.5% is not a
false-positive rate for the gate without those counts, and the right comparison is against
P(outcome | region) measured on the original grid.

### The pool sweep, which is the axis that could have reversed this

The shipped comment at `:1595-1603` says the gate's sign depends on the pool and the
ceiling exists to make that dependence explicit, so a single pool could not speak for the
predicate. Four pools x four widths, 120 cells each, 144 rows parsed with none unplaced.
Positive = the gate selects the slower mode; the ratio is mode 0 over mode 3.

| pool | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| 2 | +1.99 us, x1.068 | +2.44, x1.099 | +2.50, x1.077 | +3.01, x1.086 |
| 4 | +3.75 us, x1.174 | +4.25, x1.232 | +4.27, x1.185 | +4.28, x1.185 |
| 8 | +14.12 us, x1.618 | +12.83, x1.573 | +15.43, x1.623 | +13.71, x1.568 |
| 16 | +12.19 us, x1.534 | +11.03, x1.489 | +12.01, x1.494 | +10.94, x1.460 |

Every cell is positive, every interval excludes zero, and the sign count is 120 of 120 in
15 of the 16 cells and 118 of 120 in the other. Tax-free cell scales run 20.5 to 34.6 us.

**The floor multiple is computed per cell, not from two ranges.** An earlier version of
this section said the floors were "0.01 to 0.029 us against effects of 2 to 15 us, so 100
to 1000 times too small" -- which multiplies the ends of two separate ranges and is only
the range of the ratio if the extremes co-occur. Those inputs in fact permit 69x to 1500x.
Measured per configuration, against the larger of that cell's own two controls:

| pool | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| 2 | 249x | 610x | 227x | 431x |
| 4 | 1877x | 425x | 329x | 329x |
| 8 | 1765x | 6416x | 3856x | 2284x |
| 16 | 12185x | 3675x | 1200x | 377x |

The minimum is **227x** at pool 2, k=8. The upper end is not a stable quantity: the largest
entries divide by floors of 0.001-0.002 us, which is a timer quantum rather than a
measurement, and a ratio against a quantum is one over the quantum. The claim is the
minimum. Note that the published 100-to-1000 figure was wrong in *both* directions:
arrived at by an invalid method, and understating the measured separation by an order of
magnitude at the top. It needed correcting irrespective of which way it moved.

**The sign does not depend on the pool across the gate's whole active range on x86.** But
the magnitude does not simply grow, and saying so pooled two levels that should be
separated. By pool: about 7-10% at pool 2, 17-23% at pool 4, **57-62% at pool 8**, and
**46-53% at pool 16**. Those last two supports are disjoint and pool 16 is lower at all
four widths, 4 of 4, so the shape is a rise to a maximum at pool 8 followed by a fall.

A reversal at the low end was the outcome that would have made the gate defensible with
only its threshold placement in question, and it does not occur.

**The turn is real and it is NOT a trend -- a registered expectation that did not hold.**
Before pools 20 and 24 ran, the prediction recorded here was that R2 would read *below*
pool 16's 46-53% if the decline continued. Measured:

| pool | R2 ratio, mode 0 over mode 3 | as a percentage |
|---|---|---|
| 8 | 1.568 - 1.623 | 56.8 - 62.3% |
| 16 | 1.460 - 1.534 | 46.0 - 53.4% |
| 20 | 1.379 - 1.477 | 37.9 - 47.7% |
| 24 | 1.459 - 1.518 | 45.9 - 51.8% |

Pool 24 reads **above** pool 20 at all four widths, and at k=16 pool 20's 1.477 is above
pool 16's 1.460. So the sharp drop from pool 8 to pool 16 is reproduced, and past it the
three higher pools form a plateau in the 1.38-1.52 band with no monotone ordering. The
prediction is withdrawn: what the data support is one step down after pool 8, not a
continuing decline. And the header
supplies a prior for it at `:1600`: its pool-24 anchor is 0.785, which in this file's
time-ratio convention is **1.274** -- below pool 16 and consistent with a continuing
decline. Different host and different population, so suggestive rather than matched.
The header's `:1605-1612` carries a second x86 cost figure, "that is the configuration x86
measured at 4-9% cost" -- same sign as this whole table and far below the pool 8 peak. But
it is **not an independent source**: `git log -S` places both it and the 0.785 in
`3b676df`, which closes "Both knobs stay off until the crossed grid runs on both", so its
4-9% predates the crossed grid that `8ccd104` reports as 7.0% and 8.5%. Those two values
lie inside 4-9, both are described as the whole x86 corpus ungated, and neither message
states its grid -- consistent with two grids and equally consistent with one corpus
summarised twice, in mild tension with `8ccd104`'s "uniformly at every k". So the honest
count is **two commits with independence not established**, not three sources. This sweep
is independent of both.

`3b676df` also settles the slash direction from the figure's own commit rather than from
the clause around it: "the counter wins 1.0303 -- while every home-range variant loses (no
stealing 0.9826, front-stealing 0.9682). On redwood the same comparison is 0.785 **the
other way**." The counter winning at 1.0303 and the partition winning at 0.785 are both
true under exactly one reading, a performance ratio, which converts to 1.274 in this file's
time-ratio convention.

**One qualification the gate's own criterion supplies.** `dcf2e1f` defines materiality as
"falls more than 10% behind". At pool 2 all four ratios -- 1.068, 1.099, 1.077, 1.086 --
are *below* that bar. So at pool 2 the gate is harmful by a measurement that excludes zero
and unanimous in sign, but not by the standard the gate itself was fit to. From pool 4
upward it clears that bar by 1.7x to 6x.

**And this does not contradict the comment's pool-6 anchor.** That anchor -- the counter
winning at pool 6 -- is an ARM reading, identified by `8ccd104`'s test line and by
`:1595-1603` itself ("The pool the surrounding pipeline runs is 6 there and 24 on
redwood"). Nothing in the header claims the counter wins at low pool on x86, which is the
row that was empty. Pool 6 is not measured here; it lies between pool 4 at +3.75 us and
pool 8 at +14.12 us, both positive.

### Why the gate reaches x86 at all: the ceiling is a two-host discriminator

`3b676df` states the design in terms: "the resolved worker count is 24 on redwood and 6 on
the M5, because the composition adoption takes the caller's pool, so a ceiling anywhere in
**7..23** lets the gate fire on one host and not the other." And the header at `:1605-1612`
gives the purpose: "A ceiling between the two pools makes the gate PROVABLY INERT on the
larger host."

So 16 was not chosen as a pool threshold with a measured crossover behind it. It was chosen
to separate two specific machines, and many values separate them equally well.

**But the quoted range does not transfer to the shipped predicate, and the header records
why.** At `3b676df` the test was `gate_maxthreads > 0 && (long)nthreads > gate_maxthreads`
-- the RESOLVED worker count. The shipped test at `:2614` is `pool > gate_maxthreads`. What
happened between them is narrower than a change of comparand, and more interesting: the
range was derived under an EQUALITY between the two quantities, which the ARM grid later
falsified. `3b676df` says the resolved count is 24 and 6 "**because** the composition
adoption takes the caller's pool" -- so the two were taken to coincide for the shapes in
question, and the comment three lines above that commit's own test already said "where the
pool is small enough" while the code compared `nthreads`. Comment and code therefore
appeared to agree. The ARM grid exercised shapes where the count is raised above the pool
and broke the equality: per `:1587-1592`, "the gated and gated-with-a-ceiling-of-16 arms
must read identically on a six-thread host, and they differed by 0.7%", because the
resolved count "is raised per shape and is bounded by omp_get_num_procs(), which is 18
there". So the 0.7% was the symptom and not the discovery, and the shipped `pool`
comparison is a correction toward what had always been documented. Either way 7..23 does
not transfer: it was computed where a per-shape quantity and a per-call-site one were
assumed equal, and they are not.

**And the range is off by one under either comparator.** Both versions block with
`count > ceiling`, so the gate fires iff `count <= ceiling`, and the ceilings that fire at
6 and not at 24 are **6..23** -- eighteen values, not seventeen. Excluding 6 is defensible
on other grounds (at exactly 6 a per-shape-raised count makes the property
shape-dependent), but it does not fall out of the inequality.

**"Equivalent within the range" is true of the goal and false of the collateral, which
sharpens the point rather than softening it.** Every ceiling in 6..23 separates 6 from 24.
But the number of x86 pools each one makes live *is* the ceiling: 6 exposes pools 1-6, 16
exposes 1-16, 23 exposes 1-23. So the members differ by nearly a factor of four in how much
untested x86 they admit, and pool 8 -- where the effect above is measured -- is admitted
only by ceilings of 8 or more. At a ceiling of 6 or 7 this sweep's treatment cells would be
inert and the measurement could not have been taken. The value is unconstrained by the
purpose it was chosen for *and* fully determines how much unmeasured territory goes live.

That is an account of why the predicate's reach exceeds its evidence, and it is a reading
of the two commits rather than an inference. It is still not a cause of the regression
measured above: it explains why nobody had looked, not why mode 0 is slower when they do.

**One name doing double duty, and it is the distinction the code was changed over.**
`3b676df` calls the ARM figure of 6 a "resolved worker count"; the header at `:1593` calls
the same 6 the "pool the surrounding pipeline runs"; the same passage gives that host's
`omp_get_num_procs()` as 18; and `:1601` calls the host "the M5's 12". And the shipped block defines `pool`
itself as `nthreads_override > 0 ? nthreads_override : omp_get_max_threads()`, a fifth
quantity. So the neighbourhood holds a resolved worker count of 6, a pipeline pool of 6, an
`omp_get_num_procs()` of 18, "the M5's 12", and an `omp_get_max_threads()` fallback -- with
6 labelled two incompatible ways, inside the paragraph whose own subject is that those two
are different and that conflating them cost a falsified version.

For this concern's measurements the ambiguity does not propagate: the harness passes
`nthreads_override`, so its `--pool` is the compared quantity exactly rather than a proxy
for it, and the work proxy in the same block is `nnz * (k > 16 ? k : 16)`, the documented
`nnz*max(k,16)` unchanged from `3b676df` under older variable names. The predicate did not
move when the comparand did.

### The apparatus checked its own premises on these runs

| premise | reading |
|---|---|
| the shim flushed on every arm switch | 32401 flushes against 9720 switches, per width |
| callers pass stable literals | 1 string-compare hit against 1.73e9 by pointer identity |
| the estimator is not carrying the effect | min-of-reps and median-of-reps agree to 0.005 us |
| concern #1's adoption path is inert | `omp_get_num_procs()` capped below 2*pool at every pool |

The one non-literal caller in 1.7 billion lookups is why the cache owns its keys rather
than borrowing them: with a borrowed pointer that single case could have left a dangling
key, and the diagnostic that found it costs nothing.

### How much this matters, measured rather than implied

The gate fires only at pool <= 16, so the question is what pool a caller actually gets.
**It is not a host constant -- it tracks the allocation**, measured on mkt1:

| allocation | affinity | `torch.get_num_threads()` | `omp_get_max_threads()` | gate |
|---|---|---|---|---|
| `-c 8` | 8 | 8 | 8 | **fires** |
| `-c 16` | 16 | 16 | 16 | **fires** |
| `-c 32` | 32 | 32 | 32 | inert |
| `-c 64` | 64 | 64 | 64 | inert |
| `-c 128` | 128 | 64 | 64 | inert |

So the gate fires **with nothing set by the caller** under any allocation of 16 or fewer
CPUs. And the live case is wider than that: this repo's own job scripts record that a
submission with no CPU request at all lands at **pool 2** on this node.
`adg_screen_mkt.sbatch:7-10` says "without a cpus-per-task request Slurm gives the default
allocation and torch.get_num_threads() reads 2 on this node, so OMP_NUM_THREADS cannot help
because the cgroup caps it", and names the artefact left behind when this project made that
mistake -- `/scr/bobbyy/pfs/csvdlmc_pool2_abandoned`, which still exists. So the absence of
a flag is sufficient; no deliberately constrained pool is needed.

Against that, the balancing fact from this file's own materiality clause: **pool 2 is the
weakest cell in the table.** Its four ratios are 1.068, 1.099, 1.077 and 1.086, all below
the 10% bar `dcf2e1f` sets. So the configuration that needs no flag is also the one where
the gate is harmful by a measurement excluding zero and *not* by the standard it was fit
to. Both halves, and only together. An earlier version of this section measured one allocation size, reported
`torch.get_num_threads()` as 64, called that "the default", and concluded the predicate was
unreachable except under a deliberately constrained pool. That under-stated the finding: no
deliberation is required.

What remains true is the narrower claim, and it is checkable from the repo rather than from
a live reading: **every** `#SBATCH` script in the tree requests `-c 32` --
`adg_screen_mkt`, `pfs_screen_fused`, `nt_all392_mkt`, `pfs_screen_dlmc`, `pfs_screen_f64`
and `pfs_screen` -- and the complement, files carrying `#SBATCH` without that directive, is
empty. So those runs sit at pool `min(32, 64) = 32`, above the ceiling, and the gate is
inert in all six. The header gives redwood's pool as 24, likewise above. The runs producing
this project's headline MKL comparisons do not exercise the gate, and this finding does not
move those numbers.

**The two columns are ONE measurement propagated, and the first test of this was vacuous.**
The objection to the table is that importing torch might set the OpenMP count, making the
two columns one reading. The first test -- `omp_get_max_threads()` before and after the
import, reading 8/8 under `-c 8` and 32/32 under `-c 32` -- had no power to answer it:
libgomp's default max-threads *is* `omp_get_num_procs()`, so the value already present
equalled what a torch writing `omp_set_num_threads(torch.get_num_threads())` would have
written. Both hypotheses predict "unchanged" there.

Separated by making the pre-import value disagree with the cgroup. Under `-c 32`, after
`omp_set_num_threads(5)`: `omp_get_max_threads()` reads **5 before the import and 5 after**,
so torch does not write it -- but `torch.get_num_threads()` also reads **5**, so torch
*reads* it. The dependence runs **cgroup -> libgomp -> torch**, a chain. So the columns are
one quantity propagated rather than two observations, and the earlier claim that "all three
track the cgroup independently" is withdrawn.

The finding does not rest on it. `omp_get_num_procs()` equals the allocation *before* any
torch is loaded, which is a cgroup fact with no torch in it, and that is what carries "the
pool tracks the allocation rather than being a host constant". The gate reads one value and
that value is the same under either account, so the boundary is unmoved.

Combined with the prevalence figures above, the exposed population wherever the gate does
fire is 2069 of 4446 SuiteSparse operands at k <= 16 and 1154 of 3820 DLMC matrices, at
6.8% to 62.3% depending on pool.

**Two production paths read two different quantities, and they agree here.**
`scorch_spmm_partition_mode` takes `nthreads_override`, and `prebuilt_kernels.py` declares
`nthreads: Optional[int] = None`, passing the override only when it is not None and calling
`kernel_fn(*args)` otherwise -- in which case the policy receives 0 and `:2612` falls back
to `omp_get_max_threads()`. The comment at `:292` confirms the override is supplied only for
the drop-in SpMM. So the predicate's input differs by call path, and an inertness claim has
to name the quantity per path rather than quote one number. On this host they coincide at
every allocation size measured above, so the boundary is the same either way; on a host where
torch's count and libgomp's diverge it would not be.

### Scope

One host, float32, 120 cells per configuration, isolated allocations where concern #1
cannot fire. Pools 20 and 24 are running: above the ceiling R1 becomes a declined argument,
which supplies a floor on the treatment's own cells, and `8ccd104` already measured that
construction at about 1% on redwood at pool 24 -- so those pools are a replication of a
prior reading rather than a new instrument. Two registered expectations for them: the floor
at or below that 1%, where materially above would indict the shim or the affinity cap
rather than the gate; and the R2 effect below pool 16's 46-53%, per the turn above.

To be plain about what that floor is and is not: above the ceiling the gate is inert in
*both* of R1's arms by construction, so R1 there measures the apparatus on the treatment's
own cells. It is the matched-population floor this concern lacked, and it is not evidence
about the gate's inertness -- that inertness is what makes it a floor in the first place.
R2 is the only live route at those pools.

**Measured, and it is the registered expectation that did hold.** R1 at pools 20 and 24 is
labelled an A/A floor by the harness, from the ceiling predicate rather than by hand, and
reads a maximum absolute ratio departure of **0.08% to 0.21%** across all three populations
and four widths -- against the ~1% that `8ccd104`'s two capped arms measured on redwood at
pool 24. So the shim and the affinity cap add about a fifth of the dispersion that prior
reading had, which was the outcome that would have indicted the apparatus had it gone the
other way. It is also a replication of a construction that already had a reading, on a
different host, rather than a new instrument.

**And the mode comparison corroborates the header's own pool-24 anchor in direction.** That
anchor says the partition wins at pool 24; R2 measures mode 3 winning there by 45.9-51.8%.
The anchor's 0.785 converts to 1.274 in this convention, so the magnitudes differ -- a
different host, and a corpus-pooled figure against gate-region cells -- but the sign is the
same, and that is the one claim in the header that a reader could have taken backwards from
its direction-free ratio.

**What the pool axis therefore shows overall: mode 3 beats mode 0 at every pool measured,
2 through 24.** By 6.8-9.9% at pool 2, 17-23% at pool 4, 56.8-62.3% at pool 8, and 37.9-53.4%
across 16, 20 and 24. The gate forces mode 0 wherever it fires, which is pools 1-16, so it
is harmful across its whole active range and inert above it -- and above it the default is
already mode 3, which is the better arm at those pools too.

**Two ambiguities in the shipped comment itself, both affecting how a later reader uses
it.** At `:1600` the anchor is written "no-partition / partition = 0.785 pooled, 65% of
cells more than 10% behind". Read as a time ratio that says the counter wins, contradicting
the clause it sits inside; read as a performance ratio it agrees. The clause settles it --
"65% of cells more than 10% behind" only parses with the counter behind -- but a later
reader extracts the number, not the clause, and this file's own tables had to state their
direction explicitly for the same reason. Separately, `:1593` gives the ARM host's pipeline
pool as 6 and `:1601` calls it "the M5's 12"; the surrounding text gives that host's
`omp_get_num_procs()` as 18, so three counts are in play under similar phrasing and only
the 6 is the pool the contention argument is about.

Not established: a cause. The explanation attempted above is withdrawn, and the remaining
candidate questions are about `dcf2e1f`'s grid -- its arm labels, and the regressed and
winning cell counts needed to convert its P(region | outcome) scores into the
P(slower | region) measured here. That grid is not in this tree.

---

# The DLMC float32 deficit is ONE WIDTH, and it is the half-vector width

Measured 2026-10-03 from the completed first unit of the live multi-row screen
(`mrhv_screen.sh`, provenance bound in `PREREG_multirow_halfvec.md` amendments 12-13: driver
sha `32c9b0f077040980`, corpus `/scratch/bobbyy/mklsweep/csrcache/dlmc` read from
`/proc/2904667/environ`, key list sha `a8132f50d2c44fe4` at 3820 keys, object
`25a5f0c4425671aa`). Baseline build `mr0`, pass p1 of 4, 30560 cells complete in all three
arms, redwood pool 24.

This is not a new experiment -- it is the **baseline arm** of a screen that was already
running, read for the standing MKL question instead of only for the arm contrast it was built
for. The treatment arms are still being collected.

## The floor comes from the same pass, same process, same object bytes

`ours` and `oursb` are the same object (`25a5f0c4425671aa` for both `mr0` and `mr0b`, checked
by digest, not inferred from the build recipe) timed interleaved in one process. Their ratio
over all 30560 cells: **median 1.0000, p05 0.9394, p95 1.0656**.

So the per-cell floor for a single-pass reading is about -6%/+6.6%. Pooled, it is 1.0000.

## The board

MKL time over scorch time, so above 1 means scorch is faster.

| k | n | geomean | median | p05 | p95 | behind parity | below the floor's p05 |
|---|---|---|---|---|---|---|---|
| 2 | 3820 | 1.3506 | 1.1387 | 0.9791 | 3.1074 | 236 | 147 = 3.8% |
| **4** | 3820 | **1.2508** | 1.0979 | **0.8188** | 3.1106 | **1102** | **676 = 17.7%** |
| 8 | 3820 | 1.3474 | 1.1641 | 1.0187 | 2.9358 | 128 | 60 = 1.6% |
| 12 | 3820 | 1.3465 | 1.1757 | 1.0207 | 2.9002 | 141 | 67 = 1.8% |
| 16 | 3820 | 1.5231 | 1.3398 | 1.1944 | 3.0043 | 11 | 8 = 0.2% |
| 20 | 3820 | 1.5380 | 1.3103 | 1.1265 | 3.0113 | 39 | 29 = 0.8% |
| 24 | 3820 | 1.6543 | 1.4347 | 1.2293 | 3.1121 | 12 | 11 = 0.3% |
| 32 | 3820 | 1.6052 | 1.4363 | 1.2393 | 2.9666 | 18 | 10 = 0.3% |

Overall geomean **1.4455**, behind parity 1687/30560 = 5.5%, below the floor's p05
1008/30560 = 3.3%.

## The null for that last column is 5%, not zero -- and it changes which widths count

A criterion "below the A/A distribution's 5th percentile" flags **5% of cells by
construction** when scorch and MKL are genuinely equal and the only variation is the
variation the A/A measures. Reading the column against zero would have made every width look
like a deficit. Against its own 5%:

| k | below p05 | vs the 5% the floor's tail gives at parity |
|---|---|---|
| 4 | 17.7% | **3.5x over**, 676 observed against 191 expected |
| 2 | 3.8% | under |
| 8, 12, 16, 20, 24, 32 | 0.2-1.8% | all well under |

**Of eight widths, exactly one exceeds the rate its own floor produces at parity, and it is
k=4.** 676 of the 1008 sub-floor cells -- 67% -- are at that single width, which carries 12.5%
of the cells. k=2, the other narrow width and the one whose dispatch rule was closed earlier,
is *under* its null at 3.8%.

## The sub-floor cells are high-degree, which matches the proposed mechanism

| | p05 | median | p95 |
|---|---|---|---|
| deg over the 1008 sub-floor cells | 101.5 | **217.6** | 589.1 |
| deg over the whole corpus | 6.7 | **102.4** | 614.4 |

The deficit lives at roughly **twice the corpus median degree**, and the worst cells are
extreme: 0.2629 at k=4 deg 1024, 0.2896 at k=4 deg 512, 0.3360 at k=4 deg 256 -- 2.7-3.8x
slower than MKL, all `transformer/` and `rn50/` layers at 0.5 pruning, i.e. the *densest* rows
in the collection.

**WITHDRAWN 2026-10-03, and the mechanism it named is not available.** This paragraph read:
"`scorch_simd<T>` has no 128-bit form, so k=4 issues `_mm256_maskload_ps` per nonzero and uses
4 lanes of 8", attributing the deficit to a missing 128-bit path whose absence would scale with
degree. **The 128-bit path exists and already ships.** `scorch_policy.h:1512` sets
`SCORCH_SPMM_HALFVEC_F32` to **1**, and `spmm.h:2200` defines `scorch_simd_half<float>` on
`__m128` with `lanes = 4`. So the single-row half-vector kernel is live in the baseline object
`mr0`, and the k=4 deficit measured below is measured **with it already active**. A missing
128-bit form cannot explain a deficit observed while the 128-bit form is running.

Note the two macros are different and only one ships off: `SCORCH_SPMM_HALFVEC_F32` = 1 at
`:1512` (single-row), `SCORCH_SPMM_MULTIROW_HALFVEC` = 0 at `:662` (the conjunction), the latter
carrying "Ships at 0: the mechanism is new and unmeasured, and the exclusion it lifts was
measured." Conflating them is what produced the withdrawn sentence.

**What the header says is being closed is the CONJUNCTION**, at `scorch_policy.h:655-658`: "the
two mechanisms at this width are independent: the single-row 128-bit kernel drops the mask (11.5
emitted instructions per nonzero against the masked path's 13.5), and multi-row adds streams.
Being unable to have both is the gap this constant closes."

**So the k=4 localisation stands and its cause is open.** The localisation is a geomean ordering
and needs no mechanism; what is withdrawn is the account of why. The degree scaling in the table
below is still a fact about the deficit, no longer evidence for the withdrawn cause.

**This is a located deficit, not a fixed one.** The mechanism is a hypothesis that the
screen's own treatment arms (`mrhv`, `mrhv4`, distinct objects `713b9353f62f55c0` and
`f2ecbf401439f742`) are collected to test; `mr0b` is 41% through the second of four units.
Nothing here licenses a dispatch change, and the one-pass per-cell reading carries the
+-6% floor above.

## Two cautions that bound what this pass can say

**One pass, and it was the pass that led the rotation.** `mr0` ran first and paid a cold page
cache over 3820 `.npz` files: its wall rate was 7.71 rows/s against `mr0b`'s 8.34, an 8.2%
gap. That gap is **entirely outside the kernel** -- summed kernel time over the 37559 matched
cells is 1477.0 s for `mr0` against 1477.5 s for `mr0b`, 0.03% apart -- because each matrix is
loaded once and both arms are then timed inside that load. So the leading position costs wall
clock and not measurements, and the Latin-square rotation is guarding an I/O effect rather
than a timing one. Worth knowing before any future screen spends rotation budget on it.

**The widths are not independent reads of one quantity.** k=2 routes to the narrow-k exact
kernel, k=4 to the half-vector kernel, k>=8 to the register-blocked kernel. Three kernels, so
"one width out of eight" is really "one of three kernels", and the correct comparison for k=4
is the 5% null above rather than the other widths' rates.

## Correction: the "3.5x over its own null" framing is withdrawn, and the repair strengthens the result

A peer showed the 5% null is the at-parity rate for a distribution **centred at the floor's
median**, and none of the scored distributions have that location. The floor is median 1.0000
with log-symmetric tails (-0.06251 / +0.06354), so sigma = 0.03801; k=4's arm is centred at
1.2508, which puts the 0.9394 threshold **-7.53 sigma** below that centre, where the expected
share is ~1e-13 rather than 5%. So the sub-floor count is a statement about the arm's
**dispersion**, not a rate excess over 5%.

**The pooled row showed this and I read past it:** 3.3% observed against a stated 5% null,
while every width sits well above parity. A criterion firing *below* its own null in aggregate,
on data centred at 1.4455, is the signature of a null whose location does not match what it
scores. Withdrawn: "3.5x over its own null", "676 observed against 191 expected", and "exactly
one of eight widths exceeds the rate its own floor produces at parity".

**What never needed a floor, and is the result:** k=4 has the **lowest geomean of the eight
widths**. No threshold, no null, no floor enters that.

### The repair: per-width floor on matched batch divergence

`mrgate.py:55-77` records that the 5-95 spread rises 13.6% -> 19.9% -> 36.6% -> 78.2% as the two
arms' calibrated batch counts diverge, and `:128-129` calibrates MKL separately because "it can
be twenty times a different speed". The floor pair's divergence is zero by construction, so an
unrestricted floor understates the scored arm's noise. Restricting both to
`batch_mkl/batch_ours` within 1.2x scores each width against its own noise:

| k | n | floor p05 | geomean | median | behind parity | below floor p05 |
|---|---|---|---|---|---|---|
| 2 | 1990 | 0.8617 | 1.1036 | 1.1102 | 9.6% | 3.1% |
| **4** | 2139 | 0.8135 | **0.9987** | 1.0382 | **43.7%** | 7.3% |
| 8 | 1628 | 0.9140 | 1.1240 | 1.1336 | 6.4% | 2.5% |
| 12 | 1572 | 0.9226 | 1.1204 | 1.1314 | 7.3% | 3.2% |
| 16 | 462 | 0.9484 | 1.2540 | 1.2532 | 1.5% | 1.1% |
| 20 | 698 | 0.9218 | 1.2138 | 1.2265 | 2.7% | 1.9% |
| 24 | 255 | 0.9112 | 1.2684 | 1.2943 | 2.7% | 2.4% |
| 32 | 175 | 0.8980 | 1.2993 | 1.3080 | 1.7% | 1.1% |

**On the low-noise subset, k=4 is at 0.9987 -- scorch is at parity-or-behind MKL**, with 43.7%
of 2139 cells behind. The gap to the next-worst width *widens* under restriction, 7.7% to 10.5%.

**The restriction is not the population, and the direction is known.** Requiring similar batch
counts selects cells where the two arms run at similar speed, so every geomean falls (k=32
1.6052 -> 1.2993). The restricted figure is therefore **not** an estimate of DLMC k=4
performance; it is the cleanest available contrast, and it is biased toward parity by
construction. Both tables agree on the ordering, which is what the restriction was for.

The `below floor p05` column keeps no 5% interpretation at either restriction, for the location
reason above. It is reported as a tail measure; the ordering is what compares.

## The mechanism, read from the correct tree -- and a 128-bit form DOES exist

An earlier note here recorded that `scorch_simd<T>` has no 128-bit form. **That described the
state before one was added, and the addition is what the screen is testing.** In
`src/scorch/csrc/spmm.h`:

    :2153  template <typename T> struct scorch_simd;        // <float> wraps __m256
    :2200  template <typename T> struct scorch_simd_half;   // <float> uses __m128
    :2196  "...load and throwing away half the FMA width. That is a per-nonzero cost"

So the half-width path exists and is gated behind `SCORCH_SPMM_MULTIROW_HALFVEC`
(`scorch_policy.h:661-662`, default 0), with a `getenv` override at `spmm.h:4333-4334`. The
comment names the same cost the degree scaling in the board shows.

**Two readings in this file's earlier draft came from the wrong tree.** They were run from
`/Users/bobby/scorch`, which is on `refactor/compiler-ir-phase3-std-move-call`, not from the
`scorch-beat-mkl` worktree. Both are corrected above: `scorch_simd` does exist, and the
"`__AVX2__ && __FMA__` with no `!__AVX512F__`" gating claim is withdrawn as read from the wrong
source. Checking the branch is part of reading a path.

## What the four arms are, decoded from the objects rather than from a build log

The `pfs/` trees are byte-identical in every file except the `.so`, and hold no build log, so the
arms are distinguished only by `-D` flags. Decoded statically, with no process started next to
the live screen:

| arm | `.text` | `scorch_spmm_*` symbols | multirow symbols |
|---|---|---|---|
| mr0 | 715033 | 29 | **0** |
| mr0b | 715033 | 29 | 0 |
| mrhv | 735417 | 36 | present |
| mrhv4 | 734921 | 36 | present |

So the baseline has multirow **entirely absent**, not merely configured off -- `mr0` -> `mrhv`
therefore moves multirow and half-vector together and does not isolate either. And **mrhv and
mrhv4 have identical symbol sets**, differing by 496 bytes of `.text`, so they differ by a
compile-time constant rather than by which templates are instantiated; the instantiated set in
both is `<float, {2,3,4}, false, {2,4}, scorch_simd<float>>`.

**One input remains unbound: which constants separate mrhv from mrhv4.** It is resolvable by
importing each object and reading the policy, and that is **deferred until the screen completes**
rather than run now, because four torch imports on the host running a 32-thread timing screen is
a perturbation of a live measurement. Recorded as a gap with a scheduled resolution, not left
implicit.

## Per-unit row counts, the quantity the sentinel drops

| unit | arm | rows | rc | window |
|---|---|---|---|---|
| p1.1 | mr0 | 91680 | 0 | 20:50:29 - 00:08:43 |


## The screen's design: three cells of a 2x2, and the missing one carries both main effects

Correcting the earlier reading here that "the baseline has multirow entirely absent, so
`mr0 -> mrhv` moves multirow and half-vector together". The first half is right and the second
mis-decomposes it. The shipped defaults are two *different* macros:

| macro | `scorch_policy.h` | ships | what it controls |
|---|---|---|---|
| `SCORCH_SPMM_MULTIROW_ROWS` (`_F32` aliases it at `:623`) | `:620` | **0** | multi-row, absent from mr0 |
| `SCORCH_SPMM_HALFVEC_F32` | `:1512` | **1** | single-row 128-bit, **live in mr0** |
| `SCORCH_SPMM_MULTIROW_HALFVEC` | `:662` | **0** | the conjunction |

So `mr0 -> mrhv` does not introduce the half-vector kernel. It introduces the **conjunction**,
which is what `:655-658` says the constant is for. The design is:

| | mrhv = 0 | mrhv = 1 |
|---|---|---|
| rows 0 | **mr0** (+ `mr0b`, byte-identical) | — |
| rows 2 | **MISSING** | **mrhv** |
| rows 4 | — | **mrhv4** |

Three of four cells. The missing `(rows 2, mrhv 0)` is the only arm that makes both main effects
identifiable: `mr0` vs `(2,0)` isolates multi-row, and `(2,0)` vs `mrhv` isolates the
half-vector. It is one `-D`, and it is the configuration this study's own prereg describes at
line 52 as "multi-row with the masked 256-bit kernel taking the width". **Recorded as a design
gap, not requested mid-run** -- the screen is 41% through unit 2 of 16 and will not be disturbed.

Consequence for the result: whatever the screen shows at k=4, the contrast `mr0 -> mrhv` is
attributable to the conjunction as a package and to neither component alone.

## Three corrections to the restricted table's reading

**The widening figure changed comparand, and the fixed-comparand version is larger.** I wrote
"7.7% to 10.5%", but the next-worst width is k=12 in the full table and k=2 in the restricted
one. Holding the comparand at k=12: `1.3465/1.2508` = **7.65%** becomes `1.1204/0.9987` =
**12.19%**. The gap widens more than claimed, so the published version was the conservative one.

**The original pooled threshold moved both ends of the comparison in the headline's favour.** The
per-width floor p05 spans **0.8135 at k=4 to 0.9484 at k=16**, against the single pooled 0.9394
the first scoring used. That threshold therefore sat well *above* k=4's own floor -- counting
more cells than k=4's floor would -- and essentially *at* k=16's, counting fewer than its own
floor would. Inflating k=4 and deflating k=16 is exactly the comparison the headline made. The
caveat belongs with it: those p05 values are measured on the restricted subpopulations, so this
establishes the direction and is not a measurement of the original mis-scoring. The geomean leg
is untouched by all of it.

**The restriction selects at very different rates by width, which is a second bias beyond the
toward-parity one.** Retained n runs from 175 at k=32 (4.6% of 3820) to 2139 at k=4 (56.0%), a
12x range. So the cross-width comparison is between populations selected at very different
rates, and "both tables agree on the ordering" is strongest where n is largest -- k=4 at 2139 --
and thinnest at the right-hand end, where k=24 (n=255) and k=32 (n=175) carry much wider
intervals than the ordering statement implies.

**Why the restriction is biased toward parity, stated once properly.** Requiring
`batch_mkl/batch_ours` within 1.2x keeps cells where the two arms' calibrated repeat counts are
similar, and `mrgate.py:46-50` derives that count from each arm's own measured call time. So the
filter conditions on the arms running at similar speed, which is a function of the very ratio
being scored. Every geomean falls under it (k=32 1.6052 -> 1.2993) because cells where we are
much faster are preferentially removed. It buys a floor matched in dispersion at the cost of
conditioning on the outcome -- acceptable for a within-width noise comparison, not for a
population estimate, and the reason the full table stays the headline figure.

---

# The k=4 shortfall is a DEGREE effect, confirmed on both collections

Measured 2026-10-03. The earlier section here localised a DLMC float32 deficit at k=4 and left
its cause open. Two further readings change the finding's scope in both directions: it is **not**
specific to DLMC, and it is **not** present at low degree.

## First, the reading that looked like a refutation

SuiteSparse float32, 22920 cells over 4584 matrices, min over the six passes of `c196`'s `sh_c0`
set, A/A floor (oursb/ours) median 1.0000 with p05 0.9764:

| k | geomean MKL/ours | behind parity |
|---|---|---|
| 2 | 1.7351 | 19/4584 = 0.4% |
| 4 | 1.9346 | 39/4584 = 0.9% |
| 8 | 2.3038 | 12/4584 = 0.3% |
| 16 | 2.7101 | 3/4584 = 0.1% |
| 32 | 2.9065 | 3/4584 = 0.1% |

**k=4 is not the worst width here -- k=2 is -- and the curve is monotone in k.** Taken at face
value that says the DLMC k=4 result does not generalise, which under this project's standard
would make a k=4 change an overfit to one collection.

## It was a composition effect, and the dip is in both corpora

The statistic that settles it is **paired within each matrix**: for every key, the log of its k=4
ratio minus the mean of the logs of its k=2 and k=8 ratios. Each matrix's own level cancels, so
this measures the *shape* of the width curve and not how fast the matrix is. Bootstrapped over
matrices, 4000 resamples:

| degree band | SuiteSparse n | dip, 95% CI | DLMC n | dip, 95% CI |
|---|---|---|---|---|
| 0-32 | 4167 | -2.82% [-3.40, -2.22] | 1051 | **+0.95%** [+0.61, +1.31] |
| 32-100 | 308 | -6.13% [-7.58, -4.60] | 781 | -2.39% [-3.01, -1.74] |
| 100-256 | 58 | **-11.23%** [-14.13, -8.57] | 1275 | **-11.72%** [-12.55, -10.85] |
| 256+ | 51 | **-9.66%** [-12.45, -6.80] | 713 | **-15.59%** [-17.25, -13.91] |

Sign counts at 100-256 are 51 of 58 and 1083 of 1275. Every interval excludes zero, the two
corpora's intervals **overlap** at 100-256, and the effect is monotone in degree on both.

**Why the aggregate hid it:** 90.9% of SuiteSparse keys have mean degree below 32, where the dip
is small, against 52.1% of DLMC keys at degree >= 100, where it is -11% to -16%. The pooled
SuiteSparse number is an average over a population that is almost entirely outside the affected
region. So the previous paragraph's "k=4 is not the worst width" is true of the collection and
false of the mechanism.

| | deg 0-32 | 32-100 | 100-256 | 256+ |
|---|---|---|---|---|
| SuiteSparse share of keys | 90.9% | 6.7% | 1.3% | 1.1% |
| DLMC share of keys | 27.5% | 20.4% | 33.4% | 18.7% |

**This is the two-corpus confirmation the no-overfitting standard asks for.** The population is
named by a structural property of the matrix (mean nonzeros per row), the width is fixed by the
SIMD lane count rather than chosen (`scorch_simd_half<float>::lanes == 4`), and the same effect
with overlapping intervals appears in two corpora collected for different purposes.

## In absolute terms, which is what the MKL goal is about

The dip is a shape statistic; the goal is parity. Geomean MKL/ours at k=4:

| | deg 100-256 | deg 256+ |
|---|---|---|
| SuiteSparse | 1.1894 | 1.0853 |
| DLMC | 1.0198 | **0.9823** |

So k=4 at high degree is the weakest cell on both collections, and on DLMC above degree 256 it
is **below parity**. Everywhere else both collections are comfortably ahead.

## Two caveats that bound the claim

**The interpolation baseline mixes three kernels.** float32 k=2 is owned by the narrow-k exact
kernel, k=4 by the half-vector kernel (`HALFVEC_F32=1`, taken when `B1_size == lanes == 4`), k=8
by the register block. So the dip is k=4's kernel measured against the geometric mean of two
*other* kernels, and it cannot by itself separate "k=4 is bad" from "k=2 and k=8 are both good".
The absolute table above is what carries the parity claim; the dip is what shows the effect is
degree-dependent and cross-corpus.

**The two boards differ in method.** SuiteSparse is min-over-six-passes with a p05 floor of
0.9764; DLMC is a single pass with 0.9394. The dip is computed within a board and within a degree
band, so the method difference moves levels rather than the shape -- but the SuiteSparse
high-degree bands rest on 58 and 51 matrices, and that is the thinnest part of the evidence.

## What this implies for the open cause, without closing it

A prior half-vector study on this host measured the **multi-row plus half-vector conjunction** at
+12.95% to +39.28% on DLMC float32 at degree >= 256, n=712 -- the same population, the same
width, and the band where the dip is largest. That conjunction is
`SCORCH_SPMM_MULTIROW_HALFVEC`, which ships at 0 (`scorch_policy.h:662`, "the mechanism is new
and unmeasured"), and it is what the running screen is collecting. The already-shipping
single-row half-vector (`HALFVEC_F32=1`) does not close the dip, which is what the measurements
above establish.

**Still not a mechanism.** What is established is the population and the width; what the
conjunction would need in order to ship is the screen's own result plus the missing
`(rows 2, halfvec 0)` arm, since without it `mr0 -> mrhv` attributes to the package and not to
either component.

## Scoping the cross-corpus claim: the SIGN of the degree trend is shared, the mechanism is not

Three corrections to the section above, two of them material. The headline that survives is
narrower than what was written.

### "The two corpora overlap at 100-256" is true of one band in four

| band | SuiteSparse CI | DLMC CI | relation |
|---|---|---|---|
| 0-32 | -3.40 .. -2.22 | +0.61 .. +1.31 | **disjoint, OPPOSITE SIGN** |
| 32-100 | -7.58 .. -4.60 | -3.01 .. -1.74 | disjoint |
| 100-256 | -14.13 .. -8.57 | -12.55 .. -10.85 | DLMC **contained** in SuiteSparse |
| 256+ | -12.45 .. -6.80 | -17.25 .. -13.91 | disjoint |

Three of four disjoint, and at low degree the sign flips (-2.82% against **+0.95%**). So the
collections share the **monotonicity in degree, not the level** -- the curves are offset and cross
zero in different places. A shared sign of the derivative is what a mechanism predicts and is
what the conclusion needs, but the sentence "the two corpora's intervals overlap" invited the
stronger reading and is corrected here. At 100-256 the right word is **containment** (1.70 pp
inside 5.56 pp), which is the stronger claim -- weakened by the SuiteSparse band being n=58, so
overlap between a wide interval and a narrow one is mostly a statement about the wide one.

### The material correction: only DLMC shows k=4 as the worst-degrading width

Each width's geomean normalised to its own deg 0-32 value, so 1.000 means that width does not
degrade with degree:

| | k=2 | k=4 | k=8 | k=16 | k=32 | nbr geo | k=4/nbr |
|---|---|---|---|---|---|---|---|
| **SuiteSparse** 32-100 | 0.816 | 0.697 | 0.638 | 0.652 | 0.650 | 0.722 | 0.966 |
| 100-256 | 0.781 | 0.592 | 0.538 | 0.567 | 0.529 | 0.648 | 0.913 |
| 256+ | 0.682 | 0.540 | 0.495 | 0.476 | 0.455 | 0.581 | 0.930 |
| **DLMC** 32-100 | 0.678 | 0.660 | 0.687 | 0.709 | 0.797 | 0.683 | 0.967 |
| 100-256 | 0.616 | 0.539 | 0.618 | 0.704 | 0.830 | 0.617 | 0.874 |
| 256+ | 0.587 | **0.520** | 0.657 | 0.694 | 0.785 | 0.621 | 0.836 |

**Every width degrades with degree on both collections**, so "the dip grows because the
neighbours improve" is refuted outright -- that was the live alternative and it is dead.

But the two corpora then disagree on what is special. **On DLMC, k=4 degrades fastest of all five
widths** (0.520 against 0.587 / 0.657 / 0.694 / 0.785), which is what a k=4-specific
degree-dependent cost predicts. **On SuiteSparse it does not**: k=8, k=16 and k=32 all degrade
*more* than k=4 (0.495, 0.476, 0.455 against 0.540), and the outlier is **k=2 being unusually
robust** at 0.682. So on SuiteSparse the dip is produced by k=2 holding up, not by k=4 collapsing.

**Consequence, stated plainly: the claim "this is the two-corpus confirmation the no-overfitting
standard asks for" was too strong and is withdrawn in that form.** What is confirmed on both
corpora is that k=4 falls below the interpolation of its neighbours and that the gap grows with
degree. What is confirmed on **DLMC only** is that k=4 is the width responsible. A dispatch change
keyed on k=4 therefore still rests on one collection for its mechanism, and SuiteSparse
contributes the degree dependence rather than the width attribution. The absolute parity numbers
are unaffected.

### The reduction is a geometric mean, so subgroup arithmetic is valid here

Checked rather than assumed, because this project's `red` helper is `exp(median(log v))`, which
does not decompose. The n-weighted reconstruction of the pooled k=4 geomean from the four bands
is **exact to four decimals on both corpora** -- 1.9346 against 1.9346 (SuiteSparse, n=4584) and
1.2508 against 1.2508 (DLMC, n=3820). The implied deg<100 figures follow: 1.9597 over n=4475 and
**1.5839 over n=1832**, so k=4 at low degree on DLMC is within 2% of that collection's best width.

### 712 against 713 is ONE KEY and not a boundary convention -- but the convention moves 26%

The earlier half-vector study reports n=712 for "DLMC float32, deg >= 256"; the band partition
here gives 713. Resolved: the bands are `lo <= deg < hi`, and on DLMC

    keys with deg == 256.0 exactly : 186
    keys with deg >= 256           : 713
    keys with deg > 256            : 527

so neither convention yields 712, and the discrepancy is a single key the earlier filter dropped
rather than a predicate difference. It is left as a one-key gap, not reconciled.

**The convention itself matters far more than the off-by-one.** 186 of 713 keys sit at exactly
degree 256, so `>=` against `>` moves this population by **26%**. Any claim stated as "deg >= 256"
on this corpus must name the side the boundary belongs to; the n summing to the corpus total
proves the bands partition but not which convention they use.

## The 712/713 gap is RESOLVED: a rounded covariate, not a dropped key

`mrgate.py:143` writes `deg="%.2f" % (nnz / max(M, 1))`, so a band computed from the CSV column
bins a **rounded** degree while one recomputing `nnz/rows` bins the exact ratio. With keys sitting
on the boundary value, that is enough. Binned both ways on the same file:

| reading | >= 256 | > 256 | == 256.0 |
|---|---|---|---|
| CSV `deg` column (rounded to 2dp) | **713** | 527 | 186 |
| recomputed `nnz/rows` (exact) | **712** | 527 | 185 |

Exactly one key flips, and it is the expected shape:

    transformer/magnitude_pruning/0.5/body_decoder_layer_2_encdec_attention_
      multihead_attention_k_fully_connected
        column 256.00      exact 255.9980468750

So the earlier study's n=712 computed degree exactly and this section's 713 binned the rounded
column. **No key was lost**, and "the earlier filter dropped one key" is withdrawn. Band n under
the exact reading: 1051 / 781 / 1276 / 712.

The 26% convention exposure is unchanged and is the larger point: **185 keys have `nnz/rows`
exactly 256** -- integer `nnz` being an exact multiple of `rows`, which pruned power-of-two layers
produce in quantity -- so `>=` against `>` still moves this population from 712 to 527. A
covariate written through `%.2f` and a boundary landing on a value 185 matrices hit exactly are
two independent reasons to state the predicate precisely.

## Correction to the normalised table: the `k=4/nbr` column is not the dip

Every width in that table is normalised to **its own** deg 0-32 value, so a ratio taken inside it
inherits the normalisation. `k=4/nbr` is therefore the dip **relative to the deg 0-32 band's own
dip**, not the dip. Read as the dip it understates by up to 2.7 pp:

| SuiteSparse band | `k=4/nbr` read directly | the actual dip |
|---|---|---|
| 32-100 | -3.40% | -6.13% |
| 100-256 | -8.67% | -11.23% |
| 256+ | -7.06% | -9.66% |

The relation is `(1 + dip_band) = (1 + dip_{0-32}) x (k=4/nbr)`, which recovers the published
figures to **0.03 pp** -- the "0.01 pp" an earlier revision claimed here is wrong, and is corrected
in place. Recomposed from the printed two-decimal figures with `dip_{0-32} = -2.82`:

| band | `k=4/nbr` | recomposed | published | gap | own input envelope |
|---|---|---|---|---|---|
| 32-100 | -3.40 | -6.1241 | -6.13 | 0.0059 | +/-0.0097 (within) |
| 100-256 | -8.67 | -11.2455 | -11.23 | **0.0155** | +/-0.0094 (**exceeds**) |
| 256+ | -7.06 | -9.6809 | -9.66 | **0.0209** | +/-0.0095 (**exceeds**) |

Two of three rows exceed the old tolerance, and the same two exceed the envelope their own printed
inputs carry over all four corners at +/-0.005. Either the claim is about unrounded internals, in
which case it has to say so, or the tolerance is 0.03 pp. It now says 0.03. Peer reviewer's
recomputation. The column is kept because the normalised view is what separates the widths'
trends, but its base is now named beside it.

## The disagreement as a rank reversal, which is sharper than "what is special"

Ranking k=4's degree-robustness among the five widths in the top band:

| | normalised values, k=2..k=32 | k=4's rank |
|---|---|---|
| SuiteSparse | 0.682 0.540 0.495 0.476 0.455 | **2nd most robust of 5** |
| DLMC | 0.587 0.520 0.657 0.694 0.785 | **least robust of 5** |

And the log-step sequence across widths at deg 256+:

    SuiteSparse  -0.2335  -0.0870  -0.0391  -0.0451   monotone decline, no local minimum
    DLMC         -0.1212  +0.2339  +0.0548  +0.1232   local minimum at k=4, trend REVERSES

**The width-versus-robustness trend above k=4 has opposite sign on the two collections.** That is
a larger disagreement than which width is responsible, and it bounds what SuiteSparse contributes:
"every width degrades with degree, k=4 included and not worst" is a fact about the kernel family,
not evidence for k=4.

## The same ranks on the {2,4,8} subset, computed before MKT's result lands

MKT's screen measures three widths, {2, 4, 8}. Restricting the two rows above to those three
columns, so that a three-width rank from MKT is never cross-quoted against a five-width one.
**Independently co-registered**: the peer reviewer registered the same restriction first, in the
same six figures and with the same cross-quoting warning, and recomputed the ranks under the
higher-is-better reading as SS [2,4,8] and DL [8,2,4]. Two independent derivations agreeing digit
for digit is worth more than either alone, and this paragraph is not a correction to theirs:

| | normalised values, k=2 / k=4 / k=8 | k=4's rank |
|---|---|---|
| SuiteSparse | 0.682 / 0.540 / 0.495 | **2nd most robust of 3** |
| DLMC | 0.587 / 0.520 / 0.657 | **least robust of 3** |

The rank reversal survives the restriction, so **the registered prediction for MKT's SuiteSparse
phase is k=4 ranking 2nd of 3, not last of 3.** If MKT reads k=4 last of three on SuiteSparse that
is a between-host disagreement on the same collection -- a new finding, not a confirmation of the
DLMC one.

**What is base-robust here and what is not.** These values are normalised to each width's own deg
0-32 band. Recomputing DLMC's ordering from a different base *and* the opposite convention -- the
`chain207-dlmc-redwood` degree-band table, median ours/mkl where lower is better, deg>=256 over
deg 16-64 -- gives k=8 1.0743, k=2 1.0755, k=4 1.2154. Same ordering, so **k=4 last on DLMC is
base-robust**. The k=2-versus-k=8 ordering is **not**, and the separation is stated in one unit
because the two bases use opposite conventions: as a **difference** between the two widths' values
it is 7.000 pp under this base against 0.120 pp under that one, **58x**; as a **ratio minus one**,
11.93% against 0.112%, **107x**. Either form supports the conclusion. A mixed pair -- "7.0 pp
against 0.11%" -- reads as 64x, which is neither, and a later citation would pick whichever
reading suited it. Only k=4's position should be carried across bases.

### One feature of the SuiteSparse sequence a width-specific account does predict

The monotone decline is **not uniform**: the k=2 -> k=4 step is -0.2335 against the next step's
-0.0870, a factor of **2.7**. So SuiteSparse does carry a discontinuity at exactly the boundary
where the narrow-k exact kernel hands off to the half-vector one, and DLMC's reversal is at the
same boundary. That is the one cross-corpus feature located at a kernel transition rather than at
an arbitrary width.

It does **not** restore the attribution, because it reads equally as "the exact kernel is
unusually robust to degree" -- and on SuiteSparse that is the better description, since k=4 is
second-most robust there. But the previous section's withdrawal was slightly harsher than the data
requires, and this is the part of the cross-corpus content that survives: an anomalous step at the
exact-to-half-vector boundary on both collections, with opposite sign of everything above it.

### Why the surviving agreement is partly built into the statistic

The paired contrast is constructed from k=2 and k=8 -- the two widths whose behaviour differs most
between the collections. On SuiteSparse k=8 degrades **more** than k=4 (0.495 against 0.540), which
produces a negative dip with k=4 behaving normally; on DLMC k=8 degrades **less** (0.657 against
0.520), which produces a negative dip because k=4 collapses. **Same dip sign, opposite cause**, and
the sign agreement is what the construction guarantees once both neighbours move.

So "the dip grows with degree on both corpora" is true and is weaker support for a shared mechanism
than it reads. The general form: **when a contrast agrees across populations while its components
disagree, the statistic's construction is a candidate for what is doing the agreeing** -- check the
components before reading the contrast as replication.

### The band CIs recomputed on exact binning, so the two tables share one partition

Rather than caveat the n mismatch, both bins were run. The dip CIs are unchanged to within
0.04 pp, so nothing in the finding depended on the convention -- but the exact column is now the
published one and the agreement is a measurement rather than an assumption:

| band | rounded column | exact `nnz/rows` |
|---|---|---|
| 0-32 | n=1051, +0.95% [+0.61, +1.31] | n=1051, +0.95% [+0.61, +1.31] |
| 32-100 | n=781, -2.39% [-3.01, -1.74] | n=781, -2.39% [-3.01, -1.74] |
| 100-256 | n=1275, -11.72% [-12.55, -10.85] | n=1276, -11.72% [-12.59, -10.85] |
| 256+ | n=713, -15.59% [-17.25, -13.91] | n=712, **-15.60%** [-17.24, -13.90] |

**The flipping key has a checkable fingerprint**, which is what makes the single flip a near-miss
rather than a suspicious coincidence: `rows = 512`, `nnz = 131071 = 2^17 - 1`, so
`nnz/rows = 255.998046875 = 256 - 1/512` exactly, and `nnz == 256*rows - rows/512` holds. For that
ratio to arise with integer `nnz`, `rows` must be a multiple of 512 -- so the value is a real
structural ratio from a pruned power-of-two layer one nonzero short of dense-256, not a formatting
artefact.

### The k-set discriminator is enforced by the driver, not merely true of one run

A sealed ledger states the float32/float64 discriminator as a property of the concurrent run. It is
structural: `arm_screen.sh:92` sets `LANES=4` for float64, `:120` computes `hi = 4 * lanes` = 16,
and `:123` collects `over = [k for k in ks if k > hi]` and **refuses**. So no float64 grid on this
driver can contain 20, 24 or 32 at all, and "presence of {20,24,32} implies float32" fails closed
rather than holding by coincidence. The sets are not disjoint -- intersection {2,8,12,16}, float32
only {20,24,32}, float64 only {4,6,10} -- so the implication runs one way only.

## State of the k=4 question, with the shared feature and the disagreement on separate lines

Written out separately because a reader who takes the step as shared evidence will otherwise take
the sign as shared too.

**Established on both collections, absolute:** k=4 at high degree is the weakest cell --
SuiteSparse 1.1894 (deg 100-256) and 1.0853 (256+); DLMC 1.0198 and 0.9823, the last below parity.

**Established on both collections, shape:** an anomalous step at the k=2 -> k=4 boundary, which is
where the narrow-k exact kernel hands off to the half-vector kernel. On SuiteSparse it is a step
2.7x the next one in an otherwise monotone decline; on DLMC it is a sign reversal.

**A DISAGREEMENT, not a shared feature:** the width-versus-robustness trend *above* k=4 runs
opposite directions -- SuiteSparse declining (-0.0870, -0.0391, -0.0451), DLMC rising (+0.2339,
+0.0548, +0.1232). k=4 ranks 2nd most robust of five widths on SuiteSparse and least robust of five
on DLMC.

**Established on DLMC only:** that k=4 is the responsible width.

**What SuiteSparse contributes:** that every width degrades with degree, k=4 included and not
worst. That is a fact about the kernel family, not evidence for k=4.

**Established nowhere:** the mechanism. And `mr0 -> mrhv` cannot supply it, since the conjunction
moves as a package with the `(rows 2, halfvec 0)` arm missing.

### Correction: "max movement 0.04 pp" compared two separate bootstrap runs

The two CI columns above were produced by two bootstrap calls. Both used seed 11, but the bands
differ in n (1275/1276 and 713/712), so `randrange(n)` consumes different streams and the draws are
**not matched** -- the difference between the columns is the key's effect *plus* resampling jitter,
and the table did not separate them. The tell was inside the table: band 100-256 **gained** a key
while its lower bound moved *down* 0.04 pp and its point estimate did not move at all, which is
the signature of resampling variation rather than of one added observation.

**Both quantities measured separately.** The key's effect needs no bootstrap at all, because the
dip is a **mean** of per-key log contrasts, so adding or removing one observation moves it exactly:

| band | without the key | with it | exact shift |
|---|---|---|---|
| 256+ | n=712, -15.599% | n=713, -15.594% | **+0.0052 pp** |
| 100-256 | n=1275, -11.723% | n=1276, -11.723% | **-0.0001 pp** |

And the jitter, measured on **identical** data (exact binning, band 256+, five seeds, nboot=4000):

    seed 11  [-17.241, -13.896]      seed 14  [-17.253, -13.959]
    seed 12  [-17.334, -13.885]      seed 15  [-17.236, -13.885]
    seed 13  [-17.211, -13.900]
    spread:  lower 0.123 pp   upper 0.074 pp

**The jitter exceeds the 0.04 pp difference I published**, so that figure is withdrawn as a
statement about the key. The key's actual effect is 0.005 pp, roughly a twentieth of the method's
own run-to-run spread. The conclusion is unchanged and better stated as a bound: one key in ~1000
moves these intervals by less than the bootstrap's own reproducibility, and the point estimates
moving 0.00 and 0.01 pp is exactly what one observation of ~1000 must produce.

**The moved key's own contrast, which is what the band tables cannot show:**

    r_2 1.0236   r_4 0.9753   r_8 1.1954      contrast -11.833%

So this matrix is itself a clean instance of the effect -- ahead of MKL at k=2 and k=8, **behind at
k=4** -- and its contrast nearly equals the 100-256 band's average of -11.723%, which is why
adding it there moves the mean by 0.0001 pp.

### A reporting defect this exposes in every CI above

Endpoints that move 0.123 pp across seeds cannot be reported to two decimals. The published
intervals print figures like `[-17.24, -13.90]` whose last digit is not reproducible.

**My first repair -- "read them to 0.1 pp" -- is wrong, and the five seeds above refute it:**

    lower  -17.241 -17.334 -17.211 -17.253 -17.236  ->  -17.2 -17.3 -17.2 -17.3 -17.2
    upper  -13.896 -13.885 -13.900 -13.959 -13.885  ->  -13.9 -13.9 -13.9 -14.0 -13.9

Two distinct values at **each** endpoint, and the upper one crosses a whole unit in the first
decimal at `-13.959`. **A rounding quantum equal to the spread leaves the last digit flipping; the
quantum has to EXCEED it**, conventionally by 2-3x -- here 0.5 pp.

**The better form carries the jitter instead of hiding it:** `-17.24 +- 0.12 across seeds`, which
is strictly more informative than any rounding because it reports the method's reproducibility
rather than concealing it. Otherwise pin the resampling with an index matrix.

**Resampling harder is not the escape, and the arithmetic is worse than I implied.** Jitter falls
as 1/sqrt(nboot), so 100x the draws (400k) buys 0.0123 pp -- which still exceeds the 0.01 pp
printing quantum. A *reproducible* second decimal needs the spread under half a unit in the last
place, 0.005 pp, which is about **605x = 2.4M draws**. So "~100x" was the right factor for
0.0123 pp and badly understates the cost if read as "then two decimals would be fine". Report
coarser; do not resample harder.

The per-band dips, their signs, the band ordering and the rank reversal are all unaffected. At
0.5 pp granularity every interval in this finding is stable and every conclusion drawn from one
still holds. What was wrong is printed precision the method never had to offer. See the note on an
index matrix being what pins a bootstrap bound -- a seed plus nboot does not.

### The degenerate back-derivation, which is sharper than calling one branch lucky

Recovering the moved key's own contrast from the published 2-decimal band figures was tried two
ways: removing it from band 256+ implied -8.16%, adding it to 100-256 implied -11.72%, against the
truth of -11.833%. I attributed the near-agreement of the second to the key's contrast nearly
equalling that band's mean.

**It is stronger than luck: that branch had no content at all.** When the published before and
after values are *equal* -- as they are at -11.72% either side, the band moving by -0.0001 pp --
the back-derivation returns the reported mean **identically**. It is an algebraic identity, not an
estimate, so its 0.11 pp accuracy measures only how near the key sits to the band mean, which is
the same fact as the band not moving. **Check whether a derivation degenerates on the inputs
available before reading its output as an estimate.**

## Pending: the two-host confirmation for the width attribution

The width attribution is DLMC-only and single-host. MKT can test it -- the half-vector block is
`__AVX2__ && __FMA__`, which Zen4 satisfies, so the kernel is live there on a different ISA
(AVX-512 vs redwood's AVX2). Job **17701493** submitted 2026-10-03, measuring k in {2,4,8} at 9
reps over **both** collections from MKT's own caches (ss 4594, dlmc 3820), using the existing
release build `ship` (`4c11296ce2a68c3d`).

Checked before submitting, because a hooks build would make the MKL columns unreadable rather than
merely noisy: the object carries **one** `SCORCH_SPMM_*` string (`SCORCH_SPMM_ATPARALLEL`), not the
dozens a `SCORCH_TUNE_HOOKS` build carries, and ambient `SCORCH_*` is empty under the allocation.
The key lists are generated from each cache by `find`, so the group-qualified `ss` layout cannot
silently drop keys -- and `mrgate.py`'s `resolve_all` refuses on any unresolved key rather than
proceeding short.

## The two hosts' SuiteSparse corpora are NOT the same corpus

Checked **before** reading job 17701502's SuiteSparse result rather than after, because the DLMC
half had a byte-identical key-list digest across hosts and the SuiteSparse half had no counterpart
-- so "different corpus" was not excluded as an explanation for any cross-host disagreement.

**Key path structure differs, so comparing key strings across hosts is meaningless.** Redwood's
cache stores `<name>/<name>`; MKT's is group-qualified, `<GROUP>/<name>/<name>`. A naive set
comparison of the generated lists reports 916 common of ~4590, which is an artefact of the layout
and not a corpus statement. The identity is the matrix **basename**:

| | redwood | MKT |
|---|---|---|
| keys | 4584 | 4594 |
| distinct basenames | 4584 | **4578** |
| common basenames | **4578** | |
| redwood-only | **6** | 0 MKT-only |

**Redwood has 6 matrices MKT lacks:** `bone010_B`, `bone010_C`, `bone010_M`, `boneS10_B`,
`boneS10_C`, `boneS10_M`. The published SuiteSparse board covers 4584 and therefore includes
these; MKT cannot measure them.

**MKT has 16 extra PATHS over 15 basenames**, matrices mirrored under two collection groups:

    Nasa/barth + Pothen/barth          Boeing/nasa1824 + Nasa/nasa1824
    Nasa/barth4 + Pothen/barth4        Boeing/nasa2910 + Nasa/nasa2910
    Nasa/barth5 + Pothen/barth5        Boeing/nasa4704 + Nasa/nasa4704
    GHS_indef/copter2 + GHS_psdef/copter2    Andrianov/{ex3sta1,fxm3_6,fxm4_6,pf2177}
    Newman/football + Pajek/football          + Meszaros/{same four}
    Nasa/shuttle_eddy + Pothen/shuttle_eddy   Nasa/skirt + Pothen/skirt
    pwt: GHS_psdef/pwt + Nasa/pwt + Pothen/pwt   (three paths)

14 pairs plus one triple = 31 paths for 15 basenames = the 16 surplus.

### Three consequences for reading the MKT result

1. **The cross-host comparison must be restricted to the 4578 common basenames.** Comparing a
   4584-matrix board against a 4594-path one would differ by 22 keys for reasons that have nothing
   to do with the kernel or the ISA.
2. **MKT's 16 duplicate paths must be deduplicated before any per-matrix statistic**, or 14
   matrices carry double weight and `pwt` triple. The dip is a mean over matrices, so duplicates
   bias it toward whatever those 15 do. Whether the mirrored pairs are byte-identical is checkable
   from the emitted `rows`/`nnz` columns at no cost and is deferred to the analysis.
3. **This is the exact case `csrresolve.py` refuses on.** Its resolution is exact-path then
   *unique* basename, and on MKT 15 basenames are not unique -- so a flat corpus list would be
   ambiguous for them. The job passes exact paths generated from the cache, so it resolves; the
   refusal would have fired on a hand-written flat list.

**The completeness check when it lands is 9 rows per key, not the 75,726 total.** A total is
satisfied by one key with 6 rows and another with 12, which is the defect already on record in
this project's own sentinel.

### And the ISA guards are not interchangeable

Worth separating because they sit in the same file family and give opposite answers on Zen4:

| guard | condition | Zen4 |
|---|---|---|
| `spmm.h:2139` (half-vector kernel) | `__AVX2__ && __FMA__` | **true** |
| `scorch_policy.h:1451` (narrow-k W1 unroll mindeg) | `__AVX2__ && __FMA__ && !__AVX512F__` | **false** |

So the half-vector mechanism is live on MKT and the W1-unroll mechanism is inert there by
construction. Cite the line, not "the AVX2 gate".

### Correction: `comm` under a different collation than the `sort` fabricated three numbers

The "916 common of ~4590" above was wrong, and so was every figure derived from it. The lists were
written with `LC_ALL=C sort`, but `comm` ran under the shell's `en_US.UTF-8`, whose collation
reorders punctuation -- so the merge desynchronised and reported keys as one-sided that `grep`
finds in both files.

| comparison | UTF-8 `comm` | `LC_ALL=C comm` |
|---|---|---|
| full-list common | 916 | **1226** |
| redwood 3-component vs MKT | 931 common, 295 rw-only | **1226 common, 0 rw-only** |
| basenames: common / rw-only / MKT-only | 4578 / 6 / 0 | **4578 / 6 / 0, unchanged** |

**All 1226 of redwood's group-qualified keys exact-match MKT's** -- 0 of them one-sided. So the
"295-310 group-name disagreements" that the wrong number implied **do not exist**, and the
remaining differences are only the two already recorded: 6 basenames and 16 duplicate paths.

> **CORRECTED.** This first read "1:1 with nothing left over", which is false and was not
> supported by what I printed. I reported `comm -12` and `comm -23` and **never printed
> `comm -13`** -- the MKT-only count, which is the column that distinguishes a real cross-dataset
> match from a self-comparison, because a self-comparison also zeroes the left-only side. It is
> **3368**: 4594 MKT paths minus 1226 matched, exactly as it must be, since redwood's other 3358
> keys use forms that cannot exact-match a uniformly-qualified list. So the true claim is
> **one-sided containment** -- redwood's qualified subset is contained in MKT's key set -- not a
> bijection.
>
> **And the rule I first wrote from this -- "print all three `comm` sides, or the output cannot
> rule out what the second input was" -- is itself false.** The three sides satisfy two identities,
> `left_only + common = |A|` and `right_only + common = |B|`, so the third column adds exactly one
> new number, `|B|`, and nothing else. "Exactly as it must be" was the concession. Against the
> three failure modes: an **empty** second input is excluded by the common count being 1226 rather
> than 0, not by the third column; a **substituted** second input with 4594 lines and 1226 matches
> produces identical output in all three columns; and a **self-comparison** is separated only by
> `|B|`, 4584 against 4594, which is identification by **cardinality, not identity** -- the same
> weakness as a log line reading `corpus 3820 matrices`. It is a partition identity of the same
> shape as `ahead = n - behind - band`, where the total summing correctly let a misrouted cell
> through.

**The basename comparison -- which the finding above rests on -- is unchanged.** It was correct,
and is now correct for a checkable reason rather than by luck.

**The tell was an internal contradiction, not a suspicious value.** Redwood's 3-component keys are
a *subset* of its full list, yet reported 931 matches against the full list's 916. A subset cannot
match more than its superset, and that is visible without knowing which number is right. Both
numbers were individually plausible: 916 of ~4590 reads like a layout mismatch, 931 like the same
thing measured slightly differently.

**The rule: `comm`, `join` and `uniq` are merge algorithms whose correctness depends on the input
order matching their own comparison.** Setting the locale on `sort` and not on the consumer is
silently wrong in both directions. Set it on both, or verify with a membership test -- `grep -c
'^key$'` on each file -- which is order-independent and is what exposed this.

### Redwood's cache has three key forms, so "redwood stores `<name>/<name>`" described 39% of it

From the sealed dims census (`chain209-dlmc-dims/ss_dims.csv`, 4584 rows), counting path components:

| form | example | n | share |
|---|---|---|---|
| group-qualified | `ss/AG-Monien/3elt/3elt` | 1226 | 26.75% |
| flat | `ss/barth4/barth4` | 1772 | 38.66% |
| group dir + suffixed name | `ss/bone010/bone010_M` | 1586 | 34.60% |

So the earlier sentence covered 1772 of 4584. Only the 1226 qualified keys can exact-match a
uniformly-qualified list, and all 1226 do.

### The 6 redwood-only matrices: 4 are not SpMM operands, so the loss is 2

From the same census:

    ss/bone010/bone010_B,986703,1,1          ss/boneS10/boneS10_B,914898,1,1
    ss/bone010/bone010_C,3,986703,3          ss/boneS10/boneS10_C,3,914898,3
    ss/bone010/bone010_M,986703,986703,23888775
    ss/boneS10/boneS10_M,914898,914898,18489474

The `_B` pair are single-column right-hand sides and the `_C` pair are 3-row blocks. Only the `_M`
pair are operators, so a cross-host comparable set loses **2** real matrices, not 6.

### Non-operands distort the board but are provably inert on the k=4 finding

150 of the 4584 census keys have `min(rows, cols) == 1` -- 127 with `cols == 1`, 23 with
`rows == 1`. Excluding them from the SuiteSparse dip, measured on this project's own board:

| band | all keys | operands only | shift |
|---|---|---|---|
| 0-32 | n=4167, -2.82% | n=4018, **-3.34%** | **-0.52 pp** |
| 32-100 | n=308, -6.13% | n=307, -6.16% | -0.03 pp |
| 100-256 | n=58, -11.23% | n=58, **-11.23%** | **0.00 pp** |
| 256+ | n=51, -9.66% | n=51, **-9.66%** | **0.00 pp** |

Pooled geomeans move k=2 1.7351 -> 1.7113, k=4 1.9346 -> 1.8979, k=8 2.3038 -> 2.2707, deepening
the pooled dip -3.24% -> -3.72%.

**The distortion is real, runs against the dip, and does not touch the finding** -- but "by
construction" was wrong for 23 of the 150, and the sentence that said so was refuted by the next
sentence in its own paragraph ("149 in that band and one is in 32-100" cannot be true if the bound
holds).

With `deg = nnz/rows` (`mrgate.py:143`), the two halves of the predicate behave differently:

| | bound on degree | measured max |
|---|---|---|
| `cols == 1` (127 keys) | **deg <= 1 structurally** -- one column puts at most one entry in each row | 1.0000 |
| `rows == 1` (23 keys) | **none** -- `deg == nnz`, which the predicate does not bound | nnz 1 (x20), 5, 6, **64** |

So the 64-nonzero one-row matrix is the entry in band 32-100, and the disjointness from the
high-degree bands is a **fact about this corpus with a thin margin** -- 64 against a band edge of
100 -- not a property of the predicate. A single one-row matrix with nnz >= 256 would land in the
256+ band, where it would be 1 of 51 and that band carries the entire k=4 result.

**The safe form, which is what to run on any corpus: read `max(nnz)` over the `rows == 1` entries
and compare it to the band edges.** Here that is 64 < 100, so the two high-degree bands contain
none of the 150 and move by exactly 0.00 pp -- verified, with the margin named, rather than
asserted structurally.

What does change: SuiteSparse's low-degree dip is **-3.34%**, not -2.82%, which widens rather than
narrows its separation from DLMC's +0.95% at the same band.

### The diagnostic worth keeping over the locale rule

A peer reproduced my `931 / 295` **from redwood's list alone**, comparing its own subsets against
its own full list under the two locales -- no MKT data enters it. That is stronger than "the merge
desynchronised": the step carried no information about the second dataset at all.

**So the test is: if a cross-dataset comparison's output is reproducible from one dataset alone, it
told you nothing about the other.** That is cheaper than auditing a merge, needs no knowledge of
which tools are order-sensitive, and also catches an empty, substituted, or self-compared second
input -- which a locale audit does not. It is the same failure as a validation whose two quantities
share a residual, and the same remedy: ask what the output would be if the second input were
absent.

**One apparent label conflict that is a frame difference, not a defect.** The peer's component
counts are off by one from mine because the sealed census keys carry an `ss/` prefix and my
generated lists are relative to the `ss` directory. Group-qualified entries are therefore
**4-component in the census and 3-component in the generated list**, both 1226; flat-plus-suffixed
are 3-component in the census and 2-component generated, both 3358. Each label is right in its own
frame, and the two are not comparable without naming which. Verified: my list has 1226 three-
component and 3358 two-component keys.

### What actually discriminates: the CONTENT of the 3368, derived then summed

A peer proposed a decomposition whose terms come from the basename census and the 7/8 mirror
partition, with no merge involved, and it is falsifiable by a form census of the MKT-only keys.
Measured:

| MKT-only keys whose basename is in | count | predicted |
|---|---|---|
| redwood's **group-qualified** class | **8** | 8 |
| redwood's **non-qualified** class | **3360** | 3352 + 8 |
| neither | **0** | 0 |

8 + 3360 = 3368, and the mirror structure closes independently: the 3368 keys span 3359 distinct
basenames, so **9** surplus paths -- which is 1 + 8, not 16, for a reason that confirms the
partition rather than complicating it. Where redwood stores a mirrored basename **qualified**, its
own path matches MKT and only the twin is MKT-only, so the 6 pairs contribute 1 each and `pwt`'s
triple contributes 2: 8 keys over 7 basenames, 1 surplus. Where redwood stores one **flat**, its
2-component form matches neither MKT path, so **both** are MKT-only: 8 pairs, 16 keys over 8
basenames, 8 surplus.

**This is a bijection claim with nothing left over because the terms were derived independently and
then summed -- not because a subtraction forced them to**, which is the difference between it and
the claim retracted above.

## The backstop job's gate was the defect this document had already named

Job 17701571 was queued with a skip condition of `total rows >= 34380` -- a **total**, which
`REVIEW_shipped_dispatch_rules.md` already said was insufficient in this very file: "A total is
satisfied by one key with 6 rows and another with 12." Publishing the rule and then writing the
violation into a control path is the failure mode this project has on record twice.

**And it was worse here than in the sentinel case it repeats.** There the total was a reported
*field*, so a wrong value at least appears in an artefact. Here it was a control-flow decision
whose passing branch **writes nothing** -- the job would exit 0 silently and partial data would
become complete data with no record that a check had been made.

**Fixed by cancel-and-resubmit, because `sbatch` spools its own copy at submit and editing the
staged file cannot change a queued job.** The spooled content was captured first, while the job was
still queued, since it is unrecoverable afterwards: `scontrol write batch_script 17701571 -` gives
`b825da8309043311`, identical to the local file, with the bad gate at line 13. Job 17701571
cancelled; job **17701584** submitted, spooled digest `0e154a9b737704fd`, also identical to
`harnesses/pfstream/k4dip_dlmc.sbatch` as committed here.

**The new gate counts keys with exactly 9 rows and requires 3820, with zero short and zero over,
and its passing branch is LOUD** -- the per-key census prints either way, so a skip leaves a
record.

**Shown to fire on the defect it forbids**, which a gate asserting a property the defect also has
would not. Injecting a file with the correct total and an uneven distribution:

| file | total | exactly-9 | short | over | new gate | old total gate |
|---|---|---|---|---|---|---|
| injected | 34380 | 3818 | 1 | 1 | **RE-MEASURE** | SKIP |
| complete | 34380 | 3820 | 0 | 0 | **SKIP** | SKIP |

Both directions: it catches the injected defect and does not wrongly refuse a complete file.

**One trap inside the verification itself.** The first run of that test was executed under `zsh`,
which does not word-split unquoted variables, so `set -- $CENSUS` produced one word and both
verdicts were reached by `[` erroring rather than by the comparison -- reporting "catches it" and
"wrongly refuses" for the same wrong reason. The sbatch script is `#!/bin/bash`, where splitting
works, so the script was never affected; the *test* was. Re-run under `bash`, which is what the job
uses, it gives the table above. **Run a verification under the interpreter the artefact declares.**

### The per-key gate counted the right property over the wrong population

The per-key census fixed the total-row defect and left a hole the per-key form does not close: it
counted keys **with** 9 rows and never checked **which** keys. "9 rows per key" is
corpus-independent -- 3 widths x 3 arms -- so any corpus satisfies it, and a file holding all 4594
SuiteSparse keys plus 1000 of the 3820 DLMC keys gives 5594 exactly-9, 0 short, 0 over, and
**skips**.

**Not hypothetical for this family.** Job 17701502 measures both corpora in one job, and
`arm_screen.sh:49-56` records that this project already had "a float32 DLMC screen wrote files
named for a float64 SuiteSparse one". A dlmc-named file carrying ss keys is the documented failure
mode, and it was the one case where the skip was silent.

**The census computed four values, printed four, and the gate tested three.** `length(c)`, the
distinct-key count, was the only one bounding the *population* rather than the per-key property,
and it was printed and not tested.

**Fixed by set identity rather than another count.** The gate now compares the CSV's key set
against `$KEYS` by `awk` membership -- not `sort` + `comm`, because a merge is order-sensitive and a
locale mismatch between the sort and the merge silently fabricates set differences, which this
document records happening earlier in the same session. It reports and requires `exactly9 == nw`,
`short == 0`, `over == 0`, `keys == nw`, `unknown == 0`, `missing == 0`.

**Driven through one case per failure mode, under `bash` as the artefact declares:**

| case | rows | e9 | short | over | keys | unknown | missing | identity gate | count gate |
|---|---|---|---|---|---|---|---|---|---|
| zero-byte | -1 | 0 | 0 | 0 | 0 | 0 | 3820 | REMEASURE | REMEASURE |
| header-only | 0 | 0 | 0 | 0 | 0 | 0 | 3820 | REMEASURE | REMEASURE |
| complete | 34380 | 3820 | 0 | 0 | 3820 | 0 | 0 | **SKIP** | SKIP |
| one key over | 34381 | 3819 | 0 | **1** | 3820 | 0 | 0 | REMEASURE | **SKIP** |
| one key short | 34377 | 3819 | **1** | 0 | 3820 | 0 | 0 | REMEASURE | REMEASURE |
| one key missing | 34371 | 3819 | 0 | 0 | 3819 | 0 | **1** | REMEASURE | REMEASURE |
| extra unknown key | 34389 | 3821 | 0 | 0 | 3821 | **1** | 0 | REMEASURE | **SKIP** |
| 4594 ss + 1000 dlmc | 50346 | 5594 | 0 | 0 | 5594 | **4594** | **2820** | REMEASURE | **SKIP** |

**The count-based gate wrongly skips 3 of 8**, and `complete` is the only SKIP under the identity
gate. Stated honestly: the six clauses are **coupled** -- `e9`, `keys`, `unknown` and `missing` are
all functions of the same two sets -- so a single-clause isolation is not always constructible. What
the table demonstrates is that each failure *mode* is caught and which field catches it, which is
the useful form.

**Two printed expectations became comparisons.** The script carried
`(expect 4c11296ce2a68c3d)` and `(expect 3820, a8132f50d2c44fe4)` beside measured values and never
compared them. For audit purposes that is **worse than no check** -- a reader scanning the log sees
"expect X" next to a measured value and takes it for provenance. Both now `REFUSE` on mismatch, and
the key-list digest is the second, independent route to catching the wrong-corpus case above.

**Job 17701584 cancelled while PENDING; job 17701598 submitted**, spooled digest
`f14b7b86c77af257`, identical to `harnesses/pfstream/k4dip_dlmc.sbatch` as committed.

**And the zsh trap recurred on the reviewer's side one message after the rule was published**, with
a tell worth recording: every branch returned the **same** verdict for the same wrong reason --
REMEASURE for all cases including the complete one that must SKIP -- so a broken sweep reads as a
conservative gate. **Uniform output across a branch sweep is the tell.** `bash -n` passes and proves
nothing, since it parses without running the comparisons. The table above is varied, which is the
minimum evidence that the sweep itself works.

### The same defect recurs one level down, where no census field registers it

A key whose nine rows are nine copies of a single `(k, arm)` cell has **no k=4 and no k=8 data**,
and reads identically to a complete key on every field above: present, exactly 9 rows, not short,
not over, not unknown, not missing -- and the row total is exactly 34380. This is not a coupling
limitation. **All six fields and the total read the same as `complete`, so there is no clause to
isolate: the census was blind.**

"9 rows per key" counted the right property over the wrong population **inside** a key, exactly as
"keys with 9 rows" did across the corpus one revision earlier.

**The general form, now twice in two revisions: a completeness gate must name the population at
every level of the key it aggregates over -- corpus, key, cell -- because a count at one level is
invariant to a permutation at the next.**

Fixed with one clause: distinct `(key, k, arm)` triples per key must equal 9, via
`t[$3 SUBSEP $4 SUBSEP $8]`. It subsumes the short case, which also reads `cells!=9 = 1`.

| case | rows | e9 | sh | ov | keys | unk | miss | **c!=9** | identity+cell | total-row gate | per-key-count gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero-byte | -1 | 0 | 0 | 0 | 0 | 0 | 3820 | 0 | REMEASURE | REMEASURE | REMEASURE |
| header-only | 0 | 0 | 0 | 0 | 0 | 0 | 3820 | 0 | REMEASURE | REMEASURE | REMEASURE |
| complete | 34380 | 3820 | 0 | 0 | 3820 | 0 | 0 | 0 | **SKIP** | SKIP | SKIP |
| one key over | 34381 | 3819 | 0 | 1 | 3820 | 0 | 0 | 0 | REMEASURE | **SKIP** | REMEASURE |
| one key short | 34377 | 3819 | 1 | 0 | 3820 | 0 | 0 | 1 | REMEASURE | REMEASURE | REMEASURE |
| one key missing | 34371 | 3819 | 0 | 0 | 3819 | 0 | 1 | 0 | REMEASURE | REMEASURE | REMEASURE |
| extra unknown key | 34389 | 3821 | 0 | 0 | 3821 | 1 | 0 | 0 | REMEASURE | **SKIP** | **SKIP** |
| 4594 ss + 1000 dlmc | 50346 | 5594 | 0 | 0 | 5594 | 4594 | 2820 | 0 | REMEASURE | **SKIP** | **SKIP** |
| **repeated cell** | **34380** | **3820** | **0** | **0** | **3820** | **0** | **0** | **1** | REMEASURE | **SKIP** | **SKIP** |
| | | | | | | | | | 1 SKIP | **4 of 9 wrong** | **3 of 9 wrong** |

**Reachability, stated rather than assumed:** the repeated-cell case needs duplicate
`(key, k, arm)` rows, which means an appending or retried write by the *producing* job. The
backstop writes to a separate path and cannot do this to itself; the file it judges is written by
17701502, whose write behaviour this gate cannot see. So it is not claimed to be likely -- the
point is that the gate's reach stopped one level above where its own data lives, and the fix is one
clause.

### Correcting my own scorecard: "wrongly skips 3 of 8" named the wrong gate

The two figures disagreed because the column was mislabelled, not because either count was wrong.
My table's column headed "count gate" computed `total rows >= 34380` -- the **total-row** gate,
which is the original defect -- and against that, 3 of 8 is right. Against the **per-key-count**
gate (`exactly9 >= wanted && short == 0 && over == 0`), which is what "count gate" reads as, it is
2 of 8: `one key over` has `e9 = 3819`, so `3819 >= 3820` is false and that gate refuses. Both
columns are now computed and labelled separately above.

**The overstatement ran in favour of the change I was making**, which is the direction to be
suspicious of in one's own scorecard, and it is the reason to recompute a comparison rather than
read it off the version you are replacing.

### Two smaller repairs

`nw` was incremented per **line** of `$KEYS` while the set was built as `want[$0]`, so a duplicated
or blank line would make `wanted` disagree with the set actually used -- failing closed, but with a
log reading "3820 of 3821 wanted" and nothing pointing at the key list. Now `length(want)`, with
blank lines skipped.

And `awk` membership over `sort` + `comm` was chosen because a merge is order-sensitive; it is also
why this gate is immune to the locale defect recorded earlier in this document, so the two fixes
compose rather than merely coexist.

Job 17701598 cancelled while PENDING; job **17701612** submitted, spooled digest
`b99b6bc5c4a5379e`, identical to `harnesses/pfstream/k4dip_dlmc.sbatch` as committed. Superseded
spools on record: `b825da8309043311` (total gate), `0e154a9b737704fd` (per-key count),
`f14b7b86c77af257` (key-set identity, no cell clause).

### Identity versus count, and the cell clause was a count

The asymmetry was visible in the source with no test: keys were checked by **identity**
(`!(k in want)`, `!(k in c)`) and cells by a **number** (`nt[k] != 9`). Two standards in one
script, with the weaker one at the level the previous revision was meant to strengthen. A count is
invariant to a permutation of *which* members, so a file with all 3820 keys at 9 distinct
`(k, arm)` cells and **zero k=8 rows anywhere** passed -- in a study whose question is k=4 against
k=2 and k=8.

**Reachable, not hypothetical, and it is the same hole as the screen's.** `mrgate.py:27` is
`ap.add_argument("--ks", default="1,2,4,8,16,64")` -- six widths, matching neither this study's
{2,4,8} nor any screen's set. The width set is argv-supplied with a differing default, and the gate
judges a file written by a job whose `--ks` it cannot see. Argv is not the invocation, and nothing
in the output record binds the widths.

**Fixed by enumerating the wanted `(k, arm)` pairs in the gate.** Counting to 9 over exactly 9
enumerated members *while requiring zero non-members* forces set **equality**, which is identity
rather than cardinality -- the count alone was not.

**And one level below both: no clause read the measurement.** `csv.DictWriter` emits a missing key
as an **empty field**, and this harness family uses it in eight files under `harnesses/pfstream/`,
so a dropped value is a complete-looking row. The gate verified that a cell *exists* and never that
it holds a usable number. Added: rows with `ms == ""` or `ms <= 0`.

| case | rows | e9 | keys | missing | cells!=9 | outside | bad-ms | verdict |
|---|---|---|---|---|---|---|---|---|
| complete | 34380 | 3820 | 3820 | 0 | 0 | 0 | 0 | **SKIP** |
| widths {2,4,16}, no k=8 | 34380 | 3820 | 3820 | 0 | **3820** | **11460** | 0 | REMEASURE |
| one blank `ms` | 34380 | 3820 | 3820 | 0 | 0 | 0 | **1** | REMEASURE |
| repeated cell | 34380 | 3820 | 3820 | 0 | **1** | 0 | 0 | REMEASURE |
| `mrgate.py:27` default widths | 68760 | 0 | 3820 | 0 | 0 | **34380** | 0 | REMEASURE |

Every one of the first four reads **34380 rows**, which is why no total could separate them.

### The transferable form, after four revisions

"Name the population at every level" is **satisfiable by a count**, and a count is invariant to a
permutation of which members -- so that rule restated the defect rather than repairing it. The
sharper question at each level is:

> **Is this check an IDENTITY against an enumerated population, or a COUNT over an unenumerated
> one?**

The one level that used identity from the start -- keys against `$KEYS` -- is the one level with no
residual across four revisions. That is the evidence for the question being the right one, and it
is why the fix is "enumerate the members", not "add another counter".

Job 17701612 cancelled while PENDING. Spooled digests on record, one per revision:
`b825da8309043311` total rows, `0e154a9b737704fd` per-key count, `f14b7b86c77af257` key-set
identity, `b99b6bc5c4a5379e` plus cell count, and the current one plus cell identity and value
validity.

### The object digest guarded the branch that does not run

The gate digested `$BUILD/src/scorch_ops*.so` against `4c11296ce2a68c3d` -- the object this job
would use **if it re-measured** -- and never read the `build` column of the file it was
**accepting**. Columns 1 and 2, `build` and `slice`, were the only two the gate never touched, and
`build` is the one naming the object.

| case | e9 | keys | cellbad | outside | bad-ms | not-`ship` | distinct builds | verdict |
|---|---|---|---|---|---|---|---|---|
| complete | 3820 | 3820 | 0 | 0 | 0 | 0 | 1 | **SKIP** |
| mixed build | 3820 | 3820 | 0 | 0 | 0 | **17190** | **2** | REMEASURE |
| other build | 3820 | 3820 | 0 | 0 | 0 | **34380** | 1 | REMEASURE |

All nine prior clauses read clean in every row. **The mixed case is the worse of the two**: a
per-key `ours`/`mkl` ratio then spans two objects, so the comparison is corrupted rather than
merely unattributed.

**Both new clauses are needed and they are complementary, not redundant.** `other build` is
*uniform* -- distinct builds is 1 -- so a "one build per file" check alone misses it; the
`!= BUILDTAG` count catches it. The mixed case trips both.

**Fixed with a single source of truth rather than a duplicated literal:** `BUILDTAG=ship` is
defined once and passed both to the gate (`awk -v BUILD="$BUILDTAG"`) and to the measurement
(`--build "$BUILDTAG"`), so the expected tag and the written tag cannot drift.

**Reachability is the weak part and is stated rather than leaned on:** one invocation writes one
build and the filename encodes it, so a mixed file needs two writes to one path. What makes it
worth the clause is that the documented failure in this family is precisely a filename not matching
its content (`arm_screen.sh:49-56`), and the gate was taking the filename's word for the build while
the column stating it went unread.

**The general shape is new rather than a fifth instance of identity-versus-count:**

> **A provenance check scoped to the path not taken proves nothing about the data kept. Ask which
> branch the digest guards.**

Job 17701623 cancelled while PENDING. **Gate work stops here** -- the proportionality rule being
applied is: report what changes a verdict, not what could. This one changed three.

## MKT SuiteSparse, {2,4,8}: the k=4 dip INVERTS, and k=2 is the width behind MKL

Job 17701502, build `ship` (hooks-clean, so MKL comparisons are not taxed), ss phase complete
04:41:07 with `rows=41346 expect 41346`, rc=0. Transported with a producing-host digest
(`5762e6913dc283d4…`, byte-identical locally). Read in the order registered before the numbers were
seen.

**Corpus restriction, step 1.** 4594 measured paths -> 4578 distinct basenames, which is exactly
the common set with redwood's sealed census (`chain209-dlmc-dims/ss_dims.csv`): 6 redwood-only
(`bone010_B/C/M`, `boneS10_B/C/M`), **0 MKT-only**, and 15 basenames covering 31 paths, so
4578 + 16 duplicates = 4594 closes as a partition. Computed with Python set operations, not `comm`.

**Step 2, the mirrored-path shape check — and it mattered.** 8 of the 31 mirrored paths disagree
with redwood's sealed `(rows, nnz)`, and those 8 are exactly the basenames that disagree *with each
other* on MKT. They are genuinely different matrices sharing a name across groups:
`Newman/football` is 115x1226 against `Pajek/football` 35x118; `Pothen/barth` 6691x46187 against
`Nasa/barth` 6691x26439. In every one of the 8, exactly one path matches redwood — so
dedup-by-basename would have chosen arbitrarily, and the `barth4` precedent (exclude) is right.
Excluded: barth, barth4, barth5, ex3sta1, football, fxm3_6, fxm4_6, pf2177. **4570 matrices kept.**

**Step 3, non-operands: 148 here against the registered 150, and both are correct.** Redwood's
census has 150 keys with `min(rows, cols) == 1`; two of them (`bone010_B`, `boneS10_B`) are among
the 6 redwood-only basenames, so MKT measures 148. The 127 `cols==1` / 23 `rows==1` split
reproduces. Dropping them moves no verdict below.

**THE BOARD.** Speedup = `mkl_ms / ours_ms`, so above 1.0 is ours faster:

| k | n | geomean | median | behind MKL | `oursb` geomean |
|---|---|---|---|---|---|
| 2 | 4570 | **0.9480** | 0.8850 | **2828 (61.9%)** | 0.9482 |
| 4 | 4570 | 1.1074 | 1.0681 | 1864 (40.8%) | 1.1074 |
| 8 | 4570 | 1.2300 | 1.2612 | 1038 (22.7%) | 1.2299 |
| all | 13710 | | | **5730 (41.8%)** | |

`oursb` reproduces `ours` to four decimals at all three widths, which is the internal consistency
check that the two our-side builds are the same measurement.

**The width curve is MONOTONE INCREASING in k — 0.9480 -> 1.1074 -> 1.2300 — so there is no k=4 dip
on this host and corpus at all.** The registered paired statistic
`log r_4 - mean(log r_2, log r_8)` confirms it with the sign reversed:

| paired k=4 statistic | pooled | deg 0-32 | 32-100 | 100-256 | 256+ |
|---|---|---|---|---|---|
| redwood SuiteSparse (registered) | **-3.24%** | -2.82 | -6.13 | -11.23 | -9.66 |
| MKT SuiteSparse (measured) | **+2.56%** | +1.86 | +10.95 | +9.30 | +3.27 |
| MKT 95% CI | [+2.07, +3.06] | [+1.34, +2.38] | [+9.54, +12.41] | [+6.54, +12.42] | [+1.46, +5.17] |

Every band is a bump with a CI excluding zero. **So the k=4 deficit is not a property of the kernel
family: it reverses sign between the two confirmation hosts on the same collection.** A plausible
mechanism is already on record -- redwood compiles AVX2-only and MKT compiles Zen4 AVX-512 under
`-march=native`, and `scorch_simd_half<float>` is a 128-bit `__m128` form, so the half-vector width
is not the same code on the two hosts. That is a mechanism to test, not a conclusion.

**Against the registered prediction, which was wrong.** I registered "k=4 ranks 2nd most robust of
3 on SuiteSparse" from redwood's normalised table. MKT reads **1st of 3** -- but the rank is close
to meaningless here, because the three widths' top-band normalised values are 0.8088 / 0.8204 /
0.8095, a spread of **1.2 pp** against redwood's 18.7 pp. The honest statement is not "the rank
moved by one" but **"the quantity that discriminated the widths on redwood does not discriminate
them on MKT"**, and separately the paired statistic's sign flipped. Registering a rank was the
wrong choice of prediction: a rank cannot express "the spread collapsed", so it would have read as
a near-miss either way.

**What this means for the standing task.** On MKT SuiteSparse at narrow widths, scorch is **not**
ahead of MKL: 41.8% of 13710 cells are behind, concentrated at k=2 (61.9%, geomean 0.9480, median
0.8850). The deficit deepens with degree -- at k=2, 59% of matrices are behind in deg 0-32 and
92% at deg 256+. **k=2 on MKT, not k=4, is the live problem.**

**Scope, stated.** One pass (`p1`), one reading per cell, so the per-cell "behind" counts are an
upper bound on a confirmed deficit -- the established standard elsewhere in this work is "behind in
every reading" over 12-16 readings. A geomean of 0.9480 over 4570 matrices is not noise; an
individual cell at 0.98 may be. One host for this board: the matching redwood screen at these three
widths does not exist yet, and the redwood figures quoted above come from a different screen
(`chain207`, five widths), so the cross-host comparison is of statistics, not of a shared design.

### Pre-registration for the dlmc phase of 17701502 (started 04:41:07, no rows read)

The sign reversal above has two candidate explanations and the dlmc phase discriminates them,
because it is the one missing cell of a 2x2:

| | redwood | MKT |
|---|---|---|
| SuiteSparse | dip -3.24% | **bump +2.56%** |
| DLMC | dip (k=4 least robust of 5) | **pending** |

**Registered predictions, before any dlmc row is read.** If the cause is the host -- AVX2 against
Zen4 AVX-512 changing what the 128-bit `scorch_simd_half<float>` path compiles to -- MKT DLMC
should read a **bump**, pooled CI above zero. If the cause is an interaction with the corpus, MKT
DLMC should read a **dip**, matching redwood's sign on the same collection. A pooled CI spanning
zero supports neither and would mean the effect is smaller than this design resolves.

**Registered as a separation, not a rank**, per the lesson above: the quantity is the pooled paired
`log r_4 - mean(log r_2, log r_8)` with a bootstrap CI, and the comparison is to MKT SuiteSparse's
`+2.56% [+2.07, +3.06]` and to redwood DLMC's negative sign -- not an ordering of three widths.
Secondary and also registered: whether k=2's behind-MKL share on DLMC exceeds its 61.9% on
SuiteSparse, since k=2 is the live deficit and DLMC's matrices are smaller and lower-degree.

### Amendment to that pre-registration, made before reading any dlmc row

Written while the dlmc phase runs. **Stated as a bound rather than an assurance**, which is the
checkable form: this section committed at 04:58:44 against a dlmc start of 04:41:07, so 1057 s had
elapsed, and at the ss phase's logged 4.52 rows/s **at most ~4.8k of 34380 rows (~14%) existed** --
about 531 of 3846 keys, and since the driver walks the key list in order that is a **specific
non-random prefix**, more informative about whatever the list ordering correlates with than about
the corpus. The dlmc rate may differ from ss's, so 14% is an estimate with its rate named. The
access record, separately: no dlmc CSV has been opened this session; the only dlmc-related read was
`tail -3` of the job's stdout, which printed the phase's start line. Peer reviewer's framing, and it
is better than "no rows informed it" because a bound can be checked and a negative cannot.

**First, the redwood SuiteSparse anchor is not inconsistent — the two figures are different
populations.** A peer computed `-7.06%` from the co-registered summaries `{0.682, 0.540, 0.495}`
and read it against the doc's `-3.24%`, a 2.2x gap. But those three numbers are **top-degree-band
values normalised to each width's own deg 0-32 band**, so a contrast over them is a deg-256+
quantity. The registered redwood band figures are -2.82 / -6.13 / -11.23 / **-9.66**, and -7.06 sits
beside the top band's -9.66, not beside the pooled -3.24. The pooled figure is dominated by band 0,
which holds 4153 of 4570 SuiteSparse matrices. Two populations, not two estimators.

**And the gap to the top band is not a residual — it is an identity, which closes the question
rather than leaving it open.** Each value is normalised to *its own* deg 0-32 band, so a contrast
over normalised values is algebraically the **top-band contrast minus the band-0 contrast**:

**The composition is a ratio of factors, never a subtraction of percents** -- the difference is
exact in logs and the percent form must be written as a ratio, or the printed operator does not
produce the printed number:

| | contrast of normalised values | `(1+c_top)/(1+c_0) - 1` | gap | *(wrong form: `c_top - c_0`)* |
|---|---|---|---|---|
| redwood SuiteSparse | -7.0608% | `(1-0.0966)/(1-0.0282) - 1` = **-7.0385%** | **0.022 pp** | *-6.84%* |
| redwood DLMC | -16.2660% | `(1-0.1560)/(1+0.0095) - 1` = **-16.3943%** | 0.128 pp | *-16.55%* |

An earlier revision of this table printed the subtraction form while carrying the ratio form's
values, which a peer correctly read as the identity failing by 0.22 pp on SuiteSparse -- ten times
the real gap. In logs the subtraction *is* exact: `log(1+c_top) - log(1+c_0)` = -7.2985 and
-17.9058 pp of log. Three forms, two correct, and the incorrect one was the one printed.

**This same algebra is stated correctly at `:1990`, two sections above**, as
`(1 + dip_band) = (1 + dip_{0-32}) x (k=4/nbr)`. So the subtraction form was not a missing insight
-- the document contradicted its own earlier equation, which is the worse of the two failures.

**What the gap establishes, and what it does not -- and a rounding envelope must name WHICH
publication it is the envelope of.** Perturbing the DLMC triple over all eight corners of +/-0.0005
gives an envelope of +/-0.148 pp, which contains 0.128 pp *and* 0.022 pp, so publication rounding is
**consistent with** both and **identifies** neither. Worse for the SuiteSparse half: that number is
reachable by two different input chains that disagree about whether anything is unexplained --
inside +/-0.148 pp if its input is the three-decimal normalised triple, and **2.2x** the +/-0.0095
envelope if its input is the two-decimal band figures, where it appears as the 256+ row of the table
at `:1990`. **One residual, two available attributions, opposite verdicts.** For DLMC the statement
is unconditional, because the triple is the only chain available and the `:1990` table is
SuiteSparse-only, so there is no second appearance to cross-check. Peer reviewer's enumeration and
his localisation of the double appearance.

**The premise the identity rests on, now stated and tested rather than assumed.** The identity is
exact only if the per-width normalisation geomeans and the three-width *paired* band dips are taken
over **one key set per band** -- a paired dip is a mean of per-key contrasts, so its population is
the paired set, while a per-width geomean's need not be. MKT's grid is balanced, so the premise is
checkable there, and it holds to machine precision:

| band | contrast of normalised geomeans | `(1+c_b)/(1+c_0) - 1` | gap |
|---|---|---|---|
| 32-100 | +8.92244045% | +8.92244045% | 0.00e+00 pp |
| 100-256 | +7.30735329% | +7.30735329% | 0.00e+00 pp |
| 256+ | +1.38783240% | +1.38783240% | 2.22e-14 pp |

with the three per-width key sets and the paired set verified identical in all four bands
(4153 / 308 / 58 / 51). **So the identity contributes nothing to a gap: when the populations
coincide it is exact by linearity of the mean, and a nonzero gap on redwood is therefore
publication rounding or a population mismatch, nothing else.** The one partition route the file
already quantifies is closed by a number measured earlier in it -- rounded-versus-exact degree
binning moves DLMC 256+ from n=713 to n=712 and the figure by **+0.0052 pp**, 25x too small to be
0.128. The pairing route is untestable from what is held locally, since redwood's per-key five-width
board is not on this host.

**And the high bands are the same KEYS on both hosts, which removes a population difference from the
registered test.** Band censuses over exact `nnz/rows`:

| | deg 0-32 | 32-100 | 100-256 | 256+ | sum |
|---|---|---|---|---|---|
| redwood sealed census | 4167 | 308 | 58 | 51 | 4584 |
| MKT keep-set | 4153 | **308** | **58** | **51** | 4570 |

Band 0 loses exactly 14 and the three high bands lose exactly 0 -- verified by naming all 14, not
by the partition sums: the 6 redwood-only keys and the 8 shape-disagreeing ones have exact degrees
0.000, 1.000, 24.211, 0.000, 1.000, 20.209, 3.951, 3.903, 3.940, 3.943, 3.371, 9.310, 11.850,
3.185, every one inside band 0 and the largest at 24.2 against the 32 edge. With equal cardinality
and containment in the 4578 common basenames, **the high bands are the same 308, 58 and 51 matrices
on both hosts**, so at exactly the bands the band-matched registration rests on, a population
difference is not available as an explanation of a between-host disagreement. Peer reviewer's
finding.

**The census is convention-invariant in every band, not only the top one.** The harness writes
`deg="%.2f" % (nnz / max(M, 1))` (`mrgate.py:143`), which can move a degree by at most 0.005, and
**zero keys sit within 0.005 of an edge** -- so the 2-decimal census equals the exact one exactly:

| convention | census |
|---|---|
| exact `nnz/rows` | 4167 / 308 / 58 / 51 |
| 2-decimal, the shipped harness form | **4167 / 308 / 58 / 51** |
| integer rounding | 4164 / 310 / 59 / 51 |

The third row is what moves 3 keys across 32 and 1 across 100 -- `Goodwin_054` 31.71,
`Goodwin_071` 32.09, `Goodwin_095` 32.25, `Goodwin_127` 32.38, `e40r0100` 32.03, `lock_700` 31.68,
`m_t1` 99.96, `mcfe` 31.87, the 8 keys within 0.5 of an edge. But integer rounding is a convention
nothing in this work uses, so those 8 bound a sensitivity that is not live. A peer's
4164 / 310 / 59 / 51 is that convention's census, not this one's.

**One narrowing I decline, because it is a different population.** It does not follow from the
between-host set identity that the SuiteSparse top band's 0.022 pp gap has only rounding left. Two
distinct questions are in play: (a) are redwood's and MKT's high-band key sets the same -- answered
yes above; and (b) within redwood, does each per-width geomean's population equal the paired-dip
population -- which is the identity's premise. The 0.022 pp lives entirely inside redwood's own
data, as the gap between redwood's normalised triple and redwood's band dips, so what MKT measures
cannot bear on it. What would settle (b) is **redwood's own grid completeness**, and the precedent
is on record for the other corpus: chain207's DLMC board is 19100 cells with 229200 observations
against an expected 229200, fully balanced. If the SuiteSparse board is likewise complete then (b)
holds within redwood and rounding is the only live branch -- but that is an argument from
completeness, not from between-host identity.

**That route is now closed, by a peer who found the board where I said it was not.** Redwood's
per-key five-width SuiteSparse board is in the sealed ledgers on this machine --
`chain196-fullcollection-redwood/sh_{c0,c0b,pfd0}_float32_p{1..6}.csv` plus the chain205 and
chain206 recount/drift objects, 42 files over 7 objects x 6 passes -- and **every one is exactly
balanced**: 4584 keys x 5 widths x 3 arms = 68760 rows, distinct `(key,k,arm)` equal to the row
count, zero duplicates. Balance at cell level makes the per-width populations coincide inside every
band including the 51-key top one, so premise (b) holds on redwood too and **the pairing route is
dead on both hosts**. The 0.022 pp has only the rounding branch left, which is the conclusion I had
declined to draw from the wrong evidence and which now follows from the right evidence. "I could
not find it locally" was a statement about my search, not about the ledger.

**And the published band table is now independently identified, not merely consistent.** A sweep of
7 objects x {median, geomean, min} x {ours, oursb} = 42 combinations matches all four published
values at all four counts in exactly one place: `chain196.../sh_c0_float32_p{1..6}.csv`, with
`r_k = mkl_ms/ours_ms`, **min over the six passes**, exact `nnz/rows` bands, and the estimator
`exp(mean_keys[log r_4 - (log r_2 + log r_8)/2]) - 1`. So artefact, arm, pass aggregation, binning
and estimator are all pinned.

### The estimator mismatch in the 2x2, stated and bounded

**The registered comparison is not estimator-matched: redwood's published cell is best-of-6 passes
and MKT's is single-pass.** The ss phase is 41346 rows = 4594 keys x 3 widths x 3 arms, which
leaves no room for a second pass. (75726 is this job's *total*, 41346 ss + 34380 dlmc.) On redwood's
own data the choice is not free -- holding the object fixed, aggregation x arm moves the top band by
1.30 pp on c196/c0 and 2.37 pp on c205/c0, which is 23-42% of that band's published CI width of
5.65 pp.

**Bounded on the MKT side with the two our-side objects as a pseudo-replicate pair**, which is the
only replication a single pass admits:

| aggregation | pooled | deg 0-32 | 32-100 | 100-256 | 256+ |
|---|---|---|---|---|---|
| `ours` only (as published) | +2.56% | +1.86 | +10.95 | +9.30 | +3.27 |
| `oursb` only | +2.54% | +1.85 | +10.95 | +9.02 | +3.22 |
| max ratio (= min of our ms) | +2.50% | +1.82 | +10.73 | +8.94 | +2.90 |
| min ratio (pessimistic) | +2.60% | +1.89 | +11.17 | +9.39 | +3.59 |

Spreads: 0.105 pp pooled, 0.071 / 0.432 / 0.452 / **0.697** pp by band, and **every aggregation
gives the same sign in every band**. So the mismatch is real and must be declared -- it is -- but it
cannot reach the registered branches: the between-host separation the 2x2 turns on is about
**18.5 pp** (MKT's most pessimistic top-band +2.90 against redwood DLMC's -15.60), which is **26x**
the MKT aggregation spread and **8x** redwood's own 2.37 pp. A two-object spread is not a six-pass
spread, so this bounds rather than reproduces the house convention; the useful fact is the ratio
between the sensitivity and the effect, not the sensitivity alone. Peer reviewer's finding; the MKT
bound is measured here.

**And the 0.12 pp appears twice but is one quantity.** The additive residual against the top band is
`r = -c_0 (1+c_top)/(1+c_0)`, which gives **+2.62 pp** for SuiteSparse against an observed +2.60 and
**-0.79 pp** for DLMC against an observed -0.67. That second 0.12 pp *is* the identity gap above,
not a separate discrepancy, so it must not be counted twice.

So the quantity is not supposed to equal any single band's figure, and **nearest-band matching is
the wrong test** — it would pick band 1 for SuiteSparse (-6.13, 0.93 pp away) and the top band for
DLMC (-15.60, 0.67 pp away), an apparent 4x asymmetry that is simply the band-0 contrast appearing
with its own sign and size: -2.82 on SuiteSparse against +0.95 on DLMC. Under the identity both
corpora agree to within the rounding of the published three-decimal values, and nothing is left
unexplained. My earlier phrasing, "both agree on the band they came from", invited exactly the
nearest-band reading a peer then applied to it; the identity is the statement that does not.

The provenance was never inferred from closeness in any case — it is stated in this file at the
table's own heading, "Ranking k=4's degree-robustness among the five widths **in the top band**",
over values defined one section earlier as "Each width's geomean normalised to its own deg 0-32
value".

**Second, and this is a defect in my own registration: the pooled branch is not well posed.**
Redwood DLMC's paired dip by band is **+0.95** / -2.39 / -11.72 / -15.60 — the lowest band is a
**bump**. So "MKT DLMC reads a dip -> the cause interacts with the corpus" has no clean comparand,
because redwood DLMC is itself a bump at low degree and only a deep dip at high degree. Worse,
pooling across corpora compares different band *mixtures*: band 0 holds 91% of SuiteSparse but a far
smaller share of DLMC, whose degree distribution is much higher. **The comparison is therefore
registered band by band, and the discriminating feature is the high-degree bands**, where redwood
DLMC reads -11.72 and -15.60 against MKT SuiteSparse's +9.30 and +3.27:

| registered read, MKT DLMC deg 100-256 and 256+ | conclusion it supports |
|---|---|
| both bumps, CIs above zero | the inversion is **host**-caused (AVX2 vs Zen4 AVX-512 on the 128-bit path) |
| both deep dips near -12 and -16, CIs below zero | it tracks the **corpus**, matching redwood DLMC |
| dips, but an order of magnitude shallower | the **sign** tracks the corpus and the magnitude does not — which is the reading a sign-only branch would have hidden |
| CIs spanning zero | below this design's resolution; report as such |

The third row is the one this amendment exists to make sayable. A sign-only branch inherits exactly
the defect the rank did: it cannot express that a dip of -0.5% and a dip of -15.6% are different
findings.

**Third, the three registered ss reads are all applied, and none moves the verdict.** The `+2.56%`
already carried the common-basename restriction and the shape check; recomputing across all three
populations:

| population | n | pooled paired k=4 | 95% CI |
|---|---|---|---|
| one path per basename, no shape check | 4578 | +2.54% | [+2.06, +3.02] |
| 4578 common basenames, shape check applied | 4570 | **+2.56%** | [+2.07, +3.06] |
| same, 148 non-operands dropped | 4422 | +2.33% | [+1.83, +2.83] |

The non-operand correction is **-0.23 pp** here, and the worst case across all three is **+1.83%** —
a bump in every population, with the margin named.

**Fourth, the secondary's base.** k=2's **61.9%** behind-MKL share is over the **4570**
common-basename set *including* the 148 non-operands; operands only it is **63.1%** (2790/4422), so
dropping the degenerate keys raises it rather than lowering it. The size of that, exactly, since
the counts are integers and need no rounding: 2828 - 2790 = **38 of the 148 non-operands are behind
MKL, 25.68%, against 63.09% for operands — a 37.4 pp gap.** Degenerate keys are behind less than
half as often, which is the whole reason the share moves the way it does. The reconciliation to the
registered 150 is exact and not a 14-key shift: 150 redwood-census non-operands minus the 2 that
are redwood-only basenames (`bone010_B`, `boneS10_B`) = 148, and all 148 survive the shape check,
since 4570 - 148 = 4422. Verified against the sealed census: of the 6 redwood-only keys, exactly the
two `_B` are `cols==1`; `bone010_C` and `boneS10_C` are `rows==3` and the two `_M` are square, so
2 of 6 is the right count and the census predicate still splits 23 `rows==1` + 127 `cols==1` = 150.
**A different sensitivity would lose 4, not 2:** the `_C` pair is `rows==3`, so both sit inside the
`min(rows,cols) <= 4` set of 194 (the -1.93% to -2.08% pooled sensitivity) while neither is in the
150. If that sensitivity is ever carried to the MKT side the shift is **4 keys**. Peer reviewer's
note. The DLMC side must use the same predicate — one path per basename, shape-checked where a census exists, and the share reported on
both bases — or the two shares are differently defined and not comparable.
