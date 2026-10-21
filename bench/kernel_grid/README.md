# Runtime grid for kernels whose C++ changed

Bobby's rule (2026-10-11 and 2026-10-12): a kernel whose emitted C++ changed ships when it is
strictly correct and its measured runtime is neutral-or-better than the kernel it replaces,
on a shape / density / dtype / format grid, on two hosts, judged against same-binary A/A
noise. A kernel that was wrong before owes no timing, because there is no correct old kernel
to compare against.

The pairs (program, element type) that owe this grid for the merge of 2026-10-12 come from
mlsys's lists in `~/.cache/scorch-codex/changed-shipped-7ac70f2-nc0740/` and
`~/.cache/scorch-codex/cnonempty-nc0740/`. They are collected, with each program's
expression and formats, into `~/.cache/scorch-codex/merge-grid/pairs.json` (306 pairs):

- 146 pairs whose shipped C++ changed from 7ac70f2 to the merge tree, where the 7ac70f2 kernel
  was correct on every run (comparison "shipped change");
- 159 pairs where the new compiler's C++ declares `C_nonempty` (comparison "new compiler",
  against the merge tree's shipped kernel, and against 7ac70f2's beside it);
- 1 pair in both lists (`SDDMM ss x dd x dd -> ds`, float64), timed with all three kernels in
  one interleaved run.

## How it runs

`run_host.sh <old tree> <new tree> <pairs.json> <out dir> [capture parts] [processes]`

1. `capture.py` runs each pair on each grid cell through the production entry. For the shipped
   compiler that is the public `scorch.einsum`; for the new compiler it is
   `execute_cin_via_loopir` on the program einsum hands to its scheduler, under the automatic
   plan (`requested_schedule=Schedule()`), once per register-blocking setting: the
   environment's setting is the variant `new`, the other is `new_other`, which is always
   judged and is timed only when its kernels differ from `new`'s. It records the `.so`
   and the exact arguments of every generated kernel the call runs, in order: the public call
   may convert an operand or the result with a kernel of its own beside the program's kernel,
   and which operand is converted is a schedule decision, so a route's cost is all of its
   kernel calls. It judges the result against float64 `torch.einsum`. Only correct routes are
   timed.
2. `time_cells.py` loads every variant's kernels, plus a copy of each (the A/A control), into
   one process that imports no Scorch tree. One counted call of an arm runs every kernel call
   of its route. It interleaves them in random order, 21 rounds per
   cell, with a batch of calls per arm that lasts at least 2 ms and is the same for every arm
   in the cell. It runs in several processes, one after another.
3. `analyze.py` writes the verdicts.

Grid: three sizes (budgets of 4e3, 2.6e5 and 4e6 elements for the largest tensor, the same
extent for every index) times two fills (5% and 55% of a sparse operand's cells nonzero).
The extent is also capped so that the dense iteration space (the extent to the power of the
number of distinct indices) is at most budget**1.5, the iteration space of a square matrix
product whose matrices hold the budget.  The cap changes only programs with more than 1.5
indices per mode of the largest tensor (the four-factor chain and the TTMc-shaped
`ijk,jm,kn->imn`, extent 95 instead of 158 or 2000 at 4e6); without it their 4e6 cells
would need on the order of 1e15 multiplies per call at fill 0.55.
A pair whose name fixes a degenerate shape runs at that shape only.  Each kernel is judged on
two input families: the operands as drawn, and the same operands with the first and last
slices along the first mode of every operand with a compressed level emptied (from extent 3),
and the middle slice too from extent 5 (below that the middle is kept, or a six-index operand
of extent 3 would store nothing).  A kernel is correct only when it is correct on both.  The empty-row family is the
one timed, because real matrices have empty rows; it is the class of the 7ac70f2 defect in
`MM ss x dd -> ds` and `SDDMM ss x dd x dd -> ds`.  `analyze.py` names the failing family.

## Decision rule, fixed before any run

- Estimate: the median of per-round paired ratios, candidate / baseline, pooled over processes.
- 95% interval: a two-level bootstrap (processes, then rounds within each process), 2000 draws.
- A/A floor of a cell: the same estimate for each compared arm over its own copy. The floor is
  the largest distance from 1 of either interval bound.
- Per cell:
  - NEUTRAL-OR-BETTER if the upper bound is at most 1 + floor;
  - REGRESSION if the lower bound is above 1 + floor;
  - UNRESOLVED otherwise.
- Every captured cell of every pair gets a row, timed or not. A cell whose candidate raised
  or returned wrong values is CANDIDATE INCORRECT (a correctness finding); a cell whose baseline
  did is BASELINE INCORRECT and owes no timing, because there is no correct kernel to be slower
  than.
- Per pair, the first that applies: candidate incorrect; baseline incorrect; regression (any
  REGRESSION cell); incomplete (a cell not captured or not timed); pass (all cells
  NEUTRAL-OR-BETTER); unresolved. Any REGRESSION is a finding to report, not a shape to tune
  around.
- Positive control: the first arm called twice per counted call must come out as a REGRESSION
  with a ratio between 1.6 and 2.4 in every cell. If it does not, the run measured nothing and
  no verdict in it stands.

Both hosts: the M5 Max (ARM) and mkt1 (AMD EPYC 9334, x86).
