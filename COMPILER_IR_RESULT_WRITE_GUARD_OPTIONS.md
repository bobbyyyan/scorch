# What `result_write_pass` should key on: ten options, and the table that decides

Date: 2026-08-14.  Written at tip `01ae768` on
`refactor/compiler-ir-phase3-std-move-call`.  **No production code changed to
produce this document.**  It exists because review §61.7 says the first of the two
recorded narrownesses "means deciding what a result-write guard should key on,
which is a design question rather than a repair", and that decision is Bobby's.

Nothing here is a recommendation until §9, which is one paragraph and clearly
labelled as an opinion.

---

## 1. The problem, stated from the code

`result_write_pass` turns a serial assembly body into a counting body and a
filling body.  A statement that writes the result's storage has to be rewritten
(into a counter bump, or into a store through an exactly-sized pointer) or
dropped.  A statement the pass fails to recognize is *retained* — and in the
count body the retained statement keeps appending to a vector whose declaration
the surrounding transformation has already dropped, so the generated C++ either
fails to compile or computes the wrong positions.

So the pass has to answer one question about every statement it meets: **does this
statement write result storage?**  Today it answers by looking at a string.

`_touches_result_storage` (`src/scorch/compiler/result_write_pass.py:468`) builds
the names `{R}_values`, `{R}{L}_pos` and `{R}{L}_crd` for every compressed level
`L`, and asks whether the statement's `FunctionCallStmt.name` starts with one of
them followed by a dot.  That catches a member call whose receiver is spelled into
the call name, which is how both lowerings spell an append:
`C1_crd.push_back` from the legacy CIN lowerer, `C1_crd.emplace_back` from the
LoopIR ordered-key target.

Two gaps follow, both recorded in §61.4 and neither reachable on today's survey
matrix.

**Gap A — the property is argument-shaped, and the guard reads the callee.**  A
free function that receives a result array as an *argument* has a callee name like
`scorch_concat_chunks`, which starts with none of the result-array prefixes.
`scorch_vector_set` is handled, but only by a hand-written special case at
`:439` that looks at `args[0]` and only accepts a `Var` named `{R}{L}_pos`.

This is less hypothetical than §61.4 makes it sound. The compiler **already
emits four argument-shaped result writes today**, in
`src/scorch/compiler/loopir/parallel_chunk_assembly.py`:

| statement | the result array it writes | line |
| --- | --- | --- |
| `scorch_shift_chunk_positions(C{shared}_pos, …)` | `C{shared}_pos` | `:357` |
| `scorch_concat_chunk_positions(C{L}_pos, …)` | `C{L}_pos` | `:371` |
| `scorch_concat_chunks(C{L}_crd, …)` | `C{L}_crd` | `:382` |
| `scorch_concat_chunks(C_values, …)` | `C_values` | `:392` |

Every one of the four would pass `_touches_result_storage` silently.  The only
thing keeping them away from the guard is that they belong to a *different*
assembly strategy: `owns_two_phase_output()` returns
`assembly_strategy() in TWO_PASS_STRATEGIES` (`lower_llir.py:5584`), and the chunk
merge is `single_pass_chunk_parallel`, a single-pass strategy.  One program gets
one strategy, so the chunk merge and the two-phase pass never run over the same
body.  `parallel_chunk_assembly` also has no importer outside
`src/scorch/compiler/loopir/`, so legacy cannot reach it either.

That is a real separation and it is also a thin one.  The next milestone's own
task list is to fix the two-pass position reconstruction, and a two-pass strategy
that also chunks is the obvious shape for a parallel exact-allocation assembly.
On the day those two meet, four statements the guard cannot see land in front of
it.

**Gap B — `llir.MemberCallStmt` is never inspected at all.**
`rewrite_statement_sequence_member` (`:268`) dispatches `Assign`, `Increment`,
`FunctionCallStmt`, `VarInit` and `IfThenElse`, and anything else falls to
`super()`, which descends into children and returns the statement.  The guard runs
only inside `_rewrite_call_statement`, so a
`MemberCallStmt(base=Var("C1_crd"), member="push_back")` is never offered to it.

`MemberCallStmt` is the structured spelling — `base: Expr`, `member: str` — as
against `FunctionCallStmt`, which carries the receiver inside the dotted name.
Five modules already build `MemberCallStmt`
(`dense_pointer_hoist_pass`, `schedule_lowerer`, `compressed_where_openmp_pass`,
`cin_lowerer`, `single_iteration_loop_pass`), none of them on a result array at
this point in the pipeline.  §61.4 measured **zero** `MemberCallStmt` inspected on
either route.

**The third gap is the one that already bit us.**  §60.6 stage 2: three spellings
(`{R}{L}_crd.emplace_back`, `{R}_values.emplace_back`, and
`scorch_vector_set` on a position array) reached the external C++ compiler
unrewritten, because the pass had been written against legacy's vocabulary only
and its fallback is `return (node,)`.  That gap is closed — both spellings are
recognized and an unrecognized one now raises
`unsupported_result_write_statement` — and it is the historical evidence any
candidate design has to be judged against.

## 2. What the pass legitimately does today, so an allow-list has something to enumerate

§61.4 drove the whole 1,139-record survey matrix in both automatic arms through
legacy's lowering chain — 2,278 lowerings, 102 cell-arms reaching the pass, 204
invocations — and recorded the complete inventory of result-storage-shaped
statements legacy presents:

| form | count | what the pass does with it |
| --- | --- | --- |
| `Assign ArrayAccess` carrying `RESULT_WRITE` metadata | 102 | dropped in count, rewritten in fill |
| `FunctionCallStmt *.sort` | 102 | dropped in count, kept in fill |
| `Assign ArrayAccess C2_crd` | 58 | rewritten |
| `Assign ArrayAccess C2_pos` | 58 | dropped |
| `IfThenElse` on the `C{L}_pos.back()` boundary | 58 | rewritten |
| `Assign ArrayAccess C1_crd` | 44 | rewritten |
| `Assign ArrayAccess C1_pos` | 4 | dropped |

Three facts from that table matter for the options below.

- **The `.sort` is on the workspace, not the result.**  It is
  `{workspace}.sort` (`cin_lowerer.py:1126`, `:1574`;
  `lower_llir.py:7538` and four more), so a rule phrased over result-storage names
  does not have to make an exception for it.
- **The pass's own output uses different names than its input.**  The rewrites
  store into `{R}{L}_crd_data`, `{R}{L}_pos_data` and `{R}_values_data` — the
  exactly-sized pointers — never into the bare vector names.  So "no bare
  result-array name survives in the output" is a well-formed postcondition rather
  than an approximation.
- **LLIR already has a typed marker for a result write, and it is already used
  here.**  `TensorAccessMetadata(role=TensorAccessRole.RESULT_WRITE)`
  (`llir.py:316`, `:324`) is what `_is_result_value_target` (`:323`) matches on to
  find the workspace drain's value store.  The value store is *already* found by
  type rather than by name.  Only the position and coordinate arrays are found by
  name.

**And the statement-type census exists, which options D and H both need.**  The
same instrumented run records every statement the dispatch inspects, not only the
result-storage-shaped ones.  Over the 204 invocations, **ten** distinct statement
types reach `rewrite_statement_sequence_member`:

| type | inspected | dispatched explicitly? |
| --- | --- | --- |
| `Comment` | 2,496 | no — falls to `super()` |
| `VarInit` | 2,232 | yes |
| `BlankLine` | 2,052 | no |
| `Assign` | 868 | yes |
| `ForLoop` | 488 | no |
| `FunctionCallStmt` | 408 | yes |
| `Increment` | 320 | yes |
| `ForLoopAuto` | 204 | no |
| `IfThenElse` | 168 | yes |
| `WhileLoop` | 52 | no |

So the five types the pass names are handled and five more arrive and are passed
through.  A closed list for option H is those ten, measured rather than guessed —
and the census also shows the pass re-inspecting **its own output**
(`Assign ArrayAccess C1_crd_data` and `C2_pos_data`, 58 each in fill mode, both
passed through), because `rewrite_statement_sequence` re-dispatches each returned
statement.  That is worth knowing before writing any rule over names: the `_data`
pointer spellings appear on the input side of the dispatch too.

## 3. The options

Each is stated as: what it keys on, what it catches, what it still misses, which
way it fails on a form nobody anticipated, what it costs, and whether it can be
adopted a piece at a time.

Line counts are estimates of new or changed lines, counted against the current
code.  "Changes emitted C++" means: could a user's generated kernel differ.

---

### Option A — keep name matching, patch the two known holes

**Keys on:** the callee name, as today, plus a hand-written argument check for
each free function known to write a result array, plus a `MemberCallStmt` branch
in the dispatch.

**Catches:** exactly the holes someone has already found.  Both §61.4 gaps, and
gap 1 retroactively.

**Misses:** the next spelling.  There is no mechanism, only a list, and the list
grows only when a gap is found — which on this branch has meant "found by a
kernel that miscompiled".

**Fails:** OPEN.  Any new free function, any new receiver spelling, any new
statement type goes through the `return (node,)` fallback.

**Cost:** `result_write_pass.py` only.  Roughly 15 lines to make
`_touches_result_storage` also scan `node.args` for a `Var` named as a result
array, and 20–30 lines for a `MemberCallStmt` branch that either rewrites the
append or refuses it.  No schema bump.  Emitted C++ unchanged — the guard only
adds refusals on paths nothing reaches today.

**Incremental:** yes, and it is the smallest possible step.  Each hole is
independent.

---

### Option B — key on the arguments, with a signature registry

**Keys on:** which parameter of which helper is written.  A registry maps a
free-function name to the argument positions it writes; a statement passing a
result array into one of those positions is a result write.

**Catches:** every argument-shaped write whose helper is registered.  Today that
would be `scorch_vector_set` plus the four chunk-assembly helpers in §1, and
`scorch_zero_dense` if it is judged to write a result buffer.

**Misses:** an unregistered helper.  Also member-call spellings, since a
receiver embedded in `FunctionCallStmt.name` is not an argument — so B does not
replace the name half, it adds to it.

**Fails:** OPEN for an unregistered helper.  Registration is the same "someone
remembered" mechanism as A, one level up.

**Cost:** a new registry — natural home is `sparse_assembly.py`, which already
owns the one definition of the strategy vocabulary that four layers import, or a
new `result_write_signatures.py` (~40 lines).  `result_write_pass.py` gains an
argument scan against it (~30 lines).  `parallel_chunk_assembly.py` should import
the registry rather than keep its own `PARALLEL_CHUNK_RUNTIME_SPELLINGS` tuple in
sync (~10 lines), and `dynamic_vector_access_pass.py` should register
`scorch_vector_set` where it configures it (~5 lines).  Call it 100–140 lines
across three or four files.  No schema bump.  Emitted C++ unchanged.

**Incremental:** yes.  The registry can start with the five known helpers.

**One thing B buys that nothing else does:** it makes the write *direction*
explicit, so a helper that only reads a result array (there are none today) stays
distinguishable from one that writes it.

---

### Option C — key on the variable: any mention must be recognized or allowed

**Keys on:** the appearance of a result-storage name anywhere in the statement.
Every statement mentioning one is either handled by a rewrite or matched against
an explicit list of permitted mentions; anything else is refused.

**Catches:** all three historical gaps, and every future spelling that reaches
the pass, because it does not care how the statement is shaped.

**Misses:** nothing about writes.  Its failure mode is the opposite one.

**Fails:** CLOSED.

**The cost that matters is not lines, it is the read exemptions.**  Result-storage
names appear in expressions the pass must leave alone: `{R}{L}_crd.size` inside a
`FunctionCall` used as an index (`cin_lowerer.py:2031`, `:2069`;
`iter_lattice.py:674`, `:1314`, `:1339`; `lower_llir.py:4670`, `:7571`, `:7702`,
`:7815`, `:7904`, and more), and `{R}{L}_pos.back` inside the boundary
conditional's own condition (`cin_lowerer.py:1987`, `iter_lattice.py:1257`,
`lower_llir.py:7555`).  A statement containing any of those *mentions* a result
array while writing nothing.  So C must either distinguish statement position from
expression position, or carry an allow-list of read spellings that is exactly as
reactive as A's list of write spellings — with the difference that forgetting an
entry over-refuses a legal program instead of miscompiling it.

**Cost:** `result_write_pass.py`, and the check has to move out of
`_rewrite_call_statement` to the top of `rewrite_statement_sequence_member`,
because a check inside a per-type handler is exactly why gap B exists.  60–100
lines plus the exemption list.  No schema bump.  Emitted C++ unchanged.

**Incremental:** awkwardly.  Turning it on refuses programs unless the exemption
list is already complete, so the honest sequence is: run it in report-only mode
over the frontier, enumerate what it flags, then turn it on.  That is one census
run, and §61.4's harness is most of the instrumentation.

---

### Option D — invert the default: allow-list the permitted forms, refuse the rest

**Keys on:** the statement form.  A closed list of forms is permitted to appear in
an assembly region at all; every one has a declared disposition (rewrite, drop,
keep); anything not on the list is refused.

**Catches:** every unanticipated form, by construction.  All three historical
gaps.

**Misses:** nothing structurally.  In practice it misses whatever a permitted form
can be *made* to do — an allow-listed `Assign ArrayAccess` whose array name is new
still matches the form.  So D's granularity has to be the form *plus* the array
identity, which is where it starts to need C's or E's machinery underneath.

**Fails:** CLOSED, and loudly: new codegen breaks until registered, which is the
intended behaviour.

**Cost:** `result_write_pass.py`, 120–180 lines.  §2 gives both halves of the
permitted list as *measurements over legacy* rather than guesses: seven
result-storage-shaped forms, and ten statement types with their counts.  The typed
route needs the same census, which is one flag on the same harness.
`ResultWriteContext` already carries `compile_options` (`:82`), so a staged
rollout has somewhere to hang a gate.  No schema bump.  Emitted C++ unchanged.

**Incremental:** as a flag, yes.  As a default, it is one switch — you cannot
half-invert a default.

**The cost nobody should be surprised by:** every future lowering that emits an
assembly region will hit this before it works, including on days when the person
hitting it did not know this pass existed.  That is the price of failing closed and
it is a real one.

---

### Option E — make result storage a typed thing in LLIR, not a naming convention

**Keys on:** the type.  A result-storage reference carries
`TensorAccessMetadata(role=RESULT_WRITE)`; "touches result storage" becomes a
metadata query.

**This is less of a leap than it looks, and the reason is worth reading before
costing it.**  `Var` already has the field —
`tensor_access: Optional[TensorAccessMetadata]` at `llir.py:366`, declared
`compare=False, repr=False` — and `ArrayAccess` has it at `:950`.  The role enum
already has `RESULT_WRITE`.  The pass already uses exactly this mechanism for one
of the three arrays: `_is_result_value_target` (`:323`) matches the drain's value
store by `tensor_id` and `role`, not by name.  So E is not "add a type system", it
is "finish attaching the marker the coordinate and position arrays never got".

**Catches:** every spelling, in every statement shape, because the marker travels
on the reference rather than on the syntax around it.

**Misses:** a reference the producing lowering forgot to tag.  That is the whole
risk and it is not small: the marker is only as complete as the emission sites.

**Fails:** OPEN on producer omission — which is the same failure class as A, moved
from the consumer to the producer.  E is only fail-closed when paired with a
postcondition (option F) that catches an untagged reference by its residue.

**Cost, in concrete terms:**

- LLIR has **no canonical schema string** — `printer.CANONICAL_SCHEMA` and
  `plan_identity.CANONICAL_PLAN_SCHEMA` are LoopIR's, and `grep -n schema
  src/scorch/compiler/llir.py` is empty.  So **no schema bump**, unlike the v11→v12
  move §60.4 records.
- The traversal validators currently **refuse** `tensor_access` in seven
  positions: an assignment-target `Var` (`llir.py:1107`), a `Var` or `ArrayAccess`
  inside an assignment index (`:1033`, `:1058`), an assignment `ArrayAccess`'s own
  `array` field (`:1195`), a `MemberAccess` root (`:1172`), and two `AddressOf`
  positions (`:1467`, `:1530`).  Any position where a tagged result-array `Var`
  would legitimately appear has to be admitted, one validator at a time, each with
  a reason.  A `FunctionCallStmt` argument and a `MemberCallStmt.base` are *not*
  currently refused, which is convenient: the two gap positions need no validator
  change.
- Emission sites that build a dotted result-array name today:
  `loopir/lower_llir.py` 73, `torch_cpp_abi.py` 38, `cin_lowerer.py` 26,
  `iter_lattice.py` 12.  Not all of those build a `Var` that needs tagging, but
  that is the search space.  A shared constructor helper (~30 lines) plus the
  sites that matter puts this at **300–600 lines across five to eight files**.
- **Emitted C++ unchanged**, and this is provable rather than hoped: `codegen`
  never reads `tensor_access`, and the field is `compare=False`, so it cannot
  change any equality-keyed decision either.

**Incremental:** yes per producer, and that is also the trap — the guard cannot
*rely* on the marker until every producer participates, so during the migration E
is E-plus-A, with two mechanisms live and the weaker one still load-bearing.

---

### Option F — check the postcondition instead of the input

**Keys on:** what is left.  After the rewrite, walk the output and assert that no
statement still writes the result's own storage.  It never asks what was matched;
it asks what survived.

**There is a working prototype, and it has already been run.**
`~/.cache/scorch-codex/assembly-strategy/harness/result_write_reach.py` contains
`residual_writes()` (~55 lines), which walks the rewritten body and reports every
statement still writing result storage.  It inspects `FunctionCallStmt` appends by
name, `scorch_vector_set` **by its argument**, `MemberCallStmt` **by its base**,
`Assign` by array name and by `RESULT_WRITE` metadata, and `Increment`/`VarInit`
on the position pointers, descending seven body fields.  §61.4 drove it over 204
invocations of the pass on the legacy route and 2 on the typed route: **zero
residual result writes**.

So the adoption cost of F is not merely low, it is *measured*: promoting it to
production refuses nothing that runs today.  That is a stronger statement than any
other option here can make, and it is the only one that has already been run
against the whole matrix.

**Two versions, and the difference decides whether F is really fail-closed.**

- **F-weak**, the prototype as written: it enumerates the write *forms* it knows
  (append by name, `scorch_vector_set` by argument, indexed assign, pointer
  increment).  A brand-new helper — `scorch_concat_chunks(C_values, …)` — is not
  in that enumeration, so F-weak misses gap A.
- **F-strong**: after the pass, no statement anywhere in the output may mention a
  bare result-array name at all.  §2 establishes this is well formed — the pass's
  own output uses the `_data` pointer names, and the workspace `.sort` is not a
  result array — so the bare names should be absent, and any occurrence is a
  residue regardless of the shape wrapping it.  F-strong catches unknown spellings
  by construction, which is the entire point of checking the output.

**Catches:** F-strong catches all three historical gaps, and any future one, in
the count/fill bodies.

**Misses:** anything outside the pass's output.  F says nothing about a result
write that was *correctly removed from the body* and reappears somewhere the pass
did not produce.  It also localizes badly: it says "something survived", not
"this statement was not understood", so a defect it catches still needs
diagnosing.  And it cannot catch a *wrong* rewrite, only an absent one.

**Fails:** CLOSED, and closed against exactly the class nobody anticipated.

**Cost:** one new function in `result_write_pass.py` or a sibling module, 60–90
lines, called at the end of `rewrite_result_writes` (`:594`).  No schema bump.
Emitted C++ unchanged.

**Incremental:** yes, and independently of every other option — F composes with A
through E and G rather than competing with them.  It can also ship in report-only
mode first, though §61.4's run means that step is already done for the current
matrix.

---

### Option G — normalize first, then match exactly one thing

**Keys on:** nothing, at the guard.  A new pass ahead of `RESULT_WRITE`
canonicalizes every result-write spelling into one LLIR form, and the guard then
recognizes exactly that form.

**Catches:** whatever the normalizer handles — so the recognition problem moves
rather than disappears.  It becomes a *better-placed* problem: one pass, whose
only job is this, refusing what it cannot canonicalize.

**Misses:** the same unknown spellings, now at the normalizer.  G's value is that
there is one place to look and one refusal to write, not that the set of
recognized spellings grows on its own.

**Fails:** CLOSED if the normalizer refuses what it cannot canonicalize, OPEN if
it passes it through.  That is a choice G has to make explicitly, and it is the
same choice as D.

**Cost, and this is the option with the real risk:**

- A new pass module, 150–250 lines.
- `llir_pass_manager.py`: `CURRENT_LLIR_PASSES` is a **frozen order that
  `run_production_pipeline` validates before executing any user program work**
  (§61.4).  Adding a position changes that frozen order — the nearest thing in
  this design space to a schema bump, and it needs whatever proof the freeze
  exists to demand.
- **Emitted C++ is NOT obviously unchanged.**  Every pass after the new one sees a
  different body, and `DYNAMIC_VECTOR_ACCESS` at position 7 — the pass that
  *produces* `emplace_back` and `scorch_vector_set` — is one of them.  Legacy runs
  this whole order on release default dispatch.  So G needs the full
  release-neutrality measurement §61.2 built (506 production case-arms in both
  arms, the 20-source corpus, the 42-case grid, the 86-case audit), and it could
  genuinely fail it.  No other option here can change a shipped byte; G can.

**Incremental:** no.  A normalizer that runs on some bodies and not others gives
the guard two forms to match, which is the situation it was meant to end.

---

### Option H — make the dispatch total over the statement vocabulary

*Not in the original list.  Added because it is the only cheap option that closes
gap B by construction.*

**Keys on:** the statement *type*.  `rewrite_statement_sequence_member` names five
types and falls through for the rest.  H replaces the fallthrough with an explicit
set of types the pass has considered, and refuses a type not in it.

**Catches:** gap B — `MemberCallStmt` is an unconsidered type, so it is refused
rather than silently retained.  Any future statement type, likewise.

**Misses:** gaps A and 1 entirely.  Both arrive as `FunctionCallStmt`, a type the
pass already dispatches, so H sees a known type and waves it through.

**Fails:** CLOSED for unknown types, OPEN for unknown spellings inside a known
type.

**Cost:** 25–40 lines in one file.  No schema bump.  Emitted C++ unchanged.  **The
census the permitted set has to be built from already exists** — §2's type table:
ten types reach the dispatch, five named and five falling through, measured over
204 invocations of the legacy route.  So H needs no preparatory measurement on the
legacy side, only the same census on the typed route, which is one flag on the
same harness.

**Incremental:** yes, trivially, and it is nearly free.

**Why it is worth listing separately:** it is the one option whose catch set is
disjoint from the argument-shaped problem, so it composes cleanly with B — H
closes the type hole, B closes the argument hole, and the two together cost less
than C.

---

### Option I — have the producer declare its own writes

*Not in the original list.  Added because it inverts who does the work, which
changes the failure mode in a way none of A–H does.*

**Keys on:** a declaration.  `ResultWriteContext` gains an inventory of the
result writes the emitting lowering says it produced; the pass rewrites exactly
those, refuses one it cannot rewrite, and refuses a statement claiming to be a
result write that is not in the inventory.

**Catches:** everything a participating producer declares.  The producer knows
which statements it built and why, which is information the pass is currently
reverse-engineering from strings.

**Misses:** whatever a producer forgets to declare.

**Fails:** OPEN on producer omission, same as E.  It also has a failure mode E
does not: an inventory that goes *stale* when a lowering is edited but its
declaration is not.

**Cost:** `ResultWriteContext` gains a field (~10 lines);
`compressed_where_openmp_pass.py` builds the context at two sites (`:1078`,
`:1153`) and would have to thread the inventory through; every family that emits
result writes fills one.  200–350 lines.  `ResultWriteContext` is not serialized
into any canonical dump, so no schema bump.  Emitted C++ unchanged.

**Incremental:** per family, with the same trap as E — the guard cannot trust the
inventory until every family declares one.

---

### Option J — register the generator, not the statement

*Not in the original list.  Added because it is the cheapest structurally-closed
option, and because it is closed against precisely the way gap 1 arrived.*

**Keys on:** provenance at the region level.  The body handed to the pass carries
the identity of the lowering that built it, and the pass accepts only identities
it has been taught.  A body from a lowering nobody registered is refused whole.

**Catches:** gap 1 exactly.  The ordered-key target's body arrived from a lowering
the pass had never seen, and every one of its three unrecognized spellings would
have been refused together, at a refusal that names the family — which is also the
diagnosis a reader wants.

**Misses:** a *registered* lowering that starts emitting a new spelling.  That is
how gap A would arrive: `parallel_chunk_assembly` is an existing module gaining
reach, not a new one.  So J is closed against new producers and open against
existing producers changing.

**Fails:** CLOSED for a new generator, OPEN for a registered generator's new
statement.

**Cost:** `result_write_pass.py` plus `compressed_where_openmp_pass.py` plus a
registry, 60–100 lines.  No schema bump.  Emitted C++ unchanged.

**Incremental:** yes.

---

## 4. The table: run every option against the three gaps that actually happened

Three real gaps.  §60.6 stage 2's fail-open, which shipped a miscompiling kernel
before it was caught.  §61.4's argument-shaped gap A.  §61.4's `MemberCallStmt`
gap B.

**Two columns per gap, and the second column is why the table is worth reading.**
Several options are defined partly by the gaps they were told about, so "does it
catch it" is trivially yes for them.  The question that separates the designs is
whether the option would have caught the gap **on the day the code arrived, before
anyone knew to look for it**.  That is the "blind" column.  The "known" column is
whether the option catches it once the gap has been described.

The three gaps, by the short names used in the column headers:

- **gap 1** — `emplace_back` / `values.push_back` / `scorch_vector_set` fail-open,
  §60.6 stage 2.  The one that shipped a miscompiling kernel.
- **gap A** — the argument-shaped write, §61.4 narrowness 1.
- **gap B** — `MemberCallStmt` never inspected, §61.4 narrowness 2.

| option | 1 blind | 1 known | A blind | A known | B blind | B known |
| --- | --- | --- | --- | --- | --- | --- |
| **A** name matching, patch the holes | **no** | yes | **no** | yes | **no** | yes |
| **B** argument keying + signature registry | partly¹ | yes | **no**² | yes | **no** | no³ |
| **C** any mention must be recognized or allowed | **yes** | yes | **yes** | yes | **yes**⁴ | yes |
| **D** invert the default, allow-list the forms | **yes** | yes | **yes** | yes | **yes**⁴ | yes |
| **E** typed result storage in LLIR | yes⁵ | yes | yes⁵ | yes | yes⁵ | yes |
| **F-weak** postcondition, enumerated forms | **yes** | yes | **no**⁶ | yes | **yes** | yes |
| **F-strong** postcondition, no bare name survives | **yes** | yes | **yes** | yes | **yes** | yes |
| **G** normalize first | yes⁷ | yes | yes⁷ | yes | yes⁷ | yes |
| **H** total statement-type dispatch | **no** | no³ | **no** | no³ | **yes** | yes |
| **I** producer-declared inventory | yes⁵ | yes | yes⁵ | yes | yes⁵ | yes |
| **J** registered generator | **yes** | yes | **no**⁸ | yes | **no**⁸ | yes |

1. B catches the `scorch_vector_set` third of gap 1, because that one is
   argument-shaped.  The two `emplace_back` spellings put the receiver in the
   callee name, which is not an argument, so B alone does not see them.
2. Blind-no for a reason worth stating: the four chunk helpers in §1 exist in the
   tree today and are in no registry.  B is only closed for what someone
   registered, and nobody registered them.
3. Not merely "would not have caught it" — cannot catch it.  B keys on arguments
   and never inspects `MemberCallStmt`; H keys on types and gap 1 and gap A both
   arrive as an already-dispatched type.  These are structural blind spots, not
   omissions.
4. Only if the check runs at the top of `rewrite_statement_sequence_member`.  A
   C-style or D-style check placed inside `_rewrite_call_statement`, where the
   current guard lives, never sees a `MemberCallStmt` — which is gap B's whole
   mechanism, reproduced.
5. Conditional on the producing lowering having attached the marker (E) or filed
   the declaration (I).  For gap 1 specifically the producer was a *new* lowering
   whose author did not know the pass existed, which is exactly the case where a
   producer obligation is most likely to be missed.  Read these three cells as
   "yes if the discipline held", and the discipline is the thing that failed.
6. F-weak enumerates `scorch_vector_set` by name.  `scorch_concat_chunks` is not
   in the enumeration and would survive as an unflagged residue.  This is the
   single cell that separates the two versions of F, and it is why the
   distinction is worth making rather than calling both "F".
7. Conditional on the normalizer refusing what it cannot canonicalize.  If it
   passes an unknown spelling through, G inherits the fail-open it was built to
   remove.
8. J refuses a body from an *unregistered* lowering.  Gap A's four statements come
   from `parallel_chunk_assembly`, and gap B's spelling would come from a module
   that already builds `MemberCallStmt` — both existing, both registered.

### What the table says without recommending anything

- **Option A would have caught none of the three blind.**  By the prompt's own
  standard — "an option that would not have caught the gap that already bit us is
  not a candidate" — A is not a candidate on its own.  It remains the correct
  *first commit* of several other options.
- **Two options are blind-closed on all three without depending on anyone
  remembering anything: C, D, and F-strong.**  That is three, not two, and C's
  cost is paid in over-refusal rather than in lines.
- **E and I are blind-closed only if the producer participated**, and gap 1 is the
  case where the producer did not.  Both remove the bug class *in the long run*
  and neither closes it on the day a new lowering lands.
- **H and J each close exactly one gap and are structurally blind to others**, so
  neither is a whole answer.  Both are cheap enough to be part of one.
- **F is the only option whose adoption cost is measured rather than estimated.**
  §61.4 ran its prototype over 204 invocations of the pass and found zero
  residuals, so it refuses nothing that runs today.
- **G is the only option that can change a shipped byte.**  Everything else is
  guard-only.

## 5. What none of these fixes

- **A wrong rewrite.**  Every option here answers "did the pass see this
  statement".  None answers "did it rewrite it correctly".  §60.6 stage 3 is the
  live example: the two-pass position reconstruction on the ordered-key family
  compiles, executes, and produces wrong positions.  No guard in this document
  would have caught it; the differential run did.
- **The `_find_serial_coordinate` silent-`None` hazard** (§61.4).
  `_rewrite_if_statement` calls it in fill mode, and on `None` the fill body emits
  **no parent coordinate store at all**, silently.  Probed at 58 of 58 successes,
  so it never fires — but it is a *missing* write rather than an unrecognized one,
  and only F, which inspects the output, is even the right shape of check for it.
  Even F would need to assert a store is present rather than that none survives.
- **`codegen.py` silently dropping `pre_parallel_body` / `post_parallel_body` on a
  non-parallel `ForLoop`** (§60.10).  A different fail-open in a different layer,
  recorded and unfixed.

## 6. If two are combined

Not a recommendation, an observation about how the options compose, since several
are additive rather than exclusive.

- **F composes with everything.**  It checks the output; every other option
  changes what the input recognizer does.
- **B + H is the cheap pair with disjoint catch sets**: B closes the argument
  hole, H closes the type hole, ~140–180 lines total, and between them they cover
  gaps A and B blind.  Neither covers gap 1 blind.
- **E + F is the pair that removes the bug class and stays closed while it is
  being removed**: E moves recognition to a type over however many milestones it
  takes, F catches every reference E has not reached yet.
- **D subsumes C** for practical purposes; running both is redundant.
- **G makes A adequate**, which is the argument for G: with one canonical form
  there is one thing to recognize.  It is also the only option that has to clear
  the release-neutrality gate.

## 7. Where the numbers in this document come from

| claim | source |
| --- | --- |
| the guard's current shape | `src/scorch/compiler/result_write_pass.py:468`, `:439`, `:268` |
| four argument-shaped result writes exist today | `src/scorch/compiler/loopir/parallel_chunk_assembly.py:357`, `:371`, `:382`, `:392` |
| chunk assembly and the two-phase pass cannot co-occur | `lower_llir.py:5584`, `sparse_assembly.py:73` and `:79`; `parallel_chunk_assembly` has no importer outside `loopir/` |
| the seven statement forms legacy presents, 102 cell-arms, 204 invocations, zero residuals | review §61.4, receipts `result_write_reach_legacy_head.json`, `result_write_boundary_legacy.json` |
| `Var` and `ArrayAccess` already carry `tensor_access`; `RESULT_WRITE` already exists and is already matched | `llir.py:366`, `:950`, `:316`, `:324`; `result_write_pass.py:323` |
| seven traversal positions refuse `tensor_access` | `llir.py:1033`, `:1058`, `:1107`, `:1172`, `:1195`, `:1467`, `:1530` |
| LLIR has no canonical schema string | `grep -n schema src/scorch/compiler/llir.py` is empty; the v11→v12 bump in §60.4 is LoopIR's |
| emission-site counts (73 / 38 / 26 / 12) | dotted result-array name constructions in `loopir/lower_llir.py`, `torch_cpp_abi.py`, `cin_lowerer.py`, `iter_lattice.py` |
| the `.sort` is on the workspace | `cin_lowerer.py:1126`, `:1574`; `lower_llir.py:7538`, `:7827`, `:8099`, `:9226`, `:10102`, `:10473` |
| the frozen pass order and its validation | review §61.4; `llir_pass_manager.py` `CURRENT_LLIR_PASSES`, `run_production_pipeline` |
| F's prototype and its measured result | `~/.cache/scorch-codex/assembly-strategy/harness/result_write_reach.py`, `residual_writes()` |

## 8. What this document does not do

- **No production code changed.**  `git diff` against `01ae768` touches no file
  under `src/`.
- **No option is built, and no option is half-built.**  There is no flag, no
  scaffolding and no "while I was in there".
- **No new measurement was taken for it.**  Every number is read out of the code
  at the tip or out of §61.4's sealed receipts, cited in §7.
- **The two gaps stay unreachable.**  Nothing here makes gap A or gap B live, and
  nothing here refuses a program that compiles today.

## 9. The one paragraph that is an opinion

If it were mine to pick: **F-strong now, E over the next few milestones, and A's
two patches in the same commit as F** — and I would not build C, D, G, H, I or J.
F-strong is the only option that closes all three gaps blind without depending on
anyone remembering to register anything, its adoption cost is measured rather than
estimated (zero residuals over 204 invocations, so it refuses nothing that runs
today), it composes with every other option instead of competing, and it is 60–90
lines in one file that cannot change a shipped byte.  Its real weakness is
diagnosis — it reports that something survived, not which statement was
misunderstood — and A's two patches are the cheap fix for exactly that, turning
the two known shapes into named refusals at the point of misunderstanding while
F stands behind them for the shapes nobody has thought of.  E is the only option
that removes the bug class rather than containing it, LLIR already has the field
and the role enum and already uses them for one of the three arrays, so it is
finishing something rather than starting it; it is worth the several hundred lines
precisely because F makes it safe to do slowly, catching every reference E has not
reached yet.  I would not build D or C: both are unconditionally closed, which is
right, but both pay for it by needing a complete enumeration of legal *reads*
before they can be turned on, and a guard that over-refuses a legal program is a
different bug that will be fixed by widening the allow-list until it stops
failing — which is A's reactive list again, wearing better clothes.  G is the only
option that can change what ships, and it buys a tidier recognizer for that risk,
which is the wrong trade on a branch whose release-neutrality gate has cost this
much to establish.  H and J are each cheap and each structurally blind to two of
the three gaps, so they are tempting and they are not answers.
