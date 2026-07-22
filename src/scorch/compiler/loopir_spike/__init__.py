"""Phase-3.5 sparse LoopIR feasibility spike.

This package is a strictly experimental, target-neutral prototype used only by
the Phase-3.5 go/no-go review.  It is not imported by production compilation,
JIT, public APIs, caches, LLIR, or C++ codegen, and it must stay independent of
``iter_lattice``, ``cin_lowerer``, Torch, native code, rendered-name parsing,
and callbacks into the existing lowerer.  The only production dependency it may
reuse is the stable identity module.

The schema here is a feasibility candidate, not a frozen design.  This is the
repeat-review revision required by the corrected NO-GO: tensors declare stable
logical dimensions and an explicit physical-level-to-logical-mode mapping,
sparse iteration binds physical positions separately from coordinates with
explicit dominating-parent linkage, scalar values belong to the leaf level
only, and the interpreter executes through a format-neutral level-storage
interface with CSR as one adapter.  Modules:

- ``nodes``: frozen, tuple-owned generic IR nodes and their semantics;
- ``verifier``: the fail-closed structural verifier;
- ``csr``: the canonical plain-Python CSR container;
- ``levels``: the format-neutral level storage plus the CSR adapters;
- ``interp``: the small plain-Python interpreter (test/debug oracle);
- ``programs``: the hand-authored feasibility programs (CSR SpMV, CSR+CSR
  union add, two-CSR intersection multiply, DCSR SpMV, CSC SpMV, and a
  three-level CSF-like row contraction).
"""
