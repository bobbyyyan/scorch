"""Phase-3.5 sparse LoopIR feasibility spike.

This package is a strictly experimental, target-neutral prototype used only by
the Phase-3.5 go/no-go review.  It is not imported by production compilation,
JIT, public APIs, caches, LLIR, or C++ codegen, and it must stay independent of
``iter_lattice``, ``cin_lowerer``, Torch, native code, rendered-name parsing,
and callbacks into the existing lowerer.  The only production dependency it may
reuse is the stable identity module.

The schema here is a feasibility candidate, not a frozen design.  The initial
go/no-go review was corrected to NO-GO for general level-based LoopIR: the spike
must gain explicit parent positions and logical mode/domain identity, then pass
a repeated review before Phase 4 begins.  Modules:

- ``nodes``: frozen, tuple-owned generic IR nodes and their semantics;
- ``verifier``: the fail-closed structural verifier;
- ``csr``: the canonical plain-Python CSR container;
- ``interp``: the small plain-Python interpreter (test/debug oracle);
- ``programs``: the hand-authored feasibility programs (CSR SpMV, CSR+CSR
  union add, two-CSR intersection multiply).
"""
