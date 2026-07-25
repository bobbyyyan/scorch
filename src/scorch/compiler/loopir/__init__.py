"""Production LoopIR: the migrated vertical slices of the target design.

This package owns the frozen production LoopIR subset — the Phase-4 dense
families, the Phase-5 sparse level families, and the Phase-6 scheduled
forms (reorders, affine direct tiles, stack-accumulation workspace
regions, sparse panels, staged operand relayouts, and heap result-tile
accumulation regions (currently a rank-2 emitted slice) — and its
strangler-path pipeline:

- ``nodes``: frozen, tuple-owned production LoopIR nodes;
- ``build``: the identity-allocating construction API;
- ``verifier``: the single fail-closed structural verifier;
- ``printer``: deterministic printing and canonical serialization;
- ``oracle``: the production-owned test/debug semantic oracle;
- ``lower_cin``: normalized-CIN-to-LoopIR lowering for the migrated
  families;
- ``iterdomain``: the pure iteration-domain/merge-lattice analysis;
- ``levels``: the format-neutral level-storage interface and CSR adapters;
- ``schedule_passes``: pure typed loop-reorder, affine-tiling,
  stack-workspace, sparse-panel, staged-relayout, and heap result-tile
  passes applying a verified ``LoopPlan`` to a verified base program;
- ``lower_llir``: LoopIR lowering into the existing structured LLIR (the
  current target-specific CxxIR boundary) and the managed pass pipeline;
- ``pipeline``: the test/debug compile-and-execute driver plus the curated
  legacy/LoopIR shadow comparison helpers.

Neutrality contract: nothing in production imports this package.  Normal
``import scorch``, default compilation, legacy correctness paths, and release
JIT never load or execute it; dedicated LoopIR tests import the submodules
directly.  This ``__init__`` deliberately imports nothing so loading the
package namespace cannot drag pipeline modules into an unrelated process.
"""
