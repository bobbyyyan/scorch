"""Production LoopIR: the Phase-4 dense vertical slice of the target design.

This package owns the first frozen production LoopIR subset and its
strangler-path pipeline for the migrated dense elementwise and dense
reduction/matmul families:

- ``nodes``: frozen, tuple-owned production LoopIR nodes (dense subset);
- ``build``: the identity-allocating construction API;
- ``verifier``: the single fail-closed structural verifier;
- ``printer``: deterministic printing and canonical serialization;
- ``oracle``: the production-owned test/debug semantic oracle;
- ``lower_cin``: normalized-CIN-to-LoopIR lowering for the dense families;
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
