"""Frozen production LoopIR nodes for the Phase-4 dense vertical slice.

This is the first *production* LoopIR subset, frozen from the Phase-3.5 spike
after the Phase-4 responsibility/gap audit (see
``COMPILER_IR_REFACTOR_PHASE4_REVIEW.md``).  It deliberately contains only the
nodes the migrated dense elementwise and dense reduction/matmul families need:
dense iteration, coordinate-addressed loads, plain and ADD-reducing
coordinate-addressed stores, and value-typed binary arithmetic.  Sparse
cursors, physical positions, merges, accumulators, workspaces, tiling, and
parallel nodes are *not* declared here; they are introduced only when a
migrated operation needs them, with the spike as design input.

Discipline carried over from the spike and the binding design decisions:

- every node is a plain frozen dataclass whose children are owned tuples;
- constructors perform no validation — ``verifier.verify_program`` is the
  single fail-closed authority, and adversarial tests must be able to build
  malformed nodes directly;
- identity is never spelling: tensors are identified by production
  ``SymbolId`` values and bound loop coordinates by production ``IndexId``
  values, so CIN provenance survives lowering; ``LoopIRNodeId`` and
  ``DimensionId`` are LoopIR-artifact-local identities allocated by
  :class:`~scorch.compiler.loopir.build.LoopIRBuilder`;
- no C++ spelling, Torch storage field, rendered name, or target policy
  appears in any node; target lowering owns those exclusively.

Differences from the spike recorded by the audit and realized here:

- **Scalar typing.**  The spike was untyped ``float``-only.  Every production
  ``TensorDecl`` carries a :class:`ScalarType`; the verifier requires one
  uniform scalar type per program for this slice and fails closed on mixtures.
- **Reduction semantics without accumulators.**  The dense slice reduces by
  ``StoreReduce`` (ADD) into zero-initialized dense outputs, matching the
  legacy generated kernels; the output-initialization contract is therefore
  explicit on the node rather than implied by a scalar accumulator.
- **Level kinds.**  All four production level kinds are declared so the
  schema is stable across later phases, but only ``DENSE`` is executable in
  this subset; the others fail closed in the verifier with
  ``unsupported_level_kind`` (their iteration is Phase-5 surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Tuple

from ..identity import IndexId, SymbolId


@dataclass(frozen=True, order=True)
class LoopIRNodeId:
    """Identity of one LoopIR node within one LoopIR program artifact."""

    value: int


@dataclass(frozen=True, order=True)
class DimensionId:
    """Identity of one logical dimension (coordinate domain) of a program."""

    value: int


@unique
class ScalarType(Enum):
    """Scalar value type of a declared tensor, target-neutrally."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


@unique
class LevelKind(Enum):
    """Per-level storage layout of a declared tensor, target-neutrally.

    ``DENSE`` is the only kind the Phase-4 production subset executes.  The
    other production kinds are declared so their disposition is explicit and
    the schema does not change shape when later phases represent them; the
    verifier fails closed on them with the stable ``unsupported_level_kind``
    defect until then.
    """

    DENSE = "dense"
    COMPRESSED = "compressed"
    COORDINATE = "coordinate"
    SINGLETON = "singleton"


@unique
class BinaryOp(Enum):
    """Value-typed binary operators; coordinates never flow through these."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"


@unique
class ReduceOp(Enum):
    """Reduction combiners admitted by ``StoreReduce``.

    Only ADD is declared: its identity matches the explicit dense-output
    zero-initialization contract.  Another operator may be added only
    together with an explicit output-initialization contract carrying its
    identity.
    """

    ADD = "add"


class LoopIRNode:
    """Marker base for every production LoopIR node; carries no behaviour."""


@dataclass(frozen=True)
class Expr(LoopIRNode):
    """Base of all value- or coordinate-producing nodes."""

    node_id: LoopIRNodeId


@dataclass(frozen=True)
class Stmt(LoopIRNode):
    """Base of all statement nodes."""

    node_id: LoopIRNodeId


@dataclass(frozen=True)
class IndexValue(Expr):
    """The current coordinate bound by an enclosing :class:`DenseFor`.

    Its coordinate domain is the logical dimension the binding loop
    iterates, so the verifier can reject a coordinate used outside its
    domain.
    """

    index: IndexId


@dataclass(frozen=True)
class Load(Expr):
    """A coordinate-addressed read of an all-dense input tensor.

    ``indices`` are given in logical mode order; each must be a coordinate
    in the domain of the corresponding declared dimension.  The result is
    value-typed with the tensor's scalar type.
    """

    tensor: SymbolId
    indices: Tuple[Expr, ...]


@dataclass(frozen=True)
class BinaryExpr(Expr):
    """A value-typed binary operation over two value-typed operands."""

    op: BinaryOp
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Block(Stmt):
    """An ordered, tuple-owned statement sequence forming one lexical scope."""

    statements: Tuple[Stmt, ...]


@dataclass(frozen=True)
class DenseFor(Stmt):
    """Dense iteration over one declared logical dimension.

    Binds ``index`` to ``0 .. extent - 1`` in order, where ``extent`` is the
    dimension's runtime-resolved extent; the bound coordinate's domain is
    ``dimension``.
    """

    index: IndexId
    dimension: DimensionId
    body: Block


@dataclass(frozen=True)
class Store(Stmt):
    """A coordinate-addressed write of a value into an all-dense output.

    ``indices`` are in logical mode order and domain-checked like
    :class:`Load` indices.
    """

    tensor: SymbolId
    indices: Tuple[Expr, ...]
    value: Expr


@dataclass(frozen=True)
class StoreReduce(Stmt):
    """A coordinate-addressed read-modify-write into an all-dense output.

    Combines the stored element with ``value`` using ``op``
    (``target = target op value``).  Only ADD is admitted, and the contract
    it depends on is explicit: every dense output this subset executes is
    zero-initialized before the program body runs, which is exactly ADD's
    identity.  This is how the dense reduction/matmul family reduces —
    there is deliberately no scalar-accumulator node in this subset.
    """

    tensor: SymbolId
    indices: Tuple[Expr, ...]
    op: ReduceOp
    value: Expr


@dataclass(frozen=True)
class DimensionDecl(LoopIRNode):
    """A declared logical dimension: stable identity plus a display name.

    Dimension identity is the program's shape and domain contract: every
    tensor mode mapped to this identity must have the same runtime extent,
    and only coordinates of the same dimension are comparable.
    """

    node_id: LoopIRNodeId
    dimension: DimensionId
    name: str


@dataclass(frozen=True)
class LevelDecl(LoopIRNode):
    """One physical storage level of a tensor and the logical mode it stores.

    ``mode`` indexes the owning tensor's ``dimensions`` tuple; across one
    tensor the modes must form a permutation of the logical modes, which is
    what keeps physically permuted layouts structurally distinct.
    """

    node_id: LoopIRNodeId
    kind: LevelKind
    mode: int


@dataclass(frozen=True)
class TensorDecl(LoopIRNode):
    """A declared tensor: stable symbol, display name, type, dims, levels.

    ``dimensions`` maps each logical mode to its declared dimension
    identity; ``levels`` lists physical storage levels in storage order;
    ``dtype`` is the tensor's scalar value type.
    """

    node_id: LoopIRNodeId
    symbol: SymbolId
    name: str
    dtype: ScalarType
    dimensions: Tuple[DimensionId, ...]
    levels: Tuple[LevelDecl, ...]


@dataclass(frozen=True)
class LoopProgram(LoopIRNode):
    """One executable LoopIR program: declarations plus a top-level block."""

    node_id: LoopIRNodeId
    dimensions: Tuple[DimensionDecl, ...]
    tensors: Tuple[TensorDecl, ...]
    inputs: Tuple[SymbolId, ...]
    outputs: Tuple[SymbolId, ...]
    body: Block
