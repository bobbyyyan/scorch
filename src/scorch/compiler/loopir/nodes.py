"""Frozen production LoopIR nodes for the dense and sparse vertical slices.

The dense subset was frozen by Phase 4 (see
``COMPILER_IR_REFACTOR_PHASE4_REVIEW.md``).  Phase 5 extends it deliberately
with the sparse concepts the migrated level-based families exercise —
compressed position iteration, parent-position-linked cursors, coordinate
and leaf-value access, structured UNION/INTERSECTION merging, and ordered
CSR output assembly — revised from the reviewed Phase-3.5 spike, not
copied wholesale.  Concepts the migrated families
do not exercise are deliberately *not* declared: there are no accumulators,
integer constants, physical position loads (dense value-bearing leaves below
sparse levels), workspaces, tiles, or parallel nodes yet.

Discipline carried over from the spike and the binding design decisions:

- every node is a plain frozen dataclass whose children are owned tuples;
- constructors perform no validation — ``verifier.verify_program`` is the
  single fail-closed authority, and adversarial tests must be able to build
  malformed nodes directly;
- identity is never spelling: tensors are identified by production
  ``SymbolId`` values and bound loop coordinates by production ``IndexId``
  values, so CIN provenance survives lowering; ``LoopIRNodeId``,
  ``DimensionId``, ``CursorId``, and ``PositionId`` are
  LoopIR-artifact-local identities allocated by
  :class:`~scorch.compiler.loopir.build.LoopIRBuilder`;
- no C++ spelling, Torch storage field, rendered name, or target policy
  appears in any node; target lowering owns those exclusively.

The three separated spaces the sparse schema rests on (proven by the spike):

- **Logical dimensions** are the shape and coordinate-domain contract:
  two extents are equal because they are the same :class:`DimensionId`, and
  two coordinates are comparable because they index the same dimension.
- **Physical levels** list storage order; each :class:`LevelDecl` records
  the logical mode it stores, keeping physically permuted layouts (CSR
  versus CSC) structurally distinct.
- **Physical positions** are bound and linked explicitly: sparse iteration
  binds a :class:`PositionId` beside the coordinate it resolves, and a
  compressed child level names its dominating parent position
  (:class:`RootPosition`, :class:`DensePosition`, or a bound
  :class:`PositionValue`).  Positions are never recovered from coordinates,
  rendered names, callbacks, or interpreter state.

Scalar typing, reduction-without-accumulators, and the declared-but-
fail-closed disposition of ``COORDINATE``/``SINGLETON`` levels are unchanged
from Phase 4; ``COMPRESSED`` becomes executable in this subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Tuple

from ..identity import IndexId, SymbolId


@dataclass(frozen=True, order=True)
class LoopIRNodeId:
    """Identity of one LoopIR node within one LoopIR program artifact."""

    value: int


@dataclass(frozen=True, order=True)
class DimensionId:
    """Identity of one logical dimension (coordinate domain) of a program."""

    value: int


@dataclass(frozen=True, order=True)
class CursorId:
    """Identity of one sparse cursor binding within a LoopIR program."""

    value: int


@dataclass(frozen=True, order=True)
class PositionId:
    """Identity of one bound physical storage position within a program."""

    value: int


@unique
class ScalarType(Enum):
    """Scalar value type of a declared tensor, target-neutrally."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


@unique
class LevelKind(Enum):
    """Per-level storage layout of a declared tensor, target-neutrally.

    ``DENSE`` and ``COMPRESSED`` are the kinds this subset represents and
    executes.  ``COORDINATE`` and ``SINGLETON`` are declared so their
    disposition is explicit and the schema does not change shape when a
    later phase represents them; the verifier fails closed on them with the
    stable ``unsupported_level_kind`` defect until then.
    """

    DENSE = "dense"
    COMPRESSED = "compressed"
    COORDINATE = "coordinate"
    SINGLETON = "singleton"


@unique
class MergeMode(Enum):
    """Body-emission policy of a :class:`MergedSparseFor`."""

    UNION = "union"
    INTERSECTION = "intersection"


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
class FloatConst(Expr):
    """A value-typed floating-point constant.

    Its principal production use is the explicit unaligned-read default a
    UNION-merged :class:`CursorValue` carries.
    """

    value: float


@dataclass(frozen=True)
class RootPosition(Expr):
    """The unique physical root position that dominates every level-0 level.

    It is position-typed and is the only admissible parent of a level-0
    storage level of any tensor.
    """


@dataclass(frozen=True)
class DensePosition(Expr):
    """The physical position within one DENSE level of a declared input.

    Dense storage positions are arithmetic: the position equals
    ``parent * extent + coord``, where ``extent`` is the resolved extent of
    the logical dimension the level stores.  ``parent`` must be the
    dominating position of the immediately enclosing level of the same
    tensor (the root position for level 0), and ``coord`` must be a
    coordinate in that dimension's domain.
    """

    tensor: SymbolId
    level: int
    parent: Expr
    coord: Expr


@dataclass(frozen=True)
class PositionValue(Expr):
    """The physical position bound by an enclosing :class:`SparseFor`.

    Position-typed with the (tensor, level) linkage of the binding loop's
    cursor; this is how a compressed child level names its dominating parent
    position.
    """

    position: PositionId


@dataclass(frozen=True)
class CursorValue(Expr):
    """The stored scalar at an in-scope sparse cursor's current position.

    Only a cursor over the value-bearing leaf level (the last physical
    level) of its tensor owns scalar values; reading a structural non-leaf
    cursor is the ``non_leaf_value`` defect.

    Inside a :class:`SparseFor` or an INTERSECTION :class:`MergedSparseFor`
    the owning cursor is aligned with the loop coordinate whenever the body
    runs, so ``default`` must be ``None`` (it could never be observed).
    Inside a UNION merge the cursor may be unaligned at the emitted
    coordinate, so a value-typed ``default`` is required and is evaluated
    exactly when the cursor is not aligned.
    """

    cursor: CursorId
    default: Optional[Expr]


@dataclass(frozen=True)
class Load(Expr):
    """A coordinate-addressed read of an all-dense input tensor.

    ``indices`` are given in logical mode order; each must be a coordinate
    in the domain of the corresponding declared dimension.  The result is
    value-typed with the tensor's scalar type.  Compressed tensors are
    deliberately not loadable by coordinate: their values are only reachable
    through :class:`CursorValue`, which is what keeps coordinate search and
    format-specific probing out of the schema.
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
class SparseCursorDecl(LoopIRNode):
    """One sparse cursor over a COMPRESSED level of a declared input tensor.

    ``parent`` is the explicit dominating physical position that selects the
    stored segment the cursor walks: the root position for level 0, or a
    position of level ``level - 1`` of the same tensor (a
    :class:`DensePosition` for a dense parent, a :class:`PositionValue`
    bound by an enclosing :class:`SparseFor` for a compressed parent).
    Coordinate search, rendered names, callbacks, and implicit state are not
    admissible parents.
    """

    node_id: LoopIRNodeId
    cursor: CursorId
    tensor: SymbolId
    level: int
    parent: Expr


@dataclass(frozen=True)
class SparseFor(Stmt):
    """Position iteration over one sparse cursor's selected segment.

    Each iteration binds ``position`` to the cursor's current physical
    storage position and ``coord_index`` to the stored coordinate (whose
    domain is the dimension of the cursor level's logical mode), then runs
    ``body`` once per stored entry, in storage order.  The bound position is
    what a compressed child level references as its dominating parent.
    There is exactly one cursor, so this node never exercises sparse-sparse
    merge semantics.
    """

    cursor: SparseCursorDecl
    position: PositionId
    coord_index: IndexId
    body: Block


@dataclass(frozen=True)
class MergedSparseFor(Stmt):
    """Coordinate-synchronized iteration over two or more sparse cursors.

    Every cursor must target the value-bearing leaf level of its tensor, and
    all cursor levels must store the same logical dimension — the merged
    coordinate domain.  Hierarchical merge descent (merging non-leaf levels
    and descending into children) is deliberately not represented by this
    subset.

    Semantics (intrinsic to the node; the oracle and any target lowering
    must implement exactly this):

    - coordinate selection: while the mode-specific termination condition
      has not fired, the loop's candidate coordinate is the minimum of the
      non-exhausted cursors' current coordinates; a cursor is *aligned* when
      its current coordinate equals that candidate;
    - body emission: UNION runs ``body`` at every candidate coordinate;
      INTERSECTION runs ``body`` only when every cursor is aligned;
    - cursor advancement: after each candidate coordinate is considered,
      every aligned cursor advances by exactly one position — bodies never
      advance, rewind, or rebind cursors;
    - exhaustion/termination: UNION terminates when every cursor is
      exhausted; INTERSECTION terminates as soon as any cursor is exhausted;
    - progress: every step advances at least one cursor by one position, so
      the loop terminates after at most the sum of the segment lengths.

    ``coord_index`` binds the candidate coordinate for the body.
    """

    mode: MergeMode
    cursors: Tuple[SparseCursorDecl, ...]
    coord_index: IndexId
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
class AppendEntry(Stmt):
    """Target-neutral ordered sparse-output assembly: append one entry.

    Appends carry the full coordinate tuple (logical mode order,
    domain-checked) plus the value.  Successive appends to one output must
    be lexicographically strictly increasing in their coordinates; the
    canonical container is assembled from that stream alone, so no node
    encodes positions arrays, offsets, or target storage layout.  The
    oracle fails closed on out-of-order appends at runtime.
    """

    tensor: SymbolId
    coords: Tuple[Expr, ...]
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
