"""Frozen, tuple-owned generic IR nodes for the Phase-3.5 LoopIR spike.

Every node is a plain frozen dataclass whose children are owned tuples.  The
constructors deliberately perform no validation: the fail-closed authority for
structure, scope, typing, layout, and domains is ``verifier.verify_program``,
and the adversarial verifier tests need to be able to build malformed nodes
directly.  Semantics that a target backend or the spike interpreter must honour
are documented on the node that owns them; nothing here names a target
language, a rendered symbol, or an operation.

The schema separates three spaces that the superseded candidate conflated:

- **Logical dimensions.**  A program declares :class:`DimensionDecl` entries;
  every tensor maps each of its logical modes to one declared
  :class:`DimensionId`.  Shared dimension identity is the shape contract (two
  extents are equal because they are the same dimension) and the coordinate
  domain contract (two coordinates are comparable because they index the same
  dimension).
- **Physical levels.**  :class:`TensorDecl.levels` lists physical storage
  levels in storage order; each :class:`LevelDecl` records which logical mode
  the level stores, so CSR ``(dense@0, compressed@1)`` and CSC
  ``(dense@1, compressed@0)`` are structurally distinct.
- **Physical positions.**  Sparse iteration binds a :class:`PositionId` for
  the storage position it walks, separately from the coordinate it resolves.
  A compressed child level must name its dominating parent position
  explicitly (:class:`RootPosition`, :class:`DensePosition`, or a bound
  :class:`PositionValue`); positions are never recovered from coordinates,
  names, or interpreter state.

Identity reuses the stable production identity types where they fit:
``SymbolId`` identifies tensors and reduction accumulators, ``IndexId``
identifies bound iteration coordinates.  ``LoopNodeId``, ``CursorId``,
``DimensionId``, and ``PositionId`` are spike-local because production
``NodeId``/``AccessId`` are documented as CIN identities and this schema is
explicitly a revisable candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from itertools import count
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from ..identity import IndexId, SymbolId


@dataclass(frozen=True, order=True)
class LoopNodeId:
    """Identity of one LoopIR node within a spike program."""

    value: int


@dataclass(frozen=True, order=True)
class CursorId:
    """Identity of one sparse cursor binding within a spike program."""

    value: int


@dataclass(frozen=True, order=True)
class DimensionId:
    """Identity of one logical dimension (coordinate domain) of a program."""

    value: int


@dataclass(frozen=True, order=True)
class PositionId:
    """Identity of one bound physical storage position within a program."""

    value: int


_loop_node_ids = count()
_cursor_ids = count()
_dimension_ids = count()
_position_ids = count()
_id_lock = Lock()


def new_loop_node_id() -> LoopNodeId:
    """Allocate a program-unique identity for one LoopIR node."""

    with _id_lock:
        return LoopNodeId(next(_loop_node_ids))


def new_cursor_id() -> CursorId:
    """Allocate a program-unique identity for one sparse cursor."""

    with _id_lock:
        return CursorId(next(_cursor_ids))


def new_dimension_id() -> DimensionId:
    """Allocate a program-unique identity for one logical dimension."""

    with _id_lock:
        return DimensionId(next(_dimension_ids))


def new_position_id() -> PositionId:
    """Allocate a program-unique identity for one bound physical position."""

    with _id_lock:
        return PositionId(next(_position_ids))


@unique
class LevelKind(Enum):
    """Per-level storage layout of a declared tensor, target-neutrally.

    ``DENSE`` and ``COMPRESSED`` are the kinds this spike represents and
    executes.  ``COORDINATE`` and ``SINGLETON`` mirror the production level
    types of the same names; they are declared here so the disposition is
    explicit, and the verifier fails closed on them with the stable
    ``unsupported_level_kind`` defect until a later milestone represents
    their iteration.
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
    """Reduction combiners with a declared algebraic identity."""

    ADD = "add"
    MUL = "mul"


REDUCE_IDENTITIES: Mapping[ReduceOp, float] = MappingProxyType(
    {ReduceOp.ADD: 0.0, ReduceOp.MUL: 1.0}
)


@unique
class MergeMode(Enum):
    """Body-emission policy of a :class:`MergedSparseFor`."""

    UNION = "union"
    INTERSECTION = "intersection"


class LoopIRNode:
    """Marker base for every spike IR node; carries no behaviour."""


@dataclass(frozen=True)
class Expr(LoopIRNode):
    """Base of all value-, coordinate-, or position-producing nodes."""

    node_id: LoopNodeId


@dataclass(frozen=True)
class Stmt(LoopIRNode):
    """Base of all statement nodes."""

    node_id: LoopNodeId


@dataclass(frozen=True)
class IntConst(Expr):
    """A coordinate-typed integer constant, valid in any coordinate domain."""

    value: int


@dataclass(frozen=True)
class FloatConst(Expr):
    """A value-typed floating-point constant."""

    value: float


@dataclass(frozen=True)
class IndexValue(Expr):
    """The current coordinate bound by an enclosing loop.

    Its coordinate domain is the logical dimension the binding loop iterates,
    so the verifier can reject a coordinate used outside its domain.
    """

    index: IndexId


@dataclass(frozen=True)
class RootPosition(Expr):
    """The unique physical root position that dominates every level-0 level.

    It is position-typed and is the only admissible parent of a level-0
    storage level of any tensor.
    """


@dataclass(frozen=True)
class DensePosition(Expr):
    """The physical position within one DENSE level of a declared tensor.

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
    cursor; this is how a compressed child names its dominating parent
    position.
    """

    position: PositionId


@dataclass(frozen=True)
class PositionLoad(Expr):
    """Read the scalar owned by one tensor's value-bearing leaf position.

    ``position`` must be position-typed for the final physical level of
    ``tensor``.  Unlike :class:`Load`, this is a physical level access and is
    therefore valid for a DENSE leaf below sparse structural levels.  Unlike
    :class:`CursorValue`, it carries no merge-alignment/default semantics; a
    merged compressed leaf continues to use the cursor-owned form.
    """

    tensor: SymbolId
    position: Expr


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
class AccumValue(Expr):
    """The current value of a live reduction accumulator (value-typed)."""

    accumulator: SymbolId


@dataclass(frozen=True)
class Load(Expr):
    """A coordinate-addressed read of an all-dense input tensor.

    ``indices`` are given in logical mode order, and each must lie in the
    domain of the corresponding declared dimension.  Compressed tensors are
    deliberately not loadable by coordinate: their values are only reachable
    through :class:`CursorValue`, which is what keeps coordinate search,
    name rendering, and format-specific probing out of the schema.
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
    """An ordered, tuple-owned statement sequence forming one lexical scope.

    Reduction accumulators declared by a block die when the block exits and
    are re-initialized on re-entry.
    """

    statements: Tuple[Stmt, ...]


@dataclass(frozen=True)
class DenseFor(Stmt):
    """Dense iteration over one declared logical dimension.

    Binds ``index`` to ``0 .. extent - 1`` in order, where ``extent`` is the
    dimension's resolved extent; the bound coordinate's domain is
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
    Coordinate search, rendered names, callbacks, and implicit interpreter
    state are not admissible parents.
    """

    node_id: LoopNodeId
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
    spike.

    Semantics (intrinsic to the node; the interpreter and any later backend
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
class DeclAccum(Stmt):
    """Declare a reduction accumulator initialized to its operator identity.

    ``init`` must be the literal identity of ``op`` — the verifier enforces
    it — so re-entry semantics never depend on stale state.
    """

    accumulator: SymbolId
    op: ReduceOp
    init: Expr


@dataclass(frozen=True)
class Accumulate(Stmt):
    """Combine ``value`` into a live accumulator with its declared operator."""

    accumulator: SymbolId
    value: Expr


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
    (``target = target op value``).  The spike currently admits only ADD,
    whose identity matches dense output zero-initialization.  Other operators
    need an explicit output-initialization contract before they are executable.
    This is the target-neutral scatter accumulation a permuted traversal needs
    (for example CSC SpMV, where one output element receives contributions
    from several outer iterations); it deliberately does not make outputs
    generally readable.
    """

    tensor: SymbolId
    indices: Tuple[Expr, ...]
    op: ReduceOp
    value: Expr


@dataclass(frozen=True)
class AppendEntry(Stmt):
    """Target-neutral sparse-output assembly: append one stored entry.

    Appends carry the full coordinate tuple (logical mode order,
    domain-checked) plus the value.  Successive appends to one output must
    be lexicographically strictly increasing in their coordinates; the
    canonical container is assembled from that stream alone, so no node
    encodes positions arrays, offsets, or target storage layout.  The
    interpreter fails closed on out-of-order appends.
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

    node_id: LoopNodeId
    dimension: DimensionId
    name: str


@dataclass(frozen=True)
class LevelDecl(LoopIRNode):
    """One physical storage level of a tensor and the logical mode it stores.

    ``mode`` indexes the owning tensor's ``dimensions`` tuple; across one
    tensor the modes must form a permutation of the logical modes, which is
    what makes CSR and CSC (and any other mode permutation) structurally
    distinct.
    """

    node_id: LoopNodeId
    kind: LevelKind
    mode: int


@dataclass(frozen=True)
class TensorDecl(LoopIRNode):
    """A declared tensor: stable symbol, display name, dimensions, levels.

    ``dimensions`` maps each logical mode to its declared dimension
    identity; ``levels`` lists physical storage levels in storage order.
    """

    node_id: LoopNodeId
    symbol: SymbolId
    name: str
    dimensions: Tuple[DimensionId, ...]
    levels: Tuple[LevelDecl, ...]


@dataclass(frozen=True)
class LoopProgram(LoopIRNode):
    """One executable spike program: declarations plus a top-level block."""

    node_id: LoopNodeId
    dimensions: Tuple[DimensionDecl, ...]
    tensors: Tuple[TensorDecl, ...]
    inputs: Tuple[SymbolId, ...]
    outputs: Tuple[SymbolId, ...]
    body: Block
