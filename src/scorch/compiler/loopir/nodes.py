"""Frozen production LoopIR nodes for the dense and sparse vertical slices.

The dense subset was frozen by Phase 4 (see
``COMPILER_IR_REFACTOR_PHASE4_REVIEW.md``).  Phase 5 extends it deliberately
with the sparse concepts the migrated level-based families exercise —
compressed position iteration, parent-position-linked cursors, coordinate
and leaf-value access, structured UNION/INTERSECTION merging, and ordered
CSR output assembly — revised from the reviewed Phase-3.5 spike, not
copied wholesale.  Phase 6 adds the affine schedule subset: one structured
:class:`TileOuterFor`/:class:`TileInnerFor` pair per strip-mined dense
loop, linked by an artifact-local :class:`TileId`, with ragged-tail
coverage intrinsic to the node semantics, and — for the stack-accumulation
schedule family — the structured workspace region: one
:class:`WorkspaceRegion` owning a :class:`WorkspaceDecl` whose extent is
the point domain of one affine split, with allocation and zero-reset
intrinsic to region entry, producer-only :class:`WorkspaceReduce` writes,
and consumer-only :class:`WorkspaceRead` reads.  The sparse-panel slice
adds the coordinate-window pair: one :class:`PanelOuterFor` iterating the
clamped window origins of a compressed coordinate's dimension, paired by
the same artifact-local :class:`TileId` with one :class:`SparseWindowFor`
that visits exactly the stored entries whose coordinate falls inside the
current window.  The operand-relayout slice adds the staging region: one
:class:`RelayoutStage` owning a :class:`RelayoutDecl` that stages a dense
operand's current pack strip (panel-window rows or the whole panel axis)
at region entry, with :class:`StagedRead` — the staged twin of
:class:`Load` — reading the operand through the region.  The abstract
parallel-selection slice adds one optional program-level fact naming a dense
logical or affine-origin loop, its canonical structural work estimate, and
its verifier-proved race discipline; it deliberately carries no OpenMP or
target-policy spelling.  Concepts the migrated families do not exercise are
still not declared: there are no accumulators, integer constants, or
dimension-extent expressions.  The serial sparse-result slice adds a
target-neutral :class:`SparseWorkspaceRegion`: coordinate/value insertions
merge under ADD and one ordered drain exposes the merged entries, while
capacity, hashing, sorting, and the backing container remain target concerns.

Discipline carried over from the spike and the binding design decisions:

- every node is a plain frozen dataclass whose children are owned tuples;
- constructors perform no validation — ``verifier.verify_program`` is the
  single fail-closed authority, and adversarial tests must be able to build
  malformed nodes directly;
- identity is never spelling: tensors are identified by production
  ``SymbolId`` values and bound loop coordinates by production ``IndexId``
  values, so CIN provenance survives lowering; ``LoopIRNodeId``,
  ``DimensionId``, ``CursorId``, ``PositionId``, ``TileId``, and
  ``WorkspaceId``/``RelayoutId``/``ResultTileId`` are LoopIR-artifact-local
  identities allocated by
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

Scalar typing and reduction-without-accumulators are unchanged from Phase 4;
``COMPRESSED`` becomes executable in this subset.  ``COORDINATE`` levels are
executable on inputs whose levels are all ``COORDINATE`` (a COO tensor, see
:class:`LevelKind`); ``SINGLETON`` stays declared and fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, Optional, Tuple

from ...format import LevelType
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


@dataclass(frozen=True, order=True)
class TileId:
    """Identity of one scheduled loop transformation within a LoopIR program.

    An affine split is realized as a :class:`TileOuterFor` /
    :class:`TileInnerFor` pair, while a sparse coordinate panel is realized
    as a :class:`PanelOuterFor` / :class:`SparseWindowFor` pair.  Each pair
    shares one ``TileId`` so its ownership remains unambiguous when other
    loops sit between the origin and point/window loops.
    """

    value: int


@dataclass(frozen=True, order=True)
class WorkspaceId:
    """Identity of one scoped accumulation workspace within a program.

    A workspace is one schedule fact — "this region accumulates into a
    scratch buffer instead of final storage" — realized structurally as a
    :class:`WorkspaceRegion` whose :class:`WorkspaceDecl` owns the identity.
    Reads and reductions name the workspace by this identity, never by a
    rendered C++ name.
    """

    value: int


@dataclass(frozen=True, order=True)
class RelayoutId:
    """Identity of one staged-operand relayout region within a program.

    A relayout is one schedule fact — "this dense operand's current pack
    strip is read through contiguous staged storage instead of its
    declared layout" — realized structurally as a :class:`RelayoutStage`
    whose :class:`RelayoutDecl` owns the identity.  Staged reads name the
    relayout by this identity, never by a rendered C++ name; after the
    typed staging pass redirects the operand's unique read occurrence,
    this identity is the stable anchor and nothing downstream re-identifies
    the original access.
    """

    value: int


@dataclass(frozen=True, order=True)
class ResultTileId:
    """Identity of one heap-backed compact result-tile region within a program.

    A result tile is one schedule fact — "this dense result's current
    trailing-axis strip accumulates into reusable compact storage and is
    copied out exactly once at strip exit" — realized structurally as a
    :class:`ResultTileRegion` whose :class:`ResultTileDecl` owns the
    identity.  Tiled reductions name the result tile by this identity,
    never by a rendered C++ name; after the typed accumulation pass
    redirects the result's unique write occurrence, this identity is the
    stable anchor and nothing downstream re-identifies the original write.
    """

    value: int


@unique
class ScalarType(Enum):
    """Scalar value type of a declared tensor, target-neutrally.

    The element types the shipped compiler builds kernels for.  A program's
    tensors may hold different types: every value expression has the type
    :func:`promoted_scalar_type` gives its operands, and a stored value must
    have the type of the tensor or workspace it is stored in (the verifier's
    ``mixed_dtype``).  Integer arithmetic is exact and wraps to the declared
    width wherever a value is stored, as the C++ conversion to the element
    type does.
    """

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    UINT8 = "uint8"

    @property
    def is_floating(self) -> bool:
        return self in (ScalarType.FLOAT32, ScalarType.FLOAT64)


# Bits of each integer type, for promotion.
_INTEGER_BITS = {
    ScalarType.INT64: 64,
    ScalarType.INT32: 32,
    ScalarType.INT8: 8,
    ScalarType.UINT8: 8,
}


def promoted_scalar_type(lhs: ScalarType, rhs: ScalarType) -> Optional[ScalarType]:
    """The type of a binary operation on values of ``lhs`` and ``rhs``.

    PyTorch's promotion rule (``torch.promote_types``) over the types Scorch
    stores: a floating-point type wins over an integer one, and the wider of
    two types of one kind wins.  ``None`` for int8 with uint8, which promotes
    to int16, a type Scorch does not store.  C++ computes each admitted pair
    in this type too, except that int8 and uint8 operands are computed in
    ``int``; for ADD, SUB and MUL that value wraps to the same result where
    it is stored.  So the emitted arithmetic needs no casts.
    """

    if lhs is rhs:
        return lhs
    if lhs.is_floating or rhs.is_floating:
        if lhs.is_floating and rhs.is_floating:
            return ScalarType.FLOAT64
        return lhs if lhs.is_floating else rhs
    if {lhs, rhs} == {ScalarType.INT8, ScalarType.UINT8}:
        return None
    return lhs if _INTEGER_BITS[lhs] > _INTEGER_BITS[rhs] else rhs


@unique
class LevelKind(Enum):
    """Per-level storage layout of a declared tensor, target-neutrally.

    ``DENSE`` and ``COMPRESSED`` are executed on inputs and outputs.

    ``COORDINATE`` is executed on an input whose levels are all
    ``COORDINATE``: a COO tensor.  Such a tensor stores ``n`` entries; every
    level holds one coordinate per entry and all levels share the position
    space ``0 .. n - 1``; the entries are in lexicographic order of their
    coordinates, read in level order.  Level 0 is a compressed level whose
    coordinates may repeat and whose one segment is every entry; each later
    level is a singleton level: its segment is the run of entries that share
    the coordinates of the levels above it (TACO's COO level formats).  A
    coordinate level beside a level of another kind has no such segment and
    fails closed, as does a coordinate level of an output.

    ``SINGLETON`` is declared so its disposition is explicit and the schema
    does not change shape when a later phase represents it; the verifier
    fails closed on it with the stable ``unsupported_level_kind`` defect.
    """

    DENSE = "dense"
    COMPRESSED = "compressed"
    COORDINATE = "coordinate"
    SINGLETON = "singleton"


#: The vocabulary correspondence between a LoopIR level kind and the core tensor
#: model's level type, in one place.  Two independent copies of it already
#: existed (one per direction) and a third was about to be written; a mapping
#: between two enums is vocabulary, so it belongs beside the enum that needs
#: translating.  Only the executable kinds appear: ``SINGLETON`` is declared
#: so its disposition is explicit, and every consumer must fail closed on its
#: absence rather than defaulting.
LEVEL_KIND_TO_LEVEL_TYPE: Dict[LevelKind, LevelType] = {
    LevelKind.DENSE: LevelType.DENSE,
    LevelKind.COMPRESSED: LevelType.COMPRESSED,
    LevelKind.COORDINATE: LevelType.COORDINATE,
}


@unique
class MergeMode(Enum):
    """Body-emission policy of a :class:`MergedSparseFor`.

    ``DENSE_UNION`` is the union of the cursors' stored coordinates with
    every coordinate of the merged dimension: the iteration of a sum one of
    whose terms is present at every coordinate (a dense operand, or an
    operand that does not depend on the loop variable and stores nothing
    sparse).
    """

    UNION = "union"
    INTERSECTION = "intersection"
    DENSE_UNION = "dense_union"


@unique
class RelayoutScope(Enum):
    """Row coverage of one staged-operand relayout region.

    ``PANEL`` stages exactly the owning panel's current clamped window rows
    (the region must execute inside the panel's origin loop); ``PACK_AXIS``
    stages every row of the operand's panel axis whenever the region is
    entered.  The typed schedule pass places that region once per pack
    origin.  In both scopes the staged columns are the pack split's current
    clamped point window.
    """

    PANEL = "panel"
    PACK_AXIS = "pack_axis"


@unique
class ParallelPart(Enum):
    """Which derived part of a logical loop a parallel selection names.

    ``LOGICAL`` selects the unsplit binding loop of the index; ``OUTER``
    selects the origin loop of the index's affine split.  A split's point
    loop is deliberately unrepresentable: its ragged-tail clamp makes it a
    non-canonical work partition, matching the legacy scheduler's
    rejection of ``*_in`` parallel anchors.
    """

    LOGICAL = "logical"
    OUTER = "outer"


@unique
class ParallelDiscipline(Enum):
    """The race-freedom argument class of one parallel selection.

    ``RESULT_PARTITION``: distinct iterations of the selected loop write
    disjoint cells of the declared dense result (the selected coordinate
    participates in every enclosed result write), and any accumulation
    state lives strictly inside one iteration.  ``COMPACT_PARTITION``: the
    program accumulates through a heap result-tile region and distinct
    iterations of the selected dense prefix loop address disjoint compact
    cells (the linearized prefix position partitions the tile).
    """

    RESULT_PARTITION = "result_partition"
    COMPACT_PARTITION = "compact_partition"


@unique
class ParallelIntent(Enum):
    """Why the selection exists — schedule provenance, never target policy.

    ``EXPLICIT`` records a plan-carried explicit ``parallel_loop`` fact.
    Automatic selection intents arrive only with the automatic-plan
    migration; the member set is deliberately closed until then.
    """

    EXPLICIT = "explicit"


@unique
class VectorLoopPart(Enum):
    """Which loop of an index a recorded vector mark names.

    ``LOGICAL`` is the unsplit loop that binds the index; ``OUTER`` and
    ``INNER`` are the origin loop and the point loop of the index's split.
    Unlike :class:`ParallelPart`, the point loop is representable: it is the
    innermost loop of a strip-mined sum, the loop vector instructions run.
    """

    LOGICAL = "logical"
    OUTER = "outer"
    INNER = "inner"


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
class IntegerConst(Expr):
    """A value-typed integer constant, the counterpart of :class:`FloatConst`
    in a program whose scalar type is an integer type.

    A program holds value constants of its own kind only: ``FloatConst`` in a
    floating-point program, ``IntegerConst`` in an integer one.
    """

    value: int


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
    Inside a UNION or DENSE_UNION merge the cursor may be unaligned at the
    emitted coordinate, so a value-typed ``default`` is required and is
    evaluated exactly when the cursor is not aligned.
    """

    cursor: CursorId
    default: Optional[Expr]


@dataclass(frozen=True)
class PositionLoad(Expr):
    """Read the scalar owned by one tensor's value-bearing leaf position.

    ``position`` must be position-typed for the final physical level of
    ``tensor``.  Unlike :class:`Load`, this is a physical level access, so it
    is valid for a DENSE leaf below sparse structural levels and for an
    independently bound compressed-leaf position.

    Without ``guard`` the load is unconditional: its address must be bound
    wherever the load runs.  A DENSE leaf below a level that a UNION or
    DENSE_UNION merge iterates has an address only where that merge's
    cursor aligned (the operand stores nothing at a one-sided coordinate).
    Such a load names that cursor as ``guard``: the cursor of the deepest
    compressed level of ``tensor``, every level below it DENSE.  The load is
    evaluated exactly when the guard is aligned at its merge's current
    coordinate, and the value-typed ``default`` is evaluated otherwise, as
    for a UNION :class:`CursorValue`.  ``guard`` and ``default`` are both
    set or both ``None``.
    """

    tensor: SymbolId
    position: Expr
    guard: Optional[CursorId] = None
    default: Optional[Expr] = None


@dataclass(frozen=True)
class Load(Expr):
    """A coordinate-addressed read of an all-dense input tensor.

    ``indices`` are given in logical mode order; each must be a coordinate
    in the domain of the corresponding declared dimension.  The result is
    value-typed with the tensor's scalar type.  Compressed tensors are
    deliberately not loadable by coordinate: their values are only reachable
    through :class:`CursorValue` at a compressed leaf, or through
    :class:`PositionLoad` when the value-bearing leaf is a DENSE level below
    compressed structure, which is what keeps coordinate search and
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
    """One sparse cursor over a COMPRESSED or COORDINATE level of an input.

    ``parent`` is the explicit dominating physical position that selects the
    stored segment the cursor walks: the root position for level 0, or a
    position of level ``level - 1`` of the same tensor (a
    :class:`DensePosition` for a dense parent, a :class:`PositionValue`
    bound by an enclosing :class:`SparseFor` for a compressed or coordinate
    parent).  Coordinate search, rendered names, callbacks, and implicit
    state are not admissible parents.

    On a COMPRESSED level the segment is the position range the parent
    selects, and the cursor visits every position in it.  A parent that
    hangs from a UNION-bound position whose cursor did not align, directly
    or through dense levels, selects the empty segment.  On a COORDINATE
    level the segment is every entry (level 0) or the parent's run of
    entries (a later level); the cursor visits each distinct coordinate once,
    at the first entry that stores it, and that entry's position is what a
    child level names as its parent.  On the last level it visits every
    entry.
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
    ``body`` once per position the cursor visits, in storage order.  The
    bound position is what a compressed or coordinate child level references
    as its dominating parent.
    There is exactly one cursor, so this node never exercises sparse-sparse
    merge semantics.
    """

    cursor: SparseCursorDecl
    position: PositionId
    coord_index: IndexId
    body: Block


@dataclass(frozen=True)
class TileOuterFor(Stmt):
    """The origin loop of one affine strip-mine of a dense logical loop.

    Iterates the tile origins ``0, width, 2*width, ...`` of ``dimension``,
    strictly below the dimension's runtime extent, in order.  The origin is
    deliberately not a readable coordinate: the split's only observable
    binding is the logical coordinate the paired :class:`TileInnerFor`
    reconstructs, so no body expression can depend on target spelling of the
    origin.  ``index`` records which logical loop was split (schedule
    provenance — the same production ``IndexId`` the unscheduled loop
    bound); the paired inner loop must agree on ``index``, ``dimension``,
    and ``width`` and is matched by ``tile``.
    """

    tile: TileId
    index: IndexId
    dimension: DimensionId
    width: int
    body: Block


@dataclass(frozen=True)
class TileInnerFor(Stmt):
    """The point loop of one affine strip-mine of a dense logical loop.

    Must execute within the :class:`TileOuterFor` binding the same
    ``tile``.  For the enclosing origin ``o`` it binds ``index`` to
    ``o, o + 1, ..., min(o + width, extent) - 1`` in order, where ``extent``
    is the runtime extent of ``dimension`` — the ragged tail is intrinsic
    node semantics, not an emitted guard detail, so every coordinate of the
    dimension is visited exactly once across the origin loop's iterations.
    ``unroll`` is a target-independent unrolling preference carried from the
    schedule; it never changes iteration semantics.
    """

    tile: TileId
    index: IndexId
    dimension: DimensionId
    width: int
    unroll: bool
    body: Block


@dataclass(frozen=True)
class PanelOuterFor(Stmt):
    """The origin loop of one sparse coordinate panel (window).

    Iterates the window origins ``0, width, 2*width, ...`` of ``dimension``,
    strictly below the dimension's runtime extent, in order.  Like an
    affine origin loop, the origin is deliberately not a readable
    coordinate: the panel's only observable binding is the stored
    coordinate the paired :class:`SparseWindowFor` binds.  ``index``
    records which logical compressed-coordinate loop was panel-tiled
    (schedule provenance); the paired window loop is matched by ``tile``
    and must bind exactly that index.

    ``bound_tensor``/``bound_level`` name one declared DENSE storage level
    whose logical mode stores ``dimension`` — the plan's ``PanelBound``
    fact, materialized structurally.  It is semantically redundant with
    the dimension identity (extent equality is the dimension contract) and
    exists so a target lowering can reproduce the exact bound the legacy
    panel loop reads; the verifier enforces its consistency.
    """

    tile: TileId
    index: IndexId
    dimension: DimensionId
    width: int
    bound_tensor: SymbolId
    bound_level: int
    body: Block


@dataclass(frozen=True)
class SparseWindowFor(Stmt):
    """Windowed position iteration over one sparse cursor's segment.

    Must execute within the :class:`PanelOuterFor` binding the same
    ``tile``.  For the enclosing window origin ``o`` it visits, in storage
    order, exactly the stored entries of the cursor's selected segment
    whose coordinate ``c`` satisfies ``o <= c < min(o + width, extent)``
    (``extent`` is the runtime extent of the panel's dimension) — the
    clamped coordinate window is intrinsic node semantics, so every stored
    entry of the segment is visited exactly once across the origin loop's
    iterations.  Each visit binds ``position`` to the entry's physical
    storage position and ``coord_index`` to its stored coordinate, exactly
    like :class:`SparseFor`.  How the position sub-range is found (for a
    canonical sorted segment it is derivable by coordinate search) is a
    target concern; no search spelling appears in the node.
    """

    tile: TileId
    cursor: SparseCursorDecl
    position: PositionId
    coord_index: IndexId
    body: Block


@dataclass(frozen=True)
class WorkspaceDecl(LoopIRNode):
    """One scoped stack workspace spanning the point domain of one split.

    ``tile`` names the affine split whose point domain this workspace
    buffers: the workspace has exactly ``width`` cells (the split's width),
    of scalar type ``dtype``, and the cell addressed by a point coordinate
    ``c`` is ``c - origin`` for the current origin of that split — always in
    ``[0, width)`` because point coordinates are clamped to the tile.
    ``name`` is presentation only, like every other display name.  There is
    deliberately no dimension-extent, multi-dimensional, or sparse (hashed)
    workspace form in this subset; those remain legacy-only families.
    """

    node_id: LoopIRNodeId
    workspace: WorkspaceId
    name: str
    dtype: ScalarType
    tile: TileId


@dataclass(frozen=True)
class WorkspaceRegion(Stmt):
    """Structured allocation/reset, producer, and consumer of one workspace.

    Region semantics are intrinsic to the node (the oracle and any target
    lowering must implement exactly this):

    - on entry a fresh workspace of ``width`` cells is allocated with every
      cell zero — the explicit reset whose value is ADD's identity, which is
      what makes :class:`WorkspaceReduce` well-defined without a separate
      initialization statement;
    - ``producer`` runs first and owns all writes: :class:`WorkspaceReduce`
      into this workspace is legal only inside it, and it must not write
      declared outputs;
    - ``consumer`` runs second and owns all reads: :class:`WorkspaceRead`
      of this workspace is legal only inside it (copy-out to the output is
      an ordinary store of a read value);
    - the workspace ceases to exist at region exit — its lifetime is exactly
      the region, so a fresh zeroed buffer is observed on every execution of
      the region (once per iteration of the enclosing scope).

    The region must execute between its tile's origin loop and point loops:
    it needs a current tile origin (so cells are addressable) and must not
    sit inside a point loop of its own tile (a per-point workspace would
    never accumulate).
    """

    workspace: WorkspaceDecl
    producer: Block
    consumer: Block


@dataclass(frozen=True)
class WorkspaceRead(Expr):
    """The value stored at one cell of an in-scope workspace (consumer side).

    ``coord`` must be the owning tile's point coordinate bound by a
    :class:`TileInnerFor` inside the owning region's consumer; the cell read
    is ``coord - origin``.  The result is value-typed with the workspace's
    scalar type.
    """

    workspace: WorkspaceId
    coord: Expr


@dataclass(frozen=True)
class WorkspaceReduce(Stmt):
    """A read-modify-write into one cell of an in-scope workspace.

    Combines the cell with ``value`` using ``op`` (``cell = cell op value``).
    Only ADD is admitted, and its identity is exactly the zero the owning
    region's entry reset established — this is the reduction-legality
    contract, stated structurally.  ``coord`` must be the owning tile's
    point coordinate bound by a :class:`TileInnerFor` inside the owning
    region's producer; the cell written is ``coord - origin``.
    """

    workspace: WorkspaceId
    coord: Expr
    op: ReduceOp
    value: Expr


@dataclass(frozen=True)
class SparseWorkspaceDecl(LoopIRNode):
    """One scoped serial sparse workspace over an ordered key domain.

    The workspace buffers key/value pairs over ``key_dimensions``, an ordered
    tuple of one or more declared dimensions: insertions merge by the whole
    key tuple under ADD, and the consumer observes the merged entries in
    strictly increasing **lexicographic** key order.  ``name`` is presentation
    only.  Capacity, hashing, and the backing container are target concerns —
    no size or container spelling appears in the node.

    The key domain is the workspace's own, declared independently of any
    result layout: no result level structure, dense-prefix extent, or result
    rank appears here.  That separation is the point.  A workspace whose key
    rank differs from the result rank is ordinary — a rank-1 key draining
    into the trailing level of a rank-3 result is the TTM shape — and
    conflating the two is precisely the defect that made the legacy
    multi-compressed reduction path reject every insertion.

    ``len(key_dimensions) == 1`` is the serial single-drain form the migrated
    B1/row-scope families use; it is the ``K == 1`` instance of this node, not
    a separate kind.
    """

    node_id: LoopIRNodeId
    workspace: WorkspaceId
    name: str
    dtype: ScalarType
    key_dimensions: Tuple[DimensionId, ...]


@dataclass(frozen=True)
class SparseWorkspaceRegion(Stmt):
    """Structured allocation, producer, and ordered consumer of one sparse
    workspace.

    Region semantics are intrinsic to the node (the oracle and any target
    lowering must implement exactly this):

    - on entry the workspace is empty — ADD's identity is the absent entry,
      which is what makes :class:`SparseWorkspaceInsert` well-defined
      without a separate initialization statement;
    - ``producer`` runs first and owns all insertions:
      :class:`SparseWorkspaceInsert` into this workspace is legal only
      inside it, and it must not write declared outputs;
    - ``consumer`` runs second and owns the one ordered drain:
      :class:`SparseWorkspaceDrainFor` of this workspace is legal only
      inside it and observes every merged entry exactly once in strictly
      increasing lexicographic key order;
    - the workspace ceases to exist at region exit — its lifetime is
      exactly the region, so an empty workspace is observed on every
      execution of the region (once per iteration of the enclosing scope).
    """

    workspace: SparseWorkspaceDecl
    producer: Block
    consumer: Block


@dataclass(frozen=True)
class SparseWorkspaceInsert(Stmt):
    """A merging insertion into one in-scope sparse workspace.

    Combines the entry at ``coords`` with ``value`` using ``op``
    (``entry = entry op value``; an absent entry is created with the
    value).  Only ADD is admitted, and its identity is exactly the absent
    entry the owning region's empty-entry contract established.  ``coords``
    must have one coordinate-typed expression per declared key dimension, in
    key order, and is legal only inside the owning region's producer.
    """

    workspace: WorkspaceId
    coords: Tuple[Expr, ...]
    op: ReduceOp
    value: Expr


@dataclass(frozen=True)
class SparseWorkspaceDrainFor(Stmt):
    """The ordered drain of one in-scope sparse workspace.

    Visits every merged entry of the workspace exactly once in strictly
    increasing **lexicographic** key order, binding ``indices`` — one index
    per declared key dimension, in key order — to the entry's key components
    for the body; the entry's merged value is read through
    :class:`SparseWorkspaceValue`.  Legal only inside the owning region's
    consumer, at most once per region.  How the ordering is realized (the
    serial container sorts before iteration) is a target concern — no sort
    spelling appears in the node.

    Lexicographic key order is what makes multi-level ordered result
    assembly correct without any sort in the IR: when the key dimensions are
    listed in result level order, the drain visits entries in exactly the
    order the result's levels must be appended.
    """

    workspace: WorkspaceId
    indices: Tuple[IndexId, ...]
    body: Block


@dataclass(frozen=True)
class SparseWorkspaceValue(Expr):
    """The current drained entry's merged value (drain-loop scope only).

    Value-typed with the owning workspace's scalar type.  Legal only inside
    the body of the :class:`SparseWorkspaceDrainFor` naming the same
    workspace.
    """

    workspace: WorkspaceId


@dataclass(frozen=True)
class RelayoutDecl(LoopIRNode):
    """One staged copy of a dense operand's current pack strip.

    ``operand`` names a rank-2 all-dense declared input whose *last*
    physical storage level stores the pack split's dimension and whose
    first stores the panel's dimension — the audited legacy family's
    contiguous-last-level fact, verified structurally.  ``panel`` names the
    sparse-panel pair whose window rows are staged and ``pack`` names the
    affine split whose clamped point window is the staged column range;
    both are the same artifact-local :class:`TileId` space the pairs
    already own.  ``scope`` selects the staged row coverage
    (:class:`RelayoutScope`).  There is deliberately no display name, C++
    buffer spelling, or level index in the node: the staged storage's
    naming, capacity arithmetic, and pack-loop emission are target
    concerns with no degrees of freedom given these identities.
    """

    node_id: LoopIRNodeId
    relayout: RelayoutId
    operand: SymbolId
    panel: TileId
    pack: TileId
    scope: RelayoutScope


@dataclass(frozen=True)
class RelayoutStage(Stmt):
    """Structured staging region of one relayout declaration.

    Region semantics are intrinsic to the node (the oracle and any target
    lowering must implement exactly this):

    - on entry the operand's current strip is staged: for every row ``r``
      of the scope's row coverage (``PANEL``: the owning panel's current
      clamped window ``[origin, min(origin + width, extent))``;
      ``PACK_AXIS``: the whole panel axis ``[0, extent)``) and every
      column ``c`` of the pack split's current clamped point window, the
      staged cell ``(r, c)`` holds exactly ``operand[r, c]``;
    - ``body`` runs with the staged strip valid throughout; a
      :class:`StagedRead` of this region is legal only inside it and
      observes exactly the staged cells;
    - the staged strip ceases to exist at region exit — its lifetime is
      exactly the region, so re-entering the region (the next scope
      iteration) observes a freshly staged strip.

    The region must execute inside its pack split's origin loop (the
    staged columns need a current pack origin), and a ``PANEL``-scoped
    region additionally inside its panel's origin loop.  How the staging
    copy is realized (buffer reuse across iterations, parallel pack
    loops, capacity arithmetic) is a target concern; only the staged
    contents and lifetime are semantics.
    """

    decl: RelayoutDecl
    body: Block


@dataclass(frozen=True)
class StagedRead(Expr):
    """A staged read of an in-scope relayout region's operand.

    The staged twin of :class:`Load`: ``indices`` are given in the
    operand's logical mode order and domain-checked identically, and the
    value read is exactly ``operand[indices]`` — served through the
    enclosing region's staged strip.  The row index (the operand's
    panel-axis mode) must be the panel's window coordinate bound by the
    owning :class:`SparseWindowFor`, and the column index (the pack-axis
    mode) must be the pack split's point coordinate bound by its
    :class:`TileInnerFor`, which is what keeps every staged read inside
    the staged strip by construction.  The typed staging pass produces
    this node by replacing the operand's verifier-proven unique ``Load``
    occurrence; no occurrence identity is needed because the region
    identity is the anchor from then on.
    """

    relayout: RelayoutId
    indices: Tuple[Expr, ...]


@dataclass(frozen=True)
class ResultTileDecl(LoopIRNode):
    """One heap-backed compact result tile over a pack split's point window.

    ``result`` names a declared all-dense output of rank at least two whose
    *last* physical storage level stores the pack split's dimension — the
    audited legacy trailing-free-axis fact, verified structurally.  ``pack``
    names the affine split whose clamped point window is the compact
    column range; it is the same artifact-local :class:`TileId` space the
    split pair already owns.  The compact tile spans every dense
    result-prefix position and one clamped strip of the trailing axis.
    There is deliberately no display name, C++ buffer spelling, or level
    index in the node: the compact storage's naming, capacity arithmetic,
    and init/copy loop emission are target concerns with no degrees of
    freedom given these identities.
    """

    node_id: LoopIRNodeId
    result_tile: ResultTileId
    result: SymbolId
    pack: TileId


@dataclass(frozen=True)
class ResultTileRegion(Stmt):
    """Structured accumulation region of one heap result-tile declaration.

    Region semantics are intrinsic to the node (the oracle and any target
    lowering must implement exactly this):

    - on entry a fresh compact tile is observed with every cell zero — one
      cell per (dense result-prefix position, clamped pack-window column),
      the explicit reset whose value is ADD's identity, which is what makes
      :class:`TiledReduce` well-defined without a separate initialization
      statement;
    - ``body`` runs with the compact tile live throughout: a
      :class:`TiledReduce` of this region is legal only inside it and
      accumulates into the addressed compact cell; the declared result is
      not written directly inside the region;
    - at region exit every compact cell is copied to the declared result
      exactly once — ``result[prefix, c] = tile[prefix, c - origin]`` for
      every dense prefix position and every clamped-window column ``c`` —
      and the compact tile ceases to exist.  Because the region executes
      once per pack origin and the origin loop covers the whole trailing
      axis in strips, this exactly-once copy-out is what discharges the
      result's whole-tensor zero-initialization contract: cells that
      received no accumulation copy the entry zero.

    The region must execute directly inside its pack split's outermost
    origin loop (the compact columns need a current pack origin, and
    copy-out coverage needs exactly one region execution per origin
    iteration) and never inside a point loop or another repeating scope.
    How the compact copy is realized (buffer reuse across iterations,
    parallel init/copy loops, capacity arithmetic) is a target concern;
    only the accumulated contents, freshness, and copy-out are semantics.
    """

    decl: ResultTileDecl
    body: Block


@dataclass(frozen=True)
class TiledReduce(Stmt):
    """A read-modify-write into one cell of an in-scope compact result tile.

    The staged twin of :class:`StoreReduce`: ``indices`` are the declared
    result's access indices in logical mode order and domain-checked
    identically, and the cell combined is the one addressing
    ``result[indices]`` through the enclosing region's compact tile.
    Combines the cell with ``value`` using ``op`` (``cell = cell op
    value``).  Only ADD is admitted, and its identity is exactly the zero
    the owning region's entry reset established — the reduction-legality
    contract, stated structurally.  The trailing index (the result's
    pack-axis mode) must be the pack split's point coordinate bound by its
    :class:`TileInnerFor`, which is what keeps every reduce inside the
    live compact strip by construction.  The typed accumulation pass
    produces this node by replacing the result's verifier-proven unique
    :class:`StoreReduce` occurrence; no occurrence identity is needed
    because the region identity is the anchor from then on.
    """

    result_tile: ResultTileId
    indices: Tuple[Expr, ...]
    op: ReduceOp
    value: Expr


@dataclass(frozen=True)
class MergedSparseFor(Stmt):
    """Coordinate-synchronized iteration over sparse cursors (two or more, or
    one or more beside a dense domain under ``DENSE_UNION``).

    Every cursor level must store the same logical dimension — the merged
    coordinate domain.  A merge admits non-leaf cursors when the loop binds
    their position through ``positions``: each entry pairs with the
    same-index cursor and names the :class:`PositionId` the body observes
    for that cursor's entry, which is the descent anchor a child level's
    :class:`SparseCursorDecl` parent expression consumes.  Under
    INTERSECTION every cursor is aligned at the body, so a bound position
    is always its aligned entry; a position-binding UNION merge must bind
    every cursor, and an entry anchors descent only while its cursor is
    aligned — a child stream chained from an unaligned parent is the empty
    segment for that candidate coordinate, also through dense levels between
    the two.  A possibly absent UNION position, and a dense position computed
    from one, is therefore valid only as the parent of a compressed-level
    cursor or as the address of a :class:`PositionLoad` guarded by the
    union's cursor; it is not an unconditional scalar-load address.
    ``positions`` is empty in the historical
    no-descent form; when nonempty it has one entry (a ``PositionId`` or
    ``None``) per cursor.

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

    ``DENSE_UNION`` differs only in coordinate selection: the candidate
    coordinates are every coordinate of the merged dimension, ``0 .. extent
    - 1`` in order, whatever the cursors store; a cursor is aligned when it
    is not exhausted and its current coordinate equals the candidate; the
    body runs at every candidate and aligned cursors advance after it.  It
    terminates after the last coordinate, when every cursor is exhausted
    because stored coordinates lie in the dimension's domain.  It needs one
    cursor or more, and reads and binds positions exactly as UNION does
    (an unaligned cursor's bound position anchors only the empty child
    stream).

    ``coord_index`` binds the candidate coordinate for the body.
    """

    mode: MergeMode
    cursors: Tuple[SparseCursorDecl, ...]
    coord_index: IndexId
    body: Block
    positions: Tuple[Optional[PositionId], ...] = ()


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
    """A coordinate-addressed semantic read-modify-write.

    Combines the stored element with ``value`` using ``op``
    (``target = target op value``).  Only ADD is admitted.  On an all-dense
    output, the target is zero-initialized before the body runs, exactly
    matching ADD's identity.  On an identity-ordered DENSE/COMPRESSED
    output, this is the target-neutral semantic accumulation form: the
    oracle combines contributions by coordinate, while target lowering
    requires a sparse-workspace pass to rewrite it into ordered assembly.
    There is deliberately no scalar-accumulator node in this subset.
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

    One exception, which the verifier admits structurally: under a recorded
    two-pass assembly strategy, the appends of a relabelled copy of one input
    -- one loop per input level in its storage order, appending each stored
    entry at a permutation of its coordinates -- may arrive in any order.
    Every coordinate is appended once, and the container is assembled from the
    appended entries in coordinate order (count, prefix sum, place), which is
    what an operand conversion needs (``verifier._check_append_order``).
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
class SparseWorkSource(LoopIRNode):
    """A target-independent per-iteration work measure: one stored level.

    ``tensor``/``level`` name a compressed physical level of a declared
    tensor reached by the first sparse cursor in a selected subtree that
    contains no merged loop.  That cursor must be immediately parented by the
    selected dense coordinate, and that parent tensor/level must be the exact
    physical input driver that supplies the target loop bound.  The count of
    its stored entries is the work the selection distributes.  No
    position-array name, C++ spelling, or policy text appears here — target
    lowering revalidates the fact and combines it with the selected loop's
    concrete bound.
    """

    node_id: LoopIRNodeId
    tensor: SymbolId
    level: int


@dataclass(frozen=True)
class ParallelWork(LoopIRNode):
    """The target-independent work estimate of one parallel selection.

    ``rows`` is the selected loop's trip-count source — the declared
    dimension the loop iterates (an ``OUTER`` selection still names the
    split dimension; the origin count is derived from it and the width).
    ``nnz`` optionally names the canonical sparse work measure dominating
    one iteration; ``None`` states that the target must use the row-count-only
    policy.  It does not claim that row costs are uniform (merged sparse
    nests deliberately use the row-only legacy policy).  Both fields are
    facts about the program, re-derivable and verified structurally — never
    trusted spelling.
    """

    node_id: LoopIRNodeId
    rows: DimensionId
    nnz: Optional[SparseWorkSource]


@dataclass(frozen=True)
class ParallelSelection(LoopIRNode):
    """One abstract parallel-loop selection over a stable loop identity.

    Names the selected loop by ``(index, part)`` — the same identity
    scheme the schedule passes already use — and carries the
    target-independent facts the design assigns to abstract
    parallelization: a work estimate, the race-freedom discipline the
    verifier re-proves, and the scheduling intent.  OpenMP spelling,
    thread counts, chunking, and scheduling policy are deliberately
    absent; target lowering revalidates and realizes the owned work fact
    against the selected loop exactly as the legacy explicit-parallel route
    does.
    """

    node_id: LoopIRNodeId
    index: IndexId
    part: ParallelPart
    discipline: ParallelDiscipline
    work: ParallelWork
    intent: ParallelIntent


@dataclass(frozen=True)
class VectorLoop(LoopIRNode):
    """One loop a schedule keeps marked for vector instructions.

    Names the loop by ``(index, part)``, the identity scheme of
    :class:`ParallelSelection`.  Which loops are safe to mark is a target
    fact (the shared rule ``iter_lattice.mark_simd_on_reduction_loops``: a
    loop of a sum kept in a register, with no loop over a compressed level's
    positions in its body, not parallel or unrolled, and with no early exit);
    the recorded marks choose among them.  The pragma spelling and any
    reduction clause are target decisions and appear nowhere in LoopIR.
    """

    node_id: LoopIRNodeId
    index: IndexId
    part: VectorLoopPart


@unique
class AssemblyStrategy(Enum):
    """How a sparse result is assembled, target-neutrally.

    One traversal or two, distributed across workers or not.  The member values
    are the same tokens the public schedule and the ``LoopPlan`` record, defined
    once in :mod:`scorch.compiler.sparse_assembly`; this enum is the typed form
    the program carries.  The names describe assembly *structure* only —
    OpenMP spelling, thread counts, chunk widths and runtime gates are target
    decisions and appear nowhere in LoopIR, exactly as they appear nowhere in
    :class:`ParallelSelection`.
    """

    SINGLE_PASS_SERIAL = "single_pass_serial"
    SINGLE_PASS_CHUNK_PARALLEL = "single_pass_chunk_parallel"
    TWO_PASS_SERIAL = "two_pass_serial"
    TWO_PASS_PARALLEL = "two_pass_parallel"


@unique
class AccumulatorStructure(Enum):
    """Which structure holds a sparse reduction's accumulation, target-neutrally.

    A list of the keys inserted, or a chain threaded through an array indexed by
    the key.  The member values are the same tokens the public schedule and the
    ``LoopPlan`` record, defined once in
    :mod:`scorch.compiler.sparse_accumulator`; this enum is the typed form the
    program carries.  The names describe accumulation *structure* only — the
    container, its capacity, whether it is pooled per worker and how an insert is
    spelled are target decisions and appear nowhere in LoopIR, exactly as they
    appear nowhere in :class:`AssemblyStrategy`.
    """

    COORDINATE_LIST = "coordinate_list"
    LINKED_LIST = "linked_list"


@dataclass(frozen=True)
class LoopProgram(LoopIRNode):
    """One executable LoopIR program: declarations plus a top-level block.

    ``parallel`` optionally carries the program's abstract parallel-loop
    selection.  ``None`` means no explicit selection exists and target
    lowering keeps its legacy structural derivation; the field is part of
    program semantics (canonically serialized, verified, erased with the
    schedule), never a target annotation.

    ``assembly`` optionally carries the program's sparse-output assembly
    strategy, on the same terms and for the same reason: which of the four
    strategies is fastest depends on the operands rather than on the lowering
    class, so it is a recorded scheduling decision.  ``None`` means no strategy
    decision was recorded and target lowering keeps its own per-receiver choice,
    which is why an automatically scheduled program emits exactly what it
    emitted before this field existed.

    ``accumulator`` optionally carries which structure holds a sparse
    reduction's accumulation, on the same terms again.  Which structure is
    faster depends on the density and on the receiver's compressed extent, so it
    is a scheduling decision rather than something an assembly transform may
    impose on the way past.  ``None`` means no structure decision was recorded
    and every existing layer keeps choosing what it chooses today — including
    the two-phase transform's substitution, which is why a program that records
    nothing emits exactly what it emitted before this field existed.

    ``entries_of`` optionally names the input whose stored entries are the
    sparse result's entries.  ``None`` means the result stores the entries its
    statements write: an :class:`AppendEntry` or :class:`StoreReduce` that
    executes.  With an input named, every stored entry of that input is a
    result entry, written or not, and its value is what the statements add
    there (zero where nothing reaches it); a statement may write no entry the
    input does not store.  The two give the same values and differ in which
    zeros are stored.  The front end names an input only when the plan does
    (``LoopPlan.entries_of``, which a schedule requests as
    ``Schedule.entries_of``), and only where a reduction into a COO result is a
    product with one factor that has exactly the result's indices, in the
    result's logical and storage order, under the result's loops
    (:func:`~scorch.compiler.sparse_assembly.entry_source`): that is the
    structure the shipped compiler builds under scalar accumulation, and its
    size is known before the loops run, so the result is allocated once and
    written by position.  With no plan decision it names none.  Unlike the
    three fields above, this one is program semantics (it decides which zeros
    are stored): the front end records it from the plan, every scheduling pass
    keeps it, and schedule erasure does not remove it.

    ``vector_loops`` optionally records which loops keep a vector mark
    (:class:`VectorLoop`), on the same terms as ``parallel``, ``assembly`` and
    ``accumulator``: a scheduling decision, stamped last, erased with the
    schedule.  ``None`` means no decision was recorded and the target places
    the marks it placed before this field existed, which is none: the shared
    rule runs only on a recorded decision, so a program that records nothing
    emits exactly what it emitted before.  (The shipped compiler, with no
    decision, marks the safe loops below a sparse result entry whose sum is
    kept in a register.)  A tuple, the empty one included, keeps the mark on
    exactly the named loops; naming a loop the rule does not find safe is
    refused (``unsupported_vector_loop``).  The shipped compiler reads the same
    decision from ``LoopPlan.vector_loops``, so one tuple marks the same loops
    in both compilers.
    """

    node_id: LoopIRNodeId
    dimensions: Tuple[DimensionDecl, ...]
    tensors: Tuple[TensorDecl, ...]
    inputs: Tuple[SymbolId, ...]
    outputs: Tuple[SymbolId, ...]
    body: Block
    parallel: Optional[ParallelSelection] = None
    assembly: Optional[AssemblyStrategy] = None
    accumulator: Optional[AccumulatorStructure] = None
    entries_of: Optional[SymbolId] = None
    vector_loops: Optional[Tuple[VectorLoop, ...]] = None
