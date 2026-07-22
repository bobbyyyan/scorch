"""Frozen, tuple-owned generic IR nodes for the Phase-3.5 LoopIR spike.

Every node is a plain frozen dataclass whose children are owned tuples.  The
constructors deliberately perform no validation: the fail-closed authority for
structure, scope, typing, and layout is ``verifier.verify_program``, and the
adversarial verifier tests need to be able to build malformed nodes directly.
Semantics that a target backend or the spike interpreter must honour are
documented on the node that owns them; nothing here names a target language,
a rendered symbol, or an operation.

Identity reuses the stable production identity types where they fit:
``SymbolId`` identifies tensors and reduction accumulators, ``IndexId``
identifies bound iteration coordinates.  ``LoopNodeId`` and ``CursorId`` are
spike-local because production ``NodeId``/``AccessId`` are documented as CIN
identities and this schema is explicitly a revisable candidate.
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


_loop_node_ids = count()
_cursor_ids = count()
_id_lock = Lock()


def new_loop_node_id() -> LoopNodeId:
    """Allocate a program-unique identity for one LoopIR node."""

    with _id_lock:
        return LoopNodeId(next(_loop_node_ids))


def new_cursor_id() -> CursorId:
    """Allocate a program-unique identity for one sparse cursor."""

    with _id_lock:
        return CursorId(next(_cursor_ids))


@unique
class LevelKind(Enum):
    """Per-mode storage layout of a declared tensor, target-neutrally."""

    DENSE = "dense"
    COMPRESSED = "compressed"


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
    """Base of all value-producing nodes."""

    node_id: LoopNodeId


@dataclass(frozen=True)
class Stmt(LoopIRNode):
    """Base of all statement nodes."""

    node_id: LoopNodeId


@dataclass(frozen=True)
class IntConst(Expr):
    """A coordinate/extent-typed integer constant."""

    value: int


@dataclass(frozen=True)
class FloatConst(Expr):
    """A value-typed floating-point constant."""

    value: float


@dataclass(frozen=True)
class DimSize(Expr):
    """The extent of one dimension of a declared tensor (coordinate-typed)."""

    tensor: SymbolId
    dim: int


@dataclass(frozen=True)
class IndexValue(Expr):
    """The current coordinate bound by an enclosing loop (coordinate-typed)."""

    index: IndexId


@dataclass(frozen=True)
class CursorValue(Expr):
    """The stored value at an in-scope sparse cursor's current position.

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

    Compressed tensors are deliberately not loadable by coordinate: their
    values are only reachable through :class:`CursorValue`, which is what
    keeps coordinate search, name rendering, and format-specific probing out
    of the schema.
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
    """Dense iteration binding ``index`` to ``0 .. extent - 1`` in order."""

    index: IndexId
    extent: Expr
    body: Block


@dataclass(frozen=True)
class SparseCursorDecl(LoopIRNode):
    """One sparse cursor over a compressed level of a declared input tensor.

    ``outer_indices`` supplies the already-bound coordinates of every level
    above ``level`` (so its length equals ``level``); together they select
    the stored segment the cursor walks.  A cursor exposes, intrinsically,
    its current position, the coordinate stored at that position, its stored
    value, and whether it is exhausted; no node ever re-derives these from
    rendered names or target syntax.
    """

    node_id: LoopNodeId
    cursor: CursorId
    tensor: SymbolId
    level: int
    outer_indices: Tuple[Expr, ...]


@dataclass(frozen=True)
class SparseFor(Stmt):
    """Position iteration over one sparse cursor's selected segment.

    Each iteration binds ``coord_index`` to the cursor's stored coordinate
    and runs ``body`` once per stored entry, in storage order.  There is
    exactly one cursor, so this node never exercises sparse-sparse merge
    semantics.
    """

    cursor: SparseCursorDecl
    coord_index: IndexId
    body: Block


@dataclass(frozen=True)
class MergedSparseFor(Stmt):
    """Coordinate-synchronized iteration over two or more sparse cursors.

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
    """A coordinate-addressed write of a value into an all-dense output."""

    tensor: SymbolId
    indices: Tuple[Expr, ...]
    value: Expr


@dataclass(frozen=True)
class AppendEntry(Stmt):
    """Target-neutral sparse-output assembly: append one stored entry.

    Appends carry the full coordinate tuple plus the value.  Successive
    appends to one output must be lexicographically strictly increasing in
    their coordinates; the canonical container is assembled from that stream
    alone, so no node encodes positions arrays, offsets, or target storage
    layout.  The interpreter fails closed on out-of-order appends.
    """

    tensor: SymbolId
    coords: Tuple[Expr, ...]
    value: Expr


@dataclass(frozen=True)
class TensorDecl(LoopIRNode):
    """A declared tensor: stable symbol, display name, per-level layout."""

    node_id: LoopNodeId
    symbol: SymbolId
    name: str
    levels: Tuple[LevelKind, ...]


@dataclass(frozen=True)
class LoopProgram(LoopIRNode):
    """One executable spike program: declarations plus a top-level block."""

    node_id: LoopNodeId
    tensors: Tuple[TensorDecl, ...]
    inputs: Tuple[SymbolId, ...]
    outputs: Tuple[SymbolId, ...]
    body: Block
