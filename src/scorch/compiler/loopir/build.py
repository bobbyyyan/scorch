"""Construction API for production LoopIR programs.

:class:`LoopIRBuilder` is the supported way to construct well-formed LoopIR:
it owns the artifact-local :class:`~scorch.compiler.loopir.nodes.LoopIRNodeId`
and :class:`~scorch.compiler.loopir.nodes.DimensionId` allocation for one
program, so identical construction sequences always allocate identical
identities regardless of process history.  Tensor symbols and loop indices
are production :class:`~scorch.compiler.identity.SymbolId` /
:class:`~scorch.compiler.identity.IndexId` values: callers lowering from CIN
pass the CIN identities through unchanged (provenance), while hand-built
programs may ask the builder for fresh ones.

The builder performs no semantic validation — it only allocates identities
and owns tuple conversion.  ``verifier.verify_program`` remains the single
fail-closed authority; ``program`` deliberately does not call it so
adversarial tests can build malformed programs through the same API surface.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from ..identity import IndexId, SymbolId, new_index_id, new_symbol_id
from .nodes import (
    BinaryExpr,
    BinaryOp,
    Block,
    DenseFor,
    DimensionDecl,
    DimensionId,
    Expr,
    IndexValue,
    LevelDecl,
    LevelKind,
    Load,
    LoopIRNodeId,
    LoopProgram,
    ReduceOp,
    ScalarType,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
)


class LoopIRBuilder:
    """Allocate artifact-local identities and assemble one LoopIR program."""

    def __init__(self) -> None:
        self._next_node_id = 0
        self._next_dimension_id = 0

    def _node_id(self) -> LoopIRNodeId:
        node_id = LoopIRNodeId(self._next_node_id)
        self._next_node_id += 1
        return node_id

    def new_dimension_id(self) -> DimensionId:
        """Allocate the next artifact-local dimension identity."""

        dimension = DimensionId(self._next_dimension_id)
        self._next_dimension_id += 1
        return dimension

    @staticmethod
    def new_symbol_id() -> SymbolId:
        """Allocate a fresh production tensor symbol for a hand-built program."""

        return new_symbol_id()

    @staticmethod
    def new_index_id() -> IndexId:
        """Allocate a fresh production loop index for a hand-built program."""

        return new_index_id()

    def dimension(self, name: str) -> DimensionDecl:
        """Declare a dimension with an identity owned by this builder.

        Callers that need malformed or externally identified declarations can
        construct :class:`DimensionDecl` directly. Keeping that adversarial
        surface out of the supported builder prevents an explicit identity
        from being reissued later by the automatic allocator.
        """

        return DimensionDecl(self._node_id(), self.new_dimension_id(), name)

    def level(self, kind: LevelKind, mode: int) -> LevelDecl:
        return LevelDecl(self._node_id(), kind, mode)

    def dense_levels(self, rank: int) -> Tuple[LevelDecl, ...]:
        """Identity-mode-order all-dense levels for a rank-``rank`` tensor."""

        return tuple(self.level(LevelKind.DENSE, mode) for mode in range(rank))

    def tensor(
        self,
        symbol: SymbolId,
        name: str,
        dtype: ScalarType,
        dimensions: Sequence[DimensionId],
        levels: Sequence[LevelDecl],
    ) -> TensorDecl:
        return TensorDecl(
            self._node_id(),
            symbol,
            name,
            dtype,
            tuple(dimensions),
            tuple(levels),
        )

    def index_value(self, index: IndexId) -> IndexValue:
        return IndexValue(self._node_id(), index)

    def load(self, tensor: SymbolId, indices: Sequence[Expr]) -> Load:
        return Load(self._node_id(), tensor, tuple(indices))

    def binary(self, op: BinaryOp, lhs: Expr, rhs: Expr) -> BinaryExpr:
        return BinaryExpr(self._node_id(), op, lhs, rhs)

    def block(self, statements: Sequence[Stmt]) -> Block:
        return Block(self._node_id(), tuple(statements))

    def dense_for(
        self, index: IndexId, dimension: DimensionId, body: Block
    ) -> DenseFor:
        return DenseFor(self._node_id(), index, dimension, body)

    def store(self, tensor: SymbolId, indices: Sequence[Expr], value: Expr) -> Store:
        return Store(self._node_id(), tensor, tuple(indices), value)

    def store_reduce(
        self,
        tensor: SymbolId,
        indices: Sequence[Expr],
        op: ReduceOp,
        value: Expr,
    ) -> StoreReduce:
        return StoreReduce(self._node_id(), tensor, tuple(indices), op, value)

    def program(
        self,
        dimensions: Sequence[DimensionDecl],
        tensors: Sequence[TensorDecl],
        inputs: Sequence[SymbolId],
        outputs: Sequence[SymbolId],
        body: Block,
    ) -> LoopProgram:
        return LoopProgram(
            self._node_id(),
            tuple(dimensions),
            tuple(tensors),
            tuple(inputs),
            tuple(outputs),
            body,
        )
