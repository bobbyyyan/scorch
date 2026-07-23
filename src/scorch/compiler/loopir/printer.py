"""Deterministic printing and canonical serialization for production LoopIR.

Both surfaces operate only on verified programs (they call the fail-closed
verifier first) and never depend on allocation-order identity values, memory
addresses, or dictionary ordering:

- :func:`print_program` renders a human-readable stage dump.  Dimensions,
  tensors, and loop indices are shown with traversal-canonical local labels
  (``d0``/``t0``/``x0``) plus their display names, so two independently
  constructed equivalent programs — including ones whose global
  ``SymbolId``/``IndexId`` allocation histories differ — print identically.
- :func:`canonical_program_dump` serializes the complete semantic content as
  compact JSON with traversal-renumbered identities and an explicit schema
  version.  Display names are deliberately omitted: they are presentation,
  not semantics, and must not enter cache or fingerprint identity.  There is
  currently no deserializer, so no round-trip contract is declared.
"""

from __future__ import annotations

import json
from typing import Dict, List

from ..identity import IndexId, SymbolId
from .nodes import (
    BinaryExpr,
    Block,
    DenseFor,
    DimensionId,
    Expr,
    IndexValue,
    Load,
    LoopProgram,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
)
from .verifier import verify_program

CANONICAL_SCHEMA = "scorch.loopir.canonical.v1"


class _CanonicalIds:
    """First-appearance renumbering for every identity family."""

    def __init__(self) -> None:
        self._dimensions: Dict[DimensionId, int] = {}
        self._symbols: Dict[SymbolId, int] = {}
        self._indices: Dict[IndexId, int] = {}

    def dimension(self, dimension: DimensionId) -> int:
        return self._dimensions.setdefault(dimension, len(self._dimensions))

    def symbol(self, symbol: SymbolId) -> int:
        return self._symbols.setdefault(symbol, len(self._symbols))

    def index(self, index: IndexId) -> int:
        return self._indices.setdefault(index, len(self._indices))


def _serialize_expr(expr: Expr, ids: _CanonicalIds) -> Dict[str, object]:
    if type(expr) is IndexValue:
        return {"kind": "index", "index": ids.index(expr.index)}
    if type(expr) is Load:
        return {
            "kind": "load",
            "tensor": ids.symbol(expr.tensor),
            "indices": [_serialize_expr(index, ids) for index in expr.indices],
        }
    if type(expr) is BinaryExpr:
        return {
            "kind": "binary",
            "op": expr.op.value,
            "lhs": _serialize_expr(expr.lhs, ids),
            "rhs": _serialize_expr(expr.rhs, ids),
        }
    raise TypeError(f"unsupported LoopIR expression {type(expr).__name__}")


def _serialize_stmt(stmt: Stmt, ids: _CanonicalIds) -> Dict[str, object]:
    if type(stmt) is Block:
        return {
            "kind": "block",
            "statements": [_serialize_stmt(child, ids) for child in stmt.statements],
        }
    if type(stmt) is DenseFor:
        return {
            "kind": "dense_for",
            "index": ids.index(stmt.index),
            "dimension": ids.dimension(stmt.dimension),
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is Store:
        return {
            "kind": "store",
            "tensor": ids.symbol(stmt.tensor),
            "indices": [_serialize_expr(index, ids) for index in stmt.indices],
            "value": _serialize_expr(stmt.value, ids),
        }
    if type(stmt) is StoreReduce:
        return {
            "kind": "store_reduce",
            "tensor": ids.symbol(stmt.tensor),
            "op": stmt.op.value,
            "indices": [_serialize_expr(index, ids) for index in stmt.indices],
            "value": _serialize_expr(stmt.value, ids),
        }
    raise TypeError(f"unsupported LoopIR statement {type(stmt).__name__}")


def _serialize_tensor(decl: TensorDecl, ids: _CanonicalIds) -> Dict[str, object]:
    return {
        "symbol": ids.symbol(decl.symbol),
        "dtype": decl.dtype.value,
        "dimensions": [ids.dimension(dimension) for dimension in decl.dimensions],
        "levels": [
            {"kind": level.kind.value, "mode": level.mode} for level in decl.levels
        ],
    }


def canonical_program_dump(program: LoopProgram) -> str:
    """Serialize one verified program canonically, independent of ID history."""

    verify_program(program)
    ids = _CanonicalIds()
    for dimension_decl in program.dimensions:
        ids.dimension(dimension_decl.dimension)
    for tensor_decl in program.tensors:
        ids.symbol(tensor_decl.symbol)
    payload = {
        "schema": CANONICAL_SCHEMA,
        "dimensions": [
            {"dimension": ids.dimension(decl.dimension)} for decl in program.dimensions
        ],
        "tensors": [_serialize_tensor(decl, ids) for decl in program.tensors],
        "inputs": [ids.symbol(symbol) for symbol in program.inputs],
        "outputs": [ids.symbol(symbol) for symbol in program.outputs],
        "body": _serialize_stmt(program.body, ids),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _render_expr(expr: Expr, ids: _CanonicalIds, names: Dict[int, str]) -> str:
    if type(expr) is IndexValue:
        return f"x{ids.index(expr.index)}"
    if type(expr) is Load:
        rendered = ", ".join(_render_expr(index, ids, names) for index in expr.indices)
        return f"load t{ids.symbol(expr.tensor)}[{rendered}]"
    if type(expr) is BinaryExpr:
        return (
            f"{expr.op.value}({_render_expr(expr.lhs, ids, names)}, "
            f"{_render_expr(expr.rhs, ids, names)})"
        )
    raise TypeError(f"unsupported LoopIR expression {type(expr).__name__}")


def _render_stmt(
    stmt: Stmt,
    ids: _CanonicalIds,
    names: Dict[int, str],
    indent: int,
    lines: List[str],
) -> None:
    pad = "  " * indent
    if type(stmt) is Block:
        for child in stmt.statements:
            _render_stmt(child, ids, names, indent, lines)
        return
    if type(stmt) is DenseFor:
        label = f"x{ids.index(stmt.index)}"
        lines.append(f"{pad}for {label} in d{ids.dimension(stmt.dimension)} {{")
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is Store:
        rendered = ", ".join(_render_expr(index, ids, names) for index in stmt.indices)
        lines.append(
            f"{pad}store t{ids.symbol(stmt.tensor)}[{rendered}] = "
            f"{_render_expr(stmt.value, ids, names)}"
        )
        return
    if type(stmt) is StoreReduce:
        rendered = ", ".join(_render_expr(index, ids, names) for index in stmt.indices)
        lines.append(
            f"{pad}store_reduce({stmt.op.value}) "
            f"t{ids.symbol(stmt.tensor)}[{rendered}] = "
            f"{_render_expr(stmt.value, ids, names)}"
        )
        return
    raise TypeError(f"unsupported LoopIR statement {type(stmt).__name__}")


def print_program(program: LoopProgram) -> str:
    """Render one verified program as a deterministic human-readable dump."""

    verify_program(program)
    ids = _CanonicalIds()
    lines: List[str] = ["loopir.program {"]
    for dimension_decl in program.dimensions:
        lines.append(
            f"  dimension d{ids.dimension(dimension_decl.dimension)} "
            f"{dimension_decl.name!r}"
        )
    for decl in program.tensors:
        levels = ", ".join(f"{level.kind.value}@{level.mode}" for level in decl.levels)
        dimensions = ", ".join(
            f"d{ids.dimension(dimension)}" for dimension in decl.dimensions
        )
        lines.append(
            f"  tensor t{ids.symbol(decl.symbol)} {decl.name!r} "
            f"{decl.dtype.value} dims({dimensions}) levels({levels})"
        )
    inputs = ", ".join(f"t{ids.symbol(symbol)}" for symbol in program.inputs)
    outputs = ", ".join(f"t{ids.symbol(symbol)}" for symbol in program.outputs)
    lines.append(f"  inputs({inputs})")
    lines.append(f"  outputs({outputs})")
    lines.append("  body {")
    _render_stmt(program.body, ids, {}, 2, lines)
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"
