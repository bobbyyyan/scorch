"""Typed, stable identities for logical compiler entities."""

from dataclasses import dataclass
from itertools import count
from threading import Lock


@dataclass(frozen=True, order=True)
class SymbolId:
    """Identity of a logical tensor or workspace symbol within an artifact."""

    value: int


@dataclass(frozen=True, order=True)
class IndexId:
    """Identity of a logical index independent of its display name."""

    value: int


_symbol_ids = count()
_index_ids = count()
_id_lock = Lock()


def new_symbol_id() -> SymbolId:
    """Allocate an identity that remains stable across copies of one artifact."""

    with _id_lock:
        return SymbolId(next(_symbol_ids))


def new_index_id() -> IndexId:
    """Allocate an identity that remains stable across copies of one artifact."""

    with _id_lock:
        return IndexId(next(_index_ids))
