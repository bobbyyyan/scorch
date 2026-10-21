"""A sparse result's entries bounded at dispatch, before its kernel is built or run.

A generated kernel that assembles a sparse result one entry at a time grows each level's
coordinates and values by appends that check nothing.  Such a result was refused only
after the kernel returned (``stensor._refuse_oversized_result_arrays``), once every entry
was in memory: for a result past ``2**31 - 1`` entries about 26 GB at float64 before the
refusal, or the process killed for memory first, which is not fail-closed.  The kernel's
entry check bounds only the levels down to the deepest dense one, and 864745d refused at
entry every result whose logical element count passed ``2**31 - 1``, so for those inputs
the merged compiler's behaviour was a regression.

This module bounds every level from the result's first compressed, coordinate or
singleton level on, from the shapes and the operands' stored counts, with no loop over
the data.  Levels in storage order ``0 .. n-1``; for a level ``L``:

``P_L``
    the logical cells of the prefix: the product of the extents of levels ``0 .. L``.
``S_L``
    a bound from the stored counts.  Write ``R`` for the result indices of levels
    ``0 .. L``.  In a product term every stored prefix needs a stored entry of every
    factor, so with ``p_k = min(stored_k, cells of the indices of R factor k carries)``
    and ``q_k`` the cells of the indices of ``R`` it does not carry, a term holds at most
    ``min(prod_k p_k * u, min_k p_k * q_k)`` prefixes, ``u`` being the cells of the
    indices of ``R`` no factor carries.  A sum holds at most the sum over its terms.
``B_L = min(P_L, S_L)``
    bounds ``E_L``, the entries the level holds.  A compressed level holds one per
    stored prefix.  A dense level below it holds its parent's entries times its extent,
    which ``P_L`` and ``S_L`` count the same way.  A coordinate or singleton level holds
    one entry per entry of the deepest level of its run, which the deepest level's bound
    covers.

The limit is 0038's: ``2**31 - 1`` entries, one fewer when a compressed level lies below
(its position array holds one entry more).  A result whose shape cannot pass the limit at
any level is never bounded again: :func:`entry_bound` returns ``None`` for it.  einsum keeps
that ``None`` in its dispatch cache entry, so such a warm call pays one test of a cached
value; addition and the conversions first compare the product of the result's extents
with ``MAX_INDEX_POSITION`` (no Python call), which no level can pass while it is smaller.  A level whose total the kernel checks itself before allocating
it (the two-pass assembly, 0038c's ``scorch_check_level_total``) is left to that exact
check: the bound can overstate, and refusing there would refuse a result that fits.
"""

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .exceptions import IndexCapacityError, ScorchAssertionError
from .format import MAX_INDEX_POSITION, LevelType

#: csrc/header.h's level total check as the shipped code generator writes it
#: (``scorch_check_level_total(_total1, false, 1);``): its last argument is the level.
_LEVEL_TOTAL_CALL = re.compile(
    r"\bscorch_check_level_total\(\s*[A-Za-z_]\w*\s*,\s*(?:true|false|0|1)\s*,"
    r"\s*(?:static_cast<int>\()?\s*(\d+)\s*\)?\s*\)"
)


def level_totals_in_source(source: str) -> FrozenSet[int]:
    """The result levels whose totals a kernel's source checks before allocating them."""

    return frozenset(int(level) for level in _LEVEL_TOTAL_CALL.findall(source))


@dataclass(frozen=True)
class LevelBound:
    level: int
    kind: LevelType
    prefix_cells: int
    stored_bound: int
    limit: int

    @property
    def bound(self) -> int:
        return min(self.prefix_cells, self.stored_bound)


def _cells(extents: Dict[str, int], indices: Iterable[str]) -> int:
    product = 1
    for index in indices:
        product *= extents[index]
    return product


def _term_bound(
    term: Sequence[Tuple[FrozenSet[str], int]],
    prefix: FrozenSet[str],
    extents: Dict[str, int],
) -> int:
    """The stored prefixes one product term can make: ``term`` is (indices, stored) per
    factor."""

    carried = [indices & prefix for indices, _ in term]
    distinct = [
        min(stored, _cells(extents, indices))
        for (_, stored), indices in zip(term, carried)
    ]
    uncarried = prefix.difference(*carried) if carried else prefix
    joint = _cells(extents, uncarried)
    for count in distinct:
        joint *= count
    each = [
        count * _cells(extents, prefix - indices)
        for count, indices in zip(distinct, carried)
    ]
    return min([joint, *each])


@dataclass(frozen=True)
class EntryBound:
    """The bound of one dispatch entry's result: its levels' names, extents and kinds,
    the expression's terms as the operand positions and the result indices each factor
    carries, and the levels whose totals the kernel checks itself.  Built only for a
    result whose shape can pass the limit at some level (:func:`entry_bound`)."""

    what: str
    names: Tuple[str, ...]
    extents: Tuple[int, ...]
    kinds: Tuple[LevelType, ...]
    #: per term, per factor: (operand position, the result's index names it carries)
    terms: Tuple[Tuple[Tuple[int, FrozenSet[str]], ...], ...]
    #: (level, P_L, limit) of each level from the first sparse one on
    limits: Tuple[Tuple[int, int, int], ...]
    checked: FrozenSet[int]

    def levels(self, stored: Sequence[int]) -> List[LevelBound]:
        """Every bounded level, from the operands' stored counts (by operand position)."""

        extents = dict(zip(self.names, self.extents))
        bounds = []
        for level, cells, limit in self.limits:
            prefix = frozenset(self.names[: level + 1])
            total = sum(
                _term_bound(
                    [(indices, stored[k]) for k, indices in term],
                    prefix,
                    extents,
                )
                for term in self.terms
            )
            bounds.append(LevelBound(level, self.kinds[level], cells, total, limit))
        if not bounds:
            return bounds
        deepest = bounds[-1]
        # A coordinate or singleton level stores one entry per entry of the deepest
        # level of its run: the deepest level's bound covers it.
        return [
            (
                LevelBound(
                    bound.level,
                    bound.kind,
                    max(bound.prefix_cells, deepest.prefix_cells),
                    max(bound.stored_bound, deepest.stored_bound),
                    bound.limit,
                )
                if bound.kind in (LevelType.COORDINATE, LevelType.SINGLETON)
                else bound
            )
            for bound in bounds
        ]

    def with_checked(self, checked: FrozenSet[int]) -> "EntryBound":
        return EntryBound(
            self.what,
            self.names,
            self.extents,
            self.kinds,
            self.terms,
            self.limits,
            frozenset(checked),
        )

    def refuse_past_limit(self, stored: Sequence[int]) -> None:
        """Raise IndexCapacityError (index_position_exceeds_int32) naming the first level
        assembled one entry at a time whose bound passes its limit."""

        for bound in self.levels(stored):
            if bound.bound <= bound.limit or bound.level in self.checked:
                continue
            counts = "; ".join(
                " * ".join(f"operand {k} {stored[k]}" for k, _ in term)
                for term in self.terms
            )
            raise IndexCapacityError(
                f"{self.what}: result level {bound.level} ({bound.kind.name.lower()}) "
                f"may hold {bound.bound} entries: at most {bound.prefix_cells} logical "
                f"cells and at most {bound.stored_bound} from the operands' stored "
                f"entries ({counts}).  A 32-bit position addresses at most "
                f"{bound.limit}, and the kernel assembles this level one entry at a "
                "time without checking its total, so the result is refused before "
                "the kernel runs"
            )


def entry_bound(
    what: str,
    names: Sequence[str],
    extents: Sequence[int],
    kinds: Sequence[LevelType],
    terms: Sequence[Sequence[Tuple[int, Iterable[str]]]],
) -> Optional[EntryBound]:
    """The bound of a result whose shape can pass the limit at some sparse level, or
    ``None`` when no level's logical cells can: then no stored count can either.

    ``names``, ``extents`` and ``kinds`` describe the result's levels in storage order;
    ``terms`` the expression's sum of products, each factor as (operand position, the
    index names it carries)."""

    if not (len(names) == len(extents) == len(kinds)):
        # An internal invariant, not an input check: each caller builds the three from one
        # rank (einsum from the result's indices, its shape and its format; addition and
        # the conversions from the result's shape and format), and every public entry
        # refuses a format or shape of another rank first, with its own code.
        raise ScorchAssertionError(
            "a result bound needs one name, extent and kind per level: "
            f"{len(names)} names, {len(extents)} extents, {len(kinds)} kinds",
            code="result_bound_levels_misaligned",
        )
    sparse = [level for level, kind in enumerate(kinds) if kind is not LevelType.DENSE]
    if not sparse:
        return None
    limits = []
    cells = 1
    for level, extent in enumerate(extents):
        cells *= int(extent)
        if level < sparse[0]:
            continue
        below = any(kind is LevelType.COMPRESSED for kind in kinds[level + 1 :])
        limits.append((level, cells, MAX_INDEX_POSITION - int(below)))
    if not any(cells > limit for _, cells, limit in limits):
        return None
    return EntryBound(
        what,
        tuple(names),
        tuple(int(extent) for extent in extents),
        tuple(kinds),
        tuple(
            tuple((int(k), frozenset(indices)) for k, indices in term) for term in terms
        ),
        tuple(limits),
        frozenset(),
    )
