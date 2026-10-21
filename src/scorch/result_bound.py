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
singleton level on.  Levels in storage order ``0 .. n-1``; for a level ``L``:

``P_L``
    the logical cells of the prefix: the product of the extents of levels ``0 .. L``.
``S_L``
    a bound from the stored counts, with no loop over the data.  Write ``R`` for the
    result indices of levels ``0 .. L``.  In a product term every stored prefix needs a
    stored entry of every factor, so with ``p_k = min(stored_k, cells of the indices of R
    factor k carries)`` and ``q_k`` the cells of the indices of ``R`` it does not carry, a
    term holds at most ``min(prod_k p_k * u, min_k p_k * q_k)`` prefixes, ``u`` being the
    cells of the indices of ``R`` no factor carries.  A sum holds at most the sum over its
    terms.
``J``
    a bound from the operands' coordinates, read only when ``min(P_L, S_L)`` passes the
    limit at a level the kernel does not check (so a call accepted without it pays
    nothing).  ``S_L`` never sees an index the factors contract: for ``C(i,k) =
    A(i,j) B(j,k)`` it stays at ``m n`` however few products there are.  Every stored
    coordinate of a product term is the projection of one tuple of the join of its
    factors' stored entries on the indices they share, so the join's size bounds the
    term's entries at every level.  Each factor is counted at the deepest of its levels
    holding an index the result or another factor uses (its positions there: an upper
    bound on the distinct tuples of those indices, exact when they are a prefix of its
    storage order), grouped by its coordinates at the indices it shares.  Two factors
    sharing the indices ``K`` join in exactly ``sum over k in K of n_A(k) n_B(k)``
    tuples, which for SpGEMM is its multiply count; a further factor multiplies that by
    its largest group on the indices it shares with those joined before it (the standard
    degree bound), the pair and the order chosen to make the product smallest.  A dense
    prefix's groups are equal and need no pass over data.  J is exact below
    ``_SATURATION`` (``2**42``); at or past it the count stops and J is a lower bound,
    reported as one and never used as the level's upper bound.  The value is cached per
    operand structure (:func:`_cached_join`), so a warm call on unchanged index arrays
    does not count again.
``B_L``
    bounds ``E_L``, the entries the level holds.  At a compressed level, one per stored
    prefix: ``min(P_L, S_L, J)``.  At a dense level below a sparse one, its parent's
    entries times its extent: ``min(P_L, B_{L-1} * extent)``; ``S_L`` counts distinct
    stored coordinates, which do not bound a dense level's fill.  A coordinate or
    singleton level holds one entry per entry of the deepest level of its run, which the
    deepest level's bound covers.

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
import weakref
from dataclasses import dataclass
from itertools import combinations
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch

from .exceptions import IndexCapacityError, ScorchAssertionError
from .format import MAX_INDEX_POSITION, LevelType
from .storage import level_keys, level_positions, positions_beneath
from .storage import structure_of, structure_stamp

#: csrc/header.h's level total check as the shipped code generator writes it
#: (``scorch_check_level_total(_total1, false, 1);``): its last argument is the level.
_LEVEL_TOTAL_CALL = re.compile(
    r"\bscorch_check_level_total\(\s*[A-Za-z_]\w*\s*,\s*(?:true|false|0|1)\s*,"
    r"\s*(?:static_cast<int>\()?\s*(\d+)\s*\)?\s*\)"
)

#: A group weight at or past this is past every limit by itself (``2**31 > 2**31 - 1``);
#: two weights below it multiply exactly in int64.
_WEIGHT_CAP = 2**31

#: J is counted exactly below this and is a lower bound at or past it, far past every
#: limit.  Each product is held at or below it before summing.
_SATURATION = 2**42

#: Products summed per int64 sum: ``2**20`` values of at most ``2**42`` stay below
#: ``2**63``.
_DOT_CHUNK = 1 << 20

#: Group a factor's keys into a dense vector while its key cells are at most this or
#: twice its positions, else sort them.
_VECTOR_CELLS = 1 << 22


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
    #: J, when the operands' coordinates were read (``None`` otherwise)
    join_bound: Optional[int] = None

    @property
    def bound(self) -> int:
        if self.join_bound is None:
            return min(self.prefix_cells, self.stored_bound)
        return min(self.prefix_cells, self.stored_bound, self.join_bound)


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
class _Uniform:
    """One factor's positions at its counted level, grouped by its coordinates at the
    indices it shares with the factors joined to it, when every one of ``domain`` keys
    holds the same ``weight`` (a dense prefix)."""

    total: int
    largest: int
    domain: int
    weight: int


@dataclass(frozen=True)
class _Weighted:
    """As :class:`_Uniform`, with a weight per key: over the keys ``0 .. domain-1`` when
    ``keys`` is ``None``, else for the sorted distinct ``keys``."""

    total: int
    largest: int
    domain: int
    weights: torch.Tensor
    keys: Optional[torch.Tensor] = None


_Groups = Union[_Uniform, _Weighted]


def _weights_at(groups: _Weighted, keys: torch.Tensor) -> torch.Tensor:
    """``groups``' weight at each of ``keys`` (0 where it holds none)."""

    if groups.keys is None:
        return groups.weights[keys]
    if groups.keys.numel() == 0:
        return torch.zeros_like(keys)
    found = torch.searchsorted(groups.keys, keys).clamp_(max=groups.keys.numel() - 1)
    held = groups.keys[found] == keys
    return torch.where(held, groups.weights[found], torch.zeros_like(keys))


def _saturating_dot(left: torch.Tensor, right: torch.Tensor) -> Tuple[int, bool]:
    """``sum(left * right)`` (non-negative int64 weights) and whether it saturated.

    Exact unless it saturates: when the sum reaches ``_SATURATION``, or a weight at or
    past ``_WEIGHT_CAP`` meets a nonzero one.  Then the value returned is a lower bound
    of the true sum, and past every limit: at least ``_WEIGHT_CAP``."""

    capped = bool(
        (
            ((left >= _WEIGHT_CAP) & (right > 0))
            | ((right >= _WEIGHT_CAP) & (left > 0))
        ).any()
    )
    products = left.clamp(max=_WEIGHT_CAP) * right.clamp(max=_WEIGHT_CAP)
    products.clamp_(max=_SATURATION)
    total = 0
    for chunk in products.split(_DOT_CHUNK):
        total += int(chunk.sum())
        if total >= _SATURATION:
            return total, True
    return total, capped


def _pair_join(first: _Groups, second: _Groups) -> Tuple[int, bool]:
    """The tuples of the join of two factors on the indices they share."""

    if isinstance(first, _Uniform) and isinstance(second, _Uniform):
        return first.weight * second.weight * min(first.domain, second.domain), False
    if isinstance(first, _Uniform):
        return first.weight * second.total, False
    if isinstance(second, _Uniform):
        return second.weight * first.total, False
    # Both weighted: look up the factor with fewer keys in the other's.
    if second.keys is not None and (
        first.keys is None or second.keys.numel() < first.keys.numel()
    ):
        return _saturating_dot(second.weights, _weights_at(first, second.keys))
    if first.keys is not None:
        return _saturating_dot(first.weights, _weights_at(second, first.keys))
    return _saturating_dot(first.weights, second.weights)


class _Factor:
    """One operand of a product term, read through its storage, ``names`` the index name
    of each of its logical modes (as many as its levels: :meth:`EntryBound.join_bound`
    checks)."""

    def __init__(self, storage: Any, names: Sequence[str]) -> None:
        kinds, arrays, extents, permutation = structure_of(storage)
        self.kinds = kinds
        self.arrays = arrays
        self.extents = extents
        #: the index name each storage level holds
        self.names = tuple(names[permutation[level]] for level in range(len(kinds)))
        self.indices = frozenset(self.names)
        self.depth = -1
        self._groups: Dict[FrozenSet[str], Optional[_Groups]] = {}

    def extent_of(self, name: str) -> int:
        return self.extents[self.names.index(name)]

    def count_at(self, kept: FrozenSet[str]) -> None:
        """Count positions at the deepest level holding an index in ``kept``."""
        levels = [level for level, name in enumerate(self.names) if name in kept]
        self.depth = max(levels) if levels else -1

    def groups(self, key: FrozenSet[str], extents: Dict[str, int]) -> Optional[_Groups]:
        """Positions at ``self.depth`` grouped by the coordinates at ``key`` (indices of
        this factor, all held at or above that level), ``extents`` the extent each key
        index is linearized with: the largest any factor gives it, so that every
        factor's coordinates map to distinct keys.  ``None`` for a storage whose parents
        are not recorded (a coordinate level below a dense, compressed or singleton one,
        ``storage._records_its_parent``)."""

        if key in self._groups:
            return self._groups[key]
        total = level_positions(self.kinds, self.arrays, self.extents, self.depth)
        found: Optional[_Groups]
        if not key:
            found = _Uniform(total, total, domain=1, weight=total)
        else:
            first = {name: self.names.index(name) for name in key}
            if all(kind == LevelType.DENSE for kind in self.kinds[: self.depth + 1]):
                own = 1
                for level in first.values():
                    own *= self.extents[level]
                found = _Uniform(total, total // own, domain=own, weight=total // own)
            else:
                found = self._explicit_groups(total, first, extents)
        self._groups[key] = found
        return found

    def _explicit_groups(
        self, total: int, first: Dict[str, int], extents: Dict[str, int]
    ) -> Optional[_Groups]:
        strides: Dict[int, int] = {}
        domain = 1
        for name in sorted(first, reverse=True):
            strides[first[name]] = domain
            domain *= extents[name]
        deepest = max(first.values())
        keys = level_keys(self.kinds, self.arrays, self.extents, strides, deepest)
        weights = positions_beneath(
            self.kinds, self.arrays, self.extents, deepest, self.depth
        )
        if keys is None or weights is None:
            return None
        if domain <= max(_VECTOR_CELLS, 2 * keys.numel()):
            vector = torch.zeros(domain, dtype=torch.int64).index_add_(0, keys, weights)
            largest = int(vector.max()) if domain else 0
            return _Weighted(total, largest, domain=domain, weights=vector)
        distinct, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        summed = torch.zeros(distinct.numel(), dtype=torch.int64)
        summed.index_add_(0, inverse, weights)
        largest = int(summed.max()) if summed.numel() else 0
        return _Weighted(total, largest, domain=domain, weights=summed, keys=distinct)


def _term_join(
    factors: List[_Factor], result: FrozenSet[str], cells: Dict[str, int]
) -> Optional[Tuple[int, bool]]:
    """J for one product term: (the bound, whether it saturated)."""

    extents: Dict[str, int] = {}
    for factor in factors:
        for name in factor.indices:
            extents[name] = max(extents.get(name, 0), factor.extent_of(name))
    for position, factor in enumerate(factors):
        others: Set[str] = set()
        for other, peer in enumerate(factors):
            if other != position:
                others |= peer.indices
        factor.count_at(factor.indices & (result | others))
    uncarried = 1
    for name in result:
        if not any(name in factor.indices for factor in factors):
            uncarried *= cells[name]
    if not factors:
        # A term with no factor (a constant) writes every cell of its indices.
        return uncarried, False
    if len(factors) == 1:
        only = factors[0]
        total = level_positions(only.kinds, only.arrays, only.extents, only.depth)
        return total * uncarried, False
    found: List[Tuple[int, bool]] = []
    for a, b in combinations(range(len(factors)), 2):
        shared = factors[a].indices & factors[b].indices
        left = factors[a].groups(shared, extents)
        right = factors[b].groups(shared, extents)
        if left is None or right is None:
            return None
        joined, saturated = _pair_join(left, right)
        seen = factors[a].indices | factors[b].indices
        rest = [k for k in range(len(factors)) if k not in (a, b)]
        while rest:
            # The degree bound: each joined tuple extends to at most the largest group
            # of the next factor on the indices it shares with those joined so far.
            degrees = []
            for k in rest:
                groups = factors[k].groups(factors[k].indices & seen, extents)
                if groups is None:
                    return None
                degrees.append((groups.largest, k))
            degree, chosen = min(degrees)
            joined *= degree
            seen |= factors[chosen].indices
            rest.remove(chosen)
        found.append((joined, saturated))
    # Two factors or more: at least one pair.  A saturated pair's count is a lower bound
    # past every limit, so the smallest is an upper bound only when it is exact.
    joined, saturated = min(found, key=lambda pair: (pair[1], pair[0]))
    return joined * uncarried, saturated


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
    #: per operand position, the index name of each of its logical modes, when the caller
    #: gives them (einsum); without them J is never read
    operand_names: Tuple[Tuple[str, ...], ...] = ()

    def levels(
        self, stored: Sequence[int], join: Optional[int] = None
    ) -> List[LevelBound]:
        """Every bounded level, from the operands' stored counts (by operand position)
        and, when given, J."""

        extents = dict(zip(self.names, self.extents))
        bounds: List[LevelBound] = []
        for level, cells, limit in self.limits:
            if self.kinds[level] is LevelType.DENSE and bounds:
                # A dense level below a sparse one: its parent's entries times its extent.
                above = bounds[-1].bound * self.extents[level]
                bounds.append(LevelBound(level, self.kinds[level], cells, above, limit))
                continue
            prefix = frozenset(self.names[: level + 1])
            total = sum(
                _term_bound(
                    [(indices, stored[k]) for k, indices in term],
                    prefix,
                    extents,
                )
                for term in self.terms
            )
            bounds.append(
                LevelBound(level, self.kinds[level], cells, total, limit, join)
            )
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
                    deepest.join_bound,
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
            self.operand_names,
        )

    def _first_past_limit(self, bounds: List[LevelBound]) -> Optional[LevelBound]:
        for bound in bounds:
            if bound.bound > bound.limit and bound.level not in self.checked:
                return bound
        return None

    def join_bound(self, operands: Sequence[Any]) -> Optional[Tuple[int, bool]]:
        """J from the operands' coordinates, and whether it saturated: ``None`` when the
        caller gave no index names, or an operand is not stored the way this reads
        (no storage, a rank the names do not match, or a coordinate level below a dense,
        compressed or singleton one)."""

        if not self.operand_names or len(operands) != len(self.operand_names):
            return None
        result = frozenset(self.names)
        cells = dict(zip(self.names, self.extents))
        total, saturated = 0, False
        for term in self.terms:
            factors = []
            for k, _ in term:
                storage = getattr(operands[k], "storage", None)
                layout = getattr(storage, "_layout", None)
                if layout is None or len(layout.format.level_types) != len(
                    self.operand_names[k]
                ):
                    return None
                factors.append(_Factor(storage, self.operand_names[k]))
            found = _term_join(factors, result, cells)
            if found is None:
                return None
            total += found[0]
            saturated = saturated or found[1]
        return total, saturated

    def refuse_past_limit(
        self, stored: Sequence[int], operands: Optional[Sequence[Any]] = None
    ) -> None:
        """Raise IndexCapacityError (index_position_exceeds_int32) naming the first level
        assembled one entry at a time whose bound passes its limit.  With ``operands``
        (and the index names, :func:`entry_bound`), J is read before refusing, and only
        then."""

        failing = self._first_past_limit(self.levels(stored))
        if failing is None:
            return
        join = None if operands is None else _cached_join(self, operands)
        if join is not None and not join[1]:
            # An exact J bounds the levels; a saturated one is only a lower bound, past
            # every limit, and the levels keep their bounds from the shapes and counts.
            failing = self._first_past_limit(self.levels(stored, join[0]))
            if failing is None:
                return
        counts = "; ".join(
            " * ".join(f"operand {k} {stored[k]}" for k, _ in term)
            for term in self.terms
        )
        if failing.kind is LevelType.DENSE:
            why = (
                f"at most {failing.prefix_cells} logical cells, and the level above "
                f"may hold {failing.stored_bound // self.extents[failing.level]} "
                f"entries, each with {self.extents[failing.level]} here"
            )
        else:
            why = (
                f"at most {failing.prefix_cells} logical cells and at most "
                f"{failing.stored_bound} from the operands' stored entries ({counts})"
            )
            if join is not None and join[1]:
                why += (
                    ", and a count of the tuples in the join of the operands' stored "
                    f"coordinates on the indices they share stopped at {join[0]}, past "
                    "the limit"
                )
            elif join is not None:
                # Exact for two factors; with more, the degree bound over their join.
                why += (
                    f", and at most {join[0]} tuples in the join of the operands' "
                    "stored coordinates on the indices they share"
                )
        raise IndexCapacityError(
            f"{self.what}: result level {failing.level} ({failing.kind.name.lower()}) "
            f"may hold {failing.bound} entries: {why}.  A 32-bit position addresses at "
            f"most {failing.limit}, and the kernel assembles this level one entry at a "
            "time without checking its total, so the result is refused before the "
            "kernel runs"
        )


#: J per operand structure: (the bound, the operands' storages' ids) -> (weak references
#: to those storages, their index stamps, J).  An entry is used only while every
#: reference still names the same live storage and its stamp is unchanged, and dropped
#: when one of them dies.
_JOIN_CACHE: Dict[
    Tuple[Any, ...],
    Tuple[Tuple["weakref.ref[Any]", ...], Tuple[Any, ...], Optional[Tuple[int, bool]]],
] = {}
_JOIN_CACHE_ENTRIES = 64


def _cached_join(
    bound: EntryBound, operands: Sequence[Any]
) -> Optional[Tuple[int, bool]]:
    """``bound.join_bound(operands)``, counted once per operand structure."""

    found = [getattr(operand, "storage", None) for operand in operands]
    storages = [storage for storage in found if storage is not None]
    if not bound.operand_names or len(storages) != len(found):
        return None
    key = (
        bound.names,
        bound.extents,
        bound.terms,
        bound.operand_names,
        tuple(id(storage) for storage in storages),
    )
    stamps = tuple(structure_stamp(storage) for storage in storages)
    held = _JOIN_CACHE.get(key)
    if held is not None:
        references, held_stamps, value = held
        if held_stamps == stamps and all(
            reference() is storage for reference, storage in zip(references, storages)
        ):
            return value
    value = bound.join_bound(operands)
    while len(_JOIN_CACHE) >= _JOIN_CACHE_ENTRIES:
        _JOIN_CACHE.pop(next(iter(_JOIN_CACHE)))

    def forget(_reference: Any, key: Tuple[Any, ...] = key) -> None:
        _JOIN_CACHE.pop(key, None)

    references = tuple(weakref.ref(storage, forget) for storage in storages)
    _JOIN_CACHE[key] = (references, stamps, value)
    return value


def entry_bound(
    what: str,
    names: Sequence[str],
    extents: Sequence[int],
    kinds: Sequence[LevelType],
    terms: Sequence[Sequence[Tuple[int, Iterable[str]]]],
    operand_names: Optional[Sequence[Sequence[str]]] = None,
) -> Optional[EntryBound]:
    """The bound of a result whose shape can pass the limit at some sparse level, or
    ``None`` when no level's logical cells can: then no stored count can either.

    ``names``, ``extents`` and ``kinds`` describe the result's levels in storage order;
    ``terms`` the expression's sum of products, each factor as (operand position, the
    index names it carries); ``operand_names``, per operand position, the index name of
    each of its logical modes, which lets the bound read J."""

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
        tuple(tuple(names_k) for names_k in operand_names or ()),
    )
