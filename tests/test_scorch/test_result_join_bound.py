"""A sparse result bounded by the join of its operands' coordinates (result_bound, J).

Patch 0056 bounds each sparse result level from the shapes and the operands' stored
counts.  That bound never sees an index the factors contract, so for ``C(i,k) =
A(i,j) B(j,k)`` it stays at ``m n``: 16 of 34 SpGEMM cells of the SparseWS grid (results
of at most 1.1 * 10**8 entries) were refused before any kernel ran.  Where the counts pass
the limit at a level the kernel does not check, the bound now reads the operands'
coordinates: the join of the factors on the indices they share holds every stored
coordinate of the product, and for two factors its size is exactly the multiply count.

1. The storage walks against the prefixes each level stores, for every level kind and
   several mode orders: the walk up (``_walk_up``) as ``level_keys`` and
   ``stored_coordinates`` read it, from every level to every level above, and the walk
   down (``positions_beneath``).  The one parent rule they share: a coordinate level below
   a dense, compressed or singleton level records no parent, so the walks refuse it (with
   a code) and never guess one.  Below a singleton level that refuses nothing a storage
   can hold: every constructor that takes level arrays, and every entry that builds a
   result in a given format, refuses such a hierarchy with a code.
2. J against brute force over the operands' stored entries: exact for two factors whose
   counted indices are a prefix of their storage order, an upper bound of the product's
   stored coordinates always, over formats, mode orders and term shapes; ``None`` where a
   storage records no parents.  The count is exact below ``2**42`` and a lower bound at
   or past it, and only an exact count bounds a level.
3. A dense level below a sparse one is bounded by its parent's entries times its extent.
4. einsum at the real limit with few stored entries: SpGEMM the counts refused is
   computed (float64, the same coordinates as torch's product and every value within
   2 gamma_n |A||B| of it); a join past the limit is still refused before the product's
   kernel is built; a warm call counts once per index structure and again after an
   in-place write to an index array; a call the counts accept never reads coordinates.
   The new compiler's entry (execute_cin_via_loopir) refuses and computes the same
   SpGEMM programs through its automatic schedule, with the same code.
"""

from __future__ import annotations

import itertools
import os
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import pytest
import torch

import scorch
from scorch import ops, result_bound, stensor
from scorch.compiler.cin import (
    BinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.compiler.scheduler import Schedule
from scorch.exceptions import IndexCapacityError, ScorchAssertionError, defect_code
from scorch.format import LevelType, TensorFormat
from scorch.layout import TensorLayout
from scorch.result_bound import entry_bound
from scorch.stensor import STensor
from scorch.storage import (
    SparseStorage,
    TensorIndex,
    TensorStorage,
    _parents_recorded,
    _walk_up,
    _walked_coordinates,
    level_keys,
    positions_beneath,
    stored_coordinates,
    structure_of,
)

CODE = "index_position_exceeds_int32"
D, S, COO, SINGLETON = (
    LevelType.DENSE,
    LevelType.COMPRESSED,
    LevelType.COORDINATE,
    LevelType.SINGLETON,
)
KIND = {"d": D, "s": S, "o": COO}
U = 2.0**-53


@pytest.fixture
def fresh_caches(monkeypatch):
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_kernel_level_totals", {})
    monkeypatch.setattr(stensor, "_RESULT_BOUNDS", {})
    monkeypatch.setattr(stensor, "_GENERATED_KERNELS", {})
    monkeypatch.setattr(result_bound, "_JOIN_CACHE", {})


def _kernel_sources() -> List[Path]:
    return sorted(Path(os.environ["TORCH_EXTENSIONS_DIR"]).rglob("main.cpp"))


def _random_dense(shape: Sequence[int], density: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(*shape, generator=generator) < density
    values = torch.rand(*shape, generator=generator, dtype=torch.float64) + 0.5
    return torch.where(mask, values, torch.zeros_like(values))


Built = Tuple[STensor, List[Tuple[int, ...]], List[List[Tuple[int, ...]]]]


def _build(
    dense: torch.Tensor,
    fmt: str,
    order: Sequence[int],
    index_dtype: torch.dtype = torch.int32,
) -> Built:
    """``dense`` stored in ``fmt`` (letters d, s, o), storage level l holding logical mode
    ``order[l]``, built from its own arrays (no conversion kernel) of ``index_dtype``.
    Returns the tensor, the logical coordinate of every stored value (a dense level stores
    every cell under a stored parent, zeros included), and per level the physical prefixes
    it stores."""

    physical = dense.permute(*order).contiguous()
    shape = tuple(physical.shape)
    nonzero = sorted(map(tuple, torch.nonzero(physical).tolist()))
    levels: List[List[Tuple[int, ...]]] = []
    if set(fmt) == {"o"}:
        mode_indices = [
            [torch.tensor([e[level] for e in nonzero], dtype=index_dtype)]
            for level in range(len(shape))
        ]
        levels = [[e[: level + 1] for e in nonzero] for level in range(len(shape))]
        prefixes = nonzero
    else:
        prefixes = [()]
        mode_indices = []
        for level, letter in enumerate(fmt):
            grown: List[Tuple[int, ...]] = []
            if letter == "d":
                grown = [p + (c,) for p in prefixes for c in range(shape[level])]
                mode_indices.append([])
            else:
                pos, crd = [0], []
                for p in prefixes:
                    for c in sorted({e[level] for e in nonzero if e[:level] == p}):
                        grown.append(p + (c,))
                        crd.append(c)
                    pos.append(len(crd))
                mode_indices.append(
                    [
                        torch.tensor(pos, dtype=index_dtype),
                        torch.tensor(crd, dtype=index_dtype),
                    ]
                )
            prefixes = grown
            levels.append(prefixes)
    values = torch.tensor(
        [physical[p].item() for p in prefixes], dtype=torch.float64
    ).reshape(-1)
    tensor = STensor.from_components(
        shape,
        fmt,
        mode_indices,
        values,
        mode_order=list(order),
        index_dtype=index_dtype,
    )
    logical = []
    for p in prefixes:
        coordinate = [0] * len(order)
        for level, mode in enumerate(order):
            coordinate[mode] = p[level]
        logical.append(tuple(coordinate))
    return tensor, logical, levels


def _positions_below(
    fmt: str, stored_levels: List[List[Tuple[int, ...]]], level: int, depth: int
) -> List[int]:
    """For every position of storage level ``level`` of a storage ``_build`` made, how
    many positions of level ``depth`` descend from it.  In a COO storage every level
    holds one position per stored value, and a position's parent is the same position
    one level up, so a prefix stored twice is two positions; elsewhere each stored
    prefix is one position, whose parent holds the prefix one level shorter."""

    index = [{prefix: k for k, prefix in enumerate(held)} for held in stored_levels]
    counts = [0] * len(stored_levels[level])
    for position in range(len(stored_levels[depth])):
        owner = position
        for at in range(depth, level, -1):
            if set(fmt) != {"o"}:
                owner = index[at - 1][stored_levels[at][owner][:-1]]
        counts[owner] += 1
    return counts


# --------------------------------------------------------------------------------------
# 1. The storage walks
# --------------------------------------------------------------------------------------

_WALK_CASES = [
    ("ds", (0, 1)),
    ("ss", (1, 0)),
    ("sd", (0, 1)),
    ("dd", (1, 0)),
    ("oo", (0, 1)),
    ("dss", (2, 0, 1)),
    ("sds", (0, 1, 2)),
    ("sss", (1, 2, 0)),
    ("ooo", (0, 2, 1)),
]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("fmt, order", _WALK_CASES)
def test_the_walks_read_every_prefix_and_its_positions(fmt, order, index_dtype):
    shape = (5, 7, 6)[: len(fmt)]
    dense = _random_dense(shape, 0.3, seed=len(fmt) * 10 + order[0])
    tensor, _, stored_levels = _build(dense, fmt, order, index_dtype)
    kinds, arrays, extents, _ = structure_of(tensor.storage)
    for level in range(len(fmt)):
        expected = stored_levels[level]
        strides = {}
        stride = 1
        for chosen in reversed(range(level + 1)):
            strides[chosen] = stride
            stride *= extents[chosen]
        keys = level_keys(kinds, arrays, extents, strides, level)
        assert keys is not None
        assert keys.tolist() == [
            sum(c * strides[at] for at, c in enumerate(p)) for p in expected
        ]
        # One key level only, each level above in turn: the walk stops at that level.
        for top in range(level + 1):
            alone = level_keys(kinds, arrays, extents, {top: 1}, level)
            assert alone is not None and alone.tolist() == [p[top] for p in expected]
        for depth in range(level, len(fmt)):
            beneath = positions_beneath(kinds, arrays, extents, level, depth)
            assert beneath is not None
            assert beneath.tolist() == _positions_below(
                fmt, stored_levels, level, depth
            )
    root = positions_beneath(kinds, arrays, extents, -1, len(fmt) - 1)
    assert root is not None and root.tolist() == [tensor.values.numel()]
    # Every stored value's full coordinate, the walk stored_coordinates reads: new
    # tensors, never the storage's own index arrays (int64 ones included).
    storage = tensor.storage
    count = tensor.values.numel()
    coordinates = stored_coordinates(storage._layout, arrays, count)
    assert [c.tolist() for c in coordinates] == [
        [p[level] for p in stored_levels[-1]] for level in range(len(fmt))
    ]
    if set(fmt) != {"o"}:
        held = {array.data_ptr() for level_arrays in arrays for array in level_arrays}
        assert not any(c.data_ptr() in held for c in coordinates if c.numel())


_TWO_CRD = (
    torch.tensor([0, 2], dtype=torch.int32),
    torch.tensor([1, 3], dtype=torch.int32),
)

# (kinds, arrays, extents, whether a walk from the last level up to level 0 finds every
# parent it needs): one hierarchy per kind of level a coordinate level can sit below, and
# two that record every parent, the counts of every array consistent with
# _validate_index_storage.
_PARENT_CASES = [
    pytest.param(
        (S, COO),
        (_TWO_CRD, (torch.tensor([0, 4], dtype=torch.int32),)),
        (5, 5),
        False,
        id="coordinate-below-compressed",
    ),
    pytest.param(
        (D, COO),
        ((), (torch.tensor([0, 4], dtype=torch.int32),)),
        (2, 5),
        False,
        id="coordinate-below-dense",
    ),
    pytest.param(
        (S, SINGLETON, COO),
        (
            _TWO_CRD,
            (torch.tensor([0, 4], dtype=torch.int32),),
            (torch.tensor([2, 1], dtype=torch.int32),),
        ),
        (5, 5, 5),
        False,
        id="coordinate-below-singleton",
    ),
    pytest.param(
        (COO, SINGLETON, COO),
        tuple((torch.tensor(c, dtype=torch.int32),) for c in ([1, 3], [0, 4], [2, 1])),
        (5, 5, 5),
        False,
        id="coordinate-below-singleton-counts-agree",
    ),
    pytest.param(
        (S, SINGLETON),
        (_TWO_CRD, (torch.tensor([0, 4], dtype=torch.int32),)),
        (5, 5),
        True,
        id="singleton-below-compressed",
    ),
    pytest.param(
        (COO, COO, S),
        (
            (torch.tensor([1, 3], dtype=torch.int32),),
            (torch.tensor([0, 4], dtype=torch.int32),),
            (
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([2, 1], dtype=torch.int32),
            ),
        ),
        (5, 5, 5),
        True,
        id="compressed-below-coordinate-run",
    ),
]


@pytest.mark.parametrize("kinds, arrays, extents, recorded", _PARENT_CASES)
def test_one_parent_rule_for_every_walk(kinds, arrays, extents, recorded):
    """A coordinate level stores one coordinate per stored value; only in the leading run
    of coordinate levels does that say which parent each belongs to.  Below a dense,
    compressed or singleton level the walks refuse it -- level_keys and positions_beneath
    return None, the walk itself and _walked_coordinates raise with a code -- even where
    the counts happen to agree, and never read a position of the level above."""

    last = len(kinds) - 1
    count = arrays[last][-1].numel()
    strides = {level: 5 ** (last - level) for level in range(len(kinds))}
    assert _parents_recorded(kinds, last, 0) is recorded
    keys = level_keys(kinds, arrays, extents, strides, last)
    beneath = positions_beneath(kinds, arrays, extents, -1, last)
    if recorded:
        coordinates = _walked_coordinates(kinds, arrays, extents, count)
        assert keys is not None and beneath is not None
        assert keys.tolist() == [
            sum(int(coordinates[level][p]) * strides[level] for level in strides)
            for p in range(count)
        ]
        assert beneath.tolist() == [count]
        return
    assert keys is None and beneath is None
    with pytest.raises(ScorchAssertionError) as raised:
        list(_walk_up(kinds, arrays, extents, last, 0))
    assert defect_code(raised.value) == "storage_parent_unrecorded"
    with pytest.raises(ScorchAssertionError) as raised:
        _walked_coordinates(kinds, arrays, extents, count)
    assert defect_code(raised.value) == "storage_parent_unrecorded"
    # A walk that stops at the deepest level needs no parent and reads it as stored.
    own = level_keys(kinds, arrays, extents, {last: 1}, last)
    assert own is not None and own.tolist() == arrays[last][-1].tolist()


def _coordinate_level_below_singleton(kinds) -> bool:
    """Whether a coordinate level follows the leading coordinate run and every such level
    sits directly below a singleton level or another coordinate level, never a dense or
    compressed one: the hierarchies on which reading a coordinate level's parents through
    a singleton level would differ from the one rule above."""

    leading = next((n for n, kind in enumerate(kinds) if kind != COO), len(kinds))
    later = [n for n in range(leading, len(kinds)) if kinds[n] == COO]
    return bool(later) and all(kinds[n - 1] in (SINGLETON, COO) for n in later)


_TEXT = {D: "d", S: "s", COO: "o", SINGLETON: "singleton"}
_EXTENT = 2
_BELOW_SINGLETON = [
    kinds
    for rank in (2, 3)
    for kinds in itertools.product((D, S, COO, SINGLETON), repeat=rank)
    if _coordinate_level_below_singleton(kinds)
]


def _text(kinds) -> str:
    return ",".join(_TEXT[kind] for kind in kinds)


def _consistent_arrays(kinds):
    """Arrays for ``kinds`` that pass _validate_index_storage's count checks at the
    smallest stored count that allows them (each compressed level one child per parent, a
    singleton level one coordinate per parent position, a coordinate level one per stored
    value, every coordinate 0), or ``None`` when no stored count does."""

    for stored in range(1, 9):
        parents, arrays = 1, []
        for kind in kinds:
            if kind == D:
                arrays.append(())
                parents *= _EXTENT
            elif kind == S:
                arrays.append(
                    (
                        torch.arange(parents + 1, dtype=torch.int32),
                        torch.zeros(parents, dtype=torch.int32),
                    )
                )
            elif kind == SINGLETON:
                arrays.append((torch.zeros(parents, dtype=torch.int32),))
            else:
                arrays.append((torch.zeros(stored, dtype=torch.int32),))
                parents = stored
        if parents == stored:
            return tuple(arrays), torch.ones(stored, dtype=torch.float64)
    return None


def _refusal(make):
    """The code ``make`` refuses with, or ``None`` when it builds."""

    try:
        make()
    except Exception as error:  # every refusal is read by its code; uncoded ones fail
        return defect_code(error) or f"uncoded {type(error).__name__}: {error}"
    return None


def test_the_hierarchies_below_a_singleton():
    """The frame of the two tests below: one rank-2 and eight rank-3 hierarchies, all but
    one with arrays the count checks accept (below a coordinate level, a dense level
    multiplies the positions, so no stored count is consistent)."""

    assert [_text(kinds) for kinds in _BELOW_SINGLETON if len(kinds) == 2] == [
        "singleton,o"
    ]
    assert len(_BELOW_SINGLETON) == 9
    assert [k for k in _BELOW_SINGLETON if _consistent_arrays(k) is None] == [
        (SINGLETON, COO, D)
    ]


_ARRAY_CONSTRUCTORS = (
    "storage",
    "storage_index",
    "tensor_storage",
    "trusted",
    "from_components",
    "stensor_index",
)


@pytest.mark.parametrize("constructor", _ARRAY_CONSTRUCTORS)
@pytest.mark.parametrize(
    "kinds, expected",
    [
        pytest.param(kinds, "singleton_level_unsupported", id=_text(kinds))
        for kinds in _BELOW_SINGLETON
        if _consistent_arrays(kinds) is not None
    ]
    + [
        pytest.param((D, S), None, id="control-d,s"),
        pytest.param((S, S, S), None, id="control-s,s,s"),
    ],
)
def test_every_constructor_refuses_a_coordinate_level_below_a_singleton(
    kinds, expected, constructor
):
    """No constructor that takes level arrays builds storage holding a coordinate level
    below a singleton level, so the one parent rule refuses nothing a storage can hold.
    Each refuses with a code, from arrays the count checks accept; the two controls build
    through the same calls."""

    arrays, values = _consistent_arrays(kinds)
    text, rank = _text(kinds), len(kinds)
    shape = (_EXTENT,) * rank

    def layout():
        return TensorLayout(shape, shape, TensorFormat(text), list(range(rank)))

    make = {
        "storage": lambda: SparseStorage(layout(), values, mode_indices=arrays),
        "storage_index": lambda: SparseStorage(
            layout(), values, index=TensorIndex(text, arrays)
        ),
        "tensor_storage": lambda: TensorStorage(
            index=TensorIndex(text, arrays), value=values, shape=shape
        ),
        "trusted": lambda: SparseStorage(layout(), values, arrays, _trusted_index=True),
        "from_components": lambda: STensor.from_components(shape, text, arrays, values),
        "stensor_index": lambda: STensor(
            shape=shape, index=TensorIndex(text, arrays), value=values
        ),
    }[constructor]
    assert _refusal(make) == expected


def _all_compressed(rank: int) -> STensor:
    arrays, values = _consistent_arrays((S,) * rank)
    return STensor.from_components((_EXTENT,) * rank, "s" * rank, arrays, values)


@pytest.mark.parametrize(
    "kinds, entry, expected",
    [
        pytest.param(kinds, entry, expected, id=f"{_text(kinds)}-{entry}")
        for kinds in _BELOW_SINGLETON
        for entry, expected in (
            ("to_sparse", "unsupported_mixed_coordinate_hierarchy"),
            ("einsum", "singleton_level_unsupported"),
            ("matmul", "singleton_level_unsupported"),
            ("change_mode_order", "unsupported_mixed_coordinate_hierarchy"),
            ("change_mode_order_reversed", "unsupported_mixed_coordinate_hierarchy"),
        )
        if entry != "matmul" or len(kinds) == 2
    ],
)
def test_every_kernel_entry_refuses_a_coordinate_level_below_a_singleton(
    kinds, entry, expected
):
    """The entries that build a result in a format they are given refuse these
    hierarchies with a code.  The operands are made from arrays, so no kernel is built
    for them."""

    text, rank = _text(kinds), len(kinds)
    dense = torch.zeros((_EXTENT,) * rank, dtype=torch.float64)
    dense.view(-1)[::3] = 1.0
    product = "ij,ij->ij" if rank == 2 else "ijk,ijk->ijk"
    make = {
        "to_sparse": lambda: STensor.from_torch(dense).to_sparse(text),
        "einsum": lambda: scorch.einsum(
            product, _all_compressed(rank), _all_compressed(rank), format=text
        ),
        "matmul": lambda: scorch.matmul(
            _all_compressed(2), _all_compressed(2), format=text
        ),
        "change_mode_order": lambda: _all_compressed(rank).change_mode_order(
            list(range(rank)), _target_format=TensorFormat(text)
        ),
        "change_mode_order_reversed": lambda: _all_compressed(rank).change_mode_order(
            list(reversed(range(rank))), _target_format=TensorFormat(text)
        ),
    }[entry]
    assert _refusal(make) == expected


# --------------------------------------------------------------------------------------
# 2. J against brute force
# --------------------------------------------------------------------------------------


def _pair_join(stored_a, names_a, stored_b, names_b, result, project):
    """sum over the shared indices' values of (A's tuples) * (B's tuples), over the
    stored entries, each factor first projected on the indices the result or the other
    factor uses when ``project``."""

    shared = sorted(set(names_a) & set(names_b))
    counts = []
    for stored, names, other in (
        (stored_a, names_a, names_b),
        (stored_b, names_b, names_a),
    ):
        kept = [n for n in names if n in result or n in other]
        rows = [tuple(e[names.index(n)] for n in kept) for e in stored]
        if project:
            rows = sorted(set(rows))
        counts.append(Counter(tuple(r[kept.index(n)] for n in shared) for r in rows))
    return sum(counts[0][key] * counts[1][key] for key in counts[0])


def _product_coordinates(expression: str, stored, shapes) -> int:
    """The product's stored coordinates: the einsum of the stored patterns."""

    patterns = []
    for entries, shape in zip(stored, shapes):
        pattern = torch.zeros(shape, dtype=torch.float64)
        for e in entries:
            pattern[e] = 1.0
        patterns.append(pattern)
    return int(torch.count_nonzero(torch.einsum(expression, *patterns)))


def _join(expression: str, operands: Sequence[object]):
    inputs, result = expression.split("->")
    names = inputs.split(",")
    original = result_bound.MAX_INDEX_POSITION
    result_bound.MAX_INDEX_POSITION = 0  # every sparse level is bounded
    try:
        bound = entry_bound(
            "t",
            list(result),
            [10] * len(result),
            [S] * len(result),
            [[(k, n) for k, n in enumerate(names)]],
            operand_names=names,
        )
    finally:
        result_bound.MAX_INDEX_POSITION = original
    assert bound is not None
    return bound.join_bound(operands)


_FORMATS_2D = ["ds", "ss", "dd", "sd", "oo"]
_ORDERS_2D = [(0, 1), (1, 0)]


@pytest.mark.parametrize(
    "fa, oa, fb, ob",
    list(itertools.product(_FORMATS_2D, _ORDERS_2D, _FORMATS_2D, _ORDERS_2D)),
)
def test_two_factor_spgemm_join_is_the_multiply_count(fa, oa, fb, ob):
    a, stored_a, _ = _build(_random_dense((9, 8), 0.3, seed=1), fa, oa)
    b, stored_b, _ = _build(_random_dense((8, 10), 0.25, seed=2), fb, ob)
    found = _join("ij,jk->ik", [a, b])
    assert found is not None
    value, saturated = found
    assert not saturated
    # Every index of both factors is kept (i and k by the result, j shared), so every
    # counted position is a stored entry: J is the multiply count.
    columns = Counter(e[1] for e in stored_a)
    rows = Counter(e[0] for e in stored_b)
    assert value == sum(columns[j] * rows[j] for j in columns)
    assert value >= _product_coordinates(
        "ij,jk->ik", [stored_a, stored_b], [(9, 8), (8, 10)]
    )


@pytest.mark.parametrize(
    "expression, shapes, formats, exact",
    [
        ("ij,ij->ij", [(9, 8), (9, 8)], [("ds", (0, 1)), ("ss", (1, 0))], True),
        ("i,j->ij", [(9,), (8,)], [("s", (0,)), ("s", (0,))], True),
        ("ij,j->i", [(9, 8), (8,)], [("ds", (0, 1)), ("s", (0,))], True),
        ("ij,j->i", [(9, 8), (8,)], [("dd", (1, 0)), ("d", (0,))], True),
        ("ij,kj->ik", [(9, 8), (7, 8)], [("ss", (0, 1)), ("ds", (1, 0))], True),
        # l summed in A alone: counted at (i, j), a prefix of i, j, l.
        ("ijl,jk->ik", [(6, 5, 4), (5, 7)], [("dss", (0, 1, 2)), ("ds", (0, 1))], True),
        # Stored i, l, j: counted at (i, l, j), an upper bound of the (i, j) tuples.
        (
            "ijl,jk->ik",
            [(6, 5, 4), (5, 7)],
            [("sss", (0, 2, 1)), ("ss", (1, 0))],
            False,
        ),
        (
            "ijl,jk->ik",
            [(6, 5, 4), (5, 7)],
            [("ooo", (2, 0, 1)), ("oo", (0, 1))],
            False,
        ),
    ],
)
def test_two_factor_joins_match_brute_force(expression, shapes, formats, exact):
    built = [
        _build(_random_dense(shape, 0.3, seed=7 + k), f, o)
        for k, (shape, (f, o)) in enumerate(zip(shapes, formats))
    ]
    operands = [b[0] for b in built]
    stored = [b[1] for b in built]
    inputs, result = expression.split("->")
    names = inputs.split(",")
    found = _join(expression, operands)
    assert found is not None and not found[1]
    projected = _pair_join(stored[0], names[0], stored[1], names[1], result, True)
    entries = _pair_join(stored[0], names[0], stored[1], names[1], result, False)
    assert projected <= found[0] <= entries
    assert found[0] >= _product_coordinates(expression, stored, shapes)
    if exact:
        assert found[0] == projected


@pytest.mark.parametrize(
    "formats",
    [
        [("ds", (0, 1)), ("ss", (1, 0)), ("ds", (0, 1))],
        [("dd", (0, 1)), ("ds", (0, 1)), ("oo", (1, 0))],
        [("ss", (1, 0)), ("dd", (1, 0)), ("ss", (0, 1))],
    ],
)
def test_a_chain_of_three_bounds_the_product(formats):
    shapes = [(8, 7), (7, 9), (9, 6)]
    built = [
        _build(_random_dense(shape, 0.25, seed=20 + k), f, o)
        for k, (shape, (f, o)) in enumerate(zip(shapes, formats))
    ]
    stored = [b[1] for b in built]
    found = _join("ij,jk,kl->il", [b[0] for b in built])
    assert found is not None
    assert found[0] >= _product_coordinates("ij,jk,kl->il", stored, shapes)
    # The best pair joined exactly, times the third factor's largest group: never
    # above a pair's join times every entry of the third factor.
    pair_ab = _pair_join(stored[0], "ij", stored[1], "jk", "il", False)
    pair_bc = _pair_join(stored[1], "jk", stored[2], "kl", "il", False)
    assert found[0] <= min(pair_ab * len(stored[2]), pair_bc * len(stored[0]))


def test_a_factor_whose_parents_are_unrecorded_gives_no_join(monkeypatch):
    a, _, _ = _build(_random_dense((6, 6), 0.3, seed=30), "ds", (0, 1))
    b, _, _ = _build(_random_dense((6, 6), 0.3, seed=31), "ds", (0, 1))
    real = result_bound.structure_of

    def unrecorded(storage):
        kinds, arrays, extents, permutation = real(storage)
        if storage is not a.storage:
            return kinds, arrays, extents, permutation
        # A coordinate level under a compressed one: the walk from level 1 up to the
        # shared index i at level 0 has no parent to follow.
        compressed = (
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([1, 4], dtype=torch.int32),
        )
        return (S, COO), (compressed, (compressed[1],)), extents, permutation

    monkeypatch.setattr(result_bound, "structure_of", unrecorded)
    assert _join("ij,ik->jk", [a, b]) is None


def test_the_count_is_exact_below_its_saturation_and_a_lower_bound_past_it():
    dot = result_bound._saturating_dot
    # A weight past 2**31 meeting a zero adds nothing, and the rest is exact.
    left = torch.tensor([3, 0, 2**31 + 5, 7], dtype=torch.int64)
    right = torch.tensor([4, 9, 0, 2**30], dtype=torch.int64)
    assert dot(left, right) == (12 + 7 * 2**30, False)
    # Exact past 2**31 - 1: the outer product of two 46,341-entry vectors.
    side = torch.tensor([46_341], dtype=torch.int64)
    assert dot(side, side) == (46_341**2, False)
    # Exact across the int64 chunks it sums in.
    ones = torch.ones(result_bound._DOT_CHUNK + 1, dtype=torch.int64)
    assert dot(ones, ones) == (result_bound._DOT_CHUNK + 1, False)
    # A weight past 2**31 meeting a nonzero one: a lower bound, at least 2**31.
    total, saturated = dot(torch.tensor([2**31 + 5]), torch.tensor([1]))
    assert saturated and 2**31 <= total <= 2**31 + 5
    # A sum reaching 2**42 stops: a lower bound, at least 2**42.
    half = torch.full((2,), 2**21, dtype=torch.int64)
    total, saturated = dot(half, half)
    assert saturated and total >= result_bound._SATURATION


def test_a_term_without_factors_writes_every_cell():
    assert result_bound._term_join([], frozenset("ik"), {"i": 3, "k": 4}) == (12, False)


def test_only_an_exact_join_bounds_a_level(monkeypatch):
    # C(i,k) = A(i,j) B(j,k) into ss, 100 x 100, at a limit of 1000: with 50 stored
    # entries each, level 1's bound from the counts is 50 * 50 = 2,500.
    monkeypatch.setattr(result_bound, "MAX_INDEX_POSITION", 1000)
    bound = entry_bound(
        "t", "ik", (100, 100), (S, S), [[(0, "ij"), (1, "jk")]], ["ij", "jk"]
    )
    assert bound is not None
    operands = [object(), object()]

    def joined(value, saturated):
        monkeypatch.setattr(
            result_bound, "_cached_join", lambda _bound, _operands: (value, saturated)
        )

    joined(900, False)
    bound.refuse_past_limit([50, 50], operands)  # an exact join under the limit
    joined(1500, False)
    with pytest.raises(IndexCapacityError) as caught:
        bound.refuse_past_limit([50, 50], operands)
    assert "may hold 1500 entries" in str(caught.value)
    assert "at most 1500 tuples in the join" in str(caught.value)
    # A count that stopped is a lower bound: the level keeps its bound from the counts.
    joined(1001, True)
    with pytest.raises(IndexCapacityError) as caught:
        bound.refuse_past_limit([50, 50], operands)
    assert caught.value.code == CODE
    assert "may hold 2500 entries" in str(caught.value)
    assert "stopped at 1001, past the limit" in str(caught.value)


# --------------------------------------------------------------------------------------
# 3. A dense level below a sparse one
# --------------------------------------------------------------------------------------


def test_a_dense_level_below_a_sparse_one_counts_its_parents_fill(monkeypatch):
    # Result sd for C(i,k) = A(i,j) B(j,k): A's 20 rows store column 0 and B's row 0 one
    # column, so C has 20 nonzeros, stored as 20 rows of 100 cells.  The stored counts
    # give at most min(20 * 1, 20 * 100, 1 * 100) = 20 distinct coordinates; level 1
    # holds 2,000 positions, its parent's 20 entries times its extent.
    assert entry_bound("t", "ik", (100, 100), (S, D), [[(0, "ij"), (1, "jk")]]) is None
    monkeypatch.setattr(result_bound, "MAX_INDEX_POSITION", 1000)
    bound = entry_bound("t", "ik", (100, 100), (S, D), [[(0, "ij"), (1, "jk")]])
    assert bound is not None
    level0, level1 = bound.levels([20, 1])
    assert (level0.bound, level1.bound) == (20, 2000)
    with pytest.raises(IndexCapacityError) as caught:
        bound.refuse_past_limit([20, 1])
    assert "result level 1 (dense)" in str(caught.value)
    assert "the level above may hold 20 entries, each with 100 here" in str(
        caught.value
    )


# --------------------------------------------------------------------------------------
# 4. einsum at the real limit
# --------------------------------------------------------------------------------------

M = 100_000  # M * M passes 2**31 - 1
#: Stored entries per M x M operand where the counts must pass the limit: nnz(A) nnz(B)
#: = 2.5 * 10**9 bounds C(i,k) = A(i,j) B(j,k) at level 1 from the counts, while the
#: join on j holds about nnz(A) nnz(B) / M = 25,000 tuples.
STORED = 50_000


def _sparse_matrix(
    rows: int, cols: int, stored: int, seed: int
) -> Tuple[STensor, torch.Tensor]:
    """A ds matrix with up to ``stored`` random entries, and the same as a torch COO."""

    generator = torch.Generator().manual_seed(seed)
    flat = torch.unique(torch.randint(0, rows * cols, (stored,), generator=generator))
    r, c = flat // cols, flat % cols
    values = torch.rand(flat.numel(), generator=generator, dtype=torch.float64) + 0.5
    pos = torch.zeros(rows + 1, dtype=torch.int64)
    pos[1:] = torch.cumsum(torch.bincount(r, minlength=rows), 0)
    tensor = STensor.from_components(
        (rows, cols), "ds", [[], [pos.to(torch.int32), c.to(torch.int32)]], values
    )
    reference = torch.sparse_coo_tensor(
        torch.stack([r, c]), values, (rows, cols)
    ).coalesce()
    return tensor, reference


def _stored_coo(result: STensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The result's stored coordinates (logical, sorted) and values, read from its
    levels: no conversion, which could densify a 10**10-cell result."""

    kinds, arrays, extents, permutation = structure_of(result.storage)
    last = len(kinds) - 1
    strides = {last: 1, last - 1: extents[last]}
    keys = level_keys(kinds, arrays, extents, strides, last)
    assert keys is not None
    physical = torch.stack([keys // extents[last], keys % extents[last]])
    logical = torch.empty_like(physical)
    for level, mode in enumerate(permutation):
        logical[mode] = physical[level]
    order = torch.argsort(logical[0] * extents[last] + logical[1])
    return logical[:, order], result.values[order]


def _assert_product(result: STensor, a_ref: torch.Tensor, b_ref: torch.Tensor) -> None:
    """The result stores exactly the product's coordinates (zeros a dense level stores
    aside, which a ds result has none of here), every value within 2 gamma_n |A||B| of
    torch's float64 product, n that entry's multiply count."""

    def ones(t: torch.Tensor) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            t.indices(), torch.ones_like(t.values()), t.shape
        )

    expected = torch.sparse.mm(a_ref, b_ref).coalesce()
    magnitude = torch.sparse.mm(a_ref.abs(), b_ref.abs()).coalesce()
    count = torch.sparse.mm(ones(a_ref), ones(b_ref)).coalesce()
    indices, values = _stored_coo(result)
    assert torch.equal(indices, count.indices())
    assert torch.equal(expected.indices(), count.indices())
    n = count.values()
    gamma = n * U / (1 - n * U)
    error = (values - expected.values()).abs()
    assert bool((error <= 2 * gamma * magnitude.values()).all())


@pytest.mark.parametrize("result_format", ["ds", "ss", "oo"])
@pytest.mark.parametrize("loop_order", [None, ("j", "i", "k")])
def test_spgemm_the_counts_refused_is_computed(
    fresh_caches, monkeypatch, result_format, loop_order
):
    a, a_ref = _sparse_matrix(M, M, STORED, seed=40)
    b, b_ref = _sparse_matrix(M, M, STORED, seed=41)
    bound = entry_bound(
        "t",
        "ik",
        (M, M),
        [KIND[letter] for letter in result_format],
        [[(0, "ij"), (1, "jk")]],
    )
    assert bound is not None
    stored = [a.values.numel(), b.values.numel()]
    assert any(level.bound > level.limit for level in bound.levels(stored))
    read = []
    real = result_bound.EntryBound.join_bound

    def spy(self, operands):
        read.append(len(operands))
        return real(self, operands)

    monkeypatch.setattr(result_bound.EntryBound, "join_bound", spy)
    before = set(_kernel_sources())
    kwargs = {} if loop_order is None else {"schedule": Schedule(loop_order=loop_order)}
    result = scorch.einsum("ij,jk->ik", a, b, format=result_format, **kwargs)
    _assert_product(result, a_ref, b_ref)
    assert len(read) <= 1
    if not read:
        # Not read only where the kernel checks every failing level's total itself.
        built = [p.read_text() for p in _kernel_sources() if p not in before]
        assert any("scorch_check_level_total(" in text for text in built)


def test_a_join_past_the_limit_is_refused_before_the_product_is_built(
    fresh_caches, monkeypatch
):
    side = 46_341  # side * side = 2,147,488,281
    column = STensor.from_components(
        (side, 1),
        "ds",
        [
            [],
            [
                torch.arange(side + 1, dtype=torch.int32),
                torch.zeros(side, dtype=torch.int32),
            ],
        ],
        torch.rand(side, dtype=torch.float64) + 0.5,
    )
    row = STensor.from_components(
        (1, side),
        "ds",
        [
            [],
            [
                torch.tensor([0, side], dtype=torch.int32),
                torch.arange(side, dtype=torch.int32),
            ],
        ],
        torch.rand(side, dtype=torch.float64) + 0.5,
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the product's kernel was built")

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", refuse)
    for result_format in ("ss", "oo"):
        for loop_order in (None, ("j", "i", "k")):
            kwargs = (
                {}
                if loop_order is None
                else {"schedule": Schedule(loop_order=loop_order)}
            )
            with pytest.raises(IndexCapacityError) as caught:
                scorch.einsum("ij,jk->ik", column, row, format=result_format, **kwargs)
            assert caught.value.code == CODE
            assert f"{side * side} tuples in the join" in str(caught.value)


def test_a_warm_call_counts_once_per_index_structure(fresh_caches, monkeypatch):
    a, _ = _sparse_matrix(M, M, STORED, seed=50)
    b, _ = _sparse_matrix(M, M, STORED, seed=51)
    counted = []
    real = result_bound.EntryBound.join_bound

    def spy(self, operands):
        counted.append(1)
        return real(self, operands)

    monkeypatch.setattr(result_bound.EntryBound, "join_bound", spy)
    for _ in range(3):
        scorch.einsum("ij,jk->ik", a, b, format="ss")
    assert len(counted) == 1
    # An in-place write to an index array moves its version: counted again, once.
    a.storage._mode_indices[1][1].add_(0)
    for _ in range(2):
        scorch.einsum("ij,jk->ik", a, b, format="ss")
    assert len(counted) == 2


def _spgemm_cin(result_format: str, operand_format: str = "ds"):
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt=operand_format, dtype=torch.float64)
    b = TensorVar("B", fmt=operand_format, dtype=torch.float64)
    c = TensorVar("C", fmt=result_format, dtype=torch.float64)
    return ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    c[i, k], BinaryOp(Operation.MUL, a[i, j], b[j, k]), op=Operation.ADD
                ),
            ),
        ),
    )


def _automatic_options() -> CompileOptions:
    """The new compiler's automatic schedule.  It lowers SpGEMM into a sparse result
    only through its sparse-workspace schedule: unscheduled, or with a loop order
    requested, the program is refused at lowering with unsupported_program_shape, before
    any bound is read."""

    return CompileOptions.from_environment(requested_schedule=Schedule())


@pytest.mark.parametrize("result_format", ["ss", "oo"])
def test_the_new_compilers_entry_refuses_the_same_join(
    fresh_caches, monkeypatch, result_format
):
    side = 46_341
    column = STensor.from_components(
        (side, 1),
        "ds",
        [
            [],
            [
                torch.arange(side + 1, dtype=torch.int32),
                torch.zeros(side, dtype=torch.int32),
            ],
        ],
        torch.rand(side, dtype=torch.float64) + 0.5,
    )
    row = STensor.from_components(
        (1, side),
        "ds",
        [
            [],
            [
                torch.tensor([0, side], dtype=torch.int32),
                torch.arange(side, dtype=torch.int32),
            ],
        ],
        torch.rand(side, dtype=torch.float64) + 0.5,
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the product's kernel was built")

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", refuse)
    with pytest.raises(IndexCapacityError) as caught:
        execute_cin_via_loopir(
            _spgemm_cin(result_format),
            (side, side),
            column,
            row,
            compile_options=_automatic_options(),
        )
    assert caught.value.code == CODE
    assert f"{side * side} tuples in the join" in str(caught.value)


@pytest.mark.parametrize("result_format", ["ds", "ss", "oo"])
def test_the_new_compiler_computes_spgemm_the_counts_refused(
    fresh_caches, result_format
):
    a, a_ref = _sparse_matrix(M, M, STORED, seed=70)
    b, b_ref = _sparse_matrix(M, M, STORED, seed=71)
    result, _ = execute_cin_via_loopir(
        _spgemm_cin(result_format), (M, M), a, b, compile_options=_automatic_options()
    )
    _assert_product(result, a_ref, b_ref)


def test_a_call_the_counts_accept_never_reads_coordinates(fresh_caches, monkeypatch):
    def refuse(self, operands):
        raise AssertionError("coordinates read")

    monkeypatch.setattr(result_bound.EntryBound, "join_bound", refuse)
    small, _ = _sparse_matrix(300, 300, 2_000, seed=60)
    scorch.einsum("ij,jk->ik", small, small, format="ss")  # 90,000 cells: no bound
    x, _ = _sparse_matrix(M, 1, 10, seed=61)
    y, _ = _sparse_matrix(1, M, 12, seed=62)
    # At most 120 entries from the counts: accepted without the coordinates.
    result = scorch.einsum("ij,jk->ik", x, y, format="ss")
    assert result.values.numel() == x.values.numel() * y.values.numel()
