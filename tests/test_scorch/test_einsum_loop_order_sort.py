"""The einsum loop-order sort returns an order for every expression, and the order it
always returned wherever it used to return one.

``utils.topo_sort_characters`` orders an expression's index letters so that each operand
is read in its storage order: an edge from each letter of a term to the next, a
topological sort, and where the terms contradict one another ``resolve_cycles`` gives up
edges, cheapest first.  As shipped, that resolution did not return on every graph:

* a term that repeats a letter side by side (``"ii"``) gave an edge from the letter to
  itself, which the loop removed and put back for ever;
* a cycle through an edge the loop itself had put in (the reverse of one it gave up)
  raised a bare ``ValueError``, ``max()`` of an empty list: such an edge belongs to no
  operand and has no cost;
* nothing bounded its passes: an edge it removed can come back as the reverse of
  another, so the graphs it visits need not all differ.  (No expression whose terms have
  distinct letters was found to make it repeat one; the edge from a letter to itself
  does.)

The second is not rare.  ``"ac,dba->adcb"`` and ``"ab,daeb->bd"``, two operands each,
raise it whatever the letters are called, and so do about 1.1% of all expressions of up
to three operands on four letters (26,058 of 2,271,084 at ``PYTHONHASHSEED=0``).

Now a letter beside itself adds no edge; the shipped rule runs first, unchanged; and only
when it meets a cycle through an edge of no operand, or a state it has already had, is
the graph resolved by a second rule, which drops such an edge instead of inverting it
and takes the letters in order of first appearance.  So:

* wherever the shipped sort returned an order, the sort returns that order (compared here
  against the shipped code itself, kept in ``shipped_loop_order_sort.py``);
* everywhere else it returns an order too; on that path the order does not depend on the
  order Python iterates a set of letters in, and it keeps every operand edge that lies on
  no cycle;
* the expressions that had no loop order are computed, checked against PyTorch.

The shipped sort iterates a set of letters, so which of its three outcomes an expression
gets can follow ``PYTHONHASHSEED``.  A test that names an expression therefore renames
its letters in every possible way: the set order is the same in each renaming, so the
renamings between them put the letters in every order relative to the expression.
"""

import collections
import itertools
import random
from types import SimpleNamespace

import pytest
import torch

import scorch
from scorch import utils
from scorch.compiler.compile_options import CompileOptions
from scorch.exceptions import CompileSpecError, EinsumLoopOrderError
from scorch.utils import topo_sort_characters
from tests.test_scorch import shipped_loop_order_sort as shipped
from tests.test_scorch.level_array_support import assert_matches, reference, stored
from tests.test_scorch.test_einsum_repeated_index import time_limit

ASCENDING = dict(zip("abcde", (2, 3, 5, 7, 11)))
DESCENDING = dict(zip("abcde", (11, 7, 5, 3, 2)))


def terms(letters, longest, distinct):
    """Every string of 1..``longest`` of ``letters``; ``distinct``: no letter twice."""

    return [
        "".join(chosen)
        for length in range(1, longest + 1)
        for chosen in itertools.product(letters, repeat=length)
        if not distinct or len(set(chosen)) == length
    ]


def operands_of(strings, extents):
    """Stand-ins for the operands: the sort reads nothing of a tensor but its shape."""

    return [
        SimpleNamespace(shape=tuple(extents[letter] for letter in term))
        for term in strings[:-1]
    ]


def shipped_outcome(strings, extents):
    """``("order", letters)``, ``("empty max()", None)`` or ``("never returns", None)``."""

    try:
        order = shipped.topo_sort_characters(
            list(strings), operands_of(strings, extents)
        )
    except shipped.NeverReturns:
        return "never returns", None
    except ValueError as error:
        assert "max()" in str(error), error
        return "empty max()", None
    return "order", order


def sorted_now(strings, extents):
    order = topo_sort_characters(list(strings), operands_of(strings, extents))
    assert sorted(order) == sorted(set("".join(strings))), (strings, order)
    return order


def renamings(strings, extents):
    """``strings`` with its letters renamed in every way, each with its extents and the
    renaming itself."""

    letters = sorted(set("".join(strings)))
    for image in itertools.permutations(letters):
        rename = dict(zip(letters, image))
        yield (
            ["".join(rename[letter] for letter in term) for term in strings],
            {rename[letter]: extents[letter] for letter in letters},
            rename,
        )


def beside_itself(strings):
    """Whether some term has a letter next to itself."""

    return any(
        left == right for term in strings for left, right in zip(term, term[1:])
    )


def edges_on_no_cycle(strings):
    """Pairs ``(left, right)`` of letters side by side in a term such that no chain of
    such pairs leads from ``right`` back to ``left``."""

    letters = sorted(set("".join(strings)))
    edges = {
        (left, right)
        for term in strings
        for left, right in zip(term, term[1:])
        if left != right
    }
    reaches = {letter: {letter} for letter in letters}
    changed = True
    while changed:
        changed = False
        for left, right in edges:
            new = reaches[right] - reaches[left]
            if new:
                reaches[left] |= new
                changed = True
    return [(left, right) for left, right in edges if left not in reaches[right]]


def expressions():
    """Small expressions: every one of a few letters, and a seeded sample of larger ones.

    Each is the terms of an einsum (operands, then a result whose letters are distinct
    and occur in an operand) with an extent per letter."""

    three, four, five = "abc", "abcd", "abcde"

    result_terms = {letters: terms(letters, 3, True) for letters in (three, four, five)}

    def results(operands, letters):
        present = set("".join(operands))
        return [term for term in result_terms[letters] if set(term) <= present]

    # Three letters, terms that may repeat a letter, one or two operands.
    for count in (1, 2):
        for operands in itertools.product(terms(three, 3, False), repeat=count):
            for result in results(operands, three):
                for extents in (ASCENDING, DESCENDING):
                    yield list(operands) + [result], extents
    # Four letters, terms of distinct letters, one or two operands.
    for count in (1, 2):
        for operands in itertools.product(terms(four, 3, True), repeat=count):
            for result in results(operands, four):
                yield list(operands) + [result], ASCENDING
    # Sampled: three to six operands, up to five letters, with and without repeats.
    rng = random.Random(20261012)
    for letters, longest, distinct, counts, samples in [
        (four, 3, True, (3, 4), 12000),
        (five, 4, True, (2, 3, 4, 5, 6), 12000),
        (five, 4, False, (1, 2, 3, 4, 5), 6000),
    ]:
        pool = terms(letters, longest, distinct)
        for _ in range(samples):
            operands = [rng.choice(pool) for _ in range(rng.choice(counts))]
            result = rng.choice(results(operands, letters))
            extents = {letter: rng.randint(1, 9) for letter in letters}
            yield operands + [result], extents


def test_same_order_wherever_the_shipped_sort_returned_one():
    outcomes = collections.Counter()
    with time_limit(600.0):
        for strings, extents in expressions():
            kind, order = shipped_outcome(strings, extents)
            now = sorted_now(strings, extents)
            if kind == "order":
                assert now == order, (strings, extents)
            elif not beside_itself(strings):
                # The second rule resolved it (a term with a letter beside itself may
                # have been resolved by the shipped rule once that edge is left out).
                for left, right in edges_on_no_cycle(strings):
                    assert now.index(left) < now.index(right), (strings, now)
            outcomes[kind] += 1
    # The sweep is only a test of the claim if the shipped sort did all three things.
    assert outcomes["order"] > 50000, outcomes
    assert outcomes["empty max()"] > 0 and outcomes["never returns"] > 0, outcomes


@pytest.mark.parametrize(
    ("strings", "order"),
    [
        (["ii", "i"], ["i"]),
        (["iij", "ij"], ["i", "j"]),
        (["jii", "ij"], ["j", "i"]),
        (["iii", "i"], ["i"]),
        (["ij", "jj", "i"], ["i", "j"]),
        (["ijj", "kk", "ik"], ["i", "j", "k"]),
    ],
)
def test_a_letter_beside_itself_adds_no_edge(strings, order):
    extents = dict(i=3, j=4, k=5)
    assert shipped_outcome(strings, extents) == ("never returns", None)
    with time_limit(5.0):
        assert sorted_now(strings, extents) == order


# Expressions the shipped sort had no order for, the smallest of each kind that
# work/sort_scan.py found (see the patch's note): (terms, extents, shipped outcome).
NO_SHIPPED_ORDER = [
    (["ac", "dba", "adcb"], dict(a=3, b=8, c=9, d=4), "empty max()"),
    (["ab", "daeb", "bd"], dict(a=1, b=7, d=7, e=2), "empty max()"),
    (["cdb", "ac", "cbda"], dict(a=4, b=3, c=3, d=7), "empty max()"),
    (["ed", "ecdb", "dbe"], dict(b=3, c=6, d=2, e=7), "empty max()"),
]


@pytest.mark.parametrize(("strings", "extents", "kind"), NO_SHIPPED_ORDER)
def test_expression_the_shipped_sort_had_no_order_for(strings, extents, kind):
    outcomes = collections.Counter()
    second_rule_orders = set()
    with time_limit(60.0):
        for renamed, renamed_extents, rename in renamings(strings, extents):
            shipped_kind, shipped_order = shipped_outcome(renamed, renamed_extents)
            outcomes[shipped_kind] += 1
            now = sorted_now(renamed, renamed_extents)
            if shipped_kind == "order":
                assert now == shipped_order
                continue
            back = {image: letter for letter, image in rename.items()}
            second_rule_orders.add(tuple(back[letter] for letter in now))
            for left, right in edges_on_no_cycle(renamed):
                assert now.index(left) < now.index(right), (renamed, now)
    # The shipped sort has no order for these whatever the letters are called, that is,
    # under every hash seed.
    assert set(outcomes) == {kind}, outcomes
    # One order, whatever the letters are called: the second rule does not follow the
    # order of the set.
    assert len(second_rule_orders) == 1, second_rule_orders


def test_a_cycle_left_after_resolution_is_a_coded_error(monkeypatch):
    """The resolution leaves no cycle; if it ever did, the expression is refused with a
    code, not lowered with some of its indices missing."""

    monkeypatch.setattr(utils, "resolve_cycles", lambda *args: True)
    with pytest.raises(EinsumLoopOrderError) as raised:
        topo_sort_characters(
            ["ab", "ba", "ab"], operands_of(["ab", "ba", "ab"], ASCENDING)
        )
    assert raised.value.code == "einsum_loop_order_unresolved"
    assert isinstance(raised.value, CompileSpecError)
    assert "a, b" in str(raised.value)


def test_resolution_ends_on_a_graph_whose_passes_repeat():
    """``resolve_cycles`` called on its own, with an edge from a letter to itself (the
    sort no longer builds one): the shipped rule removes that edge and puts it back, a
    state it has already had, and the second rule then drops it."""

    def graph_with_a_self_edge():
        graph = collections.defaultdict(list, {"i": ["i", "j"], "j": []})
        in_degree = collections.defaultdict(int, {"i": 1, "j": 1})
        return {"i", "j"}, graph, in_degree

    strings = ["iij", "ij"]
    tensors = [SimpleNamespace(shape=(3, 3, 4))]
    with pytest.raises(shipped.NeverReturns):
        shipped.resolve_cycles(*graph_with_a_self_edge(), strings, tensors)
    nodes, graph, in_degree = graph_with_a_self_edge()
    with time_limit(5.0):
        assert utils.resolve_cycles(nodes, graph, in_degree, strings, tensors) is False
    assert dict(graph) == {"i": ["j"], "j": []}
    assert dict(in_degree) == {"i": 0, "j": 1}


def test_resolution_drops_an_edge_no_operand_has():
    """A cycle through an edge that belongs to no term (here ``b -> a``, with terms that
    only say ``a`` before ``b``) is broken by dropping that edge."""

    nodes = {"a", "b"}
    graph = collections.defaultdict(list, {"a": ["b"], "b": ["a"]})
    in_degree = collections.defaultdict(int, {"a": 1, "b": 1})
    strings, tensors = ["ab", "ab"], [SimpleNamespace(shape=(2, 3))]
    with time_limit(5.0):
        assert utils.resolve_cycles(nodes, graph, in_degree, strings, tensors) is False
    assert dict(graph) == {"a": ["b"], "b": []}
    assert dict(in_degree) == {"a": 0, "b": 1}


# Expressions with no shipped loop order, executed: (expression, operand formats, result
# format).  Operands in a compressed, a coordinate and a dense layout, so that the loop
# order the second rule selects is reached through operand moves; a sparse and a dense
# result.
COMPUTED = [
    ("ab,daeb->bd", ["ds", "dsss"], "ds"),
    ("ab,daeb->bd", ["ds", "dsss"], "dd"),
    ("ab,daeb->bd", ["oo", "oooo"], "oo"),
    ("ab,daeb->bd", ["dd", "ssss"], "ss"),
    ("ac,dba->adcb", ["ds", "dss"], "dsss"),
    ("ac,dba->adcb", ["ss", "ddd"], "dddd"),
]
SMALL = dict(a=3, b=5, c=4, d=6, e=2)
LARGER = dict(a=7, b=23, c=19, d=17, e=5)


@pytest.mark.parametrize("blocking", [False, True], ids=["baseline", "regblock"])
@pytest.mark.parametrize("extents", [SMALL, LARGER], ids=["small", "larger"])
@pytest.mark.parametrize(
    ("expression", "formats", "result_format"),
    COMPUTED,
    ids=[f"{e} {','.join(f)}->[{r}]" for e, f, r in COMPUTED],
)
def test_expression_with_no_shipped_loop_order_is_computed(
    expression, formats, result_format, extents, blocking
):
    terms_, result_term = expression.split("->")[0].split(","), expression.split("->")[1]
    strings = terms_ + [result_term]
    # Under every renaming, so under every hash seed, the shipped sort has no order for
    # this expression at these extents: the call below runs on the second rule's order.
    assert {
        shipped_outcome(renamed, renamed_extents)[0]
        for renamed, renamed_extents, _ in renamings(strings, extents)
    } == {"empty max()"}

    dense, operands = [], []
    for position, (term, fmt) in enumerate(zip(terms_, formats)):
        generator = torch.Generator().manual_seed(31 + position)
        shape = tuple(extents[letter] for letter in term)
        values = torch.rand(shape, generator=generator, dtype=torch.float64)
        values = values * (torch.rand(shape, generator=generator) < 0.4)
        values[0] = 0.0  # a first slice that stores nothing
        dense.append(values)
        operands.append(stored(values, fmt, "ABDEFGH"[position]))
    result = scorch.einsum(
        expression,
        *operands,
        format=result_format,
        _compile_options=CompileOptions.from_environment().with_regblock_enabled(
            blocking
        ),
    )
    assert_matches(result, reference(expression, dense), torch.float64)
