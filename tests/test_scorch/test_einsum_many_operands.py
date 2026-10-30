"""einsum with 26 operands or more.

einsum names its operands ``A``, ``B``, ... in order and gives its result the next name.
It had 26 names, so a 26th operand left none for the result and the call raised
``IndexError: pop from empty list``.  The names now go on: ``AA``, ``AB``, ...
(``ops._einsum_tensor_names``).

Two things have to hold for that, and each has its tests here.

* The names.  The first 26 are the single letters, in order, so a call of 25 operands
  or fewer names its tensors as before.  No two names give one identifier in any of the
  forms a kernel builds from a tensor name.  No name is left out: that no such
  identifier is a macro where the kernel is compiled, for a name of any length, is held
  by the spelling of the merged coordinate and its tests
  (tests/test_scorch/test_merged_coordinate_name.py).
* The one place in the shipped lowering that recognised a tensor's position variable by
  its spelling: ``p``, one capital letter, digits
  (``CINLowerer._rewrite_output_pos_vars``, which makes the writes of a coordinate
  result safe inside the parallel loop over a coordinate operand's rows).  With a result
  named ``AA`` it left the shared position ``pAA1``, and its increment, in that loop.
  It goes by the program's tensor names now.

The executed cases compare with ``torch.einsum`` on float64 to 1e-12, name the result
format, run with register blocking off and on, and use small and larger extents.  The
larger have 67 rows: a position shared between the threads of a parallel loop shows from
about 32 rows on.  A coordinate result is also compared entry by entry with the stored
coordinates of the operand that decides them.
"""

import itertools
import re
import string

import pytest
import torch

import scorch
from scorch import ops
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.merged_coordinate import merged_coordinate_name
from scorch.layout import TensorSpec
from scorch.ops import _einsum_tensor_names
from tests.test_scorch.kernel_cache_support import empty_kernel_caches
from tests.test_scorch.level_array_support import (
    assert_matches,
    coordinate_entries,
    reference,
    stored,
)

SMALL = dict(a=3, b=4, c=2)
LARGER = dict(a=67, b=41, c=5)
# 25 is the most operands einsum took before; 26 and 27 are the first results named with
# two letters; 53 has operands named with two letters as well.
COUNTS = (25, 26, 27, 30, 53)
# The counts that are executed in every family, layout, extent and arm (each is a kernel
# build); 30 operands run in test_one_tensor_as_every_vector_operand.
EXECUTED = (25, 26, 27, 53)


def family(name, count):
    """(expression, operand formats, result format) of ``count`` operands.

    sddmm               ab,ac,bc,c,...,c->ab   a coordinate mask, the rest dense, a
                                              coordinate result: the parallel loop over
                                              the mask's rows
    sddmm_compressed    the same with the mask and the result in ``ds``
    matvec              ab,b,...,b->a          the sparse operand first
    matvec_sparse_last  b,...,b,ab->a          the sparse operand last, as coordinates
    """

    if name == "sddmm":
        terms = ["ab", "ac", "bc"] + ["c"] * (count - 3)
        return ",".join(terms) + "->ab", ["oo", "dd", "dd"] + ["d"] * (count - 3), "oo"
    if name == "sddmm_compressed":
        terms = ["ab", "ac", "bc"] + ["c"] * (count - 3)
        return ",".join(terms) + "->ab", ["ds", "dd", "dd"] + ["d"] * (count - 3), "ds"
    if name == "matvec":
        terms = ["ab"] + ["b"] * (count - 1)
        return ",".join(terms) + "->a", ["ds"] + ["d"] * (count - 1), "d"
    assert name == "matvec_sparse_last", name
    terms = ["b"] * (count - 1) + ["ab"]
    return ",".join(terms) + "->a", ["d"] * (count - 1) + ["oo"], "d"


FAMILIES = ("sddmm", "sddmm_compressed", "matvec", "matvec_sparse_last")


def operands(expression, formats, extents):
    """Dense values and the tensors holding them.  Dense operands lie in 0.9 to 1.1, so
    a product of 52 of them stays near 1; a sparse operand stores about 40% of its cells
    and nothing in its first row."""

    names = _einsum_tensor_names()
    dense, tensors = [], []
    terms = expression.split("->")[0].split(",")
    for position, (term, fmt) in enumerate(zip(terms, formats)):
        generator = torch.Generator().manual_seed(31 + position)
        shape = tuple(extents[letter] for letter in term)
        values = 0.9 + 0.2 * torch.rand(shape, generator=generator, dtype=torch.float64)
        if set(fmt) != {"d"}:
            values = values * (torch.rand(shape, generator=generator) < 0.4)
            values[0] = 0.0
        dense.append(values)
        tensors.append(stored(values, fmt, next(names)))
    return dense, tensors


def emitted(monkeypatch, expression, formats, result_format, extents):
    """The kernel public einsum emits; nothing is built.  (``level_array_support``'s
    helper names at most seven operands.)"""

    captured = []

    def stop_before_native(prepared):
        captured.append(prepared.request.cpp_sources[-1])
        return object()

    empty_kernel_caches(monkeypatch)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", stop_before_native)
    names = _einsum_tensor_names()
    specs = [
        TensorSpec(
            fmt,
            tuple(extents[letter] for letter in term),
            dtype=torch.float64,
            name=next(names),
        )
        for term, fmt in zip(expression.split("->")[0].split(","), formats)
    ]
    ops.einsum(expression, *specs, compile_only=True, format=result_format)
    assert len(captured) == 1, f"{len(captured)} kernels prepared"
    return captured[0]


# --------------------------------------------------------------------------------------
# The names
# --------------------------------------------------------------------------------------


def test_the_first_26_names_are_the_single_letters_in_order():
    assert list(itertools.islice(_einsum_tensor_names(), 26)) == list(
        string.ascii_uppercase
    )


def test_the_names_go_on_with_two_and_then_three_letters():
    two_letter_names = 26 * 26
    names = list(itertools.islice(_einsum_tensor_names(), 26 + two_letter_names + 2))
    assert names[26:29] == ["AA", "AB", "AC"]
    assert names[26 + 25] == "AZ" and names[26 + 26] == "BA"
    assert names[26 + two_letter_names - 1] == "ZZ"
    assert names[26 + two_letter_names :] == ["AAA", "AAB"]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[A-Z]+", name) for name in names)


def test_no_name_is_left_out():
    """``OK`` and ``PI`` are given like any other name (the 401st and the 425th).
    Behind an upper-case index letter and an underscore each would be a macro of the C
    library (``F_OK``, ``M_PI``), and the merged coordinate is not spelled that way."""

    names = list(itertools.islice(_einsum_tensor_names(), 26 + 26 * 26))
    assert names.index("OK") == 26 + 14 * 26 + 10 == 400
    assert names.index("PI") == 26 + 15 * 26 + 8 == 424
    assert merged_coordinate_name("F", "OK") not in ("F_OK", "R_OK", "W_OK", "X_OK")
    assert merged_coordinate_name("M", "PI") != "M_PI"


# Every form in which a kernel spells an identifier from a tensor name N (L is a level
# number), counted over the kernel sources both compilers emit for the raw programs, the
# conversions and the sums.
TENSOR_SUFFIXES = (
    "values",
    "shape",
    "mode_indices",
    "val",
    "values_torch",
    "values_data",
    "val_vec",
    "val_tensor",
    "capacity",
    "reduction",
    "nonempty",
)
LEVEL_SUFFIXES = (
    "pos",
    "crd",
    "size",
    "pos_index",
    "pos_torch",
    "pos_data",
    "crd_torch",
    "crd_tensor",
    "crd_vec",
    "crd_data",
)
IDENTIFIER_FORMS = (
    tuple("{N}_" + suffix for suffix in TENSOR_SUFFIXES)
    + tuple("{N}{L}_" + suffix for suffix in LEVEL_SUFFIXES)
    + ("p{N}{L}", "p{N}{L}_end", "p{N}", "i_{N}")
)


def test_no_two_names_share_an_identifier():
    """Every form a kernel builds from a tensor name, for the first 80 names and levels
    0 to 11: all distinct, and the run of capitals in each is the name it was built
    from."""

    names = list(itertools.islice(_einsum_tensor_names(), 80))
    built = {}
    for name in names:
        for level in range(12):
            for form in IDENTIFIER_FORMS:
                identifier = form.format(N=name, L=level)
                assert built.setdefault(identifier, name) == name, identifier
                assert re.findall(r"[A-Z]+", identifier) == [name]
    # The merged coordinate of an upper-case index letter holds two runs of capitals,
    # the letter and the name; it is no identifier above and no two pairs share one.
    for index in "AMZ":
        for name in names:
            identifier = merged_coordinate_name(index, name)
            assert re.findall(r"[A-Z]+", identifier) == [index, name]
            assert built.setdefault(identifier, (index, name)) == (index, name)


def test_a_name_ending_in_a_digit_would_share_an_identifier():
    """Why the names hold no digit: tensor ``T26`` at level 1 and tensor ``T261``."""

    assert "p{N}{L}".format(N="T26", L=1) == "p{N}".format(N="T261")


# --------------------------------------------------------------------------------------
# The lowering's test for a tensor position variable
# --------------------------------------------------------------------------------------


def test_a_position_variable_is_recognised_by_the_programs_tensor_names():
    pattern = CINLowerer._tensor_position_pattern(["A", "AA", "Z", "Result"])
    assert pattern.match("pA1").group(1) == "A"
    assert pattern.match("pAA1").group(1) == "AA"
    assert pattern.match("pZ10").group(1) == "Z"
    assert pattern.match("pResult0").group(1) == "Result"
    for other in ("pB1", "pA", "pA1_end", "A1", "p1", "pAAA1", "ptr1", "xpA1"):
        assert pattern.match(other) is None, other


def test_no_tensor_names_means_no_position_variable():
    assert CINLowerer._tensor_position_pattern([]).match("p1") is None
    assert CINLowerer._tensor_position_pattern([]).match("pA1") is None


def test_an_array_belongs_to_the_tensor_whose_whole_name_it_starts_with():
    for array in ("AA1_crd", "AA_values", "AA0_pos", "AA"):
        assert CINLowerer._is_array_of_tensor(array, "AA"), array
        assert not CINLowerer._is_array_of_tensor(array, "A"), array
    for array in ("A1_crd", "A_values", "A"):
        assert CINLowerer._is_array_of_tensor(array, "A"), array
        assert not CINLowerer._is_array_of_tensor(array, "AA"), array
    assert not CINLowerer._is_array_of_tensor("B1_crd", "A")


# --------------------------------------------------------------------------------------
# The kernel text: a coordinate result under the parallel loop over the mask's rows
# --------------------------------------------------------------------------------------


def result_name(count):
    return list(itertools.islice(_einsum_tensor_names(), count + 1))[count]


@pytest.mark.parametrize("count", COUNTS)
def test_a_coordinate_result_is_written_at_the_masks_position(monkeypatch, count):
    """The loop over the mask's stored entries is parallel, so the result's entries are
    written at the mask's position and the result's own position is neither used nor
    advanced there: for a result named ``Z`` (25 operands) and for ``AA`` and later."""

    name = result_name(count)
    assert (name == "Z") == (count == 25) and (len(name) == 2) == (count > 25)
    source = emitted(monkeypatch, *family("sddmm", count), LARGER)
    assert "#pragma omp parallel for" in source
    for array in (f"{name}_values", f"{name}0_crd", f"{name}1_crd"):
        assert re.search(rf"\b{array}\[pA\d\] = ", source), array
    assert re.search(rf"\b{name}(_values|\d_crd)\[p{name}\d\]", source) is None
    assert re.search(rf"\bp{name}\d\s*(\+\+|\+=)", source) is None


def test_the_one_letter_test_left_a_two_letter_results_position_shared(monkeypatch):
    """The control for the test above: with the lowering's position test put back to the
    spelling it had (``p``, one capital letter, digits), the 26-operand kernel writes
    the result at the result's own position inside the parallel loop."""

    monkeypatch.setattr(
        CINLowerer,
        "_tensor_position_pattern",
        staticmethod(lambda names: re.compile(r"^p([A-Z])\d+$")),
    )
    source = emitted(monkeypatch, *family("sddmm", 26), LARGER)
    assert "#pragma omp parallel for" in source
    assert re.search(r"\bAA(_values|\d_crd)\[pAA\d\]", source)


# --------------------------------------------------------------------------------------
# Executed
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("blocking", [False, True], ids=["baseline", "regblock"])
@pytest.mark.parametrize("extents", [SMALL, LARGER], ids=["small", "larger"])
@pytest.mark.parametrize("count", EXECUTED)
@pytest.mark.parametrize("name", FAMILIES)
def test_many_operands_are_computed(name, count, extents, blocking):
    expression, formats, result_format = family(name, count)
    dense, tensors = operands(expression, formats, extents)
    result = scorch.einsum(
        expression,
        *tensors,
        format=result_format,
        _compile_options=CompileOptions.from_environment().with_regblock_enabled(
            blocking
        ),
    )
    assert_matches(result, reference(expression, dense), torch.float64)
    if result_format == "oo":
        # One entry per stored entry of the mask, in the mask's order: an entry written
        # twice, or at another entry's position, fails here even when the values add up.
        coordinates, _ = coordinate_entries(result)
        assert torch.equal(coordinates, dense[0].nonzero())


@pytest.mark.parametrize("count", (26, 30))
def test_one_tensor_as_every_vector_operand(count):
    """The same tensor object 25 and 29 times: each use is an operand of its own."""

    expression, formats, result_format = family("matvec", count)
    dense, tensors = operands(expression, formats, LARGER)
    shared_dense = [dense[0]] + [dense[1]] * (count - 1)
    shared = [tensors[0]] + [tensors[1]] * (count - 1)
    result = scorch.einsum(expression, *shared, format=result_format)
    assert_matches(result, reference(expression, shared_dense), torch.float64)
