"""The 32-bit position bound: per storage level, not per logical element count.

Kernels keep a level's positions, their loop variables and their coordinates in ``int``.
What a tensor may hold is therefore bounded by the positions each of its levels stores
(csrc/native_abi.h, ``checked_dense_positions`` for an operand and
``check_result_positions`` for a result), and a tensor with a sparse level is no longer
refused because rows x columns passes ``2**31 - 1``.  Every refusal under the bound is an
``IndexCapacityError`` with the code ``index_position_exceeds_int32``.

The first part needs no kernel: the rule's Python side, the kernel ABI object and the
mapping of the native text.  The second part compiles one small extension against
``header.h`` and holds the native rule to the same table.  The third part runs the
prebuilt kernels and the generated ones at extents whose product passes the old bound,
with references computed from the stored entries, never from a dense array of that size.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import re
import resource
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load_inline

import scorch
import scorch_ops
from scorch import STensor
from scorch.compiler import llir
from scorch.compiler.cin import TensorVar
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.torch_cpp_abi import (
    KernelTensorABI,
    TorchCppKernelABI,
    _refuse_unaddressable_dense_prefix,
    _refuse_unaddressable_result,
)
from scorch.exceptions import (
    IndexCapacityError,
    RepeatedCoordinateError,
    TensorValidationError,
    raise_if_native_coded,
)
from scorch.format import MAX_INDEX_POSITION, LevelType, dense_prefix_unaddressable
from scorch.stensor import _refuse_oversized_result_arrays
from tests.test_scorch.level_array_support import emitted_einsum_cpp

M = MAX_INDEX_POSITION
CODE = "index_position_exceeds_int32"
NATIVE_CODE = " [index_position_exceeds_int32]"
_CSRC = Path(__file__).resolve().parents[2] / "src" / "scorch" / "csrc"
_KIND = {"d": LevelType.DENSE, "s": LevelType.COMPRESSED, "o": LevelType.COORDINATE}
_KIND_CODE = {"d": 0, "s": 1, "o": 2}


def _levels(letters):
    return tuple(_KIND[letter] for letter in letters)


def assert_capacity_error(error):
    assert type(error) is IndexCapacityError
    assert error.code == CODE
    # The native check raised a bare RuntimeError before it had a code, and a validation
    # error is a ValueError everywhere else in Scorch: callers of either keep working.
    assert isinstance(error, RuntimeError)
    assert isinstance(error, TensorValidationError)
    assert NATIVE_CODE not in str(error)


# --------------------------------------------------------------------------------------
# The rule, as a table.  ``True`` = refused.
#
# 46,340 ** 2 = 2,147,395,600 is the largest square below 2**31 - 1 = 2,147,483,647, and
# 46,341 ** 2 = 2,147,488,281 the smallest above it.
# --------------------------------------------------------------------------------------

RESULT_TABLE = [
    # all-dense: the element count, as before
    ("dd", (46340, 46340), False),
    ("dd", (46341, 46341), True),
    ("d", (M,), False),
    ("ddd", (1291, 1291, 1289), True),  # 2,148,351,809
    ("ddd", (1290, 1290, 1290), False),  # 2,146,689,000
    ("dd", (0, M), False),
    ("dd", (M, 0), False),
    ("ddd", (M, 5, 0), True),  # the kernel forms row * 5 + column before the empty level
    ("ddd", (0, M, M), False),
    # a dense prefix: only the prefix, and one position entry more than it has parents
    ("ds", (64000, 64000), False),
    ("ds", (1000000, 1000000), False),
    ("ds", (M - 1, M), False),
    ("ds", (M, 1), True),
    ("dss", (46341, 46341, M), False),
    ("dds", (46340, 46340, M), False),
    ("dds", (46341, 46341, 5), True),
    ("do", (M, M), False),
    # no dense level: nothing is known before the kernel runs
    ("ss", (M, M), False),
    ("oo", (M, M), False),
    ("sss", (M, M, 2), False),  # the largest: (M, M, 3) has more elements than int64 counts
    ("s", (M,), False),
    # a dense level below a sparse one: the logical size down to that level
    ("sd", (46340, 46340), False),
    ("sd", (46341, 46341), True),
    ("sd", (64000, 64000), True),
    ("sds", (46341, 46341, 3), True),
    ("sds", (46340, 46340, M), False),
    ("sds", (M, 1, 7), True),  # level 2 would have M parents and M + 1 position entries
    ("dsd", (10, 46341, 4635), True),
    ("dsd", (10, 46340, 4634), False),
    ("ssd", (1291, 1291, 1289), True),
    ("dsd", (M, 1, 1), True),  # level 1 would have M parents
    ("dsd", (M - 1, 1, 1), False),
]

OPERAND_PREFIX_TABLE = [
    ("dd", (46340, 46340), False),
    ("dd", (46341, 46341), True),
    ("ds", (64000, 64000), False),
    ("ds", (M, M), False),  # one more position entry than rows: left to the arrays
    ("dds", (46341, 46341, 2), True),
    ("sd", (M, M), False),  # under a sparse level: decided by what the level stores
    ("ss", (M, M), False),
    ("ddd", (M, 5, 0), True),
    ("ddd", (0, M, M), False),
]

RELAYOUT_TABLE = [
    ("ds", (M - 1, 7), False),
    ("ds", (M, 7), True),  # M + 1 position entries
    ("do", (M, 7), False),  # a coordinate level has no position array
    ("dds", (46340, 46340, 7), False),
    ("dds", (46341, 46341, 7), True),
    ("dd", (M, M), False),  # all-dense: stated by the kernel that reads it
    ("ss", (M, M), False),
    ("sd", (M, M), False),
    ("ds", (0, M), False),
]


@pytest.mark.parametrize("letters, shape, refused", RESULT_TABLE)
def test_the_result_rule_before_a_kernel_is_built(letters, shape, refused):
    if not refused:
        _refuse_unaddressable_result(shape, _levels(letters), "evaluate")
        return
    with pytest.raises(IndexCapacityError) as caught:
        _refuse_unaddressable_result(shape, _levels(letters), "evaluate")
    assert_capacity_error(caught.value)
    assert str(caught.value).startswith(f"evaluate: result_shape {list(shape)}: ")


@pytest.mark.parametrize("letters, shape, refused", OPERAND_PREFIX_TABLE)
def test_the_operand_rule_for_the_dense_levels_a_layout_starts_with(
    letters, shape, refused
):
    if not refused:
        _refuse_unaddressable_dense_prefix("A", shape, _levels(letters), "evaluate")
        return
    with pytest.raises(IndexCapacityError) as caught:
        _refuse_unaddressable_dense_prefix("A", shape, _levels(letters), "evaluate")
    assert_capacity_error(caught.value)
    assert str(caught.value).startswith("evaluate: A dense level ")


@pytest.mark.parametrize("letters, extents, unaddressable", RELAYOUT_TABLE)
def test_the_layout_a_relayout_would_build(letters, extents, unaddressable):
    assert dense_prefix_unaddressable(letters, extents) is unaddressable


# --------------------------------------------------------------------------------------
# The kernel ABI object: it carries the result's layout, emits it, and refuses a known
# shape no kernel could address when it is constructed, which is before anything is built.
# --------------------------------------------------------------------------------------


def _operand(letters, shape, name="A"):
    return KernelTensorABI(
        name, _levels(letters), tuple(range(len(letters))), shape, torch.float64
    )


def test_the_emitted_result_check_carries_the_result_layout():
    abi = TorchCppKernelABI(
        result_shape=(64000, 64000),
        result_level_types=_levels("ds"),
        input_tensors=(_operand("ds", (64000, 64000)),),
    )
    assert abi.result_rank == 2
    lines = LLIRLowerer().lower_llir(abi.emit_validation()).split("\n")
    assert lines[0] == (
        "scorch_native::validate_jit_result_shape(result_shape, {64000, 64000}, "
        '{0, 1}, "evaluate");'
    )
    kinds = abi.emit_validation()[0].args[2]
    assert kinds == llir.Array(
        values=(
            llir.Literal(value=0, data_type=llir.DataType.INT),
            llir.Literal(value=1, data_type=llir.DataType.INT),
        ),
        data_type=llir.DataType.INT,
    )
    unknown = TorchCppKernelABI(
        result_shape=(), result_level_types=_levels("oo"), input_tensors=()
    )
    assert LLIRLowerer().lower_llir(unknown.emit_validation()) == (
        'scorch_native::validate_jit_result_shape(result_shape, {}, {2, 2}, "evaluate");'
    )


@pytest.mark.parametrize("letters, shape, refused", RESULT_TABLE)
def test_a_kernel_abi_refuses_an_unaddressable_known_result(letters, shape, refused):
    def build():
        return TorchCppKernelABI(
            result_shape=shape, result_level_types=_levels(letters), input_tensors=()
        )

    if not refused:
        assert build().result_shape == shape
        return
    with pytest.raises(IndexCapacityError) as caught:
        build()
    assert_capacity_error(caught.value)


@pytest.mark.parametrize("letters, shape, refused", OPERAND_PREFIX_TABLE)
def test_a_kernel_abi_refuses_an_unaddressable_known_operand(letters, shape, refused):
    def build():
        return TorchCppKernelABI(
            result_shape=(3,),
            result_level_types=_levels("d"),
            input_tensors=(_operand(letters, shape),),
        )

    if not refused:
        assert build().input_tensors[0].shape == shape
        return
    with pytest.raises(IndexCapacityError) as caught:
        build()
    assert_capacity_error(caught.value)
    assert "A dense level" in str(caught.value)


def test_a_kernel_abi_of_unknown_shapes_leaves_the_bound_to_the_kernel():
    abi = TorchCppKernelABI(
        result_shape=(),
        result_level_types=_levels("dd"),
        input_tensors=(_operand("dd", ()),),
    )
    assert abi.result_shape == ()


def test_both_compilers_build_the_abi_with_the_result_layout():
    """The shipped lowering's snapshot and the new compiler's go through one class, so one
    test of the class's field covers both construction sites' emitted text; this pins that
    the shipped snapshot reads the layout of the result it is given."""
    from scorch.compiler.cin_lowerer import _torch_cpp_kernel_abi

    for letters, kinds in (("ds", (0, 1)), ("oo", (2, 2)), ("sd", (1, 0)), ("dd", (0, 0))):
        result = TensorVar("Result", shape=(5, 3), fmt=letters, dtype=torch.float64)
        operand = TensorVar("A", shape=(5, 3), fmt="ds", dtype=torch.float64)
        abi = _torch_cpp_kernel_abi(result, [operand], None)
        assert abi.result_level_types == _levels(letters)
        text = LLIRLowerer().lower_llir(abi.emit_validation()[:1])
        assert text == (
            "scorch_native::validate_jit_result_shape(result_shape, {5, 3}, "
            f'{{{kinds[0]}, {kinds[1]}}}, "evaluate");'
        )


# --------------------------------------------------------------------------------------
# The native text and its structured form.
# --------------------------------------------------------------------------------------


def test_the_native_refusal_text_becomes_the_structured_error():
    line = (
        "evaluate: A dense level 1 of extent 64000 under 64001 parent positions needs "
        "more than 2147483647 positions, the most a 32-bit position addresses"
    )
    native = RuntimeError(
        line + NATIVE_CODE + "\nException raised from checked_dense_positions at "
        "native_abi.h:1470 (most recent call first):\nframe #0: c10::Error::Error"
    )
    with pytest.raises(IndexCapacityError) as caught:
        raise_if_native_coded(native)
    assert_capacity_error(caught.value)
    assert str(caught.value) == line
    assert caught.value.__cause__ is native


def test_the_mapper_keeps_the_repeat_code_and_passes_everything_else_through():
    repeat = RuntimeError(
        "op: x COO coordinate (3) is stored at positions 0 and 1; call coalesce() to sum "
        "repeated coordinates [coo_repeated_coordinate]"
    )
    with pytest.raises(RepeatedCoordinateError):
        raise_if_native_coded(repeat)
    for text in (
        "evaluate: A must have rank 2, got 3",
        "exceeds the current native int32 allocation bound",  # no code: not ours
        "index_position_exceeds_int32",  # the bare word is not the code's spelling
    ):
        assert raise_if_native_coded(RuntimeError(text)) is None
    # A refusal raised in Python is already structured and has no code in its text.
    assert raise_if_native_coded(IndexCapacityError("evaluate: too large")) is None


# --------------------------------------------------------------------------------------
# What a kernel returns: array lengths alone (a real case needs 24 GB).
# --------------------------------------------------------------------------------------


class _Length:
    """Stands in for a level array or a values tensor: the check reads lengths only."""

    def __init__(self, count):
        self._count = count

    def numel(self):
        return self._count


@pytest.mark.parametrize(
    "lengths, values, refused",
    [
        ([[], [M, M]], M, False),
        ([[], [64001, M + 1]], M + 1, True),  # a two-pass CSR result past the bound
        ([[], [M + 1, 5]], 5, True),  # M rows: one position entry too many
        ([[M + 1], [M + 1]], M + 1, True),  # COO
        ([[3, 5], [6, 9]], M + 1, True),  # values alone
        ([[3, 5], [6, 9]], 9, False),
    ],
)
def test_a_sparse_result_past_the_bound_is_refused_from_its_array_lengths(
    lengths, values, refused
):
    arrays = [[_Length(count) for count in level] for level in lengths]
    if not refused:
        _refuse_oversized_result_arrays(arrays, _Length(values))
        return
    with pytest.raises(IndexCapacityError) as caught:
        _refuse_oversized_result_arrays(arrays, _Length(values))
    assert_capacity_error(caught.value)


def test_a_small_sparse_result_passes_the_wrapper_and_a_dense_one_does_not_reach_it():
    a = torch.eye(4, dtype=torch.float64).to_sparse_csr()
    b = torch.arange(16, dtype=torch.float64).reshape(4, 4).to_sparse_csr()
    product = scorch.einsum(
        "ij,ij->ij", STensor.from_csr(a), STensor.from_csr(b), format="ds"
    )
    expected = torch.diag(torch.tensor([0.0, 5, 10, 15], dtype=torch.float64))
    assert torch.equal(product.to_torch(), expected)


# --------------------------------------------------------------------------------------
# The native rule, held to the same table.
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _native():
    source = r"""
#include "header.h"

void result_positions(std::vector<int64_t> shape, std::vector<int> kinds) {
  scorch_native::validate_jit_result_shape(shape, {}, kinds, "evaluate");
}

void known_result_positions(std::vector<int64_t> shape, std::vector<int> kinds) {
  scorch_native::validate_jit_result_shape(shape, shape, kinds, "evaluate");
}

// The level kinds as a generated kernel writes them, a braced list of constants, for each
// level list the result table names.  The two functions above pass a vector.
void braced_result_positions(std::vector<int64_t> shape, std::string letters) {
#define BRACED(spelling, ...)                                                       \
  if (letters == spelling) {                                                        \
    scorch_native::validate_jit_result_shape(shape, {}, {__VA_ARGS__}, "evaluate"); \
    return;                                                                         \
  }
  BRACED("d", 0)
  BRACED("s", 1)
  BRACED("dd", 0, 0)
  BRACED("ds", 0, 1)
  BRACED("do", 0, 2)
  BRACED("sd", 1, 0)
  BRACED("ss", 1, 1)
  BRACED("oo", 2, 2)
  BRACED("ddd", 0, 0, 0)
  BRACED("dds", 0, 0, 1)
  BRACED("dsd", 0, 1, 0)
  BRACED("dss", 0, 1, 1)
  BRACED("sds", 1, 0, 1)
  BRACED("ssd", 1, 1, 0)
  BRACED("sss", 1, 1, 1)
#undef BRACED
  TORCH_CHECK(false, "no braced level list is written here for ", letters);
}

// The rule as a yes or no (what a call whose result fits pays for), and the rule with its
// refusals worded, each on its own.  `expected` empty: compiled for the rank only.
bool result_positions_fit(std::vector<int64_t> shape, std::vector<int64_t> expected,
                          std::vector<int> kinds) {
  return scorch_native::result_positions_fit(
      shape, expected.data(), expected.size(), kinds.data(), kinds.size());
}

void refuse_result_positions(std::vector<int64_t> shape, std::vector<int64_t> expected,
                             std::vector<int> kinds) {
  scorch_native::refuse_result_positions(
      shape, expected, kinds.data(), kinds.size(), "evaluate");
}

int64_t dense_positions(int64_t parent_positions, int64_t extent, int64_t level) {
  return scorch_native::checked_dense_positions(
      parent_positions, extent, static_cast<size_t>(level), "evaluate", "A");
}

void operand(std::vector<int64_t> shape,
             std::vector<std::vector<torch::Tensor>> mode_indices,
             torch::Tensor values, std::vector<int> kinds) {
  std::vector<int> order(kinds.size());
  std::iota(order.begin(), order.end(), 0);
  scorch_native::validate_jit_tensor("evaluate", "A", shape, mode_indices, values,
                                     torch::kFloat64, kinds, order, {});
}

void merged_positions(int64_t child_coordinates) {
  scorch_check_position_capacity<int>(child_coordinates);
}

// One append to a vector that already holds `size` one-byte entries.
int64_t append_at(int64_t size) {
  std::vector<int8_t> values(static_cast<size_t>(size), int8_t{0});
  scorch_vector_set(values, static_cast<int>(size), int8_t{1});
  return static_cast<int64_t>(values.size());
}
"""
    digest = hashlib.sha256(
        (_CSRC / "header.h").read_bytes()
        + (_CSRC / "native_abi.h").read_bytes()
        + source.encode()
    ).hexdigest()[:12]
    return load_inline(
        name=f"scorch_index_position_bound_{digest}",
        cpp_sources=source,
        functions=[
            "result_positions",
            "known_result_positions",
            "braced_result_positions",
            "result_positions_fit",
            "refuse_result_positions",
            "dense_positions",
            "operand",
            "merged_positions",
            "append_at",
        ],
        extra_include_paths=[str(_CSRC)],
        extra_cflags=["-O0"],
        verbose=False,
    )


def _native_line(error):
    """The line of a native refusal that carries the code, without the code."""
    text = str(error)
    assert NATIVE_CODE in text, text
    line = next(part for part in text.splitlines() if NATIVE_CODE in part)
    return line.replace(NATIVE_CODE, "")


@pytest.mark.parametrize("letters, shape, refused", RESULT_TABLE)
@pytest.mark.parametrize("known", [False, True])
def test_the_native_result_rule_is_the_python_one(letters, shape, refused, known):
    kinds = [_KIND_CODE[letter] for letter in letters]
    check = _native().known_result_positions if known else _native().result_positions
    if not refused:
        check(list(shape), kinds)
        return
    with pytest.raises(RuntimeError) as native:
        check(list(shape), kinds)
    with pytest.raises(IndexCapacityError) as python:
        _refuse_unaddressable_result(shape, _levels(letters), "evaluate")
    # One rule, one text: the refusal reads the same whichever side raises it.
    assert _native_line(native.value) == str(python.value)
    with pytest.raises(IndexCapacityError) as mapped:
        raise_if_native_coded(native.value)
    assert str(mapped.value) == str(python.value)


@pytest.mark.parametrize("letters, shape, refused", RESULT_TABLE)
def test_a_braced_level_list_is_checked_as_a_vector_of_kinds_is(letters, shape, refused):
    """A generated kernel names its result's levels with a braced list, ``{0, 1}``, which
    the check takes without constructing a ``std::vector``; the tests above pass a vector.
    Both reach one rule, so they accept the same shapes and refuse with the same text."""
    native = _native()
    if not refused:
        native.braced_result_positions(list(shape), letters)
        return
    with pytest.raises(RuntimeError) as braced:
        native.braced_result_positions(list(shape), letters)
    with pytest.raises(RuntimeError) as vector:
        native.result_positions(list(shape), [_KIND_CODE[letter] for letter in letters])
    assert _native_line(braced.value) == _native_line(vector.value)
    with pytest.raises(IndexCapacityError) as python:
        _refuse_unaddressable_result(shape, _levels(letters), "evaluate")
    assert _native_line(braced.value) == str(python.value)


def _refuses(native, shape, expected, kinds):
    try:
        native.refuse_result_positions(list(shape), list(expected), list(kinds))
    except RuntimeError:
        return True
    return False


@pytest.mark.parametrize("letters, shape, refused", RESULT_TABLE)
@pytest.mark.parametrize("known", [False, True])
def test_the_yes_or_no_form_of_the_result_rule_follows_the_table(
    letters, shape, refused, known
):
    """A kernel call asks the rule as a yes or no first, which builds no vector and no
    message, and only a no reaches the worded form.  Each form alone, against the table."""
    native = _native()
    kinds = [_KIND_CODE[letter] for letter in letters]
    expected = list(shape) if known else []
    assert native.result_positions_fit(list(shape), expected, kinds) == (not refused)
    assert _refuses(native, shape, expected, kinds) == refused


def test_the_two_forms_of_the_result_rule_agree_near_every_boundary():
    """Ranks 0 to 3, every layout over the three level kinds, every combination of the
    extents below, compiled for the shape and for the rank only: the yes-or-no form says
    yes exactly where the worded form raises nothing.  Then the calls that are refused
    whatever the extents multiply to."""
    native = _native()
    extents = (0, 1, 3, 46341, M - 1, M)
    checked = refused = 0
    for rank in range(4):
        for kinds in itertools.product((0, 1, 2), repeat=rank):
            for shape in itertools.product(extents, repeat=rank):
                for expected in ((), shape):
                    fits = native.result_positions_fit(
                        list(shape), list(expected), list(kinds)
                    )
                    raised = _refuses(native, shape, expected, kinds)
                    assert fits == (not raised), (kinds, shape, expected)
                    checked += 1
                    refused += raised
    # 2 x (1 + 3 x 6 + 9 x 36 + 27 x 216) calls, and both answers occur often.
    assert checked == 12350
    assert 1000 < refused < checked - 1000, refused
    always_refused = [
        ((3, 4, 5), (), (0, 1)),  # three extents, two levels
        ((3, 4), (), (0, 1, 1)),  # two extents, three levels
        ((), (), (0,)),  # no extent, one level
        ((5,), (5,), ()),  # an extent and no level
        ((3, 4), (3, 4, 5), (0, 1)),  # compiled for another rank
        ((3, 4), (3, 5), (0, 1)),  # compiled for another shape
        ((3, -1), (), (0, 1)),  # a negative extent
        ((3, M + 1), (), (1, 1)),  # an extent past int, under no dense level
        ((3, 4), (), (0, 7)),  # a level kind that does not exist
        ((3, 4), (), (0, -1)),
    ]
    for shape, expected, kinds in always_refused:
        assert not native.result_positions_fit(list(shape), list(expected), list(kinds))
        assert _refuses(native, shape, expected, kinds), (shape, expected, kinds)


def test_the_native_result_check_still_refuses_what_it_always_did():
    native = _native()
    with pytest.raises(RuntimeError, match="must have rank 2"):
        native.result_positions([3, 4, 5], [0, 1])
    with pytest.raises(RuntimeError, match="negative dimension"):
        native.result_positions([3, -1], [0, 1])
    with pytest.raises(RuntimeError, match="int32 loop bound") as extent:
        native.result_positions([3, M + 1], [1, 1])
    assert NATIVE_CODE in str(extent.value)
    with pytest.raises(RuntimeError, match="unsupported level kind"):
        native.result_positions([3, 4], [0, 7])


@pytest.mark.parametrize("letters, shape, refused", OPERAND_PREFIX_TABLE)
def test_the_native_dense_level_rule_is_the_python_one(letters, shape, refused):
    native = _native()

    def walk():
        positions = 1
        for level, (letter, extent) in enumerate(zip(letters, shape)):
            if letter != "d":
                return
            positions = native.dense_positions(positions, extent, level)

    if not refused:
        walk()
        return
    with pytest.raises(RuntimeError) as caught:
        walk()
    with pytest.raises(IndexCapacityError) as python:
        _refuse_unaddressable_dense_prefix("A", shape, _levels(letters), "evaluate")
    assert _native_line(caught.value) == str(python.value)


def _int32(values):
    return torch.tensor(values, dtype=torch.int32)


def test_a_dense_level_under_a_sparse_level_counts_the_end_position():
    """``sd`` with c stored rows of extent n: a kernel that merges this operand with
    another leaves its row cursor at c when it is exhausted and may still form c * n + j,
    so (c + 1) * n positions must fit, one unit more than the positions that are read."""
    native = _native()
    values = torch.zeros(4, dtype=torch.float64)

    def call(stored_rows, extent):
        rows = list(range(stored_rows))
        native.operand(
            [1000, extent],
            [[_int32([0, stored_rows]), _int32(rows)], []],
            values,
            [1, 0],
        )

    # 2 stored rows: (2 + 1) * n <= M  <=>  n <= 715,827,882
    with pytest.raises(RuntimeError, match="values length must equal") as at_bound:
        call(2, 715827882)  # the bound passes; the 4 values are then too few
    assert NATIVE_CODE not in str(at_bound.value)
    with pytest.raises(RuntimeError) as past:
        call(2, 715827883)
    assert _native_line(past.value) == (
        "evaluate: A dense level 1 of extent 715827883 under 3 parent positions needs "
        "more than 2147483647 positions, the most a 32-bit position addresses"
    )
    # The same extent with one stored row fits: the bound follows what is stored.
    with pytest.raises(RuntimeError, match="values length must equal"):
        call(1, 715827883)


def test_an_operand_with_a_sparse_level_is_not_bounded_by_its_logical_size():
    native = _native()
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    # ds, (M - 1) x M would need M position entries; a 3-row one is enough to show that
    # rows x columns = 3 * M is not what is checked.
    native.operand(
        [3, M], [[], [_int32([0, 1, 2, 3]), _int32([0, 5, M - 1])]], values, [0, 1]
    )
    native.operand(
        [M, M],
        [[_int32([0, 3]), _int32([0, 7, M - 1])], [_int32([0, 1, 2, 3]), _int32([4, 4, 4])]],
        values,
        [1, 1],
    )
    native.operand([M, M], [[_int32([0, 7, M - 1])], [_int32([4, 4, 4])]], values, [2, 2])
    # An all-dense operand is: its last level's positions are its elements.
    with pytest.raises(RuntimeError) as dense:
        native.operand([46341, 46341], [[], []], values, [0, 0])
    assert _native_line(dense.value).startswith("evaluate: A dense level 1 of extent 46341")


def test_the_chunk_merges_refuse_a_position_past_the_bound():
    native = _native()
    native.merged_positions(0)
    native.merged_positions(M)
    with pytest.raises(RuntimeError) as caught:
        native.merged_positions(M + 1)
    assert _native_line(caught.value).startswith(
        "a result level's coordinates, merged from the parallel chunks, would hold "
        "2147483648 entries"
    )


def _memory_gb():
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


@pytest.mark.skipif(
    os.environ.get("SCORCH_BIGMEM_TESTS") != "1" or _memory_gb() < 8,
    reason="allocates 2 GB twice: needs SCORCH_BIGMEM_TESTS=1 and 8 GB of memory",
)
def test_an_append_stops_at_the_last_position_an_int_addresses():
    native = _native()
    assert native.append_at(5) == 6
    assert native.append_at(M - 1) == M  # entry number M, at position M - 1
    with pytest.raises(RuntimeError) as caught:
        native.append_at(M)
    assert _native_line(caught.value) == (
        "a level array a kernel appends to would hold 2147483648 entries; a 32-bit "
        "position addresses at most 2147483647"
    )


@lru_cache(maxsize=1)
def _optimized_kernel_entry():
    """A generated kernel's first statement, compiled at the kernels' -O3."""
    source = r"""
#include "header.h"

int64_t entry(std::vector<int64_t> result_shape) {
  scorch_native::validate_jit_result_shape(result_shape, {5, 3}, {0, 1}, "evaluate");
  return result_shape[0];
}
"""
    digest = hashlib.sha256(
        (_CSRC / "header.h").read_bytes()
        + (_CSRC / "native_abi.h").read_bytes()
        + source.encode()
    ).hexdigest()[:12]
    return load_inline(
        name=f"scorch_result_check_call_{digest}",
        cpp_sources=source,
        functions=["entry"],
        extra_include_paths=[str(_CSRC)],
        extra_cflags=["-O3"],
        verbose=False,
    )


def test_the_result_check_stays_a_call_in_an_optimized_kernel():
    """Inlined into evaluate(), the result check changed the register allocation of the
    kernel's loops (csrc/native_abi.h, validate_jit_result_shape).  At -O3 it must stay
    its own function: an inlined check leaves no symbol of it in the extension."""
    module = _optimized_kernel_entry()
    assert module.entry([5, 3]) == 5
    nm = shutil.which("nm")
    assert nm is not None, "nm reads the extension's symbols"
    symbols = subprocess.run(
        [nm, "-C", module.__file__], check=True, capture_output=True, text=True
    ).stdout
    assert "scorch_native::validate_jit_result_shape(" in symbols


# --------------------------------------------------------------------------------------
# Prebuilt kernels at extents whose product passes the old bound.
# --------------------------------------------------------------------------------------

OLD_BOUND = "exceeds the current native int32 allocation bound"


def _sparse_entries(rows, cols, count, seed):
    """``count`` distinct coordinates in row-major order, and float32 values."""
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randint(0, rows * cols, (count * 2,), generator=generator)
    flat = torch.unique(flat)[:count]
    values = torch.rand(flat.numel(), generator=generator) + 0.5
    return flat // cols, flat % cols, values


def _csr_arrays(rows, row, col):
    counts = torch.bincount(row, minlength=rows)
    positions = torch.zeros(rows + 1, dtype=torch.int64)
    positions[1:] = torch.cumsum(counts, 0)
    return positions.to(torch.int32), col.to(torch.int32)


@pytest.mark.parametrize("extent", [1024, 64000, 1000000])
def test_the_prebuilt_sddmm_is_bounded_by_its_stored_entries(extent):
    rank = 4
    row, col, values = _sparse_entries(extent, extent, 5000, seed=extent)
    generator = torch.Generator().manual_seed(1)
    a = torch.rand(extent, rank, generator=generator)
    b = torch.rand(extent, rank, generator=generator)
    result = scorch_ops.sddmm_coo_float_prebuilt(
        [extent, extent],
        [extent, extent],
        [[row.to(torch.int32)], [col.to(torch.int32)]],
        values,
        [extent, rank],
        [[], []],
        a.reshape(-1),
        [extent, rank],
        [[], []],
        b.reshape(-1),
    )
    expected = values.double() * (a[row].double() * b[col].double()).sum(1)
    got = result.storage.value.double()
    assert got.shape == expected.shape
    assert (got - expected).abs().max() <= 1e-5 * expected.abs().max()
    assert torch.equal(result.storage.index.mode_indices[0][0].long(), row)
    assert torch.equal(result.storage.index.mode_indices[1][0].long(), col)


@pytest.mark.parametrize("extent", [512, 64000, 1000000])
def test_the_prebuilt_csr_product_is_bounded_by_its_stored_entries(extent):
    row_a, col_a, values_a = _sparse_entries(extent, extent, 4000, seed=extent + 1)
    row_b, col_b, values_b = _sparse_entries(extent, extent, 4000, seed=extent + 2)
    # Rows of B that A's columns reach, so that the product is not empty at any extent.
    row_b = col_a[torch.randperm(col_a.numel(), generator=torch.Generator().manual_seed(3))]
    row_b, order = torch.sort(row_b[: col_b.numel()], stable=True)
    flat = torch.unique(row_b * extent + col_b[order][: row_b.numel()])
    row_b, col_b = flat // extent, flat % extent
    values_b = values_b[: flat.numel()]
    pos_a, crd_a = _csr_arrays(extent, row_a, col_a)
    pos_b, crd_b = _csr_arrays(extent, row_b, col_b)
    result = scorch_ops.spmspm_csr_float(
        [extent, extent],
        [extent, extent],
        [[], [pos_a, crd_a]],
        values_a,
        [extent, extent],
        [[], [pos_b, crd_b]],
        values_b,
    )
    a = torch.sparse_coo_tensor(
        torch.stack([row_a, col_a]), values_a.double(), (extent, extent)
    ).coalesce()
    b = torch.sparse_coo_tensor(
        torch.stack([row_b, col_b]), values_b.double(), (extent, extent)
    ).coalesce()
    expected = torch.sparse.mm(a, b).coalesce()
    positions, coordinates = result.storage.index.mode_indices[1]
    assert positions.numel() == extent + 1
    got_rows = torch.repeat_interleave(
        torch.arange(extent), torch.diff(positions.long())
    )
    assert expected._nnz() > 0
    assert torch.equal(got_rows, expected.indices()[0])
    assert torch.equal(coordinates.long(), expected.indices()[1])
    got = result.storage.value.double()
    assert (got - expected.values()).abs().max() <= 1e-5 * expected.values().abs().max()


def test_the_prebuilt_coo_product_is_bounded_by_its_stored_entries():
    extent = 1000000
    one = [torch.tensor([extent - 1], dtype=torch.int32)]
    result = scorch_ops.spmspm_coo_float(
        [extent, extent],
        [extent, 3],
        [one, [torch.tensor([2], dtype=torch.int32)]],
        torch.tensor([2.0]),
        [3, extent],
        [[torch.tensor([2], dtype=torch.int32)], one],
        torch.tensor([4.0]),
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [extent - 1]
    assert result.storage.index.mode_indices[1][0].tolist() == [extent - 1]
    assert result.storage.value.tolist() == [8.0]


def test_a_csr_product_refuses_one_row_too_many_with_the_code():
    """A CSR result of M rows needs M + 1 row offsets.  Shapes alone: the bound is checked
    before the arrays are."""
    empty = [[], [torch.tensor([0], dtype=torch.int32), _int32([])]]
    with pytest.raises(RuntimeError) as caught:
        scorch_ops.spmspm_csr_float(
            [M, 4], [M, 4], empty, torch.zeros(0), [4, 4], empty, torch.zeros(0)
        )
    assert _native_line(caught.value) == (
        f"spmspm_csr_float: result_shape [{M}, 4]: compressed level 1 would have {M} "
        "parents and one position entry more than that; a 32-bit position addresses at "
        f"most {M} entries"
    )


def test_the_kernels_with_a_dense_result_keep_the_element_bound_with_the_code():
    positions = torch.tensor([0, 2, 3], dtype=torch.int32)
    coordinates = torch.tensor([0, 2, 1], dtype=torch.int32)
    args = (
        [2**16, 2**16],
        [2, 3],
        [[], [positions, coordinates]],
        torch.tensor([1.0, 2.0, 3.0]),
        [3, 2],
        [[], []],
        torch.arange(6, dtype=torch.float32),
    )
    with pytest.raises(RuntimeError, match=OLD_BOUND) as caught:
        scorch_ops.spmm_csr_float(*args)
    assert NATIVE_CODE in str(caught.value)
    with pytest.raises(IndexCapacityError) as mapped:
        raise_if_native_coded(caught.value)
    assert_capacity_error(mapped.value)
    assert OLD_BOUND in str(mapped.value)


# --------------------------------------------------------------------------------------
# Generated kernels and the public operations at 64,000 x 64,000 (4.1e9 cells).
# --------------------------------------------------------------------------------------

N = 64000


def _csr_operand(per_row, seed, dtype=torch.float64, rows=N, cols=N):
    """``per_row`` distinct columns in every row, as a torch CSR tensor."""
    generator = torch.Generator().manual_seed(seed)
    col = torch.randint(0, cols, (rows, per_row * 2), generator=generator)
    col = torch.sort(col, dim=1).values
    keep = torch.ones_like(col, dtype=torch.bool)
    keep[:, 1:] = col[:, 1:] != col[:, :-1]
    counts = keep.sum(1)
    crow = torch.zeros(rows + 1, dtype=torch.int64)
    crow[1:] = torch.cumsum(counts, 0)
    col = col[keep]
    values = torch.rand(col.numel(), generator=generator, dtype=torch.float64) + 0.5
    return torch.sparse_csr_tensor(crow, col, values.to(dtype), size=(rows, cols))


def _entries(tensor):
    """(row, column, value) of every stored entry of a rank-2 STensor, read from its
    level arrays, in storage order."""
    letters = str(tensor.format).replace(",", "")
    levels = tensor.storage._mode_indices
    values = tensor.values
    if letters == "ds":
        positions, coordinates = levels[1]
        rows = torch.repeat_interleave(
            torch.arange(tensor.shape[0]), torch.diff(positions.long())
        )
        return rows, coordinates.long(), values
    if letters == "ss":
        outer_positions, outer = levels[0]
        positions, coordinates = levels[1]
        assert outer_positions.tolist() == [0, outer.numel()]
        rows = torch.repeat_interleave(outer.long(), torch.diff(positions.long()))
        return rows, coordinates.long(), values
    assert letters == "oo", letters
    return levels[0][0].long(), levels[1][0].long(), values


def _assert_entries(tensor, rows, cols, values):
    got_rows, got_cols, got_values = _entries(tensor)
    assert torch.equal(got_rows, rows)
    assert torch.equal(got_cols, cols)
    assert got_values.dtype == values.dtype
    scale = values.abs().max()
    assert (got_values - values).abs().max() <= 1e-12 * scale


def _rows_of(csr):
    return torch.repeat_interleave(
        torch.arange(csr.shape[0]), torch.diff(csr.crow_indices())
    )


@pytest.mark.parametrize("result_format", ["ds", "ss", "oo"])
def test_sddmm_on_a_64000_square_mask(result_format):
    rank = 16
    mask = _csr_operand(8, seed=11)
    generator = torch.Generator().manual_seed(12)
    a = torch.rand(N, rank, generator=generator, dtype=torch.float64)
    b = torch.rand(N, rank, generator=generator, dtype=torch.float64)
    result = scorch.einsum(
        "ij,ik,jk->ij", STensor.from_csr(mask), a, b, format=result_format
    )
    assert tuple(result.shape) == (N, N)
    rows, cols = _rows_of(mask), mask.col_indices()
    expected = mask.values() * (a[rows] * b[cols]).sum(1)
    _assert_entries(result, rows, cols, expected)


def test_an_elementwise_product_of_two_64000_square_matrices():
    left = _csr_operand(8, seed=21)
    # Half of the right operand's entries are the left's, so the product is not empty.
    shared = _csr_operand(8, seed=21)
    other = _csr_operand(8, seed=22)
    right = (shared.to_sparse_coo() * 2.0 + other.to_sparse_coo()).coalesce().to_sparse_csr()
    result = scorch.einsum(
        "ij,ij->ij", STensor.from_csr(left), STensor.from_csr(right), format="ds"
    )
    product = (left.to_sparse_coo() * right.to_sparse_coo()).coalesce()
    keep = product.values() != 0
    _assert_entries(
        result, product.indices()[0][keep], product.indices()[1][keep], product.values()[keep]
    )


def test_matrix_times_dense_and_matrix_times_vector_at_64000_square():
    matrix = _csr_operand(8, seed=31)
    generator = torch.Generator().manual_seed(32)
    dense = torch.rand(N, 16, generator=generator, dtype=torch.float64)
    vector = torch.rand(N, generator=generator, dtype=torch.float64)
    product = scorch.einsum("ij,jk->ik", STensor.from_csr(matrix), dense, format="dd")
    expected = torch.sparse.mm(matrix, dense)
    assert (product.to_torch() - expected).abs().max() <= 1e-12 * expected.abs().max()
    image = scorch.einsum("ij,j->i", STensor.from_csr(matrix), vector, format="d")
    expected = torch.sparse.mm(matrix, vector.reshape(-1, 1)).reshape(-1)
    assert (image.to_torch() - expected).abs().max() <= 1e-12 * expected.abs().max()


def test_a_sum_of_two_64000_square_matrices():
    left = _csr_operand(8, seed=41)
    right = _csr_operand(8, seed=42)
    result = STensor.from_csr(left) + STensor.from_csr(right)
    assert str(result.format) == "d,s"
    expected = (left.to_sparse_coo() + right.to_sparse_coo()).coalesce()
    _assert_entries(
        result, expected.indices()[0], expected.indices()[1], expected.values()
    )


@pytest.mark.parametrize("target", ["ss", "oo"])
def test_a_conversion_of_a_64000_square_matrix(target):
    matrix = _csr_operand(8, seed=51)
    converted = STensor.from_csr(matrix).to_sparse(target)
    _assert_entries(converted, _rows_of(matrix), matrix.col_indices(), matrix.values())


# --------------------------------------------------------------------------------------
# What stays refused, with the code, before any kernel is built.
# --------------------------------------------------------------------------------------


def _kernel_sources():
    """Every kernel source written so far.  A build writes its ``main.cpp`` into the
    extensions directory before it compiles (tests/conftest.py gives each session its
    own), whichever entry started it, so an unchanged list means nothing was built."""
    return sorted(Path(os.environ["TORCH_EXTENSIONS_DIR"]).rglob("main.cpp"))


def test_a_dense_result_past_the_bound_is_refused_before_a_kernel_is_built():
    before = _kernel_sources()
    left = _csr_operand(1, seed=61, rows=46341, cols=4)
    right = _csr_operand(1, seed=62, rows=4, cols=46341)
    with pytest.raises(IndexCapacityError) as caught:
        scorch.einsum(
            "ik,kj->ij", STensor.from_csr(left), STensor.from_csr(right), format="dd"
        )
    assert_capacity_error(caught.value)
    assert "result_shape [46341, 46341]" in str(caught.value)
    assert _kernel_sources() == before


def test_the_same_result_stored_sparsely_is_computed():
    left = _csr_operand(1, seed=61, rows=46341, cols=4)
    right = _csr_operand(1, seed=62, rows=4, cols=46341)
    result = scorch.einsum(
        "ik,kj->ij", STensor.from_csr(left), STensor.from_csr(right), format="ds"
    )
    expected = torch.sparse.mm(left.to_sparse_coo(), right.to_sparse_coo()).coalesce()
    _assert_entries(
        result, expected.indices()[0], expected.indices()[1], expected.values()
    )


def test_a_dense_block_result_is_bounded_where_each_block_opens(monkeypatch):
    """``sd``: the kernel grows a dense block under each stored row, and checks each
    block where it opens (``scorch_check_dense_block``, patch 0046), so the entry check
    counts one block and not every row the result could store: a 46,341 x 46,341 result
    is emitted, its column level of the kind ``DenseBoundedInKernel`` (3).  The block
    that would cross the bound is refused by the kernel where it opens
    (tests/test_scorch/test_dense_block_position_check.py); this product, run, would
    store 2**31 values before that block, so it is only emitted here."""
    source = emitted_einsum_cpp(
        monkeypatch,
        "ik,kj->ij",
        ["ds", "ds"],
        {"i": 46341, "k": 4, "j": 46341},
        format="sd",
    )
    assert "scorch_check_dense_block(" in source
    assert re.search(
        r"validate_jit_result_shape\(result_shape, \{[^}]*\}, \{1, 3\}, ", source
    )


# --------------------------------------------------------------------------------------
# An operand whose moved layout would start with a dense level of 2**31 - 1 positions.
# --------------------------------------------------------------------------------------


def _peak_rss_gb():
    """The process's peak resident set in GiB.  getrusage reports ``ru_maxrss`` in bytes
    on macOS and in KiB on Linux (getrusage(2) on each)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**30 if sys.platform == "darwin" else 2**20)


@pytest.mark.parametrize(
    "platform, ru_maxrss", [("darwin", 3 * 2**30), ("linux", 3 * 2**20)]
)
def test_peak_rss_reads_each_platforms_unit(monkeypatch, platform, ru_maxrss):
    class _Usage:
        pass

    usage = _Usage()
    usage.ru_maxrss = ru_maxrss
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(resource, "getrusage", lambda who: usage)
    assert _peak_rss_gb() == 3.0


def _wide_row():
    return STensor.from_components(
        (1, M),
        "ds",
        [[], [_int32([0, 2]), _int32([0, M - 1])]],
        torch.tensor([3.0, 5.0], dtype=torch.float64),
    )


def test_einsum_converts_an_operand_it_cannot_move_into_a_dense_level():
    """``ik,i->i`` with A in "ds" of shape (1, 2**31 - 1) and two stored entries: moving A
    to its column order with its letters kept would allocate 2**31 position entries (8 GB)
    for a level no kernel can address.  It is moved with every level compressed, which
    holds it in two positions, and the product is computed."""
    before = _peak_rss_gb()
    weight = torch.tensor([2.0], dtype=torch.float64)
    result = scorch.einsum("ik,i->i", _wide_row(), weight, format="d")
    assert result.to_torch().tolist() == [16.0]
    result = scorch.einsum("ik,i->i", _wide_row(), weight, format="s")
    assert result.values.tolist() == [16.0]
    assert _peak_rss_gb() - before < 1.0


def test_change_mode_order_refuses_an_unaddressable_layout_before_allocating():
    before = _peak_rss_gb()
    with pytest.raises(IndexCapacityError) as caught:
        _wide_row().change_mode_order([1, 0])
    assert_capacity_error(caught.value)
    assert "change_mode_order" in str(caught.value)
    assert _peak_rss_gb() - before < 1.0
    # A compressed first level over the columns holds the same tensor in two positions.
    moved = _wide_row().to_sparse("ss").change_mode_order([1, 0])
    assert moved.values.tolist() == [3.0, 5.0]
