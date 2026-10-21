"""The 32-bit position bound on a result's dense block, checked where the block opens.

A dense block is a run of dense result levels below a compressed level (``sd``, ``sdd``,
``sds``, ``dsd``): each coordinate the compressed level stores owns one block of cells,
the product of the run's extents, at consecutive positions from the coordinate's position
times that product.  A kernel narrows those positions to ``int`` as it forms them, before
it stores anything.  The entry check used to bound them the only way it could before the
kernel ran: by counting the compressed level by its extent, as if every coordinate were
stored, which refused results whose stored entries fit (``sd`` at 64,000 x 64,000).

Now each kernel checks every block where it opens (csrc/header.h,
``scorch_check_dense_block``), and its entry check gives the block's levels the kind
``DenseBoundedInKernel``: one block must fit, and nothing else about them is counted.  A
block whose end would pass the bound is refused when it opens, with the same code
``index_position_exceeds_int32``.

1. The rule with the new kind: Python and native, one table, one sweep of layouts.
2. The kernel's check alone.
3. The ABI object: the record of checked blocks, the form before the body is built, and
   the check of the record against the finished body.
4. Every program of the dense-below-compressed table, in both compilers: an emitted entry
   check gives a level the new kind exactly when the body checks its block, and every
   block is checked.  Read from the kernels' LLIR, never from their C++ text.
5. Both compilers run results the old rule refused and whose stored blocks fit, against
   float64 references built from the stored entries; with ``SCORCH_BIGMEM_TESTS=1``, the
   two sides of the bound itself, one block apart.
"""

from __future__ import annotations

import hashlib
import itertools
import os
from functools import lru_cache
from typing import List, Tuple

import pytest
import torch
from torch.utils.cpp_extension import load_inline

import scorch
from scorch import ops
from scorch.compiler import llir
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.llir_traversal import LLIRPath, LLIRTraversalContext, LLIRWalker
from scorch.compiler.loopir import lower_llir
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.torch_cpp_abi import (
    DENSE_BLOCK_CHECK,
    DenseBlockCheckError,
    TorchCppKernelABI,
    _refuse_unaddressable_result,
    dense_block_check,
    dense_blocks,
    levels_bounded_by_checks,
)
from scorch.exceptions import IndexCapacityError, raise_if_native_coded
from scorch.format import LevelType
from scorch.layout import TensorSpec
from scorch.stensor import STensor
from tests.test_scorch.test_dense_below_compressed_result import CELLS, EXTENT, _cell_id
from tests.test_scorch.test_index_position_bound import (
    M,
    NATIVE_CODE,
    _memory_gb,
    _native,
    _native_line,
    _refuses,
    assert_capacity_error,
)
from tests.test_scorch.test_loopir_dense_below_compressed_target import _statement
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

_CSRC = os.path.join(os.path.dirname(__file__), "..", "..", "src", "scorch", "csrc")
_KIND = {
    "d": LevelType.DENSE,
    "D": LevelType.DENSE,
    "s": LevelType.COMPRESSED,
    "o": LevelType.COORDINATE,
}
_KIND_CODE = {"d": 0, "s": 1, "o": 2, "D": 3}


def _levels(letters):
    return tuple(_KIND[letter] for letter in letters)


def _bounded(letters):
    """``D`` is a dense level of a block the kernel bounds itself."""
    return frozenset(level for level, letter in enumerate(letters) if letter == "D")


# --------------------------------------------------------------------------------------
# 1. The rule.  ``True`` = refused.  ``D`` marks the levels of blocks a kernel bounds.
# --------------------------------------------------------------------------------------

BLOCK_TABLE = [
    # One block must fit; the compressed level above counts nothing.
    ("sD", (M, M), False),
    ("sD", (64000, 64000), False),
    ("sD", (1, M), False),
    ("sDD", (5, 46340, 46340), False),
    ("sDD", (5, 46341, 46341), True),
    ("sDD", (5, M, 2), True),
    ("sDD", (5, 0, M), False),
    ("sDD", (5, 46341, 0), False),
    # The kernel forms M x 2 positions before it reaches the empty level.
    ("sDDD", (5, M, 2, 0), True),
    # A compressed level below the block has one position entry more than its cells.
    ("sDs", (5, M, 7), True),
    ("sDs", (5, M - 1, 7), False),
    ("sDDs", (5, 46340, 46340, 7), False),
    # A dense prefix above keeps its own rule.
    ("dsD", (M - 1, 1, M), False),
    ("dsD", (M, 1, 1), True),  # level 1 would have M parents
    ("ssD", (M, M, M), False),
    # Two blocks, each held by the kernel.
    ("sDsD", (M, 46341, M, 46341), False),
    # One block the kernel does not bound: its dense level counts everything above it.
    ("sdsD", (M, 46341, 5, 46341), True),
    ("sDsd", (3, 46341, 5, 46341), True),
    ("sDsd", (3, 5, 7, 11), False),
]


@pytest.mark.parametrize("letters, shape, refused", BLOCK_TABLE)
def test_the_rule_with_blocks_bounded_in_the_kernel(letters, shape, refused):
    def check():
        _refuse_unaddressable_result(
            shape, _levels(letters), "evaluate", _bounded(letters)
        )

    if not refused:
        check()
        return
    with pytest.raises(IndexCapacityError) as caught:
        check()
    assert_capacity_error(caught.value)
    assert str(caught.value).startswith(f"evaluate: result_shape {list(shape)}: ")


def test_without_the_kernels_bound_the_old_rule_refuses_what_the_new_kind_admits():
    bounded = _bounded("sD")
    _refuse_unaddressable_result((64000, 64000), _levels("sd"), "evaluate", bounded)
    with pytest.raises(IndexCapacityError):
        _refuse_unaddressable_result((64000, 64000), _levels("sd"), "evaluate")


@pytest.mark.parametrize("letters, shape, refused", BLOCK_TABLE)
@pytest.mark.parametrize("known", [False, True])
def test_the_native_rule_with_blocks_is_the_python_one(letters, shape, refused, known):
    native = _native()
    kinds = [_KIND_CODE[letter] for letter in letters]
    check = native.known_result_positions if known else native.result_positions
    expected = list(shape) if known else []
    assert native.result_positions_fit(list(shape), expected, kinds) == (not refused)
    assert _refuses(native, shape, expected, kinds) == refused
    if not refused:
        check(list(shape), kinds)
        return
    with pytest.raises(RuntimeError) as caught:
        check(list(shape), kinds)
    with pytest.raises(IndexCapacityError) as python:
        _refuse_unaddressable_result(
            shape, _levels(letters), "evaluate", _bounded(letters)
        )
    assert _native_line(caught.value) == str(python.value)
    with pytest.raises(IndexCapacityError) as mapped:
        raise_if_native_coded(caught.value)
    assert str(mapped.value) == str(python.value)


def _well_formed(kinds):
    """Whether every level of kind 3 sits where a block can: below a compressed level
    or another level of kind 3, and above one of those or nothing."""
    for level, kind in enumerate(kinds):
        if kind != 3:
            continue
        if level == 0 or kinds[level - 1] not in (1, 3):
            return False
        if level + 1 < len(kinds) and kinds[level + 1] not in (1, 3):
            return False
    return True


def test_the_three_forms_of_the_rule_agree_on_every_layout_with_the_new_kind():
    """Ranks 0 to 3, every layout over the four level kinds, every combination of the
    extents below, compiled for the shape and for the rank only.  The native yes-or-no
    form says yes exactly where the worded form raises nothing.  On a layout where kind 3
    is placed as a block, Python refuses exactly the same calls; anywhere else the
    contract is malformed, and both native forms refuse it without the capacity code."""
    native = _native()
    extents = (0, 1, 3, 46341, M - 1, M)
    checked = refused = malformed = 0
    for rank in range(4):
        for kinds in itertools.product((0, 1, 2, 3), repeat=rank):
            letters = "".join("dsoD"[kind] for kind in kinds)
            for shape in itertools.product(extents, repeat=rank):
                for expected in ((), shape):
                    fits = native.result_positions_fit(
                        list(shape), list(expected), list(kinds)
                    )
                    raised = _refuses(native, shape, expected, kinds)
                    assert fits == (not raised), (kinds, shape, expected)
                    checked += 1
                    refused += raised
                    if not _well_formed(kinds):
                        assert raised, (kinds, shape)
                        malformed += 1
                        continue
                    try:
                        _refuse_unaddressable_result(
                            shape, _levels(letters), "evaluate", _bounded(letters)
                        )
                        python_refused = False
                    except IndexCapacityError:
                        python_refused = True
                    assert python_refused == raised, (kinds, shape, expected)
    # 2 x (1 + 4 x 6 + 16 x 36 + 64 x 216) calls.
    assert checked == 28850
    assert 1000 < refused - malformed < checked - malformed - 1000
    with pytest.raises(RuntimeError, match="unsupported level kind at 1") as caught:
        native.result_positions([3, 4], [0, 3])  # below a dense level
    assert NATIVE_CODE not in str(caught.value)
    with pytest.raises(RuntimeError, match="unsupported level kind at 0"):
        native.result_positions([3, 4], [3, 1])  # the first level
    with pytest.raises(RuntimeError, match="unsupported level kind at 1"):
        native.result_positions([3, 4, 5], [1, 3, 0])  # above a dense level
    with pytest.raises(RuntimeError, match="unsupported level kind at 1"):
        native.result_positions([3, 4], [2, 3])  # below a coordinate level


# --------------------------------------------------------------------------------------
# 2. The kernel's check, where a block opens.
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _kernel_check():
    source = r"""
#include "header.h"

void dense_block(int64_t parent_position, int64_t block, bool compressed_below,
                 int64_t first_level, int64_t last_level) {
  scorch_check_dense_block(parent_position, block, compressed_below,
                           static_cast<int>(first_level), static_cast<int>(last_level));
}
"""
    digest = hashlib.sha256(
        open(os.path.join(_CSRC, "header.h"), "rb").read()
        + open(os.path.join(_CSRC, "native_abi.h"), "rb").read()
        + source.encode()
    ).hexdigest()[:12]
    return load_inline(
        name=f"scorch_dense_block_check_{digest}",
        cpp_sources=source,
        functions=["dense_block"],
        extra_include_paths=[os.path.abspath(_CSRC)],
        extra_cflags=["-O0"],
        verbose=False,
    )


@pytest.mark.parametrize(
    "parent_position, block, compressed_below, refused",
    [
        (0, M, False, False),  # one block of every position
        (0, M, True, True),  # and the compressed level below needs one entry more
        (M - 1, 1, False, False),
        (M - 1, 1, True, True),
        (M - 2, 1, True, False),
        (2046, 2**20, False, False),  # 2,146,435,072
        (2047, 2**20, False, True),  # 2**31
        (M - 1, 0, True, False),  # an empty block never passes the bound
        (M - 1, M, False, True),  # about 2**62: exact in int64
    ],
)
def test_the_kernel_refuses_a_block_whose_end_passes_the_bound(
    parent_position, block, compressed_below, refused
):
    check = _kernel_check().dense_block
    if not refused:
        check(parent_position, block, compressed_below, 1, 2)
        return
    with pytest.raises(RuntimeError) as caught:
        check(parent_position, block, compressed_below, 1, 2)
    with pytest.raises(IndexCapacityError) as mapped:
        raise_if_native_coded(caught.value)
    assert_capacity_error(mapped.value)


def test_the_kernels_refusal_names_the_block_and_where_it_would_end():
    check = _kernel_check().dense_block
    with pytest.raises(RuntimeError) as caught:
        check(2047, 2**20, False, 1, 2)
    assert _native_line(caught.value).endswith(
        "result dense levels 1 to 2: the block of 1048576 cells under position 2047 "
        "of compressed level 0 would end at position 2147483648; a 32-bit position "
        "addresses at most 2147483647 entries"
    )
    with pytest.raises(RuntimeError) as caught:
        check(0, M, True, 1, 1)
    assert _native_line(caught.value).endswith(
        "result dense levels 1 to 1: the block of 2147483647 cells under position 0 "
        "of compressed level 0 would end at position 2147483647, and compressed level "
        "2 below it would have one position entry more; a 32-bit position addresses "
        "at most 2147483647 entries"
    )


# --------------------------------------------------------------------------------------
# 3. The ABI object and the record.
# --------------------------------------------------------------------------------------


def _abi(letters, shape, bounded=()):
    return TorchCppKernelABI(
        result_shape=shape,
        result_level_types=_levels(letters),
        input_tensors=(),
        result_levels_bounded_in_kernel=bounded,
    )


def test_the_dense_blocks_of_a_layout():
    assert dense_blocks(_levels("sd")) == ((1, 1),)
    assert dense_blocks(_levels("sdd")) == ((1, 2),)
    assert dense_blocks(_levels("sds")) == ((1, 1),)
    assert dense_blocks(_levels("dsd")) == ((2, 2),)
    assert dense_blocks(_levels("sdsdd")) == ((1, 1), (3, 4))
    # A dense prefix is not a block, and neither is a run above a coordinate level.
    assert dense_blocks(_levels("dds")) == ()
    assert dense_blocks(_levels("sdo")) == ()
    assert dense_blocks(_levels("odd")) == ()
    assert dense_blocks(()) == ()


def test_before_the_body_is_built_the_abi_refuses_only_what_no_body_could_address():
    pending = _abi("sd", (64000, 64000), None)
    with pytest.raises(ValueError, match="not known until the body is built"):
        pending.emit_validation()
    with pytest.raises(ValueError, match="not known until the body is built"):
        pending.require_bounded_levels_checked([])
    checked = pending.with_levels_bounded_in_kernel({1})
    assert LLIRLowerer().lower_llir(checked.emit_validation()[:1]) == (
        "scorch_native::validate_jit_result_shape(result_shape, {64000, 64000}, "
        '{1, 3}, "evaluate");'
    )
    with pytest.raises(IndexCapacityError):
        pending.with_levels_bounded_in_kernel(())
    with pytest.raises(IndexCapacityError):
        _abi("sdd", (5, 46341, 46341), None)


def test_the_abi_takes_whole_blocks_only():
    with pytest.raises(ValueError, match="one dense block"):
        _abi("sdd", (5, 4, 3), (1,))
    with pytest.raises(ValueError, match="not in a dense block"):
        _abi("dd", (5, 4), (1,))
    with pytest.raises(ValueError, match="not in a dense block"):
        _abi("sdo", (5, 4, 3), (1,))
    with pytest.raises(TypeError, match="sorted tuple"):
        _abi("sdsd", (5, 4, 3, 2), (3, 1))
    with pytest.raises(TypeError, match="sorted tuple"):
        _abi("sd", (5, 4), [1])  # type: ignore[arg-type]
    assert _abi("sdsd", (5, 4, 3, 2), (1, 3)).result_levels_bounded_in_kernel == (1, 3)


def test_the_check_statement_and_its_record():
    record: set = set()
    check = dense_block_check(record, "C", _levels("sdds"), 1)
    assert record == {1, 2}
    assert check.name == DENSE_BLOCK_CHECK
    assert LLIRLowerer().lower_llir([check]) == (
        "scorch_check_dense_block(pC0, C1_size * C2_size, true, 1, 2);"
    )
    at = llir.Var("pC0_at", llir.DataType.INT64)
    leaf = dense_block_check(record, "C", _levels("sd"), 1, at)
    assert LLIRLowerer().lower_llir([leaf]) == (
        "scorch_check_dense_block(pC0_at, C1_size, false, 1, 1);"
    )
    for letters, first in (("sdd", 2), ("dd", 1), ("sdo", 1)):
        with pytest.raises(ValueError, match="does not open a dense block"):
            dense_block_check(set(), "C", _levels(letters), first)
    assert levels_bounded_by_checks([check, llir.BlankLine()]) == frozenset({1, 2})


def test_the_finished_body_must_check_exactly_the_blocks_the_entry_check_declares():
    abi = _abi("sdds", (5, 4, 3, 2), (1, 2))
    check = dense_block_check(set(), "C", _levels("sdds"), 1)
    abi.require_bounded_levels_checked(abi.assemble_function([check]))
    with pytest.raises(DenseBlockCheckError) as caught:
        abi.require_bounded_levels_checked(abi.assemble_function([]))
    assert caught.value.code == "dense_block_check_mismatch"
    unchecked = _abi("sdds", (5, 4, 3, 2))
    with pytest.raises(DenseBlockCheckError):
        unchecked.require_bounded_levels_checked(unchecked.assemble_function([check]))
    malformed = llir.FunctionCallStmt(
        name=DENSE_BLOCK_CHECK,
        args=[
            llir.Var("pC0", llir.DataType.INT64),
            llir.Var("C1_size", llir.DataType.INT64),
            llir.Literal(True, llir.DataType.BOOL),
            llir.Var("first", llir.DataType.INT),
            llir.Literal(2, llir.DataType.INT),
        ],
    )
    with pytest.raises(DenseBlockCheckError, match="two integer literals"):
        levels_bounded_by_checks([malformed])


# --------------------------------------------------------------------------------------
# 4. Every program of the table, in both compilers, from the kernels' LLIR.
# --------------------------------------------------------------------------------------

_WALK = LLIRTraversalContext(stage="test", pass_name="result_check_kinds")


class _ResultCheckKinds(LLIRWalker):
    def __init__(self) -> None:
        super().__init__(_WALK)
        self.kinds: List[Tuple[int, ...]] = []

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        if (
            type(node) is llir.FunctionCallStmt
            and node.name == "scorch_native::validate_jit_result_shape"
        ):
            array = node.args[2]
            assert type(array) is llir.Array
            self.kinds.append(tuple(literal.value for literal in array.values))


def _recorded_kernels(monkeypatch):
    """Every (ABI, finished function) pair either compiler checks, as it checks it."""
    recorded: list = []
    original = TorchCppKernelABI.require_bounded_levels_checked

    def record(self, body):
        original(self, body)
        recorded.append((self, body))

    monkeypatch.setattr(TorchCppKernelABI, "require_bounded_levels_checked", record)
    return recorded


# The new compiler's targets that check every dense block of their result where it
# opens.  Every other target declares none, so its result's blocks keep the entry
# check's rule, which counts the compressed level above a block by its extent.
_TARGETS_CHECKING_BLOCKS = frozenset({"_MultiCompressedAssemblyLowering"})
# The other targets that lower a program of this file with a dense block below a
# compressed level.  Strict: a target that starts checking must leave this table.
_TARGETS_LEAVING_BLOCKS_TO_THE_ENTRY_CHECK = frozenset(
    {
        "_DenseDomainMixedLowering",
        "_OrderedKeySparseWorkspaceLowering",
        "_RowScopeSparseWorkspaceLowering",
        "_TargetLowering",
    }
)


def _recorded_targets(monkeypatch):
    """The class of each new-compiler target whose ABI is asked for, in order."""
    targets: list = []
    original = lower_llir._TargetLowering.kernel_abi

    def record(self):
        targets.append(type(self).__name__)
        return original(self)

    monkeypatch.setattr(lower_llir._TargetLowering, "kernel_abi", record)
    return targets


def _entry_check_and_body(function):
    """The levels the kernel's entry check gives the new kind, and the levels whose
    blocks its body checks."""
    walker = _ResultCheckKinds()
    walker.walk(function)
    assert len(walker.kinds) == 1, walker.kinds
    declared = {level for level, kind in enumerate(walker.kinds[0]) if kind == 3}
    return declared, set(levels_bounded_by_checks(function))


def _block_levels(level_types):
    return {
        level
        for first, last in dense_blocks(level_types)
        for level in range(first, last + 1)
    }


@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_the_shipped_compiler_declares_exactly_the_blocks_it_checks(monkeypatch, cell):
    expr, operand_fmts, result_fmt = cell
    recorded = _recorded_kernels(monkeypatch)
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )
    terms = expr.split("->")[0].split(",")
    specs = [
        TensorSpec(
            fmt,
            tuple(EXTENT[letter] for letter in term),
            dtype=torch.float32,
            name="ABD"[position],
        )
        for position, (term, fmt) in enumerate(zip(terms, operand_fmts))
    ]
    ops.einsum(expr, *specs, compile_only=True, format=result_fmt)
    assert recorded, "no kernel was lowered"
    for abi, body in recorded:
        declared, checked = _entry_check_and_body(body)
        assert declared == checked, (abi.result_level_types, declared, checked)
        # Every block is checked where it opens: none is left to the entry check.
        blocks = _block_levels(abi.result_level_types)
        assert checked == blocks, (abi.result_level_types, checked, blocks)
    if _block_levels(_levels(result_fmt)):
        lowered = [_block_levels(abi.result_level_types) for abi, _ in recorded]
        assert any(lowered), "no kernel lowered has the result's block"


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_the_new_compiler_declares_exactly_the_blocks_it_checks(monkeypatch, cell, arm):
    expr, operand_fmts, result_fmt = cell
    recorded = _recorded_kernels(monkeypatch)
    targets = _recorded_targets(monkeypatch)
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    compile_cin_via_loopir(
        _statement(expr, operand_fmts, result_fmt),
        tuple(EXTENT[letter] for letter in output),
        tuple(
            (tuple(EXTENT[letter] for letter in term), torch.float32) for term in terms
        ),
        compile_options=auto_options(arm),
    )
    assert recorded and targets, "no kernel was lowered"
    target = targets[-1]
    for abi, body in recorded:
        declared, checked = _entry_check_and_body(body)
        assert declared == checked, (target, declared, checked)
        blocks = _block_levels(abi.result_level_types)
        if target in _TARGETS_CHECKING_BLOCKS:
            assert checked == blocks, (target, checked, blocks)
            continue
        assert not checked, f"{target} checks its blocks: take it out of the table"
        leaves = target in _TARGETS_LEAVING_BLOCKS_TO_THE_ENTRY_CHECK
        assert leaves or not blocks, (target, blocks)


# --------------------------------------------------------------------------------------
# 5. Execution in both compilers.
# --------------------------------------------------------------------------------------

ROWS = 100_000
STORED_ROWS = (5, ROWS - 1)


def _sparse_vector(extent, stored, values, dtype):
    return STensor.from_components(
        (extent,),
        "s",
        [[torch.tensor([0, len(stored)]), torch.tensor(stored, dtype=torch.int64)]],
        torch.tensor(values, dtype=dtype),
    )


def _dense(tensor):
    return STensor.from_components(
        tuple(tensor.shape),
        "d" * tensor.dim(),
        [[] for _ in range(tensor.dim())],
        tensor.reshape(-1).clone(),
    )


def _rows_matrix(columns, stored_columns, dtype=torch.float64):
    """A ``ss`` matrix of ROWS rows storing ``stored_columns`` in each of STORED_ROWS."""
    count = len(STORED_ROWS)
    values = torch.arange(1, count * len(stored_columns) + 1, dtype=dtype) / 7
    return (
        STensor.from_components(
            (ROWS, columns),
            "ss",
            [
                [torch.tensor([0, count]), torch.tensor(STORED_ROWS)],
                [
                    torch.arange(count + 1) * len(stored_columns),
                    torch.tensor(list(stored_columns) * count),
                ],
            ],
            values,
        ),
        values.reshape(count, len(stored_columns)),
    )


def _cin(expr, operand_fmts, result_fmt, dtype):
    """``expr`` as CIN, loops in order of first appearance, values of ``dtype``."""
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    letters = "".join(dict.fromkeys("".join(terms)))
    variables = {letter: IndexVar(letter) for letter in letters}
    product = None
    for name, term, fmt in zip("ABD", terms, operand_fmts):
        access = TensorVar(name, fmt=fmt, dtype=dtype)[
            tuple(variables[letter] for letter in term)
        ]
        product = (
            access if product is None else CINBinaryOp(Operation.MUL, product, access)
        )
    result = TensorVar("C", fmt=result_fmt, dtype=dtype)[
        tuple(variables[letter] for letter in output)
    ]
    statement = TensorAssign(
        result, product, op=Operation.ADD if set(letters) - set(output) else None
    )
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def _new_compiler_target(monkeypatch, statement, shape, operands, arm):
    """The class of the new compiler's target for a program, from a lowering that
    builds nothing.  A refusal for the position bound at emission still names it."""
    with monkeypatch.context() as patch:
        targets = _recorded_targets(patch)
        try:
            compile_cin_via_loopir(
                statement,
                shape,
                tuple(
                    (tuple(operand.shape), operand.values.dtype) for operand in operands
                ),
                compile_options=auto_options(arm),
            )
        except IndexCapacityError:
            pass
    assert targets, "no target was lowered"
    return targets[-1]


def _first(result):
    return result[0] if isinstance(result, tuple) else result


def _runs(
    monkeypatch, expr, operands, operand_fmts, result_fmt, shape, dtype=torch.float64
):
    """Each compiler's run and whether it checks the result's blocks itself: the
    shipped compiler, then the new one on both arms.  A new-compiler target that
    leaves the blocks to the entry check (``_TARGETS_LEAVING_...``) still refuses at
    emission a shape past the entry check's rule, as before the blocks were checked."""

    def shipped():
        return scorch.einsum(expr, *operands, format=result_fmt)

    yield "shipped", shipped, True
    for arm in (False, True):
        statement = _cin(expr, operand_fmts, result_fmt, dtype)
        target = _new_compiler_target(monkeypatch, statement, shape, operands, arm)
        checks = target in _TARGETS_CHECKING_BLOCKS
        assert checks or target in _TARGETS_LEAVING_BLOCKS_TO_THE_ENTRY_CHECK, target

        def new(arm=arm):
            return _first(
                execute_cin_via_loopir(
                    _cin(expr, operand_fmts, result_fmt, dtype),
                    shape,
                    *operands,
                    compile_options=auto_options(arm, jit=True),
                )
            )

        yield f"new-{'blocked' if arm else 'plain'} {target}", new, checks


def _assert_left_to_the_entry_check(name, run):
    with pytest.raises(IndexCapacityError) as caught:
        run()
    assert_capacity_error(caught.value)
    assert "the block of" not in str(caught.value), name


def _assert_rows(result, result_fmt, row_values, below=None, bound=None):
    """The result stores STORED_ROWS at level 0, each with its block ``row_values[r]``.

    ``below``: the coordinates a compressed level under the block stores in each cell.
    ``bound``: for a sum of products, how far each cell may be from the reference: the
    rounding of its terms, which a cell near zero can exceed relative to itself.
    """
    assert str(result.format) == ",".join(result_fmt)
    order = tuple(int(mode) for mode in result.mode_order)
    assert order == tuple(range(len(result_fmt)))
    levels = result.storage.index.mode_indices
    positions, coordinates = levels[0]
    assert positions.tolist() == [0, len(STORED_ROWS)]
    assert coordinates.tolist() == list(STORED_ROWS)
    expected = row_values.reshape(-1)
    if below is not None:
        segments = expected.numel() // len(below)
        positions, coordinates = levels[-1]
        assert positions.tolist() == list(range(0, expected.numel() + 1, len(below)))
        assert coordinates.tolist() == list(below) * segments
    actual = result.storage.value.reshape(-1)
    if bound is None:
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-12)
    else:
        assert bool(((actual - expected).abs() <= bound.reshape(-1)).all())


def _each_result(runs):
    """Each run's result where its compiler checks the result's blocks; a run whose
    target leaves them to the entry check is refused at emission instead."""
    for name, run, checks in runs:
        if checks:
            yield name, run()
        else:
            _assert_left_to_the_entry_check(name, run)


def test_both_compilers_store_a_block_the_old_rule_refused(monkeypatch):
    """``sd`` of 100,000 x 30,000 (3 x 10**9 cells): two stored rows of 30,000."""
    columns = 30_000
    b = _sparse_vector(ROWS, list(STORED_ROWS), [1.5, -2.25], torch.float64)
    c = torch.linspace(-3.0, 5.0, columns, dtype=torch.float64)
    with pytest.raises(IndexCapacityError):
        _refuse_unaddressable_result((ROWS, columns), _levels("sd"), "evaluate")
    expected = torch.outer(torch.tensor([1.5, -2.25], dtype=torch.float64), c)
    operands = (b, _dense(c))
    runs = _runs(monkeypatch, "i,j->ij", operands, ("s", "d"), "sd", (ROWS, columns))
    for name, result in _each_result(runs):
        _assert_rows(result, "sd", expected)


def test_both_compilers_store_a_block_of_two_dense_levels(monkeypatch):
    """``sdd`` of 100,000 x 200 x 150: one block is 30,000 cells."""
    b = _sparse_vector(ROWS, list(STORED_ROWS), [0.5, 3.0], torch.float64)
    c = torch.linspace(-1.0, 1.0, 200, dtype=torch.float64)
    d = torch.linspace(2.0, 4.0, 150, dtype=torch.float64)
    expected = torch.einsum(
        "i,j,k->ijk", torch.tensor([0.5, 3.0], dtype=torch.float64), c, d
    )
    operands = (b, _dense(c), _dense(d))
    runs = _runs(
        monkeypatch, "i,j,k->ijk", operands, ("s", "d", "d"), "sdd", (ROWS, 200, 150)
    )
    for name, result in _each_result(runs):
        _assert_rows(result, "sdd", expected)


def test_both_compilers_store_a_block_with_a_compressed_level_below(monkeypatch):
    """``sds`` of 100,000 x 30,000 x 10: each of the 60,000 cells owns a segment."""
    columns = 30_000
    b = _sparse_vector(ROWS, list(STORED_ROWS), [2.0, -1.0], torch.float64)
    c = torch.linspace(1.0, 2.0, columns, dtype=torch.float64)
    d = _sparse_vector(10, [3, 7], [0.25, 4.0], torch.float64)
    with pytest.raises(IndexCapacityError):
        _refuse_unaddressable_result((ROWS, columns, 10), _levels("sds"), "evaluate")
    expected = torch.einsum(
        "i,j,k->ijk",
        torch.tensor([2.0, -1.0], dtype=torch.float64),
        c,
        torch.tensor([0.25, 4.0], dtype=torch.float64),
    )
    operands = (b, _dense(c), d)
    runs = _runs(
        monkeypatch, "i,j,k->ijk", operands, ("s", "d", "s"), "sds", (ROWS, columns, 10)
    )
    for name, result in _each_result(runs):
        _assert_rows(result, "sds", expected, below=(3, 7))


def test_both_compilers_store_a_reduction_into_a_block(monkeypatch):
    """``ik,kj->ij`` into ``sd`` of 100,000 x 30,000, two stored rows of B."""
    columns = 30_000
    b, b_values = _rows_matrix(4, (0, 2))
    c = torch.linspace(-2.0, 2.0, 4 * columns, dtype=torch.float64).reshape(4, columns)
    rows = c[[0, 2]]
    expected = b_values @ rows
    # Each cell is a sum of two products; the reference and the kernel may each round
    # it differently, by at most 2 x 2 x eps x the sum of the products' magnitudes.
    bound = 4 * torch.finfo(torch.float64).eps * (b_values.abs() @ rows.abs())
    operands = (b, _dense(c))
    runs = _runs(
        monkeypatch, "ik,kj->ij", operands, ("ss", "dd"), "sd", (ROWS, columns)
    )
    for name, result in _each_result(runs):
        _assert_rows(result, "sd", expected, bound=bound)


# The bound itself, one block apart: 2,047 stored rows of 2**20 one-byte cells end at
# 2,146,435,072 positions; a 2,048th row's block would end at 2**31.
BOUND_ROWS, BOUND_COLUMNS = 4096, 2**20


@pytest.mark.skipif(
    os.environ.get("SCORCH_BIGMEM_TESTS") != "1" or _memory_gb() < 12,
    reason="stores 2 GB of values: needs SCORCH_BIGMEM_TESTS=1 and 12 GB of memory",
)
@pytest.mark.parametrize("stored, refused", [(2047, False), (2048, True)])
def test_both_compilers_refuse_the_block_that_crosses_the_bound_where_it_opens(
    monkeypatch, stored, refused
):
    b = _sparse_vector(BOUND_ROWS, list(range(stored)), [1] * stored, torch.int8)
    c = (torch.arange(BOUND_COLUMNS) % 7 - 3).to(torch.int8)
    operands = (b, _dense(c))
    shape = (BOUND_ROWS, BOUND_COLUMNS)
    runs = _runs(monkeypatch, "i,j->ij", operands, ("s", "d"), "sd", shape, torch.int8)
    for name, run, checks in runs:
        if not checks:
            _assert_left_to_the_entry_check(name, run)
            continue
        if refused:
            with pytest.raises(IndexCapacityError) as caught:
                run()
            assert_capacity_error(caught.value)
            assert (
                "the block of 1048576 cells under position 2047 of compressed level 0 "
                "would end at position 2147483648"
            ) in str(caught.value), name
            continue
        result = run()
        positions, coordinates = result.storage.index.mode_indices[0]
        assert positions.tolist() == [0, stored], name
        assert torch.equal(coordinates, torch.arange(stored, dtype=coordinates.dtype))
        values = result.storage.value.reshape(stored, BOUND_COLUMNS)
        for row in (0, 1023, stored - 1):
            assert torch.equal(values[row], c), (name, row)
        del result, values
