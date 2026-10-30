"""The 32-bit position bound on a two-pass result's compressed levels, and on a drain.

The two-pass assembly of a ``d,s[,s...]`` result (compressed_where_openmp_pass.py, used by
both compilers) counts each cell's entries of every compressed level, sums the counts into
an int64_t prefix sum ``_offset{L}`` whose last entry ``_total{L}`` is the level's entry
count, allocates the level's arrays from it, and fills them, storing every prefix-sum entry
and every position as ``int``.  Before patch 0038c nothing held ``_total{L}`` to the bound:
a level past it was allocated, filled through wrapped positions, and refused only on
return, by the result wrapper.  And a deeper level's count in one cell was an ``int``,
which overflows in the count pass once a cell holds more than 2**31 - 1 entries, so the
total was summed from wrapped counts.

Now (csrc/header.h):

* ``scorch_check_level_total`` holds each ``_total{L}`` to the bound between the two
  passes, before anything is allocated; a compressed level below has one position entry
  more;
* a deeper level's per-cell count, fill cursor and previous count are int64_t
  (result_write_pass.phase_state_type);
* ``scorch_check_drained_entries`` holds a sparse workspace that the shipped compiler
  drains into a coordinate tensor and reads back with ``int pT0_end =
  T0_crd_tensor.size(0)``.

1. The two native checks alone, at the bound.
2. The emitted LLIR, in both compilers: one check per compressed level, after every total
   and before every allocation of the result, with the right ``compressed_below``; the
   counters' types by level; the drain check after the drain and before the read-back.
3. Execution of small two-pass and drain programs in both compilers, against float64
   references.
4. With ``SCORCH_BIGMEM_TESTS=1``, the bound itself, one entry apart, in one cell: a result
   whose level 2 holds exactly 2**31 - 1 entries is stored and checked, and one of 2**31 is
   refused with the code before anything is allocated -- in a child process whose address
   space is limited to 4 GiB above its size when it calls, while the arrays the kernel
   would allocate need 10 GiB.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pytest
import torch
from torch.utils.cpp_extension import load_inline

import scorch
from scorch.compiler import llir
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.llir_traversal import LLIRPath, LLIRTraversalContext, LLIRWalker
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Scheduler
from scorch.compiler.sparse_assembly import TWO_PASS_STRATEGIES
from scorch.compiler.torch_cpp_abi import (
    DRAINED_ENTRIES_CHECK,
    LEVEL_TOTAL_CHECK,
    drained_entries_check,
    level_total_check,
)
from scorch.exceptions import IndexCapacityError, raise_if_native_coded
from scorch.stensor import STensor
from tests.test_scorch.kernel_cache_support import empty_kernel_caches
from tests.test_scorch.test_cin_lowerer import _build_outer_workspace_statement
from tests.test_scorch.test_dense_block_position_check import (
    _recorded_kernels,
    _recorded_targets,
)
from tests.test_scorch.test_index_position_bound import (
    M,
    _memory_gb,
    _native_line,
    assert_capacity_error,
)
from tests.test_scorch.test_sparse_assembly_strategy import auto_options

_ROOT = Path(__file__).resolve().parents[2]
_CSRC = _ROOT / "src" / "scorch" / "csrc"

# --------------------------------------------------------------------------------------
# 1. The two native checks alone.
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _kernel_checks():
    source = r"""
#include "header.h"

void level_total(int64_t total, bool compressed_below, int64_t level) {
  scorch_check_level_total(total, compressed_below, static_cast<int>(level));
}

void drained_entries(int64_t entries) { scorch_check_drained_entries(entries); }
"""
    digest = hashlib.sha256(
        (_CSRC / "header.h").read_bytes()
        + (_CSRC / "native_abi.h").read_bytes()
        + source.encode()
    ).hexdigest()[:12]
    return load_inline(
        name=f"scorch_level_total_check_{digest}",
        cpp_sources=source,
        functions=["level_total", "drained_entries"],
        extra_include_paths=[str(_CSRC)],
        extra_cflags=["-O0"],
        verbose=False,
    )


def _assert_refused(call, *args):
    with pytest.raises(RuntimeError) as caught:
        call(*args)
    with pytest.raises(IndexCapacityError) as mapped:
        raise_if_native_coded(caught.value)
    assert_capacity_error(mapped.value)
    return caught.value


@pytest.mark.parametrize(
    "total, compressed_below, refused",
    [
        (0, False, False),
        (M, False, False),  # the last level may hold every position an int addresses
        (M + 1, False, True),
        (M - 1, True, False),  # a level above another needs one position entry more
        (M, True, True),
        (2**40, False, True),  # far past the bound: exact in int64
    ],
)
def test_the_level_total_check_refuses_a_level_past_the_bound(
    total, compressed_below, refused
):
    check = _kernel_checks().level_total
    if refused:
        _assert_refused(check, total, compressed_below, 2)
    else:
        check(total, compressed_below, 2)


def test_the_level_total_refusal_names_the_level_and_its_count():
    check = _kernel_checks().level_total
    assert _native_line(_assert_refused(check, M + 1, False, 2)).endswith(
        "result compressed level 2 would hold 2147483648 entries; a 32-bit position "
        "addresses at most 2147483647 entries"
    )
    assert _native_line(_assert_refused(check, M, True, 1)).endswith(
        "result compressed level 1 would hold 2147483647 entries, and compressed level "
        "2 below it one position entry more; a 32-bit position addresses at most "
        "2147483647 entries"
    )


@pytest.mark.parametrize("entries, refused", [(0, False), (M, False), (M + 1, True)])
def test_the_drained_entries_check_refuses_a_drain_past_the_bound(entries, refused):
    check = _kernel_checks().drained_entries
    if refused:
        error = _assert_refused(check, entries)
        assert _native_line(error).endswith(
            "a sparse workspace drained 2147483648 entries, which the kernel reads back "
            "with 32-bit positions; a 32-bit position addresses at most 2147483647 "
            "entries"
        )
    else:
        check(entries)


def test_the_builders_take_only_the_int64_counts_they_check():
    total = llir.Var("_total2", llir.DataType.INT64)
    statement = level_total_check(total, 2, compressed_below=True)
    assert statement.name == LEVEL_TOTAL_CHECK
    assert statement.args[0] == total and statement.args[0] is not total
    with pytest.raises(TypeError):
        level_total_check(llir.Var("_total2", llir.DataType.INT), 2, False)
    with pytest.raises(ValueError):
        level_total_check(total, 0, False)
    drained = llir.Var("pT", llir.DataType.INT64)
    statement = drained_entries_check(drained)
    assert statement.name == DRAINED_ENTRIES_CHECK
    assert statement.args[0] == drained and statement.args[0] is not drained
    with pytest.raises(TypeError):
        drained_entries_check(llir.Var("pT", llir.DataType.INT))


# --------------------------------------------------------------------------------------
# 2. The emitted LLIR.
# --------------------------------------------------------------------------------------

_WALK = LLIRTraversalContext(stage="test", pass_name="two_pass_level_total")
_ALLOCATION = re.compile(r"\w+(?:\d+_pos|\d+_crd|_values)_torch")
_STATE = re.compile(r"_(?:cnt|pos|prev)(\d+)")


class _Events(LLIRWalker):
    """The statements this file is about, in program order."""

    def __init__(self) -> None:
        super().__init__(_WALK)
        self.events: List[Tuple[str, object]] = []

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        kind = type(node)
        if kind is llir.VarInit and type(node.var) is llir.Var:
            name, value = node.var.name, node.value
            if re.fullmatch(r"_total\d+", name):
                self.events.append(("total", int(name[len("_total") :])))
            elif _STATE.fullmatch(name):
                self.events.append(("state", (name, node.var.type)))
            elif name == "pT0_end":
                self.events.append(("read_back", name))
            elif type(value) is llir.FunctionCall:
                if value.name == "torch::empty" and _ALLOCATION.fullmatch(name):
                    self.events.append(("allocate", name))
                elif value.name == "scorch_tensor_from_vector" and name.startswith("T"):
                    self.events.append(("move", name))
        elif kind is llir.DirectInit and re.fullmatch(r"_count\d+", node.var.name):
            level = int(node.var.name[len("_count") :])
            self.events.append(("count_vector", (level, node.var.type)))
        elif kind is llir.FunctionCallStmt and node.name == LEVEL_TOTAL_CHECK:
            total, below, level = node.args
            assert type(total) is llir.Var and total.type is llir.DataType.INT64
            assert type(below) is llir.Literal and type(level) is llir.Literal
            self.events.append(("check", (level.value, below.value, total.name)))
        elif kind is llir.FunctionCallStmt and node.name == DRAINED_ENTRIES_CHECK:
            (entries,) = node.args
            assert type(entries) is llir.Var and entries.type is llir.DataType.INT64
            self.events.append(("drain_check", entries.name))
        elif kind is llir.ForLoopAuto:
            self.events.append(("drain", None))


def _events(lowered) -> List[Tuple[str, object]]:
    walker = _Events()
    walker.walk(lowered)
    return walker.events


def _positions(events, kind):
    return [index for index, (event, _) in enumerate(events) if event == kind]


def _two_pass_levels(lowered) -> Optional[Tuple[int, ...]]:
    """The compressed levels of a two-pass kernel, after checking its bound and its
    counters; None for a kernel without the two-pass assembly."""
    events = _events(lowered)
    totals = _positions(events, "total")
    checks = _positions(events, "check")
    if not totals:
        assert not checks, events
        return None
    levels = tuple(events[index][1] for index in totals)
    assert levels == tuple(range(1, len(levels) + 1)), levels
    # One check per level, each with a compressed level below except the last.
    assert [events[index][1] for index in checks] == [
        (level, level < levels[-1], f"_total{level}") for level in levels
    ]
    # Every total is read before any check, and every check precedes every allocation
    # of the result's arrays: nothing is allocated, and nothing narrowed, unchecked.
    allocations = _positions(events, "allocate")
    assert allocations, events
    assert max(totals) < min(checks) and max(checks) < min(allocations)
    # A cell's count is an int at level 1 only.
    for event, value in events:
        if event == "state":
            name, data_type = value
            level = int(_STATE.fullmatch(name).group(1))
            expected = llir.DataType.INT if level == 1 else llir.DataType.INT64
            assert data_type is expected, (name, data_type)
        elif event == "count_vector":
            level, data_type = value
            expected = (
                llir.DataType.STD_VECTOR_C_INT
                if level == 1
                else llir.DataType.STD_VECTOR_INT
            )
            assert data_type is expected, (level, data_type)
    return levels


def _bmm_cin(dtype, fmt="dss"):
    """``C[b,i,k] += A[b,i,j] * B[b,j,k]``, loops in order of first appearance."""
    b, i, j, k = (IndexVar(name) for name in "bijk")
    left = TensorVar("A", fmt=fmt, dtype=dtype)[b, i, j]
    right = TensorVar("B", fmt=fmt, dtype=dtype)[b, j, k]
    result = TensorVar("C", fmt=fmt, dtype=dtype)[b, i, k]
    statement = TensorAssign(
        result, CINBinaryOp(Operation.MUL, left, right), op=Operation.ADD
    )
    for variable in (k, j, i, b):
        statement = ForAll(variable, statement)
    return statement


def _spgemm_cin(dtype):
    """``C[i,k] += A[i,j] * B[j,k]``, all ``ds``: the new compiler's parallel
    sparse-workspace family, whose completion check rebuilds the two-pass function."""
    i, j, k = (IndexVar(name) for name in "ijk")
    left = TensorVar("A", fmt="ds", dtype=dtype)[i, j]
    right = TensorVar("B", fmt="ds", dtype=dtype)[j, k]
    result = TensorVar("C", fmt="ds", dtype=dtype)[i, k]
    statement = TensorAssign(
        result, CINBinaryOp(Operation.MUL, left, right), op=Operation.ADD
    )
    for variable in (k, j, i):
        statement = ForAll(variable, statement)
    return statement


# The new compiler's configurations: both automatic arms, and each two-pass strategy
# requested.
_NEW_CONFIGS: Dict[str, Tuple[bool, Optional[str]]] = {
    "auto-plain": (False, None),
    "auto-blocked": (True, None),
    **{strategy: (False, strategy) for strategy in TWO_PASS_STRATEGIES},
}


def _new_options(config: str, jit: bool = False):
    arm, assembly = _NEW_CONFIGS[config]
    return auto_options(arm, assembly=assembly, jit=jit)


def test_the_shipped_compiler_checks_every_level_of_a_two_pass_result():
    """The batched product into ``dss``: the shipped compiler's parallel two-pass
    assembly of two compressed levels (as in test_codegen_perf_optimizations)."""
    lowered = CINLowerer().lower_IndexStmt(
        Scheduler.auto_schedule(_bmm_cin(torch.float32))
    )
    assert _two_pass_levels(lowered) == (1, 2)


# A requested strategy a family cannot emit is refused with a code, before any kernel.
# The parallel sparse-workspace family emits the parallel two-pass form only
# (``_ParallelSparseWorkspaceLowering.supported_assemblies``).
_REFUSED_REQUESTS = {("spgemm-ds", "two_pass_serial"): "unsupported_assembly_host"}


@pytest.mark.parametrize("config", sorted(_NEW_CONFIGS))
@pytest.mark.parametrize(
    "program, build, shape, operands, levels",
    [
        ("bmm-dss", _bmm_cin, (3, 4, 5), ((3, 4, 6), (3, 6, 5)), (1, 2)),
        ("spgemm-ds", _spgemm_cin, (4, 5), ((4, 6), (6, 5)), (1,)),
    ],
    ids=["bmm-dss", "spgemm-ds"],
)
def test_the_new_compiler_checks_every_level_of_a_two_pass_result(
    monkeypatch, config, program, build, shape, operands, levels
):
    recorded = _recorded_kernels(monkeypatch)
    targets = _recorded_targets(monkeypatch)
    refusal = _REFUSED_REQUESTS.get((program, config))
    try:
        compile_cin_via_loopir(
            build(torch.float32),
            shape,
            tuple((operand, torch.float32) for operand in operands),
            compile_options=_new_options(config),
        )
    except LoopIRTargetError as error:
        if refusal is None or error.defect.code != refusal:
            raise
        return
    assert refusal is None, f"{program} under {config} was not refused"
    assert recorded and targets, "no kernel was lowered"
    found = [_two_pass_levels(body) for _, body in recorded]
    # Every kernel checks; record which configurations assemble in two passes.
    two_pass = [found_levels for found_levels in found if found_levels is not None]
    if config in TWO_PASS_STRATEGIES:
        assert two_pass == [levels], (targets[-1], found)
    for found_levels in two_pass:
        assert found_levels == levels, (targets[-1], found_levels)


def test_the_parallel_sparse_workspace_family_checks_its_total():
    """The family whose completion check rebuilds the two-pass function itself
    (``_ParallelSparseWorkspaceLowering.complete_sparse_workspace``): the rebuilt
    function carries the check too, or the completion would refuse the kernel."""
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            _spgemm_cin(torch.float32),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        source = kernel.cpp_source
        load = "  int64_t _total1 = _offset1[A0_size];\n"
        assert load + "  scorch_check_level_total(_total1, false, 1);\n" in source
        assert source.index(load) < source.index("C1_pos_torch = torch::empty(")


def test_the_shipped_drain_is_checked_before_it_is_read_back():
    statement, result = _build_outer_workspace_statement("ds")
    lowerer = CINLowerer()
    lowerer.outermost_stmt = statement
    lowerer.result_tensor_var = result
    lowerer.result_tensor_access = statement.consumer.get_result_tensor_accesses()[0]
    events = _events(lowerer.lower_outer_ConsumerIndexStmt(statement.consumer))
    (drain,) = _positions(events, "drain")
    (check,) = _positions(events, "drain_check")
    assert events[check][1] == "pT"
    moves = _positions(events, "move")
    (read_back,) = _positions(events, "read_back")
    assert moves and drain < check < min(moves) and max(moves) < read_back


# --------------------------------------------------------------------------------------
# 3. Execution of small programs.
# --------------------------------------------------------------------------------------


def _bmm_operands(rows, columns, last_row_columns, dtype):
    """``A`` (1 x rows x 2) and ``B`` (1 x 2 x columns), both ``dss``.

    Every row of ``A`` but the last stores ``j = 0`` and the last stores ``j = 1``, each
    with value 1; ``B``'s row 0 stores every column, its row 1 the first
    ``last_row_columns``.  The product's level 2 holds (rows - 1) * columns +
    last_row_columns entries, all in the one cell of level 0, and every value is one
    small integer, exact in every dtype.
    """
    left_columns = torch.zeros(rows, dtype=torch.int64)
    left_columns[-1] = 1
    left = STensor.from_components(
        (1, rows, 2),
        "dss",
        [
            [],
            [torch.tensor([0, rows]), torch.arange(rows)],
            [torch.arange(rows + 1), left_columns],
        ],
        torch.ones(rows, dtype=dtype),
    )
    right_values = torch.cat(
        [torch.arange(columns) % 7 - 3, torch.arange(last_row_columns) % 5 - 2]
    ).to(dtype)
    right = STensor.from_components(
        (1, 2, columns),
        "dss",
        [
            [],
            [torch.tensor([0, 2]), torch.tensor([0, 1])],
            [
                torch.tensor([0, columns, columns + last_row_columns]),
                torch.cat([torch.arange(columns), torch.arange(last_row_columns)]),
            ],
        ],
        right_values,
    )
    return left, right


def _assert_bmm_rows(result, rows, columns, last_row_columns, dtype, block=512):
    """The product of :func:`_bmm_operands`, read in blocks of rows against float64."""
    assert str(result.format) == "d,s,s"
    levels = result.storage.index.mode_indices
    positions1, coordinates1 = levels[1]
    assert positions1.tolist() == [0, rows]
    assert torch.equal(coordinates1.to(torch.int64), torch.arange(rows))
    positions2, coordinates2 = levels[2]
    full = (rows - 1) * columns
    expected_positions = torch.cat(
        [
            torch.arange(rows, dtype=torch.int64) * columns,
            torch.tensor([full + last_row_columns]),
        ]
    )
    assert torch.equal(positions2.to(torch.int64), expected_positions)
    values = result.storage.value
    assert values.dtype == dtype and values.numel() == full + last_row_columns
    head = (torch.arange(columns) % 7 - 3).to(torch.float64)
    tail = (torch.arange(last_row_columns) % 5 - 2).to(torch.float64)
    column_ids = torch.arange(columns, dtype=coordinates2.dtype)
    for start in range(0, rows - 1, block):
        stop = min(rows - 1, start + block)
        span = slice(start * columns, stop * columns)
        assert torch.equal(
            coordinates2[span].view(stop - start, columns),
            column_ids.expand(stop - start, columns),
        ), start
        assert torch.equal(
            values[span].view(stop - start, columns).to(torch.float64),
            head.expand(stop - start, columns),
        ), start
    assert torch.equal(
        coordinates2[full:].to(torch.int64), torch.arange(last_row_columns)
    )
    assert torch.equal(values[full:].to(torch.float64), tail)


def _first(result):
    return result[0] if isinstance(result, tuple) else result


def _runner(config: str, dtype) -> Callable:
    """One compiler's call of the batched product: the shipped compiler's public einsum,
    or the new compiler under one configuration."""
    if config == "shipped":
        return lambda left, right, shape: scorch.einsum(
            "bij,bjk->bik", left, right, format="dss"
        )
    return lambda left, right, shape: _first(
        execute_cin_via_loopir(
            _bmm_cin(dtype),
            shape,
            left,
            right,
            compile_options=_new_options(config, jit=True),
        )
    )


_CONFIGS = ("shipped", *sorted(_NEW_CONFIGS))


@pytest.mark.parametrize("config", _CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.int8])
def test_both_compilers_store_the_small_product_exactly(config, dtype):
    rows, columns, last_row_columns = 9, 11, 4
    left, right = _bmm_operands(rows, columns, last_row_columns, dtype)
    result = _runner(config, dtype)(left, right, (1, rows, columns))
    _assert_bmm_rows(result, rows, columns, last_row_columns, dtype, block=3)


def test_a_drained_sum_matches_the_float64_reference(monkeypatch):
    """``ij->i`` into ``s``: the shipped compiler drains its workspace into ``T`` and
    reads it back (test_sparse_rank_one_reduction_drain); the drain carries the check.
    """
    torch.manual_seed(20261013)
    dense = torch.randn(40, 30, dtype=torch.float64)
    dense *= torch.rand(40, 30) < 0.3
    dense[::5] = 0
    operand = STensor.from_torch(dense.clone(), "A").to_sparse("ss")
    recorded = _recorded_kernels(monkeypatch)
    empty_kernel_caches(monkeypatch)
    result = scorch.einsum("ij->i", operand, format="s")
    checked = [body for _, body in recorded if _positions(_events(body), "drain_check")]
    assert checked, "no kernel lowered here drains into a coordinate tensor"
    torch.testing.assert_close(
        result.to_torch(), dense.sum(dim=1), atol=0.0, rtol=1e-12
    )


# --------------------------------------------------------------------------------------
# 4. The bound itself, one entry apart, in one cell.
# --------------------------------------------------------------------------------------

# 46,340 full rows of 46,341 entries: 2,147,441,940.  The last row holds the rest.
BOUND_ROWS = BOUND_COLUMNS = 46_341
AT_THE_BOUND = M - (BOUND_ROWS - 1) * BOUND_COLUMNS  # 41,707: exactly 2**31 - 1
PAST_THE_BOUND = AT_THE_BOUND + 1  # 2**31
# What the kernel would allocate for 2**31 entries of int8: 8 GiB of coordinates and
# 2 GiB of values.  The child's address space is limited to this much above its size.
ADDRESS_SPACE_MARGIN = 4 * 2**30
# The most the refused call may raise the child's peak resident memory.
RESIDENT_RISE_BOUND_KIB = 512 * 1024

_BIGMEM = pytest.mark.skipif(
    os.environ.get("SCORCH_BIGMEM_TESTS") != "1"
    or _memory_gb() < 24
    or not sys.platform.startswith("linux"),
    reason="stores 10 GiB: needs SCORCH_BIGMEM_TESTS=1, 24 GB of memory and Linux",
)


@lru_cache(maxsize=1)
def _two_pass_configs() -> Tuple[str, ...]:
    """The configurations whose kernel for the batched product is two-pass, read from
    the LLIR at a small shape."""
    configs = ["shipped"]
    for config in sorted(_NEW_CONFIGS):
        with pytest.MonkeyPatch.context() as patch:
            recorded = _recorded_kernels(patch)
            compile_cin_via_loopir(
                _bmm_cin(torch.int8),
                (1, 5, 7),
                (((1, 5, 2), torch.int8), ((1, 2, 7), torch.int8)),
                compile_options=_new_options(config),
            )
        if any(_two_pass_levels(body) for _, body in recorded):
            configs.append(config)
    return tuple(configs)


@_BIGMEM
@pytest.mark.parametrize("config", _CONFIGS)
def test_both_compilers_store_exactly_the_last_position_an_int_addresses(config):
    if config not in _two_pass_configs():
        pytest.skip(f"{config} does not assemble this product in two passes")
    left, right = _bmm_operands(BOUND_ROWS, BOUND_COLUMNS, AT_THE_BOUND, torch.int8)
    result = _runner(config, torch.int8)(left, right, (1, BOUND_ROWS, BOUND_COLUMNS))
    _assert_bmm_rows(result, BOUND_ROWS, BOUND_COLUMNS, AT_THE_BOUND, torch.int8)


def _vm_bytes(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    raise AssertionError(f"no {field} in /proc/self/status")


def refused_in_this_process(config: str) -> Dict[str, object]:
    """Run the product past the bound under an address-space limit (the child's side).

    The kernel is built, or loaded, at a small shape first.  Then the soft limit on the
    address space is set to the process's size plus ADDRESS_SPACE_MARGIN, and the call
    past the bound is made: an allocation of the result's arrays would fail there, with
    an error that carries no code.
    """
    import resource

    run = _runner(config, torch.int8)
    run(*_bmm_operands(5, 7, 3, torch.int8), (1, 5, 7))
    left, right = _bmm_operands(BOUND_ROWS, BOUND_COLUMNS, PAST_THE_BOUND, torch.int8)
    size = _vm_bytes("VmSize")
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (size + ADDRESS_SPACE_MARGIN, hard))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    outcome: Dict[str, object] = {"size_bytes": size, "margin": ADDRESS_SPACE_MARGIN}
    try:
        run(left, right, (1, BOUND_ROWS, BOUND_COLUMNS))
        outcome["outcome"] = "returned"
    except IndexCapacityError as error:
        outcome.update(
            outcome="refused",
            type=type(error).__name__,
            code=getattr(error, "code", None),
            message=str(error),
        )
    except BaseException as error:  # reported to the parent, which fails the test
        outcome.update(outcome="failed", type=type(error).__name__, message=str(error))
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    outcome["resident_rise_kib"] = after - before
    return outcome


_CHILD = (
    "import json, sys\n"
    "from tests.test_scorch.test_two_pass_level_total import refused_in_this_process\n"
    "print('OUTCOME ' + json.dumps(refused_in_this_process(sys.argv[1])))\n"
)


@_BIGMEM
@pytest.mark.parametrize("config", _CONFIGS)
def test_both_compilers_refuse_the_next_position_before_allocating(config):
    if config not in _two_pass_configs():
        pytest.skip(f"{config} does not assemble this product in two passes")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT / "src"), str(_ROOT), environment.get("PYTHONPATH", "")]
    )
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, config],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    lines = [line for line in child.stdout.splitlines() if line.startswith("OUTCOME ")]
    assert child.returncode == 0 and len(lines) == 1, (child.returncode, child.stderr)
    outcome = json.loads(lines[0][len("OUTCOME ") :])
    print(config, outcome)
    assert outcome["outcome"] == "refused", outcome
    assert outcome["type"] == "IndexCapacityError"
    assert outcome["code"] == "index_position_exceeds_int32"
    assert "result compressed level 2 would hold 2147483648 entries" in str(
        outcome["message"]
    )
    assert int(outcome["resident_rise_kib"]) < RESIDENT_RISE_BOUND_KIB, outcome
