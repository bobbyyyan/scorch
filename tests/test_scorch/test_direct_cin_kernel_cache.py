"""A warm ``scorch.ops.lower_and_exec_cin`` call runs its kept kernel and lowers nothing.

The developer entry used to normalize, bind, lower, run the LLIR passes, emit, hash
and prepare a build on every call, and only then find the module in the build cache:
32,337 Python calls per warm call of a CSR float64 1,024 x 1,024 times dense 1,024 x 8
product on the tip, against 5,697 on 864745d, which did the same work with a lighter
pipeline (row 40; mkt1 stage 044: 6,270 us against 1,812 us).  It now keeps its kernel
in the generated-kernel cache the conversions and ``spmv`` use
(``scorch.stensor._GENERATED_KERNELS``), under a key read from the caller's statement on
every call (``cin_lowering_key``, after ``verify_cin_structure``), the result shape,
each operand's layout and dtypes, and the options key.  These tests pin:

1. an identical second call hits: it compiles nothing and makes no options snapshot, no
   environment parse and no compilation context, and its result is the cold call's
   exactly, level array by level array;
2. a second call that changes one input misses, its key differs from the first call's,
   and it does what the full path does with the cache off; a statement a warm call
   used, then written in place in any field the lowering reads (as 864745d's entry
   wrote shapes into it), is seen; a call writes nothing into the caller's statement;
   the options key holds the build options (a directory put on ``PATH`` misses), and an
   environment that parses to the same options hits;
3. another operand of the same layout hits and reads its own arrays, as do one tensor
   passed as both operands and values written in place between two hits;
4. the hit check (``SCORCH_CHECK_KERNEL_CACHE_HITS``) passes a correct hit and refuses
   one whose key left out an input the kernel depends on (negative controls);
5. a hit still refuses what a cold call refuses: a malformed operand, positions
   written in place after a call, a malformed or badly typed statement, a wrong
   operand count, a requested schedule, a context that does not hold the call's
   snapshot; and, in a child interpreter, every result or operand extent that
   disagrees (``IndexExtentMismatchError``), with the agreeing call's kernel kept; it
   runs full verification when the policy asks for it;
6. a context handed in sees nothing recorded on a hit, and the cold call's stages are
   the ones it recorded before this cache;
7. a call that raises keeps nothing;
8. the statement key holds every field the normalization copies, and nothing it drops,
   and per ID type how the IDs' values compare (the lowering sorts accesses by them): a
   statement built in reverse has another key and is right under the hit check.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import pytest
import torch

import scorch.ops as ops  # type: ignore[import-untyped]
import scorch.stensor as stensor_module  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.compiler import compile_options  # type: ignore[import-untyped]
from scorch.compiler.cin import (  # type: ignore[import-untyped]
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_analysis import (  # type: ignore[import-untyped]
    cin_lowering_key,
    full_cin_verification,
)
from scorch.compiler.compilation_context import (  # type: ignore[import-untyped]
    CompilationContext,
)
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    CompileOptionsError,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    schedule_force,
)
from scorch.exceptions import (  # type: ignore[import-untyped]
    GeneratedKernelCacheError,
)
from tests.test_scorch.level_array_support import assert_matches, stored
from tests.test_scorch.sum_cin_support import operands as sum_operands
from tests.test_scorch.sum_cin_support import reference as sum_reference
from tests.test_scorch.sum_cin_support import sum_cin

F64, F32, I32, I64 = torch.float64, torch.float32, torch.int32, torch.int64


# -- what a test sees -------------------------------------------------------------------


class _Recording(dict):
    """The kernel store, recording every lookup as (key, hit)."""

    def __init__(self, keep: bool = True) -> None:
        super().__init__()
        self.keep = keep
        self.lookups: List[Tuple[tuple, bool]] = []

    def _stored(self, key: tuple) -> tuple:
        return key

    def get(self, key, default=None):  # type: ignore[override]
        found = super().get(self._stored(key), default)
        self.lookups.append((key, found is not None))
        return found

    def __setitem__(self, key, value) -> None:
        if self.keep:
            super().__setitem__(self._stored(key), value)

    def direct_keys(self) -> List[tuple]:
        return [key for key, _ in self.lookups if key[0] == "lower_and_exec_cin"]


class _KeyWithout(_Recording):
    """The store with some key positions left out of every key (a negative control)."""

    def __init__(self, positions: Tuple[int, ...]) -> None:
        super().__init__()
        self.positions = positions

    def _stored(self, key: tuple) -> tuple:
        if key[0] != "lower_and_exec_cin":
            return key
        return tuple(
            "left out" if position in self.positions else part
            for position, part in enumerate(key)
        )


# The direct key's positions (``ops._direct_cin_dispatch_key``).
STRUCTURE, RESULT_SHAPE, OPERANDS, OPTIONS = 1, 2, 3, 4


class _Seen:
    """Compiles, snapshots, parses and contexts since the last ``reset``."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.compiles = 0
        self.snapshots = 0
        self.parses = 0
        self.contexts = 0


def _store() -> _Recording:
    return stensor_module._GENERATED_KERNELS


@pytest.fixture(autouse=True)
def _nothing_kept_between_tests(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    compile_options.clear_build_snapshot_cache()
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording())
    # These tests count snapshots, which a hit check makes; the check's own tests turn
    # it on.
    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [False])
    yield
    compile_options.clear_build_snapshot_cache()


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> _Seen:
    record = _Seen()
    compile_direct = ops._compile_direct_cin

    def counted_compile(*args: Any, **kwargs: Any) -> Any:
        record.compiles += 1
        return compile_direct(*args, **kwargs)

    monkeypatch.setattr(ops, "_compile_direct_cin", counted_compile)
    from_environment = CompileOptions.from_environment.__func__
    parse = CompileOptions._parse_environment.__func__

    def counted_snapshot(cls, *args, **kwargs):
        record.snapshots += 1
        return from_environment(cls, *args, **kwargs)

    def counted_parse(cls, *args, **kwargs):
        record.parses += 1
        return parse(cls, *args, **kwargs)

    monkeypatch.setattr(
        CompileOptions, "from_environment", classmethod(counted_snapshot)
    )
    monkeypatch.setattr(
        CompileOptions, "_parse_environment", classmethod(counted_parse)
    )
    context_init = CompilationContext.__init__

    def counted_context(self, *args, **kwargs):
        record.contexts += 1
        context_init(self, *args, **kwargs)

    monkeypatch.setattr(CompilationContext, "__init__", counted_context)
    return record


def _uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache off: every call compiles and nothing is kept."""

    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording(keep=False))


def _parts(result: STensor) -> tuple:
    """A result as its shape, format, mode order, level arrays and values."""

    return (
        tuple(result.shape),
        result.format,
        tuple(result.mode_order),
        [[array.clone() for array in level] for level in result._native_mode_indices()],
        result.values.clone(),
    )


def _assert_same(got: tuple, expected: tuple) -> None:
    assert got[:3] == expected[:3]
    assert len(got[3]) == len(expected[3])
    for got_level, expected_level in zip(got[3], expected[3]):
        assert len(got_level) == len(expected_level)
        for got_array, expected_array in zip(got_level, expected_level):
            assert got_array.dtype == expected_array.dtype
            assert torch.equal(got_array, expected_array)
    assert got[4].dtype == expected[4].dtype
    assert torch.equal(got[4], expected[4])


# -- the calls -------------------------------------------------------------------------


class Call:
    """One direct CIN call: a statement factory, a result shape, operands and the
    float64 reference of the result in logical order."""

    def __init__(
        self,
        statement: Callable[[], Any],
        result_shape: Tuple[int, ...],
        operands: Callable[[], Tuple[STensor, ...]],
        reference: torch.Tensor,
        dtype: torch.dtype = F64,
    ) -> None:
        self.statement = statement
        self.result_shape = result_shape
        self.operands = operands
        self.reference = reference
        self.dtype = dtype

    def __call__(
        self, statement: Any = None, operands: Any = None, **kwargs: Any
    ) -> Any:
        return ops.lower_and_exec_cin(
            self.statement() if statement is None else statement,
            self.result_shape,
            *(self.operands() if operands is None else operands),
            **kwargs,
        )


def _spmm_statement(
    names: Sequence[str] = ("C", "B", "Y", "i", "j", "k"),
    dtype: torch.dtype = F64,
    op: Optional[Operation] = Operation.ADD,
    parallel: Optional[bool] = None,
    fmt_c: str = "ds",
) -> Any:
    c, b, y, i_name, j_name, k_name = names
    i, j, k = IndexVar(i_name), IndexVar(j_name), IndexVar(k_name)
    cv = TensorVar(c, fmt=fmt_c, mode_order=[0, 1], dtype=dtype)
    bv = TensorVar(b, fmt="dd", mode_order=[0, 1], dtype=dtype)
    yv = TensorVar(y, fmt="dd", mode_order=[0, 1], dtype=dtype)
    return ForAll(
        i,
        ForAll(j, ForAll(k, TensorAssign(yv[i, k], cv[i, j] * bv[j, k], op=op))),
        parallel=parallel,
    )


def _spmm_dense(rows: int = 7, inner: int = 6, cols: int = 3, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    c = (torch.rand(rows, inner, generator=generator, dtype=F64) + 0.5) * (
        torch.rand(rows, inner, generator=generator) < 0.4
    )
    b = torch.rand(inner, cols, generator=generator, dtype=F64) + 0.5
    return c, b


def _spmm(
    rows: int = 7,
    seed: int = 0,
    dtype: torch.dtype = F64,
    fmt_c: str = "ds",
    b_mode_order: Optional[Sequence[int]] = None,
) -> Call:
    c, b = _spmm_dense(rows=rows, seed=seed)

    def make() -> Tuple[STensor, ...]:
        return (
            stored(c.to(dtype), fmt_c, "C"),
            stored(b.to(dtype), "dd", "B", mode_order=b_mode_order),
        )

    return Call(
        lambda: _spmm_statement(dtype=dtype, fmt_c=fmt_c),
        (rows, 3),
        make,
        c @ b,
        dtype,
    )


def _sum(formats, indices, result_indices, result_format, extents, dtype=F64) -> Call:
    shape = tuple(extents[name] for name in indices)
    dense = sum_operands(formats, shape, dtype, 29)

    def make() -> Tuple[STensor, ...]:
        return (stored(dense[0], formats[0], "A"), stored(dense[1], formats[1], "B"))

    return Call(
        lambda: sum_cin(formats, indices, result_indices, result_format, dtype),
        tuple(extents[name] for name in result_indices),
        make,
        sum_reference(dense, indices, result_indices),
        dtype,
    )


def _copy_relaid() -> Call:
    """C[i, j, k] = B[i, j, k] with B stored in mode order [1, 0, 2]: every call
    relays B out before its kernel (``_relayout_mode_order_args``)."""

    dense = torch.arange(24, dtype=F64).reshape(3, 2, 4) + 1.0

    def statement() -> Any:
        i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
        result_var = TensorVar("C", fmt="ddd")
        input_var = TensorVar("B", fmt="ddd")
        return ForAll(
            i,
            ForAll(j, ForAll(k, TensorAssign(result_var[i, j, k], input_var[i, j, k]))),
        )

    return Call(
        statement,
        (3, 2, 4),
        lambda: (stored(dense, "ddd", "B", mode_order=[1, 0, 2]),),
        dense,
    )


SMALL = {"i": 5, "j": 4, "k": 6}
CALLS = {
    "spmm_csr_dense": _spmm,
    "spmm_float32": lambda: _spmm(dtype=F32),
    "spmm_dense_dense": lambda: _spmm(fmt_c="dd"),
    "spmm_relaid_dense_operand": lambda: _spmm(b_mode_order=[1, 0]),
    "copy_relaid_rank3": _copy_relaid,
    "sum_into_csr": lambda: _sum(("ss", "ds"), "ij", "ij", "ds", SMALL),
    "sum_into_dcsr": lambda: _sum(("ss", "ss"), "ij", "ij", "ss", SMALL),
    "sum_into_rank3_sparse": lambda: _sum(("sss", "dss"), "ijk", "ijk", "sss", SMALL),
    "sum_reducing_into_dense": lambda: _sum(("dss", "ssd"), "ijk", "i", "d", SMALL),
}


# -- 1. an identical second call hits ---------------------------------------------------


@pytest.mark.parametrize("name", sorted(CALLS))
def test_an_identical_second_call_hits_and_returns_the_cold_result(
    name: str, seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = CALLS[name]()
    cold = call()
    assert seen.compiles == 1
    assert_matches(cold, call.reference, call.dtype)
    seen.reset()
    warm = call()
    assert (seen.compiles, seen.snapshots, seen.parses, seen.contexts) == (0, 0, 0, 0)
    direct = [hit for key, hit in _store().lookups if key[0] == "lower_and_exec_cin"]
    assert direct == [False, True]
    _assert_same(_parts(warm), _parts(cold))
    # And the cache-off call, on fresh operands and a fresh statement.
    _uncached(monkeypatch)
    _assert_same(_parts(call()), _parts(cold))


def test_a_warm_call_reuses_one_statement_object(seen: _Seen) -> None:
    call = _spmm()
    statement, operands = call.statement(), call.operands()
    first = call(statement, operands)
    seen.reset()
    for _ in range(3):
        again = call(statement, operands)
    assert seen.compiles == 0
    _assert_same(_parts(again), _parts(first))


# -- 2. a changed input misses ----------------------------------------------------------


def _changed_calls() -> dict:
    """Second calls, each changing one input of ``_spmm()``'s call, with the key
    positions that input enters."""

    base = _spmm()

    def statement_with(**kwargs: Any) -> Callable[[], Any]:
        return lambda: _spmm_statement(**kwargs)

    return {
        "tensor name": (
            Call(
                statement_with(names=("P", "B", "Y", "i", "j", "k")),
                base.result_shape,
                base.operands,
                base.reference,
            ),
            {STRUCTURE},
        ),
        "index name": (
            Call(
                statement_with(names=("C", "B", "Y", "r", "j", "k")),
                base.result_shape,
                base.operands,
                base.reference,
            ),
            {STRUCTURE},
        ),
        "declared dtype": (
            Call(
                statement_with(dtype=F32),
                base.result_shape,
                base.operands,
                base.reference,
            ),
            {STRUCTURE},
        ),
        "parallel flag": (
            Call(
                statement_with(parallel=False),
                base.result_shape,
                base.operands,
                base.reference,
            ),
            {STRUCTURE},
        ),
        "result shape and operand shape": (_spmm(rows=9), {RESULT_SHAPE, OPERANDS}),
        "operand dtype": (_spmm(dtype=F32), {STRUCTURE, OPERANDS}),
        "operand mode order": (_spmm(b_mode_order=[1, 0]), {OPERANDS}),
        "operand format": (_spmm(fmt_c="dd"), {STRUCTURE, OPERANDS}),
    }


CHANGES = sorted(_changed_calls())


@pytest.mark.parametrize("change", CHANGES)
def test_a_changed_input_misses_and_does_what_the_full_path_does(
    change: str, seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    _spmm()()
    second, positions = _changed_calls()[change]
    seen.reset()
    got = second()
    assert seen.compiles == 1
    first_key, second_key = _store().direct_keys()
    differs = {at for at in range(len(first_key)) if first_key[at] != second_key[at]}
    assert differs == positions
    assert_matches(got, second.reference, second.dtype)
    _uncached(monkeypatch)
    _assert_same(_parts(second()), _parts(got))


def test_an_index_dtype_enters_the_key(seen: _Seen) -> None:
    call = _spmm()
    wide = call.operands()
    narrow_c = STensor.from_components(
        tuple(wide[0].shape),
        "ds",
        [
            [array.to(I32) for array in level]
            for level in wide[0]._native_mode_indices()
        ],
        wide[0].values.clone(),
        name="C",
    )
    first = call(operands=wide)
    seen.reset()
    second = call(operands=(narrow_c, wide[1]))
    assert seen.compiles == 1
    first_key, second_key = _store().direct_keys()
    assert first_key[OPERANDS] != second_key[OPERANDS]
    assert first_key[STRUCTURE] == second_key[STRUCTURE]
    assert torch.equal(second.values, first.values)


def test_a_statement_changed_in_place_between_calls_is_seen(seen: _Seen) -> None:
    """864745d's entry wrote the operands' shapes into the caller's statement; any write
    to a field the lowering reads must reach the key of the next call."""

    call = _spmm()
    statement, operands = call.statement(), call.operands()
    call(statement, operands)
    seen.reset()
    result_var = statement.stmt.stmt.stmt.lhs.tensor
    result_var.shape = tuple(call.result_shape)
    again = call(statement, operands)
    assert seen.compiles == 1
    assert_matches(again, call.reference, call.dtype)
    first_key, second_key = _store().direct_keys()
    assert first_key[STRUCTURE] != second_key[STRUCTURE]
    seen.reset()
    call(statement, operands)
    assert seen.compiles == 0


def _outcome(call: Callable[[], Any]) -> tuple:
    """A call's result as ``_parts`` gives it, or its refusal: the exception's type,
    its message with numbers masked (a statement built again has new stable IDs) and
    its code."""

    try:
        return ("returned", _parts(call()))
    except Exception as error:
        message = re.sub(r"\d+", "#", str(error))
        return ("refused", type(error), message, getattr(error, "code", None))


def _assert_same_outcome(got: tuple, expected: tuple) -> None:
    assert got[0] == expected[0], (got, expected)
    if got[0] == "returned":
        _assert_same(got[1], expected[1])
    else:
        assert got == expected


def _assignment(statement: Any) -> Any:
    return statement.stmt.stmt.stmt


def _operand_var(statement: Any, position: int) -> Any:
    rhs = _assignment(statement).rhs
    return (rhs.left if position == 0 else rhs.right).tensor


def _format_of(fmt: str) -> Any:
    return _operand_var(_spmm_statement(fmt_c=fmt), 0)._format


# Writes into the fields of ``_spmm_statement()``'s tensors that the lowering reads.
MUTATIONS = {
    "result shape": lambda s: setattr(_assignment(s).lhs.tensor, "shape", (7, 3)),
    "result dtype": lambda s: setattr(_assignment(s).lhs.tensor, "dtype", F32),
    "result format": lambda s: setattr(
        _assignment(s).lhs.tensor, "_format", _format_of("ds")
    ),
    "result mode order": lambda s: setattr(
        _assignment(s).lhs.tensor, "mode_order", [1, 0]
    ),
    "operand format": lambda s: setattr(
        _operand_var(s, 0), "_format", _format_of("dd")
    ),
    "operand dtype": lambda s: setattr(_operand_var(s, 1), "dtype", F32),
    "operand mode order": lambda s: setattr(_operand_var(s, 1), "mode_order", [1, 0]),
    "operand shape": lambda s: setattr(_operand_var(s, 1), "shape", (6, 3)),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_a_statement_a_warm_call_used_then_written_in_place_misses(
    mutation: str, seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next call with the written statement does not run the kept kernel, and does
    what the cache-off call does with a fresh statement given the same write: the same
    result, or the same refusal."""

    call = _spmm()
    statement, operands = call.statement(), call.operands()
    call(statement, operands)
    call(statement, operands)
    MUTATIONS[mutation](statement)
    seen.reset()
    got = _outcome(lambda: call(statement, operands))
    hits = [hit for key, hit in _store().lookups if key[0] == "lower_and_exec_cin"]
    assert hits[:2] == [False, True]
    assert not any(hits[2:])
    assert seen.compiles == 1
    _uncached(monkeypatch)
    fresh = _with(call.statement(), MUTATIONS[mutation])
    _assert_same_outcome(got, _outcome(lambda: call(fresh, call.operands())))


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_call_writes_nothing_into_the_caller_s_statement(name: str) -> None:
    """The entry binds the operands' shapes and dtypes, the result's, and the aligned
    mode orders into the statement it normalized, which it owns; the caller's statement
    keeps its key through a cold and a warm call."""

    call = CALLS[name]()
    statement = call.statement()
    before = cin_lowering_key(statement)
    call(statement)
    assert cin_lowering_key(statement) == before
    call(statement)
    assert cin_lowering_key(statement) == before
    hits = [hit for key, hit in _store().lookups if key[0] == "lower_and_exec_cin"]
    assert hits == [False, True]


def test_another_options_snapshot_of_other_policy_misses(seen: _Seen) -> None:
    call = _spmm()
    call()
    seen.reset()
    with full_cin_verification(True):
        verified = call()
    assert seen.compiles == 1
    first_key, second_key = _store().direct_keys()
    assert first_key[OPTIONS] != second_key[OPTIONS]
    assert_matches(verified, call.reference, call.dtype)


def test_a_build_option_enters_the_options_key(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The options key is the snapshot's cache fingerprint, which holds its build
    options.  ``PATH`` is one of them (``build.executable_search_path``): an empty
    directory put in front of it changes only the options key, and the call builds
    again."""

    call = _spmm()
    first = call()
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    seen.reset()
    second = call()
    assert seen.compiles == 1
    first_key, second_key = _store().direct_keys()
    differs = {at for at in range(len(first_key)) if first_key[at] != second_key[at]}
    assert differs == {OPTIONS}
    _assert_same(_parts(second), _parts(first))


def test_an_environment_that_parses_to_the_same_options_hits(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The options key names the parsed options, not the variables' text: setting
    ``SCORCH_VERIFY_CIN`` to its default makes the options again and keeps the hit."""

    monkeypatch.delenv("SCORCH_VERIFY_CIN", raising=False)
    call = _spmm()
    first = call()
    monkeypatch.setenv("SCORCH_VERIFY_CIN", "0")
    seen.reset()
    second = call()
    assert seen.compiles == 0
    assert seen.parses == 1
    first_key, second_key = _store().direct_keys()
    assert first_key == second_key
    _assert_same(_parts(second), _parts(first))


# -- 3. another operand of the same layout ----------------------------------------------


def test_another_operand_of_the_same_layout_hits_and_reads_its_own_arrays(
    seen: _Seen,
) -> None:
    _spmm(seed=0)()
    other = _spmm(seed=5)
    seen.reset()
    got = other()
    assert seen.compiles == 0
    assert_matches(got, other.reference, other.dtype)


def test_one_tensor_passed_as_both_operands_hits_and_reads_it_twice(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = CALLS["sum_into_dcsr"]()
    a, b = call.operands()
    call(operands=(a, b))
    seen.reset()
    got = call(operands=(a, a))
    assert seen.compiles == 0
    _uncached(monkeypatch)
    fresh = call.operands()[0]
    _assert_same(_parts(got), _parts(call(operands=(fresh, fresh))))


def test_values_written_in_place_between_two_hits_are_read(seen: _Seen) -> None:
    call = _spmm()
    c, b = call.operands()
    call(operands=(c, b))
    call(operands=(c, b))
    seen.reset()
    with torch.no_grad():
        b.values.mul_(2.0)
    got = call(operands=(c, b))
    assert seen.compiles == 0
    assert_matches(got, 2.0 * call.reference, call.dtype)


def test_a_relaid_operand_is_relaid_on_every_hit(seen: _Seen) -> None:
    call = _copy_relaid()
    operands = call.operands()
    first = call(operands=operands)
    seen.reset()
    second = call(operands=operands)
    assert seen.compiles == 0
    assert_matches(second, call.reference, call.dtype)
    # The caller's operand is not moved: the relayout works on a copy.
    assert tuple(operands[0].mode_order) == (1, 0, 2)
    _assert_same(_parts(second), _parts(first))


# -- 4. the hit check -------------------------------------------------------------------


def test_the_hit_check_passes_a_correct_hit(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = _spmm()
    call()
    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [True])
    seen.reset()
    got = call()
    assert seen.compiles == 1  # the check's own full build
    assert _store().lookups[-1][1]
    assert_matches(got, call.reference, call.dtype)


def _entry_of(call: Call, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The entry ``call`` builds, in a store of its own."""

    store = _Recording()
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", store)
    call()
    (entry,) = [value for key, value in store.items() if key[0] == "lower_and_exec_cin"]
    return entry


@pytest.mark.parametrize("change", CHANGES)
def test_a_key_without_the_input_fails_its_row(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One negative control per row of ``test_a_changed_input_misses...``: with the key
    positions that input enters left out, the changed call hits the first call's entry.
    The hit check then refuses it when the changed call builds another entry, and passes
    it when it builds the same one.  The tensor name, the operand dtype and the operand
    format change the emitted C++, so those rows must be refused."""

    second, positions = _changed_calls()[change]
    differs = stensor_module._kept_kernel_identity(
        _entry_of(_spmm(), monkeypatch)
    ) != stensor_module._kept_kernel_identity(_entry_of(second, monkeypatch))
    if change in ("tensor name", "index name", "operand dtype", "operand format"):
        assert differs
    store = _KeyWithout(tuple(positions))
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", store)
    _spmm()()
    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [True])
    if differs:
        with pytest.raises(GeneratedKernelCacheError) as refused:
            second()
        assert refused.value.code == "kept_kernel_differs_from_built"
    else:
        assert_matches(second(), second.reference, second.dtype)
    assert store.lookups[-1][1]


def test_a_statement_key_without_names_fails_the_hit_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The statement key keeps names because the emitted C++ spells them."""

    def without_names(cin: Any) -> Optional[tuple]:
        key = cin_lowering_key(cin)
        names = {"C", "B", "Y", "P", "i", "j", "k", "r"}
        return (
            None
            if key is None
            else tuple("name" if part in names else part for part in key)
        )

    monkeypatch.setattr(ops, "cin_lowering_key", without_names)
    _spmm()()
    second, _ = _changed_calls()["tensor name"]
    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [True])
    with pytest.raises(GeneratedKernelCacheError):
        second()


# -- 5. a hit still refuses what a cold call refuses ------------------------------------


def _refusal(call: Callable[[], Any]) -> Tuple[type, str, Optional[str]]:
    """The exception's type, its message with numbers masked (a statement built
    again has new stable IDs) and its code."""

    with pytest.raises(Exception) as raised:
        call()
    message = re.sub(r"\d+", "#", str(raised.value))
    return type(raised.value), message, getattr(raised.value, "code", None)


def _cold_and_warm_refusals(
    monkeypatch: pytest.MonkeyPatch, bad_call: Callable[[], Any]
) -> Tuple[tuple, tuple]:
    _uncached(monkeypatch)
    cold = _refusal(bad_call)
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording())
    _spmm()()
    warm = _refusal(bad_call)
    return cold, warm


def _broken_positions() -> Tuple[STensor, STensor]:
    call = _spmm()
    c, b = call.operands()
    positions = c._native_mode_indices()[1][0]
    positions[-1] += 1  # the last position no longer equals the stored count
    return c, b


def _positions_written_after_a_call(call: Call) -> Any:
    c, b = call.operands()
    call(operands=(c, b))
    c._native_mode_indices()[1][0][-1] += 1
    return call(operands=(c, b))


@pytest.mark.parametrize(
    "bad",
    [
        "broken positions",
        "positions written in place after a call",
        "not an STensor",
        "one operand too few",
        "one operand too many",
        "result shape of floats",
        "statement of another type",
        "statement with a cycle",
    ],
)
def test_a_hit_refuses_what_a_cold_call_refuses(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = _spmm()

    def bad_call() -> Any:
        if bad == "broken positions":
            return call(operands=_broken_positions())
        if bad == "positions written in place after a call":
            return _positions_written_after_a_call(call)
        if bad == "not an STensor":
            return call(operands=(call.operands()[0], torch.ones(6, 3, dtype=F64)))
        if bad == "one operand too few":
            return call(operands=call.operands()[:1])
        if bad == "one operand too many":
            return call(operands=call.operands() + call.operands()[1:])
        if bad == "result shape of floats":
            return ops.lower_and_exec_cin(
                call.statement(), (7.0, 3.0), *call.operands()
            )
        if bad == "statement of another type":
            return ops.lower_and_exec_cin(
                "Y[i,k] += C[i,j] * B[j,k]", (7, 3), *call.operands()
            )
        statement = call.statement()
        innermost = statement.stmt.stmt
        innermost.stmt = statement  # a cycle: the innermost loop's body is the root
        return call(statement)

    cold, warm = _cold_and_warm_refusals(monkeypatch, bad_call)
    assert warm == cold


# A child interpreter: with a key that left the shapes out, a disagreeing call would run
# the kept kernel past an argument's end, which may crash, and must fail the test, not
# end the process (as in test_index_extent_agreement).
_EXTENT_CHILD = """
import torch

import scorch.ops as ops
import scorch.stensor as stensor_module
from scorch import STensor
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.exceptions import defect_code

F64 = torch.float64
generator = torch.Generator().manual_seed(0)
C = (torch.rand(7, 6, generator=generator, dtype=F64) + 0.5) * (
    torch.rand(7, 6, generator=generator) < 0.4
)
B = torch.rand(7, 4, generator=generator, dtype=F64) + 0.5


def statement():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="ds", mode_order=[0, 1], dtype=F64)
    b = TensorVar("B", fmt="dd", mode_order=[0, 1], dtype=F64)
    y = TensorVar("Y", fmt="dd", mode_order=[0, 1], dtype=F64)
    body = TensorAssign(y[i, k], c[i, j] * b[j, k], op=Operation.ADD)
    return ForAll(i, ForAll(j, ForAll(k, body)))


def outcome(shape, b_rows=6, b_cols=3):
    c = STensor.from_torch(C.clone(), "C").to_sparse("ds")
    b = STensor.from_torch(B[:b_rows, :b_cols].clone(), "B").to_dense()
    try:
        ops.lower_and_exec_cin(statement(), shape, c, b)
    except Exception as error:
        first = str(error).splitlines()[0]
        return f"{type(error).__name__} {defect_code(error)} {first}"
    return "RETURNED"


class Store(dict):
    def __init__(self, keep=True, without=()):
        super().__init__()
        self.keep, self.without, self.hits = keep, without, 0

    def _stored(self, key):
        if key[0] != "lower_and_exec_cin":
            return key
        return tuple("left out" if at in self.without else part for at, part in enumerate(key))

    def get(self, key, default=None):
        found = super().get(self._stored(key), default)
        if found is not None and key[0] == "lower_and_exec_cin":
            self.hits += 1
        return found

    def __setitem__(self, key, value):
        if self.keep:
            super().__setitem__(self._stored(key), value)


CASES = [
    ("result one row short", (6, 3), {}),
    ("result one row long", (8, 3), {}),
    ("result one column short", (7, 2), {}),
    ("result one column long", (7, 4), {}),
    ("B one row short", (7, 3), {"b_rows": 5}),
    ("B one row long", (7, 3), {"b_rows": 7}),
    ("B one column long", (7, 3), {"b_cols": 4}),
]


def run(mode, store, check):
    stensor_module._GENERATED_KERNELS = store
    stensor_module._CHECK_GENERATED_KERNEL_HITS = [check]
    if mode != "cold":
        print(mode, "| agreeing |", outcome((7, 3)))
    for name, shape, kwargs in CASES:
        print(mode, "|", name, "|", outcome(shape, **kwargs))
    kept = sum(1 for key in store if key[0] == "lower_and_exec_cin")
    print(mode, "| hits", store.hits, "| kept", kept)


run("cold", Store(keep=False), False)
run("warm", Store(), False)
# The key's result shape and operand positions (2 and 3) left out: every disagreeing
# call hits the agreeing call's kernel, and only the hit check's full build refuses it.
run("checked", Store(without=(2, 3)), True)
"""


def test_a_warm_process_refuses_every_disagreeing_extent_as_a_cold_one() -> None:
    """Step 15's refusals (``IndexExtentMismatchError``, patch 0076) depend only on the
    result shape and the operands' shapes, which the key holds, so a disagreeing call
    misses the agreeing call's kernel and the full path refuses it, before any kernel
    runs; nothing is kept for it.  The negative control leaves the shapes out of the
    key: the disagreeing calls then hit, and the hit check's full build refuses each
    one before the kernel runs."""

    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_EXTENT_CHILD)],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert (
        proc.returncode == 0
    ), f"child exited {proc.returncode}: {proc.stderr[-3000:]}"
    lines = [line.split(" | ") for line in proc.stdout.splitlines()]
    by_mode: dict = {}
    for mode, *rest in lines:
        by_mode.setdefault(mode, []).append(rest)
    cold = by_mode["cold"]
    cases = len(cold) - 1
    assert cases == 7
    for name, refusal in cold[:cases]:
        assert refusal.startswith(
            "IndexExtentMismatchError einsum_index_extent_mismatch "
        ), (name, refusal)
    assert cold[cases] == ["hits 0", "kept 0"]
    for mode, hits in (("warm", 0), ("checked", cases)):
        got = by_mode[mode]
        assert got[0] == ["agreeing", "RETURNED"]
        assert got[1 : cases + 1] == cold[:cases]
        assert got[cases + 1] == [f"hits {hits}", "kept 1"]


def test_a_hit_under_a_verifying_policy_runs_full_verification(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = _spmm()
    verified: List[object] = []
    verify = ops.verify_cin_if_enabled

    def recorded(cin: Any, enabled: Optional[bool] = None) -> Any:
        verified.append(enabled)
        return verify(cin, enabled)

    monkeypatch.setattr(ops, "verify_cin_if_enabled", recorded)
    with full_cin_verification(True):
        call()
        seen.reset()
        call()
    assert seen.compiles == 0
    assert verified == [True]
    verified.clear()
    call()
    call()
    assert verified == []


def test_a_requested_schedule_is_refused_on_a_warm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _spmm()

    def scheduled() -> Any:
        with schedule_force(Schedule()):
            return call()

    cold, warm = _cold_and_warm_refusals(monkeypatch, scheduled)
    assert warm == cold
    assert cold[0] is CompileOptionsError
    assert "does not support explicit scheduling" in cold[1]


def test_a_context_of_another_snapshot_is_refused_on_a_warm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _spmm()
    options = CompileOptions.from_environment()

    def mismatched() -> Any:
        return call(
            _compile_options=options,
            _compilation_context=CompilationContext(CompileOptions.from_environment()),
        )

    cold, warm = _cold_and_warm_refusals(monkeypatch, mismatched)
    assert warm == cold
    assert cold[2] == "compilation_context_type_invalid"


# -- 6. a handed context ----------------------------------------------------------------


def _stage_values(context: CompilationContext) -> List[str]:
    return [record.stage_id.value for record in context.stage_run_records]


def test_a_handed_context_sees_nothing_recorded_on_a_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _spmm()
    options = CompileOptions.from_environment()
    _uncached(monkeypatch)
    uncached_context = CompilationContext(options)
    call(_compile_options=options, _compilation_context=uncached_context)
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording())
    cold_context = CompilationContext(options)
    call(_compile_options=options, _compilation_context=cold_context)
    warm_context = CompilationContext(options)
    warm = call(_compile_options=options, _compilation_context=warm_context)
    assert _stage_values(cold_context) == _stage_values(uncached_context)
    assert _stage_values(cold_context)
    assert _stage_values(warm_context) == []
    assert warm_context.llir_pass_run_records == ()
    assert_matches(warm, call.reference, call.dtype)


# -- 7. a call that raises keeps nothing ------------------------------------------------


def test_a_call_that_raises_keeps_nothing(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = RuntimeError("stop at the loader")
    load = ops._load_validated_prepared_kernel

    def refusing(prepared: Any) -> Any:
        raise error

    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", refusing)
    call = _spmm()
    with pytest.raises(RuntimeError) as raised:
        call()
    assert raised.value is error
    assert dict(_store()) == {}
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", load)
    seen.reset()
    assert_matches(call(), call.reference, call.dtype)
    assert seen.compiles == 1


def test_options_that_are_not_reusable_keep_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ops._DeferredOptions, "still_reusable", lambda self: False)
    call = _spmm()
    assert_matches(call(), call.reference, call.dtype)
    assert dict(_store()) == {}


# -- 8. the statement key ---------------------------------------------------------------


def _spmv_with_workspace(dim: int = 0, dense: Optional[bool] = None) -> Any:
    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d", dtype=F64)
    a = TensorVar("A", fmt="ds", dtype=F64)
    x = TensorVar("x", fmt="d", dtype=F64)
    workspace = Workspace(name="wksp", dim=dim, dtype=F64)
    if dense is not None:
        workspace.dense = dense
    return ForAll(
        i,
        Where(
            producer=ForAll(
                j,
                TensorAssign(
                    workspace.get_default_access(), a[i, j] * x[j], op=Operation.ADD
                ),
            ),
            consumer=TensorAssign(y[i], workspace.get_default_access()),
        ),
    )


def test_a_statement_built_again_has_the_same_key() -> None:
    assert cin_lowering_key(_spmm_statement()) == cin_lowering_key(_spmm_statement())
    assert cin_lowering_key(_spmv_with_workspace()) == cin_lowering_key(
        _spmv_with_workspace()
    )
    assert cin_lowering_key(_spmm_statement()) is not None


def _with(statement: Any, edit: Callable[[Any], None]) -> Any:
    edit(statement)
    return statement


def _result_var(statement: Any) -> Any:
    return statement.stmt.stmt.stmt.lhs.tensor


EDITS = {
    "tensor name": lambda s: setattr(_result_var(s), "_name", "Z"),
    "tensor format": lambda s: setattr(
        s.stmt.stmt.stmt.rhs.left.tensor,
        "_format",
        _spmm_statement(fmt_c="dd").stmt.stmt.stmt.rhs.left.tensor._format,
    ),
    "tensor shape": lambda s: setattr(_result_var(s), "shape", (7, 3)),
    "tensor dtype": lambda s: setattr(_result_var(s), "dtype", F32),
    "tensor mode order": lambda s: setattr(_result_var(s), "mode_order", [1, 0]),
    "index name": lambda s: setattr(s.index_var, "_name", "r"),
    "parallel": lambda s: setattr(s, "parallel", True),
    "assignment op": lambda s: setattr(s.stmt.stmt.stmt, "op", None),
    "binary op": lambda s: object.__setattr__(
        s.stmt.stmt.stmt.rhs, "op", Operation.ADD
    ),
}


@pytest.mark.parametrize("edit", sorted(EDITS))
def test_every_field_the_normalization_copies_enters_the_key(edit: str) -> None:
    assert cin_lowering_key(_with(_spmm_statement(), EDITS[edit])) != cin_lowering_key(
        _spmm_statement()
    )


@pytest.mark.parametrize(
    "workspace", [{"dim": 1}, {"dense": True}], ids=["dim", "dense"]
)
def test_a_workspace_s_fields_enter_the_key(workspace: dict) -> None:
    assert cin_lowering_key(_spmv_with_workspace(**workspace)) != cin_lowering_key(
        _spmv_with_workspace()
    )


def test_one_index_object_and_two_of_one_name_have_different_keys() -> None:
    """The clone keeps object identity (its memos), so the key does too."""

    i, j, other_i = IndexVar("i"), IndexVar("j"), IndexVar("i")
    a = TensorVar("A", fmt="dd", dtype=F64)
    b = TensorVar("B", fmt="dd", dtype=F64)
    shared = ForAll(i, ForAll(j, TensorAssign(b[i, j], a[i, j])))
    i2, j2 = IndexVar("i"), IndexVar("j")
    a2 = TensorVar("A", fmt="dd", dtype=F64)
    b2 = TensorVar("B", fmt="dd", dtype=F64)
    split = ForAll(i2, ForAll(j2, TensorAssign(b2[i2, j2], a2[other_i, j2])))
    assert cin_lowering_key(shared) != cin_lowering_key(split)


def test_fields_the_normalization_drops_do_not_enter_the_key() -> None:
    """Parent pointers and the scheduler's marks are not copied, so they cannot change
    the kernel and do not split the key."""

    statement = _spmm_statement()
    before = cin_lowering_key(statement)
    statement.index_var.is_tiled = True
    statement.index_var.is_outer = True
    assert cin_lowering_key(statement) == before


def _spmm_statement_built_in_reverse() -> Any:
    """``_spmm_statement()``'s structure with every index, tensor and access made in the
    reverse order, so the IDs of each of those types compare the other way round."""

    k, j, i = IndexVar("k"), IndexVar("j"), IndexVar("i")
    yv = TensorVar("Y", fmt="dd", mode_order=[0, 1], dtype=F64)
    bv = TensorVar("B", fmt="dd", mode_order=[0, 1], dtype=F64)
    cv = TensorVar("C", fmt="ds", mode_order=[0, 1], dtype=F64)
    b_access, c_access, y_access = bv[j, k], cv[i, j], yv[i, k]
    assignment = TensorAssign(y_access, c_access * b_access, op=Operation.ADD)
    return ForAll(i, ForAll(j, ForAll(k, assignment)))


def test_how_each_id_type_s_values_compare_enters_the_key() -> None:
    """The lowering sorts the accesses by ``access_id.value`` (legacy_cin_adapter), so the
    key holds, per ID type, the order of its IDs' values: a statement built in reverse
    differs from the forward one there and only there, and built again in the same order
    it has the same key."""

    forward = cin_lowering_key(_spmm_statement())
    reverse = cin_lowering_key(_spmm_statement_built_in_reverse())
    assert forward is not None and reverse is not None
    assert forward[:-1] == reverse[:-1]
    assert forward[-1] != reverse[-1]
    assert reverse == cin_lowering_key(_spmm_statement_built_in_reverse())
    # Two accesses' IDs swapped: the same structure, another order of AccessIds only.
    statement = _spmm_statement()
    rhs = _assignment(statement).rhs
    before = cin_lowering_key(statement)
    rhs.left.access_id, rhs.right.access_id = rhs.right.access_id, rhs.left.access_id
    after = cin_lowering_key(statement)
    assert before is not None and after is not None
    assert before[:-1] == after[:-1]
    assert [b[0] for b, a in zip(before[-1], after[-1]) if b != a] == ["AccessId"]


def test_a_statement_built_in_reverse_is_right_under_the_hit_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal statements whose IDs compare the other way round, each called twice with the
    hit check on: every result matches the reference, no hit is refused, and the
    reversed statement does not run the forward statement's kernel."""

    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [True])
    call = _spmm()
    for build in (_spmm_statement, _spmm_statement_built_in_reverse) * 2:
        assert_matches(call(build()), call.reference, call.dtype)
    hits = [hit for key, hit in _store().lookups if key[0] == "lower_and_exec_cin"]
    assert hits == [False, False, True, True]


def test_the_key_costs_less_than_the_normalization_it_mirrors() -> None:
    """A guard on the per-call cost: the key is one walk of the statement."""

    import cProfile
    import pstats

    statement = _spmm_statement()

    def calls(function: Callable[[], Any]) -> int:
        profile = cProfile.Profile()
        profile.enable()
        function()
        profile.disable()
        return sum(value[1] for value in pstats.Stats(profile).stats.values())

    from scorch.compiler.cin_analysis import normalize_cin

    options = CompileOptions.from_environment()
    assert calls(lambda: cin_lowering_key(statement)) < calls(
        lambda: normalize_cin(statement, compile_options=options)
    )
