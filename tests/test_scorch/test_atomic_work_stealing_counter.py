"""The shared counter of atomic work stealing is int64_t whatever the loop's type.

A parallel loop over a compressed level with a per-worker workspace is emitted as atomic
work stealing (parallel_marking_pass.py marks it; codegen.py emits it, for both
compilers: the new compiler's loops reach the same branch of ``LLIRLowerer``).  Each
worker takes ``_chunk`` iterations at a time with ``fetch_add`` on one shared counter
until a fetch returns a start at or past the bound; that last fetch still adds a chunk,
so with T workers the counter ends at up to bound - 1 + (T + 1) x chunk.

Before patch 0051 the counter had the loop variable's type.  For an ``int`` loop it
wrapped once the bound came within (T + 1) x chunk of 2**31 - 1, and a worker that read
the wrapped, negative start passed ``_start >= bound`` and ran iterations below zero.
Now the counter, the start and the chunk end are int64_t, and only the end, which is at
most the bound, is narrowed to the loop's type.  No kernel of today's population takes
this branch with an ``int`` loop; the tests drive the emitter directly.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import pytest
from torch.utils.cpp_extension import load_inline

from scorch.compiler import llir
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.compile_options import CompileOptions

M = 2**31 - 1


def _atomic_loop(loop_type: llir.DataType, bound: str, body, *, threads="4"):
    """The loop ``for (i = 0; i < bound; i++)`` marked for atomic work stealing, as
    parallel_marking_pass.py marks it."""
    index = llir.Var(name="i", type=loop_type)
    loop = llir.ForLoop(
        init=llir.VarInit(var=index, value=llir.Literal(0, llir.DataType.INT)),
        cond=llir.BinOp(
            "<",
            llir.Var(name="i", type=loop_type),
            llir.Var(name=bound, type=loop_type),
        ),
        update=llir.Increment(llir.Var(name="i", type=loop_type)),
        body=body,
    )
    loop.omp_num_threads = threads
    loop.pre_parallel_body = [llir.RawStmt("int _chunk = 256")]
    setattr(loop, "_use_atomic_scheduling", True)
    setattr(loop, "_atomic_chunk_var", "_chunk")
    setattr(loop, "_atomic_counter_var", "_next_row")
    setattr(loop, "_loop_bound", bound)
    return loop


def _lines(text: str):
    return [line.strip() for line in text.splitlines()]


def test_an_int_loop_counts_in_int64_and_narrows_only_its_end():
    text = LLIRLowerer().lower_llir(
        _atomic_loop(llir.DataType.INT, "pB1_end", [llir.RawStmt("work(i)")])
    )
    lines = _lines(text)
    assert "std::atomic<int64_t> _next_row{0};" in lines
    assert (
        "const int64_t _start = _next_row.fetch_add(_chunk, std::memory_order_relaxed);"
        in lines
    )
    assert "if (_start >= pB1_end) break;" in lines
    assert (
        "const int _end = static_cast<int>(std::min<int64_t>(_start + _chunk, "
        "pB1_end));" in lines
    )
    assert "for (int i = static_cast<int>(_start); i < _end; i++) {" in lines
    assert "std::atomic<int>" not in text
    assert lines.index("if (_start >= pB1_end) break;") < lines.index(
        "for (int i = static_cast<int>(_start); i < _end; i++) {"
    )


def test_an_int64_loop_keeps_the_text_it_had():
    """The dense-level form, the one today's population reaches, is unchanged."""
    text = LLIRLowerer().lower_llir(
        _atomic_loop(llir.DataType.INT64, "A0_size", [llir.RawStmt("work(i)")])
    )
    lines = _lines(text)
    assert "std::atomic<int64_t> _next_row{0};" in lines
    assert (
        "const int64_t _start = _next_row.fetch_add(_chunk, std::memory_order_relaxed);"
        in lines
    )
    assert "const int64_t _end = std::min<int64_t>(_start + _chunk, A0_size);" in lines
    assert "for (int64_t i = _start; i < _end; i++) {" in lines
    assert "static_cast" not in text


@lru_cache(maxsize=1)
def _driver():
    """The emitted region for an ``int`` loop, built with the flags scorch's generated
    kernels get (CompileOptions' build snapshot, so the platform's OpenMP flags, as
    test_jit_build_errors.py builds).  Each iteration in
    [0, bound) marks its own byte (2 GiB at the largest bound); an iteration outside it
    is counted.  Returns the bytes not marked exactly once, and the outside count."""
    loop = _atomic_loop(
        llir.DataType.INT,
        "bound",
        [
            llir.RawStmt("if (i < 0 || i >= bound) _outside += 1; else _seen[i] += 1"),
        ],
    )
    loop.before_parallel_body = [
        llir.RawStmt("std::vector<uint8_t> _seen(static_cast<size_t>(bound), 0)"),
        llir.RawStmt("std::atomic<int64_t> _escaped{0}"),
    ]
    loop.pre_parallel_body = [
        llir.RawStmt("int _chunk = 256"),
        llir.RawStmt("int64_t _outside = 0"),
    ]
    loop.post_parallel_body = [
        llir.RawStmt("_escaped.fetch_add(_outside)"),
    ]
    region = LLIRLowerer().lower_llir(loop, 1)
    source = (
        "#include <algorithm>\n#include <atomic>\n#include <cstdint>\n#include <vector>\n"
        "#include <omp.h>\n\n"
        "std::vector<int64_t> drive(int64_t bound_argument) {\n"
        "  const int bound = static_cast<int>(bound_argument);\n"
        f"{region}\n"
        "  const int64_t wrong = std::count_if(_seen.begin(), _seen.end(),\n"
        "                                      [](uint8_t mark) { return mark != 1; });\n"
        "  return {wrong, _escaped.load()};\n"
        "}\n"
    )
    build = CompileOptions.from_environment(
        forced_schedule=None, regblock_override=None, verify_cin_override=False
    ).build
    flags = (tuple(build.extra_cflags), tuple(build.extra_ldflags))
    digest = hashlib.sha256((source + repr(flags)).encode()).hexdigest()[:12]
    return load_inline(
        name=f"scorch_atomic_counter_{digest}",
        cpp_sources=source,
        functions=["drive"],
        extra_cflags=list(flags[0]),
        extra_ldflags=list(flags[1]),
        verbose=False,
    )


@pytest.mark.parametrize("bound", [M, M - 256, 1_000_003])
def test_an_int_loop_at_the_bound_runs_each_iteration_once(bound):
    """Four workers, chunks of 256: at a bound of 2**31 - 1 the counter ends past it by
    up to five chunks, which an ``int`` counter wrapped into negative starts."""
    wrong, escaped = _driver().drive(bound)
    assert escaped == 0
    assert wrong == 0
