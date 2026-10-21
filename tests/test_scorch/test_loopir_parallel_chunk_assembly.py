"""The parallel single-pass sparse assembly, and the gate that keeps it out.

The ordered-key family emits a SERIAL single-pass builder where the legacy
pipeline emits a PARALLEL two-phase one.  A three-column source-level ablation
(``~/.cache/scorch-codex/ttm-density-mechanism/ABLATION.md``) measured that the
whole gap is parallelism: with legacy's two pragmas deleted the typed single
pass is 1.18-1.47x FASTER, so adopting the two-phase strategy would buy
parallelism by surrendering a real advantage.  The chunked strategy instead
parallelizes the single pass over per-chunk output buffers concatenated in
outer-loop order.

**The strategy is reached by request** (``Schedule(assembly=
"single_pass_chunk_parallel")``).  Where a program can also be assembled in two
parallel passes, which is the shipped compiler's strategy, that is what an
automatic plan records from ``2 * ROWS_PER_THREAD`` rows
(``loopir/assembly_rule.py``): the chunked form's two-host runtime measurement
(review section 59.9) has it faster than the two-pass on some shapes and slower
than the serial kernel on others, and no condition separates them yet.  The
tests below therefore request it, and the execution tests hold the automatic
plan's kernel to the same storage.

Four properties are locked here, and each one is a thing that would otherwise be
a claim rather than a fact.

**The serial arm is the ungated nest, verbatim, over the function's own
locals.**  Three structures were measured, because the obvious argument -- a
byte-identical arm must be inert -- is false.  Giving each arm its own copy made
the arm that RUNS up to 34% slower, with the other copy never executed.  Routing
the serial arm through a shared body instead was worse, 3-55%, and
``__restrict__`` recovered none of it: the cost is that reference parameters
make the buffers escape, not that they alias.  So the serial arm keeps the
original inline nest, and the shared body is used by the parallel arm only,
which pays the escape either way.  The locks pin that arm byte-for-byte against
the ungated emission and require no pragma on it.

**A gate that can never open is never emitted unasked.**  An outer extent below
``2 * ROWS_PER_THREAD`` cannot reach two threads for any operand, so an
automatic plan records the serial single pass there and the kernel is the
serial nest byte for byte.  That is the only way the cost is exactly zero
rather than small: an unexecuted arm is not free.  A request at that extent is
legal and is honoured.

**A receiver the transformation does not admit is refused it.**  A request for
the chunked strategy on such a receiver raises, including the
dense-prefix-deeper-than-one case that is deliberately out of scope, and the
automatic plan's kernel for it is the serial one.

**Ordering survives the chunk boundary.**  Concatenating per-chunk buffers is
only correct because the outer loop binds result level 0; the locks execute at
shapes where the thread count is genuinely above one and require exact storage
equality against the serial arm, not merely numeric agreement.

**The two kernels open on the same gate.**  The emitted thread count is spelled
from the same derivation legacy's own pragma uses, so the parallel/serial
crossover is the one the mechanism study characterized rather than a new one.
"""

import re
import textwrap
from dataclasses import replace

import pytest
import torch

from scorch.compiler import llir
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import InvalidSchedule, UnsupportedFeature
from scorch.compiler.loopir import lower_llir as lower_llir_module
from scorch.compiler.loopir.parallel_chunk_assembly import (
    GENERATED_NAMES,
    PARALLEL_CHUNK_RUNTIME_SPELLINGS,
    ROWS_PER_THREAD,
    ParallelChunkAssemblyContext,
    build_parallel_chunk_assembly,
    chunked_vector_names,
    parallel_chunk_generated_names,
)
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor

_F32 = torch.float32


CHUNKED = "single_pass_chunk_parallel"

#: What a request for a strategy the program cannot be assembled with raises.
REFUSALS = (lower_llir_module.LoopIRTargetError, UnsupportedFeature, InvalidSchedule)


def auto_options(regblock_enabled=False, *, jit=False, assembly=None):
    """The automatic schedule, with an assembly strategy requested or not.

    The chunked single pass is reached by request: where a program can also be
    assembled in two parallel passes, the shipped compiler's strategy, that is
    what the automatic rule records (``loopir/assembly_rule.py``).
    """

    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(assembly=assembly),
    )


def chunked_options(regblock_enabled=False, *, jit=False):
    return auto_options(regblock_enabled, jit=jit, assembly=CHUNKED)


def ttm_cin(a_fmt, b_fmt, c_fmt, dtype=_F32, names=("i", "j", "k", "l")):
    """``C[i,j,l] += A[i,j,k] * B[k,l]`` -- tensor times matrix."""

    # ``l`` is the contraction's own index name, so E741 is silenced rather than
    # renamed: every other TTM site in the tree spells it the same way.
    i, j, k, l = (IndexVar(name) for name in names)  # noqa: E741
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)[i, j, k]
    b = TensorVar("B", fmt=b_fmt, dtype=dtype)[k, l]
    c = TensorVar("C", fmt=c_fmt, dtype=dtype)[i, j, l]
    statement = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for variable in (l, k, j, i):
        statement = ForAll(variable, statement)
    return statement


def reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices):
    ivars = {name: IndexVar(name) for name in operand_indices}
    operand = TensorVar("A", fmt=operand_fmt, dtype=_F32)[
        tuple(ivars[name] for name in operand_indices)
    ]
    result = TensorVar("C", fmt=result_fmt, dtype=_F32)[
        tuple(ivars[name] for name in result_indices)
    ]
    statement = TensorAssign(result, operand, op=Operation.ADD)
    for name in reversed(operand_indices):
        statement = ForAll(ivars[name], statement)
    return statement


def random_dense(shape, seed, density=0.4, dtype=_F32):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64)
    mask = torch.rand(shape, generator=generator, dtype=torch.float64) < density
    return (((values + 0.5) * 2.0 - 1.5) * mask).to(dtype)


def sparse(dense, fmt, name):
    return STensor.from_torch(dense.clone(), name).to_sparse(fmt)


def emit(statement, result_shape, bindings, options=None):
    return compile_cin_via_loopir(
        statement, result_shape, bindings, compile_options=options or auto_options()
    ).cpp_source


def emit_chunked(statement, result_shape, bindings, regblock_enabled=False):
    """The program with the chunked single pass requested."""

    return emit(statement, result_shape, bindings, chunked_options(regblock_enabled))


def emit_ungated(statement, result_shape, bindings):
    """The same program with the serial single pass requested: the nest alone."""

    return emit(
        statement,
        result_shape,
        bindings,
        auto_options(assembly="single_pass_serial"),
    )


# The outer extent must clear 2 * ROWS_PER_THREAD or the gate is declined at
# compile time (see the extent locks below), which would make every emission
# lock here vacuous.
TTM_SHAPE = (32, 5, 6, 7)
TTM_RESULT = (32, 5, 7)
TTM_BINDINGS = (((32, 5, 6), _F32), ((6, 7), _F32))


def serial_arm(source):
    """The dedented body of the gate's ``else`` arm."""

    marker = "\n  } else {\n"
    start = source.index(marker)
    body = source[start + len(marker) :]
    return textwrap.dedent(body[: body.index("\n  }\n")]).strip("\n")


def shared_nest(source):
    """The one assembly nest, from inside the shared lambda, header normalized."""

    match = re.search(
        r"^    for \(int64_t (\w+) = _assembly_lo; \1 < _assembly_hi; \1\+\+\) \{$",
        source,
        re.M,
    )
    assert match is not None, "the gated emission must hold one ranged nest"
    nest = textwrap.dedent(source[match.start() : source.index("\n  };\n")]).strip("\n")
    index = match.group(1)
    return nest.replace(
        f"for (int64_t {index} = _assembly_lo; {index} < _assembly_hi; {index}++) {{",
        "<HEADER>",
        1,
    )


def base_nest(source):
    """An ungated emission's nest, header normalized the same way."""

    match = re.search(
        r"^  for \(int64_t (\w+) = 0; \1 < (\w+); \1\+\+\) \{$", source, re.M
    )
    assert match is not None, "the ungated emission must hold one top-level nest"
    tail = source.index("\n  // Assemble final result")
    nest = textwrap.dedent(source[match.start() : tail]).strip("\n")
    index, bound = match.group(1), match.group(2)
    return nest.replace(
        f"for (int64_t {index} = 0; {index} < {bound}; {index}++) {{", "<HEADER>", 1
    )


# -- admission ---------------------------------------------------------------


@pytest.mark.parametrize("arm", [False, True])
def test_dense_prefix_receiver_is_gated_in_both_arms(arm):
    source = emit_chunked(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS, arm)
    assert "int _assembly_threads = scorch_nthreads(" in source
    assert "if (_assembly_threads > 1) {" in source
    assert source.count("#pragma omp") == 1


def test_arms_generate_identical_gated_sources():
    off = emit_chunked(ttm_cin("dss", "ss", "dss"), TTM_RESULT, TTM_BINDINGS, False)
    on = emit_chunked(ttm_cin("dss", "ss", "dss"), TTM_RESULT, TTM_BINDINGS, True)
    assert off == on


@pytest.mark.parametrize(
    "operand_fmt,result_fmt,operand_indices,result_indices,shape",
    [
        ("ss", "s", "ij", "j", (4, 5)),
        ("sss", "ss", "ijk", "ik", (3, 4, 5)),
        ("dss", "ss", "ijk", "jk", (3, 4, 5)),
        ("ssss", "sss", "ijkl", "jkl", (2, 3, 4, 5)),
    ],
)
def test_receivers_without_a_dense_prefix_are_untouched(
    operand_fmt, result_fmt, operand_indices, result_indices, shape
):
    """No dense level 0 means no outer loop that partitions the result.

    The automatic plan records the serial single pass for it, and a request for
    the chunked one is refused, not answered with the serial kernel.
    """

    statement = reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices)
    result_shape = tuple(shape[operand_indices.index(name)] for name in result_indices)
    bindings = ((shape, _F32),)
    gated = emit(statement, result_shape, bindings)
    assert "_assembly_threads" not in gated
    assert "#pragma omp" not in gated
    ungated = emit_ungated(
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
        result_shape,
        bindings,
    )
    assert gated == ungated
    with pytest.raises(REFUSALS):
        emit_chunked(
            reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
            result_shape,
            bindings,
        )


def test_dense_prefix_deeper_than_one_is_declined_and_byte_neutral():
    """``dds`` indexes its first compressed level by a FLATTENED dense cell.

    The mechanism generalizes to it; the index derivation does not come for
    free, so it is declined rather than guessed at -- and declining has to cost
    the emission nothing.
    """

    gated = emit(ttm_cin("dss", "dd", "dds"), TTM_RESULT, TTM_BINDINGS)
    assert "_assembly_threads" not in gated
    assert "#pragma omp" not in gated
    ungated = emit_ungated(ttm_cin("dss", "dd", "dds"), TTM_RESULT, TTM_BINDINGS)
    assert gated == ungated
    with pytest.raises(REFUSALS):
        emit_chunked(ttm_cin("dss", "dd", "dds"), TTM_RESULT, TTM_BINDINGS)


# -- the shape of the gate ---------------------------------------------------


def test_serial_arm_is_the_ungated_nest_verbatim():
    """The arm that runs at one thread must be the base's, character for character."""

    gated = emit_chunked(ttm_cin("dss", "ss", "dss"), TTM_RESULT, TTM_BINDINGS)
    ungated = emit_ungated(ttm_cin("dss", "ss", "dss"), TTM_RESULT, TTM_BINDINGS)
    header = re.search(
        r"^  for \(int64_t (\w+) = 0; \1 < (\w+); \1\+\+\) \{$", ungated, re.M
    )
    spelling = (
        f"for (int64_t {header.group(1)} = 0; "
        f"{header.group(1)} < {header.group(2)}; {header.group(1)}++) {{"
    )
    assert serial_arm(gated) == base_nest(ungated).replace("<HEADER>", spelling)


def test_the_parallel_arm_calls_one_shared_body_holding_the_same_nest():
    source = emit_chunked(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS)
    ungated = emit_ungated(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS)
    assert source.count("auto _assembly_body = [&](") == 1
    # Called once per chunk from inside the region, and nowhere else.  The
    # definition spells "_assembly_body = [&](", so it is not a call match.
    assert source.count("_assembly_body(") == 1
    assert shared_nest(source) == base_nest(ungated)


def test_the_serial_arm_carries_no_pragma_and_no_buffer_construction():
    arm = serial_arm(
        emit_chunked(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS)
    )
    assert "#pragma" not in arm
    assert "scorch_chunk_buffers" not in arm
    assert "scorch_presize_positions" not in arm
    assert "_assembly_body(" not in arm


@pytest.mark.parametrize("extent", [1, 8, 31])
def test_an_extent_that_cannot_reach_two_threads_is_declined(extent):
    """Below 2 * ROWS_PER_THREAD a parallel assembly could only cost, never pay.

    The automatic plan records the serial single pass there, and its kernel is
    the serial nest alone.  The chunked strategy is still legal at that extent,
    so a request for it is honoured.
    """

    result_shape = (extent, 5, 7)
    bindings = (((extent, 5, 6), _F32), ((6, 7), _F32))
    automatic = emit(ttm_cin("dss", "dd", "dss"), result_shape, bindings)
    assert "_assembly_threads" not in automatic
    assert "#pragma omp" not in automatic
    ungated = emit_ungated(ttm_cin("dss", "dd", "dss"), result_shape, bindings)
    assert automatic == ungated
    assert "_assembly_threads" in emit_chunked(
        ttm_cin("dss", "dd", "dss"), result_shape, bindings
    )


def test_an_extent_that_can_reach_two_threads_gets_a_parallel_assembly():
    """The threshold is a real boundary, and the chunked pass is not its default.

    From ``2 * ROWS_PER_THREAD`` rows the automatic plan records a parallel
    strategy: the two-pass assembly, which this program can be emitted with and
    which is the shipped compiler's.  The chunked single pass is the kernel a
    request for it emits.
    """

    result_shape = (2 * ROWS_PER_THREAD, 5, 7)
    bindings = (((result_shape[0], 5, 6), _F32), ((6, 7), _F32))
    automatic = emit(ttm_cin("dss", "dd", "dss"), result_shape, bindings)
    assert "#pragma omp" in automatic
    assert "_assembly_threads" not in automatic
    assert automatic == emit(
        ttm_cin("dss", "dd", "dss"),
        result_shape,
        bindings,
        auto_options(assembly="two_pass_parallel"),
    )
    assert "_assembly_threads" in emit_chunked(
        ttm_cin("dss", "dd", "dss"), result_shape, bindings
    )


def test_the_pragma_carries_no_if_clause():
    """A false ``if()`` clause still enters the runtime and outlines the body."""

    source = emit_chunked(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS)
    pragma = next(
        line for line in source.splitlines() if line.strip().startswith("#pragma omp")
    )
    assert " if(" not in pragma and " if (" not in pragma
    assert "num_threads(" in pragma


def test_buffers_are_constructed_only_inside_the_taken_branch():
    source = emit_chunked(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS)
    branch = source.index("if (_assembly_threads > 1) {")
    otherwise = source.index("\n  } else {\n")
    for spelling in ("scorch_chunk_buffers", "scorch_presize_positions", "#pragma omp"):
        position = source.index(spelling)
        assert branch < position < otherwise, spelling


def test_thread_count_is_spelled_from_the_same_derivation_as_legacy():
    """Both kernels must open on the SAME gate at the same operands."""

    statement = ttm_cin("dss", "ss", "dss")
    typed = emit_chunked(statement, TTM_RESULT, TTM_BINDINGS)
    legacy = legacy_generated_cpp(
        ttm_cin("dss", "ss", "dss"),
        TTM_RESULT,
        TTM_BINDINGS,
        compile_options=auto_options(),
    )
    policy = re.compile(r"scorch_nthreads\([^)]*\)")
    typed_policies = set(policy.findall(typed))
    legacy_policies = set(policy.findall(legacy))
    assert typed_policies, "the gated emission must request a thread count"
    assert typed_policies == legacy_policies


def test_the_shared_body_takes_every_chunked_vector_under_its_own_spelling():
    """The nest is edit-free only because the parameters keep the names."""

    source = emit_chunked(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS)
    assert (
        "auto _assembly_body = [&](int64_t _assembly_lo, int64_t _assembly_hi, "
        "auto& C1_crd, auto& C2_pos, auto& C2_crd, auto& C_values) {" in source
    )
    # The first compressed level's positions are NOT a parameter: that array is
    # indexed by the outer dense loop, so it is shared and pre-sized, and every
    # chunk writes a disjoint slice of it.
    assert "auto& C1_pos" not in source
    assert "scorch_presize_positions(C1_pos, A0_size + 1);" in source


def test_the_merge_runs_in_chunk_order_over_every_chunked_vector():
    source = emit_chunked(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS)
    assert "scorch_shift_chunk_positions(C1_pos, _chunk_C1_crd," in source
    # Every merge helper takes the assembly thread count: the concatenation is
    # itself parallel over chunks, because once the compute term has been divided
    # by the thread count a serial merge is a large fraction of what is left.
    assert "scorch_concat_chunks(C1_crd, _chunk_C1_crd, _assembly_threads);" in source
    assert (
        "scorch_concat_chunk_positions(C2_pos, _chunk_C2_pos, _chunk_C2_crd, "
        "_assembly_threads);" in source
    )
    assert "scorch_concat_chunks(C2_crd, _chunk_C2_crd, _assembly_threads);" in source
    assert (
        "scorch_concat_chunks(C_values, _chunk_C_values, _assembly_threads);" in source
    )


def test_the_runtime_helpers_are_in_the_packaged_preamble():
    preamble = auto_options().build.preamble_source
    for spelling in PARALLEL_CHUNK_RUNTIME_SPELLINGS:
        assert f"{spelling}(" in preamble, spelling


# -- the transformation in isolation -----------------------------------------


def _context():
    return ParallelChunkAssemblyContext(
        result_name="C",
        shared_position_level=1,
        compressed_levels=(1, 2),
        value_ctype="float",
        index_ctype="int",
    )


def test_chunked_vectors_exclude_the_shared_position_array():
    assert chunked_vector_names(_context()) == (
        "C1_crd",
        "C2_pos",
        "C2_crd",
        "C_values",
    )
    names = parallel_chunk_generated_names(_context())
    assert set(GENERATED_NAMES) <= set(names)
    assert "_chunk_C1_crd" in names and "_chunk_C1_pos" not in names


def _plain_loop(bound="A0_size"):
    index = llir.Var(name="i", type=llir.DataType.INT64)
    return llir.ForLoop(
        init=llir.VarInit(var=index, value=llir.Literal(0, llir.DataType.INT)),
        cond=llir.BinOp("<", index, llir.Var(name=bound, type=llir.DataType.INT64)),
        update=llir.Increment(var=index),
        body=[
            llir.VarInit(
                var=llir.Var(name="pA1_end", type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=llir.Var(name="A1_pos", type=llir.DataType.NO_TYPE),
                    index=llir.Add(
                        llir.Var(name="pA0", type=llir.DataType.INT),
                        llir.Literal(1, llir.DataType.INT),
                    ),
                ),
            )
        ],
    )


def test_transformation_declines_without_a_derivable_work_estimate():
    """A bound the work estimate cannot be tied to leaves the list alone."""

    statements = [llir.BlankLine(), _plain_loop(bound="N")]
    assert build_parallel_chunk_assembly(statements, _context()) is None


def test_transformation_declines_without_exactly_one_top_level_loop():
    context = _context()
    assert build_parallel_chunk_assembly([llir.BlankLine()], context) is None
    assert (
        build_parallel_chunk_assembly([_plain_loop(), _plain_loop()], context) is None
    )


def _loop_with_derivable_work(bound="A0_size"):
    """A loop whose body carries a position read the work estimate can match."""

    index = llir.Var(name="i", type=llir.DataType.INT64)
    return llir.ForLoop(
        init=llir.VarInit(var=index, value=llir.Literal(0, llir.DataType.INT)),
        cond=llir.BinOp("<", index, llir.Var(name=bound, type=llir.DataType.INT64)),
        update=llir.Increment(var=index),
        body=[llir.RawStmt("int pA1_end = A1_pos[pA0 + 1]")],
    )


def test_transformation_keeps_the_original_statements_as_the_serial_arm():
    statements = [llir.BlankLine(), _loop_with_derivable_work()]
    gated = build_parallel_chunk_assembly(statements, _context())
    assert gated is not None
    lambda_def = gated[0]
    assert type(lambda_def) is llir.LambdaDef
    assert [param.name for param in lambda_def.params] == [
        "_assembly_lo",
        "_assembly_hi",
        "C1_crd",
        "C2_pos",
        "C2_crd",
        "C_values",
    ]
    branch = gated[-1]
    assert type(branch) is llir.IfThenElse
    assert branch.else_body == statements
    assert branch.else_body[-1] is statements[-1], "the serial arm is the same object"


def test_codegen_emits_the_lambda_and_walks_its_body():
    """The new statement is a first-class LLIR node, not a raw escape hatch."""

    from scorch.compiler.codegen import LLIRLowerer
    from scorch.compiler.llir_traversal import LLIRRewriter, LLIRTraversalContext

    node = llir.LambdaDef(
        var=llir.Var(name="_assembly_body", type=llir.DataType.AUTO),
        params=[
            llir.Var(name="_assembly_lo", type=llir.DataType.INT64),
            llir.Var(name="C_values", type=llir.DataType.AUTO_REF),
        ],
        body=[
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(name="C_values", type=llir.DataType.NO_TYPE),
                    index=llir.Var(name="_assembly_lo", type=llir.DataType.INT64),
                ),
                value=llir.Literal(1, llir.DataType.INT),
            )
        ],
    )
    emitted = LLIRLowerer().lower_llir(node, 0)
    assert emitted.splitlines()[0] == (
        "auto _assembly_body = [&](int64_t _assembly_lo, auto& C_values) {"
    )
    assert emitted.rstrip().endswith("};")

    # The shared rewriter must reach inside the body, or the downstream
    # dynamic-vector rewrite would silently skip everything the lambda holds.
    rewritten = LLIRRewriter(
        LLIRTraversalContext(stage="test", pass_name="test")
    ).rewrite(node)
    assert type(rewritten) is llir.LambdaDef
    assert rewritten is not node and rewritten.body[0] is not node.body[0]
    assert [p.name for p in rewritten.params] == ["_assembly_lo", "C_values"]


def test_the_dynamic_vector_rewrite_reaches_inside_the_shared_body():
    """``C_values[pC2] = v`` must still become ``emplace_back`` in the lambda."""

    source = emit_chunked(ttm_cin("dss", "dd", "dss"), TTM_RESULT, TTM_BINDINGS)
    body = source[source.index("auto _assembly_body") : source.index("\n  };\n")]
    assert "auto& C_values" in body
    assert "C_values.emplace_back(" in body
    assert "C2_crd.emplace_back(" in body
    assert "scorch_vector_set(C1_pos, C1_pos_index + 1, C1_crd.size());" in body


def test_transformation_rejects_a_foreign_context():
    with pytest.raises(TypeError):
        build_parallel_chunk_assembly([_plain_loop()], object())


# -- execution ---------------------------------------------------------------

# Shapes chosen so scorch_nthreads = min(rows/16, stored_ij/500) is genuinely
# above one: a gate that never opens would make every execution lock vacuous.
_PARALLEL_SHAPE = (64, 128, 64, 128)


@pytest.fixture(autouse=True)
def _torch_allows_the_teams_these_shapes_were_built_for(torch_threads):
    # _PARALLEL_SHAPE (and the three-level shape, 64 rows) gives min(64 // 16, stored / 500) = 4 workers.
    # torch's count must allow that, or a process started with MKL_NUM_THREADS=1 runs every kernel
    # here on one worker (tests/conftest.py, torch_threads).
    torch_threads(_PARALLEL_SHAPE[0] // ROWS_PER_THREAD, needs_team=False)


def _operands(a_fmt, b_fmt, shape, density, seed=20260812):
    i, j, k, l = shape  # noqa: E741 -- the contraction's own index names
    a_dense = random_dense((i, j, k), seed, density)
    b_dense = random_dense((k, l), seed + 7, min(1.0, density + 0.3))
    return (
        sparse(a_dense, a_fmt, "A"),
        sparse(b_dense, b_fmt, "B"),
    ), torch.einsum("ijk,kl->ijl", a_dense.to(torch.float64), b_dense.to(torch.float64))


def _storage(result):
    return (
        tuple(
            tuple(tuple(int(x) for x in part.tolist()) for part in mode)
            for mode in result.storage.index.mode_indices
        ),
        tuple(round(float(v), 5) for v in result.storage.value.tolist()),
    )


@pytest.mark.parametrize("b_fmt", ["ss", "dd"])
@pytest.mark.parametrize("density", [0.001, 0.05])
def test_parallel_assembly_matches_the_serial_assembly_exactly(b_fmt, density):
    """Bit-identical storage, not merely allclose values.

    Values agreeing proves the arithmetic; only the index arrays prove that the
    concatenation preserved lexicographic order across the chunk boundaries,
    which is the one thing this design could get wrong.  The automatic plan's
    kernel at this extent (two parallel passes) is held to the same storage.
    """

    operands, reference = _operands("dss", b_fmt, _PARALLEL_SHAPE, density)
    result_shape = (_PARALLEL_SHAPE[0], _PARALLEL_SHAPE[1], _PARALLEL_SHAPE[3])

    def run(options):
        return execute_cin_via_loopir(
            ttm_cin("dss", b_fmt, "dss"),
            result_shape,
            *operands,
            compile_options=options,
        )[0]

    parallel = run(chunked_options(jit=True))
    assert torch.allclose(
        parallel.to_torch(in_place=False).to(torch.float64),
        reference,
        atol=1e-3,
        rtol=1e-3,
    )
    serial = run(auto_options(jit=True, assembly="single_pass_serial"))
    assert _storage(parallel) == _storage(serial)
    assert _storage(run(auto_options(jit=True))) == _storage(serial)


def test_three_compressed_levels_assemble_in_parallel():
    """The strategy is rank-general, so its execution lock has to be too.

    Every other execution lock here uses a ``dss`` receiver, whose two
    compressed levels exercise one shared position array and one chunked one.
    A ``dsss`` receiver adds a second chunked position array, and its parent
    links are the part a per-chunk concatenation could plausibly get wrong at
    depth.  The ordered-key family's own suite covers rank-general receivers
    only at extents where the thread count is one, where this branch never
    runs.
    """

    shape = (64, 32, 8, 8, 8)
    dense = random_dense(shape, 7, density=0.5)
    operand = sparse(dense, "dssss", "A")
    stored = int(operand.storage.index.mode_indices[1][0][-1])
    assert min(shape[0] // 16, stored // 500) > 1, "the gate must actually open"

    result_shape = (64, 32, 8, 8)

    def run(options):
        return execute_cin_via_loopir(
            reduction_cin("dssss", "dsss", "ijklm", "ijkm"),
            result_shape,
            operand,
            compile_options=options,
        )[0]

    parallel = run(chunked_options(jit=True))
    assert torch.allclose(
        parallel.to_torch(in_place=False).to(torch.float64),
        dense.to(torch.float64).sum(dim=3),
        atol=1e-3,
        rtol=1e-3,
    )
    serial = run(auto_options(jit=True, assembly="single_pass_serial"))
    assert _storage(parallel) == _storage(serial)
    assert _storage(run(auto_options(jit=True))) == _storage(serial)


@pytest.mark.parametrize("assembly", [CHUNKED, None])
def test_an_empty_operand_assembles_the_same_empty_result(assembly):
    """The chunk whose rows are all empty must contribute nothing, not a gap.

    Asked of the chunked strategy and of the automatic plan's (``None``).
    """

    shape = _PARALLEL_SHAPE
    i, j, k, l = shape  # noqa: E741 -- the contraction's own index names
    a_dense = torch.zeros((i, j, k), dtype=_F32)
    b_dense = random_dense((k, l), 3, 0.5)
    operands = (sparse(a_dense, "dss", "A"), sparse(b_dense, "dd", "B"))
    options = auto_options(jit=True, assembly=assembly)
    result = execute_cin_via_loopir(
        ttm_cin("dss", "dd", "dss"), (i, j, l), *operands, compile_options=options
    )[0]
    assert int(result.storage.value.numel()) == 0
    positions = result.storage.index.mode_indices[1][0].tolist()
    assert positions == [0] * (i + 1)
