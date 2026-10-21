"""Phase-7 multiple-dense-prefix sparse outputs: the flattened-prefix family.

A compressed result level below TWO OR MORE dense parents owns one segment
per cell of the flattened dense prefix, so its position vector holds
``prod(dense extents) + 1`` entries -- 7 for a ``2x3`` prefix, not 4.  The
inherited lowering numbered those segments by a single dense loop variable,
which is why ``dds`` was rejected at CIN admission and ``ddss`` at target
lowering.

**The shipped compiler emits the same source for this family.**  It used to
bound its catch-up loop by the innermost dense loop variable alone
(``for (; C2_pos_index < j; ...)``), so for a ``2x3`` prefix it emitted a
4-entry position array where 7 are required, and this family was gated on
the LoopIR oracle and the dense PyTorch reference instead of byte parity.
The shipped lowering now numbers the segments by the flattened prefix too,
spelled the way this route spells it, so
``test_shipped_compiler_emits_the_same_source_for_this_family`` holds the two
compilers byte-identical here; the oracle and the PyTorch reference still
judge every executed cell.  Both automatic arms must agree with each other
exactly.

The neighbours' dispositions are recorded at the end of the file.  An
interleaved layout (a compressed level ABOVE a dense one, ``sds``) is
assembled for copies, products
(``test_interleaved_sds_copy_and_product_match_pytorch``) and, since
2026-10-12, sums
(``test_interleaved_sds_sum_reads_its_dense_level_below_the_union``).  A result with
permuted compressed structure is lowered in its storage order, with the
operand converted to that order
(``test_a_permuted_compressed_result_converts_the_operand``).
"""

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.nodes import LevelKind
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.stensor import STensor
from tests.test_scorch.loopir_executed_programs import (
    assert_program_matches_pytorch,
)
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

_KIND = {"d": LevelKind.DENSE, "s": LevelKind.COMPRESSED}
_OPS = {"mul": Operation.MUL, "add": Operation.ADD}


def build(result_fmt, operand_fmts, op, dtype=torch.float32, *, commuted=False):
    """``C = A <op> B ...`` over one shared index nest."""

    rank = len(result_fmt)
    ivars = tuple(IndexVar(name) for name in "ijklm"[:rank])
    order = list(operand_fmts)
    names = list("ABD"[: len(order)])
    if commuted:
        order = order[::-1]
        names = names[::-1]
    rhs = None
    for name, fmt in zip(names, order):
        access = TensorVar(name, fmt=fmt, dtype=dtype)[ivars]
        rhs = access if rhs is None else CINBinaryOp(_OPS[op], rhs, access)
    stmt = TensorAssign(TensorVar("C", fmt=result_fmt, dtype=dtype)[ivars], rhs)
    for index_var in reversed(ivars):
        stmt = ForAll(index_var, stmt)
    return stmt


def sparse(dense, name, fmt):
    return STensor.from_torch(dense.clone(), name).to_sparse(fmt)


def dense_prefix_cells(fmt, shape):
    """The flattened segment count the first compressed level must own."""

    cells = 1
    for level, char in enumerate(fmt):
        if char != "d":
            break
        cells *= shape[level]
    return cells


def assert_storage_is_honest(result, fmt, shape):
    """Identity-ordered storage whose position lengths follow the flattening."""

    mode_indices = result.storage.index.mode_indices
    assert len(mode_indices) == len(fmt)
    parents = 1
    for level, char in enumerate(fmt):
        if char == "d":
            assert list(mode_indices[level]) == []
            parents *= shape[level]
            continue
        pos = mode_indices[level][0].tolist()
        crd = mode_indices[level][1].tolist()
        assert len(pos) == parents + 1, (
            f"level {level} position vector holds {len(pos)} entries; the "
            f"flattened parent count requires {parents + 1}"
        )
        assert pos[0] == 0
        assert pos == sorted(pos)
        assert pos[-1] == len(crd)
        for segment in range(parents):
            entries = crd[pos[segment] : pos[segment + 1]]
            assert entries == sorted(set(entries))
            assert all(0 <= entry < shape[level] for entry in entries)
        parents = len(crd)
    assert len(result.storage.value.tolist()) == parents


def oracle_result(kernel, denses, operand_fmts, shape):
    """Base and scheduled oracle agreement; returns the level storage."""

    lowering = kernel.lowering
    inputs = {}
    for symbol, fmt, dense in zip(lowering.rhs_access_symbols, operand_fmts, denses):
        if set(fmt) == {"d"}:
            inputs[symbol] = dense.tolist()
        else:
            inputs[symbol] = LevelTensorStorage.from_dense(
                dense.tolist(),
                tuple(dense.shape),
                tuple(range(len(fmt))),
                tuple(_KIND[char] for char in fmt),
            )
    base = run_program(lowering.program, inputs, {lowering.result_symbol: shape})
    if kernel.schedule is not None:
        scheduled = run_program(
            kernel.schedule.program, inputs, {lowering.result_symbol: shape}
        )
        assert scheduled[lowering.result_symbol] == base[lowering.result_symbol]
    return base[lowering.result_symbol]


def fixture(kind, shape, dtype, seed):
    torch.manual_seed(seed)
    if kind == "dense":
        return torch.randn(shape).to(dtype)
    if kind == "random":
        return ((torch.rand(shape) < 0.4) * torch.randn(shape)).to(dtype)
    if kind == "ragged":
        stored = ((torch.rand(shape) < 0.4) * torch.randn(shape)).to(dtype)
        stored[0] = 0
        return stored
    if kind == "empty":
        return torch.zeros(shape, dtype=dtype)
    raise AssertionError(kind)


# -- the family's defining structural fact -----------------------------------


@pytest.mark.parametrize(
    "fmt,shape,expected_pos",
    [
        ("dds", (3, 4, 5), 13),
        ("ddss", (2, 3, 4, 5), 7),
        ("ddds", (2, 3, 2, 5), 13),
        ("dddss", (2, 3, 2, 4, 3), 13),
        ("dds", (1, 1, 4), 2),
        ("dds", (3, 4, 0), 13),
        ("ddss", (2, 0, 4, 5), 1),
    ],
)
def test_first_compressed_level_is_sized_by_the_flattened_prefix(
    fmt, shape, expected_pos
):
    """The position vector holds one slot per flattened dense cell, plus one."""

    assert dense_prefix_cells(fmt, shape) + 1 == expected_pos
    denses = [
        fixture("random", shape, torch.float32, 11),
        fixture("random", shape, torch.float32, 12),
    ]
    out = execute_cin_via_loopir(
        build(fmt, (fmt, fmt), "mul"),
        shape,
        sparse(denses[0], "A", fmt),
        sparse(denses[1], "B", fmt),
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    first_compressed = len(fmt) - len(fmt.lstrip("d"))
    pos = result.storage.index.mode_indices[first_compressed][0].tolist()
    assert len(pos) == expected_pos
    assert_storage_is_honest(result, fmt, shape)


@pytest.mark.parametrize("arm", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("op", ["mul", "add"])
@pytest.mark.parametrize(
    "fmt,shape",
    [
        ("dds", (3, 4, 5)),
        ("ddss", (2, 3, 4, 5)),
        ("ddds", (2, 3, 2, 5)),
        ("dddss", (2, 3, 2, 4, 3)),
    ],
)
def test_shipped_compiler_emits_the_same_source_for_this_family(fmt, shape, op, arm):
    """Both compilers number the segments by the flattened prefix, byte for byte.

    The shipped lowering used to close the compressed level against the
    innermost dense loop variable alone -- four position entries for a
    ``2x3`` prefix where seven are required.  It now emits the flattened
    bound this route emits, so the family is held by byte identity and a
    change to either compiler's spelling here fails this test.
    """

    bindings = ((shape, torch.float32), (shape, torch.float32))
    shipped = legacy_generated_cpp(
        build(fmt, (fmt, fmt), op),
        shape,
        bindings,
        compile_options=auto_options(arm),
    )
    kernel = compile_cin_via_loopir(
        build(fmt, (fmt, fmt), op),
        shape,
        bindings,
        compile_options=auto_options(arm),
    )
    first_compressed = len(fmt) - len(fmt.lstrip("d"))
    catch_up = [
        line.strip()
        for line in shipped.splitlines()
        if f"C{first_compressed}_pos_index <" in line
    ]
    assert catch_up, "the shipped compiler emitted no compressed-level catch-up"
    # Never the innermost dense loop variable alone.
    assert all("*" in line for line in catch_up)
    assert shipped == kernel.cpp_source


def test_loopir_route_flattens_the_prefix_in_the_generated_source():
    shape = (2, 3, 4, 5)
    kernel = compile_cin_via_loopir(
        build("ddss", ("ddss", "ddss"), "mul"),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    catch_up = [
        line.strip()
        for line in kernel.cpp_source.splitlines()
        if "C2_pos_index <" in line
    ]
    assert catch_up
    # The flattened bound multiplies the outer coordinate by the inner
    # loop's own bound spelling.
    assert all("j" in line and "i" in line for line in catch_up)
    assert any("*" in line for line in catch_up)


def test_one_dense_parent_keeps_the_inherited_spelling():
    """The prefix-of-one degenerate case must not move at all."""

    shape = (3, 4, 5)
    kernel = compile_cin_via_loopir(
        build("dss", ("dss", "dss"), "mul"),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    legacy = legacy_generated_cpp(
        build("dss", ("dss", "dss"), "mul"),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    assert kernel.cpp_source == legacy


# -- compiled execution ------------------------------------------------------


_EXEC_CASES = [
    ("dds", ("dds",), (3, 4, 5), "mul"),
    ("dds", ("dds", "dds"), (3, 4, 5), "mul"),
    ("dds", ("dds", "dds"), (3, 4, 5), "add"),
    ("dds", ("ddd", "dds"), (3, 4, 5), "mul"),
    ("dds", ("dds", "ddd"), (3, 4, 5), "mul"),
    ("ddss", ("ddss", "ddss"), (2, 3, 4, 5), "mul"),
    ("ddss", ("ddss", "ddss"), (2, 3, 4, 5), "add"),
    ("ddss", ("ddss", "dddd"), (2, 3, 4, 5), "mul"),
    ("ddds", ("ddds", "ddds"), (2, 3, 2, 5), "mul"),
    ("dddss", ("dddss", "dddss"), (2, 2, 2, 4, 3), "add"),
]


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "result_fmt,operand_fmts,shape,op",
    _EXEC_CASES,
    ids=[f"{c[0]}<-{'+'.join(c[1])}:{c[3]}" for c in _EXEC_CASES],
)
def test_compiled_execution_matches_pytorch_and_the_oracle(
    result_fmt, operand_fmts, shape, op, dtype, arm
):
    denses = [
        fixture("dense" if set(f) == {"d"} else "random", shape, dtype, 900 + 7 * n)
        for n, f in enumerate(operand_fmts)
    ]
    reference = denses[0]
    for extra in denses[1:]:
        reference = reference * extra if op == "mul" else reference + extra
    cin = build(result_fmt, operand_fmts, op, dtype)
    stensors = [
        sparse(dense, name, fmt)
        for name, fmt, dense in zip("ABD", operand_fmts, denses)
    ]
    kernel = compile_cin_via_loopir(
        cin,
        shape,
        tuple((shape, dtype) for _ in operand_fmts),
        compile_options=auto_options(arm),
    )
    out = execute_cin_via_loopir(
        cin, shape, *stensors, compile_options=auto_options(arm, jit=True)
    )
    result = out[0] if isinstance(out, tuple) else out
    assert_storage_is_honest(result, result_fmt, shape)
    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)
    oracle_result(kernel, denses, operand_fmts, shape)


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("kind", ["ragged", "empty"])
@pytest.mark.parametrize(
    "result_fmt,shape", [("dds", (3, 4, 5)), ("ddss", (2, 3, 4, 5))]
)
def test_ragged_and_empty_streams_assemble(result_fmt, shape, kind, arm):
    denses = [fixture(kind, shape, torch.float32, 55 + n) for n in range(2)]
    cin = build(result_fmt, (result_fmt, result_fmt), "add")
    out = execute_cin_via_loopir(
        cin,
        shape,
        sparse(denses[0], "A", result_fmt),
        sparse(denses[1], "B", result_fmt),
        compile_options=auto_options(arm, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    assert_storage_is_honest(result, result_fmt, shape)
    assert torch.allclose(result.to_torch(), denses[0] + denses[1], atol=1e-3)


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("shape", [(3, 4, 0), (1, 1, 4)])
def test_zero_extents_and_singleton_prefixes(shape, arm):
    denses = [fixture("random", shape, torch.float32, 71 + n) for n in range(2)]
    cin = build("dds", ("dds", "dds"), "mul")
    out = execute_cin_via_loopir(
        cin,
        shape,
        sparse(denses[0], "A", "dds"),
        sparse(denses[1], "B", "dds"),
        compile_options=auto_options(arm, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    assert_storage_is_honest(result, "dds", shape)
    assert torch.allclose(result.to_torch(), denses[0] * denses[1], atol=1e-3)


@pytest.mark.parametrize("arm", [False, True])
def test_cancellation_keeps_explicit_stored_zeros(arm):
    """``A + (-A)`` unites both patterns; every entry is a stored exact zero."""

    shape = (3, 4, 5)
    stored = fixture("random", shape, torch.float32, 31337)
    out = execute_cin_via_loopir(
        build("dds", ("dds", "dds"), "add"),
        shape,
        sparse(stored, "A", "dds"),
        sparse(-stored, "B", "dds"),
        compile_options=auto_options(arm, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    assert_storage_is_honest(result, "dds", shape)
    values = result.storage.value.tolist()
    assert values, "the union must keep the cancelled entries as stored zeros"
    assert all(value == 0.0 for value in values)
    assert torch.allclose(result.to_torch(), torch.zeros(shape), atol=1e-6)


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    "result_fmt,shape", [("dds", (3, 4, 5)), ("ddss", (2, 3, 4, 5))]
)
def test_commuted_operands_agree(result_fmt, shape, arm):
    denses = [fixture("random", shape, torch.float32, 8080 + n) for n in range(2)]
    straight = execute_cin_via_loopir(
        build(result_fmt, (result_fmt, result_fmt), "mul"),
        shape,
        sparse(denses[0], "A", result_fmt),
        sparse(denses[1], "B", result_fmt),
        compile_options=auto_options(arm, jit=True),
    )
    commuted = execute_cin_via_loopir(
        build(result_fmt, (result_fmt, result_fmt), "mul", commuted=True),
        shape,
        sparse(denses[0], "A", result_fmt),
        sparse(denses[1], "B", result_fmt),
        compile_options=auto_options(arm, jit=True),
    )
    left = straight[0] if isinstance(straight, tuple) else straight
    right = commuted[0] if isinstance(commuted, tuple) else commuted
    assert_storage_is_honest(left, result_fmt, shape)
    assert_storage_is_honest(right, result_fmt, shape)
    assert torch.allclose(left.to_torch(), right.to_torch(), atol=1e-3)


# -- route identity ----------------------------------------------------------


@pytest.mark.parametrize(
    "result_fmt,shape", [("dds", (3, 4, 5)), ("ddss", (2, 3, 4, 5))]
)
def test_both_automatic_arms_emit_identical_source(result_fmt, shape):
    sources = set()
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build(result_fmt, (result_fmt, result_fmt), "mul"),
            shape,
            ((shape, torch.float32), (shape, torch.float32)),
            compile_options=auto_options(arm),
        )
        assert "#pragma omp" not in kernel.cpp_source
        sources.add(kernel.cpp_source)
    assert len(sources) == 1


@pytest.mark.parametrize(
    "result_fmt,shape", [("dds", (3, 4, 5)), ("ddss", (2, 3, 4, 5))]
)
def test_canonical_dump_is_arm_stable_and_erases_to_base(result_fmt, shape):
    dumps = []
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build(result_fmt, (result_fmt, result_fmt), "mul"),
            shape,
            ((shape, torch.float32), (shape, torch.float32)),
            compile_options=auto_options(arm),
        )
        dumps.append(kernel.program_dump)
        program = (
            kernel.schedule.program
            if kernel.schedule is not None
            else kernel.lowering.program
        )
        assert canonical_program_dump(erase_schedule(program)) == (
            canonical_program_dump(kernel.lowering.program)
        )
    assert dumps[0] == dumps[1]


# -- recorded dispositions for the neighbours ---------------------------------


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("op", ["copy", "mul"])
def test_interleaved_sds_copy_and_product_match_pytorch(op, arm):
    """A compressed level above a dense one is assembled since 2026-10-11.

    The dense level's parent count is the stored-coordinate count of the
    compressed level above it, so the dense loop sits below a stream loop.  The
    ordered assembly writes the dense block first and commits the compressed
    coordinate above it only if something was written, so nothing has to be
    rolled back (``test_loopir_dense_below_compressed_target.py``).
    """

    from tests.test_scorch.test_loopir_dense_below_compressed_target import (
        assert_statement_matches_pytorch,
    )

    operands = ("sds",) if op == "copy" else ("sds", "sds")
    assert_statement_matches_pytorch(build("sds", operands, "mul"), arm)


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    "operand_fmts,result_fmt",
    [
        (("sds", "sds"), "sds"),
        (("sss", "sds"), "sss"),
        (("sds", "sds", "sds"), "sds"),
    ],
)
def test_interleaved_sds_sum_reads_its_dense_level_below_the_union(
    operand_fmts, result_fmt, arm
):
    """``sds + sds -> sds`` is computed since 2026-10-12; it was refused.

    Level 1 is a dense level below the union over ``i``.  Its position is
    arithmetic on the union-bound position of ``i``, which is absent where
    that operand stores no ``i``.  The compressed level below hangs from it
    and walks the empty segment there, as it does directly under a cursor
    that did not align.  The inner union names those dense positions as its
    cursors' parents.  Operand ``n`` stores no row ``n``, so every one-sided
    case runs; the oracle runs the same program, and the compiled result's
    arrays must equal the oracle's.  ``test_loopir_dense_union.py``'s
    sparse-result table judges more layouts on both compilers.
    """

    shape = (4, 3, 5)
    denses = []
    for n in range(len(operand_fmts)):
        dense = fixture("random", shape, torch.float32, 70 + n)
        dense[n] = 0
        denses.append(dense)
    reference = sum(denses[1:], denses[0])
    cin = build(result_fmt, operand_fmts, "add")
    kernel = compile_cin_via_loopir(
        cin,
        shape,
        tuple((shape, torch.float32) for _ in operand_fmts),
        compile_options=auto_options(arm),
    )
    inner = [
        line
        for line in kernel.program_text.splitlines()
        if "merged_union_for" in line and "level 2" in line
    ]
    assert len(inner) == 1 and inner[0].count("parent dense_pos(") == sum(
        fmt == "sds" for fmt in operand_fmts
    ), kernel.program_text
    out = execute_cin_via_loopir(
        cin,
        shape,
        *(
            sparse(dense, name, fmt)
            for name, fmt, dense in zip("ABD", operand_fmts, denses)
        ),
        compile_options=auto_options(arm, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    assert_storage_is_honest(result, result_fmt, shape)
    # Read the arrays before ``to_torch``, which densifies the storage.
    oracle = oracle_result(kernel, denses, operand_fmts, shape)
    mode_indices = result.storage.index.mode_indices
    for level, char in enumerate(result_fmt):
        if char == "s":
            assert list(oracle.positions[level]) == mode_indices[level][0].tolist()
            assert list(oracle.coordinates[level]) == mode_indices[level][1].tolist()
    assert torch.allclose(
        torch.tensor(oracle.values, dtype=torch.float32),
        result.storage.value,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(result.to_torch(), reference, atol=1e-6, rtol=1e-6)


def build_copy_into_a_permuted_result():
    """``C[i,j,k] = A[i,j,k]``, both ``dds``, with ``C`` stored in order (i, k, j)."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    return ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    TensorVar("C", fmt="dds", mode_order=[0, 2, 1])[i, j, k],
                    TensorVar("A", fmt="dds")[i, j, k],
                ),
            ),
        ),
    )


@pytest.mark.parametrize("arm", [False, True])
def test_a_permuted_compressed_result_converts_the_operand(arm):
    """Permuted compressed structure is no longer rejected before LoopIR.

    The result is lowered in the identity-order form of its storage order
    ``(i, k, j)`` (``test_loopir_storage_order_form.py``), which the operand,
    stored ``(i, j, k)``, is not in; the plan converts the operand to that
    order, keeping its ``dds`` layout, and the result is wrapped in its
    declared mode order.  A shape is bound in storage order, so the result's
    is ``(3, 5, 4)``; its logical-order shape is refused by the CIN verifier.

    The shipped compiler's plan origin still rejects the program
    (``sparse_parent_dominance``), so there is no shipped text to compare with.
    """

    shape = (3, 4, 5)
    options = auto_options(arm)
    kernel = compile_cin_via_loopir(
        build_copy_into_a_permuted_result(),
        (3, 5, 4),
        ((shape, torch.float32),),
        compile_options=options,
    )
    assert [(c.mode_order, c.format) for c in kernel.conversions] == [
        ((0, 2, 1), "dds")
    ]
    assert tuple(kernel.result_mode_order) == (0, 2, 1)

    with pytest.raises(VerificationError) as mismatch:
        compile_cin_via_loopir(
            build_copy_into_a_permuted_result(),
            shape,
            ((shape, torch.float32),),
            compile_options=options,
        )
    assert "index_extent_mismatch" in str(mismatch.value)

    with pytest.raises(InvalidSchedule) as refusal:
        legacy_generated_cpp(
            build_copy_into_a_permuted_result(),
            (3, 5, 4),
            ((shape, torch.float32),),
            compile_options=options,
        )
    assert [d.code for d in refusal.value.diagnostics] == ["sparse_parent_dominance"]


@pytest.mark.parametrize("arm", [0, 1])
def test_a_copy_into_a_permuted_compressed_result_matches_pytorch(arm):
    """The same copy executed: float64 values to 1e-12, and the result's level
    arrays, in its storage order, hold exactly the operand's stored entries."""

    kernel, _ = assert_program_matches_pytorch(
        (("A", "dds", "ijk", None),),
        ("ijk", "dds", (0, 2, 1)),
        arm,
    )
    assert tuple(kernel.result_mode_order) == (0, 2, 1)


# Sums into ``sd`` and ``sdd`` kept ``unsupported_sparse_output_domain`` here
# until the united assembly learned to drain a one-sided case into the dense
# block below a compressed coordinate (2026-10-12).  The sparse-result table of
# ``test_loopir_dense_union.py`` executes them (``sd + sd -> sd``,
# ``sdd + sdd -> sdd`` and others) on both compilers against float64.


@pytest.mark.parametrize("arm", [False, True])
def test_a_dense_domain_suffix_coordinate_assembles_every_cell(arm):
    """A dense-domain suffix coordinate assembles this result since 2026-10-11.

    Every operand is dense at the suffix level, and dense storage holds every
    cell, so the leaf appends each one -- structure, not value -- and no
    conditional is emitted for it.  The flattened prefix catch-up is unchanged.
    """

    shape = (3, 4, 5)
    kernel = compile_cin_via_loopir(
        build("dds", ("ddd", "ddd"), "mul"),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(arm),
    )
    assert "for (int64_t k = 0;" in kernel.cpp_source
    assert "C2_crd.emplace_back(k);" in kernel.cpp_source


@pytest.mark.parametrize("arm", [False, True])
def test_intersected_dense_prefix_coordinate_assembles(arm):
    """A `dds` result whose dense-prefix coordinate is an intersection.

    ``C = A * B`` with ``dsd`` operands: the second coordinate, a dense
    result level, iterates the intersection of two stored streams.  Refused as
    ``unsupported_sparse_output_domain`` until 2026-10-12; an intersection
    visits each coordinate both operands store once, in order, so it binds
    the dense level exactly as one stored stream does, and the rows it skips
    are empty segments of the compressed level below.
    """

    from tests.test_scorch.test_loopir_dense_below_compressed_target import (
        assert_statement_matches_pytorch,
    )

    assert_statement_matches_pytorch(build("dds", ("dsd", "dsd"), "mul"), arm)


# ``dsd + dsd -> dds``, whose dense-prefix coordinate is a union, kept
# ``unsupported_sparse_output_domain`` here until a union could drive a dense
# prefix level (2026-10-12); the sparse-result table of
# ``test_loopir_dense_union.py`` executes it on both compilers.


@pytest.mark.parametrize(
    "result_fmt,shape",
    [("dds", (3, 4, 5)), ("ddss", (2, 3, 4, 5)), ("dddss", (2, 2, 2, 4, 3))],
)
def test_routing_claims_the_whole_family_for_the_assembly_target(result_fmt, shape):
    """Routing is structural, so the owning target -- not the generic one --
    lowers every member of the family."""

    from scorch.compiler.loopir.lower_llir import (
        _multi_compressed_assembly_chain,
    )

    kernel = compile_cin_via_loopir(
        build(result_fmt, (result_fmt, result_fmt), "mul"),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    assert _multi_compressed_assembly_chain(kernel.lowering.program)
    # The owning target is the only one that emits the conditional parent
    # append plus child position close per structural level.
    assert "Assembly compressed _level indices" in kernel.cpp_source


def test_canonical_csr_keeps_its_dedicated_route():
    """``ds`` -- one dense parent, one compressed level -- must not move."""

    from scorch.compiler.loopir.lower_llir import (
        _multi_compressed_assembly_chain,
    )

    shape = (3, 4)
    kernel = compile_cin_via_loopir(
        build("ds", ("ds", "ds"), "mul"),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    assert not _multi_compressed_assembly_chain(kernel.lowering.program)
    legacy = legacy_generated_cpp(
        build("ds", ("ds", "ds"), "mul"),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    assert kernel.cpp_source == legacy
