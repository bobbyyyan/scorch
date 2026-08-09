"""Phase-7 multiple-dense-prefix sparse outputs: the flattened-prefix family.

A compressed result level below TWO OR MORE dense parents owns one segment
per cell of the flattened dense prefix, so its position vector holds
``prod(dense extents) + 1`` entries -- 7 for a ``2x3`` prefix, not 4.  The
inherited lowering numbered those segments by a single dense loop variable,
which is why ``dds`` was rejected at CIN admission and ``ddss`` at target
lowering.

**The legacy comparand is malformed for this family and is not a gate.**
Its catch-up loop is bounded by the innermost dense loop variable alone
(``for (; C2_pos_index < j; ...)``), so for a ``2x3`` prefix it emits a
4-entry position array where 7 are required.  ``test_legacy_comparand_is_
malformed_for_this_family`` pins that fact directly from the generated
legacy source, which is what justifies gating on the LoopIR oracle and the
dense PyTorch reference instead of byte parity.  Both automatic arms must
still agree with each other exactly.

Interleaved layouts (a compressed level ABOVE a dense one, ``sds``) and
permuted compressed structure stay fail-closed; see
``test_interleaved_and_permuted_layouts_stay_fail_closed`` and
``test_trailing_dense_output_families_keep_their_codes`` for the recorded
dispositions.
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
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
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


def test_legacy_comparand_is_malformed_for_this_family():
    """Legacy bounds its catch-up by ONE dense loop variable, so it is wrong.

    This is the evidence that byte parity cannot be this family's gate: the
    legacy source for a ``2x3`` dense prefix closes the compressed level
    against ``j`` alone, producing four position entries where seven are
    required.
    """

    shape = (2, 3, 4, 5)
    cpp = legacy_generated_cpp(
        build("ddss", ("ddss", "ddss"), "mul"),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    catch_up = [line.strip() for line in cpp.splitlines() if "C2_pos_index <" in line]
    assert catch_up, "legacy emitted no compressed-level catch-up"
    # Bounded by the innermost dense loop variable alone -- never the
    # flattened ``i * 3 + j`` this family requires.
    assert all(
        line.endswith("C2_pos_index < j; C2_pos_index++) {") for line in catch_up
    )
    assert not any("i * " in line or "i*" in line for line in catch_up)


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


# -- recorded dispositions for the neighbours that stay out ------------------


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("op", ["copy", "mul", "add"])
def test_interleaved_sds_stays_fail_closed(op, arm):
    """A compressed level ABOVE a dense one is a different mechanism.

    ``sds`` cannot join this family: its dense level's parent count is the
    dynamic stored-coordinate count of the compressed level above it, so a
    dense loop would have to sit BELOW a stream loop, and a compressed
    ancestor that turns out not to materialize would need its speculative
    per-dense-cell position closes rolled back.  Nothing in the ordered
    assembly owns that.  Its legacy comparand (census cell D9) is malformed
    AND numerically wrong, so there is no honest comparand either.
    """

    shape = (3, 4, 5)
    if op == "copy":
        cin = build("sds", ("sds",), "mul")
        bindings = ((shape, torch.float32),)
    else:
        cin = build("sds", ("sds", "sds"), op)
        bindings = ((shape, torch.float32), (shape, torch.float32))
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(cin, shape, bindings, compile_options=auto_options(arm))
    assert error.value.defect.code == "unsupported_sparse_output"


@pytest.mark.parametrize("arm", [False, True])
def test_permuted_compressed_structure_stays_fail_closed(arm):
    """Permuted compressed structure is rejected before LoopIR, unchanged."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    shape = (3, 4, 5)
    permuted = ForAll(
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
    with pytest.raises(InvalidSchedule):
        compile_cin_via_loopir(
            permuted,
            shape,
            ((shape, torch.float32),),
            compile_options=auto_options(arm),
        )


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("fmt,shape", [("sd", (3, 4)), ("sdd", (3, 4, 5))])
def test_trailing_dense_output_families_keep_their_codes(fmt, shape, arm):
    """D10 (``sd+sd``) and D11 (``sdd`` copy) are dispositioned, not migrated.

    A trailing dense leaf below a STORED-SPARSE parent is the mirror image
    of this family: the parent coordinates come from a cursor, not a dense
    loop, so the flattened-prefix model does not reach them.  They keep
    ``unsupported_sparse_output_domain`` exactly as inherited.
    """

    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            build(fmt, (fmt, fmt), "add"),
            shape,
            ((shape, torch.float32), (shape, torch.float32)),
            compile_options=auto_options(arm),
        )
    assert error.value.defect.code == "unsupported_sparse_output_domain"


@pytest.mark.parametrize("arm", [False, True])
def test_dense_domain_suffix_stays_fail_closed(arm):
    """A dense-domain suffix coordinate still cannot assemble this result."""

    shape = (3, 4, 5)
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            build("dds", ("ddd", "ddd"), "mul"),
            shape,
            ((shape, torch.float32), (shape, torch.float32)),
            compile_options=auto_options(arm),
        )
    assert error.value.defect.code == "unsupported_sparse_output"


@pytest.mark.parametrize("arm", [False, True])
def test_sparse_dense_prefix_coordinate_reports_the_domain(arm):
    """A `dds` result whose dense-prefix coordinate is stored-sparse.

    Recorded seam move: this cell used to report ``unsupported_sparse_output``
    ("layout not recognized").  The layout IS recognized now, so the precise
    reason is reported instead -- the prefix coordinate does not iterate a
    dense domain.
    """

    shape = (3, 4, 5)
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            build("dds", ("dsd", "dsd"), "mul"),
            shape,
            ((shape, torch.float32), (shape, torch.float32)),
            compile_options=auto_options(arm),
        )
    assert error.value.defect.code == "unsupported_sparse_output_domain"


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
