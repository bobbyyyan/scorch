"""An assignment whose update operator the shipped lowering cannot apply is refused.

The shipped lowering (``compiler/cin_lowerer.py``) has a rule for ``=`` and ``+=`` only.
It reads ``TensorAssign.op`` as ``op == Operation.ADD``, and its iteration lattice
rebuilds each assignment it simplifies with ``+=`` where the statement reduces and ``=``
where it does not.  So ``-=``, ``*=`` and ``/=`` were lowered as one of those two, with
no message: ``C[i,j] *= A[i,j] * B[i,j]`` emitted the kernel of ``C = A * B``, and
``C[i] -= A[i,j]`` summed with the wrong sign.  No library entry builds those operators,
so such a program reaches the lowering only as hand-written CIN.  It is now refused at
the lowering's entry, before anything is built, with ``unsupported_update_op``, the code
the new compiler already gives it; ``=`` and ``+=`` lower as before.
"""

from dataclasses import replace

import pytest
import torch

import scorch.ops as ops
from scorch.compiler.cin import (
    BinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.loopir.lower_cin import (
    LoopIRLoweringError,
    lower_normalized_cin_to_loopir,
)
from scorch.compiler.loopir.pipeline import legacy_generated_cpp
from scorch.compiler.scheduler import Schedule
from tests.test_scorch.level_array_support import level_arrays, logical_dense, stored

F64 = torch.float64
SMALL = {"i": 3, "j": 4, "k": 2}
LARGER = {"i": 37, "j": 41, "k": 5}
NON_ADDITIVE = [operation for operation in Operation if operation is not Operation.ADD]
ADDITIVE = [None, Operation.ADD]

# name: (operand index terms, operand formats, result indices, result format, combiner)
PROGRAMS = {
    "product ss": (("ij", "ij"), ("ss", "ss"), "ij", "ss", Operation.MUL),
    "product ds": (("ij", "ij"), ("ds", "ds"), "ij", "ds", Operation.MUL),
    "product dd": (("ij", "ij"), ("dd", "dd"), "ij", "dd", Operation.MUL),
    "sum ss": (("ij", "ij"), ("ss", "ss"), "ij", "ss", Operation.ADD),
    "row sum ds": (("ij",), ("ds",), "i", "d", None),
    "column sum ds": (("ij",), ("ds",), "j", "d", None),
    "row sum oo": (("ij",), ("oo",), "i", "d", None),
    "rank 3 dss": (("ijk",), ("dss",), "ik", "dd", None),
    "matvec": (("ij", "j"), ("ds", "d"), "i", "d", Operation.MUL),
    "matmul": (("ij", "jk"), ("ds", "dd"), "ik", "dd", Operation.MUL),
}
LOOP_PLAN_OPERATOR_CODES = {
    "auto_reduction_operator",
    "stack_reduction_operator",
    "workspace_reduction_operator",
}
ROUTES = {
    "as given": CompileOptions.from_environment(environ={}),
    "scheduled": replace(
        CompileOptions.from_environment(environ={}), requested_schedule=Schedule()
    ),
}


def access(name, fmt, indices, ivars):
    key = tuple(ivars[index] for index in indices)
    return TensorVar(name, fmt=fmt, dtype=F64)[key[0] if len(key) == 1 else key]


def program(name, update):
    """``C[result] <update>= A[...] <combiner> B[...]`` under one loop per index."""

    terms, formats, result_indices, result_format, combiner = PROGRAMS[name]
    letters = sorted(set("".join(terms)))
    ivars = {letter: IndexVar(letter) for letter in letters}
    factors = [
        access(tensor, fmt, term, ivars)
        for tensor, fmt, term in zip("AB", formats, terms)
    ]
    rhs = factors[0]
    for factor in factors[1:]:
        rhs = BinaryOp(combiner, rhs, factor)
    stmt = TensorAssign(
        access("C", result_format, result_indices, ivars), rhs, op=update
    )
    for letter in reversed(letters):
        stmt = ForAll(ivars[letter], stmt)
    return stmt


def bindings(name, extents):
    terms = PROGRAMS[name][0]
    return tuple((tuple(extents[index] for index in term), F64) for term in terms)


def result_shape(name, extents):
    return tuple(extents[index] for index in PROGRAMS[name][2])


def workspace_program(producer_update, consumer_update):
    """``C[i,k] = w[k]`` below ``w[k] += A[i,j] * B[j,k]``, a dense row workspace."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="ds", dtype=F64)
    b = TensorVar("B", fmt="dd", dtype=F64)
    c = TensorVar("C", fmt="dd", dtype=F64)
    w = Workspace("w", dim=1, dtype=F64, dense=True)
    producer = ForAll(
        j, ForAll(k, TensorAssign(w[k], a[i, j] * b[j, k], op=producer_update))
    )
    consumer = ForAll(k, TensorAssign(c[i, k], w[k], op=consumer_update))
    return ForAll(i, Where(producer=producer, consumer=consumer))


def lower(stmt):
    lowerer = CINLowerer(compile_options=CompileOptions.from_environment(environ={}))
    return lowerer.lower_IndexStmt(stmt)


def operand_values(name, extents, seed):
    """Float64 operands whose second row (or first index 1) stores nothing."""

    generator = torch.Generator().manual_seed(seed)
    values = []
    for term in PROGRAMS[name][0]:
        dense = torch.rand(
            tuple(extents[index] for index in term), generator=generator, dtype=F64
        )
        dense = torch.where(dense < 0.4, torch.zeros_like(dense), dense + 0.5)
        if dense.dim() > 1:
            dense[1] = 0
        values.append(dense)
    return values


def expected(name, values):
    terms, _, result_indices, _, combiner = PROGRAMS[name]
    if combiner is Operation.ADD:
        return values[0] + values[1]
    return torch.einsum(f"{','.join(terms)}->{result_indices}", *values)


def assert_refused(error, update, tensor="C"):
    (diagnostic,) = error.diagnostics
    assert diagnostic.code == "unsupported_update_op"
    assert diagnostic.stage == "legacy_cin_lowering"
    assert diagnostic.pass_name == "prepare_stmt"
    assert diagnostic.path[0] == "root" and diagnostic.path[-1] == "op"
    assert f"not '{update.value}='" in diagnostic.message
    assert f"the assignment to {tensor!r}" in diagnostic.message
    assert "unsupported_update_op" in str(error)
    return diagnostic


def test_every_operator_but_addition_is_refused():
    # The rule is "not None and not ADD", so a member added to Operation later is
    # refused until the lowering gains a rule for it; today that is these three.
    assert set(NON_ADDITIVE) == {Operation.SUB, Operation.MUL, Operation.DIV}


@pytest.mark.parametrize("update", NON_ADDITIVE, ids=str)
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_the_lowering_refuses_an_update_it_cannot_apply(name, update):
    with pytest.raises(UnsupportedFeature) as refusal:
        lower(program(name, update))

    diagnostic = assert_refused(refusal.value, update)
    depth = len(set("".join(PROGRAMS[name][0])))
    assert diagnostic.path == ("root",) + ("stmt",) * depth + ("op",)


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("update", NON_ADDITIVE, ids=str)
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_the_comparison_entry_refuses_it_on_both_routes(name, update, route):
    # legacy_generated_cpp is the shipped lowering the new compiler is compared with;
    # "scheduled" runs Scheduler.apply_schedule first, as production's scheduled
    # route does.  A plan that tiles a reduction or gives it a workspace is refused
    # there already, by the loop plan's legality check, with a code of its own
    # (loop_plan_legality.py); every other plan reached the lowering, which lowered
    # the operator as + or =.
    with pytest.raises(UnsupportedFeature) as refusal:
        legacy_generated_cpp(
            program(name, update),
            result_shape(name, SMALL),
            bindings(name, SMALL),
            compile_options=ROUTES[route],
        )

    codes = [diagnostic.code for diagnostic in refusal.value.diagnostics]
    if route == "scheduled" and codes != ["unsupported_update_op"]:
        assert len(codes) == 1 and codes[0] in LOOP_PLAN_OPERATOR_CODES, codes
    else:
        assert_refused(refusal.value, update)


@pytest.mark.parametrize("update", NON_ADDITIVE, ids=str)
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_the_developer_entry_refuses_it_before_it_builds_anything(
    name, update, monkeypatch
):
    def build(*args, **kwargs):
        raise AssertionError("a kernel build was started for a refused program")

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", build)
    values = operand_values(name, SMALL, seed=3)
    operands = [
        stored(dense, fmt, tensor)
        for dense, fmt, tensor in zip(values, PROGRAMS[name][1], "AB")
    ]

    with pytest.raises(UnsupportedFeature) as refusal:
        ops.lower_and_exec_cin(
            program(name, update), result_shape(name, SMALL), *operands
        )

    assert_refused(refusal.value, update)


@pytest.mark.parametrize("update", NON_ADDITIVE, ids=str)
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_the_new_compiler_refuses_it_with_the_same_code(name, update):
    with pytest.raises(LoopIRLoweringError) as refusal:
        lower_normalized_cin_to_loopir(normalize_cin(program(name, update)))

    assert refusal.value.defect.code == "unsupported_update_op"


@pytest.mark.parametrize("update", NON_ADDITIVE, ids=str)
@pytest.mark.parametrize(
    ("producer_update", "consumer_update", "tensor", "path"),
    [
        ("update", None, "w", ("root", "stmt", "producer", "stmt", "stmt", "op")),
        (Operation.ADD, "update", "C", ("root", "stmt", "consumer", "stmt", "op")),
        # Both: the producer runs first and is the one named.
        ("update", "update", "w", ("root", "stmt", "producer", "stmt", "stmt", "op")),
    ],
)
def test_a_workspace_producer_or_consumer_is_checked_too(
    producer_update, consumer_update, tensor, path, update
):
    stmt = workspace_program(
        update if producer_update == "update" else producer_update,
        update if consumer_update == "update" else consumer_update,
    )

    with pytest.raises(UnsupportedFeature) as refusal:
        lower(stmt)

    diagnostic = assert_refused(refusal.value, update, tensor)
    assert diagnostic.path == path


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("update", ADDITIVE, ids=str)
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_equals_and_plus_equals_still_lower(name, update, route):
    source = legacy_generated_cpp(
        program(name, update),
        result_shape(name, SMALL),
        bindings(name, SMALL),
        compile_options=ROUTES[route],
    )

    assert "evaluate(" in source


@pytest.mark.parametrize("consumer_update", ADDITIVE, ids=str)
def test_an_additive_workspace_still_lowers(consumer_update):
    assert lower(workspace_program(Operation.ADD, consumer_update)) is not None


@pytest.mark.parametrize("extents", [SMALL, LARGER], ids=["small", "larger"])
@pytest.mark.parametrize(("name", "fmt"), [("row sum ds", "d"), ("product ss", "ss")])
def test_an_additive_program_computes_and_its_non_additive_form_is_then_refused(
    name, fmt, extents
):
    values = operand_values(name, extents, seed=11)
    operands = [
        stored(dense, operand_format, tensor)
        for dense, operand_format, tensor in zip(values, PROGRAMS[name][1], "AB")
    ]
    reference = expected(name, values)

    for update in ADDITIVE:
        result = ops.lower_and_exec_cin(
            program(name, update), result_shape(name, extents), *operands
        )
        torch.testing.assert_close(
            logical_dense(result), reference, rtol=1e-12, atol=1e-12
        )
        if fmt != "d":
            arrays, stored_values = level_arrays(reference, fmt)
            for level, (actual, wanted) in enumerate(
                zip(result.storage.index.mode_indices, arrays)
            ):
                assert [array.tolist() for array in actual] == [
                    array.tolist() for array in wanted
                ], f"level {level}"
            assert result.storage.value.tolist() == stored_values.tolist()

    # The same program, its kernel now built and cached: the operator is still read.
    for update in NON_ADDITIVE:
        with pytest.raises(UnsupportedFeature) as refusal:
            ops.lower_and_exec_cin(
                program(name, update), result_shape(name, extents), *operands
            )
        assert_refused(refusal.value, update)
