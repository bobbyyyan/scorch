from typing import cast

import pytest

from scorch.compiler.cin import (
    CIN,
    ForAll,
    IndexStmt,
    IndexVar,
    PostOp,
    PostOps,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.diagnostics import CompilerInvariantError, UnsupportedFeature
from scorch.compiler.scheduler import Scheduler  # type: ignore[import-untyped]


class UnknownCIN(CIN):
    pass


class UnknownIndexStmt(IndexStmt):
    pass


def test_unknown_cin_node_fails_at_cin_lowering():
    with pytest.raises(
        CompilerInvariantError,
        match=r"stage=CIN lowering: unknown CIN node type 'UnknownCIN'",
    ):
        CINLowerer().lower_CIN(UnknownCIN())


def test_unknown_index_statement_fails_before_lowering_an_empty_kernel():
    with pytest.raises(
        CompilerInvariantError,
        match=r"stage=CIN lowering: unknown IndexStmt node type 'UnknownIndexStmt'",
    ):
        CINLowerer().lower_IndexStmt(UnknownIndexStmt(lhs=None, rhs=None))


def test_unknown_post_op_fails_at_cin_lowering():
    post_ops = PostOps(ops=[PostOp(kind="clip")], extra_tensors=[])

    with pytest.raises(
        UnsupportedFeature,
        match=r"stage=CIN lowering: unsupported post-op kind 'clip'",
    ):
        CINLowerer(post_ops=post_ops)


@pytest.mark.parametrize(
    ("kind", "tensor_name"),
    [
        ("add", "bias"),
        ("mul", "scale"),
        ("relu", None),
        ("gelu", None),
        ("tanh", None),
        ("sigmoid", None),
    ],
)
def test_supported_post_ops_still_lower(kind, tensor_name):
    extra_tensors = [tensor_name] if tensor_name else []
    post_ops = PostOps(
        ops=[PostOp(kind=kind, tensor_name=tensor_name)],
        extra_tensors=extra_tensors,
    )

    assert len(CINLowerer(post_ops=post_ops)._emit_post_ops("output", "i")) == 1


def test_mutated_unknown_post_op_cannot_be_silently_skipped():
    post_ops = PostOps(ops=[], extra_tensors=[])
    lowerer = CINLowerer(post_ops=post_ops)
    post_ops.ops.append(PostOp(kind="clip"))

    with pytest.raises(
        UnsupportedFeature,
        match=r"stage=CIN lowering: unsupported post-op kind 'clip'",
    ):
        lowerer._emit_post_ops("output", "i")


def test_nondefault_coo_intersection_keeps_live_coordinate_end_bounds():
    row, column = IndexVar("r"), IndexVar("c")
    result = TensorVar("Intersect", fmt="oo", mode_order=[1, 0])
    left = TensorVar("Left", fmt="oo", mode_order=[1, 0])
    right = TensorVar("Right", fmt="oo", mode_order=[1, 0])
    assignment = TensorAssign(
        result[row, column],
        left[row, column] * right[row, column],
    )
    scheduled = cast(
        ForAll,
        Scheduler.auto_schedule(ForAll(row, ForAll(column, assignment))),
    )

    cpp = LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(scheduled))

    assert "int pLeft1_end = 0;" in cpp
    assert "pLeft1_end = pLeft0 + 1;" in cpp
    assert "pLeft1_end < pLeft0_end" in cpp
    assert "pLeft1 < pLeft1_end && pRight1 < pRight1_end" in cpp
    assert "pLeft0 = pLeft1_end;" in cpp
    assert "int pRight1_end = 0;" in cpp
    assert "pRight1_end = pRight0 + 1;" in cpp
    assert "pRight1_end < pRight0_end" in cpp
    assert "pRight0 = pRight1_end;" in cpp
