import pytest

from scorch.compiler.cin import CIN, IndexStmt, PostOp, PostOps
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.diagnostics import CompilerInvariantError, UnsupportedFeature


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
