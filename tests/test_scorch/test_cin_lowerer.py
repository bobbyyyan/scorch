from typing import cast

import pytest

from scorch.compiler import llir  # type: ignore[import-untyped]
from scorch.compiler.cin import (
    CIN,
    ForAll,
    IndexStmt,
    IndexVar,
    Operation,
    PostOp,
    PostOps,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_lowerer import CINLowerer, ResultTensorAssembler
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.diagnostics import CompilerInvariantError, UnsupportedFeature
from scorch.compiler.llir_traversal import LLIRTraversalContext, LLIRWalker
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

    statements = CINLowerer(post_ops=post_ops)._emit_post_ops("output", "i")

    assert len(statements) == 1
    assignment = cast(llir.Assign, statements[0])
    assert type(assignment.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, assignment.var)
    assert cast(llir.Var, target.array).name == "output"
    assert cast(llir.Var, target.index).name == "i"
    assert LLIRLowerer().lower_llir(assignment).startswith("output[i] ")


def test_result_position_initialization_uses_a_frozen_structured_target() -> None:
    result = TensorVar("Result", fmt="ds")

    statements = ResultTensorAssembler(result).emit_level_indices_init()
    assignment = next(
        cast(llir.Assign, statement)
        for statement in statements
        if type(statement) is llir.Assign
    )

    assert type(assignment.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, assignment.var)
    assert cast(llir.Var, target.array).name == "Result1_pos"
    assert cast(llir.Literal, target.index).value == 0
    assert LLIRLowerer().lower_llir(assignment) == "Result1_pos[0] = 0;"


def test_all_coo_workspace_assembly_targets_use_storage_types() -> None:
    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("Result", fmt="oo")
    left = TensorVar("Left", fmt="ds", mode_order=[1, 0])
    right = TensorVar("Right", fmt="ds")
    workspace = Workspace("wksp", dim=2)
    statement = Where(
        producer=ForAll(
            reduction,
            ForAll(
                row,
                ForAll(
                    column,
                    TensorAssign(
                        workspace[row, column],
                        left[row, reduction] * right[reduction, column],
                        op=Operation.ADD,
                    ),
                ),
            ),
        ),
        consumer=ForAll(
            row,
            ForAll(
                column,
                TensorAssign(result[row, column], workspace[row, column]),
            ),
        ),
    )
    lowerer = CINLowerer()
    lowerer.outermost_stmt = statement
    lowerer.result_tensor_var = result
    lowerer.result_tensor_access = statement.consumer.get_result_tensor_accesses()[0]
    lowered = lowerer.lower_outer_ConsumerIndexStmt(statement.consumer)
    workspace_loop = next(
        cast(llir.ForLoopAuto, node)
        for node in lowered
        if type(node) is llir.ForLoopAuto
    )

    workspace_assignments = [
        cast(llir.Assign, node)
        for node in workspace_loop.body
        if type(node) is llir.Assign
    ]
    assert [LLIRLowerer().lower_llir(node) for node in workspace_assignments] == [
        "Result0_crd[pResult0] = it.first[0];",
        "Result1_crd[pResult1] = it.first[1];",
        "Result_values[pResult0] = it.second;",
    ]

    storage_types = {
        cast(llir.Var, cast(llir.ArrayAccess, node.var).array)
        .name: cast(llir.Var, cast(llir.ArrayAccess, node.var).array)
        .type
        for node in workspace_assignments
    }
    assert storage_types == {
        "Result0_crd": llir.DataType.STD_VECTOR_C_INT,
        "Result1_crd": llir.DataType.STD_VECTOR_C_INT,
        "Result_values": llir.DataType.STD_VECTOR_FLOAT32,
    }

    lowered_function = CINLowerer().lower_IndexStmt(statement)
    dynamic_vector_assignments: list[llir.Assign] = []

    class DynamicVectorAssignmentCollector(LLIRWalker):
        def visit_assign(self, node: llir.Assign, path: tuple[str, ...]) -> None:
            dynamic_vector_assignments.append(node)
            super().visit_assign(node, path)

    DynamicVectorAssignmentCollector(
        LLIRTraversalContext(
            stage="test",
            pass_name="collect_dynamic_vector_assignments",
        )
    ).walk(lowered_function)
    assert not any(
        type(node.var) is llir.ArrayAccess
        and cast(llir.Var, cast(llir.ArrayAccess, node.var).array).name
        in {"Result0_crd", "Result1_crd", "Result_values"}
        for node in dynamic_vector_assignments
    )


def test_mutated_unknown_post_op_cannot_be_silently_skipped():
    post_ops = PostOps(ops=[], extra_tensors=[])
    lowerer = CINLowerer(post_ops=post_ops)
    post_ops.ops.append(PostOp(kind="clip"))

    with pytest.raises(
        UnsupportedFeature,
        match=r"stage=CIN lowering: unsupported post-op kind 'clip'",
    ):
        lowerer._emit_post_ops("output", "i")


@pytest.mark.parametrize(
    ("fmt", "indices", "expected_index"),
    (
        ("d", ("i",), "i"),
        ("ds", ("i", "j"), "pInput1"),
    ),
)
def test_nonworkspace_tensor_reads_lower_to_frozen_structured_accesses(
    fmt: str,
    indices: tuple[str, ...],
    expected_index: str,
) -> None:
    index_vars = tuple(IndexVar(name) for name in indices)
    tensor = TensorVar("Input", fmt=fmt)
    access = tensor[index_vars[0] if len(index_vars) == 1 else index_vars]
    original_indices = tuple(access.indices)

    lowered = CINLowerer().lower_TensorAccess(access)

    assert type(lowered) is llir.ArrayAccess
    structured = cast(llir.ArrayAccess, lowered)
    assert cast(llir.Var, structured.array).name == "Input_val"
    assert cast(llir.Var, structured.array).type is llir.DataType.PTR_FLOAT32
    assert cast(llir.Var, structured.index).name == expected_index
    assert cast(llir.Var, structured.index).type is llir.DataType.INT
    assert structured.tensor_access == llir.TensorAccessMetadata(
        access_id=access.access_id,
        tensor_id=access.tensor_id,
        index_ids=access.index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    assert LLIRLowerer().lower_llir(structured) == f"Input_val[{expected_index}]"
    assert tuple(access.indices) == original_indices
    assert access.tensor is tensor


def test_cin_reference_rewrite_rebuilds_frozen_access_and_preserves_metadata() -> None:
    index = IndexVar("ix")
    tensor = TensorVar("Input", fmt="d")
    logical_access = tensor[index]
    metadata = llir.TensorAccessMetadata(
        access_id=logical_access.access_id,
        tensor_id=logical_access.tensor_id,
        index_ids=logical_access.index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    source = llir.ArrayAccess(
        array=llir.Var("Array[ix]", llir.DataType.PTR_FLOAT32),
        index=llir.Var("ix offset", llir.DataType.INT),
        tensor_access=metadata,
    )

    rewritten = CINLowerer._rewrite_expr_refs(source, {"ix": "root"})
    repeated = CINLowerer._rewrite_expr_refs(rewritten, {"ix": "root"})

    assert type(rewritten) is llir.ArrayAccess
    assert cast(llir.Var, rewritten.array).name == "Array[root]"
    assert cast(llir.Var, rewritten.index).name == "root offset"
    assert rewritten.tensor_access is metadata
    assert repeated == rewritten
    assert repeated is not rewritten
    assert cast(llir.Var, source.array).name == "Array[ix]"
    assert cast(llir.Var, source.index).name == "ix offset"


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
