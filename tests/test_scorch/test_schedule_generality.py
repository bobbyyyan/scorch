from typing import cast

import pytest
import torch

from scorch import STensor, einsum
from scorch.compiler import llir
from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.compiler.legacy_cin_adapter import legacy_cin_working_copy
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRTraversalContext,
    LLIRTraversalError,
    LLIRWalker,
)
from scorch.compiler.schedule_lowerer import (  # type: ignore[import-untyped]
    _contains_tensor_access,
    _is_operand_prefetch_guard,
    _matches_tensor_access,
    _redirect_sparse_prefetch,
    _rewrite_expr_access,
    _rewrite_stmt_accesses,
)
from scorch.compiler.sparse_prefetch_pass import (  # type: ignore[import-untyped]
    SPARSE_PREFETCH_CONTEXT,
    insert_sparse_prefetch,
)
from scorch.compiler.scheduler import (
    Schedule,
    Scheduler,
    TileSpec,
    regblock_force,
)


def _nest(index_vars, assignment):
    stmt = assignment
    for index_var in reversed(index_vars):
        stmt = ForAll(index_var, stmt)
    assert isinstance(stmt, ForAll)
    return stmt


def _lower_to_cpp(stmt: ForAll) -> str:
    return LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(stmt))


def _sparse_stensor(tensor: torch.Tensor, name: str, fmt: str = "ds") -> STensor:
    return STensor.from_torch(tensor, name).to_sparse(fmt)


def _build_elementwise(fmt: str) -> ForAll:
    row, col = IndexVar("r"), IndexVar("c")
    out = TensorVar("ElemOut", fmt=fmt)
    left = TensorVar("ElemLeft", fmt=fmt)
    right = TensorVar("ElemRight", fmt=fmt)
    assignment = TensorAssign(out[row, col], left[row, col] * right[row, col])
    return _nest((row, col), assignment)


def _build_ttm() -> ForAll:
    batch, row, reduction, feature = (
        IndexVar("a"),
        IndexVar("b"),
        IndexVar("c"),
        IndexVar("d"),
    )
    out = TensorVar("Projected", fmt="ddd")
    core = TensorVar("Core", fmt="ddd")
    factor = TensorVar("Factor", fmt="dd")
    assignment = TensorAssign(
        out[batch, row, feature],
        core[batch, row, reduction] * factor[reduction, feature],
        op=Operation.ADD,
    )
    return _nest((batch, row, reduction, feature), assignment)


def _build_sddmm() -> ForAll:
    row, col, reduction = IndexVar("r"), IndexVar("c"), IndexVar("q")
    out = TensorVar("Sampled", fmt="oo")
    mask = TensorVar("Mask", fmt="ds")
    left = TensorVar("Query", fmt="dd")
    right = TensorVar("Key", fmt="dd")
    assignment = TensorAssign(
        out[row, col],
        mask[row, col] * left[row, reduction] * right[col, reduction],
        op=Operation.ADD,
    )
    return _nest((row, col, reduction), assignment)


def _build_spgemm() -> ForAll:
    row, reduction, col = IndexVar("r"), IndexVar("q"), IndexVar("c")
    out = TensorVar("SparseProduct", fmt="ds")
    left = TensorVar("SparseLeft", fmt="ds")
    right = TensorVar("SparseRight", fmt="ds")
    assignment = TensorAssign(
        out[row, col],
        left[row, reduction] * right[reduction, col],
        op=Operation.ADD,
    )
    return _nest((row, reduction, col), assignment)


def _build_outer_workspace_spgemm(result_format: str) -> ForAll:
    row, reduction, col = IndexVar("r"), IndexVar("q"), IndexVar("c")
    out = TensorVar("SparseProduct", fmt=result_format)
    left = TensorVar("SparseLeft", fmt="ds", mode_order=[1, 0])
    right = TensorVar("SparseRight", fmt="ds")
    assignment = TensorAssign(
        out[row, col],
        left[row, reduction] * right[reduction, col],
        op=Operation.ADD,
    )
    return _nest((reduction, row, col), assignment)


def _build_nested_rank_two_workspace() -> ForAll:
    batch, reduction, row, col = (
        IndexVar("a"),
        IndexVar("q"),
        IndexVar("r"),
        IndexVar("c"),
    )
    out = TensorVar("BatchedProduct", fmt="doo")
    left = TensorVar("BatchedLeft", fmt="ddd", mode_order=[0, 2, 1])
    right = TensorVar("BatchedRight", fmt="ddd")
    assignment = TensorAssign(
        out[batch, row, col],
        left[batch, row, reduction] * right[batch, reduction, col],
        op=Operation.ADD,
    )
    return _nest((batch, reduction, row, col), assignment)


def _build_spmv() -> ForAll:
    row, reduction = IndexVar("r"), IndexVar("q")
    out = TensorVar("VectorOut", fmt="d")
    matrix = TensorVar("SparseMatrix", fmt="ds")
    vector = TensorVar("DenseVector", fmt="d")
    assignment = TensorAssign(
        out[row],
        matrix[row, reduction] * vector[reduction],
        op=Operation.ADD,
    )
    return _nest((row, reduction), assignment)


def _build_dense_matmul() -> ForAll:
    row, reduction, col = IndexVar("r"), IndexVar("q"), IndexVar("c")
    out = TensorVar("DenseProduct", fmt="dd")
    left = TensorVar("DenseLeft", fmt="dd")
    right = TensorVar("DenseRight", fmt="dd")
    assignment = TensorAssign(
        out[row, col],
        left[row, reduction] * right[reduction, col],
        op=Operation.ADD,
    )
    return _nest((row, reduction, col), assignment)


def test_dense_elementwise_affine_free_axis_tiles_are_generic_and_ragged():
    schedule = Schedule(
        loop_order=("r", "c"),
        tiles=(
            TileSpec("r", 3, accum="direct", unroll=False),
            TileSpec(
                "c",
                4,
                placement="child_of:r_in",
                accum="direct",
                unroll=False,
            ),
        ),
        tag="generic-elementwise-r-c",
    )

    scheduled = Scheduler.apply_schedule(_build_elementwise("dd"), schedule)
    cpp = _lower_to_cpp(scheduled)

    assert "constexpr int kTile_r = 3;" in cpp
    assert "constexpr int kTile_c = 4;" in cpp
    assert cpp.index("r_out = 0") < cpp.index("r_in = 0")
    assert cpp.index("r_in = 0") < cpp.index("c_out = 0")
    assert cpp.index("c_out = 0") < cpp.index("c_in = 0")
    assert "if (r >= ElemLeft0_size)" in cpp
    assert "if (c >= ElemLeft1_size)" in cpp
    assert "packed_" not in cpp

    left = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    right = (torch.arange(35, dtype=torch.float32).reshape(7, 5) + 1) / 7
    result = einsum(
        "rc,rc->rc",
        STensor.from_torch(left, "ElemLeft"),
        STensor.from_torch(right, "ElemRight"),
        format="dd",
        schedule=schedule,
    )

    assert torch.allclose(result.to_torch(), left * right)


def test_dense_elementwise_affine_tile_and_cache_support_float64():
    schedule = Schedule(
        loop_order=("r", "c"),
        tiles=(TileSpec("r", 3, accum="direct", unroll=False),),
        tag="generic-elementwise-float64",
    )
    left = torch.arange(35, dtype=torch.float64).reshape(7, 5)
    right = torch.linspace(0.5, 2.0, 35, dtype=torch.float64).reshape(7, 5)

    float_result = einsum(
        "rc,rc->rc",
        STensor.from_torch(left.float(), "ElemLeft"),
        STensor.from_torch(right.float(), "ElemRight"),
        format="dd",
        schedule=schedule,
    )
    result = einsum(
        "rc,rc->rc",
        STensor.from_torch(left, "ElemLeft"),
        STensor.from_torch(right, "ElemRight"),
        format="dd",
        schedule=schedule,
    )

    assert float_result.dtype == torch.float32
    assert result.dtype == torch.float64
    assert torch.allclose(result.to_torch(), left * right)


def test_llir_continue_has_a_general_cpp_lowering():
    assert LLIRLowerer().lower_llir(llir.Continue()) == "continue;"


def test_schedule_access_matching_uses_typed_identity_not_display_spelling():
    tensor_id = SymbolId(1)
    index_ids = (IndexId(2), IndexId(3))
    access = llir.ArrayAccess(
        array=llir.Var("SameDisplay_val", llir.DataType.PTR_FLOAT32),
        index=llir.Var("same_position", llir.DataType.INT),
        tensor_access=llir.TensorAccessMetadata(
            access_id=AccessId(4),
            tensor_id=tensor_id,
            index_ids=index_ids,
            role=llir.TensorAccessRole.INPUT_READ,
        ),
    )

    assert _matches_tensor_access(
        access,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )
    assert not _matches_tensor_access(
        access,
        SymbolId(99),
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )
    assert not _matches_tensor_access(
        access,
        tensor_id,
        (IndexId(2), IndexId(99)),
        llir.TensorAccessRole.INPUT_READ,
    )
    assert not _matches_tensor_access(
        access,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.RESULT_WRITE,
    )
    same_logical_access = llir.ArrayAccess(
        array=llir.Var("DifferentDisplay_val", llir.DataType.PTR_FLOAT32),
        index=llir.Var("different_position", llir.DataType.INT),
        tensor_access=llir.TensorAccessMetadata(
            access_id=AccessId(999),
            tensor_id=tensor_id,
            index_ids=index_ids,
            role=llir.TensorAccessRole.INPUT_READ,
        ),
    )
    assert _matches_tensor_access(
        same_logical_access,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )


def test_schedule_access_rewrite_is_detached_repeatable_and_fail_closed():
    tensor_id = SymbolId(11)
    other_tensor_id = SymbolId(12)
    index_ids = (IndexId(13),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(14),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    other_metadata = llir.TensorAccessMetadata(
        access_id=AccessId(15),
        tensor_id=other_tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    selected = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pInput", llir.DataType.INT),
        metadata,
    )
    unselected = llir.ArrayAccess(
        llir.Var("Other_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pOther", llir.DataType.INT),
        other_metadata,
    )
    source = llir.BinOp("+", selected, unselected)
    replacement = llir.ArrayAccess(
        llir.Var("packed_Input", llir.DataType.PTR_FLOAT32),
        llir.Var("packed_position", llir.DataType.INT),
    )
    lowerer = LLIRLowerer()
    source_cpp = lowerer.lower_llir(source)

    first, first_count = _rewrite_expr_access(
        source,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    second, second_count = _rewrite_expr_access(
        first,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )

    assert first_count == 1
    assert second_count == 0
    assert lowerer.lower_llir(source) == source_cpp
    assert (
        lowerer.lower_llir(first) == "packed_Input[packed_position] + Other_val[pOther]"
    )
    assert lowerer.lower_llir(second) == lowerer.lower_llir(first)
    assert first is not source
    assert second is not first
    assert first.left is not replacement
    assert first.right is not unselected
    assert first.right.tensor_access is other_metadata
    assert not _contains_tensor_access(
        [llir.Assign(llir.Var("out", llir.DataType.FLOAT32), first)],
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )

    class UnknownExpr(llir.Expr):
        pass

    malformed = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        UnknownExpr(),
    )
    with pytest.raises(LLIRTraversalError) as raised:
        _rewrite_expr_access(
            malformed,
            tensor_id,
            index_ids,
            llir.TensorAccessRole.INPUT_READ,
            replacement,
        )
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "index")


def test_schedule_statement_access_rewrite_counts_and_clones_each_replacement():
    tensor_id = SymbolId(21)
    index_ids = (IndexId(22),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(23),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    access = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pInput", llir.DataType.INT),
        metadata,
    )
    statements = [
        llir.Assign(llir.Var("left", llir.DataType.FLOAT32), access),
        llir.Assign(llir.Var("right", llir.DataType.FLOAT32), access),
    ]
    replacement = llir.ArrayAccess(
        llir.Var("packed_Input", llir.DataType.PTR_FLOAT32),
        llir.Var("packed_position", llir.DataType.INT),
    )

    first_count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    first_values = [statement.value for statement in statements]
    second_count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )

    assert first_count == 2
    assert second_count == 0
    assert first_values[0] is not first_values[1]
    assert first_values[0] is not replacement
    assert first_values[1] is not replacement
    assert all(
        LLIRLowerer().lower_llir(statement.value) == "packed_Input[packed_position]"
        for statement in statements
    )


def _prefetchable_sparse_loop() -> llir.ForLoop:
    """One canonical CSR-style position loop that activates typed P1."""

    def _nt(name: str) -> llir.Var:
        return llir.Var(name, llir.DataType.NO_TYPE)

    inner = llir.ForLoop(
        init=llir.VarInit(
            llir.Var("j", llir.DataType.INT),
            llir.Literal(0),
        ),
        cond=llir.BinOp("<", _nt("j"), _nt("B1_size")),
        update=llir.Increment(_nt("j")),
        body=[
            llir.VarInit(
                _nt("pB1"),
                llir.Add(llir.Mul(_nt("coordinate"), _nt("B1_size")), _nt("j")),
            ),
            llir.Assign(
                llir.ArrayAccess(_nt("C_val"), llir.Var("pC1", llir.DataType.INT64)),
                _nt("B_val[pB1]"),
            ),
        ],
    )
    return llir.ForLoop(
        init=llir.VarInit(
            _nt("pA1"),
            llir.ArrayAccess(
                llir.Var("A1_pos", llir.DataType.PTR_INT),
                llir.Var("pA0", llir.DataType.INT),
            ),
        ),
        cond=llir.BinOp("<", _nt("pA1"), _nt("pA1_end")),
        update=llir.Increment(_nt("pA1")),
        body=[
            llir.VarInit(
                _nt("coordinate"),
                llir.ArrayAccess(
                    llir.Var("A1_crd", llir.DataType.PTR_INT),
                    llir.Var("pA1", llir.DataType.INT),
                ),
            ),
            inner,
        ],
    )


@pytest.mark.parametrize(
    ("stage_row_origin", "staged_row"),
    (
        (None, "A1_crd[pA1 + 1]"),
        ("j_out", "(A1_crd[pA1 + 1] - j_out)"),
    ),
    ids=("no-origin", "origin"),
)
def test_redirect_composes_with_the_typed_sparse_prefetch_producer(
    stage_row_origin, staged_row
) -> None:
    """P1's actual output is removed and replaced by the packed guard."""

    output = insert_sparse_prefetch(
        [_prefetchable_sparse_loop()], SPARSE_PREFETCH_CONTEXT
    )
    sparse_loop = cast(llir.ForLoop, output[0])
    produced = sparse_loop.body[0]
    assert type(produced) is llir.GuardedCallStmt
    assert _is_operand_prefetch_guard(produced, "B_val")
    assert not _is_operand_prefetch_guard(produced, "C_val")

    _redirect_sparse_prefetch(
        sparse_loop,
        "B_val",
        "packed_B",
        "j_out",
        "j_out_end",
        "kTile_k",
        stage_row_origin,
    )

    assert produced not in sparse_loop.body
    packed_guard = sparse_loop.body[0]
    assert type(packed_guard) is llir.GuardedCallStmt
    assert not _is_operand_prefetch_guard(packed_guard, "packed_B")
    assert LLIRLowerer().lower_llir(packed_guard) == (
        "if (pA1 + 1 < pA1_end && A1_crd[pA1 + 1] >= j_out && "
        "A1_crd[pA1 + 1] < j_out_end) "
        f"__builtin_prefetch(&packed_B[{staged_row} * kTile_k], 0, 1);"
    )


def test_redirect_rejects_non_identifier_staged_guard_spellings() -> None:
    output = insert_sparse_prefetch(
        [_prefetchable_sparse_loop()], SPARSE_PREFETCH_CONTEXT
    )
    sparse_loop = cast(llir.ForLoop, output[0])
    original_body = sparse_loop.body
    original_statements = list(original_body)

    with pytest.raises(NotImplementedError, match="identifier spellings"):
        _redirect_sparse_prefetch(
            sparse_loop,
            "B_val",
            "packed_B",
            "j_out + 1",
            "j_out_end",
            "kTile_k",
            None,
        )

    assert sparse_loop.body is original_body
    assert sparse_loop.body == original_statements
    assert all(
        actual is expected
        for actual, expected in zip(sparse_loop.body, original_statements)
    )


def test_redirect_recognition_is_structural_and_ignores_decoys() -> None:
    """Only P1's complete typed shape is eligible for replacement."""

    def _nt(name: str) -> llir.Var:
        return llir.Var(name, llir.DataType.NO_TYPE)

    def _borrow(
        array: str,
        *,
        iterator: str = "pA1",
        coordinate_array: str = "A1_crd",
    ) -> llir.AddressOf:
        return llir.AddressOf(
            operand=llir.ArrayAccess(
                array=_nt(array),
                index=llir.Mul(
                    llir.ArrayAccess(
                        _nt(coordinate_array),
                        llir.Add(_nt(iterator), llir.Literal(1, llir.DataType.INT)),
                    ),
                    _nt("B1_size"),
                ),
            )
        )

    guard_condition = llir.BinOp(
        "<",
        llir.Add(_nt("pA1"), llir.Literal(1, llir.DataType.INT)),
        _nt("pA1_end"),
    )
    exact_arguments = (
        _borrow("B_val"),
        llir.Literal(0, llir.DataType.INT),
        llir.Literal(1, llir.DataType.INT),
    )
    decoys: list[llir.Stmt] = [
        llir.RawStmt(
            "if (pA1 + 1 < pA1_end) "
            "__builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size], 0, 1)"
        ),
        llir.GuardedCallStmt(
            cond=guard_condition,
            call=llir.FunctionCallStmt("__builtin_expect", (_borrow("B_val"),)),
        ),
        llir.GuardedCallStmt(
            cond=llir.BinOp(
                "<",
                llir.Add(_nt("pA1"), llir.Literal(1, llir.DataType.INT)),
                _nt("pA1_end"),
            ),
            call=llir.FunctionCallStmt("__builtin_prefetch", (_borrow("C_val"),)),
        ),
        llir.FunctionCallStmt("__builtin_prefetch", (_borrow("B_val"),)),
        llir.GuardedCallStmt(
            cond=llir.BinOp("==", _nt("unrelated"), llir.Literal(0)),
            call=llir.FunctionCallStmt("__builtin_prefetch", exact_arguments),
        ),
        llir.GuardedCallStmt(
            cond=llir.BinOp(
                "<",
                llir.Add(_nt("pA1"), llir.Literal(1, llir.DataType.INT)),
                _nt("pA1_end"),
            ),
            call=llir.FunctionCallStmt(
                "__builtin_prefetch",
                (
                    llir.AddressOf(llir.ArrayAccess(_nt("B_val"), _nt("wrong_index"))),
                    llir.Literal(0, llir.DataType.INT),
                    llir.Literal(1, llir.DataType.INT),
                ),
            ),
        ),
        llir.GuardedCallStmt(
            cond=llir.BinOp(
                "<",
                llir.Add(_nt("pA1"), llir.Literal(1, llir.DataType.INT)),
                _nt("pA1_end"),
            ),
            call=llir.FunctionCallStmt(
                "__builtin_prefetch",
                (
                    _borrow("B_val"),
                    llir.Literal(1, llir.DataType.INT),
                    llir.Literal(0, llir.DataType.INT),
                    llir.Literal(99, llir.DataType.INT),
                ),
            ),
        ),
        llir.GuardedCallStmt(
            cond=llir.BinOp(
                "<",
                llir.Add(_nt("pA1"), llir.Literal(1, llir.DataType.INT)),
                _nt("pA1_end"),
            ),
            call=llir.FunctionCallStmt(
                "__builtin_prefetch",
                template_args=(llir.DataType.INT,),
                args=exact_arguments,
            ),
        ),
    ]
    other_loop_p1 = llir.GuardedCallStmt(
        cond=llir.BinOp(
            "<",
            llir.Add(_nt("other"), llir.Literal(1, llir.DataType.INT)),
            _nt("other_end"),
        ),
        call=llir.FunctionCallStmt(
            "__builtin_prefetch",
            (
                _borrow(
                    "B_val",
                    iterator="other",
                    coordinate_array="Other_crd",
                ),
                llir.Literal(0, llir.DataType.INT),
                llir.Literal(1, llir.DataType.INT),
            ),
        ),
    )
    sparse_loop = _prefetchable_sparse_loop()
    sparse_loop.body = [
        sparse_loop.body[0],
        *decoys,
        other_loop_p1,
        sparse_loop.body[1],
    ]
    body_before = list(sparse_loop.body)

    for decoy in decoys:
        assert not _is_operand_prefetch_guard(decoy, "B_val")
    assert _is_operand_prefetch_guard(other_loop_p1, "B_val")

    _redirect_sparse_prefetch(
        sparse_loop,
        "B_val",
        "packed_B",
        "j_out",
        "j_out_end",
        "kTile_k",
        None,
    )

    assert sparse_loop.body == body_before
    assert all(
        existing is expected
        for existing, expected in zip(sparse_loop.body, body_before)
    )


def test_schedule_guarded_call_statement_rewrites_condition_and_arguments() -> None:
    tensor_id = SymbolId(31)
    index_ids = (IndexId(32),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(33),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    access = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pInput", llir.DataType.INT),
        metadata,
    )
    source = llir.GuardedCallStmt(
        cond=llir.BinOp("<", access, llir.Var("bound", llir.DataType.INT)),
        call=llir.FunctionCallStmt("__builtin_prefetch", (access,)),
    )
    statements: list[llir.Stmt] = [source]
    replacement = llir.ArrayAccess(
        llir.Var("packed_Input", llir.DataType.PTR_FLOAT32),
        llir.Var("packed_position", llir.DataType.INT),
    )

    count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    first = cast(llir.GuardedCallStmt, statements[0])
    repeated = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    second = cast(llir.GuardedCallStmt, statements[0])

    assert count == 2
    assert repeated == 0
    assert type(first) is llir.GuardedCallStmt
    assert first is not source
    assert second is not first
    assert LLIRLowerer().lower_llir(first) == (
        "if (packed_Input[packed_position] < bound) "
        "__builtin_prefetch(packed_Input[packed_position]);"
    )


def test_schedule_untagged_guarded_call_statement_is_a_detached_pass_through() -> None:
    tensor_id = SymbolId(41)
    index_ids = (IndexId(42),)
    source = llir.GuardedCallStmt(
        cond=llir.BinOp(
            "<",
            llir.Add(
                llir.Var("pA1", llir.DataType.NO_TYPE),
                llir.Literal(1, llir.DataType.INT),
            ),
            llir.Var("pA1_end", llir.DataType.NO_TYPE),
        ),
        call=llir.FunctionCallStmt(
            "__builtin_prefetch",
            (
                llir.AddressOf(
                    operand=llir.ArrayAccess(
                        array=llir.Var("B_val", llir.DataType.NO_TYPE),
                        index=llir.Mul(
                            llir.ArrayAccess(
                                array=llir.Var("A1_crd", llir.DataType.NO_TYPE),
                                index=llir.Add(
                                    llir.Var("pA1", llir.DataType.NO_TYPE),
                                    llir.Literal(1, llir.DataType.INT),
                                ),
                            ),
                            llir.Var("B1_size", llir.DataType.NO_TYPE),
                        ),
                    )
                ),
                llir.Literal(0, llir.DataType.INT),
                llir.Literal(1, llir.DataType.INT),
            ),
        ),
    )
    statements: list[llir.Stmt] = [source]
    replacement = llir.ArrayAccess(
        llir.Var("packed_B", llir.DataType.PTR_FLOAT32),
        llir.Var("packed_position", llir.DataType.INT),
    )

    count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    rewritten = cast(llir.GuardedCallStmt, statements[0])

    assert count == 0
    assert rewritten == source
    assert rewritten is not source
    assert LLIRLowerer().lower_llir(rewritten) == (
        "if (pA1 + 1 < pA1_end) "
        "__builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size], 0, 1);"
    )


def test_schedule_member_call_statement_rewrites_receiver_and_arguments() -> None:
    tensor_id = SymbolId(27)
    index_ids = (IndexId(28),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(29),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    access = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pInput", llir.DataType.INT),
        metadata,
    )
    source = llir.MemberCallStmt(
        base=access,
        member="consume",
        template_args=(llir.DataType.FLOAT32,),
        args=(access,),
    )
    statements: list[llir.Stmt] = [source]
    replacement = llir.ArrayAccess(
        llir.Var("packed_Input", llir.DataType.PTR_FLOAT32),
        llir.Var("packed_position", llir.DataType.INT),
    )

    count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    first = cast(llir.MemberCallStmt, statements[0])
    repeated = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    second = cast(llir.MemberCallStmt, statements[0])

    assert count == 2
    assert repeated == 0
    assert first is not source
    assert second is not first
    assert first.member == second.member == "consume"
    assert first.template_args == second.template_args == (llir.DataType.FLOAT32,)
    assert type(first.args) is tuple
    assert type(second.args) is tuple
    assert first.base is not replacement
    assert first.args[0] is not replacement
    assert first.base is not first.args[0]
    assert second.base is not first.base
    assert second.args[0] is not first.args[0]
    assert LLIRLowerer().lower_llir(source) == (
        "Input_val[pInput].consume<float>(Input_val[pInput]);"
    )
    assert LLIRLowerer().lower_llir(first) == (
        "packed_Input[packed_position].consume<float>("
        "packed_Input[packed_position]);"
    )
    assert LLIRLowerer().lower_llir(second) == LLIRLowerer().lower_llir(first)
    assert not _contains_tensor_access(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )


def test_schedule_statement_access_rewrite_preserves_tuple_bodies() -> None:
    tensor_id = SymbolId(24)
    index_ids = (IndexId(25),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(26),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    access = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pInput", llir.DataType.INT),
        metadata,
    )
    outer_call = llir.FunctionCallStmt(
        "consume", (access,), template_args=(llir.DataType.FLOAT32,)
    )
    inner_call = llir.FunctionCallStmt(
        "consume", (access,), template_args=(llir.DataType.FLOAT32,)
    )
    conditional = llir.IfThenElse(
        cond=llir.Var("enabled", llir.DataType.BOOL),
        then_body=cast(list[llir.Stmt], (inner_call,)),
    )
    statements = [outer_call, conditional]
    replacement = llir.ArrayAccess(
        llir.Var("packed_Input", llir.DataType.PTR_FLOAT32),
        llir.Var("packed_position", llir.DataType.INT),
    )

    count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )

    assert count == 2
    assert type(statements) is list
    rewritten_outer = cast(llir.FunctionCallStmt, statements[0])
    rewritten_conditional = cast(llir.IfThenElse, statements[1])
    assert rewritten_outer is not outer_call
    assert type(rewritten_conditional.then_body) is tuple
    rewritten_inner = cast(llir.FunctionCallStmt, rewritten_conditional.then_body[0])
    assert rewritten_inner is not inner_call
    assert rewritten_outer.template_args == (llir.DataType.FLOAT32,)
    assert rewritten_inner.template_args == (llir.DataType.FLOAT32,)
    assert LLIRLowerer().lower_llir(rewritten_outer) == (
        "consume<float>(packed_Input[packed_position]);"
    )
    assert LLIRLowerer().lower_llir(rewritten_inner) == (
        "consume<float>(packed_Input[packed_position]);"
    )


def test_schedule_result_target_rewrite_remains_a_valid_detached_lvalue() -> None:
    tensor_id = SymbolId(31)
    index_ids = (IndexId(32), IndexId(33))
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(34),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.RESULT_WRITE,
    )
    target = llir.ArrayAccess(
        llir.Var("Result_values", llir.DataType.PTR_FLOAT32),
        llir.Var("pResult", llir.DataType.INT64),
        metadata,
    )
    replacement = llir.ArrayAccess(
        llir.Var("compact_Result", llir.DataType.PTR_FLOAT32),
        llir.Add(
            llir.Mul(llir.Var("row", llir.DataType.INT64), llir.Literal(4)),
            llir.Var("lane", llir.DataType.INT64),
        ),
    )
    statements = [llir.Assign(target, llir.Var("value", llir.DataType.FLOAT32))]

    count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.RESULT_WRITE,
        replacement,
    )
    repeated = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.RESULT_WRITE,
        replacement,
    )

    rewritten = cast(llir.ArrayAccess, statements[0].var)
    assert count == 1
    assert repeated == 0
    assert rewritten is not replacement
    assert rewritten.array is not replacement.array
    assert rewritten.index is not replacement.index
    assert LLIRLowerer().lower_llir(statements[0]) == (
        "compact_Result[row * 4 + lane] = value;"
    )
    assert LLIRLowerer().lower_llir(target) == "Result_values[pResult]"

    malformed = [llir.Assign(target, llir.Var("value", llir.DataType.FLOAT32))]
    with pytest.raises(LLIRTraversalError) as raised:
        _rewrite_stmt_accesses(
            malformed,
            tensor_id,
            index_ids,
            llir.TensorAccessRole.RESULT_WRITE,
            llir.BinOp("+", llir.Var("left", llir.DataType.INT64), llir.Literal(1)),
        )
    assert raised.value.diagnostic.code == "invalid_assignment_target"
    assert raised.value.diagnostic.path == ("root", "[0]", "var")
    assert malformed[0].var is target
    assert LLIRLowerer().lower_llir(malformed[0]) == ("Result_values[pResult] = value;")


def test_schedule_nested_read_replacement_preflights_without_mutating_lvalue() -> None:
    tensor_id = SymbolId(41)
    index_ids = (IndexId(42),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(43),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    indirect_index = llir.ArrayAccess(
        llir.Var("indices", llir.DataType.PTR_INT),
        llir.Var("i", llir.DataType.INT64),
        metadata,
    )
    target = llir.ArrayAccess(
        llir.Var("output", llir.DataType.PTR_FLOAT32),
        indirect_index,
    )
    statements = [llir.Assign(target, llir.Var("value", llir.DataType.FLOAT32))]

    with pytest.raises(LLIRTraversalError) as raised:
        _rewrite_stmt_accesses(
            statements,
            tensor_id,
            index_ids,
            llir.TensorAccessRole.INPUT_READ,
            llir.Array([llir.Literal(0)], llir.DataType.INT),
        )

    assert raised.value.diagnostic.code == "invalid_assignment_target"
    assert raised.value.diagnostic.path == ("root", "[0]", "var")
    assert statements[0].var is target
    assert cast(llir.ArrayAccess, statements[0].var).index is indirect_index
    assert LLIRLowerer().lower_llir(statements[0]) == ("output[indices[i]] = value;")


def test_dense_elementwise_llir_tensor_access_metadata_survives_rewrites():
    source = _build_elementwise("dd")
    lowered = CINLowerer().lower_IndexStmt(source)
    assert isinstance(lowered, llir.Function)

    tagged_expressions = []

    def collect_expr(expr):
        if isinstance(expr, llir.Var):
            if expr.tensor_access is not None:
                tagged_expressions.append(expr)
        elif isinstance(expr, llir.BinOp):
            collect_expr(expr.left)
            collect_expr(expr.right)
        elif isinstance(expr, llir.ArrayAccess):
            if expr.tensor_access is not None:
                tagged_expressions.append(expr)
            collect_expr(expr.array)
            collect_expr(expr.index)

    def collect_stmts(stmts):
        for stmt in stmts:
            if isinstance(stmt, llir.Assign):
                collect_expr(stmt.var)
                collect_expr(stmt.value)
            elif isinstance(stmt, llir.VarInit):
                collect_expr(stmt.value)
            elif isinstance(stmt, (llir.ForLoop, llir.WhileLoop)):
                collect_stmts(stmt.body)
            elif isinstance(stmt, llir.IfThenElse):
                for body in [stmt.then_body, stmt.else_body] + (
                    stmt.then_body_list or []
                ):
                    if body:
                        collect_stmts(body)

    collect_stmts(lowered.body)
    metadata = {expression.tensor_access for expression in tagged_expressions}
    expected_metadata = {
        llir.TensorAccessMetadata(
            access_id=access.access_id,
            tensor_id=access.tensor_id,
            index_ids=access.index_ids,
            role=(
                llir.TensorAccessRole.RESULT_WRITE
                if access.tensor.name == "ElemOut"
                else llir.TensorAccessRole.INPUT_READ
            ),
        )
        for access in source.tensor_accesses
    }
    assert metadata == expected_metadata
    assert all(
        type(expression) is llir.ArrayAccess for expression in tagged_expressions
    )
    result_writes = [
        cast(llir.ArrayAccess, expression)
        for expression in tagged_expressions
        if expression.tensor_access.role is llir.TensorAccessRole.RESULT_WRITE
    ]
    assert len(result_writes) == 1
    assert cast(llir.Var, result_writes[0].array).name == "ElemOut_values"
    # Dense pointer hoisting rewrites the physical access spelling, while the
    # logical tensor/index identity remains available to later schedule passes.
    lowerer = LLIRLowerer()
    assert {lowerer.lower_llir(expression) for expression in tagged_expressions} == {
        "_ElemLeft_val_ptr[c]",
        "_ElemRight_val_ptr[c]",
        "ElemOut_values[pElemOut1]",
    }

    for expression in tagged_expressions:
        if type(expression) is llir.Var:
            without_metadata = llir.Var(
                name=expression.name,
                type=expression.type,
                is_ptr=expression.is_ptr,
                is_restrict=expression.is_restrict,
            )
        else:
            assert type(expression) is llir.ArrayAccess
            without_metadata = llir.ArrayAccess(
                array=expression.array,
                index=expression.index,
            )
        assert lowerer.lower_llir(expression) == lowerer.lower_llir(without_metadata)


def test_ttm_row_and_free_axis_tiles_compose_with_ragged_tails():
    schedule = Schedule(
        loop_order=("a", "b", "c", "d"),
        tiles=(
            TileSpec("a", 2, accum="direct", unroll=False),
            TileSpec("d", 3, placement="child_of:b", accum="stack"),
        ),
        tag="generic-ttm-a-d",
    )

    scheduled = Scheduler.apply_schedule(_build_ttm(), schedule)
    cpp = _lower_to_cpp(scheduled)

    assert "constexpr int kTile_a = 2;" in cpp
    assert "constexpr int kTile_d = 3;" in cpp
    assert cpp.index("a_out = 0") < cpp.index("a_in = 0")
    assert cpp.index("a_in = 0") < cpp.index("d_out = 0")
    assert "float wksp[kTile_d] = {};" in cpp
    assert "if (a >= Core0_size)" in cpp
    assert "if (d >= Factor1_size)" in cpp
    assert "Projected_values" in cpp
    assert "packed_" not in cpp

    torch.manual_seed(101)
    core = torch.randn(5, 3, 4)
    factor = torch.randn(4, 7)
    result = einsum(
        "abc,cd->abd",
        STensor.from_torch(core, "Core"),
        STensor.from_torch(factor, "Factor"),
        format="ddd",
        schedule=schedule,
    )
    reference = torch.einsum("abc,cd->abd", core, factor)

    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)


def test_ttm_heap_result_tile_is_generic_compact_accumulation():
    schedule = Schedule(
        loop_order=("a", "b", "c", "d"),
        tiles=(
            TileSpec(
                "d",
                3,
                placement="outermost",
                accum="heap",
                unroll=False,
            ),
        ),
        tag="generic-ttm-heap-d",
        parallel_loop="a",
    )

    scheduled = Scheduler.apply_schedule(_build_ttm(), schedule)

    def lower_with_storage_owner() -> tuple[str, llir.DirectInit]:
        lowered = CINLowerer().lower_IndexStmt(scheduled)
        owners: list[llir.DirectInit] = []

        class StorageOwnerCollector(LLIRWalker):
            def visit_direct_init(
                self,
                node: llir.DirectInit,
                path: tuple[str, ...],
            ) -> None:
                if node.var.name == "tiled_Projected_storage":
                    owners.append(node)
                super().visit_direct_init(node, path)

        StorageOwnerCollector(
            LLIRTraversalContext(
                stage="test",
                pass_name="collect_ttm_heap_storage_owner",
            )
        ).walk(lowered)
        assert len(owners) == 1
        return LLIRLowerer().lower_llir(lowered), owners[0]

    cpp, owner = lower_with_storage_owner()
    second_cpp, second_owner = lower_with_storage_owner()

    allocation = (
        "std::vector<float> tiled_Projected_storage("
        "(size_t)(Projected0_size * Projected1_size) * (size_t)kTile_d);"
    )
    assert owner.var == llir.Var(
        "tiled_Projected_storage",
        llir.DataType.STD_VECTOR_FLOAT32,
    )
    assert len(owner.args) == 1
    extent = cast(llir.Mul, owner.args[0])
    assert type(extent) is llir.Mul
    assert type(extent.left) is llir.Cast
    assert type(extent.right) is llir.Cast
    prefix = cast(llir.Cast, extent.left)
    tile = cast(llir.Cast, extent.right)
    assert prefix.data_type is llir.DataType.SIZE_T
    assert tile.data_type is llir.DataType.SIZE_T
    assert type(prefix.expr) is llir.Mul
    prefix_product = cast(llir.Mul, prefix.expr)
    assert prefix_product.left == llir.Var(
        "Projected0_size",
        llir.DataType.INT64,
    )
    assert prefix_product.right == llir.Var(
        "Projected1_size",
        llir.DataType.INT64,
    )
    assert tile.expr == llir.Var("kTile_d", llir.DataType.CONSTEXPR_INT)
    assert owner == second_owner
    assert hash(owner) == hash(second_owner)
    assert owner is not second_owner
    assert owner.var is not second_owner.var
    second_extent = cast(llir.Mul, second_owner.args[0])
    second_prefix = cast(llir.Cast, second_extent.left)
    second_prefix_product = cast(llir.Mul, second_prefix.expr)
    second_tile = cast(llir.Cast, second_extent.right)
    assert extent is not second_extent
    assert prefix is not second_prefix
    assert prefix_product is not second_prefix_product
    assert prefix_product.left is not second_prefix_product.left
    assert prefix_product.right is not second_prefix_product.right
    assert tile is not second_tile
    assert tile.expr is not second_tile.expr
    assert cpp == second_cpp
    init_loop = "Projected_tile_init < Projected0_size * Projected1_size"
    copy_loop = "Projected_tile_copy < Projected0_size * Projected1_size"
    assert allocation in cpp
    assert cpp.index(allocation) < cpp.index("d_out = 0")
    assert cpp.count(init_loop) == 1
    assert cpp.count(copy_loop) == 1
    assert "tiled_Projected[pProjected1 * kTile_d + d_in] +=" in cpp
    assert "Projected_values[pProjected2] +=" not in cpp
    assert "scorch_zero_dense(Projected_values, Projected_capacity)" not in cpp
    assert "packed_" not in cpp

    torch.manual_seed(105)
    core = torch.randn(5, 3, 4)
    factor = torch.randn(4, 7)
    result = einsum(
        "abc,cd->abd",
        STensor.from_torch(core, "Core"),
        STensor.from_torch(factor, "Factor"),
        format="ddd",
        schedule=schedule,
    )
    reference = torch.einsum("abc,cd->abd", core, factor)

    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        pytest.param(torch.float32, (5, 4, 7), id="float32-ragged"),
        pytest.param(torch.float64, (5, 4, 7), id="float64-ragged"),
        pytest.param(torch.float32, (0, 4, 7), id="zero-rows"),
        pytest.param(torch.float32, (5, 0, 7), id="zero-reduction"),
        pytest.param(torch.float32, (5, 4, 0), id="zero-output-columns"),
    ],
)
def test_spmm_heap_result_tile_without_relayout_matches_torch(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    rows, reduction, columns = shape
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                3,
                placement="outermost",
                accum="heap",
                unroll=False,
            ),
        ),
        tag="generic-spmm-heap-k-no-relayout",
        parallel_loop="i",
    )

    torch.manual_seed(109)
    sparse = torch.randn(rows, reduction, dtype=dtype)
    if sparse.numel():
        sparse[sparse.abs() < 0.5] = 0
    dense = torch.randn(reduction, columns, dtype=dtype)
    result = einsum(
        "ij,jk->ik",
        _sparse_stensor(sparse, "Sparse"),
        STensor.from_torch(dense, "Dense"),
        format="dd",
        schedule=schedule,
    )
    reference = sparse @ dense
    tolerance = 1e-3 if dtype is torch.float32 else 1e-9

    assert torch.allclose(
        result.to_torch(),
        reference,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    ("factory", "schedule", "message"),
    [
        pytest.param(
            lambda: _build_elementwise("dd"),
            Schedule(
                loop_order=("r", "c"),
                tiles=(TileSpec("c", 3, accum="heap", unroll=False),),
            ),
            "requires an enclosed reduction",
            id="elementwise-no-reduction",
        ),
        pytest.param(
            _build_sddmm,
            Schedule(
                loop_order=("r", "c", "q"),
                tiles=(TileSpec("r", 3, accum="heap", unroll=False),),
            ),
            "tiled sparse-output assembly",
            id="sddmm-sparse-result",
        ),
        pytest.param(
            _build_spgemm,
            Schedule(
                loop_order=("r", "q", "c"),
                tiles=(TileSpec("r", 3, accum="heap", unroll=False),),
            ),
            "tiled sparse-output assembly",
            id="spgemm-sparse-result",
        ),
        pytest.param(
            _build_spmv,
            Schedule(
                loop_order=("r", "q"),
                tiles=(TileSpec("r", 3, accum="heap", unroll=False),),
            ),
            "trailing storage level",
            id="spmv-rank-one-result",
        ),
        pytest.param(
            _build_ttm,
            Schedule(
                loop_order=("a", "b", "c", "d"),
                tiles=(
                    TileSpec(
                        "d",
                        3,
                        placement="outermost",
                        accum="heap",
                    ),
                ),
            ),
            "requires an explicit parallel dense result-prefix loop",
            id="ttm-heap-missing-safe-parallel-anchor",
        ),
        pytest.param(
            _build_ttm,
            Schedule(
                loop_order=("a", "b", "c", "d"),
                tiles=(
                    TileSpec(
                        "d",
                        3,
                        placement="outermost",
                        accum="heap",
                    ),
                    TileSpec(
                        "a",
                        2,
                        placement="outermost",
                        accum="direct",
                    ),
                ),
                parallel_loop="a",
            ),
            "requires.*remain outermost",
            id="ttm-later-affine-tile-wraps-heap-tile",
        ),
        pytest.param(
            _build_ttm,
            Schedule(
                loop_order=("a", "b", "c", "d"),
                tiles=(
                    TileSpec(
                        "d",
                        3,
                        placement="child_of:b",
                        accum="heap",
                    ),
                ),
            ),
            "requires.*outermost",
            id="ttm-unsafe-lifetime",
        ),
    ],
)
def test_heap_result_tile_rejects_unsupported_non_spmm_structures_before_lowering(
    factory, schedule, message
):
    stmt = factory()
    original = str(stmt)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        Scheduler.apply_schedule(stmt, schedule)
    assert str(stmt) == original


def test_sparse_elementwise_affine_tile_is_rejected_before_lowering():
    stmt = _build_elementwise("ds")
    original = str(stmt)
    schedule = Schedule(
        loop_order=("r", "c"),
        tiles=(TileSpec("r", 3, accum="direct", unroll=False),),
        tag="unsupported-sparse-elementwise-row-tile",
    )

    with pytest.raises(NotImplementedError, match="tiled sparse-output assembly"):
        Scheduler.apply_schedule(stmt, schedule)

    assert str(stmt) == original


def test_sddmm_default_scalar_accumulator_and_simd_are_unchanged():
    with regblock_force(False):
        scheduled = Scheduler.auto_schedule(_build_sddmm())
        cpp = _lower_to_cpp(scheduled)

    assert not scheduled.inserted_workspace
    assert "float _accum = 0.0f;" in cpp
    assert "#pragma omp simd" in cpp
    assert "wksp" not in cpp
    assert "packed_" not in cpp
    assert "int64_t _known_nnz = Mask_values.size(0);" in cpp
    assert "torch::Tensor Sampled0_crd_torch = torch::empty" in cpp
    assert (
        "torch::Tensor Sampled_values_torch = "
        "torch::empty({_known_nnz}, torch::kFloat32);"
    ) in cpp
    assert "std::vector<float> Sampled_values" not in cpp

    torch.manual_seed(102)
    mask = torch.randn(5, 7)
    mask *= torch.rand(5, 7) < 0.35
    query = torch.randn(5, 3)
    key = torch.randn(7, 3)
    result = einsum(
        "rc,rq,cq->rc",
        _sparse_stensor(mask, "Mask"),
        STensor.from_torch(query, "Query"),
        STensor.from_torch(key, "Key"),
        format="oo",
    )
    reference = mask * (query @ key.T)

    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)


def test_all_coo_sddmm_has_no_redundant_end_and_remains_correct():
    row, column, reduction = IndexVar("i"), IndexVar("j"), IndexVar("k")
    out = TensorVar("Sampled", fmt="oo")
    mask_var = TensorVar("Mask", fmt="oo")
    query_var = TensorVar("Query", fmt="dd")
    key_var = TensorVar("Key", fmt="dd")
    assignment = TensorAssign(
        out[row, column],
        mask_var[row, column] * query_var[row, reduction] * key_var[column, reduction],
        op=Operation.ADD,
    )

    with regblock_force(False):
        scheduled = Scheduler.auto_schedule(_nest((row, column, reduction), assignment))
        cpp = _lower_to_cpp(scheduled)

    assert "pMask1_end" not in cpp
    assert "int64_t _known_nnz = Mask_values.size(0);" in cpp
    assert "int64_t i = Mask0_crd[pMask0];" in cpp
    assert "Mask1_crd[pMask0]" in cpp
    assert "Mask_val[pMask0]" in cpp

    torch.manual_seed(103)
    mask = torch.randn(5, 7)
    mask *= torch.rand(5, 7) < 0.35
    query = torch.randn(5, 3)
    key = torch.randn(7, 3)
    result = einsum(
        "ij,ik,jk->ij",
        _sparse_stensor(mask, "Mask", "oo"),
        STensor.from_torch(query, "Query"),
        STensor.from_torch(key, "Key"),
        format="oo",
        schedule=Schedule(
            loop_order=("i", "j", "k"),
            tag="all-coo-pmask-end-native",
        ),
    )
    reference = mask * (query @ key.T)

    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)


def test_sddmm_affine_reduction_tile_is_rejected_during_validation():
    schedule = Schedule(
        loop_order=("r", "c", "q"),
        tiles=(
            TileSpec(
                "q",
                2,
                placement="child_of:c",
                accum="direct",
            ),
        ),
        tag="unsupported-sddmm-reduction-tile",
    )

    with pytest.raises(NotImplementedError, match="Affine reduction tiling"):
        Scheduler.apply_schedule(_build_sddmm(), schedule)


def test_spgemm_default_workspace_and_sparse_assembly_are_unchanged():
    with regblock_force(False):
        auto_scheduled = Scheduler.auto_schedule(_build_spgemm())
        empty_scheduled = Scheduler.apply_schedule(_build_spgemm(), Schedule())
        auto_cpp = _lower_to_cpp(auto_scheduled)
        empty_cpp = _lower_to_cpp(empty_scheduled)

    _, auto_body = Scheduler._extract_loop_chain(auto_scheduled)
    assert auto_scheduled.inserted_workspace
    assert isinstance(auto_body, Where)
    empty_working = legacy_cin_working_copy(
        empty_scheduled.normalized_cin,
        empty_scheduled.verified_loop_plan,
    )
    assert str(empty_working) == str(auto_scheduled)
    assert empty_cpp == auto_cpp
    assert "std::vector<linked_list_workspace_1d" in auto_cpp
    assert auto_cpp.count("].make_view()") == 2
    assert ", true);" in auto_cpp
    assert "std::current_exception()" not in auto_cpp
    assert "for (int _worker = 0" in auto_cpp
    assert ".insert_unchecked(" in auto_cpp
    assert "torch::Tensor SparseProduct1_crd_torch = torch::empty" in auto_cpp
    assert "SparseProduct1_pos_data" in auto_cpp
    assert auto_cpp.count("int64_t _base1 = _offset1[r];") == 1
    assert "packed_" not in auto_cpp
    assert auto_cpp.count("int64_t c = it.first;") == 2
    assert auto_cpp.count("float wksp_value = it.second;") == 2

    torch.manual_seed(103)
    left = torch.randn(5, 6)
    right = torch.randn(6, 4)
    left *= torch.rand(5, 6) < 0.35
    right *= torch.rand(6, 4) < 0.4
    result = einsum(
        "rq,qc->rc",
        _sparse_stensor(left, "SparseLeft"),
        _sparse_stensor(right, "SparseRight"),
        format="ds",
    )

    assert torch.allclose(result.to_torch(), left @ right, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    ("result_format", "expected_pair_reads", "expected_dtype_conversions"),
    [
        pytest.param(
            "oo",
            (
                "SparseProduct0_crd.emplace_back(it.first[0]);",
                "SparseProduct1_crd.emplace_back(it.first[1]);",
                "SparseProduct_values.emplace_back(it.second);",
            ),
            (
                "scorch_tensor_from_vector(std::move(SparseProduct0_crd), "
                "torch::kInt);",
                "scorch_tensor_from_vector(std::move(SparseProduct1_crd), "
                "torch::kInt);",
                "scorch_tensor_from_vector(std::move(SparseProduct_values), "
                "torch::kFloat32);",
            ),
            id="all-coordinate",
        ),
        pytest.param(
            "ds",
            (
                "scorch_vector_set(T0_crd_vec, pT, it.first[0]);",
                "scorch_vector_set(T1_crd_vec, pT, it.first[1]);",
                "scorch_vector_set(T_val_vec, pT, it.second);",
            ),
            (
                "scorch_tensor_from_vector(std::move(T0_crd_vec), torch::kInt);",
                "scorch_tensor_from_vector(std::move(T1_crd_vec), torch::kInt);",
                "scorch_tensor_from_vector(std::move(T_val_vec), torch::kFloat32);",
            ),
            id="intermediate-coordinate",
        ),
    ],
)
def test_outer_workspace_pair_reads_are_stable_and_remain_correct(
    result_format: str,
    expected_pair_reads: tuple[str, ...],
    expected_dtype_conversions: tuple[str, ...],
) -> None:
    schedule = Schedule(
        loop_order=("q", "r", "c"),
        tag=f"workspace-pair-outer-{result_format}",
    )
    first_statement = _build_outer_workspace_spgemm(result_format)
    second_statement = _build_outer_workspace_spgemm(result_format)
    first_original = str(first_statement)
    second_original = str(second_statement)

    with regblock_force(False):
        first_scheduled = Scheduler.apply_schedule(first_statement, schedule)
        second_scheduled = Scheduler.apply_schedule(second_statement, schedule)
        first_cpp = _lower_to_cpp(first_scheduled)
        second_cpp = _lower_to_cpp(second_scheduled)

    assert first_scheduled is not second_scheduled
    assert str(first_statement) == first_original
    assert str(second_statement) == second_original
    assert second_cpp == first_cpp
    for anchor in (*expected_pair_reads, *expected_dtype_conversions):
        assert anchor in first_cpp

    torch.manual_seed(104)
    left = torch.randn(4, 5)
    right = torch.randn(5, 3)
    left *= torch.rand(4, 5) < 0.45
    right *= torch.rand(5, 3) < 0.5
    sparse_left = STensor.from_torch(
        left,
        "WorkspacePairLeft",
        mode_order=[1, 0],
    ).to_sparse("ds")
    sparse_right = _sparse_stensor(right, "WorkspacePairRight")
    result = einsum(
        "rq,qc->rc",
        sparse_left,
        sparse_right,
        format=result_format,
        schedule=schedule,
    )

    assert torch.allclose(result.to_torch(), left @ right, atol=1e-3, rtol=1e-3)


def test_nested_rank_two_workspace_pair_reads_are_stable() -> None:
    schedule = Schedule(
        loop_order=("a", "q", "r", "c"),
        tag="workspace-pair-nested-rank-two",
    )
    statement = _build_nested_rank_two_workspace()
    original = str(statement)

    with regblock_force(False):
        first_cpp = _lower_to_cpp(Scheduler.apply_schedule(statement, schedule))
        second_cpp = _lower_to_cpp(Scheduler.apply_schedule(statement, schedule))

    assert str(statement) == original
    assert second_cpp == first_cpp
    assert "int64_t r = it.first[0];" in first_cpp
    assert "int64_t c = it.first[1];" in first_cpp
    assert "float wksp_value = it.second;" in first_cpp


def test_spgemm_affine_tile_is_rejected_before_sparse_output_assembly():
    schedule = Schedule(
        loop_order=("r", "q", "c"),
        tiles=(TileSpec("r", 2, accum="direct", unroll=False),),
        tag="unsupported-spgemm-row-tile",
    )

    with pytest.raises(NotImplementedError, match="tiled sparse-output assembly"):
        Scheduler.apply_schedule(_build_spgemm(), schedule)


@pytest.mark.parametrize(
    ("factory", "loop_order"),
    [
        pytest.param(_build_spmv, ("r", "q"), id="spmv"),
        pytest.param(_build_dense_matmul, ("r", "q", "c"), id="dense-matmul"),
    ],
)
def test_spmv_and_dense_matmul_empty_schedule_preserve_default_codegen(
    factory, loop_order
):
    with regblock_force(False):
        auto_scheduled = Scheduler.auto_schedule(factory())
        empty_scheduled = Scheduler.apply_schedule(factory(), Schedule())
        auto_cpp = _lower_to_cpp(auto_scheduled)
        empty_cpp = _lower_to_cpp(empty_scheduled)

    empty_working = legacy_cin_working_copy(
        empty_scheduled.normalized_cin,
        empty_scheduled.verified_loop_plan,
    )
    assert str(empty_working) == str(auto_scheduled)
    assert empty_cpp == auto_cpp
    assert "packed_" not in auto_cpp


@pytest.mark.parametrize(
    ("factory", "loop_order"),
    [
        pytest.param(_build_spmv, ("r", "q"), id="spmv"),
        pytest.param(_build_dense_matmul, ("r", "q", "c"), id="dense-matmul"),
    ],
)
def test_spmv_and_dense_matmul_reduction_tiles_are_rejected(factory, loop_order):
    schedule = Schedule(
        loop_order=loop_order,
        tiles=(TileSpec("q", 2, accum="direct"),),
        tag="unsupported-reduction-tile",
    )

    with pytest.raises(NotImplementedError, match="Affine reduction tiling"):
        Scheduler.apply_schedule(factory(), schedule)


def test_spmv_and_dense_matmul_default_numerics_are_unchanged():
    torch.manual_seed(104)

    sparse_matrix = torch.randn(7, 5)
    sparse_matrix *= torch.rand(7, 5) < 0.35
    vector = torch.randn(5)
    spmv_result = einsum(
        "rq,q->r",
        _sparse_stensor(sparse_matrix, "SparseMatrix"),
        STensor.from_torch(vector, "DenseVector"),
        format="d",
    )
    assert torch.allclose(
        spmv_result.to_torch(), sparse_matrix @ vector, atol=1e-3, rtol=1e-3
    )

    left = torch.randn(5, 3)
    right = torch.randn(3, 6)
    dense_result = einsum(
        "rq,qc->rc",
        STensor.from_torch(left, "DenseLeft"),
        STensor.from_torch(right, "DenseRight"),
        format="dd",
    )
    assert torch.allclose(dense_result.to_torch(), left @ right, atol=1e-3, rtol=1e-3)
