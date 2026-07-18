from typing import List, cast

import pytest

from scorch.compiler import llir
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.dynamic_vector_access_pass import (
    DYNAMIC_VECTOR_ACCESS_CONTEXT,
    rewrite_dynamic_vector_accesses,
)
from scorch.compiler.llir_traversal import (
    LLIRStatementValue,
    LLIRTraversalError,
)


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _access(
    array: str,
    index: str | llir.Expr,
    data_type: llir.DataType = llir.DataType.NO_TYPE,
) -> llir.ArrayAccess:
    index_expr = _var(index) if type(index) is str else index
    return llir.ArrayAccess(array=_var(array, data_type), index=index_expr)


def _legacy_dynamic_vector_fixture() -> List[llir.Stmt]:
    coordinate_store = llir.Assign(
        var=_access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
        value=_var("out_pos[q]"),
    )
    return [
        llir.VarDecl(_var("out_pos", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        llir.VarInit(
            _var("scratch", llir.DataType.STD_VECTOR_FLOAT32),
            _var("std::vector<float>(4)"),
        ),
        coordinate_store,
        llir.Assign(
            var=_access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
            value=_var("out_pos[q]"),
        ),
        llir.Assign(
            var=_access("out_values", "p", llir.DataType.STD_VECTOR_FLOAT32),
            value=_var("out_crd[q]"),
        ),
        llir.Assign(
            var=_access(
                "out_pos",
                llir.Add(_var("p"), llir.Literal(1)),
                llir.DataType.STD_VECTOR_C_INT,
            ),
            value=_var("out_values[q]"),
        ),
        llir.VarInit(_var("read", llir.DataType.INT), _var("out_pos[p]")),
        llir.Assign(
            var=_access("scratch", "i", llir.DataType.STD_VECTOR_FLOAT32),
            value=_var("out_values[q]"),
        ),
    ]


def _array_access_parts(access: llir.AssignmentTarget) -> tuple[str, str]:
    assert type(access) is llir.ArrayAccess
    assert type(access.array) is llir.Var
    assert type(access.index) is llir.Var
    return cast(llir.Var, access.array).name, cast(llir.Var, access.index).name


def _cpp(statements: List[llir.Stmt]) -> str:
    return LLIRLowerer().lower_llir(statements)


def _structural_snapshot(value: object) -> object:
    if isinstance(value, llir.Node):
        return (
            type(value).__name__,
            tuple(
                (name, _structural_snapshot(child))
                for name, child in sorted(vars(value).items())
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_structural_snapshot(child) for child in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_structural_snapshot(child) for child in value))
    return value


def test_dynamic_vector_pass_matches_legacy_transformation_structurally() -> None:
    source = _legacy_dynamic_vector_fixture()
    source_cpp = _cpp(source)

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert _cpp(source) == source_cpp
    assert rewritten is not source
    assert len(source) == 10
    assert len(rewritten) == 9
    assert [type(statement) for statement in rewritten] == [
        llir.VarDecl,
        llir.VarDecl,
        llir.VarDecl,
        llir.VarInit,
        llir.FunctionCallStmt,
        llir.FunctionCallStmt,
        llir.FunctionCallStmt,
        llir.VarInit,
        llir.Assign,
    ]

    coordinate_append = cast(llir.FunctionCallStmt, rewritten[4])
    assert coordinate_append.name == "out_crd.emplace_back"
    assert cast(llir.Var, coordinate_append.args[0]).name == "out_pos.at(q)"

    value_append = cast(llir.FunctionCallStmt, rewritten[5])
    assert value_append.name == "out_values.emplace_back"
    assert cast(llir.Var, value_append.args[0]).name == "out_crd.at(q)"

    position_store = cast(llir.FunctionCallStmt, rewritten[6])
    assert position_store.name == "scorch_vector_set"
    assert cast(llir.Var, position_store.args[0]).name == "out_pos"
    position = cast(llir.Add, position_store.args[1])
    assert cast(llir.Var, position.left).name == "p"
    assert cast(llir.Literal, position.right).value == 1
    assert cast(llir.Var, position_store.args[2]).name == "out_values.at(q)"

    pre_sized_store = cast(llir.Assign, rewritten[8])
    assert _array_access_parts(pre_sized_store.var) == ("scratch", "i")
    assert cast(llir.Var, pre_sized_store.value).name == "out_values.at(q)"

    expected_cpp = """std::vector<int> out_pos;
std::vector<int> out_crd;
std::vector<float> out_values;
std::vector<float> scratch = std::vector<float>(4);
out_crd.emplace_back(out_pos.at(q));
out_values.emplace_back(out_crd.at(q));
scorch_vector_set(out_pos, p + 1, out_values.at(q));
int read = out_pos.at(p);
scratch[i] = out_values.at(q);"""
    assert _cpp(rewritten) == expected_cpp


def test_structural_function_call_equality_is_explicit_at_deduplication_seam() -> None:
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        llir.Assign(
            _access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
            llir.FunctionCall("coordinate", [_var("p")]),
        ),
        llir.Assign(
            _access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
            llir.FunctionCall("coordinate", [_var("p")]),
        ),
    ]
    snapshot = _structural_snapshot(source)

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert len(rewritten) == 2
    assert type(rewritten[1]) is llir.FunctionCallStmt
    assert _cpp(rewritten) == (
        "std::vector<int> out_crd;\nout_crd.emplace_back(coordinate(p));"
    )
    assert _structural_snapshot(source) == snapshot


def test_fixed_stack_array_preserves_coordinate_assignment_deduplication() -> None:
    declaration = llir.FixedStackArrayDecl(
        name="wksp",
        element_type=llir.DataType.FLOAT32,
        extent=_var("kTile_k", llir.DataType.CONSTEXPR_INT),
        initializer=llir.Array([], llir.DataType.FLOAT32),
    )
    assignments = [
        llir.Assign(
            _access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
            llir.FunctionCall("coordinate", [_var("p")]),
        )
        for _ in range(2)
    ]
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        declaration,
        *assignments,
    ]
    source_snapshot = _structural_snapshot(source)

    once = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    twice = rewrite_dynamic_vector_accesses(once, DYNAMIC_VECTOR_ACCESS_CONTEXT)

    assert _structural_snapshot(source) == source_snapshot
    assert _structural_snapshot(once) == _structural_snapshot(twice)
    assert [type(statement) for statement in once] == [
        llir.VarDecl,
        llir.FixedStackArrayDecl,
        llir.FunctionCallStmt,
    ]
    once_decl = cast(llir.FixedStackArrayDecl, once[1])
    twice_decl = cast(llir.FixedStackArrayDecl, twice[1])
    assert once_decl == declaration == twice_decl
    assert once_decl is not declaration
    assert once_decl.extent is not declaration.extent
    assert once_decl.initializer is not declaration.initializer
    assert twice_decl is not once_decl
    assert twice_decl.extent is not once_decl.extent
    assert twice_decl.initializer is not once_decl.initializer
    assert _cpp(once) == _cpp(twice)
    assert _cpp(once) == (
        "std::vector<int> out_crd;\n"
        "float wksp[kTile_k] = {};\n"
        "out_crd.emplace_back(coordinate(p));"
    )


def test_phase_state_assignments_are_detached_and_never_deduplicated() -> None:
    state_assignments = [
        llir.Assign(
            _var("_prev2", llir.DataType.INT),
            _var("_cnt2", llir.DataType.INT),
        )
        for _ in range(2)
    ]
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        *state_assignments,
    ]
    snapshot = _structural_snapshot(source)

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert _structural_snapshot(source) == snapshot
    assert len(rewritten) == 3
    first = cast(llir.Assign, rewritten[1])
    second = cast(llir.Assign, rewritten[2])
    assert first == second
    assert first is not state_assignments[0]
    assert second is not state_assignments[1]
    assert first.var is not state_assignments[0].var
    assert second.var is not state_assignments[1].var
    assert first.var is not second.var
    assert first.value is not second.value
    assert _cpp(rewritten) == (
        "std::vector<int> out_crd;\n" "_prev2 = _cnt2;\n" "_prev2 = _cnt2;"
    )


def test_factor_materialization_is_detached_and_never_deduplicated() -> None:
    declarations = [
        llir.VarInit(
            _var("_inv_0", llir.DataType.FLOAT32),
            _access("Mask_val", "pMask0", llir.DataType.PTR_FLOAT32),
        )
        for _ in range(2)
    ]
    multiplications = [
        llir.Assign(
            _var("acc", llir.DataType.FLOAT32),
            _var("_inv_0", llir.DataType.FLOAT32),
            llir.AssignOp.MUL_ASSIGN,
        )
        for _ in range(2)
    ]
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        declarations[0],
        multiplications[0],
        declarations[1],
        multiplications[1],
    ]
    snapshot = _structural_snapshot(source)

    first = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    second = rewrite_dynamic_vector_accesses(first, DYNAMIC_VECTOR_ACCESS_CONTEXT)

    assert _structural_snapshot(source) == snapshot
    assert _structural_snapshot(first) == _structural_snapshot(second)
    assert [type(statement) for statement in first] == [
        llir.VarDecl,
        llir.VarInit,
        llir.Assign,
        llir.VarInit,
        llir.Assign,
    ]
    assert first[1] == first[3]
    assert first[2] == first[4]
    assert first[1] is not declarations[0]
    assert first[2] is not multiplications[0]
    assert cast(llir.VarInit, first[1]).var is not declarations[0].var
    assert cast(llir.Assign, first[2]).value is not multiplications[0].value
    assert _cpp(first) == (
        "std::vector<int> out_crd;\n"
        "float _inv_0 = Mask_val[pMask0];\n"
        "acc *= _inv_0;\n"
        "float _inv_0 = Mask_val[pMask0];\n"
        "acc *= _inv_0;"
    )


def test_fill_base_offset_loads_are_detached_and_never_deduplicated() -> None:
    def base_load() -> llir.VarInit:
        return llir.VarInit(
            _var("_base1", llir.DataType.INT64),
            llir.ArrayAccess(
                _var("_offset1", llir.DataType.STD_VECTOR_INT),
                _var("row", llir.DataType.INT64),
            ),
        )

    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        base_load(),
        base_load(),
    ]
    snapshot = _structural_snapshot(source)

    first = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    second = rewrite_dynamic_vector_accesses(first, DYNAMIC_VECTOR_ACCESS_CONTEXT)

    assert _structural_snapshot(source) == snapshot
    assert _structural_snapshot(first) == _structural_snapshot(second)
    assert [type(statement) for statement in first] == [
        llir.VarDecl,
        llir.VarInit,
        llir.VarInit,
    ]
    assert first[1] == first[2]
    assert first[1] is not source[1]
    assert first[2] is not source[2]
    first_access = cast(llir.ArrayAccess, cast(llir.VarInit, first[1]).value)
    second_access = cast(llir.ArrayAccess, cast(llir.VarInit, second[1]).value)
    source_access = cast(llir.ArrayAccess, cast(llir.VarInit, source[1]).value)
    assert first_access is not source_access
    assert second_access is not first_access
    assert first_access.array is not source_access.array
    assert first_access.index is not source_access.index
    assert second_access.array is not first_access.array
    assert second_access.index is not first_access.index
    assert _cpp(first) == (
        "std::vector<int> out_crd;\n"
        "int64_t _base1 = _offset1[row];\n"
        "int64_t _base1 = _offset1[row];"
    )


def _independent_single_step_add() -> llir.Add:
    return llir.Add(
        _var("base", llir.DataType.INT64),
        llir.Literal(1, llir.DataType.INT64),
    )


def test_structural_add_equality_deduplicates_only_coordinate_appends() -> None:
    coordinate_values = [_independent_single_step_add() for _ in range(2)]
    position_values = [_independent_single_step_add() for _ in range(2)]
    non_coordinate_values = [_independent_single_step_add() for _ in range(2)]
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarDecl(_var("out_pos", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        *[
            llir.Assign(
                _access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
                value,
            )
            for value in coordinate_values
        ],
        *[
            llir.Assign(
                _access("out_pos", "p", llir.DataType.STD_VECTOR_C_INT),
                value,
            )
            for value in position_values
        ],
        *[
            llir.Assign(
                _access("out_values", "p", llir.DataType.STD_VECTOR_FLOAT32),
                value,
            )
            for value in non_coordinate_values
        ],
    ]
    snapshot = _structural_snapshot(source)

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert _structural_snapshot(source) == snapshot
    calls = [
        cast(llir.FunctionCallStmt, statement)
        for statement in rewritten
        if type(statement) is llir.FunctionCallStmt
    ]
    assert [call.name for call in calls] == [
        "out_crd.emplace_back",
        "scorch_vector_set",
        "scorch_vector_set",
        "out_values.emplace_back",
        "out_values.emplace_back",
    ]
    rewritten_values = [
        cast(
            llir.Add, call.args[2] if call.name == "scorch_vector_set" else call.args[0]
        )
        for call in calls
    ]
    assert all(type(value) is llir.Add for value in rewritten_values)
    assert all(value == _independent_single_step_add() for value in rewritten_values)
    for rewritten_value in rewritten_values:
        assert all(
            rewritten_value is not source_value
            for source_value in (
                *coordinate_values,
                *position_values,
                *non_coordinate_values,
            )
        )
    assert _cpp(rewritten).count("out_crd.emplace_back(base + 1);") == 1
    assert _cpp(rewritten).count("scorch_vector_set(out_pos, p, base + 1);") == 2
    assert _cpp(rewritten).count("out_values.emplace_back(base + 1);") == 2


def test_binary_exact_class_difference_prevents_coordinate_deduplication() -> None:
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        llir.Assign(
            _access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
            llir.BinOp(
                "+",
                _var("base", llir.DataType.INT64),
                llir.Literal(1, llir.DataType.INT64),
            ),
        ),
        llir.Assign(
            _access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
            _independent_single_step_add(),
        ),
    ]

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    calls = [cast(llir.FunctionCallStmt, statement) for statement in rewritten[1:]]
    assert [call.name for call in calls] == [
        "out_crd.emplace_back",
        "out_crd.emplace_back",
    ]
    assert type(calls[0].args[0]) is llir.BinOp
    assert type(calls[1].args[0]) is llir.Add
    assert _cpp(rewritten).count("out_crd.emplace_back(base + 1);") == 2


def test_dynamic_vector_pass_does_not_mutate_or_alias_caller_input() -> None:
    source = _legacy_dynamic_vector_fixture()
    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    rewritten_decl = cast(llir.VarDecl, rewritten[0])
    rewritten_decl.var.name = "changed"
    rewritten_call = cast(llir.FunctionCallStmt, rewritten[4])
    cast(llir.Var, rewritten_call.args[0]).name = "changed_read"
    rewritten.append(llir.Break())

    assert cast(llir.VarDecl, source[0]).var.name == "out_pos"
    assert _array_access_parts(cast(llir.Assign, source[4]).var) == (
        "out_crd",
        "p",
    )
    assert cast(llir.Assign, source[4]).value.name == "out_pos[q]"
    assert len(source) == 10


def test_dynamic_vector_pass_is_idempotent_for_generated_access_shapes() -> None:
    source = _legacy_dynamic_vector_fixture()
    once = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    twice = rewrite_dynamic_vector_accesses(once, DYNAMIC_VECTOR_ACCESS_CONTEXT)

    assert _cpp(once) == _cpp(twice)
    assert [type(statement) for statement in once] == [
        type(statement) for statement in twice
    ]
    assert _structural_snapshot(once) == _structural_snapshot(twice)


def test_dynamic_vector_pass_preserves_compound_store_and_loop_update_shape() -> None:
    loop = llir.ForLoop(
        init=llir.VarInit(_var("i", llir.DataType.INT), llir.Literal(0)),
        cond=llir.BinOp("<", _var("i"), _var("n")),
        update=llir.Assign(
            _access("out_pos", "i", llir.DataType.STD_VECTOR_C_INT),
            _var("out_values[i]"),
            op=llir.AssignOp.ADD_ASSIGN,
        ),
        body=[],
    )
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_pos", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        loop,
    ]

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )
    rewritten_loop = cast(llir.ForLoop, rewritten[2])
    update = cast(llir.Assign, rewritten_loop.update)

    assert type(update) is llir.Assign
    assert update.op == llir.AssignOp.ADD_ASSIGN
    assert _array_access_parts(update.var) == ("out_pos", "i")
    assert cast(llir.Var, update.value).name == "out_values.at(i)"


def test_dynamic_vector_pass_no_vector_declaration_is_detached_no_op() -> None:
    source: List[llir.Stmt] = [
        llir.Assign(
            _access("scratch", "i", llir.DataType.PTR_FLOAT32),
            _var("value"),
        )
    ]
    snapshot = _structural_snapshot(source)

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert rewritten is not source
    assert _structural_snapshot(rewritten) == snapshot
    rewritten_store = cast(llir.Assign, rewritten[0])
    rewritten_access = cast(llir.ArrayAccess, rewritten_store.var)
    cast(llir.Var, rewritten_access.array).name = "changed"
    assert _array_access_parts(cast(llir.Assign, source[0]).var) == (
        "scratch",
        "i",
    )


def test_dynamic_vector_pass_detaches_without_rewriting_qualified_dtype() -> None:
    source: List[llir.Stmt] = [
        llir.VarInit(
            _var("tensor", llir.DataType.TORCH_TENSOR),
            llir.FunctionCall(
                "scorch_tensor_from_vector",
                (
                    llir.FunctionCall("std::move", (_var("values"),)),
                    llir.QualifiedName(
                        "torch",
                        "kFloat32",
                        llir.DataType.TORCH_SCALAR_TYPE,
                    ),
                ),
            ),
        )
    ]

    first = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )
    second = rewrite_dynamic_vector_accesses(
        first,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    def dtype_leaf(statements: List[llir.Stmt]) -> llir.QualifiedName:
        initializer = cast(llir.VarInit, statements[0])
        call = cast(llir.FunctionCall, initializer.value)
        return cast(llir.QualifiedName, call.args[1])

    original_dtype = dtype_leaf(source)
    first_dtype = dtype_leaf(first)
    second_dtype = dtype_leaf(second)
    assert original_dtype == first_dtype == second_dtype
    assert original_dtype is not first_dtype
    assert first_dtype is not second_dtype
    assert _cpp(first) == (
        "torch::Tensor tensor = "
        "scorch_tensor_from_vector(std::move(values), torch::kFloat32);"
    )


def test_dynamic_vector_pass_detaches_and_preserves_shape_extent_reads() -> None:
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        llir.VarInit(
            _var("Input0_size", llir.DataType.INT64),
            llir.ArrayAccess(
                array=_var("Input_shape", llir.DataType.STD_VECTOR_INT),
                index=llir.Literal(0, data_type=llir.DataType.INT64),
            ),
        ),
    ]
    source_snapshot = _structural_snapshot(source)

    once = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    twice = rewrite_dynamic_vector_accesses(once, DYNAMIC_VECTOR_ACCESS_CONTEXT)

    assert _structural_snapshot(source) == source_snapshot
    assert _structural_snapshot(once) == _structural_snapshot(twice)
    assert _cpp(once) == _cpp(twice)
    assert _cpp(once) == (
        "std::vector<float> out_values;\n" "int64_t Input0_size = Input_shape[0];"
    )
    first_access = cast(llir.ArrayAccess, cast(llir.VarInit, once[1]).value)
    repeated_access = cast(llir.ArrayAccess, cast(llir.VarInit, twice[1]).value)
    source_access = cast(llir.ArrayAccess, cast(llir.VarInit, source[1]).value)
    assert first_access is not source_access
    assert repeated_access is not first_access
    assert first_access.array is not source_access.array
    assert first_access.index is not source_access.index
    cast(llir.Var, first_access.array).name = "changed"
    assert cast(llir.Var, source_access.array).name == "Input_shape"
    assert cast(llir.Var, repeated_access.array).name == "Input_shape"


def test_pass_preserves_nested_list_and_tuple_statement_containers() -> None:
    nested: List[LLIRStatementValue] = [
        llir.VarDecl(_var("values", llir.DataType.STD_VECTOR_FLOAT32)),
        ([llir.VarInit(_var("read"), _var("values[i]"))],),
    ]

    rewritten = rewrite_dynamic_vector_accesses(
        nested,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )
    assert type(rewritten) is list
    assert type(rewritten[1]) is tuple
    tuple_body = cast(tuple, rewritten[1])
    assert type(tuple_body[0]) is list
    initializer = cast(llir.VarInit, cast(list, tuple_body[0])[0])
    assert cast(llir.Var, initializer.value).name == "values.at(i)"


def test_pass_accepts_a_scalar_function_root() -> None:
    function = llir.Function(
        return_type=llir.DataType.VOID,
        name="function",
        args=[],
        body=[
            llir.VarDecl(_var("values", llir.DataType.STD_VECTOR_FLOAT32)),
            llir.VarInit(_var("read"), _var("values[i]")),
        ],
    )

    rewritten = rewrite_dynamic_vector_accesses(
        function,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert type(rewritten) is llir.Function
    assert rewritten is not function
    initializer = cast(llir.VarInit, rewritten.body[1])
    assert cast(llir.Var, initializer.value).name == "values.at(i)"


def test_pass_unknown_node_reports_its_own_stage_and_name() -> None:
    class UnknownBreak(llir.Break):
        pass

    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_dynamic_vector_accesses(
            UnknownBreak(),
            DYNAMIC_VECTOR_ACCESS_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert diagnostic.node_type == "UnknownBreak"


def test_malformed_vector_declaration_fails_through_structured_diagnostic() -> None:
    class UnknownExpr(llir.Expr):
        pass

    declaration = llir.VarDecl(_var("values", llir.DataType.STD_VECTOR_FLOAT32))
    declaration.var = cast(llir.Var, UnknownExpr())

    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_dynamic_vector_accesses(
            [declaration],
            DYNAMIC_VECTOR_ACCESS_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert diagnostic.node_type == "UnknownExpr"
    assert diagnostic.path == ("root", "[0]", "var")
