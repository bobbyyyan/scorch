"""Audited compatibility budget for the current Phase-3 structured IR."""

import ast
from collections import Counter
from pathlib import Path
import re
from typing import Optional

_REPOSITORY_ROOT = Path(__file__).parents[2]
_COMPILER_ROOT = _REPOSITORY_ROOT / "src" / "scorch" / "compiler"


def _llir_constructor_calls(path: Path, constructor: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == constructor
    ]


def _function_calls(path: Path, function: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function
    ]


def _var_name_expression(call: ast.Call) -> Optional[ast.expr]:
    for keyword in call.keywords:
        if keyword.arg == "name":
            return keyword.value
    return call.args[0] if call.args else None


def _assign_target_expression(call: ast.Call) -> Optional[ast.expr]:
    for keyword in call.keywords:
        if keyword.arg == "var":
            return keyword.value
    return call.args[0] if call.args else None


def _is_llir_constructor(expression: ast.expr, constructor: str) -> bool:
    return bool(
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == constructor
    )


def _static_string_fragments(expression: ast.expr) -> str:
    return "".join(
        node.value
        for node in ast.walk(expression)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _string_expression_category(expression: ast.expr) -> Optional[str]:
    fragments = _static_string_fragments(expression)
    if "[" in fragments:
        return "subscript"
    if "?" in fragments:
        return "ternary"
    if "{" in fragments and "}" in fragments:
        return "initializer"
    if "(" in fragments:
        return "call"
    if "::" in fragments:
        return "qualified"
    if "." in fragments:
        return "member"
    if any(operator in fragments for operator in (" + ", " - ", " * ", " / ")):
        return "arithmetic"
    return None


def test_direct_string_encoded_var_expression_budget_is_explicit() -> None:
    """Lock every Var sink and classify direct and known indirect expressions."""

    per_file: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    constructor_counts: Counter[str] = Counter()
    unclassified_counts: Counter[str] = Counter()
    known_indirect: Counter[tuple[str, str]] = Counter()
    known_indirect_names = {
        "accumulator_name",
        "actual_size",
        "bound_name",
        "bound.name",
        "candidate.base",
        "candidate.stride",
        "expr.name.replace(old, new)",
        "f'{prefix}{level}'",
        "invariant_name",
        "loop_bound",
        "name",
        "node.name",
        "outer_end_bound.name",
        "outer_end_var",
        "pointer_name",
        "prefix_extent",
        "reserve_hint_var",
        "size_var",
        "sparse_pos",
        "sparse_values_tensor",
        "spec.extent",
        "value_array",
        "wname",
        "zero_value",
    }
    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for call in _llir_constructor_calls(path, "Var"):
            constructor_counts[path.name] += 1
            name_expression = _var_name_expression(call)
            if name_expression is None:
                continue
            category = _string_expression_category(name_expression)
            if category is not None:
                totals[category] += 1
                per_file[(path.name, category)] += 1
            else:
                unclassified_counts[path.name] += 1
                spelling = ast.unparse(name_expression)
                if spelling in known_indirect_names or (
                    path.name == "compressed_where_openmp_pass.py"
                    and spelling == "loop_var.name"
                ):
                    known_indirect[(path.name, spelling)] += 1

    assert constructor_counts == {
        "cin.py": 9,
        # 153 after the shared one-dimensional workspace-key reader replaced
        # two duplicated key expressions with one helper.
        "cin_lowerer.py": 153,
        "compressed_where_openmp_pass.py": 32,
        "dense_pointer_hoist_pass.py": 7,
        "dynamic_vector_access_pass.py": 1,
        "iter_lattice.py": 35,
        "iterator.py": 23,
        "llir_traversal.py": 1,
        "loop_invariant_factor_pass.py": 3,
        "parallel_marking_pass.py": 13,
        "result_write_pass.py": 3,
        "schedule_lowerer.py": 106,
        "single_iteration_loop_pass.py": 1,
        "sparse_prefetch_pass.py": 1,
        "torch_cpp_abi.py": 81,
    }
    assert sum(constructor_counts.values()) == 469
    assert unclassified_counts == {
        "cin.py": 9,
        "cin_lowerer.py": 145,
        "compressed_where_openmp_pass.py": 32,
        "dense_pointer_hoist_pass.py": 7,
        "dynamic_vector_access_pass.py": 1,
        "iter_lattice.py": 35,
        "iterator.py": 21,
        "llir_traversal.py": 1,
        "loop_invariant_factor_pass.py": 3,
        "parallel_marking_pass.py": 13,
        "result_write_pass.py": 3,
        "schedule_lowerer.py": 106,
        "single_iteration_loop_pass.py": 1,
        "sparse_prefetch_pass.py": 1,
        "torch_cpp_abi.py": 81,
    }
    assert sum(unclassified_counts.values()) == 459
    assert known_indirect == {
        ("cin_lowerer.py", "actual_size"): 2,
        ("cin_lowerer.py", "expr.name.replace(old, new)"): 1,
        ("cin_lowerer.py", "outer_end_bound.name"): 1,
        ("cin_lowerer.py", "outer_end_var"): 3,
        ("cin_lowerer.py", "reserve_hint_var"): 1,
        ("cin_lowerer.py", "size_var"): 2,
        ("cin_lowerer.py", "sparse_values_tensor"): 1,
        ("cin_lowerer.py", "wname"): 4,
        ("compressed_where_openmp_pass.py", "bound.name"): 3,
        ("compressed_where_openmp_pass.py", "loop_bound"): 1,
        ("compressed_where_openmp_pass.py", "loop_var.name"): 1,
        ("compressed_where_openmp_pass.py", "sparse_pos"): 1,
        ("parallel_marking_pass.py", "bound_name"): 4,
        ("parallel_marking_pass.py", "sparse_pos"): 2,
        ("parallel_marking_pass.py", "spec.extent"): 1,
        ("dense_pointer_hoist_pass.py", "candidate.base"): 1,
        ("dense_pointer_hoist_pass.py", "candidate.stride"): 1,
        ("dense_pointer_hoist_pass.py", "name"): 1,
        ("dense_pointer_hoist_pass.py", "pointer_name"): 1,
        ("dense_pointer_hoist_pass.py", "value_array"): 1,
        ("llir_traversal.py", "node.name"): 1,
        ("loop_invariant_factor_pass.py", "accumulator_name"): 1,
        ("loop_invariant_factor_pass.py", "invariant_name"): 2,
        ("result_write_pass.py", "f'{prefix}{level}'"): 1,
        ("schedule_lowerer.py", "name"): 1,
        ("schedule_lowerer.py", "prefix_extent"): 2,
        ("schedule_lowerer.py", "zero_value"): 1,
        ("single_iteration_loop_pass.py", "name"): 1,
        ("sparse_prefetch_pass.py", "name"): 1,
        ("torch_cpp_abi.py", "pointer_name"): 4,
        ("torch_cpp_abi.py", "bound_name"): 1,
    }
    assert sum(known_indirect.values()) == 49

    assert totals == {
        "subscript": 9,
        "ternary": 1,
    }
    assert sum(totals.values()) == 10
    assert per_file == {
        ("cin_lowerer.py", "subscript"): 7,
        ("cin_lowerer.py", "ternary"): 1,
        ("iterator.py", "subscript"): 2,
    }


def test_generic_workspace_reads_have_no_opaque_value_fallback() -> None:
    source = (_COMPILER_ROOT / "cin_lowerer.py").read_text()
    tensor_access_lowering = source.split("def lower_TensorAccess", 1)[1].split(
        "def lower_BinaryOp", 1
    )[0]

    assert 'name=f"{tensor_access.tensor.name}_val[{physical_index}]"' not in (
        tensor_access_lowering
    )
    assert tensor_access_lowering.count("if tensor_access_metadata is None:") == 1
    assert "workspace reads require a workspace-specific consumer" in (
        tensor_access_lowering
    )


def test_fixed_stack_array_declaration_producer_budget_is_explicit() -> None:
    constructor_counts = Counter(
        {
            path.name: len(_llir_constructor_calls(path, "FixedStackArrayDecl"))
            for path in sorted(_COMPILER_ROOT.glob("*.py"))
        }
    )
    constructor_counts += Counter()

    assert constructor_counts == {
        "cin_lowerer.py": 1,
        "llir_traversal.py": 1,
    }
    assert sum(constructor_counts.values()) == 2
    assert (
        sum(constructor_counts.values()) - constructor_counts["llir_traversal.py"] == 1
    )

    producers = _llir_constructor_calls(
        _COMPILER_ROOT / "cin_lowerer.py", "FixedStackArrayDecl"
    )
    assert len(producers) == 1
    fields = {
        keyword.arg: keyword.value
        for keyword in producers[0].keywords
        if keyword.arg is not None
    }
    assert set(fields) == {"name", "element_type", "extent", "initializer"}
    assert ast.unparse(fields["name"]) == "wksp.get_name()"
    assert ast.unparse(fields["element_type"]) == "wksp_ctype"
    assert ast.unparse(fields["extent"]) == "wksp.tile_size_var.llir_var"
    assert _is_llir_constructor(fields["initializer"], "Array")
    initializer = fields["initializer"]
    assert isinstance(initializer, ast.Call)
    initializer_fields = {
        keyword.arg: keyword.value
        for keyword in initializer.keywords
        if keyword.arg is not None
    }
    assert ast.unparse(initializer_fields["values"]) == "[]"
    assert ast.unparse(initializer_fields["data_type"]) == "wksp_ctype"

    source = (_COMPILER_ROOT / "cin_lowerer.py").read_text()
    lower_where = source.split("def lower_Where", 1)[1].split(
        "def lower_ProducerIndexStmt", 1
    )[0]
    assert 'name=f"{wksp.get_name()}[{wksp.tile_size_var.name}]"' not in lower_where
    assert lower_where.count("llir.FixedStackArrayDecl(") == 1


def test_free_move_calls_cannot_return_to_var_names() -> None:
    violations: list[tuple[str, int, str]] = []
    structured_moves: Counter[str] = Counter()
    remaining_opaque_calls: Counter[tuple[str, str]] = Counter()
    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for call in _llir_constructor_calls(path, "Var"):
            name_expression = _var_name_expression(call)
            if name_expression is None:
                continue
            fragments = _static_string_fragments(name_expression)
            if "std::move(" in fragments:
                violations.append(
                    (path.name, call.lineno, ast.unparse(name_expression))
                )
            if _string_expression_category(name_expression) == "call":
                if "data_ptr<" in fragments:
                    remaining_opaque_calls[(path.name, "data_ptr")] += 1
                elif ".data()" in fragments:
                    remaining_opaque_calls[(path.name, "storage_data")] += 1
                else:
                    remaining_opaque_calls[(path.name, "other")] += 1

        for call in _llir_constructor_calls(path, "FunctionCall"):
            name_expression = _var_name_expression(call)
            if (
                isinstance(name_expression, ast.Constant)
                and name_expression.value == "std::move"
            ):
                structured_moves[path.name] += 1

    assert violations == []
    assert structured_moves == {
        "cin_lowerer.py": 2,
        "torch_cpp_abi.py": 3,
    }
    assert remaining_opaque_calls == {}


def test_expression_producer_budgets_are_explicit() -> None:
    helper_calls: Counter[tuple[str, str]] = Counter()
    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for helper in (
            "mode_index_tensor",
            "tensor_data_ptr",
            "tensor_storage_member",
        ):
            helper_calls[(path.name, helper)] += len(_function_calls(path, helper))
    helper_calls += Counter()

    member_call_constructors = Counter(
        {
            path.name: len(_llir_constructor_calls(path, "MemberCall"))
            for path in sorted(_COMPILER_ROOT.glob("*.py"))
        }
    )
    member_call_constructors += Counter()

    assert helper_calls == {
        ("cin_lowerer.py", "tensor_data_ptr"): 2,
        ("cin_lowerer.py", "tensor_storage_member"): 1,
        ("torch_cpp_abi.py", "mode_index_tensor"): 4,
        ("torch_cpp_abi.py", "tensor_data_ptr"): 12,
        ("torch_cpp_abi.py", "tensor_storage_member"): 2,
    }
    assert member_call_constructors == {
        "cin_lowerer.py": 2,
        "compressed_where_openmp_pass.py": 1,
        "llir_traversal.py": 1,
        "schedule_lowerer.py": 2,
        "torch_cpp_abi.py": 1,
    }
    assert sum(member_call_constructors.values()) == 7


def test_panel_lower_bound_calls_cannot_return_to_var_names() -> None:
    violations: list[tuple[str, int, str]] = []
    structured_calls: Counter[str] = Counter()
    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for call in _llir_constructor_calls(path, "Var"):
            name_expression = _var_name_expression(call)
            if name_expression is None:
                continue
            fragments = _static_string_fragments(name_expression)
            if "std::lower_bound" in fragments:
                violations.append(
                    (path.name, call.lineno, ast.unparse(name_expression))
                )

        for call in _llir_constructor_calls(path, "FunctionCall"):
            name_expression = _var_name_expression(call)
            if (
                isinstance(name_expression, ast.Constant)
                and name_expression.value == "std::lower_bound"
            ):
                structured_calls[path.name] += 1

    assert violations == []
    assert structured_calls == {"schedule_lowerer.py": 1}


def test_torch_empty_extents_cannot_return_to_var_names() -> None:
    source = (_COMPILER_ROOT / "torch_cpp_abi.py").read_text()
    value_initialization = source.split("def emit_value_array_init", 1)[1].split(
        "def emit_level_indices_init", 1
    )[0]

    assert 'name=f"{{{self.name}_capacity}}"' not in value_initialization
    assert 'name=f"{{{self.known_nnz_var}}}"' not in value_initialization
    assert value_initialization.count("llir.Array(") == 2


def test_torch_dtype_constants_cannot_return_to_var_names() -> None:
    violations: list[tuple[str, int, str]] = []
    structured_counts: Counter[str] = Counter()
    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for call in _llir_constructor_calls(path, "Var"):
            name_expression = _var_name_expression(call)
            if name_expression is None:
                continue
            spelling = ast.unparse(name_expression)
            if (
                "torch::" in _static_string_fragments(name_expression)
                or "get_pytorch_c_dtype_str(" in spelling
            ):
                violations.append((path.name, call.lineno, spelling))

        structured_counts[path.name] = len(
            _llir_constructor_calls(path, "QualifiedName")
        )
    structured_counts += Counter()

    assert violations == []
    assert structured_counts == {
        "cin_lowerer.py": 2,
        "llir_traversal.py": 1,
        "torch_cpp_abi.py": 12,
    }

    producers = []
    for filename in ("cin_lowerer.py", "torch_cpp_abi.py"):
        producers.extend(
            _llir_constructor_calls(_COMPILER_ROOT / filename, "QualifiedName")
        )
    name_expressions: Counter[str] = Counter()
    for producer in producers:
        fields = {
            keyword.arg: keyword.value
            for keyword in producer.keywords
            if keyword.arg is not None
        }
        assert ast.unparse(fields["namespace"]) == "'torch'"
        assert ast.unparse(fields["data_type"]) == ("llir.DataType.TORCH_SCALAR_TYPE")
        name_expressions[ast.unparse(fields["name"])] += 1

    assert name_expressions == {
        "'kInt'": 7,
        "dtype_name": 1,
        "get_pytorch_c_dtype_name(self.dtype)": 3,
        "get_pytorch_c_dtype_name(self.extra_tensor_dtype)": 1,
        "get_pytorch_c_dtype_name(intermediate_tensor_var.dtype)": 1,
        "get_pytorch_c_dtype_name(tensor.dtype)": 1,
    }


def test_result_tensor_assembler_is_owned_by_the_torch_cpp_abi_module() -> None:
    lowerer_path = _COMPILER_ROOT / "cin_lowerer.py"
    abi_path = _COMPILER_ROOT / "torch_cpp_abi.py"
    lowerer_tree = ast.parse(lowerer_path.read_text())
    abi_tree = ast.parse(abi_path.read_text())

    lowerer_definitions = [
        node
        for node in ast.walk(lowerer_tree)
        if isinstance(node, ast.ClassDef) and node.name == "ResultTensorAssembler"
    ]
    abi_definitions = [
        node
        for node in ast.walk(abi_tree)
        if isinstance(node, ast.ClassDef) and node.name == "ResultTensorAssembler"
    ]
    assert lowerer_definitions == []
    assert len(abi_definitions) == 1

    abi_imports = [
        node
        for node in ast.walk(lowerer_tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "torch_cpp_abi"
    ]
    assert len(abi_imports) == 1
    assert "ResultTensorAssembler" in {alias.name for alias in abi_imports[0].names}
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in {"cin", "cin_lowerer"}
        for node in ast.walk(abi_tree)
    )

    producers = Counter(
        {
            path.name: len(_function_calls(path, "ResultTensorAssembler"))
            for path in sorted(_COMPILER_ROOT.glob("*.py"))
        }
    )
    producers += Counter()
    assert producers == {"cin_lowerer.py": 1}


def test_kernel_signature_and_prologue_are_owned_by_the_torch_cpp_abi_module() -> None:
    lowerer_path = _COMPILER_ROOT / "cin_lowerer.py"
    abi_path = _COMPILER_ROOT / "torch_cpp_abi.py"
    lowerer_source = lowerer_path.read_text()
    abi_source = abi_path.read_text()
    lowerer_tree = ast.parse(lowerer_source)
    abi_tree = ast.parse(abi_source)

    for class_name in ("KernelTensorABI", "TorchCppKernelABI"):
        assert [
            node
            for node in ast.walk(lowerer_tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ] == []
        assert (
            len(
                [
                    node
                    for node in ast.walk(abi_tree)
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                ]
            )
            == 1
        )
        producers = Counter(
            {
                path.name: len(_function_calls(path, class_name))
                for path in sorted(_COMPILER_ROOT.glob("*.py"))
            }
        )
        producers += Counter()
        assert producers == {"cin_lowerer.py": 1}

    abi_imports = [
        node
        for node in ast.walk(lowerer_tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "torch_cpp_abi"
    ]
    assert len(abi_imports) == 1
    imported_names = {alias.name for alias in abi_imports[0].names}
    assert {"KernelTensorABI", "TorchCppKernelABI"} <= imported_names
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in {"cin", "cin_lowerer"}
        for node in ast.walk(abi_tree)
    )

    assert _llir_constructor_calls(lowerer_path, "Function") == []
    assert len(_llir_constructor_calls(abi_path, "Function")) == 1
    assert "def get_level_arrays" not in lowerer_source
    assert "def get_val_ptr_stmt" not in lowerer_source
    for validator in (
        "validate_jit_result_shape",
        "validate_jit_tensor",
        "validate_jit_extra_tensor",
    ):
        assert validator not in lowerer_source
        assert abi_source.count(validator) == 1


def test_workspace_pair_reads_cannot_return_to_var_names() -> None:
    violations: list[tuple[str, int, str]] = []
    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for call in _llir_constructor_calls(path, "Var"):
            name_expression = _var_name_expression(call)
            if name_expression is None:
                continue
            fragments = _static_string_fragments(name_expression)
            if ".first" in fragments or ".second" in fragments:
                violations.append(
                    (path.name, call.lineno, ast.unparse(name_expression))
                )

    assert violations == []


def test_tiled_workspace_copy_read_cannot_return_to_a_var_name() -> None:
    source = (_COMPILER_ROOT / "cin_lowerer.py").read_text()

    # The one remaining spelling belongs to the separate non-tiled fallback.
    assert source.count('name=f"{wksp.get_name()}[{loop_var.name}]"') == 1
    assert source.count("type=llir.DataType.ptr_type(wksp.dtype)") == 1


def test_all_coo_coordinate_initializers_cannot_return_to_var_names() -> None:
    source = (_COMPILER_ROOT / "cin_lowerer.py").read_text()
    transform = source.split("def _transform_coo_loop_for_openmp", 1)[1].split(
        "def lower_ForAll", 1
    )[0]

    assert transform.count('name=f"{crd_array}[{iter_var}]"') == 0
    assert transform.count("name=crd_array,") == 2
    assert transform.count("name=cast(str, iter_var),") == 2


def test_all_coo_single_element_end_bound_cannot_return_to_a_var_name() -> None:
    source = (_COMPILER_ROOT / "cin_lowerer.py").read_text()
    transform = source.split("def _transform_coo_loop_for_openmp", 1)[1].split(
        "def lower_ForAll", 1
    )[0]

    assert 'name=f"{iter_var} + 1"' not in transform


def test_iterator_coordinate_reads_cannot_return_to_var_names() -> None:
    path = _COMPILER_ROOT / "iterator.py"
    violations: list[tuple[int, str]] = []
    for call in _llir_constructor_calls(path, "Var"):
        name_expression = _var_name_expression(call)
        if name_expression is None:
            continue
        fragments = _static_string_fragments(name_expression)
        if "_crd[" in fragments:
            violations.append((call.lineno, ast.unparse(name_expression)))

    assert violations == []


def test_compressed_iterator_position_bounds_cannot_return_to_var_names() -> None:
    source = (_COMPILER_ROOT / "iterator.py").read_text()
    compressed = source.split("elif self.level_type == LevelType.COMPRESSED:", 1)[
        1
    ].split("elif self.level_type == LevelType.DENSE:", 1)[0]

    assert "_pos[" not in compressed
    assert "self._compressed_position_access(0)" in compressed
    assert "self._compressed_position_access(1)" in compressed


def test_dense_level_shape_reads_cannot_return_to_var_names() -> None:
    violations: list[tuple[str, int, str]] = []
    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for call in _llir_constructor_calls(path, "Var"):
            name_expression = _var_name_expression(call)
            if name_expression is None:
                continue
            fragments = _static_string_fragments(name_expression)
            if "result_shape[" in fragments or "_shape[" in fragments:
                violations.append(
                    (path.name, call.lineno, ast.unparse(name_expression))
                )

    assert violations == []


def test_raw_statement_producer_budget_remains_explicit() -> None:
    counts = Counter(
        {
            path.name: len(_llir_constructor_calls(path, "RawStmt"))
            for path in sorted(_COMPILER_ROOT.glob("*.py"))
        }
    )
    counts += Counter()

    assert counts == {
        "compressed_where_openmp_pass.py": 2,
        "dense_pointer_hoist_pass.py": 1,
        "llir_traversal.py": 1,
    }
    assert sum(counts.values()) == 4
    assert sum(counts.values()) - counts["llir_traversal.py"] == 3


def test_workspace_clear_mutations_are_structured() -> None:
    """Lock the W2/W4 template and global member-call-statement budget."""

    member_call_stmt_constructors = Counter(
        {
            path.name: len(_llir_constructor_calls(path, "MemberCallStmt"))
            for path in sorted(_COMPILER_ROOT.glob("*.py"))
        }
    )
    member_call_stmt_constructors += Counter()
    assert member_call_stmt_constructors == {
        "cin_lowerer.py": 2,
        "compressed_where_openmp_pass.py": 3,
        "dense_pointer_hoist_pass.py": 1,
        "llir_traversal.py": 1,
        "schedule_lowerer.py": 1,
        "single_iteration_loop_pass.py": 1,
    }
    assert sum(member_call_stmt_constructors.values()) == 9

    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    compressed_source = compressed_path.read_text()
    clear_helper = compressed_source.split("def _workspace_clear_statement", 1)[
        1
    ].split("def _phase_header_copy", 1)[0]
    count_body = compressed_source.split("def _build_count_body", 1)[1].split(
        "def _build_fill_body", 1
    )[0]
    fill_body = compressed_source.split("def _build_fill_body", 1)[1].split(
        "def _should_drop_prefix_statement", 1
    )[0]

    assert "llir.RawStmt(" not in clear_helper
    assert "llir.RawStmt(" not in count_body
    assert "llir.RawStmt(" not in fill_body
    assert clear_helper.count("llir.MemberCallStmt(") == 1
    assert 'member="clear"' in clear_helper
    assert (
        "base=llir.Var(name=context.workspace_name, type=llir.DataType.NO_TYPE)"
        in clear_helper
    )
    assert count_body.count("body.append(_workspace_clear_statement(context))") == 1
    assert fill_body.count("body.append(_workspace_clear_statement(context))") == 1

    for call in _llir_constructor_calls(compressed_path, "RawStmt"):
        assert ".clear" not in _static_string_fragments(call)


def test_workspace_view_borrow_is_structured() -> None:
    """Lock the typed W1 worker-view acquisition template."""

    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    compressed_source = compressed_path.read_text()
    view_helper = compressed_source.split("def _workspace_view_statement", 1)[1].split(
        "def _workspace_clear_statement", 1
    )[0]

    assert "llir.RawStmt(" not in view_helper
    assert view_helper.count("llir.VarInit(") == 1
    assert view_helper.count("llir.MemberCall(") == 1
    assert view_helper.count("llir.ArrayAccess(") == 1
    assert view_helper.count("llir.Cast(") == 1
    assert view_helper.count("llir.FunctionCall(") == 1
    assert 'member="make_view"' in view_helper
    assert 'name="omp_get_thread_num"' in view_helper
    assert "llir.DataType.AUTO" in view_helper
    assert "type=_workspace_pool_type(context) or llir.DataType.NO_TYPE" in view_helper
    assert "llir.DataType.SIZE_T" in view_helper

    for call in _llir_constructor_calls(compressed_path, "RawStmt"):
        assert "make_view" not in _static_string_fragments(call)


def test_workspace_pool_construction_is_structured_with_legacy_fallback() -> None:
    """Lock typed W5 ownership and its deliberate compatibility fallback."""

    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    compressed_source = compressed_path.read_text()
    pool_helpers = compressed_source.split("def _workspace_pool_type", 1)[1].split(
        "def _resolve_outer_cell", 1
    )[0]

    assert pool_helpers.count("llir.RawStmt(") == 1
    assert pool_helpers.count("llir.VarInit(") == 2
    assert pool_helpers.count("llir.VarDecl(") == 1
    assert pool_helpers.count("llir.MemberCallStmt(") == 2
    assert pool_helpers.count("llir.ForLoop(") == 1
    assert pool_helpers.count("llir.Cast(") == 1
    assert pool_helpers.count("llir.ArrayAccess(") == 1
    assert pool_helpers.count("llir.Increment(") == 1
    assert pool_helpers.count("llir.BinOp(") == 1
    assert pool_helpers.count("llir.FunctionCall(") == 3
    assert pool_helpers.count("llir.Var(") == 4
    assert 'member="reserve"' in pool_helpers
    assert 'member="emplace_back"' in pool_helpers
    assert 'name="result_shape"' in pool_helpers
    assert "linked_list_workspace_pool_type" in pool_helpers
    assert '"std::max"' in pool_helpers
    assert '"omp_get_max_threads"' in pool_helpers
    assert "_legacy_workspace_pool_statement" in pool_helpers
    assert "if pool_type is None or not typed_policies" in pool_helpers

    pool_raws = [
        _static_string_fragments(call)
        for call in _llir_constructor_calls(compressed_path, "RawStmt")
        if "thread_count" in _static_string_fragments(call)
    ]
    assert len(pool_raws) == 1
    assert "reserve" in pool_raws[0]
    assert "emplace_back" in pool_raws[0]


def test_dense_workspace_zero_fill_is_structured() -> None:
    """Lock the C3/C5 typed template and the global Sizeof budget."""

    sizeof_constructors = Counter(
        {
            path.name: len(_llir_constructor_calls(path, "Sizeof"))
            for path in sorted(_COMPILER_ROOT.glob("*.py"))
        }
    )
    sizeof_constructors += Counter()
    assert sizeof_constructors == {
        "cin_lowerer.py": 2,
        "llir_traversal.py": 1,
    }
    assert sum(sizeof_constructors.values()) == 3

    lowerer_path = _COMPILER_ROOT / "cin_lowerer.py"
    lowerer_source = lowerer_path.read_text()
    lower_where = lowerer_source.split("def lower_Where", 1)[1].split(
        "def lower_ProducerIndexStmt", 1
    )[0]

    assert lower_where.count('name="memset"') == 1
    assert lower_where.count("llir.FunctionCallStmt(") == 1
    assert lower_where.count("llir.Sizeof(data_type=wksp_ctype)") == 1
    assert lower_where.count("var=size_llir") == 1
    assert "memset" not in "".join(
        _static_string_fragments(call)
        for call in _llir_constructor_calls(lowerer_path, "RawStmt")
    )
    # The whole dense-workspace zero-fill cluster in lower_Where is typed:
    # the extent alias, the restrict-qualified pool borrow (C4), and the
    # memset construct no raw statements.
    assert lower_where.count("llir.RawStmt(") == 0
    assert 'f"int64_t' not in lower_where
    assert lower_where.count("is_restrict=True") == 1
    assert lower_where.count('member="get"') == 1
    assert "__restrict__" not in lower_where

    # The pool thread count (C15) and RAII owner (C16) are typed: the
    # worker count owns a fresh typed copy of the applied parallel policy,
    # and the owner calls the templated aligned-buffer allocator. Both are
    # owned by the parallel-marking pass module.
    marking_path = _COMPILER_ROOT / "parallel_marking_pass.py"
    marking_source = marking_path.read_text()
    pools = marking_source.split("def attach_serial_workspace_pools", 1)[1].split(
        "def mark_first_for_loop_parallel", 1
    )[0]
    assert pools.count("llir.RawStmt(") == 0
    assert pools.count("llir.VarInit(") == 2
    assert pools.count('"scorch_make_aligned_buffer"') == 1
    assert pools.count('"scorch_checked_size_product"') == 1
    assert pools.count('"omp_get_max_threads"') == 1
    assert pools.count("template_args=(spec.scalar_type,)") == 1
    assert pools.count("thread_policy_factory()") == 1
    assert _llir_constructor_calls(marking_path, "RawStmt") == []
    for path in (lowerer_path, marking_path):
        for call in _llir_constructor_calls(path, "RawStmt"):
            fragments = _static_string_fragments(call)
            assert "pool_owner" not in fragments
            assert "thread_count" not in fragments
            assert "__restrict__" not in fragments


def test_dense_workspace_write_back_copy_is_structured() -> None:
    """Lock the C6 typed template and the global AddressOf budget."""

    address_of_constructors = Counter(
        {
            path.name: len(_llir_constructor_calls(path, "AddressOf"))
            for path in sorted(_COMPILER_ROOT.glob("*.py"))
        }
    )
    address_of_constructors += Counter()
    assert address_of_constructors == {
        "cin_lowerer.py": 2,
        "dense_pointer_hoist_pass.py": 2,
        "llir_traversal.py": 1,
        "schedule_lowerer.py": 1,
        "single_iteration_loop_pass.py": 1,
        "sparse_prefetch_pass.py": 1,
    }
    assert sum(address_of_constructors.values()) == 8

    lowerer_path = _COMPILER_ROOT / "cin_lowerer.py"
    lowerer_source = lowerer_path.read_text()
    consumer_lowering = lowerer_source.split("def lower_ConsumerIndexStmt", 1)[1].split(
        "def _prepare_scheduled_cin", 1
    )[0]

    assert consumer_lowering.count('name="memcpy"') == 1
    assert consumer_lowering.count("llir.AddressOf(") == 1
    assert consumer_lowering.count("llir.Sizeof(data_type=wksp_write_ctype)") == 1
    assert "llir.RawStmt(" not in consumer_lowering
    assert "memcpy" not in "".join(
        _static_string_fragments(call)
        for call in _llir_constructor_calls(lowerer_path, "RawStmt")
    )


def test_all_coo_preallocation_is_structured() -> None:
    """Lock the C17 typed template: no raw resize spelling anywhere."""

    lowerer_path = _COMPILER_ROOT / "cin_lowerer.py"
    lowerer_source = lowerer_path.read_text()
    coo_transform = lowerer_source.split("def _transform_coo_loop_for_openmp", 1)[
        1
    ].split("def lower_ForAll", 1)[0]

    assert "llir.RawStmt(" not in coo_transform
    assert coo_transform.count("llir.MemberCallStmt(") == 1
    assert coo_transform.count('member="resize"') == 1
    assert coo_transform.count("for array_var in output_arrays.values()") == 1

    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for call in _llir_constructor_calls(path, "RawStmt"):
            assert ".resize" not in _static_string_fragments(call)


def test_reserve_hint_declaration_is_structured() -> None:
    """Lock the C7 typed capped checked-product reserve-hint template."""

    lowerer_path = _COMPILER_ROOT / "cin_lowerer.py"

    for call in _llir_constructor_calls(lowerer_path, "RawStmt"):
        fragments = _static_string_fragments(call)
        assert "checked_product" not in fragments
        assert "std::min<" not in fragments

    initializers: list[dict[str, ast.expr]] = []
    for call in _llir_constructor_calls(lowerer_path, "VarInit"):
        fields = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        target = fields.get("var")
        if target is None or not _is_llir_constructor(target, "Var"):
            continue
        assert isinstance(target, ast.Call)
        name_expression = _var_name_expression(target)
        if (
            name_expression is not None
            and ast.unparse(name_expression) == "reserve_hint_var"
        ):
            assert ast.unparse(target.keywords[1].value) == "llir.DataType.INT64"
            initializers.append(fields)
    assert len(initializers) == 1

    value = initializers[0]["value"]
    assert _is_llir_constructor(value, "FunctionCall")
    assert isinstance(value, ast.Call)
    value_fields = {
        keyword.arg: keyword.value
        for keyword in value.keywords
        if keyword.arg is not None
    }
    assert ast.unparse(value_fields["name"]) == "'std::min'"
    assert ast.unparse(value_fields["template_args"]) == "(llir.DataType.INT64,)"
    outer_args = value_fields["args"]
    assert isinstance(outer_args, ast.List)
    assert len(outer_args.elts) == 2
    assert ast.unparse(outer_args.elts[1]) == (
        "llir.Literal(2048, llir.DataType.INT64)"
    )

    product = outer_args.elts[0]
    assert _is_llir_constructor(product, "FunctionCall")
    assert isinstance(product, ast.Call)
    product_fields = {
        keyword.arg: keyword.value
        for keyword in product.keywords
        if keyword.arg is not None
    }
    assert ast.unparse(product_fields["name"]) == "'scorch_native::checked_product'"
    assert "template_args" not in product_fields
    product_args = product_fields["args"]
    assert isinstance(product_args, ast.List)
    assert [ast.unparse(argument) for argument in product_args.elts] == [
        "llir.Var(name='result_shape', type=llir.DataType.STD_VECTOR_INT)",
        "llir.Literal('evaluate', llir.DataType.STRING)",
        "llir.Literal('result_shape', llir.DataType.STRING)",
        "llir.Literal(True, llir.DataType.BOOL)",
    ]


def test_atomic_work_stealing_prelude_is_structured() -> None:
    """Lock the C12/C13 typed templates and their structural-piece ownership."""

    lowerer_path = _COMPILER_ROOT / "cin_lowerer.py"
    marking_path = _COMPILER_ROOT / "parallel_marking_pass.py"
    marking_source = marking_path.read_text()
    marking_tree = ast.parse(marking_source)

    for path in (lowerer_path, marking_path):
        for call in _llir_constructor_calls(path, "RawStmt"):
            fragments = _static_string_fragments(call)
            assert "int _nnz" not in fragments
            assert "_chunk" not in fragments
            assert "omp_get_num_threads" not in fragments

    helpers = [
        node
        for node in ast.walk(marking_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "atomic_work_stealing_prelude"
    ]
    assert len(helpers) == 1
    helper = helpers[0]

    def helper_calls(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                or isinstance(node.func, ast.Name)
                and node.func.id == name
            )
        ]

    expected_template_inventory = {
        "RawStmt": 0,
        "VarInit": 2,
        "Var": 5,
        "ArrayAccess": 1,
        "FunctionCall": 3,
        "Literal": 3,
        "BinOp": 1,
        "Mul": 1,
        "Cast": 0,
    }
    assert {
        name: len(helper_calls(name)) for name in expected_template_inventory
    } == expected_template_inventory

    marking = marking_source.split("def mark_first_for_loop_parallel", 1)[1]
    assert marking.count("atomic_work_stealing_prelude(") == 1
    assert marking.count("llir.RawStmt(") == 0
    # The pragma text and the typed prelude come from the same validated
    # structural pieces; neither is recovered by parsing the other.
    assert 'f"scorch_nthreads({sparse_work}, {loop_bound})"' in marking


def test_hoisted_input_pointer_declaration_is_structured() -> None:
    """Lock D1's typed template and exact compatibility raw fallback."""

    path = _COMPILER_ROOT / "dense_pointer_hoist_pass.py"
    source = path.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    helper = functions["_hoisted_pointer_declaration"]
    apply_analysis = functions["_apply_loop_analysis"]

    def calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                or isinstance(node.func, ast.Name)
                and node.func.id == name
            )
        ]

    expected_template_inventory = {
        "RawStmt": 1,
        "VarInit": 1,
        "Var": 4,
        "AddressOf": 1,
        "ArrayAccess": 1,
        "Mul": 1,
        "const_ptr_type": 1,
        "_is_assignment_name": 3,
        "Cast": 0,
    }
    assert {
        name: len(calls(helper, name)) for name in expected_template_inventory
    } == expected_template_inventory

    # The one raw statement in this file is the deliberate compatibility
    # fallback inside the helper. It covers both free-form type text and
    # expression-bearing names outside AddressOf's grammar without parsing
    # either, and keeps the exact legacy spelling.
    assert len(_llir_constructor_calls(path, "RawStmt")) == 1
    helper_source = ast.get_source_segment(source, helper)
    assert helper_source is not None
    assert "except ValueError" in helper_source
    assert helper_source.count("llir._is_assignment_name(") == 3
    fallback = calls(helper, "RawStmt")[0]
    fragments = _static_string_fragments(fallback)
    assert "const " in fragments
    assert "__restrict__" in fragments

    assert len(calls(apply_analysis, "_hoisted_pointer_declaration")) == 1
    assert calls(apply_analysis, "RawStmt") == []

    typed_var_fields = {
        keyword.arg: keyword.value
        for call in calls(helper, "VarInit")
        for keyword in call.keywords
        if keyword.arg == "var"
    }
    target = typed_var_fields["var"]
    assert _is_llir_constructor(target, "Var")
    assert isinstance(target, ast.Call)
    target_fields = {
        keyword.arg: keyword.value
        for keyword in target.keywords
        if keyword.arg is not None
    }
    assert ast.unparse(target_fields["name"]) == "pointer_name"
    assert ast.unparse(target_fields["type"]) == "pointer_type"
    assert ast.unparse(target_fields["is_restrict"]) == "True"


def test_compressed_result_assembly_is_owned_by_the_typed_abi_epilogue() -> None:
    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    compressed_source = compressed_path.read_text()
    final_assembly = compressed_source.split("def _final_assembly", 1)[1].split(
        "def _build_transformed_statements", 1
    )[0]
    build = compressed_source.split("def _build_transformed_statements", 1)[1].split(
        "def _transform_compressed_where_for_openmp_managed", 1
    )[0]
    abi_source = (_COMPILER_ROOT / "torch_cpp_abi.py").read_text()
    declaration = abi_source.split("def emit_result_declaration", 1)[1].split(
        "def emit_storage_epilogue", 1
    )[0]
    epilogue = abi_source.split("def emit_storage_epilogue", 1)[1].split(
        "def emit_final_assembly", 1
    )[0]
    final = abi_source.split("def emit_final_assembly", 1)[1]

    assert "llir.RawStmt(" not in final_assembly
    assert "context.result_assembler.emit_result_declaration()" in final_assembly
    assert "context.result_assembler.emit_storage_epilogue()" in final_assembly
    assert "result.extend(_final_assembly(context))" in build
    for marker in (
        "Tensor {result_name}",
        "storage.index.mode_indices",
        "storage.value",
        "return {result_name}",
    ):
        assert marker not in final_assembly

    assert declaration.count("llir.VarDecl(") == 1
    assert declaration.count("llir.Var(") == 1
    assert epilogue.count("llir.Assign(") == 2
    assert epilogue.count("llir.Return(") == 1
    assert "tensor_storage_member(" in epilogue
    assert "self._get_mode_index_set(i, level_type)" in epilogue
    assert "stmts.extend(self.emit_storage_epilogue())" in final

    lowerer_source = (_COMPILER_ROOT / "cin_lowerer.py").read_text()
    compressed_context = lowerer_source.split("CompressedWhereOpenMPContext(", 1)[
        1
    ].split("workspace_name=workspace_name", 1)[0]
    assert (
        compressed_context.count(
            "result_assembler=_result_tensor_abi_assembler(result_tensor)"
        )
        == 1
    )


def test_direct_initialization_budget_and_live_owners_are_explicit() -> None:
    counts = Counter(
        {
            path.name: len(_llir_constructor_calls(path, "DirectInit"))
            for path in sorted(_COMPILER_ROOT.glob("*.py"))
        }
    )
    counts += Counter()

    assert counts == {
        "compressed_where_openmp_pass.py": 2,
        "llir_traversal.py": 1,
        "schedule_lowerer.py": 2,
        "torch_cpp_abi.py": 1,
    }
    assert sum(counts.values()) == 6

    schedule_source = (_COMPILER_ROOT / "schedule_lowerer.py").read_text()
    # The compact-storage owner moved into the shared helper the legacy
    # _apply_heap_result_tile and the typed LoopIR heap completion both
    # call; the typed DirectInit owner remains the single source of the
    # spelling.
    heap_storage = schedule_source.split("def _heap_result_storage_statements", 1)[
        1
    ].split("def _apply_heap_result_tile", 1)[0]
    assert "llir.RawStmt(" not in heap_storage
    assert "storage_declaration = _heap_result_storage_declaration(" in heap_storage

    heap_result = schedule_source.split("def _apply_heap_result_tile", 1)[1].split(
        "def _packed_storage_declaration", 1
    )[0]
    assert "llir.RawStmt(" not in heap_result
    assert (
        "tile_container[tile_index:tile_index] = _heap_result_storage_statements("
        in heap_result
    )

    # The packed-storage owner moved into the shared helper the legacy
    # _apply_relayout and the typed LoopIR relayout completion both call;
    # the typed DirectInit owner remains the single source of the spelling.
    relayout_storage = schedule_source.split("def _relayout_storage_statements", 1)[
        1
    ].split("def _apply_relayout", 1)[0]
    assert "llir.RawStmt(" not in relayout_storage
    assert "packed_storage = _packed_storage_declaration(" in relayout_storage

    relayout = schedule_source.split("def _apply_relayout", 1)[1].split(
        "def apply_schedule_to_llir", 1
    )[0]
    assert "llir.RawStmt(" not in relayout
    assert (
        "pack_container[pack_index:pack_index] = _relayout_storage_statements("
        in relayout
    )

    torch_abi_source = (_COMPILER_ROOT / "torch_cpp_abi.py").read_text()
    level_initialization = torch_abi_source.split("def emit_level_indices_init", 1)[
        1
    ].split("def _get_mode_index_set", 1)[0]
    assert "llir.RawStmt(" not in level_initialization
    assert level_initialization.count("llir.DirectInit(") == 1
    assert "data_type=llir.DataType.SIZE_T" in level_initialization
    assert "type=llir.DataType.INT64" in level_initialization


def test_known_nnz_coordinate_torch_allocation_is_structured() -> None:
    path = _COMPILER_ROOT / "torch_cpp_abi.py"
    raw_violations = [
        (call.lineno, ast.unparse(call))
        for call in _llir_constructor_calls(path, "RawStmt")
        if "known_nnz_var" in ast.unparse(call) and "_crd_torch" in ast.unparse(call)
    ]
    coordinate_owner_initializers: list[ast.Call] = []
    for call in _llir_constructor_calls(path, "VarInit"):
        fields = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        target = fields.get("var")
        if target is None or not _is_llir_constructor(target, "Var"):
            continue
        assert isinstance(target, ast.Call)
        name_expression = _var_name_expression(target)
        value = fields.get("value")
        value_name_expression = (
            _var_name_expression(value) if isinstance(value, ast.Call) else None
        )
        if (
            name_expression is not None
            and "_crd_torch" in _static_string_fragments(name_expression)
            and value is not None
            and _is_llir_constructor(value, "FunctionCall")
            and value_name_expression is not None
            and ast.unparse(value_name_expression) == "'torch::empty'"
        ):
            coordinate_owner_initializers.append(call)

    assert raw_violations == []
    assert len(coordinate_owner_initializers) == 1
    assert (
        sum(
            len(_llir_constructor_calls(path, constructor))
            for constructor in ("VarInit", "Array", "FunctionCall", "QualifiedName")
        )
        == 64
    )
    assert len(_llir_constructor_calls(path, "VarInit")) == 29
    assert len(_llir_constructor_calls(path, "Array")) == 10
    assert len(_llir_constructor_calls(path, "FunctionCall")) == 13
    assert len(_llir_constructor_calls(path, "QualifiedName")) == 12


def test_first_compressed_position_allocation_is_structured() -> None:
    """Lock the complete W10/W11 owner, borrow, and copy-loop template."""

    abi_path = _COMPILER_ROOT / "torch_cpp_abi.py"
    abi_tree = ast.parse(abi_path.read_text())
    emitters = [
        node
        for node in ast.walk(abi_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "emit_first_compressed_position_allocation"
    ]
    assert len(emitters) == 1
    emitter = emitters[0]

    def emitter_calls(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(emitter)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                or isinstance(node.func, ast.Name)
                and node.func.id == name
            )
        ]

    expected_template_inventory = {
        "RawStmt": 0,
        "VarInit": 3,
        "Var": 5,
        "Array": 1,
        "FunctionCall": 1,
        "QualifiedName": 1,
        "tensor_data_ptr": 1,
        "ForLoop": 1,
        "Assign": 1,
        "ArrayAccess": 2,
        "Add": 1,
        "Literal": 2,
        "Cast": 2,
        "BinOp": 1,
        "Increment": 1,
    }
    assert {
        name: len(emitter_calls(name)) for name in expected_template_inventory
    } == expected_template_inventory

    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    compressed_source = compressed_path.read_text()
    position_allocations = compressed_source.split(
        "def _position_and_coordinate_allocations", 1
    )[1].split("_CTYPE_TO_TORCH", 1)[0]
    assert position_allocations.count("llir.RawStmt(") == 0
    assert (
        position_allocations.count(
            "context.result_assembler.emit_first_compressed_position_allocation("
        )
        == 1
    )
    assert "_cell_count_reference(cell_count)" in position_allocations
    assert "_offset_reference(first_level)" in position_allocations
    for opaque_spelling in (
        "{first_level}_pos_torch",
        "{first_level}_pos_data",
        "for (int _i = 0; _i <= {loop_bound}; _i++)",
    ):
        assert opaque_spelling not in position_allocations


def test_compressed_coordinate_torch_allocations_are_structured() -> None:
    """Lock W12 on typed ABI nodes beside typed W10/W11 and W13 owners."""

    abi_path = _COMPILER_ROOT / "torch_cpp_abi.py"
    abi_tree = ast.parse(abi_path.read_text())
    emitters = [
        node
        for node in ast.walk(abi_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "emit_compressed_coordinate_allocations"
    ]
    assert len(emitters) == 1
    emitter = emitters[0]

    def emitter_calls(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(emitter)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                or isinstance(node.func, ast.Name)
                and node.func.id == name
            )
        ]

    assert emitter_calls("RawStmt") == []
    assert len(emitter_calls("VarInit")) == 2
    assert len(emitter_calls("Var")) == 4
    assert len(emitter_calls("Array")) == 1
    assert len(emitter_calls("FunctionCall")) == 1
    assert len(emitter_calls("QualifiedName")) == 1
    assert len(emitter_calls("tensor_data_ptr")) == 1

    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    compressed_source = compressed_path.read_text()
    position_allocations = compressed_source.split(
        "def _position_and_coordinate_allocations", 1
    )[1].split("_CTYPE_TO_TORCH", 1)[0]
    assert (
        position_allocations.count(
            "context.result_assembler.emit_compressed_coordinate_allocations("
        )
        == 1
    )
    assert "_crd_torch" not in position_allocations
    assert "_crd_data" not in position_allocations

    assert position_allocations.count("llir.RawStmt(") == 0
    assert (
        position_allocations.count(
            "context.result_assembler.emit_first_compressed_position_allocation("
        )
        == 1
    )
    assert "{first_level}_pos_torch" not in position_allocations
    assert "for (int _i = 0; _i <= {loop_bound}; _i++)" not in position_allocations
    assert "{result_name}{level}_pos_torch" not in position_allocations
    assert "{result_name}{level}_pos_data" not in position_allocations
    assert "_total{parent_level}" not in position_allocations


def test_deeper_compressed_position_torch_allocations_are_structured() -> None:
    """Lock W13's typed template and its single shared total-tuple handoff."""

    abi_path = _COMPILER_ROOT / "torch_cpp_abi.py"
    abi_tree = ast.parse(abi_path.read_text())
    emitters = [
        node
        for node in ast.walk(abi_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "emit_deeper_compressed_position_allocations"
    ]
    assert len(emitters) == 1
    emitter = emitters[0]

    def emitter_calls(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(emitter)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                or isinstance(node.func, ast.Name)
                and node.func.id == name
            )
        ]

    expected_template_inventory = {
        "RawStmt": 0,
        "VarInit": 2,
        "Var": 5,
        "Array": 1,
        "FunctionCall": 1,
        "QualifiedName": 1,
        "tensor_data_ptr": 1,
        "Assign": 1,
        "ArrayAccess": 1,
        "Add": 1,
        "Literal": 3,
        "Cast": 0,
    }
    assert {
        name: len(emitter_calls(name)) for name in expected_template_inventory
    } == expected_template_inventory

    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    compressed_source = compressed_path.read_text()
    compressed_tree = ast.parse(compressed_source)
    position_functions = [
        node
        for node in compressed_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_position_and_coordinate_allocations"
    ]
    assert len(position_functions) == 1
    position_function = position_functions[0]

    total_assignments = [
        node
        for node in position_function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "total_vars"
    ]
    assert len(total_assignments) == 1
    assert ast.unparse(total_assignments[0].value) == (
        "tuple((_total_reference(level) for level in levels))"
    )

    def position_calls(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(position_function)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                or isinstance(node.func, ast.Name)
                and node.func.id == name
            )
        ]

    assert len(position_calls("tuple")) == 1
    assert len(position_calls("_total_reference")) == 1
    coordinate_calls = position_calls("emit_compressed_coordinate_allocations")
    deeper_position_calls = position_calls(
        "emit_deeper_compressed_position_allocations"
    )
    assert len(coordinate_calls) == 1
    assert len(deeper_position_calls) == 1
    for call in coordinate_calls + deeper_position_calls:
        assert call.keywords == []
        assert len(call.args) == 1
        assert ast.unparse(call.args[0]) == "total_vars"

    deeper_position_conditionals = [
        node
        for node in position_function.body
        if isinstance(node, ast.If)
        and any(call is deeper_position_calls[0] for call in ast.walk(node))
    ]
    assert len(deeper_position_conditionals) == 1
    assert ast.unparse(deeper_position_conditionals[0].test) == "len(levels) > 1"

    raw_calls = position_calls("RawStmt")
    assert raw_calls == []
    position_source = ast.get_source_segment(compressed_source, position_function)
    assert position_source is not None
    for raw_deeper_position_spelling in (
        "{result_name}{level}_pos_torch",
        "{result_name}{level}_pos_data",
        "_total{parent_level}",
    ):
        assert raw_deeper_position_spelling not in position_source


def test_compressed_value_allocation_has_typed_and_legacy_paths() -> None:
    """Lock W14's exact typed template and spelling-preserving fallback."""

    abi_path = _COMPILER_ROOT / "torch_cpp_abi.py"
    abi_tree = ast.parse(abi_path.read_text())
    emitters = [
        node
        for node in ast.walk(abi_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "emit_compressed_value_allocation"
    ]
    assert len(emitters) == 1
    emitter = emitters[0]

    def calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                or isinstance(node.func, ast.Name)
                and node.func.id == name
            )
        ]

    expected_template_inventory = {
        "RawStmt": 0,
        "VarInit": 2,
        "Var": 3,
        "Array": 1,
        "FunctionCall": 1,
        "QualifiedName": 1,
        "tensor_data_ptr": 1,
        "Cast": 0,
        "Add": 0,
        "Literal": 0,
    }
    assert {
        name: len(calls(emitter, name)) for name in expected_template_inventory
    } == expected_template_inventory

    c_datatype_assignments = [
        statement
        for node in emitter.body
        if isinstance(node, ast.Try)
        for statement in node.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "c_datatype"
            for target in statement.targets
        )
    ]
    assert len(c_datatype_assignments) == 1
    assert ast.unparse(c_datatype_assignments[0].value) == (
        "dtype_to_c_datatype(self.dtype)"
    )
    pointer_type_assignments = [
        statement
        for node in emitter.body
        if isinstance(node, ast.Try)
        for statement in node.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pointer_type"
            for target in statement.targets
        )
    ]
    assert len(pointer_type_assignments) == 1
    assert ast.unparse(pointer_type_assignments[0].value) == (
        "llir.DataType.ptr_type(c_datatype)"
    )

    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    compressed_source = compressed_path.read_text()
    compressed_tree = ast.parse(compressed_source)
    functions = {
        node.name: node
        for node in compressed_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    legacy = functions["_legacy_value_allocation"]
    router = functions["_value_allocation"]
    builder = functions["_build_transformed_statements"]

    assert len(calls(legacy, "RawStmt")) == 1
    assert calls(router, "RawStmt") == []
    legacy_raw = calls(legacy, "RawStmt")[0]
    legacy_fields = {
        keyword.arg: keyword.value
        for keyword in legacy_raw.keywords
        if keyword.arg is not None
    }
    assert set(legacy_fields) == {"code", "add_semicolon"}
    assert ast.unparse(legacy_fields["add_semicolon"]) == "False"
    legacy_code = ast.unparse(legacy_fields["code"])
    for spelling in (
        "(long long)_total{leaf}",
        "{ctype}* {result_name}_values_data",
        "data_ptr<{ctype}>()",
    ):
        assert spelling in legacy_code

    torch_dtype_assignments = [
        node
        for node in legacy.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "torch_dtype"
            for target in node.targets
        )
    ]
    assert len(torch_dtype_assignments) == 1
    assert ast.unparse(torch_dtype_assignments[0].value) == (
        "_CTYPE_TO_TORCH.get(ctype, 'torch::kFloat32')"
    )

    ctype_maps = [
        node
        for node in compressed_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_CTYPE_TO_TORCH"
    ]
    assert len(ctype_maps) == 1
    assert ctype_maps[0].value is not None
    assert ast.literal_eval(ctype_maps[0].value) == {
        "float": "torch::kFloat32",
        "double": "torch::kFloat64",
        "int": "torch::kInt32",
        "int32_t": "torch::kInt32",
        "long long": "torch::kInt64",
        "int64_t": "torch::kInt64",
        "int8_t": "torch::kInt8",
        "uint8_t": "torch::kUInt8",
    }

    canonical_assignments = [
        statement
        for node in router.body
        if isinstance(node, ast.Try)
        for statement in node.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "canonical_ctype"
            for target in statement.targets
        )
    ]
    assert len(canonical_assignments) == 1
    assert ast.unparse(canonical_assignments[0].value) == (
        "dtype_to_c_datatype(context.result_assembler.dtype).value"
    )
    canonical_branches = [node for node in router.body if isinstance(node, ast.If)]
    assert len(canonical_branches) == 1
    assert ast.unparse(canonical_branches[0].test) == (
        "context.workspace_ctype == canonical_ctype"
    )
    typed_calls = calls(router, "emit_compressed_value_allocation")
    assert len(typed_calls) == 1
    assert typed_calls[0].keywords == []
    assert len(typed_calls[0].args) == 1
    assert ast.unparse(typed_calls[0].args[0]) == (
        "_total_reference(context.compressed_levels[-1])"
    )
    fallback_calls = calls(router, "_legacy_value_allocation")
    assert len(fallback_calls) == 1
    assert ast.unparse(router.body[-1]) == (
        "return [_legacy_value_allocation(context)]"
    )

    builder_calls = calls(builder, "_value_allocation")
    assert len(builder_calls) == 1
    builder_parents = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "extend"
        and any(call is builder_calls[0] for call in ast.walk(node))
    ]
    assert len(builder_parents) == 1


def test_known_nnz_tensor_size_cannot_return_to_raw_statements() -> None:
    path = _COMPILER_ROOT / "cin_lowerer.py"
    raw_violations: list[tuple[int, str]] = []
    for call in _llir_constructor_calls(path, "RawStmt"):
        spelling = ast.unparse(call)
        if "_known_nnz" in spelling or ".size(0)" in spelling:
            raw_violations.append((call.lineno, spelling))

    member_calls = []
    for call in _llir_constructor_calls(path, "MemberCall"):
        fields = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        member = fields.get("member")
        if isinstance(member, ast.Constant) and member.value == "size":
            member_calls.append(fields)

    assert raw_violations == []
    assert len(member_calls) == 1
    fields = member_calls[0]
    assert set(fields) == {"base", "member", "args"}
    assert _is_llir_constructor(fields["base"], "Var")
    base = fields["base"]
    assert isinstance(base, ast.Call)
    base_fields = {
        keyword.arg: keyword.value
        for keyword in base.keywords
        if keyword.arg is not None
    }
    assert ast.unparse(base_fields["name"]) == "sparse_values_tensor"
    assert ast.unparse(base_fields["type"]) == "llir.DataType.TORCH_TENSOR"
    assert ast.unparse(fields["args"]) == (
        "(llir.Literal(value=0, data_type=llir.DataType.INT64),)"
    )


def test_loop_invariant_factor_materialization_cannot_return_to_raw_statements() -> (
    None
):
    path = _COMPILER_ROOT / "loop_invariant_factor_pass.py"
    source = path.read_text()

    assert _llir_constructor_calls(path, "RawStmt") == []
    assert len(_llir_constructor_calls(path, "VarInit")) == 1
    assert len(_llir_constructor_calls(path, "Assign")) == 2
    assert len(_llir_constructor_calls(path, "Var")) == 3
    assert source.count("op=llir.AssignOp.MUL_ASSIGN") == 1
    assert "def _render_expression(" not in source


def test_compressed_phase_state_cannot_return_to_raw_statements() -> None:
    """Lock every mutable count/fill state producer on typed statements."""

    compressed_path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    result_write_path = _COMPILER_ROOT / "result_write_pass.py"
    raw_violations: list[tuple[str, int, str]] = []
    for path in (compressed_path, result_write_path):
        for call in _llir_constructor_calls(path, "RawStmt"):
            spelling = ast.unparse(call)
            if re.search(r"_(?:cnt|pos|prev)\{(?:level|parent_level)\}", spelling):
                raw_violations.append((path.name, call.lineno, spelling))

    state_initializers: Counter[str] = Counter()
    for call in _llir_constructor_calls(compressed_path, "VarInit"):
        var_expression = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "var"),
            call.args[0] if call.args else None,
        )
        if var_expression is None or not _is_llir_constructor(var_expression, "Var"):
            continue
        var_call = var_expression
        assert isinstance(var_call, ast.Call)
        name_expression = _var_name_expression(var_call)
        if name_expression is None:
            continue
        fragments = _static_string_fragments(name_expression)
        for prefix in ("_cnt", "_pos", "_prev"):
            if prefix in fragments:
                state_initializers[prefix] += 1

    assert raw_violations == []
    assert state_initializers == {"_cnt": 1, "_pos": 1, "_prev": 2}
    assert len(_llir_constructor_calls(result_write_path, "Increment")) == 6
    state_references: Counter[tuple[str, str]] = Counter()
    for call in _llir_constructor_calls(result_write_path, "_phase_state"):
        assert call.keywords == []
        assert len(call.args) == 2
        state_references[(ast.unparse(call.args[0]), ast.unparse(call.args[1]))] += 1
    assert state_references == {
        ("'_cnt'", "level"): 3,
        ("'_cnt'", "parent_level"): 1,
        ("'_pos'", "level"): 4,
        ("'_pos'", "parent_level"): 1,
        ("'_prev'", "level"): 3,
        ("prefix", "level"): 1,
    }
    progress_conditions: Counter[tuple[str, str]] = Counter()
    for call in _llir_constructor_calls(result_write_path, "_progress_condition"):
        assert call.keywords == []
        assert len(call.args) == 2
        progress_conditions[(ast.unparse(call.args[0]), ast.unparse(call.args[1]))] += 1
    assert progress_conditions == {
        ("'_cnt'", "level"): 1,
        ("'_pos'", "level"): 1,
    }

    previous_assignments = 0
    for call in _llir_constructor_calls(result_write_path, "Assign"):
        target = _assign_target_expression(call)
        if target is None or not _is_llir_constructor(target, "_phase_state"):
            continue
        if "_prev" in _static_string_fragments(target):
            previous_assignments += 1
    assert previous_assignments == 2


def test_compressed_offset_family_is_structured() -> None:
    """Lock the exact W3 and W6-W9 production templates outside RawStmt."""

    path = _COMPILER_ROOT / "compressed_where_openmp_pass.py"
    tree = ast.parse(path.read_text())

    def fields(call: ast.Call) -> dict[str, ast.expr]:
        return {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }

    def calls(function: ast.FunctionDef, constructor: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == constructor
        ]

    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    count_builder = functions["_count_and_offset_statements"]
    prefix_builder = functions["_prefix_sum_statements"]
    prefix_loop_builder = functions["_prefix_sum_loop"]

    raw_violations: list[tuple[int, str]] = []
    for call in _llir_constructor_calls(path, "RawStmt"):
        spelling = ast.unparse(call)
        fragments = _static_string_fragments(call)
        if any(
            marker in fragments
            for marker in (
                "_base{level}",
                "std::vector<int> _count{level}",
                "std::vector<int64_t> _offset{level}",
                "_offset{level}[_i + 1]",
                "int64_t _total{level} = _offset{level}[",
            )
        ):
            raw_violations.append((call.lineno, spelling))

    count_initializations = calls(count_builder, "DirectInit")
    offset_initializations = calls(prefix_builder, "DirectInit")
    prefix_loops = calls(prefix_loop_builder, "ForLoop")
    total_initializations = [
        call
        for call in calls(prefix_builder, "VarInit")
        if (var := fields(call).get("var")) is not None
        and ast.unparse(var) == "_total_reference(level)"
    ]

    assert raw_violations == []
    assert len(count_initializations) == 1
    assert len(offset_initializations) == 1
    assert len(prefix_loops) == 1
    assert len(total_initializations) == 1

    count_fields = fields(count_initializations[0])
    assert set(count_fields) == {"var", "args"}
    assert ast.unparse(count_fields["var"]) == "_count_reference(level)"
    assert isinstance(count_fields["args"], ast.Tuple)
    count_args = count_fields["args"].elts
    assert len(count_args) == 2
    assert ast.unparse(count_args[0]) == (
        "llir.Cast(_cell_count_reference(cell_count), llir.DataType.SIZE_T)"
    )
    assert ast.unparse(count_args[1]) == "llir.Literal(0, llir.DataType.INT)"

    offset_fields = fields(offset_initializations[0])
    assert set(offset_fields) == {"var", "args"}
    assert ast.unparse(offset_fields["var"]) == "_offset_reference(level)"
    assert isinstance(offset_fields["args"], ast.Tuple)
    offset_args = offset_fields["args"].elts
    assert len(offset_args) == 1
    assert ast.unparse(offset_args[0]) == (
        "llir.Add(llir.Cast(_cell_count_reference(cell_count), "
        "llir.DataType.SIZE_T), llir.Literal(1, llir.DataType.INT))"
    )

    loop_fields = fields(prefix_loops[0])
    assert set(loop_fields) == {"init", "cond", "update", "body"}
    assert ast.unparse(loop_fields["init"]) == (
        "llir.VarInit(var=_prefix_index_reference(), "
        "value=llir.Literal(0, llir.DataType.INT))"
    )
    assert ast.unparse(loop_fields["cond"]) == (
        "llir.BinOp('<', _prefix_index_reference(), "
        "_cell_count_reference(cell_count))"
    )
    assert ast.unparse(loop_fields["update"]) == (
        "llir.Increment(_prefix_index_reference())"
    )
    assert len(calls(prefix_loop_builder, "Assign")) == 1
    assert len(calls(prefix_loop_builder, "ArrayAccess")) == 3
    assert len(calls(prefix_loop_builder, "Add")) == 2
    assert len(calls(prefix_loop_builder, "Increment")) == 1
    assert (
        len(
            [
                call
                for call in ast.walk(prefix_loop_builder)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_prefix_index_reference"
            ]
        )
        == 6
    )

    total_fields = fields(total_initializations[0])
    assert set(total_fields) == {"var", "value"}
    assert ast.unparse(total_fields["var"]) == "_total_reference(level)"
    assert ast.unparse(total_fields["value"]) == (
        "llir.ArrayAccess(array=_offset_reference(level), "
        "index=_cell_count_reference(cell_count))"
    )

    base_initializers: list[dict[str, ast.expr]] = []
    for call in _llir_constructor_calls(path, "VarInit"):
        initializer_fields = fields(call)
        target = initializer_fields.get("var")
        if target is None or not _is_llir_constructor(target, "Var"):
            continue
        assert isinstance(target, ast.Call)
        name_expression = _var_name_expression(target)
        if name_expression is not None and "_base" in _static_string_fragments(
            name_expression
        ):
            base_initializers.append(initializer_fields)

    assert len(base_initializers) == 1

    base_fields = base_initializers[0]
    assert set(base_fields) == {"var", "value"}
    target = base_fields["var"]
    assert isinstance(target, ast.Call)
    target_fields = {
        keyword.arg: keyword.value
        for keyword in target.keywords
        if keyword.arg is not None
    }
    assert ast.unparse(target_fields["name"]) == "f'_base{level}'"
    assert ast.unparse(target_fields["type"]) == "llir.DataType.INT64"

    value = base_fields["value"]
    assert _is_llir_constructor(value, "ArrayAccess")
    assert isinstance(value, ast.Call)
    access_fields = {
        keyword.arg: keyword.value
        for keyword in value.keywords
        if keyword.arg is not None
    }
    assert set(access_fields) == {"array", "index"}

    array = access_fields["array"]
    assert _is_llir_constructor(array, "Var")
    assert isinstance(array, ast.Call)
    array_fields = {
        keyword.arg: keyword.value
        for keyword in array.keywords
        if keyword.arg is not None
    }
    assert ast.unparse(array_fields["name"]) == "f'_offset{level}'"
    assert ast.unparse(array_fields["type"]) == "llir.DataType.STD_VECTOR_INT"

    # The subscript is the outer cell index, and it is no longer a reference to
    # the loop variable: a loop over a STORED level has a variable that is a
    # position, so the caller states which cell an iteration assembles and the
    # pass detaches a fresh copy of that expression per use.  What this locks is
    # unchanged in kind -- the subscript is built by the pass's own typed builder
    # rather than by interpolating text -- and the builder is the one place a
    # detached copy is made.
    index = access_fields["index"]
    assert isinstance(index, ast.Call)
    assert ast.unparse(index) == "_cell_index_expression(context, cell_index)"


def test_no_direct_assign_target_reintroduces_a_string_expression() -> None:
    """Keep migrated production lvalues on the one structured representation."""

    violations: list[tuple[str, str, str]] = []
    allowed_members: list[tuple[str, str]] = []
    for path in sorted(_COMPILER_ROOT.glob("*.py")):
        for assign_call in _llir_constructor_calls(path, "Assign"):
            target = _assign_target_expression(assign_call)
            if target is None or not _is_llir_constructor(target, "Var"):
                continue
            target_call = target
            assert isinstance(target_call, ast.Call)
            name_expression = _var_name_expression(target_call)
            if name_expression is None:
                continue
            category = _string_expression_category(name_expression)
            spelling = ast.unparse(name_expression)
            if category == "member":
                allowed_members.append((path.name, spelling))
            elif category is not None:
                violations.append((path.name, category, ast.unparse(name_expression)))

    assert violations == []
    assert allowed_members == []


def test_generic_string_rewrite_compatibility_budget_is_explicit() -> None:
    markers = {
        "cin_lowerer.py": (
            "rewritten_name = rewritten_name.replace(old, new)",
            "stmt.code = stmt.code.replace(old, new)",
        ),
        "dense_pointer_hoist_pass.py": (
            "rewritten_statement = llir.FunctionCallStmt(",
            "raw_statement.code = code",
        ),
        "dynamic_vector_access_pass.py": (
            "rewritten.name = self._rewrite_name(rewritten.name)",
        ),
        "single_iteration_loop_pass.py": (
            "rewritten_statements[index] = llir.FunctionCallStmt(",
            "raw_statement.code = code",
        ),
    }

    for filename, expected_markers in markers.items():
        source = (_COMPILER_ROOT / filename).read_text()
        for marker in expected_markers:
            assert source.count(marker) == 1
    assert sum(len(expected_markers) for expected_markers in markers.values()) == 7


def test_workspace_insert_rewrite_is_exact_and_never_lexical() -> None:
    source = (_COMPILER_ROOT / "compressed_where_openmp_pass.py").read_text()

    assert source.count("if call.name == self._old:") == 1
    assert source.count("return llir.FunctionCallStmt(") == 1
    assert source.count("name=self._new,") == 1
    assert ".replace(self._old, self._new)" not in source
