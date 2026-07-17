"""Audited compatibility budget for the current Phase-3 structured IR."""

import ast
from collections import Counter
from pathlib import Path
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
        "expr.name.replace(old, new)",
        "name",
        "node.name",
        "prefix_extent",
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
                if spelling in known_indirect_names:
                    known_indirect[(path.name, spelling)] += 1

    assert constructor_counts == {
        "cin.py": 9,
        "cin_lowerer.py": 202,
        "compressed_where_openmp_pass.py": 5,
        "dense_pointer_hoist_pass.py": 3,
        "dynamic_vector_access_pass.py": 1,
        "iter_lattice.py": 35,
        "iterator.py": 23,
        "llir_traversal.py": 1,
        "result_write_pass.py": 5,
        "schedule_lowerer.py": 99,
        "single_iteration_loop_pass.py": 1,
    }
    assert sum(constructor_counts.values()) == 384
    assert unclassified_counts == {
        "cin.py": 9,
        "cin_lowerer.py": 175,
        "compressed_where_openmp_pass.py": 5,
        "dense_pointer_hoist_pass.py": 3,
        "dynamic_vector_access_pass.py": 1,
        "iter_lattice.py": 35,
        "iterator.py": 21,
        "llir_traversal.py": 1,
        "result_write_pass.py": 5,
        "schedule_lowerer.py": 97,
        "single_iteration_loop_pass.py": 1,
    }
    assert sum(unclassified_counts.values()) == 353
    assert known_indirect == {
        ("cin_lowerer.py", "expr.name.replace(old, new)"): 1,
        ("dense_pointer_hoist_pass.py", "name"): 1,
        ("llir_traversal.py", "node.name"): 1,
        ("schedule_lowerer.py", "prefix_extent"): 2,
        ("schedule_lowerer.py", "zero_value"): 1,
        ("single_iteration_loop_pass.py", "name"): 1,
    }
    assert sum(known_indirect.values()) == 7

    assert totals == {
        "subscript": 15,
        "call": 8,
        "member": 3,
        "initializer": 1,
        "qualified": 3,
        "ternary": 1,
    }
    assert sum(totals.values()) == 31
    assert per_file == {
        ("cin_lowerer.py", "subscript"): 13,
        ("cin_lowerer.py", "call"): 6,
        ("cin_lowerer.py", "member"): 3,
        ("cin_lowerer.py", "initializer"): 1,
        ("cin_lowerer.py", "qualified"): 3,
        ("cin_lowerer.py", "ternary"): 1,
        ("iterator.py", "subscript"): 2,
        ("schedule_lowerer.py", "call"): 2,
    }


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
    assert structured_moves == {"cin_lowerer.py": 5}
    assert remaining_opaque_calls == {
        ("cin_lowerer.py", "data_ptr"): 6,
        ("schedule_lowerer.py", "storage_data"): 2,
    }


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
    source = (_COMPILER_ROOT / "cin_lowerer.py").read_text()
    value_initialization = source.split("def emit_value_array_init", 1)[1].split(
        "def _get_mode_index_set", 1
    )[0]

    assert 'name=f"{{{self.name}_capacity}}"' not in value_initialization
    assert 'name=f"{{{self.known_nnz_var}}}"' not in value_initialization
    assert value_initialization.count("llir.Array(") == 2


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
        "cin_lowerer.py": 17,
        "compressed_where_openmp_pass.py": 19,
        "dense_pointer_hoist_pass.py": 1,
        "llir_traversal.py": 1,
        "loop_invariant_factor_pass.py": 2,
        "result_write_pass.py": 8,
        "schedule_lowerer.py": 3,
        "sparse_prefetch_pass.py": 1,
    }
    assert sum(counts.values()) == 52
    assert sum(counts.values()) - counts["llir_traversal.py"] == 51


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
    assert allowed_members == [
        ("cin_lowerer.py", "f'{self.name}.storage.index.mode_indices'"),
        ("cin_lowerer.py", "f'{self.name}.storage.value'"),
    ]


def test_generic_string_rewrite_compatibility_budget_is_explicit() -> None:
    markers = {
        "cin_lowerer.py": (
            "stmt.name = stmt.name.replace(old, new)",
            "stmt.code = stmt.code.replace(old, new)",
        ),
        "compressed_where_openmp_pass.py": (
            "rewritten.name = rewritten.name.replace(self._old, self._new)",
            "rewritten_call.name = call.name.replace(self._old, self._new)",
            "rewritten_raw.code = raw.code.replace(self._old, self._new)",
        ),
        "dense_pointer_hoist_pass.py": (
            "call.name = name",
            "raw_statement.code = code",
        ),
        "dynamic_vector_access_pass.py": (
            "rewritten.name = self._rewrite_name(rewritten.name)",
        ),
        "single_iteration_loop_pass.py": (
            "call.name = name",
            "raw_statement.code = code",
        ),
    }

    for filename, expected_markers in markers.items():
        source = (_COMPILER_ROOT / filename).read_text()
        for marker in expected_markers:
            assert source.count(marker) == 1
    assert sum(len(expected_markers) for expected_markers in markers.values()) == 10
