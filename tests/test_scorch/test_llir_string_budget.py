"""Audited compatibility budget after compact-result read migration."""

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
        "lower_expr",
        "upper_expr",
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
        "cin_lowerer.py": 199,
        "compressed_where_openmp_pass.py": 5,
        "dense_pointer_hoist_pass.py": 3,
        "dynamic_vector_access_pass.py": 1,
        "iter_lattice.py": 35,
        "iterator.py": 19,
        "llir_traversal.py": 1,
        "result_write_pass.py": 5,
        "schedule_lowerer.py": 96,
        "single_iteration_loop_pass.py": 1,
    }
    assert sum(constructor_counts.values()) == 374
    assert unclassified_counts == {
        "cin.py": 9,
        "cin_lowerer.py": 152,
        "compressed_where_openmp_pass.py": 5,
        "dense_pointer_hoist_pass.py": 3,
        "dynamic_vector_access_pass.py": 1,
        "iter_lattice.py": 35,
        "iterator.py": 13,
        "llir_traversal.py": 1,
        "result_write_pass.py": 5,
        "schedule_lowerer.py": 94,
        "single_iteration_loop_pass.py": 1,
    }
    assert sum(unclassified_counts.values()) == 319
    assert known_indirect == {
        ("cin_lowerer.py", "expr.name.replace(old, new)"): 1,
        ("dense_pointer_hoist_pass.py", "name"): 1,
        ("llir_traversal.py", "node.name"): 1,
        ("schedule_lowerer.py", "lower_expr"): 1,
        ("schedule_lowerer.py", "upper_expr"): 1,
        ("schedule_lowerer.py", "prefix_extent"): 2,
        ("schedule_lowerer.py", "zero_value"): 1,
        ("single_iteration_loop_pass.py", "name"): 1,
    }
    assert sum(known_indirect.values()) == 9

    assert totals == {
        "subscript": 27,
        "call": 13,
        "member": 7,
        "initializer": 3,
        "qualified": 3,
        "ternary": 1,
        "arithmetic": 1,
    }
    assert sum(totals.values()) == 55
    assert per_file == {
        ("cin_lowerer.py", "subscript"): 21,
        ("cin_lowerer.py", "call"): 11,
        ("cin_lowerer.py", "member"): 7,
        ("cin_lowerer.py", "initializer"): 3,
        ("cin_lowerer.py", "qualified"): 3,
        ("cin_lowerer.py", "ternary"): 1,
        ("cin_lowerer.py", "arithmetic"): 1,
        ("iterator.py", "subscript"): 6,
        ("schedule_lowerer.py", "call"): 2,
    }


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
