from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, fields, is_dataclass
from typing import Any

import pytest

from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAccess,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_analysis import (
    CINAnalysis,
    FrozenMap,
    analyze_cin,
    canonical_cin_dump,
    full_cin_verification,
    normalize_cin,
    verify_cin,
)
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.identity import AccessId, IndexId, NodeId, SymbolId
from scorch.compiler.legacy_cin_adapter import legacy_cin_working_copy
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.format import LevelFormat, TensorFormat


@dataclass(frozen=True)
class _ReductionNodes:
    i: IndexVar
    k: IndexVar
    result: TensorVar
    left: TensorVar
    right: TensorVar
    result_access: TensorAccess
    left_access: TensorAccess
    right_access: TensorAccess
    assignment: TensorAssign


def _reduction_program() -> tuple[ForAll, _ReductionNodes]:
    i = IndexVar("i")
    k = IndexVar("k")
    result = TensorVar("C", fmt="d")
    left = TensorVar("A", fmt="dd")
    right = TensorVar("B", fmt="d")
    result_access = result[i]
    left_access = left[i, k]
    right_access = right[k]
    assignment = TensorAssign(
        result_access,
        left_access * right_access,
        op=Operation.ADD,
    )
    program = ForAll(i, ForAll(k, assignment))
    return program, _ReductionNodes(
        i=i,
        k=k,
        result=result,
        left=left,
        right=right,
        result_access=result_access,
        left_access=left_access,
        right_access=right_access,
        assignment=assignment,
    )


def _same_name_nested_program() -> tuple[ForAll, IndexVar, IndexVar]:
    outer_i = IndexVar("i")
    inner_i = IndexVar("i")
    result = TensorVar("C", fmt="dd")
    source = TensorVar("A", fmt="dd")
    assignment = TensorAssign(
        result[outer_i, inner_i],
        source[outer_i, inner_i],
    )
    return ForAll(outer_i, ForAll(inner_i, assignment)), outer_i, inner_i


def _assert_deeply_immutable(value: object, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))

    assert not isinstance(value, (dict, list, set, bytearray))
    if isinstance(value, Mapping):
        assert isinstance(value, FrozenMap)
        for key, item in value.items():
            _assert_deeply_immutable(key, seen)
            _assert_deeply_immutable(item, seen)
        for stored_value in getattr(value, "__dict__", {}).values():
            _assert_deeply_immutable(stored_value, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__")
        assert params.frozen
        for item in fields(value):
            _assert_deeply_immutable(getattr(value, item.name), seen)
    elif isinstance(value, (tuple, frozenset)):
        for item in value:
            _assert_deeply_immutable(item, seen)


def _assert_structured_diagnostics(
    error: pytest.ExceptionInfo[VerificationError],
) -> set[str]:
    diagnostics = error.value.diagnostics
    assert isinstance(diagnostics, tuple)
    assert diagnostics
    for diagnostic in diagnostics:
        code = getattr(diagnostic, "code", None)
        path = getattr(diagnostic, "path", None)
        message = getattr(diagnostic, "message", None)
        assert isinstance(code, str) and code
        assert isinstance(path, tuple)
        assert isinstance(message, str) and message
    return {str(getattr(diagnostic, "code")) for diagnostic in diagnostics}


def test_cin_mutable_state_is_owned_per_instance() -> None:
    first_i = IndexVar("i")
    second_i = IndexVar("i")
    first_tensor = TensorVar("A", fmt="d")
    second_tensor = TensorVar("A", fmt="d")
    first_workspace = Workspace("tmp", dim=1, dense=True)
    second_workspace = Workspace("tmp", dim=1, dense=True)
    first_access = first_tensor[first_i]
    first_expr = first_access * first_access

    first = ForAll(first_i, TensorAssign(first_access, first_access))
    second_access = second_tensor[second_i]
    second_expr = second_access * second_access
    second = ForAll(second_i, TensorAssign(second_access, second_access))

    first.no_tile_list.append(first_i)
    first_i.tensor_accesses.append(first_access)
    first_workspace.workspace_accesses.append(first_workspace[first_i])
    assert first_tensor.mode_order is not None
    first_tensor.mode_order.append(17)
    first_expr.no_tile_list.append(first_i)

    assert second.no_tile_list == []
    assert second_i.tensor_accesses == []
    assert second_workspace.workspace_accesses == []
    assert second_tensor.mode_order == [0]
    assert second_expr.no_tile_list == []
    assert first.node_id != second.node_id


def test_access_construction_has_no_backreference_side_effects() -> None:
    i = IndexVar("i")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt="d")
    workspace = Workspace("tmp", dim=1, dense=True)

    lhs = result[i]
    rhs = source[i]
    workspace_access = workspace[i]
    TensorAssign(lhs, rhs)

    assert i.tensor_accesses == []
    assert workspace.workspace_accesses == []
    assert result._assignment is None
    assert source._assignment is None
    assert workspace_access not in workspace.workspace_accesses


def test_analysis_is_pure_id_keyed_and_tracks_access_order() -> None:
    program, nodes = _reduction_program()
    inner = program.stmt
    assert isinstance(inner, ForAll)
    assignment = nodes.assignment
    assert isinstance(assignment, TensorAssign)
    before = canonical_cin_dump(program)

    analysis = analyze_cin(program)

    assert canonical_cin_dump(program) == before
    assert program.parent is None
    assert inner.parent is None
    assert assignment.parent is None
    assert nodes.i.tensor_accesses == []
    assert nodes.k.tensor_accesses == []
    assert analysis.root_id == program.node_id
    assert analysis.parents[inner.node_id].parent_id == program.node_id
    assert analysis.parents[assignment.node_id].parent_id == inner.node_id
    assert all(isinstance(key, NodeId) for key in analysis.parents)
    assert all(isinstance(key, SymbolId) for key in analysis.symbol_definitions)
    assert all(isinstance(key, SymbolId) for key in analysis.symbol_uses)
    assert all(isinstance(key, IndexId) for key in analysis.index_definitions)
    assert all(isinstance(key, IndexId) for key in analysis.index_uses)
    assert all(isinstance(key, AccessId) for key in analysis.accesses)
    assert all(isinstance(key, SymbolId) for key in analysis.tensor_accesses)
    assert analysis.accesses[nodes.left_access.access_id].tensor_id == (
        nodes.left.symbol_id
    )
    assert analysis.accesses[nodes.left_access.access_id].index_ids == (
        nodes.i.index_id,
        nodes.k.index_id,
    )
    assert analysis.tensor_accesses[nodes.result.symbol_id] == (
        nodes.result_access.access_id,
    )
    assert analysis.access_order == (
        nodes.result_access.access_id,
        nodes.left_access.access_id,
        nodes.right_access.access_id,
    )

    repeated = analyze_cin(program)
    assert repeated == analysis
    assert repeated is not analysis


def test_analysis_result_is_deeply_immutable() -> None:
    program, _ = _reduction_program()
    analysis = analyze_cin(program)
    mapping_fields = (
        "parents",
        "node_scopes",
        "scope_parents",
        "symbol_definitions",
        "symbol_uses",
        "index_definitions",
        "index_uses",
        "accesses",
        "tensor_accesses",
    )
    tuple_fields = (
        "access_occurrences",
        "access_order",
        "free_index_ids",
        "reduction_index_ids",
        "diagnostics",
    )

    assert isinstance(analysis, CINAnalysis)
    assert all(
        isinstance(getattr(analysis, name), FrozenMap) for name in mapping_fields
    )
    assert all(isinstance(getattr(analysis, name), tuple) for name in tuple_fields)
    _assert_deeply_immutable(analysis)

    mutable_analysis: Any = analysis
    mutable_parents: Any = analysis.parents
    with pytest.raises(FrozenInstanceError):
        mutable_analysis.root_id = NodeId(-1)
    with pytest.raises(TypeError):
        mutable_parents[NodeId(-1)] = None


def test_same_name_nested_bindings_keep_distinct_identity_and_scopes() -> None:
    program, outer_i, inner_i = _same_name_nested_program()
    analysis = verify_cin(program)

    assert outer_i.name == inner_i.name == "i"
    assert outer_i.index_id != inner_i.index_id
    assert outer_i.index_id in analysis.index_definitions
    assert inner_i.index_id in analysis.index_definitions
    assert (
        analysis.index_definitions[outer_i.index_id]
        != analysis.index_definitions[inner_i.index_id]
    )
    outer_binding = analysis.index_definitions[outer_i.index_id].bindings[0]
    inner_binding = analysis.index_definitions[inner_i.index_id].bindings[0]
    assert outer_binding.scope_id != inner_binding.scope_id
    assert analysis.scope_parents[inner_binding.scope_id] == outer_binding.scope_id
    assert outer_i.index_id in analysis.index_uses
    assert inner_i.index_id in analysis.index_uses
    assert len(analysis.symbol_definitions) == 2

    with pytest.raises(VerificationError, match="distinct IndexId"):
        Scheduler.apply_schedule(program, Schedule())


def test_same_name_symbols_in_distinct_scopes_keep_distinct_identity() -> None:
    i = IndexVar("i")
    source = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    first = Workspace("tmp", dim=1, dense=True)
    second = Workspace("tmp", dim=1, dense=True)

    first_region = Where(
        producer=TensorAssign(first[i], source[i]),
        consumer=TensorAssign(result[i], first[i]),
    )
    second_region = Where(
        producer=TensorAssign(second[i], source[i]),
        consumer=TensorAssign(result[i], second[i]),
    )
    analysis = verify_cin(ForAll(i, Where(first_region, second_region)))

    first_definition = analysis.symbol_definitions[first.symbol_id]
    second_definition = analysis.symbol_definitions[second.symbol_id]
    assert first.symbol_id != second.symbol_id
    assert first_definition.display_name == second_definition.display_name == "tmp"
    assert first_definition.scope_id != second_definition.scope_id

    with pytest.raises(VerificationError, match="distinct SymbolId"):
        Scheduler.apply_schedule(
            ForAll(i, Where(first_region, second_region)), Schedule()
        )


def test_free_and_reduction_ids_are_classified_from_stable_identity() -> None:
    program, nodes = _reduction_program()
    analysis = verify_cin(program)

    assert analysis.free_index_ids == (nodes.i.index_id,)
    assert analysis.reduction_index_ids == (nodes.k.index_id,)
    assert set(analysis.free_index_ids).isdisjoint(analysis.reduction_index_ids)


def test_canonical_dump_renumbers_ids_and_normalization_is_idempotent() -> None:
    first, _, _ = _same_name_nested_program()
    # Consume unrelated allocation IDs between equivalent independent programs.
    TensorVar("unrelated", fmt="d")[IndexVar("unused")]
    second, _, _ = _same_name_nested_program()

    first_normalized = normalize_cin(first)
    first_again = normalize_cin(first_normalized)
    second_normalized = normalize_cin(second)

    assert first_normalized is not first
    assert first_again is not first_normalized
    assert canonical_cin_dump(first_normalized) == canonical_cin_dump(first_again)
    assert canonical_cin_dump(first_normalized) == canonical_cin_dump(second_normalized)


def test_workspace_normalization_is_idempotent_and_strips_backreferences() -> None:
    i = IndexVar("i")
    source = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    workspace = Workspace("tmp", dim=1, dense=True)
    program = ForAll(
        i,
        Where(
            producer=TensorAssign(workspace[i], source[i]),
            consumer=TensorAssign(result[i], workspace[i]),
        ),
    )

    first = normalize_cin(program)
    second = normalize_cin(first)

    assert canonical_cin_dump(first) == canonical_cin_dump(second)
    assert all(not index_var.tensor_accesses for index_var in first.index_vars)
    assert all(
        not workspace_var.workspace_accesses for workspace_var in first.get_workspaces()
    )


def test_canonical_dump_includes_parallel_and_level_width_metadata() -> None:
    def build(*, parallel: bool, bit_width: int) -> ForAll:
        i = IndexVar("i")
        level = TensorFormat([LevelFormat("d", bit_width=bit_width)])
        result = TensorVar("C", fmt=level)
        source = TensorVar("A", fmt=level)
        return ForAll(
            i,
            TensorAssign(result[i], source[i]),
            parallel=parallel,
        )

    baseline = canonical_cin_dump(build(parallel=False, bit_width=32))
    assert baseline != canonical_cin_dump(build(parallel=True, bit_width=32))
    assert baseline != canonical_cin_dump(build(parallel=False, bit_width=64))


def test_verifier_reports_rank_mismatch_with_structured_diagnostics() -> None:
    i = IndexVar("i")
    result = TensorVar("C", fmt="dd")
    source = TensorVar("A", fmt="dd")
    program = ForAll(i, TensorAssign(result[i], source[i]))

    with pytest.raises(VerificationError) as error:
        verify_cin(program)

    assert "tensor_access_rank_mismatch" in _assert_structured_diagnostics(error)


def test_verifier_reports_dangling_and_mismatched_stable_references() -> None:
    program, nodes = _reduction_program()
    result_access = nodes.result_access
    left_access = nodes.left_access
    result_access.tensor_id = SymbolId(10**9)
    left_access.index_ids = (IndexId(10**9), nodes.k.index_id)

    with pytest.raises(VerificationError) as error:
        verify_cin(program)

    codes = _assert_structured_diagnostics(error)
    assert {"dangling_symbol_reference", "symbol_reference_mismatch"} <= codes
    assert {"dangling_index_reference", "index_reference_mismatch"} <= codes


def test_verifier_reports_duplicate_access_reference() -> None:
    i = IndexVar("i")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt="d")
    shared = source[i]
    program = ForAll(i, TensorAssign(result[i], shared * shared))

    with pytest.raises(VerificationError) as error:
        verify_cin(program)

    assert "duplicate_access_reference" in _assert_structured_diagnostics(error)


@pytest.mark.parametrize(
    "field",
    ("node_id", "index_id", "symbol_id", "access_id", "tensor_id", "index_ids"),
)
def test_verifier_reports_untyped_ids_with_structured_diagnostics(field: str) -> None:
    program, nodes = _reduction_program()
    if field == "node_id":
        nodes.assignment.node_id = 7  # type: ignore[assignment]
    elif field == "index_id":
        nodes.i.index_id = 7  # type: ignore[assignment]
    elif field == "symbol_id":
        nodes.left.symbol_id = 7  # type: ignore[assignment]
    elif field == "access_id":
        nodes.left_access.access_id = 7  # type: ignore[assignment]
    elif field == "tensor_id":
        nodes.left_access.tensor_id = 7  # type: ignore[assignment]
    else:
        nodes.left_access.index_ids = [nodes.i.index_id, nodes.k.index_id]  # type: ignore[assignment]

    analysis = analyze_cin(program)
    _assert_deeply_immutable(analysis)
    with pytest.raises(VerificationError) as error:
        verify_cin(program)

    assert any(
        code.startswith("invalid_") for code in _assert_structured_diagnostics(error)
    )


@pytest.mark.parametrize(
    ("identity_kind", "expected_code"),
    (
        ("node", "duplicate_node_id"),
        ("access", "duplicate_access_id"),
        ("symbol", "duplicate_symbol_id"),
        ("index", "duplicate_index_id"),
    ),
)
def test_verifier_reports_duplicate_typed_ids(
    identity_kind: str,
    expected_code: str,
) -> None:
    program, nodes = _reduction_program()
    if identity_kind == "node":
        nodes.left.node_id = nodes.result.node_id
    elif identity_kind == "access":
        nodes.left_access.access_id = nodes.result_access.access_id
    elif identity_kind == "symbol":
        nodes.left.symbol_id = nodes.result.symbol_id
        nodes.left_access.tensor_id = nodes.result.symbol_id
    else:
        nodes.k.index_id = nodes.i.index_id
        nodes.left_access.index_ids = (nodes.i.index_id, nodes.i.index_id)
        nodes.right_access.index_ids = (nodes.i.index_id,)

    with pytest.raises(VerificationError) as error:
        verify_cin(program)

    assert expected_code in _assert_structured_diagnostics(error)


def test_verifier_rejects_extent_and_reduction_classification_mismatches() -> None:
    i = IndexVar("i")
    k = IndexVar("k")
    result = TensorVar("C", fmt="d", shape=(4,))
    source = TensorVar("A", fmt="d", shape=(5,))
    extent_program = ForAll(i, TensorAssign(result[i], source[i]))

    with pytest.raises(VerificationError) as extent_error:
        verify_cin(extent_program)
    assert "index_extent_mismatch" in _assert_structured_diagnostics(extent_error)

    missing_reduction = ForAll(
        i,
        ForAll(k, TensorAssign(TensorVar("D", fmt="d")[i], source[k])),
    )
    with pytest.raises(VerificationError) as reduction_error:
        verify_cin(missing_reduction)
    assert "index_classification_inconsistent" in _assert_structured_diagnostics(
        reduction_error
    )

    unused = ForAll(
        i,
        ForAll(k, TensorAssign(TensorVar("E", fmt="d")[i], source[i])),
    )
    unused_analysis = analyze_cin(unused)
    assert k.index_id not in unused_analysis.reduction_index_ids
    with pytest.raises(VerificationError) as unused_error:
        verify_cin(unused)
    assert "unused_index_binding" in _assert_structured_diagnostics(unused_error)


def test_full_verification_is_explicit_at_debug_compiler_boundary() -> None:
    i = IndexVar("i")
    result = TensorVar("C", fmt="dd")
    source = TensorVar("A", fmt="dd")
    invalid = ForAll(i, TensorAssign(result[i], source[i]))

    normalize_cin(invalid)
    with full_cin_verification():
        with pytest.raises(VerificationError, match="tensor_access_rank_mismatch"):
            Scheduler.apply_schedule(invalid, Schedule())


def test_verifier_reports_out_of_scope_index_reference() -> None:
    producer_i = IndexVar("i")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt="d")
    workspace = Workspace("tmp", dim=1, dense=True)
    program = Where(
        producer=ForAll(
            producer_i,
            TensorAssign(workspace[producer_i], source[producer_i]),
        ),
        consumer=TensorAssign(result[producer_i], workspace[producer_i]),
    )

    with pytest.raises(VerificationError) as error:
        verify_cin(program)

    assert "index_reference_out_of_scope" in _assert_structured_diagnostics(error)


def test_legacy_adapter_materializes_backreferences_only_on_private_copy() -> None:
    program, nodes = _reduction_program()
    inner = program.stmt
    assert isinstance(inner, ForAll)
    assert inner.parent is None
    assert nodes.i.tensor_accesses == []

    working = legacy_cin_working_copy(program)
    assert isinstance(working, ForAll)
    working_inner = working.stmt
    assert isinstance(working_inner, ForAll)
    assert working is not program
    assert working_inner.parent is working
    assert working_inner.stmt.parent is working_inner
    assert working.index_var.tensor_accesses

    working.index_var.tensor_accesses.clear()
    working_inner.parent = None
    assert nodes.i.tensor_accesses == []
    assert inner.parent is None
