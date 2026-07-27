from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, fields, is_dataclass
from typing import Any

import pytest

import scorch.compiler.analysis_runner as analysis_runner_module  # type: ignore[import-untyped]
import scorch.compiler.cin_analysis as cin_analysis_module  # type: ignore[import-untyped]
from scorch.compiler.analysis_runner import (  # type: ignore[import-untyped]
    AnalysisRunner,
    COMMON_ANALYSIS_RUNNER,
)
from scorch.compiler.cin import (
    BinaryOp,
    ForAll,
    IndexStmt,
    IndexVar,
    IndexVarAdd,
    Operation,
    TensorAccess,
    TensorAssign,
    TensorVar,
    TileSizeVar,
    Where,
    Workspace,
    WorkspaceAccess,
)
from scorch.compiler.cin_analysis import (
    AccessKind,
    AccessLayoutInfo,
    AssignmentInfo,
    CINAnalysis,
    FrozenMap,
    analyze_cin,
    canonical_cin_dump,
    full_cin_verification,
    normalize_cin,
    verify_cin,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.identity import AccessId, IndexId, NodeId, SymbolId
from scorch.compiler.legacy_cin_adapter import legacy_cin_working_copy
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.format import LevelFormat, LevelType, TensorFormat


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


def test_workspace_index_rewrite_does_not_materialize_tile_backlink() -> None:
    logical = IndexVar("i")
    outer = IndexVar("i_out")
    inner = IndexVar("i_in")
    logical.expr = outer + inner
    TileSizeVar(outer, inner, size=4)
    workspace = Workspace("tmp", dim=1, dense=True)
    access = WorkspaceAccess(workspace, logical)

    access.update_indices([inner])

    assert access.indices == [inner]
    assert access.index_ids == (inner.index_id,)
    assert not workspace.is_tiled


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
    assert all(isinstance(key, AccessId) for key in analysis.access_layouts)
    assert all(isinstance(key, NodeId) for key in analysis.assignments)
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
    left_layout = analysis.access_layouts[nodes.left_access.access_id]
    assert isinstance(left_layout, AccessLayoutInfo)
    assert left_layout.access_id == nodes.left_access.access_id
    assert left_layout.tensor_id == nodes.left.symbol_id
    assert left_layout.logical_index_ids == (
        nodes.i.index_id,
        nodes.k.index_id,
    )
    assert left_layout.storage_index_ids == (
        nodes.i.index_id,
        nodes.k.index_id,
    )
    assert left_layout.level_types == (LevelType.DENSE, LevelType.DENSE)
    assert left_layout.physical_extents == (None, None)
    assert left_layout.scope_id == assignment.node_id
    assert left_layout.kind == AccessKind.READ
    assert not left_layout.is_workspace

    assignment_info = analysis.assignments[assignment.node_id]
    assert isinstance(assignment_info, AssignmentInfo)
    assert assignment_info.assignment_id == assignment.node_id
    assert assignment_info.lhs_access_id == nodes.result_access.access_id
    assert assignment_info.rhs_access_ids == (
        nodes.left_access.access_id,
        nodes.right_access.access_id,
    )
    assert assignment_info.update_op == Operation.ADD
    assert assignment_info.lhs_index_ids == (nodes.i.index_id,)
    assert assignment_info.reduction_index_ids == (nodes.k.index_id,)
    assert assignment_info.multiplicative_access_ids == (
        nodes.left_access.access_id,
        nodes.right_access.access_id,
    )
    assert analysis.assignment_order == (assignment.node_id,)

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
        "access_layouts",
        "assignments",
        "tensor_accesses",
    )
    tuple_fields = (
        "access_occurrences",
        "access_order",
        "assignment_order",
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


def test_analysis_tracks_typed_nondefault_storage_layout() -> None:
    i = IndexVar("i")
    k = IndexVar("k")
    result = TensorVar("C", shape=(5,), fmt="d")
    source = TensorVar(
        "A",
        shape=(7, 5),
        fmt="ds",
        mode_order=[1, 0],
    )
    result_access = result[i]
    source_access = source[i, k]
    assignment = TensorAssign(result_access, source_access, op=Operation.ADD)

    analysis = verify_cin(ForAll(i, ForAll(k, assignment)))

    layout = analysis.access_layouts[source_access.access_id]
    assert layout.logical_index_ids == (i.index_id, k.index_id)
    assert layout.storage_index_ids == (k.index_id, i.index_id)
    assert layout.level_types == (LevelType.DENSE, LevelType.COMPRESSED)
    assert layout.physical_extents == (7, 5)
    assert not layout.is_workspace
    assert analysis.assignments[assignment.node_id].multiplicative_access_ids == (
        source_access.access_id,
    )


def test_common_analysis_runner_is_frozen_stateless_and_typed() -> None:
    runner = AnalysisRunner()
    program, _ = _reduction_program()

    assert isinstance(COMMON_ANALYSIS_RUNNER, AnalysisRunner)
    assert fields(runner) == ()
    assert vars(runner) == {}
    assert runner == AnalysisRunner()
    assert isinstance(runner.analyze_cin(program), CINAnalysis)

    mutable_runner: Any = runner
    with pytest.raises(FrozenInstanceError):
        mutable_runner.cache = {}


def test_common_analysis_runner_recomputes_equal_distinct_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program, _ = _reduction_program()
    before = canonical_cin_dump(program)
    calls: list[IndexStmt] = []
    compute = analysis_runner_module._compute_cin_analysis

    def counted_compute(cin: IndexStmt) -> CINAnalysis:
        calls.append(cin)
        return compute(cin)

    monkeypatch.setattr(
        analysis_runner_module,
        "_compute_cin_analysis",
        counted_compute,
    )

    direct = COMMON_ANALYSIS_RUNNER.analyze_cin(program)
    compatibility = analyze_cin(program)

    assert len(calls) == 2
    assert all(call is program for call in calls)
    assert compatibility == direct
    assert compatibility is not direct
    assert canonical_cin_dump(program) == before


def test_common_analysis_runner_results_are_immutable_and_input_independent() -> None:
    program, nodes = _reduction_program()
    before = canonical_cin_dump(program)

    first = COMMON_ANALYSIS_RUNNER.analyze_cin(program)

    assert canonical_cin_dump(program) == before
    assert program.parent is None
    assert nodes.assignment.parent is None
    assert nodes.i.tensor_accesses == []
    assert nodes.k.tensor_accesses == []
    _assert_deeply_immutable(first)

    nodes.left.name = "A_changed"
    changed_input = canonical_cin_dump(program)
    second = COMMON_ANALYSIS_RUNNER.analyze_cin(program)

    assert canonical_cin_dump(program) == changed_input
    assert first.symbol_definitions[nodes.left.symbol_id].display_name == "A"
    assert second.symbol_definitions[nodes.left.symbol_id].display_name == "A_changed"
    assert first != second
    assert first is not second
    _assert_deeply_immutable(second)


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
    analysis = analyze_cin(first)
    workspace_ids = tuple(
        symbol_id
        for symbol_id, definition in analysis.symbol_definitions.items()
        if definition.is_workspace
    )
    assert len(workspace_ids) == 1
    workspace_access_ids = analysis.tensor_accesses[workspace_ids[0]]
    assert workspace_access_ids
    assert all(
        analysis.access_layouts[access_id].is_workspace
        for access_id in workspace_access_ids
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

    codes = _assert_structured_diagnostics(error)
    assert {"duplicate_node_reference", "duplicate_access_reference"} <= codes
    assert "cyclic_cin_structure" not in codes


@pytest.mark.parametrize(
    "cycle_kind",
    ("forall", "binary", "index_add"),
)
def test_structural_preflight_reports_cycles_without_recursing(
    cycle_kind: str,
) -> None:
    i = IndexVar("i")
    j = IndexVar("j")
    result = TensorVar("C", fmt="d")
    left = TensorVar("A", fmt="d")
    right = TensorVar("B", fmt="d")

    if cycle_kind == "forall":
        program = ForAll(i, TensorAssign(result[i], left[i]))
        program.stmt = program
    elif cycle_kind == "binary":
        expression = BinaryOp(Operation.ADD, left[i], right[i])
        object.__setattr__(expression, "left", expression)
        program = ForAll(i, TensorAssign(result[i], expression))
    else:
        i._expr = IndexVarAdd(i, j)
        program = ForAll(i, TensorAssign(result[i], left[i]))

    analysis = analyze_cin(program)

    assert tuple(diagnostic.code for diagnostic in analysis.diagnostics) == (
        "cyclic_cin_structure",
    )
    with pytest.raises(VerificationError) as error:
        verify_cin(program)
    assert _assert_structured_diagnostics(error) == {"cyclic_cin_structure"}


def test_structural_preflight_reports_missing_forward_fields() -> None:
    programs = []

    i = IndexVar("i")
    missing_parallel = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    del missing_parallel.parallel
    programs.append((missing_parallel, ("root", "parallel")))

    i = IndexVar("i")
    missing_stmt = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    del missing_stmt.stmt
    programs.append((missing_stmt, ("root", "stmt")))

    i = IndexVar("i")
    missing_assign_op = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    del missing_assign_op.stmt.op
    programs.append((missing_assign_op, ("root", "stmt", "op")))

    i = IndexVar("i")
    expression = BinaryOp(
        Operation.ADD,
        TensorVar("A", fmt="d")[i],
        TensorVar("B", fmt="d")[i],
    )
    object.__delattr__(expression, "op")
    missing_binary_op = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], expression),
    )
    programs.append((missing_binary_op, ("root", "stmt", "rhs", "op")))

    for program, path in programs:
        analysis = analyze_cin(program)
        assert analysis.diagnostics[0].code == "missing_cin_field"
        assert analysis.diagnostics[0].path == path
        with pytest.raises(VerificationError) as error:
            verify_cin(program)
        assert _assert_structured_diagnostics(error) == {"missing_cin_field"}


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (
            lambda program: object.__setattr__(program, "parallel", 1),
            ("root", "parallel"),
        ),
        (
            lambda program: object.__setattr__(program.stmt, "rhs", object()),
            ("root", "stmt", "rhs"),
        ),
    ],
)
def test_structural_preflight_reports_invalid_forward_fields(mutate, path) -> None:
    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    mutate(program)

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    assert analysis.diagnostics[0].path == path
    with pytest.raises(VerificationError) as error:
        verify_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


def test_structural_preflight_reports_excessive_depth_iteratively() -> None:
    i = IndexVar("i")
    statement: IndexStmt = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    for depth in range(cin_analysis_module._MAX_CIN_STRUCTURE_DEPTH + 1):
        statement = ForAll(IndexVar(f"deep_{depth}"), statement)

    analysis = analyze_cin(statement)

    assert tuple(diagnostic.code for diagnostic in analysis.diagnostics) == (
        "cin_structure_depth_exceeded",
    )
    with pytest.raises(VerificationError) as error:
        verify_cin(statement)
    assert _assert_structured_diagnostics(error) == {"cin_structure_depth_exceeded"}


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


def test_verifier_rejects_extent_mismatches_and_classifies_implicit_reductions() -> (
    None
):
    i = IndexVar("i")
    k = IndexVar("k")
    result = TensorVar("C", fmt="d", shape=(4,))
    source = TensorVar("A", fmt="d", shape=(5,))
    extent_program = ForAll(i, TensorAssign(result[i], source[i]))

    with pytest.raises(VerificationError) as extent_error:
        verify_cin(extent_program)
    assert "index_extent_mismatch" in _assert_structured_diagnostics(extent_error)

    implicit_reduction = ForAll(
        i,
        ForAll(k, TensorAssign(TensorVar("D", fmt="d")[i], source[k])),
    )
    implicit_analysis = verify_cin(implicit_reduction)
    assert implicit_analysis.free_index_ids == (i.index_id,)
    assert implicit_analysis.reduction_index_ids == (k.index_id,)

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


def test_debug_compiler_boundary_accepts_implicit_reduction() -> None:
    i = IndexVar("i")
    k = IndexVar("k")
    result = TensorVar("C", fmt="d", shape=(4,))
    source = TensorVar("A", fmt="dd", shape=(4, 5))
    program = ForAll(i, ForAll(k, TensorAssign(result[i], source[i, k])))

    with full_cin_verification():
        scheduled = Scheduler.auto_schedule(program)
        CINLowerer().lower_IndexStmt(scheduled)


def test_public_lowerer_preserves_caller_owned_cin() -> None:
    i = IndexVar("i")
    result = TensorVar("C", fmt="d", shape=(4,))
    source = TensorVar("A", fmt="d", shape=(4,))
    program = ForAll(i, TensorAssign(result[i], source[i]))
    before = canonical_cin_dump(program)

    CINLowerer().lower_IndexStmt(program)

    assert canonical_cin_dump(program) == before
    assert program.parent is None
    assert program.stmt.parent is None
    assert i.tensor_accesses == []


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


def test_release_mode_normalize_cin_fails_closed_on_forged_structure() -> None:
    """Direct plan-free normalization owns the structural boundary.

    The clone walk recurses over stored forward edges, so release-mode
    callers (verification disabled) must get the same stable structural
    diagnostics as the analyzed and LoopIR-owned entries instead of raw
    AttributeError, RecursionError, or a hang.
    """

    from scorch.compiler.compile_options import CompileOptions

    release_options = CompileOptions.from_environment(environ={})
    assert release_options.verification.verify_cin is False

    i = IndexVar("i")
    missing = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    del missing.stmt
    with pytest.raises(VerificationError) as error:
        normalize_cin(missing, compile_options=release_options)
    assert _assert_structured_diagnostics(error) == {"missing_cin_field"}

    j = IndexVar("j")
    cyclic = ForAll(
        j,
        TensorAssign(TensorVar("C", fmt="d")[j], TensorVar("A", fmt="d")[j]),
    )
    cyclic.stmt = cyclic
    with pytest.raises(VerificationError) as error:
        normalize_cin(cyclic, compile_options=release_options)
    assert _assert_structured_diagnostics(error) == {"cyclic_cin_structure"}

    k = IndexVar("k")
    deep: IndexStmt = ForAll(
        k,
        TensorAssign(TensorVar("C", fmt="d")[k], TensorVar("A", fmt="d")[k]),
    )
    for depth in range(cin_analysis_module._MAX_CIN_STRUCTURE_DEPTH + 1):
        deep = ForAll(IndexVar(f"deep_{depth}"), deep)
    with pytest.raises(VerificationError) as error:
        normalize_cin(deep, compile_options=release_options)
    assert _assert_structured_diagnostics(error) == {"cin_structure_depth_exceeded"}


def test_structural_preflight_rejects_descriptor_divergence() -> None:
    """A __class__-swapped property cannot split the stored/getattr graphs.

    The preflight validates stored state while the recursive analyses walk
    the getattr view; a hostile data descriptor returning a different object
    (here: a self-cycle over a benign stored child) must fail closed instead
    of validating one graph and handing the recursion another.
    """

    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )

    class HostileForAll(ForAll):
        @property
        def stmt(self):  # type: ignore[override]
            return self

    program.__class__ = HostileForAll

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    assert analysis.diagnostics[0].path == ("root", "stmt")
    with pytest.raises(VerificationError) as error:
        verify_cin(program)
    assert "invalid_cin_field" in _assert_structured_diagnostics(error)


def test_structural_preflight_rejects_raising_descriptor() -> None:
    """A descriptor that raises on read is as hostile as a diverging one."""

    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )

    class ExplodingForAll(ForAll):
        @property
        def parallel(self):  # type: ignore[override]
            raise RuntimeError("hostile descriptor")

    program.__class__ = ExplodingForAll

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    assert analysis.diagnostics[0].path == ("root", "parallel")
    with pytest.raises(VerificationError) as error:
        verify_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}
