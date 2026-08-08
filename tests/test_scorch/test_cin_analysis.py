import copy
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
    UnaryOp,
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
    verify_cin_structure,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.diagnostics import CompilerInvariantError, VerificationError
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


def test_normalization_detaches_tensor_formats_across_artifacts() -> None:
    caller_format = TensorFormat("d")

    def build(prefix: str) -> ForAll:
        i = IndexVar("i")
        source = TensorVar(f"{prefix}A", fmt=caller_format)
        result = TensorVar(f"{prefix}C", fmt=caller_format)
        return ForAll(i, TensorAssign(result[i], source[i]))

    first = normalize_cin(build("first_"))
    second = normalize_cin(build("second_"))
    first_formats = tuple(access.tensor.format for access in first.tensor_accesses)
    second_formats = tuple(access.tensor.format for access in second.tensor_accesses)
    before = (canonical_cin_dump(first), canonical_cin_dump(second))

    assert all(tensor_format is not caller_format for tensor_format in first_formats)
    assert all(tensor_format is not caller_format for tensor_format in second_formats)
    assert all(left is not right for left in first_formats for right in second_formats)

    object.__setattr__(caller_format, "_level_formats", (LevelFormat("s"),))
    assert (canonical_cin_dump(first), canonical_cin_dump(second)) == before
    verify_cin(first)
    verify_cin(second)


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
    # The always-on structural preflight now owns shared-object references,
    # so the failure is structural (stage-owned) before the semantic
    # duplicate_access_reference analysis can run.
    assert "duplicate_node_reference" in codes
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


@pytest.mark.parametrize("field", ("index_var", "stmt", "lhs"))
def test_structural_preflight_never_reads_child_class_descriptors(field) -> None:
    """Typed child admission uses stored exact types, never ``isinstance``."""

    reads = 0

    class HostileChild:
        @property
        def __class__(self):  # type: ignore[override]
            nonlocal reads
            reads += 1
            raise RuntimeError("hostile child class descriptor")

    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    owner = program if field != "lhs" else program.stmt
    object.__setattr__(owner, field, HostileChild())

    analysis = analyze_cin(program)
    assert analysis.diagnostics[0].code == "invalid_cin_field"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}
    assert reads == 0


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


@pytest.mark.parametrize("target", ("tensor", "index"))
def test_structural_preflight_depth_bound_keeps_postpass_total(target: str) -> None:
    """Skipped descendants cannot escape through cross-field reconciliation."""

    class HostileDict(dict):
        def get(self, key, default=None):
            raise RuntimeError("hostile skipped-descendant lookup")

        def __iter__(self):
            raise RuntimeError("hostile skipped-descendant iteration")

    index = IndexVar("i")
    source = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    statement: IndexStmt = TensorAssign(result[index], source[index])
    # Put both accesses exactly at the supported limit. Their TensorVar and
    # IndexVar children sit one level beyond it and are intentionally skipped.
    for depth in range(cin_analysis_module._MAX_CIN_STRUCTURE_DEPTH - 1):
        statement = ForAll(IndexVar(f"deep_post_{depth}"), statement)

    skipped = source if target == "tensor" else index
    object.__setattr__(skipped, "__dict__", HostileDict(skipped.__dict__))

    analysis = analyze_cin(statement)

    assert analysis.diagnostics[0].code == "cin_structure_depth_exceeded"
    with pytest.raises(VerificationError) as error:
        normalize_cin(statement)
    assert error.value.diagnostics[0].code == "cin_structure_depth_exceeded"


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
    result = TensorVar("C", fmt="d", shape=(4,))
    source = TensorVar("A", fmt="d", shape=(5,))
    invalid = ForAll(i, TensorAssign(result[i], source[i]))

    normalize_cin(invalid)
    with full_cin_verification():
        with pytest.raises(VerificationError, match="index_extent_mismatch"):
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


def test_public_lowerer_rejects_malformed_raw_cin_before_legacy_copy() -> None:
    """Direct legacy lowering retains normalization's malformed-field boundary."""

    i = IndexVar("i")
    result = TensorVar("C", fmt="d", shape=(4,))
    source = TensorVar("A", fmt="d", shape=(4,))
    program = ForAll(i, TensorAssign(result[i], source[i]))
    program.index_var = object()  # type: ignore[assignment]

    with pytest.raises(VerificationError) as error:
        CINLowerer().lower_IndexStmt(program)

    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


def test_public_lowerer_scopes_scheduler_owned_workspace_id_aliases() -> None:
    """Release compatibility does not silently weaken requested verification."""

    from scorch.compiler.compile_options import CompileOptions

    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("SparseProduct", fmt="ds")
    left = TensorVar("SparseLeft", fmt="ds")
    right = TensorVar("SparseRight", fmt="ds")
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    release_options = CompileOptions.from_environment(
        environ={},
        regblock_override=False,
        verify_cin_override=False,
    )
    debug_options = CompileOptions.from_environment(
        environ={},
        regblock_override=False,
        verify_cin_override=True,
    )
    scheduled = Scheduler.auto_schedule(source, compile_options=release_options)

    with pytest.raises(VerificationError) as strict_error:
        verify_cin_structure(scheduled)
    assert _assert_structured_diagnostics(strict_error) == {
        "duplicate_index_id",
        "duplicate_node_id",
        "duplicate_node_reference",
    }
    with pytest.raises(VerificationError) as normalize_error:
        normalize_cin(scheduled, compile_options=debug_options)
    assert _assert_structured_diagnostics(normalize_error) == {
        "duplicate_index_id",
        "duplicate_node_id",
        "duplicate_node_reference",
    }
    scheduled_dump = canonical_cin_dump(scheduled)
    assert '"kind":"where"' in scheduled_dump

    lowered = CINLowerer(compile_options=release_options).lower_IndexStmt(scheduled)
    debug_scheduled = Scheduler.auto_schedule(
        source,
        compile_options=debug_options,
    )
    with pytest.raises(VerificationError) as debug_error:
        CINLowerer(compile_options=debug_options).lower_IndexStmt(debug_scheduled)
    assert _assert_structured_diagnostics(debug_error) == {
        "unverifiable_legacy_schedule_aliases"
    }

    # The underscore boundary receives a compiler-owned tree immediately after
    # Scheduler verified its normalized source; no caller work can intervene.
    owned_scheduled = Scheduler.auto_schedule(
        source,
        compile_options=debug_options,
    )
    owned_lowered = CINLowerer(compile_options=debug_options)._lower_owned_IndexStmt(
        owned_scheduled
    )

    assert lowered
    assert owned_lowered

    missing_marker = Scheduler.auto_schedule(
        source,
        compile_options=release_options,
    )
    missing_marker.inserted_workspace = False
    with pytest.raises(VerificationError) as missing_marker_error:
        CINLowerer(compile_options=release_options).lower_IndexStmt(missing_marker)
    assert _assert_structured_diagnostics(missing_marker_error) == {
        "duplicate_index_id",
        "duplicate_node_id",
        "duplicate_node_reference",
    }
    with pytest.raises(VerificationError) as missing_dump_marker_error:
        canonical_cin_dump(missing_marker)
    assert _assert_structured_diagnostics(missing_dump_marker_error) == {
        "duplicate_index_id",
        "duplicate_node_id",
        "duplicate_node_reference",
    }


def test_public_debug_lowerer_never_skips_semantic_checks_for_workspace_aliases() -> (
    None
):
    """A post-schedule semantic mutation cannot opt out of requested checks."""

    from scorch.compiler.compile_options import CompileOptions

    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("SparseProduct", fmt="ds", shape=(4, 7))
    left = TensorVar("SparseLeft", fmt="ds", shape=(4, 5))
    right = TensorVar("SparseRight", fmt="ds", shape=(5, 7))
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    options = CompileOptions.from_environment(
        environ={},
        regblock_override=False,
        verify_cin_override=True,
    )
    scheduled = Scheduler.auto_schedule(source, compile_options=options)
    scheduled_left = next(
        access.tensor
        for access in scheduled.tensor_accesses
        if access.tensor.name == "SparseLeft"
    )
    scheduled_left.shape = (4, 7)

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(scheduled)

    assert _assert_structured_diagnostics(error) == {
        "unverifiable_legacy_schedule_aliases"
    }


@pytest.mark.parametrize(
    ("forgery", "expected_code"),
    (
        ("same_kind_node", "duplicate_node_id"),
        ("cross_kind_node", "duplicate_node_id"),
        ("unpaired_index", "duplicate_index_id"),
        ("symbol", "duplicate_symbol_id"),
        ("access", "duplicate_access_id"),
    ),
)
def test_public_lowerer_legacy_alias_compatibility_is_narrow(
    forgery: str,
    expected_code: str,
) -> None:
    """Only same-kind node clones and same-node/name index clones are admitted."""

    program, nodes = _reduction_program()
    if forgery == "same_kind_node":
        inner = program.stmt
        assert isinstance(inner, ForAll)
        inner.node_id = program.node_id
    elif forgery == "cross_kind_node":
        nodes.assignment.node_id = program.node_id
    elif forgery == "unpaired_index":
        nodes.k.index_id = nodes.i.index_id
        nodes.left_access.index_ids = (nodes.i.index_id, nodes.i.index_id)
        nodes.right_access.index_ids = (nodes.i.index_id,)
    elif forgery == "symbol":
        nodes.right.symbol_id = nodes.left.symbol_id
        nodes.right_access.tensor_id = nodes.left.symbol_id
    else:
        nodes.right_access.access_id = nodes.left_access.access_id

    with pytest.raises(VerificationError) as error:
        CINLowerer().lower_IndexStmt(program)

    assert expected_code in _assert_structured_diagnostics(error)


@pytest.mark.parametrize("mismatch", ("node_id", "display_name"))
def test_public_lowerer_rejects_incompatible_same_id_index_aliases(
    mismatch: str,
) -> None:
    """Index aliases must retain both their cloned node identity and name."""

    program, outer, inner = _same_name_nested_program()
    inner.index_id = outer.index_id
    for access in program.tensor_accesses:
        access.index_ids = tuple(index.index_id for index in access.indices)
    if mismatch == "node_id":
        assert inner.node_id != outer.node_id
    else:
        inner.node_id = outer.node_id
        inner._name = "other"

    with pytest.raises(VerificationError) as error:
        CINLowerer().lower_IndexStmt(program)

    assert "duplicate_index_id" in _assert_structured_diagnostics(error)


@pytest.mark.parametrize(
    "forgery",
    ("parent", "tile_size_var", "is_tiled", "legacy_accesses"),
)
def test_public_lowerer_rejects_malformed_legacy_index_metadata(
    forgery: str,
) -> None:
    """Compatibility-only index metadata cannot leak legacy exceptions."""

    program, nodes = _reduction_program()
    if forgery == "parent":
        nodes.i._parent = object()  # type: ignore[assignment]
    elif forgery == "tile_size_var":
        nodes.i.tile_size_var = object()  # type: ignore[assignment]
    elif forgery == "is_tiled":
        nodes.i.is_tiled = 1  # type: ignore[assignment]
    else:
        nodes.i._legacy_tensor_accesses = object()  # type: ignore[assignment]

    with pytest.raises(VerificationError) as error:
        CINLowerer().lower_IndexStmt(program)

    assert "invalid_cin_field" in _assert_structured_diagnostics(error)


def test_workspace_alias_classifier_requires_corresponding_branch_binders() -> None:
    """A workspace marker cannot license an unrelated cross-branch identity."""

    from scorch.compiler.compile_options import CompileOptions

    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("SparseProduct", fmt="ds", shape=(4, 7))
    left = TensorVar("SparseLeft", fmt="ds", shape=(4, 5))
    right = TensorVar("SparseRight", fmt="ds", shape=(5, 7))
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    options = CompileOptions.from_environment(
        environ={},
        regblock_override=False,
        verify_cin_override=False,
    )
    scheduled = Scheduler.auto_schedule(source, compile_options=options)
    assert isinstance(scheduled, ForAll)
    region = scheduled.stmt
    assert isinstance(region, Where)
    producer_reduction = region.producer
    consumer_reduction = region.consumer
    assert isinstance(producer_reduction, ForAll)
    assert isinstance(consumer_reduction, ForAll)
    consumer_column = consumer_reduction.stmt
    assert isinstance(consumer_column, ForAll)

    forged = consumer_column.index_var
    reference = producer_reduction.index_var
    forged.node_id = reference.node_id
    forged.index_id = reference.index_id
    forged._name = reference.name
    for access in consumer_reduction.tensor_accesses:
        access.index_ids = tuple(index.index_id for index in access.indices)

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(scheduled)

    assert {
        "duplicate_index_id",
        "duplicate_node_id",
    } & _assert_structured_diagnostics(error)


@pytest.mark.parametrize("forgery", ("workspace_role", "assignment_operator"))
def test_workspace_alias_classifier_requires_exact_region_shape(
    forgery: str,
) -> None:
    """Only the exact producer-write/consumer-read clone region licenses IDs."""

    from scorch.compiler.compile_options import CompileOptions

    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("SparseProduct", fmt="ds", shape=(4, 7))
    left = TensorVar("SparseLeft", fmt="ds", shape=(4, 5))
    right = TensorVar("SparseRight", fmt="ds", shape=(5, 7))
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    options = CompileOptions.from_environment(
        environ={},
        regblock_override=False,
        verify_cin_override=False,
    )
    scheduled = Scheduler.auto_schedule(source, compile_options=options)
    assert isinstance(scheduled, ForAll)
    region = scheduled.stmt
    assert isinstance(region, Where)
    consumer: IndexStmt = region.consumer
    while isinstance(consumer, ForAll):
        consumer = consumer.stmt
    assert isinstance(consumer, TensorAssign)
    if forgery == "workspace_role":
        consumer.rhs = consumer.lhs
    else:
        consumer.op = Operation.SUB

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(scheduled)

    assert "duplicate_node_id" in _assert_structured_diagnostics(error)


@pytest.mark.parametrize(
    "forgery",
    (
        "inner_role",
        "outer_role",
        "dual_role",
        "wrong_parent",
        "foreign_parent",
        "nested_access_backlink",
        "foreign_access_backlink",
        "nested_tile_backlink",
    ),
)
def test_public_lowerer_rejects_relationally_malformed_tile_metadata(
    forgery: str,
) -> None:
    """Legacy tile metadata is checked as one linked structural contract."""

    from scorch.compiler.compile_options import CompileOptions

    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("DenseProduct", fmt="dd", shape=(4, 7))
    left = TensorVar("SparseLeft", fmt="ds", shape=(4, 5))
    right = TensorVar("DenseRight", fmt="dd", shape=(5, 7))
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    options = CompileOptions.from_environment(
        environ={},
        regblock_override=True,
        verify_cin_override=False,
    )
    scheduled = Scheduler.auto_schedule(source, compile_options=options)
    indices = {index.name: index for index in scheduled.index_vars}
    inner = indices["c_in"]
    outer = indices["c_out"]
    if forgery == "inner_role":
        inner.is_inner = False
    elif forgery == "outer_role":
        outer.is_outer = False
    elif forgery == "dual_role":
        inner.is_outer = True
    elif forgery == "wrong_parent":
        inner._parent = indices["r"]
    elif forgery == "foreign_parent":
        inner._parent = IndexVar("outside")
    elif forgery == "nested_access_backlink":
        inner._legacy_tensor_accesses = [[[[object()]]]]  # type: ignore[list-item]
    elif forgery == "foreign_access_backlink":
        foreign_tensor = TensorVar("Foreign", fmt="d")
        inner._legacy_tensor_accesses = [foreign_tensor[IndexVar("outside")]]
    else:
        assert inner.tile_size_var is not None
        inner.tile_size_var.no_tile_list = [[[[object()]]]]  # type: ignore[list-item]

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(scheduled)

    assert "invalid_cin_field" in _assert_structured_diagnostics(error)


def test_public_lowerer_forward_copy_ignores_unowned_deep_instance_state() -> None:
    """Discarded arbitrary attributes cannot reintroduce deepcopy recursion."""

    program, _ = _reduction_program()
    deep: object = None
    for _ in range(1_500):
        deep = [deep]
    program.unowned_deep_state = deep  # type: ignore[attr-defined]

    lowered = CINLowerer().lower_IndexStmt(program)

    assert lowered
    assert program.unowned_deep_state is deep  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "forgery",
    (
        "root_no_tile_list",
        "root_workspace_marker",
        "index_parent",
        "index_role",
        "index_tile",
        "index_access_backlink",
    ),
)
def test_normalization_discards_legacy_only_schedule_metadata(forgery: str) -> None:
    """Ignored scheduler backlinks cannot change semantic CIN admission."""

    program, nodes = _reduction_program()
    hostile = object()
    if forgery == "root_no_tile_list":
        program.no_tile_list = [hostile]  # type: ignore[list-item]
    elif forgery == "root_workspace_marker":
        program.inserted_workspace = hostile  # type: ignore[assignment]
    elif forgery == "index_parent":
        nodes.i._parent = hostile  # type: ignore[assignment]
    elif forgery == "index_role":
        nodes.i.is_outer = hostile  # type: ignore[assignment]
    elif forgery == "index_tile":
        nodes.i.tile_size_var = hostile  # type: ignore[assignment]
    else:
        nodes.i._legacy_tensor_accesses = [hostile]  # type: ignore[list-item]

    normalized = normalize_cin(program)

    assert normalized.inserted_workspace is False
    assert normalized.no_tile_list == []
    normalized_i = normalized.index_vars[0]
    assert normalized_i._parent is None
    assert normalized_i.is_outer is False
    assert normalized_i.tile_size_var is None
    assert normalized_i._legacy_tensor_accesses == []


@pytest.mark.parametrize(
    "forgery",
    (
        "root_no_tile_list",
        "root_workspace_marker",
        "index_parent",
        "index_role",
        "index_tile",
        "index_access_backlink",
    ),
)
def test_public_legacy_lowerer_validates_consumed_schedule_metadata(
    forgery: str,
) -> None:
    """The adapter remains strict for the compatibility fields it copies."""

    program, nodes = _reduction_program()
    hostile = object()
    if forgery == "root_no_tile_list":
        program.no_tile_list = [hostile]  # type: ignore[list-item]
    elif forgery == "root_workspace_marker":
        program.inserted_workspace = hostile  # type: ignore[assignment]
    elif forgery == "index_parent":
        nodes.i._parent = hostile  # type: ignore[assignment]
    elif forgery == "index_role":
        nodes.i.is_outer = hostile  # type: ignore[assignment]
    elif forgery == "index_tile":
        nodes.i.tile_size_var = hostile  # type: ignore[assignment]
    else:
        nodes.i._legacy_tensor_accesses = [hostile]  # type: ignore[list-item]

    with pytest.raises(VerificationError) as error:
        CINLowerer().lower_IndexStmt(program)

    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


def test_public_lowerer_rejects_missing_optional_tile_storage() -> None:
    """A stored optional tile reference must be present even when it is None."""

    from scorch.compiler.compile_options import CompileOptions

    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("DenseProduct", fmt="dd", shape=(4, 7))
    left = TensorVar("SparseLeft", fmt="ds", shape=(4, 5))
    right = TensorVar("DenseRight", fmt="dd", shape=(5, 7))
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    options = CompileOptions.from_environment(
        environ={},
        regblock_override=True,
        verify_cin_override=False,
    )
    scheduled = Scheduler.auto_schedule(source, compile_options=options)
    inner = next(index for index in scheduled.index_vars if index.name == "c_in")
    assert inner.tile_size_var is not None
    del inner.tile_size_var.__dict__["_index_var"]

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(scheduled)

    assert "missing_cin_field" in _assert_structured_diagnostics(error)


@pytest.mark.parametrize("missing_field", ("_parent", "tile_size_var"))
def test_public_lowerer_rejects_missing_detached_tile_parent_storage(
    missing_field: str,
) -> None:
    """A detached logical parent must still own every copied optional field."""

    from scorch.compiler.compile_options import CompileOptions

    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("R", fmt="dd", shape=(4, 7))
    left = TensorVar("A", fmt="ds", shape=(4, 5))
    right = TensorVar("B", fmt="dd", shape=(5, 7))
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    options = CompileOptions.from_environment(
        environ={},
        regblock_override=False,
        verify_cin_override=False,
    )
    scheduled = Scheduler.add_tile(
        source,
        column,
        4,
        compile_options=options,
    )
    accesses = {access.tensor.name: access for access in scheduled.tensor_accesses}
    logical_parent = next(
        index
        for index in scheduled.index_vars
        if index.name == "c" and index._expr is not None
    )
    forward_alias = accesses["B"].indices[1]
    accesses["R"].indices[1] = forward_alias
    accesses["R"].index_ids = tuple(index.index_id for index in accesses["R"].indices)

    # Successive legacy tiling can legitimately detach this logical parent.
    assert CINLowerer(compile_options=options).lower_IndexStmt(scheduled)

    del logical_parent.__dict__[missing_field]
    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(scheduled)

    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


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


@pytest.mark.parametrize("location", ("root", "expression"))
def test_release_normalization_requires_every_node_id(location: str) -> None:
    """Every node normalized in release mode must carry its stored NodeId."""

    from scorch.compiler.compile_options import CompileOptions

    release_options = CompileOptions.from_environment(environ={})
    i = IndexVar("i")
    expression = BinaryOp(
        Operation.ADD,
        TensorVar("A", fmt="d")[i],
        TensorVar("B", fmt="d")[i],
    )
    program = ForAll(i, TensorAssign(TensorVar("C", fmt="d")[i], expression))
    target = program if location == "root" else expression
    object.__delattr__(target, "node_id")

    analysis = analyze_cin(program)
    assert analysis.diagnostics[0].code == "missing_cin_field"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program, compile_options=release_options)
    assert _assert_structured_diagnostics(error) == {"missing_cin_field"}


def test_workspace_access_requires_one_authoritative_workspace() -> None:
    """The subclass-only wksp edge cannot disappear or diverge from tensor."""

    from scorch.compiler.compile_options import CompileOptions

    release_options = CompileOptions.from_environment(environ={})

    def build():
        i = IndexVar("i")
        source = TensorVar("A", fmt="d")
        result = TensorVar("C", fmt="d")
        workspace = Workspace("tmp", dim=1, dense=True)
        producer_access = WorkspaceAccess(workspace, i)
        program = ForAll(
            i,
            Where(
                producer=TensorAssign(producer_access, source[i]),
                consumer=TensorAssign(result[i], WorkspaceAccess(workspace, i)),
            ),
        )
        return program, producer_access

    missing, missing_access = build()
    del missing_access.wksp
    with pytest.raises(VerificationError) as error:
        normalize_cin(missing, compile_options=release_options)
    assert _assert_structured_diagnostics(error) == {"missing_cin_field"}

    divergent, divergent_access = build()
    divergent_access.wksp = Workspace("other", dim=1, dense=True)
    with pytest.raises(VerificationError) as error:
        normalize_cin(divergent, compile_options=release_options)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}

    forged, forged_access = build()
    ordinary_access = TensorAccess(forged_access.wksp, forged_access.indices)
    inner = forged.stmt
    assert isinstance(inner, Where)
    producer = inner.producer
    assert isinstance(producer, TensorAssign)
    producer.lhs = ordinary_access
    with pytest.raises(VerificationError) as error:
        normalize_cin(forged, compile_options=release_options)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


def test_structural_preflight_rejects_hostile_container_subclasses() -> None:
    """Preflight never executes attacker-controlled container iteration."""

    class HostileList(list):
        def __iter__(self):
            raise RuntimeError("hostile iteration")

    i = IndexVar("i")
    access = TensorVar("A", fmt="d")[i]
    access.indices = HostileList((i,))
    program = ForAll(i, TensorAssign(TensorVar("C", fmt="d")[i], access))

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


def test_structural_preflight_owns_malformed_mode_order_entries() -> None:
    """A heterogeneous forged permutation cannot escape through sorted()."""

    i, j = IndexVar("i"), IndexVar("j")
    source = TensorVar("A", fmt="dd")
    source.mode_order = [0, object()]  # type: ignore[list-item]
    program = ForAll(
        i,
        ForAll(j, TensorAssign(TensorVar("C", fmt="dd")[i, j], source[i, j])),
    )

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    with pytest.raises(VerificationError) as error:
        verify_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}

    malformed_format = TensorVar("D", fmt="dd")
    assert malformed_format._format is not None
    object.__setattr__(malformed_format._format, "_level_formats", object())
    malformed = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(TensorVar("E", fmt="dd")[i, j], malformed_format[i, j]),
        ),
    )
    with pytest.raises(VerificationError) as error:
        normalize_cin(malformed)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


@pytest.mark.parametrize(
    ("mutate_format", "mode_order"),
    (
        (
            lambda tensor_format: object.__setattr__(
                tensor_format,
                "_level_formats",
                (object(),),
            ),
            None,
        ),
        (
            lambda tensor_format: object.__setattr__(
                tensor_format.get_level_formats()[0],
                "_mode",
                object(),
            ),
            [0],
        ),
        (
            lambda tensor_format: object.__setattr__(
                tensor_format.get_level_formats()[0],
                "_bit_width",
                0,
            ),
            [0],
        ),
    ),
)
def test_structural_preflight_validates_complete_stored_tensor_format(
    mutate_format,
    mode_order,
) -> None:
    """Malformed exact format objects fail before format helper methods run."""

    i = IndexVar("i")
    source = TensorVar("A", fmt="d")
    source.mode_order = mode_order
    assert source._format is not None
    mutate_format(source._format)
    program = ForAll(i, TensorAssign(TensorVar("C", fmt="d")[i], source[i]))

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


@pytest.mark.parametrize(
    ("field", "bad_identity", "expected_code"),
    (
        ("node_id", NodeId([]), "invalid_node_id"),  # type: ignore[arg-type]
        ("index_id", IndexId([]), "invalid_index_id"),  # type: ignore[arg-type]
        ("symbol_id", SymbolId([]), "invalid_symbol_id"),  # type: ignore[arg-type]
        ("access_id", AccessId([]), "invalid_access_id"),  # type: ignore[arg-type]
        (
            "tensor_id",
            SymbolId([]),  # type: ignore[arg-type]
            "invalid_symbol_reference",
        ),
        (
            "index_ids",
            (IndexId([]),),  # type: ignore[arg-type]
            "invalid_index_reference",
        ),
    ),
)
def test_structural_preflight_rejects_unhashable_exact_identity_payloads(
    field: str,
    bad_identity,
    expected_code: str,
) -> None:
    """Exact wrapper classes still require safe integer payloads before hashing."""

    program, nodes = _reduction_program()
    if field == "node_id":
        nodes.assignment.node_id = bad_identity
    elif field == "index_id":
        nodes.i.index_id = bad_identity
    elif field == "symbol_id":
        nodes.left.symbol_id = bad_identity
    elif field == "access_id":
        nodes.left_access.access_id = bad_identity
    elif field == "tensor_id":
        nodes.left_access.tensor_id = bad_identity
    else:
        nodes.left_access.index_ids = bad_identity

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == expected_code
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert expected_code in _assert_structured_diagnostics(error)


def test_structural_preflight_bounds_integer_domains_before_formatting() -> None:
    """Huge runtime integers cannot escape through diagnostics or cost models."""

    i = IndexVar("i")

    huge_shape = TensorVar("A", fmt="d")
    huge_shape.shape = (1 << 63,)
    shape_program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d", shape=(1,))[i], huge_shape[i]),
    )

    product_overflow = TensorVar("B", fmt="dd")
    product_overflow.shape = ((1 << 63) - 1, 2)
    j = IndexVar("j")
    product_program = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("D", fmt="dd", shape=(1, 1))[i, j],
                product_overflow[i, j],
            ),
        ),
    )

    huge_width = TensorVar(
        "E",
        fmt=TensorFormat([LevelFormat("d", bit_width=10**10000)]),
    )
    width_program = ForAll(
        i,
        TensorAssign(TensorVar("F", fmt="d")[i], huge_width[i]),
    )

    workspace = Workspace("tmp", dim=1, dense=True)
    workspace.dim = 10**10000
    workspace_program = ForAll(
        i,
        TensorAssign(
            TensorVar("G", fmt="d")[i],
            WorkspaceAccess(workspace, i),
        ),
    )

    for program in (
        shape_program,
        product_program,
        width_program,
        workspace_program,
    ):
        analysis = analyze_cin(program)
        assert analysis.diagnostics[0].code == "invalid_cin_field"
        with pytest.raises(VerificationError) as error:
            normalize_cin(program)
        assert "invalid_cin_field" in _assert_structured_diagnostics(error)


def test_structural_preflight_bounds_stable_identity_payloads() -> None:
    """Stable IDs are finite nonnegative compiler-domain integers."""

    program, nodes = _reduction_program()
    nodes.assignment.node_id = NodeId(10**10000)

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_node_id"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_node_id"}


@pytest.mark.parametrize(
    ("forgery", "expected_code"),
    (
        ("format_rank", "invalid_cin_field"),
        ("access_rank", "tensor_access_rank_mismatch"),
        ("index_mirror", "index_reference_mismatch"),
        ("symbol_mirror", "symbol_reference_mismatch"),
        ("missing_format", "invalid_cin_field"),
        ("missing_mode_order", "invalid_cin_field"),
    ),
)
def test_structural_preflight_reconciles_cross_field_ownership(
    forgery: str,
    expected_code: str,
) -> None:
    """Release normalization never carries split rank/reference metadata."""

    i, j = IndexVar("i"), IndexVar("j")
    source = TensorVar("A", fmt="dd", shape=(2, 3))
    result = TensorVar("C", fmt="dd", shape=(2, 3))
    source_access = source[i, j]
    program = ForAll(i, ForAll(j, TensorAssign(result[i, j], source_access)))

    if forgery == "format_rank":
        source._format = TensorFormat("d")
    elif forgery == "access_rank":
        source_access.indices = [i]
        source_access.index_ids = (i.index_id,)
    elif forgery == "index_mirror":
        source_access.index_ids = (j.index_id, i.index_id)
    elif forgery == "symbol_mirror":
        source_access.tensor_id = result.symbol_id
    elif forgery == "missing_format":
        source._format = None
    else:
        source.mode_order = None

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == expected_code
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert expected_code in _assert_structured_diagnostics(error)


@pytest.mark.parametrize(
    ("identity_kind", "expected_code"),
    (
        ("node", "duplicate_node_id"),
        ("index", "duplicate_index_id"),
        ("symbol", "duplicate_symbol_id"),
        ("access", "duplicate_access_id"),
    ),
)
def test_structural_preflight_rejects_duplicate_stable_identity_owners(
    identity_kind: str,
    expected_code: str,
) -> None:
    """Distinct CIN entities cannot share one stable identity in release mode."""

    program, nodes = _reduction_program()
    if identity_kind == "node":
        nodes.assignment.node_id = program.node_id
    elif identity_kind == "index":
        nodes.k.index_id = nodes.i.index_id
    elif identity_kind == "symbol":
        nodes.right.symbol_id = nodes.left.symbol_id
    else:
        nodes.right_access.access_id = nodes.left_access.access_id

    analysis = analyze_cin(program)

    assert expected_code in {diagnostic.code for diagnostic in analysis.diagnostics}
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert expected_code in _assert_structured_diagnostics(error)


def test_structural_preflight_never_hashes_identity_subclasses() -> None:
    """A hostile stable-ID subclass cannot execute hooks during admission."""

    class HostileNodeId(NodeId):
        def __hash__(self):
            raise RuntimeError("hostile identity hash")

    program, nodes = _reduction_program()
    nodes.assignment.node_id = HostileNodeId(nodes.assignment.node_id.value)

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_node_id"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_node_id"}


def test_structural_preflight_never_hashes_or_names_hostile_node_classes() -> None:
    """Exact-node admission uses identity and a class-name-free diagnostic."""

    class HostileMeta(type):
        def __hash__(cls):
            raise RuntimeError("hostile class hash")

        def __getattribute__(cls, name):
            if name == "__name__":
                raise RuntimeError("hostile class name")
            return super().__getattribute__(name)

    class HostileForAll(ForAll, metaclass=HostileMeta):
        pass

    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    program.__class__ = HostileForAll

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


def test_structural_preflight_never_compares_hostile_container_classes() -> None:
    """Container admission uses exact identity rather than metaclass equality."""

    class HostileMeta(type):
        def __eq__(cls, other):
            raise RuntimeError("hostile class equality")

    class HostileIndices(metaclass=HostileMeta):
        pass

    i = IndexVar("i")
    access = TensorVar("A", fmt="d")[i]
    access.indices = HostileIndices()  # type: ignore[assignment]
    program = ForAll(i, TensorAssign(TensorVar("C", fmt="d")[i], access))

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


@pytest.mark.parametrize("target", ("node", "format", "level"))
def test_structural_preflight_rejects_hostile_stored_state(target: str) -> None:
    """Stored-state mappings and their keys cannot execute Python hooks."""

    class HostileDict(dict):
        def __contains__(self, key):
            raise RuntimeError("hostile state membership")

        def __iter__(self):
            raise RuntimeError("hostile state iteration")

        def get(self, key, default=None):
            raise RuntimeError("hostile state lookup")

    i = IndexVar("i")
    source = TensorVar("A", fmt="d")
    program = ForAll(i, TensorAssign(TensorVar("C", fmt="d")[i], source[i]))
    if target == "node":
        object.__setattr__(program, "__dict__", HostileDict(program.__dict__))
    else:
        assert source._format is not None
        format_target = (
            source._format
            if target == "format"
            else source._format.get_level_formats()[0]
        )
        object.__setattr__(
            format_target,
            "__dict__",
            HostileDict(format_target.__dict__),
        )

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


def test_structural_preflight_rejects_hostile_stored_state_keys() -> None:
    """Exact dictionaries are inspected without hashing attacker-owned keys."""

    class HostileKey:
        def __hash__(self):
            return hash("node_id")

        def __eq__(self, other):
            raise RuntimeError("hostile state-key equality")

    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    state = dict(program.__dict__)
    del state["node_id"]
    state[HostileKey()] = program.node_id
    object.__setattr__(program, "__dict__", state)

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


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
    assert analysis.diagnostics[0].path == ("root",)
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
    assert analysis.diagnostics[0].path == ("root",)
    with pytest.raises(VerificationError) as error:
        verify_cin(program)
    assert _assert_structured_diagnostics(error) == {"invalid_cin_field"}


def test_structural_preflight_never_executes_stateful_descriptors() -> None:
    """Exact-node admission closes the one-read descriptor TOCTOU."""

    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    reads = 0

    class StatefulForAll(ForAll):
        @property
        def stmt(self):  # type: ignore[override]
            nonlocal reads
            reads += 1
            stored = object.__getattribute__(self, "__dict__")["stmt"]
            return stored if reads == 1 else self

    program.__class__ = StatefulForAll

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    assert reads == 0
    with pytest.raises(VerificationError):
        normalize_cin(program)
    assert reads == 0


def test_structural_preflight_never_executes_root_identity_descriptor() -> None:
    """Root diagnostic construction reads only already-validated stored state."""

    reads = 0

    class HostileForAll(ForAll):
        @property
        def node_id(self):  # type: ignore[override]
            nonlocal reads
            reads += 1
            raise RuntimeError("hostile identity descriptor")

    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    program.__class__ = HostileForAll

    analysis = analyze_cin(program)

    assert analysis.diagnostics[0].code == "invalid_cin_field"
    assert reads == 0
    with pytest.raises(VerificationError):
        normalize_cin(program)
    assert reads == 0


def test_public_cin_root_boundaries_never_execute_class_descriptors() -> None:
    """Root admission uses ``type`` rather than caller-defined ``__class__``."""

    from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir

    reads = 0

    class HostileRoot:
        @property
        def __class__(self):  # type: ignore[override]
            nonlocal reads
            reads += 1
            raise RuntimeError("hostile root class descriptor")

    hostile: Any = HostileRoot()
    entrypoints = (
        analyze_cin,
        verify_cin_structure,
        normalize_cin,
        canonical_cin_dump,
        Scheduler.auto_schedule,
        Scheduler.auto_schedule_plan,
        CINLowerer().lower_IndexStmt,
        CINLowerer().lower_IndexExpr,
        CINLowerer().lower_CIN,
        lower_normalized_cin_to_loopir,
    )
    for entrypoint in entrypoints[:7]:
        with pytest.raises(TypeError):
            entrypoint(hostile)
    for entrypoint in entrypoints[7:]:
        # lower_IndexExpr now refuses outermost expression roots before any
        # node inspection (invalid_expression_entry), still without reading
        # hostile class descriptors.
        with pytest.raises((TypeError, CompilerInvariantError, VerificationError)):
            entrypoint(hostile)

    assert reads == 0


@pytest.mark.parametrize("location", ("assignment", "binary_left", "unary"))
@pytest.mark.parametrize("definition_kind", ("tensor", "index"))
def test_structural_preflight_rejects_definitions_in_expression_edges(
    location: str,
    definition_kind: str,
) -> None:
    """Definition nodes cannot reach expression walkers that cannot clone them."""

    i = IndexVar("i")
    source = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    definition = source if definition_kind == "tensor" else i
    if location == "assignment":
        rhs = definition
    elif location == "binary_left":
        rhs = BinaryOp(Operation.ADD, definition, source[i])
    else:
        rhs = UnaryOp(Operation.ADD, definition)
    program = ForAll(i, TensorAssign(result[i], rhs))  # type: ignore[arg-type]

    analysis = analyze_cin(program)

    assert "unsupported_expression" in {
        diagnostic.code for diagnostic in analysis.diagnostics
    }
    with pytest.raises(VerificationError) as error:
        normalize_cin(program)
    assert "unsupported_expression" in _assert_structured_diagnostics(error)


def test_root_assignment_rhs_collection_excludes_the_result_access() -> None:
    """Collector dispatch starts at the root's specialized assignment method."""

    i = IndexVar("i")
    operand = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    program = TensorAssign(result[i], operand[i])

    assert program.get_rhs_tensor_vars() == [operand]


def _release_compile_options():
    from scorch.compiler.compile_options import CompileOptions

    return CompileOptions.from_environment(
        environ={},
        regblock_override=False,
        verify_cin_override=False,
    )


def _workspace_scheduled_graph() -> IndexStmt:
    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("SparseProduct", fmt="ds")
    left = TensorVar("SparseLeft", fmt="ds")
    right = TensorVar("SparseRight", fmt="ds")
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    return Scheduler.auto_schedule(source, compile_options=_release_compile_options())


@pytest.mark.parametrize("mode", ("normalize", "structure", "raw_lowering"))
def test_shared_expression_object_fails_closed_everywhere(mode: str) -> None:
    """A same-object BinaryOp diamond previously leaked a raw ValueError from
    kernel-ABI assembly; every raw entry now rejects it structurally."""

    i = IndexVar("i")
    left = TensorVar("A", fmt="d", shape=(4,))
    right = TensorVar("B", fmt="d", shape=(4,))
    result = TensorVar("C", fmt="d", shape=(4,))
    shared = BinaryOp(Operation.MUL, left[i], right[i])
    expr = BinaryOp(Operation.ADD, shared, shared)
    program = ForAll(i, TensorAssign(result[i], expr, op=Operation.ADD))

    with pytest.raises(VerificationError) as error:
        if mode == "normalize":
            normalize_cin(program, compile_options=_release_compile_options())
        elif mode == "structure":
            verify_cin_structure(program)
        else:
            CINLowerer(compile_options=_release_compile_options()).lower_IndexStmt(
                program
            )

    assert "duplicate_node_reference" in _assert_structured_diagnostics(error)


def test_shared_expression_chain_is_rejected_in_linear_time() -> None:
    """A 2**60-path shared-DAG chain must die in the bounded preflight, not
    hang the recursive verifier, dump, or lowering walks."""

    i = IndexVar("i")
    source = TensorVar("A", fmt="d", shape=(4,))
    result = TensorVar("C", fmt="d", shape=(4,))
    expr: Any = source[i]
    for _ in range(60):
        expr = BinaryOp(Operation.ADD, expr, expr)
    program = ForAll(i, TensorAssign(result[i], expr, op=Operation.ADD))

    with pytest.raises(VerificationError) as error:
        verify_cin_structure(program)
    assert "duplicate_node_reference" in _assert_structured_diagnostics(error)
    with pytest.raises(VerificationError) as dump_error:
        canonical_cin_dump(program)
    assert "duplicate_node_reference" in _assert_structured_diagnostics(dump_error)


def test_shared_statement_object_fails_closed() -> None:
    """One TensorAssign object under two parents previously leaked IndexError."""

    i, j = IndexVar("i"), IndexVar("j")
    result = TensorVar("C", fmt="dd", shape=(4, 4))
    source = TensorVar("A", fmt="dd", shape=(4, 4))
    assign = TensorAssign(result[i, j], source[i, j], op=Operation.ADD)
    program = ForAll(i, Where(ForAll(j, assign), ForAll(j, assign)))

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=_release_compile_options()).lower_IndexStmt(program)

    assert "duplicate_node_reference" in _assert_structured_diagnostics(error)


def test_shared_access_object_as_lhs_and_rhs_fails_closed() -> None:
    """One TensorAccess object on both assignment sides was silently admitted."""

    i = IndexVar("i")
    result = TensorVar("C", fmt="d", shape=(4,))
    access = result[i]
    program = ForAll(i, TensorAssign(access, access, op=Operation.ADD))

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=_release_compile_options()).lower_IndexStmt(program)

    assert "duplicate_node_reference" in _assert_structured_diagnostics(error)


def test_workspace_pair_admitted_but_third_occurrence_rejected() -> None:
    """The exact producer-LHS/consumer-RHS shared pair stays admitted; any
    further syntactic occurrence of the same access object fails closed."""

    graph = _workspace_scheduled_graph()
    assert CINLowerer(compile_options=_release_compile_options()).lower_IndexStmt(graph)

    grafted = _workspace_scheduled_graph()
    producer_lhs = None
    consumer_assign = None
    stack: list[Any] = [grafted]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if type(node) is TensorAssign:
            if type(node.lhs) is WorkspaceAccess:
                producer_lhs = node.lhs
            if type(node.rhs) is WorkspaceAccess:
                consumer_assign = node
        for field in ("stmt", "producer", "consumer", "lhs", "rhs", "left", "right"):
            child = getattr(node, field, None)
            if child is not None and not isinstance(child, (str, int, float)):
                stack.append(child)
    assert producer_lhs is not None and consumer_assign is not None
    consumer_assign.rhs = BinaryOp(Operation.ADD, consumer_assign.rhs, producer_lhs)

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=_release_compile_options()).lower_IndexStmt(grafted)
    assert "duplicate_node_reference" in _assert_structured_diagnostics(error)


def _tiled_spmm_graph():
    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("R", fmt="dd", shape=(4, 7))
    left = TensorVar("A", fmt="ds", shape=(4, 5))
    right = TensorVar("B", fmt="dd", shape=(5, 7))
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    options = _release_compile_options()
    return Scheduler.add_tile(source, column, 4, compile_options=options), options


@pytest.mark.parametrize("hostile_parent", ("missing_field", "deep_chain"))
def test_split_role_index_must_be_its_tile_size_var_endpoint(
    hostile_parent: str,
) -> None:
    """A forged split-role index bound to a real TileSizeVar but a foreign
    detached parent previously leaked raw KeyError or RecursionError from the
    forward copier."""

    scheduled, options = _tiled_spmm_graph()
    outer = next(iv for iv in scheduled.index_vars if iv.name == "c_out")
    tile_size_var = outer.tile_size_var
    forged = IndexVar("x")
    forged.__dict__["is_outer"] = True
    forged.__dict__["is_tiled"] = False
    forged.__dict__["is_inner"] = False
    forged.__dict__["tile_size_var"] = tile_size_var
    if hostile_parent == "missing_field":
        parent = IndexVar("bb")
        del parent.__dict__["is_tiled"]
    else:
        parent = IndexVar("h0")
        node = parent
        for position in range(3000):
            nxt = IndexVar(f"h{position + 1}")
            node.__dict__["_parent"] = nxt
            node = nxt
    forged.__dict__["_parent"] = parent
    access = next(a for a in scheduled.tensor_accesses if a.tensor.name == "R")
    access.indices[0] = forged
    access.index_ids = tuple(index.index_id for index in access.indices)

    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(scheduled)

    assert "invalid_cin_field" in _assert_structured_diagnostics(error)


def test_aliased_index_twins_require_equivalent_schedule_state() -> None:
    """Same-identity IndexVar twins with divergent tile state must not merge.

    The forged plain twin of a validated tile component is refused with a
    structured VerificationError (here at the adapter's display-name
    boundary; the preflight's schedule-state equivalence check stands behind
    it as defense in depth), never merged and never a raw exception.
    """

    scheduled, options = _tiled_spmm_graph()
    inner = next(iv for iv in scheduled.index_vars if iv.name == "c_in")
    twin = copy.copy(inner)
    twin.__dict__["is_inner"] = False
    twin.__dict__["is_outer"] = False
    twin.__dict__["_parent"] = None
    twin.__dict__["tile_size_var"] = None
    access = next(a for a in scheduled.tensor_accesses if a.tensor.name == "R")
    access.indices[0] = twin
    access.index_ids = tuple(index.index_id for index in access.indices)

    with pytest.raises(VerificationError):
        CINLowerer(compile_options=options).lower_IndexStmt(scheduled)


def test_workspace_clone_of_tiled_logical_index_stays_admitted() -> None:
    """Workspace insertion legitimately pairs a tiled logical index with a
    plain clone across branches; that historical divergence must lower."""

    from scorch.compiler.compile_options import CompileOptions

    row, reduction, column = IndexVar("i"), IndexVar("k"), IndexVar("j")
    result = TensorVar("C", fmt="ds", shape=(4, 7))
    left = TensorVar("A", fmt="ds", shape=(4, 5))
    right = TensorVar("B", fmt="ds", shape=(5, 7))
    source = ForAll(
        row,
        ForAll(
            reduction,
            ForAll(
                column,
                TensorAssign(
                    result[row, column],
                    left[row, reduction] * right[reduction, column],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    options = CompileOptions.from_environment(
        environ={"SCORCH_REGBLOCK": "1", "SCORCH_REGBLOCK_T": "16"},
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )
    scheduled = Scheduler.auto_schedule(source, compile_options=options)

    assert CINLowerer(compile_options=options).lower_IndexStmt(scheduled)
