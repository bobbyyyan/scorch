"""Workspace insertion builds strictly structured CIN (post-merge item 38).

``Scheduler.insert_workspace`` used to deep-copy the reduction subtree twice, once
for the producer and once for the consumer.  The copies kept every ``node_id``,
duplicated every index variable, and shared one ``WorkspaceAccess`` object between
the producer's left side and the consumer's right side; the structure check refused
that tree, so the legacy lowering admitted it through a narrow exemption.  The
producer and consumer are now copies that share the subtree's index variables and
tensors and have their own node ids and their own workspace accesses, the
structure check runs on every insertion and tile, and the exemption is gone.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, Iterator, List, Tuple

import pytest

import scorch.compiler.scheduler as scheduler_module
from scorch.compiler.cin import (
    CIN,
    BinaryOp,
    ForAll,
    IndexStmt,
    IndexVar,
    Operation,
    TensorAccess,
    TensorAssign,
    TensorVar,
    UnaryOp,
    Where,
    Workspace,
    WorkspaceAccess,
    copy_statement,
)
from scorch.compiler.cin_analysis import normalize_cin, verify_cin_structure
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import UnsupportedFeature, VerificationError
from scorch.compiler.legacy_cin_adapter import (
    _canonicalize_legacy_entities,
    legacy_cin_working_copy,
)
from scorch.compiler.scheduler import Schedule, Scheduler, TileSpec
from scorch.exceptions import defect_code

ARMS = (False, True)


def _options(regblock: bool, verify: bool = False) -> CompileOptions:
    return CompileOptions.from_environment(
        environ={}, regblock_override=regblock, verify_cin_override=verify
    )


def _matmul(fa: str, fb: str, fc: str) -> Tuple[ForAll, Dict[str, IndexVar]]:
    """C[i,k] += A[i,j] * B[j,k] in loop order i, j, k (j is the reduction)."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", shape=(8, 12), fmt=fa)
    b = TensorVar("B", shape=(12, 18), fmt=fb)
    c = TensorVar("C", shape=(8, 18), fmt=fc)
    stmt = ForAll(
        i,
        ForAll(
            j, ForAll(k, TensorAssign(c[i, k], a[i, j] * b[j, k], op=Operation.ADD))
        ),
    )
    return stmt, {"i": i, "j": j, "k": k}


def _two_level_producer() -> Tuple[ForAll, Dict[str, IndexVar]]:
    """C[i,k,m] += A[i,j] * B[j,k,m]: two free loops below the reduction j, so the
    producer has two levels below it and the workspace has two dimensions."""

    i, j, k, m = IndexVar("i"), IndexVar("j"), IndexVar("k"), IndexVar("m")
    a = TensorVar("A", shape=(4, 5), fmt="ds")
    b = TensorVar("B", shape=(5, 6, 7), fmt="ddd")
    c = TensorVar("C", shape=(4, 6, 7), fmt="ddd")
    stmt = ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                ForAll(
                    m,
                    TensorAssign(c[i, k, m], a[i, j] * b[j, k, m], op=Operation.ADD),
                ),
            ),
        ),
    )
    return stmt, {"i": i, "j": j, "k": k, "m": m}


def _nodes(root: CIN) -> Iterator[Tuple[CIN, Tuple[str, ...]]]:
    """Every statement, expression and access node, with its path."""

    stack: List[Tuple[CIN, Tuple[str, ...]]] = [(root, ("root",))]
    while stack:
        node, path = stack.pop()
        yield node, path
        if isinstance(node, ForAll):
            stack.append((node.stmt, path + ("stmt",)))
        elif isinstance(node, Where):
            stack.append((node.producer, path + ("producer",)))
            stack.append((node.consumer, path + ("consumer",)))
        elif isinstance(node, TensorAssign):
            stack.append((node.lhs, path + ("lhs",)))
            stack.append((node.rhs, path + ("rhs",)))
        elif isinstance(node, BinaryOp):
            stack.append((node.left, path + ("left",)))
            stack.append((node.right, path + ("right",)))
        elif isinstance(node, UnaryOp):
            stack.append((node.expr, path + ("expr",)))


def _entity_references(root: CIN) -> List[Tuple[Tuple[str, ...], int]]:
    """(path, id(object)) for every index variable and tensor the tree names."""

    references: List[Tuple[Tuple[str, ...], int]] = []
    for node, path in _nodes(root):
        if isinstance(node, ForAll):
            references.append((path + ("index_var",), id(node.index_var)))
        elif isinstance(node, TensorAccess):
            references.append((path + ("tensor",), id(node.tensor)))
            for axis, index_var in enumerate(node.indices):
                references.append((path + (f"indices[{axis}]",), id(index_var)))
    return references


def _assert_one_object_per_entity(root: CIN) -> None:
    index_objects: Dict[object, int] = {}
    tensor_objects: Dict[object, int] = {}
    node_ids = set()
    access_ids = set()
    for node, path in _nodes(root):
        assert node.node_id not in node_ids, path
        node_ids.add(node.node_id)
        if isinstance(node, ForAll):
            index_vars = [node.index_var]
        elif isinstance(node, TensorAccess):
            assert node.access_id not in access_ids, path
            access_ids.add(node.access_id)
            assert tensor_objects.setdefault(
                node.tensor.symbol_id, id(node.tensor)
            ) == id(node.tensor), path
            index_vars = list(node.indices)
        else:
            index_vars = []
        for index_var in index_vars:
            assert index_objects.setdefault(index_var.index_id, id(index_var)) == id(
                index_var
            ), path


def _where(root: CIN) -> Where:
    for node, _ in _nodes(root):
        if isinstance(node, Where):
            return node
    raise AssertionError("no workspace was inserted")


@pytest.mark.parametrize("allow_dense", (False, True))
@pytest.mark.parametrize(
    "formats", (("ds", "dd", "dd"), ("ds", "ds", "ds"), ("ds", "dd", "ds"))
)
def test_insert_workspace_output_is_strictly_structured(
    formats: Tuple[str, str, str], allow_dense: bool
) -> None:
    stmt, _ = _matmul(*formats)
    scheduled = Scheduler.insert_workspace(stmt, allow_dense=allow_dense)
    if not scheduled.inserted_workspace:
        # A dense receiver needs no workspace unless dense ones are allowed.
        assert formats[2] == "dd" and not allow_dense
        return

    verify_cin_structure(scheduled)
    normalize_cin(scheduled)
    _assert_one_object_per_entity(scheduled)

    where = _where(scheduled)
    accesses = [node for node, _ in _nodes(where) if isinstance(node, WorkspaceAccess)]
    assert len(accesses) == 2
    assert accesses[0] is not accesses[1]
    assert accesses[0].access_id != accesses[1].access_id
    assert accesses[0].tensor is accesses[1].tensor
    assert isinstance(accesses[0].tensor, Workspace)
    assert [index.index_id for index in accesses[0].indices] == [
        index.index_id for index in accesses[1].indices
    ]


def test_producer_and_consumer_share_index_variables_and_tensors() -> None:
    stmt, names = _matmul("ds", "ds", "ds")
    scheduled = Scheduler.insert_workspace(stmt, allow_dense=True)
    where = _where(scheduled)

    def first(node: IndexStmt, name: str) -> IndexVar:
        for candidate, _ in _nodes(node):
            if isinstance(candidate, ForAll) and candidate.index_var.name == name:
                return candidate.index_var
            if isinstance(candidate, TensorAccess):
                for index_var in candidate.indices:
                    if index_var.name == name:
                        return index_var
        raise AssertionError(name)

    for name in ("i", "j", "k"):
        assert first(where.producer, name) is first(where.consumer, name)
    outer = scheduled
    assert isinstance(outer, ForAll) and outer.index_var.name == "i"
    assert first(where.producer, "i") is outer.index_var
    # The caller's tree is untouched: insertion works on its own copy.
    assert scheduled.index_var is not names["i"]
    assert stmt.index_var is names["i"]


@pytest.mark.parametrize("two_levels", (False, True))
@pytest.mark.parametrize("target", ("i", "j", "k"))
def test_tiles_after_insertion_are_strictly_structured(
    target: str, two_levels: bool
) -> None:
    stmt, names = _two_level_producer() if two_levels else _matmul("ds", "dd", "dd")
    try:
        tiled = Scheduler.add_tile(stmt, names[target], 4, use_workspace=True)
    except Exception as error:  # noqa: BLE001 - a refusal is a structured code
        # Tiling a sparse loop is refused before any tree is built.
        assert getattr(error, "code", None) == "sparse_index_affine_tiling_unsupported"
        assert target == "j"
        return
    verify_cin_structure(tiled)
    _assert_one_object_per_entity(tiled)
    # One logical index object carries the tile everywhere it is named.
    tiled_logical = [
        index_var
        for node, _ in _nodes(tiled)
        if isinstance(node, TensorAccess)
        for index_var in node.indices
        if index_var.name == target
    ]
    assert tiled_logical
    assert all(index_var is tiled_logical[0] for index_var in tiled_logical)
    assert tiled_logical[0].is_tiled


@pytest.mark.parametrize("regblock", ARMS)
@pytest.mark.parametrize(
    "formats", (("ds", "dd", "dd"), ("ds", "ds", "ds"), ("dd", "ds", "dd"))
)
def test_automatic_schedules_are_strictly_structured(
    formats: Tuple[str, str, str], regblock: bool
) -> None:
    stmt, _ = _matmul(*formats)
    scheduled = Scheduler.auto_schedule(stmt, compile_options=_options(regblock))
    verify_cin_structure(scheduled)
    _assert_one_object_per_entity(scheduled)

    replayed = Scheduler.auto_schedule_plan(stmt, compile_options=_options(regblock))
    working = legacy_cin_working_copy(
        replayed.normalized_cin,
        replayed.verified_loop_plan,
        compile_options=_options(regblock),
    )
    _assert_one_object_per_entity(working)


@pytest.mark.parametrize("regblock", ARMS)
@pytest.mark.parametrize(
    "formats", (("ds", "dd", "dd"), ("ds", "ds", "ds"), ("dd", "ds", "dd"))
)
def test_canonicalization_changes_nothing_in_scheduler_output(
    formats: Tuple[str, str, str], regblock: bool
) -> None:
    stmt, _ = _matmul(*formats)
    scheduled = Scheduler.auto_schedule(stmt, compile_options=_options(regblock))
    before = _entity_references(scheduled)
    _canonicalize_legacy_entities(scheduled)
    assert _entity_references(scheduled) == before


def _tiled_names(stmt: IndexStmt) -> List[str]:
    """The index variables a tile split (their ``_expr`` is a sum)."""

    return sorted({v.name for v in stmt.index_vars if v._expr is not None})


@pytest.mark.parametrize("regblock", ARMS)
def test_debug_lowering_verifies_caller_owned_workspace_schedules(
    regblock: bool,
) -> None:
    """The debug lowerer verifies a caller-owned untiled schedule, workspace
    insertion included, and refuses a caller-owned tiled one: full CIN
    verification is defined over normalized CIN, which has no tile."""

    options = _options(regblock, verify=True)
    for formats in (("ds", "dd", "dd"), ("ds", "ds", "ds")):
        stmt, _ = _matmul(*formats)
        scheduled = Scheduler.auto_schedule(stmt, compile_options=options)
        # Register blocking tiles k of the dense-operand product only.
        tiled = regblock and formats == ("ds", "dd", "dd")
        assert _tiled_names(scheduled) == (["k"] if tiled else [])
        if not tiled:
            assert CINLowerer(compile_options=options).lower_IndexStmt(scheduled)
            continue
        with pytest.raises(UnsupportedFeature) as error:
            CINLowerer(compile_options=options).lower_IndexStmt(scheduled)
        assert defect_code(error.value) == "caller_tiled_cin_unsupported"
        assert {d.code for d in error.value.diagnostics} == {
            "caller_tiled_cin_unsupported"
        }


def test_debug_lowering_refuses_a_caller_tile_without_a_workspace() -> None:
    """A caller's tile with no workspace insertion is refused the same way; the
    verifier used to report the split index as unbound (dangling_index_reference)."""

    stmt, names = _matmul("ds", "dd", "dd")
    options = _options(False, verify=True)
    tiled = Scheduler.add_tile(
        stmt, names["k"], 4, use_workspace=False, compile_options=options
    )
    assert not tiled.inserted_workspace and _tiled_names(tiled) == ["k"]
    with pytest.raises(UnsupportedFeature) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(tiled)
    assert defect_code(error.value) == "caller_tiled_cin_unsupported"
    # The same tile lowers without verification.
    release = _options(False)
    release_stmt, release_names = _matmul("ds", "dd", "dd")
    release_tiled = Scheduler.add_tile(
        release_stmt,
        release_names["k"],
        4,
        use_workspace=False,
        compile_options=release,
    )
    assert CINLowerer(compile_options=release).lower_IndexStmt(release_tiled)


@pytest.mark.parametrize("formats", (("ds", "dd", "dd"), ("ds", "ds", "ds")))
@pytest.mark.parametrize("enabled", ARMS)
def test_debug_lowering_of_the_compiler_owned_register_block_arms(
    enabled: bool, formats: Tuple[str, str, str]
) -> None:
    """einsum's register-block dual path lowers both arms it schedules, compiler
    owned, under full CIN verification as without it: a tiled arm is not verified
    again (the scheduler verified its normalized source before tiling it), an
    untiled one is."""

    for verify in ARMS:
        stmt, _ = _matmul(*formats)
        options = _options(True, verify=verify)
        arm = Scheduler._auto_schedule_regblock_arm(
            stmt, enabled=enabled, compile_options=options
        )
        tiled = enabled and formats == ("ds", "dd", "dd")
        assert _tiled_names(arm) == (["k"] if tiled else [])
        assert CINLowerer(compile_options=options)._lower_owned_IndexStmt(arm)


def test_copy_statement_shares_entities_and_renews_node_ids() -> None:
    stmt, names = _matmul("ds", "dd", "dd")
    copied = copy_statement(stmt)
    assert isinstance(copied, ForAll)
    original_nodes = {id(node): node for node, _ in _nodes(stmt)}
    for node, path in _nodes(copied):
        assert id(node) not in original_nodes, path
    assert {node.node_id for node, _ in _nodes(copied)}.isdisjoint(
        {node.node_id for node, _ in _nodes(stmt)}
    )
    assert _entity_references(copied) == _entity_references(stmt)
    copied_accesses = [n for n, _ in _nodes(copied) if isinstance(n, TensorAccess)]
    original_accesses = [n for n, _ in _nodes(stmt) if isinstance(n, TensorAccess)]
    assert [a.access_id for a in copied_accesses] == [
        a.access_id for a in original_accesses
    ]
    assert copied.index_var is names["i"]


def test_insert_workspace_check_refuses_deep_copied_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the old subtree copy is refused at insertion."""

    monkeypatch.setattr(scheduler_module, "copy_statement", copy.deepcopy)
    stmt, _ = _matmul("ds", "ds", "ds")
    with pytest.raises(VerificationError) as error:
        Scheduler.insert_workspace(stmt, allow_dense=True)
    codes = {diagnostic.code for diagnostic in error.value.diagnostics}
    assert "duplicate_node_id" in codes


@pytest.mark.parametrize("provenance", ("auto", "schedule"))
def test_replayed_tree_is_checked_before_canonicalization(
    monkeypatch: pytest.MonkeyPatch, provenance: str
) -> None:
    """Positive control: a duplicate planted in the replayed tree is refused
    before canonicalization could merge it."""

    stmt, _ = _matmul("ds", "dd", "dd")
    options = _options(False)
    if provenance == "auto":
        scheduled = Scheduler.auto_schedule_plan(stmt, compile_options=options)
        name = "_replay_auto_plan_owned"
    else:
        schedule = Schedule(loop_order=("i", "j", "k"), tiles=(TileSpec("k", 4),))
        scheduled = Scheduler.apply_schedule(
            stmt,
            schedule,
            compile_options=replace(options, requested_schedule=schedule),
        )
        options = replace(options, requested_schedule=schedule)
        name = "_apply_schedule_legacy"
    original = getattr(Scheduler, name)

    def plant(root: CIN) -> None:
        loops = [node for node, _ in _nodes(root) if isinstance(node, ForAll)]
        loops[1].node_id = loops[0].node_id

    if provenance == "auto":

        def replay(cin, plan, **kwargs):  # type: ignore[no-untyped-def]
            result = original(cin, plan, **kwargs)
            plant(result)
            return result

    else:

        def replay(cin, schedule, **kwargs):  # type: ignore[no-untyped-def]
            result = original(cin, schedule, **kwargs)
            plant(result.normalized_cin)
            return result

    monkeypatch.setattr(Scheduler, name, staticmethod(replay))
    with pytest.raises(VerificationError) as error:
        legacy_cin_working_copy(
            scheduled.normalized_cin,
            scheduled.verified_loop_plan,
            compile_options=options,
        )
    codes = {diagnostic.code for diagnostic in error.value.diagnostics}
    assert "duplicate_node_id" in codes
