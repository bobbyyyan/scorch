from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from time import perf_counter_ns
from typing import Callable, List, NoReturn, Sequence, Set, Tuple, cast

import pytest

from scorch.compiler import llir  # type: ignore[import-untyped]
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
import scorch.compiler.llir_pass_manager as pass_manager_module  # type: ignore[import-untyped]
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.compiler.llir_pass_manager import (  # type: ignore[import-untyped]
    DEBUG_LLIR_PASS_OPTIONS,
    LLIRPassArtifactType,
    LLIRPassContextType,
    LLIRPassDescriptor,
    LLIRPassManager,
    LLIRPassManagerError,
    LLIRPassOptions,
    LLIRRewriteArtifact,
    LLIRStatementListArtifact,
    LLIRStatementListPassResult,
    PRODUCTION_LLIR_PASS_OPTIONS,
    SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
    SingleIterationLoopEliminationPassSpec,
)
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from scorch.compiler.single_iteration_loop_pass import (  # type: ignore[import-untyped]
    SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    SINGLE_ITERATION_LOOP_ELIMINATION_TRAVERSAL_CONTEXT,
    SingleIterationLoopEliminationContext,
    eliminate_single_iteration_loops,
)


def _var(
    name: str,
    data_type: llir.DataType = llir.DataType.NO_TYPE,
    *,
    is_ptr: bool = False,
    is_restrict: bool = False,
    tensor_access: llir.TensorAccessMetadata | None = None,
) -> llir.Var:
    return llir.Var(
        name=name,
        type=data_type,
        is_ptr=is_ptr,
        is_restrict=is_restrict,
        tensor_access=tensor_access,
    )


def _access(array: str, index: str) -> llir.ArrayAccess:
    return llir.ArrayAccess(
        array=_var(array),
        index=_var(index, llir.DataType.INT64),
    )


def _bound(
    end_variable: str = "end",
    base: str = "base",
    *,
    value: llir.Expr | None = None,
) -> llir.VarInit:
    if value is None:
        value = _var(f"{base} + 1", llir.DataType.INT64)
    return llir.VarInit(_var(end_variable, llir.DataType.INT64), value)


def _loop(
    body: Sequence[llir.Stmt],
    *,
    loop_variable: str = "lane",
    base: str = "base",
    end_variable: str = "end",
    initial_value: llir.Expr | None = None,
    condition: llir.Expr | None = None,
    update: llir.Stmt | llir.FunctionCall | None = None,
) -> llir.ForLoop:
    if initial_value is None:
        initial_value = _var(base, llir.DataType.INT64)
    if condition is None:
        condition = llir.BinOp(
            "<",
            _var("ignored_condition_left", llir.DataType.INT64),
            _var(end_variable, llir.DataType.INT64),
        )
    if update is None:
        update = llir.Increment(_var(loop_variable, llir.DataType.INT64))
    return llir.ForLoop(
        init=llir.VarInit(
            _var(loop_variable, llir.DataType.INT64),
            initial_value,
        ),
        cond=condition,
        update=cast(
            llir.Increment | llir.VarInit | llir.FunctionCall | llir.Assign,
            update,
        ),
        body=list(body),
    )


def _program(
    body: Sequence[llir.Stmt] = (llir.RawStmt("consume(lane)"),),
    *,
    loop_variable: str = "lane",
    base: str = "base",
    end_variable: str = "end",
    initial_value: llir.Expr | None = None,
    bound_value: llir.Expr | None = None,
    declaration_after: bool = False,
) -> List[llir.Stmt]:
    declaration = _bound(end_variable, base, value=bound_value)
    loop = _loop(
        body,
        loop_variable=loop_variable,
        base=base,
        end_variable=end_variable,
        initial_value=initial_value,
    )
    if declaration_after:
        return [loop, declaration]
    return [declaration, loop]


def _mutable_ir_ids(value: object) -> Set[int]:
    mutable_ids: Set[int] = set()
    if isinstance(value, llir.Node):
        mutable_ids.add(id(value))
        for child in vars(value).values():
            mutable_ids.update(_mutable_ir_ids(child))
    elif isinstance(value, list):
        mutable_ids.add(id(value))
        for child in value:
            mutable_ids.update(_mutable_ir_ids(child))
    elif isinstance(value, tuple):
        for child in value:
            mutable_ids.update(_mutable_ir_ids(child))
    return mutable_ids


def _snapshot(value: object) -> object:
    if isinstance(value, llir.Node):
        return (
            type(value).__name__,
            tuple(
                (name, _snapshot(child)) for name, child in sorted(vars(value).items())
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_snapshot(child) for child in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_snapshot(child) for child in value))
    return value


def _raw_codes(statements: Sequence[object]) -> List[str]:
    return [
        cast(llir.RawStmt, statement).code
        for statement in statements
        if type(statement) is llir.RawStmt
    ]


def _p95(samples: List[int]) -> int:
    ordered = sorted(samples)
    return ordered[int((len(ordered) - 1) * 0.95)]


def test_stable_api_and_all_managed_carriers_are_frozen() -> None:
    context = SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT
    spec = SingleIterationLoopEliminationPassSpec(context)
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    result = LLIRPassManager().run_single_iteration_loop_elimination(
        artifact,
        spec,
    )
    record = result.run_records[0]

    assert context == SingleIterationLoopEliminationContext(
        SINGLE_ITERATION_LOOP_ELIMINATION_TRAVERSAL_CONTEXT
    )
    assert SINGLE_ITERATION_LOOP_ELIMINATION_PASS == LLIRPassDescriptor(
        name="eliminate_single_iteration_loops",
        version=1,
        input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        output_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        context_type=LLIRPassContextType.SINGLE_ITERATION_LOOP_ELIMINATION,
    )
    assert record.pass_name == "eliminate_single_iteration_loops"
    assert record.pass_version == 1
    assert record.configuration_name == "single_iteration_loop_elimination"
    assert record.input_artifact is LLIRPassArtifactType.STATEMENT_LIST
    assert record.output_artifact is LLIRPassArtifactType.STATEMENT_LIST
    assert record.context_type is LLIRPassContextType.SINGLE_ITERATION_LOOP_ELIMINATION
    assert record.diagnostic_stage == "LLIR transformation"
    assert record.diagnostic_pass_name == "eliminate_single_iteration_loops"

    frozen_updates: Tuple[Tuple[object, str, object], ...] = (
        (context, "traversal", LLIRTraversalContext("other", "other")),
        (context.traversal, "pass_name", "different"),
        (SINGLE_ITERATION_LOOP_ELIMINATION_PASS, "version", 2),
        (
            spec,
            "descriptor",
            replace(SINGLE_ITERATION_LOOP_ELIMINATION_PASS, version=2),
        ),
        (artifact, "statements", []),
        (result, "run_records", ()),
        (result.artifact, "statements", []),
        (record, "duration_ns", None),
    )
    for value, field_name, replacement in frozen_updates:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, replacement)


class _EqualitySpoof:
    def __eq__(self, other: object) -> bool:
        return True


def test_runner_rejects_descriptor_equality_spoof_and_wrong_exact_types() -> None:
    manager = LLIRPassManager()
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    context = SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT
    invalid_calls: Tuple[Callable[[], object], ...] = (
        lambda: manager.run_single_iteration_loop_elimination(
            artifact,
            SingleIterationLoopEliminationPassSpec(
                context,
                descriptor=cast(LLIRPassDescriptor, _EqualitySpoof()),
            ),
        ),
        lambda: manager.run_single_iteration_loop_elimination(
            artifact,
            SingleIterationLoopEliminationPassSpec(
                context,
                descriptor=replace(
                    SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
                    version=2,
                ),
            ),
        ),
        lambda: manager.run_single_iteration_loop_elimination(
            cast(LLIRStatementListArtifact, LLIRRewriteArtifact([])),
            SingleIterationLoopEliminationPassSpec(context),
        ),
        lambda: manager.run_single_iteration_loop_elimination(
            artifact,
            cast(SingleIterationLoopEliminationPassSpec, object()),
        ),
        lambda: manager.run_single_iteration_loop_elimination(
            artifact,
            SingleIterationLoopEliminationPassSpec(
                cast(SingleIterationLoopEliminationContext, object())
            ),
        ),
    )

    for invalid_call in invalid_calls:
        with pytest.raises(LLIRPassManagerError):
            invalid_call()


@pytest.mark.parametrize(
    "root",
    (
        llir.BlankLine(),
        (llir.BlankLine(),),
        None,
        "not a statement list",
    ),
)
def test_direct_pass_requires_an_exact_statement_list_root(root: object) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            cast(List[llir.Stmt], root),
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unsupported_single_iteration_loop_elimination_root"
    assert diagnostic.path == ("root",)
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "eliminate_single_iteration_loops"
    assert diagnostic.node_type == type(root).__name__


def test_nested_root_sequences_are_detached_and_semantically_omitted() -> None:
    list_program = _program([llir.RawStmt("list_body")])
    tuple_program = tuple(_program([llir.RawStmt("tuple_body")]))
    source = cast(List[llir.Stmt], [list_program, tuple_program])
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert type(output) is list
    assert type(output[0]) is list
    assert type(output[1]) is tuple
    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_every_legal_noop_is_detached_with_fields_metadata_and_raw_code() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(11),
        tensor_id=SymbolId(12),
        index_ids=(IndexId(13), IndexId(14)),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    loop = _loop(
        [
            llir.Assign(
                _var("output"),
                _var(
                    "Input_val[unrelated]",
                    llir.DataType.PTR_FLOAT32,
                    is_ptr=True,
                    is_restrict=True,
                    tensor_access=metadata,
                ),
            ),
            llir.RawStmt("raw_without_semicolon", add_semicolon=False),
        ],
        end_variable="missing_end",
    )
    loop.omp_parallel_for = True
    loop.omp_schedule = "dynamic, 7"
    loop.unroll = True
    loop.simd = True
    loop.omp_num_threads = "threads"
    loop.omp_chunk_expr = "chunk"
    loop.scorch_index_var = "logical"
    loop.before_parallel_body = [llir.RawStmt("before", add_semicolon=False)]
    loop.pre_parallel_body = [llir.RawStmt("pre")]
    loop.post_parallel_body = [llir.RawStmt("post")]
    setattr(loop, "_use_atomic_scheduling", True)
    setattr(loop, "_atomic_chunk_var", "atomic_chunk")
    setattr(loop, "_atomic_counter_var", "atomic_counter")
    setattr(loop, "_loop_bound", "bound")
    setattr(loop, "_hoisted_ptr_decls", [llir.RawStmt("hoisted")])
    source: List[llir.Stmt] = [_bound(), loop]
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))
    output_loop = cast(llir.ForLoop, output[1])
    output_assignment = cast(llir.Assign, output_loop.body[0])
    output_value = cast(llir.Var, output_assignment.value)
    assert output_value.tensor_access == metadata
    assert output_value.tensor_access is metadata
    assert cast(llir.RawStmt, output_loop.body[1]).add_semicolon is False
    assert (
        cast(llir.RawStmt, output_loop.before_parallel_body[0]).add_semicolon is False
    )
    assert getattr(output_loop, "_hoisted_ptr_decls") is not getattr(
        loop,
        "_hoisted_ptr_decls",
    )


@pytest.mark.parametrize(
    ("loop_variable", "base", "end_variable"),
    (
        ("coordinate_lane", "parent_position", "arbitrary_end"),
        ("x7", "base_9", "limit_3"),
        ("unicode_lane", "parent", "last"),
    ),
)
def test_var_initializer_matching_accepts_arbitrary_names(
    loop_variable: str,
    base: str,
    end_variable: str,
) -> None:
    source = _program(
        [llir.RawStmt(f"load(Array[{loop_variable}])")],
        loop_variable=loop_variable,
        base=base,
        end_variable=end_variable,
    )
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _raw_codes(output) == [f"load(Array[{base}])"]
    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_literal_initializer_matches_the_string_form_of_its_value() -> None:
    source = _program(
        [llir.RawStmt("read(Array[lane])")],
        base="17",
        initial_value=llir.Literal(17),
    )

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _raw_codes(output) == ["read(Array[17])"]


@pytest.mark.parametrize(
    "bound_spelling",
    (
        "base+1",
        "base  + 1",
        "base +  1",
        "base + 1 ",
        " base + 1",
        "base\t+ 1",
        "base + 2",
        "base-name + 1",
    ),
)
def test_bound_regex_and_spacing_are_exact(bound_spelling: str) -> None:
    source = _program(bound_value=_var(bound_spelling))
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_exact_binop_condition_ignores_condition_left_and_update() -> None:
    condition = llir.BinOp(
        "<",
        llir.FunctionCall("unrelated", []),
        _var("end"),
    )
    loop = _loop(
        [llir.RawStmt("read(A[lane])")],
        condition=condition,
        update=llir.Assign(_var("different"), llir.Literal(99)),
    )
    source: List[llir.Stmt] = [_bound(), loop]

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _raw_codes(output) == ["read(A[base])"]


def test_structured_add_bound_matches_and_rebuilds_detached_references() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(41),
        tensor_id=SymbolId(42),
        index_ids=(IndexId(43),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    bound_value = llir.Add(
        _var("base", llir.DataType.INT64),
        llir.Literal(1, llir.DataType.INT64),
    )
    body_value = llir.Add(
        _var(
            "lane value",
            llir.DataType.PTR_FLOAT32,
            is_ptr=True,
            is_restrict=True,
            tensor_access=metadata,
        ),
        llir.Literal(2, llir.DataType.INT64),
    )
    source = _program(
        [
            llir.Assign(_var("output"), body_value),
            llir.RawStmt("read(A[lane])"),
        ],
        bound_value=bound_value,
    )
    snapshot = _snapshot(source)

    once = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )
    twice = eliminate_single_iteration_loops(
        once,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(source) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once))
    assert _mutable_ir_ids(once).isdisjoint(_mutable_ir_ids(twice))
    assert _snapshot(twice) == _snapshot(once)
    assert _raw_codes(once) == ["read(A[base])"]
    rewritten = cast(llir.Add, cast(llir.Assign, once[0]).value)
    repeated = cast(llir.Add, cast(llir.Assign, twice[0]).value)
    assert type(rewritten) is llir.Add
    assert rewritten is not body_value
    assert rewritten == llir.Add(
        _var(
            "base value",
            llir.DataType.PTR_FLOAT32,
            is_ptr=True,
            is_restrict=True,
            tensor_access=metadata,
        ),
        llir.Literal(2, llir.DataType.INT64),
    )
    assert cast(llir.Var, rewritten.left).tensor_access is metadata
    assert repeated == rewritten
    assert repeated is not rewritten
    assert repeated.left is not rewritten.left
    assert repeated.right is not rewritten.right


@pytest.mark.parametrize(
    "bound_value",
    (
        llir.BinOp(
            "+",
            _var("base", llir.DataType.INT64),
            llir.Literal(1, llir.DataType.INT64),
        ),
        llir.Mul(
            _var("base", llir.DataType.INT64),
            llir.Literal(1, llir.DataType.INT64),
        ),
        llir.Add(
            llir.Literal(1, llir.DataType.INT64),
            _var("base", llir.DataType.INT64),
        ),
        llir.Add(
            _var("base", llir.DataType.INT64),
            llir.Literal(2, llir.DataType.INT64),
        ),
        llir.Add(
            _var("base", llir.DataType.INT64),
            llir.Literal(1, llir.DataType.INT),
        ),
        llir.Add(
            _var("base", llir.DataType.INT),
            llir.Literal(1, llir.DataType.INT64),
        ),
        llir.Add(
            _var("base", llir.DataType.INT64, is_ptr=True),
            llir.Literal(1, llir.DataType.INT64),
        ),
        llir.Add(
            _var("base", llir.DataType.INT64, is_restrict=True),
            llir.Literal(1, llir.DataType.INT64),
        ),
        llir.Add(
            _var(
                "base",
                llir.DataType.INT64,
                tensor_access=llir.TensorAccessMetadata(
                    access_id=AccessId(51),
                    tensor_id=SymbolId(52),
                    index_ids=(IndexId(53),),
                    role=llir.TensorAccessRole.INPUT_READ,
                ),
            ),
            llir.Literal(1, llir.DataType.INT64),
        ),
    ),
)
def test_structured_single_step_bound_requires_the_exact_legal_shape(
    bound_value: llir.Expr,
) -> None:
    source = _program(bound_value=bound_value)
    snapshot = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(output) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_structured_single_step_bound_requires_a_word_base_name() -> None:
    source = _program(
        base="base-name",
        bound_value=llir.Add(
            _var("base-name", llir.DataType.INT64),
            llir.Literal(1, llir.DataType.INT64),
        ),
    )
    snapshot = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(output) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


class _UnknownSingleStepAdd(llir.Add):
    pass


def test_structured_single_step_bound_unknown_subclass_fails_closed() -> None:
    source = _program(
        bound_value=_UnknownSingleStepAdd(
            _var("base", llir.DataType.INT64),
            llir.Literal(1, llir.DataType.INT64),
        )
    )

    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            source,
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.path == ("root", "[0]", "value")
    assert diagnostic.node_type == "_UnknownSingleStepAdd"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "eliminate_single_iteration_loops"


def test_structured_single_step_bound_forged_operator_fails_validation() -> None:
    value = llir.Add(
        _var("base", llir.DataType.INT64),
        llir.Literal(1, llir.DataType.INT64),
    )
    object.__setattr__(value, "op", "-")

    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            _program(bound_value=value),
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_add_operator"
    assert diagnostic.path == ("root", "[0]", "value", "op")
    assert diagnostic.node_type == "str"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "eliminate_single_iteration_loops"


@pytest.mark.parametrize(
    "loop",
    (
        _loop([], condition=llir.BinOp("<=", _var("lane"), _var("end"))),
        _loop([], condition=llir.Add(_var("lane"), _var("end"))),
        _loop([], condition=llir.Mul(_var("lane"), _var("end"))),
        _loop([], condition=llir.BinOp("<", _var("lane"), llir.Literal(1))),
        _loop([], initial_value=llir.FunctionCall("initial", [])),
        _loop([], loop_variable="base"),
    ),
)
def test_candidate_shape_misses_are_detached_noops(loop: llir.ForLoop) -> None:
    source: List[llir.Stmt] = [_bound(), loop]
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_declaration_after_loop_is_visible_to_analysis() -> None:
    source = _program(
        [llir.RawStmt("read(A[lane])")],
        declaration_after=True,
    )

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _raw_codes(output) == ["read(A[base])"]


def test_last_matching_bound_wins_and_only_its_loop_matches() -> None:
    first_loop = _loop(
        [llir.RawStmt("first")],
        base="first_base",
    )
    second_loop = _loop(
        [llir.RawStmt("second")],
        base="second_base",
    )
    source: List[llir.Stmt] = [
        _bound(base="first_base"),
        first_loop,
        _bound(base="second_base"),
        second_loop,
    ]

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert len(output) == 2
    assert type(output[0]) is llir.ForLoop
    assert cast(
        llir.Var, cast(llir.VarInit, cast(llir.ForLoop, output[0]).init).value
    ).name == ("first_base")
    assert _raw_codes(output) == ["second"]


def test_success_removes_every_direct_declaration_with_the_matched_end() -> None:
    loop = _loop([llir.RawStmt("body")])
    source: List[llir.Stmt] = [
        _bound(),
        llir.VarInit(_var("end"), llir.Literal(99)),
        loop,
        llir.VarInit(_var("end"), llir.FunctionCall("unrelated", [])),
    ]

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _raw_codes(output) == ["body"]
    assert not any(
        type(statement) is llir.VarInit
        and cast(llir.VarInit, statement).var.name == "end"
        for statement in output
    )


def test_bound_snapshot_eliminates_multiple_loops_and_preserves_body_order() -> None:
    source: List[llir.Stmt] = [
        _bound(),
        _loop([llir.RawStmt("first"), llir.RawStmt("second")]),
        llir.Comment("between"),
        _loop([]),
        _loop([llir.RawStmt("third")]),
        llir.RawStmt("tail"),
    ]

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert [type(statement) for statement in output] == [
        llir.RawStmt,
        llir.RawStmt,
        llir.Comment,
        llir.RawStmt,
        llir.RawStmt,
    ]
    assert _raw_codes(output) == ["first", "second", "third", "tail"]


def test_exact_substring_rewrite_order_and_positive_scope() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(21),
        tensor_id=SymbolId(22),
        index_ids=(IndexId(23),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    decorated = _var(
        "prefix_ix]_[ix]_ix tail",
        llir.DataType.PTR_FLOAT64,
        is_ptr=True,
        is_restrict=True,
        tensor_access=metadata,
    )
    nested_loop = _loop(
        [llir.Assign(_access("Nested", "ix"), _var("ix nested"))],
        loop_variable="nested",
        base="nested_base",
        end_variable="nested_end",
    )
    conditional = llir.IfThenElse(
        cond=_var("condition_ix]"),
        then_body=[llir.RawStmt("then(Array[ix])")],
        else_body=[llir.Assign(_access("Else", "ix"), _var("ix else"))],
        cond_list=[_var("branch_ix]")],
        then_body_list=[
            [llir.VarInit(_var("declared_ix]"), _var("Value[ix]"))],
            [llir.FunctionCallStmt("branch_ix]_[ix]_ix call", [_var("Arg[ix]")])],
        ],
    )
    body: List[llir.Stmt] = [
        llir.Assign(
            _access("Target", "ix"),
            llir.BinOp(
                "+",
                llir.ArrayAccess(_var("Array[ix]"), _var("ix offset")),
                decorated,
            ),
        ),
        llir.VarInit(_var("declared_ix]"), _var("Initial[ix]")),
        llir.FunctionCallStmt(
            "call_ix]_[ix]_ix tail",
            [_var("Argument[ix]"), llir.BinOp("+", _var("ix left"), _var("r"))],
        ),
        llir.RawStmt("raw_ix]_[ix]_ix tail"),
        nested_loop,
        conditional,
    ]
    source = _program(body, loop_variable="ix", base="root")

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assignment = cast(llir.Assign, output[0])
    assignment_target = cast(llir.ArrayAccess, assignment.var)
    assert cast(llir.Var, assignment_target.array).name == "Target"
    assert cast(llir.Var, assignment_target.index).name == "root"
    binary = cast(llir.BinOp, assignment.value)
    access = cast(llir.ArrayAccess, binary.left)
    assert cast(llir.Var, access.array).name == "Array[root]"
    assert cast(llir.Var, access.index).name == "root offset"
    rewritten_decorated = cast(llir.Var, binary.right)
    assert rewritten_decorated.name == "prefix_root]_[root]_root tail"
    assert rewritten_decorated.type is llir.DataType.PTR_FLOAT64
    assert rewritten_decorated.is_ptr is True
    assert rewritten_decorated.is_restrict is True
    assert rewritten_decorated.tensor_access == metadata

    initializer = cast(llir.VarInit, output[1])
    assert initializer.var.name == "declared_ix]"
    assert cast(llir.Var, initializer.value).name == "Initial[root]"
    call = cast(llir.FunctionCallStmt, output[2])
    assert call.name == "call_root]_[root]_root tail"
    assert cast(llir.Var, call.args[0]).name == "Argument[root]"
    call_binary = cast(llir.BinOp, call.args[1])
    assert cast(llir.Var, call_binary.left).name == "root left"
    assert cast(llir.RawStmt, output[3]).code == "raw_root]_[root]_root tail"

    nested = cast(llir.ForLoop, output[4])
    nested_assignment = cast(llir.Assign, nested.body[0])
    nested_target = cast(llir.ArrayAccess, nested_assignment.var)
    assert cast(llir.Var, nested_target.array).name == "Nested"
    assert cast(llir.Var, nested_target.index).name == "root"
    assert cast(llir.Var, nested_assignment.value).name == "root nested"
    rewritten_if = cast(llir.IfThenElse, output[5])
    assert cast(llir.Var, rewritten_if.cond).name == "condition_ix]"
    assert cast(
        llir.RawStmt, cast(List[llir.Stmt], rewritten_if.then_body)[0]
    ).code == ("then(Array[root])")
    else_assignment = cast(
        llir.Assign,
        cast(List[llir.Stmt], rewritten_if.else_body)[0],
    )
    else_target = cast(llir.ArrayAccess, else_assignment.var)
    assert cast(llir.Var, else_target.array).name == "Else"
    assert cast(llir.Var, else_target.index).name == "root"
    assert cast(llir.Var, else_assignment.value).name == "root else"
    branches = cast(List[List[llir.Stmt]], rewritten_if.then_body_list)
    branch_initializer = cast(llir.VarInit, branches[0][0])
    assert branch_initializer.var.name == "declared_ix]"
    assert cast(llir.Var, branch_initializer.value).name == "Value[root]"
    branch_call = cast(llir.FunctionCallStmt, branches[1][0])
    assert branch_call.name == "branch_root]_[root]_root call"
    assert cast(llir.Var, branch_call.args[0]).name == "Arg[root]"


def test_function_call_rewrite_preserves_tuple_statement_body() -> None:
    conditional = llir.IfThenElse(
        cond=_var("condition"),
        then_body=cast(
            List[llir.Stmt],
            (
                llir.FunctionCallStmt(
                    "call_lane]_[lane]_lane tail",
                    [_var("Argument[lane]")],
                    template_args=(llir.DataType.INT32,),
                ),
            ),
        ),
    )
    source = _program([conditional], loop_variable="lane", base="root")

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    rewritten_if = cast(llir.IfThenElse, output[0])
    assert type(rewritten_if.then_body) is tuple
    call = cast(llir.FunctionCallStmt, rewritten_if.then_body[0])
    assert call.name == "call_root]_[root]_root tail"
    assert call.template_args == (llir.DataType.INT32,)
    assert cast(llir.Var, call.args[0]).name == "Argument[root]"


def test_member_call_rewrite_owns_receiver_and_arguments() -> None:
    call = llir.MemberCallStmt(
        base=_access("Receivers", "lane"),
        member="consume",
        template_args=(llir.DataType.INT64,),
        args=(_access("Arguments", "lane"), _var("Payload[lane]")),
    )
    source = _program([call], loop_variable="lane", base="root")
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert len(output) == 1
    rewritten = cast(llir.MemberCallStmt, output[0])
    receiver = cast(llir.ArrayAccess, rewritten.base)
    assert cast(llir.Var, receiver.array).name == "Receivers"
    assert cast(llir.Var, receiver.index).name == "root"
    assert rewritten.member == "consume"
    assert rewritten.template_args == (llir.DataType.INT64,)
    assert type(rewritten.args) is tuple
    argument = cast(llir.ArrayAccess, rewritten.args[0])
    assert cast(llir.Var, argument.array).name == "Arguments"
    assert cast(llir.Var, argument.index).name == "root"
    assert cast(llir.Var, rewritten.args[1]).name == "Payload[root]"
    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_guarded_call_rewrite_owns_condition_and_arguments() -> None:
    guarded = llir.GuardedCallStmt(
        cond=llir.BinOp(
            "<",
            llir.Add(_var("position"), llir.Literal(1, llir.DataType.INT)),
            _var("position_end"),
        ),
        call=llir.FunctionCallStmt(
            "__builtin_prefetch",
            (
                llir.AddressOf(
                    operand=llir.ArrayAccess(
                        array=_var("Values"),
                        index=llir.Mul(
                            llir.ArrayAccess(
                                array=_var("Coordinates"),
                                index=_var("lane", llir.DataType.INT64),
                            ),
                            _var("stride"),
                        ),
                    )
                ),
                llir.Literal(0, llir.DataType.INT),
                llir.Literal(1, llir.DataType.INT),
            ),
        ),
    )
    source = _program([guarded], loop_variable="lane", base="root")
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert len(output) == 1
    rewritten = cast(llir.GuardedCallStmt, output[0])
    assert type(rewritten) is llir.GuardedCallStmt
    # Identifier-only guard spellings are outside the legacy compound
    # generated-string patterns; only the exact structured array-index
    # replacement fires.
    condition = cast(llir.BinOp, rewritten.cond)
    assert cast(llir.Var, cast(llir.Add, condition.left).left).name == "position"
    assert cast(llir.Var, condition.right).name == "position_end"
    borrowed = cast(llir.AddressOf, rewritten.call.args[0])
    borrowed_access = cast(llir.ArrayAccess, borrowed.operand)
    coordinate = cast(llir.ArrayAccess, cast(llir.Mul, borrowed_access.index).left)
    assert cast(llir.Var, coordinate.index).name == "root"
    assert rewritten.call.name == "__builtin_prefetch"
    assert type(rewritten.call.args) is tuple
    assert LLIRLowerer().lower_llir(rewritten) == (
        "if (position + 1 < position_end) "
        "__builtin_prefetch(&Values[Coordinates[root] * stride], 0, 1);"
    )
    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


@pytest.mark.parametrize("base", ("root", "0"))
def test_guarded_prefetch_rewrite_matches_the_legacy_whole_statement_scope(
    base: str,
) -> None:
    """Inlining rewrites both P1 condition and index, including literal zero."""

    guard = llir.GuardedCallStmt(
        cond=llir.BinOp(
            "<",
            llir.Add(_var("lane"), llir.Literal(1, llir.DataType.INT)),
            _var("p_end"),
        ),
        call=llir.FunctionCallStmt(
            "__builtin_prefetch",
            (
                llir.AddressOf(
                    llir.ArrayAccess(
                        _var("B_val"),
                        llir.Mul(
                            llir.ArrayAccess(
                                _var("A_crd"),
                                llir.Add(
                                    _var("lane"),
                                    llir.Literal(1, llir.DataType.INT),
                                ),
                            ),
                            _var("stride"),
                        ),
                    )
                ),
                llir.Literal(0, llir.DataType.INT),
                llir.Literal(1, llir.DataType.INT),
            ),
        ),
    )
    raw = llir.RawStmt(
        "if (lane + 1 < p_end) "
        "__builtin_prefetch(&B_val[A_crd[lane + 1] * stride], 0, 1)"
    )

    typed_output = eliminate_single_iteration_loops(
        _program([guard], base=base),
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )
    legacy_output = eliminate_single_iteration_loops(
        _program([raw], base=base),
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert len(typed_output) == 1
    assert len(legacy_output) == 1
    assert LLIRLowerer().lower_llir(typed_output) == LLIRLowerer().lower_llir(
        legacy_output
    )
    assert "lane" not in LLIRLowerer().lower_llir(typed_output)
    rewritten = cast(llir.GuardedCallStmt, typed_output[0])
    condition_left = cast(llir.Add, cast(llir.BinOp, rewritten.cond).left).left
    coordinate = cast(
        llir.ArrayAccess,
        cast(
            llir.Mul,
            cast(
                llir.ArrayAccess, cast(llir.AddressOf, rewritten.call.args[0]).operand
            ).index,
        ).left,
    )
    coordinate_left = cast(llir.Add, coordinate.index).left
    if base == "0":
        assert type(condition_left) is llir.Literal
        assert type(coordinate_left) is llir.Literal
        assert cast(llir.Literal, condition_left).value == 0
        assert cast(llir.Literal, coordinate_left).value == 0
    else:
        assert cast(llir.Var, condition_left).name == base
        assert cast(llir.Var, coordinate_left).name == base


def test_function_call_rewrite_traverses_product_and_sizeof_arguments() -> None:
    call = llir.FunctionCallStmt(
        name="memset",
        args=(
            _access("Receivers", "lane"),
            llir.Literal(value=0, data_type=llir.DataType.INT),
            llir.Mul(
                _var("Payload[lane]"),
                llir.Sizeof(data_type=llir.DataType.INT64),
            ),
        ),
    )
    source = _program([call], loop_variable="lane", base="root")
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert len(output) == 1
    rewritten = cast(llir.FunctionCallStmt, output[0])
    assert rewritten.name == "memset"
    assert type(rewritten.args) is tuple
    receiver = cast(llir.ArrayAccess, rewritten.args[0])
    assert cast(llir.Var, receiver.array).name == "Receivers"
    assert cast(llir.Var, receiver.index).name == "root"
    zero = cast(llir.Literal, rewritten.args[1])
    assert zero.value == 0
    assert zero.data_type is llir.DataType.INT
    product = cast(llir.Mul, rewritten.args[2])
    assert cast(llir.Var, product.left).name == "Payload[root]"
    assert type(product.right) is llir.Sizeof
    assert cast(llir.Sizeof, product.right).data_type is llir.DataType.INT64
    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_function_call_rewrite_traverses_addressed_copy_arguments() -> None:
    call = llir.FunctionCallStmt(
        name="memcpy",
        args=(
            llir.AddressOf(operand=_access("Receivers", "lane")),
            _var("workspace", llir.DataType.PTR_FLOAT32),
            llir.Mul(
                _var("Payload[lane]"),
                llir.Sizeof(data_type=llir.DataType.INT64),
            ),
        ),
    )
    source = _program([call], loop_variable="lane", base="root")
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert len(output) == 1
    rewritten = cast(llir.FunctionCallStmt, output[0])
    assert rewritten.name == "memcpy"
    assert type(rewritten.args) is tuple
    destination = cast(llir.AddressOf, rewritten.args[0])
    assert type(destination) is llir.AddressOf
    row_slot = cast(llir.ArrayAccess, destination.operand)
    assert cast(llir.Var, row_slot.array).name == "Receivers"
    assert cast(llir.Var, row_slot.index).name == "root"
    workspace = cast(llir.Var, rewritten.args[1])
    assert workspace.name == "workspace"
    byte_count = cast(llir.Mul, rewritten.args[2])
    assert cast(llir.Var, byte_count.left).name == "Payload[root]"
    assert type(byte_count.right) is llir.Sizeof
    assert cast(llir.Sizeof, byte_count.right).data_type is llir.DataType.INT64
    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_exact_structured_access_index_is_rewritten_and_reapplication_is_noop() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(31),
        tensor_id=SymbolId(32),
        index_ids=(IndexId(33), IndexId(34)),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    access = llir.ArrayAccess(
        array=_var("Mask_val", llir.DataType.PTR_FLOAT32),
        index=_var("pMask1", llir.DataType.INT64),
        tensor_access=metadata,
    )
    source = _program(
        [llir.Assign(_var("out"), access)],
        loop_variable="pMask1",
        base="pMask0",
    )
    before = _snapshot(source)

    first = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )
    second = eliminate_single_iteration_loops(
        first,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(source) == before
    assert _snapshot(second) == _snapshot(first)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    rewritten = cast(llir.ArrayAccess, cast(llir.Assign, first[0]).value)
    assert cast(llir.Var, rewritten.array).name == "Mask_val"
    assert cast(llir.Var, rewritten.index).name == "pMask0"
    assert rewritten.tensor_access is metadata
    assert rewritten is not access


def test_rewrite_omits_headers_parallel_regions_and_nested_containers() -> None:
    old = "Array[ix]"
    nested_header = _loop([], loop_variable="nested")
    nested_header.init = llir.VarInit(_var("nested"), _var(old))
    nested_header.cond = llir.BinOp("<", _var(old), _var("limit"))
    nested_header.update = llir.Assign(_access("Array", "ix"), llir.Literal(1))
    nested_header.body = [llir.RawStmt(old)]
    nested_header.before_parallel_body = [llir.RawStmt(old)]
    nested_header.pre_parallel_body = [llir.RawStmt(old)]
    nested_header.post_parallel_body = [llir.RawStmt(old)]
    while_loop = llir.WhileLoop(_var(old), [llir.RawStmt(old)])
    auto_loop = llir.ForLoopAuto(_var(old), _var(old), [llir.RawStmt(old)])
    function = llir.Function(
        llir.DataType.VOID,
        "function_ix]",
        [_var(old)],
        [llir.RawStmt(old)],
    )
    conditional = llir.IfThenElse(
        cond=_var(old),
        then_body=[llir.RawStmt(old)],
        cond_list=[_var(old)],
        then_body_list=[[llir.RawStmt(old)]],
    )
    raw_nested = [llir.RawStmt(old)]
    body = cast(
        List[llir.Stmt],
        [
            nested_header,
            while_loop,
            auto_loop,
            function,
            conditional,
            llir.Assign(_var("cast"), llir.Cast(_var(old), llir.DataType.FLOAT32)),
            llir.Assign(_var("unary"), llir.UnaryOp("-", _var(old))),
            llir.Assign(_var("call"), llir.FunctionCall("identity", [_var(old)])),
            llir.Assign(_var("array"), llir.Array([_var(old)], llir.DataType.INT64)),
            llir.Return(_var(old)),
            raw_nested,
        ],
    )
    outer = _loop(body, loop_variable="ix")
    outer.before_parallel_body = [llir.RawStmt(old)]
    outer.pre_parallel_body = [llir.RawStmt(old)]
    outer.post_parallel_body = [llir.RawStmt(old)]
    source: List[llir.Stmt] = [_bound(), outer]

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    header = cast(llir.ForLoop, output[0])
    assert cast(llir.Var, cast(llir.VarInit, header.init).value).name == old
    assert cast(llir.Var, cast(llir.BinOp, header.cond).left).name == old
    header_target = cast(llir.ArrayAccess, cast(llir.Assign, header.update).var)
    assert cast(llir.Var, header_target.array).name == "Array"
    assert cast(llir.Var, header_target.index).name == "ix"
    assert cast(llir.RawStmt, header.body[0]).code == "Array[base]"
    assert cast(llir.RawStmt, header.before_parallel_body[0]).code == old
    assert cast(llir.RawStmt, header.pre_parallel_body[0]).code == old
    assert cast(llir.RawStmt, header.post_parallel_body[0]).code == old

    output_while = cast(llir.WhileLoop, output[1])
    assert cast(llir.Var, output_while.cond).name == old
    assert cast(llir.RawStmt, output_while.body[0]).code == old
    output_auto = cast(llir.ForLoopAuto, output[2])
    assert output_auto.var.name == old
    assert cast(llir.Var, output_auto.array).name == old
    assert cast(llir.RawStmt, output_auto.body[0]).code == old
    output_function = cast(llir.Function, output[3])
    assert output_function.name == "function_ix]"
    assert cast(llir.Var, output_function.args[0]).name == old
    assert cast(llir.RawStmt, output_function.body[0]).code == old
    output_if = cast(llir.IfThenElse, output[4])
    assert cast(llir.Var, output_if.cond).name == old
    assert cast(llir.Var, output_if.cond_list[0]).name == old
    assert cast(llir.RawStmt, cast(List[llir.Stmt], output_if.then_body)[0]).code == (
        "Array[base]"
    )
    assert cast(llir.RawStmt, output_if.then_body_list[0][0]).code == "Array[base]"

    cast_expression = cast(llir.Cast, cast(llir.Assign, output[5]).value)
    assert cast(llir.Var, cast_expression.expr).name == old
    unary = cast(llir.UnaryOp, cast(llir.Assign, output[6]).value)
    assert cast(llir.Var, unary.operand).name == old
    call = cast(llir.FunctionCall, cast(llir.Assign, output[7]).value)
    assert cast(llir.Var, call.args[0]).name == old
    array = cast(llir.Array, cast(llir.Assign, output[8]).value)
    assert cast(llir.Var, array.values[0]).name == old
    assert cast(llir.Var, cast(llir.Return, output[9]).value).name == old
    assert type(output[10]) is list
    assert cast(llir.RawStmt, cast(List[llir.Stmt], output[10])[0]).code == old


def test_nested_for_loop_and_if_then_else_analysis_is_postorder() -> None:
    inner_program = _program(
        [llir.RawStmt("inner(Array[inner_lane])")],
        loop_variable="inner_lane",
        base="inner_base",
        end_variable="inner_end",
    )
    outer_source = _program(
        inner_program,
        loop_variable="outer_lane",
        base="outer_base",
        end_variable="outer_end",
    )

    nested_output = eliminate_single_iteration_loops(
        outer_source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _raw_codes(nested_output) == ["inner(Array[inner_base])"]

    conditional = llir.IfThenElse(
        cond=_var("condition"),
        then_body=_program(
            [llir.RawStmt("then(Array[then_lane])")],
            loop_variable="then_lane",
            base="then_base",
            end_variable="then_end",
        ),
        else_body=_program(
            [llir.RawStmt("else(Array[else_lane])")],
            loop_variable="else_lane",
            base="else_base",
            end_variable="else_end",
        ),
    )
    if_output = eliminate_single_iteration_loops(
        [conditional],
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )
    output_if = cast(llir.IfThenElse, if_output[0])
    assert _raw_codes(cast(List[llir.Stmt], output_if.then_body)) == [
        "then(Array[then_base])"
    ]
    assert _raw_codes(cast(List[llir.Stmt], output_if.else_body)) == [
        "else(Array[else_base])"
    ]


def _parallel_analysis_container() -> llir.ForLoop:
    container = _loop([], end_variable="unmatched")
    container.before_parallel_body = _program([llir.RawStmt("before_body")])
    container.pre_parallel_body = _program([llir.RawStmt("pre_body")])
    container.post_parallel_body = _program([llir.RawStmt("post_body")])
    return container


@pytest.mark.parametrize(
    "factory",
    (
        lambda: llir.IfThenElse(
            cond_list=[_var("first"), _var("second")],
            then_body_list=[
                _program([llir.RawStmt("first_branch_body")]),
                _program([llir.RawStmt("second_branch_body")]),
            ],
        ),
        _parallel_analysis_container,
        lambda: llir.WhileLoop(_var("condition"), _program()),
        lambda: llir.ForLoopAuto(_var("item"), _var("items"), _program()),
        lambda: llir.Function(llir.DataType.VOID, "function", [], _program()),
    ),
)
def test_analysis_omits_every_surviving_container(
    factory: Callable[[], llir.Stmt],
) -> None:
    source = [factory()]
    before = _snapshot(source)

    output = eliminate_single_iteration_loops(
        source,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_second_application_is_structurally_idempotent_but_fully_detached() -> None:
    first = eliminate_single_iteration_loops(
        _program([llir.RawStmt("read(Array[lane])")]),
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )
    second = eliminate_single_iteration_loops(
        first,
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    )

    assert _snapshot(second) == _snapshot(first)
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))


class _UnknownStatement(llir.Stmt):
    pass


class _UnknownExpression(llir.Expr):
    pass


@pytest.mark.parametrize(
    ("source", "expected_path"),
    (
        ([cast(llir.Stmt, _UnknownStatement())], ("root", "[0]")),
        (
            [
                llir.WhileLoop(
                    _var("condition"),
                    [cast(llir.Stmt, _UnknownStatement())],
                )
            ],
            ("root", "[0]", "body", "[0]"),
        ),
        (
            [llir.Return(cast(llir.Expr, _UnknownExpression()))],
            ("root", "[0]", "value"),
        ),
    ),
)
def test_unknown_nodes_fail_closed_inside_semantically_omitted_containers(
    source: List[llir.Stmt],
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            source,
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.path == expected_path
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "eliminate_single_iteration_loops"
    assert diagnostic.node_type in {"_UnknownStatement", "_UnknownExpression"}


@pytest.mark.parametrize(
    ("source", "expected_path"),
    (
        (
            [
                llir.WhileLoop(
                    _var("condition"),
                    cast(List[llir.Stmt], [object()]),
                )
            ],
            ("root", "[0]", "body", "[0]"),
        ),
        (
            [
                llir.Function(
                    llir.DataType.VOID,
                    "function",
                    [],
                    cast(List[llir.Stmt], [object()]),
                )
            ],
            ("root", "[0]", "body", "[0]"),
        ),
        (
            [
                llir.IfThenElse(
                    cond=_var("condition"),
                    then_body_list=[cast(List[llir.Stmt], [object()])],
                )
            ],
            ("root", "[0]", "then_body_list", "[0]", "[0]"),
        ),
    ),
)
def test_malformed_children_fail_inside_semantically_omitted_containers(
    source: List[llir.Stmt],
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            source,
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_statement_sequence_member"
    assert diagnostic.path == expected_path
    assert diagnostic.node_type == "object"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "eliminate_single_iteration_loops"


def test_invalid_top_level_member_uses_pass_owned_root_diagnostic() -> None:
    source = cast(List[llir.Stmt], [llir.BlankLine(), object()])

    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            source,
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_single_iteration_loop_elimination_root_member"
    assert diagnostic.path == ("root", "[1]")
    assert diagnostic.node_type == "object"


@pytest.mark.parametrize(
    ("context", "expected_code", "expected_path"),
    (
        (
            object(),
            "invalid_single_iteration_loop_elimination_context",
            ("context",),
        ),
        (
            SingleIterationLoopEliminationContext(cast(LLIRTraversalContext, object())),
            "invalid_single_iteration_loop_elimination_traversal_context",
            ("context", "traversal"),
        ),
        (
            SingleIterationLoopEliminationContext(
                LLIRTraversalContext("", "eliminate_single_iteration_loops")
            ),
            "invalid_single_iteration_loop_elimination_traversal_context",
            ("context", "traversal"),
        ),
    ),
)
def test_invalid_direct_context_fails_with_pass_diagnostic(
    context: object,
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            [llir.BlankLine()],
            cast(SingleIterationLoopEliminationContext, context),
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == expected_code
    assert diagnostic.path == expected_path
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "eliminate_single_iteration_loops"


def _malformed_bound_name() -> List[llir.Stmt]:
    source = _program()
    cast(llir.Var, cast(llir.VarInit, source[0]).value).name = cast(str, 7)
    return source


def _malformed_condition_operator() -> List[llir.Stmt]:
    source = _program()
    object.__setattr__(
        cast(llir.BinOp, cast(llir.ForLoop, source[1]).cond),
        "op",
        cast(str, 7),
    )
    return source


def _malformed_function_name() -> List[llir.Stmt]:
    call = llir.FunctionCallStmt("touch", [_var("Array[lane]")])
    object.__setattr__(call, "name", 7)
    return _program([call])


def _malformed_raw_code() -> List[llir.Stmt]:
    return _program([llir.RawStmt(cast(str, 7))])


def _malformed_rewritten_var_name() -> List[llir.Stmt]:
    return _program([llir.Assign(_var("output"), _var(cast(str, 7)))])


@pytest.mark.parametrize(
    ("factory", "expected_code", "expected_path"),
    (
        (
            _malformed_bound_name,
            "invalid_single_iteration_loop_var_name",
            ("root", "[0]", "value", "name"),
        ),
        (
            _malformed_function_name,
            "invalid_function_call_stmt_name",
            ("root", "[1]", "body", "[0]", "name"),
        ),
        (
            _malformed_raw_code,
            "invalid_single_iteration_loop_raw_statement",
            ("root", "[1]", "body", "[0]", "code"),
        ),
        (
            _malformed_rewritten_var_name,
            "invalid_single_iteration_loop_var_name",
            ("root", "[1]", "body", "[0]", "value", "name"),
        ),
    ),
)
def test_malformed_consumed_scalars_use_pass_owned_diagnostics(
    factory: Callable[[], List[llir.Stmt]],
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            factory(),
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == expected_code
    assert diagnostic.path == expected_path
    assert diagnostic.node_type == "int"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "eliminate_single_iteration_loops"


def test_forged_condition_operator_uses_common_binary_validation() -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        eliminate_single_iteration_loops(
            _malformed_condition_operator(),
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_binary_operator"
    assert diagnostic.path == ("root", "[1]", "cond", "op")
    assert diagnostic.node_type == "int"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "eliminate_single_iteration_loops"


def test_production_skips_manager_walks_and_debug_verifies_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walks: List[Tuple[str, str]] = []

    class RecordingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            walks.append((self.context.stage, self.context.pass_name))
            super().walk(value)

    monkeypatch.setattr(pass_manager_module, "LLIRWalker", RecordingWalker)
    artifact = LLIRStatementListArtifact(_program())
    spec = SingleIterationLoopEliminationPassSpec()

    production = LLIRPassManager(PRODUCTION_LLIR_PASS_OPTIONS)
    production_result = production.run_single_iteration_loop_elimination(
        artifact,
        spec,
    )
    assert walks == []
    assert production_result.run_records[0].verified_before is False
    assert production_result.run_records[0].verified_after is False
    assert _mutable_ir_ids(artifact.statements).isdisjoint(
        _mutable_ir_ids(production_result.artifact.statements)
    )

    debug = LLIRPassManager(DEBUG_LLIR_PASS_OPTIONS)
    debug_result = debug.run_single_iteration_loop_elimination(artifact, spec)
    assert walks == [
        ("LLIR transformation", "eliminate_single_iteration_loops"),
        ("LLIR transformation", "eliminate_single_iteration_loops"),
    ]
    assert debug_result.run_records[0].verified_before is True
    assert debug_result.run_records[0].verified_after is True
    assert _snapshot(debug_result.artifact.statements) == _snapshot(
        production_result.artifact.statements
    )


def test_failure_adds_no_run_record_and_stops_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_single_iteration_failure",
            message="single-iteration pass failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name="eliminate_single_iteration_loops",
        )
    )

    def fail_once(
        statements: List[llir.Stmt],
        context: SingleIterationLoopEliminationContext,
    ) -> NoReturn:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(
        pass_manager_module,
        "eliminate_single_iteration_loops",
        fail_once,
    )
    manager = LLIRPassManager()
    with pytest.raises(LLIRTraversalError) as raised:
        manager.run_single_iteration_loop_elimination(
            LLIRStatementListArtifact([llir.BlankLine()]),
            SingleIterationLoopEliminationPassSpec(),
        )

    assert raised.value is failure
    assert calls == 1


def test_timing_and_run_records_are_nonsemantic_and_optional() -> None:
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    spec = SingleIterationLoopEliminationPassSpec()
    timed = LLIRPassManager().run_single_iteration_loop_elimination(artifact, spec)
    untimed = LLIRPassManager(
        LLIRPassOptions(record_timing=False)
    ).run_single_iteration_loop_elimination(artifact, spec)

    timed_record = timed.run_records[0]
    untimed_record = untimed.run_records[0]
    assert timed_record.duration_ns is not None
    assert cast(int, timed_record.duration_ns) >= 0
    assert untimed_record.duration_ns is None
    assert timed_record == replace(timed_record, duration_ns=10**9)
    shared_artifact = LLIRStatementListArtifact([llir.BlankLine()])
    assert LLIRStatementListPassResult(
        shared_artifact,
        (timed_record,),
    ) == LLIRStatementListPassResult(
        shared_artifact,
        (replace(timed_record, duration_ns=10**9),),
    )


def test_empty_and_single_pass_incremental_plumbing_p95_is_below_one_ms() -> None:
    sample_count = 2000
    source: List[llir.Stmt] = [llir.BlankLine()]
    context = SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT
    spec = SingleIterationLoopEliminationPassSpec(context)
    manager = LLIRPassManager(LLIRPassOptions(record_timing=False))

    for _ in range(100):
        manager.run_empty(LLIRRewriteArtifact(source))
        eliminate_single_iteration_loops(source, context)
        manager.run_single_iteration_loop_elimination(
            LLIRStatementListArtifact(source),
            spec,
        )

    empty_ns: List[int] = []
    incremental_ns: List[int] = []
    for sample in range(sample_count):
        empty_started = perf_counter_ns()
        manager.run_empty(LLIRRewriteArtifact(source))
        empty_ns.append(perf_counter_ns() - empty_started)

        if sample % 2:
            managed_started = perf_counter_ns()
            manager.run_single_iteration_loop_elimination(
                LLIRStatementListArtifact(source),
                spec,
            )
            managed_elapsed = perf_counter_ns() - managed_started
            direct_started = perf_counter_ns()
            eliminate_single_iteration_loops(source, context)
            direct_elapsed = perf_counter_ns() - direct_started
        else:
            direct_started = perf_counter_ns()
            eliminate_single_iteration_loops(source, context)
            direct_elapsed = perf_counter_ns() - direct_started
            managed_started = perf_counter_ns()
            manager.run_single_iteration_loop_elimination(
                LLIRStatementListArtifact(source),
                spec,
            )
            managed_elapsed = perf_counter_ns() - managed_started
        incremental_ns.append(managed_elapsed - direct_elapsed)

    assert _p95(empty_ns) <= 1_000_000
    assert _p95(incremental_ns) <= 1_000_000
