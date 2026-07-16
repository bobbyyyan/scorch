from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
from time import perf_counter_ns
from typing import Callable, List, NoReturn, Sequence, Set, Tuple, cast

import pytest

from scorch.compiler import llir
import scorch.compiler.llir_pass_manager as pass_manager_module
import scorch.compiler.schedule_lowerer as schedule_lowerer_module
import scorch.compiler.sparse_prefetch_pass as sparse_prefetch_module
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer, ResultTensorAssembler
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.compiler.llir_pass_manager import (
    DEBUG_LLIR_PASS_OPTIONS,
    LOOP_INVARIANT_FACTOR_HOIST_PASS,
    PRODUCTION_LLIR_PASS_OPTIONS,
    SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
    SPARSE_PREFETCH_PASS,
    DensePointerHoistPassSpec,
    DynamicVectorAccessPassSpec,
    LLIRPassArtifactType,
    LLIRPassContextType,
    LLIRPassDescriptor,
    LLIRPassManager,
    LLIRPassManagerError,
    LLIRPassOptions,
    LLIRRewriteArtifact,
    LLIRRewritePassResult,
    LLIRStatementListArtifact,
    LLIRStatementListPassResult,
    LoopInvariantFactorHoistPassSpec,
    ResultWritePassSpec,
    SingleIterationLoopEliminationPassSpec,
    SparsePrefetchPassSpec,
)
from scorch.compiler.llir_traversal import (
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from scorch.compiler.scheduler import Schedule, Scheduler, regblock_force
from scorch.compiler.sparse_prefetch_pass import (
    SPARSE_PREFETCH_CONTEXT,
    SPARSE_PREFETCH_TRAVERSAL_CONTEXT,
    SparsePrefetchContext,
    insert_sparse_prefetch,
)


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _access(array: str, index: str) -> llir.ArrayAccess:
    return llir.ArrayAccess(
        array=_var(array),
        index=_var(index, llir.DataType.INT64),
    )


def _position_init(
    position: str,
    base: str,
    stride: str,
    offset: str = "j",
) -> llir.VarInit:
    return llir.VarInit(
        _var(position),
        llir.Add(
            llir.Mul(_var(base), _var(stride)),
            _var(offset),
        ),
    )


def _mode_position_begin(
    array: str = "A1_pos",
    index: str = "pA0",
) -> llir.ArrayAccess:
    return llir.ArrayAccess(
        _var(array, llir.DataType.PTR_INT),
        _var(index, llir.DataType.INT),
    )


def _inner_loop(body: List[llir.Stmt]) -> llir.ForLoop:
    return llir.ForLoop(
        init=llir.VarInit(_var("j", llir.DataType.INT), llir.Literal(0)),
        cond=llir.BinOp("<", _var("j"), _var("B1_size")),
        update=llir.Increment(_var("j")),
        body=body,
    )


def _string_dense_loop(
    *,
    array: str = "B_val",
    position: str = "pB1",
    base: str = "pB0",
    stride: str = "B1_size",
) -> llir.ForLoop:
    return _inner_loop(
        [
            _position_init(position, base, stride),
            llir.Assign(_access("C_val", "pC1"), _var(f"{array}[{position}]")),
        ]
    )


def _sparse_loop(
    *,
    iterator: str = "pA1",
    position_begin: llir.Expr | None = None,
    end: str = "pA1_end",
    coordinate_array: str = "A1_crd",
    tail: List[llir.Stmt] | None = None,
) -> llir.ForLoop:
    if tail is None:
        tail = [_string_dense_loop()]
    if position_begin is None:
        position_begin = _mode_position_begin()
    return llir.ForLoop(
        init=llir.VarInit(_var(iterator), position_begin),
        cond=llir.BinOp("<", _var(iterator), _var(end)),
        update=llir.Increment(_var(iterator)),
        body=[
            llir.VarInit(
                _var("coordinate"),
                llir.ArrayAccess(
                    _var(coordinate_array, llir.DataType.PTR_INT),
                    _var(iterator, llir.DataType.INT),
                ),
            ),
            *tail,
        ],
    )


def _prefetches(statements: List[llir.Stmt]) -> List[llir.RawStmt]:
    return [
        cast(llir.RawStmt, statement)
        for statement in statements
        if type(statement) is llir.RawStmt
        and "__builtin_prefetch" in cast(llir.RawStmt, statement).code
    ]


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


def _expected_prefetch(
    array: str = "B_val",
    stride: str = "B1_size",
    *,
    iterator: str = "pA1",
    end: str = "pA1_end",
    coordinate_array: str = "A1_crd",
) -> str:
    return (
        f"if ({iterator} + 1 < {end}) "
        f"__builtin_prefetch(&{array}[{coordinate_array}[{iterator} + 1] * "
        f"{stride}], 0, 1)"
    )


def test_sparse_prefetch_context_descriptor_artifact_result_and_records_are_frozen() -> (
    None
):
    context = SparsePrefetchContext()
    spec = SparsePrefetchPassSpec(context)
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    result = LLIRPassManager().run_sparse_prefetch(artifact, spec)
    record = result.run_records[0]

    frozen_updates: Tuple[Tuple[object, str, object], ...] = (
        (context, "traversal", LLIRTraversalContext("other", "other")),
        (SPARSE_PREFETCH_PASS, "version", 2),
        (spec, "descriptor", replace(SPARSE_PREFETCH_PASS, version=2)),
        (artifact, "statements", []),
        (result, "run_records", ()),
        (record, "duration_ns", 0),
    )
    for value, field, replacement in frozen_updates:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)


def test_sparse_prefetch_has_stable_v1_statement_list_boundary() -> None:
    assert SPARSE_PREFETCH_CONTEXT == SparsePrefetchContext(
        traversal=SPARSE_PREFETCH_TRAVERSAL_CONTEXT
    )
    assert SPARSE_PREFETCH_PASS == LLIRPassDescriptor(
        name="insert_sparse_prefetch",
        version=1,
        input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        output_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        context_type=LLIRPassContextType.SPARSE_PREFETCH,
    )


def test_success_and_legal_noop_are_fully_detached_without_input_mutation() -> None:
    source = [_sparse_loop()]
    source_snapshot = _structural_snapshot(source)

    output = insert_sparse_prefetch(source, SPARSE_PREFETCH_CONTEXT)

    assert _structural_snapshot(source) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))
    output_loop = cast(llir.ForLoop, output[0])
    assert [statement.code for statement in _prefetches(output_loop.body)] == [
        _expected_prefetch()
    ]

    noop_source: List[llir.Stmt] = [llir.BlankLine()]
    noop_snapshot = _structural_snapshot(noop_source)
    noop = insert_sparse_prefetch(noop_source, SPARSE_PREFETCH_CONTEXT)
    assert _structural_snapshot(noop) == noop_snapshot
    assert _mutable_ir_ids(noop_source).isdisjoint(_mutable_ir_ids(noop))


def test_nested_sparse_loops_are_processed_post_order_through_direct_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    insertion_order: List[str] = []
    original_prepend = sparse_prefetch_module._prepend_prefetches

    def record_prepend(
        loop: llir.ForLoop,
        iterator: str,
        end: str,
        coordinate_array: str,
        dense_arrays: Sequence[Tuple[str, str]],
    ) -> None:
        insertion_order.append(iterator)
        original_prepend(loop, iterator, end, coordinate_array, dense_arrays)

    monkeypatch.setattr(sparse_prefetch_module, "_prepend_prefetches", record_prepend)
    nested = _sparse_loop(
        iterator="pX1",
        position_begin=_mode_position_begin("X1_pos", "pX0"),
        end="pX1_end",
        coordinate_array="X1_crd",
        tail=[
            _string_dense_loop(
                array="Y_val",
                position="pY1",
                base="pY0",
                stride="Y1_size",
            )
        ],
    )
    outer = _sparse_loop(tail=[_string_dense_loop(), nested])

    output = insert_sparse_prefetch([outer], SPARSE_PREFETCH_CONTEXT)

    assert insertion_order == ["pX1", "pA1"]
    output_outer = cast(llir.ForLoop, output[0])
    assert cast(llir.RawStmt, output_outer.body[0]).code == _expected_prefetch()
    output_nested = next(
        cast(llir.ForLoop, statement)
        for statement in output_outer.body
        if type(statement) is llir.ForLoop
        and type(cast(llir.ForLoop, statement).init) is llir.VarInit
        and cast(llir.VarInit, cast(llir.ForLoop, statement).init).var.name == "pX1"
    )
    assert cast(llir.RawStmt, output_nested.body[0]).code == _expected_prefetch(
        "Y_val",
        "Y1_size",
        iterator="pX1",
        end="pX1_end",
        coordinate_array="X1_crd",
    )


def test_string_and_typed_array_accesses_collect_all_pairs_in_first_seen_order() -> (
    None
):
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(1),
        tensor_id=SymbolId(2),
        index_ids=(IndexId(3), IndexId(4)),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    inner = _inner_loop(
        [
            _position_init("pB1", "pB0", "B1_size"),
            _position_init("pD1", "pD0", "D1_size"),
            llir.Assign(
                _access("C_val", "pC1"),
                llir.BinOp(
                    "+",
                    _var("B_val[pB1]"),
                    llir.ArrayAccess(
                        _var("D_val"),
                        _var("pD1"),
                        tensor_access=metadata,
                    ),
                ),
            ),
            llir.Assign(_var("duplicate"), _var("B_val[pB1]")),
        ]
    )

    output = insert_sparse_prefetch(
        [_sparse_loop(tail=[inner])], SPARSE_PREFETCH_CONTEXT
    )

    loop = cast(llir.ForLoop, output[0])
    prefetches = _prefetches(loop.body)
    assert [(statement.code, statement.add_semicolon) for statement in prefetches] == [
        (_expected_prefetch("B_val", "B1_size"), True),
        (_expected_prefetch("D_val", "D1_size"), True),
    ]
    assert loop.body[:2] == prefetches
    assert LLIRLowerer().lower_llir(prefetches) == (
        _expected_prefetch("B_val", "B1_size")
        + ";\n"
        + _expected_prefetch("D_val", "D1_size")
        + ";"
    )
    rewritten_inner = next(
        statement for statement in loop.body if type(statement) is llir.ForLoop
    )
    assignment = cast(llir.Assign, rewritten_inner.body[2])
    binary = cast(llir.BinOp, assignment.value)
    assert cast(llir.ArrayAccess, binary.right).tensor_access is metadata


def test_raw_hoisted_pointer_augments_assignment_discovery_in_body_order() -> None:
    raw_pointer = llir.RawStmt(
        "const double* __restrict__ _E_val_ptr = &E_val[pE0 * E1_size]",
        add_semicolon=False,
    )
    raw_decoys = [
        llir.RawStmt("use_E_val_ptr_without_a_declaration"),
        llir.RawStmt("const int* __restrict__ _F_val_ptr = &F_val[pF0 * F1_size]"),
    ]
    source = [_sparse_loop(tail=[raw_pointer, *raw_decoys, _string_dense_loop()])]

    output = insert_sparse_prefetch(source, SPARSE_PREFETCH_CONTEXT)

    loop = cast(llir.ForLoop, output[0])
    assert [statement.code for statement in _prefetches(loop.body)] == [
        _expected_prefetch("B_val", "B1_size"),
        _expected_prefetch("E_val", "E1_size"),
    ]


def test_sparse_gate_is_structural_and_uses_the_condition_right_hand_var() -> None:
    loop = _sparse_loop()
    loop.cond = llir.Add(llir.Literal(7), _var("named_rhs_end"))

    output = insert_sparse_prefetch([loop], SPARSE_PREFETCH_CONTEXT)

    rewritten = cast(llir.ForLoop, output[0])
    assert [statement.code for statement in _prefetches(rewritten.body)] == [
        _expected_prefetch(end="named_rhs_end")
    ]

    legacy = _sparse_loop(
        position_begin=_var("prefix_A1_pos[pA0] + suffix", llir.DataType.INT)
    )
    legacy_output = insert_sparse_prefetch([legacy], SPARSE_PREFETCH_CONTEXT)
    assert _prefetches(cast(llir.ForLoop, legacy_output[0]).body) == []
    assert _mutable_ir_ids([legacy]).isdisjoint(_mutable_ir_ids(legacy_output))


def _legal_noop_sources() -> List[Tuple[str, List[llir.Stmt]]]:
    missing_init = _sparse_loop()
    missing_init.init = None

    non_var_init_value = _sparse_loop()
    cast(llir.VarInit, non_var_init_value.init).value = llir.Literal(0)

    non_sparse = _sparse_loop(position_begin=_var("A1_begin"))

    legacy_flat_position = _sparse_loop(
        position_begin=_var("A1_pos[pA0]", llir.DataType.INT)
    )

    wrong_position_array_type = _sparse_loop(
        position_begin=llir.ArrayAccess(
            _var("A1_pos", llir.DataType.STD_VECTOR_C_INT),
            _var("pA0", llir.DataType.INT),
        )
    )

    wrong_position_index_type = _sparse_loop(
        position_begin=llir.ArrayAccess(
            _var("A1_pos", llir.DataType.PTR_INT),
            _var("pA0", llir.DataType.INT64),
        )
    )

    wrong_condition = _sparse_loop()
    wrong_condition.cond = llir.Literal(True)

    unnamed_end = _sparse_loop()
    unnamed_end.cond = llir.BinOp("<", _var("pA1"), llir.Literal(4))

    missing_coordinate = _sparse_loop()
    missing_coordinate.body = missing_coordinate.body[1:]

    legacy_flat_coordinate = _sparse_loop()
    cast(llir.VarInit, legacy_flat_coordinate.body[0]).value = _var(
        "A1_crd[pA1]", llir.DataType.INT
    )

    malformed_structured_coordinate = _sparse_loop()
    cast(llir.VarInit, malformed_structured_coordinate.body[0]).value = (
        llir.ArrayAccess(
            llir.Literal(0),
            _var("pA1", llir.DataType.INT),
        )
    )

    wrong_index_structured_coordinate = _sparse_loop()
    cast(llir.VarInit, wrong_index_structured_coordinate.body[0]).value = (
        llir.ArrayAccess(
            _var("A1_crd", llir.DataType.PTR_INT),
            _var("pOther", llir.DataType.INT),
        )
    )

    no_inner_loop = _sparse_loop(tail=[llir.BlankLine()])

    no_position = _sparse_loop(
        tail=[_inner_loop([llir.Assign(_var("out"), _var("B_val[pB1]"))])]
    )

    wrong_position_shape = _sparse_loop(
        tail=[
            _inner_loop(
                [
                    llir.VarInit(
                        _var("pB1"),
                        llir.BinOp("-", _var("pB0"), _var("B1_size")),
                    ),
                    llir.Assign(_var("out"), _var("B_val[pB1]")),
                ]
            )
        ]
    )

    non_binary_position_left = _sparse_loop(
        tail=[
            _inner_loop(
                [
                    llir.VarInit(
                        _var("pB1"),
                        llir.Add(llir.Literal(0), _var("j")),
                    ),
                    llir.Assign(_var("out"), _var("B_val[pB1]")),
                ]
            )
        ]
    )

    non_multiply_position = _sparse_loop(
        tail=[
            _inner_loop(
                [
                    llir.VarInit(
                        _var("pB1"),
                        llir.Add(
                            llir.BinOp("+", _var("pB0"), _var("B1_size")),
                            _var("j"),
                        ),
                    ),
                    llir.Assign(_var("out"), _var("B_val[pB1]")),
                ]
            )
        ]
    )

    non_var_stride = _sparse_loop(
        tail=[
            _inner_loop(
                [
                    llir.VarInit(
                        _var("pB1"),
                        llir.Add(
                            llir.Mul(_var("pB0"), llir.Literal(4)),
                            _var("j"),
                        ),
                    ),
                    llir.Assign(_var("out"), _var("B_val[pB1]")),
                ]
            )
        ]
    )

    no_assign = _sparse_loop(
        tail=[
            llir.RawStmt(
                "const float* __restrict__ _E_val_ptr = &E_val[pE0 * E1_size]"
            ),
            _inner_loop([_position_init("pB1", "pB0", "B1_size")]),
        ]
    )

    target_only_access = _sparse_loop(
        tail=[
            _inner_loop(
                [
                    _position_init("pB1", "pB0", "B1_size"),
                    llir.Assign(_access("B_val", "pB1"), _var("value")),
                ]
            )
        ]
    )

    cast_hidden_access = _sparse_loop(
        tail=[
            _inner_loop(
                [
                    _position_init("pB1", "pB0", "B1_size"),
                    llir.Assign(
                        _var("out"),
                        llir.Cast(_var("B_val[pB1]"), llir.DataType.FLOAT32),
                    ),
                ]
            )
        ]
    )

    unmatched_typed_accesses = _sparse_loop(
        tail=[
            _inner_loop(
                [
                    _position_init("pB1", "pB0", "B1_size"),
                    llir.Assign(
                        _var("out0"),
                        llir.ArrayAccess(llir.Literal(0), _var("pB1")),
                    ),
                    llir.Assign(
                        _var("out1"),
                        llir.ArrayAccess(_var("B_data"), _var("pB1")),
                    ),
                    llir.Assign(
                        _var("out2"),
                        llir.ArrayAccess(_var("B_val"), llir.Literal(0)),
                    ),
                    llir.Assign(
                        _var("out3"),
                        llir.ArrayAccess(_var("B_val"), _var("unknown_position")),
                    ),
                ]
            )
        ]
    )

    conditional = llir.IfThenElse(
        cond=llir.Literal(True),
        then_body=[_sparse_loop()],
    )
    switch = llir.Switch(
        cond=_var("choice"),
        cases=[llir.Case(llir.Literal(0), [_sparse_loop()])],
        default=[_sparse_loop()],
    )
    function = llir.Function(
        return_type=llir.DataType.VOID,
        name="nested",
        args=[],
        body=[_sparse_loop()],
    )
    while_loop = llir.WhileLoop(llir.Literal(True), [_sparse_loop()])
    automatic_loop = llir.ForLoopAuto(
        _var("item"),
        _var("items"),
        [_sparse_loop()],
    )
    optional_regions = _inner_loop([llir.BlankLine()])
    optional_regions.before_parallel_body = [_sparse_loop()]
    optional_regions.pre_parallel_body = [_sparse_loop()]
    optional_regions.post_parallel_body = [_sparse_loop()]

    return [
        ("empty", []),
        ("non-loop", [llir.BlankLine()]),
        (
            "nested root statement sequence is semantically omitted",
            cast(List[llir.Stmt], [[_sparse_loop()]]),
        ),
        ("missing loop init", [missing_init]),
        ("non-Var sparse init value", [non_var_init_value]),
        ("non-sparse init spelling", [non_sparse]),
        ("legacy flat position spelling", [legacy_flat_position]),
        ("position array has the wrong type", [wrong_position_array_type]),
        ("position index has the wrong type", [wrong_position_index_type]),
        ("non-binary condition", [wrong_condition]),
        ("unnamed condition rhs", [unnamed_end]),
        ("missing coordinate", [missing_coordinate]),
        ("legacy flat coordinate spelling", [legacy_flat_coordinate]),
        (
            "structured coordinate has a non-variable base",
            [malformed_structured_coordinate],
        ),
        (
            "structured coordinate uses the wrong iterator",
            [wrong_index_structured_coordinate],
        ),
        ("no direct inner loop", [no_inner_loop]),
        ("no direct position init", [no_position]),
        ("wrong position expression", [wrong_position_shape]),
        ("position Add has non-binary left child", [non_binary_position_left]),
        ("position Add left child is not multiplication", [non_multiply_position]),
        ("position multiplication stride is not a Var", [non_var_stride]),
        ("raw pointer without assignment discovery", [no_assign]),
        ("assignment target is not scanned", [target_only_access]),
        ("cast expression is not scanned", [cast_hidden_access]),
        ("typed access shapes do not match", [unmatched_typed_accesses]),
        ("conditional container is omitted", [conditional]),
        ("switch container is omitted", [switch]),
        ("function container is omitted", [function]),
        ("while container is omitted", [while_loop]),
        ("automatic-for container is omitted", [automatic_loop]),
        ("optional loop regions are omitted", [optional_regions]),
    ]


def test_every_unmatched_structural_gate_is_a_detached_legal_noop() -> None:
    for name, source in _legal_noop_sources():
        source_snapshot = _structural_snapshot(source)

        output = insert_sparse_prefetch(source, SPARSE_PREFETCH_CONTEXT)

        assert _structural_snapshot(output) == source_snapshot, name
        assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output)), name


def test_repeated_application_preserves_non_idempotent_duplicate_insertion() -> None:
    first = insert_sparse_prefetch([_sparse_loop()], SPARSE_PREFETCH_CONTEXT)
    first_snapshot = _structural_snapshot(first)

    second = insert_sparse_prefetch(first, SPARSE_PREFETCH_CONTEXT)

    assert _structural_snapshot(first) == first_snapshot
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    first_loop = cast(llir.ForLoop, first[0])
    second_loop = cast(llir.ForLoop, second[0])
    assert [statement.code for statement in _prefetches(first_loop.body)] == [
        _expected_prefetch()
    ]
    assert [statement.code for statement in _prefetches(second_loop.body)] == [
        _expected_prefetch(),
        _expected_prefetch(),
    ]


def test_detachment_preserves_all_loop_fields_and_known_compatibility_attributes() -> (
    None
):
    loop = _sparse_loop()
    loop.omp_parallel_for = True
    loop.omp_schedule = "dynamic, 16"
    loop.unroll = True
    loop.simd = True
    loop.omp_num_threads = "threads"
    loop.omp_chunk_expr = "chunk"
    loop.scorch_index_var = "reduction"
    loop.before_parallel_body = [llir.RawStmt("before")]
    loop.pre_parallel_body = [llir.RawStmt("pre")]
    loop.post_parallel_body = [llir.RawStmt("post")]
    setattr(loop, "_use_atomic_scheduling", True)
    setattr(loop, "_atomic_chunk_var", "atomic_chunk")
    setattr(loop, "_atomic_counter_var", "counter")
    setattr(loop, "_loop_bound", "bound")
    setattr(loop, "_hoisted_ptr_decls", [llir.RawStmt("hoisted")])

    output = insert_sparse_prefetch([loop], SPARSE_PREFETCH_CONTEXT)

    rewritten = cast(llir.ForLoop, output[0])
    assert rewritten.omp_parallel_for is True
    assert rewritten.omp_schedule == "dynamic, 16"
    assert rewritten.unroll is True
    assert rewritten.simd is True
    assert rewritten.omp_num_threads == "threads"
    assert rewritten.omp_chunk_expr == "chunk"
    assert rewritten.scorch_index_var == "reduction"
    assert (
        cast(
            llir.RawStmt, cast(List[llir.Stmt], rewritten.before_parallel_body)[0]
        ).code
        == "before"
    )
    assert (
        cast(llir.RawStmt, cast(List[llir.Stmt], rewritten.pre_parallel_body)[0]).code
        == "pre"
    )
    assert (
        cast(llir.RawStmt, cast(List[llir.Stmt], rewritten.post_parallel_body)[0]).code
        == "post"
    )
    assert getattr(rewritten, "_use_atomic_scheduling") is True
    assert getattr(rewritten, "_atomic_chunk_var") == "atomic_chunk"
    assert getattr(rewritten, "_atomic_counter_var") == "counter"
    assert getattr(rewritten, "_loop_bound") == "bound"
    assert (
        cast(llir.RawStmt, getattr(rewritten, "_hoisted_ptr_decls")[0]).code
        == "hoisted"
    )
    assert _mutable_ir_ids([loop]).isdisjoint(_mutable_ir_ids(output))


class _UnknownStatement(llir.Stmt):
    pass


def test_unknown_nodes_and_malformed_children_fail_closed_with_pass_diagnostics() -> (
    None
):
    with pytest.raises(LLIRTraversalError) as unknown:
        insert_sparse_prefetch(
            [cast(llir.Stmt, _UnknownStatement())],
            SPARSE_PREFETCH_CONTEXT,
        )
    assert unknown.value.diagnostic.code == "unknown_llir_node"
    assert unknown.value.diagnostic.path == ("root", "[0]")
    assert unknown.value.diagnostic.node_type == "_UnknownStatement"
    assert unknown.value.diagnostic.stage == "LLIR transformation"
    assert unknown.value.diagnostic.pass_name == "insert_sparse_prefetch"

    omitted_container = llir.IfThenElse(
        cond=llir.Literal(True),
        then_body=[cast(llir.Stmt, _UnknownStatement())],
    )
    with pytest.raises(LLIRTraversalError) as nested_unknown:
        insert_sparse_prefetch([omitted_container], SPARSE_PREFETCH_CONTEXT)
    assert nested_unknown.value.diagnostic.code == "unknown_llir_node"
    assert nested_unknown.value.diagnostic.path == (
        "root",
        "[0]",
        "then_body",
        "[0]",
    )
    assert nested_unknown.value.diagnostic.pass_name == "insert_sparse_prefetch"

    malformed = _sparse_loop()
    malformed.body = cast(List[llir.Stmt], object())
    with pytest.raises(LLIRTraversalError) as child:
        insert_sparse_prefetch([malformed], SPARSE_PREFETCH_CONTEXT)
    assert child.value.diagnostic.code == "invalid_statement_sequence"
    assert child.value.diagnostic.path == ("root", "[0]", "body")
    assert child.value.diagnostic.node_type == "object"
    assert child.value.diagnostic.stage == "LLIR transformation"
    assert child.value.diagnostic.pass_name == "insert_sparse_prefetch"


@pytest.mark.parametrize(
    ("statements", "context", "code", "path", "node_type"),
    [
        (
            cast(List[llir.Stmt], (llir.BlankLine(),)),
            SPARSE_PREFETCH_CONTEXT,
            "unsupported_sparse_prefetch_root",
            ("root",),
            "tuple",
        ),
        (
            cast(List[llir.Stmt], [object()]),
            SPARSE_PREFETCH_CONTEXT,
            "invalid_sparse_prefetch_root_member",
            ("root", "[0]"),
            "object",
        ),
        (
            [llir.BlankLine()],
            cast(SparsePrefetchContext, object()),
            "invalid_sparse_prefetch_context",
            ("context",),
            "object",
        ),
        (
            [llir.BlankLine()],
            SparsePrefetchContext(traversal=cast(LLIRTraversalContext, object())),
            "invalid_sparse_prefetch_traversal_context",
            ("context", "traversal"),
            "object",
        ),
    ],
)
def test_wrong_roots_and_contexts_fail_closed(
    statements: List[llir.Stmt],
    context: SparsePrefetchContext,
    code: str,
    path: Tuple[str, ...],
    node_type: str,
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        insert_sparse_prefetch(statements, context)

    assert raised.value.diagnostic.code == code
    assert raised.value.diagnostic.path == path
    assert raised.value.diagnostic.node_type == node_type
    assert raised.value.diagnostic.stage == "LLIR transformation"
    assert raised.value.diagnostic.pass_name == "insert_sparse_prefetch"


def test_manager_returns_exact_detached_artifact_and_ordered_record() -> None:
    source = [_sparse_loop()]

    result = LLIRPassManager().run_sparse_prefetch(
        LLIRStatementListArtifact(source),
        SparsePrefetchPassSpec(),
    )

    assert type(result) is LLIRStatementListPassResult
    assert type(result.artifact) is LLIRStatementListArtifact
    assert type(result.artifact.statements) is list
    assert _mutable_ir_ids(source).isdisjoint(
        _mutable_ir_ids(result.artifact.statements)
    )
    assert len(result.run_records) == 1
    record = result.run_records[0]
    assert record.sequence_index == 0
    assert record.pass_name == "insert_sparse_prefetch"
    assert record.pass_version == 1
    assert record.input_artifact is LLIRPassArtifactType.STATEMENT_LIST
    assert record.output_artifact is LLIRPassArtifactType.STATEMENT_LIST
    assert record.context_type is LLIRPassContextType.SPARSE_PREFETCH
    assert record.configuration_name == "sparse_prefetch"
    assert record.diagnostic_stage == "LLIR transformation"
    assert record.diagnostic_pass_name == "insert_sparse_prefetch"


def test_manager_rejects_wrong_descriptor_artifact_spec_and_context() -> None:
    manager = LLIRPassManager()
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    bad_descriptor = replace(SPARSE_PREFETCH_PASS, version=2)

    class SpoofedDescriptor:
        def __eq__(self, other: object) -> bool:
            return True

        def __ne__(self, other: object) -> bool:
            return False

    invalid_calls: Tuple[Callable[[], object], ...] = (
        lambda: manager.run_sparse_prefetch(
            artifact,
            SparsePrefetchPassSpec(descriptor=bad_descriptor),
        ),
        lambda: manager.run_sparse_prefetch(
            artifact,
            SparsePrefetchPassSpec(
                descriptor=cast(LLIRPassDescriptor, SpoofedDescriptor())
            ),
        ),
        lambda: manager.run_sparse_prefetch(
            cast(LLIRStatementListArtifact, object()),
            SparsePrefetchPassSpec(),
        ),
        lambda: manager.run_sparse_prefetch(
            artifact,
            cast(SparsePrefetchPassSpec, object()),
        ),
        lambda: manager.run_sparse_prefetch(
            artifact,
            SparsePrefetchPassSpec(context=cast(SparsePrefetchContext, object())),
        ),
    )
    for invalid in invalid_calls:
        with pytest.raises(LLIRPassManagerError):
            invalid()


def test_manager_preserves_the_direct_pass_structured_failure() -> None:
    source = [cast(llir.Stmt, _UnknownStatement())]

    with pytest.raises(LLIRTraversalError) as direct:
        insert_sparse_prefetch(source, SPARSE_PREFETCH_CONTEXT)
    with pytest.raises(LLIRTraversalError) as managed:
        LLIRPassManager().run_sparse_prefetch(LLIRStatementListArtifact(source))

    assert managed.value.diagnostic == direct.value.diagnostic


def test_production_skips_extra_walks_and_debug_verifies_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager_walks: List[Tuple[str, str]] = []

    class RecordingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            manager_walks.append((self.context.stage, self.context.pass_name))
            super().walk(value)

    monkeypatch.setattr(pass_manager_module, "LLIRWalker", RecordingWalker)
    artifact = LLIRStatementListArtifact([llir.BlankLine()])

    production = LLIRPassManager(PRODUCTION_LLIR_PASS_OPTIONS).run_sparse_prefetch(
        artifact
    )
    assert manager_walks == []
    assert production.run_records[0].verified_before is False
    assert production.run_records[0].verified_after is False

    debug = LLIRPassManager(DEBUG_LLIR_PASS_OPTIONS).run_sparse_prefetch(artifact)
    assert manager_walks == [
        ("LLIR transformation", "insert_sparse_prefetch"),
        ("LLIR transformation", "insert_sparse_prefetch"),
    ]
    assert debug.run_records[0].verified_before is True
    assert debug.run_records[0].verified_after is True


def test_timing_and_run_record_equality_are_nonsemantic_and_optional() -> None:
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    timed = LLIRPassManager().run_sparse_prefetch(artifact)
    untimed = LLIRPassManager(LLIRPassOptions(record_timing=False)).run_sparse_prefetch(
        artifact
    )

    assert timed.run_records[0].duration_ns is not None
    assert cast(int, timed.run_records[0].duration_ns) >= 0
    assert untimed.run_records[0].duration_ns is None
    assert timed.run_records[0] == replace(timed.run_records[0], duration_ns=999)
    assert LLIRStatementListPassResult(
        timed.artifact,
        timed.run_records,
    ) == LLIRStatementListPassResult(timed.artifact, untimed.run_records)


def _p95(samples: List[int]) -> int:
    ordered = sorted(samples)
    return ordered[int((len(ordered) - 1) * 0.95)]


def test_sparse_prefetch_incremental_manager_plumbing_p95_is_below_one_ms() -> None:
    source = [llir.BlankLine()]
    artifact = LLIRStatementListArtifact(source)
    manager = LLIRPassManager(LLIRPassOptions(record_timing=False))

    for _ in range(100):
        insert_sparse_prefetch(source, SPARSE_PREFETCH_CONTEXT)
        manager.run_sparse_prefetch(artifact)

    incremental_ns: List[int] = []
    for sample in range(2000):
        if sample % 2:
            managed_start = perf_counter_ns()
            manager.run_sparse_prefetch(artifact)
            managed_elapsed = perf_counter_ns() - managed_start
            direct_start = perf_counter_ns()
            insert_sparse_prefetch(source, SPARSE_PREFETCH_CONTEXT)
            direct_elapsed = perf_counter_ns() - direct_start
        else:
            direct_start = perf_counter_ns()
            insert_sparse_prefetch(source, SPARSE_PREFETCH_CONTEXT)
            direct_elapsed = perf_counter_ns() - direct_start
            managed_start = perf_counter_ns()
            manager.run_sparse_prefetch(artifact)
            managed_elapsed = perf_counter_ns() - managed_start
        incremental_ns.append(managed_elapsed - direct_elapsed)

    assert _p95(incremental_ns) <= 1_000_000


def _build_activating_spmm_source() -> ForAll:
    row, reduction, column = IndexVar("i"), IndexVar("k"), IndexVar("j")
    result = TensorVar("C", fmt="dd")
    left = TensorVar("A", fmt="ds")
    right = TensorVar("B", fmt="dd")
    assignment = TensorAssign(
        result[row, column],
        left[row, reduction] * right[reduction, column],
        op=Operation.ADD,
    )
    return ForAll(row, ForAll(reduction, ForAll(column, assignment)))


def _build_activating_spmm_cin() -> ForAll:
    return cast(
        ForAll,
        Scheduler.auto_schedule(_build_activating_spmm_source()),
    )


def _build_activating_all_coo_sddmm_cin() -> ForAll:
    row, column, reduction = IndexVar("r"), IndexVar("c"), IndexVar("q")
    result = TensorVar("Sampled", fmt="oo")
    mask = TensorVar("Mask", fmt="oo")
    query = TensorVar("Query", fmt="dd")
    key = TensorVar("Key", fmt="dd")
    assignment = TensorAssign(
        result[row, column],
        mask[row, column] * query[row, reduction] * key[column, reduction],
        op=Operation.ADD,
    )
    return cast(
        ForAll,
        Scheduler.auto_schedule(
            ForAll(row, ForAll(column, ForAll(reduction, assignment)))
        ),
    )


def _build_compressed_ds_cin() -> ForAll:
    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("SparseProduct", fmt="ds")
    left = TensorVar("SparseLeft", fmt="ds")
    right = TensorVar("SparseRight", fmt="ds")
    assignment = TensorAssign(
        result[row, column],
        left[row, reduction] * right[reduction, column],
        op=Operation.ADD,
    )
    return cast(
        ForAll,
        Scheduler.auto_schedule(
            ForAll(row, ForAll(reduction, ForAll(column, assignment)))
        ),
    )


def _all_raw_codes(value: object) -> List[str]:
    codes: List[str] = []
    if type(value) is llir.RawStmt:
        codes.append(cast(llir.RawStmt, value).code)
    elif isinstance(value, llir.Node):
        for child in vars(value).values():
            codes.extend(_all_raw_codes(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            codes.extend(_all_raw_codes(child))
    return codes


def _coordinate_reads(
    value: object,
    *,
    target_name: str,
    array_name: str,
    index_name: str,
) -> List[llir.ArrayAccess]:
    matches: List[llir.ArrayAccess] = []

    class CoordinateReadCollector(LLIRWalker):
        def visit_var_init(
            self,
            node: llir.VarInit,
            path: Tuple[str, ...],
        ) -> None:
            if type(node.value) is llir.ArrayAccess:
                access = cast(llir.ArrayAccess, node.value)
                if (
                    node.var.name == target_name
                    and type(access.array) is llir.Var
                    and cast(llir.Var, access.array).name == array_name
                    and type(access.index) is llir.Var
                    and cast(llir.Var, access.index).name == index_name
                ):
                    matches.append(access)
            super().visit_var_init(node, path)

    CoordinateReadCollector(
        LLIRTraversalContext(
            stage="test",
            pass_name="collect_iterator_coordinate_read",
        )
    ).walk(value)
    return matches


def test_production_routes_the_detached_list_at_the_original_optimization_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statement = _build_activating_spmm_cin()
    original_statement = str(statement)
    events: List[str] = []
    detached_sparse_output: List[List[llir.Stmt]] = []
    detached_dense_output: List[List[llir.Stmt]] = []
    detached_single_output: List[List[llir.Stmt]] = []
    detached_factor_output: List[List[llir.Stmt]] = []
    original_sparse = LLIRPassManager.run_sparse_prefetch
    original_dense = LLIRPassManager.run_dense_pointer_hoist
    original_single = LLIRPassManager.run_single_iteration_loop_elimination
    original_factor = LLIRPassManager.run_loop_invariant_factor_hoist
    original_dynamic = LLIRPassManager.run_dynamic_vector_access

    def record_sparse(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: SparsePrefetchPassSpec = SparsePrefetchPassSpec(),
    ) -> LLIRStatementListPassResult:
        events.append("sparse_prefetch")
        assert not any(
            "validate_jit_tensor" in code
            for code in _all_raw_codes(artifact.statements)
        )
        result = original_sparse(manager, artifact, pass_spec)
        assert _mutable_ir_ids(artifact.statements).isdisjoint(
            _mutable_ir_ids(result.artifact.statements)
        )
        detached_sparse_output.append(result.artifact.statements)
        return result

    def record_dense(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: DensePointerHoistPassSpec,
    ) -> LLIRStatementListPassResult:
        events.append("dense_pointer")
        assert artifact.statements is detached_sparse_output[-1]
        assert pass_spec.context.value_array_ctypes == (
            ("A_val", "float"),
            ("B_val", "float"),
        )
        result = original_dense(manager, artifact, pass_spec)
        assert _mutable_ir_ids(artifact.statements).isdisjoint(
            _mutable_ir_ids(result.artifact.statements)
        )
        detached_dense_output.append(result.artifact.statements)
        return result

    def record_single(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: SingleIterationLoopEliminationPassSpec = (
            SingleIterationLoopEliminationPassSpec()
        ),
    ) -> LLIRStatementListPassResult:
        events.append("single_iteration")
        assert artifact.statements is detached_dense_output[-1]
        result = original_single(manager, artifact, pass_spec)
        assert _mutable_ir_ids(artifact.statements).isdisjoint(
            _mutable_ir_ids(result.artifact.statements)
        )
        detached_single_output.append(result.artifact.statements)
        return result

    def record_factor(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: LoopInvariantFactorHoistPassSpec = (
            LoopInvariantFactorHoistPassSpec()
        ),
    ) -> LLIRStatementListPassResult:
        events.append("factor_hoist")
        assert artifact.statements is detached_single_output[-1]
        assert not any(
            "validate_jit_tensor" in code
            for code in _all_raw_codes(artifact.statements)
        )
        result = original_factor(manager, artifact, pass_spec)
        assert _mutable_ir_ids(artifact.statements).isdisjoint(
            _mutable_ir_ids(result.artifact.statements)
        )
        detached_factor_output.append(result.artifact.statements)
        return result

    def record_dynamic(
        manager: LLIRPassManager,
        artifact: LLIRRewriteArtifact[List[llir.Stmt]],
        pass_spec: DynamicVectorAccessPassSpec = DynamicVectorAccessPassSpec(),
    ) -> LLIRRewritePassResult[List[llir.Stmt]]:
        events.append("dynamic_vector")
        assert any(
            "validate_jit_tensor" in code for code in _all_raw_codes(artifact.value)
        )
        assert all(
            any(statement is candidate for candidate in artifact.value)
            for statement in detached_factor_output[-1]
        )
        return original_dynamic(manager, artifact, pass_spec)

    monkeypatch.setattr(LLIRPassManager, "run_sparse_prefetch", record_sparse)
    monkeypatch.setattr(LLIRPassManager, "run_dense_pointer_hoist", record_dense)
    monkeypatch.setattr(
        LLIRPassManager,
        "run_single_iteration_loop_elimination",
        record_single,
    )
    monkeypatch.setattr(LLIRPassManager, "run_dynamic_vector_access", record_dynamic)
    monkeypatch.setattr(
        LLIRPassManager,
        "run_loop_invariant_factor_hoist",
        record_factor,
    )

    def lower() -> tuple[CINLowerer, llir.ArrayAccess, str]:
        lowerer = CINLowerer()
        lowered = lowerer.lower_IndexStmt(statement)
        coordinate_reads = _coordinate_reads(
            lowered,
            target_name="k",
            array_name="A1_crd",
            index_name="pA1",
        )
        assert len(coordinate_reads) == 1
        return lowerer, coordinate_reads[0], LLIRLowerer().lower_llir(lowered)

    lowerer, coordinate_read, cpp = lower()
    second_lowerer, second_coordinate_read, second_cpp = lower()

    expected_events = [
        "sparse_prefetch",
        "dense_pointer",
        "single_iteration",
        "factor_hoist",
        "dynamic_vector",
    ]
    assert events == expected_events * 2
    assert str(statement) == original_statement
    assert second_cpp == cpp
    assert coordinate_read == llir.ArrayAccess(
        llir.Var("A1_crd", llir.DataType.PTR_INT),
        llir.Var("pA1", llir.DataType.INT),
    )
    assert coordinate_read is not second_coordinate_read
    assert coordinate_read.array is not second_coordinate_read.array
    assert coordinate_read.index is not second_coordinate_read.index
    assert coordinate_read.tensor_access is None
    assert second_coordinate_read.tensor_access is None
    assert second_lowerer.llir_pass_run_records == lowerer.llir_pass_run_records
    assert len(cpp) == 2505
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151"
    )
    assert "int pA1_end = A1_pos[pA0 + 1];" in cpp
    assert "pA1 < pA1_end; pA1++" in cpp
    assert "pA1 + 1 < pA1_end" in cpp
    assert _expected_prefetch() + ";" in cpp
    assert ("const float* __restrict__ _B_val_ptr = " "&B_val[pB0 * B1_size];") in cpp
    assert [record.pass_name for record in lowerer.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.sequence_index for record in lowerer.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_production_all_coo_sddmm_activates_single_iteration_and_factor_hoist() -> None:
    statement = _build_activating_all_coo_sddmm_cin()
    original = str(statement)

    def lower() -> tuple[CINLowerer, llir.ArrayAccess, llir.ArrayAccess, str]:
        lowerer = CINLowerer()
        lowered = lowerer.lower_IndexStmt(statement)
        transformed_coordinate_reads = _coordinate_reads(
            lowered,
            target_name="r",
            array_name="Mask0_crd",
            index_name="pMask0",
        )
        iterator_coordinate_reads = _coordinate_reads(
            lowered,
            target_name="c",
            array_name="Mask1_crd",
            index_name="pMask0",
        )
        assert len(transformed_coordinate_reads) == 1
        assert len(iterator_coordinate_reads) == 1
        return (
            lowerer,
            transformed_coordinate_reads[0],
            iterator_coordinate_reads[0],
            LLIRLowerer().lower_llir(lowered),
        )

    with regblock_force(False):
        lowerer, coordinate_read, iterator_coordinate_read, cpp = lower()
        (
            second_lowerer,
            second_coordinate_read,
            second_iterator_coordinate_read,
            second_cpp,
        ) = lower()

    assert str(statement) == original
    assert second_cpp == cpp
    assert coordinate_read == llir.ArrayAccess(
        llir.Var("Mask0_crd", llir.DataType.PTR_INT),
        llir.Var("pMask0", llir.DataType.INT64),
    )
    assert coordinate_read is not second_coordinate_read
    assert coordinate_read.array is not second_coordinate_read.array
    assert coordinate_read.index is not second_coordinate_read.index
    assert coordinate_read.tensor_access is None
    assert iterator_coordinate_read == llir.ArrayAccess(
        llir.Var("Mask1_crd", llir.DataType.PTR_INT),
        llir.Var("pMask0", llir.DataType.INT),
    )
    assert iterator_coordinate_read is not second_iterator_coordinate_read
    assert iterator_coordinate_read.array is not second_iterator_coordinate_read.array
    assert iterator_coordinate_read.index is not second_iterator_coordinate_read.index
    assert iterator_coordinate_read.tensor_access is None
    assert second_iterator_coordinate_read.tensor_access is None
    assert second_lowerer.llir_pass_run_records == lowerer.llir_pass_run_records
    assert len(cpp) == 3521
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "53d6faaee132a5d82515235b529d7d88d16cbeefe388eba5cfae9ace5528d667"
    )
    assert "pMask1_end" not in cpp
    assert "pMask1 = pMask0" not in cpp
    assert "pMask1 <" not in cpp
    assert "pMask1++" not in cpp
    assert "int pMask0_end = Mask0_crd_tensor.size(0);" in cpp
    assert "pMask0 < pMask0_end; pMask0++" in cpp
    assert "Mask1_crd[pMask0]" in cpp
    assert "Mask_val[pMask0]" in cpp
    assert (
        "    float _inv_17 = Mask_val[pMask0];\n"
        "    #pragma omp simd\n"
        "    for (int64_t q = 0; q < Query1_size; q++) {\n"
        "      // Resolve dense coordinates\n"
        "      _accum += _Query_val_ptr[q] * _Key_val_ptr[q];\n"
        "    }\n"
        "    _accum *= _inv_17;"
    ) in cpp
    assert [
        (record.pass_name, record.configuration_name)
        for record in lowerer.llir_pass_run_records
    ] == [
        ("insert_sparse_prefetch", "sparse_prefetch"),
        ("hoist_dense_pointers", "dense_pointer_hoist"),
        (
            "eliminate_single_iteration_loops",
            "single_iteration_loop_elimination",
        ),
        ("hoist_loop_invariant_factors", "loop_invariant_factor_hoist"),
        ("rewrite_dynamic_vector_accesses", "dynamic_vector_access"),
    ]


def test_sparse_prefetch_failure_stops_remaining_optimizations_and_managed_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: List[str] = []
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_sparse_prefetch_failure",
            message="sparse prefetch failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name="insert_sparse_prefetch",
        )
    )

    def fail_sparse(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: SparsePrefetchPassSpec = SparsePrefetchPassSpec(),
    ) -> NoReturn:
        events.append("sparse_prefetch")
        raise failure

    def record_dense(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: DensePointerHoistPassSpec,
    ) -> NoReturn:
        events.append("dense_pointer")
        raise AssertionError("dense-pointer pass must not run after failure")

    def record_single(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: SingleIterationLoopEliminationPassSpec = (
            SingleIterationLoopEliminationPassSpec()
        ),
    ) -> NoReturn:
        events.append("single_iteration")
        raise AssertionError("single-iteration pass must not run after failure")

    def record_factor(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: LoopInvariantFactorHoistPassSpec = (
            LoopInvariantFactorHoistPassSpec()
        ),
    ) -> NoReturn:
        events.append("factor_hoist")
        raise AssertionError("factor hoisting must not run after failure")

    def record_dynamic(
        manager: LLIRPassManager,
        artifact: LLIRRewriteArtifact[List[llir.Stmt]],
        pass_spec: DynamicVectorAccessPassSpec = DynamicVectorAccessPassSpec(),
    ) -> LLIRRewritePassResult[List[llir.Stmt]]:
        events.append("dynamic_vector")
        raise AssertionError("dynamic-vector pass must not run after failure")

    monkeypatch.setattr(LLIRPassManager, "run_sparse_prefetch", fail_sparse)
    monkeypatch.setattr(LLIRPassManager, "run_dense_pointer_hoist", record_dense)
    monkeypatch.setattr(
        LLIRPassManager,
        "run_single_iteration_loop_elimination",
        record_single,
    )
    monkeypatch.setattr(LLIRPassManager, "run_dynamic_vector_access", record_dynamic)
    monkeypatch.setattr(
        LLIRPassManager,
        "run_loop_invariant_factor_hoist",
        record_factor,
    )

    lowerer = CINLowerer()
    with pytest.raises(LLIRTraversalError) as raised:
        lowerer.lower_IndexStmt(_build_activating_spmm_cin())

    assert raised.value is failure
    assert events == ["sparse_prefetch"]
    assert lowerer.llir_pass_run_records == ()


def test_compressed_fill_failure_preserves_count_record_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: List[str] = []
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_compressed_fill_failure",
            message="compressed fill failed",
            path=("root",),
            node_type="RawStmt",
            stage="LLIR transformation",
            pass_name="transform_compressed_where_for_openmp",
        )
    )
    original_result_write = LLIRPassManager.run_result_write

    def fail_fill(
        manager: LLIRPassManager,
        artifact: LLIRRewriteArtifact[object],
        pass_spec: ResultWritePassSpec,
    ) -> LLIRRewritePassResult[object]:
        events.append(pass_spec.context.mode)
        if pass_spec.context.mode == "fill":
            raise failure
        return original_result_write(manager, artifact, pass_spec)

    def reject_sparse(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: SparsePrefetchPassSpec = SparsePrefetchPassSpec(),
    ) -> NoReturn:
        events.append("sparse_prefetch")
        raise AssertionError("sparse prefetch must not run after fill failure")

    monkeypatch.setattr(LLIRPassManager, "run_result_write", fail_fill)
    monkeypatch.setattr(LLIRPassManager, "run_sparse_prefetch", reject_sparse)

    lowerer = CINLowerer()
    with pytest.raises(LLIRTraversalError) as raised:
        lowerer.lower_IndexStmt(_build_compressed_ds_cin())

    assert raised.value is failure
    assert events == ["count", "fill"]
    assert [
        (record.pass_name, record.configuration_name, record.sequence_index)
        for record in lowerer.llir_pass_run_records
    ] == [("rewrite_result_writes", "count", 0)]


def test_dense_pointer_failure_preserves_prior_records_and_stops_all_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: List[str] = []
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_dense_pointer_failure",
            message="dense pointer failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name="hoist_dense_pointers",
        )
    )

    def fail_dense(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: DensePointerHoistPassSpec,
    ) -> NoReturn:
        events.append("dense_pointer")
        raise failure

    def record_single(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: SingleIterationLoopEliminationPassSpec = (
            SingleIterationLoopEliminationPassSpec()
        ),
    ) -> NoReturn:
        events.append("single_iteration")
        raise AssertionError("single-iteration pass must not run after failure")

    def record_factor(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: LoopInvariantFactorHoistPassSpec = (
            LoopInvariantFactorHoistPassSpec()
        ),
    ) -> NoReturn:
        events.append("factor_hoist")
        raise AssertionError("factor hoisting must not run after failure")

    def record_dynamic(
        manager: LLIRPassManager,
        artifact: LLIRRewriteArtifact[List[llir.Stmt]],
        pass_spec: DynamicVectorAccessPassSpec = DynamicVectorAccessPassSpec(),
    ) -> LLIRRewritePassResult[List[llir.Stmt]]:
        events.append("dynamic_vector")
        raise AssertionError("dynamic-vector pass must not run after failure")

    monkeypatch.setattr(LLIRPassManager, "run_dense_pointer_hoist", fail_dense)
    monkeypatch.setattr(
        LLIRPassManager,
        "run_single_iteration_loop_elimination",
        record_single,
    )
    monkeypatch.setattr(LLIRPassManager, "run_dynamic_vector_access", record_dynamic)
    monkeypatch.setattr(
        LLIRPassManager,
        "run_loop_invariant_factor_hoist",
        record_factor,
    )

    lowerer = CINLowerer()
    with pytest.raises(LLIRTraversalError) as raised:
        lowerer.lower_IndexStmt(_build_compressed_ds_cin())

    assert raised.value is failure
    assert events == ["dense_pointer"]
    assert [
        (record.pass_name, record.configuration_name)
        for record in lowerer.llir_pass_run_records
    ] == [
        ("transform_compressed_where_for_openmp", "compressed_where_openmp"),
        ("rewrite_result_writes", "count"),
        ("rewrite_result_writes", "fill"),
        ("insert_sparse_prefetch", "sparse_prefetch"),
    ]
    assert [record.sequence_index for record in lowerer.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
    ]


def test_single_iteration_failure_preserves_prior_records_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: List[str] = []
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_single_iteration_failure",
            message="single-iteration elimination failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name=SINGLE_ITERATION_LOOP_ELIMINATION_PASS.name,
        )
    )

    def fail_single(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: SingleIterationLoopEliminationPassSpec = (
            SingleIterationLoopEliminationPassSpec()
        ),
    ) -> NoReturn:
        events.append("single_iteration")
        raise failure

    def record_factor(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: LoopInvariantFactorHoistPassSpec = (
            LoopInvariantFactorHoistPassSpec()
        ),
    ) -> NoReturn:
        events.append("factor_hoist")
        raise AssertionError("factor hoisting must not run after failure")

    def record_assembly(assembler: ResultTensorAssembler) -> NoReturn:
        events.append("assembly")
        raise AssertionError("result assembly must not run after failure")

    def record_dynamic(
        manager: LLIRPassManager,
        artifact: LLIRRewriteArtifact[List[llir.Stmt]],
        pass_spec: DynamicVectorAccessPassSpec = DynamicVectorAccessPassSpec(),
    ) -> NoReturn:
        events.append("dynamic_vector")
        raise AssertionError("dynamic-vector pass must not run after failure")

    def record_schedule(*args: object, **kwargs: object) -> NoReturn:
        events.append("schedule_lowering")
        raise AssertionError("schedule lowering must not run after failure")

    def record_codegen(*args: object, **kwargs: object) -> NoReturn:
        events.append("codegen")
        raise AssertionError("code generation must not run after failure")

    monkeypatch.setattr(
        LLIRPassManager,
        "run_single_iteration_loop_elimination",
        fail_single,
    )
    monkeypatch.setattr(
        LLIRPassManager,
        "run_loop_invariant_factor_hoist",
        record_factor,
    )
    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_final_assembly",
        record_assembly,
    )
    monkeypatch.setattr(LLIRPassManager, "run_dynamic_vector_access", record_dynamic)
    monkeypatch.setattr(
        schedule_lowerer_module,
        "apply_schedule_to_llir",
        record_schedule,
    )
    monkeypatch.setattr(LLIRLowerer, "lower_llir", record_codegen)

    scheduled = Scheduler.apply_schedule(_build_activating_spmm_source(), Schedule())
    lowerer = CINLowerer()
    with pytest.raises(LLIRTraversalError) as raised:
        LLIRLowerer().lower_llir(lowerer.lower_IndexStmt(scheduled))

    assert raised.value is failure
    assert events == ["single_iteration"]
    assert [
        (record.pass_name, record.configuration_name)
        for record in lowerer.llir_pass_run_records
    ] == [
        ("insert_sparse_prefetch", "sparse_prefetch"),
        ("hoist_dense_pointers", "dense_pointer_hoist"),
    ]
    assert [record.sequence_index for record in lowerer.llir_pass_run_records] == [
        0,
        1,
    ]


def test_factor_failure_preserves_prior_records_and_stops_all_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: List[str] = []
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_loop_invariant_factor_failure",
            message="loop-invariant factor hoisting failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name=LOOP_INVARIANT_FACTOR_HOIST_PASS.name,
        )
    )

    def fail_factor(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: LoopInvariantFactorHoistPassSpec = (
            LoopInvariantFactorHoistPassSpec()
        ),
    ) -> NoReturn:
        events.append("factor_hoist")
        raise failure

    def record_assembly(assembler: ResultTensorAssembler) -> NoReturn:
        events.append("assembly")
        raise AssertionError("result assembly must not run after factor failure")

    def record_dynamic(
        manager: LLIRPassManager,
        artifact: LLIRRewriteArtifact[List[llir.Stmt]],
        pass_spec: DynamicVectorAccessPassSpec = DynamicVectorAccessPassSpec(),
    ) -> NoReturn:
        events.append("dynamic_vector")
        raise AssertionError("dynamic-vector pass must not run after factor failure")

    def record_function(*args: object, **kwargs: object) -> NoReturn:
        events.append("function")
        raise AssertionError("Function construction must not run after factor failure")

    def record_schedule(*args: object, **kwargs: object) -> NoReturn:
        events.append("schedule_lowering")
        raise AssertionError("schedule lowering must not run after factor failure")

    def record_codegen(*args: object, **kwargs: object) -> NoReturn:
        events.append("codegen")
        raise AssertionError("code generation must not run after factor failure")

    monkeypatch.setattr(
        LLIRPassManager,
        "run_loop_invariant_factor_hoist",
        fail_factor,
    )
    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_final_assembly",
        record_assembly,
    )
    monkeypatch.setattr(LLIRPassManager, "run_dynamic_vector_access", record_dynamic)
    monkeypatch.setattr(llir, "Function", record_function)
    monkeypatch.setattr(
        schedule_lowerer_module,
        "apply_schedule_to_llir",
        record_schedule,
    )
    monkeypatch.setattr(LLIRLowerer, "lower_llir", record_codegen)

    scheduled = Scheduler.apply_schedule(_build_activating_spmm_source(), Schedule())
    lowerer = CINLowerer()
    with pytest.raises(LLIRTraversalError) as raised:
        LLIRLowerer().lower_llir(lowerer.lower_IndexStmt(scheduled))

    assert raised.value is failure
    assert events == ["factor_hoist"]
    assert [
        (record.pass_name, record.configuration_name)
        for record in lowerer.llir_pass_run_records
    ] == [
        ("insert_sparse_prefetch", "sparse_prefetch"),
        ("hoist_dense_pointers", "dense_pointer_hoist"),
        (
            "eliminate_single_iteration_loops",
            "single_iteration_loop_elimination",
        ),
    ]
    assert [record.sequence_index for record in lowerer.llir_pass_run_records] == [
        0,
        1,
        2,
    ]


def test_dynamic_vector_failure_preserves_prior_records_and_stops_function_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: List[str] = []
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_dynamic_vector_failure",
            message="dynamic-vector rewriting failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name="rewrite_dynamic_vector_accesses",
        )
    )

    def fail_dynamic(
        manager: LLIRPassManager,
        artifact: LLIRRewriteArtifact[List[llir.Stmt]],
        pass_spec: DynamicVectorAccessPassSpec = DynamicVectorAccessPassSpec(),
    ) -> NoReturn:
        events.append("dynamic_vector")
        raise failure

    def reject_function(*args: object, **kwargs: object) -> NoReturn:
        events.append("function")
        raise AssertionError("function construction must not run after pass failure")

    monkeypatch.setattr(LLIRPassManager, "run_dynamic_vector_access", fail_dynamic)
    monkeypatch.setattr(llir, "Function", reject_function)

    lowerer = CINLowerer()
    with pytest.raises(LLIRTraversalError) as raised:
        lowerer.lower_IndexStmt(_build_activating_spmm_cin())

    assert raised.value is failure
    assert events == ["dynamic_vector"]
    assert [
        (record.pass_name, record.configuration_name, record.sequence_index)
        for record in lowerer.llir_pass_run_records
    ] == [
        ("insert_sparse_prefetch", "sparse_prefetch", 0),
        ("hoist_dense_pointers", "dense_pointer_hoist", 1),
        (
            "eliminate_single_iteration_loops",
            "single_iteration_loop_elimination",
            2,
        ),
        ("hoist_loop_invariant_factors", "loop_invariant_factor_hoist", 3),
    ]


def test_compressed_factor_failure_preserves_all_prior_managed_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_compressed_loop_invariant_factor_failure",
            message="loop-invariant factor hoisting failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name=LOOP_INVARIANT_FACTOR_HOIST_PASS.name,
        )
    )

    def fail_factor(
        manager: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: LoopInvariantFactorHoistPassSpec = (
            LoopInvariantFactorHoistPassSpec()
        ),
    ) -> NoReturn:
        raise failure

    monkeypatch.setattr(
        LLIRPassManager,
        "run_loop_invariant_factor_hoist",
        fail_factor,
    )

    lowerer = CINLowerer()
    with pytest.raises(LLIRTraversalError) as raised:
        lowerer.lower_IndexStmt(_build_compressed_ds_cin())

    assert raised.value is failure
    assert [
        (record.pass_name, record.configuration_name)
        for record in lowerer.llir_pass_run_records
    ] == [
        ("transform_compressed_where_for_openmp", "compressed_where_openmp"),
        ("rewrite_result_writes", "count"),
        ("rewrite_result_writes", "fill"),
        ("insert_sparse_prefetch", "sparse_prefetch"),
        ("hoist_dense_pointers", "dense_pointer_hoist"),
        (
            "eliminate_single_iteration_loops",
            "single_iteration_loop_elimination",
        ),
    ]
    assert [record.sequence_index for record in lowerer.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


def test_applied_compressed_production_records_factor_before_dynamic_vector() -> None:
    lowerer = CINLowerer()

    LLIRLowerer().lower_llir(lowerer.lower_IndexStmt(_build_compressed_ds_cin()))

    assert [
        (record.pass_name, record.configuration_name)
        for record in lowerer.llir_pass_run_records
    ] == [
        ("transform_compressed_where_for_openmp", "compressed_where_openmp"),
        ("rewrite_result_writes", "count"),
        ("rewrite_result_writes", "fill"),
        ("insert_sparse_prefetch", "sparse_prefetch"),
        ("hoist_dense_pointers", "dense_pointer_hoist"),
        (
            "eliminate_single_iteration_loops",
            "single_iteration_loop_elimination",
        ),
        ("hoist_loop_invariant_factors", "loop_invariant_factor_hoist"),
        ("rewrite_dynamic_vector_accesses", "dynamic_vector_access"),
    ]
    assert [record.sequence_index for record in lowerer.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
