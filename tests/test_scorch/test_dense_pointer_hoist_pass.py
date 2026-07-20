from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from time import perf_counter_ns
from typing import Callable, List, NoReturn, Sequence, Set, Tuple, cast

import pytest

from scorch.compiler import llir  # type: ignore[import-untyped]
import scorch.compiler.llir_pass_manager as pass_manager_module  # type: ignore[import-untyped]
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.compiler.dense_pointer_hoist_pass import (  # type: ignore[import-untyped]
    DENSE_POINTER_HOIST_TRAVERSAL_CONTEXT,
    DensePointerHoistContext,
    hoist_dense_pointers,
)
from scorch.compiler.llir_pass_manager import (  # type: ignore[import-untyped]
    DEBUG_LLIR_PASS_OPTIONS,
    DENSE_POINTER_HOIST_PASS,
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
    DensePointerHoistPassSpec,
)
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
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


def _position_init(
    position: str = "position",
    base: str = "base",
    stride: str = "stride",
    loop_variable: str = "lane",
) -> llir.VarInit:
    return llir.VarInit(
        _var(position, llir.DataType.INT64),
        llir.Add(
            llir.BinOp("*", _var(base), _var(stride)),
            _var(loop_variable),
        ),
    )


def _loop(
    body: Sequence[llir.Stmt],
    *,
    loop_variable: str = "lane",
    update: llir.Stmt | llir.FunctionCall | None = None,
) -> llir.ForLoop:
    if update is None:
        update = llir.Increment(_var(loop_variable, llir.DataType.INT64))
    return llir.ForLoop(
        init=llir.VarInit(
            _var(loop_variable, llir.DataType.INT64),
            llir.Literal(0),
        ),
        cond=llir.BinOp(
            "<",
            _var(loop_variable, llir.DataType.INT64),
            _var("extent", llir.DataType.INT64),
        ),
        update=cast(
            llir.Increment | llir.VarInit | llir.FunctionCall | llir.Assign,
            update,
        ),
        body=list(body),
    )


def _activating_loop(
    *,
    value_array: str = "Input_val",
    position: str = "position",
    base: str = "base",
    stride: str = "stride",
    loop_variable: str = "lane",
) -> llir.ForLoop:
    return _loop(
        [
            _position_init(position, base, stride, loop_variable),
            llir.Assign(_var("output"), _var(f"{value_array}[{position}]")),
        ],
        loop_variable=loop_variable,
    )


def _context(
    *entries: Tuple[str, str],
    traversal: LLIRTraversalContext = DENSE_POINTER_HOIST_TRAVERSAL_CONTEXT,
) -> DensePointerHoistContext:
    return DensePointerHoistContext(entries, traversal)


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
    context = _context(("Input_val", "float"))
    spec = DensePointerHoistPassSpec(context)
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    result = LLIRPassManager().run_dense_pointer_hoist(artifact, spec)
    record = result.run_records[0]

    assert DENSE_POINTER_HOIST_PASS == LLIRPassDescriptor(
        name="hoist_dense_pointers",
        version=1,
        input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        output_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        context_type=LLIRPassContextType.DENSE_POINTER_HOIST,
    )
    assert record.pass_name == "hoist_dense_pointers"
    assert record.pass_version == 1
    assert record.configuration_name == "dense_pointer_hoist"
    assert record.input_artifact is LLIRPassArtifactType.STATEMENT_LIST
    assert record.output_artifact is LLIRPassArtifactType.STATEMENT_LIST
    assert record.context_type is LLIRPassContextType.DENSE_POINTER_HOIST
    assert record.diagnostic_stage == "LLIR transformation"
    assert record.diagnostic_pass_name == "hoist_dense_pointers"

    frozen_updates: Tuple[Tuple[object, str, object], ...] = (
        (context, "value_array_ctypes", ()),
        (context.traversal, "pass_name", "different"),
        (DENSE_POINTER_HOIST_PASS, "version", 2),
        (spec, "descriptor", replace(DENSE_POINTER_HOIST_PASS, version=2)),
        (artifact, "statements", []),
        (result, "run_records", ()),
        (record, "duration_ns", None),
    )
    for value, field_name, replacement in frozen_updates:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, replacement)


def test_context_snapshots_mutable_mapping_ownership() -> None:
    mutable_mapping = {"Input_val": "float"}
    context = DensePointerHoistContext(tuple(mutable_mapping.items()))

    mutable_mapping["Input_val"] = "double"
    mutable_mapping["Other_val"] = "int32_t"

    assert context.value_array_ctypes == (("Input_val", "float"),)
    output = hoist_dense_pointers(
        [_activating_loop()],
        context,
    )
    assert _raw_codes(output) == [
        "const float* __restrict__ _Input_val_ptr = " "&Input_val[base * stride]"
    ]


class _EqualitySpoof:
    def __eq__(self, other: object) -> bool:
        return True


def test_runner_rejects_descriptor_equality_spoof_and_wrong_exact_types() -> None:
    manager = LLIRPassManager()
    context = _context(("Input_val", "float"))
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    invalid_calls: Tuple[Callable[[], object], ...] = (
        lambda: manager.run_dense_pointer_hoist(
            artifact,
            DensePointerHoistPassSpec(
                context,
                descriptor=cast(LLIRPassDescriptor, _EqualitySpoof()),
            ),
        ),
        lambda: manager.run_dense_pointer_hoist(
            artifact,
            DensePointerHoistPassSpec(
                context,
                descriptor=replace(DENSE_POINTER_HOIST_PASS, version=2),
            ),
        ),
        lambda: manager.run_dense_pointer_hoist(
            cast(LLIRStatementListArtifact, LLIRRewriteArtifact([])),
            DensePointerHoistPassSpec(context),
        ),
        lambda: manager.run_dense_pointer_hoist(
            artifact,
            cast(DensePointerHoistPassSpec, object()),
        ),
        lambda: manager.run_dense_pointer_hoist(
            artifact,
            DensePointerHoistPassSpec(cast(DensePointerHoistContext, object())),
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
        hoist_dense_pointers(
            cast(List[llir.Stmt], root),
            _context(("Input_val", "float")),
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unsupported_dense_pointer_hoist_root"
    assert diagnostic.path == ("root",)
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_dense_pointers"
    assert diagnostic.node_type == type(root).__name__


def test_nested_root_sequences_are_validated_detached_and_semantically_omitted() -> (
    None
):
    nested_loop = _activating_loop()
    source = [[nested_loop], (_activating_loop(),)]
    before = _snapshot(source)

    output = hoist_dense_pointers(
        cast(List[llir.Stmt], source),
        _context(("Input_val", "float")),
    )

    assert type(output) is list
    assert type(output[0]) is list
    assert type(output[1]) is tuple
    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


@pytest.mark.parametrize("c_type", ("float", "double", "int32_t", "scalar_t"))
def test_core_transformation_accepts_arbitrary_names_and_c_type_spellings(
    c_type: str,
) -> None:
    source = [
        _activating_loop(
            value_array="UnrelatedTensor_val",
            position="pUnrelated7",
            base="parent_position",
            stride="Unrelated7_size",
            loop_variable="coordinate_lane",
        )
    ]
    source_ids = _mutable_ir_ids(source)

    output = hoist_dense_pointers(
        source,
        _context(("UnrelatedTensor_val", c_type)),
    )

    assert source_ids.isdisjoint(_mutable_ir_ids(output))
    assert type(output) is list
    assert len(output) == 2
    declaration = cast(llir.RawStmt, output[0])
    assert declaration.code == (
        f"const {c_type}* __restrict__ _UnrelatedTensor_val_ptr = "
        "&UnrelatedTensor_val[parent_position * Unrelated7_size]"
    )
    assert declaration.add_semicolon is True
    loop = cast(llir.ForLoop, output[1])
    assert len(loop.body) == 1
    assignment = cast(llir.Assign, loop.body[0])
    assert cast(llir.Var, assignment.value).name == (
        "_UnrelatedTensor_val_ptr[coordinate_lane]"
    )
    assert len(cast(llir.ForLoop, source[0]).body) == 2


def test_every_legal_noop_is_detached_and_preserves_all_fields_and_metadata() -> None:
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
                    "Input_val[unrelated_position]",
                    llir.DataType.PTR_FLOAT32,
                    is_ptr=True,
                    is_restrict=True,
                    tensor_access=metadata,
                ),
            )
        ]
    )
    loop.omp_parallel_for = True
    loop.omp_schedule = "dynamic, 7"
    loop.unroll = True
    loop.simd = True
    loop.omp_num_threads = "threads"
    loop.omp_chunk_expr = "chunk"
    loop.scorch_index_var = "logical"
    loop.before_parallel_body = [llir.RawStmt("before")]
    loop.pre_parallel_body = [llir.RawStmt("pre")]
    loop.post_parallel_body = [llir.RawStmt("post")]
    setattr(loop, "_use_atomic_scheduling", True)
    setattr(loop, "_atomic_chunk_var", "atomic_chunk")
    setattr(loop, "_atomic_counter_var", "atomic_counter")
    setattr(loop, "_loop_bound", "bound")
    source: List[llir.Stmt] = [loop]
    before = _snapshot(source)

    output = hoist_dense_pointers(source, _context(("Input_val", "float")))

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))
    output_loop = cast(llir.ForLoop, output[0])
    output_value = cast(llir.Var, cast(llir.Assign, output_loop.body[0]).value)
    assert output_value.tensor_access == metadata
    assert output_value.tensor_access is metadata


@pytest.mark.parametrize(
    "loop_factory",
    (
        lambda: _loop(
            [_position_init(), llir.Assign(_var("out"), _var("Input_val[position]"))],
            update=llir.Assign(_var("lane"), llir.Literal(1)),
        ),
        lambda: _loop(
            [
                llir.VarInit(
                    _var("position"),
                    llir.BinOp(
                        "+",
                        llir.BinOp("*", _var("base"), _var("stride")),
                        _var("lane"),
                    ),
                ),
                llir.Assign(_var("out"), _var("Input_val[position]")),
            ]
        ),
        lambda: _loop(
            [
                llir.VarInit(
                    _var("position"),
                    llir.Add(
                        _var("lane"),
                        llir.BinOp("*", _var("base"), _var("stride")),
                    ),
                ),
                llir.Assign(_var("out"), _var("Input_val[position]")),
            ]
        ),
        lambda: _loop(
            [
                llir.VarInit(
                    _var("position"),
                    llir.Add(
                        llir.BinOp("*", llir.Literal(1), _var("stride")),
                        _var("lane"),
                    ),
                ),
                llir.Assign(_var("out"), _var("Input_val[position]")),
            ]
        ),
    ),
)
def test_exact_affine_shape_and_increment_update_are_required(
    loop_factory: Callable[[], llir.ForLoop],
) -> None:
    source = [loop_factory()]
    before = _snapshot(source)

    output = hoist_dense_pointers(source, _context(("Input_val", "float")))

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_typed_array_access_activates_and_preserves_structured_provenance() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(21),
        tensor_id=SymbolId(22),
        index_ids=(IndexId(23),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    typed_access = llir.ArrayAccess(
        _var("Input_val"),
        _var("position"),
        tensor_access=metadata,
    )
    loop = _loop(
        [
            _position_init(),
            llir.VarInit(_var("from_initializer"), _var("Input_val[position]")),
            llir.Assign(_var("output"), typed_access),
        ]
    )
    source = [loop]
    before = _snapshot(source)

    output = hoist_dense_pointers(source, _context(("Input_val", "float")))

    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))
    declaration = cast(llir.RawStmt, output[0])
    assert declaration.code == (
        "const float* __restrict__ _Input_val_ptr = " "&Input_val[base * stride]"
    )
    rewritten_loop = cast(llir.ForLoop, output[1])
    initializer = cast(llir.VarInit, rewritten_loop.body[0])
    assert cast(llir.Var, initializer.value).name == "_Input_val_ptr[lane]"
    assignment = cast(llir.Assign, rewritten_loop.body[1])
    access = cast(llir.ArrayAccess, assignment.value)
    assert cast(llir.Var, access.array).name == "_Input_val_ptr"
    assert cast(llir.Var, access.index).name == "lane"
    assert access.tensor_access is metadata
    assert access is not typed_access


def test_add_rewrite_is_rebuilt_detached_repeatable_and_caller_owned() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(24),
        tensor_id=SymbolId(25),
        index_ids=(IndexId(26),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    value = llir.Add(
        _var(
            "Input_val[position]",
            llir.DataType.PTR_FLOAT32,
            is_ptr=True,
            is_restrict=True,
            tensor_access=metadata,
        ),
        llir.Literal(1, llir.DataType.INT64),
    )
    source = [
        _loop(
            [
                _position_init(),
                llir.Assign(_var("output"), value),
            ]
        )
    ]
    snapshot = _snapshot(source)
    context = _context(("Input_val", "float"))

    once = hoist_dense_pointers(source, context)
    twice = hoist_dense_pointers(once, context)

    assert _snapshot(source) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once))
    assert _mutable_ir_ids(once).isdisjoint(_mutable_ir_ids(twice))
    assert _snapshot(twice) == _snapshot(once)
    assert _raw_codes(once) == [
        "const float* __restrict__ _Input_val_ptr = " "&Input_val[base * stride]"
    ]
    rewritten = cast(
        llir.Add,
        cast(llir.Assign, cast(llir.ForLoop, once[1]).body[0]).value,
    )
    repeated = cast(
        llir.Add,
        cast(llir.Assign, cast(llir.ForLoop, twice[1]).body[0]).value,
    )
    assert type(rewritten) is llir.Add
    assert rewritten is not value
    assert rewritten == llir.Add(
        _var(
            "_Input_val_ptr[lane]",
            llir.DataType.PTR_FLOAT32,
            is_ptr=True,
            is_restrict=True,
            tensor_access=metadata,
        ),
        llir.Literal(1, llir.DataType.INT64),
    )
    assert cast(llir.Var, rewritten.left).tensor_access is metadata
    assert repeated == rewritten
    assert repeated is not rewritten
    assert repeated.left is not rewritten.left
    assert repeated.right is not rewritten.right


def test_varinit_only_value_access_remains_a_detached_legal_miss() -> None:
    source = [
        _loop(
            [
                _position_init(),
                llir.VarInit(
                    _var("from_initializer"),
                    _var("Input_val[position]"),
                ),
            ]
        )
    ]
    before = _snapshot(source)

    output = hoist_dense_pointers(source, _context(("Input_val", "float")))

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_analysis_is_direct_for_loop_postorder_but_rewrite_reaches_nested_bodies() -> (
    None
):
    inner = _activating_loop(
        value_array="Inner_val",
        position="inner_position",
        base="inner_base",
        stride="inner_stride",
        loop_variable="inner_lane",
    )
    outer = _loop(
        [
            _position_init(
                "outer_position",
                "outer_base",
                "outer_stride",
                "outer_lane",
            ),
            llir.Assign(_var("outer_out"), _var("Outer_val[outer_position]")),
            inner,
            llir.IfThenElse(
                cond=_var("condition"),
                then_body=[
                    _activating_loop(
                        value_array="Omitted_val",
                        position="omitted_position",
                        loop_variable="omitted_lane",
                    )
                ],
            ),
        ],
        loop_variable="outer_lane",
    )

    output = hoist_dense_pointers(
        [outer],
        _context(
            ("Outer_val", "float"),
            ("Inner_val", "double"),
            ("Omitted_val", "int32_t"),
        ),
    )

    assert _raw_codes(output) == [
        "const float* __restrict__ _Outer_val_ptr = "
        "&Outer_val[outer_base * outer_stride]"
    ]
    output_outer = cast(llir.ForLoop, output[1])
    assert _raw_codes(output_outer.body) == [
        "const double* __restrict__ _Inner_val_ptr = "
        "&Inner_val[inner_base * inner_stride]"
    ]
    output_inner = cast(llir.ForLoop, output_outer.body[2])
    assert cast(llir.Var, cast(llir.Assign, output_inner.body[0]).value).name == (
        "_Inner_val_ptr[inner_lane]"
    )
    conditional = cast(llir.IfThenElse, output_outer.body[3])
    omitted = cast(llir.ForLoop, cast(List[llir.Stmt], conditional.then_body)[0])
    assert len(omitted.body) == 2
    assert not hasattr(omitted, "_hoisted_ptr_decls")


def test_assign_scan_is_value_then_var_and_later_matches_win() -> None:
    loop = _loop(
        [
            _position_init(),
            llir.Assign(
                _var("first_target"),
                llir.BinOp(
                    "+",
                    _var("First_val[position]"),
                    _var("Second_val[position]"),
                ),
            ),
            llir.Assign(
                _access("Target_val", "position"),
                _var("LastValue_val[position]"),
            ),
        ]
    )
    output = hoist_dense_pointers(
        [loop],
        _context(
            ("First_val", "first_t"),
            ("Second_val", "second_t"),
            ("LastValue_val", "last_t"),
            ("Target_val", "target_t"),
        ),
    )

    assert _raw_codes(output) == [
        "const target_t* __restrict__ _Target_val_ptr = " "&Target_val[base * stride]"
    ]
    output_loop = cast(llir.ForLoop, output[1])
    final_assignment = cast(llir.Assign, output_loop.body[1])
    final_target = cast(llir.ArrayAccess, final_assignment.var)
    assert cast(llir.Var, final_target.array).name == "_Target_val_ptr"
    assert cast(llir.Var, final_target.index).name == "lane"
    assert cast(llir.Var, final_assignment.value).name == ("LastValue_val[position]")


def test_multiple_candidates_reverse_declaration_order_without_deduplication() -> None:
    loop = _loop(
        [
            _position_init("first_position", "first_base", "first_stride"),
            _position_init("second_position", "second_base", "second_stride"),
            llir.Assign(_var("out1"), _var("Shared_val[first_position]")),
            llir.Assign(_var("out2"), _var("Shared_val[second_position]")),
        ]
    )

    output = hoist_dense_pointers(
        [loop],
        _context(("Shared_val", "float")),
    )

    assert _raw_codes(output) == [
        "const float* __restrict__ _Shared_val_ptr = "
        "&Shared_val[second_base * second_stride]",
        "const float* __restrict__ _Shared_val_ptr = "
        "&Shared_val[first_base * first_stride]",
    ]
    output_loop = cast(llir.ForLoop, output[2])
    assert len(output_loop.body) == 2
    assert [
        cast(llir.Var, cast(llir.Assign, statement).value).name
        for statement in output_loop.body
    ] == ["_Shared_val_ptr[lane]", "_Shared_val_ptr[lane]"]


def test_missing_type_mapping_keeps_only_that_position_initializer() -> None:
    loop = _loop(
        [
            _position_init("mapped_position", "mapped_base", "mapped_stride"),
            _position_init("missing_position", "missing_base", "missing_stride"),
            llir.Assign(_var("out1"), _var("Mapped_val[mapped_position]")),
            llir.Assign(_var("out2"), _var("Missing_val[missing_position]")),
        ]
    )

    output = hoist_dense_pointers(
        [loop],
        _context(("Mapped_val", "float")),
    )

    output_loop = cast(llir.ForLoop, output[1])
    assert type(output_loop.body[0]) is llir.VarInit
    assert cast(llir.VarInit, output_loop.body[0]).var.name == "missing_position"
    assert cast(llir.Var, cast(llir.Assign, output_loop.body[1]).value).name == (
        "_Mapped_val_ptr[lane]"
    )
    assert cast(llir.Var, cast(llir.Assign, output_loop.body[2]).value).name == (
        "Missing_val[missing_position]"
    )


def test_substring_rewrite_preserves_every_var_field_and_tensor_metadata() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(31),
        tensor_id=SymbolId(32),
        index_ids=(IndexId(33),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    decorated = _var(
        "prefix_Arbitrary_val[position]_suffix",
        llir.DataType.PTR_FLOAT64,
        is_ptr=True,
        is_restrict=True,
        tensor_access=metadata,
    )
    loop = _loop(
        [
            _position_init(),
            llir.Assign(_var("discover"), _var("Arbitrary_val[position]")),
            llir.Assign(_var("output"), decorated),
        ]
    )

    output = hoist_dense_pointers(
        [loop],
        _context(("Arbitrary_val", "double")),
    )

    output_loop = cast(llir.ForLoop, output[1])
    rewritten = cast(llir.Var, cast(llir.Assign, output_loop.body[1]).value)
    assert rewritten.name == "prefix__Arbitrary_val_ptr[lane]_suffix"
    assert rewritten.type is llir.DataType.PTR_FLOAT64
    assert rewritten.is_ptr is True
    assert rewritten.is_restrict is True
    assert rewritten.tensor_access == metadata


def test_positive_rewrite_scope_covers_calls_raw_nested_loops_and_all_if_bodies() -> (
    None
):
    old = "Input_val[position]"
    nested = _loop(
        [llir.Assign(_access("Input_val", "position"), _var(old))],
        loop_variable="nested_lane",
    )
    conditional = llir.IfThenElse(
        cond=_var("condition"),
        then_body=[llir.Assign(_var("then"), _var(old))],
        else_body=[llir.RawStmt(f"consume({old})")],
        cond_list=[_var("condition1"), _var("condition2")],
        then_body_list=[
            [llir.VarInit(_var("branch0"), _var(old))],
            [llir.FunctionCallStmt(f"invoke_{old}", [_var(old)])],
        ],
    )
    loop = _loop(
        [
            _position_init(),
            llir.Assign(_var("discover"), _var(old)),
            llir.VarInit(_var("initialized"), _var(old)),
            llir.FunctionCallStmt(f"call_{old}", [_var(old)]),
            llir.RawStmt(f"raw({old})"),
            nested,
            conditional,
        ]
    )

    output = hoist_dense_pointers(
        [loop],
        _context(("Input_val", "float")),
    )

    rewritten = "_Input_val_ptr[lane]"
    output_loop = cast(llir.ForLoop, output[1])
    assert cast(llir.Var, cast(llir.Assign, output_loop.body[0]).value).name == (
        rewritten
    )
    assert cast(llir.Var, cast(llir.VarInit, output_loop.body[1]).value).name == (
        rewritten
    )
    call = cast(llir.FunctionCallStmt, output_loop.body[2])
    assert call.name == f"call_{rewritten}"
    assert cast(llir.Var, call.args[0]).name == rewritten
    assert cast(llir.RawStmt, output_loop.body[3]).code == f"raw({rewritten})"
    nested_output = cast(llir.ForLoop, output_loop.body[4])
    nested_assignment = cast(llir.Assign, nested_output.body[0])
    nested_target = cast(llir.ArrayAccess, nested_assignment.var)
    assert cast(llir.Var, nested_target.array).name == "_Input_val_ptr"
    assert cast(llir.Var, nested_target.index).name == "lane"
    assert cast(llir.Var, nested_assignment.value).name == rewritten
    output_if = cast(llir.IfThenElse, output_loop.body[5])
    assert (
        cast(
            llir.Var,
            cast(llir.Assign, cast(List[llir.Stmt], output_if.then_body)[0]).value,
        ).name
        == rewritten
    )
    assert cast(llir.RawStmt, cast(List[llir.Stmt], output_if.else_body)[0]).code == (
        f"consume({rewritten})"
    )
    branches = cast(List[List[llir.Stmt]], output_if.then_body_list)
    assert cast(llir.Var, cast(llir.VarInit, branches[0][0]).value).name == rewritten
    branch_call = cast(llir.FunctionCallStmt, branches[1][0])
    assert branch_call.name == f"invoke_{rewritten}"
    assert cast(llir.Var, branch_call.args[0]).name == rewritten


def test_function_call_rewrite_preserves_tuple_loop_body() -> None:
    old = "Input_val[position]"
    loop = _loop([])
    loop.body = cast(
        List[llir.Stmt],
        (
            _position_init(),
            llir.Assign(_var("discover"), _var(old)),
            llir.FunctionCallStmt(f"consume_{old}", [_var(old)]),
        ),
    )
    source = [loop]
    before = _snapshot(source)

    output = hoist_dense_pointers(source, _context(("Input_val", "float")))

    output_loop = cast(llir.ForLoop, output[1])
    assert type(output_loop.body) is tuple
    rewritten_call = cast(llir.FunctionCallStmt, output_loop.body[1])
    assert rewritten_call.name == "consume__Input_val_ptr[lane]"
    assert cast(llir.Var, rewritten_call.args[0]).name == "_Input_val_ptr[lane]"
    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_member_call_rewrite_owns_receiver_and_arguments() -> None:
    old = "Input_val[position]"
    call = llir.MemberCallStmt(
        base=_access("Input_val", "position"),
        member="consume",
        template_args=(llir.DataType.FLOAT32,),
        args=(_var(old), _access("Input_val", "position")),
    )
    loop = _loop(
        [
            _position_init(),
            llir.Assign(_var("discover"), _var(old)),
            call,
        ]
    )
    source = [loop]
    before = _snapshot(source)

    output = hoist_dense_pointers(
        source,
        _context(("Input_val", "float")),
    )

    output_loop = cast(llir.ForLoop, output[1])
    rewritten = cast(llir.MemberCallStmt, output_loop.body[1])
    receiver = cast(llir.ArrayAccess, rewritten.base)
    assert cast(llir.Var, receiver.array).name == "_Input_val_ptr"
    assert cast(llir.Var, receiver.index).name == "lane"
    assert rewritten.member == "consume"
    assert rewritten.template_args == (llir.DataType.FLOAT32,)
    assert type(rewritten.args) is tuple
    assert cast(llir.Var, rewritten.args[0]).name == "_Input_val_ptr[lane]"
    argument = cast(llir.ArrayAccess, rewritten.args[1])
    assert cast(llir.Var, argument.array).name == "_Input_val_ptr"
    assert cast(llir.Var, argument.index).name == "lane"
    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_legacy_rewrite_omits_headers_parallel_regions_and_nested_containers() -> None:
    old = "Input_val[position]"
    nested_header = _loop([], loop_variable="nested")
    nested_header.init = llir.VarInit(_var("nested"), _var(old))
    nested_header.cond = llir.BinOp("<", _var(old), _var("limit"))
    nested_header.update = llir.Assign(
        _access("Input_val", "position"), llir.Literal(1)
    )
    nested_header.before_parallel_body = [llir.RawStmt(old)]
    nested_header.pre_parallel_body = [llir.RawStmt(old)]
    nested_header.post_parallel_body = [llir.RawStmt(old)]
    while_loop = llir.WhileLoop(_var("condition"), [llir.RawStmt(old)])
    auto_loop = llir.ForLoopAuto(_var("item"), _var("items"), [llir.RawStmt(old)])
    function = llir.Function(
        llir.DataType.VOID,
        "omitted",
        [],
        [llir.RawStmt(old)],
    )
    conditional = llir.IfThenElse(
        cond=_var(old),
        then_body=[llir.BlankLine()],
        cond_list=[_var(old)],
        then_body_list=[[llir.BlankLine()]],
    )
    loop = _loop(
        [
            _position_init(),
            llir.Assign(_var("discover"), _var(old)),
            nested_header,
            while_loop,
            auto_loop,
            function,
            conditional,
            llir.Assign(_var("cast"), llir.Cast(_var(old), llir.DataType.FLOAT32)),
            llir.Assign(_var("unary"), llir.UnaryOp("-", _var(old))),
            llir.Assign(_var("call"), llir.FunctionCall("identity", [_var(old)])),
            llir.Return(_var(old)),
        ]
    )
    loop.before_parallel_body = [llir.RawStmt(old)]
    loop.pre_parallel_body = [llir.RawStmt(old)]
    loop.post_parallel_body = [llir.RawStmt(old)]

    output = hoist_dense_pointers(
        [loop],
        _context(("Input_val", "float")),
    )

    output_loop = cast(llir.ForLoop, output[1])
    assert cast(llir.RawStmt, output_loop.before_parallel_body[0]).code == old
    assert cast(llir.RawStmt, output_loop.pre_parallel_body[0]).code == old
    assert cast(llir.RawStmt, output_loop.post_parallel_body[0]).code == old
    header = cast(llir.ForLoop, output_loop.body[1])
    assert cast(llir.Var, cast(llir.VarInit, header.init).value).name == old
    assert cast(llir.Var, cast(llir.BinOp, header.cond).left).name == old
    header_target = cast(llir.ArrayAccess, cast(llir.Assign, header.update).var)
    assert cast(llir.Var, header_target.array).name == "Input_val"
    assert cast(llir.Var, header_target.index).name == "position"
    assert cast(llir.RawStmt, header.before_parallel_body[0]).code == old
    assert (
        cast(llir.RawStmt, cast(llir.WhileLoop, output_loop.body[2]).body[0]).code
        == old
    )
    assert (
        cast(llir.RawStmt, cast(llir.ForLoopAuto, output_loop.body[3]).body[0]).code
        == old
    )
    assert (
        cast(llir.RawStmt, cast(llir.Function, output_loop.body[4]).body[0]).code == old
    )
    output_if = cast(llir.IfThenElse, output_loop.body[5])
    assert cast(llir.Var, output_if.cond).name == old
    assert cast(llir.Var, output_if.cond_list[0]).name == old
    assert (
        cast(
            llir.Var, cast(llir.Cast, cast(llir.Assign, output_loop.body[6]).value).expr
        ).name
        == old
    )
    assert (
        cast(
            llir.Var,
            cast(llir.UnaryOp, cast(llir.Assign, output_loop.body[7]).value).operand,
        ).name
        == old
    )
    assert (
        cast(
            llir.Var,
            cast(llir.FunctionCall, cast(llir.Assign, output_loop.body[8]).value).args[
                0
            ],
        ).name
        == old
    )
    assert cast(llir.Var, cast(llir.Return, output_loop.body[9]).value).name == old


def test_preexisting_nonempty_hoisted_declarations_are_consumed_in_reverse() -> None:
    loop = _loop([])
    setattr(
        loop,
        "_hoisted_ptr_decls",
        [llir.RawStmt("first"), llir.RawStmt("second")],
    )
    source = [loop]

    output = hoist_dense_pointers(source, _context())

    assert _raw_codes(output) == ["second", "first"]
    assert type(output[2]) is llir.ForLoop
    assert not hasattr(output[2], "_hoisted_ptr_decls")
    assert hasattr(source[0], "_hoisted_ptr_decls")
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_empty_preexisting_declarations_remain_detached() -> None:
    loop = _loop([])
    declarations: List[llir.Stmt] = []
    setattr(loop, "_hoisted_ptr_decls", declarations)

    output = hoist_dense_pointers([loop], _context())

    output_loop = cast(llir.ForLoop, output[0])
    assert hasattr(output_loop, "_hoisted_ptr_decls")
    output_declarations = getattr(output_loop, "_hoisted_ptr_decls")
    assert output_declarations == []
    assert output_declarations is not declarations


def test_new_match_overwrites_preexisting_declarations() -> None:
    loop = _activating_loop()
    setattr(loop, "_hoisted_ptr_decls", [llir.RawStmt("discarded")])

    output = hoist_dense_pointers(
        [loop],
        _context(("Input_val", "float")),
    )

    assert _raw_codes(output) == [
        "const float* __restrict__ _Input_val_ptr = " "&Input_val[base * stride]"
    ]
    assert not hasattr(output[1], "_hoisted_ptr_decls")


def test_second_application_is_structurally_idempotent_but_fully_detached() -> None:
    first = hoist_dense_pointers(
        [_activating_loop()],
        _context(("Input_val", "float")),
    )
    second = hoist_dense_pointers(
        first,
        _context(("Input_val", "float")),
    )

    assert _snapshot(second) == _snapshot(first)
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))


class _UnknownStatement(llir.Stmt):
    pass


class _UnknownExpression(llir.Expr):
    pass


@pytest.mark.parametrize(
    "source, expected_path",
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
def test_unknown_nodes_fail_closed_even_inside_semantically_omitted_containers(
    source: List[llir.Stmt],
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        hoist_dense_pointers(source, _context())

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.path == expected_path
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_dense_pointers"
    assert diagnostic.node_type in {"_UnknownStatement", "_UnknownExpression"}


def test_malformed_child_fails_with_common_traversal_diagnostic() -> None:
    malformed = llir.WhileLoop(
        _var("condition"),
        cast(List[llir.Stmt], [object()]),
    )

    with pytest.raises(LLIRTraversalError) as raised:
        hoist_dense_pointers([malformed], _context())

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_statement_sequence_member"
    assert diagnostic.path == ("root", "[0]", "body", "[0]")
    assert diagnostic.node_type == "object"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_dense_pointers"


@pytest.mark.parametrize(
    "context",
    (
        object(),
        DensePointerHoistContext(cast(Tuple[Tuple[str, str], ...], [])),
        DensePointerHoistContext(cast(Tuple[Tuple[str, str], ...], (("only",),))),
        DensePointerHoistContext(cast(Tuple[Tuple[str, str], ...], ((1, "float"),))),
        DensePointerHoistContext((("Input_val", "float"), ("Input_val", "double"))),
        DensePointerHoistContext(
            (),
            traversal=cast(LLIRTraversalContext, object()),
        ),
        DensePointerHoistContext(
            (),
            traversal=LLIRTraversalContext("", "hoist_dense_pointers"),
        ),
    ),
)
def test_invalid_context_and_type_map_entries_fail_closed(context: object) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        hoist_dense_pointers(
            [llir.BlankLine()],
            cast(DensePointerHoistContext, context),
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code.startswith(("invalid_", "duplicate_"))
    assert diagnostic.path[0] == "context"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_dense_pointers"


def test_malformed_consumed_scalar_field_uses_pass_diagnostic() -> None:
    loop = _activating_loop()
    cast(llir.Increment, loop.update).var.name = cast(str, 7)

    with pytest.raises(LLIRTraversalError) as raised:
        hoist_dense_pointers(
            [loop],
            _context(("Input_val", "float")),
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_dense_pointer_hoist_var_name"
    assert diagnostic.path == ("root", "[0]", "update", "var", "name")
    assert diagnostic.node_type == "int"


def test_production_skips_extra_manager_walks_and_debug_verifies_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walks: List[Tuple[str, str]] = []

    class RecordingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            walks.append((self.context.stage, self.context.pass_name))
            super().walk(value)

    monkeypatch.setattr(pass_manager_module, "LLIRWalker", RecordingWalker)
    artifact = LLIRStatementListArtifact([_activating_loop()])
    spec = DensePointerHoistPassSpec(_context(("Input_val", "float")))

    production = LLIRPassManager(PRODUCTION_LLIR_PASS_OPTIONS)
    production_result = production.run_dense_pointer_hoist(artifact, spec)
    assert walks == []
    assert production_result.run_records[0].verified_before is False
    assert production_result.run_records[0].verified_after is False
    assert _mutable_ir_ids(artifact.statements).isdisjoint(
        _mutable_ir_ids(production_result.artifact.statements)
    )

    debug = LLIRPassManager(DEBUG_LLIR_PASS_OPTIONS)
    debug_result = debug.run_dense_pointer_hoist(artifact, spec)
    assert walks == [
        ("LLIR transformation", "hoist_dense_pointers"),
        ("LLIR transformation", "hoist_dense_pointers"),
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
            code="synthetic_dense_failure",
            message="dense pass failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name="hoist_dense_pointers",
        )
    )

    def fail_once(
        statements: List[llir.Stmt],
        context: DensePointerHoistContext,
    ) -> NoReturn:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(pass_manager_module, "hoist_dense_pointers", fail_once)
    manager = LLIRPassManager()
    with pytest.raises(LLIRTraversalError) as raised:
        manager.run_dense_pointer_hoist(
            LLIRStatementListArtifact([llir.BlankLine()]),
            DensePointerHoistPassSpec(_context()),
        )
    assert raised.value is failure
    assert calls == 1


def test_timing_and_run_records_are_nonsemantic_and_optional() -> None:
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    spec = DensePointerHoistPassSpec(_context())
    timed = LLIRPassManager().run_dense_pointer_hoist(artifact, spec)
    untimed = LLIRPassManager(
        LLIRPassOptions(record_timing=False)
    ).run_dense_pointer_hoist(artifact, spec)

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


def test_empty_and_dense_one_pass_incremental_plumbing_p95_is_below_one_ms() -> None:
    sample_count = 2000
    source: List[llir.Stmt] = [llir.BlankLine()]
    context = _context()
    spec = DensePointerHoistPassSpec(context)
    manager = LLIRPassManager(LLIRPassOptions(record_timing=False))

    for _ in range(100):
        manager.run_empty(LLIRRewriteArtifact(source))
        hoist_dense_pointers(source, context)
        manager.run_dense_pointer_hoist(LLIRStatementListArtifact(source), spec)

    empty_ns: List[int] = []
    incremental_ns: List[int] = []
    for sample in range(sample_count):
        empty_started = perf_counter_ns()
        manager.run_empty(LLIRRewriteArtifact(source))
        empty_ns.append(perf_counter_ns() - empty_started)

        if sample % 2:
            managed_started = perf_counter_ns()
            manager.run_dense_pointer_hoist(LLIRStatementListArtifact(source), spec)
            managed_elapsed = perf_counter_ns() - managed_started
            direct_started = perf_counter_ns()
            hoist_dense_pointers(source, context)
            direct_elapsed = perf_counter_ns() - direct_started
        else:
            direct_started = perf_counter_ns()
            hoist_dense_pointers(source, context)
            direct_elapsed = perf_counter_ns() - direct_started
            managed_started = perf_counter_ns()
            manager.run_dense_pointer_hoist(LLIRStatementListArtifact(source), spec)
            managed_elapsed = perf_counter_ns() - managed_started
        incremental_ns.append(managed_elapsed - direct_elapsed)

    assert _p95(empty_ns) <= 1_000_000
    assert _p95(incremental_ns) <= 1_000_000
