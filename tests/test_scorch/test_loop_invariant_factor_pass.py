from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from time import perf_counter_ns
from typing import Callable, List, NoReturn, Sequence, Set, Tuple, cast

import pytest

from scorch.compiler import llir  # type: ignore[import-untyped]
import scorch.compiler.loop_invariant_factor_pass as factor_pass_module  # type: ignore[import-untyped]
import scorch.compiler.llir_pass_manager as pass_manager_module  # type: ignore[import-untyped]
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.compiler.llir_pass_manager import (  # type: ignore[import-untyped]
    DEBUG_LLIR_PASS_OPTIONS,
    LOOP_INVARIANT_FACTOR_HOIST_PASS,
    PRODUCTION_LLIR_PASS_OPTIONS,
    LoopInvariantFactorHoistPassSpec,
    LLIRPassArtifactType,
    LLIRPassContextType,
    LLIRPassDescriptor,
    LLIRPassManager,
    LLIRPassManagerError,
    LLIRPassOptions,
    LLIRRewriteArtifact,
    LLIRStatementListArtifact,
    LLIRStatementListPassResult,
)
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRRewriter,
    LLIRWalker,
)
from scorch.compiler.loop_invariant_factor_pass import (  # type: ignore[import-untyped]
    LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    LOOP_INVARIANT_FACTOR_HOIST_TRAVERSAL_CONTEXT,
    LoopInvariantFactorHoistContext,
    hoist_loop_invariant_factors,
)

_BINOP_FAMILY = (llir.BinOp, llir.Add, llir.Mul)


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


def _right_associated_multiply(factors: Sequence[llir.Expr]) -> llir.Expr:
    assert len(factors) >= 2
    expression = factors[-1]
    for factor in reversed(factors[:-1]):
        expression = llir.BinOp("*", factor, expression)
    return expression


def _left_associated_multiply(factors: Sequence[llir.Expr]) -> llir.Expr:
    assert len(factors) >= 2
    expression = factors[0]
    for factor in factors[1:]:
        expression = llir.BinOp("*", expression, factor)
    return expression


def _assignment(
    factors: Sequence[llir.Expr],
    *,
    target: llir.Expr | None = None,
    op: llir.AssignOp = llir.AssignOp.ADD_ASSIGN,
    right_associated: bool = False,
) -> llir.Assign:
    if target is None:
        target = _var("_accum")
    value = (
        _right_associated_multiply(factors)
        if right_associated
        else _left_associated_multiply(factors)
    )
    return llir.Assign(target, value, op)


def _loop(
    body: List[llir.Stmt],
    *,
    loop_variable: str = "k",
    init_variable: str = "ignored_init",
    condition: llir.Expr | None = None,
    update: (
        llir.Increment | llir.VarInit | llir.FunctionCall | llir.Assign | None
    ) = None,
) -> llir.ForLoop:
    if condition is None:
        condition = llir.FunctionCall("ignored_condition", [_var("ignored_arg")])
    if update is None:
        update = llir.Increment(_var(loop_variable, llir.DataType.INT64))
    return llir.ForLoop(
        init=llir.VarInit(_var(init_variable), llir.Literal(37)),
        cond=condition,
        update=update,
        body=body,
    )


def _activating_loop(
    *,
    loop_variable: str = "k",
    accumulator: str = "_accum",
    invariant: llir.Expr | None = None,
    variant: llir.Expr | None = None,
    tail: Sequence[llir.Stmt] = (),
) -> llir.ForLoop:
    if invariant is None:
        invariant = _var("scale")
    if variant is None:
        variant = _var(f"_values_ptr[{loop_variable}]")
    return _loop(
        [
            _assignment(
                [invariant, variant],
                target=_var(accumulator),
            ),
            *tail,
        ],
        loop_variable=loop_variable,
    )


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


def _compatibility_codes(statements: Sequence[object]) -> List[str]:
    """Render only legacy raw prefixes and generated factor materializations."""

    lowerer = LLIRLowerer()
    codes: List[str] = []
    for statement in statements:
        include = type(statement) is llir.RawStmt
        if type(statement) is llir.VarInit:
            initialized = cast(llir.VarInit, statement).var
            include = type(initialized) is llir.Var and initialized.name.startswith(
                "_inv_"
            )
        elif type(statement) is llir.Assign:
            assignment = cast(llir.Assign, statement)
            include = (
                assignment.op is llir.AssignOp.MUL_ASSIGN
                and type(assignment.value) is llir.Var
                and cast(llir.Var, assignment.value).name.startswith("_inv_")
            )
        if include:
            codes.append(
                lowerer.lower_llir(cast(llir.Node, statement)).removesuffix(";")
            )
    return codes


def _factor_names(expression: llir.Expr) -> List[str]:
    if type(expression) in _BINOP_FAMILY:
        binary = cast(llir.BinOp, expression)
        if binary.op == "*":
            return _factor_names(binary.left) + _factor_names(binary.right)
    if type(expression) is llir.Var:
        return [cast(llir.Var, expression).name]
    if type(expression) is llir.Literal:
        return [str(cast(llir.Literal, expression).value)]
    return [str(expression)]


def _p95(samples: List[int]) -> int:
    ordered = sorted(samples)
    return ordered[int((len(ordered) - 1) * 0.95)]


def test_stable_api_and_all_managed_carriers_are_frozen() -> None:
    context = LOOP_INVARIANT_FACTOR_HOIST_CONTEXT
    spec = LoopInvariantFactorHoistPassSpec(context)
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    result = LLIRPassManager().run_loop_invariant_factor_hoist(artifact, spec)
    record = result.run_records[0]

    assert context == LoopInvariantFactorHoistContext(
        LOOP_INVARIANT_FACTOR_HOIST_TRAVERSAL_CONTEXT
    )
    assert LOOP_INVARIANT_FACTOR_HOIST_PASS == LLIRPassDescriptor(
        name="hoist_loop_invariant_factors",
        version=1,
        input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        output_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        context_type=LLIRPassContextType.LOOP_INVARIANT_FACTOR_HOIST,
    )
    assert record.pass_name == "hoist_loop_invariant_factors"
    assert record.pass_version == 1
    assert record.configuration_name == "loop_invariant_factor_hoist"
    assert record.input_artifact is LLIRPassArtifactType.STATEMENT_LIST
    assert record.output_artifact is LLIRPassArtifactType.STATEMENT_LIST
    assert record.context_type is LLIRPassContextType.LOOP_INVARIANT_FACTOR_HOIST
    assert record.diagnostic_stage == "LLIR transformation"
    assert record.diagnostic_pass_name == "hoist_loop_invariant_factors"

    frozen_updates: Tuple[Tuple[object, str, object], ...] = (
        (context, "traversal", LLIRTraversalContext("other", "other")),
        (context.traversal, "pass_name", "different"),
        (LOOP_INVARIANT_FACTOR_HOIST_PASS, "version", 2),
        (
            spec,
            "descriptor",
            replace(LOOP_INVARIANT_FACTOR_HOIST_PASS, version=2),
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
    context = LOOP_INVARIANT_FACTOR_HOIST_CONTEXT
    invalid_calls: Tuple[Callable[[], object], ...] = (
        lambda: manager.run_loop_invariant_factor_hoist(
            artifact,
            LoopInvariantFactorHoistPassSpec(
                context,
                descriptor=cast(LLIRPassDescriptor, _EqualitySpoof()),
            ),
        ),
        lambda: manager.run_loop_invariant_factor_hoist(
            artifact,
            LoopInvariantFactorHoistPassSpec(
                context,
                descriptor=replace(LOOP_INVARIANT_FACTOR_HOIST_PASS, version=2),
            ),
        ),
        lambda: manager.run_loop_invariant_factor_hoist(
            cast(LLIRStatementListArtifact, LLIRRewriteArtifact([])),
            LoopInvariantFactorHoistPassSpec(context),
        ),
        lambda: manager.run_loop_invariant_factor_hoist(
            artifact,
            cast(LoopInvariantFactorHoistPassSpec, object()),
        ),
        lambda: manager.run_loop_invariant_factor_hoist(
            artifact,
            LoopInvariantFactorHoistPassSpec(
                cast(LoopInvariantFactorHoistContext, object())
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
        hoist_loop_invariant_factors(
            cast(List[llir.Stmt], root),
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unsupported_loop_invariant_factor_hoist_root"
    assert diagnostic.path == ("root",)
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_loop_invariant_factors"
    assert diagnostic.node_type == type(root).__name__


def test_nested_root_sequences_are_detached_and_semantically_omitted() -> None:
    source = cast(
        List[llir.Stmt],
        [[_activating_loop()], (_activating_loop(invariant=_var("other")),)],
    )
    before = _snapshot(source)

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert type(output) is list
    assert type(output[0]) is list
    assert type(output[1]) is tuple
    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_success_is_fully_detached_and_does_not_mutate_the_input() -> None:
    source = [_activating_loop()]
    before = _snapshot(source)

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _snapshot(source) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))
    assert _compatibility_codes(output) == [
        "float _inv_0 = scale",
        "_accum *= _inv_0",
    ]
    rewritten_loop = cast(llir.ForLoop, output[1])
    rewritten_assignment = cast(llir.Assign, rewritten_loop.body[0])
    assert _factor_names(rewritten_assignment.value) == ["_values_ptr[k]"]
    assert rewritten_assignment.var is not cast(llir.ForLoop, source[0]).body[0].var


def test_success_constructs_exact_owned_structural_materialization_nodes() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(41),
        tensor_id=SymbolId(42),
        index_ids=(IndexId(43),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    invariant = llir.ArrayAccess(
        _var("Mask_val", llir.DataType.PTR_FLOAT32),
        _var("pMask0", llir.DataType.INT64),
        tensor_access=metadata,
    )
    target = _var(
        "accumulator",
        llir.DataType.FLOAT64,
        is_ptr=True,
        is_restrict=True,
    )
    source = [
        _activating_loop(
            loop_variable="q",
            accumulator=target.name,
            invariant=invariant,
            variant=llir.ArrayAccess(
                _var("values", llir.DataType.PTR_FLOAT64),
                _var("q", llir.DataType.INT64),
            ),
        )
    ]
    cast(llir.Assign, source[0].body[0]).var = target
    before = _snapshot(source)

    first = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    independent = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    repeated = hoist_loop_invariant_factors(
        first,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    declaration = cast(llir.VarInit, first[0])
    rewritten_loop = cast(llir.ForLoop, first[1])
    rewritten_assignment = cast(llir.Assign, rewritten_loop.body[0])
    post = cast(llir.Assign, first[2])
    assert [type(statement) for statement in first] == [
        llir.VarInit,
        llir.ForLoop,
        llir.Assign,
    ]
    assert declaration.var == _var("_inv_0", llir.DataType.FLOAT32)
    assert declaration.var.is_ptr is False
    assert declaration.var.is_restrict is False
    assert declaration.var.tensor_access is None
    assert declaration.op == "="
    assert declaration.cast is False
    assert type(declaration.value) is llir.ArrayAccess
    assert cast(llir.ArrayAccess, declaration.value).tensor_access is metadata
    assert type(post.var) is llir.Var
    assert post.var == target
    assert cast(llir.Var, post.var).tensor_access is None
    assert post.value == declaration.var
    assert post.value is not declaration.var
    assert post.op is llir.AssignOp.MUL_ASSIGN
    assert post.cast is False
    assert post.var is not rewritten_assignment.var
    assert hash(declaration.var) == hash(cast(llir.Var, post.value))
    assert hash(declaration) == hash(declaration.var)
    assert hash(cast(llir.ArrayAccess, declaration.value)) == hash(
        cast(llir.ArrayAccess, cast(llir.VarInit, independent[0]).value)
    )
    with pytest.raises(TypeError):
        hash(post)
    assert _snapshot(source) == before
    assert _snapshot(independent) == _snapshot(first)
    assert _snapshot(repeated) == _snapshot(first)
    assert cast(llir.VarInit, independent[0]) == declaration
    assert cast(llir.Assign, independent[2]) == post
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(independent))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(repeated))
    assert all(type(statement) is not llir.RawStmt for statement in first)


def test_structured_access_factors_preserve_exact_all_coo_hoist_behavior() -> None:
    mask_metadata = llir.TensorAccessMetadata(
        access_id=AccessId(31),
        tensor_id=SymbolId(32),
        index_ids=(IndexId(33), IndexId(34)),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    query_metadata = llir.TensorAccessMetadata(
        access_id=AccessId(35),
        tensor_id=SymbolId(36),
        index_ids=(IndexId(33), IndexId(37)),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    key_metadata = llir.TensorAccessMetadata(
        access_id=AccessId(38),
        tensor_id=SymbolId(39),
        index_ids=(IndexId(34), IndexId(37)),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    mask = llir.ArrayAccess(
        _var("Mask_val", llir.DataType.PTR_FLOAT32),
        _var("pMask0"),
        tensor_access=mask_metadata,
    )
    query = llir.ArrayAccess(
        _var("_Query_val_ptr", llir.DataType.PTR_FLOAT32),
        _var("q"),
        tensor_access=query_metadata,
    )
    key = llir.ArrayAccess(
        _var("_Key_val_ptr", llir.DataType.PTR_FLOAT32),
        _var("q"),
        tensor_access=key_metadata,
    )
    source = [_loop([_assignment([mask, query, key])], loop_variable="q")]
    before = _snapshot(source)

    first = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    second = hoist_loop_invariant_factors(
        first,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _snapshot(source) == before
    assert _snapshot(second) == _snapshot(first)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert _compatibility_codes(first) == [
        "float _inv_0 = Mask_val[pMask0]",
        "_accum *= _inv_0",
    ]
    assignment = cast(llir.Assign, cast(llir.ForLoop, first[1]).body[0])
    product = cast(llir.BinOp, assignment.value)
    rewritten_query = cast(llir.ArrayAccess, product.left)
    rewritten_key = cast(llir.ArrayAccess, product.right)
    assert cast(llir.Var, rewritten_query.index).name == "q"
    assert cast(llir.Var, rewritten_key.index).name == "q"
    assert rewritten_query.tensor_access is query_metadata
    assert rewritten_key.tensor_access is key_metadata


def test_legal_noop_preserves_fields_metadata_raw_settings_and_compatibility() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(11),
        tensor_id=SymbolId(12),
        index_ids=(IndexId(13), IndexId(14)),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    decorated = _var(
        "Input_val[position]",
        llir.DataType.PTR_FLOAT64,
        is_ptr=True,
        is_restrict=True,
        tensor_access=metadata,
    )
    assignment = llir.Assign(_var("output"), decorated)
    assignment.cast = True
    loop = _loop(
        [
            assignment,
            llir.RawStmt("raw_without_semicolon", add_semicolon=False),
        ],
        update=llir.Assign(_var("k"), llir.Literal(2)),
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
    source: List[llir.Stmt] = [loop]
    before = _snapshot(source)

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))
    output_loop = cast(llir.ForLoop, output[0])
    output_assignment = cast(llir.Assign, output_loop.body[0])
    output_value = cast(llir.Var, output_assignment.value)
    assert output_assignment.cast is True
    assert output_value.tensor_access == metadata
    assert output_value.tensor_access is metadata
    assert output_value.type is llir.DataType.PTR_FLOAT64
    assert output_value.is_ptr is True
    assert output_value.is_restrict is True
    assert cast(llir.RawStmt, output_loop.body[1]).add_semicolon is False
    assert (
        cast(llir.RawStmt, output_loop.before_parallel_body[0]).add_semicolon is False
    )
    assert getattr(output_loop, "_hoisted_ptr_decls") is not getattr(
        loop,
        "_hoisted_ptr_decls",
    )


@pytest.mark.parametrize(
    ("loop_variable", "accumulator", "invariant", "variant"),
    (
        ("iteration", "sum", "outer", "dense[iteration]"),
        ("q9", "aggregate_value", "Mask_val[pMask0]", "_Query_val_ptr[q9]"),
        ("$lane", "accumulator.with.punctuation", "factor-name", "values[$lane]"),
    ),
)
def test_arbitrary_loop_accumulator_and_factor_names(
    loop_variable: str,
    accumulator: str,
    invariant: str,
    variant: str,
) -> None:
    output = hoist_loop_invariant_factors(
        [
            _activating_loop(
                loop_variable=loop_variable,
                accumulator=accumulator,
                invariant=_var(invariant),
                variant=_var(variant),
            )
        ],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        f"float _inv_0 = {invariant}",
        f"{accumulator} *= _inv_0",
    ]
    assignment = cast(llir.Assign, cast(llir.ForLoop, output[1]).body[0])
    assert _factor_names(assignment.value) == [variant]


def test_loop_identity_comes_only_from_increment_and_init_condition_are_ignored() -> (
    None
):
    loop = _loop(
        [
            _assignment(
                [_var("outside"), _var("value[actual_driver]")],
                target=_var("total"),
            )
        ],
        loop_variable="actual_driver",
        init_variable="different_initializer",
        condition=llir.FunctionCall("unrelated_predicate", [_var("not_a_bound")]),
    )

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = outside",
        "total *= _inv_0",
    ]
    assignment = cast(llir.Assign, cast(llir.ForLoop, output[1]).body[0])
    assert _factor_names(assignment.value) == ["value[actual_driver]"]


@pytest.mark.parametrize("binary_type", (llir.BinOp, llir.Mul))
def test_every_exact_multiply_node_can_be_a_root_multiply(
    binary_type: type[llir.BinOp],
) -> None:
    value = (
        llir.BinOp("*", _var("scale"), _var("value[k]"))
        if binary_type is llir.BinOp
        else llir.Mul(_var("scale"), _var("value[k]"))
    )
    loop = _loop([llir.Assign(_var("acc"), value, llir.AssignOp.ADD_ASSIGN)])

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = scale",
        "acc *= _inv_0",
    ]


def _legal_miss_loops() -> Tuple[llir.ForLoop, ...]:
    non_root_multiply = llir.BinOp(
        "+",
        llir.BinOp("*", _var("scale"), _var("value[k]")),
        _var("offset"),
    )
    return (
        _loop(
            [
                _assignment(
                    [_var("scale"), _var("value[k]")],
                    op=llir.AssignOp.ASSIGN,
                )
            ]
        ),
        _loop([llir.Assign(_var("acc"), non_root_multiply, llir.AssignOp.ADD_ASSIGN)]),
        _loop(
            [_assignment([_var("left"), _var("right")])],
            update=llir.Assign(_var("k"), llir.Literal(2)),
        ),
        _loop(
            [
                _assignment(
                    [_var("scale"), _var("value[k]")],
                    target=llir.ArrayAccess(_var("result"), _var("index")),
                )
            ]
        ),
        _loop([_assignment([_var("outside_a"), _var("outside_b")])]),
        _loop([_assignment([_var("value[k]"), _var("other[k]")])]),
        _loop(
            [
                llir.Assign(
                    _var("acc"),
                    llir.Add(_var("scale"), _var("value[k]")),
                    llir.AssignOp.ADD_ASSIGN,
                )
            ]
        ),
        _loop([]),
    )


@pytest.mark.parametrize("loop", _legal_miss_loops())
def test_all_legal_structural_misses_are_detached_noops(loop: llir.ForLoop) -> None:
    source = [loop]
    before = _snapshot(source)

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_indexed_target_is_a_legal_detached_noop_after_candidate_gates() -> None:
    qualifying = _loop(
        [
            _assignment(
                [_var("scale"), _var("value[k]")],
                target=llir.ArrayAccess(_var("acc"), _var("index")),
            )
        ]
    )
    source = [qualifying]
    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    assert _snapshot(output) == _snapshot(source)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))

    earlier_miss = _loop(
        [
            _assignment(
                [_var("scale"), _var("value[k]")],
                target=llir.ArrayAccess(_var("acc"), _var("index")),
                op=llir.AssignOp.ASSIGN,
            )
        ]
    )
    output = hoist_loop_invariant_factors(
        [earlier_miss],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    assert _snapshot(output) == _snapshot([earlier_miss])


def test_arbitrary_rvalue_target_fails_at_the_common_boundary() -> None:
    assignment = _assignment([_var("scale"), _var("value[k]")])
    assignment.var = llir.BinOp("+", _var("left"), _var("right"))

    with pytest.raises(LLIRTraversalError) as raised:
        hoist_loop_invariant_factors(
            [_loop([assignment])],
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
        )

    assert raised.value.diagnostic.code == "invalid_assignment_target"
    assert raised.value.diagnostic.path == ("root", "[0]", "body", "[0]", "var")
    assert raised.value.diagnostic.node_type == "BinOp"


def test_flatten_partition_and_rebuild_orders_are_exact() -> None:
    factors = [
        _var("inv_a"),
        _var("variant_a[k]"),
        _var("inv_b"),
        _var("_variant_b_ptr[k]"),
        _var("inv_c"),
        _var("variant_c[k]"),
    ]
    loop = _loop(
        [
            _assignment(
                factors,
                right_associated=True,
            )
        ]
    )

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = inv_a * inv_b * inv_c",
        "_accum *= _inv_0",
    ]
    assignment = cast(llir.Assign, cast(llir.ForLoop, output[1]).body[0])
    assert _factor_names(assignment.value) == [
        "variant_a[k]",
        "_variant_b_ptr[k]",
        "variant_c[k]",
    ]
    variant_root = cast(llir.BinOp, assignment.value)
    assert type(variant_root.left) is llir.BinOp
    assert _factor_names(variant_root.left) == [
        "variant_a[k]",
        "_variant_b_ptr[k]",
    ]


def test_ptr_marker_loop_name_and_raw_substring_collisions_are_variant() -> None:
    loop = _loop(
        [
            llir.VarInit(_var("row"), llir.Literal(0)),
            _assignment(
                [
                    _var("outside"),
                    _var("_unrelated_ptr[index]"),
                    _var("arrow_value"),
                    _var("prefix_k_suffix"),
                ]
            ),
        ],
        loop_variable="k",
    )

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = outside",
        "_accum *= _inv_0",
    ]
    assignment = cast(llir.Assign, cast(llir.ForLoop, output[1]).body[1])
    assert _factor_names(assignment.value) == [
        "_unrelated_ptr[index]",
        "arrow_value",
        "prefix_k_suffix",
    ]


def test_supported_non_var_leaves_are_invariant_and_preserve_order() -> None:
    literal = llir.Literal(2.5)
    call = llir.FunctionCall("scale", [_var("argument")])
    loop = _loop(
        [
            _assignment(
                [literal, call, _var("value[k]")],
            )
        ]
    )
    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    declaration = cast(llir.VarInit, output[0])
    invariant = cast(llir.BinOp, declaration.value)
    assert type(declaration) is llir.VarInit
    assert type(invariant.left) is llir.Literal
    assert type(invariant.right) is llir.FunctionCall
    assert cast(llir.Literal, invariant.left).value == 2.5
    assert cast(llir.FunctionCall, invariant.right).name == "scale"
    assert _compatibility_codes(output) == [
        "float _inv_0 = 2.5 * scale(argument)",
        "_accum *= _inv_0",
    ]
    assignment = cast(llir.Assign, cast(llir.ForLoop, output[1]).body[0])
    assert _factor_names(assignment.value) == ["value[k]"]


def test_structured_function_call_ignores_expression_str_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llir.FunctionCall,
        "__str__",
        lambda self: f"CALL<{self.name}>",
    )
    loop = _loop(
        [_assignment([llir.FunctionCall("scale", [_var("x")]), _var("value[k]")])]
    )

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    declaration = cast(llir.VarInit, output[0])
    assert type(declaration.value) is llir.FunctionCall
    assert cast(llir.FunctionCall, declaration.value).name == "scale"
    assert _compatibility_codes(output) == [
        "float _inv_0 = scale(x)",
        "_accum *= _inv_0",
    ]


def test_declarations_after_use_and_duplicate_definitions_are_visible() -> None:
    loop = _loop(
        [
            _assignment([_var("outside"), _var("late_value"), _var("late_again")]),
            llir.VarInit(_var("late"), llir.Literal(1)),
            llir.VarInit(_var("late"), llir.Literal(2)),
        ]
    )

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = outside",
        "_accum *= _inv_0",
    ]
    output_loop = cast(llir.ForLoop, output[1])
    assignment = cast(llir.Assign, output_loop.body[0])
    assert _factor_names(assignment.value) == ["late_value", "late_again"]


def test_fixed_stack_array_declaration_marks_workspace_reads_variant() -> None:
    declaration = llir.FixedStackArrayDecl(
        name="wksp",
        element_type=llir.DataType.FLOAT32,
        extent=_var("kTile_k", llir.DataType.CONSTEXPR_INT),
        initializer=llir.Array([], llir.DataType.FLOAT32),
    )
    workspace_read = llir.ArrayAccess(
        _var("wksp", llir.DataType.PTR_FLOAT32),
        _var("column", llir.DataType.INT64),
    )
    loop = _activating_loop(
        loop_variable="q",
        invariant=llir.BinOp("*", _var("outside"), workspace_read),
        variant=_var("_values_ptr[q]"),
        tail=[declaration],
    )
    source: List[llir.Stmt] = [loop]
    source_snapshot = _snapshot(source)

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _snapshot(source) == source_snapshot
    assert _compatibility_codes(output) == [
        "float _inv_0 = outside",
        "_accum *= _inv_0",
    ]
    output_loop = cast(llir.ForLoop, output[1])
    assignment = cast(llir.Assign, output_loop.body[0])
    assert assignment.value == llir.BinOp(
        "*",
        workspace_read,
        _var("_values_ptr[q]"),
    )
    assert LLIRLowerer().lower_llir(assignment) == (
        "_accum += wksp[column] * _values_ptr[q];"
    )
    rewritten_decl = cast(llir.FixedStackArrayDecl, output_loop.body[1])
    assert type(rewritten_decl) is llir.FixedStackArrayDecl
    assert rewritten_decl == declaration
    assert rewritten_decl is not declaration
    assert rewritten_decl.extent is not declaration.extent
    assert rewritten_decl.initializer is not declaration.initializer
    assert LLIRLowerer().lower_llir(rewritten_decl) == "float wksp[kTile_k] = {};"


def test_defined_variable_analysis_recurses_through_legacy_supported_bodies() -> None:
    nested_for = _loop(
        [llir.VarInit(_var("nested_body"), llir.Literal(0))],
        init_variable="nested_init",
        loop_variable="nested_update",
    )
    nested_while = llir.WhileLoop(
        _var("while_condition"),
        [llir.VarInit(_var("while_defined"), llir.Literal(0))],
    )
    nested_if = llir.IfThenElse(
        cond=_var("if_condition"),
        then_body=[llir.VarInit(_var("then_defined"), llir.Literal(0))],
        else_body=[llir.VarInit(_var("else_defined"), llir.Literal(0))],
    )
    loop = _loop(
        [
            _assignment(
                [
                    _var("outside"),
                    _var("direct_value"),
                    _var("nested_init_value"),
                    _var("nested_body_value"),
                    _var("while_defined_value"),
                    _var("then_defined_value"),
                    _var("else_defined_value"),
                ]
            ),
            llir.VarInit(_var("direct"), llir.Literal(0)),
            nested_for,
            nested_while,
            nested_if,
        ]
    )

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = outside",
        "_accum *= _inv_0",
    ]
    output_loop = cast(llir.ForLoop, output[1])
    assignment = cast(llir.Assign, output_loop.body[0])
    assert _factor_names(assignment.value) == [
        "direct_value",
        "nested_init_value",
        "nested_body_value",
        "while_defined_value",
        "then_defined_value",
        "else_defined_value",
    ]


def test_definition_analysis_omits_every_surviving_unsupported_container() -> None:
    parallel_owner = _loop([], loop_variable="parallel_owner")
    parallel_owner.before_parallel_body = [
        llir.VarInit(_var("before_defined"), llir.Literal(0))
    ]
    parallel_owner.pre_parallel_body = [
        llir.VarInit(_var("pre_defined"), llir.Literal(0))
    ]
    parallel_owner.post_parallel_body = [
        llir.VarInit(_var("post_defined"), llir.Literal(0))
    ]
    conditional = llir.IfThenElse(
        cond_list=[
            _var("branch_condition"),
            _var("alternate_condition"),
            _var("final_condition"),
        ],
        then_body_list=[
            [llir.VarInit(_var("branch_defined"), llir.Literal(0))],
            [llir.VarInit(_var("alternate_defined"), llir.Literal(0))],
            [llir.VarInit(_var("final_defined"), llir.Literal(0))],
        ],
        make_last_case_else=True,
    )
    auto_loop = llir.ForLoopAuto(
        _var("auto_defined"),
        _var("array"),
        [llir.VarInit(_var("auto_body_defined"), llir.Literal(0))],
    )
    function = llir.Function(
        llir.DataType.VOID,
        "ignored_function",
        [],
        [llir.VarInit(_var("function_defined"), llir.Literal(0))],
    )
    omitted_names = [
        "before_defined",
        "pre_defined",
        "post_defined",
        "branch_defined",
        "auto_defined",
        "auto_body_defined",
        "function_defined",
        "alternate_defined",
        "final_defined",
        "raw_list_defined",
        "raw_tuple_defined",
    ]
    body = cast(
        List[llir.Stmt],
        [
            _assignment(
                [
                    *[_var(f"{name}_value") for name in omitted_names],
                    _var("value[k]"),
                ]
            ),
            parallel_owner,
            conditional,
            auto_loop,
            function,
            [llir.VarInit(_var("raw_list_defined"), llir.Literal(0))],
            (llir.VarInit(_var("raw_tuple_defined"), llir.Literal(0)),),
        ],
    )
    loop = _loop(body)

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    expected_invariant = _factor_names(
        cast(llir.BinOp, cast(llir.Assign, loop.body[0]).value)
    )[:-1]
    declaration = LLIRLowerer().lower_llir(cast(llir.VarInit, output[0]))
    for name in expected_invariant:
        assert name in declaration
    output_loop = cast(llir.ForLoop, output[1])
    assignment = cast(llir.Assign, output_loop.body[0])
    assert _factor_names(assignment.value) == ["value[k]"]


def _omitted_transform_roots() -> Tuple[List[llir.Stmt], ...]:
    parallel_owner = _loop([], loop_variable="parallel_owner")
    parallel_owner.before_parallel_body = [_activating_loop(invariant=_var("before"))]
    parallel_owner.pre_parallel_body = [_activating_loop(invariant=_var("pre"))]
    parallel_owner.post_parallel_body = [_activating_loop(invariant=_var("post"))]
    return (
        [llir.WhileLoop(_var("condition"), [_activating_loop()])],
        [
            llir.ForLoopAuto(
                _var("item"),
                _var("items"),
                [_activating_loop()],
            )
        ],
        [
            llir.Function(
                llir.DataType.VOID,
                "function",
                [],
                [_activating_loop()],
            )
        ],
        [
            llir.IfThenElse(
                cond_list=[_var("first_condition"), _var("final_condition")],
                then_body_list=[
                    [_activating_loop(invariant=_var("first_branch"))],
                    [_activating_loop(invariant=_var("final_branch"))],
                ],
                make_last_case_else=True,
            )
        ],
        [parallel_owner],
        cast(List[llir.Stmt], [[_activating_loop()]]),
        cast(List[llir.Stmt], [(_activating_loop(),)]),
    )


@pytest.mark.parametrize("source", _omitted_transform_roots())
def test_transform_recursion_omits_every_surviving_unsupported_container(
    source: List[llir.Stmt],
) -> None:
    before = _snapshot(source)

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _snapshot(output) == before
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))


def test_nested_for_loop_transform_is_postorder_with_independent_indices() -> None:
    inner = _activating_loop(
        loop_variable="inner",
        accumulator="inner_accum",
        invariant=_var("child_scale"),
        variant=_var("inner_value[inner]"),
    )
    outer = _loop(
        [
            llir.RawStmt("prefix"),
            inner,
            _assignment(
                [_var("top_scale"), _var("outer_value[outer]")],
                target=_var("outer_accum"),
            ),
        ],
        loop_variable="outer",
    )

    output = hoist_loop_invariant_factors(
        [outer],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = top_scale",
        "outer_accum *= _inv_0",
    ]
    output_outer = cast(llir.ForLoop, output[1])
    assert _compatibility_codes(output_outer.body) == [
        "prefix",
        "float _inv_1 = child_scale",
        "inner_accum *= _inv_1",
    ]
    assert type(output_outer.body[2]) is llir.ForLoop


def test_nested_typed_materialization_remains_invisible_to_legacy_name_analysis() -> (
    None
):
    inner = _activating_loop(
        loop_variable="inner",
        accumulator="inner_accum",
        invariant=_var("scale"),
        variant=_var("inner_value[inner]"),
    )
    outer = _loop(
        [
            inner,
            _assignment(
                [_var("_inv_0_scale"), _var("outer_value[outer]")],
                target=_var("outer_accum"),
            ),
        ],
        loop_variable="outer",
    )

    first = hoist_loop_invariant_factors(
        [outer],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    second = hoist_loop_invariant_factors(
        first,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(first) == [
        "float _inv_0 = _inv_0_scale",
        "outer_accum *= _inv_0",
    ]
    output_outer = cast(llir.ForLoop, first[1])
    assert _compatibility_codes(output_outer.body) == [
        "float _inv_0 = scale",
        "inner_accum *= _inv_0",
    ]
    assert _snapshot(second) == _snapshot(first)
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))


def test_ordinary_materialization_lookalike_remains_a_visible_definition() -> None:
    declaration = llir.VarInit(
        _var("_inv_0", llir.DataType.FLOAT32),
        _var("local_scale", llir.DataType.FLOAT32),
    )
    candidate = _assignment([_var("_inv_0", llir.DataType.FLOAT32), _var("value[k]")])
    unrelated_post = llir.Assign(
        _var("other_accumulator"),
        _var("_inv_0", llir.DataType.FLOAT32),
        llir.AssignOp.MUL_ASSIGN,
    )
    source = [_loop([declaration, candidate, unrelated_post])]
    snapshot = _snapshot(source)

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _snapshot(output) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(output))
    assert _compatibility_codes(output) == []
    output_loop = cast(llir.ForLoop, output[0])
    assert _factor_names(cast(llir.Assign, output_loop.body[1]).value) == [
        "_inv_0",
        "value[k]",
    ]


def test_nested_generated_wrapper_block_remains_invisible_after_reapplication() -> None:
    inner = _loop(
        [
            _assignment(
                [_var("first_scale"), _var("first_value[inner]")],
                target=_var("first_accumulator"),
            ),
            _assignment(
                [_var("second_scale"), _var("second_value[inner]")],
                target=_var("second_accumulator"),
            ),
        ],
        loop_variable="inner",
    )
    first = hoist_loop_invariant_factors(
        [inner],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    wrapped_inner = hoist_loop_invariant_factors(
        first,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    outer = _loop(
        [
            *wrapped_inner,
            _assignment(
                [_var("_inv_0_scale"), _var("outer_value[outer]")],
                target=_var("outer_accumulator"),
            ),
        ],
        loop_variable="outer",
    )

    output = hoist_loop_invariant_factors(
        [outer],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = _inv_0_scale",
        "outer_accumulator *= _inv_0",
    ]
    output_outer = cast(llir.ForLoop, output[1])
    assert _compatibility_codes(output_outer.body) == [
        "float _inv_0 = first_scale",
        "float _inv_1 = second_scale",
        "second_accumulator *= _inv_1",
        "first_accumulator *= _inv_0",
    ]


def test_if_then_else_transform_is_postorder_and_branches_number_independently() -> (
    None
):
    conditional = llir.IfThenElse(
        cond=_var("condition"),
        then_body=[
            llir.RawStmt("then_prefix"),
            _activating_loop(
                loop_variable="then_lane",
                accumulator="then_accum",
                invariant=_var("then_scale"),
                variant=_var("then_value[then_lane]"),
            ),
        ],
        else_body=[
            _activating_loop(
                loop_variable="else_lane",
                accumulator="else_accum",
                invariant=_var("else_scale"),
                variant=_var("else_value[else_lane]"),
            )
        ],
    )

    output = hoist_loop_invariant_factors(
        [conditional],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    output_conditional = cast(llir.IfThenElse, output[0])
    assert output_conditional.then_body is not None
    assert output_conditional.else_body is not None
    assert _compatibility_codes(output_conditional.then_body) == [
        "then_prefix",
        "float _inv_1 = then_scale",
        "then_accum *= _inv_1",
    ]
    assert _compatibility_codes(output_conditional.else_body) == [
        "float _inv_0 = else_scale",
        "else_accum *= _inv_0",
    ]


def test_typed_materialization_uses_float_mul_assign_and_canonical_precedence() -> None:
    invariant = llir.BinOp(
        "+",
        _var("left"),
        llir.Mul(_var("middle"), llir.Literal(3)),
    )
    loop = _loop(
        [
            _assignment(
                [invariant, _var("value[k]")],
                target=_var("acc"),
            )
        ]
    )

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    declaration = cast(llir.VarInit, output[0])
    post = cast(llir.Assign, output[2])
    assert type(declaration) is llir.VarInit
    assert declaration.var == _var("_inv_0", llir.DataType.FLOAT32)
    assert type(declaration.value) is llir.BinOp
    assert declaration.value == invariant
    assert declaration.op == "="
    assert declaration.cast is False
    assert type(post) is llir.Assign
    assert post.var == _var("acc")
    assert post.value == _var("_inv_0", llir.DataType.FLOAT32)
    assert post.op is llir.AssignOp.MUL_ASSIGN
    assert post.cast is False
    assert LLIRLowerer().lower_llir(declaration) == (
        "float _inv_0 = left + middle * 3;"
    )
    assert LLIRLowerer().lower_llir(post) == "acc *= _inv_0;"


def test_generated_name_uses_current_index_without_collision_checks() -> None:
    source: List[llir.Stmt] = [
        llir.RawStmt("float _inv_1 = existing"),
        _activating_loop(),
    ]

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_1 = existing",
        "float _inv_1 = scale",
        "_accum *= _inv_1",
    ]
    assert type(output[2]) is llir.ForLoop


def test_successful_siblings_shift_later_suffixes_by_two_and_scanning_continues() -> (
    None
):
    source = [
        _activating_loop(invariant=_var("first"), accumulator="first_accum"),
        _activating_loop(invariant=_var("second"), accumulator="second_accum"),
        _activating_loop(invariant=_var("third"), accumulator="third_accum"),
    ]

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = first",
        "first_accum *= _inv_0",
        "float _inv_3 = second",
        "second_accum *= _inv_3",
        "float _inv_6 = third",
        "third_accum *= _inv_6",
    ]
    assert [
        index for index, value in enumerate(output) if type(value) is llir.ForLoop
    ] == [
        1,
        4,
        7,
    ]


def test_earlier_accumulation_misses_do_not_hide_first_success() -> None:
    miss_op = _assignment(
        [_var("miss_scale"), _var("miss_value[k]")],
        target=_var("miss"),
        op=llir.AssignOp.ASSIGN,
    )
    miss_target = _assignment(
        [_var("array_scale"), _var("array_value[k]")],
        target=llir.ArrayAccess(_var("array"), _var("index")),
    )
    first = _assignment(
        [_var("first_scale"), _var("first_value[k]")],
        target=_var("first_accum"),
    )
    second = _assignment(
        [_var("second_scale"), _var("second_value[k]")],
        target=_var("second_accum"),
    )
    loop = _loop([miss_op, miss_target, first, second])

    output = hoist_loop_invariant_factors(
        [loop],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(output) == [
        "float _inv_0 = first_scale",
        "first_accum *= _inv_0",
    ]
    output_loop = cast(llir.ForLoop, output[1])
    assert _snapshot(output_loop.body[0]) == _snapshot(miss_op)
    assert _snapshot(output_loop.body[1]) == _snapshot(miss_target)
    assert _factor_names(cast(llir.Assign, output_loop.body[2]).value) == [
        "first_value[k]"
    ]
    assert _snapshot(output_loop.body[3]) == _snapshot(second)


def test_one_match_reapplication_is_structurally_stable_but_fully_detached() -> None:
    first = hoist_loop_invariant_factors(
        [_activating_loop()],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    second = hoist_loop_invariant_factors(
        first,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _snapshot(second) == _snapshot(first)
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))


def test_multiple_accumulations_transform_one_per_application() -> None:
    source = [
        _loop(
            [
                _assignment(
                    [_var("first_scale"), _var("first_value[k]")],
                    target=_var("first_accum"),
                ),
                _assignment(
                    [_var("second_scale"), _var("second_value[k]")],
                    target=_var("second_accum"),
                ),
            ]
        )
    ]

    first = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    second = hoist_loop_invariant_factors(
        first,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    third = hoist_loop_invariant_factors(
        second,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    assert _compatibility_codes(first) == [
        "float _inv_0 = first_scale",
        "first_accum *= _inv_0",
    ]
    assert _compatibility_codes(second) == [
        "float _inv_0 = first_scale",
        "float _inv_1 = second_scale",
        "second_accum *= _inv_1",
        "first_accum *= _inv_0",
    ]
    assert _snapshot(second) != _snapshot(first)
    assert _snapshot(third) == _snapshot(second)
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert _mutable_ir_ids(second).isdisjoint(_mutable_ir_ids(third))


def test_successful_replacement_preserves_target_fields_factor_metadata_and_cast() -> (
    None
):
    factor_metadata = llir.TensorAccessMetadata(
        access_id=AccessId(23),
        tensor_id=SymbolId(24),
        index_ids=(IndexId(25),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    target = _var(
        "accumulator",
        llir.DataType.FLOAT64,
        is_ptr=True,
        is_restrict=True,
    )
    variant = _var(
        "Input_val[k]",
        llir.DataType.FLOAT64,
        tensor_access=factor_metadata,
    )
    assignment = _assignment([_var("scale"), variant], target=target)
    assignment.cast = True
    source = [_loop([assignment])]

    output = hoist_loop_invariant_factors(
        source,
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )

    rewritten = cast(llir.Assign, cast(llir.ForLoop, output[1]).body[0])
    rewritten_target = cast(llir.Var, rewritten.var)
    rewritten_variant = cast(llir.Var, rewritten.value)
    assert rewritten.op is llir.AssignOp.ADD_ASSIGN
    assert rewritten.cast is True
    assert rewritten_target.name == "accumulator"
    assert rewritten_target.type is llir.DataType.FLOAT64
    assert rewritten_target.is_ptr is True
    assert rewritten_target.is_restrict is True
    assert rewritten_target.tensor_access is None
    assert rewritten_variant.tensor_access is factor_metadata
    assert rewritten_target is not target
    assert rewritten_variant is not variant


class _UnknownStatement(llir.Stmt):
    pass


class _UnknownExpression(llir.Expr):
    pass


class _UnknownBinOp(llir.BinOp):
    pass


@pytest.mark.parametrize(
    ("source", "expected_path", "expected_type"),
    (
        (
            [cast(llir.Stmt, _UnknownStatement())],
            ("root", "[0]"),
            "_UnknownStatement",
        ),
        (
            [
                llir.WhileLoop(
                    _var("condition"),
                    [cast(llir.Stmt, _UnknownStatement())],
                )
            ],
            ("root", "[0]", "body", "[0]"),
            "_UnknownStatement",
        ),
        (
            [llir.Return(cast(llir.Expr, _UnknownExpression()))],
            ("root", "[0]", "value"),
            "_UnknownExpression",
        ),
        (
            [
                _loop(
                    [
                        llir.Assign(
                            _var("acc"),
                            cast(
                                llir.Expr,
                                _UnknownBinOp(
                                    "*",
                                    _var("scale"),
                                    _var("value[k]"),
                                ),
                            ),
                            llir.AssignOp.ADD_ASSIGN,
                        )
                    ]
                )
            ],
            ("root", "[0]", "body", "[0]", "value"),
            "_UnknownBinOp",
        ),
    ),
)
def test_unknown_nodes_and_subclasses_fail_closed_even_when_semantically_omitted(
    source: List[llir.Stmt],
    expected_path: Tuple[str, ...],
    expected_type: str,
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        hoist_loop_invariant_factors(
            source,
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.path == expected_path
    assert diagnostic.node_type == expected_type
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_loop_invariant_factors"


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
        (
            [
                _loop(
                    [],
                )
            ],
            ("root", "[0]", "pre_parallel_body", "[0]"),
        ),
    ),
)
def test_malformed_children_fail_inside_semantically_omitted_containers(
    source: List[llir.Stmt],
    expected_path: Tuple[str, ...],
) -> None:
    if expected_path[-2:] == ("pre_parallel_body", "[0]"):
        cast(llir.ForLoop, source[0]).pre_parallel_body = cast(
            List[llir.Stmt], [object()]
        )

    with pytest.raises(LLIRTraversalError) as raised:
        hoist_loop_invariant_factors(
            source,
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_statement_sequence_member"
    assert diagnostic.path == expected_path
    assert diagnostic.node_type == "object"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_loop_invariant_factors"


def test_invalid_top_level_member_uses_pass_owned_root_diagnostic() -> None:
    source = cast(List[llir.Stmt], [llir.BlankLine(), object()])

    with pytest.raises(LLIRTraversalError) as raised:
        hoist_loop_invariant_factors(
            source,
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_loop_invariant_factor_hoist_root_member"
    assert diagnostic.path == ("root", "[1]")
    assert diagnostic.node_type == "object"


@pytest.mark.parametrize(
    ("context", "expected_code", "expected_path"),
    (
        (
            object(),
            "invalid_loop_invariant_factor_hoist_context",
            ("context",),
        ),
        (
            LoopInvariantFactorHoistContext(cast(LLIRTraversalContext, object())),
            "invalid_loop_invariant_factor_hoist_traversal_context",
            ("context", "traversal"),
        ),
        (
            LoopInvariantFactorHoistContext(
                LLIRTraversalContext("", "hoist_loop_invariant_factors")
            ),
            "invalid_loop_invariant_factor_hoist_traversal_context",
            ("context", "traversal"),
        ),
        (
            LoopInvariantFactorHoistContext(
                LLIRTraversalContext("LLIR transformation", "")
            ),
            "invalid_loop_invariant_factor_hoist_traversal_context",
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
        hoist_loop_invariant_factors(
            [llir.BlankLine()],
            cast(LoopInvariantFactorHoistContext, context),
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == expected_code
    assert diagnostic.path == expected_path
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_loop_invariant_factors"


class _AssignOperatorSpoof:
    value = "+="


def _malformed_update_name() -> List[llir.Stmt]:
    loop = _activating_loop()
    cast(llir.Increment, loop.update).var.name = cast(str, 7)
    return [loop]


def _malformed_defined_name() -> List[llir.Stmt]:
    loop = _activating_loop(tail=[llir.VarInit(_var("defined"), llir.Literal(0))])
    cast(llir.VarInit, loop.body[1]).var.name = cast(str, 7)
    return [loop]


def _malformed_binary_operator() -> List[llir.Stmt]:
    loop = _activating_loop()
    object.__setattr__(
        cast(llir.BinOp, cast(llir.Assign, loop.body[0]).value),
        "op",
        cast(str, 7),
    )
    return [loop]


def _malformed_assign_operator() -> List[llir.Stmt]:
    loop = _activating_loop()
    cast(llir.Assign, loop.body[0]).op = cast(
        llir.AssignOp,
        _AssignOperatorSpoof(),
    )
    return [loop]


def _malformed_target_name() -> List[llir.Stmt]:
    loop = _activating_loop()
    cast(llir.Var, cast(llir.Assign, loop.body[0]).var).name = cast(str, 7)
    return [loop]


def _malformed_factor_name() -> List[llir.Stmt]:
    loop = _activating_loop()
    value = cast(llir.BinOp, cast(llir.Assign, loop.body[0]).value)
    cast(llir.Var, value.left).name = cast(str, 7)
    return [loop]


@pytest.mark.parametrize(
    ("factory", "expected_code", "expected_path"),
    (
        (
            _malformed_update_name,
            "invalid_loop_invariant_factor_var_name",
            ("root", "[0]", "update", "var", "name"),
        ),
        (
            _malformed_defined_name,
            "invalid_loop_invariant_factor_var_name",
            ("root", "[0]", "body", "[1]", "var", "name"),
        ),
        (
            _malformed_binary_operator,
            "invalid_binary_operator",
            ("root", "[0]", "body", "[0]", "value", "op"),
        ),
        (
            _malformed_assign_operator,
            "invalid_assign_op",
            ("root", "[0]", "body", "[0]", "op"),
        ),
        (
            _malformed_target_name,
            "invalid_assignment_target",
            ("root", "[0]", "body", "[0]", "var"),
        ),
        (
            _malformed_factor_name,
            "invalid_loop_invariant_factor_var_name",
            ("root", "[0]", "body", "[0]", "value", "left", "name"),
        ),
    ),
)
def test_malformed_consumed_scalars_use_the_owning_boundary_diagnostics(
    factory: Callable[[], List[llir.Stmt]],
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        hoist_loop_invariant_factors(
            factory(),
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == expected_code
    assert diagnostic.path == expected_path
    assert diagnostic.node_type in {"int", "Var", "_AssignOperatorSpoof"}
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_loop_invariant_factors"


def test_manager_matches_direct_output_and_returns_one_ordered_record() -> None:
    source = [_activating_loop()]
    context = LOOP_INVARIANT_FACTOR_HOIST_CONTEXT
    direct = hoist_loop_invariant_factors(source, context)

    managed = LLIRPassManager().run_loop_invariant_factor_hoist(
        LLIRStatementListArtifact(source),
        LoopInvariantFactorHoistPassSpec(context),
    )

    assert _snapshot(managed.artifact.statements) == _snapshot(direct)
    assert _mutable_ir_ids(source).isdisjoint(
        _mutable_ir_ids(managed.artifact.statements)
    )
    assert len(managed.run_records) == 1
    assert managed.run_records[0].sequence_index == 0


def test_production_skips_manager_walks_and_debug_verifies_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walks: List[Tuple[str, str]] = []

    class RecordingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            walks.append((self.context.stage, self.context.pass_name))
            super().walk(value)

    monkeypatch.setattr(pass_manager_module, "LLIRWalker", RecordingWalker)
    artifact = LLIRStatementListArtifact([_activating_loop()])
    spec = LoopInvariantFactorHoistPassSpec()

    production = LLIRPassManager(PRODUCTION_LLIR_PASS_OPTIONS)
    production_result = production.run_loop_invariant_factor_hoist(artifact, spec)
    assert walks == []
    assert production_result.run_records[0].verified_before is False
    assert production_result.run_records[0].verified_after is False
    assert _mutable_ir_ids(artifact.statements).isdisjoint(
        _mutable_ir_ids(production_result.artifact.statements)
    )

    debug = LLIRPassManager(DEBUG_LLIR_PASS_OPTIONS)
    debug_result = debug.run_loop_invariant_factor_hoist(artifact, spec)
    assert walks == [
        ("LLIR transformation", "hoist_loop_invariant_factors"),
        ("LLIR transformation", "hoist_loop_invariant_factors"),
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
            code="synthetic_loop_invariant_failure",
            message="loop-invariant factor pass failed",
            path=("root",),
            node_type="ForLoop",
            stage="LLIR transformation",
            pass_name="hoist_loop_invariant_factors",
        )
    )

    def fail_once(
        statements: List[llir.Stmt],
        context: LoopInvariantFactorHoistContext,
    ) -> NoReturn:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(
        pass_manager_module,
        "hoist_loop_invariant_factors",
        fail_once,
    )
    manager = LLIRPassManager()
    with pytest.raises(LLIRTraversalError) as raised:
        manager.run_loop_invariant_factor_hoist(
            LLIRStatementListArtifact([llir.BlankLine()]),
            LoopInvariantFactorHoistPassSpec(),
        )

    assert raised.value is failure
    assert calls == 1


def test_generated_materialization_walk_order_is_deterministic() -> None:
    output = hoist_loop_invariant_factors(
        [
            _activating_loop(
                invariant=llir.ArrayAccess(
                    _var("weights", llir.DataType.PTR_FLOAT32),
                    _var("row", llir.DataType.INT64),
                )
            )
        ],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    materialization = [output[0], output[2]]

    def record_walk() -> List[Tuple[str, Tuple[str, ...]]]:
        entered: List[Tuple[str, Tuple[str, ...]]] = []

        class RecordingWalker(LLIRWalker):
            def enter_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
                entered.append((type(node).__name__, path))

        RecordingWalker(LOOP_INVARIANT_FACTOR_HOIST_TRAVERSAL_CONTEXT).walk(
            materialization
        )
        return entered

    expected = [
        ("VarInit", ("root", "[0]")),
        ("Var", ("root", "[0]", "var")),
        ("ArrayAccess", ("root", "[0]", "value")),
        ("Var", ("root", "[0]", "value", "array")),
        ("Var", ("root", "[0]", "value", "index")),
        ("Assign", ("root", "[1]")),
        ("Var", ("root", "[1]", "var")),
        ("Var", ("root", "[1]", "value")),
    ]
    assert record_walk() == expected
    assert record_walk() == expected


def test_generated_materialization_identity_and_replacement_rewrites_are_owned() -> (
    None
):
    output = hoist_loop_invariant_factors(
        [_activating_loop()],
        LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    )
    materialization = [output[0], output[2]]
    context = LOOP_INVARIANT_FACTOR_HOIST_TRAVERSAL_CONTEXT

    first = cast(List[llir.Stmt], LLIRRewriter(context).rewrite(materialization))
    repeated = cast(List[llir.Stmt], LLIRRewriter(context).rewrite(first))
    assert _snapshot(first) == _snapshot(materialization)
    assert _snapshot(repeated) == _snapshot(first)
    assert _mutable_ir_ids(materialization).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(repeated))

    replacement = llir.VarInit(
        _var("_inv_0", llir.DataType.FLOAT32),
        llir.ArrayAccess(
            _var("replacement", llir.DataType.PTR_FLOAT32),
            _var("offset", llir.DataType.INT64),
        ),
    )

    class DeclarationReplacingRewriter(LLIRRewriter):
        def rewrite_statement_sequence_member(
            self,
            node: llir.Stmt,
            path: Tuple[str, ...],
        ) -> Sequence[llir.Stmt]:
            if path == ("root", "[0]"):
                return (replacement,)
            return super().rewrite_statement_sequence_member(node, path)

    replaced = cast(
        List[llir.Stmt],
        DeclarationReplacingRewriter(context).rewrite(materialization),
    )
    replaced_again = cast(
        List[llir.Stmt],
        DeclarationReplacingRewriter(context).rewrite(materialization),
    )
    assert _snapshot(replaced[0]) == _snapshot(replacement)
    assert _snapshot(replaced_again) == _snapshot(replaced)
    assert _mutable_ir_ids(replacement).isdisjoint(_mutable_ir_ids(replaced[0]))
    assert _mutable_ir_ids(materialization).isdisjoint(_mutable_ir_ids(replaced))
    assert _mutable_ir_ids(replaced).isdisjoint(_mutable_ir_ids(replaced_again))


class _GeneratedVarInitSubclass(llir.VarInit):
    pass


class _GeneratedAssignSubclass(llir.Assign):
    pass


class _GeneratedVarSubclass(llir.Var):
    pass


class _GeneratedUnknownExpression(llir.Expr):
    pass


def _valid_generated_materialization() -> Tuple[llir.VarInit, llir.Assign]:
    return (
        llir.VarInit(
            _var("_inv_0", llir.DataType.FLOAT32),
            _var("scale", llir.DataType.FLOAT32),
        ),
        llir.Assign(
            _var("accumulator"),
            _var("_inv_0", llir.DataType.FLOAT32),
            llir.AssignOp.MUL_ASSIGN,
        ),
    )


def _forged_declaration_node() -> Tuple[object, object]:
    _, post = _valid_generated_materialization()
    return llir.RawStmt("float _inv_0 = scale"), post


def _forged_declaration_subclass() -> Tuple[object, object]:
    _, post = _valid_generated_materialization()
    return (
        _GeneratedVarInitSubclass(
            _var("_inv_0", llir.DataType.FLOAT32),
            _var("scale", llir.DataType.FLOAT32),
        ),
        post,
    )


def _forged_assignment_node() -> Tuple[object, object]:
    declaration, _ = _valid_generated_materialization()
    return declaration, llir.RawStmt("accumulator *= _inv_0")


def _forged_assignment_subclass() -> Tuple[object, object]:
    declaration, _ = _valid_generated_materialization()
    return (
        declaration,
        _GeneratedAssignSubclass(
            _var("accumulator"),
            _var("_inv_0", llir.DataType.FLOAT32),
            llir.AssignOp.MUL_ASSIGN,
        ),
    )


def _forged_declaration_var_subclass() -> Tuple[object, object]:
    _, post = _valid_generated_materialization()
    return (
        llir.VarInit(
            _GeneratedVarSubclass("_inv_0", llir.DataType.FLOAT32),
            _var("scale", llir.DataType.FLOAT32),
        ),
        post,
    )


def _forged_declaration_var_field() -> Tuple[object, object]:
    declaration, post = _valid_generated_materialization()
    declaration.var.is_ptr = cast(bool, 1)
    return declaration, post


def _forged_declaration_fields() -> Tuple[object, object]:
    declaration, post = _valid_generated_materialization()
    declaration.op = "+="
    return declaration, post


def _forged_assignment_target_field() -> Tuple[object, object]:
    declaration, post = _valid_generated_materialization()
    cast(llir.Var, post.var).type = cast(llir.DataType, object())
    return declaration, post


def _forged_assignment_target_name() -> Tuple[object, object]:
    declaration, post = _valid_generated_materialization()
    cast(llir.Var, post.var).name = "wrong_accumulator"
    return declaration, post


def _forged_assignment_reference_field() -> Tuple[object, object]:
    declaration, post = _valid_generated_materialization()
    cast(llir.Var, post.value).name = "_inv_7"
    return declaration, post


def _forged_assignment_fields() -> Tuple[object, object]:
    declaration, post = _valid_generated_materialization()
    post.op = llir.AssignOp.ADD_ASSIGN
    return declaration, post


def _forged_unknown_declaration_child() -> Tuple[object, object]:
    declaration, post = _valid_generated_materialization()
    declaration.value = cast(llir.Expr, _GeneratedUnknownExpression())
    return declaration, post


@pytest.mark.parametrize(
    ("factory", "expected_code", "expected_path", "expected_type"),
    (
        (
            _forged_declaration_node,
            "invalid_loop_invariant_factor_materialization_declaration",
            ("root", "[0]"),
            "RawStmt",
        ),
        (
            _forged_declaration_subclass,
            "invalid_loop_invariant_factor_materialization_declaration",
            ("root", "[0]"),
            "_GeneratedVarInitSubclass",
        ),
        (
            _forged_assignment_node,
            "invalid_loop_invariant_factor_materialization_assignment",
            ("root", "[2]"),
            "RawStmt",
        ),
        (
            _forged_assignment_subclass,
            "invalid_loop_invariant_factor_materialization_assignment",
            ("root", "[2]"),
            "_GeneratedAssignSubclass",
        ),
        (
            _forged_declaration_var_subclass,
            "invalid_loop_invariant_factor_materialization_var",
            ("root", "[0]", "var"),
            "_GeneratedVarSubclass",
        ),
        (
            _forged_declaration_var_field,
            "invalid_loop_invariant_factor_materialization_var",
            ("root", "[0]", "var"),
            "Var",
        ),
        (
            _forged_declaration_fields,
            "invalid_loop_invariant_factor_materialization_declaration_fields",
            ("root", "[0]"),
            "VarInit",
        ),
        (
            _forged_assignment_target_field,
            "invalid_loop_invariant_factor_materialization_var",
            ("root", "[2]", "var"),
            "Var",
        ),
        (
            _forged_assignment_target_name,
            "invalid_loop_invariant_factor_materialization_var",
            ("root", "[2]", "var"),
            "Var",
        ),
        (
            _forged_assignment_reference_field,
            "invalid_loop_invariant_factor_materialization_var",
            ("root", "[2]", "value"),
            "Var",
        ),
        (
            _forged_assignment_fields,
            "invalid_loop_invariant_factor_materialization_assignment_fields",
            ("root", "[2]"),
            "Assign",
        ),
        (
            _forged_unknown_declaration_child,
            "unknown_llir_node",
            ("root", "[0]", "value"),
            "_GeneratedUnknownExpression",
        ),
    ),
)
def test_forged_generated_materialization_fails_at_the_owning_boundary(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], Tuple[object, object]],
    expected_code: str,
    expected_path: Tuple[str, ...],
    expected_type: str,
) -> None:
    calls = 0

    def forge_once(
        loop: llir.ForLoop,
        sequence_index: int,
        context: LoopInvariantFactorHoistContext,
        path: Tuple[str, ...],
    ) -> Tuple[llir.VarInit, llir.Assign, int, llir.Expr]:
        nonlocal calls
        calls += 1
        assert sequence_index == 0
        assert path == ("root", "[0]")
        declaration, post = factory()
        invariant_expression = (
            cast(llir.VarInit, declaration).value
            if type(declaration) is llir.VarInit
            else _var("scale", llir.DataType.FLOAT32)
        )
        return cast(
            Tuple[llir.VarInit, llir.Assign, int, llir.Expr],
            (declaration, post, 0, invariant_expression),
        )

    monkeypatch.setattr(factor_pass_module, "_try_hoist_from_loop", forge_once)
    with pytest.raises(LLIRTraversalError) as raised:
        hoist_loop_invariant_factors(
            [_activating_loop(accumulator="accumulator")],
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert calls == 1
    assert diagnostic.code == expected_code
    assert diagnostic.path == expected_path
    assert diagnostic.node_type == expected_type
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_loop_invariant_factors"


def test_forged_generated_declaration_value_fails_at_the_owning_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forge_declaration_value(
        loop: llir.ForLoop,
        sequence_index: int,
        context: LoopInvariantFactorHoistContext,
        path: Tuple[str, ...],
    ) -> Tuple[llir.VarInit, llir.Assign, int, llir.Expr]:
        del loop, sequence_index, context, path
        declaration, post = _valid_generated_materialization()
        selected_invariant = declaration.value
        declaration.value = _var("wrong_scale", llir.DataType.FLOAT32)
        return declaration, post, 0, selected_invariant

    monkeypatch.setattr(
        factor_pass_module,
        "_try_hoist_from_loop",
        forge_declaration_value,
    )
    with pytest.raises(LLIRTraversalError) as raised:
        hoist_loop_invariant_factors(
            [_activating_loop(accumulator="accumulator")],
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == (
        "invalid_loop_invariant_factor_materialization_declaration_value"
    )
    assert diagnostic.path == ("root", "[0]", "value")
    assert diagnostic.node_type == "Var"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "hoist_loop_invariant_factors"


def test_managed_forged_materialization_adds_no_record_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    record_attempts: List[object] = []

    def forge_invalid_assignment(
        loop: llir.ForLoop,
        sequence_index: int,
        context: LoopInvariantFactorHoistContext,
        path: Tuple[str, ...],
    ) -> Tuple[llir.VarInit, llir.Assign, int, llir.Expr]:
        nonlocal calls
        calls += 1
        declaration, post = _valid_generated_materialization()
        post.op = llir.AssignOp.ADD_ASSIGN
        return declaration, post, 0, declaration.value

    def record_attempt(**kwargs: object) -> NoReturn:
        record_attempts.append(kwargs)
        raise AssertionError("failed pass must not create a run record")

    monkeypatch.setattr(
        factor_pass_module,
        "_try_hoist_from_loop",
        forge_invalid_assignment,
    )
    monkeypatch.setattr(pass_manager_module, "_record", record_attempt)
    with pytest.raises(LLIRTraversalError) as raised:
        LLIRPassManager().run_loop_invariant_factor_hoist(
            LLIRStatementListArtifact([_activating_loop(accumulator="accumulator")]),
            LoopInvariantFactorHoistPassSpec(),
        )

    assert raised.value.diagnostic.code == (
        "invalid_loop_invariant_factor_materialization_assignment_fields"
    )
    assert calls == 1
    assert record_attempts == []


def test_timing_and_run_records_are_nonsemantic_and_optional() -> None:
    artifact = LLIRStatementListArtifact([llir.BlankLine()])
    spec = LoopInvariantFactorHoistPassSpec()
    timed = LLIRPassManager().run_loop_invariant_factor_hoist(artifact, spec)
    untimed = LLIRPassManager(
        LLIRPassOptions(record_timing=False)
    ).run_loop_invariant_factor_hoist(artifact, spec)

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


def test_empty_and_factor_pass_incremental_plumbing_p95_is_below_one_ms() -> None:
    sample_count = 2000
    source: List[llir.Stmt] = [llir.BlankLine()]
    context = LOOP_INVARIANT_FACTOR_HOIST_CONTEXT
    spec = LoopInvariantFactorHoistPassSpec(context)
    manager = LLIRPassManager(LLIRPassOptions(record_timing=False))

    for _ in range(100):
        manager.run_empty(LLIRRewriteArtifact(source))
        hoist_loop_invariant_factors(source, context)
        manager.run_loop_invariant_factor_hoist(
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
            manager.run_loop_invariant_factor_hoist(
                LLIRStatementListArtifact(source),
                spec,
            )
            managed_elapsed = perf_counter_ns() - managed_started
            direct_started = perf_counter_ns()
            hoist_loop_invariant_factors(source, context)
            direct_elapsed = perf_counter_ns() - direct_started
        else:
            direct_started = perf_counter_ns()
            hoist_loop_invariant_factors(source, context)
            direct_elapsed = perf_counter_ns() - direct_started
            managed_started = perf_counter_ns()
            manager.run_loop_invariant_factor_hoist(
                LLIRStatementListArtifact(source),
                spec,
            )
            managed_elapsed = perf_counter_ns() - managed_started
        incremental_ns.append(managed_elapsed - direct_elapsed)

    assert _p95(empty_ns) <= 1_000_000
    assert _p95(incremental_ns) <= 1_000_000
