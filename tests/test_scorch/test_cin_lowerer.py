from dataclasses import FrozenInstanceError
from typing import cast

import pytest
import torch

from scorch.compiler import llir  # type: ignore[import-untyped]
from scorch.compiler.cin import (
    CIN,
    ForAll,
    IndexStmt,
    IndexVar,
    Operation,
    PostOp,
    PostOps,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_lowerer import CINLowerer, _torch_cpp_kernel_abi
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.diagnostics import CompilerInvariantError, UnsupportedFeature
from scorch.compiler.iterator import (  # type: ignore[import-untyped]
    ModeIterator,
    collect_mode_position_arrays,
    match_mode_position_access,
    match_mode_position_begin,
    match_mode_position_bounds,
)
from scorch.compiler.llir_traversal import (
    LLIRRewriter,
    LLIRTraversalContext,
    LLIRTraversalError,
    LLIRWalker,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Scheduler,
    regblock_force,
)
from scorch.compiler.torch_cpp_abi import (  # type: ignore[import-untyped]
    KernelTensorABI,
    ResultTensorAssembler,
    TorchCppKernelABI,
    mode_index_tensor,
    tensor_data_ptr,
    tensor_storage_member,
)
from scorch.format import LevelType  # type: ignore[import-untyped]


class UnknownCIN(CIN):
    pass


class UnknownIndexStmt(IndexStmt):
    pass


def _result_tensor_assembler(
    tensor_var: TensorVar,
    *,
    known_nnz_var: str | None = None,
    exact_dense_parent_positions: bool = False,
    reserve_hint_var: str | None = None,
) -> ResultTensorAssembler:
    return ResultTensorAssembler(
        name=tensor_var.get_name(),
        level_types=tuple(tensor_var.get_level_types()),
        dtype=tensor_var.dtype,
        known_nnz_var=known_nnz_var,
        exact_dense_parent_positions=exact_dense_parent_positions,
        reserve_hint_var=reserve_hint_var,
    )


def _kernel_tensor_abi(tensor_var: TensorVar) -> KernelTensorABI:
    level_types = tuple(tensor_var.get_level_types())
    return KernelTensorABI(
        name=tensor_var.get_name(),
        level_types=level_types,
        mode_order=tuple(tensor_var.get_mode_order() or tuple(range(len(level_types)))),
        shape=tuple(tensor_var.shape or ()),
        dtype=tensor_var.dtype,
    )


def _build_all_coo_transform_loop() -> llir.ForLoop:
    return llir.ForLoop(
        init=llir.VarInit(
            llir.Var("pMask0", llir.DataType.INT),
            llir.Literal(0),
        ),
        cond=llir.BinOp(
            "<",
            llir.Var("pMask0", llir.DataType.INT),
            llir.Var("pMask0_end", llir.DataType.INT),
        ),
        update=llir.Assign(
            llir.Var("pMask0", llir.DataType.INT),
            llir.Var("pMask1_end", llir.DataType.INT),
        ),
        body=[
            llir.VarInit(
                llir.Var("r", llir.DataType.INT),
                llir.ArrayAccess(
                    array=llir.Var("Mask0_crd", llir.DataType.PTR_INT),
                    index=llir.Var("pMask0", llir.DataType.INT),
                ),
            )
        ],
    )


def _all_coo_transform_coordinate_read(
    *, scalar_accumulator: bool, source: llir.ForLoop | None = None
) -> tuple[llir.ForLoop, llir.VarInit, llir.ArrayAccess]:
    lowerer = CINLowerer()
    lowerer._used_scalar_accum = scalar_accumulator
    transformed = lowerer._transform_coo_loop_for_openmp(
        [source if source is not None else _build_all_coo_transform_loop()]
    )
    parallel_loop = next(
        cast(llir.ForLoop, statement)
        for statement in transformed
        if type(statement) is llir.ForLoop
        and cast(llir.ForLoop, statement).omp_parallel_for
    )
    initializer = next(
        cast(llir.VarInit, statement)
        for statement in parallel_loop.body
        if type(statement) is llir.VarInit
        and cast(llir.VarInit, statement).var.name == "r"
    )
    assert type(initializer.value) is llir.ArrayAccess
    return parallel_loop, initializer, cast(llir.ArrayAccess, initializer.value)


def _all_coo_transform_end_bound(
    *, source: llir.ForLoop | None = None
) -> tuple[llir.ForLoop, llir.VarInit, llir.Add]:
    lowerer = CINLowerer()
    lowerer._used_scalar_accum = True
    transformed = lowerer._transform_coo_loop_for_openmp(
        [source if source is not None else _build_all_coo_transform_loop()]
    )
    parallel_loop = next(
        cast(llir.ForLoop, statement)
        for statement in transformed
        if type(statement) is llir.ForLoop
        and cast(llir.ForLoop, statement).omp_parallel_for
    )
    initializer = next(
        cast(llir.VarInit, statement)
        for statement in parallel_loop.body
        if type(statement) is llir.VarInit
        and cast(llir.VarInit, statement).var.name == "pMask1_end"
    )
    assert type(initializer.value) is llir.Add
    return parallel_loop, initializer, cast(llir.Add, initializer.value)


def _build_activating_all_coo_sddmm() -> ForAll:
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


def _build_outer_workspace_statement(
    result_format: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[Where, TensorVar]:
    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("Result", fmt=result_format, dtype=dtype)
    left = TensorVar("Left", fmt="ds", mode_order=[1, 0], dtype=dtype)
    right = TensorVar("Right", fmt="ds", dtype=dtype)
    workspace = Workspace("wksp", dim=2, dtype=dtype)
    return (
        Where(
            producer=ForAll(
                reduction,
                ForAll(
                    row,
                    ForAll(
                        column,
                        TensorAssign(
                            workspace[row, column],
                            left[row, reduction] * right[reduction, column],
                            op=Operation.ADD,
                        ),
                    ),
                ),
            ),
            consumer=ForAll(
                row,
                ForAll(
                    column,
                    TensorAssign(result[row, column], workspace[row, column]),
                ),
            ),
        ),
        result,
    )


def _assert_workspace_pair_read(
    value: llir.Expr,
    expected_member: str,
    expected_index: int | None,
) -> llir.Var:
    if expected_index is None:
        assert type(value) is llir.MemberAccess
        member_access = cast(llir.MemberAccess, value)
    else:
        assert type(value) is llir.ArrayAccess
        array_access = cast(llir.ArrayAccess, value)
        assert array_access.tensor_access is None
        assert type(array_access.index) is llir.Literal
        literal = cast(llir.Literal, array_access.index)
        assert literal.value == expected_index
        assert literal.data_type is llir.DataType.INT64
        assert type(array_access.array) is llir.MemberAccess
        member_access = cast(llir.MemberAccess, array_access.array)

    assert member_access.member == expected_member
    assert type(member_access.base) is llir.Var
    base = cast(llir.Var, member_access.base)
    assert base.name == "it"
    assert base.type is llir.DataType.CONST_AUTO_REF
    assert base.tensor_access is None
    return base


def _assert_vector_move_initialization(
    statement: llir.Stmt,
    *,
    target_name: str,
    vector_name: str,
    vector_type: llir.DataType,
    dtype_name: str,
) -> tuple[llir.FunctionCall, llir.QualifiedName]:
    assert type(statement) is llir.VarInit
    initializer = cast(llir.VarInit, statement)
    assert initializer.var.name == target_name
    assert initializer.var.type is llir.DataType.TORCH_TENSOR
    assert type(initializer.value) is llir.FunctionCall
    conversion = cast(llir.FunctionCall, initializer.value)
    assert conversion.name == "scorch_tensor_from_vector"
    assert type(conversion.args) is tuple
    assert len(conversion.args) == 2
    assert type(conversion.args[0]) is llir.FunctionCall
    move = cast(llir.FunctionCall, conversion.args[0])
    assert move.name == "std::move"
    assert type(move.args) is tuple
    assert len(move.args) == 1
    assert type(move.args[0]) is llir.Var
    vector = cast(llir.Var, move.args[0])
    assert vector.name == vector_name
    assert vector.type is vector_type
    assert vector.tensor_access is None
    dtype = _assert_torch_qualified_name(conversion.args[1], dtype_name)
    return move, dtype


def _assert_torch_qualified_name(
    expression: llir.Expr,
    expected_spelling: str,
) -> llir.QualifiedName:
    assert type(expression) is llir.QualifiedName
    qualified = cast(llir.QualifiedName, expression)
    namespace, separator, name = expected_spelling.partition("::")
    assert separator == "::"
    assert qualified.namespace == namespace == "torch"
    assert qualified.name == name
    assert qualified.data_type is llir.DataType.TORCH_SCALAR_TYPE
    assert LLIRLowerer().lower_llir(qualified) == expected_spelling
    return qualified


def _assert_data_ptr_call(
    expression: llir.Expr,
    data_type: llir.DataType,
) -> llir.MemberCall:
    assert type(expression) is llir.MemberCall
    call = cast(llir.MemberCall, expression)
    assert call.member == "data_ptr"
    assert type(call.template_args) is tuple
    assert call.template_args == (data_type,)
    assert type(call.args) is tuple
    assert call.args == ()
    return call


def _assert_torch_tensor_var(expression: llir.Expr, name: str) -> llir.Var:
    assert type(expression) is llir.Var
    variable = cast(llir.Var, expression)
    assert variable.name == name
    assert variable.type is llir.DataType.TORCH_TENSOR
    assert variable.is_ptr is False
    assert variable.is_restrict is False
    assert variable.tensor_access is None
    return variable


def _assert_mode_index_tensor(
    expression: llir.Expr,
    *,
    tensor_name: str,
    level: int,
    slot: int,
) -> tuple[llir.ArrayAccess, llir.ArrayAccess, llir.Var]:
    assert type(expression) is llir.ArrayAccess
    outer = cast(llir.ArrayAccess, expression)
    assert outer.tensor_access is None
    assert type(outer.index) is llir.Literal
    outer_index = cast(llir.Literal, outer.index)
    assert outer_index.value == slot
    assert outer_index.data_type is llir.DataType.INT

    assert type(outer.array) is llir.ArrayAccess
    inner = cast(llir.ArrayAccess, outer.array)
    assert inner.tensor_access is None
    assert type(inner.index) is llir.Literal
    inner_index = cast(llir.Literal, inner.index)
    assert inner_index.value == level
    assert inner_index.data_type is llir.DataType.INT

    assert type(inner.array) is llir.Var
    root = cast(llir.Var, inner.array)
    assert root.name == f"{tensor_name}_mode_indices"
    assert root.type is llir.DataType.STD_VECTOR_2D_TORCH_TENSOR
    assert root.is_ptr is False
    assert root.is_restrict is False
    assert root.tensor_access is None
    return outer, inner, root


def _assert_tensor_storage_member(
    expression: llir.Expr,
    tensor_name: str,
    *members: str,
) -> tuple[llir.Var, tuple[llir.MemberAccess, ...]]:
    chain: list[llir.MemberAccess] = []
    current = expression
    while type(current) is llir.MemberAccess:
        member = cast(llir.MemberAccess, current)
        chain.append(member)
        current = member.base

    assert [member.member for member in reversed(chain)] == list(members)
    assert type(current) is llir.Var
    root = cast(llir.Var, current)
    assert root.name == tensor_name
    assert root.type is llir.DataType.TACO_TENSOR
    assert root.is_ptr is False
    assert root.is_restrict is False
    assert root.tensor_access is None
    return root, tuple(chain)


def _assert_mode_index_initializer(
    expression: llir.Expr,
    expected_names: tuple[tuple[str, ...], ...],
) -> tuple[llir.Array, tuple[llir.Array, ...], tuple[llir.Var, ...]]:
    assert type(expression) is llir.Array
    outer = cast(llir.Array, expression)
    assert outer.data_type is llir.DataType.STD_VECTOR_2D_TORCH_TENSOR
    assert type(outer.values) is tuple
    assert len(outer.values) == len(expected_names)

    inner_arrays: list[llir.Array] = []
    children: list[llir.Var] = []
    for expression_set, names in zip(outer.values, expected_names):
        assert type(expression_set) is llir.Array
        inner = cast(llir.Array, expression_set)
        inner_arrays.append(inner)
        assert inner.data_type is llir.DataType.STD_VECTOR_TORCH_TENSOR
        assert type(inner.values) is tuple
        assert len(inner.values) == len(names)
        for child, name in zip(inner.values, names):
            children.append(_assert_torch_tensor_var(child, name))
    return outer, tuple(inner_arrays), tuple(children)


def test_torch_cpp_abi_helpers_reject_malformed_boundary_inputs() -> None:
    for tensor_name in ("", "Input.value", 1, None):
        with pytest.raises(TypeError, match="root"):
            tensor_storage_member(  # type: ignore[arg-type]
                tensor_name,
                "storage",
            )
        with pytest.raises(TypeError, match="root"):
            mode_index_tensor(  # type: ignore[arg-type]
                tensor_name,
                0,
                0,
            )

    with pytest.raises(TypeError, match="cannot be empty"):
        tensor_storage_member("Input")
    with pytest.raises(TypeError, match="non-empty identifier"):
        tensor_storage_member("Input", "storage.value")

    for level in (-1, True, 1.0):
        with pytest.raises(TypeError, match="level"):
            mode_index_tensor("Input", level, 0)  # type: ignore[arg-type]
    for slot in (-1, True, 1.0):
        with pytest.raises(TypeError, match="slot"):
            mode_index_tensor("Input", 0, slot)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="receiver"):
        tensor_data_ptr(object(), llir.DataType.FLOAT32)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="template argument"):
        tensor_data_ptr(  # type: ignore[arg-type]
            llir.Var("Input_values", llir.DataType.TORCH_TENSOR),
            "float",
        )

    valid_metadata = {
        "name": "Result",
        "level_types": (LevelType.DENSE, LevelType.COMPRESSED),
        "dtype": torch.float32,
    }
    for name in ("", "Result.value", 1):
        with pytest.raises(TypeError, match="result tensor name"):
            ResultTensorAssembler(  # type: ignore[arg-type]
                **{**valid_metadata, "name": name}
            )
    for level_types in (
        [LevelType.DENSE],
        (LevelType.DENSE, "compressed"),
        "ds",
    ):
        with pytest.raises(TypeError, match="immutable LevelType tuple"):
            ResultTensorAssembler(  # type: ignore[arg-type]
                **{**valid_metadata, "level_types": level_types}
            )
    for dtype in ("float32", None):
        with pytest.raises(TypeError, match="torch.dtype"):
            ResultTensorAssembler(  # type: ignore[arg-type]
                **{**valid_metadata, "dtype": dtype}
            )
    for known_nnz_var in ("", "known-nnz", 1):
        with pytest.raises(TypeError, match="known nnz"):
            ResultTensorAssembler(  # type: ignore[arg-type]
                **valid_metadata,
                known_nnz_var=known_nnz_var,
            )
    with pytest.raises(TypeError, match="exact dense parent"):
        ResultTensorAssembler(  # type: ignore[arg-type]
            **valid_metadata,
            exact_dense_parent_positions=1,
        )
    for reserve_hint_var in ("", "reserve.hint", 1):
        with pytest.raises(TypeError, match="reserve hint"):
            ResultTensorAssembler(  # type: ignore[arg-type]
                **valid_metadata,
                reserve_hint_var=reserve_hint_var,
            )


def test_result_tensor_assembler_owns_one_frozen_abi_snapshot() -> None:
    tensor = TensorVar("Result", fmt="ds", dtype=torch.float32)
    assembler = _result_tensor_assembler(
        tensor,
        known_nnz_var="_known_nnz",
        exact_dense_parent_positions=True,
        reserve_hint_var="_reserve_hint",
    )
    equal = _result_tensor_assembler(
        TensorVar("Result", fmt="ds", dtype=torch.float32),
        known_nnz_var="_known_nnz",
        exact_dense_parent_positions=True,
        reserve_hint_var="_reserve_hint",
    )

    assert ResultTensorAssembler.__module__ == "scorch.compiler.torch_cpp_abi"
    assert assembler == equal
    assert hash(assembler) == hash(equal)
    assert assembler is not equal
    assert assembler.name == "Result"
    assert assembler.level_types == (LevelType.DENSE, LevelType.COMPRESSED)
    assert assembler.levels == 2
    assert assembler.is_dense is False
    assert assembler.dtype is torch.float32
    assert assembler.known_nnz_var == "_known_nnz"
    assert assembler.exact_dense_parent_positions is True
    assert assembler.reserve_hint_var == "_reserve_hint"
    assert not hasattr(assembler, "tensor_var")

    tensor.name = "Mutated"
    tensor.format = TensorVar("Other", fmt="oo").format
    tensor.dtype = torch.float64
    assert assembler == equal
    lowerer = LLIRLowerer()
    assert lowerer.lower_llir(assembler.emit_value_array_init()) == lowerer.lower_llir(
        equal.emit_value_array_init()
    )
    assert lowerer.lower_llir(
        assembler.emit_level_indices_init()
    ) == lowerer.lower_llir(equal.emit_level_indices_init())
    assert lowerer.lower_llir(assembler.emit_final_assembly()) == lowerer.lower_llir(
        equal.emit_final_assembly()
    )

    with pytest.raises(FrozenInstanceError):
        assembler.name = "Other"
    with pytest.raises(FrozenInstanceError):
        assembler.level_types = ()
    with pytest.raises(FrozenInstanceError):
        assembler.known_nnz_var = None


def test_kernel_abi_contracts_reject_malformed_and_forged_fields() -> None:
    valid_tensor = {
        "name": "Input",
        "level_types": (LevelType.DENSE, LevelType.COMPRESSED),
        "mode_order": (1, 0),
        "shape": (3, 5),
        "dtype": torch.float32,
    }
    for name in ("", "Input.value", 1):
        with pytest.raises(TypeError, match="kernel tensor name"):
            KernelTensorABI(**{**valid_tensor, "name": name})  # type: ignore[arg-type]
    for levels in ([LevelType.DENSE], (LevelType.DENSE, "compressed"), "ds"):
        with pytest.raises(TypeError, match="immutable LevelType tuple"):
            KernelTensorABI(  # type: ignore[arg-type]
                **{**valid_tensor, "level_types": levels}
            )
    for malformed_mode_order in ([1, 0], (True, 0), (1.0, 0)):
        with pytest.raises(TypeError, match="mode order"):
            KernelTensorABI(  # type: ignore[arg-type]
                **{**valid_tensor, "mode_order": malformed_mode_order}
            )
    for invalid_mode_order in ((0,), (0, 0), (0, 2)):
        with pytest.raises(ValueError, match="rank-sized permutation"):
            KernelTensorABI(**{**valid_tensor, "mode_order": invalid_mode_order})
    for shape in ([3, 5], (True, 5), (3.0, 5)):
        with pytest.raises(TypeError, match="kernel tensor shape"):
            KernelTensorABI(**{**valid_tensor, "shape": shape})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape must match the level rank"):
        KernelTensorABI(**{**valid_tensor, "shape": (3,)})
    with pytest.raises(ValueError, match="extents must be non-negative"):
        KernelTensorABI(**{**valid_tensor, "shape": (3, -1)})
    with pytest.raises(TypeError, match="kernel tensor dtype"):
        KernelTensorABI(**{**valid_tensor, "dtype": "float32"})  # type: ignore[arg-type]

    tensor = KernelTensorABI(**valid_tensor)
    valid_kernel = {
        "result_shape": (3, 5),
        "result_rank": 2,
        "input_tensors": (tensor,),
    }
    for function_name in ("", "scorch.evaluate", 1):
        with pytest.raises(TypeError, match="function name"):
            TorchCppKernelABI(  # type: ignore[arg-type]
                **valid_kernel,
                function_name=function_name,
            )
    for result_shape in ([3, 5], (True, 5), (3.0, 5)):
        with pytest.raises(TypeError, match="kernel result shape"):
            TorchCppKernelABI(  # type: ignore[arg-type]
                **{**valid_kernel, "result_shape": result_shape}
            )
    for result_rank in (-1, True, 2.0):
        with pytest.raises(TypeError, match="kernel result rank"):
            TorchCppKernelABI(  # type: ignore[arg-type]
                **{**valid_kernel, "result_rank": result_rank}
            )
    with pytest.raises(ValueError, match="shape must match the result rank"):
        TorchCppKernelABI(**{**valid_kernel, "result_shape": (3,)})
    with pytest.raises(ValueError, match="extents must be non-negative"):
        TorchCppKernelABI(**{**valid_kernel, "result_shape": (3, -1)})
    for input_tensors in ([tensor], (object(),)):
        with pytest.raises(TypeError, match="exact KernelTensorABI tuple"):
            TorchCppKernelABI(  # type: ignore[arg-type]
                **{**valid_kernel, "input_tensors": input_tensors}
            )

    class UnknownKernelTensorABI(KernelTensorABI):
        pass

    unknown_tensor = UnknownKernelTensorABI(**valid_tensor)
    with pytest.raises(TypeError, match="exact KernelTensorABI tuple"):
        TorchCppKernelABI(**{**valid_kernel, "input_tensors": (unknown_tensor,)})
    for extra_names in (["bias"], ("bias.value",), (1,)):
        with pytest.raises(TypeError, match="extra tensor names"):
            TorchCppKernelABI(  # type: ignore[arg-type]
                **valid_kernel,
                extra_tensor_names=extra_names,
                extra_tensor_dtype=torch.float32,
            )
    with pytest.raises(TypeError, match="when present"):
        TorchCppKernelABI(**valid_kernel, extra_tensor_names=("bias",))
    with pytest.raises(TypeError, match="without extra tensors"):
        TorchCppKernelABI(**valid_kernel, extra_tensor_dtype=torch.float32)
    with pytest.raises(ValueError, match="argument names must be unique"):
        TorchCppKernelABI(**{**valid_kernel, "input_tensors": (tensor, tensor)})
    with pytest.raises(ValueError, match="argument names must be unique"):
        TorchCppKernelABI(
            **valid_kernel,
            extra_tensor_names=("Input",),
            extra_tensor_dtype=torch.float32,
        )
    result_name_collision = KernelTensorABI(
        "result",
        (LevelType.DENSE,),
        (0,),
        (),
        torch.float32,
    )
    with pytest.raises(ValueError, match="argument names must be unique"):
        TorchCppKernelABI(
            result_shape=(),
            result_rank=1,
            input_tensors=(result_name_collision,),
        )

    forged_tensor = KernelTensorABI(**valid_tensor)
    object.__setattr__(forged_tensor, "shape", [3, 5])
    with pytest.raises(TypeError, match="kernel tensor shape"):
        forged_tensor.emit_value_pointer()

    forged_kernel = TorchCppKernelABI(**valid_kernel)
    object.__setattr__(forged_kernel, "input_tensors", (forged_tensor,))
    with pytest.raises(TypeError, match="kernel tensor shape"):
        forged_kernel.emit_arguments()

    forged_order = KernelTensorABI(**valid_tensor)
    object.__setattr__(forged_order, "mode_order", (0, 0))
    with pytest.raises(ValueError, match="rank-sized permutation"):
        forged_order.emit_level_array_bindings()


def test_kernel_abi_owns_one_frozen_semantic_snapshot() -> None:
    result = TensorVar("Result", shape=(5, 3), fmt="dd", dtype=torch.float64)
    left = TensorVar(
        "Left",
        shape=(3, 5),
        fmt="ds",
        dtype=torch.float64,
        mode_order=[1, 0],
    )
    mask = TensorVar("Mask", shape=(5, 3), fmt="oo", dtype=torch.float32)
    post_ops = PostOps(
        ops=[PostOp(kind="add", tensor_name="bias")],
        extra_tensors=["bias"],
    )
    abi = _torch_cpp_kernel_abi(result, [left, mask], post_ops)
    equal = TorchCppKernelABI(
        result_shape=(5, 3),
        result_rank=2,
        input_tensors=(
            KernelTensorABI(
                "Left",
                (LevelType.DENSE, LevelType.COMPRESSED),
                (1, 0),
                (3, 5),
                torch.float64,
            ),
            KernelTensorABI(
                "Mask",
                (LevelType.COORDINATE, LevelType.COORDINATE),
                (0, 1),
                (5, 3),
                torch.float32,
            ),
        ),
        extra_tensor_names=("bias",),
        extra_tensor_dtype=torch.float64,
    )

    assert abi == equal
    assert hash(abi) == hash(equal)
    assert abi is not equal
    assert not hasattr(abi, "result_tensor_var")
    assert not hasattr(abi, "post_ops")
    assert all(not hasattr(tensor, "tensor_var") for tensor in abi.input_tensors)

    result.name = "ChangedResult"
    result.shape = (9, 9)
    result.format = TensorVar("OtherResult", fmt="s").format
    result.dtype = torch.float32
    left.name = "ChangedLeft"
    left.shape = (7, 11)
    left.mode_order.reverse()
    left.format = TensorVar("OtherLeft", fmt="oo").format
    left.dtype = torch.float32
    mask.name = "ChangedMask"
    post_ops.extra_tensors.append("scale")

    assert abi == equal
    assert abi.emit_arguments() == equal.emit_arguments()
    assert abi.emit_validation() == equal.emit_validation()
    assert LLIRLowerer().lower_llir(
        abi.emit_input_prologue()
    ) == LLIRLowerer().lower_llir(equal.emit_input_prologue())
    assert LLIRLowerer().lower_llir(
        abi.emit_extra_tensor_prologue()
    ) == LLIRLowerer().lower_llir(equal.emit_extra_tensor_prologue())

    with pytest.raises(FrozenInstanceError):
        abi.result_rank = 3
    with pytest.raises(FrozenInstanceError):
        abi.input_tensors = ()
    with pytest.raises(FrozenInstanceError):
        abi.input_tensors[0].shape = ()


def test_kernel_abi_arguments_validation_and_function_are_fresh_and_byte_exact() -> (
    None
):
    abi = TorchCppKernelABI(
        result_shape=(5, 3),
        result_rank=2,
        input_tensors=(
            KernelTensorABI(
                "Left",
                (LevelType.DENSE, LevelType.COMPRESSED),
                (1, 0),
                (3, 5),
                torch.float64,
            ),
            KernelTensorABI(
                "Mask",
                (LevelType.COORDINATE, LevelType.COORDINATE),
                (0, 1),
                (5, 3),
                torch.float32,
            ),
        ),
        extra_tensor_names=("bias",),
        extra_tensor_dtype=torch.float64,
    )
    first_args = abi.emit_arguments()
    second_args = abi.emit_arguments()
    expected_arguments = (
        ("result_shape", llir.DataType.STD_VECTOR_INT),
        ("Left_shape", llir.DataType.STD_VECTOR_INT),
        ("Left_mode_indices", llir.DataType.STD_VECTOR_2D_TORCH_TENSOR),
        ("Left_values", llir.DataType.TORCH_TENSOR),
        ("Mask_shape", llir.DataType.STD_VECTOR_INT),
        ("Mask_mode_indices", llir.DataType.STD_VECTOR_2D_TORCH_TENSOR),
        ("Mask_values", llir.DataType.TORCH_TENSOR),
        ("bias_values", llir.DataType.TORCH_TENSOR),
    )
    assert tuple((argument.name, argument.type) for argument in first_args) == (
        expected_arguments
    )
    assert first_args == second_args
    assert first_args is not second_args
    assert all(first is not second for first, second in zip(first_args, second_args))
    assert all(
        not argument.is_ptr
        and not argument.is_restrict
        and argument.tensor_access is None
        for argument in first_args
    )

    first_validation = abi.emit_validation()
    second_validation = abi.emit_validation()
    expected_validation = (
        "scorch_native::validate_jit_result_shape(result_shape, {5, 3}, 2, "
        '"evaluate")',
        'scorch_native::validate_jit_tensor("evaluate", "Left", Left_shape, '
        "Left_mode_indices, Left_values, torch::kFloat64, {0, 1}, {1, 0}, {3, 5})",
        'scorch_native::validate_jit_tensor("evaluate", "Mask", Mask_shape, '
        "Mask_mode_indices, Mask_values, torch::kFloat32, {2, 2}, {0, 1}, {5, 3})",
        "scorch_native::validate_jit_extra_tensor(bias_values, torch::kFloat64, "
        '"evaluate", "bias_values")',
    )
    assert all(
        type(statement) is llir.FunctionCallStmt for statement in first_validation
    )
    assert all(type(statement.args) is tuple for statement in first_validation)
    assert first_validation == second_validation
    assert all(
        first is not second
        for first, second in zip(first_validation, second_validation)
    )
    assert LLIRLowerer().lower_llir(first_validation) == "\n".join(
        f"{code};" for code in expected_validation
    )

    artifact: list[llir.Stmt] = [
        *first_validation,
        *abi.emit_input_prologue(),
        *abi.emit_extra_tensor_prologue(),
    ]

    def traversal_snapshot(value: list[llir.Stmt]) -> tuple[tuple[str, ...], ...]:
        paths: list[tuple[str, ...]] = []

        class OrderCollector(LLIRWalker):
            def enter_node(
                self,
                node: llir.Node,
                path: tuple[str, ...],
            ) -> None:
                paths.append((*path, type(node).__name__))

        OrderCollector(
            LLIRTraversalContext(stage="test", pass_name="kernel_abi_order")
        ).walk(value)
        return tuple(paths)

    first_snapshot = traversal_snapshot(artifact)
    assert traversal_snapshot(artifact) == first_snapshot
    rewritten = cast(
        list[llir.Stmt],
        LLIRRewriter(
            LLIRTraversalContext(stage="test", pass_name="rewrite_kernel_abi")
        ).rewrite(artifact),
    )
    assert rewritten is not artifact
    assert len(rewritten) == len(artifact)
    assert traversal_snapshot(rewritten) == first_snapshot
    assert LLIRLowerer().lower_llir(rewritten) == LLIRLowerer().lower_llir(artifact)

    def node_ids(value: list[llir.Stmt]) -> set[int]:
        identities: set[int] = set()

        class IdentityCollector(LLIRWalker):
            def enter_node(
                self,
                node: llir.Node,
                path: tuple[str, ...],
            ) -> None:
                del path
                identities.add(id(node))

        IdentityCollector(
            LLIRTraversalContext(stage="test", pass_name="kernel_abi_ownership")
        ).walk(value)
        return identities

    assert node_ids(artifact).isdisjoint(node_ids(rewritten))

    body = [
        llir.Comment("body"),
        llir.Return(llir.Var("Result", llir.DataType.TACO_TENSOR)),
    ]
    first_function = abi.assemble_function(body)
    second_function = abi.assemble_function(body)
    assert first_function.return_type is llir.DataType.TACO_TENSOR
    assert first_function.name == "evaluate"
    assert first_function.args == second_function.args == first_args
    assert first_function.args is not second_function.args
    assert first_function.body == second_function.body == body
    assert first_function.body is not second_function.body
    assert first_function.body is not body
    body.append(llir.BlankLine())
    assert len(first_function.body) == len(second_function.body) == 2
    assert (
        LLIRLowerer()
        .lower_llir(first_function)
        .startswith(
            "Tensor evaluate(std::vector<int64_t> result_shape, "
            "std::vector<int64_t> Left_shape, "
            "std::vector<std::vector<torch::Tensor>> Left_mode_indices, "
            "torch::Tensor Left_values, std::vector<int64_t> Mask_shape, "
            "std::vector<std::vector<torch::Tensor>> Mask_mode_indices, "
            "torch::Tensor Mask_values, torch::Tensor bias_values) {"
        )
    )

    with pytest.raises(TypeError, match="exact LLIR list"):
        abi.assemble_function(tuple(body))  # type: ignore[arg-type]


def test_kernel_abi_rejects_unsupported_singleton_validation() -> None:
    with pytest.raises(ValueError, match="unsupported JIT level type"):
        KernelTensorABI(
            "Input",
            (LevelType.SINGLETON,),
            (0,),
            (4,),
            torch.float32,
        )

    forged = KernelTensorABI(
        "Input",
        (LevelType.DENSE,),
        (0,),
        (4,),
        torch.float32,
    )
    object.__setattr__(forged, "level_types", (LevelType.SINGLETON,))
    with pytest.raises(ValueError, match="unsupported JIT level type"):
        forged.emit_level_array_bindings()


def _collect_move_calls(value: llir.Node) -> list[llir.FunctionCall]:
    calls: list[llir.FunctionCall] = []

    class MoveCollector(LLIRWalker):
        def visit_function_call(
            self,
            node: llir.FunctionCall,
            path: tuple[str, ...],
        ) -> None:
            if node.name == "std::move":
                calls.append(node)
            super().visit_function_call(node, path)

    MoveCollector(
        LLIRTraversalContext(stage="test", pass_name="collect_move_calls")
    ).walk(value)
    return calls


def _collect_qualified_names(value: llir.Node) -> list[llir.QualifiedName]:
    names: list[llir.QualifiedName] = []

    class QualifiedNameCollector(LLIRWalker):
        def visit_qualified_name(
            self,
            node: llir.QualifiedName,
            path: tuple[str, ...],
        ) -> None:
            names.append(node)
            super().visit_qualified_name(node, path)

    QualifiedNameCollector(
        LLIRTraversalContext(stage="test", pass_name="collect_qualified_names")
    ).walk(value)
    return names


def test_unknown_cin_node_fails_at_cin_lowering():
    with pytest.raises(
        CompilerInvariantError,
        match=r"stage=CIN lowering: unknown CIN node type 'UnknownCIN'",
    ):
        CINLowerer().lower_CIN(UnknownCIN())


def test_unknown_index_statement_fails_before_lowering_an_empty_kernel():
    with pytest.raises(
        CompilerInvariantError,
        match=r"stage=CIN lowering: unknown IndexStmt node type 'UnknownIndexStmt'",
    ):
        CINLowerer().lower_IndexStmt(UnknownIndexStmt(lhs=None, rhs=None))


def test_unknown_post_op_fails_at_cin_lowering():
    post_ops = PostOps(ops=[PostOp(kind="clip")], extra_tensors=[])

    with pytest.raises(
        UnsupportedFeature,
        match=r"stage=CIN lowering: unsupported post-op kind 'clip'",
    ):
        CINLowerer(post_ops=post_ops)


@pytest.mark.parametrize(
    ("kind", "tensor_name"),
    [
        ("add", "bias"),
        ("mul", "scale"),
        ("relu", None),
        ("gelu", None),
        ("tanh", None),
        ("sigmoid", None),
    ],
)
def test_supported_post_ops_still_lower(kind, tensor_name):
    extra_tensors = [tensor_name] if tensor_name else []
    post_ops = PostOps(
        ops=[PostOp(kind=kind, tensor_name=tensor_name)],
        extra_tensors=extra_tensors,
    )

    statements = CINLowerer(post_ops=post_ops)._emit_post_ops("output", "i")

    assert len(statements) == 1
    assignment = cast(llir.Assign, statements[0])
    assert type(assignment.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, assignment.var)
    assert cast(llir.Var, target.array).name == "output"
    assert cast(llir.Var, target.index).name == "i"
    assert LLIRLowerer().lower_llir(assignment).startswith("output[i] ")


def test_post_op_extra_tensor_data_ptr_is_live_typed_and_fresh() -> None:
    index = IndexVar("i")
    result = TensorVar("Result", fmt="d")
    source = TensorVar("Input", fmt="d")
    statement = ForAll(index, TensorAssign(result[index], source[index]))
    post_ops = PostOps(
        ops=[PostOp(kind="add", tensor_name="bias")],
        extra_tensors=["bias"],
    )
    original = str(statement)

    first_function = CINLowerer(post_ops=post_ops).lower_IndexStmt(statement)
    second_function = CINLowerer(post_ops=post_ops).lower_IndexStmt(statement)
    assert type(first_function) is llir.Function
    assert type(second_function) is llir.Function
    first_function = cast(llir.Function, first_function)
    second_function = cast(llir.Function, second_function)
    assert first_function.return_type is llir.DataType.TACO_TENSOR
    assert first_function.name == second_function.name == "evaluate"
    assert tuple(
        (argument.name, argument.type) for argument in first_function.args
    ) == (
        ("result_shape", llir.DataType.STD_VECTOR_INT),
        ("Input_shape", llir.DataType.STD_VECTOR_INT),
        ("Input_mode_indices", llir.DataType.STD_VECTOR_2D_TORCH_TENSOR),
        ("Input_values", llir.DataType.TORCH_TENSOR),
        ("bias_values", llir.DataType.TORCH_TENSOR),
    )
    assert first_function.args == second_function.args
    assert first_function.args is not second_function.args
    assert all(
        first is not second
        for first, second in zip(first_function.args, second_function.args)
    )
    validation = first_function.body[:3]
    assert all(type(node) is llir.FunctionCallStmt for node in validation)
    assert not any(type(node) is llir.RawStmt for node in validation)
    assert [LLIRLowerer().lower_llir(node) for node in validation] == [
        'scorch_native::validate_jit_result_shape(result_shape, {}, 1, "evaluate");',
        'scorch_native::validate_jit_tensor("evaluate", "Input", Input_shape, '
        "Input_mode_indices, Input_values, torch::kFloat32, {0}, {0}, {});",
        "scorch_native::validate_jit_extra_tensor(bias_values, torch::kFloat32, "
        '"evaluate", "bias_values");',
    ]

    def pointer_initializer(function: llir.Function) -> llir.VarInit:
        return next(
            cast(llir.VarInit, node)
            for node in function.body
            if type(node) is llir.VarInit and node.var.name == "bias_val"
        )

    first = pointer_initializer(first_function)
    second = pointer_initializer(second_function)
    assert first.var.type is llir.DataType.PTR_FLOAT32
    assert first.var.is_restrict is True
    assert first.var.tensor_access is None
    first_call = _assert_data_ptr_call(first.value, llir.DataType.FLOAT32)
    second_call = _assert_data_ptr_call(second.value, llir.DataType.FLOAT32)
    first_base = _assert_torch_tensor_var(first_call.base, "bias_values")
    second_base = _assert_torch_tensor_var(second_call.base, "bias_values")
    assert first_call == second_call
    assert hash(first_call) == hash(second_call)
    assert first_call is not second_call
    assert first_base is not second_base
    assert first_base is not first.var
    assert LLIRLowerer().lower_llir(first) == (
        "float* __restrict__ bias_val = bias_values.data_ptr<float>();"
    )
    assert LLIRLowerer().lower_llir(second) == LLIRLowerer().lower_llir(first)
    assert str(statement) == original
    assert post_ops.extra_tensors == ["bias"]


def test_result_position_initialization_uses_a_frozen_structured_target() -> None:
    result = TensorVar("Result", fmt="ds")

    statements = _result_tensor_assembler(result).emit_level_indices_init()
    assignment = next(
        cast(llir.Assign, statement)
        for statement in statements
        if type(statement) is llir.Assign
    )

    assert type(assignment.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, assignment.var)
    assert cast(llir.Var, target.array).name == "Result1_pos"
    assert cast(llir.Literal, target.index).value == 0
    assert LLIRLowerer().lower_llir(assignment) == "Result1_pos[0] = 0;"


def test_fixed_result_position_owner_is_typed_fresh_direct_initialization() -> None:
    tensor = TensorVar("Result", fmt="ds")
    assembler = _result_tensor_assembler(
        tensor,
        exact_dense_parent_positions=True,
    )

    first = next(
        cast(llir.DirectInit, statement)
        for statement in assembler.emit_level_indices_init()
        if type(statement) is llir.DirectInit
    )
    second = next(
        cast(llir.DirectInit, statement)
        for statement in assembler.emit_level_indices_init()
        if type(statement) is llir.DirectInit
    )

    assert first == second
    assert hash(first) == hash(second)
    assert first is not second
    assert first.var is not second.var
    assert first.args is not second.args
    assert first.var.name == "Result1_pos"
    assert first.var.type is llir.DataType.STD_VECTOR_C_INT
    assert first.var.is_ptr is False
    assert first.var.is_restrict is False
    assert first.var.tensor_access is None
    assert type(first.args) is tuple
    assert len(first.args) == 2

    extent = cast(llir.Add, first.args[0])
    equal_extent = cast(llir.Add, second.args[0])
    assert type(extent) is llir.Add
    assert extent is not equal_extent
    assert type(extent.left) is llir.Cast
    parent_cast = cast(llir.Cast, extent.left)
    equal_parent_cast = cast(llir.Cast, equal_extent.left)
    assert parent_cast is not equal_parent_cast
    assert parent_cast.data_type is llir.DataType.SIZE_T
    assert type(parent_cast.expr) is llir.Var
    parent_size = cast(llir.Var, parent_cast.expr)
    equal_parent_size = cast(llir.Var, equal_parent_cast.expr)
    assert parent_size is not equal_parent_size
    assert parent_size.name == "Result0_size"
    assert parent_size.type is llir.DataType.INT64
    assert type(extent.right) is llir.Literal
    assert cast(llir.Literal, extent.right).value == 1
    assert cast(llir.Literal, extent.right).data_type is llir.DataType.INT
    assert extent.right is not equal_extent.right
    assert type(first.args[1]) is llir.Literal
    assert cast(llir.Literal, first.args[1]).value == 0
    assert cast(llir.Literal, first.args[1]).data_type is llir.DataType.INT
    assert first.args[1] is not second.args[1]

    assert LLIRLowerer().lower_llir(first) == (
        "std::vector<int> Result1_pos((size_t) Result0_size + 1, 0);"
    )
    assert "scorch_tensor_from_vector(std::move(Result1_pos), torch::kInt)" in (
        LLIRLowerer().lower_llir(assembler.emit_final_assembly())
    )


def _assert_torch_empty_extent(
    statements: list[llir.Stmt],
    *,
    extent_name: str,
    dtype_name: str = "torch::kFloat32",
) -> tuple[llir.Array, llir.QualifiedName, llir.VarInit]:
    initializer = next(
        cast(llir.VarInit, statement)
        for statement in statements
        if type(statement) is llir.VarInit
        and cast(llir.VarInit, statement).var.name == "Result_values_torch"
    )
    assert type(initializer.value) is llir.FunctionCall
    call = cast(llir.FunctionCall, initializer.value)
    assert call.name == "torch::empty"
    assert type(call.args) is tuple
    assert len(call.args) == 2
    assert type(call.args[0]) is llir.Array
    extent = cast(llir.Array, call.args[0])
    assert extent.data_type is llir.DataType.INT64
    assert type(extent.values) is tuple
    assert len(extent.values) == 1
    assert type(extent.values[0]) is llir.Var
    child = cast(llir.Var, extent.values[0])
    assert child.name == extent_name
    assert child.type is llir.DataType.INT64
    assert child.is_ptr is False
    assert child.is_restrict is False
    assert child.tensor_access is None
    dtype = _assert_torch_qualified_name(call.args[1], dtype_name)
    return extent, dtype, initializer


def test_dense_result_torch_empty_extent_is_structured_typed_and_fresh() -> None:
    tensor = TensorVar("Result", fmt="dd")

    first_statements = _result_tensor_assembler(tensor).emit_value_array_init()
    second_statements = _result_tensor_assembler(tensor).emit_value_array_init()
    first, first_dtype, first_initializer = _assert_torch_empty_extent(
        first_statements,
        extent_name="Result_capacity",
    )
    second, second_dtype, second_initializer = _assert_torch_empty_extent(
        second_statements,
        extent_name="Result_capacity",
    )
    capacity_initializer = cast(llir.VarInit, first_statements[0])

    assert first == second
    assert hash(first) == hash(second)
    assert first is not second
    assert first.values[0] is not second.values[0]
    assert first.values[0] is not capacity_initializer.var
    assert first_dtype == second_dtype
    assert hash(first_dtype) == hash(second_dtype)
    assert first_dtype is not second_dtype
    assert LLIRLowerer().lower_llir(first_initializer) == (
        "torch::Tensor Result_values_torch = "
        "torch::empty({Result_capacity}, torch::kFloat32);"
    )
    assert LLIRLowerer().lower_llir(second_initializer) == (
        "torch::Tensor Result_values_torch = "
        "torch::empty({Result_capacity}, torch::kFloat32);"
    )


def test_known_nnz_result_torch_empty_extent_is_structured_typed_and_fresh() -> None:
    tensor = TensorVar("Result", fmt="ds")

    first_statements = _result_tensor_assembler(
        tensor,
        known_nnz_var="_known_nnz",
    ).emit_value_array_init()
    second_statements = _result_tensor_assembler(
        tensor,
        known_nnz_var="_known_nnz",
    ).emit_value_array_init()
    first, first_dtype, first_initializer = _assert_torch_empty_extent(
        first_statements,
        extent_name="_known_nnz",
    )
    second, second_dtype, second_initializer = _assert_torch_empty_extent(
        second_statements,
        extent_name="_known_nnz",
    )

    assert first == second
    assert hash(first) == hash(second)
    assert first is not second
    assert first.values[0] is not second.values[0]
    assert first_dtype == second_dtype
    assert hash(first_dtype) == hash(second_dtype)
    assert first_dtype is not second_dtype
    assert LLIRLowerer().lower_llir(first_initializer) == (
        "torch::Tensor Result_values_torch = "
        "torch::empty({_known_nnz}, torch::kFloat32);"
    )
    assert LLIRLowerer().lower_llir(second_initializer) == (
        "torch::Tensor Result_values_torch = "
        "torch::empty({_known_nnz}, torch::kFloat32);"
    )


def test_known_nnz_coordinate_torch_owners_and_pointers_are_structured_and_fresh() -> (
    None
):
    tensor = TensorVar("Result", fmt="oo")
    first_statements = _result_tensor_assembler(
        tensor,
        known_nnz_var="_known_nnz",
    ).emit_level_indices_init()
    second_statements = _result_tensor_assembler(
        tensor,
        known_nnz_var="_known_nnz",
    ).emit_level_indices_init()

    def initializers(statements: list[llir.Stmt]) -> dict[str, llir.VarInit]:
        return {
            statement.var.name: statement
            for statement in statements
            if type(statement) is llir.VarInit
        }

    first = initializers(first_statements)
    second = initializers(second_statements)
    assert set(first) == {
        "Result0_crd_torch",
        "Result0_crd",
        "pResult0",
        "Result1_crd_torch",
        "Result1_crd",
        "pResult1",
    }
    assert set(second) == set(first)

    owner_children: list[tuple[llir.Var, llir.QualifiedName]] = []
    pointer_receivers: list[llir.Var] = []
    for level in range(2):
        owner_name = f"Result{level}_crd_torch"
        pointer_name = f"Result{level}_crd"
        first_owner = first[owner_name]
        second_owner = second[owner_name]
        first_pointer = first[pointer_name]
        second_pointer = second[pointer_name]

        assert first_owner == second_owner
        assert hash(first_owner) == hash(second_owner)
        assert first_owner is not second_owner
        assert first_owner.var is not second_owner.var
        assert first_owner.var.name == owner_name
        assert first_owner.var.type is llir.DataType.TORCH_TENSOR
        assert first_owner.var.is_ptr is False
        assert first_owner.var.is_restrict is False
        assert first_owner.var.tensor_access is None
        assert first_owner.op == "="
        assert first_owner.cast is False

        assert type(first_owner.value) is llir.FunctionCall
        assert type(second_owner.value) is llir.FunctionCall
        first_empty = cast(llir.FunctionCall, first_owner.value)
        second_empty = cast(llir.FunctionCall, second_owner.value)
        assert first_empty == second_empty
        assert hash(first_empty) == hash(second_empty)
        assert first_empty is not second_empty
        assert first_empty.name == "torch::empty"
        assert type(first_empty.args) is tuple
        assert len(first_empty.args) == 2
        assert type(first_empty.args[0]) is llir.Array
        assert type(second_empty.args[0]) is llir.Array
        first_extent = cast(llir.Array, first_empty.args[0])
        second_extent = cast(llir.Array, second_empty.args[0])
        assert first_extent == second_extent
        assert hash(first_extent) == hash(second_extent)
        assert first_extent is not second_extent
        assert first_extent.data_type is llir.DataType.INT64
        assert type(first_extent.values) is tuple
        assert len(first_extent.values) == 1
        assert first_extent.values[0] is not second_extent.values[0]
        assert type(first_extent.values[0]) is llir.Var
        extent = cast(llir.Var, first_extent.values[0])
        assert extent.name == "_known_nnz"
        assert extent.type is llir.DataType.INT64
        assert extent.is_ptr is False
        assert extent.is_restrict is False
        assert extent.tensor_access is None
        dtype = _assert_torch_qualified_name(first_empty.args[1], "torch::kInt")
        second_dtype = _assert_torch_qualified_name(second_empty.args[1], "torch::kInt")
        assert dtype == second_dtype
        assert hash(dtype) == hash(second_dtype)
        assert dtype is not second_dtype
        owner_children.append((extent, dtype))

        assert first_pointer == second_pointer
        assert hash(first_pointer) == hash(second_pointer)
        assert first_pointer is not second_pointer
        assert first_pointer.var is not second_pointer.var
        assert first_pointer.var.name == pointer_name
        assert first_pointer.var.type is llir.DataType.PTR_INT
        assert first_pointer.var.is_ptr is False
        assert first_pointer.var.is_restrict is False
        assert first_pointer.var.tensor_access is None
        first_data_ptr = _assert_data_ptr_call(
            first_pointer.value,
            llir.DataType.INT,
        )
        second_data_ptr = _assert_data_ptr_call(
            second_pointer.value,
            llir.DataType.INT,
        )
        assert first_data_ptr == second_data_ptr
        assert hash(first_data_ptr) == hash(second_data_ptr)
        assert first_data_ptr is not second_data_ptr
        first_receiver = _assert_torch_tensor_var(first_data_ptr.base, owner_name)
        second_receiver = _assert_torch_tensor_var(second_data_ptr.base, owner_name)
        assert first_receiver is not second_receiver
        assert first_receiver is not first_owner.var
        pointer_receivers.append(first_receiver)

        assert LLIRLowerer().lower_llir([first_owner, first_pointer]) == (
            f"torch::Tensor {owner_name} = "
            "torch::empty({_known_nnz}, torch::kInt);\n"
            f"int* {pointer_name} = {owner_name}.data_ptr<int>();"
        )

    assert len({id(child) for pair in owner_children for child in pair}) == 4
    assert len({id(receiver) for receiver in pointer_receivers}) == 2
    assert not any(
        type(statement) is llir.RawStmt and "_known_nnz" in statement.code
        for statement in first_statements
    )

    first["Result0_crd_torch"].var.name = "owned_owner"
    owner_children[0][0].name = "owned_extent"
    pointer_receivers[0].name = "owned_receiver"
    assert second["Result0_crd_torch"].var.name == "Result0_crd_torch"
    second_empty = cast(
        llir.FunctionCall,
        second["Result0_crd_torch"].value,
    )
    second_extent = cast(llir.Array, second_empty.args[0])
    assert cast(llir.Var, second_extent.values[0]).name == "_known_nnz"
    second_pointer = cast(
        llir.MemberCall,
        second["Result0_crd"].value,
    )
    assert cast(llir.Var, second_pointer.base).name == "Result0_crd_torch"
    assert first["Result1_crd_torch"].var.name == "Result1_crd_torch"
    assert owner_children[1][0].name == "_known_nnz"
    assert pointer_receivers[1].name == "Result1_crd_torch"


@pytest.mark.parametrize(
    "level_types",
    (
        pytest.param(
            (LevelType.DENSE, LevelType.COMPRESSED),
            id="ds",
        ),
        pytest.param(
            (
                LevelType.DENSE,
                LevelType.COMPRESSED,
                LevelType.COMPRESSED,
            ),
            id="dss",
        ),
    ),
)
@pytest.mark.parametrize(
    "bound_type",
    (llir.DataType.INT, llir.DataType.INT64),
    ids=("int-bound", "int64-bound"),
)
def test_first_compressed_position_allocation_is_typed_fresh_and_rewritable(
    level_types: tuple[LevelType, ...],
    bound_type: llir.DataType,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=level_types,
        dtype=torch.float32,
    )
    caller_bound = llir.Var("extent", bound_type)
    caller_offsets = llir.Var("_offset1", llir.DataType.STD_VECTOR_INT)

    first = assembler.emit_first_compressed_position_allocation(
        caller_bound,
        caller_offsets,
    )
    second = assembler.emit_first_compressed_position_allocation(
        caller_bound,
        caller_offsets,
    )

    assert [type(statement) for statement in first] == [
        llir.VarInit,
        llir.VarInit,
        llir.ForLoop,
    ]
    assert [type(statement) for statement in second] == [
        llir.VarInit,
        llir.VarInit,
        llir.ForLoop,
    ]
    owner = cast(llir.VarInit, first[0])
    pointer = cast(llir.VarInit, first[1])
    copy_loop = cast(llir.ForLoop, first[2])
    equal_owner = cast(llir.VarInit, second[0])
    equal_pointer = cast(llir.VarInit, second[1])
    equal_loop = cast(llir.ForLoop, second[2])

    assert owner == equal_owner
    assert hash(owner) == hash(equal_owner)
    assert owner is not equal_owner
    assert owner.var is not equal_owner.var
    assert owner.var.name == "Result1_pos_torch"
    assert owner.var.type is llir.DataType.TORCH_TENSOR
    assert owner.var.is_ptr is False
    assert owner.var.is_restrict is False
    assert owner.var.tensor_access is None
    assert owner.op == "="
    assert owner.cast is False
    assert type(owner.value) is llir.FunctionCall
    empty = cast(llir.FunctionCall, owner.value)
    equal_empty = cast(llir.FunctionCall, equal_owner.value)
    assert empty == equal_empty
    assert hash(empty) == hash(equal_empty)
    assert empty is not equal_empty
    assert empty.name == "torch::empty"
    assert type(empty.args) is tuple
    assert len(empty.args) == 2

    assert type(empty.args[0]) is llir.Array
    extent = cast(llir.Array, empty.args[0])
    equal_extent = cast(llir.Array, equal_empty.args[0])
    assert extent == equal_extent
    assert hash(extent) == hash(equal_extent)
    assert extent is not equal_extent
    assert extent.data_type is llir.DataType.INT64
    assert type(extent.values) is tuple
    assert len(extent.values) == 1
    assert type(extent.values[0]) is llir.Cast
    extent_cast = cast(llir.Cast, extent.values[0])
    equal_extent_cast = cast(llir.Cast, equal_extent.values[0])
    assert extent_cast == equal_extent_cast
    assert hash(extent_cast) == hash(equal_extent_cast)
    assert extent_cast is not equal_extent_cast
    assert extent_cast.data_type is llir.DataType.INT64
    assert type(extent_cast.expr) is llir.Add
    cardinality = cast(llir.Add, extent_cast.expr)
    equal_cardinality = cast(llir.Add, equal_extent_cast.expr)
    assert cardinality == equal_cardinality
    assert hash(cardinality) == hash(equal_cardinality)
    assert cardinality is not equal_cardinality
    assert cardinality.op == "+"
    assert type(cardinality.left) is llir.Var
    extent_bound = cast(llir.Var, cardinality.left)
    assert extent_bound.name == "extent"
    assert extent_bound.type is bound_type
    assert extent_bound.is_ptr is False
    assert extent_bound.is_restrict is False
    assert extent_bound.tensor_access is None
    assert extent_bound is not caller_bound
    assert type(cardinality.right) is llir.Literal
    extent_increment = cast(llir.Literal, cardinality.right)
    assert extent_increment.value == 1
    assert extent_increment.data_type is llir.DataType.INT
    dtype = _assert_torch_qualified_name(empty.args[1], "torch::kInt")
    equal_dtype = _assert_torch_qualified_name(
        equal_empty.args[1],
        "torch::kInt",
    )
    assert dtype == equal_dtype
    assert hash(dtype) == hash(equal_dtype)
    assert dtype is not equal_dtype

    assert pointer == equal_pointer
    assert hash(pointer) == hash(equal_pointer)
    assert pointer is not equal_pointer
    assert pointer.var is not equal_pointer.var
    assert pointer.var.name == "Result1_pos_data"
    assert pointer.var.type is llir.DataType.PTR_INT
    assert pointer.var.is_ptr is False
    assert pointer.var.is_restrict is False
    assert pointer.var.tensor_access is None
    assert pointer.op == "="
    assert pointer.cast is False
    data_ptr = _assert_data_ptr_call(pointer.value, llir.DataType.INT)
    equal_data_ptr = _assert_data_ptr_call(
        equal_pointer.value,
        llir.DataType.INT,
    )
    assert data_ptr == equal_data_ptr
    assert hash(data_ptr) == hash(equal_data_ptr)
    assert data_ptr is not equal_data_ptr
    receiver = _assert_torch_tensor_var(data_ptr.base, "Result1_pos_torch")
    equal_receiver = _assert_torch_tensor_var(
        equal_data_ptr.base,
        "Result1_pos_torch",
    )
    assert receiver is not owner.var
    assert receiver is not equal_receiver

    assert copy_loop is not equal_loop
    assert copy_loop.omp_parallel_for is False
    assert copy_loop.omp_schedule is None
    assert copy_loop.omp_num_threads is None
    assert copy_loop.omp_chunk_expr is None
    assert copy_loop.unroll is False
    assert copy_loop.simd is False
    assert copy_loop.before_parallel_body is None
    assert copy_loop.pre_parallel_body is None
    assert copy_loop.post_parallel_body is None
    assert copy_loop.scorch_index_var is None
    assert type(copy_loop.init) is llir.VarInit
    initialization = cast(llir.VarInit, copy_loop.init)
    assert initialization.var.name == "_i"
    assert initialization.var.type is llir.DataType.INT
    assert initialization.var.is_ptr is False
    assert initialization.var.is_restrict is False
    assert initialization.var.tensor_access is None
    assert initialization.op == "="
    assert initialization.cast is False
    assert type(initialization.value) is llir.Literal
    initial_zero = cast(llir.Literal, initialization.value)
    assert initial_zero.value == 0
    assert initial_zero.data_type is llir.DataType.INT

    assert type(copy_loop.cond) is llir.BinOp
    condition = cast(llir.BinOp, copy_loop.cond)
    assert type(condition) is llir.BinOp
    assert condition.op == "<="
    assert type(condition.left) is llir.Var
    condition_index = cast(llir.Var, condition.left)
    assert condition_index.name == "_i"
    assert condition_index.type is llir.DataType.INT
    assert type(condition.right) is llir.Var
    condition_bound = cast(llir.Var, condition.right)
    assert condition_bound.name == "extent"
    assert condition_bound.type is bound_type
    assert condition_bound.is_ptr is False
    assert condition_bound.is_restrict is False
    assert condition_bound.tensor_access is None
    assert condition_bound is not caller_bound
    assert condition_bound is not extent_bound

    assert type(copy_loop.update) is llir.Increment
    update = cast(llir.Increment, copy_loop.update)
    assert update.var.name == "_i"
    assert update.var.type is llir.DataType.INT
    assert len(copy_loop.body) == 1
    assert type(copy_loop.body[0]) is llir.Assign
    assignment = cast(llir.Assign, copy_loop.body[0])
    assert assignment.op is llir.AssignOp.ASSIGN
    assert assignment.cast is False
    assert type(assignment.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, assignment.var)
    assert target.tensor_access is None
    assert type(target.array) is llir.Var
    target_pointer = cast(llir.Var, target.array)
    assert target_pointer.name == "Result1_pos_data"
    assert target_pointer.type is llir.DataType.PTR_INT
    assert target_pointer.is_ptr is False
    assert target_pointer.is_restrict is False
    assert target_pointer.tensor_access is None
    assert target_pointer is not pointer.var
    assert type(target.index) is llir.Var
    target_index = cast(llir.Var, target.index)
    assert target_index.name == "_i"
    assert target_index.type is llir.DataType.INT

    assert type(assignment.value) is llir.Cast
    value_cast = cast(llir.Cast, assignment.value)
    assert value_cast.data_type is llir.DataType.INT
    assert type(value_cast.expr) is llir.ArrayAccess
    source_access = cast(llir.ArrayAccess, value_cast.expr)
    assert source_access.tensor_access is None
    assert type(source_access.array) is llir.Var
    source_offsets = cast(llir.Var, source_access.array)
    assert source_offsets.name == "_offset1"
    assert source_offsets.type is llir.DataType.STD_VECTOR_INT
    assert source_offsets.is_ptr is False
    assert source_offsets.is_restrict is False
    assert source_offsets.tensor_access is None
    assert source_offsets is not caller_offsets
    assert type(source_access.index) is llir.Var
    source_index = cast(llir.Var, source_access.index)
    assert source_index.name == "_i"
    assert source_index.type is llir.DataType.INT
    index_references = (
        initialization.var,
        condition_index,
        update.var,
        target_index,
        source_index,
    )
    assert len({id(index) for index in index_references}) == len(index_references)

    expected_cpp = (
        "torch::Tensor Result1_pos_torch = "
        "torch::empty({(int64_t) (extent + 1)}, torch::kInt);\n"
        "int* Result1_pos_data = Result1_pos_torch.data_ptr<int>();\n"
        "for (int _i = 0; _i <= extent; _i++) {\n"
        "  Result1_pos_data[_i] = (int) _offset1[_i];\n"
        "}"
    )
    assert LLIRLowerer().lower_llir(first) == expected_cpp
    assert LLIRLowerer().lower_llir(second) == expected_cpp

    def traversal_snapshot_and_ids(
        value: list[llir.Stmt],
        pass_name: str,
    ) -> tuple[tuple[tuple[str, ...], ...], set[int]]:
        paths: list[tuple[str, ...]] = []
        identities: set[int] = set()

        class Collector(LLIRWalker):
            def enter_node(
                self,
                node: llir.Node,
                path: tuple[str, ...],
            ) -> None:
                paths.append((*path, type(node).__name__))
                identities.add(id(node))

        Collector(LLIRTraversalContext(stage="test", pass_name=pass_name)).walk(value)
        return tuple(paths), identities

    first_snapshot, first_ids = traversal_snapshot_and_ids(
        first,
        "first_position_first",
    )
    second_snapshot, second_ids = traversal_snapshot_and_ids(
        second,
        "first_position_second",
    )
    assert first_snapshot == second_snapshot
    assert len(first_ids) == len(first_snapshot)
    assert len(second_ids) == len(second_snapshot)
    assert first_ids.isdisjoint(second_ids)
    assert id(caller_bound) not in first_ids | second_ids
    assert id(caller_offsets) not in first_ids | second_ids

    rewritten = cast(
        list[llir.Stmt],
        LLIRRewriter(
            LLIRTraversalContext(
                stage="test",
                pass_name="rewrite_first_compressed_position",
            )
        ).rewrite(first),
    )
    rewritten_snapshot, rewritten_ids = traversal_snapshot_and_ids(
        rewritten,
        "first_position_rewritten",
    )
    assert rewritten_snapshot == first_snapshot
    assert rewritten_ids.isdisjoint(first_ids)
    assert LLIRLowerer().lower_llir(rewritten) == expected_cpp

    class RenameFirstPositionReferences(LLIRRewriter):
        def rewrite_var(
            self,
            node: llir.Var,
            path: tuple[str, ...],
        ) -> llir.Var:
            rewritten_var = super().rewrite_var(node, path)
            rewritten_var.name = {
                "Result1_pos_torch": "Renamed1_pos_torch",
                "Result1_pos_data": "Renamed1_pos_data",
                "extent": "renamed_extent",
                "_offset1": "renamed_offsets",
                "_i": "renamed_i",
            }.get(node.name, node.name)
            return rewritten_var

    renamed = cast(
        list[llir.Stmt],
        RenameFirstPositionReferences(
            LLIRTraversalContext(
                stage="test",
                pass_name="rename_first_compressed_position",
            )
        ).rewrite(first),
    )
    renamed_cpp = LLIRLowerer().lower_llir(renamed)
    assert "Renamed1_pos_torch" in renamed_cpp
    assert "Renamed1_pos_data" in renamed_cpp
    assert "renamed_extent" in renamed_cpp
    assert "renamed_offsets" in renamed_cpp
    assert "renamed_i" in renamed_cpp
    assert LLIRLowerer().lower_llir(first) == expected_cpp

    caller_bound.name = "caller_owned_bound"
    caller_offsets.name = "caller_owned_offsets"
    extent_bound.name = "first_owned_bound"
    source_offsets.name = "first_owned_offsets"
    assert LLIRLowerer().lower_llir(second) == expected_cpp
    assert LLIRLowerer().lower_llir(rewritten) == expected_cpp


@pytest.mark.parametrize(
    "level_types",
    (
        (),
        (LevelType.DENSE,),
        (LevelType.COMPRESSED,),
        (LevelType.DENSE, LevelType.COORDINATE),
        (LevelType.DENSE, LevelType.COMPRESSED, LevelType.DENSE),
    ),
)
def test_first_compressed_position_allocation_rejects_other_layouts(
    level_types: tuple[LevelType, ...],
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=level_types,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="one dense level"):
        assembler.emit_first_compressed_position_allocation(
            llir.Var("extent", llir.DataType.INT64),
            llir.Var("_offset1", llir.DataType.STD_VECTOR_INT),
        )


@pytest.mark.parametrize(
    ("argument", "field", "forged_value", "expected_message"),
    (
        pytest.param(
            "bound",
            "name",
            "not.an.identifier",
            "bound must be an identifier",
            id="bound-name",
        ),
        pytest.param(
            "bound",
            "type",
            llir.DataType.FLOAT32,
            "bound must be INT or INT64",
            id="bound-type",
        ),
        pytest.param(
            "bound",
            "is_ptr",
            True,
            "bound cannot be a pointer",
            id="bound-pointer",
        ),
        pytest.param(
            "bound",
            "is_restrict",
            True,
            "bound cannot be restrict-qualified",
            id="bound-restrict",
        ),
        pytest.param(
            "bound",
            "tensor_access",
            object(),
            "bound cannot carry tensor provenance",
            id="bound-provenance",
        ),
        pytest.param(
            "offsets",
            "name",
            "other_offsets",
            "offsets must be named _offset1",
            id="offsets-name",
        ),
        pytest.param(
            "offsets",
            "type",
            llir.DataType.STD_VECTOR_C_INT,
            "offsets must have STD_VECTOR_INT type",
            id="offsets-type",
        ),
        pytest.param(
            "offsets",
            "is_ptr",
            True,
            "offsets cannot be a pointer",
            id="offsets-pointer",
        ),
        pytest.param(
            "offsets",
            "is_restrict",
            True,
            "offsets cannot be restrict-qualified",
            id="offsets-restrict",
        ),
        pytest.param(
            "offsets",
            "tensor_access",
            object(),
            "offsets cannot carry tensor provenance",
            id="offsets-provenance",
        ),
    ),
)
def test_first_compressed_position_allocation_validates_every_boundary_field(
    argument: str,
    field: str,
    forged_value: object,
    expected_message: str,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    bound = llir.Var("extent", llir.DataType.INT64)
    offsets = llir.Var("_offset1", llir.DataType.STD_VECTOR_INT)
    object.__setattr__(bound if argument == "bound" else offsets, field, forged_value)

    with pytest.raises((TypeError, ValueError), match=expected_message):
        assembler.emit_first_compressed_position_allocation(bound, offsets)


@pytest.mark.parametrize("argument", ("bound", "offsets"))
@pytest.mark.parametrize(
    "missing_field",
    ("name", "type", "is_ptr", "is_restrict", "tensor_access"),
)
def test_first_compressed_position_allocation_rejects_missing_boundary_fields(
    argument: str,
    missing_field: str,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    complete_fields = {
        "name": "extent" if argument == "bound" else "_offset1",
        "type": (
            llir.DataType.INT64 if argument == "bound" else llir.DataType.STD_VECTOR_INT
        ),
        "is_ptr": False,
        "is_restrict": False,
        "tensor_access": None,
    }
    incomplete = object.__new__(llir.Var)
    for field, value in complete_fields.items():
        if field != missing_field:
            object.__setattr__(incomplete, field, value)
    bound = (
        incomplete if argument == "bound" else llir.Var("extent", llir.DataType.INT64)
    )
    offsets = (
        incomplete
        if argument == "offsets"
        else llir.Var("_offset1", llir.DataType.STD_VECTOR_INT)
    )

    with pytest.raises(TypeError, match=f"{argument} must be a complete LLIR Var"):
        assembler.emit_first_compressed_position_allocation(bound, offsets)


@pytest.mark.parametrize("argument", ("bound", "offsets"))
def test_first_compressed_position_allocation_rejects_extra_boundary_fields(
    argument: str,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    bound = llir.Var("extent", llir.DataType.INT64)
    offsets = llir.Var("_offset1", llir.DataType.STD_VECTOR_INT)
    object.__setattr__(bound if argument == "bound" else offsets, "extra", object())

    with pytest.raises(TypeError, match=f"{argument} must be a complete LLIR Var"):
        assembler.emit_first_compressed_position_allocation(bound, offsets)


@pytest.mark.parametrize("argument", ("bound", "offsets"))
def test_first_compressed_position_allocation_rejects_boundary_subclasses(
    argument: str,
) -> None:
    class UnknownVar(llir.Var):
        pass

    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    bound = (
        UnknownVar("extent", llir.DataType.INT64)
        if argument == "bound"
        else llir.Var("extent", llir.DataType.INT64)
    )
    offsets = (
        UnknownVar("_offset1", llir.DataType.STD_VECTOR_INT)
        if argument == "offsets"
        else llir.Var("_offset1", llir.DataType.STD_VECTOR_INT)
    )

    with pytest.raises(TypeError, match=f"{argument} must be an exact LLIR Var"):
        assembler.emit_first_compressed_position_allocation(bound, offsets)


@pytest.mark.parametrize("argument", ("bound", "offsets"))
def test_first_compressed_position_allocation_rejects_non_vars(argument: str) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    bound = (
        cast(llir.Var, object())
        if argument == "bound"
        else llir.Var(
            "extent",
            llir.DataType.INT64,
        )
    )
    offsets = (
        cast(llir.Var, object())
        if argument == "offsets"
        else llir.Var(
            "_offset1",
            llir.DataType.STD_VECTOR_INT,
        )
    )

    with pytest.raises(TypeError, match=f"{argument} must be an exact LLIR Var"):
        assembler.emit_first_compressed_position_allocation(bound, offsets)


def test_first_compressed_position_allocation_rejects_assembler_subclasses() -> None:
    class ResultAssemblerSubclass(ResultTensorAssembler):
        pass

    assembler = ResultAssemblerSubclass(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )

    with pytest.raises(TypeError, match="exact ResultTensorAssembler"):
        assembler.emit_first_compressed_position_allocation(
            llir.Var("extent", llir.DataType.INT64),
            llir.Var("_offset1", llir.DataType.STD_VECTOR_INT),
        )


@pytest.mark.parametrize(
    "level_types",
    (
        pytest.param(
            (LevelType.DENSE, LevelType.COMPRESSED),
            id="ds",
        ),
        pytest.param(
            (
                LevelType.DENSE,
                LevelType.COMPRESSED,
                LevelType.COMPRESSED,
            ),
            id="dss",
        ),
    ),
)
def test_compressed_coordinate_torch_owners_are_typed_fresh_and_ordered(
    level_types: tuple[LevelType, ...],
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=level_types,
        dtype=torch.float32,
    )
    compressed_levels = tuple(range(1, len(level_types)))
    totals = tuple(
        llir.Var(f"_total{level}", llir.DataType.INT64) for level in compressed_levels
    )

    first = assembler.emit_compressed_coordinate_allocations(totals)
    second = assembler.emit_compressed_coordinate_allocations(totals)

    expected_names = [
        name
        for level in compressed_levels
        for name in (f"Result{level}_crd_torch", f"Result{level}_crd_data")
    ]
    assert [type(statement) for statement in first] == [llir.VarInit] * (
        2 * len(compressed_levels)
    )
    assert [cast(llir.VarInit, statement).var.name for statement in first] == (
        expected_names
    )
    assert first == second
    assert [hash(statement) for statement in first] == [
        hash(statement) for statement in second
    ]

    owned_children: list[llir.Node] = []
    expected_cpp: list[str] = []
    for index, level in enumerate(compressed_levels):
        owner = cast(llir.VarInit, first[2 * index])
        pointer = cast(llir.VarInit, first[2 * index + 1])
        equal_owner = cast(llir.VarInit, second[2 * index])
        equal_pointer = cast(llir.VarInit, second[2 * index + 1])
        owner_name = f"Result{level}_crd_torch"
        pointer_name = f"Result{level}_crd_data"

        assert owner is not equal_owner
        assert owner.var is not equal_owner.var
        assert owner.var.name == owner_name
        assert owner.var.type is llir.DataType.TORCH_TENSOR
        assert owner.var.is_ptr is False
        assert owner.var.is_restrict is False
        assert owner.var.tensor_access is None
        assert owner.op == "="
        assert owner.cast is False
        assert type(owner.value) is llir.FunctionCall
        empty = cast(llir.FunctionCall, owner.value)
        equal_empty = cast(llir.FunctionCall, equal_owner.value)
        assert empty.name == "torch::empty"
        assert type(empty.args) is tuple
        assert len(empty.args) == 2
        assert type(empty.args[0]) is llir.Array
        extent = cast(llir.Array, empty.args[0])
        equal_extent = cast(llir.Array, equal_empty.args[0])
        assert extent.data_type is llir.DataType.INT64
        assert type(extent.values) is tuple
        assert len(extent.values) == 1
        assert type(extent.values[0]) is llir.Var
        total = cast(llir.Var, extent.values[0])
        assert total.name == f"_total{level}"
        assert total.type is llir.DataType.INT64
        assert total.is_ptr is False
        assert total.is_restrict is False
        assert total.tensor_access is None
        assert total is not totals[index]
        dtype = _assert_torch_qualified_name(empty.args[1], "torch::kInt")
        equal_dtype = _assert_torch_qualified_name(
            equal_empty.args[1],
            "torch::kInt",
        )
        assert extent == equal_extent
        assert hash(extent) == hash(equal_extent)
        assert extent is not equal_extent
        assert total is not equal_extent.values[0]
        assert dtype == equal_dtype
        assert hash(dtype) == hash(equal_dtype)
        assert dtype is not equal_dtype

        assert pointer is not equal_pointer
        assert pointer.var is not equal_pointer.var
        assert pointer.var.name == pointer_name
        assert pointer.var.type is llir.DataType.PTR_INT
        assert pointer.var.is_ptr is False
        assert pointer.var.is_restrict is False
        assert pointer.var.tensor_access is None
        assert pointer.op == "="
        assert pointer.cast is False
        data_ptr = _assert_data_ptr_call(pointer.value, llir.DataType.INT)
        equal_data_ptr = _assert_data_ptr_call(
            equal_pointer.value,
            llir.DataType.INT,
        )
        receiver = _assert_torch_tensor_var(data_ptr.base, owner_name)
        equal_receiver = _assert_torch_tensor_var(
            equal_data_ptr.base,
            owner_name,
        )
        assert data_ptr == equal_data_ptr
        assert hash(data_ptr) == hash(equal_data_ptr)
        assert data_ptr is not equal_data_ptr
        assert receiver is not equal_receiver
        assert receiver is not owner.var
        owned_children.extend((total, dtype, receiver))
        expected_cpp.extend(
            (
                f"torch::Tensor {owner_name} = "
                f"torch::empty({{{total.name}}}, torch::kInt);",
                f"int* {pointer_name} = {owner_name}.data_ptr<int>();",
            )
        )

    assert len({id(child) for child in owned_children}) == len(owned_children)
    assert LLIRLowerer().lower_llir(first) == "\n".join(expected_cpp)
    totals[0].name = "caller_owned_total"
    first_source_owner = cast(llir.VarInit, first[0])
    first_source_empty = cast(llir.FunctionCall, first_source_owner.value)
    first_source_extent = cast(llir.Array, first_source_empty.args[0])
    assert cast(llir.Var, first_source_extent.values[0]).name == "_total1"
    first_owner = cast(llir.VarInit, first[0])
    first_empty = cast(llir.FunctionCall, first_owner.value)
    first_extent = cast(llir.Array, first_empty.args[0])
    cast(llir.Var, first_extent.values[0]).name = "owned_total"
    first_pointer = cast(llir.VarInit, first[1])
    first_call = cast(llir.MemberCall, first_pointer.value)
    cast(llir.Var, first_call.base).name = "owned_receiver"
    second_owner = cast(llir.VarInit, second[0])
    second_empty = cast(llir.FunctionCall, second_owner.value)
    second_extent = cast(llir.Array, second_empty.args[0])
    assert cast(llir.Var, second_extent.values[0]).name == "_total1"
    second_pointer = cast(llir.VarInit, second[1])
    second_call = cast(llir.MemberCall, second_pointer.value)
    assert cast(llir.Var, second_call.base).name == "Result1_crd_torch"
    if len(compressed_levels) == 2:
        sibling_owner = cast(llir.VarInit, first[2])
        sibling_empty = cast(llir.FunctionCall, sibling_owner.value)
        sibling_extent = cast(llir.Array, sibling_empty.args[0])
        assert cast(llir.Var, sibling_extent.values[0]).name == "_total2"


@pytest.mark.parametrize(
    "level_types",
    (
        (),
        (LevelType.DENSE,),
        (LevelType.COMPRESSED,),
        (LevelType.DENSE, LevelType.COORDINATE),
        (LevelType.DENSE, LevelType.COMPRESSED, LevelType.DENSE),
    ),
)
def test_compressed_coordinate_torch_owner_rejects_other_layouts(
    level_types: tuple[LevelType, ...],
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=level_types,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        assembler.emit_compressed_coordinate_allocations(())


def test_compressed_coordinate_torch_owner_rejects_invalid_total_references() -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    valid = llir.Var("_total1", llir.DataType.INT64)

    with pytest.raises(TypeError, match="immutable Var tuple"):
        assembler.emit_compressed_coordinate_allocations([valid])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="match the compressed levels"):
        assembler.emit_compressed_coordinate_allocations(())

    class UnknownTotal(llir.Var):
        pass

    invalid_totals = (
        UnknownTotal("_total1", llir.DataType.INT64),
        llir.Var("_total2", llir.DataType.INT64),
        llir.Var("_total1", llir.DataType.INT),
        llir.Var("_total1", llir.DataType.INT64, is_ptr=True),
        llir.Var("_total1", llir.DataType.INT64, is_restrict=True),
        llir.Var(
            "_total1",
            llir.DataType.INT64,
            tensor_access=cast(llir.TensorAccessMetadata, object()),
        ),
    )
    for total in invalid_totals:
        with pytest.raises((TypeError, ValueError)):
            assembler.emit_compressed_coordinate_allocations((total,))

    class ResultAssemblerSubclass(ResultTensorAssembler):
        pass

    subclass = ResultAssemblerSubclass(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    with pytest.raises(TypeError, match="exact ResultTensorAssembler"):
        subclass.emit_compressed_coordinate_allocations((valid,))


@pytest.mark.parametrize(
    "level_types",
    (
        pytest.param(
            (
                LevelType.DENSE,
                LevelType.COMPRESSED,
                LevelType.COMPRESSED,
            ),
            id="dss",
        ),
        pytest.param(
            (
                LevelType.DENSE,
                LevelType.COMPRESSED,
                LevelType.COMPRESSED,
                LevelType.COMPRESSED,
            ),
            id="dsss",
        ),
    ),
)
def test_deeper_compressed_position_allocations_are_typed_fresh_and_ordered(
    level_types: tuple[LevelType, ...],
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=level_types,
        dtype=torch.float32,
    )
    compressed_levels = tuple(range(1, len(level_types)))
    deeper_levels = compressed_levels[1:]
    totals = tuple(
        llir.Var(f"_total{level}", llir.DataType.INT64) for level in compressed_levels
    )

    first = assembler.emit_deeper_compressed_position_allocations(totals)
    second = assembler.emit_deeper_compressed_position_allocations(totals)
    coordinate_allocations = assembler.emit_compressed_coordinate_allocations(totals)

    expected_types = [
        statement_type
        for _ in deeper_levels
        for statement_type in (llir.VarInit, llir.VarInit, llir.Assign)
    ]
    expected_names = [
        name
        for level in deeper_levels
        for name in (
            f"Result{level}_pos_torch",
            f"Result{level}_pos_data",
        )
    ]
    assert [type(statement) for statement in first] == expected_types
    assert [
        cast(llir.VarInit, statement).var.name
        for statement in first
        if type(statement) is llir.VarInit
    ] == expected_names
    assert first == second

    owned_children: list[llir.Node] = []
    equal_children: list[llir.Node] = []
    expected_cpp: list[str] = []
    parent_references: list[llir.Var] = []
    for index, level in enumerate(deeper_levels):
        offset = 3 * index
        owner = cast(llir.VarInit, first[offset])
        pointer = cast(llir.VarInit, first[offset + 1])
        sentinel = cast(llir.Assign, first[offset + 2])
        equal_owner = cast(llir.VarInit, second[offset])
        equal_pointer = cast(llir.VarInit, second[offset + 1])
        equal_sentinel = cast(llir.Assign, second[offset + 2])
        owner_name = f"Result{level}_pos_torch"
        pointer_name = f"Result{level}_pos_data"
        parent_name = f"_total{level - 1}"

        assert owner == equal_owner
        assert hash(owner) == hash(equal_owner)
        assert owner is not equal_owner
        assert owner.var is not equal_owner.var
        assert owner.var.name == owner_name
        assert owner.var.type is llir.DataType.TORCH_TENSOR
        assert owner.var.is_ptr is False
        assert owner.var.is_restrict is False
        assert owner.var.tensor_access is None
        assert owner.op == "="
        assert owner.cast is False
        assert type(owner.value) is llir.FunctionCall
        assert type(equal_owner.value) is llir.FunctionCall
        empty = cast(llir.FunctionCall, owner.value)
        equal_empty = cast(llir.FunctionCall, equal_owner.value)
        assert empty == equal_empty
        assert hash(empty) == hash(equal_empty)
        assert empty is not equal_empty
        assert empty.name == "torch::empty"
        assert type(empty.args) is tuple
        assert len(empty.args) == 2
        assert type(empty.args[0]) is llir.Array
        assert type(equal_empty.args[0]) is llir.Array
        extent = cast(llir.Array, empty.args[0])
        equal_extent = cast(llir.Array, equal_empty.args[0])
        assert extent == equal_extent
        assert hash(extent) == hash(equal_extent)
        assert extent is not equal_extent
        assert extent.data_type is llir.DataType.INT64
        assert type(extent.values) is tuple
        assert len(extent.values) == 1
        assert type(extent.values[0]) is llir.Add
        assert type(equal_extent.values[0]) is llir.Add
        parent_extent = cast(llir.Add, extent.values[0])
        equal_parent_extent = cast(llir.Add, equal_extent.values[0])
        assert parent_extent == equal_parent_extent
        assert hash(parent_extent) == hash(equal_parent_extent)
        assert parent_extent is not equal_parent_extent
        assert parent_extent.op == "+"
        assert type(parent_extent.left) is llir.Var
        assert type(equal_parent_extent.left) is llir.Var
        parent_total = cast(llir.Var, parent_extent.left)
        equal_parent_total = cast(llir.Var, equal_parent_extent.left)
        assert parent_total is not equal_parent_total
        assert parent_total is not totals[index]
        assert parent_total.name == parent_name
        assert parent_total.type is llir.DataType.INT64
        assert parent_total.is_ptr is False
        assert parent_total.is_restrict is False
        assert parent_total.tensor_access is None
        assert type(parent_extent.right) is llir.Literal
        assert type(equal_parent_extent.right) is llir.Literal
        extent_sentinel = cast(llir.Literal, parent_extent.right)
        equal_extent_sentinel = cast(llir.Literal, equal_parent_extent.right)
        assert extent_sentinel is not equal_extent_sentinel
        assert extent_sentinel.value == 1
        assert extent_sentinel.data_type is llir.DataType.INT
        dtype = _assert_torch_qualified_name(empty.args[1], "torch::kInt")
        equal_dtype = _assert_torch_qualified_name(
            equal_empty.args[1],
            "torch::kInt",
        )
        assert dtype == equal_dtype
        assert hash(dtype) == hash(equal_dtype)
        assert dtype is not equal_dtype

        assert pointer == equal_pointer
        assert hash(pointer) == hash(equal_pointer)
        assert pointer is not equal_pointer
        assert pointer.var is not equal_pointer.var
        assert pointer.var.name == pointer_name
        assert pointer.var.type is llir.DataType.PTR_INT
        assert pointer.var.is_ptr is False
        assert pointer.var.is_restrict is False
        assert pointer.var.tensor_access is None
        assert pointer.op == "="
        assert pointer.cast is False
        data_ptr = _assert_data_ptr_call(pointer.value, llir.DataType.INT)
        equal_data_ptr = _assert_data_ptr_call(
            equal_pointer.value,
            llir.DataType.INT,
        )
        assert data_ptr == equal_data_ptr
        assert hash(data_ptr) == hash(equal_data_ptr)
        assert data_ptr is not equal_data_ptr
        receiver = _assert_torch_tensor_var(data_ptr.base, owner_name)
        equal_receiver = _assert_torch_tensor_var(
            equal_data_ptr.base,
            owner_name,
        )
        assert receiver is not equal_receiver
        assert receiver is not owner.var

        assert sentinel == equal_sentinel
        assert sentinel is not equal_sentinel
        assert sentinel.op is llir.AssignOp.ASSIGN
        assert sentinel.cast is False
        assert type(sentinel.var) is llir.ArrayAccess
        assert type(equal_sentinel.var) is llir.ArrayAccess
        target = cast(llir.ArrayAccess, sentinel.var)
        equal_target = cast(llir.ArrayAccess, equal_sentinel.var)
        assert target == equal_target
        assert hash(target) == hash(equal_target)
        assert target is not equal_target
        assert target.tensor_access is None
        assert type(target.array) is llir.Var
        assert type(equal_target.array) is llir.Var
        target_pointer = cast(llir.Var, target.array)
        equal_target_pointer = cast(llir.Var, equal_target.array)
        assert target_pointer is not equal_target_pointer
        assert target_pointer is not pointer.var
        assert target_pointer.name == pointer_name
        assert target_pointer.type is llir.DataType.PTR_INT
        assert target_pointer.is_ptr is False
        assert target_pointer.is_restrict is False
        assert target_pointer.tensor_access is None
        assert type(target.index) is llir.Literal
        assert type(equal_target.index) is llir.Literal
        target_index = cast(llir.Literal, target.index)
        equal_target_index = cast(llir.Literal, equal_target.index)
        assert target_index is not equal_target_index
        assert target_index.value == 0
        assert target_index.data_type is llir.DataType.INT
        assert type(sentinel.value) is llir.Literal
        assert type(equal_sentinel.value) is llir.Literal
        sentinel_value = cast(llir.Literal, sentinel.value)
        equal_sentinel_value = cast(llir.Literal, equal_sentinel.value)
        assert sentinel_value is not equal_sentinel_value
        assert sentinel_value.value == 0
        assert sentinel_value.data_type is llir.DataType.INT

        owned_children.extend(
            (
                owner.var,
                empty,
                extent,
                parent_extent,
                parent_total,
                extent_sentinel,
                dtype,
                pointer.var,
                data_ptr,
                receiver,
                target,
                target_pointer,
                target_index,
                sentinel_value,
            )
        )
        equal_children.extend(
            (
                equal_owner.var,
                equal_empty,
                equal_extent,
                equal_parent_extent,
                equal_parent_total,
                equal_extent_sentinel,
                equal_dtype,
                equal_pointer.var,
                equal_data_ptr,
                equal_receiver,
                equal_target,
                equal_target_pointer,
                equal_target_index,
                equal_sentinel_value,
            )
        )
        parent_references.append(parent_total)
        expected_cpp.extend(
            (
                f"torch::Tensor {owner_name} = "
                f"torch::empty({{{parent_name} + 1}}, torch::kInt);",
                f"int* {pointer_name} = {owner_name}.data_ptr<int>();",
                f"{pointer_name}[0] = 0;",
            )
        )

    assert len({id(child) for child in owned_children}) == len(owned_children)
    assert len({id(child) for child in equal_children}) == len(equal_children)
    assert not {id(child) for child in owned_children} & {
        id(child) for child in equal_children
    }
    assert not any(type(statement) is llir.RawStmt for statement in first)
    assert LLIRLowerer().lower_llir(first) == "\n".join(expected_cpp)

    first_coordinate_owner = cast(llir.VarInit, coordinate_allocations[0])
    first_coordinate_empty = cast(llir.FunctionCall, first_coordinate_owner.value)
    first_coordinate_extent = cast(llir.Array, first_coordinate_empty.args[0])
    coordinate_total = cast(llir.Var, first_coordinate_extent.values[0])
    assert coordinate_total is not parent_references[0]
    assert coordinate_total is not totals[0]
    totals[0].name = "caller_owned_total"
    assert coordinate_total.name == "_total1"
    assert parent_references[0].name == "_total1"
    coordinate_total.name = "coordinate_owned_total"
    assert parent_references[0].name == "_total1"
    parent_references[0].name = "position_owned_total"
    assert coordinate_total.name == "coordinate_owned_total"
    equal_first_owner = cast(llir.VarInit, second[0])
    equal_first_empty = cast(llir.FunctionCall, equal_first_owner.value)
    equal_first_extent = cast(llir.Array, equal_first_empty.args[0])
    equal_first_add = cast(llir.Add, equal_first_extent.values[0])
    assert cast(llir.Var, equal_first_add.left).name == "_total1"
    if len(deeper_levels) == 2:
        sibling_owner = cast(llir.VarInit, first[3])
        sibling_empty = cast(llir.FunctionCall, sibling_owner.value)
        sibling_extent = cast(llir.Array, sibling_empty.args[0])
        sibling_add = cast(llir.Add, sibling_extent.values[0])
        assert cast(llir.Var, sibling_add.left).name == "_total2"


@pytest.mark.parametrize(
    "level_types",
    (
        (),
        (LevelType.DENSE,),
        (LevelType.COMPRESSED,),
        (LevelType.DENSE, LevelType.COMPRESSED),
        (LevelType.COMPRESSED, LevelType.COMPRESSED, LevelType.COMPRESSED),
        (LevelType.DENSE, LevelType.COORDINATE, LevelType.COMPRESSED),
        (LevelType.DENSE, LevelType.COMPRESSED, LevelType.DENSE),
    ),
)
def test_deeper_compressed_position_allocations_reject_other_layouts(
    level_types: tuple[LevelType, ...],
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=level_types,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="one dense level"):
        assembler.emit_deeper_compressed_position_allocations(())


def test_deeper_compressed_position_allocations_reject_invalid_tuple_shape() -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(
            LevelType.DENSE,
            LevelType.COMPRESSED,
            LevelType.COMPRESSED,
        ),
        dtype=torch.float32,
    )
    totals = (
        llir.Var("_total1", llir.DataType.INT64),
        llir.Var("_total2", llir.DataType.INT64),
    )

    with pytest.raises(TypeError, match="immutable Var tuple"):
        assembler.emit_deeper_compressed_position_allocations(  # type: ignore[arg-type]
            list(totals)
        )
    with pytest.raises(ValueError, match="match the compressed levels"):
        assembler.emit_deeper_compressed_position_allocations(totals[:-1])
    with pytest.raises(ValueError, match="match the compressed levels"):
        assembler.emit_deeper_compressed_position_allocations(
            totals + (llir.Var("_total3", llir.DataType.INT64),)
        )


@pytest.mark.parametrize(
    "total_index",
    (0, 1, 2),
    ids=("used-total1", "used-total2", "unused-total3"),
)
@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        pytest.param("name", "forged_total", id="name"),
        pytest.param("type", llir.DataType.INT, id="type"),
        pytest.param("is_ptr", True, id="pointer"),
        pytest.param("is_restrict", True, id="restrict"),
        pytest.param("tensor_access", object(), id="provenance"),
    ),
)
def test_deeper_compressed_position_allocations_validate_every_total_field(
    total_index: int,
    field: str,
    forged_value: object,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(
            LevelType.DENSE,
            LevelType.COMPRESSED,
            LevelType.COMPRESSED,
            LevelType.COMPRESSED,
        ),
        dtype=torch.float32,
    )
    totals = [llir.Var(f"_total{level}", llir.DataType.INT64) for level in range(1, 4)]
    object.__setattr__(totals[total_index], field, forged_value)

    with pytest.raises((TypeError, ValueError)):
        assembler.emit_deeper_compressed_position_allocations(tuple(totals))


@pytest.mark.parametrize(
    "total_index",
    (0, 1, 2),
    ids=("used-total1", "used-total2", "unused-total3"),
)
@pytest.mark.parametrize(
    "missing_field",
    ("name", "type", "is_ptr", "is_restrict", "tensor_access"),
)
def test_deeper_compressed_position_allocations_reject_missing_total_fields(
    total_index: int,
    missing_field: str,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(
            LevelType.DENSE,
            LevelType.COMPRESSED,
            LevelType.COMPRESSED,
            LevelType.COMPRESSED,
        ),
        dtype=torch.float32,
    )
    totals = [llir.Var(f"_total{level}", llir.DataType.INT64) for level in range(1, 4)]
    missing = object.__new__(llir.Var)
    complete_fields = {
        "name": f"_total{total_index + 1}",
        "type": llir.DataType.INT64,
        "is_ptr": False,
        "is_restrict": False,
        "tensor_access": None,
    }
    for field, value in complete_fields.items():
        if field != missing_field:
            object.__setattr__(missing, field, value)
    totals[total_index] = missing

    with pytest.raises(TypeError, match="complete LLIR Vars"):
        assembler.emit_deeper_compressed_position_allocations(tuple(totals))


@pytest.mark.parametrize("total_index", (0, 1, 2))
def test_deeper_compressed_position_allocations_reject_total_subclasses(
    total_index: int,
) -> None:
    class UnknownTotal(llir.Var):
        pass

    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(
            LevelType.DENSE,
            LevelType.COMPRESSED,
            LevelType.COMPRESSED,
            LevelType.COMPRESSED,
        ),
        dtype=torch.float32,
    )
    totals = [llir.Var(f"_total{level}", llir.DataType.INT64) for level in range(1, 4)]
    totals[total_index] = UnknownTotal(
        f"_total{total_index + 1}",
        llir.DataType.INT64,
    )

    with pytest.raises(TypeError, match="exact LLIR Vars"):
        assembler.emit_deeper_compressed_position_allocations(tuple(totals))


def test_deeper_compressed_position_allocations_reject_assembler_subclasses() -> None:
    class ResultAssemblerSubclass(ResultTensorAssembler):
        pass

    assembler = ResultAssemblerSubclass(
        name="Result",
        level_types=(
            LevelType.DENSE,
            LevelType.COMPRESSED,
            LevelType.COMPRESSED,
        ),
        dtype=torch.float32,
    )
    totals = (
        llir.Var("_total1", llir.DataType.INT64),
        llir.Var("_total2", llir.DataType.INT64),
    )

    with pytest.raises(TypeError, match="exact ResultTensorAssembler"):
        assembler.emit_deeper_compressed_position_allocations(totals)


@pytest.mark.parametrize("dimension", range(1, 6), ids=lambda value: f"dim-{value}")
@pytest.mark.parametrize(
    ("scalar_type", "enum_prefix"),
    (
        pytest.param(llir.DataType.FLOAT32, "FLOAT32", id="float32"),
        pytest.param(llir.DataType.FLOAT64, "FLOAT64", id="float64"),
        pytest.param(llir.DataType.INT32, "INT32", id="int32"),
        pytest.param(llir.DataType.INT64, "INT64", id="int64"),
        pytest.param(llir.DataType.INT8, "INT8", id="int8"),
        pytest.param(llir.DataType.UINT8, "UINT8", id="uint8"),
    ),
)
def test_canonical_coo_workspace_types_cover_every_supported_dimension(
    scalar_type: llir.DataType,
    enum_prefix: str,
    dimension: int,
) -> None:
    expected = llir.DataType[f"COO_WORKSPACE_{enum_prefix}_{dimension}"]

    actual = llir.DataType.coo_workspace_type_with_dim(scalar_type, dimension)

    assert actual is expected
    assert actual.value == f"coo_workspace<{scalar_type.value}, {dimension}>"


@pytest.mark.parametrize(
    "scalar_type",
    (
        llir.DataType.FLOAT32,
        llir.DataType.FLOAT64,
        llir.DataType.INT32,
        llir.DataType.INT64,
        llir.DataType.INT8,
        llir.DataType.UINT8,
    ),
)
def test_canonical_coo_workspace_type_zero_dimension_is_the_scalar(
    scalar_type: llir.DataType,
) -> None:
    assert llir.DataType.coo_workspace_type_with_dim(scalar_type, 0) is scalar_type


@pytest.mark.parametrize("dimension", (-1, 6), ids=("negative", "too-large"))
@pytest.mark.parametrize(
    "scalar_type",
    (
        llir.DataType.FLOAT32,
        llir.DataType.FLOAT64,
        llir.DataType.INT32,
        llir.DataType.INT64,
        llir.DataType.INT8,
        llir.DataType.UINT8,
    ),
)
def test_canonical_coo_workspace_types_fail_closed_outside_supported_dimensions(
    scalar_type: llir.DataType,
    dimension: int,
) -> None:
    with pytest.raises(ValueError, match="is not a valid DataType"):
        llir.DataType.coo_workspace_type_with_dim(scalar_type, dimension)


@pytest.mark.parametrize(
    ("level_types", "leaf"),
    (
        pytest.param(
            (LevelType.DENSE, LevelType.COMPRESSED),
            1,
            id="ds",
        ),
        pytest.param(
            (
                LevelType.DENSE,
                LevelType.COMPRESSED,
                LevelType.COMPRESSED,
            ),
            2,
            id="dss",
        ),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "c_type", "pointer_type", "torch_name"),
    (
        pytest.param(
            torch.float32,
            llir.DataType.FLOAT32,
            llir.DataType.PTR_FLOAT32,
            "kFloat32",
            id="float32",
        ),
        pytest.param(
            torch.float64,
            llir.DataType.FLOAT64,
            llir.DataType.PTR_FLOAT64,
            "kFloat64",
            id="float64",
        ),
        pytest.param(
            torch.int32,
            llir.DataType.INT32,
            llir.DataType.PTR_INT_32,
            "kInt32",
            id="int32",
        ),
        pytest.param(
            torch.int64,
            llir.DataType.INT64,
            llir.DataType.PTR_INT_64,
            "kInt64",
            id="int64",
        ),
        pytest.param(
            torch.int8,
            llir.DataType.INT8,
            llir.DataType.PTR_INT8,
            "kInt8",
            id="int8",
        ),
        pytest.param(
            torch.uint8,
            llir.DataType.UINT8,
            llir.DataType.PTR_UINT8,
            "kUInt8",
            id="uint8",
        ),
    ),
)
def test_compressed_value_allocation_is_typed_fresh_and_rewritable(
    level_types: tuple[LevelType, ...],
    leaf: int,
    dtype: torch.dtype,
    c_type: llir.DataType,
    pointer_type: llir.DataType,
    torch_name: str,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=level_types,
        dtype=dtype,
    )
    caller_total = llir.Var(f"_total{leaf}", llir.DataType.INT64)

    first = assembler.emit_compressed_value_allocation(caller_total)
    second = assembler.emit_compressed_value_allocation(caller_total)

    assert [type(statement) for statement in first] == [llir.VarInit, llir.VarInit]
    assert not any(type(statement) is llir.RawStmt for statement in first)
    assert first == second
    assert [hash(statement) for statement in first] == [
        hash(statement) for statement in second
    ]

    owner = cast(llir.VarInit, first[0])
    pointer = cast(llir.VarInit, first[1])
    equal_owner = cast(llir.VarInit, second[0])
    equal_pointer = cast(llir.VarInit, second[1])
    assert owner is not equal_owner
    assert owner.var is not equal_owner.var
    assert owner.var.name == "Result_values_torch"
    assert owner.var.type is llir.DataType.TORCH_TENSOR
    assert owner.var.is_ptr is False
    assert owner.var.is_restrict is False
    assert owner.var.tensor_access is None
    assert owner.op == "="
    assert owner.cast is False
    assert type(owner.value) is llir.FunctionCall
    empty = cast(llir.FunctionCall, owner.value)
    equal_empty = cast(llir.FunctionCall, equal_owner.value)
    assert empty == equal_empty
    assert hash(empty) == hash(equal_empty)
    assert empty is not equal_empty
    assert empty.name == "torch::empty"
    assert type(empty.args) is tuple
    assert len(empty.args) == 2
    assert type(empty.args[0]) is llir.Array
    extent = cast(llir.Array, empty.args[0])
    equal_extent = cast(llir.Array, equal_empty.args[0])
    assert extent == equal_extent
    assert hash(extent) == hash(equal_extent)
    assert extent is not equal_extent
    assert extent.data_type is llir.DataType.INT64
    assert type(extent.values) is tuple
    assert len(extent.values) == 1
    assert type(extent.values[0]) is llir.Var
    total = cast(llir.Var, extent.values[0])
    equal_total = cast(llir.Var, equal_extent.values[0])
    assert total == equal_total
    assert hash(total) == hash(equal_total)
    assert total is not equal_total
    assert total is not caller_total
    assert total.name == f"_total{leaf}"
    assert total.type is llir.DataType.INT64
    assert total.is_ptr is False
    assert total.is_restrict is False
    assert total.tensor_access is None
    dtype_constant = _assert_torch_qualified_name(
        empty.args[1],
        f"torch::{torch_name}",
    )
    equal_dtype_constant = _assert_torch_qualified_name(
        equal_empty.args[1],
        f"torch::{torch_name}",
    )
    assert dtype_constant == equal_dtype_constant
    assert hash(dtype_constant) == hash(equal_dtype_constant)
    assert dtype_constant is not equal_dtype_constant

    assert pointer is not equal_pointer
    assert pointer.var is not equal_pointer.var
    assert pointer.var.name == "Result_values_data"
    assert pointer.var.type is pointer_type
    assert pointer.var.is_ptr is False
    assert pointer.var.is_restrict is False
    assert pointer.var.tensor_access is None
    assert pointer.op == "="
    assert pointer.cast is False
    assert llir.DataType.ptr_type(c_type) is pointer_type
    data_ptr = _assert_data_ptr_call(pointer.value, c_type)
    equal_data_ptr = _assert_data_ptr_call(equal_pointer.value, c_type)
    assert data_ptr == equal_data_ptr
    assert hash(data_ptr) == hash(equal_data_ptr)
    assert data_ptr is not equal_data_ptr
    receiver = _assert_torch_tensor_var(data_ptr.base, "Result_values_torch")
    equal_receiver = _assert_torch_tensor_var(
        equal_data_ptr.base,
        "Result_values_torch",
    )
    assert receiver is not owner.var
    assert receiver is not equal_receiver

    expected_cpp = (
        "torch::Tensor Result_values_torch = "
        f"torch::empty({{{total.name}}}, torch::{torch_name});\n"
        f"{c_type.value}* Result_values_data = "
        f"Result_values_torch.data_ptr<{c_type.value}>();"
    )
    assert LLIRLowerer().lower_llir(first) == expected_cpp
    assert LLIRLowerer().lower_llir(second) == expected_cpp

    def traversal_snapshot_and_ids(
        value: list[llir.Stmt],
        pass_name: str,
    ) -> tuple[tuple[tuple[str, ...], ...], set[int]]:
        paths: list[tuple[str, ...]] = []
        identities: set[int] = set()

        class Collector(LLIRWalker):
            def enter_node(
                self,
                node: llir.Node,
                path: tuple[str, ...],
            ) -> None:
                paths.append((*path, type(node).__name__))
                identities.add(id(node))

        Collector(LLIRTraversalContext(stage="test", pass_name=pass_name)).walk(value)
        return tuple(paths), identities

    first_snapshot, first_ids = traversal_snapshot_and_ids(first, "value_first")
    second_snapshot, second_ids = traversal_snapshot_and_ids(second, "value_second")
    assert first_snapshot == second_snapshot
    assert len(first_ids) == len(first_snapshot)
    assert len(second_ids) == len(second_snapshot)
    assert first_ids.isdisjoint(second_ids)
    assert id(caller_total) not in first_ids | second_ids

    rewritten = cast(
        list[llir.Stmt],
        LLIRRewriter(
            LLIRTraversalContext(stage="test", pass_name="rewrite_value_allocation")
        ).rewrite(first),
    )
    rewritten_snapshot, rewritten_ids = traversal_snapshot_and_ids(
        rewritten,
        "value_rewritten",
    )
    assert rewritten_snapshot == first_snapshot
    assert rewritten_ids.isdisjoint(first_ids)
    assert LLIRLowerer().lower_llir(rewritten) == expected_cpp

    caller_total.name = "caller_owned_total"
    total.name = "first_owned_total"
    receiver.name = "first_owned_receiver"
    assert LLIRLowerer().lower_llir(second) == expected_cpp
    assert LLIRLowerer().lower_llir(rewritten) == expected_cpp


@pytest.mark.parametrize(
    "level_types",
    (
        (),
        (LevelType.DENSE,),
        (LevelType.COMPRESSED,),
        (LevelType.DENSE, LevelType.COORDINATE),
        (LevelType.DENSE, LevelType.COMPRESSED, LevelType.DENSE),
    ),
)
def test_compressed_value_allocation_rejects_other_layouts(
    level_types: tuple[LevelType, ...],
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=level_types,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="one dense level"):
        assembler.emit_compressed_value_allocation(
            llir.Var("_total1", llir.DataType.INT64)
        )


@pytest.mark.parametrize(
    ("field", "forged_value", "expected_message"),
    (
        pytest.param(
            "name",
            "_total2",
            "name must match the leaf",
            id="name",
        ),
        pytest.param(
            "type",
            llir.DataType.INT,
            "must have INT64 type",
            id="type",
        ),
        pytest.param(
            "is_ptr",
            True,
            "cannot be a pointer",
            id="pointer",
        ),
        pytest.param(
            "is_restrict",
            True,
            "cannot be restrict-qualified",
            id="restrict",
        ),
        pytest.param(
            "tensor_access",
            object(),
            "cannot carry tensor provenance",
            id="provenance",
        ),
    ),
)
def test_compressed_value_allocation_validates_every_total_field(
    field: str,
    forged_value: object,
    expected_message: str,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    total = llir.Var("_total1", llir.DataType.INT64)
    object.__setattr__(total, field, forged_value)

    with pytest.raises((TypeError, ValueError), match=expected_message):
        assembler.emit_compressed_value_allocation(total)


@pytest.mark.parametrize(
    "missing_field",
    ("name", "type", "is_ptr", "is_restrict", "tensor_access"),
)
def test_compressed_value_allocation_rejects_missing_total_fields(
    missing_field: str,
) -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    total = object.__new__(llir.Var)
    complete_fields = {
        "name": "_total1",
        "type": llir.DataType.INT64,
        "is_ptr": False,
        "is_restrict": False,
        "tensor_access": None,
    }
    for field, value in complete_fields.items():
        if field != missing_field:
            object.__setattr__(total, field, value)

    with pytest.raises(TypeError, match="complete LLIR Var"):
        assembler.emit_compressed_value_allocation(total)


def test_compressed_value_allocation_rejects_extra_fields_and_subclasses() -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    extra = llir.Var("_total1", llir.DataType.INT64)
    object.__setattr__(extra, "forged", True)
    with pytest.raises(TypeError, match="complete LLIR Var"):
        assembler.emit_compressed_value_allocation(extra)

    class TotalSubclass(llir.Var):
        pass

    with pytest.raises(TypeError, match="exact LLIR Var"):
        assembler.emit_compressed_value_allocation(
            TotalSubclass("_total1", llir.DataType.INT64)
        )
    with pytest.raises(TypeError, match="exact LLIR Var"):
        assembler.emit_compressed_value_allocation(object())  # type: ignore[arg-type]

    class ResultAssemblerSubclass(ResultTensorAssembler):
        pass

    subclass = ResultAssemblerSubclass(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
    )
    with pytest.raises(TypeError, match="exact ResultTensorAssembler"):
        subclass.emit_compressed_value_allocation(
            llir.Var("_total1", llir.DataType.INT64)
        )


def test_compressed_value_allocation_rejects_unsupported_result_dtype() -> None:
    assembler = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="supported result dtype"):
        assembler.emit_compressed_value_allocation(
            llir.Var("_total1", llir.DataType.INT64)
        )


@pytest.mark.parametrize(
    ("child", "expected_code", "expected_path"),
    (
        pytest.param(
            "function_args",
            "invalid_function_call_args",
            ("root", "[0]", "value", "args"),
            id="function-args",
        ),
        pytest.param(
            "template_args",
            "invalid_member_call_template_args",
            ("root", "[1]", "value", "template_args"),
            id="template-args",
        ),
    ),
)
def test_compressed_value_allocation_malformed_children_fail_traversal(
    child: str,
    expected_code: str,
    expected_path: tuple[str, ...],
) -> None:
    statements = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.int8,
    ).emit_compressed_value_allocation(llir.Var("_total1", llir.DataType.INT64))
    if child == "function_args":
        owner = cast(llir.VarInit, statements[0])
        empty = cast(llir.FunctionCall, owner.value)
        object.__setattr__(empty, "args", list(empty.args))
    else:
        pointer = cast(llir.VarInit, statements[1])
        data_ptr = cast(llir.MemberCall, pointer.value)
        object.__setattr__(data_ptr, "template_args", list(data_ptr.template_args))

    with pytest.raises(LLIRTraversalError) as raised:
        LLIRWalker(
            LLIRTraversalContext(stage="test", pass_name="value_allocation")
        ).walk(statements)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == expected_code
    assert diagnostic.path == expected_path
    assert diagnostic.stage == "test"
    assert diagnostic.pass_name == "value_allocation"


@pytest.mark.parametrize(
    ("fmt", "known_nnz_var", "is_restrict"),
    (
        pytest.param("dd", None, True, id="dense"),
        pytest.param("ds", "_known_nnz", False, id="known-nnz"),
    ),
)
def test_result_value_data_ptr_producers_are_structured_typed_and_fresh(
    fmt: str,
    known_nnz_var: str | None,
    is_restrict: bool,
) -> None:
    tensor = TensorVar("Result", fmt=fmt)

    first_statements = _result_tensor_assembler(
        tensor,
        known_nnz_var=known_nnz_var,
    ).emit_value_array_init()
    second_statements = _result_tensor_assembler(
        tensor,
        known_nnz_var=known_nnz_var,
    ).emit_value_array_init()

    def pointer_initializer(statements: list[llir.Stmt]) -> llir.VarInit:
        return next(
            cast(llir.VarInit, statement)
            for statement in statements
            if type(statement) is llir.VarInit
            and cast(llir.VarInit, statement).var.name == "Result_values"
        )

    first = pointer_initializer(first_statements)
    second = pointer_initializer(second_statements)
    assert first.var.name == "Result_values"
    assert first.var.type is llir.DataType.PTR_FLOAT32
    assert first.var.is_restrict is is_restrict
    assert first.var.tensor_access is None
    first_call = _assert_data_ptr_call(first.value, llir.DataType.FLOAT32)
    second_call = _assert_data_ptr_call(second.value, llir.DataType.FLOAT32)
    first_base = _assert_torch_tensor_var(
        first_call.base,
        "Result_values_torch",
    )
    second_base = _assert_torch_tensor_var(
        second_call.base,
        "Result_values_torch",
    )

    assert first_call == second_call
    assert hash(first_call) == hash(second_call)
    assert first_call is not second_call
    assert first_base is not second_base
    assert first_base is not first.var
    assert second_base is not second.var
    assert LLIRLowerer().lower_llir(first) == (
        "float* "
        + ("__restrict__ " if is_restrict else "")
        + "Result_values = Result_values_torch.data_ptr<float>();"
    )
    assert LLIRLowerer().lower_llir(second) == LLIRLowerer().lower_llir(first)


def test_multi_compressed_mode_index_data_ptrs_are_nested_typed_and_fresh() -> None:
    tensor = TensorVar("Input", fmt="dss")
    tensor_abi = _kernel_tensor_abi(tensor)
    first_statements = tensor_abi.emit_level_array_bindings()
    second_statements = tensor_abi.emit_level_array_bindings()
    expected = (
        ("Input1_pos", 1, 0),
        ("Input1_crd", 1, 1),
        ("Input2_pos", 2, 0),
        ("Input2_crd", 2, 1),
    )

    def pointer_initializers(
        statements: list[llir.Stmt],
    ) -> dict[str, llir.VarInit]:
        return {
            statement.var.name: statement
            for statement in statements
            if type(statement) is llir.VarInit
            and type(statement.value) is llir.MemberCall
        }

    first = pointer_initializers(first_statements)
    second = pointer_initializers(second_statements)
    first_nodes: list[llir.Expr] = []
    second_nodes: list[llir.Expr] = []
    for name, level, slot in expected:
        first_initializer = first[name]
        second_initializer = second[name]
        assert first_initializer.var.type is llir.DataType.PTR_INT
        assert first_initializer.var.is_restrict is True
        assert first_initializer.var.tensor_access is None
        first_call = _assert_data_ptr_call(
            first_initializer.value,
            llir.DataType.INT,
        )
        second_call = _assert_data_ptr_call(
            second_initializer.value,
            llir.DataType.INT,
        )
        first_access, first_inner, first_root = _assert_mode_index_tensor(
            first_call.base,
            tensor_name="Input",
            level=level,
            slot=slot,
        )
        second_access, second_inner, second_root = _assert_mode_index_tensor(
            second_call.base,
            tensor_name="Input",
            level=level,
            slot=slot,
        )
        assert first_call == second_call
        assert hash(first_call) == hash(second_call)
        assert first_call is not second_call
        assert first_access is not second_access
        assert first_inner is not second_inner
        assert first_root is not second_root
        first_nodes.extend((first_call, first_access, first_inner, first_root))
        second_nodes.extend((second_call, second_access, second_inner, second_root))
        assert LLIRLowerer().lower_llir(first_initializer) == (
            f"int* __restrict__ {name} = "
            f"Input_mode_indices[{level}][{slot}].data_ptr<int>();"
        )

    assert len({id(node) for node in first_nodes}) == len(first_nodes)
    assert len({id(node) for node in second_nodes}) == len(second_nodes)
    assert {id(node) for node in first_nodes}.isdisjoint(
        {id(node) for node in second_nodes}
    )


def test_coordinate_mode_index_tensor_and_pointer_are_independently_owned() -> None:
    tensor = TensorVar("Mask", fmt="oo")
    tensor_abi = _kernel_tensor_abi(tensor)
    first_statements = tensor_abi.emit_level_array_bindings()
    second_statements = tensor_abi.emit_level_array_bindings()

    def initializers(statements: list[llir.Stmt]) -> dict[str, llir.VarInit]:
        return {
            statement.var.name: statement
            for statement in statements
            if type(statement) is llir.VarInit
        }

    first = initializers(first_statements)
    second = initializers(second_statements)
    for level in range(2):
        tensor_name = f"Mask{level}_crd_tensor"
        pointer_name = f"Mask{level}_crd"
        first_tensor = first[tensor_name]
        second_tensor = second[tensor_name]
        assert first_tensor.var.type is llir.DataType.TORCH_TENSOR
        assert first_tensor.var.is_restrict is False
        first_access, first_inner, first_root = _assert_mode_index_tensor(
            first_tensor.value,
            tensor_name="Mask",
            level=level,
            slot=0,
        )
        second_access, second_inner, second_root = _assert_mode_index_tensor(
            second_tensor.value,
            tensor_name="Mask",
            level=level,
            slot=0,
        )

        first_pointer = first[pointer_name]
        second_pointer = second[pointer_name]
        assert first_pointer.var.type is llir.DataType.PTR_INT
        assert first_pointer.var.is_restrict is True
        first_call = _assert_data_ptr_call(first_pointer.value, llir.DataType.INT)
        second_call = _assert_data_ptr_call(second_pointer.value, llir.DataType.INT)
        first_call_access, first_call_inner, first_call_root = (
            _assert_mode_index_tensor(
                first_call.base,
                tensor_name="Mask",
                level=level,
                slot=0,
            )
        )
        second_call_access, second_call_inner, second_call_root = (
            _assert_mode_index_tensor(
                second_call.base,
                tensor_name="Mask",
                level=level,
                slot=0,
            )
        )

        assert first_access == first_call_access == second_access == second_call_access
        assert hash(first_access) == hash(first_call_access) == hash(second_access)
        assert first_access is not first_call_access
        assert first_inner is not first_call_inner
        assert first_root is not first_call_root
        assert second_access is not second_call_access
        assert second_inner is not second_call_inner
        assert second_root is not second_call_root
        assert first_access is not second_access
        assert first_call_access is not second_call_access
        assert LLIRLowerer().lower_llir(first_tensor) == (
            f"torch::Tensor {tensor_name} = Mask_mode_indices[{level}][0];"
        )
        assert LLIRLowerer().lower_llir(first_pointer) == (
            f"int* __restrict__ {pointer_name} = "
            f"Mask_mode_indices[{level}][0].data_ptr<int>();"
        )


def test_input_value_data_ptr_is_structured_typed_and_fresh() -> None:
    tensor = TensorVar("Input", fmt="d", dtype=torch.float64)
    tensor_abi = _kernel_tensor_abi(tensor)
    first = tensor_abi.emit_value_pointer()
    second = tensor_abi.emit_value_pointer()

    assert first.var.name == "Input_val"
    assert first.var.type is llir.DataType.PTR_FLOAT64
    assert first.var.is_restrict is True
    assert first.var.tensor_access is None
    first_call = _assert_data_ptr_call(first.value, llir.DataType.FLOAT64)
    second_call = _assert_data_ptr_call(second.value, llir.DataType.FLOAT64)
    first_base = _assert_torch_tensor_var(first_call.base, "Input_values")
    second_base = _assert_torch_tensor_var(second_call.base, "Input_values")
    assert first_call == second_call
    assert hash(first_call) == hash(second_call)
    assert first_call is not second_call
    assert first_base is not second_base
    assert first_base is not first.var
    assert LLIRLowerer().lower_llir(first) == (
        "double* __restrict__ Input_val = Input_values.data_ptr<double>();"
    )
    assert LLIRLowerer().lower_llir(second) == LLIRLowerer().lower_llir(first)


@pytest.mark.parametrize(
    ("dtype", "vector_type", "dtype_name"),
    [
        pytest.param(
            torch.float32,
            llir.DataType.STD_VECTOR_FLOAT32,
            "torch::kFloat32",
            id="float32",
        ),
        pytest.param(
            torch.float64,
            llir.DataType.STD_VECTOR_FLOAT64,
            "torch::kFloat64",
            id="float64",
        ),
        pytest.param(
            torch.int32,
            llir.DataType.STD_VECTOR_INT32,
            "torch::kInt32",
            id="int32",
        ),
        pytest.param(
            torch.int64,
            llir.DataType.STD_VECTOR_INT,
            "torch::kInt64",
            id="int64",
        ),
        pytest.param(
            torch.int8,
            llir.DataType.STD_VECTOR_INT8,
            "torch::kInt8",
            id="int8",
        ),
        pytest.param(
            torch.uint8,
            llir.DataType.STD_VECTOR_UINT8,
            "torch::kUInt8",
            id="uint8",
        ),
    ],
)
def test_final_value_qualified_dtype_preserves_every_supported_mapping(
    dtype: torch.dtype,
    vector_type: llir.DataType,
    dtype_name: str,
) -> None:
    initializers = {
        statement.var.name: statement
        for statement in _result_tensor_assembler(
            TensorVar("Result", fmt="ds", dtype=dtype)
        ).emit_final_assembly()
        if type(statement) is llir.VarInit
    }
    _, qualified = _assert_vector_move_initialization(
        initializers["Result_values_torch"],
        target_name="Result_values_torch",
        vector_name="Result_values",
        vector_type=vector_type,
        dtype_name=dtype_name,
    )

    assert qualified.name == dtype_name.removeprefix("torch::")
    assert LLIRLowerer().lower_llir(initializers["Result_values_torch"]) == (
        "torch::Tensor Result_values_torch = "
        f"scorch_tensor_from_vector(std::move(Result_values), {dtype_name});"
    )


def test_final_result_assembly_uses_structured_typed_move_calls() -> None:
    statements = _result_tensor_assembler(
        TensorVar("Result", fmt="ds")
    ).emit_final_assembly()
    initializers = {
        statement.var.name: statement
        for statement in statements
        if type(statement) is llir.VarInit
    }

    expected = (
        (
            "Result1_pos_torch",
            "Result1_pos",
            llir.DataType.STD_VECTOR_C_INT,
            "torch::kInt",
        ),
        (
            "Result1_crd_torch",
            "Result1_crd",
            llir.DataType.STD_VECTOR_C_INT,
            "torch::kInt",
        ),
        (
            "Result_values_torch",
            "Result_values",
            llir.DataType.STD_VECTOR_FLOAT32,
            "torch::kFloat32",
        ),
    )
    move_and_dtypes = [
        _assert_vector_move_initialization(
            initializers[target_name],
            target_name=target_name,
            vector_name=vector_name,
            vector_type=vector_type,
            dtype_name=dtype_name,
        )
        for target_name, vector_name, vector_type, dtype_name in expected
    ]
    moves = [move for move, _ in move_and_dtypes]
    dtypes = [dtype for _, dtype in move_and_dtypes]

    assert [LLIRLowerer().lower_llir(initializers[name]) for name, *_ in expected] == [
        "torch::Tensor Result1_pos_torch = "
        "scorch_tensor_from_vector(std::move(Result1_pos), torch::kInt);",
        "torch::Tensor Result1_crd_torch = "
        "scorch_tensor_from_vector(std::move(Result1_crd), torch::kInt);",
        "torch::Tensor Result_values_torch = "
        "scorch_tensor_from_vector(std::move(Result_values), torch::kFloat32);",
    ]
    assert len({id(cast(llir.Var, move.args[0])) for move in moves}) == len(moves)
    assert len({id(dtype) for dtype in dtypes}) == len(dtypes)
    assert dtypes[0] == dtypes[1]
    assert hash(dtypes[0]) == hash(dtypes[1])


@pytest.mark.parametrize(
    ("fmt", "known_nnz_var", "mode_index_names"),
    (
        pytest.param("dd", None, ((), ()), id="dense"),
        pytest.param(
            "oo",
            "_known_nnz",
            (("Result0_crd_torch",), ("Result1_crd_torch",)),
            id="known-nnz-coordinate",
        ),
        pytest.param(
            "ds",
            None,
            ((), ("Result1_pos_torch", "Result1_crd_torch")),
            id="dynamic-sparse",
        ),
        pytest.param(
            "oo",
            None,
            (("Result0_crd_torch",), ("Result1_crd_torch",)),
            id="dynamic-coordinate",
        ),
        pytest.param(
            "dss",
            None,
            (
                (),
                ("Result1_pos_torch", "Result1_crd_torch"),
                ("Result2_pos_torch", "Result2_crd_torch"),
            ),
            id="multi-compressed",
        ),
    ),
)
def test_final_result_storage_assembly_is_structured_typed_and_fresh(
    fmt: str,
    known_nnz_var: str | None,
    mode_index_names: tuple[tuple[str, ...], ...],
) -> None:
    tensor = TensorVar("Result", fmt=fmt)
    first_statements = _result_tensor_assembler(
        tensor,
        known_nnz_var=known_nnz_var,
    ).emit_final_assembly()
    second_statements = _result_tensor_assembler(
        tensor,
        known_nnz_var=known_nnz_var,
    ).emit_final_assembly()

    def storage_assignments(
        statements: list[llir.Stmt],
    ) -> tuple[llir.Assign, llir.Assign, llir.Return]:
        assignments = [
            cast(llir.Assign, statement)
            for statement in statements
            if type(statement) is llir.Assign
        ]
        returns = [
            cast(llir.Return, statement)
            for statement in statements
            if type(statement) is llir.Return
        ]
        assert len(assignments) == 2
        assert len(returns) == 1
        return assignments[0], assignments[1], returns[0]

    first_modes, first_values, first_return = storage_assignments(first_statements)
    second_modes, second_values, second_return = storage_assignments(second_statements)
    first_mode_root, first_mode_chain = _assert_tensor_storage_member(
        first_modes.var,
        "Result",
        "storage",
        "index",
        "mode_indices",
    )
    second_mode_root, second_mode_chain = _assert_tensor_storage_member(
        second_modes.var,
        "Result",
        "storage",
        "index",
        "mode_indices",
    )
    first_value_root, first_value_chain = _assert_tensor_storage_member(
        first_values.var,
        "Result",
        "storage",
        "value",
    )
    second_value_root, second_value_chain = _assert_tensor_storage_member(
        second_values.var,
        "Result",
        "storage",
        "value",
    )
    first_outer, first_inner, first_children = _assert_mode_index_initializer(
        first_modes.value,
        mode_index_names,
    )
    second_outer, second_inner, second_children = _assert_mode_index_initializer(
        second_modes.value,
        mode_index_names,
    )
    first_value = _assert_torch_tensor_var(
        first_values.value,
        "Result_values_torch",
    )
    second_value = _assert_torch_tensor_var(
        second_values.value,
        "Result_values_torch",
    )

    assert first_modes.var == second_modes.var
    assert hash(first_modes.var) == hash(second_modes.var)
    assert first_values.var == second_values.var
    assert hash(first_values.var) == hash(second_values.var)
    assert first_outer == second_outer
    assert hash(first_outer) == hash(second_outer)
    assert first_modes.var is not second_modes.var
    assert first_values.var is not second_values.var
    assert first_mode_root is not second_mode_root
    assert first_value_root is not second_value_root
    assert first_mode_root is not first_value_root
    assert second_mode_root is not second_value_root
    assert {id(node) for node in first_mode_chain + first_value_chain}.isdisjoint(
        {id(node) for node in second_mode_chain + second_value_chain}
    )
    assert first_outer is not second_outer
    assert all(first is not second for first, second in zip(first_inner, second_inner))
    assert all(
        first is not second for first, second in zip(first_children, second_children)
    )
    assert len({id(node) for node in first_inner}) == len(first_inner)
    assert len({id(node) for node in first_children}) == len(first_children)
    assert first_value is not second_value

    expected_initializer = (
        "{"
        + ", ".join("{" + ", ".join(names) + "}" for names in mode_index_names)
        + "}"
    )
    assert LLIRLowerer().lower_llir(first_modes) == (
        "Result.storage.index.mode_indices = " + expected_initializer + ";"
    )
    assert LLIRLowerer().lower_llir(first_values) == (
        "Result.storage.value = Result_values_torch;"
    )
    assert type(first_return.value) is llir.Var
    assert cast(llir.Var, first_return.value).name == "Result"
    assert cast(llir.Var, first_return.value).type is llir.DataType.TACO_TENSOR
    assert LLIRLowerer().lower_llir(first_return) == "return Result;"
    assert LLIRLowerer().lower_llir(second_modes) == LLIRLowerer().lower_llir(
        first_modes
    )
    assert LLIRLowerer().lower_llir(second_values) == LLIRLowerer().lower_llir(
        first_values
    )
    assert LLIRLowerer().lower_llir(second_return) == "return Result;"


def test_result_storage_epilogue_is_the_shared_validated_abi_boundary() -> None:
    assembler = _result_tensor_assembler(TensorVar("Result", fmt="dss"))

    declaration = assembler.emit_result_declaration()
    epilogue = assembler.emit_storage_epilogue()
    full_assembly = assembler.emit_final_assembly()

    assert [type(statement) for statement in epilogue] == [
        llir.Assign,
        llir.Assign,
        llir.Return,
    ]
    assert LLIRLowerer().lower_llir(declaration) == "Tensor Result;"
    assert LLIRLowerer().lower_llir(epilogue) == LLIRLowerer().lower_llir(
        full_assembly[-3:]
    )
    assert declaration is not full_assembly[1]
    assert all(
        shared is not ordinary for shared, ordinary in zip(epilogue, full_assembly[-3:])
    )
    assert not any(type(statement) is llir.VarInit for statement in epilogue)

    object.__setattr__(assembler, "level_types", (LevelType.DENSE, "compressed"))
    with pytest.raises(TypeError, match="immutable LevelType tuple"):
        assembler.validate()
    with pytest.raises(TypeError, match="immutable LevelType tuple"):
        assembler.emit_result_declaration()
    with pytest.raises(TypeError, match="immutable LevelType tuple"):
        assembler.emit_storage_epilogue()
    with pytest.raises(TypeError, match="immutable LevelType tuple"):
        assembler.emit_final_assembly()


def test_tensor_storage_value_read_is_structured_typed_and_fresh() -> None:
    tensor = TensorVar("Input", fmt="ds")
    first = cast(llir.VarInit, CINLowerer.get_value_array_statement(tensor))
    second = cast(llir.VarInit, CINLowerer.get_value_array_statement(tensor))

    assert first.var.name == "Input_values"
    assert first.var.type is llir.DataType.TORCH_TENSOR
    assert first.var.tensor_access is None
    first_root, first_chain = _assert_tensor_storage_member(
        first.value,
        "Input",
        "storage",
        "value",
    )
    second_root, second_chain = _assert_tensor_storage_member(
        second.value,
        "Input",
        "storage",
        "value",
    )
    assert first.value == second.value
    assert hash(first.value) == hash(second.value)
    assert first.value is not second.value
    assert first_root is not second_root
    assert all(first is not second for first, second in zip(first_chain, second_chain))
    assert first_root is not first.var
    assert LLIRLowerer().lower_llir(first) == (
        "torch::Tensor Input_values = Input.storage.value;"
    )
    assert LLIRLowerer().lower_llir(second) == LLIRLowerer().lower_llir(first)


@pytest.mark.parametrize(
    ("dtype", "scalar_type", "pointer_type", "workspace_type", "torch_name"),
    (
        pytest.param(
            torch.int8,
            llir.DataType.INT8,
            llir.DataType.PTR_INT8,
            llir.DataType.COO_WORKSPACE_INT8_2,
            "kInt8",
            id="int8",
        ),
        pytest.param(
            torch.uint8,
            llir.DataType.UINT8,
            llir.DataType.PTR_UINT8,
            llir.DataType.COO_WORKSPACE_UINT8_2,
            "kUInt8",
            id="uint8",
        ),
    ),
)
def test_production_compressed_where_lowers_narrow_integer_workspace_types(
    dtype: torch.dtype,
    scalar_type: llir.DataType,
    pointer_type: llir.DataType,
    workspace_type: llir.DataType,
    torch_name: str,
) -> None:
    statement, _ = _build_outer_workspace_statement("ds", dtype)
    original = str(statement)

    lowered = CINLowerer().lower_IndexStmt(statement)

    assert type(lowered) is llir.Function
    initializers: list[llir.VarInit] = []

    class InitializerCollector(LLIRWalker):
        def visit_var_init(
            self,
            node: llir.VarInit,
            path: tuple[str, ...],
        ) -> None:
            initializers.append(node)
            super().visit_var_init(node, path)

    InitializerCollector(
        LLIRTraversalContext(
            stage="test",
            pass_name="collect_narrow_integer_where_initializers",
        )
    ).walk(lowered)
    by_name = {initializer.var.name: initializer for initializer in initializers}

    workspace = by_name["wksp"]
    assert workspace.var.type is workspace_type
    assert type(workspace.value) is llir.FunctionCall
    workspace_call = cast(llir.FunctionCall, workspace.value)
    assert workspace_call.name == f"coo_workspace<{scalar_type.value}, 2>"
    assert type(workspace_call.args) is tuple
    assert len(workspace_call.args) == 2

    for pointer_name in ("Left_val", "Right_val", "T_val"):
        pointer = by_name[pointer_name]
        assert pointer.var.type is pointer_type
        data_ptr = _assert_data_ptr_call(pointer.value, scalar_type)
        assert type(data_ptr.base) is llir.Var

    cpp = LLIRLowerer().lower_llir(lowered)
    assert (
        f"coo_workspace<{scalar_type.value}, 2> wksp = "
        f"coo_workspace<{scalar_type.value}, 2>(1024, result_shape);"
    ) in cpp
    assert (
        "torch::Tensor Result_values_torch = "
        "scorch_tensor_from_vector(std::move(Result_values), "
        f"torch::{torch_name});"
    ) in cpp
    assert str(statement) == original


def test_all_coo_workspace_assembly_targets_use_storage_types() -> None:
    statement, result = _build_outer_workspace_statement("oo")
    lowerer = CINLowerer()
    lowerer.outermost_stmt = statement
    lowerer.result_tensor_var = result
    lowerer.result_tensor_access = statement.consumer.get_result_tensor_accesses()[0]
    lowered = lowerer.lower_outer_ConsumerIndexStmt(statement.consumer)
    workspace_loop = next(
        cast(llir.ForLoopAuto, node)
        for node in lowered
        if type(node) is llir.ForLoopAuto
    )

    workspace_assignments = [
        cast(llir.Assign, node)
        for node in workspace_loop.body
        if type(node) is llir.Assign
    ]
    assert [LLIRLowerer().lower_llir(node) for node in workspace_assignments] == [
        "Result0_crd[pResult0] = it.first[0];",
        "Result1_crd[pResult1] = it.first[1];",
        "Result_values[pResult0] = it.second;",
    ]
    pair_bases = [
        _assert_workspace_pair_read(node.value, member, index)
        for node, member, index in zip(
            workspace_assignments,
            ("first", "first", "second"),
            (0, 1, None),
        )
    ]
    assert len({id(base) for base in pair_bases}) == len(pair_bases)
    assert all(base is not workspace_loop.var for base in pair_bases)

    storage_types = {
        cast(llir.Var, cast(llir.ArrayAccess, node.var).array)
        .name: cast(llir.Var, cast(llir.ArrayAccess, node.var).array)
        .type
        for node in workspace_assignments
    }
    assert storage_types == {
        "Result0_crd": llir.DataType.STD_VECTOR_C_INT,
        "Result1_crd": llir.DataType.STD_VECTOR_C_INT,
        "Result_values": llir.DataType.STD_VECTOR_FLOAT32,
    }

    lowered_function = CINLowerer().lower_IndexStmt(statement)
    dynamic_vector_assignments: list[llir.Assign] = []

    class DynamicVectorAssignmentCollector(LLIRWalker):
        def visit_assign(self, node: llir.Assign, path: tuple[str, ...]) -> None:
            dynamic_vector_assignments.append(node)
            super().visit_assign(node, path)

    DynamicVectorAssignmentCollector(
        LLIRTraversalContext(
            stage="test",
            pass_name="collect_dynamic_vector_assignments",
        )
    ).walk(lowered_function)
    assert not any(
        type(node.var) is llir.ArrayAccess
        and cast(llir.Var, cast(llir.ArrayAccess, node.var).array).name
        in {"Result0_crd", "Result1_crd", "Result_values"}
        for node in dynamic_vector_assignments
    )


def test_outer_workspace_intermediate_reads_use_structured_pair_members() -> None:
    statement, result = _build_outer_workspace_statement("ds")
    original = str(statement)
    lowerer = CINLowerer()
    lowerer.outermost_stmt = statement
    lowerer.result_tensor_var = result
    lowerer.result_tensor_access = statement.consumer.get_result_tensor_accesses()[0]

    lowered = lowerer.lower_outer_ConsumerIndexStmt(statement.consumer)
    workspace_loop = next(
        cast(llir.ForLoopAuto, node)
        for node in lowered
        if type(node) is llir.ForLoopAuto
    )
    workspace_assignments = [
        cast(llir.Assign, node)
        for node in workspace_loop.body
        if type(node) is llir.Assign
    ]

    assert [LLIRLowerer().lower_llir(node) for node in workspace_assignments] == [
        "T0_crd_vec[pT] = it.first[0];",
        "T1_crd_vec[pT] = it.first[1];",
        "T_val_vec[pT] = it.second;",
    ]
    pair_bases = [
        _assert_workspace_pair_read(node.value, member, index)
        for node, member, index in zip(
            workspace_assignments,
            ("first", "first", "second"),
            (0, 1, None),
        )
    ]
    assert len({id(base) for base in pair_bases}) == len(pair_bases)
    assert all(base is not workspace_loop.var for base in pair_bases)

    assembly_initializers = {
        node.var.name: node
        for node in lowered
        if type(node) is llir.VarInit
        and type(node.value) is llir.FunctionCall
        and cast(llir.FunctionCall, node.value).name == "scorch_tensor_from_vector"
    }
    expected_moves = (
        (
            "T0_crd_tensor",
            "T0_crd_vec",
            llir.DataType.STD_VECTOR_C_INT,
            "torch::kInt",
        ),
        (
            "T1_crd_tensor",
            "T1_crd_vec",
            llir.DataType.STD_VECTOR_C_INT,
            "torch::kInt",
        ),
        (
            "T_val_tensor",
            "T_val_vec",
            llir.DataType.STD_VECTOR_FLOAT32,
            "torch::kFloat32",
        ),
    )
    move_and_dtypes = [
        _assert_vector_move_initialization(
            assembly_initializers[target_name],
            target_name=target_name,
            vector_name=vector_name,
            vector_type=vector_type,
            dtype_name=dtype_name,
        )
        for target_name, vector_name, vector_type, dtype_name in expected_moves
    ]
    moves = [move for move, _ in move_and_dtypes]
    dtypes = [dtype for _, dtype in move_and_dtypes]
    assert [
        LLIRLowerer().lower_llir(assembly_initializers[name])
        for name, *_ in expected_moves
    ] == [
        "torch::Tensor T0_crd_tensor = "
        "scorch_tensor_from_vector(std::move(T0_crd_vec), torch::kInt);",
        "torch::Tensor T1_crd_tensor = "
        "scorch_tensor_from_vector(std::move(T1_crd_vec), torch::kInt);",
        "torch::Tensor T_val_tensor = "
        "scorch_tensor_from_vector(std::move(T_val_vec), torch::kFloat32);",
    ]
    assert len({id(cast(llir.Var, move.args[0])) for move in moves}) == len(moves)
    assert len({id(dtype) for dtype in dtypes}) == len(dtypes)
    assert dtypes[0] == dtypes[1]
    assert hash(dtypes[0]) == hash(dtypes[1])
    assert str(statement) == original


def test_intermediate_tensor_data_ptr_producers_are_typed_and_fresh() -> None:
    statement, result = _build_outer_workspace_statement("ds")
    original = str(statement)

    def lower() -> list[llir.Stmt]:
        lowerer = CINLowerer()
        lowerer.outermost_stmt = statement
        lowerer.result_tensor_var = result
        lowerer.result_tensor_access = statement.consumer.get_result_tensor_accesses()[
            0
        ]
        return lowerer.lower_outer_ConsumerIndexStmt(statement.consumer)

    first_statements = lower()
    second_statements = lower()

    def initializers(statements: list[llir.Stmt]) -> dict[str, llir.VarInit]:
        return {
            statement.var.name: statement
            for statement in statements
            if type(statement) is llir.VarInit
        }

    first = initializers(first_statements)
    second = initializers(second_statements)
    expected = (
        ("T0_crd", "T0_crd_tensor", llir.DataType.PTR_INT, llir.DataType.INT),
        ("T1_crd", "T1_crd_tensor", llir.DataType.PTR_INT, llir.DataType.INT),
        (
            "T_val",
            "T_val_tensor",
            llir.DataType.PTR_FLOAT32,
            llir.DataType.FLOAT32,
        ),
    )
    for name, receiver_name, pointer_type, data_type in expected:
        first_initializer = first[name]
        second_initializer = second[name]
        assert first_initializer.var.type is pointer_type
        assert first_initializer.var.is_restrict is False
        assert first_initializer.var.tensor_access is None
        first_call = _assert_data_ptr_call(first_initializer.value, data_type)
        second_call = _assert_data_ptr_call(second_initializer.value, data_type)
        first_base = _assert_torch_tensor_var(first_call.base, receiver_name)
        second_base = _assert_torch_tensor_var(second_call.base, receiver_name)
        assert first_call == second_call
        assert hash(first_call) == hash(second_call)
        assert first_call is not second_call
        assert first_base is not second_base
        assert first_base is not first[receiver_name].var
        assert second_base is not second[receiver_name].var

    assert LLIRLowerer().lower_llir(first["T0_crd"]) == (
        "int* T0_crd = T0_crd_tensor.data_ptr<int>();"
    )
    assert LLIRLowerer().lower_llir(first["T1_crd"]) == (
        "int* T1_crd = T1_crd_tensor.data_ptr<int>();"
    )
    assert LLIRLowerer().lower_llir(first["T_val"]) == (
        "float* T_val = T_val_tensor.data_ptr<float>();"
    )
    assert str(statement) == original


def test_move_calls_and_qualified_names_survive_with_independent_ownership() -> None:
    statement, _ = _build_outer_workspace_statement("ds")
    original = str(statement)

    first = CINLowerer().lower_IndexStmt(statement)
    second = CINLowerer().lower_IndexStmt(statement)

    assert type(first) is llir.Function
    assert type(second) is llir.Function
    first_calls = _collect_move_calls(cast(llir.Function, first))
    second_calls = _collect_move_calls(cast(llir.Function, second))
    first_qualified = _collect_qualified_names(cast(llir.Function, first))
    second_qualified = _collect_qualified_names(cast(llir.Function, second))
    expected = [
        ("T0_crd_vec", llir.DataType.STD_VECTOR_C_INT),
        ("T1_crd_vec", llir.DataType.STD_VECTOR_C_INT),
        ("T_val_vec", llir.DataType.STD_VECTOR_FLOAT32),
        ("Result1_pos", llir.DataType.STD_VECTOR_C_INT),
        ("Result1_crd", llir.DataType.STD_VECTOR_C_INT),
        ("Result_values", llir.DataType.STD_VECTOR_FLOAT32),
    ]

    def call_values(
        calls: list[llir.FunctionCall],
    ) -> list[tuple[str, llir.DataType]]:
        return [
            (cast(llir.Var, call.args[0]).name, cast(llir.Var, call.args[0]).type)
            for call in calls
        ]

    assert call_values(first_calls) == call_values(second_calls) == expected
    # The two leading constants are the typed validation-call dtype arguments.
    expected_qualified = [
        "torch::kFloat32",
        "torch::kFloat32",
        "torch::kInt",
        "torch::kInt",
        "torch::kFloat32",
        "torch::kInt",
        "torch::kInt",
        "torch::kFloat32",
    ]
    assert (
        [LLIRLowerer().lower_llir(node) for node in first_qualified]
        == ([LLIRLowerer().lower_llir(node) for node in second_qualified])
        == expected_qualified
    )
    assert all(
        node.data_type is llir.DataType.TORCH_SCALAR_TYPE
        for node in first_qualified + second_qualified
    )
    assert LLIRLowerer().lower_llir(first) == LLIRLowerer().lower_llir(second)
    assert str(statement) == original
    assert {id(cast(llir.Var, call.args[0])) for call in first_calls}.isdisjoint(
        {id(cast(llir.Var, call.args[0])) for call in second_calls}
    )
    assert {id(node) for node in first_qualified}.isdisjoint(
        {id(node) for node in second_qualified}
    )

    cast(llir.Var, first_calls[0].args[0]).name = "owned_by_first"
    assert call_values(second_calls) == expected
    with pytest.raises(FrozenInstanceError):
        first_qualified[0].name = "owned_by_first"
    assert [LLIRLowerer().lower_llir(node) for node in second_qualified] == (
        expected_qualified
    )
    assert str(statement) == original


@pytest.mark.parametrize(
    ("result_format", "index_names", "expected_initializers"),
    [
        pytest.param(
            "s",
            ("i",),
            ("int64_t i = it.first;", "float wksp_value = it.second;"),
            id="rank-one",
        ),
        pytest.param(
            "oo",
            ("i", "j"),
            (
                "int64_t i = it.first[0];",
                "int64_t j = it.first[1];",
                "float wksp_value = it.second;",
            ),
            id="rank-two",
        ),
    ],
)
def test_nested_workspace_reads_use_structured_pair_members(
    result_format: str,
    index_names: tuple[str, ...],
    expected_initializers: tuple[str, ...],
) -> None:
    index_vars = tuple(IndexVar(name) for name in index_names)
    result = TensorVar("Result", fmt=result_format)
    workspace = Workspace("wksp", dim=len(index_vars))
    access_key = index_vars[0] if len(index_vars) == 1 else index_vars
    consumer = TensorAssign(result[access_key], workspace[access_key])
    original = str(consumer)
    lowerer = CINLowerer()
    lowerer.outermost_stmt = ForAll(IndexVar("outer"), consumer)

    lowered = lowerer.lower_ConsumerIndexStmt(consumer)
    workspace_loop = next(
        cast(llir.ForLoopAuto, node)
        for node in lowered
        if type(node) is llir.ForLoopAuto
    )
    initializers = [
        cast(llir.VarInit, node)
        for node in workspace_loop.body
        if type(node) is llir.VarInit
    ][: len(index_vars) + 1]

    assert [LLIRLowerer().lower_llir(node) for node in initializers] == list(
        expected_initializers
    )
    expected_reads: list[tuple[str, int | None]] = (
        [("first", None)]
        if len(index_vars) == 1
        else [("first", index) for index in range(len(index_vars))]
    )
    expected_reads.append(("second", None))
    pair_bases = [
        _assert_workspace_pair_read(node.value, member, index)
        for node, (member, index) in zip(initializers, expected_reads)
    ]
    assert len({id(base) for base in pair_bases}) == len(pair_bases)
    assert all(base is not workspace_loop.var for base in pair_bases)
    assert str(consumer) == original


@pytest.mark.parametrize(
    ("level_type", "fmt"),
    (
        pytest.param(LevelType.COORDINATE, "oo", id="coordinate"),
        pytest.param(LevelType.COMPRESSED, "ds", id="compressed"),
    ),
)
def test_sparse_mode_iterator_coordinate_reads_are_structured_typed_and_owned(
    level_type: LevelType,
    fmt: str,
) -> None:
    tensor = TensorVar("Input", fmt=fmt)
    index = IndexVar("column")
    parent = IndexVar("row")
    tensor_state = (
        tensor.name,
        tensor.symbol_id,
        tensor.format,
        tensor.shape,
        tensor.dtype,
        tuple(tensor.mode_order or ()),
        tensor._assignment,
    )

    def index_state(value: IndexVar) -> tuple[object, ...]:
        return (
            value.name,
            value.index_id,
            value._expr,
            value._parent,
            value.is_tiled,
            value.is_outer,
            value.is_inner,
            value.tile_size_var,
            tuple(value.tensor_accesses),
        )

    index_before = index_state(index)
    parent_before = index_state(parent)

    first_iterator = ModeIterator(
        _tensor_var=tensor,
        index_var=index,
        parent_index_var=parent,
        _level=1,
        level_type=level_type,
    )
    second_iterator = ModeIterator(
        _tensor_var=tensor,
        index_var=index,
        parent_index_var=parent,
        _level=1,
        level_type=level_type,
    )
    first = first_iterator.get_coord_var_value_llir()
    second = second_iterator.get_coord_var_value_llir()

    expected = llir.ArrayAccess(
        array=llir.Var("Input1_crd", llir.DataType.PTR_INT),
        index=llir.Var("pInput1", llir.DataType.INT),
    )
    assert type(first) is llir.ArrayAccess
    assert type(second) is llir.ArrayAccess
    first_access = cast(llir.ArrayAccess, first)
    second_access = cast(llir.ArrayAccess, second)
    assert first_access == expected == second_access
    assert hash(first_access) == hash(expected) == hash(second_access)
    assert first_access.tensor_access is None
    assert second_access.tensor_access is None
    assert type(first_access.array) is llir.Var
    assert type(first_access.index) is llir.Var
    first_array = cast(llir.Var, first_access.array)
    first_index = cast(llir.Var, first_access.index)
    assert first_array.name == "Input1_crd"
    assert first_array.type is llir.DataType.PTR_INT
    assert first_array.tensor_access is None
    assert first_index.name == "pInput1"
    assert first_index.type is llir.DataType.INT
    assert first_index.tensor_access is None

    assert first_access is not second_access
    assert first_access.array is not second_access.array
    assert first_access.index is not second_access.index
    assert first_iterator.iterator_var_llir is not second_iterator.iterator_var_llir
    assert first_access.index is not first_iterator.iterator_var_llir
    assert second_access.index is not second_iterator.iterator_var_llir
    assert LLIRLowerer().lower_llir(first_access) == "Input1_crd[pInput1]"
    with pytest.raises(FrozenInstanceError):
        first_access.index = llir.Var("other", llir.DataType.INT)

    assert first_iterator.tensor_var is tensor
    assert first_iterator.index_var is index
    assert first_iterator.parent_index_var is parent
    assert tensor_state == (
        tensor.name,
        tensor.symbol_id,
        tensor.format,
        tensor.shape,
        tensor.dtype,
        tuple(tensor.mode_order or ()),
        tensor._assignment,
    )
    assert index_state(index) == index_before
    assert index_state(parent) == parent_before


@pytest.mark.parametrize("with_parent", (False, True), ids=("root", "parent"))
def test_compressed_mode_iterator_position_bounds_are_structured_typed_and_owned(
    with_parent: bool,
) -> None:
    tensor = TensorVar("Input", fmt="ds" if with_parent else "s")
    index = IndexVar("column")
    parent = IndexVar("row") if with_parent else None
    tensor_access = tensor[parent, index] if parent is not None else tensor[index]
    tensor_state = (
        tensor.name,
        tensor.symbol_id,
        tensor.format,
        tensor.shape,
        tensor.dtype,
        tuple(tensor.mode_order or ()),
        tensor._assignment,
    )

    def index_state(value: IndexVar) -> tuple[object, ...]:
        return (
            value.name,
            value.index_id,
            value._expr,
            value._parent,
            value.is_tiled,
            value.is_outer,
            value.is_inner,
            value.tile_size_var,
            tuple(value.tensor_accesses),
        )

    index_before = index_state(index)
    parent_before = index_state(parent) if parent is not None else None
    access_state = (
        tensor_access.access_id,
        tensor_access.tensor,
        tensor_access.tensor_id,
        tuple(tensor_access.indices),
        tensor_access.index_ids,
    )

    def build() -> ModeIterator:
        return ModeIterator(
            tensor_access=tensor_access,
            index_var=index,
        )

    first_iterator = build()
    second_iterator = build()
    first_begin = first_iterator.get_iterator_var_begin_value_llir()
    first_end = first_iterator.get_iterator_var_end_value_llir()
    second_begin = second_iterator.get_iterator_var_begin_value_llir()
    second_end = second_iterator.get_iterator_var_end_value_llir()
    level = 1 if with_parent else 0
    array_name = f"Input{level}_pos"
    expected_match: tuple[str, str | None]
    if with_parent:
        expected_begin = llir.ArrayAccess(
            llir.Var(array_name, llir.DataType.PTR_INT),
            llir.Var("pInput0", llir.DataType.INT),
        )
        expected_end = llir.ArrayAccess(
            llir.Var(array_name, llir.DataType.PTR_INT),
            llir.Add(
                llir.Var("pInput0", llir.DataType.INT),
                llir.Literal(1, llir.DataType.INT),
            ),
        )
        expected_begin_cpp = "Input1_pos[pInput0]"
        expected_end_cpp = "Input1_pos[pInput0 + 1]"
        expected_match = ("Input1_pos", "pInput0")
    else:
        expected_begin = llir.ArrayAccess(
            llir.Var(array_name, llir.DataType.PTR_INT),
            llir.Literal(0, llir.DataType.INT),
        )
        expected_end = llir.ArrayAccess(
            llir.Var(array_name, llir.DataType.PTR_INT),
            llir.Literal(1, llir.DataType.INT),
        )
        expected_begin_cpp = "Input0_pos[0]"
        expected_end_cpp = "Input0_pos[1]"
        expected_match = ("Input0_pos", None)

    assert type(first_begin) is llir.ArrayAccess
    assert type(first_end) is llir.ArrayAccess
    assert first_begin == expected_begin == second_begin
    assert first_end == expected_end == second_end
    assert hash(first_begin) == hash(expected_begin) == hash(second_begin)
    assert hash(first_end) == hash(expected_end) == hash(second_end)
    assert match_mode_position_begin(first_begin) == expected_match
    assert match_mode_position_access(first_begin) == array_name
    assert match_mode_position_access(first_end) == array_name
    matched_begin = match_mode_position_bounds(first_begin, first_end)
    assert type(matched_begin) is llir.ArrayAccess
    assert matched_begin == expected_begin
    assert matched_begin is not first_begin
    assert matched_begin.array is not first_begin.array
    assert matched_begin.index is not first_begin.index
    assert LLIRLowerer().lower_llir(first_begin) == expected_begin_cpp
    assert LLIRLowerer().lower_llir(first_end) == expected_end_cpp
    assert LLIRLowerer().lower_llir(matched_begin) == expected_begin_cpp

    accesses = [
        cast(llir.ArrayAccess, value)
        for value in (first_begin, first_end, second_begin, second_end)
    ]
    assert all(access.tensor_access is None for access in accesses)
    assert all(type(access.array) is llir.Var for access in accesses)
    arrays = [cast(llir.Var, access.array) for access in accesses]
    assert all(array.name == array_name for array in arrays)
    assert all(array.type is llir.DataType.PTR_INT for array in arrays)
    assert all(array.is_ptr is False for array in arrays)
    assert all(array.is_restrict is False for array in arrays)
    assert all(array.tensor_access is None for array in arrays)
    assert len({id(access) for access in accesses}) == 4
    assert len({id(array) for array in arrays}) == 4
    assert first_begin is not first_end
    assert first_begin is not second_begin
    assert first_end is not second_end
    assert first_begin.array is not first_end.array
    assert first_begin.index is not first_end.index
    assert first_begin.index is not first_iterator.iterator_var_llir
    assert first_end.index is not first_iterator.iterator_var_llir
    assert second_begin.index is not second_iterator.iterator_var_llir
    assert second_end.index is not second_iterator.iterator_var_llir

    def owned_children(
        begin: llir.ArrayAccess,
        end: llir.ArrayAccess,
    ) -> list[llir.Expr]:
        children: list[llir.Expr] = [
            begin.array,
            begin.index,
            end.array,
            end.index,
        ]
        if type(end.index) is llir.Add:
            add = cast(llir.Add, end.index)
            children.extend((add.left, add.right))
        return children

    first_children = owned_children(
        cast(llir.ArrayAccess, first_begin),
        cast(llir.ArrayAccess, first_end),
    )
    second_children = owned_children(
        cast(llir.ArrayAccess, second_begin),
        cast(llir.ArrayAccess, second_end),
    )
    assert len({id(child) for child in first_children}) == len(first_children)
    assert len({id(child) for child in second_children}) == len(second_children)
    assert {id(child) for child in first_children}.isdisjoint(
        {id(child) for child in second_children}
    )

    for access in accesses:
        if type(access.index) is llir.Var:
            index_var = cast(llir.Var, access.index)
            assert index_var.type is llir.DataType.INT
            assert index_var.is_ptr is False
            assert index_var.is_restrict is False
            assert index_var.tensor_access is None
        elif type(access.index) is llir.Literal:
            literal = cast(llir.Literal, access.index)
            assert type(literal.value) is int
            assert literal.data_type is llir.DataType.INT
        else:
            add = cast(llir.Add, access.index)
            assert type(add.left) is llir.Var
            assert cast(llir.Var, add.left).type is llir.DataType.INT
            assert type(add.right) is llir.Literal
            assert cast(llir.Literal, add.right).value == 1
            assert cast(llir.Literal, add.right).data_type is llir.DataType.INT

    with pytest.raises(FrozenInstanceError):
        cast(llir.ArrayAccess, first_begin).index = llir.Literal(2)
    if with_parent:
        with pytest.raises(FrozenInstanceError):
            cast(llir.Add, cast(llir.ArrayAccess, first_end).index).left = llir.Var(
                "other", llir.DataType.INT
            )

    assert tensor_state == (
        tensor.name,
        tensor.symbol_id,
        tensor.format,
        tensor.shape,
        tensor.dtype,
        tuple(tensor.mode_order or ()),
        tensor._assignment,
    )
    assert index_state(index) == index_before
    if parent is not None:
        assert index_state(parent) == parent_before
    assert access_state == (
        tensor_access.access_id,
        tensor_access.tensor,
        tensor_access.tensor_id,
        tuple(tensor_access.indices),
        tensor_access.index_ids,
    )


def test_mode_position_matching_is_exact_and_collection_is_structural() -> None:
    parent_begin = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        llir.Var("pA0", llir.DataType.INT),
    )
    parent_end = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        llir.Add(
            llir.Var("pA0", llir.DataType.INT),
            llir.Literal(1, llir.DataType.INT),
        ),
    )
    root_begin = llir.ArrayAccess(
        llir.Var("B0_pos", llir.DataType.PTR_INT),
        llir.Literal(0, llir.DataType.INT),
    )
    raw = llir.RawStmt("use(C1_pos[pC0], A1_pos[pA0])")
    body: list[llir.Stmt] = [
        llir.VarInit(llir.Var("pA1_end", llir.DataType.INT), parent_end),
        llir.ForLoop(
            init=llir.VarInit(llir.Var("pA1", llir.DataType.INT), parent_begin),
            cond=llir.BinOp(
                "<",
                llir.Var("pA1", llir.DataType.INT),
                llir.Var("pA1_end", llir.DataType.INT),
            ),
            update=llir.Increment(llir.Var("pA1", llir.DataType.INT)),
            body=[llir.VarInit(llir.Var("pB0", llir.DataType.INT), root_begin), raw],
        ),
    ]
    context = LLIRTraversalContext("test", "collect_mode_position_arrays")

    assert collect_mode_position_arrays(body, context) == [
        "A1_pos",
        "B0_pos",
        "C1_pos",
    ]
    assert CINLowerer._has_sparse_inner_loop(body) is True
    assert CINLowerer._find_sparse_pos_array(body) == "A1_pos"
    assert (
        CINLowerer._has_sparse_inner_loop(
            [
                llir.ForLoop(
                    init=llir.VarInit(
                        llir.Var("pA1", llir.DataType.INT),
                        llir.Var("A1_pos[pA0]", llir.DataType.INT),
                    ),
                    cond=llir.Literal(True),
                    update=llir.Increment(llir.Var("pA1", llir.DataType.INT)),
                    body=[],
                )
            ]
        )
        is False
    )
    assert (
        CINLowerer._find_sparse_pos_array(
            [
                llir.VarInit(
                    llir.Var("pA1", llir.DataType.INT),
                    llir.Var("A1_pos[pA0]", llir.DataType.INT),
                )
            ]
        )
        is None
    )
    assert CINLowerer._find_sparse_pos_array([llir.RawStmt("A1_pos[pA0]")]) == (
        "A1_pos"
    )

    wrong_type = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.STD_VECTOR_C_INT),
        llir.Var("pA0", llir.DataType.INT),
    )
    wrong_index = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        llir.Var("pA0", llir.DataType.INT64),
    )
    wrong_literal = llir.ArrayAccess(
        llir.Var("A0_pos", llir.DataType.PTR_INT),
        llir.Literal(0),
    )
    access_metadata = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        llir.Var("pA0", llir.DataType.INT),
    )
    object.__setattr__(access_metadata, "tensor_access", object())
    base_metadata = llir.ArrayAccess(
        llir.Var(
            "A1_pos",
            llir.DataType.PTR_INT,
            tensor_access=cast(llir.TensorAccessMetadata, object()),
        ),
        llir.Var("pA0", llir.DataType.INT),
    )
    index_metadata = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        llir.Var(
            "pA0",
            llir.DataType.INT,
            tensor_access=cast(llir.TensorAccessMetadata, object()),
        ),
    )
    semantic_misses = [
        wrong_type,
        wrong_index,
        wrong_literal,
        access_metadata,
        base_metadata,
        index_metadata,
        llir.ArrayAccess(
            llir.Var("A1_pos", llir.DataType.PTR_INT, is_ptr=True),
            llir.Var("pA0", llir.DataType.INT),
        ),
        llir.ArrayAccess(
            llir.Var("A1_pos", llir.DataType.PTR_INT, is_restrict=True),
            llir.Var("pA0", llir.DataType.INT),
        ),
        llir.ArrayAccess(
            llir.Var("A1_pos", llir.DataType.PTR_INT),
            llir.Var("not-an-identifier", llir.DataType.INT),
        ),
        llir.ArrayAccess(
            llir.Var("not-position", llir.DataType.PTR_INT),
            llir.Var("pA0", llir.DataType.INT),
        ),
    ]
    for value in semantic_misses:
        assert match_mode_position_begin(value) is None
        assert match_mode_position_access(value) is None

    forged_end = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        llir.Add(
            llir.Var("pA0", llir.DataType.INT),
            llir.Literal(1, llir.DataType.INT),
        ),
    )
    object.__setattr__(cast(llir.Add, forged_end.index), "op", "-")
    invalid_ends = [
        root_begin,
        forged_end,
        llir.ArrayAccess(
            llir.Var("A1_pos", llir.DataType.PTR_INT),
            llir.Add(
                llir.Literal(1, llir.DataType.INT),
                llir.Var("pA0", llir.DataType.INT),
            ),
        ),
        llir.ArrayAccess(
            llir.Var("A1_pos", llir.DataType.PTR_INT),
            llir.Add(
                llir.Var("pA0", llir.DataType.INT),
                llir.Literal(1, llir.DataType.INT64),
            ),
        ),
    ]
    assert all(
        match_mode_position_bounds(parent_begin, end) is None for end in invalid_ends
    )

    class UnknownPositionAccess(llir.ArrayAccess):
        pass

    class UnknownPositionAdd(llir.Add):
        pass

    class UnknownPositionVar(llir.Var):
        pass

    unknown_access = UnknownPositionAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        llir.Var("pA0", llir.DataType.INT),
    )
    unknown_base = llir.ArrayAccess(
        UnknownPositionVar("A1_pos", llir.DataType.PTR_INT),
        llir.Var("pA0", llir.DataType.INT),
    )
    unknown_index = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        UnknownPositionVar("pA0", llir.DataType.INT),
    )
    unknown_end = llir.ArrayAccess(
        llir.Var("A1_pos", llir.DataType.PTR_INT),
        UnknownPositionAdd(
            llir.Var("pA0", llir.DataType.INT),
            llir.Literal(1, llir.DataType.INT),
        ),
    )
    for value in (unknown_access, unknown_base, unknown_index, unknown_end):
        assert match_mode_position_access(value) is None
    assert match_mode_position_bounds(parent_begin, unknown_end) is None
    with pytest.raises(LLIRTraversalError, match="unknown_llir_node"):
        collect_mode_position_arrays(
            [llir.VarInit(llir.Var("x", llir.DataType.INT), unknown_access)], context
        )


@pytest.mark.parametrize("scalar_accumulator", (False, True))
def test_all_coo_transform_coordinate_initializers_are_structured_typed_and_owned(
    scalar_accumulator: bool,
) -> None:
    source = _build_all_coo_transform_loop()
    source_initializer = cast(llir.VarInit, source.body[0])
    source_access = cast(llir.ArrayAccess, source_initializer.value)
    source_cpp = LLIRLowerer().lower_llir(source)
    assert source_cpp == (
        "for (int pMask0 = 0; pMask0 < pMask0_end; pMask0 = pMask1_end) {\n"
        "  int r = Mask0_crd[pMask0];\n"
        "}"
    )

    first_loop, first_initializer, first = _all_coo_transform_coordinate_read(
        scalar_accumulator=scalar_accumulator,
        source=source,
    )
    second_loop, _, second = _all_coo_transform_coordinate_read(
        scalar_accumulator=scalar_accumulator
    )

    expected = llir.ArrayAccess(
        array=llir.Var("Mask0_crd", llir.DataType.PTR_INT),
        index=llir.Var("pMask0", llir.DataType.INT64),
    )
    assert first == expected == second
    assert hash(first) == hash(expected) == hash(second)
    assert first.tensor_access is None
    assert type(first.array) is llir.Var
    assert cast(llir.Var, first.array).type is llir.DataType.PTR_INT
    assert type(first.index) is llir.Var
    assert cast(llir.Var, first.index).type is llir.DataType.INT64
    assert first is not second
    assert first.array is not second.array
    assert first.index is not second.index
    assert first is not source_access
    assert first != source_access
    assert not any(statement is source_initializer for statement in first_loop.body)
    assert not any(statement == source_initializer for statement in first_loop.body)

    first_candidates = [cast(llir.VarInit, first_loop.init)] + [
        cast(llir.VarInit, statement)
        for statement in first_loop.body
        if type(statement) is llir.VarInit
    ]
    second_candidates = [cast(llir.VarInit, second_loop.init)] + [
        cast(llir.VarInit, statement)
        for statement in second_loop.body
        if type(statement) is llir.VarInit
    ]
    first_iterator = next(
        initializer.var
        for initializer in first_candidates
        if initializer.var.name == "pMask0"
    )
    second_iterator = next(
        initializer.var
        for initializer in second_candidates
        if initializer.var.name == "pMask0"
    )
    assert first.index is not first_iterator
    assert second.index is not second_iterator
    assert LLIRLowerer().lower_llir(first_initializer) == (
        "int64_t r = Mask0_crd[pMask0];"
    )
    first_loop_cpp = LLIRLowerer().lower_llir(first_loop)
    assert first_loop_cpp.count("  int64_t r = Mask0_crd[pMask0];") == 1
    assert "  int r = Mask0_crd[pMask0];" not in first_loop_cpp
    assert LLIRLowerer().lower_llir(source) == source_cpp
    with pytest.raises(FrozenInstanceError):
        first.index = llir.Var("other", llir.DataType.INT64)


def test_all_coo_transform_end_bound_is_structured_typed_owned_and_byte_exact() -> None:
    source = _build_all_coo_transform_loop()
    source_cpp = LLIRLowerer().lower_llir(source)
    first_loop, first_initializer, first = _all_coo_transform_end_bound(source=source)
    second_loop, second_initializer, second = _all_coo_transform_end_bound()

    expected = llir.Add(
        llir.Var("pMask0", llir.DataType.INT64),
        llir.Literal(1, data_type=llir.DataType.INT64),
    )
    assert type(first) is llir.Add
    assert type(second) is llir.Add
    assert first == expected == second
    assert hash(first) == hash(expected) == hash(second)
    assert first != llir.BinOp(
        "+",
        llir.Var("pMask0", llir.DataType.INT64),
        llir.Literal(1, data_type=llir.DataType.INT64),
    )
    assert type(first.left) is llir.Var
    first_left = cast(llir.Var, first.left)
    assert first_left.name == "pMask0"
    assert first_left.type is llir.DataType.INT64
    assert first_left.tensor_access is None
    assert first_left.is_ptr is False
    assert first_left.is_restrict is False
    assert type(first.right) is llir.Literal
    first_right = cast(llir.Literal, first.right)
    assert first_right.value == 1
    assert type(first_right.value) is int
    assert first_right.data_type is llir.DataType.INT64

    assert first is not second
    assert first.left is not second.left
    assert first.right is not second.right
    assert first.left is not cast(llir.VarInit, first_loop.init).var
    assert second.left is not cast(llir.VarInit, second_loop.init).var
    assert first_initializer is not second_initializer
    assert first_initializer.var is not second_initializer.var
    assert first_initializer.value is first
    assert second_initializer.value is second
    assert first_initializer.var.type is llir.DataType.INT64
    assert LLIRLowerer().lower_llir(first_initializer) == (
        "int64_t pMask1_end = pMask0 + 1;"
    )
    assert "  int64_t pMask1_end = pMask0 + 1;" in LLIRLowerer().lower_llir(first_loop)
    assert LLIRLowerer().lower_llir(source) == source_cpp


def test_production_all_coo_end_bound_activates_then_is_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statement = _build_activating_all_coo_sddmm()
    statement_before = str(statement)
    original_transform = CINLowerer._transform_coo_loop_for_openmp
    captured: list[llir.Add] = []

    def record_end_bound(
        self: CINLowerer,
        statements: list[llir.Stmt],
    ) -> list[llir.Stmt]:
        transformed = original_transform(self, statements)

        def collect(value: object) -> None:
            if type(value) is llir.VarInit:
                initializer = cast(llir.VarInit, value)
                if (
                    initializer.var.name == "pMask1_end"
                    and type(initializer.value) is llir.Add
                ):
                    captured.append(cast(llir.Add, initializer.value))
            if isinstance(value, llir.Node):
                for child in vars(value).values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(transformed)
        return transformed

    monkeypatch.setattr(
        CINLowerer,
        "_transform_coo_loop_for_openmp",
        record_end_bound,
    )
    with regblock_force(False):
        lowerer = CINLowerer()
        lowered = lowerer.lower_IndexStmt(statement)
    cpp = LLIRLowerer().lower_llir(lowered)

    assert str(statement) == statement_before
    assert captured == [
        llir.Add(
            llir.Var("pMask0", llir.DataType.INT64),
            llir.Literal(1, data_type=llir.DataType.INT64),
        )
    ]
    assert "pMask1_end" not in cpp
    assert "pMask1 = pMask0" not in cpp
    assert "pMask1 <" not in cpp
    assert "pMask1++" not in cpp
    assert "pMask0 < pMask0_end; pMask0++" in cpp
    assert [record.pass_name for record in lowerer.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]


def test_production_known_nnz_size_is_structured_typed_owned_and_byte_exact() -> None:
    with regblock_force(False):
        statement = _build_activating_all_coo_sddmm()
        statement_before = str(statement)
        first_lowerer = CINLowerer()
        first_function = cast(
            llir.Function,
            first_lowerer.lower_IndexStmt(statement),
        )
        second_lowerer = CINLowerer()
        second_function = cast(
            llir.Function,
            second_lowerer.lower_IndexStmt(statement),
        )

    def known_nnz_initializer(function: llir.Function) -> llir.VarInit:
        matches = [
            cast(llir.VarInit, candidate)
            for candidate in function.body
            if type(candidate) is llir.VarInit
            and cast(llir.VarInit, candidate).var.name == "_known_nnz"
        ]
        assert len(matches) == 1
        return matches[0]

    first = known_nnz_initializer(first_function)
    second = known_nnz_initializer(second_function)
    expected = llir.VarInit(
        var=llir.Var("_known_nnz", llir.DataType.INT64),
        value=llir.MemberCall(
            base=llir.Var("Mask_values", llir.DataType.TORCH_TENSOR),
            member="size",
            args=(llir.Literal(0, llir.DataType.INT64),),
        ),
    )

    assert type(first) is llir.VarInit
    assert first == expected == second
    assert hash(first) == hash(expected) == hash(second)
    assert first is not second
    assert first.var is not second.var
    assert first.var.name == "_known_nnz"
    assert first.var.type is llir.DataType.INT64
    assert first.var.tensor_access is None
    assert first.var.is_ptr is False
    assert first.var.is_restrict is False
    assert first.op == "="
    assert first.cast is False

    assert type(first.value) is llir.MemberCall
    assert type(second.value) is llir.MemberCall
    first_call = cast(llir.MemberCall, first.value)
    second_call = cast(llir.MemberCall, second.value)
    assert first_call == second_call
    assert hash(first_call) == hash(second_call)
    assert first_call is not second_call
    assert first_call.member == "size"
    assert first_call.template_args == ()
    assert type(first_call.args) is tuple
    assert len(first_call.args) == 1
    assert first_call.args[0] is not second_call.args[0]
    assert type(first_call.args[0]) is llir.Literal
    extent = cast(llir.Literal, first_call.args[0])
    assert type(extent.value) is int
    assert extent.value == 0
    assert extent.data_type is llir.DataType.INT64

    assert type(first_call.base) is llir.Var
    assert type(second_call.base) is llir.Var
    first_base = cast(llir.Var, first_call.base)
    second_base = cast(llir.Var, second_call.base)
    first_argument = next(
        argument
        for argument in first_function.args
        if type(argument) is llir.Var and argument.name == "Mask_values"
    )
    assert first_base is not second_base
    assert first_base is not first_argument
    assert first_base.name == "Mask_values"
    assert first_base.type is first_argument.type is llir.DataType.TORCH_TENSOR
    assert first_base.tensor_access is None
    assert first_base.is_ptr is False
    assert first_base.is_restrict is False

    first_cpp = LLIRLowerer().lower_llir(first_function)
    second_cpp = LLIRLowerer().lower_llir(second_function)
    assert first_cpp == second_cpp
    assert LLIRLowerer().lower_llir(first) == (
        "int64_t _known_nnz = Mask_values.size(0);"
    )
    assert first_cpp.count("int64_t _known_nnz = Mask_values.size(0);") == 1
    for level in range(2):
        owner_name = f"Sampled{level}_crd_torch"
        pointer_name = f"Sampled{level}_crd"
        first_owner = next(
            cast(llir.VarInit, candidate)
            for candidate in first_function.body
            if type(candidate) is llir.VarInit
            and cast(llir.VarInit, candidate).var.name == owner_name
        )
        second_owner = next(
            cast(llir.VarInit, candidate)
            for candidate in second_function.body
            if type(candidate) is llir.VarInit
            and cast(llir.VarInit, candidate).var.name == owner_name
        )
        first_pointer = next(
            cast(llir.VarInit, candidate)
            for candidate in first_function.body
            if type(candidate) is llir.VarInit
            and cast(llir.VarInit, candidate).var.name == pointer_name
        )
        second_pointer = next(
            cast(llir.VarInit, candidate)
            for candidate in second_function.body
            if type(candidate) is llir.VarInit
            and cast(llir.VarInit, candidate).var.name == pointer_name
        )
        assert first_owner == second_owner
        assert first_owner is not second_owner
        assert first_owner.var is not second_owner.var
        assert first_owner.var.type is llir.DataType.TORCH_TENSOR
        assert type(first_owner.value) is llir.FunctionCall
        owner_call = cast(llir.FunctionCall, first_owner.value)
        assert owner_call.name == "torch::empty"
        assert type(owner_call.args[0]) is llir.Array
        owner_extent = cast(llir.Array, owner_call.args[0])
        assert type(owner_extent.values[0]) is llir.Var
        assert cast(llir.Var, owner_extent.values[0]).name == "_known_nnz"
        _assert_torch_qualified_name(owner_call.args[1], "torch::kInt")

        assert first_pointer == second_pointer
        assert first_pointer is not second_pointer
        assert first_pointer.var is not second_pointer.var
        assert first_pointer.var.type is llir.DataType.PTR_INT
        pointer_call = _assert_data_ptr_call(
            first_pointer.value,
            llir.DataType.INT,
        )
        receiver = _assert_torch_tensor_var(pointer_call.base, owner_name)
        assert receiver is not first_owner.var
        assert (
            first_cpp.count(
                f"torch::Tensor {owner_name} = "
                "torch::empty({_known_nnz}, torch::kInt);"
            )
            == 1
        )
        assert (
            first_cpp.count(f"int* {pointer_name} = {owner_name}.data_ptr<int>();") == 1
        )
    assert not any(
        type(candidate) is llir.RawStmt
        and (
            candidate.code.startswith("int64_t _known_nnz =")
            or ("_known_nnz" in candidate.code and "_crd_torch" in candidate.code)
        )
        for candidate in first_function.body
    )
    assert str(statement) == statement_before
    assert first_lowerer.llir_pass_run_records == second_lowerer.llir_pass_run_records
    assert [record.pass_name for record in first_lowerer.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]

    first.var.name = "owned_target"
    first_base.name = "owned_receiver"
    assert second.var.name == "_known_nnz"
    assert second_base.name == "Mask_values"
    assert first_argument.name == "Mask_values"
    assert str(statement) == statement_before


def test_mutated_unknown_post_op_cannot_be_silently_skipped():
    post_ops = PostOps(ops=[], extra_tensors=[])
    lowerer = CINLowerer(post_ops=post_ops)
    post_ops.ops.append(PostOp(kind="clip"))

    with pytest.raises(
        UnsupportedFeature,
        match=r"stage=CIN lowering: unsupported post-op kind 'clip'",
    ):
        lowerer._emit_post_ops("output", "i")


@pytest.mark.parametrize(
    ("fmt", "indices", "expected_index"),
    (
        ("d", ("i",), "i"),
        ("ds", ("i", "j"), "pInput1"),
    ),
)
def test_nonworkspace_tensor_reads_lower_to_frozen_structured_accesses(
    fmt: str,
    indices: tuple[str, ...],
    expected_index: str,
) -> None:
    index_vars = tuple(IndexVar(name) for name in indices)
    tensor = TensorVar("Input", fmt=fmt)
    access = tensor[index_vars[0] if len(index_vars) == 1 else index_vars]
    original_indices = tuple(access.indices)

    lowered = CINLowerer().lower_TensorAccess(access)

    assert type(lowered) is llir.ArrayAccess
    structured = cast(llir.ArrayAccess, lowered)
    assert cast(llir.Var, structured.array).name == "Input_val"
    assert cast(llir.Var, structured.array).type is llir.DataType.PTR_FLOAT32
    assert cast(llir.Var, structured.index).name == expected_index
    assert cast(llir.Var, structured.index).type is llir.DataType.INT
    assert structured.tensor_access == llir.TensorAccessMetadata(
        access_id=access.access_id,
        tensor_id=access.tensor_id,
        index_ids=access.index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    assert LLIRLowerer().lower_llir(structured) == f"Input_val[{expected_index}]"
    assert tuple(access.indices) == original_indices
    assert access.tensor is tensor


@pytest.mark.parametrize(
    ("dim", "dense", "index_names"),
    (
        pytest.param(0, False, (), id="scalar"),
        pytest.param(1, True, ("i",), id="dense"),
        pytest.param(2, False, ("i", "j"), id="coordinate"),
    ),
)
def test_generic_workspace_reads_fail_closed_before_llir_construction(
    dim: int,
    dense: bool,
    index_names: tuple[str, ...],
) -> None:
    workspace = Workspace("wksp", dim=dim, dense=dense)
    index_vars = tuple(IndexVar(name) for name in index_names)
    if not index_vars:
        access = workspace.get_default_access()
    else:
        key = index_vars[0] if len(index_vars) == 1 else index_vars
        access = workspace[key]
    original_indices = tuple(access.indices)

    with pytest.raises(
        CompilerInvariantError,
        match=(
            r"stage=CIN lowering pass=lower_TensorAccess: workspace tensor 'wksp' "
            r"reached generic value lowering; workspace reads require a "
            r"workspace-specific consumer"
        ),
    ):
        CINLowerer().lower_TensorAccess(access)

    assert tuple(access.indices) == original_indices
    assert access.tensor is workspace


def test_direct_workspace_rhs_fails_before_an_invalid_kernel_is_built() -> None:
    index = IndexVar("i")
    result = TensorVar("Result", fmt="d")
    workspace = Workspace("wksp", dim=1, dense=True)
    statement = ForAll(
        index,
        TensorAssign(result[index], workspace[index]),
    )
    original = str(statement)
    lowerer = CINLowerer()

    with pytest.raises(
        CompilerInvariantError,
        match=(
            r"stage=CIN lowering pass=lower_TensorAccess: workspace tensor 'wksp' "
            r"reached generic value lowering"
        ),
    ):
        lowerer.lower_IndexStmt(statement)

    assert lowerer.llir_pass_run_records == ()
    assert str(statement) == original


def test_dense_level_shape_reads_use_frozen_structured_array_accesses() -> None:
    row, column = IndexVar("row"), IndexVar("column")
    result = TensorVar("Result", shape=(3, 5), fmt="dd")
    operand = TensorVar("Input", shape=(3, 5), fmt="dd")
    statement = ForAll(
        row,
        ForAll(
            column,
            TensorAssign(result[row, column], operand[row, column]),
        ),
    )

    lowered = CINLowerer().lower_IndexStmt(statement)

    assert type(lowered) is llir.Function
    function = cast(llir.Function, lowered)
    initializers = {
        node.var.name: cast(llir.VarInit, node)
        for node in function.body
        if type(node) is llir.VarInit
    }
    expected = {
        "Result0_size": ("result_shape", 0),
        "Result1_size": ("result_shape", 1),
        "Input0_size": ("Input_shape", 0),
        "Input1_size": ("Input_shape", 1),
    }
    for initializer_name, (shape_name, level) in expected.items():
        initializer = initializers[initializer_name]
        assert type(initializer.value) is llir.ArrayAccess
        access = cast(llir.ArrayAccess, initializer.value)
        assert access == llir.ArrayAccess(
            array=llir.Var(shape_name, llir.DataType.STD_VECTOR_INT),
            index=llir.Literal(level, data_type=llir.DataType.INT64),
        )
        assert type(access.array) is llir.Var
        assert cast(llir.Var, access.array).type is llir.DataType.STD_VECTOR_INT
        assert type(access.index) is llir.Literal
        assert cast(llir.Literal, access.index).data_type is llir.DataType.INT64
        assert access.tensor_access is None
        assert LLIRLowerer().lower_llir(initializer) == (
            f"int64_t {initializer_name} = {shape_name}[{level}];"
        )

    result_access = cast(llir.ArrayAccess, initializers["Result0_size"].value)
    with pytest.raises(FrozenInstanceError):
        result_access.index = llir.Literal(1)


def test_cin_reference_rewrite_rebuilds_frozen_access_and_preserves_metadata() -> None:
    index = IndexVar("ix")
    tensor = TensorVar("Input", fmt="d")
    logical_access = tensor[index]
    metadata = llir.TensorAccessMetadata(
        access_id=logical_access.access_id,
        tensor_id=logical_access.tensor_id,
        index_ids=logical_access.index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    source = llir.ArrayAccess(
        array=llir.Var("Array[ix]", llir.DataType.PTR_FLOAT32),
        index=llir.Var("ix offset", llir.DataType.INT),
        tensor_access=metadata,
    )

    rewritten = CINLowerer._rewrite_expr_refs(source, {"ix": "root"})
    repeated = CINLowerer._rewrite_expr_refs(rewritten, {"ix": "root"})

    assert type(rewritten) is llir.ArrayAccess
    assert cast(llir.Var, rewritten.array).name == "Array[root]"
    assert cast(llir.Var, rewritten.index).name == "root offset"
    assert rewritten.tensor_access is metadata
    assert repeated == rewritten
    assert repeated is not rewritten
    assert cast(llir.Var, source.array).name == "Array[ix]"
    assert cast(llir.Var, source.index).name == "ix offset"


def test_nondefault_coo_intersection_keeps_live_coordinate_end_bounds():
    row, column = IndexVar("r"), IndexVar("c")
    result = TensorVar("Intersect", fmt="oo", mode_order=[1, 0])
    left = TensorVar("Left", fmt="oo", mode_order=[1, 0])
    right = TensorVar("Right", fmt="oo", mode_order=[1, 0])
    assignment = TensorAssign(
        result[row, column],
        left[row, column] * right[row, column],
    )
    scheduled = cast(
        ForAll,
        Scheduler.auto_schedule(ForAll(row, ForAll(column, assignment))),
    )

    cpp = LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(scheduled))

    assert "int pLeft1_end = 0;" in cpp
    assert "pLeft1_end = pLeft0 + 1;" in cpp
    assert "pLeft1_end < pLeft0_end" in cpp
    assert "pLeft1 < pLeft1_end && pRight1 < pRight1_end" in cpp
    assert "pLeft0 = pLeft1_end;" in cpp
    assert "int pRight1_end = 0;" in cpp
    assert "pRight1_end = pRight0 + 1;" in cpp
    assert "pRight1_end < pRight0_end" in cpp
    assert "pRight0 = pRight1_end;" in cpp
