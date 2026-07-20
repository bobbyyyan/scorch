import hashlib
from collections import Counter
from dataclasses import FrozenInstanceError, replace
import re
from typing import List, Set, Tuple, cast

import pytest
import torch

from scorch.compiler import llir
from scorch.compiler import compressed_where_openmp_pass as compressed_where_module
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.compressed_where_openmp_pass import (
    CompressedWhereOpenMPContext,
    CompressedWhereOpenMPPolicy,
    CompressedWhereOpenMPResult,
    transform_compressed_where_for_openmp,
)
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.format import LevelType
from scorch.compiler.llir_traversal import (
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from scorch.compiler.llir_pass_manager import DEBUG_LLIR_PASS_OPTIONS
from scorch.compiler.scheduler import Scheduler
from scorch.compiler.torch_cpp_abi import (  # type: ignore[import-untyped]
    ResultTensorAssembler,
)


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _result_assembler(
    compressed_levels: Tuple[int, ...] = (1,),
    *,
    name: str = "Result",
    dtype: torch.dtype = torch.float32,
) -> ResultTensorAssembler:
    return ResultTensorAssembler(
        name=name,
        level_types=(LevelType.DENSE,)
        + tuple(LevelType.COMPRESSED for _ in compressed_levels),
        dtype=dtype,
    )


def _context(
    compressed_levels: Tuple[int, ...] = (1,),
    *,
    policy: CompressedWhereOpenMPPolicy = CompressedWhereOpenMPPolicy(),
    result_id: SymbolId = SymbolId(1),
) -> CompressedWhereOpenMPContext:
    return CompressedWhereOpenMPContext(
        result_name="Result",
        result_id=result_id,
        compressed_levels=compressed_levels,
        result_assembler=_result_assembler(compressed_levels),
        workspace_name="wksp",
        workspace_ctype="float",
        policy=policy,
    )


def _workspace_init() -> llir.VarInit:
    return llir.VarInit(
        _var("wksp", llir.DataType.AUTO),
        llir.FunctionCall("coo_workspace_1d<float, 1>", [llir.Literal(1024)]),
    )


def _compatible_loop(
    body: List[llir.Stmt],
    *,
    bound: str = "A0_size",
    bound_type: llir.DataType = llir.DataType.INT64,
    cond_op: str = "<",
    update: llir.Assign | None = None,
) -> llir.ForLoop:
    row = _var("row", llir.DataType.INT)
    return llir.ForLoop(
        init=llir.VarInit(row, llir.Literal(0)),
        cond=llir.BinOp(cond_op, row, _var(bound, bound_type)),
        update=update or llir.Increment(_var("row", llir.DataType.INT)),
        body=body,
    )


def _ds_work_body(
    *, workspace: bool = True, both_operands: bool = True
) -> List[llir.Stmt]:
    body: List[llir.Stmt] = []
    if workspace:
        body.append(_workspace_init())
    body.append(llir.RawStmt("int pA1 = A1_pos[row]"))
    if both_operands:
        body.append(llir.RawStmt("int pB1 = B1_pos[reduction]"))
    body.extend(
        [
            llir.FunctionCallStmt("wksp.insert", [_var("value")]),
            llir.FunctionCallStmt("Result1_crd.push_back", [_var("column")]),
            llir.FunctionCallStmt("Result_values.push_back", [_var("value")]),
        ]
    )
    return body


def _structured_ds_work_body() -> List[llir.Stmt]:
    def end_init(array: str, parent: str, iterator: str) -> llir.VarInit:
        return llir.VarInit(
            _var(f"{iterator}_end", llir.DataType.INT),
            llir.ArrayAccess(
                _var(array, llir.DataType.PTR_INT),
                llir.Add(
                    _var(parent, llir.DataType.INT),
                    llir.Literal(1, llir.DataType.INT),
                ),
            ),
        )

    return [
        _workspace_init(),
        end_init("A1_pos", "pA0", "pA1"),
        end_init("B1_pos", "pB0", "pB1"),
        llir.FunctionCallStmt("wksp.insert", [_var("value")]),
        llir.FunctionCallStmt("Result1_crd.push_back", [_var("column")]),
        llir.FunctionCallStmt("Result_values.push_back", [_var("value")]),
    ]


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


def _raw_codes(value: object) -> List[str]:
    codes: List[str] = []
    if type(value) is llir.RawStmt:
        codes.append(cast(llir.RawStmt, value).code)
    elif isinstance(value, llir.Node):
        for child in vars(value).values():
            codes.extend(_raw_codes(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            codes.extend(_raw_codes(child))
    return codes


def _assignments(value: object) -> List[llir.Assign]:
    assignments: List[llir.Assign] = []
    if type(value) is llir.Assign:
        assignments.append(cast(llir.Assign, value))
    elif isinstance(value, llir.Node):
        for child in vars(value).values():
            assignments.extend(_assignments(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            assignments.extend(_assignments(child))
    return assignments


def _offset_zero_assignments(value: object) -> List[llir.Assign]:
    return [
        assignment
        for assignment in _assignments(value)
        if type(assignment.var) is llir.ArrayAccess
        and type(cast(llir.ArrayAccess, assignment.var).array) is llir.Var
        and re.fullmatch(
            r"_offset\d+",
            cast(llir.Var, cast(llir.ArrayAccess, assignment.var).array).name,
        )
        and type(cast(llir.ArrayAccess, assignment.var).index) is llir.Literal
        and cast(llir.Literal, cast(llir.ArrayAccess, assignment.var).index).value == 0
    ]


def _direct_initializations(value: object) -> List[llir.DirectInit]:
    declarations: List[llir.DirectInit] = []
    if type(value) is llir.DirectInit:
        declarations.append(cast(llir.DirectInit, value))
    elif isinstance(value, llir.Node):
        for child in vars(value).values():
            declarations.extend(_direct_initializations(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            declarations.extend(_direct_initializations(child))
    return declarations


def _offset_family_direct_initializations(value: object) -> List[llir.DirectInit]:
    return [
        declaration
        for declaration in _direct_initializations(value)
        if re.fullmatch(r"_(?:count|offset)\d+", declaration.var.name)
    ]


def _prefix_sum_loops(value: object) -> List[llir.ForLoop]:
    loops: List[llir.ForLoop] = []
    if type(value) is llir.ForLoop:
        loop = cast(llir.ForLoop, value)
        if (
            type(loop.init) is llir.VarInit
            and cast(llir.VarInit, loop.init).var.name == "_i"
            and type(loop.cond) is llir.BinOp
            and cast(llir.BinOp, loop.cond).op == "<"
            and not loop.omp_parallel_for
        ):
            loops.append(loop)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            loops.extend(_prefix_sum_loops(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            loops.extend(_prefix_sum_loops(child))
    return loops


def _total_offset_loads(value: object) -> List[llir.VarInit]:
    loads: List[llir.VarInit] = []
    if type(value) is llir.VarInit:
        initializer = cast(llir.VarInit, value)
        access = initializer.value
        if (
            re.fullmatch(r"_total\d+", initializer.var.name)
            and type(access) is llir.ArrayAccess
            and type(cast(llir.ArrayAccess, access).array) is llir.Var
            and re.fullmatch(
                r"_offset\d+",
                cast(llir.Var, cast(llir.ArrayAccess, access).array).name,
            )
        ):
            loads.append(initializer)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            loads.extend(_total_offset_loads(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            loads.extend(_total_offset_loads(child))
    return loads


def _compressed_coordinate_initializations(value: object) -> List[llir.VarInit]:
    initializations: List[llir.VarInit] = []
    if type(value) is llir.VarInit:
        initialization = cast(llir.VarInit, value)
        if re.fullmatch(r"Result\d+_crd_(?:torch|data)", initialization.var.name):
            initializations.append(initialization)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            initializations.extend(_compressed_coordinate_initializations(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            initializations.extend(_compressed_coordinate_initializations(child))
    return initializations


def _value_initializations(value: object) -> List[llir.VarInit]:
    initializations: List[llir.VarInit] = []
    if type(value) is llir.VarInit:
        initialization = cast(llir.VarInit, value)
        if re.fullmatch(r"[A-Za-z_]\w*_values_(?:torch|data)", initialization.var.name):
            initializations.append(initialization)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            initializations.extend(_value_initializations(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            initializations.extend(_value_initializations(child))
    return initializations


def _first_position_initializations(value: object) -> List[llir.VarInit]:
    initializations: List[llir.VarInit] = []
    if type(value) is llir.VarInit:
        initialization = cast(llir.VarInit, value)
        if initialization.var.name in {
            "Result1_pos_torch",
            "Result1_pos_data",
        }:
            initializations.append(initialization)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            initializations.extend(_first_position_initializations(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            initializations.extend(_first_position_initializations(child))
    return initializations


def _first_position_copy_loops(value: object) -> List[llir.ForLoop]:
    loops: List[llir.ForLoop] = []
    if type(value) is llir.ForLoop:
        loop = cast(llir.ForLoop, value)
        condition = loop.cond
        if (
            type(condition) is llir.BinOp
            and condition.op == "<="
            and len(loop.body) == 1
            and type(loop.body[0]) is llir.Assign
        ):
            assignment = cast(llir.Assign, loop.body[0])
            target = assignment.var
            if (
                type(target) is llir.ArrayAccess
                and type(target.array) is llir.Var
                and target.array.name == "Result1_pos_data"
            ):
                loops.append(loop)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            loops.extend(_first_position_copy_loops(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            loops.extend(_first_position_copy_loops(child))
    return loops


def _deeper_position_owner_initializations(value: object) -> List[llir.VarInit]:
    initializations: List[llir.VarInit] = []
    if type(value) is llir.VarInit:
        initialization = cast(llir.VarInit, value)
        match = re.fullmatch(r"Result(\d+)_pos_torch", initialization.var.name)
        if match is not None and int(match.group(1)) > 1:
            initializations.append(initialization)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            initializations.extend(_deeper_position_owner_initializations(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            initializations.extend(_deeper_position_owner_initializations(child))
    return initializations


def _deeper_position_pointer_initializations(value: object) -> List[llir.VarInit]:
    initializations: List[llir.VarInit] = []
    if type(value) is llir.VarInit:
        initialization = cast(llir.VarInit, value)
        match = re.fullmatch(r"Result(\d+)_pos_data", initialization.var.name)
        if match is not None and int(match.group(1)) > 1:
            initializations.append(initialization)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            initializations.extend(_deeper_position_pointer_initializations(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            initializations.extend(_deeper_position_pointer_initializations(child))
    return initializations


def _deeper_position_zero_sentinels(value: object) -> List[llir.Assign]:
    assignments: List[llir.Assign] = []
    if type(value) is llir.Assign:
        assignment = cast(llir.Assign, value)
        target = assignment.var
        if type(target) is llir.ArrayAccess and type(target.array) is llir.Var:
            match = re.fullmatch(r"Result(\d+)_pos_data", target.array.name)
            if (
                match is not None
                and int(match.group(1)) > 1
                and type(target.index) is llir.Literal
                and target.index.value == 0
                and type(assignment.value) is llir.Literal
                and assignment.value.value == 0
            ):
                assignments.append(assignment)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            assignments.extend(_deeper_position_zero_sentinels(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            assignments.extend(_deeper_position_zero_sentinels(child))
    return assignments


def _named_vars(value: object, name: str) -> List[llir.Var]:
    variables: List[llir.Var] = []
    if type(value) is llir.Var and cast(llir.Var, value).name == name:
        variables.append(cast(llir.Var, value))
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            variables.extend(_named_vars(child, name))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            variables.extend(_named_vars(child, name))
    return variables


def _assignment_codes(value: object) -> List[str]:
    return [
        LLIRLowerer().lower_llir([assignment]).removesuffix(";")
        for assignment in _assignments(value)
    ]


def _phase_state_statements(value: object) -> List[llir.Stmt]:
    statements: List[llir.Stmt] = []
    if type(value) is llir.VarInit:
        initialization = cast(llir.VarInit, value)
        if initialization.var.name.startswith(("_cnt", "_pos", "_prev")):
            statements.append(initialization)
    elif type(value) is llir.Increment:
        increment = cast(llir.Increment, value)
        if increment.var.name.startswith(("_cnt", "_pos")):
            statements.append(increment)
    elif type(value) is llir.Assign:
        assignment = cast(llir.Assign, value)
        if type(assignment.var) is llir.Var and assignment.var.name.startswith("_prev"):
            statements.append(assignment)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            statements.extend(_phase_state_statements(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            statements.extend(_phase_state_statements(child))
    return statements


def _phase_state_codes(value: object) -> List[str]:
    return [
        LLIRLowerer().lower_llir(statement).removesuffix(";")
        for statement in _phase_state_statements(value)
    ]


def _phase_state_vars(value: object) -> List[llir.Var]:
    if type(value) is llir.Var:
        variable = cast(llir.Var, value)
        if re.fullmatch(r"_(?:cnt|pos|prev)\d+", variable.name):
            return [variable]
        return []
    variables: List[llir.Var] = []
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            variables.extend(_phase_state_vars(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            variables.extend(_phase_state_vars(child))
    return variables


def _base_offset_loads(value: object) -> List[llir.VarInit]:
    loads: List[llir.VarInit] = []
    if type(value) is llir.VarInit:
        initializer = cast(llir.VarInit, value)
        access = initializer.value
        if (
            re.fullmatch(r"_base\d+", initializer.var.name)
            and type(access) is llir.ArrayAccess
            and type(cast(llir.ArrayAccess, access).array) is llir.Var
            and re.fullmatch(
                r"_offset\d+",
                cast(llir.Var, cast(llir.ArrayAccess, access).array).name,
            )
        ):
            loads.append(initializer)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            loads.extend(_base_offset_loads(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            loads.extend(_base_offset_loads(child))
    return loads


def _call_names(value: object) -> List[str]:
    names: List[str] = []
    if type(value) is llir.FunctionCallStmt:
        names.append(cast(llir.FunctionCallStmt, value).name)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            names.extend(_call_names(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            names.extend(_call_names(child))
    return names


def _phase_loops(result: CompressedWhereOpenMPResult) -> Tuple[llir.ForLoop, ...]:
    loops = tuple(
        cast(llir.ForLoop, statement)
        for statement in result.statements
        if type(statement) is llir.ForLoop and statement.omp_parallel_for
    )
    assert len(loops) == 2
    return loops


def _result_assembly_tail(
    statements: List[llir.Stmt],
) -> Tuple[llir.VarDecl, llir.Assign, llir.Assign, llir.Return]:
    tail = statements[-4:]
    assert [type(statement) for statement in tail] == [
        llir.VarDecl,
        llir.Assign,
        llir.Assign,
        llir.Return,
    ]
    declaration, mode_indices, values, return_statement = tail
    return (
        cast(llir.VarDecl, declaration),
        cast(llir.Assign, mode_indices),
        cast(llir.Assign, values),
        cast(llir.Return, return_statement),
    )


def test_policy_context_and_result_are_frozen() -> None:
    policy = CompressedWhereOpenMPPolicy()
    context = _context(policy=policy)
    result = CompressedWhereOpenMPResult(statements=[], applied=False)

    with pytest.raises(FrozenInstanceError):
        policy.omp_schedule = "static"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.result_name = "Other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.result_assembler = _result_assembler()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.applied = True  # type: ignore[misc]


def test_ds_transform_builds_exact_count_fill_allocation_and_policy() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_ds_work_body())]

    result = transform_compressed_where_for_openmp(source, _context())

    assert result.applied is True
    count_loop, fill_loop = _phase_loops(result)
    count_codes = _raw_codes(count_loop.body)
    fill_codes = _raw_codes(fill_loop.body)
    count_state_codes = _phase_state_codes(count_loop.body)
    fill_state_codes = _phase_state_codes(fill_loop.body)
    count_assignment_codes = _assignment_codes(count_loop.body)
    fill_assignment_codes = _assignment_codes(fill_loop.body)
    all_assignment_codes = _assignment_codes(result.statements)
    top_level_codes = [
        statement.code
        for statement in result.statements
        if type(statement) is llir.RawStmt
    ]
    offset_family_codes = [
        LLIRLowerer().lower_llir(declaration)
        for declaration in _offset_family_direct_initializations(result.statements)
    ]
    prefix_sum_codes = [
        LLIRLowerer().lower_llir(loop) for loop in _prefix_sum_loops(result.statements)
    ]
    total_load_codes = [
        LLIRLowerer().lower_llir(load)
        for load in _total_offset_loads(result.statements)
    ]

    assert "int _cnt1 = 0" in count_state_codes
    assert "_cnt1++" in count_state_codes
    assert "_count1[row] = _cnt1" in count_assignment_codes
    assert "_count1[row] = _cnt1" not in count_codes
    assert "wksp.clear()" in count_codes
    base_load_codes = [
        LLIRLowerer().lower_llir(load).removesuffix(";")
        for load in _base_offset_loads(fill_loop.body)
    ]
    assert base_load_codes == ["int64_t _base1 = _offset1[row]"]
    assert "int64_t _base1 = _offset1[row]" not in fill_codes
    assert "int _pos1 = 0" in fill_state_codes
    assert "Result1_crd_data[_base1 + _pos1] = column" in fill_assignment_codes
    assert "Result1_crd_data[_base1 + _pos1] = column" not in fill_codes
    assert "_pos1++" in fill_state_codes
    assert "Result_values_data[_base1 + _pos1] = value" in fill_assignment_codes
    assert "Result_values_data[_base1 + _pos1] = value" not in fill_codes
    assert "wksp.clear()" in fill_codes
    assert "_offset1[0] = 0" in all_assignment_codes
    assert "_offset1[0] = 0" not in _raw_codes(result.statements)
    assert all(
        type(assignment.var) is llir.ArrayAccess
        for assignment in _assignments(count_loop.body) + _assignments(fill_loop.body)
    )
    assert _call_names(count_loop.body).count("wksp.insert_unchecked") == 1
    assert _call_names(fill_loop.body).count("wksp.insert_unchecked") == 1

    assert offset_family_codes == [
        "std::vector<int> _count1((size_t) A0_size, 0);",
        "std::vector<int64_t> _offset1((size_t) A0_size + 1);",
    ]
    assert prefix_sum_codes == [
        "for (int _i = 0; _i < A0_size; _i++) {\n"
        "  _offset1[_i + 1] = _offset1[_i] + _count1[_i];\n"
        "}"
    ]
    assert total_load_codes == ["int64_t _total1 = _offset1[A0_size];"]
    assert not any(
        re.search(
            r"std::vector<(?:int|int64_t)> _(?:count|offset)\d+|"
            r"_offset\d+\[_i \+ 1\] = _offset\d+\[_i\] \+ _count\d+\[_i\]|"
            r"int64_t _total\d+ = _offset\d+\[",
            code,
        )
        for code in top_level_codes
    )
    assert not any(
        "Result1_pos_torch" in code
        or "Result1_pos_data" in code
        or "_offset1[_i]" in code
        for code in top_level_codes
    )
    assert LLIRLowerer().lower_llir(
        [
            *_first_position_initializations(result.statements),
            *_first_position_copy_loops(result.statements),
        ]
    ) == (
        "torch::Tensor Result1_pos_torch = "
        "torch::empty({(int64_t) (A0_size + 1)}, torch::kInt);\n"
        "int* Result1_pos_data = Result1_pos_torch.data_ptr<int>();\n"
        "for (int _i = 0; _i <= A0_size; _i++) {\n"
        "  Result1_pos_data[_i] = (int) _offset1[_i];\n"
        "}"
    )
    assert not any("Result1_crd_torch" in code for code in top_level_codes)
    assert [
        LLIRLowerer().lower_llir(initialization)
        for initialization in _compressed_coordinate_initializations(result.statements)
    ] == [
        "torch::Tensor Result1_crd_torch = " "torch::empty({_total1}, torch::kInt);",
        "int* Result1_crd_data = Result1_crd_torch.data_ptr<int>();",
    ]
    assert [
        LLIRLowerer().lower_llir(initialization)
        for initialization in _value_initializations(result.statements)
    ] == [
        "torch::Tensor Result_values_torch = "
        "torch::empty({_total1}, torch::kFloat32);",
        "float* Result_values_data = Result_values_torch.data_ptr<float>();",
    ]
    assert not any(
        "Result_values_torch" in code or "Result_values_data" in code
        for code in top_level_codes
    )
    declaration, mode_indices, values, return_statement = _result_assembly_tail(
        result.statements
    )
    assert LLIRLowerer().lower_llir(declaration) == "Tensor Result;"
    assert LLIRLowerer().lower_llir(mode_indices) == (
        "Result.storage.index.mode_indices = "
        "{{}, {Result1_pos_torch, Result1_crd_torch}};"
    )
    assert LLIRLowerer().lower_llir(values) == (
        "Result.storage.value = Result_values_torch;"
    )
    assert LLIRLowerer().lower_llir(return_statement) == "return Result;"
    assert not any(
        "Tensor Result;" in code
        or "storage.index.mode_indices" in code
        or "storage.value" in code
        or "return Result" in code
        for code in top_level_codes
    )

    flop = (
        "(long)A1_pos[A0_size] * (B0_size > 0 ? " "(B1_pos[B0_size] / B0_size) + 1 : 1)"
    )
    for loop in (count_loop, fill_loop):
        assert loop.omp_schedule == "dynamic, 64"
        assert loop.omp_num_threads == (
            f"scorch_nthreads({flop}, A0_size, " "SCORCH_GRAIN_CODEGEN_SPGEMM)"
        )
        assert loop.omp_chunk_expr == (
            f"scorch_chunk(A0_size, {flop}, SCORCH_GRAIN_CODEGEN_SPGEMM)"
        )


def test_structured_position_bounds_drive_the_exact_spgemm_policy() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_structured_ds_work_body())]

    result = transform_compressed_where_for_openmp(source, _context())

    flop = (
        "(long)A1_pos[A0_size] * (B0_size > 0 ? " "(B1_pos[B0_size] / B0_size) + 1 : 1)"
    )
    for loop in _phase_loops(result):
        assert loop.omp_num_threads == (
            f"scorch_nthreads({flop}, A0_size, SCORCH_GRAIN_CODEGEN_SPGEMM)"
        )
        assert loop.omp_chunk_expr == (
            f"scorch_chunk(A0_size, {flop}, SCORCH_GRAIN_CODEGEN_SPGEMM)"
        )


def test_dss_transform_builds_each_compressed_boundary_once() -> None:
    boundary = llir.IfThenElse(
        cond=llir.BinOp(
            "<",
            llir.FunctionCall("Result2_pos.back", []),
            _var("pResult2", llir.DataType.INT64),
        ),
        then_body=[
            llir.FunctionCallStmt("Result1_crd.push_back", [_var("parent_coordinate")])
        ],
    )
    body: List[llir.Stmt] = [
        _workspace_init(),
        llir.FunctionCallStmt("Result1_crd.push_back", [_var("row_coordinate")]),
        boundary,
        llir.FunctionCallStmt("Result2_crd.push_back", [_var("leaf_coordinate")]),
        llir.FunctionCallStmt("Result_values.push_back", [_var("value")]),
    ]

    result = transform_compressed_where_for_openmp(
        [_compatible_loop(body)], _context((1, 2))
    )

    count_loop, fill_loop = _phase_loops(result)
    count_state_codes = _phase_state_codes(count_loop.body)
    all_codes = _raw_codes(result.statements)
    count_boundary = cast(
        llir.IfThenElse,
        next(
            statement
            for statement in count_loop.body
            if type(statement) is llir.IfThenElse
        ),
    )
    fill_boundary = cast(
        llir.IfThenElse,
        next(
            statement
            for statement in fill_loop.body
            if type(statement) is llir.IfThenElse
        ),
    )

    assert "int _cnt1 = 0" in count_state_codes
    assert "int _cnt2 = 0" in count_state_codes
    assert "int _prev2 = 0" in count_state_codes
    assert cast(llir.BinOp, count_boundary.cond).op == ">"
    assert _phase_state_codes(count_boundary.then_body) == [
        "_cnt1++",
        "_prev2 = _cnt2",
    ]
    assert cast(llir.BinOp, fill_boundary.cond).op == ">"
    assert _phase_state_codes(fill_boundary.then_body) == [
        "_pos1++",
        "_prev2 = _pos2",
    ]
    assert [
        code
        for code in _assignment_codes(fill_boundary.then_body)
        if not code.startswith("_prev")
    ] == [
        "Result1_crd_data[_base1 + _pos1] = parent_coordinate",
        "Result2_pos_data[_base1 + _pos1] = _base2 + _pos2",
    ]
    assert "Result2_crd_data[_base2 + _pos2] = leaf_coordinate" in _assignment_codes(
        fill_loop.body
    )
    assert "Result2_pos_data[0] = 0" in _assignment_codes(result.statements)
    assert "Result2_pos_data[0] = 0" not in all_codes
    assert _assignment_codes(result.statements).count("Result2_pos_data[0] = 0") == 1
    _, mode_indices, _, _ = _result_assembly_tail(result.statements)
    assert LLIRLowerer().lower_llir(mode_indices) == (
        "Result.storage.index.mode_indices = "
        "{{}, {Result1_pos_torch, Result1_crd_torch}, "
        "{Result2_pos_torch, Result2_crd_torch}};"
    )


@pytest.mark.parametrize(
    "bound_type",
    (llir.DataType.INT, llir.DataType.INT64),
)
def test_first_compressed_position_allocation_is_typed_ordered_and_detached(
    bound_type: llir.DataType,
) -> None:
    source = [
        _compatible_loop(
            _ds_work_body(),
            bound="extent",
            bound_type=bound_type,
        )
    ]
    source_snapshot = _structural_snapshot(source)
    source_bound = cast(
        llir.Var,
        cast(llir.BinOp, cast(llir.ForLoop, source[0]).cond).right,
    )

    first = transform_compressed_where_for_openmp(source, _context())
    second = transform_compressed_where_for_openmp(source, _context())
    first_initializations = _first_position_initializations(first.statements)
    second_initializations = _first_position_initializations(second.statements)
    first_copy_loops = _first_position_copy_loops(first.statements)
    second_copy_loops = _first_position_copy_loops(second.statements)
    first_family: List[llir.Stmt] = [*first_initializations, *first_copy_loops]
    second_family: List[llir.Stmt] = [*second_initializations, *second_copy_loops]

    assert [type(statement) for statement in first_family] == [
        llir.VarInit,
        llir.VarInit,
        llir.ForLoop,
    ]
    assert [initialization.var.name for initialization in first_initializations] == [
        "Result1_pos_torch",
        "Result1_pos_data",
    ]
    assert len(first_copy_loops) == len(second_copy_loops) == 1
    assert _structural_snapshot(first_family) == _structural_snapshot(second_family)
    assert _mutable_ir_ids(first_family).isdisjoint(_mutable_ir_ids(second_family))
    assert _structural_snapshot(source) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(first.statements))

    owner, pointer = first_initializations
    copy_loop = first_copy_loops[0]
    assert owner.var.type is llir.DataType.TORCH_TENSOR
    assert type(owner.value) is llir.FunctionCall
    empty = cast(llir.FunctionCall, owner.value)
    assert empty.name == "torch::empty"
    extent = cast(llir.Array, empty.args[0])
    assert type(extent) is llir.Array
    assert extent.data_type is llir.DataType.INT64
    assert type(extent.values[0]) is llir.Cast
    extent_cast = cast(llir.Cast, extent.values[0])
    assert extent_cast.data_type is llir.DataType.INT64
    assert type(extent_cast.expr) is llir.Add
    extent_add = cast(llir.Add, extent_cast.expr)
    assert type(extent_add.left) is llir.Var
    assert extent_add.left.name == "extent"
    assert extent_add.left.type is bound_type
    assert type(extent_add.right) is llir.Literal
    assert extent_add.right.value == 1
    assert extent_add.right.data_type is llir.DataType.INT
    assert type(empty.args[1]) is llir.QualifiedName
    assert cast(llir.QualifiedName, empty.args[1]).namespace == "torch"
    assert cast(llir.QualifiedName, empty.args[1]).name == "kInt"

    assert pointer.var.type is llir.DataType.PTR_INT
    assert type(pointer.value) is llir.MemberCall
    data_ptr = cast(llir.MemberCall, pointer.value)
    assert data_ptr.member == "data_ptr"
    assert data_ptr.template_args == (llir.DataType.INT,)
    assert data_ptr.args == ()
    assert type(data_ptr.base) is llir.Var
    assert data_ptr.base.name == "Result1_pos_torch"
    assert data_ptr.base.type is llir.DataType.TORCH_TENSOR

    assert type(copy_loop.init) is llir.VarInit
    assert copy_loop.init.var.name == "_i"
    assert copy_loop.init.var.type is llir.DataType.INT
    assert type(copy_loop.init.value) is llir.Literal
    assert copy_loop.init.value.value == 0
    assert type(copy_loop.cond) is llir.BinOp
    assert copy_loop.cond.op == "<="
    assert type(copy_loop.cond.left) is llir.Var
    assert copy_loop.cond.left.name == "_i"
    assert type(copy_loop.cond.right) is llir.Var
    assert copy_loop.cond.right.name == "extent"
    assert copy_loop.cond.right.type is bound_type
    assert type(copy_loop.update) is llir.Increment
    assert copy_loop.update.var.name == "_i"
    assert len(copy_loop.body) == 1
    assignment = cast(llir.Assign, copy_loop.body[0])
    assert type(assignment) is llir.Assign
    assert type(assignment.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, assignment.var)
    assert type(target.array) is llir.Var
    assert target.array.name == "Result1_pos_data"
    assert target.array.type is llir.DataType.PTR_INT
    assert type(target.index) is llir.Var
    assert target.index.name == "_i"
    assert type(assignment.value) is llir.Cast
    value_cast = cast(llir.Cast, assignment.value)
    assert value_cast.data_type is llir.DataType.INT
    assert type(value_cast.expr) is llir.ArrayAccess
    offset_access = cast(llir.ArrayAccess, value_cast.expr)
    assert type(offset_access.array) is llir.Var
    assert offset_access.array.name == "_offset1"
    assert offset_access.array.type is llir.DataType.STD_VECTOR_INT
    assert type(offset_access.index) is llir.Var
    assert offset_access.index.name == "_i"

    assert LLIRLowerer().lower_llir(first_family) == (
        "torch::Tensor Result1_pos_torch = "
        "torch::empty({(int64_t) (extent + 1)}, torch::kInt);\n"
        "int* Result1_pos_data = Result1_pos_torch.data_ptr<int>();\n"
        "for (int _i = 0; _i <= extent; _i++) {\n"
        "  Result1_pos_data[_i] = (int) _offset1[_i];\n"
        "}"
    )
    assert not any(
        "Result1_pos_torch" in code
        or "Result1_pos_data" in code
        or "_offset1[_i]" in code
        for code in _raw_codes(first.statements)
    )
    bound_references = _named_vars(first_family, "extent")
    assert len(bound_references) == 2
    assert all(bound.type is bound_type for bound in bound_references)
    assert len({id(bound) for bound in bound_references}) == 2
    assert all(bound is not source_bound for bound in bound_references)

    owner.var.name = "owned_owner"
    cast(llir.Var, copy_loop.cond.right).name = "owned_bound"
    second_owner = second_initializations[0]
    second_loop = second_copy_loops[0]
    assert second_owner.var.name == "Result1_pos_torch"
    assert cast(llir.Var, second_loop.cond.right).name == "extent"
    assert source_bound.name == "extent"


@pytest.mark.parametrize(
    ("compressed_levels", "expected_coordinate_names"),
    (
        (
            (1,),
            ("Result1_crd_torch", "Result1_crd_data"),
        ),
        (
            (1, 2),
            (
                "Result1_crd_torch",
                "Result1_crd_data",
                "Result2_crd_torch",
                "Result2_crd_data",
            ),
        ),
    ),
)
def test_compressed_coordinate_allocations_are_typed_ordered_and_detached(
    compressed_levels: Tuple[int, ...],
    expected_coordinate_names: Tuple[str, ...],
) -> None:
    source = [_compatible_loop(_ds_work_body())]
    source_snapshot = _structural_snapshot(source)

    first = transform_compressed_where_for_openmp(source, _context(compressed_levels))
    second = transform_compressed_where_for_openmp(source, _context(compressed_levels))

    first_coordinates = _compressed_coordinate_initializations(first.statements)
    second_coordinates = _compressed_coordinate_initializations(second.statements)
    first_position_initializations = _first_position_initializations(first.statements)
    first_position_copy_loops = _first_position_copy_loops(first.statements)
    first_deeper_owners = _deeper_position_owner_initializations(first.statements)
    first_deeper_pointers = _deeper_position_pointer_initializations(first.statements)
    first_deeper_sentinels = _deeper_position_zero_sentinels(first.statements)
    assert [initialization.var.name for initialization in first_coordinates] == list(
        expected_coordinate_names
    )
    assert _structural_snapshot(first_coordinates) == _structural_snapshot(
        second_coordinates
    )
    assert _mutable_ir_ids(first_coordinates).isdisjoint(
        _mutable_ir_ids(second_coordinates)
    )
    assert _structural_snapshot(source) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(first.statements))

    raw_codes = _raw_codes(first.statements)
    assert not any(
        re.search(
            r"torch::Tensor Result\d+_crd_torch|Result\d+_crd_data\s*=",
            code,
        )
        or re.search(
            r"torch::Tensor Result\d+_pos_torch|Result\d+_pos_data\s*=",
            code,
        )
        or "Result1_pos_data[_i]" in code
        for code in raw_codes
    )

    statements = first.statements
    coordinate_indices = [
        index
        for index, statement in enumerate(statements)
        if type(statement) is llir.VarInit
        and cast(llir.VarInit, statement).var.name in expected_coordinate_names
    ]
    first_position_ids = {
        id(statement)
        for statement in (
            *first_position_initializations,
            *first_position_copy_loops,
        )
    }
    first_position_indices = [
        index
        for index, statement in enumerate(statements)
        if id(statement) in first_position_ids
    ]
    value_initializations = _value_initializations(statements)
    value_ids = {id(statement) for statement in value_initializations}
    value_indices = [
        index
        for index, statement in enumerate(statements)
        if id(statement) in value_ids
    ]
    fill_loop = next(
        index
        for index, statement in enumerate(statements)
        if type(statement) is llir.ForLoop and statement is _phase_loops(first)[1]
    )

    assert coordinate_indices == list(
        range(coordinate_indices[0], coordinate_indices[0] + len(coordinate_indices))
    )
    assert [
        initialization.var.name for initialization in first_position_initializations
    ] == [
        "Result1_pos_torch",
        "Result1_pos_data",
    ]
    assert len(first_position_copy_loops) == 1
    assert first_position_indices == list(
        range(first_position_indices[0], first_position_indices[0] + 3)
    )
    assert first_position_indices[-1] < coordinate_indices[0]
    if compressed_levels == (1, 2):
        assert [owner.var.name for owner in first_deeper_owners] == [
            "Result2_pos_torch"
        ]
        assert [pointer.var.name for pointer in first_deeper_pointers] == [
            "Result2_pos_data"
        ]
        assert len(first_deeper_sentinels) == 1
        deeper_ids = {
            id(statement)
            for statement in (
                *first_deeper_owners,
                *first_deeper_pointers,
                *first_deeper_sentinels,
            )
        }
        deeper_indices = [
            index
            for index, statement in enumerate(statements)
            if id(statement) in deeper_ids
        ]
        assert deeper_indices == list(
            range(deeper_indices[0], deeper_indices[0] + len(deeper_indices))
        )
        assert coordinate_indices[-1] < deeper_indices[0]
        assert deeper_indices[-1] < value_indices[0]
    else:
        assert first_deeper_owners == []
        assert first_deeper_pointers == []
        assert first_deeper_sentinels == []
        assert not any("Result2_pos_torch" in code for code in raw_codes)
    assert [value.var.name for value in value_initializations] == [
        "Result_values_torch",
        "Result_values_data",
    ]
    assert value_indices == list(range(value_indices[0], value_indices[0] + 2))
    assert coordinate_indices[-1] < value_indices[0]
    assert value_indices[-1] < fill_loop


def test_dss_deeper_position_allocation_is_typed_ordered_and_detached() -> None:
    source = [_compatible_loop(_ds_work_body())]
    source_snapshot = _structural_snapshot(source)

    first = transform_compressed_where_for_openmp(source, _context((1, 2)))
    second = transform_compressed_where_for_openmp(source, _context((1, 2)))

    first_owners = _deeper_position_owner_initializations(first.statements)
    first_pointers = _deeper_position_pointer_initializations(first.statements)
    first_sentinels = _deeper_position_zero_sentinels(first.statements)
    second_family: List[llir.Stmt] = [
        *_deeper_position_owner_initializations(second.statements),
        *_deeper_position_pointer_initializations(second.statements),
        *_deeper_position_zero_sentinels(second.statements),
    ]
    assert len(first_owners) == 1
    assert len(first_pointers) == 1
    assert len(first_sentinels) == 1
    owner = first_owners[0]
    pointer = first_pointers[0]
    sentinel = first_sentinels[0]
    first_family: List[llir.Stmt] = [owner, pointer, sentinel]

    assert _structural_snapshot(first_family) == _structural_snapshot(second_family)
    assert _structural_snapshot(source) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(first.statements))
    assert _mutable_ir_ids(first_family).isdisjoint(_mutable_ir_ids(second_family))

    assert type(owner.var) is llir.Var
    assert owner.var.name == "Result2_pos_torch"
    assert owner.var.type is llir.DataType.TORCH_TENSOR
    assert owner.var.is_ptr is False
    assert owner.var.is_restrict is False
    assert owner.var.tensor_access is None
    assert type(owner.value) is llir.FunctionCall
    empty = cast(llir.FunctionCall, owner.value)
    assert empty.name == "torch::empty"
    assert type(empty.args) is tuple
    assert len(empty.args) == 2
    extent = cast(llir.Array, empty.args[0])
    dtype = cast(llir.QualifiedName, empty.args[1])
    assert type(extent) is llir.Array
    assert type(extent.values) is tuple
    assert extent.data_type is llir.DataType.INT64
    assert len(extent.values) == 1
    cardinality = cast(llir.Add, extent.values[0])
    assert type(cardinality) is llir.Add
    assert cardinality.op == "+"
    total = cast(llir.Var, cardinality.left)
    one = cast(llir.Literal, cardinality.right)
    assert type(total) is llir.Var
    assert total.name == "_total1"
    assert total.type is llir.DataType.INT64
    assert total.is_ptr is False
    assert total.is_restrict is False
    assert total.tensor_access is None
    assert type(one) is llir.Literal
    assert one.value == 1
    assert one.data_type is llir.DataType.INT
    assert type(dtype) is llir.QualifiedName
    assert dtype.namespace == "torch"
    assert dtype.name == "kInt"
    assert dtype.data_type is llir.DataType.TORCH_SCALAR_TYPE

    assert type(pointer.var) is llir.Var
    assert pointer.var.name == "Result2_pos_data"
    assert pointer.var.type is llir.DataType.PTR_INT
    assert pointer.var.is_ptr is False
    assert pointer.var.is_restrict is False
    assert pointer.var.tensor_access is None
    assert type(pointer.value) is llir.MemberCall
    data_ptr = cast(llir.MemberCall, pointer.value)
    assert data_ptr.member == "data_ptr"
    assert data_ptr.template_args == (llir.DataType.INT,)
    assert data_ptr.args == ()
    assert type(data_ptr.base) is llir.Var
    assert data_ptr.base.name == "Result2_pos_torch"
    assert data_ptr.base.type is llir.DataType.TORCH_TENSOR
    assert data_ptr.base.is_ptr is False
    assert data_ptr.base.is_restrict is False
    assert data_ptr.base.tensor_access is None

    assert type(sentinel.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, sentinel.var)
    assert type(target.array) is llir.Var
    assert target.array.name == "Result2_pos_data"
    assert target.array.type is llir.DataType.PTR_INT
    assert target.array.is_ptr is False
    assert target.array.is_restrict is False
    assert target.array.tensor_access is None
    assert type(target.index) is llir.Literal
    assert target.index.value == 0
    assert target.index.data_type is llir.DataType.INT
    assert target.tensor_access is None
    assert type(sentinel.value) is llir.Literal
    assert sentinel.value.value == 0
    assert sentinel.value.data_type is llir.DataType.INT
    assert sentinel.op is llir.AssignOp.ASSIGN
    assert sentinel.cast is False

    assert [LLIRLowerer().lower_llir(statement) for statement in first_family] == [
        "torch::Tensor Result2_pos_torch = "
        "torch::empty({_total1 + 1}, torch::kInt);",
        "int* Result2_pos_data = Result2_pos_torch.data_ptr<int>();",
        "Result2_pos_data[0] = 0;",
    ]
    family_ids = {id(statement) for statement in first_family}
    family_indices = [
        index
        for index, statement in enumerate(first.statements)
        if id(statement) in family_ids
    ]
    coordinate_indices = [
        index
        for index, statement in enumerate(first.statements)
        if type(statement) is llir.VarInit
        and cast(llir.VarInit, statement).var.name
        in {
            "Result1_crd_torch",
            "Result1_crd_data",
            "Result2_crd_torch",
            "Result2_crd_data",
        }
    ]
    value_initializations = _value_initializations(first.statements)
    value_ids = {id(statement) for statement in value_initializations}
    value_indices = [
        index
        for index, statement in enumerate(first.statements)
        if id(statement) in value_ids
    ]
    fill_loop = next(
        index
        for index, statement in enumerate(first.statements)
        if type(statement) is llir.ForLoop and statement is _phase_loops(first)[1]
    )
    assert family_indices == list(
        range(family_indices[0], family_indices[0] + len(family_indices))
    )
    assert coordinate_indices[-1] < family_indices[0]
    assert [value.var.name for value in value_initializations] == [
        "Result_values_torch",
        "Result_values_data",
    ]
    assert value_indices == list(range(value_indices[0], value_indices[0] + 2))
    assert family_indices[-1] < value_indices[0]
    assert value_indices[-1] < fill_loop
    assert not any(
        "Result2_pos_torch" in code or "Result2_pos_data" in code
        for code in _raw_codes(first.statements)
    )


@pytest.mark.parametrize(
    ("compressed_levels", "leaf_level"),
    [
        ((1,), 1),
        ((1, 2), 2),
    ],
)
def test_compressed_value_allocation_is_typed_ordered_and_detached(
    compressed_levels: Tuple[int, ...],
    leaf_level: int,
) -> None:
    source = [_compatible_loop(_ds_work_body())]
    source_snapshot = _structural_snapshot(source)

    first = transform_compressed_where_for_openmp(source, _context(compressed_levels))
    second = transform_compressed_where_for_openmp(source, _context(compressed_levels))
    first_values = _value_initializations(first.statements)
    second_values = _value_initializations(second.statements)

    assert [initialization.var.name for initialization in first_values] == [
        "Result_values_torch",
        "Result_values_data",
    ]
    assert first_values == second_values
    assert [hash(initialization) for initialization in first_values] == [
        hash(initialization) for initialization in second_values
    ]
    assert _structural_snapshot(source) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(first.statements))
    assert _mutable_ir_ids(first_values).isdisjoint(_mutable_ir_ids(second_values))

    owner, pointer = first_values
    assert owner.var.type is llir.DataType.TORCH_TENSOR
    assert owner.var.is_ptr is False
    assert owner.var.is_restrict is False
    assert owner.var.tensor_access is None
    assert type(owner.value) is llir.FunctionCall
    empty = cast(llir.FunctionCall, owner.value)
    assert empty.name == "torch::empty"
    assert type(empty.args) is tuple
    assert len(empty.args) == 2
    assert type(empty.args[0]) is llir.Array
    extent = cast(llir.Array, empty.args[0])
    assert type(extent.values) is tuple
    assert extent.data_type is llir.DataType.INT64
    assert len(extent.values) == 1
    assert type(extent.values[0]) is llir.Var
    total = cast(llir.Var, extent.values[0])
    assert total.name == f"_total{leaf_level}"
    assert total.type is llir.DataType.INT64
    assert total.is_ptr is False
    assert total.is_restrict is False
    assert total.tensor_access is None
    assert type(empty.args[1]) is llir.QualifiedName
    dtype = cast(llir.QualifiedName, empty.args[1])
    assert dtype.namespace == "torch"
    assert dtype.name == "kFloat32"
    assert dtype.data_type is llir.DataType.TORCH_SCALAR_TYPE

    assert pointer.var.type is llir.DataType.PTR_FLOAT32
    assert pointer.var.is_ptr is False
    assert pointer.var.is_restrict is False
    assert pointer.var.tensor_access is None
    assert type(pointer.value) is llir.MemberCall
    data_ptr = cast(llir.MemberCall, pointer.value)
    assert data_ptr.member == "data_ptr"
    assert data_ptr.template_args == (llir.DataType.FLOAT32,)
    assert data_ptr.args == ()
    assert type(data_ptr.base) is llir.Var
    assert data_ptr.base.name == "Result_values_torch"
    assert data_ptr.base.type is llir.DataType.TORCH_TENSOR
    assert data_ptr.base.is_ptr is False
    assert data_ptr.base.is_restrict is False
    assert data_ptr.base.tensor_access is None
    assert owner.var is not data_ptr.base

    expected = [
        llir.VarInit(
            llir.Var("Result_values_torch", llir.DataType.TORCH_TENSOR),
            llir.FunctionCall(
                "torch::empty",
                (
                    llir.Array(
                        (llir.Var(f"_total{leaf_level}", llir.DataType.INT64),),
                        llir.DataType.INT64,
                    ),
                    llir.QualifiedName(
                        "torch",
                        "kFloat32",
                        llir.DataType.TORCH_SCALAR_TYPE,
                    ),
                ),
            ),
        ),
        llir.VarInit(
            llir.Var("Result_values_data", llir.DataType.PTR_FLOAT32),
            llir.MemberCall(
                llir.Var("Result_values_torch", llir.DataType.TORCH_TENSOR),
                "data_ptr",
                (llir.DataType.FLOAT32,),
            ),
        ),
    ]
    assert first_values == expected
    assert [LLIRLowerer().lower_llir(value) for value in first_values] == [
        "torch::Tensor Result_values_torch = "
        f"torch::empty({{_total{leaf_level}}}, torch::kFloat32);",
        "float* Result_values_data = Result_values_torch.data_ptr<float>();",
    ]
    assert not any(
        "Result_values_torch" in code or "Result_values_data" in code
        for code in _raw_codes(first.statements)
    )

    totals = _total_offset_loads(first.statements)
    assert total is not totals[-1].var
    value_ids = {id(statement) for statement in first_values}
    value_indices = [
        index
        for index, statement in enumerate(first.statements)
        if id(statement) in value_ids
    ]
    fill_index = first.statements.index(_phase_loops(first)[1])
    assert value_indices == list(range(value_indices[0], value_indices[0] + 2))
    if compressed_levels == (1, 2):
        deeper = [
            *_deeper_position_owner_initializations(first.statements),
            *_deeper_position_pointer_initializations(first.statements),
            *_deeper_position_zero_sentinels(first.statements),
        ]
        assert max(first.statements.index(statement) for statement in deeper) < (
            value_indices[0]
        )
    else:
        coordinates = _compressed_coordinate_initializations(first.statements)
        assert max(first.statements.index(statement) for statement in coordinates) < (
            value_indices[0]
        )
    assert value_indices[-1] < fill_index

    owner.var.name = "owned_values_torch"
    total.name = "owned_total"
    cast(llir.Var, data_ptr.base).name = "owned_receiver"
    assert [value.var.name for value in second_values] == [
        "Result_values_torch",
        "Result_values_data",
    ]
    second_empty = cast(llir.FunctionCall, second_values[0].value)
    second_extent = cast(llir.Array, second_empty.args[0])
    assert cast(llir.Var, second_extent.values[0]).name == f"_total{leaf_level}"
    second_call = cast(llir.MemberCall, second_values[1].value)
    assert cast(llir.Var, second_call.base).name == "Result_values_torch"
    assert totals[-1].var.name == f"_total{leaf_level}"


@pytest.mark.parametrize(
    ("compressed_levels", "expected_mode_indices"),
    [
        (
            (1,),
            "{{}, {Result1_pos_torch, Result1_crd_torch}}",
        ),
        (
            (1, 2),
            "{{}, {Result1_pos_torch, Result1_crd_torch}, "
            "{Result2_pos_torch, Result2_crd_torch}}",
        ),
    ],
)
def test_compressed_result_assembly_reuses_frozen_abi_epilogue(
    compressed_levels: Tuple[int, ...],
    expected_mode_indices: str,
) -> None:
    context = _context(compressed_levels)
    source = [_compatible_loop(_ds_work_body())]

    first = transform_compressed_where_for_openmp(source, context)
    second = transform_compressed_where_for_openmp(source, context)

    first_tail = list(_result_assembly_tail(first.statements))
    second_tail = list(_result_assembly_tail(second.statements))
    expected_tail = [
        context.result_assembler.emit_result_declaration(),
        *context.result_assembler.emit_storage_epilogue(),
    ]
    expected_cpp = (
        "Tensor Result;\n"
        f"Result.storage.index.mode_indices = {expected_mode_indices};\n"
        "Result.storage.value = Result_values_torch;\n"
        "return Result;"
    )

    assert _structural_snapshot(first_tail) == _structural_snapshot(expected_tail)
    assert _structural_snapshot(second_tail) == _structural_snapshot(first_tail)
    assert _mutable_ir_ids(first_tail).isdisjoint(_mutable_ir_ids(second_tail))
    assert _mutable_ir_ids(first_tail).isdisjoint(_mutable_ir_ids(expected_tail))
    assert LLIRLowerer().lower_llir(first_tail) == expected_cpp
    assert LLIRLowerer().lower_llir(second_tail) == expected_cpp
    assert not any(
        marker in code
        for code in _raw_codes(first.statements)
        for marker in (
            "Tensor Result;",
            "storage.index.mode_indices",
            "storage.value",
            "return Result",
        )
    )

    first_modes = cast(llir.Assign, first_tail[1]).value
    second_modes = cast(llir.Assign, second_tail[1]).value
    assert first_modes == second_modes
    assert hash(first_modes) == hash(second_modes)
    assert first_modes is not second_modes


def test_count_fill_state_is_typed_structural_fresh_and_never_raw() -> None:
    boundary = llir.IfThenElse(
        cond=llir.BinOp(
            "<",
            llir.FunctionCall("Result2_pos.back"),
            _var("pResult2", llir.DataType.INT64),
        ),
        then_body=[
            llir.FunctionCallStmt(
                "Result1_crd.push_back",
                [_var("parent_coordinate")],
            )
        ],
    )
    source: List[llir.Stmt] = [
        _compatible_loop(
            [
                _workspace_init(),
                llir.FunctionCallStmt(
                    "Result1_crd.push_back",
                    [_var("row_coordinate")],
                ),
                boundary,
                llir.FunctionCallStmt(
                    "Result2_crd.push_back",
                    [_var("leaf_coordinate")],
                ),
            ]
        )
    ]
    snapshot = _structural_snapshot(source)

    first = transform_compressed_where_for_openmp(source, _context((1, 2)))
    second = transform_compressed_where_for_openmp(source, _context((1, 2)))
    first_state = _phase_state_statements(first.statements)
    second_state = _phase_state_statements(second.statements)
    first_state_vars = _phase_state_vars(first.statements)
    second_state_vars = _phase_state_vars(second.statements)

    assert _structural_snapshot(source) == snapshot
    assert first_state == second_state
    assert _mutable_ir_ids(first_state).isdisjoint(_mutable_ir_ids(second_state))
    assert first_state_vars == second_state_vars
    assert {id(var) for var in first_state_vars}.isdisjoint(
        {id(var) for var in second_state_vars}
    )
    assert _phase_state_codes(first.statements) == _phase_state_codes(second.statements)
    assert not any(
        re.search(r"\b_(?:cnt|pos|prev)\d+\b", code)
        for code in _raw_codes(first.statements)
    )

    for statement in first_state:
        if type(statement) is llir.VarInit:
            initialization = cast(llir.VarInit, statement)
            assert type(initialization.var) is llir.Var
            assert initialization.var.type is llir.DataType.INT
            assert type(initialization.value) is llir.Literal
            literal = cast(llir.Literal, initialization.value)
            assert literal.value == 0
            assert literal.data_type is llir.DataType.INT
        elif type(statement) is llir.Increment:
            increment = cast(llir.Increment, statement)
            assert type(increment.var) is llir.Var
            assert increment.var.type is llir.DataType.INT
        else:
            assignment = cast(llir.Assign, statement)
            assert type(assignment.var) is llir.Var
            assert type(assignment.value) is llir.Var
            assert assignment.var.type is llir.DataType.INT
            assert assignment.value.type is llir.DataType.INT
            assert assignment.var is not assignment.value

    assert all(type(var) is llir.Var for var in first_state_vars)
    assert all(var.type is llir.DataType.INT for var in first_state_vars)
    assert Counter(var.name for var in first_state_vars) == {
        "_cnt1": 4,
        "_cnt2": 5,
        "_pos1": 7,
        "_pos2": 7,
        "_prev2": 6,
    }
    assert len({id(var) for var in first_state_vars}) == len(first_state_vars)


def test_fill_base_offset_loads_are_typed_owned_structural_and_never_raw() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_ds_work_body(), bound="A0_size")]
    source_header = cast(llir.VarInit, cast(llir.ForLoop, source[0]).init).var
    snapshot = _structural_snapshot(source)

    first = transform_compressed_where_for_openmp(source, _context((1, 2)))
    second = transform_compressed_where_for_openmp(source, _context((1, 2)))
    first_loads = _base_offset_loads(first.statements)
    second_loads = _base_offset_loads(second.statements)

    assert _structural_snapshot(source) == snapshot
    assert [load.var.name for load in first_loads] == ["_base1", "_base2"]
    assert first_loads == second_loads
    assert [hash(load) for load in first_loads] == [hash(load) for load in second_loads]
    assert _mutable_ir_ids(first_loads).isdisjoint(_mutable_ir_ids(second_loads))
    assert _mutable_ir_ids(first_loads[0]).isdisjoint(_mutable_ir_ids(first_loads[1]))
    assert [LLIRLowerer().lower_llir(load) for load in first_loads] == [
        "int64_t _base1 = _offset1[row];",
        "int64_t _base2 = _offset2[row];",
    ]
    assert not any(
        re.search(r"int64_t _base\d+ = _offset\d+\[", code)
        for code in _raw_codes(first.statements)
    )

    for level, initializer in enumerate(first_loads, start=1):
        assert type(initializer) is llir.VarInit
        assert type(initializer.var) is llir.Var
        assert initializer.var.name == f"_base{level}"
        assert initializer.var.type is llir.DataType.INT64
        assert initializer.var.is_ptr is False
        assert initializer.var.is_restrict is False
        assert initializer.var.tensor_access is None
        assert initializer.op == "="
        assert initializer.cast is False

        assert type(initializer.value) is llir.ArrayAccess
        access = cast(llir.ArrayAccess, initializer.value)
        assert access.tensor_access is None
        assert type(access.array) is llir.Var
        offset = cast(llir.Var, access.array)
        assert offset.name == f"_offset{level}"
        assert offset.type is llir.DataType.STD_VECTOR_INT
        assert offset.is_ptr is False
        assert offset.is_restrict is False
        assert offset.tensor_access is None
        assert type(access.index) is llir.Var
        index = cast(llir.Var, access.index)
        assert index.name == "row"
        assert index.type is llir.DataType.INT
        assert index.is_ptr is False
        assert index.is_restrict is False
        assert index.tensor_access is None
        assert index is not source_header
        assert len({id(initializer.var), id(offset), id(index)}) == 3

        expected = llir.VarInit(
            llir.Var(f"_base{level}", llir.DataType.INT64),
            llir.ArrayAccess(
                llir.Var(f"_offset{level}", llir.DataType.STD_VECTOR_INT),
                llir.Var("row", llir.DataType.INT),
            ),
        )
        assert initializer == expected
        assert hash(initializer) == hash(expected)

    first_access = cast(llir.ArrayAccess, first_loads[0].value)
    cast(llir.Var, first_access.array).name = "owned_offset"
    cast(llir.Var, first_access.index).name = "owned_index"
    sibling_access = cast(llir.ArrayAccess, first_loads[1].value)
    assert cast(llir.Var, sibling_access.array).name == "_offset2"
    assert cast(llir.Var, sibling_access.index).name == "row"
    assert cast(llir.Var, cast(llir.ArrayAccess, second_loads[0].value).array).name == (
        "_offset1"
    )
    assert cast(llir.Var, cast(llir.ArrayAccess, second_loads[0].value).index).name == (
        "row"
    )
    assert source_header.name == "row"


def test_offset_family_is_typed_owned_structural_fresh_and_never_raw() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_ds_work_body(), bound="A0_size")]
    source_loop = cast(llir.ForLoop, source[0])
    source_bound = cast(llir.Var, cast(llir.BinOp, source_loop.cond).right)
    snapshot = _structural_snapshot(source)

    first = transform_compressed_where_for_openmp(source, _context((1, 2)))
    second = transform_compressed_where_for_openmp(source, _context((1, 2)))
    first_owners = _offset_family_direct_initializations(first.statements)
    second_owners = _offset_family_direct_initializations(second.statements)
    first_prefix_loops = _prefix_sum_loops(first.statements)
    second_prefix_loops = _prefix_sum_loops(second.statements)
    first_totals = _total_offset_loads(first.statements)
    second_totals = _total_offset_loads(second.statements)
    first_zeroes = _offset_zero_assignments(first.statements)
    second_zeroes = _offset_zero_assignments(second.statements)

    assert _structural_snapshot(source) == snapshot
    assert _structural_snapshot(first.statements) == _structural_snapshot(
        second.statements
    )
    assert [owner.var.name for owner in first_owners] == [
        "_count1",
        "_count2",
        "_offset1",
        "_offset2",
    ]
    assert first_owners == second_owners
    assert [hash(owner) for owner in first_owners] == [
        hash(owner) for owner in second_owners
    ]
    assert len(first_prefix_loops) == len(second_prefix_loops) == 2
    assert [load.var.name for load in first_totals] == ["_total1", "_total2"]
    assert first_totals == second_totals
    assert [hash(load) for load in first_totals] == [
        hash(load) for load in second_totals
    ]
    assert len(first_zeroes) == len(second_zeroes) == 2

    count_loop, _ = _phase_loops(first)
    family_order: List[str] = []
    for statement in first.statements:
        if type(statement) is llir.DirectInit and re.fullmatch(
            r"_(?:count|offset)\d+", statement.var.name
        ):
            family_order.append(statement.var.name)
        elif statement is count_loop:
            family_order.append("count-loop")
        elif statement in first_zeroes:
            target = cast(llir.ArrayAccess, cast(llir.Assign, statement).var)
            family_order.append(f"{cast(llir.Var, target.array).name}[0]")
        elif statement in first_prefix_loops:
            target = cast(
                llir.ArrayAccess,
                cast(llir.Assign, cast(llir.ForLoop, statement).body[0]).var,
            )
            family_order.append(
                f"prefix-{cast(llir.Var, target.array).name.removeprefix('_offset')}"
            )
        elif statement in first_totals:
            family_order.append(cast(llir.VarInit, statement).var.name)
    assert family_order == [
        "_count1",
        "_count2",
        "count-loop",
        "_offset1",
        "_offset1[0]",
        "prefix-1",
        "_offset2",
        "_offset2[0]",
        "prefix-2",
        "_total1",
        "_total2",
    ]

    family_nodes: List[llir.Node] = [
        *first_owners,
        *first_zeroes,
        *first_prefix_loops,
        *first_totals,
    ]
    for index, node in enumerate(family_nodes):
        for sibling in family_nodes[index + 1 :]:
            assert _mutable_ir_ids(node).isdisjoint(_mutable_ir_ids(sibling))
    assert _mutable_ir_ids(family_nodes).isdisjoint(
        _mutable_ir_ids(
            [
                *second_owners,
                *second_zeroes,
                *second_prefix_loops,
                *second_totals,
            ]
        )
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(family_nodes))

    for level, owner in enumerate(first_owners[:2], start=1):
        assert type(owner) is llir.DirectInit
        assert type(owner.var) is llir.Var
        assert owner.var.name == f"_count{level}"
        assert owner.var.type is llir.DataType.STD_VECTOR_C_INT
        assert owner.var.is_ptr is False
        assert owner.var.is_restrict is False
        assert owner.var.tensor_access is None
        assert type(owner.args) is tuple
        assert len(owner.args) == 2
        extent = cast(llir.Cast, owner.args[0])
        assert type(extent) is llir.Cast
        assert extent.data_type is llir.DataType.SIZE_T
        assert type(extent.expr) is llir.Var
        bound = cast(llir.Var, extent.expr)
        assert (bound.name, bound.type) == ("A0_size", llir.DataType.INT64)
        assert bound.is_ptr is False
        assert bound.is_restrict is False
        assert bound.tensor_access is None
        fill = cast(llir.Literal, owner.args[1])
        assert type(fill) is llir.Literal
        assert (fill.value, fill.data_type) == (0, llir.DataType.INT)
        expected = llir.DirectInit(
            llir.Var(f"_count{level}", llir.DataType.STD_VECTOR_C_INT),
            (
                llir.Cast(
                    llir.Var("A0_size", llir.DataType.INT64),
                    llir.DataType.SIZE_T,
                ),
                llir.Literal(0, llir.DataType.INT),
            ),
        )
        assert owner == expected
        assert hash(owner) == hash(expected)

    for level, owner in enumerate(first_owners[2:], start=1):
        assert type(owner) is llir.DirectInit
        assert type(owner.var) is llir.Var
        assert owner.var.name == f"_offset{level}"
        assert owner.var.type is llir.DataType.STD_VECTOR_INT
        assert owner.var.is_ptr is False
        assert owner.var.is_restrict is False
        assert owner.var.tensor_access is None
        assert type(owner.args) is tuple
        assert len(owner.args) == 1
        extent = cast(llir.Add, owner.args[0])
        assert type(extent) is llir.Add
        assert extent.op == "+"
        assert type(extent.left) is llir.Cast
        extent_cast = cast(llir.Cast, extent.left)
        assert extent_cast.data_type is llir.DataType.SIZE_T
        assert type(extent_cast.expr) is llir.Var
        bound = cast(llir.Var, extent_cast.expr)
        assert (bound.name, bound.type) == ("A0_size", llir.DataType.INT64)
        assert bound.is_ptr is False
        assert bound.is_restrict is False
        assert bound.tensor_access is None
        assert type(extent.right) is llir.Literal
        increment = cast(llir.Literal, extent.right)
        assert (increment.value, increment.data_type) == (1, llir.DataType.INT)
        expected = llir.DirectInit(
            llir.Var(f"_offset{level}", llir.DataType.STD_VECTOR_INT),
            (
                llir.Add(
                    llir.Cast(
                        llir.Var("A0_size", llir.DataType.INT64),
                        llir.DataType.SIZE_T,
                    ),
                    llir.Literal(1, llir.DataType.INT),
                ),
            ),
        )
        assert owner == expected
        assert hash(owner) == hash(expected)

    assert [LLIRLowerer().lower_llir(assignment) for assignment in first_zeroes] == [
        "_offset1[0] = 0;",
        "_offset2[0] = 0;",
    ]
    for level, assignment in enumerate(first_zeroes, start=1):
        target = cast(llir.ArrayAccess, assignment.var)
        assert target.tensor_access is None
        assert type(target.array) is llir.Var
        assert (
            cast(llir.Var, target.array).name,
            cast(llir.Var, target.array).type,
        ) == (f"_offset{level}", llir.DataType.STD_VECTOR_INT)
        assert type(target.index) is llir.Literal
        assert (
            cast(llir.Literal, target.index).value,
            cast(llir.Literal, target.index).data_type,
        ) == (0, llir.DataType.INT)
        assert type(assignment.value) is llir.Literal
        assert (
            cast(llir.Literal, assignment.value).value,
            cast(llir.Literal, assignment.value).data_type,
        ) == (0, llir.DataType.INT)

    for level, loop in enumerate(first_prefix_loops, start=1):
        assert type(loop) is llir.ForLoop
        assert loop.omp_parallel_for is False
        assert loop.omp_schedule is None
        assert loop.omp_num_threads is None
        assert loop.omp_chunk_expr is None
        assert type(loop.init) is llir.VarInit
        initialization = cast(llir.VarInit, loop.init)
        assert (initialization.var.name, initialization.var.type) == (
            "_i",
            llir.DataType.INT,
        )
        assert type(initialization.value) is llir.Literal
        assert (
            cast(llir.Literal, initialization.value).value,
            cast(llir.Literal, initialization.value).data_type,
        ) == (0, llir.DataType.INT)
        assert type(loop.cond) is llir.BinOp
        condition = cast(llir.BinOp, loop.cond)
        assert condition.op == "<"
        assert type(condition.left) is llir.Var
        assert (
            cast(llir.Var, condition.left).name,
            cast(llir.Var, condition.left).type,
        ) == (
            "_i",
            llir.DataType.INT,
        )
        assert type(condition.right) is llir.Var
        assert (
            cast(llir.Var, condition.right).name,
            cast(llir.Var, condition.right).type,
        ) == ("A0_size", llir.DataType.INT64)
        assert type(loop.update) is llir.Increment
        assert (
            cast(llir.Increment, loop.update).var.name,
            cast(llir.Increment, loop.update).var.type,
        ) == ("_i", llir.DataType.INT)
        assert len(loop.body) == 1
        assert type(loop.body[0]) is llir.Assign
        assignment = cast(llir.Assign, loop.body[0])
        assert type(assignment.var) is llir.ArrayAccess
        target = cast(llir.ArrayAccess, assignment.var)
        assert type(target.array) is llir.Var
        assert (
            cast(llir.Var, target.array).name,
            cast(llir.Var, target.array).type,
        ) == (f"_offset{level}", llir.DataType.STD_VECTOR_INT)
        assert type(target.index) is llir.Add
        target_index = cast(llir.Add, target.index)
        assert type(target_index.left) is llir.Var
        assert (
            cast(llir.Var, target_index.left).name,
            cast(llir.Var, target_index.left).type,
        ) == ("_i", llir.DataType.INT)
        assert type(target_index.right) is llir.Literal
        assert (
            cast(llir.Literal, target_index.right).value,
            cast(llir.Literal, target_index.right).data_type,
        ) == (1, llir.DataType.INT)
        assert type(assignment.value) is llir.Add
        value = cast(llir.Add, assignment.value)
        assert type(value.left) is llir.ArrayAccess
        assert type(value.right) is llir.ArrayAccess
        offset_access = cast(llir.ArrayAccess, value.left)
        count_access = cast(llir.ArrayAccess, value.right)
        assert type(offset_access.array) is llir.Var
        assert (
            cast(llir.Var, offset_access.array).name,
            cast(llir.Var, offset_access.array).type,
        ) == (f"_offset{level}", llir.DataType.STD_VECTOR_INT)
        assert type(count_access.array) is llir.Var
        assert (
            cast(llir.Var, count_access.array).name,
            cast(llir.Var, count_access.array).type,
        ) == (f"_count{level}", llir.DataType.STD_VECTOR_C_INT)
        assert type(offset_access.index) is llir.Var
        assert type(count_access.index) is llir.Var
        assert (
            cast(llir.Var, offset_access.index).name,
            cast(llir.Var, offset_access.index).type,
        ) == ("_i", llir.DataType.INT)
        assert (
            cast(llir.Var, count_access.index).name,
            cast(llir.Var, count_access.index).type,
        ) == ("_i", llir.DataType.INT)
        assert len({id(var) for var in _named_vars(loop, "_i")}) == 6

    for level, initializer in enumerate(first_totals, start=1):
        assert type(initializer) is llir.VarInit
        assert (initializer.var.name, initializer.var.type) == (
            f"_total{level}",
            llir.DataType.INT64,
        )
        assert initializer.var.is_ptr is False
        assert initializer.var.is_restrict is False
        assert initializer.var.tensor_access is None
        assert initializer.op == "="
        assert initializer.cast is False
        assert type(initializer.value) is llir.ArrayAccess
        access = cast(llir.ArrayAccess, initializer.value)
        assert access.tensor_access is None
        assert type(access.array) is llir.Var
        assert (
            cast(llir.Var, access.array).name,
            cast(llir.Var, access.array).type,
        ) == (f"_offset{level}", llir.DataType.STD_VECTOR_INT)
        assert type(access.index) is llir.Var
        assert (
            cast(llir.Var, access.index).name,
            cast(llir.Var, access.index).type,
        ) == ("A0_size", llir.DataType.INT64)
        expected = llir.VarInit(
            llir.Var(f"_total{level}", llir.DataType.INT64),
            llir.ArrayAccess(
                llir.Var(f"_offset{level}", llir.DataType.STD_VECTOR_INT),
                llir.Var("A0_size", llir.DataType.INT64),
            ),
        )
        assert initializer == expected
        assert hash(initializer) == hash(expected)

    assert [LLIRLowerer().lower_llir(owner) for owner in first_owners] == [
        "std::vector<int> _count1((size_t) A0_size, 0);",
        "std::vector<int> _count2((size_t) A0_size, 0);",
        "std::vector<int64_t> _offset1((size_t) A0_size + 1);",
        "std::vector<int64_t> _offset2((size_t) A0_size + 1);",
    ]
    assert [LLIRLowerer().lower_llir(loop) for loop in first_prefix_loops] == [
        "for (int _i = 0; _i < A0_size; _i++) {\n"
        f"  _offset{level}[_i + 1] = _offset{level}[_i] + _count{level}[_i];\n"
        "}"
        for level in (1, 2)
    ]
    assert [LLIRLowerer().lower_llir(load) for load in first_totals] == [
        "int64_t _total1 = _offset1[A0_size];",
        "int64_t _total2 = _offset2[A0_size];",
    ]
    assert not any(
        re.search(
            r"std::vector<(?:int|int64_t)> _(?:count|offset)\d+|"
            r"_offset\d+\[_i \+ 1\] = _offset\d+\[_i\] \+ _count\d+\[_i\]|"
            r"int64_t _total\d+ = _offset\d+\[",
            code,
        )
        for code in _raw_codes(first.statements)
    )

    first_bounds = _named_vars(family_nodes, "A0_size")
    second_bounds = _named_vars(
        [*second_owners, *second_prefix_loops, *second_totals],
        "A0_size",
    )
    assert len(first_bounds) == len(second_bounds) == 8
    assert all(bound.type is llir.DataType.INT64 for bound in first_bounds)
    assert len({id(bound) for bound in first_bounds}) == len(first_bounds)
    assert {id(bound) for bound in first_bounds}.isdisjoint(
        {id(bound) for bound in second_bounds}
    )
    assert all(bound is not source_bound for bound in first_bounds)

    first_owners[0].var.name = "owned_count"
    first_prefix_target = cast(
        llir.ArrayAccess,
        cast(llir.Assign, first_prefix_loops[0].body[0]).var,
    )
    cast(llir.Var, first_prefix_target.array).name = "owned_offset"
    assert first_owners[1].var.name == "_count2"
    assert second_owners[0].var.name == "_count1"
    second_prefix_target = cast(
        llir.ArrayAccess,
        cast(llir.Assign, second_prefix_loops[0].body[0]).var,
    )
    assert cast(llir.Var, second_prefix_target.array).name == "_offset1"
    assert source_bound.name == "A0_size"


@pytest.mark.parametrize(
    "bound_type",
    (llir.DataType.INT, llir.DataType.INT64),
)
def test_offset_family_preserves_exact_loop_bound_type(
    bound_type: llir.DataType,
) -> None:
    source = [
        _compatible_loop(
            _ds_work_body(),
            bound="extent",
            bound_type=bound_type,
        )
    ]
    source_bound = cast(
        llir.Var,
        cast(llir.BinOp, cast(llir.ForLoop, source[0]).cond).right,
    )

    result = transform_compressed_where_for_openmp(source, _context())
    family: List[llir.Node] = [
        *_offset_family_direct_initializations(result.statements),
        *_prefix_sum_loops(result.statements),
        *_total_offset_loads(result.statements),
    ]
    bounds = _named_vars(family, "extent")

    assert len(bounds) == 4
    assert all(type(bound) is llir.Var for bound in bounds)
    assert all(bound.type is bound_type for bound in bounds)
    assert all(bound.is_ptr is False for bound in bounds)
    assert all(bound.is_restrict is False for bound in bounds)
    assert all(bound.tensor_access is None for bound in bounds)
    assert len({id(bound) for bound in bounds}) == len(bounds)
    assert all(bound is not source_bound for bound in bounds)


def test_first_top_level_compatible_loop_is_selected_and_suffix_is_discarded() -> None:
    auto_prefix = llir.ForLoopAuto(
        _var("item"), _var("items"), [llir.RawStmt("keep_auto_prefix")]
    )
    incompatible = _compatible_loop(
        [llir.RawStmt("keep_incompatible")],
        bound="Ignored0_size",
    )
    cast(llir.VarInit, incompatible.init).var.name = "other"
    first = _compatible_loop([llir.RawStmt("first_work")], bound="First0_size")
    second = _compatible_loop([llir.RawStmt("second_work")], bound="Second0_size")
    source: List[llir.Stmt] = [
        auto_prefix,
        incompatible,
        first,
        second,
        llir.RawStmt("suffix"),
    ]

    result = transform_compressed_where_for_openmp(source, _context())

    count_loop, fill_loop = _phase_loops(result)
    assert cast(llir.Var, cast(llir.BinOp, count_loop.cond).right).name == (
        "First0_size"
    )
    assert cast(llir.Var, cast(llir.BinOp, fill_loop.cond).right).name == (
        "First0_size"
    )
    codes = _raw_codes(result.statements)
    assert "keep_auto_prefix" in codes
    assert "keep_incompatible" in codes
    assert codes.count("first_work") == 2
    assert "second_work" not in codes
    assert "suffix" not in codes


@pytest.mark.parametrize("nested", [False, True])
def test_no_top_level_loop_or_immediate_unextractable_bound_is_detached_no_op(
    nested: bool,
) -> None:
    valid_later = _compatible_loop([llir.RawStmt("later")], bound="Later0_size")
    if nested:
        source: List[llir.Stmt] = [
            llir.IfThenElse(cond=_var("guard"), then_body=[valid_later])
        ]
    else:
        source = [
            _compatible_loop(
                [llir.RawStmt("first")],
                bound="First0_size",
                cond_op="<=",
            ),
            valid_later,
        ]
    snapshot = _structural_snapshot(source)

    result = transform_compressed_where_for_openmp(source, _context())

    assert result.applied is False
    assert _structural_snapshot(result.statements) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(result.statements))


def test_legacy_prefix_and_work_body_filters_are_top_level_only() -> None:
    assembly_loop = _compatible_loop(
        [llir.RawStmt("drop_assembly_loop")], bound="limit"
    )
    assembly_loop.cond = llir.BinOp("<", _var("Result1_pos_index"), _var("limit"))
    position_loop = _compatible_loop([llir.RawStmt("drop_position_loop")])
    cast(llir.VarInit, position_loop.init).var.name = "pResult1"
    nested = llir.IfThenElse(
        cond=_var("guard"),
        then_body=[
            llir.RawStmt("keep_nested"),
            llir.DirectInit(
                llir.Var("Result1_pos_nested", llir.DataType.STD_VECTOR_C_INT),
                (llir.Literal(2), llir.Literal(0)),
            ),
        ],
    )
    selected = _compatible_loop(
        [
            _workspace_init(),
            assembly_loop,
            position_loop,
            llir.Assign(
                llir.ArrayAccess(_var("Result1_pos"), _var("row")),
                _var("pResult1"),
            ),
            llir.RawStmt("keep_work"),
            nested,
        ]
    )
    prefix_position_loop = _compatible_loop([llir.RawStmt("drop_prefix_loop")])
    cast(llir.VarInit, prefix_position_loop.init).var.name = "pResultPrefix"
    source = [
        llir.Comment("keep-prefix"),
        llir.VarDecl(_var("Result_values")),
        llir.VarDecl(_var("Result1_pos_dynamic")),
        llir.DirectInit(
            llir.Var("Result1_pos", llir.DataType.STD_VECTOR_C_INT),
            (llir.Literal(8), llir.Literal(0)),
        ),
        llir.DirectInit(
            llir.Var("Other1_pos", llir.DataType.STD_VECTOR_C_INT),
            (llir.Literal(8), llir.Literal(0)),
        ),
        llir.VarDecl(_var("Result1_crd")),
        llir.VarInit(_var("pResult1"), llir.Literal(0)),
        llir.VarInit(_var("Result1_pos_index"), llir.Literal(0)),
        llir.Assign(
            llir.ArrayAccess(_var("Result1_pos"), llir.Literal(0)),
            llir.Literal(0),
        ),
        prefix_position_loop,
        selected,
        llir.RawStmt("drop_suffix"),
    ]

    result = transform_compressed_where_for_openmp(source, _context())

    codes = _raw_codes(result.statements)
    assert codes.count("keep_work") == 2
    assert codes.count("keep_nested") == 2
    assert "drop_assembly_loop" not in codes
    assert "drop_position_loop" not in codes
    assert "drop_prefix_loop" not in codes
    assert "drop_suffix" not in codes
    assert [
        statement.value
        for statement in result.statements
        if type(statement) is llir.Comment
    ] == ["keep-prefix"]
    assert not any(
        type(statement) is llir.VarDecl
        and statement.var.name
        in {"Result_values", "Result1_pos_dynamic", "Result1_crd"}
        for statement in result.statements
    )
    assert [
        declaration.var.name
        for declaration in _direct_initializations(result.statements)
    ] == [
        "Other1_pos",
        "_count1",
        "Result1_pos_nested",
        "_offset1",
        "Result1_pos_nested",
    ]


def test_nested_control_flow_and_statement_containers_follow_legacy_scopes() -> None:
    def writes(marker: str) -> List[llir.Stmt]:
        return [
            llir.FunctionCallStmt("Result1_crd.push_back", [_var(marker)]),
            llir.FunctionCallStmt("wksp.insert", [_var(marker)]),
        ]

    nested_for = _compatible_loop(writes("for_value"), bound="inner_limit")
    auto_loop = llir.ForLoopAuto(_var("item"), _var("items"), writes("auto_value"))
    while_loop = llir.WhileLoop(_var("keep_going"), writes("while_value"))
    conditional = llir.IfThenElse(
        cond=_var("guard"),
        then_body=writes("then_value"),
        else_body=writes("else_value"),
        cond_list=[_var("other_guard")],
        then_body_list=[writes("branch_value")],
    )
    nested_list: List[LLIRStatementValue] = cast(
        List[LLIRStatementValue], writes("list_value")
    )
    nested_tuple = tuple(writes("tuple_value"))
    body = cast(
        List[llir.Stmt],
        [
            _workspace_init(),
            nested_for,
            auto_loop,
            while_loop,
            conditional,
            nested_list,
            nested_tuple,
        ],
    )

    result = transform_compressed_where_for_openmp([_compatible_loop(body)], _context())

    count_loop, _ = _phase_loops(result)
    count_state_codes = _phase_state_codes(count_loop.body)
    calls = _call_names(count_loop.body)
    assert count_state_codes.count("_cnt1++") == 8
    # The exact call-name rewrite descends into ForLoop bodies and If branches,
    # but not ForLoopAuto, WhileLoop, or bare nested statement containers.
    assert calls.count("wksp.insert_unchecked") == 4
    assert calls.count("wksp.insert") == 4


def test_workspace_insert_rewrite_matches_only_the_exact_call_name() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(1),
        tensor_id=SymbolId(2),
        index_ids=(IndexId(3),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    assignment = llir.Assign(
        llir.ArrayAccess(_var("wksp.insert_targets"), _var("wksp.insert_target_index")),
        llir.Add(
            _var("wksp.insert_value"),
            llir.ArrayAccess(
                _var("wksp.insert_values"),
                _var("wksp.insert_value_index"),
                tensor_access=metadata,
            ),
        ),
    )
    initialization = llir.VarInit(
        _var("initialized"), _var("wksp.insert_initial_value")
    )
    raw = llir.RawStmt("wksp.insert(raw_value)", add_semicolon=False)
    calls = [
        llir.FunctionCallStmt("wksp.insert", [_var("wksp.insert_argument")]),
        llir.FunctionCallStmt("prefix_wksp.insert", [_var("prefix_argument")]),
        llir.FunctionCallStmt("wksp.insert_suffix", [_var("suffix_argument")]),
        llir.FunctionCallStmt("wksp.insert.more", [_var("member_argument")]),
        llir.FunctionCallStmt("wksp.insert_unchecked", [_var("unchecked_argument")]),
    ]
    source: List[llir.Stmt] = [
        _compatible_loop([_workspace_init(), assignment, initialization, raw, *calls])
    ]
    snapshot = _structural_snapshot(source)

    result = transform_compressed_where_for_openmp(source, _context())
    repeated = transform_compressed_where_for_openmp(source, _context())

    count_loop, fill_loop = _phase_loops(result)
    rewritten_assignment = cast(
        llir.Assign,
        next(
            statement for statement in count_loop.body if type(statement) is llir.Assign
        ),
    )
    target = cast(llir.ArrayAccess, rewritten_assignment.var)
    value = cast(llir.Add, rewritten_assignment.value)
    value_access = cast(llir.ArrayAccess, value.right)
    rewritten_initialization = cast(
        llir.VarInit,
        next(
            statement
            for statement in count_loop.body
            if type(statement) is llir.VarInit and statement.var.name == "initialized"
        ),
    )

    assert cast(llir.Var, target.array).name == "wksp.insert_targets"
    assert cast(llir.Var, target.index).name == "wksp.insert_target_index"
    assert cast(llir.Var, value.left).name == "wksp.insert_value"
    assert cast(llir.Var, value_access.array).name == "wksp.insert_values"
    assert cast(llir.Var, value_access.index).name == "wksp.insert_value_index"
    assert value_access.tensor_access is metadata
    fill_assignment = cast(
        llir.Assign,
        next(
            statement for statement in fill_loop.body if type(statement) is llir.Assign
        ),
    )
    fill_value = cast(llir.Add, fill_assignment.value)
    assert type(value) is llir.Add
    assert type(fill_value) is llir.Add
    assert value == fill_value
    assert value is not assignment.value
    assert fill_value is not assignment.value
    assert fill_value is not value
    assert value.left is not cast(llir.Add, assignment.value).left
    assert value.right is not cast(llir.Add, assignment.value).right
    assert fill_value.left is not value.left
    assert fill_value.right is not value.right
    assert cast(llir.ArrayAccess, fill_value.right).tensor_access is metadata
    assert cast(llir.Var, rewritten_initialization.value).name == (
        "wksp.insert_initial_value"
    )

    expected_names = {
        "wksp.insert_argument": "wksp.insert_unchecked",
        "prefix_argument": "prefix_wksp.insert",
        "suffix_argument": "wksp.insert_suffix",
        "member_argument": "wksp.insert.more",
        "unchecked_argument": "wksp.insert_unchecked",
    }
    for phase_loop in (count_loop, fill_loop):
        phase_calls = [
            cast(llir.FunctionCallStmt, statement)
            for statement in phase_loop.body
            if type(statement) is llir.FunctionCallStmt
        ]
        actual_names = {
            cast(llir.Var, call.args[0]).name: call.name for call in phase_calls
        }
        assert actual_names == expected_names
        assert all(type(call.args[0]) is llir.Var for call in phase_calls)
        rewritten_raw = next(
            cast(llir.RawStmt, statement)
            for statement in phase_loop.body
            if type(statement) is llir.RawStmt
            and statement.code == "wksp.insert(raw_value)"
        )
        assert rewritten_raw.add_semicolon is False

    assert _structural_snapshot(source) == snapshot
    assert _structural_snapshot(result.statements) == _structural_snapshot(
        repeated.statements
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(result.statements))
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(repeated.statements))
    assert _mutable_ir_ids(result.statements).isdisjoint(
        _mutable_ir_ids(repeated.statements)
    )


@pytest.mark.parametrize(
    "name",
    [None, "", "   ", type("WorkspaceCallName", (str,), {})("wksp.insert")],
    ids=["none", "empty", "whitespace", "str-subclass"],
)
def test_workspace_insert_rewrite_rejects_malformed_call_names(
    name: object,
) -> None:
    malformed = llir.FunctionCallStmt("wksp.insert", [_var("value")])
    object.__setattr__(malformed, "name", name)
    source = [_compatible_loop([_workspace_init(), malformed])]
    snapshot = _structural_snapshot(source)

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, _context())

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_function_call_stmt_name"
    assert diagnostic.stage == _context().traversal.stage
    assert diagnostic.pass_name == _context().traversal.pass_name
    assert diagnostic.path == ("root", "[0]", "body", "[1]", "name")
    assert _structural_snapshot(source) == snapshot


@pytest.mark.parametrize(
    ("args", "expected_code", "expected_path"),
    [
        (
            object(),
            "invalid_function_call_stmt_args",
            ("root", "[0]", "body", "[1]", "args"),
        ),
        (
            ("not_an_expression",),
            "invalid_expression_sequence_member",
            ("root", "[0]", "body", "[1]", "args", "[0]"),
        ),
    ],
    ids=["container", "member"],
)
def test_workspace_insert_rewrite_rejects_malformed_call_args(
    args: object,
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    malformed = llir.FunctionCallStmt("wksp.insert", [])
    object.__setattr__(malformed, "args", args)
    source = [_compatible_loop([_workspace_init(), malformed])]
    snapshot = _structural_snapshot(source)

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, _context())

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == expected_code
    assert diagnostic.stage == _context().traversal.stage
    assert diagnostic.pass_name == _context().traversal.pass_name
    assert diagnostic.path == expected_path
    assert _structural_snapshot(source) == snapshot


def test_retained_metadata_is_preserved_but_selected_loop_policy_is_reset() -> None:
    inner = _compatible_loop([llir.RawStmt("inner")], bound="inner_limit")
    inner.omp_parallel_for = True
    inner.omp_schedule = "static, 2"
    inner.unroll = True
    inner.simd = True
    inner.before_parallel_body = [llir.RawStmt("inner_before")]
    inner.pre_parallel_body = [llir.RawStmt("inner_pre")]
    inner.post_parallel_body = [llir.RawStmt("inner_post")]
    inner.omp_num_threads = "inner_threads"
    inner.omp_chunk_expr = "inner_chunk"
    inner.scorch_index_var = "inner_tag"
    setattr(inner, "_use_atomic_scheduling", True)
    setattr(inner, "_hoisted_ptr_decls", [llir.RawStmt("inner_hoisted")])

    selected = _compatible_loop([inner])
    selected.omp_parallel_for = True
    selected.omp_schedule = "static"
    selected.unroll = True
    selected.simd = True
    selected.before_parallel_body = [llir.RawStmt("selected_before")]
    selected.pre_parallel_body = [llir.RawStmt("selected_pre")]
    selected.post_parallel_body = [llir.RawStmt("selected_post")]
    selected.scorch_index_var = "selected_tag"
    setattr(selected, "_hoisted_ptr_decls", [llir.RawStmt("selected_hoisted")])

    result = transform_compressed_where_for_openmp([selected], _context())

    count_loop, fill_loop = _phase_loops(result)
    for phase in (count_loop, fill_loop):
        rewritten_inner = cast(
            llir.ForLoop,
            next(
                statement for statement in phase.body if type(statement) is llir.ForLoop
            ),
        )
        assert rewritten_inner.omp_parallel_for is True
        assert rewritten_inner.omp_schedule == "static, 2"
        assert rewritten_inner.unroll is True
        assert rewritten_inner.simd is True
        assert rewritten_inner.omp_num_threads == "inner_threads"
        assert rewritten_inner.omp_chunk_expr == "inner_chunk"
        assert rewritten_inner.scorch_index_var == "inner_tag"
        assert _raw_codes(rewritten_inner.before_parallel_body) == ["inner_before"]
        assert _raw_codes(rewritten_inner.pre_parallel_body) == ["inner_pre"]
        assert _raw_codes(rewritten_inner.post_parallel_body) == ["inner_post"]
        assert getattr(rewritten_inner, "_use_atomic_scheduling") is True
        assert _raw_codes(getattr(rewritten_inner, "_hoisted_ptr_decls")) == [
            "inner_hoisted"
        ]

        # The selected serial loop is replaced, not rebuilt in place. Its old
        # region metadata is intentionally not copied to either generated phase.
        assert phase.omp_parallel_for is True
        assert phase.omp_schedule == "dynamic, 64"
        assert phase.unroll is False
        assert phase.simd is False
        assert phase.before_parallel_body is None
        assert phase.pre_parallel_body is None
        assert phase.post_parallel_body is None
        assert phase.scorch_index_var is None
        assert not hasattr(phase, "_hoisted_ptr_decls")


@pytest.mark.parametrize(
    ("body", "update", "expected_work", "expected_rows", "expected_grain"),
    [
        (
            _ds_work_body(workspace=False, both_operands=False),
            None,
            "A1_pos[A0_size]",
            "A0_size",
            False,
        ),
        (
            [llir.RawStmt("work")],
            llir.Assign(
                _var("row", llir.DataType.INT),
                llir.Literal(4),
                op=llir.AssignOp.ADD_ASSIGN,
            ),
            "-1",
            "((A0_size + 4 - 1) / 4)",
            False,
        ),
    ],
)
def test_parallel_policy_fallbacks_preserve_work_and_trip_count_decisions(
    body: List[llir.Stmt],
    update: llir.Assign | None,
    expected_work: str,
    expected_rows: str,
    expected_grain: bool,
) -> None:
    loop = _compatible_loop(body, update=update)

    result = transform_compressed_where_for_openmp([loop], _context())

    for phase in _phase_loops(result):
        assert phase.omp_num_threads == (
            f"scorch_nthreads({expected_work}, {expected_rows})"
        )
        assert phase.omp_chunk_expr == (
            f"scorch_chunk({expected_rows}, {expected_work})"
        )
        assert ("SCORCH_GRAIN" in phase.omp_num_threads) is expected_grain


def test_custom_parallel_policy_is_an_explicit_context_input() -> None:
    policy = CompressedWhereOpenMPPolicy(
        omp_schedule="guided, 7",
        flop_grain="CUSTOM_FLOP_GRAIN",
    )

    result = transform_compressed_where_for_openmp(
        [_compatible_loop(_ds_work_body())],
        _context(policy=policy),
    )

    for phase in _phase_loops(result):
        assert phase.omp_schedule == "guided, 7"
        assert "CUSTOM_FLOP_GRAIN" in cast(str, phase.omp_num_threads)
        assert "CUSTOM_FLOP_GRAIN" in cast(str, phase.omp_chunk_expr)


@pytest.mark.parametrize(
    (
        "workspace_ctype",
        "result_dtype",
        "scalar_type",
        "pointer_type",
        "torch_dtype_name",
    ),
    [
        (
            "float",
            torch.float32,
            llir.DataType.FLOAT32,
            llir.DataType.PTR_FLOAT32,
            "kFloat32",
        ),
        (
            "double",
            torch.float64,
            llir.DataType.FLOAT64,
            llir.DataType.PTR_FLOAT64,
            "kFloat64",
        ),
        (
            "int32_t",
            torch.int32,
            llir.DataType.INT32,
            llir.DataType.PTR_INT_32,
            "kInt32",
        ),
        (
            "int64_t",
            torch.int64,
            llir.DataType.INT64,
            llir.DataType.PTR_INT_64,
            "kInt64",
        ),
        (
            "int8_t",
            torch.int8,
            llir.DataType.INT8,
            llir.DataType.PTR_INT8,
            "kInt8",
        ),
        (
            "uint8_t",
            torch.uint8,
            llir.DataType.UINT8,
            llir.DataType.PTR_UINT8,
            "kUInt8",
        ),
    ],
)
def test_canonical_workspace_ctype_builds_typed_value_owner_and_borrow(
    workspace_ctype: str,
    result_dtype: torch.dtype,
    scalar_type: llir.DataType,
    pointer_type: llir.DataType,
    torch_dtype_name: str,
) -> None:
    context = CompressedWhereOpenMPContext(
        result_name="Result",
        result_id=SymbolId(1),
        compressed_levels=(1,),
        result_assembler=_result_assembler(dtype=result_dtype),
        workspace_name="wksp",
        workspace_ctype=workspace_ctype,
    )

    result = transform_compressed_where_for_openmp(
        [_compatible_loop(_ds_work_body())], context
    )

    owner, pointer = _value_initializations(result.statements)
    assert owner.var.type is llir.DataType.TORCH_TENSOR
    assert type(owner.value) is llir.FunctionCall
    empty = cast(llir.FunctionCall, owner.value)
    assert empty.name == "torch::empty"
    assert type(empty.args[0]) is llir.Array
    extent = cast(llir.Array, empty.args[0])
    assert extent.data_type is llir.DataType.INT64
    assert type(extent.values) is tuple
    assert len(extent.values) == 1
    assert type(extent.values[0]) is llir.Var
    total = cast(llir.Var, extent.values[0])
    assert (total.name, total.type) == ("_total1", llir.DataType.INT64)
    assert type(empty.args[1]) is llir.QualifiedName
    dtype = cast(llir.QualifiedName, empty.args[1])
    assert (dtype.namespace, dtype.name, dtype.data_type) == (
        "torch",
        torch_dtype_name,
        llir.DataType.TORCH_SCALAR_TYPE,
    )
    assert pointer.var.type is pointer_type
    assert type(pointer.value) is llir.MemberCall
    data_ptr = cast(llir.MemberCall, pointer.value)
    assert data_ptr.template_args == (scalar_type,)
    assert data_ptr.args == ()
    assert type(data_ptr.base) is llir.Var
    assert (data_ptr.base.name, data_ptr.base.type) == (
        "Result_values_torch",
        llir.DataType.TORCH_TENSOR,
    )
    assert [LLIRLowerer().lower_llir(value) for value in (owner, pointer)] == [
        "torch::Tensor Result_values_torch = "
        f"torch::empty({{_total1}}, torch::{torch_dtype_name});",
        f"{workspace_ctype}* Result_values_data = "
        f"Result_values_torch.data_ptr<{workspace_ctype}>();",
    ]
    assert not any(
        "Result_values_torch" in code or "Result_values_data" in code
        for code in _raw_codes(result.statements)
    )
    value_targets = [
        cast(llir.ArrayAccess, assignment.var)
        for assignment in _assignments(result.statements)
        if type(assignment.var) is llir.ArrayAccess
        and type(cast(llir.ArrayAccess, assignment.var).array) is llir.Var
        and cast(llir.Var, cast(llir.ArrayAccess, assignment.var).array).name
        == "Result_values_data"
    ]
    assert value_targets
    assert all(
        cast(llir.Var, target.array).type is pointer_type for target in value_targets
    )


@pytest.mark.parametrize(
    ("workspace_ctype", "result_dtype", "torch_dtype", "pointer_type"),
    [
        ("int", torch.int32, "torch::kInt32", llir.DataType.PTR_INT),
        ("long long", torch.int64, "torch::kInt64", llir.DataType.NO_TYPE),
        (
            "custom_scalar",
            torch.float32,
            "torch::kFloat32",
            llir.DataType.NO_TYPE,
        ),
    ],
)
def test_noncanonical_workspace_ctype_preserves_exact_raw_value_allocation(
    workspace_ctype: str,
    result_dtype: torch.dtype,
    torch_dtype: str,
    pointer_type: llir.DataType,
) -> None:
    context = CompressedWhereOpenMPContext(
        result_name="Result",
        result_id=SymbolId(1),
        compressed_levels=(1,),
        result_assembler=_result_assembler(dtype=result_dtype),
        workspace_name="wksp",
        workspace_ctype=workspace_ctype,
    )

    result = transform_compressed_where_for_openmp(
        [_compatible_loop(_ds_work_body())], context
    )

    assert _value_initializations(result.statements) == []
    allocations = [
        code
        for code in _raw_codes(result.statements)
        if "torch::Tensor Result_values_torch" in code
    ]
    assert allocations == [
        "torch::Tensor Result_values_torch = "
        f"torch::empty({{(long long)_total1}}, {torch_dtype});\n"
        f"  {workspace_ctype}* Result_values_data = "
        f"Result_values_torch.data_ptr<{workspace_ctype}>();"
    ]
    value_targets = [
        cast(llir.ArrayAccess, assignment.var)
        for assignment in _assignments(result.statements)
        if type(assignment.var) is llir.ArrayAccess
        and type(cast(llir.ArrayAccess, assignment.var).array) is llir.Var
        and cast(llir.Var, cast(llir.ArrayAccess, assignment.var).array).name
        == "Result_values_data"
    ]
    assert value_targets
    assert all(
        cast(llir.Var, target.array).type is pointer_type for target in value_targets
    )


@pytest.mark.parametrize(
    ("workspace_ctype", "result_dtype"),
    [
        ("float", torch.float64),
        ("double", torch.float32),
        ("int", torch.int64),
        ("int32_t", torch.int64),
        ("long long", torch.int32),
        ("int64_t", torch.int32),
        ("int8_t", torch.uint8),
        ("uint8_t", torch.int8),
    ],
)
def test_recognized_workspace_ctype_dtype_mismatches_fail_before_transform(
    workspace_ctype: str,
    result_dtype: torch.dtype,
) -> None:
    source = [_compatible_loop(_ds_work_body())]
    snapshot = _structural_snapshot(source)
    context = CompressedWhereOpenMPContext(
        result_name="Result",
        result_id=SymbolId(1),
        compressed_levels=(1,),
        result_assembler=_result_assembler(dtype=result_dtype),
        workspace_name="wksp",
        workspace_ctype=workspace_ctype,
    )

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "mismatched_compressed_where_result_assembler"
    assert diagnostic.path == ("context", "result_assembler", "dtype")
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert _structural_snapshot(source) == snapshot


def test_transform_does_not_mutate_or_alias_caller_owned_llir() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_ds_work_body())]
    snapshot = _structural_snapshot(source)

    result = transform_compressed_where_for_openmp(source, _context())

    assert _structural_snapshot(source) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(result.statements))
    original_loop = cast(llir.ForLoop, source[0])
    original_insert = cast(llir.FunctionCallStmt, original_loop.body[3])
    assert original_insert.name == "wksp.insert"

    result.statements.append(llir.Break())
    assert len(source) == 1


def test_unknown_node_fails_with_pass_specific_structured_diagnostic() -> None:
    class UnknownStmt(llir.Stmt):
        pass

    source = [
        _compatible_loop([llir.RawStmt("work")]),
        UnknownStmt(),
    ]
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.node_type == "UnknownStmt"
    assert diagnostic.path == ("root", "[1]")


def test_malformed_typed_child_fails_with_pass_specific_diagnostic() -> None:
    malformed = _compatible_loop([])
    malformed.body = cast(List[llir.Stmt], [_var("not_a_statement")])
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp([malformed], context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_statement_sequence_member"
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path == ("root", "[0]", "body", "[0]")


@pytest.mark.parametrize(
    ("malformation", "expected_code", "expected_suffix"),
    (
        ("unknown_array", "unknown_llir_node", ("value", "array")),
        ("unknown_index", "unknown_llir_node", ("value", "index")),
        ("access_subclass", "unknown_llir_node", ("value",)),
        (
            "forged_metadata",
            "invalid_tensor_access_metadata",
            ("value", "tensor_access"),
        ),
    ),
)
def test_generated_fill_base_loads_fail_closed_at_compressed_owner(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    expected_code: str,
    expected_suffix: Tuple[str, ...],
) -> None:
    class UnknownExpr(llir.Expr):
        pass

    class UnknownArrayAccess(llir.ArrayAccess):
        pass

    injected: List[str] = []

    class InjectingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            loads = _base_offset_loads(value)
            if loads and not injected:
                initializer = loads[0]
                access = cast(llir.ArrayAccess, initializer.value)
                if malformation == "unknown_array":
                    initializer.value = llir.ArrayAccess(
                        UnknownExpr(),
                        access.index,
                    )
                elif malformation == "unknown_index":
                    initializer.value = llir.ArrayAccess(
                        access.array,
                        UnknownExpr(),
                    )
                elif malformation == "access_subclass":
                    unknown = object.__new__(UnknownArrayAccess)
                    object.__setattr__(unknown, "array", access.array)
                    object.__setattr__(unknown, "index", access.index)
                    object.__setattr__(unknown, "tensor_access", None)
                    initializer.value = unknown
                else:
                    forged = object.__new__(llir.ArrayAccess)
                    object.__setattr__(forged, "array", access.array)
                    object.__setattr__(forged, "index", access.index)
                    object.__setattr__(forged, "tensor_access", object())
                    initializer.value = forged
                injected.append(initializer.var.name)
            super().walk(value)

    monkeypatch.setattr(compressed_where_module, "LLIRWalker", InjectingWalker)
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(
            [_compatible_loop(_ds_work_body())],
            context,
        )

    diagnostic = raised.value.diagnostic
    assert injected == ["_base1"]
    assert diagnostic.code == expected_code
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path[-len(expected_suffix) :] == expected_suffix


@pytest.mark.parametrize(
    ("producer", "expected_code", "expected_suffix"),
    (
        ("W6", "invalid_direct_init_args", ("args",)),
        ("W7", "invalid_add_operator", ("args", "[0]", "op")),
        ("W8", "invalid_add_operator", ("body", "[0]", "value", "op")),
        (
            "W9",
            "invalid_tensor_access_metadata",
            ("value", "tensor_access"),
        ),
    ),
)
def test_generated_offset_family_fails_closed_at_compressed_owner(
    monkeypatch: pytest.MonkeyPatch,
    producer: str,
    expected_code: str,
    expected_suffix: Tuple[str, ...],
) -> None:
    source = [_compatible_loop(_ds_work_body())]
    snapshot = _structural_snapshot(source)
    injected: List[str] = []

    class InjectingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            owners = _offset_family_direct_initializations(value)
            prefix_loops = _prefix_sum_loops(value)
            totals = _total_offset_loads(value)
            if owners and prefix_loops and totals and not injected:
                if producer == "W6":
                    count = next(
                        owner for owner in owners if owner.var.name == "_count1"
                    )
                    object.__setattr__(count, "args", list(count.args))
                elif producer == "W7":
                    offset = next(
                        owner for owner in owners if owner.var.name == "_offset1"
                    )
                    object.__setattr__(cast(llir.Add, offset.args[0]), "op", "-")
                elif producer == "W8":
                    prefix = prefix_loops[0]
                    assignment = cast(llir.Assign, prefix.body[0])
                    object.__setattr__(cast(llir.Add, assignment.value), "op", "-")
                else:
                    access = cast(llir.ArrayAccess, totals[0].value)
                    object.__setattr__(access, "tensor_access", object())
                injected.append(producer)
            super().walk(value)

    monkeypatch.setattr(compressed_where_module, "LLIRWalker", InjectingWalker)
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert injected == [producer]
    assert diagnostic.code == expected_code
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path[-len(expected_suffix) :] == expected_suffix
    assert _structural_snapshot(source) == snapshot


@pytest.mark.parametrize(
    ("malformation", "expected_code", "expected_suffix"),
    (
        ("owner_args", "invalid_function_call_args", ("value", "args")),
        (
            "pointer_template_args",
            "invalid_member_call_template_args",
            ("value", "template_args"),
        ),
        ("unknown_owner_call", "unknown_llir_node", ("value",)),
        (
            "copy_target_metadata",
            "invalid_tensor_access_metadata",
            ("body", "[0]", "var", "tensor_access"),
        ),
        (
            "copy_cast_type",
            "invalid_cast_data_type",
            ("body", "[0]", "value", "data_type"),
        ),
    ),
)
def test_generated_first_position_allocation_fails_closed_at_owner(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    expected_code: str,
    expected_suffix: Tuple[str, ...],
) -> None:
    class UnknownFunctionCall(llir.FunctionCall):
        pass

    source = [_compatible_loop(_ds_work_body())]
    snapshot = _structural_snapshot(source)
    injected: List[str] = []

    class InjectingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            initializations = _first_position_initializations(value)
            copy_loops = _first_position_copy_loops(value)
            if len(initializations) == 2 and len(copy_loops) == 1 and not injected:
                owner, pointer = initializations
                copy_loop = copy_loops[0]
                if malformation == "owner_args":
                    empty = cast(llir.FunctionCall, owner.value)
                    object.__setattr__(empty, "args", list(empty.args))
                elif malformation == "pointer_template_args":
                    data_ptr = cast(llir.MemberCall, pointer.value)
                    object.__setattr__(
                        data_ptr,
                        "template_args",
                        list(data_ptr.template_args),
                    )
                elif malformation == "unknown_owner_call":
                    empty = cast(llir.FunctionCall, owner.value)
                    unknown = object.__new__(UnknownFunctionCall)
                    object.__setattr__(unknown, "name", empty.name)
                    object.__setattr__(unknown, "args", empty.args)
                    owner.value = unknown
                elif malformation == "copy_target_metadata":
                    assignment = cast(llir.Assign, copy_loop.body[0])
                    target = cast(llir.ArrayAccess, assignment.var)
                    object.__setattr__(target, "tensor_access", object())
                else:
                    assignment = cast(llir.Assign, copy_loop.body[0])
                    value_cast = cast(llir.Cast, assignment.value)
                    object.__setattr__(value_cast, "data_type", object())
                injected.append(malformation)
            super().walk(value)

    monkeypatch.setattr(compressed_where_module, "LLIRWalker", InjectingWalker)
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert injected == [malformation]
    assert diagnostic.code == expected_code
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path[-len(expected_suffix) :] == expected_suffix
    assert _structural_snapshot(source) == snapshot


@pytest.mark.parametrize(
    ("malformation", "expected_code", "expected_suffix"),
    (
        ("owner_args", "invalid_function_call_args", ("value", "args")),
        (
            "pointer_template_args",
            "invalid_member_call_template_args",
            ("value", "template_args"),
        ),
        ("unknown_owner_call", "unknown_llir_node", ("value",)),
        (
            "extent_metadata",
            "invalid_tensor_access_metadata",
            ("value", "args", "[0]", "values", "[0]", "tensor_access"),
        ),
    ),
)
def test_generated_compressed_coordinate_allocations_fail_closed_at_owner(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    expected_code: str,
    expected_suffix: Tuple[str, ...],
) -> None:
    class UnknownFunctionCall(llir.FunctionCall):
        pass

    source = [_compatible_loop(_ds_work_body())]
    snapshot = _structural_snapshot(source)
    injected: List[str] = []

    class InjectingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            initializations = _compressed_coordinate_initializations(value)
            if initializations and not injected:
                owner, pointer = initializations
                if malformation == "owner_args":
                    empty = cast(llir.FunctionCall, owner.value)
                    object.__setattr__(empty, "args", list(empty.args))
                elif malformation == "pointer_template_args":
                    data_ptr = cast(llir.MemberCall, pointer.value)
                    object.__setattr__(
                        data_ptr,
                        "template_args",
                        list(data_ptr.template_args),
                    )
                elif malformation == "unknown_owner_call":
                    empty = cast(llir.FunctionCall, owner.value)
                    unknown = object.__new__(UnknownFunctionCall)
                    object.__setattr__(unknown, "name", empty.name)
                    object.__setattr__(unknown, "args", empty.args)
                    owner.value = unknown
                else:
                    empty = cast(llir.FunctionCall, owner.value)
                    extent = cast(llir.Array, empty.args[0])
                    total = cast(llir.Var, extent.values[0])
                    total.tensor_access = object()  # type: ignore[assignment]
                injected.append(malformation)
            super().walk(value)

    monkeypatch.setattr(compressed_where_module, "LLIRWalker", InjectingWalker)
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert injected == [malformation]
    assert diagnostic.code == expected_code
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path[-len(expected_suffix) :] == expected_suffix
    assert _structural_snapshot(source) == snapshot


@pytest.mark.parametrize(
    ("malformation", "expected_code", "expected_suffix"),
    (
        ("owner_args", "invalid_function_call_args", ("value", "args")),
        (
            "pointer_template_args",
            "invalid_member_call_template_args",
            ("value", "template_args"),
        ),
        ("unknown_owner_call", "unknown_llir_node", ("value",)),
        (
            "extent_add_operator",
            "invalid_add_operator",
            ("value", "args", "[0]", "values", "[0]", "op"),
        ),
        (
            "sentinel_metadata",
            "invalid_tensor_access_metadata",
            ("var", "tensor_access"),
        ),
    ),
)
def test_generated_deeper_position_allocations_fail_closed_at_owner(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    expected_code: str,
    expected_suffix: Tuple[str, ...],
) -> None:
    class UnknownFunctionCall(llir.FunctionCall):
        pass

    source = [_compatible_loop(_ds_work_body())]
    snapshot = _structural_snapshot(source)
    injected: List[str] = []

    class InjectingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            owners = _deeper_position_owner_initializations(value)
            pointers = _deeper_position_pointer_initializations(value)
            sentinels = _deeper_position_zero_sentinels(value)
            if owners and pointers and sentinels and not injected:
                owner = owners[0]
                pointer = pointers[0]
                sentinel = sentinels[0]
                if malformation == "owner_args":
                    empty = cast(llir.FunctionCall, owner.value)
                    object.__setattr__(empty, "args", list(empty.args))
                elif malformation == "pointer_template_args":
                    data_ptr = cast(llir.MemberCall, pointer.value)
                    object.__setattr__(
                        data_ptr,
                        "template_args",
                        list(data_ptr.template_args),
                    )
                elif malformation == "unknown_owner_call":
                    empty = cast(llir.FunctionCall, owner.value)
                    unknown = object.__new__(UnknownFunctionCall)
                    object.__setattr__(unknown, "name", empty.name)
                    object.__setattr__(unknown, "args", empty.args)
                    owner.value = unknown
                elif malformation == "extent_add_operator":
                    empty = cast(llir.FunctionCall, owner.value)
                    extent = cast(llir.Array, empty.args[0])
                    cardinality = cast(llir.Add, extent.values[0])
                    object.__setattr__(cardinality, "op", "-")
                else:
                    target = cast(llir.ArrayAccess, sentinel.var)
                    object.__setattr__(target, "tensor_access", object())
                injected.append(malformation)
            super().walk(value)

    monkeypatch.setattr(compressed_where_module, "LLIRWalker", InjectingWalker)
    context = _context((1, 2))

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert injected == [malformation]
    assert diagnostic.code == expected_code
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path[-len(expected_suffix) :] == expected_suffix
    assert _structural_snapshot(source) == snapshot


@pytest.mark.parametrize(
    ("malformation", "expected_code", "expected_suffix"),
    (
        ("owner_args", "invalid_function_call_args", ("value", "args")),
        (
            "pointer_template_args",
            "invalid_member_call_template_args",
            ("value", "template_args"),
        ),
        ("unknown_owner_call", "unknown_llir_node", ("value",)),
        (
            "extent_metadata",
            "invalid_tensor_access_metadata",
            ("value", "args", "[0]", "values", "[0]", "tensor_access"),
        ),
        (
            "dtype_data_type",
            "invalid_qualified_name_data_type",
            ("value", "args", "[1]", "data_type"),
        ),
    ),
)
def test_generated_value_allocation_fails_closed_at_compressed_owner(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    expected_code: str,
    expected_suffix: Tuple[str, ...],
) -> None:
    class UnknownFunctionCall(llir.FunctionCall):
        pass

    source = [_compatible_loop(_ds_work_body())]
    snapshot = _structural_snapshot(source)
    injected: List[str] = []

    class InjectingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            initializations = _value_initializations(value)
            if initializations and not injected:
                owner, pointer = initializations
                empty = cast(llir.FunctionCall, owner.value)
                if malformation == "owner_args":
                    object.__setattr__(empty, "args", list(empty.args))
                elif malformation == "pointer_template_args":
                    data_ptr = cast(llir.MemberCall, pointer.value)
                    object.__setattr__(
                        data_ptr,
                        "template_args",
                        list(data_ptr.template_args),
                    )
                elif malformation == "unknown_owner_call":
                    unknown = object.__new__(UnknownFunctionCall)
                    object.__setattr__(unknown, "name", empty.name)
                    object.__setattr__(unknown, "args", empty.args)
                    owner.value = unknown
                elif malformation == "extent_metadata":
                    extent = cast(llir.Array, empty.args[0])
                    total = cast(llir.Var, extent.values[0])
                    total.tensor_access = object()  # type: ignore[assignment]
                else:
                    dtype = cast(llir.QualifiedName, empty.args[1])
                    object.__setattr__(dtype, "data_type", object())
                injected.append(malformation)
            super().walk(value)

    monkeypatch.setattr(compressed_where_module, "LLIRWalker", InjectingWalker)
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert injected == [malformation]
    assert diagnostic.code == expected_code
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path[-len(expected_suffix) :] == expected_suffix
    assert _structural_snapshot(source) == snapshot


@pytest.mark.parametrize(
    ("context", "expected_code", "expected_path"),
    [
        (
            cast(CompressedWhereOpenMPContext, object()),
            "invalid_compressed_where_context",
            ("context",),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                result_assembler=_result_assembler(),
                workspace_name="wksp",
                workspace_ctype="float",
            ),
            "invalid_compressed_where_result_name",
            ("context", "result_name"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=cast(SymbolId, object()),
                compressed_levels=(1,),
                result_assembler=_result_assembler(),
                workspace_name="wksp",
                workspace_ctype="float",
            ),
            "invalid_compressed_where_result_id",
            ("context", "result_id"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(),
                result_assembler=_result_assembler(),
                workspace_name="wksp",
                workspace_ctype="float",
            ),
            "invalid_compressed_where_levels",
            ("context", "compressed_levels"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(2,),
                result_assembler=_result_assembler(),
                workspace_name="wksp",
                workspace_ctype="float",
            ),
            "unsupported_compressed_where_layout",
            ("context", "compressed_levels"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                result_assembler=_result_assembler(),
                workspace_name="",
                workspace_ctype="float",
            ),
            "invalid_compressed_where_workspace_name",
            ("context", "workspace_name"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                result_assembler=_result_assembler(),
                workspace_name="wksp",
                workspace_ctype="",
            ),
            "invalid_compressed_where_workspace_ctype",
            ("context", "workspace_ctype"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                result_assembler=_result_assembler(),
                workspace_name="wksp",
                workspace_ctype="float",
                policy=CompressedWhereOpenMPPolicy(omp_schedule="", flop_grain="grain"),
            ),
            "invalid_compressed_where_schedule",
            ("context", "policy.omp_schedule"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                result_assembler=_result_assembler(),
                workspace_name="wksp",
                workspace_ctype="float",
                traversal=LLIRTraversalContext(stage="", pass_name="pass"),
            ),
            "invalid_compressed_where_traversal_context",
            ("context", "traversal"),
        ),
    ],
)
def test_invalid_contexts_fail_structurally(
    context: CompressedWhereOpenMPContext,
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp([llir.Break()], context)

    assert raised.value.diagnostic.code == expected_code
    assert raised.value.diagnostic.path == expected_path


def test_result_assembler_context_contract_is_exact_and_fail_closed() -> None:
    class ResultAssemblerSubclass(ResultTensorAssembler):
        pass

    mismatched_name = _result_assembler(name="Other")
    mismatched_levels = ResultTensorAssembler(
        name="Result",
        level_types=(LevelType.DENSE, LevelType.COORDINATE),
        dtype=torch.float32,
    )
    mismatched_dtype = _result_assembler(dtype=torch.float64)
    forged = _result_assembler()
    object.__setattr__(forged, "dtype", "float32")
    missing_field = _result_assembler()
    object.__delattr__(missing_field, "level_types")
    cases = (
        (
            cast(ResultTensorAssembler, object()),
            "invalid_compressed_where_result_assembler",
            ("context", "result_assembler"),
        ),
        (
            mismatched_name,
            "mismatched_compressed_where_result_assembler",
            ("context", "result_assembler", "name"),
        ),
        (
            mismatched_levels,
            "mismatched_compressed_where_result_assembler",
            ("context", "result_assembler", "level_types"),
        ),
        (
            mismatched_dtype,
            "mismatched_compressed_where_result_assembler",
            ("context", "result_assembler", "dtype"),
        ),
        (
            forged,
            "invalid_compressed_where_result_assembler",
            ("context", "result_assembler"),
        ),
        (
            missing_field,
            "invalid_compressed_where_result_assembler",
            ("context", "result_assembler"),
        ),
        (
            ResultAssemblerSubclass(
                name="Result",
                level_types=(LevelType.DENSE, LevelType.COMPRESSED),
                dtype=torch.float32,
            ),
            "invalid_compressed_where_result_assembler",
            ("context", "result_assembler"),
        ),
    )

    for result_assembler, expected_code, expected_path in cases:
        context = replace(_context(), result_assembler=result_assembler)
        with pytest.raises(LLIRTraversalError) as raised:
            transform_compressed_where_for_openmp([llir.Break()], context)

        diagnostic = raised.value.diagnostic
        assert diagnostic.code == expected_code
        assert diagnostic.path == expected_path
        assert diagnostic.stage == context.traversal.stage
        assert diagnostic.pass_name == context.traversal.pass_name

    missing = _context()
    object.__delattr__(missing, "result_assembler")
    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp([llir.Break()], missing)
    assert raised.value.diagnostic.code == ("invalid_compressed_where_result_assembler")
    assert raised.value.diagnostic.path == ("context", "result_assembler")


def test_unknown_shared_result_epilogue_node_fails_at_compressed_pass_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnknownResultEpilogue(llir.Stmt):
        pass

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_storage_epilogue",
        lambda self: [UnknownResultEpilogue()],
    )
    context = _context()
    source = [_compatible_loop(_ds_work_body())]
    source_snapshot = _structural_snapshot(source)

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.node_type == "UnknownResultEpilogue"
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path[:1] == ("root",)
    assert _structural_snapshot(source) == source_snapshot


@pytest.mark.parametrize(
    ("root", "expected_code", "expected_path"),
    [
        (
            cast(List[llir.Stmt], (llir.Break(),)),
            "unsupported_compressed_where_root",
            ("root",),
        ),
        (
            cast(List[llir.Stmt], [llir.Break(), _var("not_a_statement")]),
            "invalid_compressed_where_root_member",
            ("root", "[1]"),
        ),
    ],
)
def test_invalid_roots_fail_structurally(
    root: List[llir.Stmt],
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(root, _context())

    assert raised.value.diagnostic.code == expected_code
    assert raised.value.diagnostic.path == expected_path


def test_successful_transform_is_single_use_and_not_idempotent() -> None:
    context = _context()
    first = transform_compressed_where_for_openmp(
        [_compatible_loop(_ds_work_body())], context
    )

    second = transform_compressed_where_for_openmp(first.statements, context)

    assert first.applied is True
    assert second.applied is True
    assert _structural_snapshot(second.statements) != _structural_snapshot(
        first.statements
    )
    assert _mutable_ir_ids(first.statements).isdisjoint(
        _mutable_ir_ids(second.statements)
    )


def test_production_ds_generated_cpp_locks_typed_offset_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("SparseProduct", fmt="ds")
    left = TensorVar("SparseLeft", fmt="ds")
    right = TensorVar("SparseRight", fmt="ds")
    assignment = TensorAssign(
        result[row, column],
        left[row, reduction] * right[reduction, column],
        op=Operation.ADD,
    )
    cin = cast(
        ForAll,
        Scheduler.auto_schedule(
            ForAll(row, ForAll(reduction, ForAll(column, assignment)))
        ),
    )

    def reject_ordinary_final_assembly(
        assembler: ResultTensorAssembler,
    ) -> List[llir.Stmt]:
        del assembler
        raise AssertionError(
            "compressed-output ABI must not run ordinary final assembly"
        )

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_final_assembly",
        reject_ordinary_final_assembly,
    )
    lowerer = CINLowerer()
    lowered = lowerer.lower_IndexStmt(cin)
    assert type(lowered) is llir.Function
    function = cast(llir.Function, lowered)
    declaration, mode_indices, values, return_statement = _result_assembly_tail(
        function.body
    )
    assert type(declaration) is llir.VarDecl
    assert type(mode_indices) is llir.Assign
    assert type(values) is llir.Assign
    assert type(return_statement) is llir.Return
    assert not any(
        marker in code
        for code in _raw_codes(function.body)
        for marker in (
            "Tensor SparseProduct;",
            "storage.index.mode_indices",
            "storage.value",
            "return SparseProduct",
        )
    )
    assert [cast(llir.Var, argument).name for argument in function.args] == [
        "result_shape",
        "SparseLeft_shape",
        "SparseLeft_mode_indices",
        "SparseLeft_values",
        "SparseRight_shape",
        "SparseRight_mode_indices",
        "SparseRight_values",
    ]
    assert all(
        type(statement) is llir.FunctionCallStmt for statement in function.body[:3]
    )
    validation = [
        LLIRLowerer().lower_llir(statement) for statement in function.body[:3]
    ]
    assert validation[0] == (
        'scorch_native::validate_jit_result_shape(result_shape, {}, 2, "evaluate");'
    )
    assert '"SparseLeft"' in validation[1]
    assert '"SparseRight"' in validation[2]
    assert all("wksp" not in statement for statement in validation)
    production_base_loads = _base_offset_loads(function)
    assert [load.var.name for load in production_base_loads] == ["_base1"]
    production_access = cast(llir.ArrayAccess, production_base_loads[0].value)
    assert cast(llir.Var, production_access.array).type is (
        llir.DataType.STD_VECTOR_INT
    )
    assert cast(llir.Var, production_access.index).name == "r"
    assert cast(llir.Var, production_access.index).type is llir.DataType.INT64
    production_owners = _offset_family_direct_initializations(function)
    production_prefix_loops = _prefix_sum_loops(function)
    production_totals = _total_offset_loads(function)
    production_family: List[llir.Node] = [
        *production_owners,
        *production_prefix_loops,
        *production_totals,
    ]
    assert [owner.var.name for owner in production_owners] == [
        "_count1",
        "_offset1",
    ]
    assert len(production_prefix_loops) == 1
    assert [total.var.name for total in production_totals] == ["_total1"]
    production_values = _value_initializations(function)
    assert [value.var.name for value in production_values] == [
        "SparseProduct_values_torch",
        "SparseProduct_values_data",
    ]
    assert [LLIRLowerer().lower_llir(value) for value in production_values] == [
        "torch::Tensor SparseProduct_values_torch = "
        "torch::empty({_total1}, torch::kFloat32);",
        "float* SparseProduct_values_data = "
        "SparseProduct_values_torch.data_ptr<float>();",
    ]
    assert not any(
        "SparseProduct_values_torch" in code or "SparseProduct_values_data" in code
        for code in _raw_codes(function)
    )
    production_bounds = _named_vars(production_family, "SparseLeft0_size")
    assert len(production_bounds) == 4
    assert all(bound.type is llir.DataType.INT for bound in production_bounds)
    assert len({id(bound) for bound in production_bounds}) == 4
    cpp = LLIRLowerer().lower_llir(function)

    assert len(cpp) == 7117
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "02043a5a8625c596d183385bbe58063ebccf3a2dc75cc2f8bb05e45921ce9f12"
    )
    assert cpp.count("wksp.insert_unchecked(") == 2
    assert "wksp.insert(" not in cpp
    assert cpp.count("int pSparseLeft1_end = SparseLeft1_pos[pSparseLeft0 + 1];") == 2
    assert cpp.count("pSparseLeft1 < pSparseLeft1_end; pSparseLeft1++") == 2
    assert (
        cpp.count("int pSparseRight1_end = SparseRight1_pos[pSparseRight0 + 1];") == 2
    )
    assert cpp.count("pSparseRight1 < pSparseRight1_end; pSparseRight1++") == 2
    assert [record.pass_name for record in lowerer.llir_pass_run_records] == [
        "transform_compressed_where_for_openmp",
        "rewrite_result_writes",
        "rewrite_result_writes",
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.configuration_name for record in lowerer.llir_pass_run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
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
    assert all(
        not record.verified_before and not record.verified_after
        for record in lowerer.llir_pass_run_records
    )
    debug_lowerer = CINLowerer(llir_pass_options=DEBUG_LLIR_PASS_OPTIONS)
    debug_cpp = LLIRLowerer().lower_llir(debug_lowerer.lower_IndexStmt(cin))
    assert debug_cpp == cpp
    assert all(
        record.verified_before and record.verified_after
        for record in debug_lowerer.llir_pass_run_records
    )
    assert not hasattr(lowerer, "_compressed_output_parallel")
    assert [tensor.get_name() for tensor in lowerer.need_compute] == [
        "SparseProduct",
        "wksp",
        "SparseProduct",
        "wksp",
        "SparseProduct",
        "wksp",
        "wksp",
        "wksp",
    ]


def test_production_dss_generated_cpp_locks_typed_offset_family() -> None:
    batch, row, reduction, column = (
        IndexVar("batch"),
        IndexVar("row"),
        IndexVar("reduction"),
        IndexVar("column"),
    )
    result = TensorVar("Result", fmt="dss")
    left = TensorVar("Left", fmt="dss")
    right = TensorVar("Right", fmt="dss")
    result[batch, row, column] = (
        left[batch, row, reduction] * right[batch, reduction, column]
    )
    assignment = result._assignment
    assert assignment is not None
    cin = cast(
        ForAll,
        Scheduler.auto_schedule(
            ForAll(
                batch,
                ForAll(row, ForAll(reduction, ForAll(column, assignment))),
            )
        ),
    )

    lowerer = CINLowerer()
    lowered = lowerer.lower_IndexStmt(cin)
    production_base_loads = _base_offset_loads(lowered)
    assert [load.var.name for load in production_base_loads] == ["_base1", "_base2"]
    for level, load in enumerate(production_base_loads, start=1):
        access = cast(llir.ArrayAccess, load.value)
        assert cast(llir.Var, access.array).name == f"_offset{level}"
        assert cast(llir.Var, access.array).type is llir.DataType.STD_VECTOR_INT
        assert cast(llir.Var, access.index).name == "batch"
        assert cast(llir.Var, access.index).type is llir.DataType.INT64
    production_owners = _offset_family_direct_initializations(lowered)
    production_prefix_loops = _prefix_sum_loops(lowered)
    production_totals = _total_offset_loads(lowered)
    production_family: List[llir.Node] = [
        *production_owners,
        *production_prefix_loops,
        *production_totals,
    ]
    assert [owner.var.name for owner in production_owners] == [
        "_count1",
        "_count2",
        "_offset1",
        "_offset2",
    ]
    assert len(production_prefix_loops) == 2
    assert [total.var.name for total in production_totals] == [
        "_total1",
        "_total2",
    ]
    production_values = _value_initializations(lowered)
    assert [value.var.name for value in production_values] == [
        "Result_values_torch",
        "Result_values_data",
    ]
    assert [LLIRLowerer().lower_llir(value) for value in production_values] == [
        "torch::Tensor Result_values_torch = "
        "torch::empty({_total2}, torch::kFloat32);",
        "float* Result_values_data = Result_values_torch.data_ptr<float>();",
    ]
    assert not any(
        "Result_values_torch" in code or "Result_values_data" in code
        for code in _raw_codes(lowered)
    )
    production_bounds = _named_vars(production_family, "Left0_size")
    assert len(production_bounds) == 8
    assert all(bound.type is llir.DataType.INT for bound in production_bounds)
    assert len({id(bound) for bound in production_bounds}) == 8
    cpp = LLIRLowerer().lower_llir(lowered)

    assert len(cpp) == 8648
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "adc0b71fd3f98ea3a437ffd980c237f1a649e3cf0b1d00cd64faf81701e49fbc"
    )
    assert cpp.count("wksp.insert(") == 2
    assert "wksp.insert_unchecked(" not in cpp
    assert cpp.count("int pLeft1_end = Left1_pos[pLeft0 + 1];") == 2
    assert cpp.count("pLeft1 < pLeft1_end; pLeft1++") == 2
    assert cpp.count("int pLeft2_end = Left2_pos[pLeft1 + 1];") == 2
    assert cpp.count("pLeft2 < pLeft2_end && pRight1 < pRight1_end") == 2
    assert cpp.count("int pRight1_end = Right1_pos[pRight0 + 1];") == 2
    assert cpp.count("int pRight2_end = Right2_pos[pRight1 + 1];") == 2
    assert cpp.count("pRight2 < pRight2_end; pRight2++") == 2
    assert [record.configuration_name for record in lowerer.llir_pass_run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
    ]


def _lower_production_ds(dtype: torch.dtype) -> llir.Function:
    row, reduction, column = IndexVar("row"), IndexVar("reduction"), IndexVar("column")
    result = TensorVar("Result", fmt="ds", dtype=dtype)
    left = TensorVar("Left", fmt="ds", dtype=dtype)
    right = TensorVar("Right", fmt="ds", dtype=dtype)
    assignment = TensorAssign(
        result[row, column],
        left[row, reduction] * right[reduction, column],
        op=Operation.ADD,
    )
    cin = cast(
        ForAll,
        Scheduler.auto_schedule(
            ForAll(row, ForAll(reduction, ForAll(column, assignment)))
        ),
    )

    lowered = CINLowerer().lower_IndexStmt(cin)

    assert type(lowered) is llir.Function
    return cast(llir.Function, lowered)


def test_production_ds_float64_locks_typed_value_source() -> None:
    lowered = _lower_production_ds(torch.float64)
    values = _value_initializations(lowered)

    assert [LLIRLowerer().lower_llir(value) for value in values] == [
        "torch::Tensor Result_values_torch = "
        "torch::empty({_total1}, torch::kFloat64);",
        "double* Result_values_data = Result_values_torch.data_ptr<double>();",
    ]
    assert not any(
        "Result_values_torch" in code or "Result_values_data" in code
        for code in _raw_codes(lowered)
    )
    cpp = LLIRLowerer().lower_llir(lowered)
    assert len(cpp) == 6277
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "b94eb205c39ea975ce1cc746930b776c64d78028c055c854815e198836e980a2"
    )


@pytest.mark.parametrize(
    ("dtype", "scalar_type", "pointer_type", "torch_dtype_name", "ctype"),
    [
        (
            torch.int8,
            llir.DataType.INT8,
            llir.DataType.PTR_INT8,
            "kInt8",
            "int8_t",
        ),
        (
            torch.uint8,
            llir.DataType.UINT8,
            llir.DataType.PTR_UINT8,
            "kUInt8",
            "uint8_t",
        ),
    ],
)
def test_production_ds_narrow_integer_values_are_structured(
    dtype: torch.dtype,
    scalar_type: llir.DataType,
    pointer_type: llir.DataType,
    torch_dtype_name: str,
    ctype: str,
) -> None:
    lowered = _lower_production_ds(dtype)
    owner, pointer = _value_initializations(lowered)

    assert owner.var.type is llir.DataType.TORCH_TENSOR
    empty = cast(llir.FunctionCall, owner.value)
    qualified = cast(llir.QualifiedName, empty.args[1])
    assert qualified.name == torch_dtype_name
    assert qualified.data_type is llir.DataType.TORCH_SCALAR_TYPE
    assert pointer.var.type is pointer_type
    data_ptr = cast(llir.MemberCall, pointer.value)
    assert data_ptr.template_args == (scalar_type,)
    assert [LLIRLowerer().lower_llir(value) for value in (owner, pointer)] == [
        "torch::Tensor Result_values_torch = "
        f"torch::empty({{_total1}}, torch::{torch_dtype_name});",
        f"{ctype}* Result_values_data = " f"Result_values_torch.data_ptr<{ctype}>();",
    ]
    value_targets = [
        cast(llir.ArrayAccess, assignment.var)
        for assignment in _assignments(lowered)
        if type(assignment.var) is llir.ArrayAccess
        and type(cast(llir.ArrayAccess, assignment.var).array) is llir.Var
        and cast(llir.Var, cast(llir.ArrayAccess, assignment.var).array).name
        == "Result_values_data"
    ]
    assert value_targets
    assert all(
        cast(llir.Var, target.array).type is pointer_type for target in value_targets
    )
    assert not any(
        "Result_values_torch" in code or "Result_values_data" in code
        for code in _raw_codes(lowered)
    )
