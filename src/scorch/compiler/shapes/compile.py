import torch

from scorch.compiler import cin
from scorch import tensor
from scorch.compiler.shapes import cfir, codegen, cpp
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence


def GetLevelAndValueArrays(argument: cin.TensorVar) -> Sequence[cpp.Cpp]:
    name: str = argument.get_name()
    stmts: list[cpp.Cpp] = []
    for i, level_type in enumerate(argument.get_level_types()):
        match level_type:
            case LevelType.DENSE:
                stmts.append(
                    cpp.Define(
                        cpp.Int32(),
                        cpp.Variable(f"{name}{i}_size"),
                        cpp.Variable(f"{name}_shape[{i}]"),
                    )
                )
            case LevelType.COMPRESSED:
                raise NotImplementedError(level_type)
            case LevelType.COORDINATE | _:
                raise NotImplementedError(level_type)
    return stmts


def InitializeResultTensor(result: cin.TensorVar) -> Sequence[cpp.Cpp]:
    stmts: list[cpp.Cpp] = []
    name: str = result.name
    if result.is_dense():
        # For dense results, we'll use malloc.
        capacity = cpp.Variable(f"{name}0_size")
        for i in range(1, result.levels):
            capacity = cpp.Mul(capacity, rhs=cpp.Variable(f"{name}{i}_size"))
        stmts.append(
            cpp.Define(cpp.Int32(), cpp.Variable(f"{name}_capacity"), capacity)
        )
        result_type: cpp.CppType = cpp.TypeFrom(result.dtype)

        bytesize = cpp.Mul(cpp.SizeOf(result_type), cpp.Variable(f"{name}_capacity"))
        stmts.append(
            cpp.Define(
                cpp.Pointer(result_type),
                cpp.Variable(f"{name}_values"),
                cpp.Cast(
                    cpp.Pointer(result_type),
                    cpp.FunctionCall(name="malloc", args=[bytesize]),
                ),
            )
        )
        # Additionally, we need to memset to 0.
        stmts.append(
            cpp.Expression(
                cpp.FunctionCall(
                    name="memset",
                    args=[cpp.Variable(f"{name}_values"), cpp.Constant(0), bytesize],
                )
            )
        )

    else:
        stmts.append(
            cpp.Declare(
                cpp.StdVector(cpp.TypeFrom(result.dtype)),
                cpp.Variable(f"{name}_values"),
            )
        )
    return stmts


def DefineFunction(
    functionname: str,
    stmt: cpp.Cpp,
    result: cin.TensorVar,
    arguments: List[cin.TensorVar],
) -> cpp.Function:
    # 1. Define function header.
    args: Sequence[Tuple[cpp.Type, cpp.Variable]] = []
    args.append((cpp.StdVector(cpp.Int32()), cpp.Variable("result_shape")))
    for tensor in arguments:
        argname = tensor.get_name()
        args.append((cpp.StdVector(cpp.Int32()), cpp.Variable(f"{argname}_shape")))
        args.append(
            (
                cpp.StdVector(cpp.StdVector(cpp.TorchTensor())),
                cpp.Variable(f"{argname}_mode_indices"),
            )
        )
        args.append((cpp.TorchTensor(), cpp.Variable(f"{argname}_tensor")))

    # 2. Prologue: insert local variables used in sparse iteration algorithm.
    resultname = result.get_name()
    prologue: Sequence[cpp.Cpp] = []
    for i, level_type in enumerate(result.get_level_types()):
        match level_type:
            case LevelType.DENSE:
                prologue.append(
                    cpp.Define(
                        cpp.Int32(),
                        cpp.Variable(f"{resultname}{i}_size"),
                        cpp.Variable(f"result_shape[{i}]"),
                    )
                )
            case LevelType.COMPRESSED:
                prologue.append(
                    cpp.Declare(
                        cpp.StdVector(cpp.Int32()), cpp.Variable(f"{resultname}{i}_pos")
                    )
                )
                prologue.append(
                    cpp.Declare(
                        cpp.StdVector(cpp.Int32()), cpp.Variable(f"{resultname}{i}_crd")
                    )
                )
                prologue.append(
                    cpp.Assign(cpp.Variable(f"{resultname}{i}_pos[0]"), cpp.Constant(0))
                )
                prologue.append(
                    cpp.Define(
                        cpp.Int32(), cpp.Variable(f"p{resultname}{i}"), cpp.Constant(0)
                    )
                )
                prologue.append(
                    cpp.Define(
                        cpp.Int32(),
                        cpp.Variable(f"{resultname}{i}_pos_index"),
                        cpp.Constant(0),
                    )
                )
                raise NotImplementedError(f"prologue for {level_type} format")
                # See: https://github.com/bobbyyyan/scorch/blob/56c27071644acb5cc4cd0c0f5486c713d6cbf06e/src/scorch/compiler/cin_lowerer.py#L1173
            case LevelType.COORDINATE | _:
                raise NotImplementedError(level_type)
    for argument in arguments:
        prologue.extend(GetLevelAndValueArrays(argument))

        cpptype: cpp.CppType = cpp.TypeFrom(argument.dtype)
        name: str = argument.name
        prologue.append(
            cpp.Define(
                cpp.Pointer(cpptype),
                cpp.Variable(f"{name}_values"),
                cpp.Variable(f"{name}_tensor.data_ptr<{cpptype}>()"),
            )
        )

    prologue.extend(InitializeResultTensor(result))

    # 3. Epilogue: initialize and return the final argument.
    epilogue: Sequence[cpp.Cpp] = []
    epilogue.append(cpp.Declare(cpp.TacoTensor(), resultname))

    if result.is_dense():
        # Dense tensors require a lambda deleter to delete the result value array.
        epilogue.append(
            cpp.Define(
                cpp.Auto(),
                cpp.Variable(f"{resultname}_tensor_deleter"),
                cpp.Variable("[](void* ptr) { free(ptr); }"),
            )
        )
        epilogue.append(
            cpp.Define(
                cpp.TorchTensor(),
                cpp.Variable(f"{resultname}_tensor_torch"),
                cpp.FunctionCall(
                    name="torch::from_blob",
                    args=[
                        cpp.Variable(f"{resultname}_values"),
                        cpp.Variable(f"{{{resultname}_capacity}}"),
                        cpp.Variable(f"{resultname}_tensor_deleter"),
                        cpp.Variable(cpp.PyTorchTypeToString(result.dtype)),
                    ],
                ),
            )
        )
    else:
        raise NotImplementedError("sparse result initialization")

    def mode_index_set(level: int, level_type: LevelType) -> str:
        name = f"{resultname}{level}"
        match level_type:
            case LevelType.DENSE:
                return "{}"
            case LevelType.COMPRESSED:
                return f"{{{name}_pos_torch, {name}_crd_torch}}"
            case LevelType.COORDINATE:
                return f"{{{name}_crd_torch}}"
            case _:
                raise NotImplementedError(level_type)

    epilogue.append(
        cpp.Assign(
            cpp.Variable(f"{resultname}._storage._index.mode_indices"),
            cpp.Variable(
                f"{{{', '.join(mode_index_set(i, l) for i, l in enumerate(result.get_level_types()))}}}"
            ),
        )
    )
    epilogue.append(
        cpp.Assign(
            cpp.Variable(f"{resultname}._storage._value"),
            cpp.Variable(f"{resultname}_tensor_torch"),
        )
    )
    epilogue.append(cpp.Return(resultname))

    body = cpp.Block(stmts=[*prologue, stmt, *epilogue])
    return cpp.Function(
        returntype=cpp.TacoTensor(), name=functionname, body=body, args=args
    )


def Compile(cin: cin.CIN) -> cpp.Cpp:
    """Compiles CIN -> CFIR -> CPP"""
    s0: cfir.CFIR = cfir.Lower(cin)
    s1: cpp.Cpp = codegen.Lower(s0)
    return s1


def CompileAndPrint(cin: cin.CIN) -> str:
    """Compiles CIN -> CFIR -> CPP, and then pretty prints it."""
    return codegen.PrettyPrint(Compile(cin))


def ConvertToTensorVar(t: tensor.Tensor) -> cin.TensorVar:
    return cin.TensorVar(name=t.name, shape=t.shape, fmt=t.format, dtype=t.dtype)


def CompileAndExecuteFunction(
    stmt: cpp.Cpp, result: tensor.Tensor, arguments: List[tensor.Tensor]
) -> tensor.Tensor:
    fn: cpp.Function = DefineFunction(
        "evaluate",
        stmt,
        ConvertToTensorVar(result),
        [ConvertToTensorVar(a) for a in arguments],
    )

    from pathlib import Path

    path = Path(__file__)
    while not (path / "setup.py").exists():
        path = path.parent
    with open(path / "csrc" / "header.cpp", "r") as file:
        header_cpp = file.read()

    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp, codegen.PrettyPrint(fn)],
        functions=["evaluate"],
        extra_cflags=["-O3"],
    )

    args = [result.shape]
    for t in arguments:
        args.extend([t.shape, t.index.mode_indices, t.values])

    output = module.evaluate(*args)
    return tensor.Tensor(
        shape=result.shape,
        index=tensor.TensorIndex(
            mode_indices=output._storage._index.mode_indices,
            tensor_format=result.format,
        ),
        value=output._storage._value,
    )
