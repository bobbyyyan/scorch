from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
from scorch.compiler import cin
from scorch.compiler.shapes import cfir, cpp, cpputil, iterator as it
import scorch.format as format


def LowerLoop(idx: cin.IndexVar, sexpr: cin.Seq, body: cfir.CFIR, first: bool):
    loop = cpp.While(
        it.Valid(sexpr),
        cpp.Block(
            stmts=[
                cpp.Define(
                    type=cpp.IndexType(), lhs=cpp.Variable(idx.name), rhs=it.Eval(sexpr)
                ),
                Lower(body),
                it.Next(cpp.Variable(idx.name), sexpr)
                if isinstance(body, cfir.Switch)
                else it.UnconditionalNext(sexpr),
            ]
        ),
    )
    return cpp.Block(stmts=[it.Init(sexpr), loop]) if first else loop


def LowerIndexExpr(expr: cin.IndexExpr) -> cpp.Cpp:
    match expr:
        case cin.TensorAccess():
            tensor: cin.TensorVar = expr.tensor
            return cpp.Access(
                cpp.Variable(f"{tensor.name}.data"),
                LowerIndexExprRec(expr, i=expr.num_levels),
            )
        case cin.BinaryOp():
            lhs: cpp.Cpp = LowerIndexExpr(expr.left)
            rhs: cpp.Cpp = LowerIndexExpr(expr.right)
            match op := expr.op:
                case cin.Operation.ADD:
                    return cpp.Add(lhs, rhs)
                case cin.Operation.MUL:
                    return cpp.Mul(lhs, rhs)
                case _:
                    raise NotImplementedError(op)
        case _:
            raise NotImplementedError(type(expr))


def LowerIndexExprRec(expr: cin.IndexExpr, i: int):
    i = i - 1  # Off by one, since we'll pass in the total number of indices.
    match expr:
        case cin.TensorAccess():
            if not (0 <= i < expr.num_levels):
                return cpp.Constant(0)
            if expr.num_levels == 0:
                return cpp.Variable(expr.tensor.name)
            idx: cin.IndexVar = expr.get_index_vars()[i]
            fmt: format.LevelType = expr.level_types()[i]
            match fmt:
                case format.LevelType.COMPRESSED:
                    return cpp.Variable(f"{idx.name}p_{expr.tensor.name}")
                case format.LevelType.DENSE:
                    return cpp.Add(
                        cpp.Mul(LowerIndexExprRec(expr, i), expr.get_tensor().shape[i]),
                        cpp.Variable(idx.name),
                    )
                case _:
                    raise NotImplementedError(fmt)
        case _:
            raise NotImplementedError(expr)


def LowerAssign(lhs: cin.TensorAccess, rhs: cin.IndexExpr):
    iterators = cpputil.UpdateCompressedIterators(lhs)
    return cpp.Block(
        stmts=[
            # Write to compressed dimensions.
            *IndexWriteList(lhs),
            # Perform assignment/compute.
            cpp.Assign(LowerIndexExpr(lhs), LowerIndexExpr(rhs)),
            # Update compressed iterators.
            *([] if iterators is None else [iterators]),
        ]
    )


def IndexWriteList(ta: cin.TensorAccess) -> list[cpp.Cpp]:
    types: List[format.LevelType] = ta.level_types()
    indices: List[cin.IndexVar] = ta.get_index_vars()

    def is_compressed(pair: Tuple[format.LevelType, cin.IndexVar]):
        return pair[0] == format.LevelType.COMPRESSED

    compressed_levels = list(filter(is_compressed, zip(types, indices)))
    return [IndexWrite(ta, idx) for (_, idx) in compressed_levels]


def IndexWrite(ta: cin.TensorAccess, idx: cin.IndexVar) -> cpp.Cpp:
    index: int = ta.level_of_index_var(idx)
    format: format.LevelType = ta.level_types()[index]
    return cpp.Assign(
        cpputil.ArrayAccessCrd(ta.tensor, idx, index, format), cpp.Variable(idx.name)
    )


def Lower(stmt: cfir.CFIR) -> cpp.Cpp:
    def _Lower(stmt: cfir.CFIR, first=False):
        match stmt:
            case cfir.Loop(idx, sexpr, body):
                return LowerLoop(idx, sexpr, body, first)
            case cfir.Assign(lhs, rhs):
                return LowerAssign(lhs, rhs)
            case cfir.Block(stmts):
                return cpp.Block(
                    stmts=[_Lower(s, first=(i == 0)) for (i, s) in enumerate(stmts)]
                )
            case cfir.Switch(idx, cases):
                return cpp.IfBlock(
                    pairs=list(
                        map(
                            lambda c: (
                                it.Equals(cpp.Variable(idx.name), c.sexpr),
                                Lower(c.stmt),
                            ),
                            cases,
                        )
                    )
                )
            case _:
                raise NotImplementedError(type(stmt))

    return _Lower(stmt, first=True)


########################################
############# Pretty Print #############
########################################


def PrettyPrint(stmt: cpp.Cpp, indent_level: int = 0) -> str:
    """
    Pretty print for the CPP intermediate representation.
    This will handle indentation.
    """

    def PpExpr(e: cpp.Cpp) -> str:
        return str(e)

    def indent() -> str:
        return indent_level * " "

    def PpIf(cond: Optional[cpp.Cpp], body: cpp.Cpp) -> str:
        pp: str = ""
        if cond is not None:
            pp += f"if ({PpExpr(cond)})"
        pp += " {"
        pp += "\n"
        pp += PrettyPrint(body, indent_level + 2)
        pp += "\n"
        pp += indent()
        pp += "}"
        return pp

    pp: str = ""
    match stmt:
        case cpp.Block(stmts):
            pp += "\n".join(PrettyPrint(stmt, indent_level) for stmt in stmts)
        case cpp.IfBlock(pairs):
            pp += indent()
            pp += " else ".join(PpIf(p[0], p[1]) for p in pairs)
        case cpp.While(cond, block):
            # while (cond) {
            #   stmt0
            #   stmt1
            #   ...
            # }
            pp += indent()
            pp += f"while ({PpExpr(cond)}) "
            pp += "{"
            pp += "\n"
            pp += PrettyPrint(block, indent_level + 2)
            pp += "\n"
            pp += indent()
            pp += "}"
        case _:
            pp += indent()
            pp += PpExpr(stmt)
    return pp
