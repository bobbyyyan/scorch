"""Structured C++ ABI expressions shared by compiler lowering stages."""

from typing import cast

from . import llir


def tensor_storage_member(tensor_name: str, *members: str) -> llir.MemberAccess:
    """Build a fresh nested member path rooted at one generated ``Tensor``."""

    if type(tensor_name) is not str or not tensor_name.isidentifier():
        raise TypeError("tensor storage root must be a non-empty identifier")
    if not members:
        raise TypeError("tensor storage member path cannot be empty")
    expression: llir.Expr = llir.Var(
        name=tensor_name,
        type=llir.DataType.TACO_TENSOR,
    )
    for member in members:
        expression = llir.MemberAccess(base=expression, member=member)
    return cast(llir.MemberAccess, expression)


def mode_index_tensor(tensor_name: str, level: int, slot: int) -> llir.ArrayAccess:
    """Build a fresh access to one Torch tensor in the mode-index ABI argument."""

    if type(tensor_name) is not str or not tensor_name.isidentifier():
        raise TypeError("mode-index tensor root must be a non-empty identifier")
    if type(level) is not int or level < 0:
        raise TypeError("mode-index level must be a non-negative int")
    if type(slot) is not int or slot < 0:
        raise TypeError("mode-index slot must be a non-negative int")
    return llir.ArrayAccess(
        array=llir.ArrayAccess(
            array=llir.Var(
                name=f"{tensor_name}_mode_indices",
                type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
            ),
            index=llir.Literal(level, llir.DataType.INT),
        ),
        index=llir.Literal(slot, llir.DataType.INT),
    )


def tensor_data_ptr(receiver: llir.Expr, data_type: llir.DataType) -> llir.MemberCall:
    """Build a typed ``receiver.data_ptr<T>()`` expression."""

    if not isinstance(receiver, llir.Expr):
        raise TypeError("tensor data_ptr receiver must be an LLIR Expr")
    if type(data_type) is not llir.DataType:
        raise TypeError("tensor data_ptr template argument must be a DataType")
    return llir.MemberCall(
        base=receiver,
        member="data_ptr",
        template_args=(data_type,),
    )
