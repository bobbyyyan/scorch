from __future__ import annotations

import pdb
from copy import deepcopy
from typing import Optional, Tuple, Union, List

import torch

from .compiler.cin import TensorVar, ForAll, IndexVar
from .compiler.cin_lowerer import CINLowerer
from .compiler.codegen import LLIRLowerer
from .format import TensorFormat, LevelFormat, LevelType
from .storage import TensorStorage, TensorIndex, TensorStorageView
from .utils import PROJECT_ROOT_DIR, parse_format


class Window(object):
    """A tensor window object that describes the slice into a physical storage (TensorStorage)
    or another logical tensor (Tensor)
    Contains:
        - an offset for the starting coordinate of the window
        - a shape tuple for the shape of the window
        - a step tuple for the step of the window
    """

    def __init__(self, offset: Tuple[int], shape: Tuple[int], step: Tuple[int]):
        self.offset = offset
        self.shape = shape
        self.step = step

    def __str__(self):
        return f"Window(offset={self.offset}, shape={self.shape}, step={self.step})"

    def __repr__(self):
        return f"Window(offset={self.offset}, shape={self.shape}, step={self.step})"

    def __copy__(self):
        return Window(deepcopy(self.offset), deepcopy(self.shape), deepcopy(self.step))


class Tensor(torch.nn.Module):
    """A tensor stored in custom format."""

    _name: Optional[str]

    _shape: Optional[Tuple[int, ...]]

    # (Logical) component type, which might be different from the physical component type in TensorStorage
    _dtype: torch.dtype = torch.float32

    # TODO: storage can also be a secondary index (TensorStorageView)
    _storage: Optional[Union[TensorStorage, TensorStorageView]]

    def __init__(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        storage: Optional[Union[TensorStorage, TensorStorageView]] = None,
        index: Optional[TensorIndex] = None,
        value: Optional[torch.Tensor] = None,
        requires_grad: Optional[bool] = False,
    ) -> None:
        super().__init__()
        if storage is not None:
            self._storage = storage
        else:
            self._storage = TensorStorage(index=index, value=value, shape=shape)
        self._name = name
        self._shape = shape

        if value is not None:
            self._dtype = value.dtype
        elif storage and storage.value is not None:
            self._dtype = storage.value.dtype

        self.requires_grad = requires_grad

    def insert(self, indices, values):
        """Insert values into the tensor."""
        # TODO: Implement this.
        pass

    def _nnz(self):
        """Get the number of non-zero elements in the tensor."""
        return self.values.numel()

    @property
    def has_index(self) -> bool:
        """Return whether the tensor has an index."""
        return self.storage.has_index

    @property
    def name(self) -> str:
        """Get the tensor name."""
        assert self._name is not None, "Tensor name is not set."
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def values(self) -> torch.Tensor:
        """Get the tensor value."""
        return self.storage.value

    @property
    def index(self) -> TensorIndex:
        """Get the tensor index."""
        return self.storage.index

    @property
    def format(self) -> TensorFormat:
        """Get the tensor format."""
        tensor_format = self.index.format
        assert tensor_format is not None, "Tensor format is not set."
        return tensor_format

    @property
    def storage(self) -> TensorStorage:
        """Get the tensor storage."""
        assert self._storage is not None, "Tensor storage is not set."
        return self._storage

    @property
    def dtype(self):
        """Get the tensor logical dtype."""
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the tensor shape."""
        return self._shape if self._shape is not None else tuple()

    def __str__(self):
        """Get a string representation of the tensor."""
        # return f"TacoTensor_{self._name}({self._storage})"
        return "Tensor"

    def __repr__(self):
        """Get a string representation of the tensor."""
        return self.__str__()

    def validate(self):
        """Validate the tensor."""
        # TODO: Implement this.
        raise NotImplementedError()

    def to(self, device):
        """Move the tensor to a device."""
        # TODO: Implement this.
        raise NotImplementedError()

    def cuda(self):
        """Move the tensor to the GPU."""
        return self.to(torch.cuda.current_device())

    def clone(self):
        """Clone the tensor."""
        # TODO: Implement this.
        raise NotImplementedError()

    def dim(self):
        """Get the number of dimensions."""
        return len(self.shape)

    def __add__(self, other) -> Tensor:
        """Add two tensors together."""
        # Perform element-wise addition
        # TODO: support broadcasting
        a_index_vars = ([IndexVar(f"i{i}") for i in self._storage._index.mode_order])
        b_index_vars = ([IndexVar(f"i{i}") for i in self._storage._index.mode_order])
        c_index_vars = ([IndexVar(f"i{i}") for i in other._storage._index.mode_order])
        # TODO: output format inferred from input formats
        output_format = self.format
        result_shape = self.shape

        # TODO: should infer mode_order, fmt from LHS of addition expression
        A = TensorVar(
            name="A",
            fmt=output_format,
            mode_order=self._storage._index.mode_order
        )
        B = TensorVar(
            name="B",
            fmt=self.format,
            mode_order=self._storage._index.mode_order
        )
        C = TensorVar(
            name="C",
            fmt=other.format,
            mode_order=other._storage._index.mode_order
        )

        # Assert A, B, C, and index_vars are defined
        assert A is not None, "Tensor A is not defined."
        assert B is not None, "Tensor B is not defined."
        assert C is not None, "Tensor C is not defined."
        assert a_index_vars is not None, "Index variables for A are not defined."
        assert b_index_vars is not None, "Index variables for B are not defined."
        assert c_index_vars is not None, "Index variables for C are not defined."


        # Generate the python code for the element-wise addition
        # e.g. A[i0, i1, ...] = B[i0, i1, ...] + C[i0, i1, ...]
        lhs = f'A[{", ".join(["a_index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs = f'B[{", ".join(["b_index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs += f' + C[{", ".join(["c_index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        code = f"{lhs} = {rhs}"
        # pdb.set_trace()
        exec(code)

        # Generate the python code for constructing the ForAll's and execute it
        # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
        rhs = "A._assignment"
        assert ForAll is not None, "ForAll is not imported"
        for i in range(len(self.shape))[::-1]:
            rhs = f"ForAll(a_index_vars[{i}], {rhs})"
        cin_stmt = eval(rhs)
        pdb.set_trace()

        lowerer = CINLowerer()
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        print("\n\ncpp_code:\n\n", cpp_code)

        # Read header_cpp_code from csrc/header.cpp
        with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
            header_cpp_code = f.read()

        module = torch.utils.cpp_extension.load_inline(
            name="kernel",
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=["-O3"],
        )

        result_cpp = module.evaluate(
            result_shape,
            self.shape,
            self.index.mode_indices,
            self.storage.value,
            other.shape,
            other.index.mode_indices,
            other.storage.value,
        )

        result = Tensor(
            shape=result_shape,
            index=TensorIndex(
                mode_indices=result_cpp._storage._index.mode_indices,
                tensor_format=output_format,
            ),
            value=result_cpp._storage._value,
        )

        return result

    def __mul__(self, other) -> Tensor:
        """Multiply two tensors together."""
        raise NotImplementedError()

    def copy(self) -> Tensor:
        """Copy the tensor."""
        return Tensor(
            name=self._name,
            shape=self.shape,
            storage=self.storage.copy(),
        )

    @staticmethod
    def from_coo(
        indices: torch.Tensor,
        values: torch.Tensor,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
    ) -> Tensor:
        """
        Create a Tensor from a COO tensor.
        :param indices:
        :param values:
        :param shape:
        :param name:
        :return:
        """
        # If name is not provided, use the default name
        if name is None:
            name = "tensor"

        mode_indices = []
        for i in range(len(shape)):
            mode_indices.append([indices[i]])

        tt_tensor = Tensor(
            name=name,
            shape=tuple(shape),
            storage=TensorStorage(
                index=TensorIndex(
                    tensor_format=TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.COORDINATE)
                            for _ in range(len(shape))
                        ]
                    ),
                    mode_indices=mode_indices,
                ),
                value=values,
            ),
        )

        return tt_tensor

    @staticmethod
    def from_torch(tensor: torch.Tensor, name: Optional[str] = None, mode_order: Optional[List[int]] = None) -> Tensor:
        """Create a Tensor from a torch.Tensor."""
        # torch.Tensor is dense, so shape is the same as torch tensor,
        # and format is dense at every level

        # If name is not provided, use the default name
        if name is None:
            name = "tensor"

        # TODO: Should insert some error-checking with mode-order here?
        if mode_order:
            tensor = tensor.permute(*mode_order)

        tt_tensor = Tensor(
            name=name,
            shape=tuple(tensor.shape),
            storage=TensorStorage(
                index=TensorIndex(
                    tensor_format=TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.DENSE)
                            for _ in range(len(tensor.shape))
                        ]
                    ),
                    mode_indices=[[] for _ in range(len(tensor.shape))],
                    mode_order=mode_order
                ),
                value=tensor.flatten(),
            ),
        )

        # pdb.set_trace()

        return tt_tensor

    def to_torch(self, in_place=True) -> torch.Tensor:
        """Convert a Scorch Tensor to a torch.Tensor."""
        # Get a dense Scorch tensor
        dense_tensor = self.to_dense(in_place=in_place)
        # Convert the dense Scorch tensor to a torch.Tensor
        # torch_tensor = dense_tensor.storage.value.clone().detach()
        torch_tensor = dense_tensor.storage.value
        if torch_tensor.dtype != self.dtype:
            torch_tensor = torch_tensor.type(self.dtype)
        # Reshape the torch.Tensor to the original shape
        torch_tensor = torch_tensor.reshape(dense_tensor.shape)

        def generate_mode_order_permutation(mode_order_start: List[int], mode_order_end: List[int]) -> List[int]:
            permutation_ = []
            for dim in mode_order_end:
                permutation_.append(mode_order_start.index(dim))
            return permutation_

        # permute torch_tensor if it has non-default mode order
        default_mode_order = [i for i in range(self.dim())]

        if self._storage._index.mode_order and self._storage._index.mode_order != default_mode_order:
            permutation = generate_mode_order_permutation(self._storage._index.mode_order, default_mode_order)
            torch_tensor = torch_tensor.permute(*permutation)

        return torch_tensor

    def to_dense(
        self,
        fmt: Optional[Union[TensorFormat, str, List[str]]] = None,
        in_place: bool = False,
    ) -> Tensor:
        """Convert the Scorch tensor to a dense Scorch tensor."""

        # If self is already dense at every level, return self
        if self.format.is_dense():
            if in_place:
                return self
            else:
                return self.copy()

        default_index_vars = [IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]]

        if len(self.shape) > len(default_index_vars):
            index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
        else:
            index_vars = default_index_vars[: len(self.shape)]

        pdb.set_trace()
        # permute index_vars based on self._storage._index.mode_order
        if self._storage._index.mode_order:
            index_vars = [index_vars[i] for i in self._storage._index.mode_order]

        if self.has_index:
            B = TensorVar(
                name="B",
                fmt=self.format,
                dtype=self.dtype,
                mode_order=self._storage._index.mode_order
            )
        else:
            B = TensorVar(
                name="B",
                fmt=TensorFormat(
                    level_formats=[
                        LevelFormat(mode=LevelType.DENSE)
                        for _ in range(len(self.shape))
                    ]
                ),
                dtype=self.dtype,
                mode_order=self._storage._index.mode_order
            )

        if fmt is None:
            # TODO: infer output format from input format
            # For now, make every level COMPRESSED
            output_format = TensorFormat(
                level_formats=[
                    LevelFormat(mode=LevelType.DENSE) for _ in range(len(self.shape))
                ]
            )
        else:
            output_format = parse_format(fmt)

        A = TensorVar(
            name="A",
            fmt=output_format,
            dtype=self.dtype,
            mode_order=self._storage._index.mode_order
        )

        # Assert A, B, and index_vars are defined
        assert A is not None, "A is not defined"
        assert B is not None, "B is not defined"
        assert index_vars is not None, "index_vars is not defined"

        # Generate the python code for A[i0, i1, etc.] = B[i0, i1, etc.] and execute it
        lhs = f'A[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs = f'B[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        code = f"{lhs} = {rhs}"
        exec(code)

        # Generate the python code for constructing the ForAll's and execute it
        # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
        rhs = "A._assignment"
        assert ForAll is not None, "ForAll is not imported"
        for i in range(len(self.shape))[::-1]:
            rhs = f"ForAll(index_vars[{i}], {rhs})"
        cin_stmt = eval(rhs)

        lowerer = CINLowerer(filter_zeros=True)
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        # print("\n\ncpp_code:\n\n", cpp_code)

        # Read header_cpp_code from csrc/header.cpp
        with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
            header_cpp_code = f.read()

        module = torch.utils.cpp_extension.load_inline(
            name="kernel",
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=["-O3"],
        )

        result_cpp = module.evaluate(
            self.shape,
            self.shape,
            self.index.mode_indices,
            self.storage.value,
        )

        new_storage = TensorStorage(
            index=TensorIndex(
                tensor_format=output_format,
                mode_indices=result_cpp._storage._index.mode_indices,
                mode_order=self._storage._index.mode_order
            ),
            value=result_cpp._storage._value,
        )

        if in_place:
            self._storage = new_storage
            return self

        new_tensor = Tensor(
            name=self._name,
            shape=self.shape,
            storage=new_storage,
        )

        return new_tensor

    def to_sparse(
        self, fmt: Optional[Union[TensorFormat, str, List[str]]] = None
    ) -> Tensor:
        # pdb.set_trace()
        """Convert the Scorch tensor to a sparse Scorch tensor."""
        if len(self.shape) == 1:
            # Find indexes of non-zero elements in self.values, flatten them
            nonzero_indices = torch.nonzero(self.values).flatten()
            size = len(nonzero_indices)
            # Create a filtered value tensor that only contains non-zero elements
            filtered_values = self.values[nonzero_indices]
            self._storage = TensorStorage(
                index=TensorIndex(
                    tensor_format=TensorFormat(
                        level_formats=[LevelFormat(mode=LevelType.COMPRESSED)]
                    ),
                    mode_indices=[
                        [
                            torch.Tensor([0, size]),
                            nonzero_indices,
                        ]
                    ],
                ),
                value=filtered_values,
            )
        else:
            default_index_vars = [
                IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]
            ]
            if len(self.shape) > len(default_index_vars):
                index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
            else:
                index_vars = default_index_vars[: len(self.shape)]

            # permute index_vars based off self._storage._index.mode_order
            if self._storage._index.mode_order:
                index_vars = [index_vars[i] for i in self._storage._index.mode_order]

            if self.has_index:
                B = TensorVar(
                    name="B",
                    fmt=self.format,
                    dtype=self.dtype,
                    mode_order=self._storage._index.mode_order
                )
            else:
                B = TensorVar(
                    name="B",
                    fmt=TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.DENSE)
                            for _ in range(len(self.shape))
                        ]
                    ),
                    mode_order=self._storage._index.mode_order
                )

            if fmt is None:
                # TODO: infer output format from input format
                # For now, make every level COMPRESSED
                output_format = TensorFormat(
                    level_formats=[
                        LevelFormat(mode=LevelType.COMPRESSED)
                        for _ in range(len(self.shape))
                    ]
                )
            else:
                output_format = parse_format(fmt)

            A = TensorVar(
                name="A",
                fmt=output_format,
                shape=self.shape,
                dtype=self.dtype,
                mode_order=self._storage._index.mode_order
            )

            # Assert A, B, and index_vars are defined
            assert A is not None, "A is not defined"
            assert B is not None, "B is not defined"
            assert index_vars is not None, "index_vars is not defined"

            # Generate the python code for A[i0, i1, etc.] = B[i0, i1, etc.] and execute it
            lhs = f'A[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
            rhs = f'B[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
            code = f"{lhs} = {rhs}"
            exec(code)

            # Generate the python code for constructing the ForAll's and execute it
            # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
            rhs = "A._assignment"
            assert ForAll is not None, "ForAll is not imported"
            for i in range(len(self.shape))[::-1]:
                rhs = f"ForAll(index_vars[{i}], {rhs})"
            cin_stmt = eval(rhs)

            # print("\n\ncin_stmt: ", cin_stmt)

            lowerer = CINLowerer(filter_zeros=True)
            lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
            llir_lowerer = LLIRLowerer()
            cpp_code = llir_lowerer.lower_llir(lowered_llir)
            # pdb.set_trace()

            # print("\n\ncpp_code:\n\n", cpp_code)

            # Read header_cpp_code from csrc/header.cpp
            with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
                header_cpp_code = f.read()

            module = torch.utils.cpp_extension.load_inline(
                name="kernel",
                cpp_sources=[header_cpp_code, cpp_code],
                functions=["evaluate"],
                extra_cflags=["-O3"],
            )

            result_cpp = module.evaluate(
                self.shape,
                self.shape,
                self.index.mode_indices,
                self.storage.value,
            )

            # TODO: bypassing result_cpp for mode_order, C++ code should be completely unaware?
            self._storage = TensorStorage(
                index=TensorIndex(
                    tensor_format=output_format,
                    mode_indices=result_cpp._storage._index.mode_indices,
                    mode_order=self._storage._index.mode_order
                ),
                value=result_cpp._storage._value,
            )

        return self
