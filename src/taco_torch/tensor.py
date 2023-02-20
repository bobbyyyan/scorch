from __future__ import annotations
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch

from src.taco_torch.format import TensorFormat, LevelFormat, LevelType
from src.taco_torch.storage import TensorStorage, TensorIndex, TensorStorageView


class Window(object):
    """A tensor window object that describes the slice into a physical storage (TensorStorage)
    or another logical tensor (TacoTensor)
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


class TacoTensor(torch.nn.Module):
    """A tensor stored in custom format."""

    _name: Optional[str]

    _shape: Optional[Tuple[int, ...]]

    # (Logical) component type, which might be different from the physical component type in TensorStorage
    _dtype: torch.dtype = torch.float32

    # TODO: storage can also be a secondary index (TensorStorageView)
    _storage: Optional[Union[TensorStorage, TensorStorageView]]

    _value: Optional[torch.Tensor]

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

        self.requires_grad = requires_grad

    def insert(self, indices, values):
        """Insert values into the tensor."""
        # TODO: Implement this.
        pass

    @property
    def name(self) -> str:
        """Get the tensor name."""
        assert self._name is not None, "Tensor name is not set."
        return self._name

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

    # dtype property
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
        return "TacoTensor"

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

    def __add__(self, other) -> TacoTensor:
        """Add two tensors together."""
        raise NotImplementedError()

    def __mul__(self, other) -> TacoTensor:
        """Multiply two tensors together."""
        raise NotImplementedError()

    # function to create a TacoTensor from a torch.Tensor
    @staticmethod
    def from_torch(tensor: torch.Tensor, name: Optional[str] = None) -> TacoTensor:
        """Create a TacoTensor from a torch.Tensor."""
        # torch.Tensor is dense, so shape is the same as torch tensor,
        # and format is dense at every level
        tt_tensor = TacoTensor(
            name=name,
            shape=tuple(tensor.shape),
            storage=TensorStorage(
                value=tensor,
            ),
        )

        return tt_tensor

    # function to sparsify a TacoTensor
    def to_sparse(self, fmt: Optional[TensorFormat] = None) -> TacoTensor:
        """Convert the tensor to a sparse tensor."""
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
        return self
