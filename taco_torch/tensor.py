from typing import Optional, Tuple, Union

from copy import deepcopy

import torch

from .storage import TensorStorage, TensorIndex, TensorStorageView


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
    ) -> object:
        super().__init__()
        if storage is not None:
            self._storage = storage
        else:
            self._storage = TensorStorage(index=index, value=value, shape=shape)
        self._name = name

        self.requires_grad = requires_grad

    def insert(self, indices, values):
        """Insert values into the tensor."""
        # TODO: Implement this.
        pass

    @property
    def value(self):
        """Get the tensor value."""
        return self._storage.value

    # dtype property
    @property
    def dtype(self):
        """Get the tensor logical dtype."""
        return self._dtype

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
