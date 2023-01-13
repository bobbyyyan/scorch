from typing import Optional, List, Tuple

import torch

from .format import TensorFormat, LevelType


class TensorLevelIndex(object):
    """A TensorLevelIndex contains a list of indices for a single level of a tensor."""

    _indices: Optional[List[torch.Tensor]]
    _level_type: Optional[LevelType]

    def __init__(
        self,
        indices: Optional[List[torch.Tensor]] = None,
        level_type: Optional[LevelType] = None,
    ):
        self._indices = indices
        self._level_type = level_type

    def __str__(self):
        return (
            f"TensorLevelIndex(level_type={self._level_type}, indices={self._indices})"
        )

    def __repr__(self):
        return (
            f"TensorLevelIndex(level_type={self._level_type}, indices={self._indices})"
        )


class TensorIndex(object):
    """An index contains the index data structure for a tensor, but not its values.
    Thus, an index has a format and zero or more mode indices that describe the
    non-empty coordinates in each mode.
    """

    def __init__(
        self,
        tensor_format: Optional[TensorFormat] = None,
        mode_indices: Optional[List[torch.Tensor]] = None,
    ):
        self.format = tensor_format
        self.mode_indices = mode_indices

    def __str__(self):
        return "TensorIndex({})".format(self.format)

    def __repr__(self):
        return "TensorIndex({})".format(self.format)

    def __eq__(self, other):
        return self.format == other.format and self.mode_indices == other.mode_indices

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_mode_index(self, mode: int) -> torch.Tensor:
        """Get the mode index of a mode."""
        return self.mode_indices[mode]

    def get_mode_indices(self) -> List[torch.Tensor]:
        """Get the mode indices of all modes."""
        return self.mode_indices

    def get_order(self) -> int:
        """Get the order of the tensor."""
        return self.format.get_order()

    def get_format(self) -> TensorFormat:
        """Get the format of the tensor."""
        return self.format

    def get_mode_format(self, mode: int) -> LevelType:
        """Get the format of a mode."""
        return self.format.get_mode_format(mode)

    def get_mode_formats(self) -> List[LevelType]:
        """Get the format of all modes."""
        return self.format.get_mode_formats()

    def get_dimension(self, mode: int) -> int:
        """Get the dimension of a mode."""
        return self.mode_indices[mode].shape(0)

    def get_dimensions(self) -> List[int]:
        """Get the dimensions of all modes."""
        return [self.mode_indices[mode].shape(0) for mode in range(self.get_order())]

    def get_size(self) -> int:
        """Get the number of non-zero elements in the tensor."""
        return self.mode_indices[0].shape(0)

    def get_sizes(self) -> List[int]:
        """Get the number of non-zero elements in each mode."""
        return [self.mode_indices[mode].shape(0) for mode in range(self.get_order())]


class TensorStorage(object):
    """A tensor storage."""

    _index: Optional[TensorIndex]
    _value: Optional[torch.Tensor]
    # (Physical) component type. default to float32
    _dtype: torch.dtype = torch.float32
    _shape: Optional[Tuple[int, ...]]

    def __init__(
        self,
        index: Optional[TensorIndex] = None,
        value: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self._index = index
        self._value = value
        self._shape = shape
        # self._fill_value = fill_value

    def __str__(self):
        return "TensorStorage({})"

    def __repr__(self):
        return "TensorStorage({})"

    @property
    def value(self) -> torch.Tensor:
        """Get the value of the tensor."""
        return self._value


class TensorStorageView(TensorStorage):
    """
    A tensor storage view is a tensor storage that is a view of another tensor storage.
    In database terms, a tensor storage view can be thought of as a secondary index
    into a physical tensor storage.

    """

    # View must point to another storage
    _storage: TensorStorage

    def __str__(self):
        return "TensorStorageView"

    def __repr__(self):
        return "TensorStorageView"

    pass
