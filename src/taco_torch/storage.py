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
        return self.get_mode_indices()[mode]

    def get_mode_indices(self) -> List[torch.Tensor]:
        """Get the mode indices of all modes."""
        assert self.mode_indices, "mode_indices is None"
        return self.mode_indices

    def get_order(self) -> int:
        """Get the order of the tensor."""
        return self.get_format().get_order()

    def get_format(self) -> TensorFormat:
        """Get the format of the tensor."""
        assert self.format, "format is None"
        return self.format

    def get_level_type(self, mode: int) -> LevelType:
        """Get the format of a mode."""
        return self.get_level_types()[mode]

    def get_level_types(self) -> List[LevelType]:
        """Get the format of all modes."""
        return self.get_format().get_level_types()

    def get_size(self, mode: int) -> int:
        """Get the size of a mode."""
        return self.get_mode_index(mode).shape[0]

    def get_sizes(self) -> List[int]:
        """Get the number of non-zero elements in each mode."""
        return [self.get_size(mode) for mode in range(self.get_order())]


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
        assert self._value is not None, "value is None"
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
