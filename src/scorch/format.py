from enum import Enum
from typing import Optional, Union, List

from .compiler import llir


class LevelType(Enum):
    DENSE = "d"
    COMPRESSED = "s"
    SINGLETON = "singleton"
    COORDINATE = "o"


# Canonical mapping from string aliases to LevelType.
_STR_TO_LEVEL_TYPE = {
    "dense": LevelType.DENSE,
    "d": LevelType.DENSE,
    "compressed": LevelType.COMPRESSED,
    "sparse": LevelType.COMPRESSED,
    "c": LevelType.COMPRESSED,
    "s": LevelType.COMPRESSED,
    "singleton": LevelType.SINGLETON,
    "single": LevelType.SINGLETON,
    "coordinate": LevelType.COORDINATE,
    "coord": LevelType.COORDINATE,
    "o": LevelType.COORDINATE,
}


def _parse_level_type(s: str) -> LevelType:
    """Convert a string alias to a LevelType, or raise ValueError."""
    try:
        return _STR_TO_LEVEL_TYPE[s]
    except KeyError:
        raise ValueError(f"Invalid format string: {s}")


class LevelFormat(object):
    """
    A level format has a type: compressed, dense, or singleton
    Also has bit width (for optimization)
    """

    _mode: LevelType
    _bit_width: Optional[int]

    def __init__(
        self,
        mode: Union[str, LevelType],
        bit_width: Optional[int] = None,
    ):
        if isinstance(mode, str):
            mode = _parse_level_type(mode)
        assert isinstance(mode, LevelType)
        self._mode = mode
        self._bit_width = bit_width

    def get_level_type(self) -> LevelType:
        return self._mode

    def __str__(self):
        # return f'"{self._mode.value}"'
        return str(self._mode.value)

    def __repr__(self):
        return str(self)


class LevelPack:
    def __init__(self, level_type: LevelType, tensor: llir.Expr, mode: int, level: int):
        self.level_type = level_type
        self.tensor = tensor
        self.mode = mode
        self.level = level
        self.arrays = self.get_arrays(level_type, tensor, mode, level)

    @staticmethod
    def get_arrays(
        level_type: LevelType, tensor: llir.Expr, mode: int, level: int
    ) -> List[llir.Expr]:
        # TODO: implement this
        raise NotImplementedError


class TensorFormat(object):
    """A tensor format"""

    _level_formats: List[LevelFormat]

    # Fill value default to 0.0
    # TODO: extend to support other fill values
    _fill_value: Optional[float] = 0.0

    def __init__(
        self,
        level_formats: Optional[
            Union[LevelFormat, List[LevelFormat], List[str], str]
        ] = None,
    ):
        if level_formats is None:
            self._level_formats = []
        elif isinstance(level_formats, LevelFormat):
            self._level_formats = [level_formats]
        else:
            if isinstance(level_formats, str):
                level_formats = list(level_formats)
            self._level_formats = [
                lf if isinstance(lf, LevelFormat) else LevelFormat(mode=_parse_level_type(lf))
                for lf in level_formats
            ]

    def get_level_formats(self) -> List[LevelFormat]:
        assert self._level_formats is not None, "level_formats is None"
        return self._level_formats

    def get_level_types(self) -> List[LevelType]:
        return [level_format.get_level_type() for level_format in self._level_formats]

    def get_order(self) -> int:
        return len(self.get_level_formats())

    def is_dense(self) -> bool:
        return all(
            [
                level_format.get_level_type() == LevelType.DENSE
                for level_format in self._level_formats
            ]
        )

    def __str__(self):
        # return "TensorFormat({})".format(self._level_formats)
        # return str(self._level_formats)
        return ",".join([str(level_format) for level_format in self._level_formats])

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self._level_formats == other._level_formats

    def __ne__(self, other):
        return not self.__eq__(other)
