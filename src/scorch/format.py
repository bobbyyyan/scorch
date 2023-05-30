from enum import Enum
from typing import Optional, Union, List

from .compiler import llir


class LevelType(Enum):
    DENSE = "d"
    COMPRESSED = "s"
    SINGLETON = "singleton"
    COORDINATE = "o"


class LevelFormat(object):
    """
    A _level format has a type: compressed, dense, or singleton
    Also has bit width (for optimization)
    """

    _mode: LevelType
    _bit_width: Optional[int]

    # For the constructor, mode may be a string or a LevelType
    # if mode is a string, then we need to convert it to a LevelType

    def __init__(
        self,
        mode: Union[str, LevelType],
        bit_width: Optional[int] = None,
    ):
        if isinstance(mode, str):
            # convert string to LevelType
            if mode == "dense":
                mode = LevelType.DENSE
            elif mode == "compressed":
                mode = LevelType.COMPRESSED
            elif mode == "singleton":
                mode = LevelType.SINGLETON
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

    # Initializer takes in a single LevelFormat or a list of LevelFormats
    # or None (for a 0-order tensor/scalar)
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
        elif isinstance(level_formats, str):
            level_formats = list(level_formats)

        if isinstance(level_formats, list):
            self._level_formats = []
            for level_format in level_formats:
                if isinstance(level_format, str):
                    if level_format in ["dense", "d"]:
                        self._level_formats.append(LevelFormat(mode=LevelType.DENSE))
                    elif level_format in ["compressed", "sparse", "c", "s"]:
                        self._level_formats.append(
                            LevelFormat(mode=LevelType.COMPRESSED)
                        )
                    elif level_format in ["singleton", "single", "s"]:
                        self._level_formats.append(
                            LevelFormat(mode=LevelType.SINGLETON)
                        )
                    elif level_format in ["coordinate", "coord", "o"]:
                        self._level_formats.append(
                            LevelFormat(mode=LevelType.COORDINATE)
                        )
                    else:
                        raise ValueError(f"Invalid format string: {level_format}")
                elif isinstance(level_format, LevelFormat):
                    self._level_formats.append(level_format)

    def get_level_formats(self) -> List[LevelFormat]:
        assert self._level_formats is not None, "level_formats is None"
        return self._level_formats

    def get_level_types(self) -> List[LevelType]:
        return [level_format.get_level_type() for level_format in self._level_formats]

    def get_order(self) -> int:
        return len(self.get_level_formats())

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
