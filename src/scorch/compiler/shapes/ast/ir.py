from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

from scorch.compiler import cin

# Constructs used throughout the compiler, i.e., we choose to use the same
# representation rather than having a per-pass variant.
#
# TODO(cgyurgyik): The other CIN constructs that appear in CFIR/CPP phases
# should be translated to a "general" construct here too, e.g., TensorVar
# and TensorAccess.


@dataclass
class IndexExpression:
    pass


@dataclass
class IndexVar(IndexExpression):
    """A tensor index variable.
    e.g. i, j, k, ...
    An index variable is bound to a set of coordinates by a forall statement.
    """

    name: str
    parent: Optional[IndexVar]
    tensor_accesses: list[cin.TensorAccess] = field(default_factory=list)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_cin(i: cin.IndexVar):
        """Translates from a CIN index variable."""
        return IndexVar(
            name=i.name,
            parent=i.parent if i.has_parent else None,
            tensor_accesses=i.tensor_accesses,
        )
