"""Canonical plain-Python CSR container for the Phase-3.5 LoopIR spike.

The container is the canonical CSR adapter on the interpreter's input side and
its current sparse-output representation. Other level layouts bind through
``LevelTensorStorage``. It is deliberately Torch-free and list/tuple-based so
the interpreter stays an independent oracle. Canonical means: exact int/float
scalars, ``indptr`` of length ``n_rows + 1`` starting at zero and nondecreasing,
per-row column indices strictly increasing and in range, and value/index
streams of equal length. Explicit zero values are permitted (a UNION add can
cancel to an exact zero at an overlapping coordinate); ``from_dense`` never
manufactures them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


class CsrFormatError(Exception):
    """A container construction violated the canonical CSR contract."""


def _expect_exact_int(value: object, what: str) -> int:
    if type(value) is not int:
        raise CsrFormatError(f"{what} must be an exact int, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class CsrMatrix:
    """One canonical, immutable rank-2 dense-by-compressed matrix."""

    n_rows: int
    n_cols: int
    indptr: Tuple[int, ...]
    indices: Tuple[int, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        n_rows = _expect_exact_int(self.n_rows, "n_rows")
        n_cols = _expect_exact_int(self.n_cols, "n_cols")
        if n_rows < 0 or n_cols < 0:
            raise CsrFormatError("matrix dimensions must be nonnegative")
        for name, stream in (
            ("indptr", self.indptr),
            ("indices", self.indices),
            ("values", self.values),
        ):
            if type(stream) is not tuple:
                raise CsrFormatError(f"{name} must be an owned tuple")
        if len(self.indptr) != n_rows + 1:
            raise CsrFormatError(f"indptr length {len(self.indptr)} for {n_rows} rows")
        previous = 0
        for position, offset in enumerate(self.indptr):
            _expect_exact_int(offset, f"indptr[{position}]")
            if position == 0:
                if offset != 0:
                    raise CsrFormatError("indptr must start at zero")
            elif offset < previous:
                raise CsrFormatError("indptr must be nondecreasing")
            previous = offset
        if len(self.indices) != len(self.values):
            raise CsrFormatError("indices and values must have equal length")
        if self.indptr[-1] != len(self.indices):
            raise CsrFormatError("indptr must terminate at the stored-entry count")
        for position, value in enumerate(self.values):
            if type(value) is not float:
                raise CsrFormatError(f"values[{position}] must be an exact float")
        for row in range(n_rows):
            last_column = -1
            for position in range(self.indptr[row], self.indptr[row + 1]):
                column = _expect_exact_int(
                    self.indices[position], f"indices[{position}]"
                )
                if not 0 <= column < n_cols:
                    raise CsrFormatError(
                        f"column {column} outside [0, {n_cols}) in row {row}"
                    )
                if column <= last_column:
                    raise CsrFormatError(
                        f"row {row} columns must be strictly increasing"
                    )
                last_column = column

    @property
    def nnz(self) -> int:
        """Number of stored entries, including explicit zeros."""

        return len(self.values)

    @classmethod
    def from_dense(cls, rows: Sequence[Sequence[float]], n_cols: int) -> "CsrMatrix":
        """Build a canonical matrix from a rectangular nested sequence.

        Exact zeros are dropped; ``n_cols`` is explicit so zero-row and
        zero-column shapes stay representable.
        """

        n_cols = _expect_exact_int(n_cols, "n_cols")
        if type(rows) not in (list, tuple):
            raise CsrFormatError("rows must be an owned list or tuple")
        indptr: List[int] = [0]
        indices: List[int] = []
        values: List[float] = []
        for row_number, row in enumerate(rows):
            if type(row) not in (list, tuple):
                raise CsrFormatError(f"row {row_number} must be an owned list or tuple")
            if len(row) != n_cols:
                raise CsrFormatError(
                    f"row {row_number} has {len(row)} columns, expected {n_cols}"
                )
            for column, value in enumerate(row):
                if type(value) not in (int, float):
                    raise CsrFormatError(
                        f"row {row_number} column {column} must be an exact "
                        "int or float"
                    )
                try:
                    value = float(value)
                except (OverflowError, TypeError, ValueError) as error:
                    raise CsrFormatError(
                        f"row {row_number} holds an unrepresentable numeric entry"
                    ) from error
                if value != 0.0:
                    indices.append(column)
                    values.append(value)
            indptr.append(len(indices))
        return cls(
            n_rows=len(indptr) - 1,
            n_cols=n_cols,
            indptr=tuple(indptr),
            indices=tuple(indices),
            values=tuple(values),
        )

    def to_dense(self) -> List[List[float]]:
        """Materialize the stored matrix as nested lists of floats."""

        dense = [[0.0] * self.n_cols for _ in range(self.n_rows)]
        for row in range(self.n_rows):
            for position in range(self.indptr[row], self.indptr[row + 1]):
                dense[row][self.indices[position]] = self.values[position]
        return dense

    def row_segment(self, row: int) -> Tuple[int, int]:
        """Half-open stored-position range for one row."""

        row = _expect_exact_int(row, "row")
        if not 0 <= row < self.n_rows:
            raise CsrFormatError(f"row {row} outside [0, {self.n_rows})")
        return self.indptr[row], self.indptr[row + 1]
