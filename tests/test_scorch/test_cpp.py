import scorch.compiler.cin as cin
import scorch.compiler.cfir as cfir
from scorch.format import LevelType
import scorch.compiler.cppcodegen as cppcodegen
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import scorch.compiler.cpp as cpp

# Tests CFIR -> CPP lowering phase.


def assert_equal(actual: cpp.Cpp, expected: str):
    """Asserts `actual` is equal to `expected` while ignoring white space."""

    def strip(s: Any) -> str:
        return str(s).replace(" ", "").strip()

    actual = strip(actual)
    expected = strip(actual)
    assert actual == expected, f"\nactual:{actual}\nexpected:{expected}\n"


def test_slice():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i] = B[j]

    assert_equal(
        cppcodegen.Lower(
            cfir.Loop(
                idx=i,
                sexpr=cin.SliceSeq(
                    cin.IndexSeq(
                        j, B, size=8, index=0, parent=None, format=LevelType.COMPRESSED
                    ),
                    start=0,
                    end=8,
                    stride=2,
                ),
                body=cfir.Assign(A[i], B[j]),
            )
        ),
        """while (((jp_B < B.pos[0][1]) && (B.crd[0][jp_B] < 8))) {
            size_t i = ((B.crd[0][jp_B] - 0) / 2);
            A.data[ip_A] = B.data[jp_B];
            ip_A += 1;
            jp_B += 1;
            while (((jp_B < B.pos[0][1]) && (! (((B.crd[0][jp_B] - 0) % 2) == 0)))) {
              jp_B += 1;
            }
          }""",
    )
