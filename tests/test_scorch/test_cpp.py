from scorch.compiler import cin, scfir as cfir, scpp as cpp, scodegen as codegen
from scorch.format import LevelType

from typing import List, Optional, Any, Tuple, Callable, Union, Sequence


# Tests CFIR -> CPP lowering phase.


def assert_equal(actual: cpp.Cpp, expected: str):
    """Asserts `actual` is equal to `expected` while ignoring white space,
    e.g.,
       assert_equal("a", "\n  a  \t ") # true
       assert_equal("ab", "a")         # false
    """

    def strip(s: Any) -> str:
        return str(s).replace(" ", "").strip()

    actual = strip(actual)
    expected = strip(expected)
    assert actual == expected, f"\nactual:{actual}\nexpected:{expected}\n"


def test_slice():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i] = B[j]

    assert_equal(
        codegen.Lower(
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
             A.crd[0][ip_A] = i;
             A.data[ip_A] = B.data[jp_B];
             ip_A += 1;
             jp_B += 1;
             while (((jp_B < B.pos[0][1]) && (!(((B.crd[0][jp_B] - 0) % 2) == 0)))) {
               jp_B += 1;
             }
           }""",
    )


def test_2d_assign():
    A = cin.TensorVar("A", fmt=["s", "d"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "d"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    assert_equal(
        codegen.Lower(
            cfir.Loop(
                idx=i,
                sexpr=cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED),
                body=cfir.Loop(
                    idx=j,
                    sexpr=cin.IndexSeq(
                        j, B, size=10, index=1, format=LevelType.DENSE, parent=i
                    ),
                    body=cfir.Assign(A[i, j], B[i, j]),
                ),
            )
        ),
        """while ((ip_B < B.pos[0][1])) {
             size_t i = B.crd[0][ip_B];
             while ((j_B < 10)) {
               size_t j = j_B;
               A.crd[0][ip_A] = i;
               A.data[ip_A] = B.data[ip_B];
               j_B += 1;
             }
             ip_B += 1;
           }""",
    )
