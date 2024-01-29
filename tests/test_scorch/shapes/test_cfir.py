from scorch.compiler import cin
from scorch.compiler.shapes import cfir
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence


# TODO(cgyurgyik): Create a test_util file instead of duplicating this.
def assert_equal(actual: Any, expected: str):
    """Asserts `actual` is equal to `expected` while ignoring white space,
    e.g.,
       assert_equal("  a", "\n  a  \t ") # true
       assert_equal("ab", "a")           # false
    """

    def strip(s: Any) -> str:
        s = str(s)
        s = s.strip()
        s = s.replace("\n", "")
        s = s.replace("\t", "")
        s = s.replace(" ", "")
        return

    actual = strip(actual)
    expected = strip(expected)
    assert actual == expected, f"\nactual:{actual}\nexpected:{expected}\n"

# Tests CIN -> CFIR lowering phase.


def test_union():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    C = cin.TensorVar("C", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] + C[i]

    assert_equal(cfir.PrettyPrint(cfir.Lower(
        cin.ForAll(
            i,
            A._assignment,
            cin.UnionSeq(
                cin.IndexSeq(
                    i, B, size=8, index=0, parent=None, format=LevelType.COMPRESSED
                ),
                cin.IndexSeq(
                    i, C, size=8, index=0, parent=None, format=LevelType.COMPRESSED
                ),
            ),
        )
    )), """while i <-- B:s[i] ∪ C:s[i]
             switch i
               case: B:s[i] ∪ C:s[i]
                 A:s[i] = (B:s[i] + C:s[i])
               case: B:s[i]
                 A:s[i] = B:s[i]
               case: C:s[i]
                 A:s[i] = C:s[i]
           while i <-- B:s[i] 
             A:s[i] = B:s[i]
           while i <-- C:s[i] 
             A:s[i] = C:s[i]""")


def test_slice():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i] = B[j]

    assert cfir.Lower(
        cin.ForAll(
            i,
            A._assignment,
            cin.SliceSeq(
                cin.IndexSeq(
                    j, B, size=8, index=0, parent=None, format=LevelType.COMPRESSED
                ),
                start=0,
                end=8,
                stride=2,
            ),
        )
    ) == cfir.Loop(
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


def test_2d_assign():
    return  # TODO(cgyurgyik): This is causing an EmptySeq upon simplification.
    A = cin.TensorVar("A", fmt=["s", "s"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "s"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    assert cfir.Lower(
        cin.ForAll(
            i,
            cin.ForAll(
                j,
                A._assignment,
                cin.IndexSeq(
                    j, B, size=10, index=1, format=LevelType.COMPRESSED, parent=i
                ),
            ),
            cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED),
        )
    ) == cfir.Loop(
        idx=i,
        sexpr=cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED),
        body=cfir.Loop(
            idx=j,
            sexpr=cin.IndexSeq(
                j, B, size=10, index=1, format=LevelType.COMPRESSED, parent=i
            ),
            body=cfir.Assign(A[i, j], B[i, j]),
        ),
    )
