from scorch.compiler import cin
from scorch.compiler.shapes import cfir
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import tests.utility as util

# Tests CIN -> CFIR lowering phase.


def test_intersection():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    C = cin.TensorVar("C", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] * C[i]

    util.assert_equal(
        cfir.PrettyPrint(
            cfir.Lower(
                cin.ForAll(
                    i,
                    A._assignment,
                    cin.IntersectionSeq(
                        cin.IndexSeq(
                            i,
                            B,
                            size=8,
                            index=0,
                            format=LevelType.COMPRESSED,
                        ),
                        cin.IndexSeq(
                            i,
                            C,
                            size=8,
                            index=0,
                            format=LevelType.COMPRESSED,
                        ),
                    ),
                )
            )
        ),
        """while i <-- (B:s[i] ∩ C:s[i])
             switch i
               case: (B:s[i] ∩ C:s[i])
                 A:s[i] = (B:s[i] * C:s[i])""",
    )


def test_union():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    C = cin.TensorVar("C", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] + C[i]

    util.assert_equal(
        cfir.PrettyPrint(
            cfir.Lower(
                cin.ForAll(
                    i,
                    A._assignment,
                    cin.UnionSeq(
                        cin.IndexSeq(
                            i,
                            B,
                            size=8,
                            index=0,
                            format=LevelType.COMPRESSED,
                        ),
                        cin.IndexSeq(
                            i,
                            C,
                            size=8,
                            index=0,
                            format=LevelType.COMPRESSED,
                        ),
                    ),
                )
            )
        ),
        """while i <-- (B:s[i] ∪ C:s[i])
             switch i
               case: (B:s[i] ∪ C:s[i])
                 A:s[i] = (B:s[i] + C:s[i])
               case: B:s[i]
                 A:s[i] = B:s[i]
               case: C:s[i]
                 A:s[i] = C:s[i]
           while i <-- B:s[i] 
             A:s[i] = B:s[i]
           while i <-- C:s[i] 
             A:s[i] = C:s[i]""",
    )


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
                cin.IndexSeq(j, B, size=8, index=0, format=LevelType.COMPRESSED),
                start=0,
                end=8,
                stride=2,
            ),
        )
    ) == cfir.Loop(
        idx=i,
        sexpr=cin.SliceSeq(
            cin.IndexSeq(j, B, size=8, index=0, format=LevelType.COMPRESSED),
            start=0,
            end=8,
            stride=2,
        ),
        body=cfir.Assign(A[i], B[j]),
        locs=[],
    )


def test_collapse():
    A = cin.TensorVar("A", fmt=["d", "s"], shape=[8, 8])
    b = cin.TensorVar("b", fmt=["s"], shape=[8])
    c = cin.TensorVar("c", fmt=["d"], shape=[8])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    k = cin.IndexVar("k")
    c[k] = A[i, j] + b[k]

    Ai = cin.IndexSeq(i, A, size=8, index=0, format=LevelType.DENSE)
    Aj = cin.IndexSeq(j, A, size=8, index=1, format=LevelType.COMPRESSED)
    bk = cin.IndexSeq(k, b, size=8, index=0, format=LevelType.COMPRESSED)

    util.assert_equal(
        cfir.PrettyPrint(
            cfir.Lower(
                cin.ForAll(k, c._assignment, cin.UnionSeq(cin.Product(Ai, Aj), bk))
            )
        ),
        """
    while k <-- ((A:d,s[i] × A:d,s[j]) ∪ b:s[k]) 
      switch k
        case: ((A:d,s[i] × A:d,s[j]) ∪ b:s[k])
          c:d[k] = (A:d,s[i, j] + b:s[k])
        case: (A:d,s[i] × A:d,s[j])
          c:d[k] = A:d,s[i, j]
        case: b:s[k]
          c:d[k] = b:s[k]

    while k <-- (A:d,s[i] × A:d,s[j]) 
      c:d[k] = A:d,s[i, j]

    while k <-- b:s[k] 
      c:d[k] = b:s[k]
    """,
    )


def test_assign_2d_ss():
    A = cin.TensorVar("A", fmt=["s", "s"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "s"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.COMPRESSED)

    assert cfir.Lower(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)) == cfir.Loop(
        idx=i,
        sexpr=Bi,
        body=cfir.Loop(
            idx=j,
            sexpr=Bj,
            body=cfir.Assign(A[i, j], B[i, j]),
            locs=[],
        ),
        locs=[],
    )
