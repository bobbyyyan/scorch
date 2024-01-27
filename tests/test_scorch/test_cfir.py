import scorch.compiler.cin as cin
import scorch.compiler.scfir as cfir
from scorch.format import LevelType

# Tests CIN -> CFIR lowering phase.


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
    A = cin.TensorVar("A", fmt=["s", "d"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "d"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    assert cfir.Lower(
        cin.ForAll(
            i,
            cin.ForAll(
                j,
                A._assignment,
                cin.IndexSeq(j, B, size=10, index=1, format=LevelType.DENSE, parent=i),
            ),
            cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED),
        )
    ) == cfir.Loop(
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
