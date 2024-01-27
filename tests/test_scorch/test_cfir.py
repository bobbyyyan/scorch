import scorch.compiler.cin as cin
import scorch.compiler.cfir as cfir
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
    ) == [
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
    ]
