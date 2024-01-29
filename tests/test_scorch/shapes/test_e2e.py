from scorch.compiler import cin
from scorch.compiler.shapes import cfir, codegen, cpp
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import tests.utility as util

# Tests CIN -> CFIR -> CPP lowering phase(s).


def Compile(cin: cin.CIN) -> str:
    """Compiles CIN -> CFIR -> CIN, and then pretty prints it."""
    s0: cfir.CFIR = cfir.Lower(cin)
    s1: cpp.Cpp = codegen.Lower(s0)
    return codegen.PrettyPrint(s1)


# TODO(cgyurgyik): Dense version causing issues
# during CFIR phase - probably need to remove-dense.
def test_assign_1d_s():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i]

    util.assert_equal(
        Compile(
            cin.ForAll(
                i,
                A._assignment,
                cin.IndexSeq(
                    i, B, size=8, index=0, parent=None, format=LevelType.COMPRESSED
                ),
            )
        ),
        """
      size_t ip_B = B.pos[0][0];
      while ((ip_B < B.pos[0][1])) {
        size_t i = B.crd[0][ip_B];
        A.crd[0][ip_A] = i;
        A.data[ip_A] = B.data[ip_B];
        ip_A += 1;
        ip_B += 1;
      }""",
    )


def test_union():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    C = cin.TensorVar("C", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] + C[i]

    return  # TODO(cgyurgyik): Provide `union` support in codegen.

    util.assert_equal(
        Compile(
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
        ),
        """""",
    )
