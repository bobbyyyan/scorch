from scorch.compiler import cin
from scorch.compiler.shapes import cfir, codegen, cpp
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import tests.utility as util

# Tests CIN -> CFIR -> CPP lowering phase(s).

# TODO(cgyurgyik): Nondeterminism when iterating over sets.


def Compile(cin: cin.CIN) -> str:
    """Compiles CIN -> CFIR -> CIN, and then pretty prints it."""
    s0: cfir.CFIR = cfir.Lower(cin)
    s1: cpp.Cpp = codegen.Lower(s0)
    return codegen.PrettyPrint(s1)


def test_assign_1d_d():
    A = cin.TensorVar("A", fmt=["d"], shape=[8])
    B = cin.TensorVar("B", fmt=["d"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i]

    util.assert_equal(
        Compile(
            cin.ForAll(
                i,
                A._assignment,
                cin.IndexSeq(
                    i, B, size=8, index=0, parent=None, format=LevelType.DENSE
                ),
            )
        ),
        """
        size_t i_B = 0;
        while ((i_B < 8)) {
          size_t i = i_B;
          A.data[((0 * 8) + i)] = B.data[((0 * 8) + i)];
          i_B += 1;
        }""",
    )


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
        """
        size_t ip_B = B.pos[0][0];
        size_t ip_C = C.pos[0][0];
        while (((ip_B < B.pos[0][1]) && (ip_C < C.pos[0][1]))) {
          size_t i = min(B.crd[0][ip_B], C.crd[0][ip_C]);
          if (((i == ip_B) && (i == ip_C))) {
            A.crd[0][ip_A] = i;
            A.data[ip_A] = (B.data[ip_B] + C.data[ip_C]);
            ip_A += 1;
          } else if ((i == ip_C)) {
            A.crd[0][ip_A] = i;
            A.data[ip_A] = C.data[ip_C];
            ip_A += 1;
          } else if ((i == ip_B)) {
            A.crd[0][ip_A] = i;
            A.data[ip_A] = B.data[ip_B];
            ip_A += 1;
          }
          ip_B += (i == B.crd[0][ip_B]);
          ip_C += (i == C.crd[0][ip_C]);
        }
        while ((ip_C < C.pos[0][1])) {
          size_t i = C.crd[0][ip_C];
          A.crd[0][ip_A] = i;
          A.data[ip_A] = C.data[ip_C];
          ip_A += 1;
          ip_C += 1;
        }
        while ((ip_B < B.pos[0][1])) {
          size_t i = B.crd[0][ip_B];
          A.crd[0][ip_A] = i;
          A.data[ip_A] = B.data[ip_B];
          ip_A += 1;
          ip_B += 1;
        }""")
