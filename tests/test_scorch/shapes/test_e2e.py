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
          A.data[i] = B.data[i];
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


def test_assign_2d_ss():
    A = cin.TensorVar("A", fmt=["s", "s"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "s"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED, parent=None)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.COMPRESSED, parent=Bi)

    util.assert_equal(
        Compile(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t ip_B = B.pos[0][0];
        while ((ip_B < B.pos[0][1])) {
          size_t i = B.crd[0][ip_B];
          size_t jp_B = B.pos[1][ip_B];
          while ((jp_B < B.pos[1][(ip_B + 1)])) {
            size_t j = B.crd[1][jp_B];
            A.crd[0][ip_A] = i;
            A.crd[1][jp_A] = j;
            A.data[jp_A] = B.data[jp_B];
            jp_A += 1;
            jp_B += 1;
          }
          ip_B += 1;
        }""",
    )


def test_assign_2d_dd():
    A = cin.TensorVar("A", fmt=["d", "d"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["d", "d"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.DENSE, parent=None)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.DENSE, parent=Bi)

    util.assert_equal(
        Compile(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t i_B = 0;
        while ((i_B < 8)) {
          size_t i = i_B;
          size_t j_B = 0;
          while ((j_B < 10)) {
            size_t j = j_B;
            A.data[((i * 10) + j)] = B.data[((i * 10) + j)];
            j_B += 1;
          }
          i_B += 1;
        }""",
    )


def test_assign_2d_sd():
    A = cin.TensorVar("A", fmt=["s", "d"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "d"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED, parent=None)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.DENSE, parent=Bi)

    util.assert_equal(
        Compile(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t ip_B = B.pos[0][0];
        while ((ip_B < B.pos[0][1])) {
          size_t i = B.crd[0][ip_B];
          size_t j_B = 0;
          while ((j_B < 10)) {
            size_t j = j_B;
            A.crd[0][ip_A] = i;
            A.data[((ip_A * 10) + j)] = B.data[((ip_B * 10) + j)];
            j_B += 1;
          }
          ip_B += 1;
        }""",
    )


def test_assign_2d_ds():
    A = cin.TensorVar("A", fmt=["d", "s"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["d", "s"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.DENSE, parent=None)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.COMPRESSED, parent=Bi)

    util.assert_equal(
        Compile(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t i_B = 0;
        while ((i_B < 8)) {
          size_t i = i_B;
          size_t jp_B = B.pos[1][ip_B];
          while ((jp_B < B.pos[1][(ip_B + 1)])) {
            size_t j = B.crd[1][jp_B];
            A.crd[1][jp_A] = j;
            A.data[jp_A] = B.data[jp_B];
            jp_A += 1;
            jp_B += 1;
          }
          i_B += 1;
        }""",
    )


def test_union_1d_s():
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
                        i,
                        B,
                        size=8,
                        index=0,
                        format=LevelType.COMPRESSED,
                        parent=None,
                    ),
                    cin.IndexSeq(
                        i, C, size=8, index=0, format=LevelType.COMPRESSED, parent=None
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
          } else if ((i == ip_B)) {
            A.crd[0][ip_A] = i;
            A.data[ip_A] = B.data[ip_B];
            ip_A += 1;
          } else if ((i == ip_C)) {
            A.crd[0][ip_A] = i;
            A.data[ip_A] = C.data[ip_C];
            ip_A += 1;
          }
          ip_B += (i == B.crd[0][ip_B]);
          ip_C += (i == C.crd[0][ip_C]);
        }
        while ((ip_B < B.pos[0][1])) {
          size_t i = B.crd[0][ip_B];
          A.crd[0][ip_A] = i;
          A.data[ip_A] = B.data[ip_B];
          ip_A += 1;
          ip_B += 1;
        }
        while ((ip_C < C.pos[0][1])) {
          size_t i = C.crd[0][ip_C];
          A.crd[0][ip_A] = i;
          A.data[ip_A] = C.data[ip_C];
          ip_A += 1;
          ip_C += 1;
        }""",
    )


def test_union_1d_d():
    A = cin.TensorVar("A", fmt=["d"], shape=[8])
    B = cin.TensorVar("B", fmt=["d"], shape=[8])
    C = cin.TensorVar("C", fmt=["d"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] + C[i]

    util.assert_equal(
        Compile(
            cin.ForAll(
                i,
                A._assignment,
                cin.UnionSeq(
                    cin.IndexSeq(
                        i,
                        B,
                        size=8,
                        index=0,
                        format=LevelType.DENSE,
                        parent=None,
                    ),
                    cin.IndexSeq(
                        i, C, size=8, index=0, format=LevelType.DENSE, parent=None
                    ),
                ),
            )
        ),
        """
        size_t i_B = 0;
        while ((i_B < 8)) {
          size_t i = i_B;
          A.data[i] = (B.data[i] + C.data[i]);
          i_B += 1;
        }
        """,
    )


def test_intersection_1d_s():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    C = cin.TensorVar("C", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] * C[i]

    util.assert_equal(
        Compile(
            cin.ForAll(
                i,
                A._assignment,
                cin.IntersectionSeq(
                    cin.IndexSeq(
                        i,
                        B,
                        size=8,
                        index=0,
                        parent=None,
                        format=LevelType.COMPRESSED,
                    ),
                    cin.IndexSeq(
                        i,
                        C,
                        size=8,
                        index=0,
                        parent=None,
                        format=LevelType.COMPRESSED,
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
            A.data[ip_A] = (B.data[ip_B] * C.data[ip_C]);
            ip_A += 1;
          }
          ip_B += (i == B.crd[0][ip_B]);
          ip_C += (i == C.crd[0][ip_C]);
        }""",
    )


def test_intersection_1d_d():
    A = cin.TensorVar("A", fmt=["d"], shape=[8])
    B = cin.TensorVar("B", fmt=["d"], shape=[8])
    C = cin.TensorVar("C", fmt=["d"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] * C[i]

    util.assert_equal(
        Compile(
            cin.ForAll(
                i,
                A._assignment,
                cin.IntersectionSeq(
                    cin.IndexSeq(
                        i,
                        B,
                        size=8,
                        index=0,
                        parent=None,
                        format=LevelType.DENSE,
                    ),
                    cin.IndexSeq(
                        i,
                        C,
                        size=8,
                        index=0,
                        parent=None,
                        format=LevelType.DENSE,
                    ),
                ),
            )
        ),
        """
      size_t i_B = 0;
      while ((i_B < 8)) {
        size_t i = i_B;
        A.data[i] = (B.data[i] * C.data[i]);
        i_B += 1;
      }""",
    )


def test_spmm():
    # SpMM: C[i, j] = A[i, k] * B[k, j]
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    k = cin.IndexVar("k")

    A = cin.TensorVar("A", fmt=["dense", "sparse"], shape=[10, 10])
    B = cin.TensorVar("B", fmt=["dense", "sparse"], shape=[10, 10])
    C = cin.TensorVar("C", fmt=["dense", "sparse"], shape=[10, 10])

    C[i, j] = A[i, k] * B[k, j]
    # TODO(cgyurgyik): Implement.
