from scorch.compiler import cin
from scorch.compiler.shapes import cfir, cpp, codegen
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import tests.utility as util

# Tests CFIR -> CPP lowering phase.


def test_pretty_print():
    assert (
        codegen.PrettyPrint(cpp.Assign(cpp.Variable("a"), cpp.Constant(1))) == "a = 1;"
    )

    assert (
        codegen.PrettyPrint(
            cpp.While(
                cond=cpp.Lt(cpp.Variable("pb_A"), cpp.Constant(42)),
                body=cpp.Block(
                    stmts=[
                        cpp.Define(cpp.Int32(), cpp.Variable("i"), cpp.Constant(0)),
                        cpp.IncAssign(
                            cpp.Access(
                                array=cin.TensorVar(name="A"),
                                idx=cin.IndexVar(name="i"),
                            ),
                            cpp.Constant(1),
                        ),
                    ]
                ),
            )
        )
        == "while ((pb_A < 42)) {\n  int32_t i = 0;\n  A[i] += 1;\n}"
    )


def test_slice():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i] = B[j]

    util.assert_equal(
        codegen.Lower(
            cfir.Loop(
                idx=i,
                sexpr=cin.SliceSeq(
                    cin.IndexSeq(
                        j, B, size=8, index=0, format=LevelType.COMPRESSED
                    ),
                    start=0,
                    end=8,
                    stride=2,
                ),
                body=cfir.Assign(A[i], B[j]),
                locs=[],
            )
        ),
        """
        size_t jp_B = B.pos[0][0];
        while (((jp_B < B.pos[0][1]) && (!(((B.crd[0][jp_B] - 0) % 2) == 0)))) {
            jp_B += 1;
        }
        while (((jp_B < B.pos[0][1]) && (B.crd[0][jp_B] < 8))) {
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


def test_assign_2d_sd():
    A = cin.TensorVar("A", fmt=["s", "d"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "d"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    util.assert_equal(
        codegen.Lower(
            cfir.Loop(
                idx=i,
                sexpr=cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED),
                body=cfir.Loop(
                    idx=j,
                    sexpr=cin.IndexSeq(
                        j, B, size=10, index=1, format=LevelType.DENSE
                    ),
                    body=cfir.Assign(A[i, j], B[i, j]),
                    locs=[],
                ),
                locs=[],
            )
        ),
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
