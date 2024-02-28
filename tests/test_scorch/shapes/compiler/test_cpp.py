from scorch.compiler import cin
from scorch.compiler.shapes import cfir, cpp, codegen
from scorch.format import LevelType
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
        codegen.PrettyPrint(
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
            )
        ),
        """
        size_t pB0 = B0_pos[0];
        while (((pB0 < B0_pos[1]) && ((B0_crd[pB0] % 2) != 0))) {
          pB0 += 1;
        }
        while (((pB0 < B0_pos[1]) && (B0_crd[pB0] < 8))) {
          size_t i = (B0_crd[pB0] / 2);
          A0_crd[pA0] = i;
          A_values[pA0] = B_values[pB0];
          pA0 += 1;
          pB0 += 1;
          while (((pB0 < B0_pos[1]) && ((B0_crd[pB0] % 2) != 0))) {
            pB0 += 1;
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
        codegen.PrettyPrint(
            codegen.Lower(
                cfir.Loop(
                    idx=i,
                    sexpr=cin.IndexSeq(
                        i, B, size=8, index=0, format=LevelType.COMPRESSED
                    ),
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
            )
        ),
        """
        size_t pB0 = B0_pos[0];
        while ((pB0 < B0_pos[1])) {
        size_t i = B0_crd[pB0];
        size_t B1 = 0;
        while ((B1 < 10)) {
            size_t j = B1;
            A0_crd[pA0] = i;
            A_values[((pA0 * 10) + j)] = B_values[((pB0 * 10) + j)];
            B1 += 1;
        }
        pB0 += 1;
        }""",
    )
