import torch

from scorch.compiler import cin
from scorch.compiler.shapes.ast import cpp
from scorch.compiler.shapes.lower import compile
from scorch.format import LevelType
import tests.utility as util

# Tests CIN -> CFIR -> CPP lowering phase(s).


def test_assign_e2e_ds():
    A = cin.TensorVar("A", fmt=["d", "s"], shape=[2, 2])
    B = cin.TensorVar("B", fmt=["d", "s"], shape=[2, 2])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    c = cin.ForAll(
        i,
        cin.ForAll(
            j,
            A._assignment,
            cin.IndexSeq(j, B, size=2, index=1, format=LevelType.COMPRESSED),
        ),
        cin.IndexSeq(i, B, size=2, index=0, format=LevelType.DENSE),
    )
    stmt: cpp.Cpp = compile.Compile(c)
    return  # TODO: Sparse result not supported.
    util.assert_equal(
        cpp.PrettyPrint(
            compile.DefineFunction(
                functionname="evaluate", stmt=stmt, result=A, arguments=[B]
            )
        ),
        """""",
    )


def test_assign_e2e_dense():
    A = cin.TensorVar("A", fmt=["d", "d"], shape=[2, 2])
    B = cin.TensorVar("B", fmt=["d", "d"], shape=[2, 2])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    c = cin.ForAll(
        i,
        cin.ForAll(
            j,
            A._assignment,
            cin.IndexSeq(j, B, size=2, index=1, format=LevelType.DENSE),
        ),
        cin.IndexSeq(i, B, size=2, index=0, format=LevelType.DENSE),
    )
    stmt: cpp.Cpp = compile.Compile(c)
    util.assert_equal(
        cpp.PrettyPrint(
            compile.DefineFunction(
                functionname="evaluate", stmt=stmt, result=A, arguments=[B]
            )
        ),
        """Tensor evaluate(std::vector<int32_t> result_shape, std::vector<int32_t> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_tensor) {
            int32_t A0_size = result_shape[0];
            int32_t A1_size = result_shape[1];
            int32_t B0_size = B_shape[0];
            int32_t B1_size = B_shape[1];
            float* B_values = B_tensor.data_ptr<float>();
            int32_t A_capacity = (A0_size * A1_size);
            float* A_values = (float*) malloc((sizeof(float) * A_capacity));
            memset(A_values, 0, (sizeof(float) * A_capacity));
            size_t B0 = 0;
            while ((B0 < 2)) {
              size_t i = B0;
              size_t B1 = 0;
              while ((B1 < 2)) {
                size_t j = B1;
                A_values[((i * 2) + j)] = B_values[((i * 2) + j)];
                B1 += 1;
              }
              B0 += 1;
            }
            Tensor A;
            auto A_tensor_deleter = [](void* ptr) { free(ptr); };
            torch::Tensor A_tensor_torch = torch::from_blob(A_values, {A_capacity}, A_tensor_deleter, torch::kFloat32);
            A._storage._index.mode_indices = {{}, {}};
            A._storage._value = A_tensor_torch;
            return A;
           }""",
    )


def test_assign_1d_d():
    A = cin.TensorVar("A", fmt=["d"], shape=[8])
    B = cin.TensorVar("B", fmt=["d"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i]

    util.assert_equal(
        compile.CompileAndPrint(
            cin.ForAll(
                i,
                A._assignment,
                cin.IndexSeq(i, B, size=8, index=0, format=LevelType.DENSE),
            )
        ),
        """
        size_t B0 = 0;
        while ((B0 < 8)) {
          size_t i = B0;
          A_values[i] = B_values[i];
          B0 += 1;
}""",
    )


def test_assign_1d_s():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i]

    util.assert_equal(
        compile.CompileAndPrint(
            cin.ForAll(
                i,
                A._assignment,
                cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED),
            )
        ),
        """
        size_t pB0 = B0_pos[0];
        while ((pB0 < B0_pos[1])) {
          size_t i = B0_crd[pB0];
          A0_crd[pA0] = i;
          A_values[pA0] = B_values[pB0];
          pA0 += 1;
          pB0 += 1;
        }""",
    )


def test_assign_2d_ss():
    A = cin.TensorVar("A", fmt=["s", "s"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "s"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.COMPRESSED)

    util.assert_equal(
        compile.CompileAndPrint(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t pB0 = B0_pos[0];
        while ((pB0 < B0_pos[1])) {
          size_t i = B0_crd[pB0];
          size_t pB1 = B1_pos[pB0];
          while ((pB1 < B1_pos[(pB0 + 1)])) {
            size_t j = B1_crd[pB1];
            A0_crd[pA0] = i;
            A1_crd[pA1] = j;
            A_values[pA1] = B_values[pB1];
            pA1 += 1;
            pB1 += 1;
          }
          pB0 += 1;
        }""",
    )


def test_assign_2d_dd():
    A = cin.TensorVar("A", fmt=["d", "d"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["d", "d"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.DENSE)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.DENSE)

    util.assert_equal(
        compile.CompileAndPrint(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t B0 = 0;
        while ((B0 < 8)) {
          size_t i = B0;
          size_t B1 = 0;
          while ((B1 < 10)) {
            size_t j = B1;
            A_values[((i * 10) + j)] = B_values[((i * 10) + j)];
            B1 += 1;
          }
          B0 += 1;
        }""",
    )


def test_assign_2d_sd():
    A = cin.TensorVar("A", fmt=["s", "d"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["s", "d"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.DENSE)

    util.assert_equal(
        compile.CompileAndPrint(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
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


def test_assign_2d_ds():
    A = cin.TensorVar("A", fmt=["d", "s"], shape=[8, 10])
    B = cin.TensorVar("B", fmt=["d", "s"], shape=[8, 10])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.DENSE)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.COMPRESSED)
    util.assert_equal(
        compile.CompileAndPrint(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t B0 = 0;
        while ((B0 < 8)) {
          size_t i = B0;
          size_t pB1 = B1_pos[B0];
          while ((pB1 < B1_pos[(B0 + 1)])) {
            size_t j = B1_crd[pB1];
            A1_crd[pA1] = j;
            A_values[pA1] = B_values[pB1];
            pA1 += 1;
            pB1 += 1;
          }
          B0 += 1;
        }""",
    )


def test_union_1d_s():
    A = cin.TensorVar("A", fmt=["s"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    C = cin.TensorVar("C", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] + C[i]

    util.assert_equal(
        compile.CompileAndPrint(
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
                    cin.IndexSeq(i, C, size=8, index=0, format=LevelType.COMPRESSED),
                ),
            )
        ),
        """
        size_t pB0 = B0_pos[0];
        size_t pC0 = C0_pos[0];
        while (((pB0 < B0_pos[1]) && (pC0 < C0_pos[1]))) {
          size_t i = std::min<size_t>(B0_crd[pB0], C0_crd[pC0]);
          if (((i == B0_crd[pB0]) && (i == C0_crd[pC0]))) {
            A0_crd[pA0] = i;
            A_values[pA0] = (B_values[pB0] + C_values[pC0]);
            pA0 += 1;
          } else if ((i == B0_crd[pB0])) {
            A0_crd[pA0] = i;
            A_values[pA0] = B_values[pB0];
            pA0 += 1;
          } else if ((i == C0_crd[pC0])) {
            A0_crd[pA0] = i;
            A_values[pA0] = C_values[pC0];
            pA0 += 1;
          }
          pB0 += (i == B0_crd[pB0]);
          pC0 += (i == C0_crd[pC0]);
        }
        while ((pB0 < B0_pos[1])) {
          size_t i = B0_crd[pB0];
          A0_crd[pA0] = i;
          A_values[pA0] = B_values[pB0];
          pA0 += 1;
          pB0 += 1;
        }
        while ((pC0 < C0_pos[1])) {
          size_t i = C0_crd[pC0];
          A0_crd[pA0] = i;
          A_values[pA0] = C_values[pC0];
          pA0 += 1;
          pC0 += 1;
        }""",
    )


def test_union_1d_d():
    A = cin.TensorVar("A", fmt=["d"], shape=[8])
    B = cin.TensorVar("B", fmt=["d"], shape=[8])
    C = cin.TensorVar("C", fmt=["d"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] + C[i]

    util.assert_equal(
        compile.CompileAndPrint(
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
                    ),
                    cin.IndexSeq(i, C, size=8, index=0, format=LevelType.DENSE),
                ),
            )
        ),
        """
        size_t B0 = 0;
        while ((B0 < 8)) {
          size_t i = B0;
          A_values[i] = (B_values[i] + C_values[i]);
          B0 += 1;
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
        compile.CompileAndPrint(
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
        ),
        """
        size_t pB0 = B0_pos[0];
        size_t pC0 = C0_pos[0];
        while (((pB0 < B0_pos[1]) && (pC0 < C0_pos[1]))) {
          size_t i = std::min<size_t>(B0_crd[pB0], C0_crd[pC0]);
          if (((i == B0_crd[pB0]) && (i == C0_crd[pC0]))) {
            A0_crd[pA0] = i;
            A_values[pA0] = (B_values[pB0] * C_values[pC0]);
            pA0 += 1;
          }
          pB0 += (i == B0_crd[pB0]);
          pC0 += (i == C0_crd[pC0]);
        }""",
    )


def test_intersection_1d_d():
    A = cin.TensorVar("A", fmt=["d"], shape=[8])
    B = cin.TensorVar("B", fmt=["d"], shape=[8])
    C = cin.TensorVar("C", fmt=["d"], shape=[8])
    i = cin.IndexVar("i")
    A[i] = B[i] * C[i]

    util.assert_equal(
        compile.CompileAndPrint(
            cin.ForAll(
                i,
                A._assignment,
                cin.IntersectionSeq(
                    cin.IndexSeq(
                        i,
                        B,
                        size=8,
                        index=0,
                        format=LevelType.DENSE,
                    ),
                    cin.IndexSeq(
                        i,
                        C,
                        size=8,
                        index=0,
                        format=LevelType.DENSE,
                    ),
                ),
            )
        ),
        """
        size_t B0 = 0;
        while ((B0 < 8)) {
          size_t i = B0;
          A_values[i] = (B_values[i] * C_values[i]);
          B0 += 1;
        }""",
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
        compile.CompileAndPrint(
            cin.ForAll(k, c._assignment, cin.UnionSeq(cin.ProductSeq(Ai, Aj), bk))
        ),
        """
        size_t A0 = 0;
        size_t pA1 = A1_pos[A0];
        while (((A0 < 8) && (pA1 >= A1_pos[(A0 + 1)]))) {
          A0 += 1;
          pA1 = A1_pos[A0];
        }
        size_t pb0 = b0_pos[0];
        while ((((A0 < 8) && (pA1 < A1_pos[(A0 + 1)])) && (pb0 < b0_pos[1]))) {
          size_t k = std::min<size_t>(((A0 * 8) + A1_crd[pA1]), b0_crd[pb0]);
          if (((((k / 8) == A0) && ((k % 8) == A1_crd[pA1])) && (k == b0_crd[pb0]))) {
            c_values[k] = (A_values[pA1] + b_values[pb0]);
          } else if ((((k / 8) == A0) && ((k % 8) == A1_crd[pA1]))) {
            c_values[k] = A_values[pA1];
          } else if ((k == b0_crd[pb0])) {
            c_values[k] = b_values[pb0];
          }
          if ((k == ((A0 * 8) + A1_crd[pA1]))) {
            pA1 += 1;
            while (((A0 < 8) && (pA1 >= A1_pos[(A0 + 1)]))) {
              A0 += 1;
              pA1 = A1_pos[A0];
            }
          }
          pb0 += (k == b0_crd[pb0]);
        }
        while (((A0 < 8) && (pA1 < A1_pos[(A0 + 1)]))) {
          size_t k = ((A0 * 8) + A1_crd[pA1]);
          c_values[k] = A_values[pA1];
          pA1 += 1;
          while (((A0 < 8) && (pA1 >= A1_pos[(A0 + 1)]))) {
            A0 += 1;
            pA1 = A1_pos[A0];
          }
        }
        while ((pb0 < b0_pos[1])) {
          size_t k = b0_crd[pb0];
          c_values[k] = b_values[pb0];
          pb0 += 1;
        }""",
    )


def test_project():
    A = cin.TensorVar("A", fmt=["s", "s"], shape=[8, 8])
    b = cin.TensorVar("b", fmt=["s"], shape=[64])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    k = cin.IndexVar("k")

    kb = cin.IndexSeq(k, b, size=64, index=0, format=LevelType.COMPRESSED)
    proj0 = cin.ProjectSeq(kb, k=0, I=i, J=j)
    proj1 = cin.ProjectSeq(kb, k=1, I=i, J=j)
    A[i, j] = b[k]

    util.assert_equal(
        compile.CompileAndPrint(
            cin.ForAll(i, cin.ForAll(j, A._assignment, proj1), proj0)
        ),
        """
        size_t pb0 = b0_pos[0];        
        while ((pb0 < b0_pos[1])) {
          size_t i = (b0_crd[pb0] / j);
          size_t proj_1 = (b0_crd[pb0] / j);
          while ((proj_1 == (b0_crd[pb0] / j))) {
            size_t j = (b0_crd[pb0] % j);
            A0_crd[pA0] = i;
            A1_crd[pA1] = j;
            A_values[pA1] = b_values[pb0];
            pA1 += 1;
            pb0 += 1;
          }
          while (((pb0 < b0_pos[1]) && (proj_1 == (b0_crd[pb0] / j)))) {
            pb0 += 1;
          }
        }""",
    )


def test_scalar_workspace_dd():
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    k = cin.IndexVar("k")

    A = cin.TensorVar("A", fmt="dd", shape=[8, 8])
    B = cin.TensorVar("B", fmt="dd", shape=[8, 8])
    C = cin.TensorVar("C", fmt="dd", shape=[8, 8])
    w = cin.Workspace(name="w", dim=0, shape=[])

    Ai = cin.IndexSeq(i, A, size=8, index=0, format=LevelType.DENSE)
    Ak = cin.IndexSeq(k, A, size=8, index=1, format=LevelType.DENSE)
    Bj = cin.IndexSeq(j, B, size=8, index=1, format=LevelType.DENSE)
    Bk = cin.IndexSeq(k, B, size=8, index=0, format=LevelType.DENSE)

    util.assert_equal(
        compile.CompileAndPrint(
            cin.ForAll(
                i,
                cin.ForAll(
                    j,
                    cin.Where(
                        workspace=w,
                        producer=cin.ForAll(
                            k,
                            cin.TensorAssign(
                                w.get_default_access(),
                                A[i, k] * B[k, j],
                                op=cin.Operation.ADD,
                            ),
                            cin.IntersectionSeq(Ak, Bk),
                        ),
                        consumer=cin.TensorAssign(C[i, j], w.get_default_access()),
                    ),
                    cin.IntersectionSeq(Bj, cin.Universe(j, 8)),
                ),
                cin.IntersectionSeq(Ai, cin.Universe(i, 8)),
            )
        ),
        """
        size_t A0 = 0;
        while ((A0 < 8)) {
          size_t i = A0;
          size_t B1 = 0;
          while ((B1 < 8)) {
            size_t j = B1;
            float w = 0;
            size_t A1 = 0;
            while ((A1 < 8)) {
              size_t k = A1;
              w += (A_values[((i * 8) + k)] * B_values[((k * 8) + j)]);
              A1 += 1;
            }
            C_values[((i * 8) + j)] = w;
            B1 += 1;
          }
          A0 += 1;
        }""",
    )


def test_scalar_workspace_sd():
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    k = cin.IndexVar("k")

    A = cin.TensorVar("A", fmt="ds", shape=[8, 8])
    B = cin.TensorVar("B", fmt="ds", shape=[8, 8])
    C = cin.TensorVar("C", fmt="dd", shape=[8, 8])
    w = cin.Workspace(name="w", dim=0, shape=[])

    Ai = cin.IndexSeq(i, A, size=8, index=0, format=LevelType.DENSE)
    Ak = cin.IndexSeq(k, A, size=8, index=1, format=LevelType.COMPRESSED)
    Bj = cin.IndexSeq(j, B, size=8, index=0, format=LevelType.DENSE)
    Bk = cin.IndexSeq(k, B, size=8, index=1, format=LevelType.COMPRESSED)

    util.assert_equal(
        compile.CompileAndPrint(
            cin.ForAll(
                i,
                cin.ForAll(
                    j,
                    cin.Where(
                        workspace=w,
                        producer=cin.ForAll(
                            k,
                            cin.TensorAssign(
                                w.get_default_access(),
                                A[i, k] * B[j, k],
                                op=cin.Operation.ADD,
                            ),
                            cin.IntersectionSeq(Ak, Bk),
                        ),
                        consumer=cin.TensorAssign(C[i, j], w.get_default_access()),
                    ),
                    cin.IntersectionSeq(Bj, cin.Universe(j, 8)),
                ),
                cin.IntersectionSeq(Ai, cin.Universe(i, 8)),
            )
        ),
        """
        size_t A0 = 0;
        while ((A0 < 8)) {
          size_t i = A0;
          size_t B0 = 0;
          while ((B0 < 8)) {
            size_t j = B0;
            float w = 0;
            size_t pA1 = A1_pos[A0];
            size_t pB1 = B1_pos[B0];
            while (((pA1 < A1_pos[(A0 + 1)]) && (pB1 < B1_pos[(B0 + 1)]))) {
              size_t k = std::min<size_t>(A1_crd[pA1], B1_crd[pB1]);
              if (((k == A1_crd[pA1]) && (k == B1_crd[pB1]))) {
                w += (A_values[pA1] * B_values[pB1]);
              }
              pA1 += (k == A1_crd[pA1]);
              pB1 += (k == B1_crd[pB1]);
            }
            C_values[((i * 8) + j)] = w;
            B0 += 1;
          }
          A0 += 1;
        }""",
    )


def test_vector_workspace_dd():
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")

    a = cin.TensorVar("A", fmt="d", shape=[8])
    b = cin.TensorVar("B", fmt="d", shape=[8])
    c = cin.TensorVar("C", fmt="d", shape=[8])
    w = cin.Workspace(name="w", dim=1, dtype=torch.float32, dense=True, shape=[8])

    ai = cin.IndexSeq(i, a, size=8, index=0, format=LevelType.DENSE)
    bi = cin.IndexSeq(i, b, size=8, index=0, format=LevelType.DENSE)
    wj = cin.IndexSeq(j, w, size=8, index=0, format=LevelType.DENSE)

    util.assert_equal(
        compile.CompileAndPrint(
            cin.Where(
                workspace=w,
                producer=cin.ForAll(
                    i, cin.TensorAssign(w[i], a[i] + b[i]), cin.IntersectionSeq(ai, bi)
                ),
                consumer=cin.ForAll(j, cin.TensorAssign(c[j], w[j]), wj),
            )
        ),
        """
        size_t A0 = 0;
        while ((A0 < 8)) {
          size_t i = A0;
          w_values[i] = (A_values[i] + B_values[i]);
          A0 += 1;
        }
        size_t w0 = 0;
        while ((w0 < 8)) {
          size_t j = w0;
          C_values[j] = w_values[j];
          w_values[j] = 0;
          w0 += 1;
        }""",
    )
