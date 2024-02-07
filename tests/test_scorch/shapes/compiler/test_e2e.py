import torch

from scorch.compiler import cin
from scorch.compiler.shapes import cfir, codegen, cpp, compile
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import tests.utility as util

# Tests CIN -> CFIR -> CPP lowering phase(s).


def test_function():
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
        codegen.PrettyPrint(
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
            size_t i_B = 0;
            while ((i_B < 2)) {
              size_t i = i_B;
              size_t j_B = 0;
              while ((j_B < 2)) {
                size_t j = j_B;
                A_values[((i * 2) + j)] = B_values[((i * 2) + j)];
                j_B += 1;
              }
              i_B += 1;
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
        size_t i_B = 0;
        while ((i_B < 8)) {
          size_t i = i_B;
          A_values[i] = B_values[i];
          i_B += 1;
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
      size_t ip_B = B.pos[0][0];
      while ((ip_B < B.pos[0][1])) {
        size_t i = B.crd[0][ip_B];
        A.crd[0][ip_A] = i;
        A_values[ip_A] = B_values[ip_B];
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

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.COMPRESSED)

    util.assert_equal(
        compile.CompileAndPrint(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t ip_B = B.pos[0][0];
        while ((ip_B < B.pos[0][1])) {
          size_t i = B.crd[0][ip_B];
          size_t jp_B = B.pos[1][ip_B];
          while ((jp_B < B.pos[1][(ip_B + 1)])) {
            size_t j = B.crd[1][jp_B];
            A.crd[0][ip_A] = i;
            A.crd[1][jp_A] = j;
            A_values[jp_A] = B_values[jp_B];
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

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.DENSE)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.DENSE)

    util.assert_equal(
        compile.CompileAndPrint(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t i_B = 0;
        while ((i_B < 8)) {
          size_t i = i_B;
          size_t j_B = 0;
          while ((j_B < 10)) {
            size_t j = j_B;
            A_values[((i * 10) + j)] = B_values[((i * 10) + j)];
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

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.COMPRESSED)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.DENSE)

    util.assert_equal(
        compile.CompileAndPrint(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t ip_B = B.pos[0][0];
        while ((ip_B < B.pos[0][1])) {
          size_t i = B.crd[0][ip_B];
          size_t j_B = 0;
          while ((j_B < 10)) {
            size_t j = j_B;
            A.crd[0][ip_A] = i;
            A_values[((ip_A * 10) + j)] = B_values[((ip_B * 10) + j)];
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

    Bi = cin.IndexSeq(i, B, size=8, index=0, format=LevelType.DENSE)
    Bj = cin.IndexSeq(j, B, size=10, index=1, format=LevelType.COMPRESSED)

    util.assert_equal(
        compile.CompileAndPrint(cin.ForAll(i, cin.ForAll(j, A._assignment, Bj), Bi)),
        """
        size_t i_B = 0;
        while ((i_B < 8)) {
          size_t i = i_B;
          size_t jp_B = B.pos[1][ip_B];
          while ((jp_B < B.pos[1][(ip_B + 1)])) {
            size_t j = B.crd[1][jp_B];
            A.crd[1][jp_A] = j;
            A_values[jp_A] = B_values[jp_B];
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
        size_t ip_B = B.pos[0][0];
        size_t ip_C = C.pos[0][0];
        while (((ip_B < B.pos[0][1]) && (ip_C < C.pos[0][1]))) {
          size_t i = min(B.crd[0][ip_B], C.crd[0][ip_C]);
          if (((i == ip_B) && (i == ip_C))) {
            A.crd[0][ip_A] = i;
            A_values[ip_A] = (B_values[ip_B] + C_values[ip_C]);
            ip_A += 1;
          } else if ((i == ip_B)) {
            A.crd[0][ip_A] = i;
            A_values[ip_A] = B_values[ip_B];
            ip_A += 1;
          } else if ((i == ip_C)) {
            A.crd[0][ip_A] = i;
            A_values[ip_A] = C_values[ip_C];
            ip_A += 1;
          }
          ip_B += (i == B.crd[0][ip_B]);
          ip_C += (i == C.crd[0][ip_C]);
        }
        while ((ip_B < B.pos[0][1])) {
          size_t i = B.crd[0][ip_B];
          A.crd[0][ip_A] = i;
          A_values[ip_A] = B_values[ip_B];
          ip_A += 1;
          ip_B += 1;
        }
        while ((ip_C < C.pos[0][1])) {
          size_t i = C.crd[0][ip_C];
          A.crd[0][ip_A] = i;
          A_values[ip_A] = C_values[ip_C];
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
        size_t i_B = 0;
        while ((i_B < 8)) {
          size_t i = i_B;
          A_values[i] = (B_values[i] + C_values[i]);
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
        size_t ip_B = B.pos[0][0];
        size_t ip_C = C.pos[0][0];
        while (((ip_B < B.pos[0][1]) && (ip_C < C.pos[0][1]))) {
          size_t i = min(B.crd[0][ip_B], C.crd[0][ip_C]);
          if (((i == ip_B) && (i == ip_C))) {
            A.crd[0][ip_A] = i;
            A_values[ip_A] = (B_values[ip_B] * C_values[ip_C]);
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
      size_t i_B = 0;
      while ((i_B < 8)) {
        size_t i = i_B;
        A_values[i] = (B_values[i] * C_values[i]);
        i_B += 1;
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
            cin.ForAll(k, c._assignment, cin.UnionSeq(cin.Product(Ai, Aj), bk))
        ),
        """
    size_t i_A = 0;
    size_t jp_A = A.pos[1][ip_A];
    while (((i_A < 8) && (!(jp_A < A.pos[1][(ip_A + 1)])))) {
      i_A += 1;
      jp_A = A.pos[1][ip_A];
    }
    size_t kp_b = b.pos[0][0];
    while ((((i_A < 8) && (jp_A < A.pos[1][(ip_A + 1)])) && (kp_b < b.pos[0][1]))) {
      size_t k = min(((i_A * 8) + A.crd[1][jp_A]), b.crd[0][kp_b]);
      if (((((k / 8) == i_A) && ((k % 8) == jp_A)) && (k == kp_b))) {
        c_values[k] = (A_values[jp_A] + b_values[kp_b]);
      } else if ((((k / 8) == i_A) && ((k % 8) == jp_A))) {
        c_values[k] = A_values[jp_A];
      } else if ((k == kp_b)) {
        c_values[k] = b_values[kp_b];
      }
      if ((k == ((i_A * 8) + A.crd[1][jp_A]))) {
        jp_A += 1;
        while (((i_A < 8) && (!(jp_A < A.pos[1][(ip_A + 1)])))) {
          i_A += 1;
          jp_A = A.pos[1][ip_A];
        }
      }
      kp_b += (k == b.crd[0][kp_b]);
    }
    while (((i_A < 8) && (jp_A < A.pos[1][(ip_A + 1)]))) {
      size_t k = ((i_A * 8) + A.crd[1][jp_A]);
      c_values[k] = A_values[jp_A];
      jp_A += 1;
      while (((i_A < 8) && (!(jp_A < A.pos[1][(ip_A + 1)])))) {
        i_A += 1;
        jp_A = A.pos[1][ip_A];
      }
    }
    while ((kp_b < b.pos[0][1])) {
      size_t k = b.crd[0][kp_b];
      c_values[k] = b_values[kp_b];
      kp_b += 1;
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
    size_t i_A = 0;
    while ((i_A < 8)) {
      size_t i = i_A;
      size_t j_B = 0;
      while ((j_B < 8)) {
        size_t j = j_B;
        float w = 0;
        size_t k_A = 0;
        while ((k_A < 8)) {
          size_t k = k_A;
          w += (A_values[((i * 8) + k)] * B_values[((k * 8) + j)]);
          k_A += 1;
        }
        C_values[((i * 8) + j)] = w;
        j_B += 1;
      }
      i_A += 1;
    }""",
    )


def test_scalar_workspace_csc():
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
        size_t i_A = 0;
        while ((i_A < 8)) {
          size_t i = i_A;
          size_t j_B = 0;
          while ((j_B < 8)) {
            size_t j = j_B;
            float w = 0;
            size_t kp_A = A.pos[1][ip_A];
            size_t kp_B = B.pos[1][jp_B];
            while (((kp_A < A.pos[1][(ip_A + 1)]) && (kp_B < B.pos[1][(jp_B + 1)]))) {
              size_t k = min(A.crd[1][kp_A], B.crd[1][kp_B]);
              if (((k == kp_A) && (k == kp_B))) {
                w += (A_values[kp_A] * B_values[kp_B]);
              }
              kp_A += (k == A.crd[1][kp_A]);
              kp_B += (k == B.crd[1][kp_B]);
            }
            C_values[((i * 8) + j)] = w;
            j_B += 1;
          }
          i_A += 1;
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
        size_t i_A = 0;
        while ((i_A < 8)) {
          size_t i = i_A;
          w_values[i] = (A_values[i] + B_values[i]);
          i_A += 1;
        }
        size_t j_w = 0;
        while ((j_w < 8)) {
          size_t j = j_w;
          C_values[j] = w_values[j];
          w_values[j] = 0;
          j_w += 1;
        }""",
    )
