from taco_torch.cin import TensorAssign, TensorAccess, IndexVar, TensorVar, ForAll
from taco_torch.compiler import CINLowerer, LLIRLowerer


def test_elementwise_vector_mul_codegen():
    # elementwise vector multiplication code generation
    # a[i] = b[i] * c[i]

    i = IndexVar("i")

    a = TensorVar("a", fmt="sparse")
    # a = TensorVar("a", fmt="dense")
    # n = TensorVar("b", fmt="sparse")
    b = TensorVar("b", fmt="dense")
    c = TensorVar("c", fmt="sparse")

    a[i] = b[i] * c[i]

    cin_stmt = ForAll(i, a._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_iteration_lattice():
    # iteration lattice generation
    # A[i] = (B[i] + C[i]) * (D[i] + E[i])
    i = IndexVar("i")
    A = TensorVar("A", fmt="sparse")
    B = TensorVar("B", fmt="sparse")
    C = TensorVar("C", fmt="sparse")
    D = TensorVar("D", fmt="sparse")
    E = TensorVar("E", fmt="sparse")
    A[i] = (B[i] + C[i]) * (D[i] + E[i])

    cin_stmt = ForAll(i, A._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_iteration_lattice_2():
    # iteration lattice generation
    # A[i] = (B[i] + C[i]) * (D[i] + E[i])
    i = IndexVar("i")
    A = TensorVar("A", fmt="sparse")
    B = TensorVar("B", fmt="dense")
    C = TensorVar("C", fmt="dense")
    D = TensorVar("D", fmt="sparse")
    E = TensorVar("E", fmt="sparse")
    A[i] = (B[i] + C[i]) * (D[i] + E[i])

    cin_stmt = ForAll(i, A._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elementwise_matrix_mul_codegen():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elementwise_matrix_add_mul_codegen():
    # elementwise matrix multiplication code generation
    # A(i,j)=(B(i,j)+C(i,j))*(D(i,j)+E(i,j))

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])
    D = TensorVar("D", fmt=["sparse", "sparse"])
    E = TensorVar("E", fmt=["sparse", "sparse"])

    A[i, j] = (B[i, j] + C[i, j]) * (D[i, j] + E[i, j])

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    print("\nCIN statement:")
    print(cin_stmt)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))
