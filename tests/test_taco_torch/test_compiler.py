from src.taco_torch.compiler.cin import IndexVar, TensorVar, ForAll
from src.taco_torch.compiler.cin_lowerer import CINLowerer
from src.taco_torch.compiler.codegen import LLIRLowerer


def test_elemwise_vector_mul_sss():
    # elementwise vector multiplication code generation
    # a[i] = b[i] * c[i]
    # taco "a(i) = b(i)*c(i)" -f=a:s -f=b:s -f=c:s -print-evaluate
    # Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)&format=a:s:0;b:s:0;c:s:0

    i = IndexVar("i")

    a = TensorVar("a", fmt="sparse")
    b = TensorVar("b", fmt="sparse")
    c = TensorVar("c", fmt="sparse")

    a[i] = b[i] * c[i]

    cin_stmt = ForAll(i, a._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_vector_mul_dss():
    # elementwise vector multiplication code generation
    # a[i] = b[i] * c[i]
    # taco "a(i) = b(i)*c(i)" -f=a:d -f=b:s -f=c:s -print-evaluate

    i = IndexVar("i")

    a = TensorVar("a", fmt="dense")
    b = TensorVar("b", fmt="sparse")
    c = TensorVar("c", fmt="sparse")

    a[i] = b[i] * c[i]

    cin_stmt = ForAll(i, a._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_vector_mul_sds():
    # elementwise vector multiplication code generation
    # a[i] = b[i] * c[i]
    # Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)&format=a:s:0;b:d:0;c:s:0

    i = IndexVar("i")

    a = TensorVar("a", fmt="sparse")
    b = TensorVar("b", fmt="dense")
    c = TensorVar("c", fmt="sparse")

    a[i] = b[i] * c[i]

    cin_stmt = ForAll(i, a._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_vector_mul_add_sssd():
    # elementwise vector multiplication and addition code generation
    # Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)+d(i)&format=a:s:0;b:s:0;c:s:0;d:d:0
    # a[i] = b[i] * c[i] + d[i]
    i = IndexVar("i")

    a = TensorVar("a", fmt="sparse")
    b = TensorVar("b", fmt="sparse")
    c = TensorVar("c", fmt="sparse")
    d = TensorVar("d", fmt="dense")

    a[i] = b[i] * c[i] + d[i]

    cin_stmt = ForAll(i, a._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_vector_add_sss():
    # elementwise vector addition code generation
    # a[i] = b[i] + c[i]
    i = IndexVar("i")

    a = TensorVar("a", fmt="sparse")
    b = TensorVar("b", fmt="sparse")
    c = TensorVar("c", fmt="sparse")
    # c = TensorVar("c", fmt="dense")

    a[i] = b[i] + c[i]

    cin_stmt = ForAll(i, a._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_vector_add_sds():
    # elementwise vector addition code generation
    # a[i] = b[i] + c[i]
    # Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)+c(i)&format=a:s:0;b:d:0;c:s:0
    i = IndexVar("i")

    a = TensorVar("a", fmt="sparse")
    # a = TensorVar("a", fmt="dense")
    # b = TensorVar("b", fmt="sparse")
    b = TensorVar("b", fmt="dense")
    c = TensorVar("c", fmt="sparse")
    # c = TensorVar("c", fmt="dense")

    a[i] = b[i] + c[i]

    cin_stmt = ForAll(i, a._assignment)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_matrix_mul_ss_dd_ds():
    # A[i, j] = B[i, j] * C[i, j]
    # A: sparse, sparse
    # B: dense, dense
    # C: dense, sparse
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_matrix_vector_mul_codegen():
    # matrix vector multiplication code generation
    # A[i, j] = B[i, j] * C[j]
    # Reference: http://tensor-compiler.org/codegen.html?expr=A(i,j)=B(i,j)*c(j)&format=A:ss:0,1;B:ss:0,1;c:s:0

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse"])

    A[i, j] = B[i, j] * C[j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_matrix_mul_ss_ss_ss():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]
    # taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ss -f=B:ss -f=C:ss -print-evaluate

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


def test_elemwise_matrix_mul_ds_ss_ss():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]
    # taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ds -f=B:ss -f=C:ss -print-evaluate

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_matrix_add_ss_ss_ss():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]
    # taco "A(i,j) = B(i,j) + C(i,j)" -f=A:ss -f=B:ss -f=C:ss -print-evaluate

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse"])

    A[i, j] = B[i, j] + C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_matrix_add_sd_ss_ss():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]
    # taco "A(i,j) = B(i,j) + C(i,j)" -f=A:ss -f=B:ss -f=C:ss -print-evaluate

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "dense"])
    B = TensorVar("B", fmt=["sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse"])

    A[i, j] = B[i, j] + C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_matrix_mul_codegen_2():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "dense"])
    C = TensorVar("C", fmt=["sparse", "dense"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_matrix_mul_diff_order_codegen():
    # http://tensor-compiler.org/codegen.html?expr=A(i,j)=B(i,j)*C(j,i)&format=A:ss:0,1;B:dd:0,1;C:dd:0,1
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[j, i]
    # A: sparse, sparse
    # B: dense, dense
    # C: dense, dense

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "dense"])

    A[i, j] = B[i, j] * C[j, i]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_matrix_add_mul_codegen():
    # elementwise matrix multiplication code generation
    # A[i, j] = (B[i, j] + C[i, j]) * (D[i, j] + E[i, j])
    # TACO reference: http://tensor-compiler.org/codegen.html?expr=A(i,j)=(B(i,j)+C(i,j))*(D(i,j)+E(i,j))&format=A:ss:0,1;B:ds:0,1;C:ds:0,1;D:ss:0,1;E:ss:0,1

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
