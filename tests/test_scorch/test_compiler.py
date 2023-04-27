from src.scorch.compiler.cin import IndexVar, TensorVar, ForAll
from src.scorch.compiler.cin_lowerer import CINLowerer
from src.scorch.compiler.codegen import LLIRLowerer


def test_elemwise_mul_1d_sss():
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


def test_elemwise_mul_1d_dss():
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


def test_elemwise_mul_1d_sds():
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


def test_elemwise_mul_add_1d_sssd():
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


def test_elemwise_add_1d_sss():
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


def test_elemwise_add_1d_dss():
    # elementwise vector addition code generation
    # a[i] = b[i] + c[i]
    # taco "a(i) = b(i)+c(i)" -f=a:d -f=b:s -f=c:s -print-evaluate
    i = IndexVar("i")

    a = TensorVar("a", fmt="dense")
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


def test_elemwise_add_1d_sds():
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


def test_elemwise_mul_3d_tensor_dss_sss_sss():
    # A[i, j, k] = B[i, j, k] * C[i, j, k]
    # taco "A(i,j,k) = B(i,j,k) * C(i,j,k)" -f=A:dss -f=B:sss -f=C:sss -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse"])

    A[i, j, k] = B[i, j, k] * C[i, j, k]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, A._assignment)))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_convert_4d_ssss_oooo():
    # taco "A(i,j,k,l) = B(i,j,k,l)" -f=A:ssss:0,1,2,3 -f=B:uccq:0,1,2,3 -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["sparse", "sparse", "sparse", "sparse"])
    B = TensorVar("B", fmt=["coord", "coord", "coord", "coord"])

    A[i, j, k, l] = B[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lowerer = CINLowerer(filter_zeros=True)

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_4d_oooo_ssss_ssss():
    # taco "A(i,j,k,l)=B(i,j,k,l)*C(i,j,k,l)" -f=A:uccq:0,1,2,3 -f=B:ssss:0,1,2,3 -f=C:ssss:0,1,2,3 -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["coord", "coord", "coord", "coord"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_4d_ssss_ssss_ssss():
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssss -f=B:ssss -f=C:ssss -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["sparse", "sparse", "sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_4d_ssss_oooo_oooo():
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssss -f=B:oooo -f=C:oooo -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["sparse", "sparse", "sparse", "sparse"])
    B = TensorVar("B", fmt=["coord", "coord", "coord", "coord"])
    C = TensorVar("C", fmt=["coord", "coord", "coord", "coord"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_4d_ssdd_ssss_ssss():
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssdd -f=B:ssss -f=C:ssss -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["sparse", "sparse", "dense", "dense"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_4d_dddd_ssss_ssss():
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:dddd -f=B:ssss -f=C:ssss -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["dense", "dense", "dense", "dense"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_4d_ddss_dddd_ssss():
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ddss -f=B:dddd -f=C:ssss -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["dense", "dense", "sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense", "dense", "dense"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_4d_ddss_ssss_ssss():
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ddss -f=B:ssss -f=C:ssss -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["dense", "dense", "sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_2d_ss_dd_ds():
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


def test_elemwise_mul_2d_ss_ss_ss():
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


def test_elemwise_mul_2d_oo_oo_oo():
    # taco "A(i,j)=B(i,j)*C(i,j)" -f=A:uq:0,1 -f=B:uq:0,1 -f=C:uq:0,1 -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["coordinate", "coordinate"])
    B = TensorVar("B", fmt=["coordinate", "coordinate"])
    C = TensorVar("C", fmt=["coordinate", "coordinate"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_2d_ds_dd_ds():
    # taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ds -f=B:dd -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_2d_dd_dd_ds():
    # taco "A(i,j) = B(i,j) * C(i,j)" -f=A:dd -f=B:dd -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "dense"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_elemwise_mul_2d_ds_ss_ss():
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


def test_elemwise_matrix_add_ds_ss_ss():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]
    # taco "A(i,j) = B(i,j) + C(i,j)" -f=A:ds -f=B:ss -f=C:ss -print-evaluate

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "sparse"])
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


def test_elemwise_mul_2d_codegen_2():
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


def test_elemwise_mul_2d_diff_order_codegen():
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


def test_spmv_codegen():
    # taco "y(i) = A(i, j) * x(j)" -f=y:d -f=A:ds -f=x:d -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")

    y = TensorVar("y", fmt=["dense"])
    A = TensorVar("A", fmt=["dense", "sparse"])
    x = TensorVar("x", fmt=["dense"])

    y[i] = A[i, j] * x[j]

    cin_stmt = ForAll(i, ForAll(j, y._assignment))

    print("\nCIN statement:")
    print(cin_stmt)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_spmm_codegen():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, k] * C[k, j]

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, A._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_spmm_dd_ds_ds():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:dd -f=B:ds -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "dense"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, k] * C[k, j]

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, A._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_spmm_sd_ds_ds():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:sd -f=B:ds -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["sparse", "dense"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, k] * C[k, j]

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, A._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))


def test_ij_i_j_ss_s_s():
    # taco "A(i, j) = B(i) * C(j)" -f=A:ss -f=B:s -f=C:s -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse"])
    C = TensorVar("C", fmt=["sparse"])

    A[i, j] = B[i] * C[j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    print("\nCIN statement:")
    print(cin_stmt)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    print("\nC++ torch extension code:")
    print(llir_lowerer.lower_llir(lowered_llir))
