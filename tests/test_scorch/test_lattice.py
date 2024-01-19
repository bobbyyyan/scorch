from scorch.compiler.cin import (
    IndexVar,
    TensorVar,
    ForAll,
    TensorAssign,
    TensorAccess,
    Operation,
    Workspace,
    Where,
    TileSizeVar,
    IntersectionSeq,
    IndexSeq,
    UnionSeq,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.iter_lattice import IterationLattice, LatticePoint


def test_mul_seq():
    A = TensorVar("a", fmt=["d"])
    B = TensorVar("b", fmt=["d"])
    C = TensorVar("c", fmt=["d"])
    i = IndexVar("i")

    A[i] = B[i] * C[i]
    assert IterationLattice(
        for_all_stmt=ForAll(
            i, A._assignment, seq=IntersectionSeq(IndexSeq(B, i), IndexSeq(C, i))
        ),
        cin_lowerer=CINLowerer(),
    ).gen_lattice_points() == [
        LatticePoint(dense_tensor_accesses=[TensorAccess(B, i), TensorAccess(C, i)])
    ]


def test_add_seq():
    A = TensorVar("a", fmt=["s", "s"])
    B = TensorVar("b", fmt=["s", "s"])
    C = TensorVar("c", fmt=["s", "s"])
    i = IndexVar("i")
    j = IndexVar("j")
    A[i, j] = B[i, j] + C[i, j]

    cin = ForAll(
        i, ForAll(j, A._assignment, UnionSeq(IndexSeq(B, [i, j]), IndexSeq(C, [i, j])))
    )
    assert IterationLattice(
        cin,
        cin_lowerer=CINLowerer(),
    ).gen_lattice_points() == [
        LatticePoint(
            sparse_tensor_accesses=[TensorAccess(B, [i, j]), TensorAccess(C, [i, j])]
        ),
        LatticePoint(sparse_tensor_accesses=[TensorAccess(B, [i, j])]),
        LatticePoint(sparse_tensor_accesses=[TensorAccess(C, [i, j])]),
    ]
