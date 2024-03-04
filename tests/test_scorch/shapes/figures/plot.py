import matplotlib.pyplot as plt
import torch
from typing import Callable, Tuple, Optional
from enum import StrEnum
import random
from dataclasses import dataclass

# Input matrix will have dimension [N, N].
N: int = 100
# The number of experiments. This will be averaged.
E: int = 10

# Density of rows / columns...?


class Format(StrEnum):
    """Format of the input matrix"""
    D = "d"  # dense (torch.rand)
    SP = "sp"  # "with probability p, A[i,j] is 0."
    CSR = "csr"  # S + "with probability p, A[:, j] is 0"
    CSC = "csc"  # S + "with probability p, A[i, :] is 0"
    DS = "dbl-sp"  # SR + SC
    USR = "uni-row"  # CSR, but guarantees each row has same # of zeros.
    USC = "uni-col"  # CSC, but guarantees each column has same # of zeros.


def dmatrix() -> torch.Tensor:
    """Returns a dense matrix with random values."""
    return torch.rand((N, N)).float()


def spmatrix(sparsity: int) -> torch.Tensor:
    """Returns a sparse matrix with random values and the provided sparsity."""
    return dmatrix() * (torch.rand((N, N)) > sparsity)


# sparsities: list[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
SPARSITIES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def count_rows(t: torch.Tensor) -> int:
    nonzeros: Tuple[torch.Tensor, ...] = torch.nonzero(t, as_tuple=True)
    if len(nonzeros) < 1:
        return 0
    return len(torch.unique(nonzeros[0]))


def count_cols(t: torch.Tensor) -> int:
    nonzeros: Tuple[torch.Tensor, ...] = torch.nonzero(t, as_tuple=True)
    if len(nonzeros) < 2:
        return 0
    return len(torch.unique(nonzeros[1]))


def count_values(t: torch.Tensor) -> int:
    return len(torch.nonzero(t))


def matrix(sparsity: float, format: Format):
    match format:
        case Format.D:
            return dmatrix()
        case Format.SP:
            return spmatrix(sparsity)
        case Format.CSC:
            m: torch.Tensor = spmatrix(sparsity)
            # Randomly select [0, sparsity * N) columns, and zero them out.
            for _ in range(0, int(sparsity * N)):
                m[random.randrange(0, N), :] = 0
            return m
        case Format.CSR:
            # Randomly select [0, sparsity * N) columns, and zero them out.
            return matrix(sparsity, Format.CSC).transpose(1, 0)
        case Format.DS:
            # Randomly select [0, sparsity * N) rows AND columns, and zero them out.
            m = matrix(sparsity, Format.CSC)
            for _ in range(0, int(sparsity * N)):
                m[:, random.randrange(0, N)] = 0
            return m
        case Format.USC:
            k: int = N - int(sparsity * N)  # The density of non-zero values.
            csr: Optional[torch.Tensor] = None
            for _ in range(0, N):
                tensor: torch.Tensor = torch.zeros(size=(N,))
                if sparsity < random.uniform(0, 1):
                    indices: torch.Tensor = torch.randperm(tensor.numel())[:k]
                    tensor[indices] = random.uniform(1, 2)  # (non-zero value)
                csr = tensor if csr is None else torch.vstack((csr, tensor))
            return csr
        case Format.USR:
            return matrix(sparsity, Format.USC).transpose(1, 0)
        case _:
            raise NotImplementedError(format)


def average(l: list[int], E: int) -> None:
    i: list[int] = []
    for _ in range(0, E):
        i.append(l.pop())
    a = sum(i) / len(i)
    assert a >= 0.0
    l.append(a)


@dataclass
class Data:
    nnzA: list[int]
    nnzB: list[int]
    nnzC: list[int]
    A0: list[int]
    A1: list[int]
    B0: list[int]
    B1: list[int]
    C0: list[int]
    C1: list[int]


def sparsity(f: Callable, input1: Format, input2: Format) -> Data:
    A0: list[int] = []
    A1: list[int] = []
    nnzA: list[int] = []
    B0: list[int] = []
    B1: list[int] = []
    nnzB: list[int] = []
    C0: list[int] = []
    C1: list[int] = []
    nnzC: list[int] = []

    for sparsity in SPARSITIES:
        for _ in range(0, E):
            a: torch.Tensor = matrix(sparsity, input1)
            b: torch.Tensor = matrix(sparsity, input2)
            c: torch.Tensor = f(a, b)
            A0_: int = count_rows(a)
            A1_: int = count_cols(a)
            va: int = count_values(a)
            B0_: int = count_rows(b)
            B1_: int = count_cols(b)
            vb: int = count_values(b)
            C0_: int = count_rows(c)
            C1_: int = count_cols(c)
            vc: int = count_values(c)
            nnzA.append(va)
            nnzB.append(vb)
            nnzC.append(vc)
            A0.append(A0_)
            A1.append(A1_)
            B0.append(B0_)
            B1.append(B1_)
            C0.append(C0_)
            C1.append(C1_)
        average(A0, E)
        average(A1, E)
        average(B0, E)
        average(B1, E)
        average(C0, E)
        average(C1, E)
        average(nnzA, E)
        average(nnzB, E)
        average(nnzC, E)

    return Data(
        nnzA=nnzA,
        nnzB=nnzB,
        nnzC=nnzC,
        A0=A0,
        A1=A1,
        B0=B0,
        B1=B1,
        C0=C0,
        C1=C1,
    )


def _plot_nnz(f: Callable, axis, input1: Format, input2: Format):
    d: Data = sparsity(f, input1, input2)
    axis.set_title(f'({input1}, {input2})')
    axis.plot(SPARSITIES, d.nnzA, label='A.nnz')
    axis.plot(SPARSITIES, d.nnzB, label='B.nnz')
    axis.plot(SPARSITIES, d.nnzC, label='C.nnz')


def _plot_density(f: Callable, axis, input1: Format, input2: Format):
    d: Data = sparsity(f, input1, input2)
    axis.set_title(f'({input1}, {input2})')
    axis.plot(SPARSITIES, d.C0, label='C.0')
    axis.plot(SPARSITIES, d.C1, label='C.1')


def plot_operations(input1: Format, input2: Format):
    figure, axes = plt.subplots(2, 2)
    figure.suptitle(f'C = f(A,B) - (A={input1},B={input2})')
    _plot_nnz(torch.add, axes[0, 0], input1, input2)
    _plot_nnz(torch.matmul, axes[0, 1], input1, input2)
    _plot_nnz(torch.mul, axes[1, 0], input1, input2)
    _plot_nnz(torch.fmax, axes[1, 1], input1, input2)

    axes[0, 0].set(xticks=[])
    axes[0, 1].set(xticks=[])

    plt.legend()
    plt.show()


def plot_nnz(f: Callable, formats: list[Format] = list(Format), block=False):
    n: int = len(formats)
    formats: list[Tuple[Format, int]] = [(format, i) for (format, i) in zip(formats, range(0, n))]
    figure, axes = plt.subplots(n, n)
    figure.suptitle(f"NNZ: C = {f.__name__}(A, B)")
    for i1, i in formats:
        for i2, j in formats:
            _plot_nnz(f, axes[i, j], i1, i2)

    for axis in axes.flat:
        axis.label_outer()
    plt.legend()
    plt.show(block=block)


def plot_density(f: Callable, formats: list[Format] = list(Format), block=False):
    n: int = len(formats)
    formats: list[Tuple[Format, int]] = [(format, i) for (format, i) in zip(formats, range(0, n))]
    figure, axes = plt.subplots(n, n)
    figure.suptitle(f"Density: C = {f.__name__}(A, B)")
    for i1, i in formats:
        for i2, j in formats:
            _plot_density(f, axes[i, j], i1, i2)

    for axis in axes.flat:
        axis.label_outer()
    plt.legend()
    plt.show(block=block)


if __name__ == "__main__":
    plot_density(torch.matmul, formats=[Format.D, Format.DS])
    plot_nnz(torch.matmul, formats=[Format.D, Format.DS], block=True)
