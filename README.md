# <img src="docs/source/_static/img/scorch-icon-pixel.svg" alt="" width="42" height="42"> Scorch

**A CPU sparse-tensor compiler for PyTorch.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/CGO%202026-paper-blueviolet?style=flat)](https://doi.org/10.1109/CGO68049.2026.11394842)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-blue?style=flat)](https://github.com/bobbyyyan/scorch)

[Installation](#installation) · [Quickstart](#quickstart) ·
[Formats](#format-system) · [Examples](#examples-and-documentation) ·
[Paper](#citation)

Scorch lets you describe a tensor's physical layout, write familiar operations
such as `matmul` or explicit `einsum`, and run specialized C++/OpenMP kernels.
It interoperates with PyTorch tensors while making sparse storage formats a
first-class part of compilation.

Scorch is research software focused on sparse inference and compiler research.

## Why Scorch?

- **Declarative formats.** Describe each tensor mode as dense, compressed, or
  coordinate. CSR is `"ds"`, COO is `"oo"`, and supported layouts can be
  composed without hand-writing an operation-specific kernel.
- **A compiler, not only a kernel library.** Generic operations lower through
  Compiler Index Notation (CIN), scheduling, LLIR, and C++ code generation.
- **PyTorch interoperability.** Build an `STensor` from dense, COO, or CSR
  `torch.Tensor` inputs and convert results back with `.to_torch()`.
- **Fast common paths.** Frequently used sparse matrix operations dispatch to
  prebuilt native kernels; other supported format and operation combinations
  are JIT-compiled and cached.

The [CGO 2026 paper](https://fredrikbk.com/publications/scorch.pdf) reports
1.05–5.80× speedups over PyTorch Sparse on its evaluated end-to-end graph neural
network, sparse autoencoder, and sparse transformer workloads. Performance is
hardware- and workload-dependent.

## Installation

Scorch requires Python 3.9 or newer (3.11 recommended), PyTorch, Ninja, a C++
toolchain, and OpenMP. The supported installation path is currently from source.

### First-time setup

On macOS, install the OpenMP headers before running setup:

```bash
brew install libomp
```

```bash
git clone https://github.com/bobbyyyan/scorch.git
cd scorch
./setup.sh
```

When Conda is available, `setup.sh` creates a fresh Python 3.11 environment
named `scorch` (after backing up an existing environment with that name). It
otherwise creates `venv/`. Activate the environment with the command printed by
the script:

```bash
conda activate scorch
# Without Conda: source venv/bin/activate
```

### Rebuild in an existing environment

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

The editable install builds the native `scorch_ops` extension. Re-run the second
command after changing code under `src/scorch/csrc/`.

Verify the installation with:

```bash
python scripts/verify_quickstart.py
```

The first verification run may compile kernels and take longer than later runs.

See the [installation guide](docs/source/getting_started/installation.md) for
platform details and troubleshooting.

## Quickstart

This example stores `A` in CSR, multiplies it by a dense PyTorch tensor, and
checks both `matmul` and `einsum` against PyTorch:

```python
import torch
import scorch

A_dense = torch.tensor(
    [
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0],
    ]
)
B = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
)

# from_torch preserves the source layout; choose CSR explicitly.
A = scorch.from_torch(A_dense, "A").to_sparse("ds")

# CSR × dense returns a dense torch.Tensor.
C = scorch.matmul(A, B)
expected = A_dense @ B
assert torch.allclose(C, expected, atol=1e-3, rtol=1e-3)

# einsum requires an explicit output and returns an STensor.
D = scorch.einsum("ij,jk->ik", A, B, format="dd")
assert torch.allclose(D.to_torch(), expected, atol=1e-3, rtol=1e-3)

print(C)
print(D.to_torch())
```

```text
tensor([[11., 14.],
        [ 9., 12.],
        [29., 38.]])
tensor([[11., 14.],
        [ 9., 12.],
        [29., 38.]])
```

Generic format/operation combinations compile on first use and reuse the cached
shared library afterward. Common `matmul` combinations may use a prebuilt kernel
instead, so not every first call incurs JIT compilation.

## Format system

A format has one level per tensor dimension, written from the outermost mode to
the innermost. Compact strings are accepted when constructing or converting an
`STensor`:

| Notation | Kind | Meaning |
|:--:|---|---|
| `d` | Level | Dense: iterate the full dimension. |
| `s` or `c` | Level | Compressed: store positions and coordinates. |
| `o` | Level | Coordinate: store explicit coordinates. |
| `"dd"` | Matrix | Fully dense. |
| `"ds"` | Matrix | CSR: dense rows, compressed columns. |
| `"oo"` | Matrix | COO: coordinate rows and columns. |
| `"ss"` | Matrix | Doubly compressed. |

For example, `to_sparse("ds")` converts a matrix to CSR and `str(A.format)`
prints `d,s`. The `singleton` level exists in the type system but is reserved
and is not currently supported by the runtime or code generator.

Read the [format-system guide](docs/source/user_guide/format_system.md) for
constructors, aliases, mode ordering, and storage details.

## Execution model

Scorch uses a hybrid runtime:

1. Dense-only `matmul` delegates to PyTorch.
2. Common sparse `matmul` layouts use prebuilt kernels from `scorch_ops`.
3. Generic supported expressions follow the compiler pipeline:

   ```text
   torch.Tensor → STensor + format → CIN → scheduling → LLIR
                → C++/OpenMP → JIT-compiled shared library
   ```

Generated kernels are cached in memory and on disk. Reuse the same formats when
benchmarking, and time warmed calls rather than compilation.

The main public operations include:

- `scorch.matmul` for sparse matrix-vector, sparse matrix-dense matrix, and
  sparse matrix-sparse matrix products;
- `scorch.einsum` for supported explicit-output contractions, elementwise
  expressions, and SDDMM;
- `scorch.sparse_linear`, `scorch.sparse_attention`, and related fused inference
  helpers; and
- `scorch.autotune` / `scorch.set_autotune` for selecting CSR × dense SpMM
  kernel and tiling variants.

## Current scope

- Scorch tensors and generated kernels are CPU-only.
- Scorch targets inference; `STensor` does not participate in autograd.
- `STensor` does not implement the `@` operator. Use `scorch.matmul(A, B)`.
- `scorch.einsum` requires an explicit output, such as `"ij,jk->ik"`.
- Linux and macOS are supported; Windows is not currently supported.

## Examples and documentation

Runnable examples live in:

| Example | What it covers |
|---|---|
| [Standard kernels](examples/kernels/README.md) | SpMV, SpMM, SpGEMM, and SDDMM. |
| [Graph convolutional networks](examples/gcn/README.md) | GCN inference across several graph datasets. |
| [Sparse autoencoder](examples/sparse_autoencoder/README.md) | Sparse-weight autoencoder inference. |
| [Sparse transformer](examples/sparse_transformer/README.md) | Sparse attention workloads. |

The documentation source provides deeper guides:

- [Getting started](docs/source/getting_started/index.md)
- [User guide](docs/source/user_guide/index.md)
- [Compiler internals](docs/source/compiler/index.md)
- [API reference](docs/source/api/index.md)
- [Performance and tuning](docs/source/performance/index.md)

## Development

After activating the Scorch environment:

```bash
# Full test suite
pytest tests

# Focused test file
pytest tests/test_scorch/test_tensor.py -q

# Formatting, type checking, and linting
bash pre-commit.sh
```

Tests may JIT-compile kernels on first execution. Add regression tests beside the
affected subsystem and compare numerical results against PyTorch with explicit
tolerances.

## Citation

If you use Scorch in academic work, please cite:

> Bobby Yan, Alexander J. Root, Trevor Gale, David Broman, and Fredrik Kjolstad.
> “Fast Autoscheduling for Sparse ML Frameworks.” 2026 IEEE/ACM International
> Symposium on Code Generation and Optimization (CGO), pp. 28–43, 2026.
> [DOI](https://doi.org/10.1109/CGO68049.2026.11394842) ·
> [PDF](https://fredrikbk.com/publications/scorch.pdf)

## License

Scorch is available under the [MIT License](LICENSE).
