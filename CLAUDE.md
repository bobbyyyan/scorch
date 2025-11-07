# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scorch is a Python library for sparse machine learning built on top of PyTorch. It provides sparse tensor implementations of key PyTorch operations, allowing users to work with sparse tensors seamlessly by importing it as a drop-in replacement (`import scorch as torch`).

## Development Commands

### Building and Installation
```bash
# Install in development mode (recommended)
pip install -e .

# Build C++ extensions with optimizations
pip install .
```

### Testing
```bash
# Run all tests
pytest tests
# or
pipenv run test

# Run specific test file
pytest tests/test_scorch/test_tensor.py -v

# Run performance benchmarks
pytest tests/test_scorch/test_perf.py -v
```

### Code Quality
```bash
# Format code (88 char line limit)
black src tests examples
# or
pipenv run format

# Lint code
flake8 src
# or
pipenv run lint

# Type checking
mypy src --install-types --non-interactive --show-error-codes --check-untyped-defs
# or
pipenv run typecheck

# Run all pre-commit checks
bash pre-commit.sh
```

### Running Examples
```bash
# Graph Neural Networks
cd examples/gcn && python scorch_gcn.py --dataset cora
cd examples/gat && python torch_gat.py

# Sparse Models
cd examples/sparse_transformer && python scorch_sparse_transformer.py
cd examples/sparse_autoencoder && python scorch_sparse_autoencoder.py

# Kernel Benchmarks
cd examples/kernels && python spmm.py
```

## Architecture Overview

### Core Components

1. **STensor** (`src/scorch/stensor.py`): Sparse tensor implementation supporting multiple formats (COO, CSR)

2. **Compiler Pipeline** (`src/scorch/compiler/`):
   - `cin.py`: Computation Index Notation (high-level representation)
   - `cin_lowerer.py`: Lowers CIN to LLIR (Low-Level IR)
   - `llir.py`: Low-level intermediate representation
   - `codegen.py`: Generates optimized C++ code
   - `scheduler.py`: Schedules operations for optimization
   - `iter_lattice.py`: Iteration lattice for sparse computations

3. **C++ Extensions** (`csrc/`): Performance-critical operations implemented in C++ with OpenMP parallelization

4. **Format System** (`src/scorch/format.py`):
   - Level types: Dense ("d"), Compressed ("s"), Coordinate ("o")
   - Format strings: "dd" (dense-dense), "ds" (dense-sparse), etc.

### Key Design Patterns

- **Visitor Pattern**: Used throughout compiler for AST traversal
- **Dataclasses**: IR nodes are implemented as dataclasses
- **Format Inference**: Automatic format selection based on tensor properties
- **Lazy Compilation**: Operations are compiled on first use and cached

### Important Implementation Details

1. **Sparse Operations**: SPMM, SPMV, SDDMM, SpMSpM with optimized kernels
2. **C++ Compilation Flags**: `-O3 -march=native -ffast-math -fopenmp -funroll-loops`
3. **Testing Pattern**: Compare Scorch vs PyTorch implementations for correctness and performance
4. **Import Convention**: `import scorch as torch` for drop-in replacement

### Development Tips

- When modifying sparse operations, check both Python (`src/scorch/ops.py`) and C++ (`csrc/`) implementations
- Compiler changes require understanding the full pipeline: CIN → LLIR → C++ codegen
- Performance tests track both compilation time and execution time separately
- Use `format.infer_format()` for automatic sparse format selection
- STensor supports both PyTorch tensor and scipy sparse matrix inputs