# Technology Stack

**Analysis Date:** 2026-03-18

## Languages

**Primary:**
- Python 3.11 - Main application language for sparse ML library
- C++ 14 - High-performance sparse tensor kernels

**Secondary:**
- CMake - Build system for C++ extensions

## Runtime

**Environment:**
- CPython 3.11.7 - Specified runtime in `Pipfile` and CI workflows
- PyTorch runtime - GPU/CPU tensor backend

**Package Manager:**
- Pipenv 1.x - Dependency management (Pipfile-based)
- pip - Direct package installation within Pipenv environments
- Lockfile: `Pipfile.lock` (present, pinned to exact versions)

## Frameworks

**Core:**
- PyTorch 2.0.1 - Deep learning framework; sparse tensor operations backend (CPU version in CI, flexible in development)
- NumPy <2.0 - Numerical computing utilities for tensor operations
- SciPy - Scientific computing for sparse matrix operations

**Testing:**
- pytest - Test runner and framework
- pytest-markers - Performance test categorization (custom `perf` marker in `pytest.ini`)

**Build/Dev:**
- setuptools - Python package installation
- torch.utils.cpp_extension (CppExtension, BuildExtension) - PyTorch C++ extension building
- CMake 3.23+ - C++ project build configuration
- Ninja - Fast parallel build system for C++ compilation

## Key Dependencies

**Critical:**
- torch - Deep learning framework; handles tensor operations, GPU acceleration
- numpy <2 - Numerical arrays; required for compatibility with legacy sparse operations
- scipy - Sparse matrix algorithms; used for baseline comparisons and utilities

**Infrastructure:**
- ninja - Build acceleration for C++ extensions (specified in requirements.txt)

**Development:**
- black - Code formatting (required in Pipenv)
- flake8 - Linting with custom config (max complexity 39, ignore E501/E203/W503)
- mypy - Static type checking
- ipdb - Interactive debugging
- jupyterlab, notebook, ipykernel - Interactive notebook support
- matplotlib - Plotting for benchmarks and analysis

**Optional (Examples):**
- torch_geometric - Graph neural network library (in `examples/gcn/requirements.txt`)
- dgl - Deep Graph Library for graph operations
- ogb - Open Graph Benchmark datasets
- torch-scatter, torch-sparse - Sparse tensor utilities for geometric ML
- torchvision, torchtext, torchdata - Domain-specific PyTorch extensions

## Configuration

**Environment:**
- Python version: 3.11.7 (enforced in Pipfile `[requires]`)
- OpenMP support: Platform-specific compilation flags
  - macOS: Xpreprocessor -fopenmp with Homebrew libomp
  - Linux: Direct -fopenmp with PyTorch's bundled libgomp

**Build:**
- `setup.py` - Package installation with C++ extension definition
- `CMakeLists.txt` - CMake configuration for optional standalone builds
- Compiler flags: `-O3 -march=native -ffast-math -funroll-loops` for optimization
- PyTorch include directories: Dynamically resolved from torch installation

**Linting/Formatting:**
- `.flake8` - Configuration with E501 (line length), E203, W503 ignored; max complexity 39
- Black formatting - 88 character line length (implicit)
- mypy static type checking

**Testing:**
- `pytest.ini` - Markers for performance tests; warnings suppressed for sparse CSR beta support

## Platform Requirements

**Development:**
- Linux, macOS (x86_64 and Apple Silicon), or Windows compatible system
- C++ compiler (clang on macOS, g++ on Linux)
- OpenMP support (libomp on macOS via Homebrew or system)
- CMake 3.23+

**Production:**
- Linux (primary target in `.github/workflows/pytest.yml`)
- CPython 3.11+
- PyTorch 2.0.1+
- libomp or libgomp (OpenMP runtime)

## Dependency Versions

**Locked Versions (from Pipfile.lock):**
- torch: 2.2.1 (in examples/gcn environment); 2.0.1 (CI/CPU)
- numpy: 1.26.4
- scipy: 1.12.0
- torch_geometric: 2.5.0 (optional)
- dgl: 2.0.0 (optional)
- scikit-learn: 1.4.1.post1 (optional)
- pandas: 2.2.1 (optional)
- matplotlib: 3.8.3 (optional)

---

*Stack analysis: 2026-03-18*
