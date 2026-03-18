# Codebase Structure

**Analysis Date:** 2026-03-18

## Directory Layout

```
scorch/
├── src/scorch/                 # Main Python package
│   ├── __init__.py            # Public API exports
│   ├── ops.py                 # High-level tensor operations (matmul, einsum, spmv)
│   ├── stensor.py             # STensor class (sparse tensor abstraction)
│   ├── storage.py             # TensorStorage and TensorIndex data structures
│   ├── format.py              # TensorFormat descriptor (dense/sparse/singleton/coordinate)
│   ├── utils.py               # Utilities (JIT compilation, kernel caching, format parsing)
│   ├── prebuilt_kernels.py    # Prebuilt kernel registry and dispatch
│   ├── _C/                    # C++ binding module (compiled extension)
│   └── compiler/              # Multi-stage compilation pipeline
│       ├── __init__.py
│       ├── cin.py             # Compiler Index Notation (high-level domain language)
│       ├── cin_lowerer.py     # CIN → LLIR lowering with scheduling
│       ├── llir.py            # Low-Level Intermediate Representation (type system, nodes)
│       ├── codegen.py         # LLIR → C++ code generation
│       ├── scheduler.py       # Loop scheduling and optimization
│       ├── iterator.py        # Iterator analysis for sparse access patterns
│       └── iter_lattice.py    # Iterator lattice construction
├── csrc/                       # C++ source templates and prebuilt kernels
│   ├── header.cpp             # C++ header template (called by every kernel)
│   ├── header.h               # C++ header definitions
│   ├── ops.cpp                # C++ extension binding implementation
│   ├── pybind.cpp             # PyBind11 bindings (legacy)
│   ├── kernels.h              # Prebuilt kernel definitions
│   ├── spmm.h                 # Sparse matrix multiplication kernels
│   ├── prebuilt_types.h       # Type definitions for prebuilt kernels
│   └── spmm_simd_optimized.h  # SIMD optimized SpMM
├── tests/                      # Test suite
│   ├── conftest.py            # Pytest configuration and fixtures
│   └── test_scorch/           # Test package
│       ├── test_kernels.py    # Kernel correctness tests
│       ├── test_tensor.py     # STensor functionality
│       ├── test_kernels_comprehensive.py  # Extended kernel tests
│       ├── test_matmul_dispatch.py        # Dispatch logic tests
│       ├── test_scheduler.py  # Scheduler behavior
│       ├── test_cin.py        # CIN AST tests
│       ├── test_compiler.py   # Compiler pipeline tests
│       ├── test_format_*.py   # Format conversion tests
│       ├── test_prebuilt_kernel_registry.py  # Prebuilt kernel tests
│       ├── test_known_compiler_gaps*.py     # Known limitation tests
│       └── codegen/           # Code generation specific tests
│           ├── test_1d_operations.py
│           ├── test_2d_operations.py
│           ├── test_matmul_operations.py
│           ├── test_higher_dim_operations.py
│           └── test_codegen_perf_optimizations.py
├── examples/                   # Example usage
│   ├── kernels/               # Kernel demonstrations
│   │   ├── spmv.py            # SpMV example
│   │   ├── spmm.py            # SpMM example
│   │   ├── sddmm.py           # SDDMM example
│   │   └── plot_spmm.py       # Visualization
│   ├── gcn/                   # Graph Neural Network example
│   ├── sparse_transformer/    # Sparse Transformer example
│   └── sparse_autoencoder/    # Sparse Autoencoder example
├── bench/                      # Benchmarks
│   ├── bench_spmm.py          # SpMM benchmarks
│   ├── bench_spmv.py          # SpMV benchmarks
│   ├── bench_sddmm.py         # SDDMM benchmarks
│   ├── bench_sparse_transformer.py
│   ├── bench_gcn.py
│   ├── bench_sparse_autoencoder.py
│   ├── bench_scheduling.py
│   ├── bench_omp_vs_tbb.py
│   ├── bench_spmm_variants.py
│   ├── _utils.py              # Benchmark utilities
│   └── bench_results/         # Benchmark output data
├── tools/                      # Utility scripts
├── setup.py                    # Package build configuration
├── setup.sh                    # Setup script (environment + build)
├── CMakeLists.txt             # CMake configuration (optional)
├── pytest.ini                  # Pytest configuration
├── requirements.txt           # Python dependencies
├── Pipfile, Pipfile.lock      # Alternative dependency specs
├── .github/                    # CI/CD workflows
├── .gitignore                 # Git exclusions
└── .planning/                 # Generated planning documents

```

## Directory Purposes

**src/scorch/:**
- Purpose: Main Python package for sparse operations
- Contains: User API, tensor abstractions, compiler pipeline
- Key files: `ops.py` (operations), `stensor.py` (tensor), `compiler/` (compilation)

**src/scorch/compiler/:**
- Purpose: Compiler infrastructure for sparse tensor code generation
- Contains: CIN DSL, LLIR, scheduling, code generation
- Key files: `cin.py` (domain language), `cin_lowerer.py` (lowering), `codegen.py` (C++ gen)

**csrc/:**
- Purpose: C++ kernel implementations and extension binding
- Contains: Header templates, prebuilt sparse kernels, SIMD optimizations
- Key files: `header.cpp` (runtime), `spmm.h` (SpMM kernels), `kernels.h` (all kernels)

**tests/:**
- Purpose: Correctness verification across all layers
- Contains: Unit tests, integration tests, known limitation tests
- Organization: One test file per module + codegen subdirectory

**examples/:**
- Purpose: Real-world usage demonstrations
- Contains: Neural network examples, kernel demonstrations
- Key files: GCN, sparse transformer, sparse autoencoder implementations

**bench/:**
- Purpose: Performance measurement and profiling
- Contains: Benchmarks for all major operations
- Key files: `bench_spmm.py`, `bench_sddmm.py`, `_utils.py` (shared utilities)

## Key File Locations

**Entry Points:**
- `src/scorch/__init__.py`: Public API (STensor, einsum, matmul, from_torch, from_coo)
- `src/scorch/ops.py`: Main operation implementations (matmul, einsum, spmv, matmul_wksp)

**Configuration:**
- `setup.py`: Package build with C++ extension configuration
- `pytest.ini`: Test discovery and settings
- `requirements.txt`: Python dependencies
- `.flake8`: Linting configuration

**Core Logic:**
- `src/scorch/stensor.py`: STensor class (467 lines) - sparse tensor wrapper
- `src/scorch/compiler/cin.py`: CIN AST nodes and operations (944 lines)
- `src/scorch/compiler/cin_lowerer.py`: Lowering logic (5,290 lines)
- `src/scorch/compiler/codegen.py`: C++ code generation (461 lines)
- `src/scorch/ops.py`: High-level operations (922 lines)

**Testing:**
- `tests/test_scorch/test_kernels.py`: Main kernel tests
- `tests/test_scorch/test_matmul_dispatch.py`: Dispatch logic
- `tests/test_scorch/codegen/test_matmul_operations.py`: Code generation tests
- `tests/conftest.py`: Pytest fixtures and configuration

## Naming Conventions

**Files:**
- Operation files: `ops.py` (operations module)
- Test files: `test_*.py` (pytest discovery)
- Benchmark files: `bench_*.py` (operation-specific benchmarks)
- Example files: Named by domain (e.g., `scorch_gcn.py`, `torch_gcn.py` for comparison)
- Compiler stages: `cin.py` → `cin_lowerer.py` → `llir.py` → `codegen.py`

**Directories:**
- Package modules: lowercase plural (`compiler/`, `tests/`)
- Example domains: lowercase with underscore (e.g., `sparse_transformer/`, `sparse_autoencoder/`)
- Benchmark output: `bench_results/` for data artifacts

**Classes:**
- Tensor classes: PascalCase (STensor, TensorStorage, TensorIndex, TensorFormat)
- IR nodes: PascalCase with suffix (ForAll, TensorAssign, Workspace, IndexVar)
- Enums: PascalCase (Operation, LevelType, DataType, AssignOp)

**Functions:**
- Public operations: lowercase_underscore (matmul, einsum, spmv, from_torch)
- Private utilities: leading underscore (_load_kernel, _kernel_name, _kernel_cache)
- Visitor pattern: snake_case + verb (visit_ForAll, lower_llir, accept)

**Variables:**
- Tensors: Single letter (A, B, C for matrices; i, j, k for indices)
- Index variables: Lowercase with number (i0, i1, or semantic i, j, k)
- Caches: Descriptive + suffix (\_kernel_cache, _so_cache, \_module_cache)

## Where to Add New Code

**New Sparse Operation:**
- Primary code: `src/scorch/ops.py` - add function following matmul/einsum pattern
- CIN construction: Define TensorVar, IndexVar, ForAll/Where chains
- Tests: Add to `tests/test_scorch/test_kernels.py` or create `tests/test_scorch/test_<operation>.py`
- Example: `examples/kernels/<operation>.py`

**New Format Type:**
- Format enum: `src/scorch/format.py` - add to LevelType enum and _STR_TO_LEVEL_TYPE
- Iterator logic: `src/scorch/compiler/iter_lattice.py` - handle in lattice construction
- Tests: `tests/test_scorch/test_format_*.py` - add format conversion/inference test

**New Optimization (scheduling/codegen):**
- Scheduler rules: `src/scorch/compiler/scheduler.py` - add transformation pass
- Code generation: `src/scorch/compiler/codegen.py` - add LLIRLowerer visitor method if needed
- Tests: `tests/test_scorch/codegen/test_codegen_perf_optimizations.py`

**Prebuilt Kernel:**
- C++ kernel: `csrc/kernels.h` (or domain-specific like csrc/spmm.h)
- Registration: `src/scorch/prebuilt_kernels.py` - add to PrebuiltKernelRegistry
- Tests: `tests/test_scorch/test_prebuilt_kernel_registry.py`

**Utilities:**
- Shared helpers: `src/scorch/utils.py`
- Compilation utilities: Already contains _load_kernel, _kernel_name
- Format utilities: Already contains parse_format, topo_sort_characters

## Special Directories

**build/:**
- Purpose: CMake/setuptools build artifacts
- Generated: Yes (from setup.py compilation)
- Committed: No (in .gitignore)

**bench_results/:**
- Purpose: Benchmark output data and CSV results
- Generated: Yes (by benchmark scripts)
- Committed: No (contains only local results)

**src/scorch/__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Yes (automatic by Python)
- Committed: No (in .gitignore)

**.git/, .github/:**
- Purpose: Git repository metadata, CI/CD workflows
- Generated: Yes (.git by git, workflows defined in .github)
- Committed: Yes (.github workflows committed, .git for repo state)

**src/scorch.egg-info/:**
- Purpose: Installed package metadata (from setup.py)
- Generated: Yes (from pip install .)
- Committed: No (in .gitignore)

---

*Structure analysis: 2026-03-18*
