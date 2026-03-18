# Architecture

**Analysis Date:** 2026-03-18

## Pattern Overview

**Overall:** Compiler-based sparse tensor computation framework with multi-stage IR lowering

**Key Characteristics:**
- Compiler pipeline: CIN (Compiler Index Notation) → LLIR (Low-Level IR) → C++ code generation
- Custom sparse tensor abstraction layer (STensor) wrapping PyTorch integration
- Just-in-time C++ kernel compilation with module caching
- Format-driven computation (explicit tensor format specification)
- Workspace-optimized accumulation for sparse operations

## Layers

**API Layer:**
- Purpose: User-facing operations for sparse tensor computation
- Location: `src/scorch/ops.py`, `src/scorch/__init__.py`
- Contains: `matmul()`, `einsum()`, `spmv()`, `matmul_wksp()` - high-level tensor operations
- Depends on: STensor, compiler chain, prebuilt kernels
- Used by: External code, examples, tests

**STensor Abstraction Layer:**
- Purpose: Unified sparse tensor representation abstracting PyTorch integration
- Location: `src/scorch/stensor.py`
- Contains: STensor class wrapping TensorStorage, conversion methods (from_torch, to_torch, to_sparse, to_dense)
- Depends on: TensorStorage, TensorFormat, format module
- Used by: All operations, compiler pipeline

**Format & Storage Layer:**
- Purpose: Describe tensor memory layout and physical storage
- Location: `src/scorch/format.py`, `src/scorch/storage.py`
- Contains: TensorFormat (declares DENSE/COMPRESSED/COORDINATE/SINGLETON levels), TensorStorage (index + values), TensorIndex
- Depends on: PyTorch tensors
- Used by: STensor, compiler for code generation

**Compiler Pipeline Layer:**
- Purpose: Multi-stage compilation from high-level notation to executable C++
- Location: `src/scorch/compiler/`
- Contains:
  - **CIN (cin.py):** Domain-specific language for sparse operations (ForAll, Where, TensorAssign, Workspace)
  - **CINLowerer (cin_lowerer.py):** Lowers CIN to LLIR with scheduling
  - **LLIR (llir.py):** Low-level intermediate representation (loops, conditionals, type system)
  - **LLIRLowerer (codegen.py):** Generates C++ code strings from LLIR
  - **Supporting modules:** iterator.py, iter_lattice.py, scheduler.py
- Depends on: Format layer
- Used by: Operations layer

**C++ Binding Layer:**
- Purpose: Bridge Python-generated C++ to PyTorch via JIT compilation
- Location: `csrc/` (C++ source), `src/scorch/utils.py` (_load_kernel, _kernel_name)
- Contains: Header templates, prebuilt kernel registry, JIT compilation infrastructure
- Depends on: Generated C++ code
- Used by: Operations layer to execute computation

## Data Flow

**Sparse Matrix Multiplication (matmul):**

1. **Input:** STensor inputs with format specifications (e.g., "cs" = compressed rows, sparse columns)
2. **Format Check:** If both operands dense, delegate to torch.matmul()
3. **Prebuilt Dispatch:** Check resolve_prebuilt_matmul() for optimized prebuilt kernels
4. **CIN Construction:** Build index notation for operation (e.g., "ij,jk->ik")
   - Create TensorVar for each input/output with format
   - Define IndexVar (i, j, k) and looping structure
5. **CINLowerer:** Lower CIN to LLIR
   - Apply iterator analysis to understand tensor sparsity structure
   - Generate loop nests with proper index ordering
6. **LLIRLowerer:** Generate C++ code from LLIR
   - Convert loop constructs to C++ for loops
   - Handle memory access patterns, indexing
7. **Compilation:** _load_kernel() uses torch.cpp_extension to JIT compile C++
   - Check disk cache (_so_cache) for .so file
   - Falls back to load_inline() with extra_cflags/extra_ldflags
   - Module caching on repeat calls with same format combination
8. **Execution:** Call module.evaluate() with tensor shapes, indices, values
9. **Result Wrapping:** Convert C++ output back to STensor

**Einsum Path:**

1. Parse einsum expression ("ij,jk->ik" format)
2. Convert torch.Tensor inputs to STensor via from_torch()
3. Special dispatch for patterns like SDDMM ("ij,ik,jk->ij")
4. Generic path: lower_and_exec_cin() → full pipeline
5. Auto-normalize mode orders to canonical [0, 1] for consistency
6. Return STensor or torch.Tensor depending on output format

**State Management:**
- _kernel_cache: Module cache keyed by (format_a, format_b, format_output)
- _einsum_dispatch_cache: Dispatch memos for expression patterns
- matmul._module_cache: Per-function module persistence
- _so_cache: Persistent .so file references (survives process)

## Key Abstractions

**STensor (Sparse Tensor):**
- Purpose: Unified abstraction for sparse tensors with custom storage format
- Examples: `src/scorch/stensor.py` (STensor class)
- Pattern: Wraps TensorStorage, provides PyTorch interop via from_torch/to_torch, format conversion via to_sparse/to_dense

**TensorFormat:**
- Purpose: Declarative format specification (dense, sparse, coordinate, singleton levels)
- Examples: "d" (dense), "cs" (compressed sparse), "o" (coordinate), "s" (singleton)
- Pattern: Immutable format descriptor used in code generation decisions

**Workspace:**
- Purpose: Temporary accumulation buffer for reduction operations
- Examples: Row-wise accumulation in SpMM (workspace[j] += A[i,k]*B[k,j])
- Pattern: Can be dense (low overhead) or sparse (COO hash map); specified in CIN

**CIN (Compiler Index Notation):**
- Purpose: Domain-specific language for sparse tensor operations
- Examples: `ForAll(i, Where(producer=ForAll(k, ...), consumer=...))` for matmul
- Pattern: AST-based, visitable for traversal and analysis

**LLIR (Low-Level IR):**
- Purpose: Intermediate representation close to C++ with type system
- Examples: llir.ForLoop, llir.WhileLoop, llir.IfThenElse, llir.Assign
- Pattern: Direct mapping to C++ constructs via LLIRLowerer visitor pattern

## Entry Points

**matmul():**
- Location: `src/scorch/ops.py` line 248
- Triggers: User calls scorch.matmul(tensor_a, tensor_b)
- Responsibilities: Dispatch to fast paths (dense, prebuilt), fallback to einsum for generic sparse

**einsum():**
- Location: `src/scorch/ops.py` line 375
- Triggers: User calls scorch.einsum("ij,jk->ik", a, b) or internal matmul dispatch
- Responsibilities: Parse expression, build CIN, lower to C++, execute

**lower_and_exec_cin():**
- Location: `src/scorch/ops.py` line 864
- Triggers: Generic sparse operation pipeline
- Responsibilities: Full compilation pipeline (CIN → LLIR → C++ → execution)

**_load_kernel():**
- Location: `src/scorch/utils.py` line 32
- Triggers: Whenever new kernel signature needed
- Responsibilities: JIT compile C++ with caching, return callable module

## Error Handling

**Strategy:** Assertions with descriptive messages, fallback to torch.matmul for dense

**Patterns:**
- Format validation: assert format is TensorFormat or parseable string
- Tensor validation: Check tensor has required index/values before access
- Compilation errors: Caught by torch.cpp_extension, propagate to user
- Missing prebuilt kernels: Silently fall through to generic path (no error)

## Cross-Cutting Concerns

**Logging:** None detected in core pipeline; performance timing captured in time_dict kwargs

**Validation:**
- Format compatibility checked implicitly via type system
- Mode order normalization to [0, 1] before dispatch (handles non-canonical formats)
- Tensor shape validation in from_torch/from_coo

**Authentication:** Not applicable

**Parallelization:**
- OpenMP enabled in C++ via compilation flags (-fopenmp)
- SpmM and header kernels use OMP pragmas (see csrc/spmm.h, csrc/header.cpp)
- Platform-specific linker configuration in setup.py and utils.py

---

*Architecture analysis: 2026-03-18*
