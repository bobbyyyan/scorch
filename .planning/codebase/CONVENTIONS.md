# Coding Conventions

**Analysis Date:** 2026-03-18

## Naming Patterns

**Files:**
- Snake case for all Python source files: `stensor.py`, `prebuilt_kernels.py`, `cin_lowerer.py`
- Test files follow pattern `test_*.py`: `test_kernels.py`, `test_compiler.py`, `test_kernels_comprehensive.py`
- C++ source files in `csrc/` use snake case: `ops.cpp`, `header.cpp`

**Functions:**
- Snake case for all function definitions: `spmv()`, `einsum()`, `matmul()`, `lower_and_print()`, `create_index_vars()`
- Private/internal functions prefixed with underscore: `_kernel_name()`, `_load_kernel()`, `_parse_level_type()`, `_resolve_symbol()`
- Helper functions in tests also follow snake case: `make_sparse_2d()`, `assert_close()`

**Variables:**
- Snake case for all local and module-level variables: `cpp_code`, `lowered_llir`, `tensor_format`, `output_format`
- Module-level constants in uppercase: `PROJECT_ROOT_DIR`, `_kernel_cache`, `_einsum_dispatch_cache`, `ATOL`, `RTOL`, `_STR_TO_LEVEL_TYPE`
- Private module variables prefixed with underscore: `_kernel_cache`, `_einsum_dispatch_cache`, `_so_cache`

**Types & Classes:**
- PascalCase for all class names: `STensor`, `Window`, `TensorVar`, `IndexVar`, `ForAll`, `Where`, `TensorAssign`, `TensorIndex`, `TensorStorage`, `TensorStorageView`, `CINLowerer`, `LLIRLowerer`, `Scheduler`, `LevelFormat`, `LevelType`
- Enum members in uppercase: `LevelType.DENSE`, `LevelType.COMPRESSED`, `LevelType.SINGLETON`, `LevelType.COORDINATE`, `Operation.ADD`
- Dataclasses in PascalCase with frozen=True for immutable specs: `PrebuiltMatmulSpec`, `ResolvedPrebuiltKernel`

**Type Aliases:**
- Snake case with descriptive names: `_UnaryOp = Callable[[Any], Any]`, `_BinaryOp = Callable[[Any, Any], Any]`, `KernelFn = Callable[..., Any]`

## Code Style

**Formatting:**
- Flake8 linter configured in `.flake8`
- Ignored rules: E501 (line too long), E203 (whitespace before colon), W503 (line break before binary operator)
- Max line length: 88 characters
- Max complexity: 39 (permissive for complex compiler code)

**Linting:**
- Use flake8 for linting: `flake8 src/scorch`
- Ignore long lines and whitespace formatting inconsistencies (allows for complex expressions)

**Line Lengths:**
- Target 88 characters (similar to Black formatter style)
- Exceptions allowed for long lines (E501 ignored) especially in:
  - Complex tensor operations and indexing expressions
  - Function signatures with multiple type annotations
  - Compiler codegen functions

## Import Organization

**Order:**
1. `from __future__ import annotations` - Future imports at top
2. Standard library imports: `import os`, `import time`, `from pathlib import Path`, `from typing import ...`, `from collections import defaultdict`
3. Third-party imports: `import torch`, `import numpy as np`, `from torch.utils.cpp_extension import load_inline, load`
4. Local package imports: `from .compiler.cin import ...`, `from .format import TensorFormat`, `from .stensor import STensor`
5. Conditional/TYPE_CHECKING imports for forward references: `if TYPE_CHECKING: from .stensor import STensor`

**Path Aliases:**
- Relative imports used throughout: `from .ops import einsum`, `from .compiler.cin_lowerer import CINLowerer`
- No custom import path aliases configured; all imports are explicit relative paths

**Type Hints:**
- Optional types: `Optional[str]`, `Optional[List[str]]`, `Optional[Union[TensorFormat, str, List[str]]]`
- Union types for flexible inputs: `Union[TensorFormat, str, List[str]]`, `Union[TensorStorage, TensorStorageView]`
- Generic types: `List[str]`, `Tuple[int, ...]`, `Dict[str, Any]`

## Error Handling

**Patterns:**
- Assertions for internal invariants: `assert self._name is not None, "Tensor name is not set."`, `assert self.format, "format is None"`
- ValueError for invalid input: `raise ValueError(f"Invalid format string: {s}")`, `raise ValueError(f"Unsupported RHS rank for prebuilt matmul kernel: {b.dim()}")`
- NotImplementedError for unimplemented features: `raise NotImplementedError()` in `stensor.py` insert, slice, other unimplemented ops
- No try/except blocks found in main code; errors propagate naturally
- Validation typically occurs at public API entry points (e.g., `resolve_prebuilt_matmul()` checks dtype match)

## Logging

**Framework:**
- No logging framework configured; uses `print()` statements for debug output
- Commented-out print statements left in code for development: `# print(f"Pybind load time: {compile_time:.5f} seconds")`
- Test code uses print for inspection: `print("\n\n", cpp_code)`, `print("CIN statement:")`, `print("C++ torch extension code:")`

**Patterns:**
- Debug output wrapped in comments for production code
- Test helpers like `lower_and_print()` print intermediate representations (CIN, LLIR, C++ code) for verification
- Performance timing via `time.time()` or `time.perf_counter()`

## Documentation

**Docstrings:**
- Module-level docstrings in test files: `"""Tests for 1D tensor operations in the CIN compiler."""`
- Class docstrings present in key classes: `class Window: """A tensor window object that describes the slice..."""`, `class STensor: """A tensor stored in custom format."""`
- Method/function docstrings inconsistent - some functions have docstrings with Args/Returns, many do not
- Helper functions in tests have docstrings with Args/Returns: `lower_and_print()`, `create_index_vars()`, `create_tensor_vars()`, `create_elementwise_operation()`

**Comments:**
- Inline comments minimal but used for clarification: `# TODO: storage can also be a secondary index (TensorStorageView)`
- TODO comments in code: `# TODO: Implement this.`, `# TODO: Handle OpenMP flags`
- References to external documentation: comments reference taco tensor compiler and http://tensor-compiler.org/codegen.html

**When to Comment:**
- Mark unimplemented features with TODO
- Explain non-obvious design choices (e.g., PyTorch libomp linking strategy in setup.py)
- Reference academic papers or external specs (e.g., tensor compiler concepts)

## Function Design

**Size:**
- Mix of small utility functions (10-20 lines) and larger compiler functions (50-100+ lines)
- Compiler phases like `lower_IndexStmt()`, `lower_llir()` are naturally longer due to complexity

**Parameters:**
- Keyword arguments used for optional/configuration: `output_format=None`, `mode_order=None`, `fmt=None`, `filter_zeros=False`, `no_comments=False`
- `**kwargs` used in public APIs for extensibility: `def spmv(a: STensor, b: STensor, output_format: Optional[...] = None, **kwargs)`
- Type hints on all function parameters in public API functions
- Optional parameters have defaults: `Optional[Union[...]] = None`

**Return Values:**
- Return types specified on public functions: `-> STensor:`, `-> Optional[ResolvedPrebuiltKernel]:`
- Return types sometimes omitted on private functions and complex internal functions
- Multiple return values as tuples: `return fn, symbol_name`, `return result_cpp, result_shape`

## Module Design

**Exports:**
- Public API defined in `__init__.py` with `__all__` list: `["STensor", "TensorFormat", "einsum", "from_torch", "from_coo", "matmul", "matmul_wksp", "__version__"]`
- Convenience imports at module level: `from_torch = STensor.from_torch`, `from_coo = STensor.from_coo`
- Fallback to PyTorch via `__getattr__`: `def __getattr__(name): return getattr(torch, name)`

**Barrel Files:**
- Compiler submodule `__init__.py` exists but minimal; imports done explicitly from submodules
- No star imports (`from .module import *`) used in codebase
- Explicit imports preferred: `from .compiler.cin import IndexVar, TensorVar, ...`

**Submodule Organization:**
- `src/scorch/` - Main package
- `src/scorch/compiler/` - Compiler infrastructure (CIN, lowering, codegen, scheduling)
- `src/scorch/format.py` - Tensor format definitions
- `src/scorch/stensor.py` - Sparse tensor class
- `src/scorch/storage.py` - Storage and indexing structures
- `src/scorch/ops.py` - High-level tensor operations (spmv, einsum, matmul)
- `src/scorch/prebuilt_kernels.py` - Precompiled kernel resolution and execution
- `src/scorch/utils.py` - Shared utilities
- `csrc/` - C++ extension source code

---

*Convention analysis: 2026-03-18*
