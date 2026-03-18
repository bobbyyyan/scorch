# Testing Patterns

**Analysis Date:** 2026-03-18

## Test Framework

**Runner:**
- pytest (configured in `pytest.ini`)
- Version: 8.3.3 and 9.0.2 (cached pycache indicates upgrades over time)
- Config file: `pytest.ini`

**Assertion Library:**
- pytest's native `assert` statements
- PyTorch's `torch.allclose()` for numerical comparison with tolerance
- Custom assertion helper: `assert_close()` function wraps torch.allclose

**Run Commands:**
```bash
pytest tests/                          # Run all tests
pytest tests/test_scorch/test_kernels.py  # Run specific test file
pytest tests/ -m "not perf"           # Skip performance tests
pytest tests/ --tb=short              # Show short traceback format
```

## Test File Organization

**Location:**
- All tests in `tests/test_scorch/` directory, mirroring source structure
- Codegen tests in `tests/test_scorch/codegen/` subdirectory
- Benchmarks separate in `bench/` directory (not part of main test suite)
- Ad-hoc test scripts at root: `test_simd_implementation.py`

**Naming:**
- Test functions: `test_*` prefix (pytest convention)
- Test classes: `Test*` prefix for pytest class-based organization
- Examples: `test_dense_copy()`, `test_sparse_to_dense()`, `test_spmv_square()`, `TestSpMV`, `TestElemwiseMulRandom`

**Structure:**
```
tests/
├── conftest.py                    # pytest fixtures (session-scoped TORCH_EXTENSIONS_DIR)
├── test_scorch/
│   ├── test_helpers.py           # Helper functions for test setup
│   ├── test_cin.py               # CIN IR tests
│   ├── test_compiler.py          # Compiler integration tests
│   ├── test_kernels.py           # Basic kernel tests
│   ├── test_kernels_comprehensive.py  # Comprehensive correctness tests
│   ├── test_kernels_mode_order.py     # Mode order variant tests
│   ├── test_format_convert.py    # Format conversion tests
│   ├── test_format_inference.py  # Format inference tests
│   ├── test_scheduler.py         # Scheduler tests
│   ├── test_known_compiler_gaps*.py   # Known limitation tests
│   └── codegen/
│       ├── test_1d_operations.py
│       ├── test_2d_operations.py
│       ├── test_higher_dim_operations.py
│       ├── test_matmul_operations.py
│       └── test_codegen_perf_optimizations.py
```

## Test Structure

**Suite Organization:**
```python
# Simple function-based tests (most common)
def test_dense_copy():
    tensor_a_torch = torch.Tensor([[1, 0], [0, 2]])
    tensor_a = STensor.from_torch(tensor_a_torch, "A")
    tensor_a_dense = tensor_a.to_dense()
    print(tensor_a_dense)

# Class-based tests with shared setup and multiple test methods
class TestSpMV:
    """SpMV: y[i] = A[i,j] * x[j]  verified against torch.mv"""

    @pytest.mark.parametrize("matrix_fmt", ["ds", "ss", "oo"])
    def test_spmv_square(self, matrix_fmt):
        torch.manual_seed(42)
        # test body
        assert_close(result, expected)
```

**Patterns:**
- Setup: Create test data with known seed, convert to STensor format
- Execution: Call scorch operation (einsum, matmul, format conversion)
- Assertion: Compare against PyTorch reference with `assert_close()`
- Teardown: Implicit (no cleanup needed for torch operations)

**Assertion Pattern:**
```python
# Numerical tolerance assertions
def assert_close(scorch_result, expected):
    """Compare an STensor result against a torch.Tensor reference."""
    if isinstance(scorch_result, STensor):
        actual = scorch_result.to_torch()
    else:
        actual = scorch_result
    assert torch.allclose(actual, expected, atol=ATOL, rtol=RTOL), (
        f"Max diff: {(actual - expected).abs().max().item()}"
    )

# Direct equality assertions
assert a_sparse_to_torch.tolist() == tensor_a_torch.tolist()
assert result.shape == (12,)
assert len(result.index.mode_indices) == 1
```

## Mocking

**Framework:**
- No mocking framework (unittest.mock) detected in tests
- Tests use real implementations throughout
- Prebuilt kernel tests check symbol resolution with `getattr(native_ops, symbol_name, None)`

**Patterns:**
- No mock objects; tests use actual STensor, TensorVar, etc.
- Performance test setup uses real sparse matrices from scipy.sparse
- Format inference tests parametrize cache usage: `@pytest.mark.parametrize("use_cache", [True, False])`

**What to Mock:**
- Generally NOT done in this codebase; prefer end-to-end testing
- Integration with C++ extensions cannot be mocked (they're real compiled code)

**What NOT to Mock:**
- Tensor operations (use real STensor)
- Compiler phases (CINLowerer, LLIRLowerer are real)
- Kernel execution (real torch.sparse.mm, real compiled kernels)

## Fixtures and Factories

**Test Data Creation:**
```python
# Sparse matrix creation helper
def make_sparse_2d(m, n, sparsity, seed):
    """Return a (m x n) torch.Tensor with the given sparsity ratio zeroed out."""
    torch.manual_seed(seed)
    t = torch.rand(m, n)
    mask = (torch.rand(m, n) > sparsity).float()
    return t * mask

# Used in tests:
a_torch = make_sparse_2d(30, 30, 0.8, seed=42)
x_torch = torch.rand(30)

# 3D tensor creation
def make_sparse_3d(d0, d1, d2, sparsity, seed):
    """Return a (d0 x d1 x d2) torch.Tensor with the given sparsity."""
    torch.manual_seed(seed)
    t = torch.rand(d0, d1, d2)
    mask = (torch.rand(d0, d1, d2) > sparsity).float()
    return t * mask
```

**Compiler Statement Builders:**
```python
# In test_helpers.py - CIN statement construction helpers
def create_index_vars(*names):
    """Create index variables with the given names."""
    return tuple(IndexVar(name) for name in names)

def create_tensor_vars(tensor_specs):
    """Create tensor variables with the given specifications.

    Args:
        tensor_specs: A dict mapping tensor names to format specifications
    Returns:
        A dict mapping tensor names to TensorVar objects
    """
    return {name: TensorVar(name, fmt=fmt) for name, fmt in tensor_specs.items()}

def create_elementwise_operation(tensors, index_vars, operation_type="mul", index_maps=None):
    """Create a common elementwise operation (multiplication or addition) on tensors."""
    # Returns a CIN statement
```

**Session-Level Fixture:**
```python
# In conftest.py
@pytest.fixture(scope="session", autouse=True)
def _set_torch_extensions_dir(tmp_path_factory):
    """Set TORCH_EXTENSIONS_DIR to temp directory for session to avoid cache conflicts."""
    old_value = os.environ.get("TORCH_EXTENSIONS_DIR")
    ext_dir = tmp_path_factory.mktemp("torch_extensions")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = old_value
```

**Location:**
- Helper functions in `tests/test_scorch/test_helpers.py`
- Data generators in test files themselves (make_sparse_2d, etc.)
- Global fixtures in `tests/conftest.py`

## Coverage

**Requirements:**
- No coverage threshold enforced (no `.coveragerc` or coverage configuration found)
- Coverage not mentioned in CI/test setup

**View Coverage:**
- Not configured; would use: `pytest --cov=src/scorch tests/`

## Test Types

**Unit Tests:**
- Individual operation tests: `test_dense_copy()`, `test_sparse_to_dense()`
- CIN compiler unit tests: `test_elementwise_sparse_vector_mul_cin()`, `test_loop_order_getter()`
- Scope: Single function or class method
- Approach: Direct testing of public API with simple inputs

**Integration Tests:**
- Format conversion roundtrips: `test_2d_ss_oo()`, `test_2d_dd_oo()`
- Full operation flow: `test_spmv_square()` tests STensor creation → sparse format → einsum → verification
- Scope: Multiple components working together
- Approach: End-to-end against PyTorch reference implementations

**End-to-End Tests:**
- Comprehensive correctness suite in `test_kernels_comprehensive.py`: Tests all tensor operations (SpMV, elemwise, matmul, outer product) across format combinations
- Approach: Real sparse matrices, real kernels, real torch comparison
- Classes: `TestSpMV`, `TestElemwiseMulRandom`, `TestElemwiseAddRandom`, `TestElemwiseMul3D`, `TestOuterProduct`

**Performance Tests:**
- Marked with `@pytest.mark.perf` decorator
- Location: `test_perf.py`, `test_perf_large.py`
- Skipped in normal test runs: `pytest -m "not perf"`
- Time-sensitive operations that may fail in CI/slow environments

## Test Parametrization

**Decorator Pattern:**
```python
@pytest.mark.parametrize("matrix_fmt", ["ds", "ss", "oo"])
def test_spmv_square(self, matrix_fmt):
    # Test runs 3 times with each format

@pytest.mark.parametrize(
    "a_fmt, b_fmt, out_fmt",
    [
        ("ds", "ds", "dd"),
        ("ds", "dd", "dd"),
        # ...
    ]
)
def test_elemwise_mul_2d_random(self, a_fmt, b_fmt, out_fmt):
    # Test runs once per combination

@pytest.mark.parametrize("use_cache", [True, False])
def test_spmm_oo_dd_fmtinf(use_cache):
    # Tests kernel caching behavior
```

## Test Markers

**Custom Markers:**
- `@pytest.mark.perf` - Performance-sensitive tests (skip in normal runs)

**Pytest Config Markers:**
```ini
[pytest]
markers =
    perf: performance tests
filterwarnings =
    ignore:Sparse CSR tensor support is in beta state:UserWarning
```

**Usage:**
```bash
pytest -m "not perf"     # Exclude performance tests
pytest -m perf           # Run only performance tests
```

## Common Test Patterns

**Sparse Matrix Testing:**
```python
torch.manual_seed(42)
a_torch = make_sparse_2d(30, 30, 0.8, seed=42)
x_torch = torch.rand(30)

a_st = STensor.from_torch(a_torch).to_sparse(matrix_fmt)
x_st = STensor.from_torch(x_torch)

result = einsum("ij,j->i", a_st, x_st, format="d")
expected = torch.mv(a_torch, x_torch)
assert_close(result, expected)
```

**Compiler Code Generation Testing:**
```python
# From test_1d_operations.py
i = IndexVar("i")
a = TensorVar("a", fmt="s")
b = TensorVar("b", fmt="s")
c = TensorVar("c", fmt="s")

a[i] = b[i] * c[i]

cin_stmt = ForAll(i, a._assignment)
cpp_code = lower_and_print(cin_stmt)
```

**Correctness Assertion with Tolerance:**
```python
ATOL = 1e-3
RTOL = 1e-3

def assert_close(scorch_result, expected):
    if isinstance(scorch_result, STensor):
        actual = scorch_result.to_torch()
    else:
        actual = scorch_result
    assert torch.allclose(actual, expected, atol=ATOL, rtol=RTOL), (
        f"Max diff: {(actual - expected).abs().max().item()}"
    )
```

**Multiple Assertion Types in Single Test:**
```python
# Shape checks
assert result.shape == (12,)

# Structure checks
assert len(result.index.mode_indices) == 1

# Index value checks
assert mode_index[0].tolist() == [0, 5]
assert mode_index[1].tolist() == [0, 2, 4, 10, 11]

# Data value checks
assert result.values.tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]
```

---

*Testing analysis: 2026-03-18*
