# Testing

Scorch's test suite is the ground truth for correctness. Because the library
JIT-generates and compiles C++ kernels per operation and format, the tests do not
check hand-written numeric expectations — they run each sparse operation and
compare it against the equivalent dense PyTorch call. This page shows how to run
the suite, how the correctness convention works, and how the suite protects the
compiler's more fragile corners with regression tests.

## Running the suite

Activate the `scorch` conda environment first — `pytest`, `python`, and the
compiled `scorch_ops` extension all live in that env:

```bash
conda activate scorch
pytest tests/
```

Bare `pytest` works too (discovery picks up `tests/` automatically). Tests live
under `tests/test_scorch/`, mirroring the package layout, with codegen-specific
cases in `tests/test_scorch/codegen/`.

### A single test

Address one test by file, class, and method with the `::` separator:

```bash
pytest tests/test_scorch/test_kernels.py::TestSpMV::test_spmv_square
```

Narrow to a whole file or class the same way, and use `-k` to filter by substring:

```bash
pytest tests/test_scorch/test_tensor.py -q       # one focused file
pytest tests/test_scorch/test_kernels.py -k SpMV # every SpMV test
```

### Skipping the performance tests

Slow performance tests are marked `@pytest.mark.perf` (declared in
`pytest.ini`). Exclude them with a marker expression for a fast correctness-only
run:

```bash
pytest -m "not perf"
```

`pytest.ini` also silences PyTorch's beta-CSR `UserWarning` so it doesn't clutter
output:

```ini
[pytest]
markers =
    perf: performance tests
filterwarnings =
    ignore:Sparse CSR tensor support is in beta state:UserWarning
```

:::{note}
The **first** execution of any operation/format combination may pause while
Scorch generates C++ and JIT-compiles a kernel — the initial hit is compilation,
not slow math. Subsequent runs reuse the cached `.so`. See
{doc}`the build guide </development/building>` for how the JIT cache works.
:::

## The correctness convention

Every kernel test follows one pattern: run the Scorch operation, run the dense
PyTorch reference, and assert the two agree to `atol = rtol = 1e-3`. There are no
hardcoded expected numbers to drift out of date — the reference *is* the oracle.

The shared helper (from `tests/test_scorch/test_kernels_comprehensive.py`)
densifies the {class}`~scorch.STensor` result before comparing:

```python
import torch
from scorch import STensor

ATOL = 1e-3
RTOL = 1e-3


def assert_close(scorch_result, expected):
    """Compare an STensor result against a torch.Tensor reference."""
    actual = scorch_result.to_torch() if isinstance(scorch_result, STensor) else scorch_result
    assert torch.allclose(actual, expected, atol=ATOL, rtol=RTOL), (
        f"Max diff: {(actual - expected).abs().max().item()}"
    )
```

A complete SpMV test reads as *build sparse input → run Scorch → run
`torch.mv` → assert*:

```python
import torch
import pytest
from scorch import STensor, einsum


class TestSpMV:
    """SpMV: y[i] = A[i, j] * x[j], verified against torch.mv."""

    @pytest.mark.parametrize("matrix_fmt", ["ds", "ss", "oo"])
    def test_spmv_square(self, matrix_fmt):
        torch.manual_seed(42)
        a_torch = torch.rand(30, 30) * (torch.rand(30, 30) < 0.8)
        x_torch = torch.rand(30)

        a_st = STensor.from_torch(a_torch).to_sparse(matrix_fmt)
        x_st = STensor.from_torch(x_torch)

        result = einsum("ij,j->i", a_st, x_st, format="d")
        expected = torch.mv(a_torch, x_torch)
        assert_close(result, expected)
```

Note how one test body sweeps several formats via `@pytest.mark.parametrize`:
CSR (`"ds"`), DCSR (`"ss"`), and COO (`"oo"`) all run the same assertion, so a
format-specific codegen bug surfaces as a single failing parameter. SpMM tests
follow the identical shape against `torch.matmul`; SDDMM and SpGEMM tests compare
against the corresponding dense expression.

:::{tip}
When you add a new operation, match this convention rather than inventing
expected values: seed the RNG, build the sparse operands, compute the dense
reference with the closest `torch.*` call, and `assert torch.allclose(...,
atol=1e-3, rtol=1e-3)`. It keeps tests robust to internal representation changes
and immediately portable across every format the op supports.
:::

## Test isolation: `TORCH_EXTENSIONS_DIR`

Generated kernels are cached on disk as compiled `.so` files, keyed by the
`(format_a, format_b, format_output)` triple. A stale cache from a previous run
can silently mask a codegen change — you edit the emitter, rerun, and the old
binary answers.

To make each session hermetic, `tests/conftest.py` defines a **session-scoped,
autouse** fixture that redirects the cache to a fresh temp directory for the whole
run and restores the original afterward:

```python
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def _set_torch_extensions_dir(tmp_path_factory):
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

Because it's `autouse=True`, no test opts in — every run compiles kernels from
scratch into a throwaway directory. That guarantees the tests exercise the code
you just changed, at the cost of paying JIT compilation once per session.

:::{warning}
Outside the test suite, Scorch's JIT cache defaults to
`~/.cache/torch_extensions/`. If you're debugging codegen by hand (not through
`pytest`), clear that directory after editing the emitter or C++ templates —
otherwise the memoized module and persistent `.so` will hide your change. Inside
the suite, the fixture above already handles this for you.
:::

## Known-limitation tests

The files matching `tests/test_scorch/test_known_compiler_gaps*.py` are, despite
the name, **passing regression tests**. Each pins a format / mode-order /
loop-order combination that was historically fragile — a transposed mode order, a
broadcast vector operand, a forced non-default loop order — and asserts the
compiler still lowers and executes it correctly:

```
test_known_compiler_gaps.py
test_known_compiler_gaps_spmm_spgemm.py
test_known_compiler_gaps_loop_orders_spmm_spgemm.py
test_known_compiler_gaps_dense_stensor_matmul.py
```

They exercise the lower-level compiler entry points directly — building
{doc}`CIN </compiler/index_notation>` by hand and calling `lower_and_exec_cin`,
sometimes forcing a specific schedule — so a regression in iterator analysis or
loop ordering fails loudly here instead of silently degrading a corner case. A
representative case pins SpMM with a transposed (`mode_order=[1, 0]`) operand and
checks it against `torch.matmul`:

```python
from scorch.ops import lower_and_exec_cin

result = lower_and_exec_cin(cin_stmt, (n, n), a, b)   # CSR B, transposed modes
expected = torch.matmul(a_torch, b_torch)
assert torch.allclose(result.to_torch(), expected, atol=1e-4, rtol=1e-4)
```

:::{admonition} Consult these before assuming a combination is unsupported
:class: tip
If you're about to add a feature that touches format handling, mode order, or
scheduling, read the relevant `test_known_compiler_gaps*.py` first. They are the
executable record of which combinations are known-good (and, by their absence,
which remain genuinely unsupported). When you fix a real gap, add its case here so
it can never silently regress.
:::

## Continuous integration

The GitHub Actions workflow (`.github/workflows/pytest.yml`, named `pytest`) runs
the suite on `ubuntu-latest` with Python 3.11.7 on every pull request to `main`.
It installs a CPU build of PyTorch, does an editable install (which compiles the
native `scorch_ops` extension with libgomp), and runs `pytest`, always uploading
`pytest-report.xml` as an artifact.

:::{note}
CI gates **tests only** — it does not run the lint/format/typecheck stage. Those
live in `pre-commit.sh` (Black, mypy, flake8) and are run locally. See
{doc}`the contributing guide </development/contributing>` for the full local
check-in workflow. CI also runs on Linux only; macOS is a supported dev platform
but is not exercised by CI.
:::

## Next steps

- {doc}`Contributing </development/contributing>` — the local check-in workflow
  (lint, type-check, commit conventions) your change should pass before a PR.
- {doc}`Building </development/building>` — how the `scorch_ops` extension and the
  JIT kernel cache are compiled, and when to rebuild.
- {doc}`Architecture </development/architecture>` — the compiler pipeline the
  known-limitation tests exercise.
