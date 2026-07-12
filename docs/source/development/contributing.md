# Contributing

Thanks for helping improve Scorch. This page covers the conventions that keep the
codebase consistent: coding style, the pre-commit checks, how to test a change
against a PyTorch reference, and how to shape commits and pull requests. For
environment setup and rebuilding the native extension see
{doc}`Building </development/building>`; for the test suite in depth see
{doc}`Testing </development/testing>`.

Before you start, activate the project's conda environment — every `python`,
`pip`, and `pytest` invocation depends on the `scorch_ops` C++ extension built
into it:

```bash
conda activate scorch
```

## Coding style

Scorch follows a small, strict set of conventions so that generated code and
hand-written kernels read the same way across the compiler stages.

- **Indentation:** four spaces, never tabs.
- **Formatting:** [Black](https://black.readthedocs.io/) with its default
  **88-column** line length. Run Black before submitting; the pre-commit check
  rejects unformatted code.
- **Naming:**
  - `snake_case` for modules, functions, fixtures, and variables
    (`lower_and_exec_cin`, `from_torch`, `resolve_prebuilt_matmul`).
  - `PascalCase` for classes — the sparse tensor {class}`~scorch.STensor`, the
    compiler's `TensorVar`, `ForAll`, `TensorAssign`, and {class}`~scorch.TensorFormat`.
  - `UPPER_CASE` for module-level constants (e.g. the `LevelType` /
    `_STR_TO_LEVEL_TYPE` tables in `format.py`).

Keep each compiler transformation in the stage where it belongs — CIN
construction in `cin.py`, lowering in `cin_lowerer.py`, scheduling in
`scheduler.py`, C++ emission in `codegen.py` — and document any non-obvious
scheduling or storage assumption with a comment. The compiler files carry
genuinely complex control flow, which is why the lint config allows a high
cyclomatic complexity (see below); it is not a license to add more.

## Pre-commit checks

Run the full local gate before every commit:

```bash
bash pre-commit.sh
```

It runs, in order, against `src`:

1. A Python version guard (**>= 3.9**).
2. `black --check --diff src` — formatting.
3. `mypy --install-types --non-interactive --show-error-codes
   --check-untyped-defs --pretty src` — type checking, including untyped
   function bodies.
4. `flake8 src` — linting.

The Flake8 configuration lives in `.flake8`:

```ini
[flake8]
ignore = E501,E203,W503
max-line-length = 88
max-complexity = 39
```

- **`max-line-length = 88`** matches Black.
- **`E501`** (line too long), **`E203`** (whitespace before `:`), and **`W503`**
  (line break before a binary operator) are ignored because they conflict with
  Black's formatting — let Black own those decisions.
- **`max-complexity = 39`** is deliberately high to accommodate the large lowering
  and scheduling functions. Prefer to stay well under it in new code.

:::{note}
CI runs the **test suite only** (Linux, Python 3.11, CPU PyTorch) — it does *not*
run Black, mypy, or Flake8. `pre-commit.sh` is a local gate, so run it yourself;
a lint or type error will not be caught for you on the pull request.
:::

If you edited anything under `csrc/`, rebuild the extension before testing so the
checks and tests exercise your change:

```bash
pip install -e . --no-build-isolation
```

For header-only edits (`spmm.h`, `kernels.h`, `scorch_policy.h`, …), `touch
csrc/ops.cpp` first so the single compilation unit is recompiled.

## Testing your change

Add regression tests **near the affected subsystem**: kernel behavior under
`tests/test_scorch/test_kernels*.py`, formats under `tests/test_scorch/test_format_*.py`,
codegen under `tests/test_scorch/codegen/`. Pytest discovers files named
`test_*.py` and functions named `test_*`.

The correctness convention throughout Scorch is to compare a generated-kernel
result against a **PyTorch reference** with explicit tolerances, rather than
hardcoding expected numbers — floating-point summation order differs between the
sparse kernel and dense PyTorch, so exact equality is the wrong test.

```python
import torch
import scorch

A_dense = torch.tensor([[1.0, 0.0, 2.0],
                        [0.0, 3.0, 0.0],
                        [4.0, 0.0, 5.0]])
B_dense = torch.tensor([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]])

A = scorch.from_torch(A_dense, "A")
B = scorch.from_torch(B_dense, "B")

C = scorch.matmul(A, B)

# Compare against the dense PyTorch reference with the project tolerances.
expected = A_dense @ B_dense
assert torch.allclose(C.to_torch(), expected, atol=1e-3, rtol=1e-3)
```

The suite's `assert_close()` helper wraps exactly this pattern (`atol = rtol =
1e-3`); match it in new tests.

A few practical notes:

- The **first** run of a generic sparse op JIT-compiles a C++ kernel, so the
  first test to hit a new `(format_a, format_b, format_output)` combination is
  slow; subsequent runs reuse the cached `.so`.
- Mark performance-only tests with `@pytest.mark.perf`, and skip them during
  normal iteration with `pytest -m "not perf"`.
- When you change codegen or a C++ template, aggressive kernel memoization can
  mask your change. The test suite sidesteps this with a session fixture that
  points `TORCH_EXTENSIONS_DIR` at a fresh temp dir; if you see a stale result
  outside the suite, clear the torch extensions build directory.

See {doc}`Testing </development/testing>` for running focused files, the CI
matrix, and the caching details.

## Commit style

Follow the scoped, focused commit style visible in the history — a type, an
optional scope in parentheses, then an imperative summary:

```text
perf(spmm): register-block dual-path kernel for narrow free-dim
bench(ae): add sparse-autoencoder harness
docs: clarify the format-string per-character split
```

Keep each commit focused on one change. Separate a performance kernel from the
benchmark that measures it, and separate a refactor from a behavior change, so
each can be reviewed and reverted independently.

:::{warning}
Do **not** add `Co-Authored-By:` trailers to commit messages — this is a project
preference. Also never commit generated extensions (`*.so`), datasets, cache
directories, or benchmark output CSVs; these are already covered by
`.gitignore`.
:::

## Pull requests

A good Scorch pull request makes the change easy to trust:

- **Explain the behavior change** — what was wrong or missing, and what the code
  now does. If it touches the compiler, name the stage (lowering, scheduling,
  codegen).
- **List the verification commands** you ran, so a reviewer can reproduce them —
  e.g. `pytest tests/test_scorch/test_kernels.py -q` and `bash pre-commit.sh`.
- **Link any relevant issue.**
- **Include benchmark data for performance work.** Scorch's performance policy is
  that an optimization must *generalize* — neutral-or-better across narrow- and
  wide-`k` SpMM, small and large row counts, sparse and near-dense inputs, and
  the real workload families (GCN, autoencoder, attention) — with **no
  regressions**. Show a before/after across a shape/format/density grid, not a
  single winning shape. If a change only helps a sub-regime, gate it behind a
  runtime condition that provably cannot fire on the shapes it would hurt.
- **Call out platform-specific OpenMP behavior.** OpenMP linking differs by
  platform (macOS links PyTorch's bundled `libomp.dylib`; Linux uses the bundled
  `libgomp`), and thread-pool behavior differs across CPUs. If your change
  touches threading or affects one platform differently, say so explicitly — and
  note that CI only exercises Linux, so macOS behavior needs your local
  measurement.

Attach plots only when the output is genuinely visual (e.g. a benchmark sweep);
otherwise a small table in the PR body is clearer.

## Next steps

- {doc}`Building </development/building>` — environment setup, the native
  extension, and the rebuild loop.
- {doc}`Testing </development/testing>` — the suite, markers, CI, and kernel-cache
  isolation.
- {doc}`Architecture </development/architecture>` — where each compiler stage
  lives, so your change lands in the right place.
