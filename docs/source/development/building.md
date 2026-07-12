# Building from source

This page covers the developer build in detail: how the editable install works,
why the native `scorch_ops` extension has to be rebuilt after you touch `csrc/`,
and how Scorch's two-tier compile cache can quietly mask codegen changes if you
don't clear it. If you only want to *use* Scorch, the shorter path is on the
{doc}`installation page </getting_started/installation>`; this page is for people
hacking on the library itself.

Scorch has two independently compiled C++ layers, and most build confusion comes
from not knowing which one you're rebuilding:

1. **The prebuilt `scorch_ops` extension** — hand-optimized kernels in `csrc/`
   (`ops.cpp`, `spmm.h`, `kernels.h`, …), compiled once at install time by
   `setup.py`. Rebuilt with `pip install -e .`.
2. **JIT-generated kernels** — the generic sparse path lowers each operation to a
   C++ source string and compiles it at runtime with `load_inline`, caching the
   `.so` under `TORCH_EXTENSIONS_DIR`. See {doc}`/compiler/codegen`.

Editing `csrc/` affects (1); editing the compiler affects (2); they have separate
build steps and separate caches.

## Activate the `scorch` conda env first

Every `pip`, `pytest`, and `python` invocation in this repo assumes the `scorch`
conda environment, because the `scorch_ops` extension is compiled *into* that
environment against its exact PyTorch build:

```bash
conda activate scorch
```

On macOS, activating the env also runs an `activate.d` hook that re-pins the
compiler to system clang (`CC=/usr/bin/clang`, `CXX=/usr/bin/clang++`). This is
deliberate: torch's C++ ABI is built with the system toolchain, and building
`scorch_ops` with Homebrew LLVM instead produces link/runtime mismatches. If you
build in the wrong environment — or with the wrong compiler — imports fail in
ways that look like bugs in Scorch but are really ABI skew.

:::{note}
If you don't have conda, `setup.sh` falls back to a `venv/` and
`requirements.txt` (Python ≥ 3.9). The conda path is canonical for development;
the venv path is the no-conda fallback. Full first-time setup lives on the
{doc}`installation page </getting_started/installation>`.
:::

## The editable install: `pip install -e . --no-build-isolation`

Once the environment exists, this single command rebuilds both the editable
Python package and the native `scorch_ops` extension:

```bash
conda activate scorch
pip install -e . --no-build-isolation
```

### Why `--no-build-isolation` is mandatory

`setup.py` imports torch at module top level to configure the extension:

```python
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
```

By default, pip builds in an isolated, freshly created virtual environment where
your project's dependencies are **not** installed. In that clean environment the
`import torch` at the top of `setup.py` raises `ModuleNotFoundError` and the build
aborts before it ever compiles a line of C++. `--no-build-isolation` tells pip to
run the build in your *current* environment — the `scorch` env, where torch is
already present — so `setup.py` can import it and hand its compiler/linker flags
to `BuildExtension`.

:::{warning}
Omitting `--no-build-isolation` is the single most common build failure. The
error surfaces as a torch import error *during the build*, not at runtime, which
misleads people into reinstalling torch. The fix is the flag, not the reinstall.
:::

## Rebuild `scorch_ops` after editing `csrc/`

The extension defined in `setup.py` compiles exactly one translation unit,
`csrc/ops.cpp`:

```python
CppExtension(
    "scorch_ops",
    ["csrc/ops.cpp"],
    extra_compile_args=["-O3", "-march=native", "-ffast-math", "-funroll-loops"],
    extra_link_args=...,   # platform-specific OpenMP, see below
)
```

`ops.cpp` `#include`s the header-only kernels (`spmm.h`, `kernels.h`, `header.*`,
`scorch_policy.h`). Any change to those headers or to `ops.cpp` requires a rebuild
before it takes effect:

```bash
pip install -e . --no-build-isolation
```

:::{admonition} Header-only edits may not trigger a rebuild
:class: tip
Because only `csrc/ops.cpp` is a compilation unit, editing a *header* it includes
(e.g. `spmm.h`) can leave the build system thinking nothing changed. Force a
recompile by touching the translation unit:

```bash
touch csrc/ops.cpp
pip install -e . --no-build-isolation
```
:::

:::{note}
`CMakeLists.txt` and `csrc/pybind.cpp` exist in the tree but are **legacy /
IDE-indexing only** — they are not the supported build path. Always build through
`pip` / `setup.py`.
:::

## The two-tier JIT cache and `TORCH_EXTENSIONS_DIR`

Operations that don't hit a prebuilt kernel go through the compiler pipeline
({doc}`/compiler/codegen`): Scorch emits a C++ source string and compiles it at
runtime with `torch.utils.cpp_extension.load_inline`. To avoid recompiling on
every call, results are memoized at two levels:

Module cache (in-memory)
: Keyed by the format triple `(format_a, format_b, format_output)` in `ops.py`.
  Lives only for the process.

On-disk `.so` cache
: torch writes the compiled shared objects under `TORCH_EXTENSIONS_DIR` (default
  `~/.cache/torch_extensions/…`). This persists across processes, so the compile
  cost for a given kernel is paid once and reused forever after.

Both caches are what make Scorch fast to *use*. They are also what make codegen
changes frustrating to *develop*: after you edit the compiler, an unchanged cache
key will happily serve you the **old** compiled kernel, and your change appears to
do nothing.

:::{warning}
When you change codegen or a C++ template and your edit seems to have no effect,
suspect a stale cache first. Clear the JIT build directory so the next run
recompiles from your new source:

```bash
rm -rf ~/.cache/torch_extensions        # or wherever TORCH_EXTENSIONS_DIR points
```

Or point `TORCH_EXTENSIONS_DIR` at a throwaway directory for the session:

```bash
export TORCH_EXTENSIONS_DIR=$(mktemp -d)
```
:::

The test suite handles this automatically. A session-scoped autouse fixture in
`tests/conftest.py` redirects `TORCH_EXTENSIONS_DIR` to a fresh temporary
directory for the whole run and restores the original afterward — so `pytest`
always exercises freshly generated kernels rather than a cache from a previous
edit. When reproducing a codegen bug outside pytest, replicate that isolation
manually with the commands above.

## Platform-specific OpenMP linking

Scorch's kernels are OpenMP-parallel, and the trickiest part of the build is
linking OpenMP *without* introducing two conflicting OpenMP runtimes into the same
process. `setup.py` handles this per platform, preferring the copy that PyTorch
itself already loaded:

**macOS (Darwin).** Compiled with `-Xpreprocessor -fopenmp` plus a headers-only
include path from Homebrew's libomp (`/opt/homebrew/opt/libomp/include` or
`/usr/local/opt/libomp/include`). For *linking*, it prefers torch's bundled
`libomp.dylib` — linking the full path and adding an rpath so it resolves at
runtime — precisely to stop the linker from picking up Homebrew's separate libomp
and creating a dual-runtime conflict. If torch's copy is missing, it falls back to
`-lomp` against Homebrew's libomp.

**Linux.** Compiled with `-fopenmp`. For linking, it prefers torch's bundled
`libgomp*.so*` (globbed from torch's `lib/` directory) plus an rpath, again to
avoid a PyTorch-copy-vs-system-copy conflict. Fallback is plain `-fopenmp`.

:::{note}
This is why `brew install libomp` is a macOS prerequisite even though the runtime
uses torch's libomp: Homebrew supplies the **headers** the compiler needs, while
the **runtime** library is torch's, to keep a single OpenMP runtime in the
process.
:::

## Verify the build

After building, confirm the extension imports and a real op runs end to end:

```bash
python scripts/verify_quickstart.py
```

or inline — this mirrors the project's correctness convention of checking against
a dense PyTorch reference:

```python
import torch
import scorch

A = scorch.from_torch(
    torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]), "A"
)
B = scorch.from_torch(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), "B")

C = scorch.matmul(A, B)
ref = torch.matmul(
    torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]),
    torch.tensor([[1., 2.], [3., 4.], [5., 6.]]),
)
assert torch.allclose(C.to_torch(), ref, atol=1e-3, rtol=1e-3)
```

Then run the suite (skip the slow perf-marked tests for a quick check):

```bash
pytest -m "not perf"
```

## Troubleshooting

`ModuleNotFoundError: No module named 'torch'` *during the build*
: You omitted `--no-build-isolation`. pip is building in an isolated env without
  torch. Re-run `pip install -e . --no-build-isolation`.

`import scorch` fails after a `csrc/` edit, or your kernel change has no effect
: The `scorch_ops` extension wasn't rebuilt. Run
  `pip install -e . --no-build-isolation`. If you only edited a header, first
  `touch csrc/ops.cpp` to force the recompile.

A codegen/compiler change appears to do nothing
: A stale JIT kernel is being served from cache. Clear `TORCH_EXTENSIONS_DIR`
  (`rm -rf ~/.cache/torch_extensions`) or point it at a fresh temp directory, then
  re-run. See {doc}`/compiler/codegen`.

`warning: libomp not found` from `setup.sh` on macOS
: Install the OpenMP headers: `brew install libomp`. The runtime still links
  torch's bundled `libomp.dylib`; Homebrew only supplies headers.

Wrong compiler picked up on macOS (link/ABI errors, symbol mismatches)
: Make sure you're in the `scorch` conda env so the `activate.d` hook pins
  `CC`/`CXX` to system clang. Building with Homebrew LLVM against torch's
  system-clang ABI produces exactly these errors.

`scorch_ops` fails to `dlopen` on Apple Silicon (libomp resolves to the wrong path)
: The rpath can resolve to a foreign libomp (e.g. `/opt/llvm-openmp`). You can
  repoint the load command in place with
  `install_name_tool -change @rpath/libomp.dylib <torch>/lib/libomp.dylib` on the
  built `.so`, without a full rebuild.

`libittnotify.so` / VTune ITT errors on x86_64 macOS/Linux
: Newer MKL (2025+) introduced this dependency. `setup.sh` pins `mkl<2025` on
  x86_64 to avoid it; do the same in a manual environment.

## Next steps

- {doc}`/getting_started/installation` — first-time setup, prerequisites, and the
  conda vs. venv paths.
- {doc}`/compiler/codegen` — how JIT kernels are generated and compiled, and what
  the cache actually stores.
- {doc}`/development/contributing` — dev workflow, style, and the performance
  policy for kernel changes.
