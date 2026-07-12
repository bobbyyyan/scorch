# Installation

Scorch is a CPU sparse-tensor library that installs as an editable Python
package with a native C++ extension. This page covers the prerequisites, the two
supported install paths — a one-command first-time setup and a fast rebuild in an
existing environment — how the native `scorch_ops` extension and its OpenMP
runtime are linked on each platform, and how to verify the result.

Scorch runs on **Linux** and **macOS** (Apple Silicon and Intel). It is a **CPU
library — there is no CUDA build**; Windows is not supported.

## Prerequisites

| Requirement | Recommended | Notes |
|---|---|---|
| Python | **3.11** | Pinned by `setup.sh` and the `Pipfile`; **3.9** is the hard floor. |
| PyTorch | **2.0+** | `setup.sh` installs `torch>=2.6`; CI runs against 2.0.1. |
| NumPy | **`numpy<2`** | Required for compatibility with the bundled PyTorch libraries. |
| C++ compiler | system clang / gcc | macOS: system clang at `/usr/bin/clang++` (**not** Homebrew LLVM — torch ABI). Linux: system gcc/g++. |
| OpenMP | libomp / libgomp | macOS: `brew install libomp` for headers. Linux: `libgomp` (ships with torch). |
| ninja | latest | Build backend for the JIT-compiled kernels. |

:::{admonition} macOS: install OpenMP first
:class: tip
Scorch parallelizes its kernels with OpenMP. On macOS the compiler needs the
Homebrew libomp headers:

```bash
brew install libomp
```

If they are missing, `setup.sh` prints a warning telling you to run exactly this
command. At runtime Scorch links PyTorch's bundled `libomp.dylib`, not
Homebrew's — see [Native extension and OpenMP](#native-extension-and-openmp).
:::

## Path 1 — first-time setup with `setup.sh`

For a clean machine, `setup.sh` does everything: it creates an isolated
environment, pins compatible versions of PyTorch and NumPy, configures the
compiler, and builds Scorch in editable mode.

```console
$ git clone https://github.com/bobbyyyan/scorch && cd scorch
$ brew install libomp          # macOS only — OpenMP headers
$ bash setup.sh
```

When `conda` is available (the preferred path) `setup.sh` will:

1. Detect macOS and export `CC=/usr/bin/clang` / `CXX=/usr/bin/clang++` — the
   system clang is required for PyTorch ABI compatibility.
2. Back up any existing `scorch` environment, then create a fresh
   `conda create -n scorch python=3.11`.
3. Install `pybind11` via conda (pinning `mkl<2025` on x86-64), then install
   PyTorch and the rest via **pip** — not the conda `pytorch` channel, which
   pins an older torch that fails to compile under recent toolchains:

   ```bash
   pip install "torch>=2.6" "numpy<2" scipy ninja \
       black flake8 mypy pytest matplotlib pandas seaborn
   ```
4. On macOS, write conda `activate.d` / `deactivate.d` hooks so that every
   `conda activate scorch` re-pins `CC`/`CXX` to the system clang.
5. Upgrade `pip`/`setuptools` (for PEP 660 editable installs), then run
   `pip install -e . --no-build-isolation`.
6. Verify the build with `python3 -c "import scorch"`.

If `conda` is not found, `setup.sh` falls back to a `venv/` virtual environment
(Python ≥ 3.9) and installs from `requirements.txt`.

:::{admonition} Always activate the `scorch` environment first
:class: important
The native extension is built into the `scorch` environment. Activate it before
running any `python`, `pip`, or `pytest` command in this repo:

```bash
conda activate scorch
```

On macOS, activation auto-sets `CC`/`CXX` to the system clang through the hooks
`setup.sh` installed.
:::

## Path 2 — rebuild in an existing environment

Once the environment exists, reinstall Scorch in editable mode to rebuild after
editing Python source or anything in `csrc/`:

```bash
conda activate scorch
pip install -e . --no-build-isolation
```

This rebuilds both the editable Python package **and** the native `scorch_ops`
extension.

### Why `--no-build-isolation` is required

`setup.py` imports `torch` at module top level (it uses
`torch.utils.cpp_extension.BuildExtension` and `CppExtension` to compile the
native extension). With pip's default build isolation, the build runs in a fresh
throwaway environment where PyTorch is *not* installed, so `setup.py`
crashes on import before it can build anything. `--no-build-isolation` runs the
build in your current environment, where torch is present. This flag is mandatory
for every Scorch install and rebuild.

For the no-conda case, the manual sequence mirrors what `setup.sh` does:

```bash
python3 -m venv venv && source venv/bin/activate   # Python >= 3.9
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

(#native-extension-and-openmp)=
## Native extension and OpenMP

`setup.py` builds one native compilation unit — `csrc/ops.cpp` — into the
`scorch_ops` pybind extension. This hosts Scorch's hand-written **prebuilt
kernels** (it is distinct from the JIT **codegen kernels**, which are compiled at
runtime; see [JIT kernel cache](#jit-kernel-cache-and-rebuilds)). The base
compile flags are `-O3 -march=native -ffast-math -funroll-loops`.

OpenMP is linked differently per platform, deliberately preferring PyTorch's
bundled runtime so a second copy of OpenMP is never loaded alongside torch's:

**macOS**
: Compiles with `-Xpreprocessor -fopenmp` and the Homebrew libomp *include* path
  (headers only). Links the full path to torch's bundled `libomp.dylib` with an
  rpath, avoiding a dual-runtime conflict with Homebrew's libomp. Falls back to
  `-lomp` against Homebrew's copy only if torch's is absent.

**Linux**
: Compiles with `-fopenmp` and links torch's bundled `libgomp` (with an rpath),
  falling back to plain `-fopenmp` otherwise.

:::{note}
The repo also contains a `CMakeLists.txt` and `csrc/pybind.cpp`, but these are
legacy / IDE-indexing only. The supported build path is `pip install -e .`
(`setup.py` + torch's `BuildExtension`) — do not build via CMake.
:::

## Verifying the install

A minimal end-to-end check — build two sparse tensors, multiply them, and confirm
the result matches a dense PyTorch reference:

```python
import torch
import scorch

A = scorch.from_torch(
    torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]), "A"
)
B = scorch.from_torch(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), "B")

C = scorch.matmul(A, B)

expected = torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]) @ \
    torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
assert torch.allclose(C.to_torch(), expected, atol=1e-3, rtol=1e-3)
```

The first call JIT-compiles a kernel, so it is slower than subsequent runs.

The repo also ships a script that runs the full README workflow (`from_torch` →
{func}`~scorch.matmul` → {func}`~scorch.einsum`):

```console
$ python scripts/verify_quickstart.py
```

It prints two identical 3×2 tensors. For a broader check, run the test suite
(skip the slow performance tests with `-m "not perf"`):

```console
$ pytest -m "not perf"
```

## Troubleshooting

**`brew install libomp` and still no OpenMP.** Ensure libomp lives under
`/opt/homebrew/opt/libomp` (Apple Silicon) or `/usr/local/opt/libomp` (Intel) —
these are the two prefixes `setup.sh` and the build probe. Reinstall Scorch after
installing libomp so the headers are picked up.

**Changes to `csrc/` don't take effect.** The native extension is only rebuilt
when you reinstall. After editing anything in `csrc/`, rerun
`pip install -e . --no-build-isolation`. For header-only edits (e.g. `spmm.h`),
`touch csrc/ops.cpp` first so the build system sees a changed compilation unit.

(jit-kernel-cache-and-rebuilds)=
**A codegen change seems to have no effect (stale JIT cache).** Generic sparse
ops are lowered to C++ and JIT-compiled at runtime, then cached as `.so` files
keyed by the `(format_a, format_b, format_output)` combination. A stale cache can
mask a codegen edit. The cache lives under `TORCH_EXTENSIONS_DIR` (default
`~/.cache/torch_extensions/…`); point it at a fresh directory to force
recompilation:

```bash
export TORCH_EXTENSIONS_DIR=$(mktemp -d)
```

The test suite already does this automatically via a session fixture, so tests
never see a stale kernel.

**macOS: `scorch_ops` fails to load (`libomp` not found).** On some Apple Silicon
setups the dynamic loader resolves libomp to the wrong copy. Repoint the extension
at torch's bundled `libomp.dylib` with `install_name_tool -change` — this does not
require a rebuild.

## Next steps

- {doc}`Quickstart </getting_started/quickstart>` — your first sparse matmul and
  einsum, end to end.
- {doc}`Building from source </development/building>` — deeper detail on the
  build system, compile flags, and the JIT codegen path.
