# Scorch documentation

The source for the official Scorch documentation site, built with
[Sphinx](https://www.sphinx-doc.org/) and the
[PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/). Pages are
authored in [MyST Markdown](https://myst-parser.readthedocs.io/); the API
reference is generated from the docstrings in `src/scorch/` via `autodoc`.

## Building locally

From the repository's `scorch` conda environment (which already has `torch` and
the compiled extension), everything you need is one install away:

```bash
cd docs
pip install -r requirements.txt
make html
open build/html/index.html   # or: xdg-open build/html/index.html
```

The build is configured to treat warnings as errors (`-W --keep-going -n`), so a
clean `make html` means the site is publishable.

### Live preview

For an auto-rebuilding preview while you edit:

```bash
pip install sphinx-autobuild
make livehtml   # serves http://127.0.0.1:8000 and reloads on save
```

### Other targets

```bash
make linkcheck   # validate external links
make clean       # remove build/
```

## How the build works

- **Import-free of C++.** `conf.py` adds `../../src` to `sys.path` and mocks the
  native `scorch_ops` extension (`autodoc_mock_imports`), so the docs build on
  any machine with `torch` + the doc requirements — no C++ toolchain, no
  `pip install -e .`, no recompilation.
- **Content lives in `source/`.** Narrative pages are `.md` (MyST); the API
  reference under `source/api/` uses `autosummary` tables that generate one stub
  page per public object from its docstring.
- **Brand assets** (logo, favicon) and custom styling live in
  `source/_static/`.

## Deployment

- **GitHub Pages** — `.github/workflows/docs.yml` builds and publishes on every
  push to `main`.
- **ReadTheDocs** — `.readthedocs.yaml` at the repo root configures the RTD
  build.

Both install a CPU-only `torch` wheel so `autodoc` can import `scorch`.

## Writing docs

- Prose pages are MyST Markdown. Useful extensions are enabled in `conf.py`
  (`colon_fence`, `dollarmath`, `deflist`, `sphinx-design` cards/grids/tabs,
  `sphinx-copybutton`).
- Keep code examples runnable and verify sparse results against a PyTorch
  reference, matching the project's testing convention.
- To add a new public API object to the reference, add it to the appropriate
  `autosummary` block under `source/api/` — the stub page is generated
  automatically from its docstring.
