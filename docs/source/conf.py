"""Sphinx configuration for the Scorch documentation site.

The docs are authored in MyST Markdown and rendered with the
``pydata-sphinx-theme`` — the same toolchain family behind the NumPy, pandas,
and SciPy documentation. The build is intentionally *import-free*: nothing here
imports :mod:`scorch` or the compiled ``scorch_ops`` extension, so the site can
be built on any machine (and in CI / ReadTheDocs) with only the lightweight
documentation dependencies in ``docs/requirements.txt``.
"""

from __future__ import annotations

import datetime
import os
import sys

# Make the ``scorch`` package importable for autodoc without installing it.
# We only need the Python sources on the path; the compiled ``scorch_ops``
# extension is mocked (see ``autodoc_mock_imports`` below), so the API
# reference can be generated on any machine — including ReadTheDocs and CI —
# without a C++ toolchain.
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# -- Project information -----------------------------------------------------

project = "Scorch"
author = "Bobby Yan"
copyright = f"{datetime.date.today().year}, {author}"

# Keep in sync with ``src/scorch/__init__.py``'s ``__version__``.
release = "0.0.1"
version = "0.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    # API reference is generated from source docstrings.
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Diagrams (client-side mermaid.js; works on static hosts).
    "sphinxcontrib.mermaid",
]

# -- autodoc / autosummary / napoleon ----------------------------------------

# Generate the per-object stub pages referenced by ``.. autosummary::`` blocks.
autosummary_generate = True
autosummary_imported_members = False

# The native pybind extension is not built in the docs environment. Mock it so
# importing ``scorch`` (which pulls it in transitively) succeeds. ``torch`` is a
# real dependency of the docs build and is imported normally.
autodoc_mock_imports = ["scorch_ops"]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
# Render type hints into the parameter descriptions rather than the signature,
# which keeps signatures readable (the NumPy/pandas convention).
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True

# Scorch docstrings follow the NumPy style.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Link Scorch's own types when they appear as parameter/return types in
# docstrings, so the API reference cross-references them nicely.
napoleon_type_aliases = {
    "STensor": ":class:`~scorch.STensor`",
    "TensorFormat": ":class:`~scorch.TensorFormat`",
    "LevelType": ":class:`~scorch.format.LevelType`",
    "LevelFormat": ":class:`~scorch.format.LevelFormat`",
}

# Some docstring parameter/return types name external or internal objects that
# are intentionally not part of the documented API (PyTorch types, builtins,
# duck-typed hints, compiler-internal storage classes). Silence the nitpicky
# "reference target not found" for exactly those so a strict (-n -W) build stays
# clean without littering the prose with unresolvable links.
nitpick_ignore_regex = [
    (r"py:.*", r"torch\..*"),
    (r"py:.*", r"(numpy|np)\..*"),
    (r"py:.*", r"scorch\.compiler\..*"),
    (r"py:(obj|data)", r"(True|False|None|Ellipsis)"),
    (r"py:class",
     r"(array-like|array_like|optional|scalar|callable|iterable|sequence|"
     r"keyword-only|path-like|file-like)"),
    (r"py:.*", r".*(TensorStorage|TensorIndex)"),
    (r"py:func", r"scorch\.utils\.parse_format"),
    (r"py:.*", r"enum\..*"),
    (r"py:attr", r"LevelType\..*"),
    (r"py:class", r"env"),
    # Unqualified in-house names referenced inside docstrings (they resolve for
    # a human reader from context, but Sphinx can't pin them without a module
    # scope). napoleon_type_aliases handles the common type fields; this covers
    # the rest in free-text / See Also sections.
    (r"py:class", r"(STensor|TensorFormat|LevelFormat|LevelType)"),
    (r"py:(func|meth)",
     r"(spmv|to_torch|to_dense|to_sparse|from_torch|from_coo|from_csr|"
     r"change_mode_order|precompile_kernels)"),
]

# MyST (Markdown) feature set.
myst_enable_extensions = [
    "colon_fence",     # ::: fenced directives
    "deflist",         # definition lists
    "dollarmath",      # $inline$ and $$block$$ math
    "amsmath",         # LaTeX amsmath environments
    "attrs_inline",    # inline attributes, e.g. {.class}
    "substitution",    # |substitutions|
    "tasklist",        # - [ ] task lists
    "fieldlist",       # :field: lists (used on API pages)
    "linkify",         # bare URLs become links
]
myst_heading_anchors = 4  # auto-anchor headings up to <h4> for deep-linking

# Substitutions available in every page.
myst_substitutions = {
    "release": release,
    "min_python": "3.11",
    "min_torch": "2.0",
}

# autosectionlabel can collide across files; prefix labels with the document
# path to keep them unique, and don't warn on the (rare) remaining dupes.
autosectionlabel_prefix_document = True
suppress_warnings = ["autosectionlabel.*"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_kb", "**/_kb"]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# The landing page.
root_doc = "index"

# Show "todo" notes only when explicitly enabled (off in production builds).
todo_include_todos = False

# -- Intersphinx: cross-link to Python / PyTorch / NumPy docs -----------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
# Don't fail the build if an inventory can't be fetched (offline / CI).
intersphinx_disabled_reftypes = ["*"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Scorch"
html_static_path = ["_static"]
html_css_files = ["css/scorch.css"]
html_favicon = "_static/img/favicon.svg"
html_show_sourcelink = False
html_copy_source = False
html_last_updated_fmt = "%b %d, %Y"

GITHUB_URL = "https://github.com/bobbyyyan/scorch"
PAPER_URL = "https://ieeexplore.ieee.org/abstract/document/11394842"

html_theme_options = {
    "logo": {
        "image_light": "_static/img/scorch-logo.svg",
        "image_dark": "_static/img/scorch-logo-dark.svg",
        "alt_text": "Scorch",
    },
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "show_prev_next": True,
    "header_links_before_dropdown": 6,
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "GitHub",
            "url": GITHUB_URL,
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "CGO 2026 Paper",
            "url": PAPER_URL,
            "icon": "fa-solid fa-file-lines",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "announcement": (
        "Scorch is research software (v0.0.1) accompanying the "
        f'<a href="{PAPER_URL}">CGO&nbsp;2026 paper</a> — APIs may change.'
    ),
}

html_context = {
    "github_user": "bobbyyyan",
    "github_repo": "scorch",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
}

# Sidebar layout: primary nav on the left, nothing extra on the right except the
# in-page TOC (handled by the theme). Landing page gets no sidebars.
html_sidebars = {
    "index": [],
}

# -- OpenGraph (social cards) ------------------------------------------------

ogp_site_name = "Scorch Documentation"
ogp_description_length = 200
ogp_enable_meta_description = True

# -- copybutton: strip prompts so pasted snippets Just Work ------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regex = True
copybutton_only_copy_prompt_lines = False
