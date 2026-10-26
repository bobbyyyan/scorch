"""PEP 517 build backend: setuptools' own, loaded from the project root.

``[tool.setuptools.cmdclass]`` in pyproject.toml names
``scorch_build.ScorchBuildExtension``, a module at the project root.  With
``package-dir = {"" = "src"}`` setuptools looks for it in ``src/`` and then through
the import system, and the process a build frontend (pip, ``python -m build``) runs
the backend in cannot import from the root, so every clean install failed with
``ModuleNotFoundError: scorch_build``.  pyproject.toml's ``backend-path = ["."]``
makes the frontend import this module from the root, and with it every other
top-level module there, ``scorch_build`` included: pip's and build's
``pyproject_hooks`` put a finder for the backend path first on ``sys.meta_path``
(older versions prepend the path to ``sys.path``).  This is the in-tree wrapper
setuptools documents for build_meta tweaks; every hook is setuptools.build_meta's.
"""

from setuptools.build_meta import *  # noqa: F401,F403
