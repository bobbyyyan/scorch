"""Setuptools command for building Scorch's PyTorch C++ extension."""

from __future__ import annotations

import glob
import os
import platform
import subprocess
from typing import Any, Iterable, Optional

import torch
from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths


def _extend_unique(target: Any, attribute: str, values: Iterable[str]) -> None:
    current = list(getattr(target, attribute, None) or [])
    for value in values:
        if value not in current:
            current.append(value)
    setattr(target, attribute, current)


def _dylib_install_name(path: str) -> Optional[str]:
    """The install name recorded inside a dylib, or None if it cannot be read."""
    try:
        result = subprocess.run(
            ["otool", "-D", path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    # otool -D prints the path it was given, then the install name.
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if len(lines) >= 2 else None


class ScorchBuildExtension(BuildExtension):
    """Populate the platform-specific settings normally added by CppExtension."""

    def build_extension(self, ext: Any) -> None:
        """Link as usual, then make the OpenMP reference resolvable on macOS.

        PyTorch ships its own ``libomp.dylib`` whose *install name* is
        ``/opt/llvm-openmp/lib/libomp.dylib`` -- an absolute path from the machine
        PyTorch was built on, which does not exist here. A Mach-O link records the
        dependency's install name, not the path the linker was handed, so linking
        against torch's copy produces an extension that asks the loader for
        ``/opt/llvm-openmp`` and fails to import. The ``-rpath`` we pass cannot help:
        an rpath only resolves references that are written ``@rpath/...``.

        So the reference is rewritten after the link to the copy we actually linked
        against. Doing it here rather than by hand means a fresh clone builds and
        imports; the alternative was a manual ``install_name_tool`` step that every
        rebuild silently needed and that failed as soon as the recorded path changed.
        """
        super().build_extension(ext)
        if platform.system() != "Darwin" or ext.name != "scorch_ops":
            return
        torch_omp = os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")
        if not os.path.exists(torch_omp):
            return
        recorded = _dylib_install_name(torch_omp)
        if not recorded or recorded == torch_omp:
            return  # already resolvable, nothing to rewrite
        built = self.get_ext_fullpath(ext.name)
        if not os.path.exists(built):
            return
        try:
            subprocess.run(
                ["install_name_tool", "-change", recorded, torch_omp, built],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            # A failed rewrite leaves exactly the extension we had before it, so the
            # build should not die here -- the import error that follows is clearer.
            pass

    def build_extensions(self) -> None:
        for extension in self.extensions:
            if extension.name != "scorch_ops":
                continue

            _extend_unique(extension, "include_dirs", include_paths())
            _extend_unique(extension, "library_dirs", library_paths())
            _extend_unique(
                extension,
                "libraries",
                ["c10", "torch", "torch_cpu", "torch_python"],
            )

            compile_args = []
            link_args = []
            torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

            if os.environ.get("SCORCH_BUILD_TUNE_HOOKS"):
                compile_args.append("-DSCORCH_TUNE_HOOKS")

            if platform.system() == "Darwin":
                compile_args.extend(["-Xpreprocessor", "-fopenmp"])
                torch_omp = os.path.join(torch_lib_path, "libomp.dylib")
                if os.path.exists(torch_omp):
                    link_args.extend([torch_omp, f"-Wl,-rpath,{torch_lib_path}"])
                    for header_path in (
                        "/opt/homebrew/opt/libomp/include",
                        "/usr/local/opt/libomp/include",
                    ):
                        if os.path.exists(header_path):
                            compile_args.append(f"-I{header_path}")
                            break
                else:
                    for libomp_path in (
                        "/opt/homebrew/opt/libomp",
                        "/usr/local/opt/libomp",
                    ):
                        if os.path.exists(libomp_path):
                            compile_args.append(f"-I{libomp_path}/include")
                            link_args.extend(["-lomp", f"-L{libomp_path}/lib"])
                            break
                    else:
                        link_args.append("-lomp")
            else:
                compile_args.append("-fopenmp")
                gomp_libraries = glob.glob(os.path.join(torch_lib_path, "libgomp*.so*"))
                if gomp_libraries:
                    link_args.extend(
                        [gomp_libraries[0], f"-Wl,-rpath,{torch_lib_path}"]
                    )
                else:
                    link_args.append("-fopenmp")

            _extend_unique(extension, "extra_compile_args", compile_args)
            _extend_unique(extension, "extra_link_args", link_args)

        super().build_extensions()
