"""Setuptools command for building Scorch's PyTorch C++ extension."""

from __future__ import annotations

import glob
import os
import platform
from typing import Any, Iterable

import torch
from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths


def _extend_unique(target: Any, attribute: str, values: Iterable[str]) -> None:
    current = list(getattr(target, attribute, None) or [])
    for value in values:
        if value not in current:
            current.append(value)
    setattr(target, attribute, current)


class ScorchBuildExtension(BuildExtension):
    """Populate the platform-specific settings normally added by CppExtension."""

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
