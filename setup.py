import os
import platform
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Handle OpenMP flags for different platforms
extra_compile_args = [
    "-O3",
    "-march=native",
    "-ffast-math",
    "-funroll-loops",
]
extra_link_args = []

if platform.system() == "Darwin":
    # macOS: use Xpreprocessor flag for OpenMP
    extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])

    # Use PyTorch's bundled libomp to avoid runtime conflicts
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    torch_omp = os.path.join(torch_lib_path, "libomp.dylib")

    if os.path.exists(torch_omp):
        # Link against PyTorch's libomp using full path to avoid linker finding Homebrew's
        extra_link_args.append(torch_omp)
        # Also add rpath so it finds the right library at runtime
        extra_link_args.append(f"-Wl,-rpath,{torch_lib_path}")
        # Also need OpenMP headers - check Homebrew as fallback for headers only
        for header_path in ["/opt/homebrew/opt/libomp/include", "/usr/local/opt/libomp/include"]:
            if os.path.exists(header_path):
                extra_compile_args.append(f"-I{header_path}")
                break
    else:
        # Fall back to Homebrew's libomp
        libomp_paths = [
            "/opt/homebrew/opt/libomp",  # Apple Silicon Homebrew
            "/usr/local/opt/libomp",      # Intel Mac Homebrew
        ]
        for path in libomp_paths:
            if os.path.exists(path):
                extra_compile_args.append(f"-I{path}/include")
                extra_link_args.extend(["-lomp", f"-L{path}/lib"])
                break
        else:
            extra_link_args.append("-lomp")
else:
    # Linux: standard OpenMP support
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")

ext_modules = [
    CppExtension(
        "scorch_ops",
        [
            "csrc/ops.cpp",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="scorch",
    version="0.0.1",
    author="Bobby Yan",
    author_email="bobbyy@cs.stanford.edu",
    description="Scorch: A library for sparse machine learning",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bobbyyyan/scorch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    install_requires=[
        "torch",
    ],
    cmdclass={"build_ext": BuildExtension},
)
