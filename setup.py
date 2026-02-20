import os
import platform
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

    # Check for libomp in common locations
    libomp_paths = [
        "/opt/homebrew/opt/libomp",  # Apple Silicon Homebrew
        "/usr/local/opt/libomp",      # Intel Mac Homebrew
    ]

    libomp_path = None
    for path in libomp_paths:
        if os.path.exists(path):
            libomp_path = path
            break

    if libomp_path:
        extra_compile_args.append(f"-I{libomp_path}/include")
        extra_link_args.extend(["-lomp", f"-L{libomp_path}/lib"])
    else:
        # Fall back to system paths, libomp may be installed elsewhere
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
