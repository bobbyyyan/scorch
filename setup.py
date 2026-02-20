import platform
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Handle OpenMP flags for different platforms
if platform.system() == "Darwin":
    # macOS: use Xpreprocessor flag and link against Homebrew libomp
    openmp_compile_args = ["-Xpreprocessor", "-fopenmp"]
    openmp_link_args = ["-lomp", "-L/opt/homebrew/opt/libomp/lib"]
else:
    # Linux: standard OpenMP support
    openmp_compile_args = ["-fopenmp"]
    openmp_link_args = ["-fopenmp"]

ext_modules = [
    CppExtension(
        "scorch_ops",
        [
            "csrc/ops.cpp",
        ],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            # "-fno-signed-zeros",
            *openmp_compile_args,
            # "-funsafe-math-optimizations",
            # "-freciprocal-math",
            # "-ftree-vectorize",
            # "-flto",
            "-funroll-loops",
        ],
        extra_link_args=openmp_link_args,
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
