from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

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
            "-fno-signed-zeros",
            "-fopenmp",
            "-funsafe-math-optimizations",
            "-freciprocal-math",
            "-ftree-vectorize",
            "-flto",
            "-funroll-loops",
        ],
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
