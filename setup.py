from setuptools import setup, find_packages

setup(
    name="scorch",
    version="0.0.1",
    description="A library for sparse machine learning",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Bobby Yan",
    author_email="bobbyy@cs.stanford.edu",
    url="https://github.com/bobbyyyan/scorch",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
