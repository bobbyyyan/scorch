# Scorch

### Setting up development environment

#### Install libtorch

* See https://pytorch.org/get-started/locally/
* Currently, latest version for macOS at https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.13.0.zip

#### Edit includepath

For VSCode, edit include path to include the following:

```
${workspaceFolder}/**
~/libtorch/include/**
~/libtorch/include/torch/**
~/libtorch/include/torch/csrc/api/include/**
```

#### Install Ninja

Ninja is required to load C++ extension. For Mac, use `brew install ninja`.

