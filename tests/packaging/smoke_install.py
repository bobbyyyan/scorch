"""Smoke-test an installed Scorch distribution from outside the checkout.

This script intentionally is not a pytest test.  Packaging CI invokes it with
the Python interpreter from a clean artifact-only environment and with a fresh
``TORCH_EXTENSIONS_DIR`` so repository files and prior JIT builds cannot mask
missing package data.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import torch

import scorch
import scorch_ops

EXPECTED_NATIVE_RESOURCES = (
    "header.h",
    "kernels.h",
    "ops.cpp",
    "prebuilt_types.h",
    "pybind.cpp",
    "scorch_policy.h",
    "spmm.h",
)


def _assert_outside_checkout(path: Path, checkout: Path, label: str) -> None:
    try:
        path.resolve().relative_to(checkout.resolve())
    except ValueError:
        return
    raise AssertionError(f"{label} unexpectedly resolved inside checkout: {path}")


def _assert_clean_install() -> None:
    checkout_value = os.environ.get("SCORCH_CHECKOUT")
    if not checkout_value:
        raise AssertionError("SCORCH_CHECKOUT must identify the source checkout")

    checkout = Path(checkout_value)
    _assert_outside_checkout(Path.cwd(), checkout, "smoke-test working directory")

    if scorch.__file__ is None or scorch_ops.__file__ is None:
        raise AssertionError("installed modules must have filesystem locations")
    _assert_outside_checkout(Path(scorch.__file__), checkout, "scorch package")
    _assert_outside_checkout(
        Path(scorch_ops.__file__), checkout, "scorch_ops extension"
    )


def _assert_packaged_native_resources() -> None:
    native_root = resources.files("scorch").joinpath("csrc")
    if not native_root.is_dir():
        raise AssertionError("installed distribution is missing scorch/csrc")

    for filename in EXPECTED_NATIVE_RESOURCES:
        resource = native_root.joinpath(filename)
        if not resource.is_file():
            raise AssertionError(f"installed distribution is missing csrc/{filename}")
        if not resource.read_bytes():
            raise AssertionError(f"installed csrc/{filename} is empty")

    if native_root.joinpath("header.cpp").is_file():
        raise AssertionError("distribution contains the removed divergent header.cpp")

    if native_root.joinpath("scorch_policy_tuned.h").is_file():
        raise AssertionError("distribution contains a host-specific tuned policy")


def _assert_native_kernel() -> None:
    lhs_torch = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    rhs_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    lhs = scorch.from_torch(lhs_torch.to_sparse_csr(), "lhs")
    rhs = scorch.from_torch(rhs_torch, "rhs")
    actual = scorch.matmul(lhs, rhs)

    if not isinstance(actual, torch.Tensor):
        raise AssertionError("native CSR x dense kernel must return a torch.Tensor")
    torch.testing.assert_close(actual, lhs_torch @ rhs_torch)


def _shared_objects(root: Path) -> set[Path]:
    return {path.resolve() for path in root.rglob("*.so")}


def _assert_jit_kernel() -> None:
    jit_dir_value = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not jit_dir_value:
        raise AssertionError("TORCH_EXTENSIONS_DIR must point to a fresh CI directory")

    jit_dir = Path(jit_dir_value)
    before = _shared_objects(jit_dir) if jit_dir.exists() else set()
    if before:
        raise AssertionError(f"JIT directory is not clean: {sorted(before)}")

    lhs_torch = torch.tensor([1.0, 0.0, 2.0])
    rhs_torch = torch.tensor([3.0, 4.0, 0.0])
    lhs = scorch.from_torch(lhs_torch, "jit_lhs")
    rhs = scorch.from_torch(rhs_torch, "jit_rhs")
    actual = (lhs + rhs).to_torch()

    torch.testing.assert_close(actual, lhs_torch + rhs_torch)
    after = _shared_objects(jit_dir)
    if not after:
        raise AssertionError("generic operation did not produce a JIT extension")


def main() -> None:
    _assert_clean_install()
    _assert_packaged_native_resources()
    _assert_native_kernel()
    _assert_jit_kernel()
    print("installed Scorch distribution passed native and JIT smoke tests")


if __name__ == "__main__":
    main()
