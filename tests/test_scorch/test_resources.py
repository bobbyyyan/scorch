import ast
from importlib import resources
from pathlib import Path

import pytest

import scorch
from scorch import utils
from scorch.exceptions import ScorchOSError, defect_code
from scorch.utils import jit_preamble_text, native_resource_text

EXPECTED_NATIVE_RESOURCES = {
    "header.h",
    "kernels.h",
    "ops.cpp",
    "prebuilt_types.h",
    "pybind.cpp",
    "scorch_policy.h",
    "spmm.h",
}

# The names scorch's own code reads today; the scan below must find each of them.
LIVE_RESOURCE_LITERALS = {"scorch_policy.h", "header.h", "native_abi.h"}


def test_native_sources_are_packaged() -> None:
    native_root = resources.files("scorch").joinpath("csrc")

    assert native_root.is_dir()
    assert EXPECTED_NATIVE_RESOURCES <= {
        resource.name for resource in native_root.iterdir() if resource.is_file()
    }
    for filename in EXPECTED_NATIVE_RESOURCES:
        assert native_resource_text(filename)
    assert not native_root.joinpath("header.cpp").is_file()


def test_jit_preamble_embeds_packaged_policy_header() -> None:
    preamble = jit_preamble_text()

    assert '#include "scorch_policy.h"' not in preamble
    assert "inline int scorch_nthreads(" in preamble
    assert "typedef struct" in preamble
    assert "scorch_tensor_from_vector" in preamble
    assert "class cvector" not in preamble


@pytest.mark.parametrize("filename", ["", ".", "..", "../header.h", "csrc/header.h"])
def test_native_resource_rejects_non_local_filename(filename: str) -> None:
    with pytest.raises(ValueError):
        native_resource_text(filename)


def test_a_native_resource_the_package_does_not_ship_is_refused_with_a_code() -> None:
    # spmm-csr.cpp is what the deleted load_to_kernel_cache read (post-merge item 26).
    with pytest.raises(ScorchOSError) as caught:
        native_resource_text("spmm-csr.cpp")

    assert isinstance(caught.value, OSError)
    assert defect_code(caught.value) == "native_resource_missing"
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def _shipped_native_resources() -> set:
    native_root = resources.files("scorch").joinpath("csrc")
    return {resource.name for resource in native_root.iterdir() if resource.is_file()}


def _resource_literals(source: str, filename: str) -> list:
    """(file, line, name) for every string literal passed as the first argument of a
    call to ``native_resource_text``, called by name or as an attribute."""
    found = []
    for node in ast.walk(ast.parse(source, filename)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if isinstance(function, ast.Name):
            called = function.id
        elif isinstance(function, ast.Attribute):
            called = function.attr
        else:
            continue
        argument = node.args[0]
        if (
            called == "native_resource_text"
            and isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        ):
            found.append((filename, node.lineno, argument.value))
    return sorted(found)


def test_every_native_resource_named_in_the_package_is_shipped() -> None:
    """Each string literal scorch's own code passes to ``native_resource_text`` names a
    file the package ships, so no code path reads a deleted native source (post-merge
    item 26: ``load_to_kernel_cache`` read csrc/spmm-csr.cpp after it was deleted)."""
    package = Path(scorch.__file__).parent
    literals = [
        found
        for path in sorted(package.rglob("*.py"))
        for found in _resource_literals(
            path.read_text(encoding="utf-8"), str(path.relative_to(package))
        )
    ]

    # The scan must reach the live readers, so that a change in how they are called
    # cannot make it pass by finding nothing.
    assert LIVE_RESOURCE_LITERALS <= {name for _, _, name in literals}, literals
    shipped = _shipped_native_resources()
    assert [found for found in literals if found[2] not in shipped] == []


def test_the_resource_scan_flags_a_name_the_package_does_not_ship() -> None:
    planted = (
        "from scorch import utils\n"
        'native_resource_text("spmm-csr.cpp")\n'
        'utils.native_resource_text("header.h")\n'
        "native_resource_text(name)\n"
    )
    literals = _resource_literals(planted, "planted.py")

    assert literals == [
        ("planted.py", 2, "spmm-csr.cpp"),
        ("planted.py", 3, "header.h"),
    ]
    shipped = _shipped_native_resources()
    assert [found for found in literals if found[2] not in shipped] == [
        ("planted.py", 2, "spmm-csr.cpp")
    ]


def test_the_special_kernel_loader_is_gone() -> None:
    # Post-merge item 26: it could only read native sources the package no longer ships.
    assert not hasattr(utils, "load_to_kernel_cache")
