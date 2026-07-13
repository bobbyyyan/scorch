from importlib import resources

import pytest

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
