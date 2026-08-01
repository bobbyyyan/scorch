"""Public ``to_sparse`` dense-suffix block materialization.

Formats whose value-bearing suffix is DENSE below ``d``/``s`` prefix
structure (``sd``, ``sdd``, ``dsd``, ``ssd``, ...) previously ran the
per-entry legacy filter kernel, which mis-assembles blocked values
(values without parent coordinates), so every runtime test needed
hand-built ``TensorIndex`` storage.  The public conversion now
materializes these layouts directly: one complete dense block per
stored prefix path, a path stored exactly when its block contains any
nonzero, and stored blocks keeping their interior zeros — the same
conditional-parent discipline the compiled assembly families use.
"""

import pytest
import torch

from scorch.stensor import STensor
from tests.test_scorch.test_loopir_mixed_operand_target import (
    sparse_dsd,
    sparse_sd,
    sparse_sdd,
)


def storage_snapshot(stensor):
    return (
        [
            [tensor.tolist() for tensor in level]
            for level in stensor.storage.index.mode_indices
        ],
        stensor.storage.value.tolist(),
    )


@pytest.mark.parametrize("fmt", ["sd", "sdd", "dsd", "ssd", "dssd"])
def test_dense_suffix_conversion_round_trips(fmt):
    torch.manual_seed(20260731)
    rank = len(fmt)
    shape = {2: (4, 5), 3: (3, 4, 5), 4: (2, 3, 4, 3)}[rank]
    dense = (torch.rand(shape) < 0.25) * torch.randn(shape)
    converted = STensor.from_torch(dense.clone(), "A").to_sparse(fmt)
    assert str(converted.format) == ",".join(fmt)
    assert torch.equal(converted.to_torch(), dense)


@pytest.mark.parametrize(
    "fmt,builder",
    [("sd", sparse_sd), ("sdd", sparse_sdd), ("dsd", sparse_dsd)],
)
def test_conversion_matches_the_hand_built_storage_exactly(fmt, builder):
    torch.manual_seed(7)
    shape = (4, 5) if fmt == "sd" else (3, 4, 5)
    dense = (torch.rand(shape) < 0.3) * torch.randn(shape)
    converted = STensor.from_torch(dense.clone(), "A").to_sparse(fmt)
    hand_built = builder(dense, "A")
    assert storage_snapshot(converted) == storage_snapshot(hand_built)


def test_stored_blocks_keep_interior_zeros():
    dense = torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("sd")
    modes, values = storage_snapshot(converted)
    assert modes[0] == [[0, 2], [0, 2]]
    assert modes[1] == []
    assert values == [0.0, 5.0, 0.0, 1.0, 0.0, 2.0]


def test_all_zero_tensors_convert_to_canonical_empty_storage():
    converted = STensor.from_torch(torch.zeros(3, 4), "A").to_sparse("sd")
    modes, values = storage_snapshot(converted)
    assert modes[0] == [[0, 0], []]
    assert values == []

    converted = STensor.from_torch(torch.zeros(3, 4, 5), "A").to_sparse("dsd")
    modes, values = storage_snapshot(converted)
    assert modes[0] == []
    assert modes[1] == [[0, 0, 0, 0], []]
    assert values == []


def test_dense_prefix_empty_slots_keep_full_position_arrays():
    dense = torch.zeros(3, 4, 5)
    dense[2, 1, 3] = 4.0
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("dsd")
    modes, values = storage_snapshot(converted)
    assert modes[1] == [[0, 0, 0, 1], [1]]
    assert values == dense[2, 1].tolist()


def test_float64_conversion_preserves_dtype_and_values():
    torch.manual_seed(3)
    dense = ((torch.rand(4, 5) < 0.3) * torch.randn(4, 5)).double()
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("sd")
    assert converted.storage.value.dtype == torch.float64
    assert torch.equal(converted.to_torch(), dense)


def test_sparse_sources_reconvert_through_densification():
    dense = torch.tensor([[0.0, 5.0], [3.0, 0.0], [0.0, 0.0]])
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("ss").to_sparse("sd")
    assert str(converted.format) == "s,d"
    assert torch.equal(converted.to_torch(), dense)


def test_compressed_leaf_formats_keep_the_kernel_path():
    """Formats with compressed value-bearing leaves are untouched."""

    dense = torch.tensor([[0.0, 5.0], [3.0, 0.0]])
    ss = STensor.from_torch(dense.clone(), "A").to_sparse("ss")
    assert str(ss.format) == "s,s"
    assert ss.storage.value.tolist() == [5.0, 3.0]
    ds = STensor.from_torch(dense.clone(), "B").to_sparse("ds")
    assert str(ds.format) == "d,s"
    assert ds.storage.value.tolist() == [5.0, 3.0]


def test_nonidentity_mode_order_keeps_the_historical_path():
    """The materialization is identity-order only; other orders keep the
    prior (failing) kernel behavior rather than guessing a layout."""

    from scorch.exceptions import TensorStorageError

    dense = torch.tensor([[0.0, 5.0, 0.0], [1.0, 0.0, 2.0]])
    tensor = STensor.from_torch(dense.clone(), "A").change_mode_order([1, 0])
    with pytest.raises(TensorStorageError):
        tensor.to_sparse("sd")


def test_converted_inputs_execute_through_the_compiled_mixed_route():
    """to_sparse('sd') inputs drive the mixed-operand family end-to-end."""

    from tests.test_scorch.test_loopir_mixed_operand_target import (
        build_copy_cin,
    )
    from tests.test_scorch.test_loopir_sparse_workspace_target import (
        auto_options,
    )
    from scorch.compiler.loopir.pipeline import execute_cin_via_loopir

    torch.manual_seed(11)
    dense = (torch.rand(4, 5) < 0.4) * torch.randn(4, 5)
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("sd")
    out = execute_cin_via_loopir(
        build_copy_cin(),
        (4, 5),
        converted,
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(result.to_torch(), dense, atol=1e-3, rtol=1e-3)
