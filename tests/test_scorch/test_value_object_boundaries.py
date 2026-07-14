from dataclasses import FrozenInstanceError
import warnings

import pytest
import torch

import scorch.ops as ops
from scorch import (
    CompileSpecError,
    ScorchError,
    SparseStorage,
    STensor,
    TensorFormat,
    TensorFormatError,
    TensorIndex,
    TensorIndexError,
    TensorLayout,
    TensorLayoutError,
    TensorMetadata,
    TensorSpec,
    TensorStorageError,
    TensorTypeError,
    TensorValidationError,
    parse_format,
)
from scorch.utils import parse_format as utils_parse_format


def _dense_storage(
    shape=(2, 3),
    *,
    permutation=None,
    index_dtype=torch.int32,
    value_dtype=torch.float32,
):
    layout = TensorLayout.from_physical_shape(
        shape,
        "d" * len(shape),
        permutation=permutation,
        index_dtype=index_dtype,
    )
    values = torch.arange(layout.element_count, dtype=value_dtype)
    return SparseStorage(
        layout,
        values,
        mode_indices=[[] for _ in shape],
    )


def _csr_storage(
    *,
    positions=None,
    coordinates=None,
    values=None,
    shape=(2, 3),
):
    positions = (
        torch.tensor([0, 2, 3], dtype=torch.int64) if positions is None else positions
    )
    coordinates = (
        torch.tensor([0, 2, 1], dtype=torch.int64)
        if coordinates is None
        else coordinates
    )
    values = torch.tensor([1.0, 2.0, 3.0]) if values is None else values
    layout = TensorLayout.from_physical_shape(shape, "ds", index_dtype=torch.int64)
    return SparseStorage(
        layout,
        values,
        mode_indices=[[], [positions, coordinates]],
    )


def test_format_parser_is_canonical_structural_and_frozen():
    compact = TensorFormat("ds")
    long_form = TensorFormat(["dense", "compressed"])
    comma_form = parse_format("dense, compressed")

    assert compact == long_form == comma_form
    assert hash(compact) == hash(long_form)
    assert compact.serialize() == long_form.serialize()
    assert compact.get_level_formats() == tuple(compact.get_level_formats())
    assert utils_parse_format is parse_format

    with pytest.raises(FrozenInstanceError):
        compact._level_formats = ()


def test_format_parser_treats_long_alias_as_one_level_and_rejects_bad_tokens():
    assert TensorFormat("dense").get_order() == 1
    assert TensorFormat("singleton").get_order() == 1
    assert TensorFormat("ds").get_order() == 2
    assert TensorFormat("d,s").get_order() == 2

    with pytest.raises(TensorFormatError):
        TensorFormat("d,not-a-level")
    with pytest.raises(TensorTypeError):
        TensorFormat(["d", object()])


def test_layout_round_trip_preserves_logical_and_physical_mode_semantics():
    layout = TensorLayout.from_logical_shape(
        (2, 3, 5),
        "dss",
        permutation=(2, 0, 1),
        index_dtype=torch.int64,
    )

    assert layout.logical_shape == (2, 3, 5)
    assert layout.physical_shape == (5, 2, 3)
    assert layout.permutation == (2, 0, 1)
    assert layout.logical_to_physical == (1, 2, 0)
    assert layout.element_count == 30
    assert TensorLayout.from_dict(layout.to_dict()) == layout
    assert TensorLayout.from_dict(layout.to_dict()).serialize() == layout.serialize()

    with pytest.raises(FrozenInstanceError):
        layout.physical_shape = (2, 3, 5)


def test_layout_rejects_rank_permutation_shape_and_overflow_mismatches():
    with pytest.raises(TensorLayoutError):
        TensorLayout((2, 3), (2, 3), TensorFormat("d"), (0, 1))
    with pytest.raises(TensorLayoutError):
        TensorLayout.from_logical_shape((2, 3), "dd", (0, 0))
    with pytest.raises(TensorLayoutError):
        TensorLayout((2, 3), (2, 3), TensorFormat("dd"), (1, 0))
    with pytest.raises(TensorLayoutError):
        TensorLayout.from_logical_shape((1 << 62, 4), "dd", index_dtype=torch.int64)
    with pytest.raises(TensorTypeError):
        TensorLayout.from_logical_shape((2, True), "dd")


def test_metadata_is_frozen_canonical_and_validates_authoritative_fields():
    layout = TensorLayout.from_logical_shape((2, 3), "dd")
    metadata = TensorMetadata("  weights  ", torch.float64, "cpu", layout)

    assert metadata.name == "weights"
    assert metadata.dtype == torch.float64
    assert metadata.device == torch.device("cpu")
    assert TensorMetadata.from_dict(metadata.to_dict()) == metadata
    assert (
        TensorMetadata.from_dict(metadata.to_dict()).serialize() == metadata.serialize()
    )

    with pytest.raises(FrozenInstanceError):
        metadata.name = "other"
    with pytest.raises(TensorLayoutError):
        TensorMetadata(" ", torch.float32, "cpu", layout)


def test_tensor_index_never_casts_or_mutates_caller_indices_and_is_frozen():
    positions = torch.tensor([0, 2], dtype=torch.int64)
    coordinates = torch.tensor([0, 3], dtype=torch.int64)
    caller_indices = [[positions, coordinates]]
    index = TensorIndex("s", caller_indices)

    assert index.index_dtype == torch.int64
    assert caller_indices == [[positions, coordinates]]
    assert caller_indices[0][0] is positions
    assert index.mode_indices[0][0] is not positions
    assert torch.equal(index.mode_indices[0][0], positions)

    exposed = index.mode_indices
    exposed[0][0][0] = 99
    exposed[0].clear()
    caller_indices[0].clear()
    assert len(index.mode_indices[0]) == 2
    assert index.mode_indices[0][0][0].item() == 0

    with pytest.raises(FrozenInstanceError):
        index._mode_order = (0,)


def test_tensor_index_equality_and_validation_include_structure_and_dtype():
    left = TensorIndex(
        "s",
        [[torch.tensor([0, 1]), torch.tensor([2])]],
    )
    equal = TensorIndex(
        ["compressed"],
        [[torch.tensor([0, 1]), torch.tensor([2])]],
    )
    different = TensorIndex(
        "s",
        [[torch.tensor([0, 1]), torch.tensor([1])]],
    )

    assert left == equal
    assert left != different
    with pytest.raises(TensorIndexError):
        TensorIndex("ds", [[]])
    with pytest.raises(TensorIndexError):
        TensorIndex(
            "s",
            [
                [
                    torch.tensor([0, 0], dtype=torch.int32),
                    torch.tensor([], dtype=torch.int64),
                ]
            ],
        )
    with pytest.raises(TensorTypeError):
        TensorIndex("dd", [[], []], mode_order=(0, "1"))


def test_tensor_index_accessors_reject_negative_and_out_of_rank_modes():
    index = TensorIndex("dd", [[], []])

    for mode in (-1, 2):
        with pytest.raises(TensorIndexError):
            index.get_mode_index(mode)
        with pytest.raises(TensorIndexError):
            index.get_level_type(mode)


def test_sparse_storage_validates_csr_and_exposes_defensive_containers():
    storage = _csr_storage()

    assert storage.layout.physical_shape == (2, 3)
    assert storage.index.index_dtype == torch.int64
    assert storage.index == TensorIndex(
        "ds",
        [
            [],
            [
                torch.tensor([0, 2, 3], dtype=torch.int64),
                torch.tensor([0, 2, 1], dtype=torch.int64),
            ],
        ],
    )
    storage.validate()
    assert storage.serialize() == storage.serialize()

    exposed = storage.mode_indices
    exposed[1].clear()
    assert len(storage.mode_indices[1]) == 2
    with pytest.raises(FrozenInstanceError):
        storage._layout = storage.layout


@pytest.mark.parametrize(
    ("positions", "coordinates", "values", "error"),
    [
        (
            torch.tensor([1, 2, 3], dtype=torch.int64),
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.ones(3),
            TensorIndexError,
        ),
        (
            torch.tensor([0, 3, 2], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.ones(2),
            TensorIndexError,
        ),
        (
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 3], dtype=torch.int64),
            torch.ones(2),
            TensorIndexError,
        ),
        (
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.ones(1),
            TensorStorageError,
        ),
    ],
)
def test_sparse_storage_rejects_invalid_csr_invariants(
    positions, coordinates, values, error
):
    with pytest.raises(error):
        _csr_storage(
            positions=positions,
            coordinates=coordinates,
            values=values,
        )


def test_sparse_storage_rejects_csr_order_native_views_cannot_consume():
    with pytest.raises(TensorIndexError):
        _csr_storage(
            positions=torch.tensor([0, 2, 3], dtype=torch.int64),
            coordinates=torch.tensor([2, 0, 1], dtype=torch.int64),
        )


def test_sparse_storage_dense_count_and_equality_include_value_dtype():
    float_storage = _dense_storage(value_dtype=torch.float32)
    int_storage = _dense_storage(value_dtype=torch.int64)

    assert float_storage != int_storage
    with pytest.raises(TensorStorageError):
        SparseStorage(
            float_storage.layout,
            torch.ones(5),
            mode_indices=[[], []],
        )


def test_stensor_requires_complete_runtime_state_and_validates_storage_metadata():
    with pytest.raises(TensorValidationError):
        STensor(name="incomplete")

    storage = _dense_storage()
    tensor = STensor(name="x", storage=storage)
    assert tensor.storage is storage
    assert tensor.metadata.layout is storage.layout
    assert tensor.shape == (2, 3)
    assert tensor.logical_shape == (2, 3)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device("cpu")

    with pytest.raises(TensorLayoutError):
        STensor(name="x", shape=(3, 2), storage=storage)


def test_stensor_from_components_is_a_validated_public_boundary():
    tensor = STensor.from_components(
        (2, 3),
        "oo",
        [
            [torch.tensor([0, 1], dtype=torch.int64)],
            [torch.tensor([2, 0], dtype=torch.int64)],
        ],
        torch.tensor([4.0, 5.0]),
        name="components",
    )
    tensor.validate()
    assert tensor.name == "components"
    assert tensor.index_dtype == torch.int64

    with pytest.raises(TensorIndexError):
        STensor.from_components(
            (2, 3),
            "oo",
            [
                [torch.tensor([0, 2], dtype=torch.int64)],
                [torch.tensor([2, 0], dtype=torch.int64)],
            ],
            torch.tensor([4.0, 5.0]),
        )


def test_dense_factory_tracks_logical_and_physical_shape_without_jit():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensor = STensor.from_torch(source, name="x", mode_order=[1, 0])

    assert tensor.shape == (3, 2)
    assert tensor.physical_shape == (3, 2)
    assert tensor.logical_shape == (2, 3)
    assert tensor.mode_order == (1, 0)
    assert torch.equal(tensor.to_torch(in_place=False), source)


def test_sparse_factories_preserve_callers_and_publish_valid_index_dtype():
    coordinates = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
    values = torch.tensor([4.0, 5.0])
    coordinate_snapshot = coordinates.clone()
    value_snapshot = values.clone()

    coo = STensor.from_coo(indices=coordinates, values=values, shape=(2, 3))
    assert torch.equal(coordinates, coordinate_snapshot)
    assert coordinates.dtype == torch.int32
    assert torch.equal(values, value_snapshot)
    assert coo.index_dtype in (torch.int32, torch.int64)
    coo.validate()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csr_source = torch.tensor([[0.0, 2.0], [3.0, 0.0]]).to_sparse_csr()
    csr = STensor.from_csr(csr_source)
    assert csr.index_dtype == csr_source.crow_indices().dtype
    csr.validate()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: STensor.from_coo(
            indices=torch.empty((1, 0), dtype=torch.int64),
            values=torch.empty(0),
            shape=3,
        ),
        lambda: STensor.from_torch(
            torch.sparse_coo_tensor(
                torch.tensor([[0, 1]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                (2, 2),
                check_invariants=False,
            ).coalesce()
        ),
    ],
)
def test_sparse_factories_translate_invalid_inputs_to_domain_exceptions(factory):
    with pytest.raises(ScorchError):
        factory()


def test_sparse_csr_factory_transpose_never_leaks_backend_exceptions():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csr = torch.tensor([[0.0, 2.0], [3.0, 0.0]]).to_sparse_csr()

    try:
        result = STensor.from_torch(csr, mode_order=[1, 0])
    except ScorchError:
        return
    assert result.logical_shape == (2, 2)
    assert result.physical_shape == (2, 2)
    assert result.mode_order == (1, 0)


def test_tensor_spec_is_frozen_payload_free_and_mode_order_updates_are_pure():
    spec = TensorSpec(
        "dd",
        (2, 3),
        dtype=torch.float64,
        mode_order=(1, 0),
        index_dtype=torch.int64,
        name="a",
    )
    relaid = spec.with_mode_order((0, 1))

    assert spec.logical_shape == (2, 3)
    assert spec.shape == (3, 2)
    assert spec.mode_order == (1, 0)
    assert relaid.logical_shape == (2, 3)
    assert relaid.shape == (2, 3)
    assert relaid.mode_order == (0, 1)
    assert not hasattr(spec, "values")
    assert not hasattr(spec, "index")
    with pytest.raises(FrozenInstanceError):
        spec.metadata = relaid.metadata


def test_tensor_spec_wraps_invalid_device_as_compile_domain_error():
    with pytest.raises(CompileSpecError):
        TensorSpec("d", (1,), device="not-a-device")


def test_tensor_spec_rejects_unsupported_compile_contracts():
    with pytest.raises(CompileSpecError):
        TensorSpec("d", (1,), dtype=torch.complex64)
    with pytest.raises(CompileSpecError):
        TensorSpec("singleton", (1,))


def test_sparse_storage_rejects_noncanonical_coo_order():
    layout = TensorLayout.from_physical_shape((2, 3), "oo", index_dtype=torch.int64)
    with pytest.raises(TensorIndexError, match="lexicographically"):
        SparseStorage(
            layout,
            torch.tensor([1.0, 2.0]),
            mode_indices=[
                [torch.tensor([1, 0], dtype=torch.int64)],
                [torch.tensor([0, 2], dtype=torch.int64)],
            ],
        )


def test_einsum_compile_only_returns_spec_without_jit_or_operand_mutation(monkeypatch):
    compiled_sources = []

    def fake_load_kernel(prepared):
        compiled_sources.append(prepared.request.cpp_sources)
        return object()

    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", fake_load_kernel)
    monkeypatch.setattr(ops, "_kernel_cache", {})
    a = TensorSpec("dd", (2, 3), mode_order=(1, 0), name="a")
    b = TensorSpec("dd", (3, 4), name="b")

    result = ops.einsum(
        "ik,kj->ij",
        a,
        b,
        compile_only=True,
        format="dd",
        output_mode_order=(1, 0),
    )

    assert isinstance(result, TensorSpec)
    assert result.logical_shape == (2, 4)
    assert result.physical_shape == (4, 2)
    assert result.mode_order == (1, 0)
    assert a.logical_shape == (2, 3)
    assert a.mode_order == (1, 0)
    assert b.logical_shape == (3, 4)
    assert b.mode_order == (0, 1)
    assert len(compiled_sources) == 1


def test_einsum_compile_only_validates_contraction_shapes_before_codegen(monkeypatch):
    def fail_if_codegen_starts(**kwargs):
        raise AssertionError("shape validation must run before native code generation")

    monkeypatch.setattr(ops, "_prepare_jit_build", fail_if_codegen_starts)
    monkeypatch.setattr(ops, "_kernel_cache", {})
    a = TensorSpec("dd", (2, 3), name="a")
    b = TensorSpec("dd", (5, 4), name="b")

    with pytest.raises(CompileSpecError):
        ops.einsum("ik,kj->ij", a, b, compile_only=True, format="dd")


def test_runtime_einsum_rejects_payload_free_spec_before_codegen(monkeypatch):
    def fail_if_codegen_starts(**kwargs):
        raise AssertionError("payload validation must run before code generation")

    monkeypatch.setattr(ops, "_prepare_jit_build", fail_if_codegen_starts)
    spec = TensorSpec("dd", (2, 3))

    with pytest.raises(CompileSpecError):
        ops.einsum("ij->ij", spec)


def test_storage_owns_structural_indices_and_protects_tensor_metadata():
    positions = torch.tensor([0, 1, 1], dtype=torch.int64)
    coordinates = torch.tensor([0], dtype=torch.int64)
    values = torch.tensor([4.0])
    tensor = STensor.from_components(
        (2, 2), "ds", [[], [positions, coordinates]], values
    )

    positions[1] = 9
    coordinates[0] = 9
    values.resize_(0)
    exposed_indices = tensor.index.mode_indices
    exposed_indices[1][0][1] = 9
    exposed_values = tensor.values
    exposed_values.resize_(0)

    tensor.validate()
    assert tensor.values.shape == (1,)
    assert tensor.index.mode_indices[1][0].tolist() == [0, 1, 1]
    assert tensor.index.mode_indices[1][1].tolist() == [0]


def test_runtime_rejects_contracts_the_compiler_cannot_execute():
    with pytest.raises(TensorValidationError, match="dtype"):
        STensor.from_torch(torch.ones(2, dtype=torch.float16))
    with pytest.raises(TensorLayoutError, match="singleton"):
        STensor.from_components(
            (3,),
            "singleton",
            [[torch.tensor([1], dtype=torch.int64)]],
            torch.tensor([2.0]),
        )
    complex_layout = TensorLayout.from_physical_shape((1,), "d")
    with pytest.raises(TensorValidationError, match="dtype"):
        SparseStorage(
            complex_layout,
            torch.tensor([1 + 2j]),
            mode_indices=[[]],
        )


@pytest.mark.parametrize("output_format", ["d", "singleton,singleton"])
def test_einsum_validates_output_contract_before_codegen(monkeypatch, output_format):
    def fail_if_codegen_starts(**kwargs):
        raise AssertionError("output validation must precede code generation")

    monkeypatch.setattr(ops, "_prepare_jit_build", fail_if_codegen_starts)
    a = TensorSpec("dd", (2, 3))
    b = TensorSpec("dd", (3, 4))

    with pytest.raises(CompileSpecError):
        ops.einsum("ik,kj->ij", a, b, compile_only=True, format=output_format)


def test_malformed_serialized_layout_and_coo_shape_raise_domain_errors():
    malformed = {
        "logical_shape": 1,
        "physical_shape": [1],
        "format": TensorFormat("d").to_dict(),
        "permutation": [0],
        "index_dtype": "int32",
    }
    with pytest.raises(TensorLayoutError):
        TensorLayout.from_dict(malformed)

    with pytest.raises(TensorTypeError):
        STensor.from_coo(
            indices=torch.empty((2, 0), dtype=torch.int64),
            values=torch.empty(0),
            shape=(2, "3"),
        )


def test_permuted_coo_sddmm_skips_canonical_native_shortcut():
    mask_dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    mask = STensor.from_torch(mask_dense.to_sparse_coo(), mode_order=[1, 0])
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[2.0, 1.0], [1.0, 3.0], [4.0, 2.0]])

    result = ops.einsum("ij,ik,jk->ij", mask, left, right)
    expected = mask_dense * (left @ right.T)
    assert torch.allclose(result.to_torch(), expected)
