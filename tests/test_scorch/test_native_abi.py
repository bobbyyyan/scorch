import pytest
import torch

import scorch.ops as ops
import scorch_ops
from scorch import STensor


def _spmm_args(index_dtype: torch.dtype = torch.int32):
    positions = torch.tensor([0, 2, 3], dtype=index_dtype)
    coordinates = torch.tensor([0, 2, 1], dtype=index_dtype)
    a_values = torch.tensor([1.0, 2.0, 3.0])
    b_values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).reshape(-1)
    return (
        [2, 2],
        [2, 3],
        [[], [positions, coordinates]],
        a_values,
        [3, 2],
        [[], []],
        b_values,
    )


def test_native_spmm_accepts_int64_indices_without_mutating_caller():
    args = _spmm_args(torch.int64)
    positions, coordinates = args[2][1]
    positions_ptr = positions.data_ptr()
    coordinates_ptr = coordinates.data_ptr()

    result = scorch_ops.spmm_csr_float_v2(*args)

    assert torch.equal(
        result.storage.value.reshape(2, 2),
        torch.tensor([[11.0, 14.0], [9.0, 12.0]]),
    )
    assert positions.dtype == torch.int64
    assert coordinates.dtype == torch.int64
    assert positions.data_ptr() == positions_ptr
    assert coordinates.data_ptr() == coordinates_ptr


def test_native_result_carrier_fields_are_read_only():
    result = scorch_ops.spmm_csr_float_v2(*_spmm_args())

    for carrier in (
        scorch_ops.Tensor,
        scorch_ops.TensorStorage,
        scorch_ops.TensorIndex,
    ):
        with pytest.raises(TypeError):
            carrier()
    with pytest.raises(AttributeError):
        result.storage = result.storage
    with pytest.raises(AttributeError):
        result.storage.value = result.storage.value
    with pytest.raises(AttributeError):
        result.storage.index.mode_indices = result.storage.index.mode_indices


@pytest.mark.parametrize(
    ("positions", "coordinates", "message"),
    [
        ([1, 2, 3], [0, 2, 1], r"positions\[0\] must be 0"),
        ([0, 3, 2], [0, 1, 2], "invalid CSR span"),
        ([0, 1, 2], [0, 2, 1], "terminal position must equal nnz"),
        ([0, 2, 3], [0, 3, 1], "outside"),
    ],
)
def test_native_spmm_rejects_invalid_csr(positions, coordinates, message):
    args = list(_spmm_args())
    args[2] = [
        [],
        [
            torch.tensor(positions, dtype=torch.int32),
            torch.tensor(coordinates, dtype=torch.int32),
        ],
    ]

    with pytest.raises(RuntimeError, match=message):
        scorch_ops.spmm_csr_float_v2(*args)


def test_native_spmm_rejects_malformed_nested_indices_before_indexing():
    args = list(_spmm_args())
    args[2] = [[]]

    with pytest.raises(RuntimeError, match="exactly 2 levels"):
        scorch_ops.spmm_csr_float_v2(*args)


def test_native_spmm_rejects_duplicate_csr_coordinates():
    args = list(_spmm_args())
    args[2] = [
        [],
        [
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([1, 1, 2], dtype=torch.int32),
        ],
    ]

    with pytest.raises(RuntimeError, match="strictly increasing"):
        scorch_ops.spmm_csr_float_v2(*args)


def test_native_shape_overflow_reaches_torch_checked_boundary():
    args = list(_spmm_args())
    args[0] = [2**40, 2]

    with pytest.raises(RuntimeError, match="exceeds the current native int32"):
        scorch_ops.spmm_csr_float_v2(*args)


def test_native_spmm_rejects_int64_index_narrowing_overflow():
    args = list(_spmm_args(torch.int64))
    args[2] = [
        [],
        [args[2][1][0], torch.tensor([0, 2**40, 1], dtype=torch.int64)],
    ]

    with pytest.raises(RuntimeError, match="cannot be represented as int32"):
        scorch_ops.spmm_csr_float_v2(*args)


def test_native_spmm_rejects_mixed_index_dtypes():
    args = list(_spmm_args())
    args[2] = [
        [],
        [args[2][1][0], args[2][1][1].to(torch.int64)],
    ]

    with pytest.raises(RuntimeError, match="common index dtype"):
        scorch_ops.spmm_csr_float_v2(*args)


def test_native_spmm_rejects_dtype_numel_contiguity_and_tile_errors():
    args = list(_spmm_args())
    wrong_dtype = list(args)
    wrong_dtype[6] = wrong_dtype[6].double()
    with pytest.raises(RuntimeError, match="must have dtype Float"):
        scorch_ops.spmm_csr_float_v2(*wrong_dtype)

    wrong_numel = list(args)
    wrong_numel[6] = wrong_numel[6][:-1].contiguous()
    with pytest.raises(RuntimeError, match="must contain 6 elements"):
        scorch_ops.spmm_csr_float_v2(*wrong_numel)

    noncontiguous = list(args)
    noncontiguous[3] = torch.arange(6.0)[::2]
    assert not noncontiguous[3].is_contiguous()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        scorch_ops.spmm_csr_float_v2(*noncontiguous)

    with pytest.raises(RuntimeError, match="tile_size must be positive"):
        scorch_ops.spmm_csr_float_v2(*args, tile_size=0)

    lazy_negative = list(args)
    lazy_negative[6] = torch._neg_view(lazy_negative[6])
    with pytest.raises(RuntimeError, match="lazy negative"):
        scorch_ops.spmm_csr_float_v2(*lazy_negative)


def test_native_csr_softmax_validates_terminal_pointer_without_mutation():
    positions = torch.tensor([0, 2, 3], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, 3.0])
    result = scorch_ops.scorch_sparse_softmax_csr_float(positions, values)

    assert positions.dtype == torch.int64
    assert torch.allclose(result, torch.tensor([0.26894143, 0.7310586, 1.0]))

    with pytest.raises(RuntimeError, match="terminal crow index"):
        scorch_ops.scorch_sparse_softmax_csr_float(
            torch.tensor([0, 2, 2], dtype=torch.int64), values
        )


def test_native_sparse_attention_rejects_out_of_bounds_column():
    q = torch.ones(2, 1, 2)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    coordinates = torch.tensor([0, 2], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="column coordinate 2"):
        scorch_ops.scorch_sparse_attention_csr_float(positions, coordinates, q, q, q)


def test_native_rectangular_coo_spgemm_sizes_workspace_from_output_columns():
    result = scorch_ops.spmspm_coo_float(
        [1, 3],
        [1, 1],
        [
            [torch.tensor([0], dtype=torch.int32)],
            [torch.tensor([0], dtype=torch.int32)],
        ],
        torch.tensor([2.0]),
        [1, 3],
        [
            [torch.tensor([0], dtype=torch.int32)],
            [torch.tensor([2], dtype=torch.int32)],
        ],
        torch.tensor([4.0]),
    )

    assert result.storage.index.mode_indices[0][0].tolist() == [0]
    assert result.storage.index.mode_indices[1][0].tolist() == [2]
    assert result.storage.value.tolist() == [8.0]


def test_native_extreme_tile_is_safely_normalized_to_runtime_extent():
    rows = 32
    result = scorch_ops.spmm_csr_float_tiled_i_k(
        [rows, 1],
        [rows, 1],
        [
            [],
            [
                torch.arange(rows + 1, dtype=torch.int32),
                torch.zeros(rows, dtype=torch.int32),
            ],
        ],
        torch.ones(rows),
        [1, 1],
        [[], []],
        torch.ones(1),
        i_tile_size=(2**31) - 16,
        k_tile_size=1,
    )

    assert result.storage.value.tolist() == [1.0] * rows


def test_jit_abi_rejects_noncanonical_coo_order_on_cached_entry():
    left = STensor.from_torch(torch.eye(2)).to_sparse("oo")
    right = STensor.from_torch(torch.diag(torch.tensor([3.0, 4.0]))).to_sparse("oo")
    expected = ops.einsum("ij,ij->ij", left, right, format="oo")
    assert torch.equal(expected.to_torch(), torch.diag(torch.tensor([3.0, 4.0])))

    for coordinate in right.storage._mode_indices:
        coordinate[0].copy_(coordinate[0].flip(0))
    right.storage._value.copy_(right.storage._value.flip(0))

    with pytest.raises(RuntimeError, match="lexicographically ordered"):
        ops.einsum("ij,ij->ij", left, right, format="oo")


def test_jit_abi_uses_physical_extents_for_rectangular_mode_order():
    source = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    tensor = STensor.from_torch(source, mode_order=[1, 0])

    # to_sparse forces a generated evaluate() call.  Its runtime shape is [3, 2]
    # in physical level order; applying mode_order to that shape a second time
    # would incorrectly validate level 0 against extent 2.
    sparse = tensor.to_sparse("ds")

    assert torch.equal(sparse.to_torch(), source)
