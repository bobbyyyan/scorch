import random
from typing import List, Optional

import torch

from scorch import STensor, einsum


def test_change_mode_order_2d_dd():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    a_row = STensor.from_torch(tensor_a_torch, "a_row", [0, 1])
    a_col = STensor.from_torch(tensor_a_torch, "a_col", [1, 0])
    a = STensor.from_torch(tensor_a_torch, "a", [0, 1])

    a.change_mode_order([1, 0])
    assert a.storage.value.tolist() == a_col.storage.value.tolist()
    assert a.storage.index.mode_order == a_col.storage.index.mode_order

    a.change_mode_order([0, 1])
    assert a.storage.value.tolist() == a_row.storage.value.tolist()
    assert a.storage.index.mode_order == a_row.storage.index.mode_order


def test_change_mode_order_2d_ds():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    a_row = STensor.from_torch(tensor_a_torch, "a_row", [0, 1]).to_sparse("ds")
    a_col = STensor.from_torch(tensor_a_torch, "a_col", [1, 0]).to_sparse("ds")
    a = STensor.from_torch(tensor_a_torch, "a", [0, 1]).to_sparse("ds")

    a.change_mode_order([1, 0])
    assert a.storage.value.tolist() == a_col.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_col.storage.index.mode_indices[1][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][1].tolist()
        == a_col.storage.index.mode_indices[1][1].tolist()
    )
    assert a.storage.index.mode_order == a_col.storage.index.mode_order

    a.change_mode_order([0, 1])
    assert a.storage.value.tolist() == a_row.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_row.storage.index.mode_indices[1][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][1].tolist()
        == a_row.storage.index.mode_indices[1][1].tolist()
    )
    assert a.storage.index.mode_order == a_row.storage.index.mode_order


def test_change_mode_order_2d_oo():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    a_row = STensor.from_torch(tensor_a_torch, "a_row", [0, 1]).to_sparse("oo")
    a_col = STensor.from_torch(tensor_a_torch, "a_col", [1, 0]).to_sparse("oo")
    a = STensor.from_torch(tensor_a_torch, "a", [0, 1]).to_sparse("oo")

    a.change_mode_order([1, 0])
    assert a.storage.value.tolist() == a_col.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[0][0].tolist()
        == a_col.storage.index.mode_indices[0][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_col.storage.index.mode_indices[1][0].tolist()
    )
    assert a.storage.index.mode_order == a_col.storage.index.mode_order

    a.change_mode_order([0, 1])
    assert a.storage.value.tolist() == a_row.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[0][0].tolist()
        == a_row.storage.index.mode_indices[0][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_row.storage.index.mode_indices[1][0].tolist()
    )
    assert a.storage.index.mode_order == a_row.storage.index.mode_order


def test_change_mode_order_2d_ss():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    a_row = STensor.from_torch(tensor_a_torch, "a_row", [0, 1]).to_sparse("ss")
    a_col = STensor.from_torch(tensor_a_torch, "a_col", [1, 0]).to_sparse("ss")
    a = STensor.from_torch(tensor_a_torch, "a", [0, 1]).to_sparse("ss")

    a.change_mode_order([1, 0])
    assert a.storage.value.tolist() == a_col.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[0][0].tolist()
        == a_col.storage.index.mode_indices[0][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[0][1].tolist()
        == a_col.storage.index.mode_indices[0][1].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_col.storage.index.mode_indices[1][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][1].tolist()
        == a_col.storage.index.mode_indices[1][1].tolist()
    )
    assert a.storage.index.mode_order == a_col.storage.index.mode_order

    a.change_mode_order([0, 1])
    assert a.storage.value.tolist() == a_row.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[0][0].tolist()
        == a_row.storage.index.mode_indices[0][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[0][1].tolist()
        == a_row.storage.index.mode_indices[0][1].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_row.storage.index.mode_indices[1][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][1].tolist()
        == a_row.storage.index.mode_indices[1][1].tolist()
    )
    assert a.storage.index.mode_order == a_row.storage.index.mode_order


def test_change_mode_order_3d_dss():
    tensor_a_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ],
        ]
    )
    a_default = STensor.from_torch(tensor_a_torch, "a_default", [0, 1, 2]).to_sparse(
        "dss"
    )
    a_reverse = STensor.from_torch(tensor_a_torch, "a_reverse", [2, 1, 0]).to_sparse(
        "dss"
    )
    a = STensor.from_torch(tensor_a_torch, "a", [0, 1, 2]).to_sparse("dss")

    a.change_mode_order([2, 1, 0])
    assert a.storage.value.tolist() == a_reverse.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_reverse.storage.index.mode_indices[1][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][1].tolist()
        == a_reverse.storage.index.mode_indices[1][1].tolist()
    )
    assert (
        a.storage.index.mode_indices[2][0].tolist()
        == a_reverse.storage.index.mode_indices[2][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[2][1].tolist()
        == a_reverse.storage.index.mode_indices[2][1].tolist()
    )
    assert a.storage.index.mode_order == a_reverse.storage.index.mode_order

    a.change_mode_order([0, 1, 2])
    assert a.storage.value.tolist() == a_default.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_default.storage.index.mode_indices[1][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][1].tolist()
        == a_default.storage.index.mode_indices[1][1].tolist()
    )
    assert (
        a.storage.index.mode_indices[2][0].tolist()
        == a_default.storage.index.mode_indices[2][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[2][1].tolist()
        == a_default.storage.index.mode_indices[2][1].tolist()
    )
    assert a.storage.index.mode_order == a_default.storage.index.mode_order


def test_change_mode_order_3d_sss():
    tensor_a_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ],
        ]
    )

    a_default = STensor.from_torch(tensor_a_torch, "a_default").to_sparse("sss")
    a_reverse = STensor.from_torch(tensor_a_torch, "a_reverse", [2, 1, 0]).to_sparse(
        "sss"
    )
    a = STensor.from_torch(tensor_a_torch, "a", [0, 1, 2]).to_sparse("sss")

    a.change_mode_order([2, 1, 0])
    assert a.storage.value.tolist() == a_reverse.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[0][0].tolist()
        == a_reverse.storage.index.mode_indices[0][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[0][1].tolist()
        == a_reverse.storage.index.mode_indices[0][1].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_reverse.storage.index.mode_indices[1][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][1].tolist()
        == a_reverse.storage.index.mode_indices[1][1].tolist()
    )
    assert (
        a.storage.index.mode_indices[2][0].tolist()
        == a_reverse.storage.index.mode_indices[2][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[2][1].tolist()
        == a_reverse.storage.index.mode_indices[2][1].tolist()
    )
    assert a.storage.index.mode_order == a_reverse.storage.index.mode_order

    a.change_mode_order([0, 1, 2])
    assert a.storage.value.tolist() == a_default.storage.value.tolist()
    assert (
        a.storage.index.mode_indices[0][0].tolist()
        == a_default.storage.index.mode_indices[0][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[0][1].tolist()
        == a_default.storage.index.mode_indices[0][1].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][0].tolist()
        == a_default.storage.index.mode_indices[1][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[1][1].tolist()
        == a_default.storage.index.mode_indices[1][1].tolist()
    )
    assert (
        a.storage.index.mode_indices[2][0].tolist()
        == a_default.storage.index.mode_indices[2][0].tolist()
    )
    assert (
        a.storage.index.mode_indices[2][1].tolist()
        == a_default.storage.index.mode_indices[2][1].tolist()
    )
    assert a.storage.index.mode_order == a_default.storage.index.mode_order


def test_tensor_addition_2d_csr_csc():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )

    a_csr = STensor.from_torch(tensor_a_torch, "a_csr", [0, 1]).to_sparse("ds")
    a_csc = STensor.from_torch(tensor_a_torch, "a_csc", [1, 0]).to_sparse("ds")

    b_csr = STensor.from_torch(tensor_b_torch, "b_csr", [0, 1]).to_sparse("ds")
    b_csc = STensor.from_torch(tensor_b_torch, "b_csc", [1, 0]).to_sparse("ds")

    c_csr = a_csr + b_csc

    assert c_csr.storage.value.tolist() == [
        2.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 3.0, 3.0, 1.0, 5.0, 3.0, 6.0,
    ]
    assert c_csr.storage.index.mode_indices[1][0].tolist() == [0, 5, 7, 9, 10, 13]
    assert c_csr.storage.index.mode_indices[1][1].tolist() == [
        0, 1, 2, 3, 4, 0, 1, 0, 2, 3, 0, 3, 4,
    ]
    assert c_csr.storage.index.mode_order == [0, 1]

    c_csc = a_csc + b_csr

    assert c_csc.storage.value.tolist() == [
        2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 3.0, 4.0, 1.0, 3.0, 5.0, 6.0,
    ]
    assert c_csc.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 8, 11, 13]
    assert c_csc.storage.index.mode_indices[1][1].tolist() == [
        0, 1, 2, 4, 0, 1, 0, 2, 0, 3, 4, 0, 4,
    ]
    assert c_csc.storage.index.mode_order == [1, 0]


def test_change_mode_order_3d_random_sss():
    tensor_a_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ],
        ]
    )
    a = STensor.from_torch(tensor_a_torch, "a", [0, 1, 2]).to_sparse("sss")
    mode_order = [0, 1, 2]
    for _ in range(10):
        random.shuffle(mode_order)
        a.change_mode_order(mode_order)

    a.change_mode_order([0, 1, 2])
    assert a.storage.value.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert a.storage.index.mode_indices[0][0].tolist() == [0, 2]
    assert a.storage.index.mode_indices[0][1].tolist() == [0, 2]
    assert a.storage.index.mode_indices[1][0].tolist() == [0, 3, 5]
    assert a.storage.index.mode_indices[1][1].tolist() == [0, 1, 2, 0, 1]
    assert a.storage.index.mode_indices[2][0].tolist() == [0, 1, 2, 3, 4, 6]
    assert a.storage.index.mode_indices[2][1].tolist() == [0, 2, 0, 1, 0, 1]
    assert a.storage.index.mode_order == [0, 1, 2]


def test_einsum_2d_concordant_ds():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )

    a_csr = STensor.from_torch(tensor_a_torch, "a_csr", [0, 1]).to_sparse("ds")
    a_csc = STensor.from_torch(tensor_a_torch, "a_csc", [1, 0]).to_sparse("ds")

    b_csr = STensor.from_torch(tensor_b_torch, "b_csr", [0, 1]).to_sparse("ds")
    b_csc = STensor.from_torch(tensor_b_torch, "b_csc", [1, 0]).to_sparse("ds")

    c_csr = einsum("ik,kj->ij", a_csr, b_csc, format="ds", output_mode_order=[0, 1])
    assert torch.allclose(c_csr.values, torch.Tensor([1, 4, 19, 5, 2, 4, 3, 5, 15, 5]))
    assert c_csr.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 7, 7, 10]
    assert c_csr.storage.index.mode_indices[1][1].tolist() == [0, 1, 3, 4, 0, 1, 0, 0, 3, 4]
    assert c_csr.storage.index.mode_order == [0, 1]

    c_csr = einsum("ik,kj->ij", a_csc, b_csr, format="ds", output_mode_order=[0, 1])
    assert torch.allclose(c_csr.values, torch.Tensor([1, 4, 19, 5, 2, 4, 3, 5, 15, 5]))
    assert c_csr.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 7, 7, 10]
    assert c_csr.storage.index.mode_indices[1][1].tolist() == [0, 1, 3, 4, 0, 1, 0, 0, 3, 4]
    assert c_csr.storage.index.mode_order == [0, 1]

    c_csc = einsum("ik,kj->ij", a_csc, b_csr, format="ds", output_mode_order=[1, 0])
    assert torch.allclose(c_csc.values, torch.Tensor([1, 2, 3, 5, 4, 4, 19, 15, 5, 5]))
    assert c_csc.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 6, 8, 10]
    assert c_csc.storage.index.mode_indices[1][1].tolist() == [
        0, 1, 2, 4, 0, 1, 0, 4, 0, 4,
    ]
    assert c_csc.storage.index.mode_order == [1, 0]

    c_csc = einsum("ik,kj->ij", a_csr, b_csc, format="ds", output_mode_order=[1, 0])
    assert torch.allclose(c_csc.values, torch.Tensor([1, 2, 3, 5, 4, 4, 19, 15, 5, 5]))
    assert c_csc.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 6, 8, 10]
    assert c_csc.storage.index.mode_indices[1][1].tolist() == [
        0, 1, 2, 4, 0, 1, 0, 4, 0, 4,
    ]
    assert c_csc.storage.index.mode_order == [1, 0]


def test_einsum_2d_discordant_ds():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )

    a_csr = STensor.from_torch(tensor_a_torch, "a_csr", [0, 1]).to_sparse("ds")
    a_csc = STensor.from_torch(tensor_a_torch, "a_csc", [1, 0]).to_sparse("ds")

    b_csr = STensor.from_torch(tensor_b_torch, "b_csr", [0, 1]).to_sparse("ds")
    b_csc = STensor.from_torch(tensor_b_torch, "b_csc", [1, 0]).to_sparse("ds")

    c_csr = einsum("ik,kj->ij", a_csc, b_csc, format="ds", output_mode_order=[0, 1])
    assert c_csr.storage.value.tolist() == [
        1.0, 4.0, 19.0, 5.0, 2.0, 4.0, 3.0, 5.0, 15.0, 5.0,
    ]
    assert c_csr.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 7, 7, 10]
    assert c_csr.storage.index.mode_indices[1][1].tolist() == [
        0, 1, 3, 4, 0, 1, 0, 0, 3, 4,
    ]
    assert c_csr.storage.index.mode_order == [0, 1]

    c_csc = einsum("ik,kj->ij", a_csr, b_csr, format="ds", output_mode_order=[1, 0])
    assert c_csc.storage.value.tolist() == [
        1.0, 2.0, 3.0, 5.0, 4.0, 4.0, 19.0, 15.0, 5.0, 5.0,
    ]
    assert c_csc.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 6, 8, 10]
    assert c_csc.storage.index.mode_indices[1][1].tolist() == [
        0, 1, 2, 4, 0, 1, 0, 4, 0, 4,
    ]
    assert c_csc.storage.index.mode_order == [1, 0]


def test_einsum_2d_concordant_oo():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )

    a_torch_coo = tensor_a_torch.to_sparse_coo()
    b_torch_coo = tensor_b_torch.to_sparse_coo()
    c_torch_coo = torch.sparse.mm(a_torch_coo, b_torch_coo)
    c_torch_coo_transposed = c_torch_coo.transpose(0, 1).coalesce()

    a_default = STensor.from_torch(tensor_a_torch, "a_csr", [0, 1]).to_sparse("oo")
    a_inverted = STensor.from_torch(tensor_a_torch, "a_csc", [1, 0]).to_sparse("oo")

    b_default = STensor.from_torch(tensor_b_torch, "b_csr", [0, 1]).to_sparse("oo")
    b_inverted = STensor.from_torch(tensor_b_torch, "b_csc", [1, 0]).to_sparse("oo")

    c_default = einsum("ik,kj->ij", a_default, b_inverted, format="oo", output_mode_order=[0, 1])
    assert torch.allclose(c_default.values, c_torch_coo.values())
    assert torch.allclose(
        c_default.index.mode_indices[0][0], c_torch_coo.indices()[0].int()
    )
    assert torch.allclose(
        c_default.index.mode_indices[1][0], c_torch_coo.indices()[1].int()
    )
    assert c_default.storage.index.mode_order == [0, 1]

    c_default = einsum("ik,kj->ij", a_inverted, b_default, format="oo", output_mode_order=[0, 1])
    assert torch.allclose(c_default.values, c_torch_coo.values())
    assert torch.allclose(
        c_default.index.mode_indices[0][0], c_torch_coo.indices()[0].int()
    )
    assert torch.allclose(
        c_default.index.mode_indices[1][0], c_torch_coo.indices()[1].int()
    )
    assert c_default.storage.index.mode_order == [0, 1]

    c_inverted = einsum("ik,kj->ij", a_default, b_inverted, format="oo", output_mode_order=[1, 0])
    assert torch.allclose(c_inverted.values, c_torch_coo_transposed.values())
    assert torch.allclose(
        c_inverted.index.mode_indices[0][0], c_torch_coo_transposed.indices()[0].int()
    )
    assert torch.allclose(
        c_inverted.index.mode_indices[1][0], c_torch_coo_transposed.indices()[1].int()
    )
    assert c_inverted.storage.index.mode_order == [1, 0]

    c_inverted = einsum("ik,kj->ij", a_inverted, b_default, format="oo", output_mode_order=[1, 0])
    assert torch.allclose(c_inverted.values, c_torch_coo_transposed.values())
    assert torch.allclose(
        c_inverted.index.mode_indices[0][0], c_torch_coo_transposed.indices()[0].int()
    )
    assert torch.allclose(
        c_inverted.index.mode_indices[1][0], c_torch_coo_transposed.indices()[1].int()
    )
    assert c_inverted.storage.index.mode_order == [1, 0]


def test_einsum_2d_discordant_oo():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )

    a_torch_coo = tensor_a_torch.to_sparse_coo()
    b_torch_coo = tensor_b_torch.to_sparse_coo()
    c_torch_coo = torch.sparse.mm(a_torch_coo, b_torch_coo)
    c_torch_coo_transposed = c_torch_coo.transpose(0, 1).coalesce()

    a_default = STensor.from_torch(tensor_a_torch, "a_csr", [0, 1]).to_sparse("oo")
    a_inverted = STensor.from_torch(tensor_a_torch, "a_csc", [1, 0]).to_sparse("oo")

    b_default = STensor.from_torch(tensor_b_torch, "b_csr", [0, 1]).to_sparse("oo")
    b_inverted = STensor.from_torch(tensor_b_torch, "b_csc", [1, 0]).to_sparse("oo")

    c_default = einsum("ik,kj->ij", a_inverted, b_inverted, format="oo", output_mode_order=[0, 1])
    assert torch.allclose(c_default.values, c_torch_coo.values())
    assert torch.allclose(
        c_default.index.mode_indices[0][0], c_torch_coo.indices()[0].int()
    )
    assert torch.allclose(
        c_default.index.mode_indices[1][0], c_torch_coo.indices()[1].int()
    )
    assert c_default.storage.index.mode_order == [0, 1]

    c_inverted = einsum("ik,kj->ij", a_default, b_default, format="oo", output_mode_order=[1, 0])
    assert torch.allclose(c_inverted.values, c_torch_coo_transposed.values())
    assert torch.allclose(
        c_inverted.index.mode_indices[0][0], c_torch_coo_transposed.indices()[0].int()
    )
    assert torch.allclose(
        c_inverted.index.mode_indices[1][0], c_torch_coo_transposed.indices()[1].int()
    )
    assert c_inverted.storage.index.mode_order == [1, 0]


def test_einsum_3d():
    tensor_a_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ],
        ]
    )

    tensor_b_torch = torch.Tensor(
        [
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
        ]
    )

    a_default = STensor.from_torch(tensor_a_torch, "a_default", [0, 1, 2]).to_sparse("dss")
    a_reverse = STensor.from_torch(tensor_a_torch, "a_reverse", [2, 1, 0]).to_sparse("dss")

    b_default = STensor.from_torch(tensor_b_torch, "b_default", [0, 1, 2]).to_sparse("dss")
    b_reverse = STensor.from_torch(tensor_b_torch, "b_reverse", [2, 1, 0]).to_sparse("dss")

    c = einsum("bij,bjk->bik", a_default, b_reverse, format="dss", output_mode_order=[0, 1, 2])
    assert c.storage.value.tolist() == [4., 12., 8., 5., 12.]
    assert c.storage.index.mode_indices[1][0].tolist() == [0, 2, 2, 4]
    assert c.storage.index.mode_indices[1][1].tolist() == [0, 2, 0, 1]
    assert c.storage.index.mode_indices[2][0].tolist() == [0, 1, 2, 3, 5]
    assert c.storage.index.mode_indices[2][1].tolist() == [1, 1, 2, 0, 2]

    c = einsum("bij,bjk->bik", a_reverse, b_default, format="dss", output_mode_order=[2, 1, 0])
    assert c.storage.value.tolist() == [5., 4., 12., 8., 12.]
    assert c.storage.index.mode_indices[1][0].tolist() == [0, 1, 3, 5]
    assert c.storage.index.mode_indices[1][1].tolist() == [1, 0, 2, 0, 1]
    assert c.storage.index.mode_indices[2][0].tolist() == [0, 1, 2, 3, 4, 5]
    assert c.storage.index.mode_indices[2][1].tolist() == [2, 0, 0, 2, 2]


# --- Element-wise add tests with mixed formats and mode orders ---


def generate_2d_tensors(
    a_fmt: str,
    b_fmt: str,
    result_fmt: str,
    a_mode_order: Optional[List[int]] = None,
    b_mode_order: Optional[List[int]] = None,
    result_mode_order: Optional[List[int]] = None,
):
    """Generate test tensors A, B, and expected results for various operations."""
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )

    tensor_result_add_torch = torch.Tensor(
        [
            [2, 2, 3, 4, 5],
            [2, 4, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 1, 0],
            [5, 0, 0, 3, 6],
        ]
    )

    tensor_result_matmul_torch = torch.Tensor(
        [
            [1, 4, 0, 19, 5],
            [2, 4, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 15, 5],
        ]
    )

    a = STensor.from_torch(tensor_a_torch, "a", a_mode_order).to_sparse(a_fmt)
    b = STensor.from_torch(tensor_b_torch, "b", b_mode_order).to_sparse(b_fmt)
    result_add = STensor.from_torch(
        tensor_result_add_torch, "result_add", result_mode_order
    ).to_sparse(result_fmt)
    result_matmul = STensor.from_torch(
        tensor_result_matmul_torch, "result_matmul", result_mode_order
    ).to_sparse(result_fmt)

    return a, b, result_add, result_matmul


def test_elemwise_2d_add_csc_csc_csc():
    a, b, result, _ = generate_2d_tensors("ds", "ds", "ds", [1, 0], [1, 0], [1, 0])
    result_cpp = a + b

    assert (
        result_cpp._storage._value.tolist() == result._storage._value.tolist()
    ), "Values are different"
    assert (
        result_cpp._storage._index.mode_indices[0]
        == result._storage._index.mode_indices[0]
    )
    assert (
        result_cpp._storage._index.mode_indices[1][0].tolist()
        == result._storage._index.mode_indices[1][0].tolist()
    )
    assert (
        result_cpp._storage._index.mode_indices[1][1].tolist()
        == result._storage._index.mode_indices[1][1].tolist()
    )


def test_elemwise_2d_add_csr_coo():
    a, b, result, _ = generate_2d_tensors("ds", "oo", "ds")
    result_cpp = a + b

    assert (
        result_cpp._storage._value.tolist() == result._storage._value.tolist()
    ), "Values are different"
    assert (
        result_cpp._storage._index.mode_indices[0]
        == result._storage._index.mode_indices[0]
    )
    assert (
        result_cpp._storage._index.mode_indices[1][0].tolist()
        == result._storage._index.mode_indices[1][0].tolist()
    )
    assert (
        result_cpp._storage._index.mode_indices[1][1].tolist()
        == result._storage._index.mode_indices[1][1].tolist()
    )


def test_elemwise_2d_add_csr_csc():
    # CSR = CSR + CSC
    a, b, result, _ = generate_2d_tensors("ds", "ds", "ds", [0, 1], [1, 0], [0, 1])
    result_cpp = a + b

    assert (
        result_cpp._storage._value.tolist() == result._storage._value.tolist()
    ), "Values are different"
    assert (
        result_cpp._storage._index.mode_indices[0]
        == result._storage._index.mode_indices[0]
    )
    assert (
        result_cpp._storage._index.mode_indices[1][0].tolist()
        == result._storage._index.mode_indices[1][0].tolist()
    )
    assert (
        result_cpp._storage._index.mode_indices[1][1].tolist()
        == result._storage._index.mode_indices[1][1].tolist()
    )


def test_elemwise_2d_add_coo_mixed():
    # COO [0,1] = COO [0,1] + COO [1, 0]
    a, b, result, _ = generate_2d_tensors("oo", "oo", "oo", [0, 1], [1, 0], [0, 1])
    result_cpp = a + b

    assert (
        result_cpp._storage._value.tolist() == result._storage._value.tolist()
    ), "Values are different"
    assert (
        result_cpp._storage._index.mode_indices[0][0].tolist()
        == result._storage._index.mode_indices[0][0].tolist()
    )
    assert (
        result_cpp._storage._index.mode_indices[1][0].tolist()
        == result._storage._index.mode_indices[1][0].tolist()
    )


def test_elemwise_2d_add_basic():
    a, b, _, _ = generate_2d_tensors("ds", "ds", "ds")
    result = a + b
    assert result is not None


def test_einsum_2d_discordant_coo():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )
    # ik,jk->ji computes C[j,i] = sum_k A[i,k]*B[j,k]
    expected = torch.einsum("ik,jk->ji", tensor_a_torch, tensor_b_torch)

    a = STensor.from_torch(tensor_a_torch, "a", [0, 1]).to_sparse("oo")
    b = STensor.from_torch(tensor_b_torch, "b", [1, 0]).to_sparse("oo")
    result_cpp = einsum("ik,jk->ji", a, b, format="oo")

    result_dense = result_cpp.to_torch()
    assert torch.allclose(result_dense, expected), f"Values differ:\n{result_dense}\nvs\n{expected}"
