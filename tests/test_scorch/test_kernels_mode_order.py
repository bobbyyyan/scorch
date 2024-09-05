import pdb

import torch

import random
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
        2.0,
        2.0,
        3.0,
        4.0,
        5.0,
        2.0,
        4.0,
        3.0,
        3.0,
        1.0,
        5.0,
        3.0,
        6.0,
    ]
    assert c_csr.storage.index.mode_indices[1][0].tolist() == [0, 5, 7, 9, 10, 13]
    assert c_csr.storage.index.mode_indices[1][1].tolist() == [
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        0,
        2,
        3,
        0,
        3,
        4,
    ]
    assert c_csr.storage.index.mode_order == [0, 1]

    c_csc = a_csc + b_csr

    assert c_csc.storage.value.tolist() == [
        2.0,
        2.0,
        3.0,
        5.0,
        2.0,
        4.0,
        3.0,
        3.0,
        4.0,
        1.0,
        3.0,
        5.0,
        6.0,
    ]
    assert c_csc.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 8, 11, 13]
    assert c_csc.storage.index.mode_indices[1][1].tolist() == [
        0,
        1,
        2,
        4,
        0,
        1,
        0,
        2,
        0,
        3,
        4,
        0,
        4,
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
        print(mode_order)
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


def test_einsum_2d_csr_csc_concordant():
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

    a_csr = STensor.from_torch(tensor_a_torch, "a_csr", [0, 1]).to_sparse("oo")
    a_csc = STensor.from_torch(tensor_a_torch, "a_csc", [1, 0]).to_sparse("oo")

    b_csr = STensor.from_torch(tensor_b_torch, "b_csr", [0, 1]).to_sparse("oo")
    b_csc = STensor.from_torch(tensor_b_torch, "b_csc", [1, 0]).to_sparse("oo")

    # c_csr = einsum("ik,kj->ij", a_csr, b_csc, format="ds", output_mode_order=[0, 1])
    # assert c_csr.storage.value.tolist() == [1., 4., 19., 5., 2., 4., 3., 5., 15., 5.]
    # assert c_csr.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 7, 7, 10]
    # assert c_csr.storage.index.mode_indices[1][1].tolist() == [0, 1, 3, 4, 0, 1, 0, 0, 3, 4]
    # assert c_csr.storage.index.mode_order == [0, 1]

    c_csr = einsum("ik,kj->ij", a_csc, b_csr, format="oo", output_mode_order=[0, 1])
    assert torch.allclose(c_csr.values, c_torch_coo.values())
    assert torch.allclose(
        c_csr.index.mode_indices[0][0], c_torch_coo.indices()[0].int()
    )
    assert torch.allclose(
        c_csr.index.mode_indices[1][0], c_torch_coo.indices()[1].int()
    )
    assert c_csr.storage.index.mode_order == [0, 1]

    c_csc = einsum("ik,kj->ij", a_csc, b_csr, format="ds", output_mode_order=[1, 0])
    assert torch.allclose(c_csc.values, torch.Tensor([1, 2, 3, 5, 4, 4, 19, 15, 5, 5]))
    assert c_csc.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 6, 8, 10]
    assert c_csc.storage.index.mode_indices[1][1].tolist() == [
        0,
        1,
        2,
        4,
        0,
        1,
        0,
        4,
        0,
        4,
    ]
    assert c_csc.storage.index.mode_order == [1, 0]

    c_csc = einsum("ik,kj->ij", a_csr, b_csc, format="ds", output_mode_order=[1, 0])
    assert torch.allclose(c_csc.values, torch.Tensor([1, 2, 3, 5, 4, 4, 19, 15, 5, 5]))
    assert c_csc.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 6, 8, 10]
    assert c_csc.storage.index.mode_indices[1][1].tolist() == [
        0,
        1,
        2,
        4,
        0,
        1,
        0,
        4,
        0,
        4,
    ]
    assert c_csc.storage.index.mode_order == [1, 0]


def test_einsum_2d_csc_discordant():
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
        1.0,
        4.0,
        19.0,
        5.0,
        2.0,
        4.0,
        3.0,
        5.0,
        15.0,
        5.0,
    ]
    assert c_csr.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 7, 7, 10]
    assert c_csr.storage.index.mode_indices[1][1].tolist() == [
        0,
        1,
        3,
        4,
        0,
        1,
        0,
        0,
        3,
        4,
    ]
    assert c_csr.storage.index.mode_order == [0, 1]

    c_csc = einsum("ik,kj->ij", a_csr, b_csr, format="ds", output_mode_order=[1, 0])
    assert c_csc.storage.value.tolist() == [
        1.0,
        2.0,
        3.0,
        5.0,
        4.0,
        4.0,
        19.0,
        15.0,
        5.0,
        5.0,
    ]
    assert c_csc.storage.index.mode_indices[1][0].tolist() == [0, 4, 6, 6, 8, 10]
    assert c_csc.storage.index.mode_indices[1][1].tolist() == [
        0,
        1,
        2,
        4,
        0,
        1,
        0,
        4,
        0,
        4,
    ]
    assert c_csc.storage.index.mode_order == [1, 0]
