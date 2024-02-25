import os

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import ToSparseTensor


def load_dataset(dataset_name, to_sparse_tensor=False):
    dataset_name = dataset_name.lower()

    split_idx = None

    transform = ToSparseTensor(layout=torch.sparse_csr) if to_sparse_tensor else None

    if dataset_name in ["cora", "pubmed", "citeseer"]:
        dataset = Planetoid(
            root=os.path.join(os.getcwd(), "data"),
            name=dataset_name,
            transform=transform,
        )
    elif dataset_name == "reddit":
        dataset = Reddit(
            root=os.path.join(os.getcwd(), "data/reddit"),
            transform=transform,
        )
    elif dataset_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv",
            root=os.path.join(os.getcwd(), "data"),
            transform=transform,
        )
        split_idx = dataset.get_idx_split()
    else:
        raise ValueError(
            f"Dataset {dataset_name} not recognized. "
            "Choose from 'cora', 'pubmed', 'citeseer', 'reddit', or 'ogbn-arxiv'."
        )

    return dataset, split_idx


def modify_state_dict_pyg_to_torch(state_dict):
    # replace .lin. with .linear.
    # and replace .bias. with .linear.bias.
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(".lin.", ".linear.")
        if ".bias" in new_key:
            new_key = new_key.replace(".bias", ".linear.bias")
        new_state_dict[new_key] = value
    return new_state_dict
