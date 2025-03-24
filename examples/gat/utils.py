import os

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Reddit, DBLP
from torch_geometric.transforms import ToSparseTensor


def load_dataset(dataset_name, to_sparse_tensor=False):
    dataset_name = dataset_name.lower()

    split_idx = None

    transform = ToSparseTensor(layout=torch.sparse_csr) if to_sparse_tensor else None

    if dataset_name in ["cora", "pubmed", "citeseer"]:
        dataset = Planetoid(
            root=os.path.join(os.getcwd(), "data", dataset_name),
            name=dataset_name,
            transform=transform,
        )
    elif dataset_name == "reddit":
        dataset = Reddit(
            root=os.path.join(os.getcwd(), "data", dataset_name),
            transform=transform,
        )
    elif dataset_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv",
            root=os.path.join(os.getcwd(), "data", dataset_name),
            transform=transform,
        )
        split_idx = dataset.get_idx_split()
    elif dataset_name == "dblp":
        dataset = DBLP(
            root=os.path.join(os.getcwd(), "data", dataset_name),
            transform=transform,
        )
        # You may need to define your own split here or leave as None if unavailable.
    else:
        raise ValueError(
            f"Dataset {dataset_name} not recognized. "
            "Choose from 'cora', 'pubmed', 'citeseer', 'reddit', 'ogbn-arxiv', or 'dblp'."
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
