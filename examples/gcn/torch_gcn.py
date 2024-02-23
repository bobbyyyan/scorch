import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_dataset, modify_state_dict_pyg_to_torch

from torch_scatter import scatter_add

args_dict = {}


class GCNConvScatterGather(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConvScatterGather, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.linear(x)

        node_dim = 0

        src_nodes = edge_index[0]  # source nodes
        source_node_feats = x.index_select(node_dim, src_nodes)  # source node features

        # Scatter source node features to destination nodes
        dest_nodes = edge_index[1]
        out = scatter_add(source_node_feats, dest_nodes, dim=0, dim_size=x.size(0))

        return out


class GCNScatterGather(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNScatterGather, self).__init__()
        self.conv1 = GCNConvScatterGather(in_channels, hidden_channels)
        self.conv2 = GCNConvScatterGather(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        start_time = time.perf_counter()
        x = self.conv1(x, edge_index)
        end_time = time.perf_counter()
        print(f"\nself.conv1(x, edge_index) took {end_time - start_time} s")

        x = F.relu(x)

        start_time = time.perf_counter()
        x = self.conv2(x, edge_index)
        end_time = time.perf_counter()
        print(f"\nself.conv2(x, edge_index) took {end_time - start_time} s")

        return x


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, adjacency):
        if not args_dict["sparse"]:
            # Print sparsity level
            x_nnz = torch.count_nonzero(x)
            x_sparsity = 1 - x_nnz.item() / (x.shape[0] * x.shape[1])
            print(f"\nx Sparsity: {x_sparsity * 100:.2f}%")

            adj_nnz = torch.count_nonzero(adjacency)
            adj_sparsity = 1 - adj_nnz.item() / (
                adjacency.shape[0] * adjacency.shape[1]
            )
            print(f"Adjacency Sparsity: {adj_sparsity * 100:.2f}%")

        start_time = time.perf_counter()
        out = torch.matmul(adjacency, x)
        end_time = time.perf_counter()
        print(f"\ntorch.matmul(adjacency, x) took {end_time - start_time} s")

        out = self.linear(out)
        return out


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, adjacency):
        start_time = time.perf_counter()
        x = self.conv1(x, adjacency)
        end_time = time.perf_counter()
        print(f"self.conv1(x, adjacency) took {end_time - start_time} s")

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        start_time = time.perf_counter()
        x = self.conv2(x, adjacency)
        end_time = time.perf_counter()
        print(f"\nself.conv2(x, adjacency) took {end_time - start_time} s")

        return F.log_softmax(x, dim=1)


def inference(model, data, device, dataset_name, split_idx=None):
    # Load weights and prepare for inference
    state_dict = torch.load(f"weights/gcn_{dataset_name.lower()}_weights.pth")
    new_state_dict = modify_state_dict_pyg_to_torch(state_dict)
    model.load_state_dict(new_state_dict)
    model.eval()

    x = data.x.clone().detach().to(torch.float).to(device)

    # Create mask for test data
    if split_idx and dataset_name in ["ogbn-arxiv"]:
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[split_idx["test"]] = True
    else:
        test_mask = data.test_mask

    adjacency = None
    if args_dict["gather"]:
        adjacency = data.edge_index
    else:
        if hasattr(data, "adj_t"):
            adjacency = (
                data.adj_t.to_dense().clone().detach().to(torch.float).to(device)
            )

            # Convert adjacency matrix to a PyTorch sparse tensor
            if args_dict["sparse"]:
                # x = x.to_sparse_csr()
                adjacency = adjacency.to_sparse_csr()
        else:
            adjacency = torch.sparse_coo_tensor(
                indices=data.edge_index,
                values=torch.ones(data.edge_index.shape[1]),
                size=(data.num_nodes, data.num_nodes),
            )

    # Perform inference and measure time
    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(x, adjacency)
        pred = logits.argmax(dim=1)

    inference_time = time.perf_counter() - start_time

    # Calculate accuracy
    correct = float((pred[test_mask] == data.y[test_mask]).sum().item())
    accuracy = correct / test_mask.sum().item()

    print(f"\nInference time: {inference_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test a GCN with PyTorch.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help='Dataset to use.',
        choices=["cora", "pubmed", "citeseer", "reddit", "ogbn-arxiv"],
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        default=False,
        help="Use sparse adjacency matrix.",
    )
    # Add argument to use gather-scatter implementation
    parser.add_argument(
        "--gather",
        dest="gather",
        action="store_true",
        default=False,
        help="Use gather-scatter implementation.",
    )

    args = parser.parse_args()

    # Convert dataset to lowercase
    args.dataset = args.dataset.lower()

    global args_dict
    args_dict = vars(args)

    # Load dataset
    if args.gather:
        dataset, split_idx = load_dataset(args.dataset, to_sparse_tensor=False)
    else:
        if args.dataset in ["ogbn-arxiv"]:
            if args_dict["sparse"]:
                dataset, split_idx = load_dataset(args.dataset)
            else:
                dataset, split_idx = load_dataset(args.dataset, to_sparse_tensor=True)
        else:
            dataset, split_idx = load_dataset(args.dataset, to_sparse_tensor=True)
    data = dataset[0]

    # Define dimensions
    in_channels = dataset.num_features
    hidden_channels = 128
    out_channels = dataset.num_classes

    # Initialize model
    if args.gather:
        model = GCNScatterGather(in_channels, hidden_channels, out_channels)
    else:
        model = GCN(in_channels, hidden_channels, out_channels)

    device = torch.device("cpu")

    # Inference
    inference(model, data, device, args.dataset, split_idx)


if __name__ == "__main__":
    main()
