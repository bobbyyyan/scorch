import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_dataset

from torch_scatter import scatter_add

import warnings

# Suppress specific PyTorch UserWarning about Sparse CSR tensor support
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Sparse CSR tensor support is in beta state.*",
)

args_dict = {}


class GCNConvScatterGather(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConvScatterGather, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, x, edge_index):
        x = self.lin(x)

        node_dim = 0

        src_nodes = edge_index[0]  # source nodes
        source_node_feats = x.index_select(node_dim, src_nodes)  # source node features

        # Scatter source node features to destination nodes
        dest_nodes = edge_index[1]
        out = scatter_add(source_node_feats, dest_nodes, dim=0, dim_size=x.size(0))
        out += self.bias
        return out


class GCNScatterGather(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNScatterGather, self).__init__()
        self.conv1 = GCNConvScatterGather(in_channels, hidden_channels)
        self.conv2 = GCNConvScatterGather(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, x, adjacency):
        # if not args_dict["sparse"]:
        #     # Print sparsity level
        #     x_nnz = torch.count_nonzero(x)
        #     x_sparsity = 1 - x_nnz.item() / (x.shape[0] * x.shape[1])
        #     print(f"x Sparsity: {x_sparsity * 100:.2f}%")

        #     adj_nnz = torch.count_nonzero(adjacency)
        #     adj_sparsity = 1 - adj_nnz.item() / (
        #         adjacency.shape[0] * adjacency.shape[1]
        #     )
        #     print(f"Adjacency Sparsity: {adj_sparsity * 100:.2f}%")

        out = torch.matmul(adjacency, x)
        out = self.lin(out)
        out += self.bias

        return out


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adjacency)
        return F.log_softmax(x, dim=1)


def inference(model, data, device, dataset_name, split_idx=None, batch_size=1):
    # Load weights and prepare for inference
    state_dict = torch.load(f"weights/gcn_{dataset_name.lower()}_weights.pth")
    model.load_state_dict(state_dict)
    model.eval()

    # Create mask for test data
    if split_idx and dataset_name in ["ogbn-arxiv"]:
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[split_idx["test"]] = True
    else:
        test_mask = data.test_mask

    # Perform inference and measure time
    start_time = time.perf_counter()

    if dataset_name == "reddit":
        # Mini-batch inference for Reddit dataset
        test_nodes = test_mask.nonzero(as_tuple=True)[0]
        perm = torch.randperm(test_nodes.size(0))
        num_batches = (test_nodes.size(0) + batch_size - 1) // batch_size
        batches = tqdm(range(num_batches), desc="Inference", unit="batch")

        correct = 0
        total = 0

        for batch_idx in batches:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, test_nodes.size(0))
            batch_nodes = test_nodes[perm[batch_start:batch_end]]
            batch_data = data.subgraph(batch_nodes)

            with torch.no_grad():
                batch_x = batch_data.x.clone().detach().to(torch.float).to(device)
                batch_adjacency = torch.sparse_coo_tensor(
                    indices=batch_data.edge_index,
                    values=torch.ones(batch_data.edge_index.shape[1]),
                    size=(batch_data.num_nodes, batch_data.num_nodes),
                )
                logits = model(batch_x, batch_adjacency)
                pred = logits.argmax(dim=1)

            batch_correct = int((pred == batch_data.y.view(-1)).sum())
            batch_total = batch_nodes.size(0)

            correct += batch_correct
            total += batch_total

        accuracy = correct / total
    else:
        # Full-batch inference for other datasets
        x = data.x.clone().detach().to(torch.float).to(device)
        adjacency = None

        if args_dict["gather"]:
            adjacency = data.edge_index
        else:
            if hasattr(data, "adj_t"):
                if args_dict["sparse"]:
                    adjacency = data.adj_t
                else:
                    adjacency = (
                        data.adj_t.to_dense().clone().detach().to(torch.float).to(device)
                    )
            else:
                if data.edge_index.dim() == 2 and data.edge_index.shape[0] != 2:
                    edge_index = data.edge_index.t()
                else:
                    edge_index = data.edge_index

                adjacency = torch.sparse_coo_tensor(
                    indices=edge_index,  # This should be a 2 x num_edges tensor
                    values=torch.ones(edge_index.shape[1]),
                    size=(data.num_nodes, data.num_nodes),
                )

        # Perform inference and measure time
        start_time = time.perf_counter()

        with torch.no_grad():
            logits = model(x, adjacency)
            pred = logits.argmax(dim=1)

        # Calculate accuracy
        correct = float((pred[test_mask] == data.y.view(-1)[test_mask]).sum().item())
        accuracy = correct / test_mask.sum().item()

    inference_time = time.perf_counter() - start_time

    print(f"\nInference time: {inference_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test a GCN with PyTorch.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cora", "pubmed", "citeseer", "reddit", "ogbn-arxiv"],
        default="cora",
        help="Dataset to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for mini-batch inference (only applicable for Reddit dataset).",
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
        if args.dataset == "reddit":
            dataset, split_idx = load_dataset(args.dataset)
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
    inference(model, data, device, args.dataset, split_idx, args.batch_size)


if __name__ == "__main__":
    main()
