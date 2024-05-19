import argparse
import time

import scorch as torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_dataset

import warnings

# Suppress specific PyTorch UserWarning about Sparse CSR tensor support
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Sparse CSR tensor support is in beta state.*",
)


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, x, adjacency):
        out = torch.matmul(adjacency, x, format="dd")
        out = self.lin(out)
        out += self.bias

        return out


class CustomGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGCN, self).__init__()
        self.conv1 = GraphConvolution(in_channels, hidden_channels)
        self.conv2 = GraphConvolution(hidden_channels, out_channels)

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
                batch_adjacency = torch.from_coo(
                    indices=batch_data.edge_index,
                    values=torch.ones(batch_data.edge_index.shape[1]),
                    shape=(batch_data.num_nodes, batch_data.num_nodes),
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

        start_time = time.perf_counter()

        if hasattr(data, "adj_t"):
            if data.adj_t.is_sparse_csr:
                adjacency = torch.from_csr(data.adj_t)
            else:
                adjacency = (
                    data.adj_t.to_dense().clone().detach().to(torch.float).to(device)
                )
                adjacency = torch.from_torch(adjacency)
                adjacency = adjacency.to_sparse("ds")
        else:
            adjacency = torch.from_coo(
                indices=data.edge_index,
                values=torch.ones(data.edge_index.shape[1]),
                shape=(data.num_nodes, data.num_nodes),
            )

        end_time = time.perf_counter()
        print(f"Adj matrix construction took {end_time - start_time} s")

        # Perform inference and measure time
        start_time = time.perf_counter()

        with torch.no_grad():
            logits = model(x, adjacency)
            pred = logits.argmax(dim=1)

        correct = float((pred[test_mask] == data.y.view(-1)[test_mask]).sum().item())
        accuracy = correct / test_mask.sum().item()

    inference_time = time.perf_counter() - start_time

    print(f"Inference time: {inference_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test a CustomGCN with Scorch.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help='Dataset to use. Options are "cora", "pubmed", "citeseer", "reddit", or "ogbn-arxiv".',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for mini-batch inference (only applicable for Reddit dataset).",
    )
    args = parser.parse_args()

    # Convert dataset to lowercase
    args.dataset = args.dataset.lower()

    # Load dataset
    if args.dataset == "reddit":
        dataset, split_idx = load_dataset(args.dataset)
    else:
        dataset, split_idx = load_dataset(args.dataset, to_sparse_tensor=True)
    data = dataset[0]

    # Define the dimensions
    in_channels = dataset.num_features
    hidden_channels = 128  # This should match the hidden dimension used in pyg_gcn.py
    out_channels = dataset.num_classes

    # Initialize model
    device = torch.device("cpu")
    model = CustomGCN(in_channels, hidden_channels, out_channels).to(device)

    # Inference
    inference(model, data, device, args.dataset, split_idx, args.batch_size)


if __name__ == "__main__":
    main()
