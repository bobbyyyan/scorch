import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from tqdm import tqdm

from utils import load_dataset


class GCNConv(MessagePassing):
    def __init__(
        self, in_channels, out_channels, normalize=False, add_self_loops=False
    ):
        super().__init__(aggr="add")  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        norm = None
        if self.normalize:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm=None):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        if norm is not None:
            return norm.view(-1, 1) * x_j
        else:
            return x_j


# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

    def forward(self, x, edge_index):
        start_time = time.perf_counter()
        x = self.conv1(x, edge_index)
        end_time = time.perf_counter()
        print(f"self.conv1(x, edge_index) took {end_time - start_time} s")

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        start_time = time.perf_counter()
        x = self.conv2(x, edge_index)
        end_time = time.perf_counter()
        print(f"self.conv2(x, edge_index) took {end_time - start_time} s")

        return F.log_softmax(x, dim=1)


def train(model, data, device, dataset_name, split_idx=None):
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Create mask for train data
    if dataset_name == "ogbn-arxiv":
        train_mask = torch.zeros(data.num_nodes, dtype=bool)
        train_mask[split_idx["train"]] = True
    else:
        train_mask = data.train_mask

    # Train the model
    model.train()
    for epoch in tqdm(range(200), desc="Training", unit="epoch"):
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.nll_loss(out[train_mask], data.y.view(-1)[train_mask].to(device))
        loss.backward()
        optimizer.step()

    # Save weights
    # Make weights/ directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), f"weights/gcn_{dataset_name.lower()}_weights.pth")


def inference(model, data, device, dataset_name, split_idx=None):
    # Load weights and prepare for inference
    model.load_state_dict(torch.load(f"weights/gcn_{dataset_name.lower()}_weights.pth"))
    model.eval()

    # Create mask for test data
    if split_idx and dataset_name in ["ogbn-arxiv"]:
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[split_idx["test"]] = True
    else:
        test_mask = data.test_mask

    # Perform inference and measure time
    start_time = time.time()

    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        pred = logits.argmax(dim=1)

    inference_time = time.time() - start_time

    # Calculate accuracy
    correct = float((pred[test_mask] == data.y.view(-1)[test_mask]).sum().item())
    accuracy = correct / test_mask.sum().item()

    print(f"Inference time: {inference_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train and test a GCN.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help='Mode to run: "train" or "test".',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "pubmed", "citeseer", "reddit", "ogbn-arxiv", "dblp"],
    )
    args = parser.parse_args()

    # Convert dataset to lowercase
    args.dataset = args.dataset.lower()

    # Load dataset
    dataset, split_idx = load_dataset(args.dataset)
    data = dataset[0]

    # Define dimensions
    in_channels = dataset.num_features
    hidden_channels = 128
    out_channels = dataset.num_classes

    # Initialize model
    device = torch.device("cpu")
    model = GCN(in_channels, hidden_channels, out_channels).to(device)

    if args.mode.lower() == "train":
        train(model, data, device, args.dataset, split_idx)
    elif args.mode.lower() == "test":
        inference(model, data, device, args.dataset, split_idx)
    else:
        raise ValueError(
            f"Mode {args.mode} not recognized. Choose from 'train' or 'test'."
        )


if __name__ == "__main__":
    main()
