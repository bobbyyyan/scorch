import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_dataset

from torch_scatter import scatter_add

args_dict = {}


class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0):
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        self.bias = nn.Parameter(torch.empty(heads * out_channels))
        nn.init.xavier_normal_(self.att_src)
        nn.init.xavier_normal_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        row, col = edge_index

        x = self.lin(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(-1, self.heads, self.out_channels)
        x_i, x_j = torch.index_select(x, 0, row), torch.index_select(x, 0, col)

        alpha = F.leaky_relu((x_i * self.att_src).sum(dim=-1) + (x_j * self.att_dst).sum(dim=-1), negative_slope=self.negative_slope)
        alpha = F.softmax(alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = scatter_add(alpha.unsqueeze(-1) * x_j, row.unsqueeze(-1), dim=0, dim_size=x.size(0))

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out += self.bias

        return out


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def inference(model, data, device, dataset_name, split_idx=None, batch_size=1):
    # Load weights and prepare for inference
    state_dict = torch.load(f"weights/gat_{dataset_name.lower()}_weights.pth")

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
                batch_edge_index = batch_data.edge_index
                logits = model(batch_x, batch_edge_index)
                pred = logits.argmax(dim=1)

            batch_correct = int((pred == batch_data.y.view(-1)).sum())
            batch_total = batch_nodes.size(0)

            correct += batch_correct
            total += batch_total

        accuracy = correct / total
    else:
        # Full-batch inference for other datasets
        x = data.x.clone().detach().to(torch.float).to(device)
        edge_index = data.edge_index

        # Perform inference and measure time
        start_time = time.perf_counter()

        with torch.no_grad():
            logits = model(x, edge_index)
            pred = logits.argmax(dim=1)

        # Calculate accuracy
        correct = float((pred[test_mask] == data.y.view(-1)[test_mask]).sum().item())
        accuracy = correct / test_mask.sum().item()

    inference_time = time.perf_counter() - start_time

    print(f"\nInference time: {inference_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test a GAT with PyTorch.")
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

    args = parser.parse_args()

    # Convert dataset to lowercase
    args.dataset = args.dataset.lower()

    global args_dict
    args_dict = vars(args)

    # Load dataset
    if args.dataset == "reddit":
        dataset, split_idx = load_dataset(args.dataset)
    else:
        dataset, split_idx = load_dataset(args.dataset, to_sparse_tensor=True)

    data = dataset[0]

    # Define dimensions
    in_channels = dataset.num_features
    hidden_channels = 8
    out_channels = dataset.num_classes
    num_heads = 8

    # Initialize model
    model = GAT(in_channels, hidden_channels, out_channels, num_heads)

    device = torch.device("cpu")

    # Inference
    inference(model, data, device, args.dataset, split_idx, args.batch_size)


if __name__ == "__main__":
    main()
