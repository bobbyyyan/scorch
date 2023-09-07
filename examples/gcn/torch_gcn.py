import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_dataset, modify_state_dict_pyg_to_torch

args_dict = {}


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
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


class CustomGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGCN, self).__init__()
        self.conv1 = GraphConvolution(in_channels, hidden_channels)
        self.conv2 = GraphConvolution(hidden_channels, out_channels)

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


def inference(model, data, device, dataset_name):
    # Load weights and prepare for inference
    state_dict = torch.load(f"weights/gcn_{dataset_name.lower()}_weights.pth")
    new_state_dict = modify_state_dict_pyg_to_torch(state_dict)
    model.load_state_dict(new_state_dict)
    model.eval()

    x = data.x.clone().detach().to(torch.float).to(device)
    adjacency = data.adj_t.to_dense().clone().detach().to(torch.float).to(device)

    # Convert adjacency matrix to a PyTorch sparse tensor
    if args_dict["sparse"]:
        # x = x.to_sparse_csr()
        adjacency = adjacency.to_sparse_csr()

    # Perform inference and measure time
    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(x, adjacency)
        pred = logits.argmax(dim=1)

    inference_time = time.perf_counter() - start_time

    # Calculate accuracy
    correct = float((pred[data.test_mask] == data.y[data.test_mask]).sum().item())
    accuracy = correct / data.test_mask.sum().item()

    print(f"\nInference time: {inference_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test a CustomGCN with Scorch.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help='Dataset to use. Options are "cora", "pubmed", "citeseer", or "reddit".',
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
    dataset, split_idx = load_dataset(args.dataset, to_sparse_tensor=True)
    data = dataset[0]

    # Define dimensions
    in_channels = dataset.num_features
    hidden_channels = 128
    out_channels = dataset.num_classes

    # Initialize model
    model = CustomGCN(in_channels, hidden_channels, out_channels)

    device = torch.device("cpu")

    # Inference
    inference(model, data, device, args.dataset)


if __name__ == "__main__":
    main()
