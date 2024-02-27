import argparse
import time

import scorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_dataset, modify_state_dict_pyg_to_torch

import ipdb


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, adjacency):
        start_time = time.perf_counter()
        time_dict = {}
        out = scorch.matmul(adjacency, x, format="dd", time_dict=time_dict)
        end_time = time.perf_counter()
        print(f"\nscorch.matmul(adjacency, x) took {end_time - start_time} s")
        eval_time = time_dict["eval_time"]
        print(f"eval_time: {eval_time}")

        start_time = time.perf_counter()
        out = out.to_torch()
        end_time = time.perf_counter()
        print(f"out.to_torch() took {end_time - start_time} s")

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
        print(f"self.conv2(x, adjacency) took {end_time - start_time} s")

        return F.log_softmax(x, dim=1)


def inference(model, data, device, dataset_name, split_idx=None):
    # Load weights and prepare for inference
    state_dict = torch.load(f"weights/gcn_{dataset_name.lower()}_weights.pth")
    new_state_dict = modify_state_dict_pyg_to_torch(state_dict)
    model.load_state_dict(new_state_dict)
    model.eval()

    x = data.x.clone().detach().to(torch.float).to(device)
    start_time = time.perf_counter()
    x = scorch.from_torch(x)
    # x = x.to_sparse("ds")
    end_time = time.perf_counter()
    # print(f"scorch.from_torch(x) took {end_time - start_time} s")

    start_time = time.perf_counter()

    if hasattr(data, "adj_t"):
        # ipdb.set_trace()
        if data.adj_t.is_sparse_csr:
            adjacency = scorch.from_csr(data.adj_t)
        else:
            adjacency = (
                data.adj_t.to_dense().clone().detach().to(torch.float).to(device)
            )
            adjacency = scorch.from_torch(adjacency)
            adjacency = adjacency.to_sparse("ds")
    else:
        adjacency = scorch.from_coo(
            indices=data.edge_index,
            values=torch.ones(data.edge_index.shape[1]),
            shape=(data.num_nodes, data.num_nodes),
        )
    end_time = time.perf_counter()
    # print(f"Adj matrix construction took {end_time - start_time} s")

    # Create mask for test data
    if split_idx and dataset_name in ["ogbn-arxiv"]:
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[split_idx["test"]] = True
    else:
        test_mask = data.test_mask

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
    parser = argparse.ArgumentParser(description="Test a CustomGCN with Scorch.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help='Dataset to use. Options are "cora", "pubmed", "citeseer", or "reddit".',
    )
    args = parser.parse_args()

    # Convert dataset to lowercase
    args.dataset = args.dataset.lower()

    # Load dataset
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
    inference(model, data, device, args.dataset, split_idx)


if __name__ == "__main__":
    main()
