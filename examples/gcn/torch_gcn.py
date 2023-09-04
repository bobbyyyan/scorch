import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as datasets
from torch_geometric.transforms import ToSparseTensor
import argparse


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
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
        # Print sparsity level
        x_nnz = torch.count_nonzero(x)
        adj_nnz = torch.count_nonzero(adjacency)

        x_sparsity = 1 - x_nnz.item() / (x.shape[0] * x.shape[1])
        adj_sparsity = 1 - adj_nnz.item() / (adjacency.shape[0] * adjacency.shape[1])

        print(f"\nx Sparsity: {x_sparsity * 100:.2f}%")
        print(f"Adjacency Sparsity: {adj_sparsity * 100:.2f}%")

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


def load_dataset(dataset_name):
    if dataset_name in ["cora", "pubmed", "citeseer"]:
        dataset = datasets.Planetoid(
            root="data",
            name=dataset_name,
            transform=ToSparseTensor(),
        )
    elif dataset_name == "reddit":
        dataset = datasets.Reddit(root="data/reddit", transform=ToSparseTensor())
    else:
        raise ValueError(
            f"Dataset {dataset_name} not recognized. Choose from 'cora', 'pubmed', 'citeseer', or 'reddit'."
        )
    return dataset


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


def inference(model, data, device, dataset_name):
    # Load weights and prepare for inference
    state_dict = torch.load(f"weights/gcn_{dataset_name.lower()}_weights.pth")
    new_state_dict = modify_state_dict_pyg_to_torch(state_dict)
    model.load_state_dict(new_state_dict)
    model.eval()

    x = data.x.clone().detach().to(torch.float)
    adjacency = data.adj_t.to_dense().clone().detach().to(torch.float)

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
    args = parser.parse_args()

    # Convert dataset to lowercase
    args.dataset = args.dataset.lower()

    # Load dataset
    dataset = load_dataset(args.dataset)
    data = dataset[0]

    # Define the dimensions
    in_channels = dataset.num_features
    hidden_channels = 128  # This should match the hidden dimension used in pyg_gcn.py
    out_channels = dataset.num_classes

    # Initialize the CustomGCN model
    model = CustomGCN(in_channels, hidden_channels, out_channels)

    device = torch.device("cpu")

    # Inference
    inference(model, data, device, args.dataset)


if __name__ == "__main__":
    main()
