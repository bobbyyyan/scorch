import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import scorch
import torch_geometric.datasets as datasets
from torch_geometric.transforms import ToSparseTensor


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        out = torch.matmul(adjacency, x)
        out = self.linear(out)
        return out


class CustomGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGCN, self).__init__()
        self.conv1 = GraphConvolution(in_channels, hidden_channels)
        self.conv2 = GraphConvolution(hidden_channels, out_channels)

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = self.conv2(x, adjacency)
        return x


class GraphConvolutionScorch(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionScorch, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        out = scorch.matmul(adjacency, x).to_torch()
        out = self.linear(out)
        return out


class CustomGCNScorch(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGCNScorch, self).__init__()
        self.conv1 = GraphConvolutionScorch(in_channels, hidden_channels)
        self.conv2 = GraphConvolutionScorch(hidden_channels, out_channels)

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = self.conv2(x, adjacency)
        return x


def train(model, dataset):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Iterate over the dataset
    for data in dataset:
        # Get the node features and adjacency matrix
        x = data.x
        adjacency = data.adj_t.to_dense()

        # Perform a forward pass
        logits = model(x, adjacency)

        # Compute the loss
        loss = criterion(logits, data.y)

        # Perform a backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the model weights
    torch.save(model.state_dict(), "weights/gcn_reddit_weights.pth")


def benchmark(model, scorch_model, dataset):
    # Load the saved weights
    model.load_state_dict(torch.load("weights/gcn_reddit_weights.pth"))
    model.eval()
    scorch_model.load_state_dict(torch.load("weights/gcn_reddit_weights.pth"))
    scorch_model.eval()

    correct = 0
    total = 0

    # Iterate over the dataset
    for data in dataset:
        # Get the node features and adjacency matrix
        x = data.x
        adjacency = data.adj_t.to_dense()

        # Get the index of the max log-probability
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.y.size(0)

        # Measure the inference time with PyTorch
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(x, adjacency)
        end_time = time.perf_counter()
        torch_time = end_time - start_time
        print(f"Inference time with PyTorch: {torch_time:.6f} seconds")

        adjacency_scorch = scorch.Tensor.from_torch(adjacency, "A").to_sparse("ds")

        # Measure the inference time with Scorch
        start_time = time.perf_counter()
        with torch.no_grad():
            output = scorch_model(x, adjacency_scorch)
        end_time = time.perf_counter()
        scorch_time = end_time - start_time
        print(f"Inference time with Scorch: {scorch_time:.6f} seconds")

        # Get the index of the max log-probability
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.y.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="GCN Benchmark")
    parser.add_argument(
        "--train", action="store_true", help="Train the model and save the weights"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark the inference time"
    )
    args = parser.parse_args()

    # Load the Reddit dataset
    dataset = datasets.Reddit(root="data/Reddit", transform=ToSparseTensor())

    # Initialize the GCN model
    in_channels = dataset.num_features
    hidden_channels = 16
    out_channels = dataset.num_classes
    model = CustomGCN(in_channels, hidden_channels, out_channels)
    scorch_model = CustomGCNScorch(in_channels, hidden_channels, out_channels)

    if args.train:
        train(model, dataset)
    elif args.benchmark:
        benchmark(model, scorch_model, dataset)


if __name__ == "__main__":
    main()
