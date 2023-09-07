import argparse
import time

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from utils import load_dataset


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
        print(f"\nself.conv1(x, edge_index) took {end_time - start_time} s")

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
    torch.save(model.state_dict(), f"weights/gcn_{dataset_name.lower()}_weights.pth")


def inference(model, data, device, dataset_name, split_idx=None):
    # Load weights and prepare for inference
    model.load_state_dict(torch.load(f"weights/gcn_{dataset_name.lower()}_weights.pth"))
    model.eval()

    # Create mask for test data
    if dataset_name == "ogbn-arxiv":
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
        "--mode", type=str, default="test", help='Mode to run: "train" or "test".'
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help='Dataset to use. Options are "cora", "pubmed", "citeseer", "reddit", or "ogbn-arxiv".',
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
