import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree
from tqdm import tqdm

from utils import load_dataset


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=num_heads, dropout=0.6
        )
        self.conv2 = GATConv(
            hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train(model, data, device, dataset_name, split_idx=None, batch_size=1, epochs=200):
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Create mask for train data
    if dataset_name == "ogbn-arxiv":
        train_mask = torch.zeros(data.num_nodes, dtype=bool)
        train_mask[split_idx["train"]] = True
    else:
        train_mask = data.train_mask

    # Train the model
    model.train()
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        if dataset_name == "reddit":
            # Mini-batch training for Reddit dataset
            perm = torch.randperm(data.num_nodes)
            num_batches = (data.num_nodes + batch_size - 1) // batch_size
            batches = tqdm(
                range(num_batches), desc=f"Epoch {epoch+1}", unit="batch", leave=False
            )
            for batch_idx in batches:
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, data.num_nodes)
                batch_nodes = perm[batch_start:batch_end]
                batch_data = data.subgraph(batch_nodes)

                optimizer.zero_grad()
                out = model(batch_data.x.to(device), batch_data.edge_index.to(device))
                loss = F.nll_loss(
                    out[batch_data.train_mask],
                    batch_data.y.view(-1)[batch_data.train_mask].to(device),
                )
                loss.backward()
                optimizer.step()

                # Update progress bar
                batches.set_postfix({"Loss": loss.item()})
        else:
            # Full-batch training for other datasets
            optimizer.zero_grad()
            out = model(data.x.to(device), data.edge_index.to(device))
            loss = F.nll_loss(out[train_mask], data.y.view(-1)[train_mask].to(device))
            loss.backward()
            optimizer.step()

    # Save weights
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), f"weights/gat_{dataset_name.lower()}_weights.pth")


def inference(model, data, device, dataset_name, split_idx=None, batch_size=1):
    # Load weights and prepare for inference
    model.load_state_dict(torch.load(f"weights/gat_{dataset_name.lower()}_weights.pth"))
    model.eval()

    # Create mask for test data
    if split_idx and dataset_name in ["ogbn-arxiv"]:
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[split_idx["test"]] = True
    else:
        test_mask = data.test_mask

    # Perform inference and measure time
    start_time = time.time()

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
                logits = model(
                    batch_data.x.to(device), batch_data.edge_index.to(device)
                )
                pred = logits.argmax(dim=1)

            batch_correct = int((pred == batch_data.y.view(-1)).sum())
            batch_total = batch_nodes.size(0)

            correct += batch_correct
            total += batch_total

        accuracy = correct / total
    else:
        # Full-batch inference for other datasets
        with torch.no_grad():
            logits = model(data.x.to(device), data.edge_index.to(device))
            pred = logits.argmax(dim=1)

        correct = int((pred[test_mask] == data.y.view(-1)[test_mask]).sum())
        total = int(test_mask.sum())
        accuracy = correct / total

    inference_time = time.time() - start_time

    print(f"Inference time: {inference_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train and test a GAT.")
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
        choices=["cora", "pubmed", "citeseer", "reddit", "ogbn-arxiv"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for mini-batch (only applicable for Reddit dataset).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200).",
    )
    args = parser.parse_args()

    # Convert dataset to lowercase
    args.dataset = args.dataset.lower()

    # Load dataset
    dataset, split_idx = load_dataset(args.dataset)
    data = dataset[0]

    # Define dimensions
    in_channels = dataset.num_features
    hidden_channels = 8
    out_channels = dataset.num_classes
    num_heads = 8

    # Initialize model
    device = torch.device("cpu")
    model = GAT(in_channels, hidden_channels, out_channels, num_heads).to(device)

    if args.mode.lower() == "train":
        train(
            model,
            data,
            device,
            args.dataset,
            split_idx,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    elif args.mode.lower() == "test":
        inference(
            model, data, device, args.dataset, split_idx, batch_size=args.batch_size
        )
    else:
        raise ValueError(
            f"Mode {args.mode} not recognized. Choose from 'train' or 'test'."
        )


if __name__ == "__main__":
    main()
