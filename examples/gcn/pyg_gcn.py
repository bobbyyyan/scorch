import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn import GCNConv
import argparse
from tqdm import tqdm


# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, num_classes)

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


def train(model, data, device, dataset_name):
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train the model
    model.train()
    for epoch in tqdm(range(200), desc="Training", unit="epoch"):
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()

    # Save weights
    torch.save(model.state_dict(), f"weights/gcn_{dataset_name.lower()}_weights.pth")


def inference(model, data, device, dataset_name):
    # Load weights and prepare for inference
    model.load_state_dict(torch.load(f"weights/gcn_{dataset_name.lower()}_weights.pth"))
    model.eval()

    # Perform inference and measure time
    start_time = time.time()

    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        pred = logits.argmax(dim=1)

    inference_time = time.time() - start_time

    # Calculate accuracy
    correct = float((pred[data.test_mask] == data.y[data.test_mask]).sum().item())
    accuracy = correct / data.test_mask.sum().item()

    print(f"\nInference time: {inference_time:.6f} seconds")
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
        help='Dataset to use. Options are "cora", "pubmed", "citeseer", or "reddit".',
    )
    args = parser.parse_args()

    # Convert dataset to lowercase
    args.dataset = args.dataset.lower()

    # Load dataset
    if args.dataset in ["cora", "pubmed", "citeseer"]:
        dataset = Planetoid(
            root=os.path.join(os.getcwd(), "data"),
            name=args.dataset,
        )
        data = dataset[0]
    elif args.dataset == "reddit":
        dataset = Reddit(root=os.path.join(os.getcwd(), "data/reddit"))
        data = dataset[0]
    else:
        raise ValueError(
            f"Dataset {args.dataset} not recognized. Choose from 'cora', 'pubmed', 'citeseer', or 'reddit'."
        )

    # Initialize model
    device = torch.device("cpu")
    model = GCN(dataset.num_features, dataset.num_classes).to(device)

    if args.mode.lower() == "train":
        train(model, data, device, args.dataset)
    elif args.mode.lower() == "test":
        inference(model, data, device, args.dataset)
    else:
        raise ValueError(
            f"Mode {args.mode} not recognized. Choose from 'train' or 'test'."
        )


if __name__ == "__main__":
    main()
