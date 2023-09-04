import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import argparse


# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, data, device, dataset_name):
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train the model
    model.train()
    for epoch in range(200):
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
        help='Dataset to use. Options are "Cora", "Pubmed", or "CiteSeer".',
    )
    args = parser.parse_args()

    # Set dataset name to lowercase
    args.dataset = args.dataset.lower()

    # Load dataset
    dataset = Planetoid(root=os.path.join(os.getcwd(), "data"), name=args.dataset)
    data = dataset[0]

    # Initialize model
    device = torch.device("cpu")
    model = GCN(dataset.num_features, dataset.num_classes).to(device)

    if args.mode == "train":
        train(model, data, device, args.dataset)
        inference(model, data, device, args.dataset)
    elif args.mode == "test":
        inference(model, data, device, args.dataset)
    else:
        raise ValueError(
            f"Mode {args.mode} not recognized. Choose from 'train' or 'test'."
        )


if __name__ == "__main__":
    main()
