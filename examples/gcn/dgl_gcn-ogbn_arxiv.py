import os
import time
import argparse
import torch
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from ogb.nodeproppred import DglNodePropPredDataset


# Define GCN model using DGL
class DGLGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(DGLGCN, self).__init__()
        self.conv1 = GraphConv(
            num_features, 128, allow_zero_in_degree=True, norm="none"
        )
        self.conv2 = GraphConv(128, num_classes, allow_zero_in_degree=True, norm="none")

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)


# Load ogbn-arxiv dataset
def load_ogbn_arxiv():
    dataset = DglNodePropPredDataset(name="ogbn-arxiv")
    split_idx = dataset.get_idx_split()
    g, labels = dataset[0]
    g.ndata["label"] = labels
    g.ndata["test_mask"] = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    g.ndata["test_mask"][split_idx["test"]] = True
    return g, split_idx["test"]


# Perform inference
def inference(model, g, device):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        g = g.to(device)
        logits = model(g, g.ndata["feat"].float().to(device))
        pred = logits.argmax(dim=1)
    inference_time = time.time() - start_time
    test_mask = g.ndata["test_mask"].to(device)
    labels = g.ndata["label"].view(-1).to(device)
    correct = float((pred[test_mask] == labels[test_mask]).sum().item())
    accuracy = correct / test_mask.sum().item()
    print(f"DGL Inference time: {inference_time:.6f} seconds")
    print(f"DGL Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="DGL GCN Inference on Citation Network"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "citeseer", "pubmed", "ogbn-arxiv"],
        help="The dataset to use for running the model.",
    )
    args = parser.parse_args()
    device = torch.device("cpu")

    # Load dataset and convert it to DGL format if needed
    if args.dataset.lower() == "ogbn-arxiv":
        g, test_idx = load_ogbn_arxiv()
        g = g.to(device)
        num_features = g.ndata["feat"].shape[1]
        num_classes = int(g.ndata["label"].max() + 1)
    else:
        raise ValueError("This script now only supports ogbn-arxiv dataset")

    model_dgl = DGLGCN(num_features, num_classes).to(device)

    # Load and adjust weights for DGL model
    weights_file = f"weights/gcn_{args.dataset.lower()}_weights.pth"
    if os.path.isfile(weights_file):
        state_dict = torch.load(weights_file, map_location=device)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace(
                "lin.weight", "weight"
            )  # Adjust key for DGL model weight
            if "bias" in new_key:
                new_key = new_key.replace("lin.bias", "bias")
            # Transpose the weight matrices
            if "weight" in new_key:
                value = value.t()
            new_state_dict[new_key] = value
        model_dgl.load_state_dict(new_state_dict, strict=False)

        # Run inference
        inference(model_dgl, g, device)
    else:
        print(f"Weight file not found: {weights_file}")


if __name__ == "__main__":
    main()
