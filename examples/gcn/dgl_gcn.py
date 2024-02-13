import os
import time
import argparse
import torch
import torch.nn.functional as F
import dgl
from dgl import nn as dglnn
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.datasets import Planetoid


# Define GCN model using DGL
class DGLGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(DGLGCN, self).__init__()
        self.conv1 = dglnn.GraphConv(
            num_features, 128, norm="none", allow_zero_in_degree=True
        )
        self.conv2 = dglnn.GraphConv(
            128, num_classes, norm="none", allow_zero_in_degree=True
        )

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)


# Load dataset based on input name
def load_dataset(name):
    root_dir = os.path.join(os.getcwd(), "data")
    if name.lower() in ["citeseer", "cora", "pubmed"]:
        dataset = Planetoid(root=root_dir, name=name.capitalize())
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return dataset


# Utility function to convert PyG graph to DGL graph
def pyg_to_dgl(data):
    edge_index = data.edge_index.t().numpy()
    num_nodes = edge_index.max() + 1
    sparse_matrix = coo_matrix(
        (np.ones(edge_index.shape[0]), (edge_index[:, 0], edge_index[:, 1])),
        shape=(num_nodes, num_nodes),
    )
    g = dgl.from_scipy(sparse_matrix)
    g.ndata["feat"] = data.x
    return g


# Prepare and run inference
def inference(model, g, data):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        logits = model(g, g.ndata["feat"])
        pred = logits.argmax(dim=1)
    inference_time = time.time() - start_time
    correct = float((pred[data.test_mask] == data.y[data.test_mask]).sum().item())
    accuracy = correct / data.test_mask.sum().item()
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
        choices=["cora", "citeseer", "pubmed"],
        help="The dataset to use for running the model.",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset)
    data = dataset[0]

    g = pyg_to_dgl(data).to(torch.device("cpu"))
    model_dgl = DGLGCN(dataset.num_features, dataset.num_classes).to(
        torch.device("cpu")
    )

    # Load and adjust weights for DGL model
    weights_file = f"weights/gcn_{args.dataset.lower()}_weights.pth"
    if os.path.isfile(weights_file):
        state_dict = torch.load(weights_file, map_location="cpu")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace(".lin.", ".")
            new_value = value.t() if ".weight" in new_key else value
            new_state_dict[new_key] = new_value
        model_dgl.load_state_dict(new_state_dict)

        # Run inference
        inference(model_dgl, g, data)
    else:
        print(f"Weight file not found: {weights_file}")


if __name__ == "__main__":
    main()
