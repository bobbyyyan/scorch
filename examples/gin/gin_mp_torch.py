import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing

# Define GINConv layer
class GINConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GINConv, self).__init__(aggr="add")  # "Add" aggregation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        edge_index, _ = self.add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i):
        return self.mlp((1 + self.eps) * x_i + x_j)


# Define GIN model
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GINConv(hidden_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        return F.log_softmax(x, dim=1)


# Define the dimensions
in_channels = 500
hidden_channels = 64
out_channels = 3

# Initialize the custom PyTorch GIN model
model_custom = GIN(in_channels, hidden_channels, out_channels)

# Load the PubMed dataset
dataset = Planetoid(root="data/PubMed", name="PubMed")
graph = dataset[0]

# Load the saved weights and prepare for inference
model_custom.load_state_dict(torch.load("weights/gin_pubmed_weights.pth"))
model_custom.eval()

# Get the node features and adjacency matrix
node_features = graph.x
adjacency_matrix = graph.edge_index

# Measure the inference time
start_time = time.perf_counter()

# Perform inference
with torch.no_grad():
    output = model_custom(node_features, adjacency_matrix)

end_time = time.perf_counter()

# Calculate the inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.6f} seconds")
