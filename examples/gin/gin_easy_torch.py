import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid


# Define GINConv layer
class GINConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GINConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, adjacency):
        out = (1 + self.eps) * x + torch.matmul(adjacency, x)
        out = self.linear(out)
        return out


class CustomGIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGIN, self).__init__()
        self.conv1 = GINConv(in_channels, hidden_channels)
        self.conv2 = GINConv(hidden_channels, out_channels)

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = self.conv2(x, adjacency)
        return F.log_softmax(x, dim=1)


# Define the dimensions
in_channels = 500
hidden_channels = 64
out_channels = 3

# Initialize the custom PyTorch GIN model
model_custom = CustomGIN(in_channels, hidden_channels, out_channels)

# Load the pre-trained weights
state_dict = torch.load("weights/gin_pubmed_weights.pth")

# Modify the keys to match the custom GIN model
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace(".lin.", ".linear.")
    if ".bias" in new_key:
        new_key = new_key.replace(".bias", ".linear.bias")
    new_state_dict[new_key] = value

# Load the modified state_dict into the custom GIN model
model_custom.load_state_dict(new_state_dict)

# Set the model to evaluation mode
model_custom.eval()

# Load the PubMed dataset
dataset = Planetoid(root="data/PubMed", name="PubMed")
graph = dataset[0]

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
