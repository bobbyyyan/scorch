import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGEConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSAGEConvolution, self).__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neighbor = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        out_self = self.linear_self(x)
        out_neighbor = self.linear_neighbor(torch.matmul(adjacency, x))
        out = F.relu(out_self + out_neighbor)
        return out


class CustomGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGraphSAGE, self).__init__()
        self.conv1 = GraphSAGEConvolution(in_channels, hidden_channels)
        self.conv2 = GraphSAGEConvolution(hidden_channels, out_channels)

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = self.conv2(x, adjacency)
        return x


# Define the dimensions
in_channels = 50  # PPI has 50 features per node
hidden_channels = 16
out_channels = 121  # PPI has 121 classes

# Initialize the custom PyTorch GraphSAGE model
model_custom = CustomGraphSAGE(in_channels, hidden_channels, out_channels)


# Load the pre-trained weights
state_dict = torch.load("weights/graphsage_ppi_weights.pth")

# Modify the keys to match the custom GraphSAGE model
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace(".lin.", ".linear.")
    if ".bias" in new_key:
        new_key = new_key.replace(".bias", ".linear.bias")
    new_state_dict[new_key] = value

# Load the modified state_dict into the custom GraphSAGE model
model_custom.load_state_dict(new_state_dict)

# Set the model to evaluation mode
model_custom.eval()

# Prepare the input data (e.g., node features and adjacency matrix)
import torch_geometric.datasets as datasets
from torch_geometric.transforms import ToDense, ToSparseTensor

# Load the PPI dataset
dataset = datasets.PPI(root="data/PPI", transform=ToDense())

# Get the first graph in the dataset
graph = dataset[0]

# Get the node features and adjacency matrix
node_features = graph.x
adjacency_matrix = (
    graph.adj
)  # For GraphSAGE, we don't need to convert it to dense matrix

x = node_features.clone().detach().to(torch.float)
adjacency = adjacency_matrix.clone().detach().to(torch.float)


# Measure the inference time
start_time = time.perf_counter()

# Perform inference
with torch.no_grad():
    output = model_custom(x, adjacency)

end_time = time.perf_counter()

# Calculate the inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.6f} seconds")
