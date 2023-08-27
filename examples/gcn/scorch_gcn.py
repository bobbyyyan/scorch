import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import scorch


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        # out = torch.matmul(adjacency, x)
        start_time = time.perf_counter()
        out = scorch.matmul(adjacency, x)
        end_time = time.perf_counter()
        print(f"scorch.matmul(adjacency, x) took {end_time - start_time} s")
        out = out.to_torch()
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


# Define the dimensions
in_channels = 1433
hidden_channels = 16
out_channels = 7

# Initialize the custom PyTorch GCN model
model_custom = CustomGCN(in_channels, hidden_channels, out_channels)

# Load the pre-trained weights
state_dict = torch.load("weights/gcn_cora_weights.pth")

# Modify the keys to match the custom GCN model
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace(".lin.", ".linear.")
    if ".bias" in new_key:
        new_key = new_key.replace(".bias", ".linear.bias")
    new_state_dict[new_key] = value

# Load the modified state_dict into the custom GCN model
model_custom.load_state_dict(new_state_dict)
# model_custom.load_state_dict(torch.load("weights/gcn_cora_weights.pth"))

# Set the model to evaluation mode
model_custom.eval()

# Prepare the input data (e.g., node features and adjacency matrix)
import torch_geometric.datasets as datasets
from torch_geometric.transforms import ToSparseTensor

# Load the Cora dataset
dataset = datasets.Planetoid(root="data/Cora", name="Cora", transform=ToSparseTensor())

# Get the first graph in the dataset
graph = dataset[0]

# Get the node features and adjacency matrix
node_features = graph.x
adjacency_matrix = graph.adj_t.to_dense()  # Convert the sparse tensor to a dense tensor

# x = torch.tensor(node_features, dtype=torch.float)
# adjacency = torch.tensor(adjacency_matrix, dtype=torch.float)
x = node_features.clone().detach().to(torch.float)
x = scorch.Tensor.from_torch(x, "x").to_sparse("ds")
adjacency = adjacency_matrix.clone().detach().to(torch.float)
adjacency = scorch.Tensor.from_torch(adjacency, "A").to_sparse("ds")

# Measure the inference time
start_time = time.perf_counter()

# Perform inference
with torch.no_grad():
    output = model_custom(x, adjacency)

end_time = time.perf_counter()

# Calculate the inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.6f} seconds")
