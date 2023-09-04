import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        # Print sparsity level
        x_nnz = torch.count_nonzero(x)
        adj_nnz = torch.count_nonzero(adjacency)

        x_sparsity = 1 - x_nnz.item() / (x.shape[0] * x.shape[1])
        adj_sparsity = 1 - adj_nnz.item() / (adjacency.shape[0] * adjacency.shape[1])

        print(f"\nx Sparsity: {x_sparsity * 100:.2f}%")
        print(f"Adjacency Sparsity: {adj_sparsity * 100:.2f}%")

        start_time = time.perf_counter()
        out = torch.matmul(adjacency, x)
        end_time = time.perf_counter()
        print(f"torch.matmul(adjacency, x) took {end_time - start_time} s")
        out = self.linear(out)
        return out


class CustomGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGCN, self).__init__()
        self.conv1 = GraphConvolution(in_channels, hidden_channels)
        self.conv2 = GraphConvolution(hidden_channels, out_channels)

    def forward(self, x, adjacency):
        start_time = time.perf_counter()
        x = self.conv1(x, adjacency)
        end_time = time.perf_counter()
        print(f"\nself.conv1(x, adjacency) took {end_time - start_time} s")

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        start_time = time.perf_counter()
        x = self.conv2(x, adjacency)
        end_time = time.perf_counter()
        print(f"\nself.conv2(x, adjacency) took {end_time - start_time} s")
        return x


# Define the dimensions
in_channels = 1433
hidden_channels = 128
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
dataset = datasets.Planetoid(root="data", name="Cora", transform=ToSparseTensor())

# Get the first data in the dataset
data = dataset[0]

# Get the node features and adjacency matrix
node_features = data.x
adjacency_matrix = data.adj_t.to_dense()  # Convert the sparse tensor to a dense tensor

# x = torch.tensor(node_features, dtype=torch.float)
# adjacency = torch.tensor(adjacency_matrix, dtype=torch.float)
x = node_features.clone().detach().to(torch.float)
adjacency = adjacency_matrix.clone().detach().to(torch.float)

# # Print sparsity level
# x_nnz = torch.count_nonzero(x)
# adj_nnz = torch.count_nonzero(adjacency)
#
# x_sparsity = 1 - x_nnz.item() / (x.shape[0] * x.shape[1])
# adj_sparsity = 1 - adj_nnz.item() / (adjacency.shape[0] * adjacency.shape[1])
#
# print(f"x Sparsity: {x_sparsity * 100:.2f}%")
# print(f"Adjacency Sparsity: {adj_sparsity * 100:.2f}%")

# Measure the inference time
start_time = time.perf_counter()

# Perform inference
with torch.no_grad():
    logits = model_custom(x, adjacency)
    pred = logits.argmax(dim=1)

inference_time = time.perf_counter() - start_time

# Calculate accuracy
correct = float((pred[data.test_mask] == data.y[data.test_mask]).sum().item())
accuracy = correct / data.test_mask.sum().item()

end_time = time.perf_counter()

# Calculate the inference time
inference_time = end_time - start_time

# Calculate accuracy
correct = float((pred[data.test_mask] == data.y[data.test_mask]).sum().item())
accuracy = correct / data.test_mask.sum().item()

print(f"\nInference time: {inference_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
