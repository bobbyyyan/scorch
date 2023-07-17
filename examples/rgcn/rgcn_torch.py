import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.transforms import ToSparseTensor


class RelationalGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RelationalGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations

        self.weight = nn.Parameter(
            torch.Tensor(num_relations, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.Tensor(num_relations, out_features))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.bias)

    def forward(self, x, adjacency):
        out = torch.stack(
            [
                torch.spmm(adjacency[i], x @ self.weight[i])
                for i in range(self.num_relations)
            ],
            dim=0,
        )
        out = out.sum(dim=0) + self.bias.mean(dim=0)
        return out


class CustomRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(CustomRGCN, self).__init__()
        self.conv1 = RelationalGraphConvolution(
            in_channels, hidden_channels, num_relations
        )
        self.conv2 = RelationalGraphConvolution(
            hidden_channels, out_channels, num_relations
        )

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = self.conv2(x, adjacency)
        return x


# Define the dimensions
in_channels = 288
hidden_channels = 16
out_channels = 4
num_relations = 91

# Initialize the custom PyTorch R-GCN model
model_custom = CustomRGCN(in_channels, hidden_channels, out_channels, num_relations)

# Load the pre-trained weights
state_dict = torch.load("weights/rgcn_aifb_weights.pth")
model_custom.load_state_dict(state_dict)

# Set the model to evaluation mode
model_custom.eval()

# Load the AIFB dataset
dataset = Entities(root="data/AIFB", name="AIFB", transform=ToSparseTensor())

# Get the first graph in the dataset
graph = dataset[0]

# Get the node features and adjacency matrix
node_features = graph.x
adjacency_matrices = (
    graph.adj_t
)  # This is a list of adjacency matrices, one for each relation

x = node_features.clone().detach().to(torch.float)
adjacency = [
    adj.to_dense() for adj in adjacency_matrices
]  # Convert the sparse tensor to a dense tensor

# Measure the inference time
start_time = time.perf_counter()

# Perform inference
with torch.no_grad():
    output = model_custom(x, adjacency)

end_time = time.perf_counter()

# Calculate the inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.6f} seconds")
