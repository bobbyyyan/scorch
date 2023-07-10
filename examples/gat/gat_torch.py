import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid


# GAT layer
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.attn_fc = nn.Linear(2 * out_features, 1)

    def forward(self, x, edge_index):
        # Attention
        alpha = self.attn_fc(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.softmax(alpha, edge_index[0])

        # Aggregation
        out = self.linear(x)
        out = torch.matmul(alpha, out[edge_index[1]])

        return out


# GAT network
class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATLayer(in_channels, hidden_channels)
        self.conv2 = GATLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Load dataset
dataset = Planetoid(root="data/Cora", name="Cora")

# Load pre-trained weights
state_dict = torch.load("weights/gat_cora_weights.pth")

# Modify the keys to match the custom GAT model
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace(".lin.", ".linear.")
    if ".bias" in new_key:
        new_key = new_key.replace(".bias", ".linear.bias")
    new_state_dict[new_key] = value

# Create model and load weights
model = GATNet(dataset.num_features, 16, dataset.num_classes)
# Load the modified state_dict into the custom GAT model
model.load_state_dict(new_state_dict)
model.eval()

# Perform inference
data = dataset[0]

start = time.time()
with torch.no_grad():
    out = model(data.x, data.edge_index)
end = time.time()

print(f"Inference time: {end - start:.4f} seconds")
