import os
import time
import torch
import torch.nn.functional as F
import dgl
from dgl import nn as dglnn
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.datasets import Planetoid
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

# Define GAT model using DGL
class DGLGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(DGLGAT, self).__init__()
        self.conv1 = dglnn.GATConv(num_features, 16, num_heads=8, dropout=0.6)
        self.conv2 = dglnn.GATConv(8 * 16, num_classes, num_heads=1, dropout=0.6)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)


# Load dataset
dataset = Planetoid(root=os.path.join(os.getcwd(), "data"), name="Cora")
data = dataset[0]

# Convert PyG graph to DGL graph
edge_index = data.edge_index.t().numpy()
num_nodes = edge_index.max() + 1
sparse_matrix = coo_matrix(
    (np.ones(edge_index.shape[0]), (edge_index[:, 0], edge_index[:, 1])),
    shape=(num_nodes, num_nodes),
)
g = dgl.from_scipy(sparse_matrix)
g.ndata["feat"] = data.x
g = g.to(torch.device("cpu"))

# Initialize DGL model
model_dgl = DGLGAT(dataset.num_features, dataset.num_classes).to(torch.device("cpu"))

# Convert weights named "conv1.weight" and "conv2.weight" to DGL format
state_dict = torch.load("weights/gat_cora_weights.pth")
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace(".lin.", ".")
    new_value = value
    if ".weight" in new_key:
        new_value = value.t()  # Transpose the weight dimensions
    new_state_dict[new_key] = new_value

# Load the modified state_dict into the model
model_dgl.load_state_dict(new_state_dict)

# Prepare for inference
model_dgl.eval()

# Perform inference and measure time
start_time = time.time()

with torch.no_grad():
    logits_dgl = model_dgl(g, g.ndata["feat"])
    pred_dgl = logits_dgl.argmax(dim=1)

inference_time_dgl = time.time() - start_time

# Calculate accuracy
correct_dgl = float((pred_dgl[data.test_mask] == data.y[data.test_mask]).sum().item())
accuracy_dgl = correct_dgl / data.test_mask.sum().item()

print(f"DGL Inference time: {inference_time_dgl:.6f} seconds")
print(f"DGL Accuracy: {accuracy_dgl:.4f}")
