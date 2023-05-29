import os
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

# Define GCN model using DGL
class DGLGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(DGLGCN, self).__init__()
        self.conv1 = GraphConv(num_features, 16)
        self.conv2 = GraphConv(16, num_classes)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)

# Load dataset
dataset = Planetoid(root=os.path.join(os.getcwd(), 'data'), name='Cora')
data = dataset[0]

# Convert PyG graph to DGL graph
g = dgl.graph(data.edge_index)
g.ndata['feat'] = data.x
g = g.to(torch.device('cpu'))

# Initialize DGL model
model_dgl = DGLGCN(dataset.num_features, dataset.num_classes).to(torch.device('cpu'))

# Load weights from PyG model
model_dgl.load_state_dict(torch.load('gcn_cora_weights.pth'))

# Prepare for inference
model_dgl.eval()

# Perform inference and measure time
start_time = time.time()

with torch.no_grad():
    logits_dgl = model_dgl(g, g.ndata['feat'])
    pred_dgl = logits_dgl.argmax(dim=1)

inference_time_dgl = time.time() - start_time

# Calculate accuracy
correct_dgl = float((pred_dgl[data.test_mask] == data.y[data.test_mask]).sum().item())
accuracy_dgl = correct_dgl / data.test_mask.sum().item()

print(f"DGL Inference time: {inference_time_dgl:.6f} seconds")
print(f"DGL Accuracy: {accuracy_dgl:.4f}")
