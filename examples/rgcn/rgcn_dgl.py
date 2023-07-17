import os
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv
from dgl.data import AIFBDataset


# Define R-GCN model using DGL
class DGLRGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations):
        super(DGLRGCN, self).__init__()
        self.conv1 = RelGraphConv(
            num_features, 16, num_relations, "basis", num_bases=30, activation=F.relu
        )
        self.conv2 = RelGraphConv(16, num_classes, num_relations, "basis", num_bases=30)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)


# Load dataset
dataset = AIFBDataset()
data = dataset[0]

num_nodes = dataset.num_nodes
num_relations = dataset.num_rels
num_classes = dataset.num_classes

# Convert to DGL graph
g = dgl.graph(data.graph)
g.ndata["feat"] = torch.FloatTensor(dataset.features)
g.edata["type"] = torch.LongTensor(dataset.edge_type)
g = g.to(torch.device("cpu"))

# Initialize DGL model
model_dgl = DGLRGCN(g.ndata["feat"].shape[1], num_classes, num_relations).to(
    torch.device("cpu")
)

# Load weights
state_dict = torch.load("weights/rgcn_aifb_weights.pth")
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
