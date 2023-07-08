import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Load dataset
dataset = Planetoid(root=os.path.join(os.getcwd(), "data"), name="Cora")
data = dataset[0]

# Initialize model and optimizer
device = torch.device("cpu")
model = GCN(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Train the model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()

# Save weights
torch.save(model.state_dict(), "weights/gcn_cora_weights.pth")

# Load weights and prepare for inference
model.load_state_dict(torch.load("weights/gcn_cora_weights.pth"))
model.eval()

# Perform inference and measure time
start_time = time.time()

with torch.no_grad():
    logits = model(data.x.to(device), data.edge_index.to(device))
    pred = logits.argmax(dim=1)

inference_time = time.time() - start_time

# Calculate accuracy
correct = float((pred[data.test_mask] == data.y[data.test_mask]).sum().item())
accuracy = correct / data.test_mask.sum().item()

print(f"Inference time: {inference_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
