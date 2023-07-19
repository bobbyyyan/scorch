import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
data = fetch_20newsgroups(subset="train")
vectorizer = CountVectorizer(max_features=5000)
X = torch.FloatTensor(vectorizer.fit_transform(data.data).toarray())
Y = torch.LongTensor(data.target)

"""
Computes the output of all experts for each input.
Then take a weighted sum of these outputs, where the weights are given
by the gate outputs (which sum to 1 for each input due to the softmax operation).
This effectively means that each input is processed by all experts,
but the contribution of each expert to the final output
is determined by the gate outputs.
This approach is more akin to the original Mixture of Experts model,
where every expert contributes to the final decision with different weights.
"""


# Define the MoE model
class SparseMoE(nn.Module):
    def __init__(self, in_features, num_experts, out_features):
        super(SparseMoE, self).__init__()
        self.gates = nn.Linear(in_features, num_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(num_experts)]
        )

    def forward(self, x):
        gate_outputs = F.softmax(self.gates(x), dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        outputs = (gate_outputs.unsqueeze(2) * expert_outputs).sum(dim=2)
        return outputs


# Load the 20 newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset="train")

# Convert text data to vectors
vectorizer = TfidfVectorizer(
    max_features=5000
)  # Limit to 5000 most frequent words for simplicity
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors = vectors.toarray()

# Convert labels to tensor
labels = torch.tensor(newsgroups_train.target)

# Define the dimensions
in_features = vectors.shape[1]
out_features = 20  # There are 20 classes in the 20 newsgroups dataset
num_experts = 10

# Initialize the model and optimizer
model = SparseMoE(in_features, out_features, num_experts)
# model = SparseMoE(X.shape[1], 10, len(set(Y)))
optimizer = Adam(model.parameters())

# Training loop
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    pred = model(X)
    loss = F.cross_entropy(pred, Y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the weights
torch.save(model.state_dict(), "moe_weights.pth")

# Load the pre-trained weights
state_dict = torch.load("weights/moe_newsgroups_weights.pth")
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Prepare the input data
x = torch.tensor(vectors, dtype=torch.float)

# Measure the inference time
start_time = time.perf_counter()

# Perform inference
with torch.no_grad():
    output = model(x)

# Calculate accuracy
pred = output.argmax(dim=1)
accuracy = accuracy_score(labels, pred)

end_time = time.perf_counter()

# Calculate the inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.6f} seconds")
print(f"Accuracy: {accuracy:.2f}")
