import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.categorical import Categorical
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
Selects one expert for each input based on the gate outputs,
which are used as probabilities in a categorical distribution.
Only the selected expert's output is used, all other expert outputs are ignored.
This is a sparser approach.
"""


# Define the MoE model
class SparseMoE(nn.Module):
    """Sparse Mixture of Experts layer."""

    def __init__(self, in_features, out_features, num_experts):
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.gates = nn.Linear(in_features, num_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(num_experts)]
        )

    def forward(self, x):
        gate_values = F.softmax(self.gates(x), dim=1)
        selected_expert = Categorical(gate_values).sample()
        outputs = torch.zeros(x.shape[0], self.num_experts).to(x.device)
        for i, expert in enumerate(self.experts):
            outputs[:, i] = expert(x).squeeze()
        expanded_selection = F.one_hot(
            selected_expert, num_classes=self.num_experts
        ).float()
        return torch.sum(outputs * expanded_selection, dim=1)


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
