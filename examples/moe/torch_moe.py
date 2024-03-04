import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import os


class SparseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index].toarray().squeeze(0)
        y = self.y[index]
        return torch.FloatTensor(x), torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]


class Expert(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_size, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, expert_output_size, num_experts, num_classes):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_size, expert_output_size) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_size, num_experts)
        self.classifier = nn.Linear(expert_output_size * num_experts, num_classes)

    def forward(self, x):
        gating_weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weighted_expert_outputs = expert_outputs * gating_weights.unsqueeze(-1)
        flattened_outputs = weighted_expert_outputs.view(weighted_expert_outputs.size(0), -1)
        return self.classifier(flattened_outputs)


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Training Loss: {avg_loss:.4f}')
    return avg_loss


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy


def preprocess_data():
    # Load the dataset
    newsgroups = fetch_20newsgroups(subset='all')
    data = newsgroups.data
    target = newsgroups.target

    # Convert the text data into TF-IDF feature vectors
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    x_vectors = vectorizer.fit_transform(data)

    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_vectors, target, test_size=0.2, random_state=42)

    # Label encoding
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    return SparseDataset(x_train, y_train), SparseDataset(x_test, y_test), vectorizer, encoder


def train(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_epoch(model, train_loader, criterion, optimizer)
    # Save the model
    torch.save(model.state_dict(), 'moe_model.pth')
    print('Training complete. Model saved.')


def test(model, test_loader):
    # Load the model
    model.load_state_dict(torch.load('moe_model.pth'))
    print('Model loaded for evaluation.')
    evaluate(model, test_loader)


def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate Mixture of Experts Model")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode to run: train or test the model.")
    args = parser.parse_args()

    train_dataset, test_dataset, vectorizer, encoder = preprocess_data()

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # MoE Parameters
    input_size = 2000  # Number of features from TF-IDF
    expert_output_size = 128  # Size of each expert's output vector
    num_experts = 10  # Number of experts
    num_classes = 20  # Number of classes in 20newsgroups

    # Model
    model = MixtureOfExperts(input_size, expert_output_size, num_experts, num_classes)

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.mode == 'train':
        train(model, train_loader, criterion, optimizer)
    elif args.mode == 'test':
        test(model, test_loader)


if __name__ == "__main__":
    main()
