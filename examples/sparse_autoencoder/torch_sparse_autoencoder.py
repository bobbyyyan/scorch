import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import warnings

# Suppress specific PyTorch UserWarning about Sparse CSR tensor support
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Sparse CSR tensor support is in beta state.*",
)


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain("relu"))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.sparse.mm(input, self.weight.t()) + self.bias


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim=256):
        super(SparseAutoencoder, self).__init__()
        self.input_size = input_size
        self.encoder = SparseLinear(input_size, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_size)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x


def get_dataset(name):
    if name == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
        )
        train_dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST("./data", train=False, transform=transform)
        input_size = 28 * 28
    elif name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform)
        input_size = 32 * 32
    elif name == "cifar100":
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        )
        train_dataset = datasets.CIFAR100(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100("./data", train=False, transform=transform)
        input_size = 32 * 32
    elif name == "celeba":
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        )
        train_dataset = datasets.CelebA(
            "./data", split="train", download=True, transform=transform
        )
        test_dataset = datasets.CelebA(
            "./data", split="test", download=True, transform=transform
        )
        input_size = 64 * 64
    else:
        raise ValueError("Unsupported dataset")

    return train_dataset, test_dataset, input_size


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_fn = nn.MSELoss()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, model.input_size).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    loss_fn = nn.MSELoss(reduction="sum")
    start_time = time.time()

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, model.input_size).to(device)
            sparse_data = data.to_sparse_csr()
            output = model(sparse_data)
            test_loss += loss_fn(output, data)

    end_time = time.time()
    test_loss /= len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}")
    print(f"Inference time: {end_time - start_time}s")


def main():
    parser = argparse.ArgumentParser(description="Sparse Autoencoder Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help="train the model or test the model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "cifar100", "celeba"],
        help="dataset for training/testing the model (default: 'mnist')",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()

    dataset_name = args.dataset

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset, test_dataset, input_size = get_dataset(args.dataset)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    model = SparseAutoencoder(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
        torch.save(model.state_dict(), f"models/{dataset_name}_sparse_autoencoder.pt")
        print("Training complete, model saved.")

    elif args.mode == "test":
        model.load_state_dict(
            torch.load(
                f"models/{dataset_name}_sparse_autoencoder.pt", map_location=device
            )
        )
        test(model, device, test_loader)


if __name__ == "__main__":
    main()
