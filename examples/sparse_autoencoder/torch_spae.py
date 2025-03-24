import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(784, 256)
        self.decoder = nn.Linear(256, 784)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_fn = nn.MSELoss()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    loss_fn = nn.MSELoss(reduction="sum")
    start_time = time.time()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784)
            data = data.to_sparse().to(device)
            output = model(data.to_dense())
            test_loss += loss_fn(output, data.to_dense())
    end_time = time.time()
    test_loss /= len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Time elapsed: {:.2f}s\n".format(
            test_loss, end_time - start_time
        )
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch Sparse Autoencoder Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="train the model or test the model",
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
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    model = SparseAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            if args.save_model:
                torch.save(model.state_dict(), "mnist_autoencoder.pt")

    elif args.mode == "test":
        model.load_state_dict(
            torch.load("mnist_autoencoder.pt", map_location=torch.device(device))
        )
        test(model, device, test_loader)


if __name__ == "__main__":
    main()
