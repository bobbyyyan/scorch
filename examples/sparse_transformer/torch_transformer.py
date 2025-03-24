import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TransformerModel(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, output_dim
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output = self.transformer_encoder(embedded)
        output = self.fc(output.mean(dim=1))
        return output


def collate_batch(batch):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(_label)
        text_list.append(torch.tensor(_text))
    return torch.tensor(label_list), torch.nn.utils.rnn.pad_sequence(
        text_list, batch_first=True, padding_value=0.0
    )


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (labels, text) in enumerate(train_loader):
        labels, text = labels.to(device), text.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(labels)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for labels, text in test_loader:
            labels, text = labels.to(device), text.to(device)
            output = model(text)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    end_time = time.time()
    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )
    print(f"Inference time: {end_time - start_time}s")


def main():
    parser = argparse.ArgumentParser(description="Transformer Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
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
        "--epochs", type=int, default=1, help="number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
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

    train_iter = IMDB(split="train")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        map(lambda x: tokenizer(x[1]), train_iter), specials=["<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x):
        return vocab(tokenizer(x))

    def label_pipeline(x):
        return 0 if x == "neg" else 1

    train_iter = IMDB(split="train")
    test_iter = IMDB(split="test")

    train_dataset = [
        (label_pipeline(label), text_pipeline(text)) for label, text in train_iter
    ]
    test_dataset = [
        (label_pipeline(label), text_pipeline(text)) for label, text in test_iter
    ]

    train_loader = DataLoader(train_dataset, collate_fn=collate_batch, **train_kwargs)
    test_loader = DataLoader(test_dataset, collate_fn=collate_batch, **test_kwargs)

    model = TransformerModel(len(vocab), 128, 2, 2, 256, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
        torch.save(model.state_dict(), "model.pt")
        print("Training complete, model saved.")

    elif args.mode == "test":
        model.load_state_dict(torch.load("model.pt", map_location=device))
        test(model, device, test_loader, criterion)


if __name__ == "__main__":
    main()
