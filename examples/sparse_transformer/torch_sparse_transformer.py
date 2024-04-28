import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB, AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class BigBirdSparseAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        block_size,
        num_random_blocks,
        num_sliding_blocks,
        inference=False,
    ):
        super(BigBirdSparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.num_sliding_blocks = num_sliding_blocks
        self.inference = inference
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

    def forward(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.size()
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = query.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (
            self.head_dim**0.5 + 1e-6
        )

        # Compute sparse indices for the attention mask
        indices = []
        for b in range(batch_size):
            for h in range(self.num_heads):
                for i in range(seq_length):
                    for j in range(
                        max(0, i - self.num_sliding_blocks),
                        min(seq_length, i + self.num_sliding_blocks + 1),
                    ):
                        if i % self.block_size == 0 and j % self.block_size == 0:
                            indices.append((b, h, i, j))

        # Convert indices to tensor
        indices = torch.tensor(indices, dtype=torch.long).t()

        # Extract the values at these indices from attention_scores
        values = attention_scores.view(-1)[
            indices[0]
            * attention_scores.size(1)
            * attention_scores.size(2)
            * attention_scores.size(3)
            + indices[1] * attention_scores.size(2) * attention_scores.size(3)
            + indices[2] * attention_scores.size(3)
            + indices[3]
        ]

        sparse_attention_scores = torch.sparse_coo_tensor(
            indices, values, attention_scores.size()
        )

        # Apply softmax to sparse attention scores
        attention_probs = torch.sparse.softmax(sparse_attention_scores, dim=-1)
        if self.inference:
            batch_size, num_heads, seq_len, _ = attention_probs.shape
            hidden_size = value.shape[-1]

            context = torch.zeros(
                batch_size, num_heads, seq_len, hidden_size, device=value.device
            )

            for b in range(batch_size):
                for h in range(num_heads):
                    attention_probs_bh = attention_probs[
                        b, h
                    ]  # (seq_len, seq_len) sparse tensor
                    value_bh = value[b, h]  # (seq_len, hidden_size) dense tensor
                    context[b, h] = torch.sparse.mm(attention_probs_bh, value_bh)
        else:
            context = torch.einsum("bhij,bhjd->bhid", attention_probs.to_dense(), value)
        context = context.contiguous().view(batch_size, seq_length, self.embed_dim)

        # Reshape back to the original context size
        context = context.transpose(1, 2).reshape(
            batch_size, seq_length, self.embed_dim
        )

        return context


class BigBirdBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        block_size,
        num_random_blocks,
        num_sliding_blocks,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        inference=False,
    ):
        super(BigBirdBlock, self).__init__()
        self.attention = BigBirdSparseAttention(
            embed_dim,
            num_heads,
            block_size,
            num_random_blocks,
            num_sliding_blocks,
            inference,
        )
        self.intermediate = nn.Linear(embed_dim, intermediate_size)
        self.output = nn.Linear(intermediate_size, embed_dim)
        nn.init.xavier_uniform_(self.intermediate.weight)
        nn.init.xavier_uniform_(self.output.weight)
        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.layer_norm_1(attention_output + hidden_states)
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        output = self.output(intermediate_output)
        output = self.hidden_dropout(output)
        hidden_states = self.layer_norm_2(output + hidden_states)
        return hidden_states


class BigBirdModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        block_size,
        num_random_blocks,
        num_sliding_blocks,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        num_classes,
        inference=False,
    ):
        super(BigBirdModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                BigBirdBlock(
                    embed_dim,
                    num_heads,
                    block_size,
                    num_random_blocks,
                    num_sliding_blocks,
                    intermediate_size,
                    hidden_dropout_prob,
                    attention_probs_dropout_prob,
                    inference,
                )
                for _ in range(num_layers)
            ]
        )
        self.pooler = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        pooled_output = hidden_states[:, 0]
        logits = self.pooler(pooled_output)
        return logits


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
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    print(f"Inference time: {end_time - start_time:.2f}s")


def YAHOO_ANSWERS(split=("train", "test")):
    train_df = pd.read_csv("data/yahoo_answers_csv/train.csv", header=None)
    test_df = pd.read_csv("data/yahoo_answers_csv/test.csv", header=None)

    # labels in first column, text in second, third, and fourth columns
    train_iter = [(row[0], f"{row[1]} {row[2]} {row[3]}") for row in train_df.itertuples(index=False, name=None)]
    test_iter = [(row[0], f"{row[1]} {row[2]} {row[3]}") for row in test_df.itertuples(index=False, name=None)]

    # train_iter = [(row[0], row[1]) for row in train_df.itertuples(index=False, name=None)]
    # test_iter = [(row[0], row[1]) for row in test_df.itertuples(index=False, name=None)]

    return train_iter, test_iter

def main():
    parser = argparse.ArgumentParser(
        description="Big Bird Sparse Transformer Benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imdb",
        choices=["imdb", "ag_news", "yahoo_answers"],
        help="dataset to use for training and testing (default: imdb)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help="train the model or test the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to save/load the model",
    )
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"model/bigbird_model_sparse_{args.dataset}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the dataset
    if args.dataset == "imdb":
        train_iter, test_iter = IMDB(split=("train", "test"))
        num_classes = 2
    elif args.dataset == "ag_news":
        train_iter, test_iter = AG_NEWS(split=("train", "test"))
        num_classes = 4
    elif args.dataset == "yahoo_answers":
        train_iter, test_iter = YAHOO_ANSWERS()
        num_classes = 10


    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x):
        return vocab(tokenizer(x))

    def label_pipeline(x):
        if args.dataset == "ag_news":
            return int(x) - 1
        elif args.dataset == "imdb":
            return 1 if x == "pos" else 0
        elif args.dataset == "yahoo_answers":
            return int(x) - 1


    train_dataset = list(train_iter)
    test_dataset = list(test_iter)

    train_dataset = [
        (label_pipeline(label), text_pipeline(text)) for label, text in train_dataset
    ]
    test_dataset = [
        (label_pipeline(label), text_pipeline(text)) for label, text in test_dataset
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=collate_batch
    )

    is_inference = args.mode == "test"

    model = BigBirdModel(
        vocab_size=len(vocab),
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        block_size=16,
        num_random_blocks=2,
        num_sliding_blocks=2,
        intermediate_size=256,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_classes=num_classes,
        inference=is_inference,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if is_inference:
        model.load_state_dict(torch.load(args.model_path))
        test(model, device, test_loader, criterion)
    else:
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            test(model, device, test_loader, criterion)
            torch.save(model.state_dict(), args.model_path + f".{epoch}_epochs")


if __name__ == "__main__":
    main()
