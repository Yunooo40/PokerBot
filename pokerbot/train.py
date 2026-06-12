"""Train a per-round win predictor.

Example:
    python -m pokerbot.train data/games.json --round 3 --epochs 100 \\
        --out models/river.pt
"""

import argparse

import numpy as np
import torch
from torch import nn

from . import dataset
from .model import WinPredictor, save_model


def train(X_train, y_train, X_test, y_test, epochs=50, hidden_size=128,
          num_layers=4, dropout=0.5, lr=1e-4, batch_size=128, seed=0,
          log_every=10):
    torch.manual_seed(seed)
    model = WinPredictor(X_train.shape[1], hidden_size, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        if epoch % log_every == 0 or epoch == epochs - 1:
            train_acc = evaluate(model, X_train, y_train)
            test_acc = evaluate(model, X_test, y_test)
            print("epoch %3d  loss %.4f  train acc %.3f  test acc %.3f"
                  % (epoch, total_loss / len(X_train), train_acc, test_acc))
    return model


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(torch.from_numpy(X))
    preds = (torch.sigmoid(logits) > 0.5).float().numpy()
    return float(np.mean(preds == y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="JSON training data from pokerbot.simulate")
    parser.add_argument("--round", type=int, default=3, choices=range(4),
                        help="betting round to train on (0=preflop ... 3=river)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="max samples (0 = all)")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="path to save the trained model (.pt)")
    args = parser.parse_args()

    samples = dataset.load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]
    X, y = dataset.build_arrays(samples, round_num=args.round)
    X_train, y_train, X_test, y_test = dataset.train_test_split(
        X, y, test_frac=args.test_frac, seed=args.seed)
    print("round %d: %d train / %d test samples, %d features"
          % (args.round, len(X_train), len(X_test), X.shape[1]))

    model = train(X_train, y_train, X_test, y_test, epochs=args.epochs,
                  hidden_size=args.hidden_size, num_layers=args.layers,
                  dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
                  seed=args.seed)

    print("final test accuracy: %.3f" % evaluate(model, X_test, y_test))
    if args.out:
        save_model(model, args.out)
        print("saved model to %s" % args.out)


if __name__ == "__main__":
    main()
