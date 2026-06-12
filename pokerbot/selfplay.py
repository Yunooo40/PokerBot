"""Iterative self-play training.

Generation 0 is trained on hands between heuristic ThresholdPlayers. Each
later generation plays the previous generation's bot against itself (half
the games) and against the heuristic (the other half, to keep diversity),
then retrains all four per-round models on the fresh data. Folded hands end
early and are labelled by who took the pot, so later generations learn from
opponents who actually fold.

Example:
    python -m pokerbot.selfplay models/selfplay --iterations 3 --games 10000
"""

import argparse
import os
import random

from . import dataset
from .arena import play_match
from .model import save_model
from .simulator import NeuralPlayer, ThresholdPlayer, generate_training_data
from .train import evaluate, train


def selfplay(out_dir, iterations=3, games=10000, epochs=40, equity_samples=100,
             eval_hands=2000, seed=0, hidden_size=128, num_layers=4,
             dropout=0.5, lr=1e-4):
    os.makedirs(out_dir, exist_ok=True)
    prev_models = None

    for it in range(iterations):
        print("=== generation %d ===" % it)
        if prev_models is None:
            matchups = [[ThresholdPlayer(), ThresholdPlayer()]]
        else:
            matchups = [
                [NeuralPlayer(prev_models), NeuralPlayer(prev_models)],
                [NeuralPlayer(prev_models), ThresholdPlayer()],
            ]
        samples = []
        for m, players in enumerate(matchups):
            samples += generate_training_data(
                games // len(matchups), players=players, seed=seed * 1000 + it * 10 + m,
                print_every=0, equity_samples=equity_samples)
        print("generated %d samples from %d games" % (len(samples), games))

        models = {}
        for round_num in range(4):
            X, y = dataset.build_arrays(samples, round_num=round_num)
            X_train, y_train, X_test, y_test = dataset.train_test_split(
                X, y, seed=seed + it)
            model = train(X_train, y_train, X_test, y_test, epochs=epochs,
                          hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, lr=lr, seed=seed + it,
                          log_every=max(1, epochs))
            acc = evaluate(model, X_test, y_test)
            print("round %d: %d samples, test acc %.3f" % (round_num, len(X), acc))
            models[round_num] = model
            save_model(model, os.path.join(out_dir, "gen%d_round%d.pt" % (it, round_num)))

        rng = random.Random(seed + 100 + it)
        _, bb100_thr, err_thr = play_match(NeuralPlayer(models), ThresholdPlayer(),
                                           num_hands=eval_hands, rng=rng,
                                           equity_samples=equity_samples)
        print("gen %d vs threshold: %+.1f +/- %.1f bb/100" % (it, bb100_thr, err_thr))
        if prev_models is not None:
            _, bb100_prev, err_prev = play_match(
                NeuralPlayer(models), NeuralPlayer(prev_models),
                num_hands=eval_hands, rng=rng, equity_samples=equity_samples)
            print("gen %d vs gen %d: %+.1f +/- %.1f bb/100"
                  % (it, it - 1, bb100_prev, err_prev))
        prev_models = models

    # also save the last generation under the plain names arena expects
    for round_num, model in prev_models.items():
        save_model(model, os.path.join(out_dir, "round%d.pt" % round_num))
    print("final models saved to %s" % out_dir)
    return prev_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="directory for generation checkpoints")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games", type=int, default=10000, help="games per generation")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--equity-samples", type=int, default=100)
    parser.add_argument("--eval-hands", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    selfplay(args.out_dir, iterations=args.iterations, games=args.games,
             epochs=args.epochs, equity_samples=args.equity_samples,
             eval_hands=args.eval_hands, seed=args.seed)


if __name__ == "__main__":
    main()
