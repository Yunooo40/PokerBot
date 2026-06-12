"""Head-to-head match runner for comparing players.

Seats are swapped every other hand and the dealer button rotates, so neither
player gets a positional edge. Results are reported in big blinds per 100
hands (bb/100), the standard poker win-rate unit.

CLI: compare two model directories (containing round0.pt ... round3.pt), or
a model directory against the built-in heuristic:

    python -m pokerbot.arena models/ threshold --hands 2000
"""

import argparse
import math
import os
import random

from .model import load_model
from .simulator import BIG_BLIND, NeuralPlayer, Table, ThresholdPlayer


def play_match(player_a, player_b, num_hands=2000, rng=None, equity_samples=100):
    """Play ``num_hands`` between two players.

    Returns (hands_won_a, bb100_a, stderr_bb100): the win rate comes with its
    standard error because pots run up to 200 big blinds in this engine, so
    bb/100 over a few thousand hands is noisy.
    """
    rng = rng or random.Random()
    # one table per seating order; player_a sits in seat 0, then seat 1
    tables = [
        Table([player_a, player_b], rng=rng, equity_samples=equity_samples),
        Table([player_b, player_a], rng=rng, equity_samples=equity_samples),
    ]
    a_seats = (0, 1)
    hands_won = 0
    outcomes = []
    for i in range(num_hands):
        table = tables[i % 2]
        seat = a_seats[i % 2]
        logs = table.play_game(dealer_index=(i // 2) % 2)
        winners = {s["playerIndex"] for s in logs if s["result"] == 1}
        bet = table.bets[seat]
        if seat in winners:
            hands_won += 1
            outcomes.append(table.pot / len(winners) - bet)
        else:
            outcomes.append(-bet)
    mean = sum(outcomes) / num_hands
    var = sum((o - mean) ** 2 for o in outcomes) / max(1, num_hands - 1)
    bb100 = 100.0 * mean / BIG_BLIND
    stderr = 100.0 * math.sqrt(var / num_hands) / BIG_BLIND
    return hands_won, bb100, stderr


def load_round_models(models_dir):
    """Load {round: model} from a directory of round{r}.pt files."""
    models = {}
    for r in range(4):
        path = os.path.join(models_dir, "round%d.pt" % r)
        if os.path.exists(path):
            models[r] = load_model(path)
    if not models:
        raise FileNotFoundError("no round*.pt models in %s" % models_dir)
    return models


def _make_player(spec):
    if spec == "threshold":
        return ThresholdPlayer()
    return NeuralPlayer(load_round_models(spec))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("player_a", help="model directory, or 'threshold'")
    parser.add_argument("player_b", help="model directory, or 'threshold'")
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--equity-samples", type=int, default=100)
    args = parser.parse_args()

    hands_won, bb100, stderr = play_match(
        _make_player(args.player_a), _make_player(args.player_b),
        num_hands=args.hands, rng=random.Random(args.seed),
        equity_samples=args.equity_samples)
    print("%s vs %s over %d hands: %d hands won (%.1f%%), %+.1f +/- %.1f bb/100"
          % (args.player_a, args.player_b, args.hands,
             hands_won, 100.0 * hands_won / args.hands, bb100, stderr))


if __name__ == "__main__":
    main()
