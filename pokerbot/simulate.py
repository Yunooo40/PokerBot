"""Generate training data by simulating poker hands.

Example:
    python -m pokerbot.simulate data/games.json --games 100000
"""

import argparse

from .simulator import generate_training_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="output JSON file")
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--equity-samples", type=int, default=100,
                        help="Monte-Carlo rollouts per equity estimate (0 = off)")
    args = parser.parse_args()
    generate_training_data(args.games, args.out, num_players=args.players,
                           seed=args.seed, equity_samples=args.equity_samples)


if __name__ == "__main__":
    main()
