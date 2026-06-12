"""Monte-Carlo hand equity estimation.

Equity is the probability of winning at showdown against random opponent
holdings, with ties counted as a split. It is the single most informative
feature before the river, where exact hand strength is not yet defined.
"""

import random

from . import cards
from .evaluator import Evaluator

_evaluator = None


def _shared_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator


def estimate_equity(hand, board, num_opponents=1, num_samples=100, rng=None,
                    evaluator=None):
    """Estimate P(win at showdown) for ``hand`` given the visible ``board``.

    Deals random opponent hands and completes the board ``num_samples``
    times. Returns a float in [0, 1]; a k-way tie contributes 1/k.
    """
    rng = rng or random
    evaluator = evaluator or _shared_evaluator()
    seen = set(hand) | set(board)
    deck = [c for c in range(cards.NUM_CARDS) if c not in seen]
    num_board_missing = 5 - len(board)
    num_drawn = 2 * num_opponents + num_board_missing

    wins = 0.0
    for _ in range(num_samples):
        drawn = rng.sample(deck, num_drawn)
        full_board = board + drawn[2 * num_opponents:]
        my_score = evaluator.evaluate(hand, full_board)
        num_better = num_tied = 0
        for i in range(num_opponents):
            opp_score = evaluator.evaluate(drawn[2 * i: 2 * i + 2], full_board)
            if opp_score < my_score:
                num_better += 1
                break
            if opp_score == my_score:
                num_tied += 1
        if num_better == 0:
            wins += 1.0 / (1 + num_tied)
    return wins / num_samples
