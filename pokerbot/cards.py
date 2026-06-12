"""Card utilities.

Cards are plain integers in [0, 51]:
    rank = card % 13   (0 = deuce ... 12 = ace)
    suit = card // 13  (0 = spades, 1 = hearts, 2 = diamonds, 3 = clubs)

This matches the encoding of the historical JSON training data
(``handCards`` / ``tableCards`` fields produced by legacy/tableQ.py).
"""

import random

NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4

STR_RANKS = "23456789TJQKA"
STR_SUITS = "shdc"

# Prime per rank, used by the evaluator's perfect-hash lookup tables.
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

# deuces encodes suits as bit flags: spades=1, hearts=2, diamonds=4, clubs=8
_SUIT_BITS = (1, 2, 4, 8)


def rank(card):
    return card % NUM_RANKS


def suit(card):
    return card // NUM_RANKS


def name(card):
    """Human-readable name, e.g. 0 -> '2s', 51 -> 'Ac'."""
    return STR_RANKS[rank(card)] + STR_SUITS[suit(card)]


def to_eval_int(card):
    """Convert a 0-51 card to the deuces bit format used by the evaluator.

    Layout: ``xxxbbbbb bbbbbbbb cdhsrrrr xxpppppp`` (bitrank, suit, rank, prime).
    """
    r = rank(card)
    return (1 << (16 + r)) | (_SUIT_BITS[suit(card)] << 12) | (r << 8) | PRIMES[r]


def new_deck(rng=None):
    """Return a freshly shuffled list of the 52 card integers."""
    deck = list(range(NUM_CARDS))
    (rng or random).shuffle(deck)
    return deck
