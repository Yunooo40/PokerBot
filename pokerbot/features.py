"""Feature extraction from logged game-state samples.

A sample is a dict with the schema produced by the simulator (and by the
historical JSON training data): handCards, tableCards (0-51 ints), rd_num,
pot, bets, remPlayers, playerIndex, dealerIndex, and on later streets
handScore/handRank and scoreDiff/rankDiff. Samples produced by the current
simulator additionally carry a Monte-Carlo ``equity`` estimate.

The feature vector length depends on the street, the player count and the
fields present, which is why one model is trained per betting round and per
dataset (the model checkpoint records its expected feature count).
"""

from . import cards

# Money amounts are normalised by the big blind so feature scales stay sane.
BET_SCALE = 100.0


def _draw_flags(all_cards):
    """Flush-draw and straight-draw flags computed from the visible cards."""
    suit_counts = [0] * cards.NUM_SUITS
    ranks = set()
    for card in all_cards:
        suit_counts[cards.suit(card)] += 1
        ranks.add(cards.rank(card))
    flush_draw = 1.0 if max(suit_counts) == 4 else 0.0
    # four consecutive ranks present; ace also plays low (A-2-3-4)
    if 12 in ranks:
        ranks = ranks | {-1}
    straight_draw = 0.0
    for low in range(-1, 10):
        if all(r in ranks for r in range(low, low + 4)):
            straight_draw = 1.0
            break
    return flush_draw, straight_draw


def extract_features(sample, cards_on=True, other_on=True):
    """Return the feature vector (list of floats) for one sample."""
    feats = []
    all_cards = sample["handCards"] + sample["tableCards"]
    if cards_on:
        suit_counts = [0] * cards.NUM_SUITS
        rank_counts = [0] * cards.NUM_RANKS
        for card in all_cards:
            suit_counts[cards.suit(card)] += 1
            rank_counts[cards.rank(card)] += 1
        feats += suit_counts + rank_counts
        feats += _draw_flags(all_cards)
    if "equity" in sample:
        feats.append(float(sample["equity"]))
    if "handScore" in sample:
        feats += [float(sample["handScore"]), float(sample["handRank"])]
    if "scoreDiff" in sample:
        feats += [float(sample["scoreDiff"]), float(sample["rankDiff"])]
    if other_on:
        num_players = len(sample["bets"])
        position = (sample["playerIndex"] - sample["dealerIndex"]) % num_players
        to_call = max(sample["bets"]) - sample["bets"][sample["playerIndex"]]
        pot_odds = to_call / (sample["pot"] + to_call) if to_call > 0 else 0.0
        feats += [b / BET_SCALE for b in sample["bets"]]
        feats.append(sample["pot"] / BET_SCALE)
        feats.append(to_call / BET_SCALE)
        feats.append(pot_odds)
        feats.append(float(position))
        feats += [1.0 if p else 0.0 for p in sample["remPlayers"]]
    return feats
