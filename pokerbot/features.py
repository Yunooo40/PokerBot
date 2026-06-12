"""Feature extraction from logged game-state samples.

A sample is a dict with the schema produced by the simulator (and by the
historical JSON training data): handCards, tableCards (0-51 ints), rd_num,
pot, bets, remPlayers, playerIndex, dealerIndex, and on later streets
handScore/handRank and scoreDiff/rankDiff.

The feature vector length depends on the street and player count, which is
why one model is trained per betting round.
"""

from . import cards

# Money amounts are normalised by the big blind so feature scales stay sane.
BET_SCALE = 100.0


def extract_features(sample, cards_on=True, other_on=True):
    """Return the feature vector (list of floats) for one sample."""
    feats = []
    if cards_on:
        suit_counts = [0] * cards.NUM_SUITS
        rank_counts = [0] * cards.NUM_RANKS
        for card in sample["handCards"] + sample["tableCards"]:
            suit_counts[cards.suit(card)] += 1
            rank_counts[cards.rank(card)] += 1
        feats += suit_counts + rank_counts
    if "handScore" in sample:
        feats += [float(sample["handScore"]), float(sample["handRank"])]
    if "scoreDiff" in sample:
        feats += [float(sample["scoreDiff"]), float(sample["rankDiff"])]
    if other_on:
        num_players = len(sample["bets"])
        position = (sample["playerIndex"] - sample["dealerIndex"]) % num_players
        feats += [b / BET_SCALE for b in sample["bets"]]
        feats.append(sample["pot"] / BET_SCALE)
        feats.append(float(position))
        feats += [1.0 if p else 0.0 for p in sample["remPlayers"]]
    return feats
