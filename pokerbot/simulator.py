"""Texas Hold'em game engine, players and training-data generation.

A cleaned-up port of the legacy table.py / tableQ.py simulator:
- cards are 0-51 ints end to end (converted only inside the evaluator)
- the (playerIndex - dealerIndex) position bug is fixed in features.py
- players are classes with a single ``act`` interface

The logged sample schema is kept compatible with the historical JSON data.
"""

import json
import random

from . import cards
from .evaluator import Evaluator
from .features import extract_features

SMALL_BLIND = 50
BIG_BLIND = 100
STACK = 200 * BIG_BLIND

# Thresholds of the legacy heuristic player (hand ranks; lower = stronger).
DUMB_THRESHOLD_SMALL = 500
DUMB_THRESHOLD_BIG = 1500
TWO_PAIR_MAX_RANK = 3325

FOLD, CALL, BET, ALL_IN = 0, 1, 2, 3

ROUND_NAMES = ("preflop", "flop", "turn", "river")


class Player:
    """Base player: always calls (capped at the stack)."""

    def __init__(self):
        self.hand = []

    def act(self, table, player_index, only_call=False):
        to_call = table.call_amount(player_index)
        sample = table.make_sample(self, player_index)
        action, amount = self.decide(table, player_index, to_call, only_call)
        amount = min(amount, STACK - table.bets[player_index])
        if table.bets[player_index] + amount >= STACK:
            action = ALL_IN
        sample["action"] = [action, amount]
        return action, amount, sample

    def decide(self, table, player_index, to_call, only_call):
        return CALL, to_call


class ThresholdPlayer(Player):
    """Legacy 'dumb' heuristic: calls until the river, then pot-bets strong hands."""

    def decide(self, table, player_index, to_call, only_call):
        if only_call or len(table.board) < 5:
            return CALL, to_call
        player_score = table.evaluator.evaluate(self.hand, table.board)
        table_score = table.evaluator.evaluate([], table.board)
        weak_edge = player_score > max(table_score - DUMB_THRESHOLD_SMALL, 1)
        if (weak_edge and table_score < TWO_PAIR_MAX_RANK) or player_score > max(
            table_score - DUMB_THRESHOLD_BIG, 1
        ):
            return CALL, to_call
        return BET, to_call + table.pot


class NeuralPlayer(Player):
    """Plays from per-round WinPredictor models: fold weak, call medium, bet strong."""

    def __init__(self, models, fold_below=0.3, bet_above=0.7):
        super().__init__()
        self.models = models  # dict: round number -> WinPredictor
        self.fold_below = fold_below
        self.bet_above = bet_above

    def decide(self, table, player_index, to_call, only_call):
        model = self.models.get(table.round_num)
        if model is None:
            return CALL, to_call
        sample = table.make_sample(self, player_index)
        p_win = model.predict_proba(extract_features(sample))
        if p_win < self.fold_below and to_call > 0:
            return FOLD, 0
        if p_win > self.bet_above and not only_call:
            return BET, to_call + table.pot
        return CALL, to_call


class Table:
    def __init__(self, players, rng=None):
        assert 2 <= len(players) <= 23
        self.players = players
        self.num_players = len(players)
        self.evaluator = Evaluator()
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.pot = 0
        self.bets = [0] * self.num_players
        self.remaining = [True] * self.num_players
        self.board = []
        self.round_num = 0
        self.dealer_index = 0
        self.logs = []

    def call_amount(self, player_index):
        return max(self.bets) - self.bets[player_index]

    def make_sample(self, player, player_index):
        """Snapshot the game state from one player's point of view."""
        sample = {
            "dealerIndex": self.dealer_index,
            "playerIndex": player_index,
            "handCards": list(player.hand),
            "tableCards": list(self.board),
            "pot": self.pot,
            "bets": list(self.bets),
            "remPlayers": list(self.remaining),
            "rd_num": self.round_num,
        }
        if len(player.hand) + len(self.board) >= 5:
            player_score = self.evaluator.evaluate(player.hand, self.board)
            hand_rank = self.evaluator.get_rank_class(player_score)
            sample["handScore"] = 10.0 * player_score / 7462
            sample["handRank"] = hand_rank
            if len(self.board) >= 5:
                table_score = self.evaluator.evaluate([], self.board)
                table_rank = self.evaluator.get_rank_class(table_score)
                sample["scoreDiff"] = 10.0 * (table_score - player_score) / table_score
                sample["rankDiff"] = table_rank - hand_rank
        return sample

    def play_game(self, dealer_index=0, log=True):
        """Play one full hand; return the logged samples with results filled in."""
        self.reset()
        self.dealer_index = dealer_index % self.num_players

        deck = cards.new_deck(self.rng)
        for player in self.players:
            player.hand = [deck.pop(), deck.pop()]

        self.bets[self.dealer_index] = SMALL_BLIND
        self.bets[(self.dealer_index + 1) % self.num_players] = BIG_BLIND
        self.pot = SMALL_BLIND + BIG_BLIND

        for round_num, num_new_cards in enumerate((0, 3, 1, 1)):
            self.round_num = round_num
            self.board += [deck.pop() for _ in range(num_new_cards)]
            self._betting_round(log)
            if sum(self.remaining) <= 1:
                break

        winners = self._find_winners()
        if log:
            for sample in self.logs:
                sample["result"] = 1 if sample["playerIndex"] in winners else -1
        return self.logs

    def _betting_round(self, log):
        """Two passes like the legacy engine: open betting, then call-only."""
        first = self.dealer_index if self.round_num == 0 else self.dealer_index + 1
        for only_call in (False, True):
            for offset in range(self.num_players):
                idx = (first + offset) % self.num_players
                if not self.remaining[idx]:
                    continue
                if only_call and self.call_amount(idx) == 0:
                    continue
                action, amount, sample = self.players[idx].act(self, idx, only_call)
                if log:
                    self.logs.append(sample)
                if action == FOLD:
                    self.remaining[idx] = False
                else:
                    self.bets[idx] += amount
                    self.pot += amount
                if sum(self.remaining) <= 1:
                    return

    def _find_winners(self):
        active = [i for i in range(self.num_players) if self.remaining[i]]
        if len(active) == 1 or len(self.board) < 5:
            return active
        best = float("inf")
        winners = []
        for i in active:
            score = self.evaluator.evaluate(self.players[i].hand, self.board)
            if score < best:
                best, winners = score, [i]
            elif score == best:
                winners.append(i)
        return winners


def generate_training_data(num_games, path, num_players=2, seed=None, print_every=10000):
    """Simulate heads-up (or multiway) hands between ThresholdPlayers to JSON."""
    rng = random.Random(seed)
    table = Table([ThresholdPlayer() for _ in range(num_players)], rng=rng)
    samples = []
    for i in range(num_games):
        samples += table.play_game(dealer_index=rng.randrange(num_players))
        if print_every and i % print_every == 0:
            print("simulated %d/%d games" % (i, num_games))
    with open(path, "w") as f:
        json.dump(samples, f)
    wins = sum(1 for s in samples if s["result"] == 1)
    print("wrote %d samples to %s (win ratio %.3f)" % (len(samples), path, wins / len(samples)))
    return samples
