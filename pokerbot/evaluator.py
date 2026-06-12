"""Poker hand evaluator for 5, 6 and 7 cards.

Adapted from the ``deuces`` library by Will Drevo (MIT licence), which the
legacy table.py vendored via Alex Beloi's nn-holdem. The public API takes
cards in this project's 0-51 integer encoding (see cards.py).

Hand ranks are in [1, 7462]; lower is stronger (1 = royal flush).
"""

import itertools

from . import cards


class LookupTable:
    """Maps a 5-card hand's unique prime product to its rank in [1, 7462]."""

    MAX_STRAIGHT_FLUSH = 10
    MAX_FOUR_OF_A_KIND = 166
    MAX_FULL_HOUSE = 322
    MAX_FLUSH = 1599
    MAX_STRAIGHT = 1609
    MAX_THREE_OF_A_KIND = 2467
    MAX_TWO_PAIR = 3325
    MAX_PAIR = 6185
    MAX_HIGH_CARD = 7462

    MAX_TO_RANK_CLASS = {
        MAX_STRAIGHT_FLUSH: 1,
        MAX_FOUR_OF_A_KIND: 2,
        MAX_FULL_HOUSE: 3,
        MAX_FLUSH: 4,
        MAX_STRAIGHT: 5,
        MAX_THREE_OF_A_KIND: 6,
        MAX_TWO_PAIR: 7,
        MAX_PAIR: 8,
        MAX_HIGH_CARD: 9,
    }

    RANK_CLASS_TO_STRING = {
        1: "Straight Flush",
        2: "Four of a Kind",
        3: "Full House",
        4: "Flush",
        5: "Straight",
        6: "Three of a Kind",
        7: "Two Pair",
        8: "Pair",
        9: "High Card",
    }

    def __init__(self):
        self.flush_lookup = {}
        self.unsuited_lookup = {}
        self._flushes()  # also fills straights and high cards
        self._multiples()

    def _flushes(self):
        # straight flush rank-bit patterns, best to worst (5-high last)
        straight_flushes = [7936, 3968, 1984, 992, 496, 248, 124, 62, 31, 4111]

        flushes = []
        gen = self._next_bit_sequence(0b11111)
        # 1277 high-card patterns + the straight flushes interleaved
        for _ in range(1277 + len(straight_flushes) - 1):
            f = next(gen)
            if all(f ^ sf for sf in straight_flushes):
                flushes.append(f)
        flushes.reverse()

        rank = 1
        for sf in straight_flushes:
            self.flush_lookup[self._prime_product_from_rankbits(sf)] = rank
            rank += 1

        rank = LookupTable.MAX_FULL_HOUSE + 1
        for f in flushes:
            self.flush_lookup[self._prime_product_from_rankbits(f)] = rank
            rank += 1

        # same bit sequences rank straights and high cards
        rank = LookupTable.MAX_FLUSH + 1
        for s in straight_flushes:
            self.unsuited_lookup[self._prime_product_from_rankbits(s)] = rank
            rank += 1

        rank = LookupTable.MAX_PAIR + 1
        for h in flushes:
            self.unsuited_lookup[self._prime_product_from_rankbits(h)] = rank
            rank += 1

    def _multiples(self):
        primes = cards.PRIMES
        backwards = list(range(cards.NUM_RANKS - 1, -1, -1))

        rank = LookupTable.MAX_STRAIGHT_FLUSH + 1
        for i in backwards:
            for k in (r for r in backwards if r != i):
                self.unsuited_lookup[primes[i] ** 4 * primes[k]] = rank
                rank += 1

        rank = LookupTable.MAX_FOUR_OF_A_KIND + 1
        for i in backwards:
            for pr in (r for r in backwards if r != i):
                self.unsuited_lookup[primes[i] ** 3 * primes[pr] ** 2] = rank
                rank += 1

        rank = LookupTable.MAX_STRAIGHT + 1
        for r in backwards:
            kickers = [k for k in backwards if k != r]
            for c1, c2 in itertools.combinations(kickers, 2):
                self.unsuited_lookup[primes[r] ** 3 * primes[c1] * primes[c2]] = rank
                rank += 1

        rank = LookupTable.MAX_THREE_OF_A_KIND + 1
        for p1, p2 in itertools.combinations(backwards, 2):
            for k in (r for r in backwards if r != p1 and r != p2):
                self.unsuited_lookup[primes[p1] ** 2 * primes[p2] ** 2 * primes[k]] = rank
                rank += 1

        rank = LookupTable.MAX_TWO_PAIR + 1
        for pairrank in backwards:
            kickers = [k for k in backwards if k != pairrank]
            for k1, k2, k3 in itertools.combinations(kickers, 3):
                self.unsuited_lookup[
                    primes[pairrank] ** 2 * primes[k1] * primes[k2] * primes[k3]
                ] = rank
                rank += 1

    @staticmethod
    def _prime_product_from_rankbits(rankbits):
        product = 1
        for i in range(cards.NUM_RANKS):
            if rankbits & (1 << i):
                product *= cards.PRIMES[i]
        return product

    @staticmethod
    def _next_bit_sequence(bits):
        """Lexicographically next bit permutations, in poker rank order.

        Bit hack from
        http://www-graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
        """
        nxt = bits
        while True:
            t = (nxt | (nxt - 1)) + 1
            nxt = t | ((((t & -t) // (nxt & -nxt)) >> 1) - 1)
            yield nxt


class Evaluator:
    """Evaluates hand strength via prime-product table lookups (Cactus Kev)."""

    def __init__(self):
        self.table = LookupTable()

    def evaluate(self, hand, board):
        """Rank the best 5-card hand from ``hand + board`` (0-51 cards).

        Accepts 5, 6 or 7 cards in total; ``hand`` may be empty to score the
        board alone. Lower is stronger.
        """
        ints = [cards.to_eval_int(c) for c in hand] + [cards.to_eval_int(c) for c in board]
        n = len(ints)
        if n == 5:
            return self._five(ints)
        if n in (6, 7):
            return min(self._five(combo) for combo in itertools.combinations(ints, 5))
        raise ValueError("evaluate() needs 5 to 7 cards, got %d" % n)

    def _five(self, ints):
        if ints[0] & ints[1] & ints[2] & ints[3] & ints[4] & 0xF000:
            hand_or = (ints[0] | ints[1] | ints[2] | ints[3] | ints[4]) >> 16
            prime = LookupTable._prime_product_from_rankbits(hand_or)
            return self.table.flush_lookup[prime]
        prime = 1
        for c in ints:
            prime *= c & 0xFF
        return self.table.unsuited_lookup[prime]

    def get_rank_class(self, hand_rank):
        """Map a hand rank to its class (1 = straight flush ... 9 = high card)."""
        for max_rank in sorted(LookupTable.MAX_TO_RANK_CLASS):
            if hand_rank <= max_rank:
                return LookupTable.MAX_TO_RANK_CLASS[max_rank]
        raise ValueError("Invalid hand rank %r" % (hand_rank,))

    def class_to_string(self, class_int):
        return LookupTable.RANK_CLASS_TO_STRING[class_int]
