"""Loading and preparing training data from simulator JSON files."""

import json

import numpy as np

from .features import extract_features


def load_samples(path):
    with open(path) as f:
        return json.load(f)


def build_arrays(samples, round_num=None, cards_on=True, other_on=True):
    """Build (X, y) float32/float32 arrays from raw samples.

    - round_num: keep only samples of that betting round (None = all; only
      sensible when all rounds share a feature length).
    - y is 1.0 for a won hand, 0.0 otherwise.
    """
    rows, labels = [], []
    for sample in samples:
        if round_num is not None and sample["rd_num"] != round_num:
            continue
        rows.append(extract_features(sample, cards_on=cards_on, other_on=other_on))
        labels.append(1.0 if sample["result"] == 1 else 0.0)
    if not rows:
        raise ValueError("No samples matched round_num=%r" % (round_num,))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def train_test_split(X, y, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - test_frac))
    train_idx, test_idx = perm[:cut], perm[cut:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
