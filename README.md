# PokerBot

A Texas Hold'em bot that learns to predict its probability of winning a hand
from simulated games, built on PyTorch. One dense MLP (default: 4 layers x
128 neurons) is trained per betting round, in the spirit of DeepStack's
per-street value networks.

## Layout

- `pokerbot/` — the package
  - `simulator.py` — game engine, heuristic and neural players, data generation
  - `evaluator.py` — 5/6/7-card hand evaluator (adapted from [deuces](https://github.com/worldveil/deuces))
  - `features.py`, `dataset.py` — feature extraction and data loading
  - `model.py`, `train.py` — the win-predictor MLP and its training CLI
- `legacy/` — the original TensorFlow 1.x notebook exports, kept for reference
- `Result/` — TF1 checkpoints trained with the legacy code (not loadable by
  the new PyTorch pipeline)

## Quickstart

```bash
pip install -r requirements.txt

# 1. Simulate hands between heuristic players to build a dataset
python -m pokerbot.simulate data/games.json --games 100000 --seed 42

# 2. Train a win predictor for the river (round 3)
python -m pokerbot.train data/games.json --round 3 --epochs 100 --out models/river.pt

# 3. Play the trained model against the heuristic player
python - <<'EOF'
from pokerbot.model import load_model
from pokerbot.simulator import Table, ThresholdPlayer, NeuralPlayer

table = Table([NeuralPlayer({3: load_model("models/river.pt")}), ThresholdPlayer()])
logs = table.play_game()
print(logs[-1])
EOF
```

Rounds are numbered 0 = preflop, 1 = flop, 2 = turn, 3 = river. Each round
has its own feature length (hand-strength features only exist once five
cards are visible), so train one model per round and pass them to
`NeuralPlayer` as a `{round: model}` dict.

## Data format

`pokerbot.simulate` writes a JSON list of per-decision samples, schema-
compatible with the historical training data: cards are integers 0-51
(`rank = card % 13`, `suit = card // 13`), plus pot, bets, positions,
hand-strength scores on later streets, and `result` (+1 won / -1 lost).

## Credits

Hand evaluation is adapted from Will Drevo's [deuces](https://github.com/worldveil/deuces)
(MIT), originally vendored via [Alex Beloi's nn-holdem](https://github.com/alexbeloi/nn-holdem).
