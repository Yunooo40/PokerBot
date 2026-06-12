"""PokerBot: a Texas Hold'em win-probability bot built on PyTorch.

Modules:
- cards:     0-51 card encoding utilities
- evaluator: 5/6/7-card hand evaluator (adapted from deuces)
- features:  feature extraction from logged game states
- simulator: game engine, players and training-data generation
- dataset:   JSON training data loading and array building
- model:     PyTorch win-predictor MLP
- train:     training CLI (python -m pokerbot.train)
- simulate:  data-generation CLI (python -m pokerbot.simulate)
"""
