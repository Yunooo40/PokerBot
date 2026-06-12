"""Win-predictor model: a dense MLP that outputs the logit of P(win).

Same shape family as the legacy TF1 models (n dense ReLU layers of equal
width, dropout before the readout), rebuilt in PyTorch with a single-logit
output trained with binary cross-entropy.
"""

import torch
from torch import nn


class WinPredictor(nn.Module):
    def __init__(self, num_features, hidden_size=128, num_layers=4, dropout=0.5):
        super().__init__()
        self.config = {
            "num_features": num_features,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        layers = []
        in_size = num_features
        for _ in range(num_layers):
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU()]
            in_size = hidden_size
        layers += [nn.Dropout(dropout), nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, feats):
        """P(win) for a single feature vector (list of floats)."""
        self.eval()
        x = torch.tensor([feats], dtype=torch.float32)
        return torch.sigmoid(self(x)).item()


def save_model(model, path):
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = WinPredictor(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
