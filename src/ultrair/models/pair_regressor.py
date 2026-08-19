import torch
import torch.nn as nn

from ultrair.models.pair_classifier import UltraPairClassifier


class UltraPairRegressor(UltraPairClassifier):
    """
    Uses the exact same pair backbone as UltraPairClassifier and replaces only the task head.
    """

    def __init__(self, *args, regression_dropout: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        pair_stats_dim = 10
        total_pair_repr_dim = self.pair_hidden_dim * 5 + pair_stats_dim
        self.regressor = nn.Sequential(
            nn.LayerNorm(total_pair_repr_dim),
            nn.Linear(total_pair_repr_dim, self.pair_hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(float(regression_dropout)),
            nn.Linear(self.pair_hidden_dim * 3, self.pair_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(regression_dropout)),
            nn.Linear(self.pair_hidden_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get("ir", None)
        if x is None:
            raise ValueError("UltraPairRegressor requires 'ir' input.")

        pair_repr = self.forward_pair_features(x)
        return self.regressor(pair_repr)

