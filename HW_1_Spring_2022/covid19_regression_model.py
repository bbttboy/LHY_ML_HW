import torch.nn as nn


class Covid19RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(self.dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
