"""Custom CNN feature extractor for MiniGrid observations."""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MinigridCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for small MiniGrid observations.

    SB3's default NatureCNN uses 8x8 kernels which are too large for
    MiniGrid's small grids (e.g., 5x5). This extractor uses smaller
    kernels appropriate for grid-world observations.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        """Initialize the CNN with two convolutional layers and a linear projection.

        Args:
            observation_space: Flattened MiniGrid observation space.
            features_dim: Output dimensionality of the linear projection.
        """
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[2]  # MiniGrid: (H, W, 3)

        self.cnn = nn.Sequential(
            # Transpose from (H, W, C) to (C, H, W) is handled by SB3
            nn.Conv2d(n_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            sample = torch.as_tensor(
                observation_space.sample()[None]
            ).permute(0, 3, 1, 2).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 CnnPolicy expects (B, C, H, W) but MiniGrid gives (B, H, W, C)
        # MiniGrid ImgObsWrapper gives encoded integers (0-10), not pixels (0-255)
        # Normalize by 10.0 to get values in [0, 1]
        x = observations.permute(0, 3, 1, 2).float() / 10.0
        return self.linear(self.cnn(x))
