"""2D convolutional autoencoder for cylinder fields."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


if nn is not None:

    class Conv2dAutoencoder(nn.Module):
        """Small Conv2D autoencoder for image-like fields."""

        def __init__(self, channels: int, latent_dim: int, hidden_channels: int = 16) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
            )
            self.to_latent = nn.Linear(hidden_channels * 64, latent_dim)
            self.from_latent = nn.Linear(latent_dim, hidden_channels * 64)
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (hidden_channels, 8, 8)),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, channels, 3, padding=1),
            )

        def encode(self, x):
            """Encode fields."""
            return self.to_latent(self.encoder(x))

        def decode(self, z):
            """Decode latent states."""
            return self.decoder(self.from_latent(z))

        def forward(self, x):
            """Reconstruct input."""
            return self.decode(self.encode(x))

else:

    class Conv2dAutoencoder:  # type: ignore[no-redef]
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for Conv2dAutoencoder")
