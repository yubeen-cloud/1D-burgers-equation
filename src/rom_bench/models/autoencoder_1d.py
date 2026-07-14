"""1D autoencoder models with PyTorch and NumPy fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


if nn is not None:

    class Conv1dAutoencoder(nn.Module):
        """Small Conv1D autoencoder for 1D fields."""

        def __init__(self, nx: int, latent_dim: int, hidden_channels: int = 16) -> None:
            super().__init__()
            self.nx = nx
            self.encoder = nn.Sequential(
                nn.Conv1d(1, hidden_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Flatten(),
            )
            encoded_size = hidden_channels * ((nx + 1) // 2)
            self.to_latent = nn.Linear(encoded_size, latent_dim)
            self.from_latent = nn.Linear(latent_dim, encoded_size)
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (hidden_channels, (nx + 1) // 2)),
                nn.Upsample(size=nx, mode="linear", align_corners=False),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_channels, 1, kernel_size=5, padding=2),
            )

        def encode(self, x):
            """Encode [batch, nx] to latent."""
            h = self.encoder(x[:, None, :])
            return self.to_latent(h)

        def decode(self, z):
            """Decode latent to [batch, nx]."""
            h = self.from_latent(z)
            return self.decoder(h)[:, 0, :]

        def forward(self, x):
            """Reconstruct input."""
            return self.decode(self.encode(x))

else:

    class Conv1dAutoencoder:  # type: ignore[no-redef]
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for Conv1dAutoencoder")


@dataclass
class DenseAutoencoderFallback:
    """NumPy dense nonlinear autoencoder fallback for smoke tests."""

    mean: np.ndarray
    scale: float
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    w4: np.ndarray
    b4: np.ndarray

    @classmethod
    def fit(
        cls,
        snapshots: np.ndarray,
        latent_dim: int,
        hidden_dim: int = 32,
        epochs: int = 500,
        learning_rate: float = 1.0e-3,
        seed: int = 0,
    ) -> "DenseAutoencoderFallback":
        """Fit a small dense autoencoder with Adam."""
        rng = np.random.default_rng(seed)
        mean = np.mean(snapshots, axis=0)
        scale = float(np.std(snapshots) + 1.0e-12)
        x = (snapshots - mean) / scale
        n, d = x.shape
        w1 = 0.1 * rng.standard_normal((d, hidden_dim)) / np.sqrt(d)
        b1 = np.zeros(hidden_dim)
        w2 = 0.1 * rng.standard_normal((hidden_dim, latent_dim)) / np.sqrt(hidden_dim)
        b2 = np.zeros(latent_dim)
        w3 = 0.1 * rng.standard_normal((latent_dim, hidden_dim)) / np.sqrt(latent_dim)
        b3 = np.zeros(hidden_dim)
        w4 = 0.1 * rng.standard_normal((hidden_dim, d)) / np.sqrt(hidden_dim)
        b4 = np.zeros(d)
        params = [w1, b1, w2, b2, w3, b3, w4, b4]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1.0e-8
        for epoch in range(1, epochs + 1):
            h1 = np.tanh(x @ w1 + b1)
            z = np.tanh(h1 @ w2 + b2)
            h3 = np.tanh(z @ w3 + b3)
            y = h3 @ w4 + b4
            dy = (2.0 / n) * (y - x) / d
            dw4, db4 = h3.T @ dy, np.sum(dy, axis=0)
            dg3 = (dy @ w4.T) * (1.0 - h3**2)
            dw3, db3 = z.T @ dg3, np.sum(dg3, axis=0)
            dg2 = (dg3 @ w3.T) * (1.0 - z**2)
            dw2, db2 = h1.T @ dg2, np.sum(dg2, axis=0)
            dg1 = (dg2 @ w2.T) * (1.0 - h1**2)
            dw1, db1 = x.T @ dg1, np.sum(dg1, axis=0)
            grads = [dw1, db1, dw2, db2, dw3, db3, dw4, db4]
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1.0 - beta1) * g
                v[i] = beta2 * v[i] + (1.0 - beta2) * g**2
                p -= learning_rate * (m[i] / (1.0 - beta1**epoch)) / (np.sqrt(v[i] / (1.0 - beta2**epoch)) + eps)
        return cls(mean, scale, w1, b1, w2, b2, w3, b3, w4, b4)

    def encode(self, snapshots: np.ndarray) -> np.ndarray:
        """Encode snapshots."""
        x = (snapshots - self.mean) / self.scale
        h1 = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h1 @ self.w2 + self.b2)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent states."""
        h3 = np.tanh(latent @ self.w3 + self.b3)
        y = h3 @ self.w4 + self.b4
        return y * self.scale + self.mean

    def reconstruct(self, snapshots: np.ndarray) -> np.ndarray:
        """Reconstruct snapshots."""
        return self.decode(self.encode(snapshots))
