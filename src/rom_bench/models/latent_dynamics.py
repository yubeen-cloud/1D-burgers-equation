"""Simple latent dynamics models."""

from __future__ import annotations

import numpy as np


class LatentLinearDynamics:
    """Linear map z_{k+1} = A z_k."""

    def __init__(self) -> None:
        self.operator: np.ndarray | None = None

    def fit(self, latent: np.ndarray) -> "LatentLinearDynamics":
        """Fit from latent trajectory shaped [nt, latent_dim]."""
        z1 = latent[:-1].T
        z2 = latent[1:].T
        self.operator = z2 @ np.linalg.pinv(z1)
        return self

    def rollout(self, z0: np.ndarray, n_steps: int) -> np.ndarray:
        """Roll out latent states."""
        if self.operator is None:
            raise RuntimeError("LatentLinearDynamics must be fit first")
        z = z0.copy()
        out = [z.copy()]
        for _ in range(1, n_steps):
            z = self.operator @ z
            out.append(z.copy())
        return np.asarray(out)
