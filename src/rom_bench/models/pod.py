"""Proper Orthogonal Decomposition models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PODModel:
    """SVD-based POD basis."""

    mean: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    subtract_mean: bool = True

    @classmethod
    def fit(
        cls,
        snapshots: np.ndarray,
        rank: int | None = None,
        energy_threshold: float | None = None,
        subtract_mean: bool = True,
    ) -> "PODModel":
        """Fit POD from snapshots shaped [n_samples, n_features]."""
        mean = np.mean(snapshots, axis=0) if subtract_mean else np.zeros(snapshots.shape[1])
        centered = snapshots - mean
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
        if rank is None:
            if energy_threshold is None:
                rank = len(s)
            else:
                energy = np.cumsum(s**2) / np.sum(s**2)
                rank = int(np.searchsorted(energy, energy_threshold) + 1)
        rank = min(rank, len(s))
        return cls(mean=mean, modes=vh[:rank], singular_values=s, subtract_mean=subtract_mean)

    @property
    def rank(self) -> int:
        """Number of modes."""
        return int(self.modes.shape[0])

    def encode(self, snapshots: np.ndarray) -> np.ndarray:
        """Project snapshots to POD coefficients."""
        return (snapshots - self.mean) @ self.modes.T

    def decode(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct snapshots from POD coefficients."""
        return coeffs @ self.modes + self.mean

    def reconstruct(self, snapshots: np.ndarray) -> np.ndarray:
        """Project and reconstruct snapshots."""
        return self.decode(self.encode(snapshots))

    def cumulative_energy(self) -> np.ndarray:
        """Cumulative singular value energy."""
        energy = self.singular_values**2
        return np.cumsum(energy) / np.sum(energy)


def rank_errors(train: np.ndarray, test: np.ndarray, ranks: list[int], subtract_mean: bool = True) -> list[dict[str, float]]:
    """Compute train/test reconstruction error for several ranks."""
    from rom_bench.evaluation.field_metrics import relative_l2

    rows = []
    for rank in ranks:
        model = PODModel.fit(train, rank=rank, subtract_mean=subtract_mean)
        rows.append(
            {
                "rank": float(rank),
                "train_relative_l2": relative_l2(train, model.reconstruct(train)),
                "test_relative_l2": relative_l2(test, model.reconstruct(test)),
            }
        )
    return rows
