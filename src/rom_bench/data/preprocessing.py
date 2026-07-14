"""Preprocessing and normalization utilities."""

from __future__ import annotations

import numpy as np


def normalize_train_stats(train: np.ndarray) -> dict[str, np.ndarray]:
    """Return mean/std normalization statistics."""
    return {"mean": np.mean(train, axis=0), "std": np.std(train, axis=0) + 1.0e-12}


def apply_normalization(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Normalize data."""
    return (data - stats["mean"]) / stats["std"]


def inverse_normalization(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Undo normalization."""
    return data * stats["std"] + stats["mean"]
