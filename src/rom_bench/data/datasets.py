"""Dataset wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SnapshotDataset:
    """In-memory snapshot dataset."""

    x: np.ndarray
    t: np.ndarray
    fields: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
