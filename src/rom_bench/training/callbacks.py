"""Training callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Minimal early stopping state."""

    patience: int
    best: float = float("inf")
    count: int = 0

    def update(self, value: float) -> bool:
        """Return True when training should stop."""
        if value < self.best:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience
