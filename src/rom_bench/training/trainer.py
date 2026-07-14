"""PyTorch autoencoder trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_torch_autoencoder(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Placeholder trainer used when PyTorch is installed in a full environment."""
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise ImportError("PyTorch is required for train_torch_autoencoder") from exc
    raise NotImplementedError(
        "This project uses a NumPy fallback for smoke tests; extend this trainer for long AE runs."
    )


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a PyTorch checkpoint when torch is available."""
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
