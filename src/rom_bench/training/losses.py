"""Training losses."""

from __future__ import annotations

import numpy as np


def relative_l2_loss_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-12) -> float:
    """NumPy relative L2 loss."""
    return float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + eps))


def gradient_loss_np(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray) -> float:
    """NumPy spatial gradient loss."""
    return relative_l2_loss_np(np.gradient(y_true, x, axis=-1), np.gradient(y_pred, x, axis=-1))


def front_weights_np(y_true: np.ndarray, x: np.ndarray, strength: float = 5.0) -> np.ndarray:
    """Give high weights near large gradients."""
    grad = np.abs(np.gradient(y_true, x, axis=-1))
    scale = np.max(grad, axis=-1, keepdims=True) + 1.0e-12
    return 1.0 + strength * grad / scale


def weighted_mse_np(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Weighted MSE."""
    return float(np.mean(weights * (y_pred - y_true) ** 2))
