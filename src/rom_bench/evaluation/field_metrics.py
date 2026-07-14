"""Field and rollout error metrics."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def l2_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Absolute L2 error."""
    return float(np.linalg.norm(y_pred - y_true))


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-12) -> float:
    """Relative L2 error."""
    return float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + eps))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-12) -> float:
    """Normalized RMSE by true field range."""
    denom = np.max(y_true) - np.min(y_true)
    return rmse(y_true, y_pred) / float(denom + eps)


def max_pointwise_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute pointwise error."""
    return float(np.max(np.abs(y_pred - y_true)))


def spatial_gradient_error(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray) -> float:
    """Relative error of spatial gradients."""
    grad_true = np.gradient(y_true, x, axis=-1)
    grad_pred = np.gradient(y_pred, x, axis=-1)
    return relative_l2(grad_true, grad_pred)


def error_over_time(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Relative L2 error for each snapshot."""
    return np.array([relative_l2(y_true[i], y_pred[i]) for i in range(y_true.shape[0])])


def first_threshold_time(times: np.ndarray, errors: np.ndarray, threshold: float) -> float:
    """First time when error exceeds a threshold, or NaN if never."""
    indices = np.flatnonzero(errors > threshold)
    return float(times[indices[0]]) if len(indices) else float("nan")


def summarize_field_metrics(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray | None = None) -> dict[str, float]:
    """Return a common metric dictionary."""
    metrics = {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "l2": l2_error(y_true, y_pred),
        "relative_l2": relative_l2(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "max_pointwise_error": max_pointwise_error(y_true, y_pred),
    }
    if x is not None:
        metrics["spatial_gradient_error"] = spatial_gradient_error(y_true, y_pred, x)
    return metrics
