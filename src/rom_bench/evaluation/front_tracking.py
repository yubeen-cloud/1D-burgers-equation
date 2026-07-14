"""Front tracking metrics for 1D moving structures."""

from __future__ import annotations

from typing import Literal

import numpy as np

FrontMethod = Literal["max_gradient", "threshold", "midpoint", "cross_correlation"]


def periodic_distance(a: float, b: float, length: float) -> float:
    """Shortest periodic distance between two positions."""
    raw = abs(a - b)
    return float(min(raw, length - raw))


def front_position(
    x: np.ndarray,
    u: np.ndarray,
    method: FrontMethod = "max_gradient",
    threshold: float | None = None,
    reference: np.ndarray | None = None,
) -> float:
    """Estimate a 1D front location."""
    if method == "max_gradient":
        return float(x[int(np.argmax(np.abs(np.gradient(u, x))))])
    if method == "threshold":
        level = float(threshold if threshold is not None else 0.5 * (np.max(u) + np.min(u)))
        idx = int(np.argmin(np.abs(u - level)))
        return float(x[idx])
    if method == "midpoint":
        level = 0.5 * (float(np.max(u)) + float(np.min(u)))
        idx = int(np.argmin(np.abs(u - level)))
        return float(x[idx])
    if method == "cross_correlation":
        if reference is None:
            raise ValueError("reference is required for cross_correlation front tracking")
        corr = np.fft.ifft(np.fft.fft(u) * np.conj(np.fft.fft(reference))).real
        shift = int(np.argmax(corr))
        dx = float(x[1] - x[0])
        return float(x[0] + shift * dx)
    raise ValueError(f"Unknown front method: {method}")


def front_positions(
    x: np.ndarray,
    snapshots: np.ndarray,
    method: FrontMethod = "max_gradient",
    periodic: bool = False,
) -> np.ndarray:
    """Estimate front location for every snapshot."""
    reference = snapshots[0] if method == "cross_correlation" else None
    pos = np.array([front_position(x, u, method=method, reference=reference) for u in snapshots])
    if periodic:
        length = float(x[-1] - x[0] + (x[1] - x[0]))
        pos = np.mod(pos - x[0], length) + x[0]
    return pos


def front_position_error(
    x: np.ndarray,
    true_snapshots: np.ndarray,
    pred_snapshots: np.ndarray,
    method: FrontMethod = "max_gradient",
    periodic: bool = False,
) -> np.ndarray:
    """Front position absolute error over time."""
    true_pos = front_positions(x, true_snapshots, method, periodic)
    pred_pos = front_positions(x, pred_snapshots, method, periodic)
    if periodic:
        length = float(x[-1] - x[0] + (x[1] - x[0]))
        return np.array([periodic_distance(a, b, length) for a, b in zip(true_pos, pred_pos)])
    return np.abs(true_pos - pred_pos)


def front_speed_error(x: np.ndarray, times: np.ndarray, true_snapshots: np.ndarray, pred_snapshots: np.ndarray) -> float:
    """Mean absolute error of front speed."""
    true_speed = np.gradient(front_positions(x, true_snapshots), times)
    pred_speed = np.gradient(front_positions(x, pred_snapshots), times)
    return float(np.mean(np.abs(true_speed - pred_speed)))
