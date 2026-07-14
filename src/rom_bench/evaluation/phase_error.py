"""Phase drift utilities."""

from __future__ import annotations

import numpy as np


def cross_correlation_lag(signal_true: np.ndarray, signal_pred: np.ndarray, dt: float) -> float:
    """Estimate lag by cross correlation."""
    a = signal_true - np.mean(signal_true)
    b = signal_pred - np.mean(signal_pred)
    corr = np.correlate(b, a, mode="full")
    lag_idx = int(np.argmax(corr) - (len(a) - 1))
    return float(lag_idx * dt)


def phase_drift_from_peaks(times: np.ndarray, true_signal: np.ndarray, pred_signal: np.ndarray) -> dict[str, float]:
    """Simple phase drift proxy using global cross-correlation lag."""
    lag = cross_correlation_lag(true_signal, pred_signal, float(np.mean(np.diff(times))))
    duration = float(times[-1] - times[0])
    return {
        "mean_phase_error_proxy_time": abs(lag),
        "final_phase_error_proxy_time": abs(lag),
        "phase_drift_rate_proxy": abs(lag) / duration if duration > 0 else float("nan"),
    }
