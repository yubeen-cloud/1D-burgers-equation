"""Lift and drag coefficient metrics."""

from __future__ import annotations

import numpy as np

from rom_bench.evaluation.frequency import fft_peak_frequency


def coefficient_metrics(times: np.ndarray, cl: np.ndarray | None, cd: np.ndarray | None) -> dict[str, float]:
    """Return force coefficient metrics when signals are available."""
    metrics: dict[str, float] = {}
    if cd is not None:
        metrics["cd_mean"] = float(np.mean(cd))
    if cl is not None:
        metrics["cl_rms"] = float(np.sqrt(np.mean((cl - np.mean(cl)) ** 2)))
        metrics["cl_dominant_frequency"] = fft_peak_frequency(times, cl)
    return metrics
