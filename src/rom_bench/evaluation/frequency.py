"""Frequency and Strouhal utilities."""

from __future__ import annotations

import numpy as np


def fft_peak_frequency(times: np.ndarray, signal: np.ndarray, start_index: int = 0) -> float:
    """Return dominant FFT frequency after start_index."""
    t = times[start_index:]
    y = signal[start_index:] - np.mean(signal[start_index:])
    if len(t) < 3:
        return float("nan")
    dt = float(np.mean(np.diff(t)))
    freqs = np.fft.rfftfreq(len(y), dt)
    spectrum = np.abs(np.fft.rfft(y))
    if len(spectrum) <= 1:
        return float("nan")
    idx = int(np.argmax(spectrum[1:]) + 1)
    return float(freqs[idx])


def strouhal_number(frequency: float, diameter: float, u_inf: float) -> float:
    """Compute St = fD/U_inf."""
    return float(frequency * diameter / u_inf)
