from __future__ import annotations

import numpy as np

from rom_bench.evaluation.field_metrics import relative_l2
from rom_bench.evaluation.frequency import fft_peak_frequency
from rom_bench.evaluation.front_tracking import periodic_distance
from rom_bench.evaluation.phase_error import cross_correlation_lag


def test_same_field_error_zero() -> None:
    x = np.arange(5.0)
    assert relative_l2(x, x) == 0.0


def test_periodic_distance() -> None:
    assert periodic_distance(0.95, 0.05, 1.0) == 0.10000000000000009


def test_fft_peak_frequency() -> None:
    dt = 0.01
    t = np.arange(1000) * dt
    y = np.sin(2 * np.pi * 3.0 * t)
    assert abs(fft_peak_frequency(t, y) - 3.0) < 0.05


def test_cross_correlation_lag() -> None:
    dt = 0.1
    t = np.arange(100) * dt
    a = np.sin(t)
    b = np.roll(a, 2)
    assert abs(cross_correlation_lag(a, b, dt) - 0.2) < 0.11
