from __future__ import annotations

import numpy as np

from rom_bench.evaluation.field_metrics import relative_l2
from rom_bench.evaluation.frequency import fft_peak_frequency
from rom_bench.evaluation.front_tracking import periodic_distance


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

