from __future__ import annotations

import numpy as np

from rom_bench.models.dmd import DMDModel


def test_dmd_recovers_linear_oscillation_frequency() -> None:
    dt = 0.05
    freq = 1.0
    t = np.arange(80) * dt
    x = np.column_stack([np.cos(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
    model = DMDModel.fit(x, dt=dt, rank=2)
    assert np.min(np.abs(model.frequencies() - freq)) < 0.05


def test_dmd_prediction_shape() -> None:
    data = np.random.default_rng(0).normal(size=(10, 4))
    model = DMDModel.fit(data, dt=0.1, rank=3)
    pred = model.predict(10)
    assert pred.shape == data.shape
    assert np.isrealobj(pred)
