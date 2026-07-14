from __future__ import annotations

import numpy as np

from rom_bench.evaluation.field_metrics import relative_l2
from rom_bench.models.pod import PODModel


def test_energy_ratio_monotone() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(12, 8))
    model = PODModel.fit(data, rank=4)
    energy = model.cumulative_energy()
    assert np.all(np.diff(energy) >= -1e-14)


def test_full_rank_reconstruction_matches() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(8, 5))
    model = PODModel.fit(data, rank=5)
    rec = model.reconstruct(data)
    assert relative_l2(data, rec) < 1e-12


def test_rank_increase_does_not_increase_train_error() -> None:
    rng = np.random.default_rng(2)
    data = rng.normal(size=(20, 10))
    e2 = relative_l2(data, PODModel.fit(data, rank=2).reconstruct(data))
    e4 = relative_l2(data, PODModel.fit(data, rank=4).reconstruct(data))
    assert e4 <= e2 + 1e-12
