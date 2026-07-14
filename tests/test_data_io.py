from __future__ import annotations

import numpy as np

from rom_bench.data.io import read_h5, write_h5


def test_hdf5_roundtrip(tmp_path) -> None:
    path = tmp_path / "data.h5"
    arrays = {
        "x": np.arange(4),
        "split/train_indices": np.array([0]),
        "split/val_indices": np.array([1]),
        "split/test_indices": np.array([2]),
    }
    metadata = {"solver": "test", "random_seed": 1}
    write_h5(path, arrays, metadata)
    loaded, meta = read_h5(path)
    assert np.array_equal(loaded["x"], arrays["x"])
    assert meta["solver"] == "test"
    assert not set(loaded["split/train_indices"]).intersection(set(loaded["split/test_indices"]))
