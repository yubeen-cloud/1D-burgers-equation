"""Synthetic and external cylinder wake data interfaces."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from rom_bench.data.io import write_h5, write_json
from rom_bench.paths import resolve_path


def synthetic_cylinder_wake(
    re_values: list[float],
    nx: int = 48,
    ny: int = 24,
    nt: int = 60,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Generate a small analytic vortex-shedding-like cylinder dataset."""
    x = np.linspace(-1.0, 4.0, nx)
    y = np.linspace(-1.2, 1.2, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    t = np.arange(nt) * dt
    n_cases = len(re_values)
    u = np.empty((n_cases, nt, ny, nx))
    v = np.empty_like(u)
    p = np.empty_like(u)
    vort = np.empty_like(u)
    cl = np.empty((n_cases, nt))
    cd = np.empty((n_cases, nt))
    mask = (xx**2 + yy**2) < 0.15**2
    for c, re in enumerate(re_values):
        freq = 0.16 + 0.00025 * (re - 100.0)
        amp = 0.25 + 0.0008 * (re - 100.0)
        for i, ti in enumerate(t):
            phase = 2.0 * np.pi * freq * ti
            envelope = np.exp(-((xx - 1.2) ** 2) / 3.0) * np.exp(-(yy**2) / 0.5)
            wake = amp * envelope * np.sin(2.0 * np.pi * (xx - 0.4) - phase)
            u[c, i] = 1.0 - 0.35 * envelope + 0.05 * wake
            v[c, i] = wake
            p[c, i] = 0.15 * envelope * np.cos(2.0 * np.pi * (xx - 0.4) - phase)
            vort[c, i] = np.gradient(v[c, i], x, axis=1) - np.gradient(u[c, i], y, axis=0)
            u[c, i, mask] = 0.0
            v[c, i, mask] = 0.0
            p[c, i, mask] = 0.0
            vort[c, i, mask] = 0.0
            cl[c, i] = amp * np.sin(phase)
            cd[c, i] = 1.2 + 0.05 * np.cos(2.0 * phase)
    return {
        "x": x,
        "y": y,
        "t": t,
        "velocity/u": u,
        "velocity/v": v,
        "pressure": p,
        "vorticity": vort,
        "coefficients/cl": cl,
        "coefficients/cd": cd,
        "parameters/reynolds_number": np.asarray(re_values),
        "parameters/u_inf": np.ones(n_cases),
        "parameters/diameter": np.ones(n_cases),
        "mask": mask.astype(np.uint8),
        "split/train_indices": np.array([0], dtype=np.int64),
        "split/val_indices": np.array([1], dtype=np.int64) if n_cases > 2 else np.array([], dtype=np.int64),
        "split/test_indices": np.array([n_cases - 1], dtype=np.int64),
    }


def save_cylinder_dataset(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Save synthetic or processed cylinder data and sidecar summaries."""
    out = resolve_path(path)
    write_h5(out, arrays, metadata)
    summary = {
        "field_shape": list(arrays["vorticity"].shape),
        "reynolds_numbers": arrays["parameters/reynolds_number"].tolist(),
        "has_cl": "coefficients/cl" in arrays,
        "has_cd": "coefficients/cd" in arrays,
    }
    write_json(out.parent / "dataset_summary.json", summary)
    index_path = out.parent / "snapshot_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "snapshot_id", "time", "reynolds_number", "split", "source_file", "processed_file", "has_cl", "has_cd", "normalization_id"])
        train = set(arrays["split/train_indices"].tolist())
        val = set(arrays["split/val_indices"].tolist())
        test = set(arrays["split/test_indices"].tolist())
        for case_id, re in enumerate(arrays["parameters/reynolds_number"]):
            split = "train" if case_id in train else "val" if case_id in val else "test" if case_id in test else "unused"
            for sid, time in enumerate(arrays["t"]):
                writer.writerow([case_id, sid, time, re, split, "synthetic", str(out), True, True, "train_stats"])


def load_external_placeholder(path: str | Path) -> dict[str, np.ndarray]:
    """Load a user-provided HDF5/NumPy cylinder dataset placeholder interface."""
    raise NotImplementedError(
        f"External cylinder loader is intentionally explicit. Convert {path} to the documented HDF5 layout first."
    )
