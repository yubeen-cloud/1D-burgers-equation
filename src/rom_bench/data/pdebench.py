"""PDEBench Burgers dataset helpers."""

from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PDEBENCH_BURGERS_URLS = {
    "0.001": "https://darus.uni-stuttgart.de/api/access/datafile/268190",
    "0.002": "https://darus.uni-stuttgart.de/api/access/datafile/268193",
    "0.004": "https://darus.uni-stuttgart.de/api/access/datafile/268191",
    "0.01": "https://darus.uni-stuttgart.de/api/access/datafile/281363",
    "0.02": "https://darus.uni-stuttgart.de/api/access/datafile/268189",
    "0.04": "https://darus.uni-stuttgart.de/api/access/datafile/281362",
    "0.1": "https://darus.uni-stuttgart.de/api/access/datafile/268185",
    "0.2": "https://darus.uni-stuttgart.de/api/access/datafile/268187",
    "0.4": "https://darus.uni-stuttgart.de/api/access/datafile/268192",
    "1.0": "https://darus.uni-stuttgart.de/api/access/datafile/281365",
    "2.0": "https://darus.uni-stuttgart.de/api/access/datafile/281364",
    "4.0": "https://darus.uni-stuttgart.de/api/access/datafile/268188",
}


def pdebench_burgers_filename(nu: str) -> str:
    """Return the canonical PDEBench Burgers filename for a viscosity value."""
    return f"1D_Burgers_Sols_Nu{nu}.hdf5"


def download_file(url: str, path: Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Download a URL to disk and return basic metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context()
    md5 = hashlib.md5()
    bytes_written = 0
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with path.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                md5.update(chunk)
                bytes_written += len(chunk)
                if total and bytes_written % (512 * chunk_size) < chunk_size:
                    pct = 100.0 * bytes_written / total
                    print(f"Downloaded {bytes_written / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)", flush=True)
    return {"bytes": bytes_written, "md5": md5.hexdigest()}


def inspect_pdebench_burgers(path: Path) -> dict[str, Any]:
    """Inspect a PDEBench Burgers HDF5 file."""
    with h5py.File(path, "r") as h5:
        return {
            "keys": list(h5.keys()),
            "tensor_shape": tuple(h5["tensor"].shape),
            "x_shape": tuple(h5["x-coordinate"].shape),
            "t_shape": tuple(h5["t-coordinate"].shape) if "t-coordinate" in h5 else None,
        }


def convert_pdebench_burgers_subset(
    source_path: Path,
    nu: float,
    n_cases: int,
    case_offset: int,
    time_stride: int,
    space_stride: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert a subset of PDEBench Burgers to the local standard HDF5 layout."""
    with h5py.File(source_path, "r") as h5:
        tensor = h5["tensor"]
        if len(tensor.shape) == 4:
            n_total, nt_total, nx_total, channels = tensor.shape
            if channels != 1:
                raise ValueError(f"Expected one channel for Burgers tensor, got {channels}")
        elif len(tensor.shape) == 3:
            n_total, nt_total, nx_total = tensor.shape
            channels = 1
        else:
            raise ValueError(f"Expected 3D or 4D Burgers tensor, got shape {tensor.shape}")
        end = min(case_offset + n_cases, n_total)
        case_slice = slice(case_offset, end)
        time_slice = slice(None, None, time_stride)
        space_slice = slice(None, None, space_stride)
        if len(tensor.shape) == 4:
            u = np.asarray(tensor[case_slice, time_slice, space_slice, 0], dtype=np.float64)
        else:
            u = np.asarray(tensor[case_slice, time_slice, space_slice], dtype=np.float64)
        x = np.asarray(h5["x-coordinate"][space_slice], dtype=np.float64)
        if "t-coordinate" in h5:
            t = np.asarray(h5["t-coordinate"][:nt_total][time_slice], dtype=np.float64)
        else:
            t = np.arange(u.shape[1], dtype=np.float64)

    n = u.shape[0]
    train_end = max(1, int(0.6 * n))
    val_end = max(train_end + 1, int(0.8 * n)) if n > 2 else train_end
    train = np.arange(0, min(train_end, n), dtype=np.int64)
    val = np.arange(train[-1] + 1, min(val_end, n), dtype=np.int64) if n > 1 else np.array([], dtype=np.int64)
    test = np.setdiff1d(np.arange(n, dtype=np.int64), np.concatenate([train, val]))

    arrays = {
        "x": x,
        "t": t,
        "u": u,
        "parameters/nu": np.full(n, float(nu), dtype=np.float64),
        "parameters/amplitude": np.full(n, np.nan, dtype=np.float64),
        "parameters/front_location": np.full(n, np.nan, dtype=np.float64),
        "parameters/front_width": np.full(n, np.nan, dtype=np.float64),
        "split/train_indices": train,
        "split/val_indices": val,
        "split/test_indices": test,
    }
    metadata = {
        "source": "PDEBench",
        "source_file": str(source_path),
        "source_url": PDEBENCH_BURGERS_URLS.get(str(nu)),
        "source_tensor_shape": [int(n_total), int(nt_total), int(nx_total), int(channels)],
        "subset": {
            "case_offset": int(case_offset),
            "n_cases_requested": int(n_cases),
            "n_cases_used": int(n),
            "time_stride": int(time_stride),
            "space_stride": int(space_stride),
        },
    }
    return arrays, metadata
