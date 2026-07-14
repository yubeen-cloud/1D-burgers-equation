"""HDF5 and metadata I/O."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from rom_bench.paths import resolve_path


def git_info() -> dict[str, Any]:
    """Return git commit and dirty status when git is available."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty, "available": True}
    except Exception as exc:
        return {"commit": None, "dirty": None, "available": False, "reason": str(exc)}


def environment_info() -> dict[str, Any]:
    """Return lightweight environment metadata."""
    packages: dict[str, str | None] = {}
    for name in ["numpy", "scipy", "pandas", "matplotlib", "h5py", "yaml", "torch", "sklearn"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git": git_info(),
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON with UTF-8 encoding."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON."""
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_h5(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Write arrays and metadata to HDF5."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h5:
        for key, value in arrays.items():
            group = h5
            parts = key.split("/")
            for part in parts[:-1]:
                group = group.require_group(part)
            group.create_dataset(parts[-1], data=value)
        h5.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)


def read_h5(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read all HDF5 datasets into memory."""
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(resolve_path(path), "r") as h5:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                arrays[name] = obj[()]

        h5.visititems(visitor)
        metadata = json.loads(h5.attrs.get("metadata_json", "{}"))
    return arrays, metadata
