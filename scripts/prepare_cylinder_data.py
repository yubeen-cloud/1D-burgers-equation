"""Prepare synthetic or external cylinder wake data."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401

from rom_bench.config import parse_config_args
from rom_bench.data.cylinder import save_cylinder_dataset, synthetic_cylinder_wake
from rom_bench.data.io import environment_info
from rom_bench.paths import resolve_path


def main() -> None:
    config, _args = parse_config_args("Prepare cylinder wake data")
    data_cfg = config["data"]
    if data_cfg.get("source", "synthetic") != "synthetic":
        raise NotImplementedError("External/OpenFOAM loaders are documented placeholders in this Phase 2 scaffold.")
    arrays = synthetic_cylinder_wake(
        re_values=[float(v) for v in data_cfg.get("reynolds_numbers", [100, 150, 200])],
        nx=int(data_cfg.get("nx", 48)),
        ny=int(data_cfg.get("ny", 24)),
        nt=int(data_cfg.get("nt", 60)),
        dt=float(data_cfg.get("dt", 0.05)),
    )
    out = resolve_path(data_cfg["path"])
    metadata = {**environment_info(), "source": "synthetic_cylinder_wake", "config_path": config.get("_config_path")}
    save_cylinder_dataset(out, arrays, metadata)
    print(json.dumps({"path": str(out), "vorticity_shape": list(arrays["vorticity"].shape)}, indent=2))


if __name__ == "__main__":
    main()
