"""Prepare a PDEBench 1D Burgers subset for ROM experiments."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import numpy as np

from rom_bench.config import parse_config_args, save_yaml
from rom_bench.data.io import environment_info, write_h5, write_json
from rom_bench.data.pdebench import (
    PDEBENCH_BURGERS_URLS,
    convert_pdebench_burgers_subset,
    download_file,
    inspect_pdebench_burgers,
    pdebench_burgers_filename,
)
from rom_bench.paths import resolve_path
from rom_bench.visualization.burgers_plots import plot_spacetime


def main() -> None:
    config, _args = parse_config_args("Prepare PDEBench Burgers public dataset subset")
    source_cfg = config["source"]
    subset_cfg = config["subset"]
    nu = str(source_cfg.get("nu", "0.01"))
    url = str(source_cfg.get("url") or PDEBENCH_BURGERS_URLS[nu])
    raw_path = resolve_path(source_cfg.get("path") or f"data/external/pdebench/{pdebench_burgers_filename(nu)}")
    if bool(source_cfg.get("download", False)) and not raw_path.exists():
        download_meta = download_file(url, raw_path)
    else:
        download_meta = {"downloaded": False, "reason": "file exists or source.download=false"}
    if not raw_path.exists():
        raise FileNotFoundError(
            f"PDEBench Burgers file is not available: {raw_path}. "
            f"Set source.download=true to download {url} or place the file there manually."
        )

    inspection = inspect_pdebench_burgers(raw_path)
    selection_strategy = str(subset_cfg.get("selection_strategy", "contiguous"))
    selection_seed = int(subset_cfg.get("selection_seed", config["experiment"].get("seed", 42)))
    n_cases = int(subset_cfg.get("n_cases", 64))
    if selection_strategy == "random_without_replacement":
        n_total = int(inspection["tensor_shape"][0])
        case_indices = np.random.default_rng(selection_seed).choice(
            n_total, size=n_cases, replace=False
        )
    elif selection_strategy == "contiguous":
        case_indices = None
    else:
        raise ValueError(f"Unsupported subset.selection_strategy: {selection_strategy}")
    arrays, source_metadata = convert_pdebench_burgers_subset(
        source_path=raw_path,
        nu=float(nu),
        n_cases=n_cases,
        case_offset=int(subset_cfg.get("case_offset", 0)),
        time_stride=int(subset_cfg.get("time_stride", 2)),
        space_stride=int(subset_cfg.get("space_stride", 8)),
        case_indices=case_indices,
        split_seed=selection_seed,
    )
    source_metadata["subset"]["selection_strategy"] = selection_strategy
    source_metadata["subset"]["selection_seed"] = selection_seed
    metadata = {
        **environment_info(),
        **source_metadata,
        "download": download_meta,
        "inspection": inspection,
        "config_path": config.get("_config_path"),
        "notes": "Converted subset of public PDEBench 1D Burgers HDF5 to local ROM benchmark layout.",
    }
    out_path = resolve_path(config["data"]["path"])
    write_h5(out_path, arrays, metadata)
    write_json(out_path.with_suffix(".metadata.json"), metadata)
    save_yaml(config, out_path.with_suffix(".resolved.yaml"))

    fig_dir = resolve_path(config["experiment"].get("output_dir", ".")) / "figures" / "data" / config["experiment"]["name"]
    plot_spacetime(arrays["x"], arrays["t"], arrays["u"][0], "PDEBench Burgers subset sample", fig_dir / "pdebench_burgers_sample")
    print(json.dumps({"path": str(out_path), "u_shape": list(arrays["u"].shape), "source": str(raw_path)}, indent=2))


if __name__ == "__main__":
    main()
