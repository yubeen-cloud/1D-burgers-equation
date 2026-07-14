"""Run cached benchmark pipelines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from rom_bench.config import load_yaml
from rom_bench.paths import resolve_path


DEFAULT_CONFIGS = {
    ("burgers", "generate"): "configs/burgers_generate.yaml",
    ("burgers", "pod"): "configs/burgers_pod.yaml",
    ("burgers", "dmd"): "configs/burgers_dmd.yaml",
    ("burgers", "autoencoder"): "configs/burgers_autoencoder.yaml",
    ("cylinder", "generate"): "configs/cylinder_data.yaml",
    ("cylinder", "pod"): "configs/cylinder_pod.yaml",
    ("cylinder", "dmd"): "configs/cylinder_dmd.yaml",
    ("cylinder", "autoencoder"): "configs/cylinder_autoencoder.yaml",
}

SCRIPTS = {
    "generate": "scripts/generate_burgers.py",
    "cylinder_generate": "scripts/prepare_cylinder_data.py",
    "pod": "scripts/train_pod.py",
    "dmd": "scripts/train_dmd.py",
    "autoencoder": "scripts/train_autoencoder.py",
}


def _is_done(config_path: str, method: str) -> bool:
    config = load_yaml(config_path)
    if method == "generate":
        return resolve_path(config["data"]["path"]).exists()
    exp = config["experiment"]["name"]
    return (resolve_path(config["experiment"].get("output_dir", "artifacts")) / "metrics" / f"{exp}_metrics.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROM benchmark pipeline")
    parser.add_argument("--problem", required=True, choices=["burgers", "cylinder"])
    parser.add_argument("--methods", nargs="+", default=["pod", "dmd", "autoencoder"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.problem == "burgers":
        stages = ["generate", *args.methods]
    else:
        stages = ["generate", *args.methods]

    for stage in stages:
        cfg = DEFAULT_CONFIGS[(args.problem, stage)]
        script_key = "cylinder_generate" if args.problem == "cylinder" and stage == "generate" else stage
        script = SCRIPTS[script_key]
        if not args.force and _is_done(cfg, stage):
            print(f"Reuse cached stage {stage}: {cfg}")
            continue
        command = [sys.executable, script, "--config", cfg]
        print(f"Running {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    subprocess.run(
        [sys.executable, "scripts/compare_models.py", "--problem", args.problem, "--results-dir", "artifacts"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )


if __name__ == "__main__":
    main()
