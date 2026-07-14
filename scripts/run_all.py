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
    ("burgers", "generate"): "configs/burgers/generate.yaml",
    ("burgers", "pod"): "configs/burgers/pod.yaml",
    ("burgers", "pod_rom"): "configs/burgers/pod_rom.yaml",
    ("burgers", "dmd"): "configs/burgers/dmd.yaml",
    ("burgers", "autoencoder"): "configs/burgers/autoencoder.yaml",
}

SCRIPTS = {
    "generate": "scripts/generate_burgers.py",
    "pod": "scripts/train_pod.py",
    "pod_rom": "scripts/train_pod_rom.py",
    "dmd": "scripts/train_dmd.py",
    "autoencoder": "scripts/train_autoencoder.py",
}


def _is_done(config_path: str, method: str) -> bool:
    config = load_yaml(config_path)
    if method == "generate":
        return resolve_path(config["data"]["path"]).exists()
    exp = config["experiment"]["name"]
    problem = config["problem"]
    return (
        resolve_path(config["experiment"].get("output_dir", "artifacts"))
        / problem
        / "metrics"
        / f"{exp}_metrics.json"
    ).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROM benchmark pipeline")
    parser.add_argument("--problem", default="burgers", choices=["burgers"])
    parser.add_argument("--methods", nargs="+", default=["pod", "dmd", "autoencoder"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stages = ["generate", *args.methods]

    for stage in stages:
        cfg = DEFAULT_CONFIGS[(args.problem, stage)]
        script = SCRIPTS[stage]
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
