"""Run the final public-data Burgers workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from rom_bench.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final Burgers ROM workflow")
    parser.add_argument("--config", default="configs/burgers/run_all.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    project_root = Path(__file__).resolve().parents[1]
    for step in config["steps"]:
        command = [sys.executable, step["script"]]
        if step.get("config"):
            command.extend(["--config", step["config"]])
        command.extend(str(value) for value in step.get("args", []))
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
