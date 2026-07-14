"""Evaluate saved predictions."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401

from rom_bench.config import parse_config_args


def main() -> None:
    config, _args = parse_config_args("Evaluate model outputs")
    print(json.dumps({"message": "Evaluation is run inside train_* scripts for Phase 1.", "config": config.get("experiment", {})}, indent=2))


if __name__ == "__main__":
    main()
