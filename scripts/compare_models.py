"""Compare model metrics and write summary reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from rom_bench.evaluation.reports import write_markdown_report
from rom_bench.paths import ensure_dir, resolve_path
from rom_bench.visualization.common import save_figure


def _load_rows(metrics_dir: Path, problem: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in metrics_dir.glob("*_metrics.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("problem") == problem:
            rows.append(data)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows: list[dict[str, object]], metric: str, path: Path) -> None:
    labels = [str(row.get("method", "unknown")) for row in rows]
    values = [float(row.get(metric, np.nan)) for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(labels, values)
    ax.set_ylabel(metric)
    ax.set_title(metric.replace("_", " "))
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ROM model metrics")
    parser.add_argument("--problem", required=True, choices=["burgers", "cylinder"])
    parser.add_argument("--results-dir", default="artifacts")
    args = parser.parse_args()

    root = resolve_path(args.results_dir)
    metric_dir = root / "metrics"
    rows = _load_rows(metric_dir, args.problem)
    if not rows:
        print(f"No metrics found for {args.problem} in {metric_dir}")
        return
    out_csv = metric_dir / f"{args.problem}_model_comparison.csv"
    _write_csv(out_csv, rows)
    fig_dir = ensure_dir(root / "figures" / args.problem / "comparison")
    for metric in ["reconstruction_relative_l2", "rollout_relative_l2", "final_rollout_error", "front_position_mae"]:
        _plot_metric(rows, metric, fig_dir / f"{metric}_comparison")
    report = root / "reports" / f"{args.problem}_comparison.md"
    write_markdown_report(
        report,
        f"{args.problem.title()} Model Comparison",
        {
            "Compared rows": rows,
            "Interpretation criteria": [
                "정확도만 보지 않고 rollout 안정성, moving structure 표현, 계산 비용을 함께 비교",
                "사용 불가능한 지표는 NaN으로 남겨 후속 실험에서 보완",
            ],
            "Output CSV": str(out_csv),
        },
    )
    print(json.dumps({"comparison_csv": str(out_csv), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
