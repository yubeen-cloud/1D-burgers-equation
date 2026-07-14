"""Train/evaluate POD for Burgers data."""

from __future__ import annotations

import csv
import json
import time

import _bootstrap  # noqa: F401

from rom_bench.config import parse_config_args, save_yaml
from rom_bench.data.io import read_h5, write_json
from rom_bench.evaluation.field_metrics import error_over_time, first_threshold_time, relative_l2
from rom_bench.evaluation.front_tracking import front_position_error, front_speed_error
from rom_bench.evaluation.reports import write_markdown_report
from rom_bench.models.pod import PODModel, rank_errors
from rom_bench.paths import ensure_dir, resolve_path
from rom_bench.seed import seed_everything
from rom_bench.visualization.burgers_plots import plot_error_curve, plot_field_comparison, plot_pod_spectrum, plot_spacetime


def _burgers(config: dict) -> None:
    start = time.perf_counter()
    seed_everything(int(config["experiment"].get("seed", 42)))
    arrays, _metadata = read_h5(config["data"]["path"])
    x, t, u = arrays["x"], arrays["t"], arrays["u"]
    train_cases = arrays["split/train_indices"].astype(int)
    train = u[train_cases].reshape(-1, u.shape[-1])
    case_index = int(config["data"].get("case_index", 0))
    all_snapshots = u[case_index]
    model_cfg = config["model"]
    rank = int(model_cfg["rank"])
    model = PODModel.fit(train, rank=rank, energy_threshold=model_cfg.get("energy_threshold"), subtract_mean=bool(model_cfg.get("subtract_mean", True)))
    reconstruction = model.reconstruct(all_snapshots)
    err = error_over_time(all_snapshots, reconstruction)
    front_err = front_position_error(x, all_snapshots, reconstruction, method=config["evaluation"].get("front_method", "max_gradient"))
    ranks = config.get("comparison", {}).get("ranks", [rank])
    rank_table = rank_errors(train, all_snapshots, [int(r) for r in ranks], subtract_mean=bool(model_cfg.get("subtract_mean", True)))
    out_root = resolve_path(config["experiment"].get("output_dir", "artifacts"))
    exp_id = config["experiment"]["name"]
    fig_dir = ensure_dir(out_root / "burgers" / "figures" / "pod" / exp_id)
    metric_dir = ensure_dir(out_root / "burgers" / "metrics")
    pred_dir = ensure_dir(out_root / "burgers" / "predictions")
    report_dir = ensure_dir(out_root / "burgers" / "reports")
    import numpy as np

    np.savez(pred_dir / f"{exp_id}_predictions.npz", x=x, t=t, truth=all_snapshots, reconstruction=reconstruction)
    plot_pod_spectrum(model.singular_values, model.cumulative_energy(), fig_dir / "pod_energy_spectrum")
    plot_field_comparison(x, all_snapshots[-1], reconstruction[-1], "POD final reconstruction", fig_dir / "pod_final_reconstruction")
    plot_error_curve(t, err, "POD reconstruction error", fig_dir / "rollout_error_vs_time")
    plot_spacetime(x, t, all_snapshots, "Burgers true", fig_dir / "burgers_spacetime_true")
    plot_spacetime(x, t, reconstruction, "Burgers POD reconstruction", fig_dir / "burgers_spacetime_reconstruction")
    plot_spacetime(x, t, abs(reconstruction - all_snapshots), "Burgers POD absolute error", fig_dir / "burgers_spacetime_error")
    metrics = {
        "method": "pod",
        "problem": "burgers",
        "rank": rank,
        "reconstruction_relative_l2": relative_l2(all_snapshots, reconstruction),
        "rollout_relative_l2": float("nan"),
        "final_rollout_error": float(err[-1]),
        "front_position_mae": float(front_err.mean()),
        "front_speed_error": front_speed_error(x, t, all_snapshots, reconstruction),
        "max_gradient_error": float(front_err.max()),
        "training_time": time.perf_counter() - start,
        "inference_time": 0.0,
    }
    write_json(metric_dir / f"{exp_id}_metrics.json", metrics)
    with (metric_dir / f"{exp_id}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    with (metric_dir / f"{exp_id}_rank_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rank_table[0].keys()))
        writer.writeheader()
        writer.writerows(rank_table)
    save_yaml(config, fig_dir / "config_resolved.yaml")
    write_json(fig_dir / "runtime.json", {"seconds": metrics["training_time"]})
    write_markdown_report(
        report_dir / f"{exp_id}_summary.md",
        "Burgers POD Summary",
        {
            "Experiment purpose": "moving front에서 POD rank와 reconstruction failure 관찰",
            "Key metrics": metrics,
            "Largest error time": float(t[int(err.argmax())]),
            "Interpretation": [
                "moving front는 위치가 이동하므로 낮은 rank의 선형 basis로 표현하기 어려움",
                "전체 L2 error가 작아도 front 위치 오차는 별도로 확인해야 함",
            ],
            "Figures": [str(p) for p in fig_dir.glob("*.png")],
        },
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    config, _args = parse_config_args("Train POD")
    if config["problem"] != "burgers":
        raise ValueError("This project now supports only the 1D Burgers equation.")
    _burgers(config)


if __name__ == "__main__":
    main()
