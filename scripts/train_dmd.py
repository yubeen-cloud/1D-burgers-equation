"""Train/evaluate DMD for Burgers data."""

from __future__ import annotations

import csv
import json
import time

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from rom_bench.config import parse_config_args, save_yaml
from rom_bench.data.io import read_h5, write_json
from rom_bench.evaluation.field_metrics import error_over_time, first_threshold_time
from rom_bench.evaluation.front_tracking import front_position_error
from rom_bench.evaluation.reports import write_markdown_report
from rom_bench.models.dmd import DMDModel
from rom_bench.paths import ensure_dir, resolve_path
from rom_bench.seed import seed_everything
from rom_bench.visualization.burgers_plots import plot_error_curve, plot_field_comparison, plot_spacetime
from rom_bench.visualization.common import save_figure


def main() -> None:
    start = time.perf_counter()
    config, _args = parse_config_args("Train DMD")
    if config["problem"] != "burgers":
        raise ValueError("This project now supports only the 1D Burgers equation.")
    seed_everything(int(config["experiment"].get("seed", 42)))
    arrays, _metadata = read_h5(config["data"]["path"])
    case_index = int(config["data"].get("case_index", 0))
    x, t, u = arrays["x"], arrays["t"], arrays["u"][case_index]
    rollout_start = int(config["evaluation"].get("rollout_start_index", int(0.6 * len(t))))
    train = u[:rollout_start]
    dt = float(np.mean(np.diff(t)))
    model = DMDModel.fit(
        train,
        dt=dt,
        rank=int(config["model"]["rank"]),
        filter_unstable=bool(config["model"].get("filter_unstable", False)),
    )
    pred = model.predict(len(t))
    err = error_over_time(u, pred)
    front_err = front_position_error(x, u, pred, method=config["evaluation"].get("front_method", "max_gradient"))
    out_root = resolve_path(config["experiment"].get("output_dir", "."))
    exp_id = config["experiment"]["name"]
    fig_dir = ensure_dir(out_root / "figures" / "dmd" / exp_id)
    metric_dir = ensure_dir(out_root / "metrics")
    pred_dir = ensure_dir(out_root / "predictions")
    report_dir = ensure_dir(out_root / "reports")
    np.savez(pred_dir / f"{exp_id}_predictions.npz", x=x, t=t, truth=u, prediction=pred, eigenvalues=model.eigenvalues)
    plot_field_comparison(x, u[-1], pred[-1], "DMD final rollout", fig_dir / "dmd_final_rollout")
    plot_error_curve(t, err, "DMD rollout error", fig_dir / "rollout_error_vs_time")
    plot_spacetime(x, t, pred, "Burgers DMD prediction", fig_dir / "burgers_spacetime_prediction")
    plot_spacetime(x, t, abs(pred - u), "Burgers DMD absolute error", fig_dir / "burgers_spacetime_error")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    unit = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    ax.plot(unit.real, unit.imag, "k--", linewidth=1.0)
    ax.scatter(model.eigenvalues.real, model.eigenvalues.imag)
    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title("DMD eigenvalues")
    ax.axis("equal")
    save_figure(fig, fig_dir / "dmd_eigenvalues")
    metrics = {
        "method": "dmd",
        "problem": "burgers",
        "rank": int(config["model"]["rank"]),
        "reconstruction_relative_l2": float(np.mean(err[:rollout_start])),
        "rollout_relative_l2": float(np.mean(err[rollout_start:])),
        "final_rollout_error": float(err[-1]),
        "front_position_mae": float(front_err.mean()),
        "front_speed_error": float("nan"),
        "max_gradient_error": float(front_err.max()),
        "dominant_frequency": model.dominant_frequency(),
        "threshold_crossing_time": first_threshold_time(t, err, float(config["evaluation"].get("error_threshold", 0.2))),
        "training_time": time.perf_counter() - start,
        "inference_time": 0.0,
    }
    write_json(metric_dir / f"{exp_id}_metrics.json", metrics)
    with (metric_dir / f"{exp_id}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    save_yaml(config, fig_dir / "config_resolved.yaml")
    write_json(fig_dir / "runtime.json", {"seconds": metrics["training_time"]})
    write_markdown_report(
        report_dir / f"{exp_id}_summary.md",
        "Burgers DMD Summary",
        {
            "Experiment purpose": "DMD free rollout 오차 누적과 front 위치 drift 관찰",
            "Key metrics": metrics,
            "Largest error time": float(t[int(err.argmax())]),
            "Interpretation": [
                "DMD는 선형 시간 연산자로 nonlinear front steepening을 장시간 예측하기 어려움",
                "threshold_crossing_time 이후 rollout 신뢰도가 급격히 낮아지는 구간으로 볼 수 있음",
            ],
            "Figures": [str(p) for p in fig_dir.glob("*.png")],
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
