"""Run POD-Galerkin ROM for Burgers and expose nonlinear RHS bottleneck."""

from __future__ import annotations

import csv
import json
import time

import _bootstrap  # noqa: F401
import numpy as np

from rom_bench.config import parse_config_args, save_yaml
from rom_bench.data.io import read_h5, write_json
from rom_bench.evaluation.field_metrics import error_over_time, first_threshold_time, relative_l2, spatial_gradient_error
from rom_bench.evaluation.front_tracking import front_position_error
from rom_bench.evaluation.reports import write_markdown_report
from rom_bench.models.pod import PODModel
from rom_bench.models.pod_galerkin import PODGalerkinBurgers
from rom_bench.paths import ensure_dir, resolve_path
from rom_bench.seed import seed_everything
from rom_bench.visualization.burgers_plots import plot_error_curve, plot_field_comparison, plot_spacetime
from rom_bench.visualization.common import save_figure


def plot_bottleneck_breakdown(values: dict[str, float], path) -> None:
    """Save a small timing breakdown figure."""
    import matplotlib.pyplot as plt

    labels = list(values)
    seconds = [values[key] for key in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(labels, seconds)
    ax.set_ylabel("seconds")
    ax.set_title("POD-Galerkin RHS timing breakdown")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, path)


def main() -> None:
    start = time.perf_counter()
    config, _args = parse_config_args("Train/evaluate Burgers POD-Galerkin ROM")
    if config["problem"] != "burgers":
        raise ValueError("POD-Galerkin script currently supports only Burgers")

    seed_everything(int(config["experiment"].get("seed", 42)))
    arrays, _metadata = read_h5(config["data"]["path"])
    x, t, u = arrays["x"], arrays["t"], arrays["u"]
    train_cases = arrays["split/train_indices"].astype(int)
    case_index = int(config["data"].get("case_index", 0))
    truth = u[case_index]
    train = u[train_cases].reshape(-1, u.shape[-1])

    model_cfg = config["model"]
    rank = int(model_cfg["rank"])
    pod = PODModel.fit(
        train,
        rank=rank,
        subtract_mean=bool(model_cfg.get("subtract_mean", True)),
    )
    viscosity = float(arrays["parameters/nu"][case_index])
    rom = PODGalerkinBurgers(
        pod=pod,
        x=x,
        viscosity=viscosity,
        boundary_condition=config.get("solver", {}).get("boundary_condition", "periodic"),
        modal_damping=float(model_cfg.get("modal_damping", 0.0)),
    )
    result = rom.rollout(truth[0], t, cfl=float(config.get("solver", {}).get("cfl", 0.2)))
    prediction = result.states
    errors = error_over_time(truth, prediction)
    front_err = front_position_error(x, truth, prediction, method=config["evaluation"].get("front_method", "max_gradient"))

    out_root = resolve_path(config["experiment"].get("output_dir", "artifacts"))
    exp_id = config["experiment"]["name"]
    fig_dir = ensure_dir(out_root / "burgers" / "figures" / "pod_rom" / exp_id)
    metric_dir = ensure_dir(out_root / "burgers" / "metrics")
    pred_dir = ensure_dir(out_root / "burgers" / "predictions")
    report_dir = ensure_dir(out_root / "burgers" / "reports")

    np.savez(
        pred_dir / f"{exp_id}_predictions.npz",
        x=x,
        t=t,
        truth=truth,
        prediction=prediction,
        coefficients=result.coefficients,
        pod_modes=pod.modes,
    )
    plot_field_comparison(x, truth[-1], prediction[-1], "POD-Galerkin final rollout", fig_dir / "pod_rom_final_rollout")
    plot_error_curve(t, errors, "POD-Galerkin rollout error", fig_dir / "rollout_error_vs_time")
    plot_spacetime(x, t, prediction, "Burgers POD-Galerkin rollout", fig_dir / "burgers_spacetime_prediction")
    plot_spacetime(x, t, abs(prediction - truth), "Burgers POD-Galerkin absolute error", fig_dir / "burgers_spacetime_error")

    timing_breakdown = {
        "reconstruct": result.stats.reconstruction_seconds,
        "full_rhs": result.stats.full_rhs_seconds,
        "project": result.stats.projection_seconds,
    }
    plot_bottleneck_breakdown(timing_breakdown, fig_dir / "pod_rom_rhs_timing_breakdown")

    elapsed = time.perf_counter() - start
    rhs_avg = result.stats.rhs_seconds / max(result.stats.rhs_calls, 1)
    metrics = {
        "method": "pod_rom",
        "problem": "burgers",
        "rank": rank,
        "case_index": case_index,
        "viscosity": viscosity,
        "reconstruction_relative_l2": float("nan"),
        "rollout_relative_l2": relative_l2(truth, prediction),
        "final_rollout_error": float(errors[-1]),
        "front_position_mae": float(front_err.mean()),
        "front_speed_error": float("nan"),
        "max_gradient_error": spatial_gradient_error(truth, prediction, x),
        "rhs_calls": result.stats.rhs_calls,
        "rhs_total_seconds": result.stats.rhs_seconds,
        "rhs_avg_seconds": rhs_avg,
        "rhs_reconstruction_seconds": result.stats.reconstruction_seconds,
        "rhs_full_grid_nonlinear_seconds": result.stats.full_rhs_seconds,
        "rhs_projection_seconds": result.stats.projection_seconds,
        "full_grid_size": int(x.size),
        "reduced_rank": rank,
        "compression_ratio": float(x.size / rank),
        "threshold_crossing_time": first_threshold_time(t, errors, float(config["evaluation"].get("error_threshold", 0.2))),
        "training_time": elapsed,
        "inference_time": elapsed,
    }
    write_json(metric_dir / f"{exp_id}_metrics.json", metrics)
    with (metric_dir / f"{exp_id}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    save_yaml(config, fig_dir / "config_resolved.yaml")
    write_json(fig_dir / "runtime.json", {"seconds": elapsed})
    write_json(
        fig_dir / "bottleneck_explanation.json",
        {
            "point": "This POD-ROM is reduced in state dimension, but each nonlinear RHS call still reconstructs a full-grid state and evaluates burgers_rhs on all grid points.",
            "rhs_calls": result.stats.rhs_calls,
            "full_grid_size": int(x.size),
            "reduced_rank": rank,
            "compression_ratio": float(x.size / rank),
        },
    )
    write_markdown_report(
        report_dir / f"{exp_id}_summary.md",
        "Burgers POD-Galerkin ROM Summary",
        {
            "Experiment purpose": "POD projection reconstruction과 달리 POD-Galerkin time rollout 및 full-grid nonlinear RHS 병목 확인",
            "Key metrics": metrics,
            "Largest error time": float(t[int(errors.argmax())]),
            "Interpretation": [
                "이 실험은 POD basis 위에서 coefficient를 시간 적분한다.",
                "Burgers 비선형항은 hyper-reduction 없이 full grid에서 계산되므로 reduced rank가 작아도 RHS 비용이 nx에 의존한다.",
                "Projection POD error와 POD-ROM rollout error는 다른 지표다.",
            ],
            "Figures": [str(p) for p in fig_dir.glob("*.png")],
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
