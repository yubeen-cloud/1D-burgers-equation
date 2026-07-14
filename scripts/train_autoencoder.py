"""Train/evaluate 1D autoencoder for Burgers data."""

from __future__ import annotations

import csv
import json
import time

import _bootstrap  # noqa: F401
import numpy as np

from rom_bench.config import parse_config_args, save_yaml
from rom_bench.data.io import read_h5, write_json
from rom_bench.evaluation.field_metrics import error_over_time, relative_l2
from rom_bench.evaluation.front_tracking import front_position_error
from rom_bench.evaluation.reports import write_markdown_report
from rom_bench.models.autoencoder_1d import DenseAutoencoderFallback
from rom_bench.models.latent_dynamics import LatentLinearDynamics
from rom_bench.paths import ensure_dir, resolve_path
from rom_bench.seed import seed_everything
from rom_bench.visualization.burgers_plots import plot_error_curve, plot_field_comparison, plot_spacetime
from rom_bench.visualization.latent_plots import plot_latent_trajectory


def main() -> None:
    start = time.perf_counter()
    config, _args = parse_config_args("Train Burgers autoencoder")
    if config["problem"] != "burgers":
        print("Cylinder autoencoder placeholder: Conv2D model is available, full trainer is Phase 2 extension.")
        return
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed)
    arrays, _metadata = read_h5(config["data"]["path"])
    case_index = int(config["data"].get("case_index", 0))
    x, t, u = arrays["x"], arrays["t"], arrays["u"][case_index]
    train_end = int(config["evaluation"].get("rollout_start_index", int(0.6 * len(t))))
    model_cfg = config["model"]
    training_cfg = config["training"]
    model = DenseAutoencoderFallback.fit(
        u[:train_end],
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        epochs=int(training_cfg.get("epochs", 500)),
        learning_rate=float(training_cfg.get("learning_rate", 1.0e-3)),
        seed=seed,
    )
    reconstruction = model.reconstruct(u)
    latent = model.encode(u)
    latent_rollout = LatentLinearDynamics().fit(latent[:train_end]).rollout(latent[0], len(t))
    rollout = model.decode(latent_rollout)
    rec_err = error_over_time(u, reconstruction)
    roll_err = error_over_time(u, rollout)
    front_err = front_position_error(x, u, rollout, method=config["evaluation"].get("front_method", "max_gradient"))
    out_root = resolve_path(config["experiment"].get("output_dir", "artifacts"))
    exp_id = config["experiment"]["name"]
    fig_dir = ensure_dir(out_root / "figures" / "burgers" / "autoencoder" / exp_id)
    metric_dir = ensure_dir(out_root / "metrics")
    pred_dir = ensure_dir(out_root / "predictions")
    ckpt_dir = ensure_dir(out_root / "checkpoints" / "burgers" / "autoencoder" / exp_id)
    report_dir = ensure_dir(out_root / "reports")
    np.savez(pred_dir / f"{exp_id}_predictions.npz", x=x, t=t, truth=u, reconstruction=reconstruction, rollout=rollout, latent=latent)
    np.savez(ckpt_dir / "best_checkpoint.npz", **model.__dict__)
    np.savez(ckpt_dir / "last_checkpoint.npz", **model.__dict__)
    plot_field_comparison(x, u[-1], reconstruction[-1], "AE final reconstruction", fig_dir / "autoencoder_final_reconstruction")
    plot_field_comparison(x, u[-1], rollout[-1], "AE latent rollout", fig_dir / "autoencoder_latent_rollout_final")
    plot_error_curve(t, rec_err, "AE reconstruction error", fig_dir / "reconstruction_error_vs_time")
    plot_error_curve(t, roll_err, "AE latent rollout error", fig_dir / "rollout_error_vs_time")
    plot_spacetime(x, t, abs(reconstruction - u), "AE reconstruction absolute error", fig_dir / "burgers_spacetime_error")
    plot_latent_trajectory(t, latent, "Autoencoder latent trajectory", fig_dir / "latent_trajectory")
    metrics = {
        "method": "autoencoder",
        "problem": "burgers",
        "latent_dim": int(model_cfg["latent_dim"]),
        "reconstruction_relative_l2": relative_l2(u, reconstruction),
        "rollout_relative_l2": float(np.mean(roll_err[train_end:])),
        "final_rollout_error": float(roll_err[-1]),
        "front_position_mae": float(front_err.mean()),
        "front_speed_error": float("nan"),
        "max_gradient_error": float(front_err.max()),
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
        "Burgers Autoencoder Summary",
        {
            "Experiment purpose": "AE nonlinear reconstruction과 latent rollout failure 관찰",
            "Key metrics": metrics,
            "Largest error time": float(t[int(roll_err.argmax())]),
            "Interpretation": [
                "AE reconstruction이 좋아도 latent linear dynamics가 정확하다는 뜻은 아님",
                "front-weighted loss는 config에 남겨두었고 긴 학습에서 PyTorch trainer로 확장 가능",
            ],
            "Figures": [str(p) for p in fig_dir.glob("*.png")],
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
