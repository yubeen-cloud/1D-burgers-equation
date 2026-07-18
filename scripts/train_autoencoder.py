"""Train/evaluate 1D autoencoder for Burgers data."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from rom_bench.config import parse_config_args, save_yaml
from rom_bench.data.io import read_h5, write_json
from rom_bench.evaluation.field_metrics import error_over_time, relative_l2
from rom_bench.evaluation.front_tracking import front_position_error
from rom_bench.evaluation.reports import write_markdown_report
from rom_bench.models.autoencoder_1d import Conv1dAutoencoder, DenseAutoencoderFallback
from rom_bench.models.latent_dynamics import LatentLinearDynamics
from rom_bench.paths import ensure_dir, resolve_path
from rom_bench.seed import seed_everything
from rom_bench.visualization.burgers_plots import plot_error_curve, plot_field_comparison, plot_spacetime
from rom_bench.visualization.latent_plots import plot_latent_trajectory


def _torch_available() -> bool:
    """Return whether PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_device(requested: str):
    """Resolve auto/cpu/cuda device for PyTorch."""
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _train_torch_conv1d(
    snapshots: np.ndarray,
    all_snapshots: np.ndarray,
    model_cfg: dict,
    training_cfg: dict,
    ckpt_dir: Path,
):
    """Train Conv1D autoencoder and return reconstruction helpers."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = _resolve_device(str(training_cfg.get("device", "auto")))
    mean = snapshots.mean(axis=0).astype(np.float32)
    scale = np.float32(snapshots.std() + 1.0e-12)
    train_x = ((snapshots - mean) / scale).astype(np.float32)
    all_x = ((all_snapshots - mean) / scale).astype(np.float32)

    model = Conv1dAutoencoder(
        nx=all_snapshots.shape[-1],
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_channels=int(model_cfg.get("hidden_channels", 16)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 1.0e-3)))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_state = None
    patience = int(training_cfg.get("patience", 50))
    epochs_without_improvement = 0
    history: list[float] = []

    for epoch in range(1, int(training_cfg.get("epochs", 500)) + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = torch.mean((pred - batch) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if epoch_loss < best_loss - 1.0e-8:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "mean": mean,
                    "scale": scale,
                    "config": {"model": model_cfg, "training": training_cfg},
                    "epoch": epoch,
                    "loss": best_loss,
                },
                ckpt_dir / "best_checkpoint.pt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "config": {"model": model_cfg, "training": training_cfg},
            "loss_history": history,
        },
        ckpt_dir / "last_checkpoint.pt",
    )

    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(all_x).to(device)
        reconstruction_norm = model(tensor).cpu().numpy()
        latent = model.encode(tensor).cpu().numpy()

    reconstruction = reconstruction_norm * scale + mean

    def decode(latent_values: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            z = torch.from_numpy(latent_values.astype(np.float32)).to(device)
            decoded = model.decode(z).cpu().numpy()
        return decoded * scale + mean

    return {
        "backend": "pytorch_conv1d",
        "model": model,
        "reconstruction": reconstruction,
        "latent": latent,
        "decode": decode,
        "best_loss": best_loss,
        "epochs_ran": len(history),
        "device": str(device),
    }


def main() -> None:
    start = time.perf_counter()
    config, _args = parse_config_args("Train Burgers autoencoder")
    if config["problem"] != "burgers":
        raise ValueError("This project now supports only the 1D Burgers equation.")
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed)
    arrays, _metadata = read_h5(config["data"]["path"])
    case_index = int(config["data"].get("case_index", 0))
    x, t, u = arrays["x"], arrays["t"], arrays["u"][case_index]
    train_end = int(config["evaluation"].get("rollout_start_index", int(0.6 * len(t))))
    model_cfg = config["model"]
    training_cfg = config["training"]
    out_root = resolve_path(config["experiment"].get("output_dir", "."))
    exp_id = config["experiment"]["name"]
    fig_dir = ensure_dir(out_root / "figures" / "autoencoder" / exp_id)
    metric_dir = ensure_dir(out_root / "metrics")
    pred_dir = ensure_dir(out_root / "predictions")
    ckpt_dir = ensure_dir(out_root / "checkpoints" / "autoencoder" / exp_id)
    report_dir = ensure_dir(out_root / "reports")
    if _torch_available() and bool(training_cfg.get("use_torch", True)):
        trained = _train_torch_conv1d(u[:train_end], u, model_cfg, training_cfg, ckpt_dir)
        reconstruction = trained["reconstruction"]
        latent = trained["latent"]
        decode = trained["decode"]
        backend = trained["backend"]
        best_loss = float(trained["best_loss"])
        epochs_ran = int(trained["epochs_ran"])
        device = str(trained["device"])
    else:
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
        decode = model.decode
        backend = "numpy_dense_fallback"
        best_loss = float("nan")
        epochs_ran = int(training_cfg.get("epochs", 500))
        device = "cpu"
        np.savez(ckpt_dir / "best_checkpoint.npz", **model.__dict__)
        np.savez(ckpt_dir / "last_checkpoint.npz", **model.__dict__)
    latent_rollout = LatentLinearDynamics().fit(latent[:train_end]).rollout(latent[0], len(t))
    rollout = decode(latent_rollout)
    rec_err = error_over_time(u, reconstruction)
    roll_err = error_over_time(u, rollout)
    front_err = front_position_error(x, u, rollout, method=config["evaluation"].get("front_method", "max_gradient"))
    np.savez(pred_dir / f"{exp_id}_predictions.npz", x=x, t=t, truth=u, reconstruction=reconstruction, rollout=rollout, latent=latent)
    plot_field_comparison(x, u[-1], reconstruction[-1], "AE final reconstruction", fig_dir / "autoencoder_final_reconstruction")
    plot_field_comparison(x, u[-1], rollout[-1], "AE latent rollout", fig_dir / "autoencoder_latent_rollout_final")
    plot_error_curve(t, rec_err, "AE reconstruction error", fig_dir / "reconstruction_error_vs_time")
    plot_error_curve(t, roll_err, "AE latent rollout error", fig_dir / "rollout_error_vs_time")
    plot_spacetime(x, t, abs(reconstruction - u), "AE reconstruction absolute error", fig_dir / "burgers_spacetime_error")
    plot_latent_trajectory(t, latent, "Autoencoder latent trajectory", fig_dir / "latent_trajectory")
    metrics = {
        "method": "autoencoder",
        "problem": "burgers",
        "backend": backend,
        "latent_dim": int(model_cfg["latent_dim"]),
        "reconstruction_relative_l2": relative_l2(u, reconstruction),
        "rollout_relative_l2": float(np.mean(roll_err[train_end:])),
        "final_rollout_error": float(roll_err[-1]),
        "front_position_mae": float(front_err.mean()),
        "front_speed_error": float("nan"),
        "max_gradient_error": float(front_err.max()),
        "best_training_loss": best_loss,
        "epochs_ran": epochs_ran,
        "device": device,
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
                f"backend = {backend}",
                "AE reconstruction이 좋아도 latent linear dynamics가 정확하다는 뜻은 아님",
                "Conv1D AE는 field reconstruction 모델이고, latent rollout은 별도 dynamics 모델의 성능에 좌우됨",
            ],
            "Figures": [str(p) for p in fig_dir.glob("*.png")],
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
