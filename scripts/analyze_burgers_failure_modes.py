"""Compare Burgers smooth/shock cases and dimension sweeps on the public PDEBench subset."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rom_bench.data.io import read_h5, write_json
from rom_bench.evaluation.field_metrics import error_over_time, relative_l2, spatial_gradient_error
from rom_bench.evaluation.front_tracking import front_position_error, front_positions
from rom_bench.models.autoencoder_1d import Conv1dAutoencoder
from rom_bench.models.dmd import DMDModel
from rom_bench.models.pod import PODModel
from rom_bench.paths import ensure_dir, resolve_path
from rom_bench.seed import seed_everything


DATA_PATH = "data/processed/burgers/pdebench_burgers_nu0.01_subset.h5"
EXP_ID = "pdebench_smooth_shock_failure_modes"
RANKS = [2, 4, 8, 16, 32]
LATENT_DIMS = [2, 4, 8, 16, 32]
TRAIN_END = 60
SEED = 42


def max_gradient_series(x: np.ndarray, snapshots: np.ndarray) -> np.ndarray:
    """Return max |du/dx| for each snapshot."""
    return np.asarray([np.max(np.abs(np.gradient(u, x))) for u in snapshots], dtype=float)


def choose_smooth_and_shock_cases(x: np.ndarray, u: np.ndarray, test_indices: np.ndarray) -> list[dict[str, Any]]:
    """Choose one smooth-like and one shock-like test case by gradient strength."""
    rows = []
    for case_index in test_indices.astype(int):
        gradients = max_gradient_series(x, u[case_index])
        rows.append(
            {
                "case_index": int(case_index),
                "mean_max_gradient": float(np.mean(gradients)),
                "max_gradient": float(np.max(gradients)),
                "time_of_max_gradient_index": int(np.argmax(gradients)),
            }
        )
    smooth = min(rows, key=lambda row: row["mean_max_gradient"])
    shock = max(rows, key=lambda row: row["mean_max_gradient"])
    smooth = {**smooth, "regime": "smooth_like"}
    shock = {**shock, "regime": "shock_like"}
    return [smooth, shock]


def train_conv1d_ae(
    train_snapshots: np.ndarray,
    all_snapshots: np.ndarray,
    latent_dim: int,
    *,
    epochs: int = 120,
    patience: int = 25,
    hidden_channels: int = 16,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train a small Conv1D AE on train_snapshots and reconstruct all_snapshots."""
    seed_everything(seed + latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train_snapshots.mean(axis=0).astype(np.float32)
    scale = np.float32(train_snapshots.std() + 1.0e-12)
    train_x = ((train_snapshots - mean) / scale).astype(np.float32)
    all_x = ((all_snapshots - mean) / scale).astype(np.float32)
    model = Conv1dAutoencoder(
        nx=all_snapshots.shape[-1],
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x)), batch_size=32, shuffle=True)
    best_state = None
    best_loss = float("inf")
    wait = 0
    for _epoch in range(epochs):
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
        if epoch_loss < best_loss - 1.0e-8:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(all_x).to(device)
        reconstruction_norm = model(tensor).cpu().numpy()
        latent = model.encode(tensor).cpu().numpy()
    return reconstruction_norm * scale + mean, latent, best_loss


def local_front_error(
    x: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    front_pos: float,
    width: float = 0.08,
) -> float:
    """Relative L2 error around a front-centered local window."""
    mask = np.abs(x - front_pos) <= width
    if not np.any(mask):
        return float("nan")
    return relative_l2(truth[..., mask], pred[..., mask])


def metric_row(
    *,
    regime: str,
    case_index: int,
    method: str,
    dim_name: str,
    dim: int,
    x: np.ndarray,
    t: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    train_end: int,
    training_time: float,
) -> dict[str, Any]:
    """Build a common metric row."""
    err = error_over_time(truth, pred)
    front_err = front_position_error(x, truth, pred, method="max_gradient")
    true_front = front_positions(x, truth, method="max_gradient")
    front_time = int(np.argmax(max_gradient_series(x, truth)))
    return {
        "regime": regime,
        "case_index": case_index,
        "method": method,
        "dimension_name": dim_name,
        "dimension": dim,
        "train_relative_l2": float(np.mean(err[:train_end])),
        "test_relative_l2": float(np.mean(err[train_end:])),
        "all_relative_l2": relative_l2(truth, pred),
        "final_relative_l2": float(err[-1]),
        "front_position_mae": float(np.mean(front_err)),
        "front_position_mae_report": "N/A" if regime == "smooth_like" else float(np.mean(front_err)),
        "front_window_relative_l2": local_front_error(x, truth[front_time], pred[front_time], true_front[front_time]),
        "spatial_gradient_error": spatial_gradient_error(truth, pred, x),
        "time_of_max_gradient": float(t[front_time]),
        "training_time": training_time,
    }


def save_case_selection_plot(rows: list[dict[str, Any]], selected: list[dict[str, Any]], fig_dir: Path) -> None:
    """Plot sharpness scores for test cases."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    case_indices = [row["case_index"] for row in rows]
    mean_grad = [row["mean_max_gradient"] for row in rows]
    ax.plot(case_indices, mean_grad, marker="o", label="test cases")
    for item in selected:
        ax.scatter([item["case_index"]], [item["mean_max_gradient"]], s=90, label=item["regime"])
        ax.annotate(item["regime"], (item["case_index"], item["mean_max_gradient"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("case index")
    ax.set_ylabel("mean max |du/dx|")
    ax.set_title("Smooth-like and shock-like case selection")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(fig_dir / "case_sharpness_selection.png", dpi=300)
    fig.savefig(fig_dir / "case_sharpness_selection.pdf")
    plt.close(fig)


def save_true_spacetime_pair(x: np.ndarray, t: np.ndarray, u: np.ndarray, cases: list[dict[str, Any]], fig_dir: Path) -> None:
    """Save smooth/shock true space-time comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for ax, item in zip(axes, cases):
        im = ax.pcolormesh(x, t, u[item["case_index"]], shading="auto")
        ax.set_title(f"{item['regime']} case {item['case_index']}")
        ax.set_xlabel("x")
        ax.set_ylabel("time")
        fig.colorbar(im, ax=ax)
    fig.savefig(fig_dir / "smooth_vs_shock_true_spacetime.png", dpi=300)
    fig.savefig(fig_dir / "smooth_vs_shock_true_spacetime.pdf")
    plt.close(fig)


def save_front_overlay(
    x: np.ndarray,
    t: np.ndarray,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    item: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Save local front overlay and local absolute-error plot."""
    front_time = int(np.argmax(max_gradient_series(x, truth)))
    true_front = front_positions(x, truth, method="max_gradient")[front_time]
    width = 0.10
    mask = np.abs(x - true_front) <= width
    if np.sum(mask) < 8:
        mask = slice(None)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    axes[0].plot(x[mask], truth[front_time][mask], linewidth=2.0, label="truth")
    for name, pred in predictions.items():
        axes[0].plot(x[mask], pred[front_time][mask], "--", label=name)
    axes[0].axvline(true_front, color="0.25", linestyle=":", linewidth=1.0, label="true front")
    axes[0].set_title(f"{item['regime']} case {item['case_index']} front overlay, t={t[front_time]:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend(fontsize=8)
    for name, pred in predictions.items():
        axes[1].plot(x[mask], np.abs(pred[front_time][mask] - truth[front_time][mask]), label=name)
    axes[1].set_title("local absolute error around front")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|error|")
    axes[1].legend(fontsize=8)
    path = fig_dir / f"front_overlay_{item['regime']}_case{item['case_index']}.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def save_sweep_plot(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    """Save POD-rank and AE-latent sweep plots."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    for regime in sorted({row["regime"] for row in rows}):
        for method, ax in [("POD", axes[0]), ("AE", axes[1])]:
            subset = [row for row in rows if row["regime"] == regime and row["method"] == method]
            subset = sorted(subset, key=lambda row: row["dimension"])
            ax.plot(
                [row["dimension"] for row in subset],
                [row["test_relative_l2"] for row in subset],
                marker="o",
                label=regime,
            )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("rank" if method == "POD" else "latent dimension")
            ax.set_ylabel("post-split relative L2")
            ax.set_title(f"{method} dimension sweep")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
    fig.savefig(fig_dir / "pod_rank_ae_latent_sweep.png", dpi=300)
    fig.savefig(fig_dir / "pod_rank_ae_latent_sweep.pdf")
    plt.close(fig)


def save_same_training_bar(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    """Plot same-training-condition reconstruction metrics for dim 8."""
    selected = [row for row in rows if row["dimension"] == 8 and row["method"] in {"POD", "AE"}]
    labels = [f"{row['regime']}\n{row['method']}" for row in selected]
    values = [row["test_relative_l2"] for row in selected]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(labels, values)
    ax.set_ylabel("post-split relative L2")
    ax.set_title("Same temporal training snapshots, dimension 8")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(fig_dir / "same_training_reconstruction_comparison.png", dpi=300)
    fig.savefig(fig_dir / "same_training_reconstruction_comparison.pdf")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the smooth/shock and dimension-sweep comparison."""
    seed_everything(SEED)
    arrays, metadata = read_h5(DATA_PATH)
    x = arrays["x"]
    t = arrays["t"]
    u = arrays["u"]
    test_indices = arrays["split/test_indices"]
    out_root = resolve_path(".")
    fig_dir = ensure_dir(out_root / "figures" / "failure_modes" / EXP_ID)
    metric_dir = ensure_dir(out_root / "metrics")
    pred_dir = ensure_dir(out_root / "predictions" / "failure_modes" / EXP_ID)

    all_case_rows = []
    for case_index in test_indices.astype(int):
        gradients = max_gradient_series(x, u[case_index])
        all_case_rows.append(
            {
                "case_index": int(case_index),
                "mean_max_gradient": float(np.mean(gradients)),
                "max_gradient": float(np.max(gradients)),
                "time_of_max_gradient_index": int(np.argmax(gradients)),
            }
        )
    selected_cases = choose_smooth_and_shock_cases(x, u, test_indices)
    save_case_selection_plot(all_case_rows, selected_cases, fig_dir)
    save_true_spacetime_pair(x, t, u, selected_cases, fig_dir)

    metric_rows: list[dict[str, Any]] = []
    overlay_predictions: dict[str, dict[str, np.ndarray]] = {}

    for item in selected_cases:
        case_index = int(item["case_index"])
        regime = str(item["regime"])
        truth = u[case_index]
        train = truth[:TRAIN_END]
        overlay_predictions[regime] = {}

        for rank in RANKS:
            start = time.perf_counter()
            pod = PODModel.fit(train, rank=rank, subtract_mean=True)
            pred = pod.reconstruct(truth)
            elapsed = time.perf_counter() - start
            metric_rows.append(
                metric_row(
                    regime=regime,
                    case_index=case_index,
                    method="POD",
                    dim_name="rank",
                    dim=rank,
                    x=x,
                    t=t,
                    truth=truth,
                    pred=pred,
                    train_end=TRAIN_END,
                    training_time=elapsed,
                )
            )
            if rank == 8:
                overlay_predictions[regime]["POD rank 8"] = pred

        start = time.perf_counter()
        dmd = DMDModel.fit(train, dt=float(np.mean(np.diff(t))), rank=8, filter_unstable=False)
        dmd_pred = dmd.predict(len(t))
        elapsed = time.perf_counter() - start
        metric_rows.append(
            metric_row(
                regime=regime,
                case_index=case_index,
                method="DMD",
                dim_name="rank",
                dim=8,
                x=x,
                t=t,
                truth=truth,
                pred=dmd_pred,
                train_end=TRAIN_END,
                training_time=elapsed,
            )
        )
        overlay_predictions[regime]["DMD rank 8"] = dmd_pred

        for latent_dim in LATENT_DIMS:
            start = time.perf_counter()
            ae_pred, latent, best_loss = train_conv1d_ae(train, truth, latent_dim)
            elapsed = time.perf_counter() - start
            row = metric_row(
                regime=regime,
                case_index=case_index,
                method="AE",
                dim_name="latent_dim",
                dim=latent_dim,
                x=x,
                t=t,
                truth=truth,
                pred=ae_pred,
                train_end=TRAIN_END,
                training_time=elapsed,
            )
            row["best_training_loss"] = float(best_loss)
            metric_rows.append(row)
            np.savez(
                pred_dir / f"{regime}_case{case_index}_ae_latent{latent_dim}.npz",
                x=x,
                t=t,
                truth=truth,
                reconstruction=ae_pred,
                latent=latent,
            )
            if latent_dim == 8:
                overlay_predictions[regime]["AE latent 8"] = ae_pred

        save_front_overlay(x, t, truth, overlay_predictions[regime], item, fig_dir)

    write_csv(metric_dir / "pdebench_failure_mode_sweep_metrics.csv", metric_rows)
    write_csv(metric_dir / "pdebench_failure_mode_case_selection.csv", all_case_rows)
    write_json(
        metric_dir / "pdebench_failure_mode_summary.json",
        {
            "experiment_id": EXP_ID,
            "data_path": DATA_PATH,
            "metadata": metadata,
            "train_end": TRAIN_END,
            "ranks": RANKS,
            "latent_dims": LATENT_DIMS,
            "selected_cases": selected_cases,
            "interpretation_scope": "Smooth/shock and dimension-sweep comparison on the same PDEBench dataset; no method ranking is intended.",
        },
    )
    save_sweep_plot(metric_rows, fig_dir)
    save_same_training_bar(metric_rows, fig_dir)
    print(json.dumps({"selected_cases": selected_cases, "n_metric_rows": len(metric_rows)}, indent=2))


if __name__ == "__main__":
    main()
