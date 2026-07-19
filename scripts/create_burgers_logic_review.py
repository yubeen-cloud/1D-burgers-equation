"""Create a public-dataset-only PDF review for Burgers ROM experiments."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

from rom_bench.evaluation.front_tracking import front_positions
from rom_bench.paths import resolve_path


PAGE_SIZE = (8.27, 11.69)


def setup_korean_font() -> None:
    """Use a Korean-capable font on Windows when available."""
    for path in [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
    ]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def read_json(path: str) -> dict[str, Any]:
    """Read JSON relative to project root."""
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def read_csv_rows(path: str) -> list[dict[str, str]]:
    """Read CSV rows relative to project root."""
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: Any) -> str:
    """Format scalar values for report tables."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "N/A" if value is None else str(value)
    if np.isnan(x):
        return "N/A"
    if abs(x) >= 1.0e3 or (abs(x) < 1.0e-2 and x != 0.0):
        return f"{x:.3e}"
    return f"{x:.4f}"


def wrap_index_list(values: list[int], chunk_size: int = 7) -> str:
    """Wrap long random-index lists inside narrow report table cells."""
    return "\n".join(
        ", ".join(str(value) for value in values[start : start + chunk_size])
        for start in range(0, len(values), chunk_size)
    )


def available_pdf_path(path: Path) -> Path:
    """Return a writable PDF path, adding a suffix if existing files are locked."""
    candidates = [path]
    candidates.extend(path.with_name(f"burgers_logic_review_public_{i}.pdf") for i in range(1, 20))
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with candidate.open("ab"):
                pass
            return candidate
        except PermissionError:
            continue
    raise PermissionError(f"No writable PDF path found in {path.parent}")


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    """Add wrapped text pages."""
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.055, 0.965, title, fontsize=17, fontweight="bold", va="top")
    y = 0.915
    for paragraph in paragraphs:
        if paragraph == "":
            y -= 0.020
            continue
        indent = "   " if paragraph.startswith("- ") else ""
        for line in textwrap.wrap(paragraph, width=86, break_long_words=False, subsequent_indent=indent):
            ax.text(0.065, y, line, fontsize=9.6, va="top")
            y -= 0.024
            if y < 0.075:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=PAGE_SIZE)
                ax = fig.add_subplot(111)
                ax.axis("off")
                y = 0.955
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    columns: list[str],
    rows: list[list[str]],
    note: str | None = None,
) -> None:
    """Add a compact table page."""
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.055, 0.965, title, fontsize=16, fontweight="bold", va="top")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="upper left",
        cellLoc="left",
        bbox=[0.055, 0.23 if note else 0.10, 0.89, 0.64 if note else 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.25)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(fontweight="bold")
    if note:
        ax.text(0.055, 0.17, textwrap.fill(note, width=92), fontsize=9.0, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_wrapped_kv_page(
    pdf: PdfPages,
    title: str,
    rows: list[tuple[str, str, str]],
    note: str | None = None,
) -> None:
    """Add key/value/description rows with wrapped long text."""
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.055, 0.965, title, fontsize=16, fontweight="bold", va="top")
    y = 0.91
    row_gap = 0.014
    line_step = 0.022

    for key, value, description in rows:
        key_lines = textwrap.wrap(key, width=18, break_long_words=False) or [key]
        value_lines = textwrap.wrap(value, width=52, break_long_words=False) or [value]
        desc_lines = textwrap.wrap(description, width=34, break_long_words=False) or [description]
        line_count = max(len(key_lines), len(value_lines), len(desc_lines))
        row_height = 0.034 + line_count * line_step
        if y - row_height < 0.11:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=PAGE_SIZE)
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.055, 0.965, f"{title} 계속", fontsize=16, fontweight="bold", va="top")
            y = 0.91

        ax.add_patch(
            plt.Rectangle(
                (0.055, y - row_height),
                0.89,
                row_height,
                facecolor="#f8f8f8",
                edgecolor="#d0d0d0",
                linewidth=0.7,
            )
        )
        x_positions = [0.072, 0.25, 0.68]
        for lines, xpos in zip([key_lines, value_lines, desc_lines], x_positions):
            cursor = y - 0.018
            for line in lines:
                ax.text(xpos, cursor, line, fontsize=8.3, va="top")
                cursor -= line_step
        y -= row_height + row_gap

    if note:
        note_lines = textwrap.wrap(note, width=90, break_long_words=False)
        if y - 0.03 * len(note_lines) < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=PAGE_SIZE)
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.94
        ax.text(0.065, y, "참고", fontsize=9.2, fontweight="bold", va="top")
        y -= 0.024
        for line in note_lines:
            ax.text(0.065, y, line, fontsize=8.6, color="0.25", va="top")
            y -= 0.022

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_record_pages(
    pdf: PdfPages,
    title: str,
    records: list[tuple[str, list[tuple[str, str]]]],
    note: str | None = None,
) -> None:
    """Add records as full-width cards to avoid cramped table cells."""
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.055, 0.965, title, fontsize=16, fontweight="bold", va="top")
    y = 0.91
    line_step = 0.022
    row_gap = 0.016

    def new_page(continued: bool = True) -> None:
        nonlocal fig, ax, y
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        fig = plt.figure(figsize=PAGE_SIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        suffix = " 계속" if continued else ""
        ax.text(0.055, 0.965, f"{title}{suffix}", fontsize=16, fontweight="bold", va="top")
        y = 0.91

    for record_title, fields in records:
        wrapped_fields: list[tuple[str, list[str]]] = []
        total_lines = 1
        for key, value in fields:
            text = f"{key}: {value}"
            lines = textwrap.wrap(text, width=92, break_long_words=False) or [text]
            wrapped_fields.append((key, lines))
            total_lines += len(lines)
        card_height = 0.046 + total_lines * line_step
        if y - card_height < 0.10:
            new_page()

        ax.add_patch(
            plt.Rectangle(
                (0.055, y - card_height),
                0.89,
                card_height,
                facecolor="#f8f8f8",
                edgecolor="#cfcfcf",
                linewidth=0.8,
            )
        )
        cursor = y - 0.018
        ax.text(0.075, cursor, record_title, fontsize=9.2, fontweight="bold", va="top")
        cursor -= 0.028
        for _key, lines in wrapped_fields:
            for line in lines:
                ax.text(0.075, cursor, line, fontsize=8.5, va="top")
                cursor -= line_step
        y -= card_height + row_gap

    if note:
        note_lines = textwrap.wrap(note, width=92, break_long_words=False)
        note_height = 0.035 + len(note_lines) * line_step
        if y - note_height < 0.08:
            new_page()
        ax.text(0.065, y, "참고", fontsize=9.4, fontweight="bold", va="top")
        y -= 0.026
        for line in note_lines:
            ax.text(0.065, y, line, fontsize=8.6, color="0.25", va="top")
            y -= line_step

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: str, caption: str) -> None:
    """Add a generated PNG figure when it exists."""
    path = resolve_path(image_path)
    if not path.exists():
        return
    image = plt.imread(path)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.05, 0.035, textwrap.fill(caption, width=135), fontsize=9.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_method_figures(
    pdf: PdfPages,
    method_title: str,
    figure_dir: str,
    ordered_files: list[tuple[str, str]],
) -> None:
    """Append all selected figures for one method in a fixed order."""
    add_text_page(
        pdf,
        f"{method_title} Figures",
        [
            f"아래 그림들은 public PDEBench subset에 {method_title} 방법을 적용해 생성한 결과다.",
            "각 그림은 원본 PNG 파일을 PDF 뒤쪽에 한 장씩 붙인 것이다.",
        ],
    )
    for filename, caption in ordered_files:
        stem = Path(filename).stem.replace("_", " ").title()
        add_image_page(
            pdf,
            f"{method_title}: {stem}",
            f"{figure_dir}/{filename}",
            caption,
        )


def load_public_dataset_summary() -> dict[str, Any]:
    """Read public PDEBench subset facts."""
    path = resolve_path("data/processed/burgers/pdebench_burgers_nu0.01_subset.h5")
    with h5py.File(path, "r") as h5:
        metadata = json.loads(h5.attrs["metadata_json"])
        return {
            "path": str(path),
            "u_shape": tuple(h5["u"].shape),
            "x_shape": tuple(h5["x"].shape),
            "t_shape": tuple(h5["t"].shape),
            "train": h5["split/train_indices"][:].tolist(),
            "val": h5["split/val_indices"][:].tolist(),
            "test": h5["split/test_indices"][:].tolist(),
            "source": metadata.get("source"),
            "source_file": metadata.get("source_file"),
            "source_url": metadata.get("source_url"),
            "source_tensor_shape": metadata.get("source_tensor_shape"),
            "subset": metadata.get("subset"),
        }


def _max_gradient_series(x: np.ndarray, snapshots: np.ndarray) -> np.ndarray:
    """Return max |du/dx| for each snapshot."""
    return np.asarray([np.max(np.abs(np.gradient(u, x))) for u in snapshots], dtype=float)


def compute_logic_audit() -> dict[str, Any]:
    """Compute concrete audit values for the public Burgers review."""
    processed_path = resolve_path("data/processed/burgers/pdebench_burgers_nu0.01_subset.h5")
    source_path = resolve_path("data/external/pdebench/1D_Burgers_Sols_Nu0.01.hdf5")
    ae_pred_path = resolve_path("predictions/pdebench_burgers_ae_latent8_predictions.npz")
    dmd_cfg = yaml.safe_load(resolve_path("configs/burgers/pdebench_dmd.yaml").read_text(encoding="utf-8"))
    ae_cfg = yaml.safe_load(resolve_path("configs/burgers/pdebench_autoencoder.yaml").read_text(encoding="utf-8"))

    with h5py.File(processed_path, "r") as h5:
        x = np.asarray(h5["x"], dtype=float)
        u = np.asarray(h5["u"], dtype=float)
        train_indices = np.asarray(h5["split/train_indices"], dtype=int)
        val_indices = np.asarray(h5["split/val_indices"], dtype=int)
        test_indices = np.asarray(h5["split/test_indices"], dtype=int)
        source_case_indices = np.asarray(h5["source_case_indices"], dtype=int)
        metadata = json.loads(h5.attrs["metadata_json"])

    case_index = int(ae_cfg["data"]["case_index"])
    ae_train_end = int(ae_cfg["evaluation"]["rollout_start_index"])
    dmd_train_end = int(dmd_cfg["evaluation"]["rollout_start_index"])
    dx_128 = float(np.mean(np.diff(x)))
    true_front = front_positions(x, u[case_index], method="max_gradient")
    front_steps = np.abs(np.diff(true_front))
    large_jump_threshold = 4.0 * dx_128

    latent = np.load(ae_pred_path)["latent"]
    z1 = latent[:ae_train_end][:-1].T
    z2 = latent[:ae_train_end][1:].T
    latent_operator = z2 @ np.linalg.pinv(z1)
    latent_eigs = np.linalg.eigvals(latent_operator)
    spectral_radius = float(np.max(np.abs(latent_eigs)))

    resolution_rows: list[dict[str, Any]] = []
    if source_path.exists():
        with h5py.File(source_path, "r") as h5:
            tensor = h5["tensor"]
            original_x = np.asarray(h5["x-coordinate"], dtype=float)
            original_t = np.asarray(h5["t-coordinate"], dtype=float) if "t-coordinate" in h5 else np.arange(tensor.shape[1])
            original_case = int(source_case_indices[case_index])
            time_stride = int(metadata["subset"]["time_stride"])
            time_slice = slice(None, None, time_stride)
            for nx_target, stride in [(128, 8), (256, 4), (1024, 1)]:
                if len(tensor.shape) == 4:
                    snapshots = np.asarray(tensor[original_case, time_slice, slice(None, None, stride), 0], dtype=float)
                else:
                    snapshots = np.asarray(tensor[original_case, time_slice, slice(None, None, stride)], dtype=float)
                x_res = original_x[::stride]
                grad = _max_gradient_series(x_res, snapshots)
                pos = front_positions(x_res, snapshots, method="max_gradient")
                steps = np.abs(np.diff(pos))
                dx = float(np.mean(np.diff(x_res)))
                resolution_rows.append(
                    {
                        "resolution": int(nx_target),
                        "space_stride": int(stride),
                        "nx_actual": int(len(x_res)),
                        "nt_actual": int(len(original_t[time_slice])),
                        "dx": dx,
                        "mean_max_abs_gradient": float(np.mean(grad)),
                        "max_abs_gradient": float(np.max(grad)),
                        "median_front_step": float(np.median(steps)) if len(steps) else 0.0,
                        "max_front_step": float(np.max(steps)) if len(steps) else 0.0,
                        "large_front_jump_count": int(np.sum(steps > 4.0 * dx)),
                    }
                )

    if resolution_rows:
        full_mean = next(row["mean_max_abs_gradient"] for row in resolution_rows if row["resolution"] == 1024)
        full_max = next(row["max_abs_gradient"] for row in resolution_rows if row["resolution"] == 1024)
        for row in resolution_rows:
            row["mean_gradient_ratio_vs_1024"] = float(row["mean_max_abs_gradient"] / full_mean)
            row["max_gradient_ratio_vs_1024"] = float(row["max_abs_gradient"] / full_max)

        metric_dir = resolve_path("metrics")
        metric_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metric_dir / "pdebench_resolution_gradient_comparison.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(resolution_rows[0].keys()))
            writer.writeheader()
            writer.writerows(resolution_rows)

        fig_dir = resolve_path("figures/data/pdebench_resolution_audit")
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
        axes[0].plot(
            [row["resolution"] for row in resolution_rows],
            [row["mean_max_abs_gradient"] for row in resolution_rows],
            marker="o",
        )
        axes[0].set_xlabel("spatial resolution nx")
        axes[0].set_ylabel("mean max |du/dx|")
        axes[0].set_title("Front sharpness vs resolution")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(
            [row["resolution"] for row in resolution_rows],
            [row["mean_gradient_ratio_vs_1024"] for row in resolution_rows],
            marker="o",
            color="#c44e52",
        )
        axes[1].axhline(1.0, color="0.3", linestyle="--", linewidth=1.0)
        axes[1].set_xlabel("spatial resolution nx")
        axes[1].set_ylabel("ratio vs nx=1024")
        axes[1].set_title("Gradient retained after downsampling")
        axes[1].grid(True, alpha=0.3)
        fig.savefig(fig_dir / "resolution_gradient_comparison.png", dpi=300)
        fig.savefig(fig_dir / "resolution_gradient_comparison.pdf")
        plt.close(fig)

    audit = {
        "pod_snapshot_matrix_orientation": "snapshots are rows: [n_samples, n_features]",
        "pod_spatial_modes_source": "np.linalg.svd(centered) returns U, S, Vh; code stores modes=Vh[:rank], so spatial modes are rows of Vh, not columns of U",
        "case_index": case_index,
        "case_index_split": "train" if case_index in train_indices else "val" if case_index in val_indices else "test" if case_index in test_indices else "unknown",
        "ae_uses_hdf5_case_train_split_only": False,
        "ae_training_snapshots": f"case {case_index}, time indices 0..{ae_train_end - 1}",
        "ae_selected_case_in_temporal_training": True,
        "ae_future_snapshots_in_training": False,
        "latent_dynamics_fit_snapshots": f"latent time indices 0..{ae_train_end - 1}",
        "latent_dynamics_future_snapshots_used_for_fit": False,
        "dmd_train_snapshots": f"case {case_index}, time indices 0..{dmd_train_end - 1}",
        "dmd_ae_temporal_split_identical": bool(
            case_index == int(dmd_cfg["data"]["case_index"]) and ae_train_end == dmd_train_end
        ),
        "latent_linear_operator_spectral_radius": spectral_radius,
        "latent_linear_operator_max_eigenvalue_abs": spectral_radius,
        "front_detector_method": "max_gradient",
        "front_detector_dx_128": dx_128,
        "front_detector_max_jump": float(np.max(front_steps)) if len(front_steps) else 0.0,
        "front_detector_large_jump_threshold": large_jump_threshold,
        "front_detector_large_jump_count": int(np.sum(front_steps > large_jump_threshold)),
        "front_detector_tracks_same_front_likely": bool(np.sum(front_steps > large_jump_threshold) == 0),
        "resolution_rows": resolution_rows,
    }
    metric_dir = resolve_path("metrics")
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / "burgers_logic_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    setup_korean_font()
    out = available_pdf_path(resolve_path("reports/burgers_logic_review.pdf"))
    data = load_public_dataset_summary()
    pod = read_json("metrics/pdebench_burgers_pod_rank8_metrics.json")
    dmd = read_json("metrics/pdebench_burgers_dmd_rank8_metrics.json")
    ae = read_json("metrics/pdebench_burgers_ae_latent8_metrics.json")
    audit = compute_logic_audit()
    failure_summary_path = resolve_path("metrics/pdebench_failure_mode_summary.json")
    failure_metrics_path = resolve_path("metrics/pdebench_failure_mode_sweep_metrics.csv")
    failure_summary = json.loads(failure_summary_path.read_text(encoding="utf-8")) if failure_summary_path.exists() else {}
    failure_rows = read_csv_rows(str(failure_metrics_path)) if failure_metrics_path.exists() else []
    aggregate_path = resolve_path("metrics/pdebench_test_aggregate_summary.json")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    case_index = int(audit["case_index"])
    selected_by_regime = {
        str(row["regime"]): int(row["case_index"])
        for row in failure_summary.get("selected_cases", [])
    }
    smooth_case = selected_by_regime.get("smooth_like", case_index)
    shock_case = selected_by_regime.get("shock_like", case_index)
    comparison_rows = [
        row
        for row in read_csv_rows("metrics/burgers_model_comparison.csv")
        if row.get("experiment_name", "").startswith("pdebench_burgers")
    ]

    with PdfPages(out) as pdf:
        pdf.infodict()["Title"] = "PDEBench Burgers ROM Review"
        pdf.infodict()["Author"] = "Codex"

        add_text_page(
            pdf,
            "PDEBench 공개 Burgers 데이터 ROM 리뷰",
            [
                "이 문서는 기존 자체 생성 Burgers 데이터 결과를 제외하고, 공개 PDEBench 1D Burgers 데이터셋으로 수행한 ROM 결과만 정리한다.",
                "",
                "사용한 공개 데이터는 PDEBench의 1D_Burgers_Sols_Nu0.01.hdf5이다. 원본 HDF5에서 동일한 subset을 추출한 뒤, 그 같은 processed HDF5 파일에 POD reconstruction, DMD-based prediction, Conv1D Autoencoder reconstruction 및 latent linear rollout을 적용했다.",
                "",
                "핵심 결론은 다음과 같다.",
                f"- 세 방법 모두 같은 public subset과 같은 held-out case_index={case_index}를 사용했다.",
                "- POD는 시간 예측이 아니라 snapshot projection reconstruction baseline이다.",
                "- DMD는 linear rollout prediction이므로 시간이 지나면서 prediction error가 증가한다.",
                "- Conv1D AE는 reconstruction과 latent linear rollout에서 서로 다른 오차 양상을 보였다.",
            ],
        )

        add_record_pages(
            pdf,
            "1. 공개 데이터셋 출처",
            [
                (
                    "PDEBench 원본 데이터",
                    [
                        ("공개 데이터셋", str(data["source"])),
                        ("원본 파일", "1D_Burgers_Sols_Nu0.01.hdf5"),
                        ("원본 URL", str(data["source_url"])),
                        ("원본 tensor shape", str(data["source_tensor_shape"])),
                    ],
                ),
                (
                    "프로젝트에서 사용한 subset",
                    [
                        ("원본 저장 위치", "data/external/pdebench/1D_Burgers_Sols_Nu0.01.hdf5"),
                        ("processed subset", "data/processed/burgers/pdebench_burgers_nu0.01_subset.h5"),
                        ("subset shape", str(data["u_shape"])),
                        ("subset rule", str(data["subset"])),
                    ],
                ),
            ],
            note="긴 URL과 파일 경로는 전체 폭을 쓰는 카드형 레이아웃으로 내려 써서 잘리지 않도록 처리했다.",
        )

        add_record_pages(
            pdf,
            "2. 세 방법에 같은 데이터가 적용되었는가?",
            [
                (
                    "POD",
                    [
                        ("config", "configs/burgers/pdebench_pod.yaml"),
                        ("data path", "data/processed/burgers/pdebench_burgers_nu0.01_subset.h5"),
                        ("case_index", str(case_index)),
                        ("rank", "8"),
                        ("동일 데이터 적용", "yes"),
                    ],
                ),
                (
                    "DMD",
                    [
                        ("config", "configs/burgers/pdebench_dmd.yaml"),
                        ("data path", "data/processed/burgers/pdebench_burgers_nu0.01_subset.h5"),
                        ("case_index", str(case_index)),
                        ("rank", "8"),
                        ("동일 데이터 적용", "yes"),
                    ],
                ),
                (
                    "Conv1D AE",
                    [
                        ("config", "configs/burgers/pdebench_autoencoder.yaml"),
                        ("data path", "data/processed/burgers/pdebench_burgers_nu0.01_subset.h5"),
                        ("case_index", str(case_index)),
                        ("latent_dim", "8"),
                        ("동일 데이터 적용", "yes"),
                    ],
                ),
            ],
            note=(
                "세 방법은 같은 processed HDF5와 같은 평가 case를 쓴다. 단, 학습/평가 의미는 다르다. "
                f"POD는 train split 전체로 basis를 만들고 case {case_index}를 projection reconstruction한다. "
                f"DMD와 AE는 case {case_index} 내부 시간 구간에서 rollout 또는 latent dynamics를 평가한다."
            ),
        )

        add_table_page(
            pdf,
            "3. PDEBench public subset split",
            ["split", "case indices", "개수"],
            [
                ["train", wrap_index_list(data["train"]), str(len(data["train"]))],
                ["validation", wrap_index_list(data["val"]), str(len(data["val"]))],
                ["test", wrap_index_list(data["test"]), str(len(data["test"]))],
            ],
            note=(
                "64개 source trajectory는 seed 42로 비복원 무작위 추출했고 local split도 seed 42 permutation으로 만들었다. "
                f"POD/DMD/AE config의 case_index={case_index}는 test split에 포함된 held-out case이다."
            ),
        )

        add_table_page(
            pdf,
            f"4. 기본 대표 결과표: public-only case {case_index}",
            ["method", "rank/latent", "reconstruction L2", "rollout L2", "final error", "front MAE"],
            [
                [
                    "POD",
                    str(pod["rank"]),
                    fmt(pod["reconstruction_relative_l2"]),
                    "N/A",
                    fmt(pod["final_rollout_error"]),
                    fmt(pod["front_position_mae"]),
                ],
                [
                    "DMD",
                    str(dmd["rank"]),
                    fmt(dmd["reconstruction_relative_l2"]),
                    fmt(dmd["rollout_relative_l2"]),
                    fmt(dmd["final_rollout_error"]),
                    fmt(dmd["front_position_mae"]),
                ],
                [
                    "Conv1D AE",
                    str(ae["latent_dim"]),
                    fmt(ae["reconstruction_relative_l2"]),
                    fmt(ae["rollout_relative_l2"]),
                    fmt(ae["final_rollout_error"]),
                    fmt(ae["front_position_mae"]),
                ],
            ],
            note=(
                f"이 표는 기본 파이프라인의 대표 결과다. 모두 같은 PDEBench processed dataset과 case {case_index}를 사용하지만, "
                "POD의 final error는 final snapshot reconstruction error이고 DMD/AE의 final error는 rollout final error이다. "
                "따라서 아래 controlled comparison 표와 분리해서 읽어야 한다."
            ),
        )

        add_text_page(
            pdf,
            "5. 결과 해석",
            [
                f"POD rank 8 reconstruction relative L2는 {pod['reconstruction_relative_l2']:.6e}이다. POD는 주어진 true snapshot을 POD basis에 projection해 복원하므로 안정적인 선형 baseline 역할을 한다.",
                f"DMD reconstruction relative L2는 {dmd['reconstruction_relative_l2']:.6e}, rollout relative L2는 {dmd['rollout_relative_l2']:.6e}, final rollout error는 {dmd['final_rollout_error']:.6e}이다. 이는 linear time-evolution model에서 시간 구간에 따라 오차가 다르게 나타남을 보여준다.",
                f"Conv1D AE reconstruction relative L2는 {ae['reconstruction_relative_l2']:.6e}이다. 같은 latent_dim=8 조건에서 nonlinear decoder가 snapshot reconstruction을 어떻게 표현하는지 관찰할 수 있다.",
                f"AE latent linear rollout relative L2는 {ae['rollout_relative_l2']:.6e}, final rollout error는 {ae['final_rollout_error']:.6e}이다. 즉 reconstruction latent와 dynamics coordinate를 분리해서 해석해야 한다.",
                "",
                "따라서 같은 공개 데이터셋에서 비교할 때, POD는 linear reconstruction baseline, DMD는 linear rollout baseline, Conv1D AE는 nonlinear reconstruction과 latent rollout 관찰 대상으로 구분해서 읽는다.",
            ],
        )

        if aggregate:
            methods = aggregate.get("methods", {})
            aggregate_records = []
            for method, values in methods.items():
                aggregate_records.append(
                    (
                        method,
                        [
                            ("rollout mean +/- std", f"{values['rollout_mean']:.4f} +/- {values['rollout_std']:.4f}"),
                            ("final mean +/- std", f"{values['final_mean']:.4f} +/- {values['final_std']:.4f}"),
                            ("front MAE median [IQR]", f"{values['front_median']:.4f} [{values['front_q1']:.4f}, {values['front_q3']:.4f}]"),
                            ("smooth rollout mean", fmt(values["smooth_rollout_mean"])),
                            ("shock rollout mean", fmt(values["shock_rollout_mean"])),
                            ("diverged cases", f"{values['diverged_case_count']} / 13"),
                        ],
                    )
                )
            add_record_pages(
                pdf,
                "5A. 13개 test trajectory 전체 temporal extrapolation",
                aggregate_records,
                note=(
                    f"범위: {aggregate['scope']}. 발산 정의: {aggregate['divergence_definition']}. "
                    "대표 extreme case 하나의 결과를 전체 성능으로 일반화하지 않기 위해 모든 test trajectory를 집계했다."
                ),
            )

        if comparison_rows:
            add_record_pages(
                pdf,
                "6. comparison CSV에 남은 public-only rows",
                [
                    (
                        row.get("experiment_name", ""),
                        [
                            ("method", row.get("method", "")),
                            ("reconstruction L2", fmt(row.get("reconstruction_relative_l2"))),
                            ("rollout L2", fmt(row.get("rollout_relative_l2"))),
                            ("final error", fmt(row.get("final_rollout_error"))),
                        ],
                    )
                    for row in comparison_rows
                ],
            )

        add_record_pages(
            pdf,
            "7. 코드 로직 검토: 어떤 파일이 무엇을 하는가",
            [
                (
                    "공개 데이터 준비",
                    [
                        ("코드 경로", "scripts/prepare_pdebench_burgers.py"),
                        ("설정 경로", "configs/burgers/pdebench_prepare.yaml"),
                        ("로직 판단", "PDEBench 원본 HDF5에서 지정한 subset을 읽어 프로젝트 표준 HDF5 형식으로 저장하므로 재실행성과 비교 기준이 일관적이다."),
                        ("확인 결과", "u=(64, 101, 128), train=38 cases, val=13 cases, test=13 cases로 저장되어 세 방법이 같은 processed file을 사용한다."),
                    ],
                ),
                (
                    "POD reconstruction",
                    [
                        ("코드 경로", "scripts/train_pod.py, src/rom_bench/models/pod.py"),
                        ("설정 경로", "configs/burgers/pdebench_pod.yaml"),
                        ("로직 판단", f"train split snapshot으로 SVD basis를 만들고 held-out case {case_index}를 rank 8 basis에 projection한다. 구현 방향은 POD reconstruction 목적에 맞다."),
                        ("중요한 해석", "POD 결과는 시간 예측이 아니라 true snapshot을 알고 있을 때의 projection reconstruction이다. 따라서 rollout error가 아니라 reconstruction baseline으로 해석해야 한다."),
                    ],
                ),
                (
                    "DMD-based prediction",
                    [
                        ("코드 경로", "scripts/train_dmd.py, src/rom_bench/models/dmd.py"),
                        ("설정 경로", "configs/burgers/pdebench_dmd.yaml"),
                        ("로직 판단", f"case {case_index}의 앞쪽 시간 구간으로 rank 8 exact DMD를 학습한 뒤 이후 시간을 free rollout한다. DMD prediction 실험 목적에 맞다."),
                        ("중요한 해석", "Burgers 방정식은 nonlinear이지만 DMD는 하나의 linear time-advance operator로 근사한다. 그래서 학습 구간 재구성과 장시간 rollout에서 서로 다른 오차 양상이 나타날 수 있다."),
                    ],
                ),
                (
                    "Conv1D Autoencoder",
                    [
                        ("코드 경로", "scripts/train_autoencoder.py, src/rom_bench/models/autoencoder_1d.py, src/rom_bench/models/latent_dynamics.py"),
                        ("설정 경로", "configs/burgers/pdebench_autoencoder.yaml"),
                        ("로직 판단", "PyTorch Conv1D encoder-decoder로 snapshot reconstruction을 학습하고, latent vector에는 별도 linear dynamics를 맞춰 rollout한다. reconstruction 실험 로직은 맞다."),
                        ("중요한 해석", "AE는 nonlinear decoder로 field reconstruction을 수행하지만, latent 좌표가 자동으로 선형 시간좌표가 되는 것은 아니다. latent linear rollout은 별도로 해석해야 한다."),
                    ],
                ),
                (
                    "공통 비교",
                    [
                        ("코드 경로", "scripts/compare_models.py"),
                        ("입력", "metrics/pdebench_burgers_*_metrics.json"),
                        ("출력", "metrics/burgers_model_comparison.csv"),
                        ("로직 판단", "세 방법의 metrics를 한 CSV로 모으는 로직은 맞다. 다만 POD reconstruction, DMD rollout, AE reconstruction/latent rollout은 물리적으로 같은 난이도의 문제는 아니므로 해석에서 구분해야 한다."),
                    ],
                ),
            ],
            note=(
                "종합 판단: 코드 흐름은 공개 데이터 준비, 동일 case 선택, 방법별 학습/평가, metrics 저장 순서로 일관적이다. "
                "주의할 점은 숫자 하나로 방법의 우열을 정하는 것이 아니라 reconstruction과 rollout을 분리해서 읽어야 한다는 점이다."
            ),
        )

        add_text_page(
            pdf,
            "8. 상세 코드 로직: 데이터 흐름과 평가 대상",
            [
                "전체 파이프라인의 출발점은 processed HDF5 파일이다. 이 파일에는 x, t, u, split/train_indices, split/val_indices, split/test_indices가 들어 있다. 여기서 u의 shape은 [case 개수, 시간 snapshot 개수, 공간 격자점 개수] = [64, 101, 128]이다.",
                f"즉 한 개의 Burgers 해는 u_case = u[case_index] 형태로 꺼내며, 이번 리뷰에서 세 방법은 모두 case_index={case_index}를 사용한다. u_case의 shape은 [101, 128]이므로, 시간 101개와 공간 격자점 128개를 가진 하나의 space-time field라고 보면 된다.",
                f"POD는 train split에 있는 여러 case의 snapshot을 모아서 basis를 만든다. 그 다음 test split에 포함된 case {case_index}를 그 basis 위에 projection한다. 따라서 POD는 '처음 보는 test case를 train basis가 얼마나 잘 복원하는가'를 보는 구조다.",
                f"DMD는 case {case_index}의 앞쪽 시간 구간을 학습 구간으로 사용한다. 앞쪽 snapshot pair로 선형 시간 전진 operator를 추정하고, 이후 시간은 true 값을 넣지 않고 free rollout한다. 따라서 DMD의 핵심 평가는 '시간이 지날수록 예측 오차가 어떻게 누적되는가'이다.",
                f"Autoencoder는 case {case_index}의 snapshot을 Conv1D encoder-decoder로 압축하고 복원한다. encoder는 u(x)를 latent vector z로 바꾸고, decoder는 z에서 다시 u(x)를 만든다. 추가 rollout은 latent z에 대해 linear dynamics를 맞춰서 수행한다.",
                "평가 지표는 세 방법이 같은 true field와 비교되도록 저장된다. relative L2 error는 ||prediction - truth|| / (||truth|| + epsilon)으로 계산된다. front 위치 오차는 공간 기울기 |du/dx|가 큰 위치를 front로 보고, true front와 predicted front의 차이를 계산한다.",
                "따라서 코드 로직 자체는 같은 공개 데이터, 같은 case, 같은 rank 또는 latent_dim=8을 기준으로 비교하도록 정리되어 있다. 다만 POD reconstruction, DMD rollout, AE latent rollout은 문제의 성격이 다르므로 결과 해석에서 반드시 구분해야 한다.",
            ],
        )

        add_text_page(
            pdf,
            "9. 상세 코드 로직: POD reconstruction",
            [
                "POD는 먼저 train snapshot들을 하나의 큰 행렬 X로 만든다. 실제 코드에서는 각 snapshot u(x)가 길이 nx=128인 벡터이고, 여러 시간과 여러 case의 snapshot을 행 방향으로 쌓는다. 따라서 X의 shape은 [n_samples, n_features]이다.",
                "mean subtraction이 켜져 있으면 train snapshot의 평균 field를 먼저 뺀다. 이렇게 하면 POD mode는 평균값 자체가 아니라 평균에서 벗어나는 주요 변형 패턴을 학습한다.",
                "그 다음 SVD를 수행한다. 코드 기준으로는 centered = U S Vh이다. snapshot이 행이고 공간 격자가 열이므로, spatial mode는 U가 아니라 Vh의 행이다. 실제 구현도 modes = Vh[:rank]로 저장한다.",
                "rank=8 실험에서는 Vh의 앞 8개 행을 Phi로 사용한다. test snapshot u_true가 들어오면 u_centered = u_true - mean을 만들고, 계수 a = u_centered Phi^T를 계산한다. reconstruction은 u_pred = a Phi + mean이다.",
                "이 로직은 POD reconstruction으로 맞다. 왜냐하면 POD는 주어진 rank에서 train 데이터의 평균 제곱 projection error를 최소화하는 선형 basis를 찾는 방법이기 때문이다.",
                "하지만 여기에는 시간 동역학 모델이 없다. POD reconstruction error가 어떤 값을 갖더라도, 그것만으로 미래를 예측했다는 뜻은 아니다. 현재 POD 결과는 '이미 주어진 snapshot을 rank 8 선형 공간에 어떻게 압축하고 복원했는가'를 의미한다.",
                "moving front에서는 같은 모양이 오른쪽이나 왼쪽으로 조금 이동하는 현상이 생긴다. 선형 basis는 이동 자체를 자연스럽게 표현하지 못해서 여러 mode를 필요로 한다. 그래서 주어진 rank에서 전체 L2 error와 gradient/front 주변 error가 함께 나타날 수 있다.",
            ],
        )

        add_text_page(
            pdf,
            "10. 상세 코드 로직: DMD prediction",
            [
                "DMD는 시간 순서가 있는 snapshot pair를 사용한다. 학습 구간의 snapshot을 X1 = [u_0, u_1, ..., u_{m-1}], X2 = [u_1, u_2, ..., u_m]처럼 두 행렬로 나눈다.",
                "목표는 X2가 A X1과 비슷해지도록 선형 operator A를 찾는 것이다. full A를 직접 만들면 너무 클 수 있으므로, 코드에서는 SVD로 rank=8 truncation을 하고 reduced operator를 계산한다.",
                "그 다음 reduced operator의 eigenvalue와 DMD mode를 구한다. eigenvalue의 크기는 mode가 시간에 따라 감쇠하는지 커지는지를 보여주고, angle은 진동 주파수와 관련된다.",
                "reconstruction에서는 초기 amplitude를 계산한 뒤 DMD mode와 eigenvalue를 이용해 학습 구간 또는 전체 시간의 field를 만든다. rollout에서는 학습 이후 시점에 true snapshot을 다시 넣지 않고 eigenvalue를 계속 거듭제곱해서 미래를 생성한다.",
                "이 로직은 DMD-based prediction으로 맞다. DMD는 nonlinear Burgers 방정식을 직접 풀지 않고, 관측된 snapshot 사이의 시간 전진을 선형 근사로 표현하는 방법이다.",
                "다만 Burgers 방정식의 실제 변화에는 u du/dx라는 nonlinear 항과 viscosity diffusion이 함께 작용한다. 하나의 고정된 선형 operator로 이 효과를 모든 시간에 정확히 표현하기 어렵다.",
                "그래서 DMD reconstruction error와 rollout error는 서로 다른 크기와 시간 분포를 가질 수 있다. 미래로 갈수록 front 위치, phase, amplitude, gradient가 조금씩 어긋나며 오차가 누적될 수 있기 때문이다.",
            ],
        )

        add_text_page(
            pdf,
            "11. 상세 코드 로직: Conv1D Autoencoder",
            [
                "Autoencoder는 각 1D snapshot u(x)를 입력으로 받는다. Conv1D encoder는 공간 방향의 국소 패턴, 예를 들면 완만한 영역과 급격한 front 주변 패턴을 convolution filter로 읽어 latent vector z로 압축한다.",
                "latent_dim=8이면 하나의 snapshot은 z1부터 z8까지 총 8개의 숫자로 표현된다. 이 숫자들은 사람이 미리 정한 물리량이 아니라, reconstruction loss를 줄이기 위해 neural network가 스스로 만든 내부 좌표다.",
                "decoder는 이 8개 latent 숫자에서 다시 길이 128의 u(x)를 만든다. loss는 기본적으로 true field와 reconstructed field 사이의 MSE 또는 설정된 추가 loss를 줄이는 방향으로 계산된다.",
                "학습 중에는 validation loss가 최소로 기록된 checkpoint를 best checkpoint라는 이름으로 저장한다. 이번 결과에서 backend가 pytorch_conv1d로 기록되어 있으므로, 실제 PyTorch Conv1D Autoencoder가 사용된 것이 맞다.",
                "latent trajectory figure는 시간에 따라 z1, z2, ..., z8이 어떻게 변하는지 그린 것이다. 이 그래프는 AE가 field의 시간 변화를 latent 공간에서 어떤 경로로 표현하는지 보는 용도다.",
                "중요한 점은 AE의 기본 목적이 reconstruction이라는 것이다. 즉 z가 field를 잘 복원하도록 학습되었지, z(t+1)=B z(t) 같은 간단한 선형 동역학을 따르도록 강하게 학습된 것은 아니다.",
                "따라서 AE reconstruction error와 POD reconstruction error는 서로 다른 표현 방식의 차이를 보여준다. nonlinear decoder는 rank 8 선형 basis와 다른 방식으로 field를 표현한다. 또한 latent linear rollout error는 reconstruction error와 별개로 해석해야 한다. latent 공간의 압축 좌표가 곧 시간 예측 좌표는 아니기 때문이다.",
            ],
        )

        add_text_page(
            pdf,
            "12. 상세 코드 로직: 비교 결과를 읽는 기준",
            [
                "현재 comparison CSV는 method, rank 또는 latent_dim, reconstruction_relative_l2, rollout_relative_l2, final_rollout_error, front_position_mae 등을 한 표에 모은다. 이 정리는 실험 결과를 나란히 보기 위한 것이며, 모든 열을 같은 의미로 읽으면 안 된다.",
                "POD의 reconstruction_relative_l2는 projection reconstruction error다. POD에는 별도 시간 전진 모델이 없으므로 rollout_relative_l2는 NaN 또는 N/A로 보는 것이 맞다.",
                "DMD의 reconstruction_relative_l2는 modal reconstruction이 학습 데이터 구조를 얼마나 잘 맞췄는지 보여준다. rollout_relative_l2와 final_rollout_error는 학습 이후 free prediction이 얼마나 무너지는지 보여준다.",
                "AE의 reconstruction_relative_l2는 Conv1D encoder-decoder의 복원 성능이고, rollout_relative_l2는 latent linear dynamics까지 붙였을 때의 예측 성능이다. 이 둘은 같은 모델에서 나온 값이지만 의미는 다르다.",
                "front_position_mae는 전체 L2 error와 다른 정보를 준다. 예를 들어 front 위치는 맞지만 amplitude가 틀리면 front error는 작고 L2 error는 클 수 있다. 반대로 전체 L2 error가 작아도 sharp gradient 위치가 조금 어긋나면 front error가 커질 수 있다.",
                "따라서 현재 결과의 판단 기준은 다음과 같다. reconstruction 능력은 POD와 AE 중심으로 비교하고, 장시간 예측 안정성은 DMD rollout과 AE latent rollout 중심으로 비교한다. front 관련 평가는 전체 L2 error와 따로 읽는다.",
            ],
        )

        add_record_pages(
            pdf,
            "13. 추가 검증 체크리스트",
            [
                (
                    "POD snapshot matrix 방향과 spatial mode",
                    [
                        ("확인 결과", str(audit["pod_snapshot_matrix_orientation"])),
                        ("spatial mode", str(audit["pod_spatial_modes_source"])),
                        ("판단", "기존 설명에서 U를 spatial mode처럼 쓴 부분은 일반적인 열-snapshot 관례 설명이었고, 실제 코드 기준으로는 Vh의 행이 spatial mode이다. PDF 설명을 이 기준으로 수정했다."),
                    ],
                ),
                (
                    f"AE 학습 데이터와 case {case_index} 포함 여부",
                    [
                        (f"case {case_index} split", str(audit["case_index_split"])),
                        ("HDF5 train split만 사용?", str(audit["ae_uses_hdf5_case_train_split_only"])),
                        ("AE training snapshots", str(audit["ae_training_snapshots"])),
                        (f"case {case_index} snapshot 포함?", "yes, time indices 0..59 are used for AE training"),
                        ("판단", f"AE는 HDF5의 train case split만으로 학습된 것이 아니라, 평가 case {case_index}의 앞 60개 시간 snapshot으로 학습되었다. 따라서 AE 결과는 temporal extrapolation 실험이지 case-generalization 실험은 아니다."),
                    ],
                ),
                (
                    "latent dynamics와 DMD/AE temporal split",
                    [
                        ("latent dynamics fit", str(audit["latent_dynamics_fit_snapshots"])),
                        ("future snapshot fitting 사용?", str(audit["latent_dynamics_future_snapshots_used_for_fit"])),
                        ("DMD training snapshots", str(audit["dmd_train_snapshots"])),
                        ("DMD와 AE temporal split 동일?", str(audit["dmd_ae_temporal_split_identical"])),
                        ("latent operator spectral radius", fmt(audit["latent_linear_operator_spectral_radius"])),
                        ("판단", f"미래 test snapshot은 latent linear operator fitting에는 쓰이지 않았다. 다만 AE encoder-decoder 자체는 case {case_index}의 앞쪽 시간구간을 보고 학습했다."),
                    ],
                ),
                (
                    "front detector 연속성",
                    [
                        ("front method", str(audit["front_detector_method"])),
                        ("dx at nx=128", fmt(audit["front_detector_dx_128"])),
                        ("max detector-position jump", fmt(audit["front_detector_max_jump"])),
                        ("large jump threshold", fmt(audit["front_detector_large_jump_threshold"])),
                        ("large jump count", str(audit["front_detector_large_jump_count"])),
                        ("same front likely?", str(audit["front_detector_tracks_same_front_likely"])),
                        ("판단", "max-gradient detector는 각 시점의 가장 큰 |du/dx| 위치를 독립적으로 고른다. 여러 sharp region이 경쟁하거나 주기 경계를 넘으면 다른 위치로 전환될 수 있다. 따라서 이 jump는 물리적 front 속도나 해상도 오차가 아니다."),
                    ],
                ),
            ],
        )

        if audit["resolution_rows"]:
            add_table_page(
                pdf,
                "14. 공간해상도 128 / 256 / 1024 front gradient 비교",
                ["nx", "stride", "mean max |du/dx|", "ratio vs 1024", "max detector jump", "jump count"],
                [
                    [
                        str(row["resolution"]),
                        str(row["space_stride"]),
                        fmt(row["mean_max_abs_gradient"]),
                        fmt(row["mean_gradient_ratio_vs_1024"]),
                        fmt(row["max_front_step"]),
                        str(row["large_front_jump_count"]),
                    ]
                    for row in audit["resolution_rows"]
                ],
                note=(
                    f"원본 PDEBench source trajectory {data['subset']['source_case_indices'][case_index]}를 같은 시간 stride로 읽고, 공간 stride만 8, 4, 1로 바꿔 nx=128, 256, 1024를 비교했다. "
                    "ratio vs 1024가 1보다 작으면 downsampling 때문에 sharp front gradient가 약해졌다는 뜻이다. "
                    "마지막 두 열은 maximum-gradient 검출기의 위치 전환 진단이며 물리적 front 이동량으로 해석하지 않는다."
                ),
            )
            add_image_page(
                pdf,
                "공간해상도에 따른 front sharpness 비교",
                "figures/data/pdebench_resolution_audit/resolution_gradient_comparison.png",
                "nx=128은 현재 ROM 실험에 쓰인 processed 해상도이고, nx=1024는 PDEBench 원본 공간해상도이다. 이 그림은 downsampling이 max |du/dx|를 얼마나 줄이는지 보여준다.",
            )

        if failure_summary and failure_rows:
            selected_cases = failure_summary.get("selected_cases", [])
            add_text_page(
                pdf,
                "15. smooth/shock 및 dimension sweep 비교",
                [
                    "이 비교는 어떤 방법의 순위를 매기기 위한 것이 아니다. 목적은 같은 공개 PDEBench 데이터 안에서 smooth-like case와 shock-like case를 명시적으로 나누고, front 주변에서 각 reconstruction 또는 rollout이 어떤 오차 모양을 만드는지 관찰하는 것이다.",
                    "사용한 데이터셋은 앞의 기본 실험과 동일하다. 같은 processed HDF5 안에서 test split의 mean max |du/dx|가 작은 case를 smooth-like, 큰 case를 shock-like로 골라 두 regime을 나란히 비교했다.",
                    "또한 POD와 AE reconstruction 비교가 공정하지 않다는 지적을 반영해, controlled comparison을 따로 만들었다. 이 controlled comparison에서는 POD와 AE 모두 같은 case의 같은 temporal training snapshots, 즉 time index 0..59만 사용한다.",
                    "따라서 이것은 다른 데이터셋을 쓴 별도 결과가 아니라, 같은 PDEBench 데이터셋 안에서 smooth/shock 및 dimension sweep 관찰 항목을 더 자세히 펼친 것이다.",
                ],
            )
            add_table_page(
                pdf,
                "16. smooth-like / shock-like case 선택",
                ["regime", "case", "mean max |du/dx|", "max |du/dx|", "max-gradient time index"],
                [
                    [
                        str(case["regime"]),
                        str(case["case_index"]),
                        fmt(case["mean_max_gradient"]),
                        fmt(case["max_gradient"]),
                        str(case["time_of_max_gradient_index"]),
                    ]
                    for case in selected_cases
                ],
                note=(
                    f"선택 기준은 test split 내부의 mean max |du/dx|이다. 값이 작은 case {smooth_case}는 smooth-like, 값이 큰 case {shock_case}는 shock-like로 표시했다. "
                    "이 구분은 물리 regime label이 아니라, 현재 subset에서 gradient 강도 차이를 보기 위한 operational label이다."
                ),
            )
            dim8_rows = [
                row
                for row in failure_rows
                if row.get("dimension") == "8" and row.get("method") in {"POD", "DMD", "AE"}
            ]
            add_table_page(
                pdf,
                "17. controlled comparison 결과표: 동일 temporal split, dim=8",
                ["regime", "method", "post-split L2", "final L2", "front MAE", "front-window L2"],
                [
                    [
                        row["regime"],
                        row["method"],
                        fmt(row["test_relative_l2"]),
                        fmt(row["final_relative_l2"]),
                        "N/A" if row["regime"] == "smooth_like" else fmt(row.get("front_position_mae_report", row["front_position_mae"])),
                        fmt(row["front_window_relative_l2"]),
                    ]
                    for row in dim8_rows
                ],
                note=(
                    "이 표는 기본 대표 결과표와 다른 controlled comparison이다. POD와 AE는 같은 case의 같은 time index 0..59로 reconstruction 모델을 맞췄고, "
                    "DMD는 같은 구간으로 rollout operator를 맞췄다. smooth-like case는 뚜렷한 shock/front가 아니므로 front MAE를 N/A로 비활성화했다."
                ),
            )
            add_text_page(
                pdf,
                "18. smooth/shock 및 dimension sweep에서 보이는 오차 양상",
                [
                    f"smooth-like case {smooth_case}에서는 true field 자체의 gradient가 약하고 시간 변화도 완만하다. 그래서 front 주변 국부 오차 그림에서도 sharp shock failure보다는 작은 amplitude 차이나 완만한 profile 차이가 주로 보인다.",
                    f"shock-like case {shock_case}에서는 mean max |du/dx|가 훨씬 크고, front 주변 국부 오차가 해석의 중심이 된다. 전체 L2 error만 보면 front가 어디서 어떻게 흐려졌는지 알기 어렵기 때문에, front overlay와 local absolute error를 함께 보도록 했다.",
                    "rank sweep과 latent-dimension sweep은 차원을 늘렸을 때 숫자가 어떻게 변하는지뿐 아니라, shock-like case에서 front-window error와 gradient error가 어떻게 남는지 보기 위한 것이다.",
                    "동일 학습 조건 비교에서는 POD와 AE 모두 같은 case-specific temporal training snapshots만 사용한다. 따라서 기존 기본 파이프라인의 POD/AE 비교보다 reconstruction 조건이 더 직접적으로 맞춰져 있다.",
                    "이 결과는 방법의 우열 판단이 아니라, smooth-like field와 shock-like field에서 오차가 나타나는 위치와 형태를 비교하기 위한 자료다.",
                ],
            )
            add_image_page(
                pdf,
                "smooth-like / shock-like case 선택 기준",
                "figures/failure_modes/pdebench_smooth_shock_failure_modes/case_sharpness_selection.png",
                f"test split case들의 mean max |du/dx|를 표시하고, smooth/shock 비교에 사용한 smooth-like case {smooth_case}와 shock-like case {shock_case}를 강조했다.",
            )
            add_image_page(
                pdf,
                "smooth-like와 shock-like true space-time field",
                "figures/failure_modes/pdebench_smooth_shock_failure_modes/smooth_vs_shock_true_spacetime.png",
                "두 case의 true solution을 나란히 보여준다. shock-like case는 공간 기울기가 큰 구조가 훨씬 뚜렷하다.",
            )
            add_image_page(
                pdf,
                "smooth-like case front 주변 overlay",
                f"figures/failure_modes/pdebench_smooth_shock_failure_modes/front_overlay_smooth_like_case{smooth_case}.png",
                "smooth-like case에서 truth, POD rank 8, DMD rank 8, AE latent 8 결과를 front 주변에서 겹쳐 그렸다.",
            )
            add_image_page(
                pdf,
                "shock-like case front 주변 overlay",
                f"figures/failure_modes/pdebench_smooth_shock_failure_modes/front_overlay_shock_like_case{shock_case}.png",
                "shock-like case에서 truth, POD rank 8, DMD rank 8, AE latent 8 결과를 front 주변에서 겹쳐 그렸다. local absolute error도 함께 표시했다.",
            )
            add_image_page(
                pdf,
                "POD rank / AE latent dimension sweep",
                "figures/failure_modes/pdebench_smooth_shock_failure_modes/pod_rank_ae_latent_sweep.png",
                "POD rank와 AE latent dimension을 2, 4, 8, 16, 32로 바꿨을 때 post-split relative L2가 어떻게 달라지는지 smooth-like와 shock-like case로 나누어 표시했다.",
            )
            add_image_page(
                pdf,
                "동일 학습 조건 reconstruction 비교",
                "figures/failure_modes/pdebench_smooth_shock_failure_modes/same_training_reconstruction_comparison.png",
                "POD와 AE가 같은 temporal training snapshots를 사용하도록 맞춘 controlled comparison이다. 이 그림도 순위가 아니라 조건을 맞춘 관찰 결과로 읽어야 한다.",
            )

        add_text_page(
            pdf,
            "19. 판단 결과가 왜 그렇게 나왔는가: POD",
            [
                f"POD rank 8의 reconstruction relative L2는 {pod['reconstruction_relative_l2']:.6e}이다. 이 값은 train split 전체에서 얻은 8개의 선형 basis가 held-out case {case_index}의 전체 시간장을 어느 정도 표현했는지를 보여준다.",
                "Burgers 해가 단순한 진폭 변화만 한다면 적은 POD mode로도 잘 표현된다. 하지만 front 또는 sharp gradient가 위치를 바꾸면, 같은 모양이 조금 이동한 것만으로도 선형 basis 입장에서는 여러 mode가 필요하다. 그래서 rank 8에서는 front 주변 오차와 gradient 오차가 남는다.",
                f"front_position_mae는 {pod['front_position_mae']:.6e}, spatial-gradient relative L2는 {pod['spatial_gradient_relative_l2']:.6e}이다. 이는 front 위치 오차와 gradient sharpness 오차가 서로 다른 정보를 준다는 뜻이다.",
                "또 하나 중요한 점은 POD의 final error가 rollout 실패를 의미하지 않는다는 것이다. 현재 POD는 매 시점 true snapshot을 basis에 projection하므로 시간 적분 오차가 쌓이지 않는다. 그래서 POD 결과는 '동일 rank에서 선형 공간이 데이터를 얼마나 담는가'를 보는 결과다.",
                "따라서 이 결과는 'POD 로직은 reconstruction 목적에 맞고, rank 8 선형 subspace에서 moving front 구조가 어떤 오차 양상으로 나타나는지 관찰할 수 있다'로 해석한다.",
            ],
        )

        add_text_page(
            pdf,
            "20. 판단 결과가 왜 그렇게 나왔는가: DMD",
            [
                f"DMD reconstruction relative L2는 {dmd['reconstruction_relative_l2']:.6e}이다. 이것은 DMD가 case {case_index}의 시간 데이터 자체에서 modal dynamics를 맞추기 때문에 학습 구간의 시간 패턴을 어떻게 재구성했는지 보여준다.",
                f"하지만 rollout relative L2는 {dmd['rollout_relative_l2']:.6e}, final rollout error는 {dmd['final_rollout_error']:.6e}로 커진다. Burgers 방정식의 실제 시간 발전에는 u du/dx라는 nonlinear 항이 있는데, DMD는 이를 고정된 linear operator 하나로 근사한다.",
                "짧은 시간에는 선형 근사가 그럴듯하게 보일 수 있다. 그러나 시간이 길어지면 front 위치, amplitude, phase, diffusion에 의한 smoothing 차이가 조금씩 쌓인다. 이 누적 오차가 rollout error 증가로 나타난다.",
                f"threshold crossing time은 {dmd['threshold_crossing_time']:.4f} 근처로 기록되었다. 이 시점 이후에는 field 전체의 shape 차이나 phase 차이가 평가 threshold를 넘어섰다고 볼 수 있다.",
                f"front_position_mae가 {dmd['front_position_mae']:.6e}로 기록되었다고 해서 field 전체가 같은 형태로 예측되었다는 뜻은 아니다. front의 대표 위치는 비슷해도, 앞뒤 profile의 amplitude나 shock-like gradient가 틀리면 L2 error는 달라질 수 있다.",
                "따라서 이 결과는 'DMD 로직은 prediction 실험에 맞고, nonlinear Burgers dynamics를 linear rollout으로 근사할 때 어떤 누적 오차가 생기는지 관찰할 수 있다'로 해석한다.",
            ],
        )

        add_text_page(
            pdf,
            "21. 판단 결과가 왜 그렇게 나왔는가: Autoencoder",
            [
                f"Conv1D AE의 reconstruction relative L2는 {ae['reconstruction_relative_l2']:.6e}이다. 같은 latent_dim=8 조건에서 encoder와 decoder의 nonlinear mapping이 snapshot을 어떻게 표현하는지 보여준다.",
                "POD는 rank 8 선형 평면 안에서 답을 찾는다. 반면 Autoencoder는 8차원 latent vector를 거쳐 decoder가 nonlinear하게 field를 만든다. 그래서 두 방법은 같은 숫자 차원이라도 표현 방식이 다르다.",
                f"AE latent linear rollout relative L2는 {ae['rollout_relative_l2']:.6e}, final rollout error는 {ae['final_rollout_error']:.6e}이다. 이 값은 AE가 기본적으로 reconstruction loss를 줄이도록 학습되었고, latent dynamics는 별도 선형 근사로 붙어 있음을 보여준다.",
                "reconstruction을 잘하는 latent 좌표가 반드시 시간에 따라 직선적이거나 선형 시스템처럼 움직이는 좌표는 아니다. latent trajectory가 휘어 있거나 서로 얽혀 있으면, linear dynamics가 미래 latent를 잘못 보낼 수 있다.",
                "또한 rollout 중 latent vector가 학습 때 보지 못한 영역으로 벗어나면 decoder는 학습 분포 밖의 field를 만들 수 있다. 이 off-manifold 현상이 final rollout error에 영향을 줄 수 있다.",
                "따라서 이 결과는 'AE reconstruction 로직과 latent linear rollout 로직을 분리해서 보며, nonlinear representation과 latent dynamics의 차이를 관찰한다'로 해석한다.",
            ],
        )

        add_text_page(
            pdf,
            "22. 최종 검토 의견",
            [
                "코드 로직 자체는 현재 목표에 대해 대체로 맞다. 공개 PDEBench 데이터만 사용하도록 정리되었고, 세 방법 모두 같은 processed HDF5와 같은 held-out case를 사용한다.",
                "다만 결과 판단은 반드시 두 층으로 나누어야 한다. 첫째, reconstruction 문제에서는 POD와 AE가 어떤 방식으로 snapshot을 표현하는지 본다. 둘째, prediction 또는 rollout 문제에서는 DMD와 AE latent dynamics의 오차 누적 양상을 본다. POD projection 숫자를 DMD rollout 숫자와 직접 경쟁시키면 공정하지 않다.",
                "현재 수치는 각 방법의 관찰 지표다. DMD reconstruction 값은 선택된 case의 시간 구조에 직접 맞춰진 modal reconstruction 결과이고, AE reconstruction 값은 nonlinear encoder-decoder의 snapshot 복원 결과이며, POD 값은 train basis projection 결과다.",
                "결론적으로, 이 실험은 방법의 순위를 정하려는 결과가 아니다. 더 정확한 결론은 '같은 공개 데이터에서 reconstruction과 rollout은 서로 다른 문제이며, nonlinear Burgers 구조 때문에 각 방법의 오차 양상이 다르게 드러난다'이다.",
            ],
        )

        add_method_figures(
            pdf,
            "POD",
            "figures/pod/pdebench_burgers_pod_rank8",
            [
                ("pod_energy_spectrum.png", "POD singular value spectrum과 cumulative energy. Rank 선택과 에너지 집중 정도를 확인한다."),
                ("pod_final_reconstruction.png", f"Held-out case {case_index}의 마지막 snapshot에서 truth와 POD reconstruction을 비교한다."),
                ("rollout_error_vs_time.png", "POD projection reconstruction의 시간별 relative L2 error. POD는 rollout이 아니라 snapshot별 projection이다."),
                ("burgers_spacetime_true.png", f"PDEBench held-out case {case_index}의 true space-time field."),
                ("burgers_spacetime_reconstruction.png", "POD rank 8 reconstruction의 space-time field."),
                ("burgers_spacetime_error.png", "POD rank 8 reconstruction absolute error contour."),
            ],
        )
        add_method_figures(
            pdf,
            "DMD",
            "figures/dmd/pdebench_burgers_dmd_rank8",
            [
                ("dmd_eigenvalues.png", "DMD eigenvalue 위치. Rollout 안정성과 mode 감쇠/성장 가능성을 확인한다."),
                ("dmd_final_rollout.png", "마지막 시점에서 truth와 DMD free rollout prediction을 비교한다."),
                ("rollout_error_vs_time.png", "DMD free rollout의 시간별 relative L2 error."),
                ("burgers_spacetime_prediction.png", "DMD rollout prediction의 space-time field."),
                ("burgers_spacetime_error.png", "DMD rollout absolute error contour."),
            ],
        )
        add_method_figures(
            pdf,
            "Autoencoder",
            "figures/autoencoder/pdebench_burgers_ae_latent8",
            [
                ("autoencoder_final_reconstruction.png", "마지막 시점에서 truth와 Conv1D AE reconstruction을 비교한다."),
                ("autoencoder_latent_rollout_final.png", "마지막 시점에서 truth와 latent linear rollout 결과를 비교한다."),
                ("reconstruction_error_vs_time.png", "Conv1D AE reconstruction의 시간별 relative L2 error."),
                ("rollout_error_vs_time.png", "AE latent linear rollout의 시간별 relative L2 error."),
                ("burgers_spacetime_error.png", "Conv1D AE reconstruction absolute error contour."),
                ("latent_trajectory.png", "latent_dim=8 좌표의 시간 변화. Latent representation을 정성적으로 확인한다."),
            ],
        )

    print(out)


if __name__ == "__main__":
    main()
