"""Common plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rom_bench.paths import resolve_path


def save_figure(fig: plt.Figure, path: str | Path, save_pdf: bool = True, dpi: int = 300) -> None:
    """Save PNG and optional PDF."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_series(x: np.ndarray, ys: dict[str, np.ndarray], xlabel: str, ylabel: str, title: str, path: str | Path) -> None:
    """Plot multiple line series."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, y in ys.items():
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_figure(fig, path)
