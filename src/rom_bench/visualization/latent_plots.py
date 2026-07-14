"""Latent representation plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rom_bench.visualization.common import save_figure


def plot_latent_trajectory(t: np.ndarray, latent: np.ndarray, title: str, path: str | Path) -> None:
    """Plot latent coordinates over time."""
    fig_width = max(7.0, 0.7 * latent.shape[1] + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    for i in range(latent.shape[1]):
        ax.plot(t, latent[:, i], label=f"z{i + 1}")
    ax.set_xlabel("time")
    ax.set_ylabel("latent value")
    ax.set_title(title)
    ax.legend(ncol=2 if latent.shape[1] > 6 else 1, fontsize=8)
    save_figure(fig, path)
