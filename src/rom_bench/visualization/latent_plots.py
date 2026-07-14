"""Latent representation plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rom_bench.visualization.common import save_figure


def plot_latent_trajectory(t: np.ndarray, latent: np.ndarray, title: str, path: str | Path) -> None:
    """Plot latent coordinates over time."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i in range(min(latent.shape[1], 6)):
        ax.plot(t, latent[:, i], label=f"z{i + 1}")
    ax.set_xlabel("time")
    ax.set_ylabel("latent value")
    ax.set_title(title)
    ax.legend()
    save_figure(fig, path)
