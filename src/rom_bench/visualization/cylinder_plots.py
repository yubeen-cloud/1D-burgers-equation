"""Cylinder wake plotting functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rom_bench.visualization.common import save_figure


def plot_cylinder_field(x: np.ndarray, y: np.ndarray, field: np.ndarray, title: str, path: str | Path) -> None:
    """Save a 2D cylinder field."""
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    im = ax.pcolormesh(x, y, field, shading="auto", cmap="coolwarm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    save_figure(fig, path)


def plot_phase_portrait(a: np.ndarray, b: np.ndarray, title: str, path: str | Path) -> None:
    """Save a two-coefficient phase portrait."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(a, b)
    ax.set_xlabel("coefficient 1")
    ax.set_ylabel("coefficient 2")
    ax.set_title(title)
    save_figure(fig, path)
