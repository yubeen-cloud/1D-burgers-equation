"""Burgers plotting functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rom_bench.visualization.common import save_figure


def plot_spacetime(x: np.ndarray, t: np.ndarray, field: np.ndarray, title: str, path: str | Path) -> None:
    """Save a space-time contour."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.pcolormesh(x, t, field, shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("time")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    save_figure(fig, path)


def plot_field_comparison(x: np.ndarray, true: np.ndarray, pred: np.ndarray, title: str, path: str | Path) -> None:
    """Compare true and predicted final field."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(x, true, label="truth")
    ax.plot(x, pred, "--", label="prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(title)
    ax.legend()
    save_figure(fig, path)


def plot_error_curve(t: np.ndarray, error: np.ndarray, title: str, path: str | Path) -> None:
    """Save semilog error curve."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(t, error)
    ax.set_xlabel("time")
    ax.set_ylabel("relative L2 error")
    ax.set_title(title)
    save_figure(fig, path)


def plot_pod_spectrum(singular_values: np.ndarray, energy: np.ndarray, path: str | Path) -> None:
    """Save POD singular value and cumulative energy plots."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].semilogy(np.arange(1, len(singular_values) + 1), singular_values, marker="o")
    axes[0].set_xlabel("mode")
    axes[0].set_ylabel("singular value")
    axes[0].set_title("POD singular values")
    axes[1].plot(np.arange(1, len(energy) + 1), energy, marker="o")
    axes[1].set_xlabel("mode")
    axes[1].set_ylabel("cumulative energy")
    axes[1].set_title("POD cumulative energy")
    save_figure(fig, path)
