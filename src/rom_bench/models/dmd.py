"""Exact DMD implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DMDModel:
    """Exact DMD model."""

    modes: np.ndarray
    eigenvalues: np.ndarray
    amplitudes: np.ndarray
    dt: float

    @classmethod
    def fit(cls, snapshots: np.ndarray, dt: float, rank: int, filter_unstable: bool = False) -> "DMDModel":
        """Fit DMD from snapshots shaped [nt, n_features]."""
        x1 = snapshots[:-1].T
        x2 = snapshots[1:].T
        u, s, vh = np.linalg.svd(x1, full_matrices=False)
        r = min(rank, len(s))
        u_r = u[:, :r]
        s_r = s[:r]
        v_r = vh.conj().T[:, :r]
        a_tilde = u_r.conj().T @ x2 @ v_r @ np.diag(1.0 / s_r)
        eigvals, w = np.linalg.eig(a_tilde)
        modes = x2 @ v_r @ np.diag(1.0 / s_r) @ w
        if filter_unstable:
            keep = np.abs(eigvals) <= 1.0 + 1.0e-10
            eigvals = eigvals[keep]
            modes = modes[:, keep]
        amplitudes = np.linalg.lstsq(modes, snapshots[0], rcond=None)[0]
        return cls(modes=modes, eigenvalues=eigvals, amplitudes=amplitudes, dt=dt)

    def predict(self, n_steps: int) -> np.ndarray:
        """Free rollout for n_steps snapshots."""
        states = []
        for k in range(n_steps):
            state = self.modes @ (self.amplitudes * self.eigenvalues**k)
            states.append(np.real(state))
        return np.asarray(states)

    def continuous_eigenvalues(self) -> np.ndarray:
        """Continuous-time eigenvalues."""
        return np.log(self.eigenvalues) / self.dt

    def frequencies(self) -> np.ndarray:
        """DMD modal frequencies in cycles per unit time."""
        return np.abs(np.imag(self.continuous_eigenvalues())) / (2.0 * np.pi)

    def growth_rates(self) -> np.ndarray:
        """DMD modal growth rates."""
        return np.real(self.continuous_eigenvalues())

    def dominant_frequency(self) -> float:
        """Frequency of the largest-amplitude mode."""
        if len(self.amplitudes) == 0:
            return float("nan")
        idx = int(np.argmax(np.abs(self.amplitudes)))
        return float(self.frequencies()[idx])
