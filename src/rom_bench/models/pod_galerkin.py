"""POD-Galerkin rollout for 1D Burgers equation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from rom_bench.data.burgers import burgers_rhs, stable_dt
from rom_bench.models.pod import PODModel


@dataclass
class PODGalerkinStats:
    """Runtime counters for the POD-Galerkin RHS."""

    rhs_calls: int = 0
    rhs_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    full_rhs_seconds: float = 0.0
    projection_seconds: float = 0.0


@dataclass
class PODGalerkinResult:
    """POD-Galerkin rollout result."""

    times: np.ndarray
    coefficients: np.ndarray
    states: np.ndarray
    stats: PODGalerkinStats


class PODGalerkinBurgers:
    """POD-Galerkin model that evaluates the nonlinear Burgers RHS on the full grid.

    This is intentionally the standard, non-hyper-reduced formulation. Each reduced
    RHS call reconstructs a full state, evaluates the full nonlinear finite-volume
    RHS, then projects the result back to POD coefficients. That full-grid RHS
    evaluation is the nonlinear-term bottleneck this experiment is meant to expose.
    """

    def __init__(
        self,
        pod: PODModel,
        x: np.ndarray,
        viscosity: float,
        boundary_condition: str = "periodic",
        modal_damping: float = 0.0,
    ) -> None:
        self.pod = pod
        self.x = x
        self.dx = float(x[1] - x[0])
        self.viscosity = float(viscosity)
        self.boundary_condition = boundary_condition
        self.modal_damping = float(modal_damping)
        self.stats = PODGalerkinStats()

    @property
    def rank(self) -> int:
        """Reduced dimension."""
        return self.pod.rank

    @property
    def nx(self) -> int:
        """Full grid size."""
        return int(self.x.size)

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Project one full state to POD coefficients."""
        return self.pod.encode(state[None, :])[0]

    def decode(self, coeffs: np.ndarray) -> np.ndarray:
        """Decode one coefficient vector to a full state."""
        return self.pod.decode(coeffs[None, :])[0]

    def rhs_coefficients(self, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate reduced RHS by full-grid nonlinear evaluation and projection."""
        t0 = perf_counter()

        t_reconstruct = perf_counter()
        state = self.decode(coeffs)
        self.stats.reconstruction_seconds += perf_counter() - t_reconstruct

        t_full_rhs = perf_counter()
        full_rhs = burgers_rhs(state, self.dx, self.viscosity, self.boundary_condition)
        self.stats.full_rhs_seconds += perf_counter() - t_full_rhs

        t_project = perf_counter()
        reduced_rhs = full_rhs @ self.pod.modes.T
        if self.modal_damping:
            reduced_rhs = reduced_rhs - self.modal_damping * coeffs
        self.stats.projection_seconds += perf_counter() - t_project

        self.stats.rhs_calls += 1
        self.stats.rhs_seconds += perf_counter() - t0
        return reduced_rhs

    def _rk3_step(self, coeffs: np.ndarray, dt: float) -> np.ndarray:
        """SSP-RK3 step in reduced coordinates."""
        a1 = coeffs + dt * self.rhs_coefficients(coeffs)
        a2 = 0.75 * coeffs + 0.25 * (a1 + dt * self.rhs_coefficients(a1))
        return (1.0 / 3.0) * coeffs + (2.0 / 3.0) * (a2 + dt * self.rhs_coefficients(a2))

    def rollout(self, initial_state: np.ndarray, target_times: np.ndarray, cfl: float = 0.2) -> PODGalerkinResult:
        """Integrate POD coefficients over the requested snapshot times."""
        coeffs = self.encode(initial_state)
        current_time = float(target_times[0])
        coeff_history = [coeffs.copy()]
        state_history = [self.decode(coeffs)]

        for target_time in target_times[1:]:
            target = float(target_time)
            while current_time < target - 1.0e-12:
                state = self.decode(coeffs)
                dt = stable_dt(state, self.dx, self.viscosity, cfl, target - current_time)
                coeffs = self._rk3_step(coeffs, dt)
                current_time += dt
                if not np.all(np.isfinite(coeffs)):
                    raise FloatingPointError("POD-Galerkin rollout produced NaN or Inf coefficients")
            current_time = target
            coeff_history.append(coeffs.copy())
            state_history.append(self.decode(coeffs))

        return PODGalerkinResult(
            times=np.asarray(target_times),
            coefficients=np.asarray(coeff_history),
            states=np.asarray(state_history),
            stats=self.stats,
        )
