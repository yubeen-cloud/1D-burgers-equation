"""Tests for POD-Galerkin Burgers rollout."""

from __future__ import annotations

import numpy as np

from rom_bench.data.burgers import BurgersCase, BurgersSolverConfig, solve_case
from rom_bench.models.pod import PODModel
from rom_bench.models.pod_galerkin import PODGalerkinBurgers


def test_pod_galerkin_rollout_shape_and_finite() -> None:
    """POD-Galerkin rollout should preserve snapshot shape and remain finite."""
    cfg = BurgersSolverConfig(nx=32, t_end=0.04, snapshot_dt=0.02, cfl=0.2)
    case = BurgersCase(nu=0.01, initial_condition="sinusoidal")
    x, t, u = solve_case(cfg, case)
    pod = PODModel.fit(u, rank=3)
    rom = PODGalerkinBurgers(pod, x, viscosity=case.nu)
    result = rom.rollout(u[0], t, cfl=0.2)

    assert result.states.shape == u.shape
    assert result.coefficients.shape == (len(t), pod.rank)
    assert result.stats.rhs_calls > 0
    assert np.all(np.isfinite(result.states))
