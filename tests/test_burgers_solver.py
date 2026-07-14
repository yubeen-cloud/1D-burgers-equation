from __future__ import annotations

import numpy as np

from rom_bench.data.burgers import BurgersCase, BurgersSolverConfig, burgers_rhs, solve_case


def test_constant_field_remains_constant_rhs() -> None:
    u = np.ones(32) * 0.7
    rhs = burgers_rhs(u, dx=1.0 / 32, nu=0.01, boundary="periodic")
    assert np.allclose(rhs, 0.0)


def test_solver_has_expected_shape_and_no_nan() -> None:
    cfg = BurgersSolverConfig(nx=64, t_end=0.05, snapshot_dt=0.01, cfl=0.3)
    case = BurgersCase(nu=0.01, initial_condition="sinusoidal")
    x, t, u = solve_case(cfg, case)
    assert x.shape == (64,)
    assert u.shape == (len(t), len(x))
    assert np.all(np.isfinite(u))


def test_periodic_rhs_shape() -> None:
    u = np.sin(2 * np.pi * np.linspace(0, 1, 32, endpoint=False))
    rhs = burgers_rhs(u, dx=1.0 / 32, nu=0.01, boundary="periodic")
    assert rhs.shape == u.shape
