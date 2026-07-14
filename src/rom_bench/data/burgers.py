"""1D viscous Burgers finite-volume solver and dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Boundary = Literal["periodic", "dirichlet", "neumann"]
InitialCondition = Literal["sinusoidal", "gaussian", "tanh_front", "shock", "random_smooth"]


@dataclass(frozen=True)
class BurgersCase:
    """Parameters for one Burgers trajectory."""

    nu: float
    amplitude: float = 1.0
    front_location: float = 0.25
    front_width: float = 0.03
    initial_condition: InitialCondition = "tanh_front"


@dataclass(frozen=True)
class BurgersSolverConfig:
    """Numerical settings for Burgers data generation."""

    nx: int = 256
    length: float = 1.0
    t_start: float = 0.0
    t_end: float = 0.5
    snapshot_dt: float = 0.01
    cfl: float = 0.35
    boundary_condition: Boundary = "periodic"
    seed: int = 42


def make_grid(cfg: BurgersSolverConfig) -> tuple[np.ndarray, float]:
    """Return cell centers and grid spacing."""
    dx = cfg.length / cfg.nx
    return (np.arange(cfg.nx) + 0.5) * dx, dx


def initial_condition(x: np.ndarray, case: BurgersCase, seed: int = 0) -> np.ndarray:
    """Create a configured initial condition."""
    if case.initial_condition == "sinusoidal":
        return case.amplitude * (
            np.sin(2.0 * np.pi * x)
            + 0.25 * np.sin(4.0 * np.pi * x)
            + 0.10 * np.cos(6.0 * np.pi * x)
        )
    if case.initial_condition == "gaussian":
        return case.amplitude * np.exp(-0.5 * ((x - case.front_location) / case.front_width) ** 2)
    if case.initial_condition in {"tanh_front", "shock"}:
        return 0.5 * case.amplitude * (1.0 - np.tanh((x - case.front_location) / case.front_width))
    if case.initial_condition == "random_smooth":
        rng = np.random.default_rng(seed)
        u = np.zeros_like(x)
        for k in range(1, 6):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            weight = rng.normal(scale=1.0 / k**2)
            u += weight * np.sin(2.0 * np.pi * k * x + phase)
        return case.amplitude * u
    raise ValueError(f"Unknown initial condition: {case.initial_condition}")


def _ghost_cells(u: np.ndarray, boundary: Boundary) -> tuple[float, float]:
    """Return left and right ghost cell values."""
    if boundary == "periodic":
        return float(u[-1]), float(u[0])
    if boundary == "dirichlet":
        return float(u[0]), float(u[-1])
    if boundary == "neumann":
        return float(u[0]), float(u[-1])
    raise ValueError(f"Unknown boundary condition: {boundary}")


def burgers_rhs(u: np.ndarray, dx: float, nu: float, boundary: Boundary = "periodic") -> np.ndarray:
    """Compute finite-volume RHS with Rusanov convection and central diffusion."""
    left_ghost, right_ghost = _ghost_cells(u, boundary)
    padded = np.concatenate(([left_ghost], u, [right_ghost]))

    ul = padded[:-1]
    ur = padded[1:]
    fl = 0.5 * ul**2
    fr = 0.5 * ur**2
    speed = np.maximum(np.abs(ul), np.abs(ur))
    flux = 0.5 * (fl + fr) - 0.5 * speed * (ur - ul)
    convection = -(flux[1:] - flux[:-1]) / dx

    diffusion = nu * (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / dx**2
    return convection + diffusion


def ssp_rk3_step(u: np.ndarray, dt: float, rhs) -> np.ndarray:
    """Third-order strong-stability-preserving Runge-Kutta step."""
    u1 = u + dt * rhs(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2))


def stable_dt(u: np.ndarray, dx: float, nu: float, cfl: float, max_dt: float) -> float:
    """Choose a stable explicit time step."""
    advective = dx / (np.max(np.abs(u)) + 1.0e-12)
    diffusive = 0.5 * dx**2 / (nu + 1.0e-12)
    return min(max_dt, cfl * min(advective, diffusive))


def solve_case(
    cfg: BurgersSolverConfig,
    case: BurgersCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve one Burgers case and return x, t, u[nt, nx]."""
    x, dx = make_grid(cfg)
    u = initial_condition(x / cfg.length, case, cfg.seed)
    time = cfg.t_start
    snapshots: list[np.ndarray] = []
    times: list[float] = []

    def rhs(v: np.ndarray) -> np.ndarray:
        return burgers_rhs(v, dx, case.nu, cfg.boundary_condition)

    snapshot_times = np.arange(cfg.t_start, cfg.t_end + 0.5 * cfg.snapshot_dt, cfg.snapshot_dt)
    if snapshot_times[-1] < cfg.t_end:
        snapshot_times = np.append(snapshot_times, cfg.t_end)

    for target_time in snapshot_times:
        while time < target_time - 1.0e-12:
            dt = stable_dt(u, dx, case.nu, cfg.cfl, target_time - time)
            u = ssp_rk3_step(u, dt, rhs)
            time += dt
        time = float(target_time)
        snapshots.append(u.copy())
        times.append(time)

    out = np.asarray(snapshots)
    validate_solution(x, np.asarray(times), out)
    return x, np.asarray(times), out


def build_cases(config: dict) -> list[BurgersCase]:
    """Build Burgers cases from config."""
    cases: list[BurgersCase] = []
    for item in config.get("cases", []):
        cases.append(
            BurgersCase(
                nu=float(item.get("nu", config.get("viscosity", 0.01))),
                amplitude=float(item.get("amplitude", 1.0)),
                front_location=float(item.get("front_location", 0.25)),
                front_width=float(item.get("front_width", 0.03)),
                initial_condition=item.get("initial_condition", config.get("initial_condition", "tanh_front")),
            )
        )
    if cases:
        return cases

    if "case_grid" in config:
        grid = config["case_grid"]
        rng = np.random.default_rng(int(config.get("seed", 42)))

        def uniform(name: str, default: list[float]) -> float:
            low, high = grid.get(name, default)
            return float(rng.uniform(float(low), float(high)))

        smooth_ics = grid.get("smooth_initial_conditions", ["sinusoidal", "random_smooth"])
        front_ics = grid.get("front_initial_conditions", ["tanh_front", "shock"])
        for _ in range(int(grid.get("smooth_count", 0))):
            cases.append(
                BurgersCase(
                    nu=uniform("smooth_nu_range", [0.012, 0.03]),
                    amplitude=uniform("amplitude_range", [0.8, 1.2]),
                    front_location=uniform("front_location_range", [0.15, 0.45]),
                    front_width=uniform("smooth_width_range", [0.06, 0.12]),
                    initial_condition=str(rng.choice(smooth_ics)),
                )
            )
        for _ in range(int(grid.get("front_count", 0))):
            cases.append(
                BurgersCase(
                    nu=uniform("front_nu_range", [0.001, 0.004]),
                    amplitude=uniform("amplitude_range", [0.8, 1.2]),
                    front_location=uniform("front_location_range", [0.15, 0.45]),
                    front_width=uniform("front_width_range", [0.015, 0.04]),
                    initial_condition=str(rng.choice(front_ics)),
                )
            )
        if cases:
            return cases

    cases.append(
        BurgersCase(
            nu=float(config.get("viscosity", 0.01)),
            initial_condition=config.get("initial_condition", "tanh_front"),
        )
    )
    return cases


def generate_dataset(solver_cfg: BurgersSolverConfig, cases: list[BurgersCase]) -> dict[str, np.ndarray]:
    """Generate a multi-case dataset in the requested HDF5 layout."""
    x0, t0, u0 = solve_case(solver_cfg, cases[0])
    data = np.empty((len(cases), len(t0), solver_cfg.nx), dtype=np.float64)
    data[0] = u0
    for i, case in enumerate(cases[1:], start=1):
        x, t, u = solve_case(solver_cfg, case)
        if not (np.allclose(x, x0) and np.allclose(t, t0)):
            raise ValueError("All cases must share the same x and t grids")
        data[i] = u

    n_cases = len(cases)
    train_end = max(1, int(0.6 * n_cases))
    val_end = max(train_end + 1, int(0.8 * n_cases)) if n_cases > 2 else train_end
    train = np.arange(0, min(train_end, n_cases), dtype=np.int64)
    val = np.arange(train[-1] + 1, min(val_end, n_cases), dtype=np.int64) if n_cases > 1 else np.array([], dtype=np.int64)
    test = np.setdiff1d(np.arange(n_cases, dtype=np.int64), np.concatenate([train, val]))

    return {
        "x": x0,
        "t": t0,
        "u": data,
        "parameters/nu": np.array([c.nu for c in cases]),
        "parameters/amplitude": np.array([c.amplitude for c in cases]),
        "parameters/front_location": np.array([c.front_location for c in cases]),
        "parameters/front_width": np.array([c.front_width for c in cases]),
        "split/train_indices": train,
        "split/val_indices": val,
        "split/test_indices": test,
    }


def validate_solution(x: np.ndarray, t: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """Validate generated solution and return summary statistics."""
    if not np.all(np.isfinite(u)):
        raise ValueError("Burgers solution contains NaN or Inf")
    if x.ndim != 1 or t.ndim != 1 or u.shape != (len(t), len(x)):
        raise ValueError("Burgers solution has inconsistent shapes")
    dt = np.diff(t)
    if len(dt) and np.any(dt <= 0.0):
        raise ValueError("Snapshot times must be strictly increasing")
    return {
        "min": float(np.min(u)),
        "max": float(np.max(u)),
        "mean_initial": float(np.mean(u[0])),
        "mean_final": float(np.mean(u[-1])),
    }
