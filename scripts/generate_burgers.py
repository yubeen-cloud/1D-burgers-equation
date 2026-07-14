"""Generate Burgers datasets."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401

from rom_bench.config import parse_config_args, save_yaml
from rom_bench.data.burgers import BurgersSolverConfig, build_cases, generate_dataset, validate_solution
from rom_bench.data.io import environment_info, write_h5, write_json
from rom_bench.paths import resolve_path
from rom_bench.seed import seed_everything
from rom_bench.visualization.burgers_plots import plot_spacetime


def main() -> None:
    config, _args = parse_config_args("Generate Burgers finite-volume data")
    seed = int(config.get("experiment", {}).get("seed", 42))
    seed_everything(seed)
    data_cfg = config["data"]
    solver_raw = config["solver"]
    solver_cfg = BurgersSolverConfig(
        nx=int(solver_raw["nx"]),
        length=float(solver_raw.get("length", 1.0)),
        t_start=float(solver_raw.get("t_start", 0.0)),
        t_end=float(solver_raw["t_end"]),
        snapshot_dt=float(solver_raw["snapshot_dt"]),
        cfl=float(solver_raw.get("cfl", 0.35)),
        boundary_condition=solver_raw.get("boundary_condition", "periodic"),
        seed=seed,
    )
    cases = build_cases(solver_raw)
    arrays = generate_dataset(solver_cfg, cases)
    stats = validate_solution(arrays["x"], arrays["t"], arrays["u"][0])
    metadata = {
        **environment_info(),
        "solver": "finite_volume_rusanov_ssp_rk3",
        "spatial_discretization": "finite_volume_rusanov_flux_plus_central_diffusion",
        "time_integrator": "SSP-RK3",
        "boundary_condition": solver_cfg.boundary_condition,
        "initial_condition": solver_raw.get("initial_condition"),
        "cfl": solver_cfg.cfl,
        "dx": solver_cfg.length / solver_cfg.nx,
        "snapshot_dt": solver_cfg.snapshot_dt,
        "random_seed": seed,
        "config_path": config.get("_config_path"),
        "validation": stats,
    }
    out_path = resolve_path(data_cfg["path"])
    write_h5(out_path, arrays, metadata)
    write_json(out_path.with_suffix(".metadata.json"), metadata)
    save_yaml(config, out_path.with_suffix(".resolved.yaml"))

    fig_dir = resolve_path(config["experiment"].get("output_dir", "artifacts")) / "burgers" / "figures" / "data" / config["experiment"]["name"]
    plot_spacetime(arrays["x"], arrays["t"], arrays["u"][0], "Burgers generated field", fig_dir / "burgers_spacetime_true")
    print(json.dumps({"path": str(out_path), "u_shape": list(arrays["u"].shape), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
