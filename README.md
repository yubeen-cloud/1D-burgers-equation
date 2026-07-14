# ROM Fluid Benchmarks

Reduced-Order Modeling benchmark code for:

1. 1D viscous Burgers equation
2. 2D cylinder wake

The project compares POD, DMD, and Autoencoder ROMs with emphasis on failure modes:

- POD difficulty with moving fronts and shock-like structures
- DMD rollout error accumulation
- Autoencoder nonlinear latent representation and latent dynamics failure
- Cylinder wake phase drift in long-horizon prediction
- Reynolds number generalization when multi-Re data are available

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -e .
```

Development install:

```bash
pip install -e ".[dev]"
```

## Burgers Workflow

Generate data:

```bash
python scripts/generate_burgers.py --config configs/burgers_generate.yaml
```

Run POD:

```bash
python scripts/train_pod.py --config configs/burgers_pod.yaml
```

Run DMD:

```bash
python scripts/train_dmd.py --config configs/burgers_dmd.yaml
```

Run Autoencoder:

```bash
python scripts/train_autoencoder.py --config configs/burgers_autoencoder.yaml
```

Compare models:

```bash
python scripts/compare_models.py --problem burgers --results-dir artifacts
```

Full Burgers pipeline:

```bash
python scripts/run_all.py --problem burgers --methods pod dmd autoencoder
```

`run_all.py` reuses cached outputs unless `--force` is passed.

## Cylinder Workflow

Phase 2 currently includes a small synthetic wake dataset and clear interfaces for external data.

Generate synthetic cylinder data:

```bash
python scripts/prepare_cylinder_data.py --config configs/cylinder_data.yaml
```

Run placeholders for method scripts:

```bash
python scripts/train_pod.py --config configs/cylinder_pod.yaml
python scripts/train_dmd.py --config configs/cylinder_dmd.yaml
python scripts/train_autoencoder.py --config configs/cylinder_autoencoder.yaml
```

External cylinder datasets should be converted to the documented HDF5 layout:

- `x`, `y`, `t`
- `velocity/u`, `velocity/v`
- `pressure`
- `vorticity`
- `coefficients/cl`, `coefficients/cd` when available
- `parameters/reynolds_number`, `parameters/u_inf`, `parameters/diameter`
- `split/train_indices`, `split/val_indices`, `split/test_indices`

OpenFOAM outputs should first be converted with `foamToVTK` or `postProcess`, then interpolated to a common grid. VTK/VTU loading is intended as an optional dependency path using `meshio` or `pyvista`.

## Config Overrides

Configs are YAML files. Simple key overrides are supported:

```bash
python scripts/train_autoencoder.py \
  --config configs/burgers_autoencoder.yaml \
  model.latent_dim=4 \
  training.epochs=200
```

## Output Layout

Artifacts are written under:

```text
artifacts/
├── checkpoints/
├── figures/
├── metrics/
├── predictions/
├── logs/
└── reports/
```

Each experiment saves:

- resolved config
- metrics JSON and CSV
- runtime JSON
- predictions NPZ
- report Markdown
- PNG and PDF figures

## Metrics

Burgers metrics include:

- MSE, RMSE, MAE
- absolute and relative L2
- normalized RMSE
- maximum pointwise error
- spatial gradient error
- rollout error over time
- threshold crossing time
- front position error
- front speed error

Cylinder-oriented modules include:

- field relative L2
- FFT peak frequency
- Strouhal number
- coefficient signal metrics when Cl/Cd are available
- phase drift proxy by cross-correlation

## Git And Reproducibility

Recommended branches:

- `main`
- `develop`
- `feature/burgers-solver`
- `feature/pod`
- `feature/dmd`
- `feature/autoencoder`
- `feature/cylinder-data`
- `feature/evaluation`

Track in Git:

- source code
- configs
- tests
- README
- small metadata examples

Do not track by default:

- HDF5 processed datasets
- checkpoints
- generated figures
- prediction arrays
- large logs

Use Git LFS for datasets or checkpoints that must be versioned.

Every HDF5 dataset stores environment information, config path, and git information when `git` is available.

Suggested initial commit order:

1. project skeleton and packaging
2. Burgers solver and data I/O
3. POD and DMD
4. Autoencoder
5. metrics and visualization
6. tests and README
7. cylinder synthetic data scaffold

## Known Limitations

- Linear POD may need many modes to represent moving structures.
- Standard DMD struggles with nonlinear transients and moving shocks in long rollout.
- Good Autoencoder reconstruction does not imply accurate latent dynamics.
- Cylinder force coefficients cannot be accurately reconstructed from fields without wall pressure and shear information.
- Reynolds number generalization needs sufficient parameter coverage.
- Phase 2 includes synthetic cylinder data and interfaces; robust OpenFOAM/VTK ingestion is an extension point.

## Tests

```bash
pytest -q
```

The current smoke tests cover Burgers solver consistency, POD, DMD, metrics, and HDF5 I/O.
