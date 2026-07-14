# 1D Burgers Equation ROM

Reduced-Order Modeling experiments for the 1D viscous Burgers equation.

The project compares POD, DMD, and Conv1D Autoencoder approaches on Burgers data, with emphasis on observing how each method behaves around moving fronts and shock-like structures. The goal is comparison and interpretation, not ranking methods.

## Main Questions

- How does POD reconstruction behave when a moving front is represented by a finite linear basis?
- How does DMD rollout error accumulate after the training time window?
- How does a Conv1D Autoencoder represent the field in a nonlinear latent space?
- How do reconstruction error, rollout error, front-local error, and gradient error tell different stories?
- How do smooth-like and shock-like cases differ within the same PDEBench dataset?

## Project Layout

```text
configs/burgers/          YAML experiment configs
data/external/pdebench/   Public PDEBench source files
data/processed/burgers/   Processed Burgers HDF5 datasets
artifacts/burgers/        Figures, metrics, predictions, reports, checkpoints
scripts/                  CLI scripts for data, training, evaluation, review PDFs
src/rom_bench/            Reusable Burgers ROM modules
tests/                    Pytest smoke tests
notebooks/                Optional Burgers exploration notebook
```

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

## Public PDEBench Burgers Data

The public-data workflow uses:

```text
data/external/pdebench/1D_Burgers_Sols_Nu0.01.hdf5
```

and converts a reproducible subset to:

```text
data/processed/burgers/pdebench_burgers_nu0.01_subset.h5
```

Prepare the processed dataset:

```bash
python scripts/prepare_pdebench_burgers.py --config configs/burgers/pdebench_prepare.yaml
```

The processed subset currently stores:

- `x`
- `t`
- `u`
- `parameters/nu`
- `split/train_indices`
- `split/val_indices`
- `split/test_indices`
- metadata describing the source file, subset rule, package versions, and config path

## Core Experiments

Run POD:

```bash
python scripts/train_pod.py --config configs/burgers/pdebench_pod.yaml
```

Run DMD:

```bash
python scripts/train_dmd.py --config configs/burgers/pdebench_dmd.yaml
```

Run Conv1D Autoencoder:

```bash
python scripts/train_autoencoder.py --config configs/burgers/pdebench_autoencoder.yaml
```

Compare saved metrics:

```bash
python scripts/compare_models.py --problem burgers --results-dir artifacts
```

Generate the detailed review PDF:

```bash
python scripts/create_burgers_logic_review.py
```

## Smooth/Shock And Dimension Sweep Comparison

To compare smooth-like and shock-like cases within the same PDEBench subset, run:

```bash
python scripts/analyze_burgers_failure_modes.py
```

This creates:

```text
artifacts/burgers/metrics/pdebench_failure_mode_sweep_metrics.csv
artifacts/burgers/metrics/pdebench_failure_mode_summary.json
artifacts/burgers/figures/failure_modes/pdebench_smooth_shock_failure_modes/
```

The comparison includes:

- smooth-like and shock-like case selection by mean max `|du/dx|`
- front-local overlay figures
- POD rank sweep
- AE latent-dimension sweep
- a controlled comparison where POD and AE use the same temporal training snapshots

For smooth-like cases, front MAE is reported as `N/A` in the review table because a sharp front is not a meaningful tracked object there.

## Metrics

Implemented metrics include:

- MSE, RMSE, MAE
- absolute and relative L2 error
- normalized RMSE
- maximum pointwise error
- spatial gradient error
- rollout error over time
- threshold crossing time
- front position error for shock-like cases
- front speed error when applicable

## Outputs

Most generated outputs are under:

```text
artifacts/burgers/
├── checkpoints/
├── figures/
├── metrics/
├── predictions/
└── reports/
```

Important review artifact:

```text
artifacts/burgers/reports/burgers_logic_review.pdf
```

## Config Overrides

Configs are YAML files. Simple key overrides are supported:

```bash
python scripts/train_autoencoder.py \
  --config configs/burgers/pdebench_autoencoder.yaml \
  model.latent_dim=16 \
  training.epochs=300
```

## Reproducibility

The project stores resolved configs, metrics, predictions, figures, and metadata where possible. HDF5 metadata records the source dataset, subset rule, environment information, and config path.

Recommended Git tracking:

- source code
- configs
- tests
- README
- small metadata files

Usually exclude:

- large HDF5 datasets
- checkpoints
- generated predictions
- generated figures
- generated logs

Use Git LFS if large public-data subsets or checkpoints must be versioned.

## Known Limitations

- Linear POD may need many modes to represent moving fronts.
- Standard DMD uses a linear time-advance approximation and can accumulate rollout error for nonlinear Burgers dynamics.
- Autoencoder reconstruction quality does not automatically imply accurate latent dynamics.
- Front tracking is meaningful mainly for shock-like cases with a clear dominant gradient.
- Downsampling from the original PDEBench resolution can weaken measured front gradients.

## Tests

```bash
pytest -q
```

The current tests cover Burgers solver consistency, POD, DMD, field metrics, front-distance logic, FFT peak detection, and HDF5 I/O.
