# 1D Burgers Equation ROM

Reproducible POD, DMD, and Conv1D autoencoder analysis of the public PDEBench
1D viscous Burgers dataset. The final workflow focuses on moving-front and
shock-like behavior, rollout error, front tracking, and model limitations.

## Project Layout

```text
configs/burgers/          Final public-data experiment configs
data/external/pdebench/   Public PDEBench source file
data/processed/burgers/   Reproducible processed HDF5 subset
checkpoints/              Autoencoder checkpoints
figures/                  Data, model, comparison, and failure-mode figures
metrics/                  JSON and CSV metrics
predictions/              Saved POD, DMD, and AE predictions
reports/                  Final PDF and generated Markdown summaries
scripts/                  Reproducible command-line workflow
src/rom_bench/            Reusable ROM modules
tests/                    Unit and numerical-consistency tests
notebooks/                Optional exploration notebook
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Public Dataset

The workflow uses:

```text
data/external/pdebench/1D_Burgers_Sols_Nu0.01.hdf5
```

Prepare the 64-case, time/space-strided subset with:

```bash
python scripts/prepare_pdebench_burgers.py --config configs/burgers/pdebench_prepare.yaml
```

If the public source file is missing, the preparer downloads it from the
configured PDEBench data URL before conversion.

The processed dataset is written to
`data/processed/burgers/pdebench_burgers_nu0.01_subset.h5` with coordinates,
solution fields, viscosity, split indices, source metadata, and resolved config.

## Final Workflow

Run the complete public-data analysis with:

```bash
make all
```

Equivalent command:

```bash
python scripts/run_all.py --config configs/burgers/run_all.yaml
```

The ordered workflow performs data preparation, POD, DMD, autoencoder
training, model comparison, smooth/shock failure-mode analysis, and final PDF
generation. Individual stages are available through `make data`, `make pod`,
`make dmd`, `make ae`, `make compare`, `make failure`, and `make review`.

## Analysis Scope

- POD rank sensitivity and moving-front reconstruction error
- DMD free-rollout error and front-position drift
- Conv1D autoencoder reconstruction and latent linear rollout
- Smooth-like versus shock-like case selection by mean maximum gradient
- Channel-consistent POD-rank and AE-latent-dimension comparison
- Global relative L2, front-local error, gradient error, and runtime

Front-position metrics are interpreted only for shock-like cases with a clear
dominant gradient. Autoencoder reconstruction quality is not treated as proof
of accurate latent dynamics.

## Final Outputs

```text
checkpoints/
figures/
metrics/
predictions/
reports/
  burgers_logic_review.pdf
```

## Config Overrides

```bash
python scripts/train_autoencoder.py \
  --config configs/burgers/pdebench_autoencoder.yaml \
  model.latent_dim=16 training.epochs=300
```

## Verification

```bash
pytest -q
ruff check src scripts tests
```
