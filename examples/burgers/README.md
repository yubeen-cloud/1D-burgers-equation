# Burgers Example

This folder is the quick entry point for the 1D Burgers equation experiments.

Main configs:

```text
configs/burgers/generate.yaml
configs/burgers/pod.yaml
configs/burgers/dmd.yaml
configs/burgers/autoencoder.yaml
```

Run the full Burgers pipeline:

```bash
python scripts/run_all.py --problem burgers --methods pod dmd autoencoder
```

Run step by step:

```bash
python scripts/generate_burgers.py --config configs/burgers/generate.yaml
python scripts/train_pod.py --config configs/burgers/pod.yaml
python scripts/train_dmd.py --config configs/burgers/dmd.yaml
python scripts/train_autoencoder.py --config configs/burgers/autoencoder.yaml
python scripts/compare_models.py --problem burgers --results-dir artifacts
```

Important outputs:

```text
data/processed/burgers/burgers_dataset.h5
artifacts/burgers/figures/
artifacts/burgers/metrics/burgers_model_comparison.csv
artifacts/burgers/reports/burgers_comparison.md
```

Focus:

- moving front or shock-like structure
- POD reconstruction failure around a translating front
- DMD rollout error accumulation
- Autoencoder latent trajectory and latent rollout failure
