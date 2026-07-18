.PHONY: install dev test lint data pod dmd ae compare failure review all

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src scripts tests

data:
	python scripts/prepare_pdebench_burgers.py --config configs/burgers/pdebench_prepare.yaml

pod:
	python scripts/train_pod.py --config configs/burgers/pdebench_pod.yaml

dmd:
	python scripts/train_dmd.py --config configs/burgers/pdebench_dmd.yaml

ae:
	python scripts/train_autoencoder.py --config configs/burgers/pdebench_autoencoder.yaml

compare:
	python scripts/compare_models.py --problem burgers --results-dir .

failure:
	python scripts/analyze_burgers_failure_modes.py

review:
	python scripts/create_burgers_logic_review.py

all:
	python scripts/run_all.py --config configs/burgers/run_all.yaml
