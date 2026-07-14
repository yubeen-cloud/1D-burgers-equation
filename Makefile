.PHONY: install dev test lint burgers cylinder

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src scripts tests
	mypy src

burgers:
	python scripts/run_all.py --problem burgers --methods pod dmd autoencoder

cylinder:
	python scripts/run_all.py --problem cylinder --methods pod dmd autoencoder
