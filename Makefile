# Developer convenience tasks. Run `make help` to list them.
.PHONY: help install format lint typecheck test check run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install the project with dev extras
	python -m pip install -e ".[dev]"

format:  ## Auto-format the code with black
	black .

lint:  ## Lint with ruff
	ruff check .

typecheck:  ## Type-check the src package with mypy
	mypy src

test:  ## Run the test suite
	pytest

check: lint typecheck test  ## Run lint + type-check + tests (what CI runs)
	black --check .

run:  ## Analyse the default tickers (needs network for Yahoo Finance)
	python -m src.main

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache output data_cache \
		*.egg-info src/__pycache__ tests/__pycache__
