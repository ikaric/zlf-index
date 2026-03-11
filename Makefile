SHELL := /bin/bash

UV ?= uv
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install install-dev build-ext test-core test-api test-unit test lint format typecheck check pre-commit benchmark-speed clean
help:
	@echo "Setup:"
	@echo "  venv         Create virtualenv with uv in $(VENV)"
	@echo "  install      Install package (production) into venv"
	@echo "  install-dev  Install package + dev dependencies into venv"
	@echo "  build-ext    Build Cython extension in-place"
	@echo ""
	@echo "Development:"
	@echo "  test-core               Run fundamental logic tests (test_01_core.py)"
	@echo "  test-api                Run Python API wrapper tests (test_02_api.py)"
	@echo "  test-unit               Run core and api checks together"
	@echo "  test                    Run entire test suite with coverage"
	@echo "  benchmark-speed         Run synthetic speed benchmark to .benchmarks/benchmark_speed_results.txt"
	@echo "  lint                    Lint code with ruff"
	@echo "  format                  Format code with ruff"
	@echo "  typecheck               Type check with mypy"
	@echo "  check                   Run lint, typecheck, and test-unit"
	@echo "  pre-commit              Run pre-commit hooks on all files"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        Remove build artifacts and venv"

venv:
	$(UV) venv $(VENV)

install: venv
	$(UV) pip install -e .

install-dev: venv
	$(UV) pip install -e ".[dev]"

build-ext:
	SUFFIX25_NATIVE=1 $(UV) run python setup.py build_ext --inplace

test-core: build-ext
	$(UV) run --extra dev pytest tests/test_01_core.py

test-api: build-ext
	$(UV) run --extra dev pytest tests/test_02_api.py

test-unit: build-ext
	$(UV) run --extra dev pytest tests/test_01_core.py tests/test_02_api.py

test: build-ext
	$(UV) run --extra dev pytest --cov=suffix25

benchmark-speed: build-ext
	mkdir -p .benchmarks && $(UV) run benchmarks/benchmark_01_speed.py | tee .benchmarks/benchmark_speed_results.txt

lint:
	$(UV) run --extra dev ruff check src/ tests/

format:
	$(UV) run --extra dev ruff format src/ tests/

typecheck:
	$(UV) run --extra dev mypy src/

check: lint typecheck test-unit

pre-commit:
	$(UV) run --extra dev pre-commit run --all-files

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .coverage coverage.xml htmlcov .ruff_cache .benchmarks/ src/suffix25/_core.c src/suffix25/_core*.so

