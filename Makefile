SHELL := /bin/bash

UV ?= uv
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install install-dev build-ext test-core test-api test-unit test lint format typecheck check pre-commit benchmark-speed benchmark-beir-quick benchmark-beir-full benchmark-mrtydi-quick benchmark-mrtydi-full benchmark-accuracy benchmark-all clean

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
	@echo "  benchmark-speed         Run synthetic speed comparison vs BM25 to .benchmarks/benchmark_speed_results.txt"
	@echo "  benchmark-beir-quick    Preview BEIR 2.0 evaluation on 6 diverse lightweight datasets"
	@echo "  benchmark-beir-full     Execute official BEIR 2.0 evaluation across all 18 datasets (Slow!)"
	@echo "  benchmark-mrtydi-quick  Preview MrTyDi evaluation across all 11 languages (Micro-Sampled)"
	@echo "  benchmark-mrtydi-full   Execute official MrTyDi evaluation across all 11 datasets (Slow!)"
	@echo "  benchmark-accuracy      Run precision/recall tests against BM25 (BEIR/MrTyDi Quick)"
	@echo "  benchmark-all           Run all benchmarks (Speed, BEIR Full, MrTyDi Full)"
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
	ZLFI_NATIVE=1 $(UV) run python setup.py build_ext --inplace

test-core: build-ext
	$(UV) run --extra dev pytest tests/test_01_core.py

test-api: build-ext
	$(UV) run --extra dev pytest tests/test_02_api.py

test-unit: build-ext
	$(UV) run --extra dev pytest tests/test_01_core.py tests/test_02_api.py

test: build-ext
	$(UV) run --extra dev pytest --cov=zlfi

benchmark-speed: build-ext
	mkdir -p .benchmarks && $(UV) run benchmarks/benchmark_01_speed.py | tee .benchmarks/benchmark_speed_results.txt

benchmark-beir-quick: build-ext
	# Note: Now fully aligned with BEIR 2.0 Eval (Recall@1000 + LocalBM25 Regex)
	mkdir -p .benchmarks
	$(UV) run benchmarks/benchmark_02_beir.py --datasets scifact nfcorpus arguana scidocs fiqa trec-covid --log-file .benchmarks/beir.log | tee .benchmarks/benchmark_beir_results.txt

benchmark-beir-full: build-ext
	mkdir -p .benchmarks
	$(UV) run benchmarks/benchmark_02_beir.py --datasets scifact trec-covid nfcorpus nq hotpotqa fiqa arguana webis-touche2020 cqadupstack quora dbpedia-entity scidocs fever climate-fever msmarco-v2 medqa-retrieval legalbench-ir codesearchnet-rag --log-file .benchmarks/beir_full.log | tee .benchmarks/benchmark_beir_full_results.txt

benchmark-mrtydi-quick: build-ext
	mkdir -p .benchmarks
	$(UV) run benchmarks/benchmark_03_mrtydi.py --languages english arabic japanese korean bengali finnish indonesian russian swahili telugu thai --max-queries 5 --max-docs 100 --log-file .benchmarks/mrtydi.log | tee .benchmarks/benchmark_mrtydi_results.txt

benchmark-mrtydi-full: build-ext
	mkdir -p .benchmarks
	$(UV) run benchmarks/benchmark_03_mrtydi.py --languages english arabic japanese korean bengali finnish indonesian russian swahili telugu thai --log-file .benchmarks/mrtydi_full.log | tee .benchmarks/benchmark_mrtydi_full_results.txt

benchmark-accuracy: benchmark-beir-quick benchmark-mrtydi-quick

benchmark-all: benchmark-speed benchmark-beir-full benchmark-mrtydi-full

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
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .coverage coverage.xml htmlcov .ruff_cache .benchmarks/ src/zlfi/_core.c src/zlfi/_core*.so

