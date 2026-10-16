.PHONY: install lint test data train serve up down clean format

# ── Variables ─────────────────────────────────────────────────────────
PYTHON   ?= python
PIP      ?= pip
PYTEST   ?= $(PYTHON) -m pytest
RUFF     ?= $(PYTHON) -m ruff
MYPY     ?= $(PYTHON) -m mypy
CONFIG   ?= configs/default.yaml

# ── Setup ─────────────────────────────────────────────────────────────
install:  ## Install package in editable mode with all optional deps
	$(PIP) install -e ".[all]"

install-dev:  ## Install only core + dev deps (fast, no torch)
	$(PIP) install -e ".[dev]"

# ── Code quality ──────────────────────────────────────────────────────
lint:  ## Run ruff linter + mypy type checker
	$(RUFF) check src/ tests/
	$(MYPY) src/fraudgnn/

format:  ## Auto-format code with ruff
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

# ── Tests ─────────────────────────────────────────────────────────────
test:  ## Run pytest test suite
	$(PYTEST) tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -v --tb=short --cov=fraudgnn --cov-report=term-missing

# ── Pipeline ──────────────────────────────────────────────────────────
data:  ## Generate synthetic transaction data
	$(PYTHON) -m fraudgnn.data.generate --config $(CONFIG)

train:  ## Train the model
	$(PYTHON) -m fraudgnn.training.train --config $(CONFIG)

serve:  ## Start the FastAPI inference server
	$(PYTHON) -m uvicorn fraudgnn.serving.app:app --host 0.0.0.0 --port 8000 --reload

# ── Docker ────────────────────────────────────────────────────────────
up:  ## Start all services via docker-compose
	docker compose up -d

down:  ## Stop all services
	docker compose down -v

# ── Cleanup ───────────────────────────────────────────────────────────
clean:  ## Remove build artifacts, caches, and generated data
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ mlruns/

# ── Help ──────────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
