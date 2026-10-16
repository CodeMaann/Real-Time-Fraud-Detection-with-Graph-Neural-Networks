# Real-Time Fraud Detection with Graph Neural Networks

Production-grade fraud detection pipeline that combines **Graph Neural Networks**
(GraphSAGE, Heterogeneous GAT) with **XGBoost** baselines over a streaming
architecture (Kafka/Redpanda → GNN inference → scored stream).

## Quick Start

```bash
# 1. Create a virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 2. Install (CPU-first — no GPU required)
pip install -e ".[all]"

# 3. Verify
python -m pytest tests/ -v
python -m ruff check src/ tests/
python -m mypy src/fraudgnn/
```

## Project Structure

```
├── configs/              # YAML configuration files (no magic numbers in code)
├── data/                 # Raw and processed data (git-ignored)
├── docs/                 # Design decisions, architecture diagrams
├── dashboard/            # Streamlit real-time monitoring dashboard
├── notebooks/            # Exploratory analysis and experiments
├── scripts/              # Utility scripts (data download, deployment, etc.)
├── src/fraudgnn/         # Main package (src-layout)
│   ├── config.py         # YAML → Pydantic config loader
│   ├── data/             # Data generation and preprocessing
│   ├── graph/            # Heterogeneous graph construction
│   ├── models/           # GNN and XGBoost model definitions
│   ├── training/         # Training loops, evaluation, MLflow logging
│   ├── explain/          # Model explainability (SHAP, GNN-Explainer)
│   ├── serving/          # FastAPI inference service
│   ├── streaming/        # Kafka/Redpanda stream processing
│   └── monitoring/       # Prometheus metrics, drift detection
├── tests/                # pytest test suite
├── pyproject.toml        # Package metadata and tool configuration
├── Makefile              # Common development commands
└── docker-compose.yml    # Infrastructure services
```

## Makefile Targets

| Target    | Description                                  |
|-----------|----------------------------------------------|
| `install` | Install package with all optional deps       |
| `lint`    | Run ruff + mypy                              |
| `test`    | Run pytest                                   |
| `data`    | Generate synthetic transaction data          |
| `train`   | Train the model                              |
| `serve`   | Start FastAPI inference server               |
| `up`      | Start Docker services                        |
| `down`    | Stop Docker services                         |
| `clean`   | Remove build artifacts and caches            |

## Key Design Decisions

See [docs/design-decisions.md](docs/design-decisions.md) for detailed rationale.

- **Temporal splits only** — no random train/test splits (prevents data leakage)
- **PR-AUC primary metric** — appropriate for highly imbalanced fraud detection
- **CPU-first** — everything runs on a laptop, GPU optional
- **Config-driven** — all hyperparameters in YAML, validated by Pydantic

## License

MIT
