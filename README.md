# Real-Time Fraud Detection with Graph Neural Networks

[![CI](https://github.com/CodeMaann/Real-Time-Fraud-Detection-with-Graph-Neural-Networks/actions/workflows/ci.yml/badge.svg)](https://github.com/CodeMaann/Real-Time-Fraud-Detection-with-Graph-Neural-Networks/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Fraud detection on a heterogeneous transaction graph (users, cards, merchants, devices, IPs),
built to answer one honest question: **does graph structure catch fraud that tabular models miss,
and does a GNN beat hand-crafted graph features once evaluation is leak-free?**

> **Status: work in progress.** Data simulation, graph construction, baselines and GNN training are done
> or in progress. Streaming, serving, explainability and the dashboard are planned (see roadmap).

## Why graphs?

Fraud rings and mule accounts hide in relationships: shared devices, shared IPs, money moving through
connected accounts. A model that scores each transaction in isolation cannot see these patterns.

## What is built so far

- **Fraud simulator** with five typologies: fraud rings, mule accounts, card testing, account takeover,
  merchant collusion. It also includes benign noise (families and offices sharing devices, public Wi-Fi)
  and 1.5% label noise.
- **Heterogeneous graph builder** (PyTorch Geometric `HeteroData`) with strict temporal train/val/test splits.
- **Causal feature engineering**: rolling windows, device sharing and merchant risk use only past data.
- **Baselines**: XGBoost on tabular features, XGBoost plus graph features (with and without label-derived counts).
- **GNNs**: GraphSAGE and a Heterogeneous GAT, trained with causal temporal neighbor sampling.
- **Leak-free evaluation**: thresholds chosen on validation only, never on test; PR-AUC as primary metric;
  business-cost metric; per-fraud-type recall.
- **Quality gates**: ruff, mypy, pytest and CI on every push.

## Findings so far

All numbers are real runs on the hardened simulator (500k transactions, 90 days, temporal split).

| Model | Test PR-AUC | Recall @ 1% FPR |
|---|---:|---:|
| XGBoost, tabular features only | 0.354 | 0.422 |
| XGBoost + graph features (label-free) | 0.595 | 0.683 |
| XGBoost + graph features (uses past fraud labels) | 0.686 | 0.743 |
| GraphSAGE / HeteroGAT (causal sampling) | _in progress_ | _in progress_ |

Key lessons (details in [docs/design-decisions.md](docs/design-decisions.md)):

1. **The first simulator was too easy.** A plain XGBoost got 0.935 PR-AUC from transaction amount alone,
   which makes a graph comparison meaningless. I diagnosed it with feature importance and per-fraud-type
   recall, then hardened the simulator until tabular PR-AUC dropped to 0.35.
2. **Graph structure matters.** Fraud-ring recall rises from 4.5% (tabular) to 68% with graph features.
3. **Evaluation leaks are easy to miss.** Beyond the usual random-split leak, a transaction can see later
   edges inside its own time window. I replaced standard neighbor sampling with causal temporal sampling and
   added a test that injects a future transaction and checks the output does not change.
4. **Label-derived features are optimistic.** Counting past known fraud on a device is strong, but real
   chargeback labels arrive weeks late, so I report a label-free baseline as the fair opponent.

## Architecture

```
 Simulator -> Graph builder -> Training (XGBoost, GraphSAGE, HeteroGAT) -> MLflow
   [done]        [done]                  [in progress]

 Kafka/Redpanda -> stream processor -> inference API (FastAPI) -> scored stream
                                                                  -> Prometheus/Grafana, dashboard
   [planned]           [planned]              [planned]                   [planned]
```

**Graph schema**

| Node | Edge (relation) | Edge features |
|---|---|---|
| user, card, merchant, device, ip | user-card, user-device, user-ip, card-merchant, user-merchant (transaction) | amount, time-of-day, time deltas, rolling-window counts and sums (27 causal features) |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e ".[train,dev]"

python -m fraudgnn.data.generate --config configs/data.yaml      # simulate transactions
python -m fraudgnn.graph.builder --config configs/graph.yaml     # build temporal graphs
python -m fraudgnn.models.baseline --config configs/baseline.yaml  # XGBoost baseline

python -m pytest tests/ -v
python -m ruff check src/ tests/
python -m mypy src/fraudgnn/
```

## Project structure

```
configs/        YAML configs (no magic numbers in code)
docs/           design decisions and limitations
scripts/        diagnostics, graph-signal check, label-free baseline
src/fraudgnn/
  data/         simulator, causal feature engineering
  graph/        graph builder, loaders, causal temporal loader
  models/       XGBoost, GraphSAGE, HeteroGAT, losses
  training/     training loops, evaluation, MLflow logging
tests/          pytest suite (leakage, causality, models, simulator)
```

## Design decisions

- **Temporal splits only**, never random.
- **PR-AUC** as the primary metric (about 0.8% fraud rate), plus recall at 1% FPR and a business-cost metric.
- **Thresholds chosen on validation**, then applied to test.
- **Causal everything**: features, node statistics and neighbor sampling only use data before the target transaction.
- **CPU-first**, config-driven, seeded and logged to MLflow.

## Limitations

Synthetic data: the simulator difficulty was tuned to land in a target range, so absolute scores are not
comparable to real-world systems. Label delay for the merchant risk feature is simplified. Account takeover
is mostly undetectable by design. See [docs/design-decisions.md](docs/design-decisions.md).

## Roadmap

- [x] Project scaffold, config system, CI
- [x] Fraud simulator and heterogeneous graph builder
- [x] Causal features and XGBoost baselines
- [ ] GraphSAGE / HeteroGAT final results with degree-aware message passing
- [ ] Explainability (GNNExplainer, SHAP)
- [ ] Real-time serving (FastAPI, Kafka/Redpanda, Redis) with latency benchmarks
- [ ] Monitoring and drift detection (Prometheus, Grafana, Evidently)
- [ ] Streamlit dashboard

## License

MIT