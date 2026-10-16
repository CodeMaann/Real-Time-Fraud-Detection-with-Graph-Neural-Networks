# Design Decisions

This document records major design choices for the FraudGNN project.

## 1. Package Layout — `src`-layout

**Choice**: `src/fraudgnn/` with `pyproject.toml` + Hatch build backend.

**Rationale**: The `src`-layout prevents accidental imports from the working
directory rather than the installed package.  `hatchling` is a minimal,
standards-compliant build backend that avoids `setup.py` boilerplate.

## 2. Configuration — YAML + Pydantic

**Choice**: YAML config files validated at load time by Pydantic `BaseModel`
subclasses with `Field` constraints.

**Rationale**: YAML is human-readable and diff-friendly. Pydantic gives us
runtime type checking, sensible defaults, and auto-generated JSON Schema.
This ensures no magic numbers leak into code and every experiment is
reproducible via its config snapshot.

## 3. Temporal Train/Val/Test Splits

**Choice**: Transactions are split by timestamp, **never** shuffled randomly.

**Rationale**: Random splits on transaction data create severe data leakage
because the model sees future fraud patterns during training.  Temporal splits
mirror real-world deployment where the model scores unseen future transactions.

## 4. Primary Metric — PR-AUC

**Choice**: Precision-Recall AUC is the primary metric; ROC-AUC, F1,
recall@1%FPR, and inference latency are always co-reported.

**Rationale**: Fraud detection is a highly imbalanced classification task
(~2% fraud).  ROC-AUC can be misleadingly high when the negative class
dominates. PR-AUC focuses on the minority (positive) class and is a more
informative metric for imbalanced problems.

## 5. CPU-First, GPU-Optional

**Choice**: All code paths must run on CPU. GPU acceleration is opt-in.

**Rationale**: Hiring managers and reviewers often lack GPU hardware. A
CPU-runnable project is more accessible and still demonstrates
production-grade engineering.

## 6. Dependency Groups

**Choice**: Three optional dependency groups — `[train]`, `[serve]`, `[dev]`.

**Rationale**: Production serving environments do not need training
dependencies (e.g., torch-geometric, MLflow), and CI does not need the
serving stack. Splitting groups keeps Docker images lean and install times
fast.

## 7. Synthetic Transaction Generator & Vectorization

**Choice**: Vectorized generation with `numpy.random.default_rng` generating
users, devices, circadian timestamps, and log-normal spending distributions in
batches.

**Rationale**: Generating 500k to 1M rows iteratively in Python loops can take
several minutes and consumes significant memory. Vectorized arrays allow the
generator to synthesize 500k transactions in ~3-5 seconds on a CPU laptop,
satisfying the CPU-first rule.

## 8. Benign Graph Noise

**Choice**: Injected benign shared devices (households, office kiosks) and
shared IPs (corporate VPNs, public Wi-Fi).

**Rationale**: In a naive graph simulation, any shared device or IP between
accounts is 100% correlated with a fraud ring. In real financial networks,
benign multi-user devices are common. Incorporating benign graph noise prevents
the GNN from degenerating into a simple connected-component lookup and forces it
to combine graph topology with temporal and tabular features.

## 9. Hidden Meta-Label (`fraud_type`)

**Choice**: The `fraud_type` column specifies the injected typology
(`fraud_ring`, `mule`, `card_testing`, `account_takeover`, `merchant_collusion`)
and is strictly marked for evaluation slicing and explainability only.

**Rationale**: In production, fraud types are not available at inference time.
The simulator maintains this hidden column for slicing model performance (e.g.,
evaluating recall on card testing vs. fraud rings separately) while keeping the
feature space free of label leakage.

## 10. Parquet File Format with PyArrow

**Choice**: Raw transactions are stored in PyArrow Parquet format with snappy
compression.

**Rationale**: Parquet provides column-oriented storage, preserving explicit
data types (datetime64, string, float64), reducing disk usage from ~100MB
(CSV) to ~18MB, and enabling fast streaming or sub-graph slicing.

## 11. Heterogeneous Graph Schema & Core Transaction Edge

**Choice**: PyG `HeteroData` with 5 node types (`user`, `card`, `merchant`,
`device`, `ip`). The core financial transaction event is modeled on the
directed relation `("user", "transacts_at", "merchant")` carrying
multidimensional continuous and cyclical edge attributes, binary fraud labels,
and microsecond timestamps. Bidirectional relational edges connect users to
their cards, devices, and IPs.

**Rationale**: Multiplex bipartite and k-partite edges enable GNN attention
layers (e.g., HeteroGAT, GraphSAGE) to aggregate structural signals across
shared infrastructure (such as fraud rings sharing devices) while keeping
transaction-level features explicit on the transaction edge.

## 12. Zero-Leakage Temporal Message Passing

**Choice**: The training graph `train_data` is constructed exclusively from
transactions occurring on or before `train_cutoff` (first 60% of timeline). Zero
validation or test edges exist in `train_data`.

**Rationale**: Unlike tabular models where masking is sufficient, GNN message
passing propagates node embeddings across edges. If test-period edges were
present in the training graph during message passing, test information would
leak into node representations, producing artificially inflated benchmark
results. Our strict inductive temporal graph partitioning guarantees zero
lookahead bias.

## 13. Causal Feature Engineering

**Choice**: Rolling features (counts, sums over 1h, 24h, 7d) are computed with
`closed='left'`, strictly excluding the current transaction. Device sharing
and merchant risk rates are computed expanding-only over past records using
Empirical Bayes smoothing.

**Rationale**: Even a one-second lookahead or inclusion of the current event in
rolling aggregates introduces target/feature leakage that fails in real-time
streaming. Using strictly causal expanding windows guarantees fidelity with
the live production inference service.

## 14. Business Cost Matrix & Operating Threshold Optimization

**Choice**: Model evaluation pairs statistical metrics (PR-AUC, ROC-AUC, F1,
recall@1%FPR) with an explicit financial loss function:
$$\text{Cost} = \text{FN} \times \text{avg\_fraud\_amount} + \text{FP} \times \text{review\_cost}$$

**Rationale**: In fraud operations, false positives incur operational review
costs (~$10/alert) while false negatives incur chargeback and direct fraud
losses (~$500–$1,000/miss). Optimizing solely for F1 (typically at threshold 0.50)
fails to minimize total financial damage. Slicing total loss across the threshold
grid identifies the cost-optimal decision threshold for production deployment.


