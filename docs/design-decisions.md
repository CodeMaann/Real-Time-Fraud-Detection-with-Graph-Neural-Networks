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

## 15. Benchmark Hardening for Meaningful GNN Comparison (Phase 2.5)

**Context**: In Phase 2, the XGBoost tabular baseline achieved test PR-AUC 0.9353 and recall@1%FPR 0.9872. Diagnostics revealed that tabular feature importance (gain) was overwhelmingly dominated by `log_amount` (gain 5,322) and `amount` (gain 3,078), catching fraud rings with 80.7% recall and mules with 94.1% recall purely through tabular amount and velocity signals. This rendered any GNN comparison meaningless because graph structure was not required to detect fraud.

**Hardening Strategy**:
1. **Normal-Looking Ring Members**: Fraud ring accounts now draw transaction amounts and circadian hours from normal-user distributions. Their individual profiles look benign; the malicious signal resides entirely in shared graph topology (dense bipartite cliques across devices/IPs/merchants) and multi-entity temporal coordination.
2. **Path-Based Mule Accounts**: Mule accounts now spend within their normal account distributions. The signal is in graph chains and flow through cash-out merchants rather than single-transaction dollar spikes.
3. **Reduced Feature Separability for Card Testing and ATO**: Micro-authorizations and cash-out charges were scaled to overlap with legitimate small/large purchases. 25% of ATOs are domestic and 15% reuse existing devices.
4. **Enhanced Benign Graph Noise**: Benign multi-user device sharing was expanded (7,700+ legitimate multi-user devices, shared IP pools, corporate VPNs), preventing graph algorithms from treating any shared node as an automatic indicator of fraud.
5. **Realistic Label Noise**: 1.5% label noise was introduced to reflect real-world imperfect ground truth.
6. **Strict Validation-Only Threshold Selection**: Decision thresholds are strictly selected on the validation partition (maximizing validation F1) and passed via `threshold_override` to test evaluation. No test-set threshold search or re-optimization is permitted.

**Resulting Baseline Operating Benchmark**:
- Validation PR-AUC: 0.3190
- Test PR-AUC: 0.3535 (target range: 0.30–0.75, down from 0.9353)
- Test Recall @ 1% FPR: 0.4219 (down from 0.9872)
- Test Recall on Fraud Rings: 0.0445 (4.5%, down from 80.7%)
- Test Recall on Mule Accounts: 0.4930 (49.3%, down from 94.1%)
- Top Feature Importance (Gain): Shifted from raw transaction amounts to `merchant_risk_rate` (4,125), `channel_web` (1,806), and `time_delta_card` (1,779).

This establishes a realistic, hardened tabular benchmark with significant headroom for heterogeneous GNNs (Phase 3) to demonstrate measurable architectural advantage on complex relational fraud topologies.

## 16. Limitations

1. **Label Delay for Merchant Risk Rate**: In production financial systems, chargeback and dispute notifications arrive with a significant lag (typically 30 to 90 days after transaction settlement). The current feature engineering pipeline computes expanding Bayesian-smoothed merchant risk rates using past transaction labels without an explicit dispute delay window. While strictly causal (zero lookahead into future timestamps), a fully mature production pipeline would model an asynchronous label arrival delay buffer.
2. **Label Noise Rate (1.5%)**: The synthetic generator intentionally flips 1.5% of positive fraud labels to `is_fraud = 0` to reflect real-world imperfect observation, unreported fraud, and noisy ground-truth chargebacks. Consequently, theoretical maximum recall is bounded below 100%.
3. **Account Takeover (ATO) Detection Ceiling**: In the hardened simulator, account takeover events intentionally emulate sophisticated attackers (25% domestic IP origin, 15% reuse of familiar device/session tokens, amounts overlapping legitimate variance). Tabular features alone achieve <1% recall on ATO; catching such compromises in practice requires upstream authentication telemetry, device fingerprinting, or multi-hop identity graph clustering.
4. **Simulator Difficulty Calibration**: The synthetic data parameters were deliberately calibrated to place baseline tabular XGBoost PR-AUC into an intermediate difficulty range (~0.30–0.75). In synthetic benchmark design, uncalibrated simulators tend to be either trivially separable (yielding false confidence in simple models) or pure noise (preventing any model from learning). The calibrated difficulty ensures a rigorous testbed for demonstrating GNN structural learning value.

## 17. Causal Temporal Sampling & Elimination of Within-Window Lookahead Leakage

**Context & Leakage Gap**:
In static graph snapshot evaluations, temporal splits are created at macro boundaries (e.g. Train: 0–60%, Val: 60–80%, Test: 80–100%). While this prevents test edges from appearing in the train graph, an intra-split leakage gap remained: inside the test window, a transaction occurring on day 25 could sample neighbor edges and aggregate node statistics from transactions occurring on day 29 of the same window. Because standard tabular models (like streaming XGBoost) are strictly causal at point-of-decision, evaluating GNNs with intra-window future lookahead is fundamentally unfair and optimistically biased.

**Architecture & Implementation Choice**:
1. **Temporal CSR Index (`FastTemporalAdjacency`)**:
   Relational edges are indexed in a CSR structure sorted by `(time, src)`. When sampling neighbors for a target edge at timestamp $t$, an $O(\log D)$ binary search (`np.searchsorted(times, t, side="left")`) restricts the visible adjacency strictly to $t' < t$ across all relations and hops.

2. **Dynamic Causal Node Features vs. Dropping Aggregations**:
   - *Option A (Chosen)*: Compute node degree, cumulative transaction volume, and entity age dynamically on-the-fly strictly before each target transaction's timestamp using the temporal CSR binary search.
   - *Option B (Rejected - Dropping Aggregations)*: Reducing node features to static 5-dimensional one-hot type indicators `[user, merchant, card, device, ip]` strips the model of velocity and volume signals, rendering it blind to high-frequency card testing and collusive ring spikes.
   - *Option C (Rejected - Freezing at Split Cutoff)*: Static-freezing node features at split cutoffs creates catastrophic train/test distribution shifts: at $t=0$, training nodes have 0 prior history, whereas at $t=24$ days, test nodes have weeks of history.
   *Rationale*: Option A faithfully replicates production feature-store serving: when a transaction arrives at time $t$, upstream graph registries aggregate only historical events $t' < t$, achieving mathematical invariance to future events in $<0.5$ ms per batch.

3. **Invariance Verification Test**:
   Every causal loader is validated with an injection test: injecting an arbitrary $1,000,000 fake transaction strictly AFTER target edge timestamp $t_0$ in the same window must yield 100% bit-identical sampled neighborhoods, node feature tensors, and model logits (`torch.equal(out_clean, out_injected)`).



