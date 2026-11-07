# Evaluation protocol: final tuned run and sealed holdout

Status: **Registered 2026-11-07.** This protocol is committed before any search or final
training starts. Changes after that are recorded under *Deviations* with a date and a reason,
before the holdout is scored.

It depends on these configs, committed with it: `configs/hparam_search.yaml`,
`configs/xgb_search.yaml`, `configs/holdout.yaml`, `configs/leakage_battery.yaml`.

## 1. Data

| World | Generator | Role |
|---|---|---|
| Original | `configs/data.yaml`, seed 2025 | train (first 60% of time), validation (next 20%), test (last 20%) |
| Holdout | `configs/data.yaml` with seed **20261107**, everything else identical | sealed: only the last 20% of time are scored; the first 80% are causal history only (never trained on, never used for selection or thresholds) |

- Holdout edge features are scaled with the **original training-split scaler**
  (`models/edge_scaler.pkl`), never refit on holdout data. The scaler was recovered by
  rebuilding the original graphs: the rebuilt files are not byte-identical to the current ones
  (the builder has changed since they were built), but every model-relevant tensor (edges, edge
  features, times, labels, masks) is exactly equal. The only differences are the static node
  features `x`, which no causal loader reads, and added device/IP indices. Details in
  `docs/verification/holdout_manifest.json`.
- File hashes of the holdout parquet, its graphs and the scaler are in
  `docs/verification/holdout_manifest.json`. No metric is computed on the holdout before the
  freeze.

## 2. Metrics

- **Primary:** PR-AUC (average precision) on the scored split, with a 95% bootstrap CI
  (500 resamples, seed 42, as in `compute_metrics`).
- **Secondary:** ROC-AUC; recall at 1% FPR; F1 at the validation-selected threshold; business
  cost ($10 per alert reviewed, transaction amount per missed fraud); per-fraud-type recall as
  k/n over rows with `is_fraud == 1`.
- **Thresholds** are chosen on the **original validation split only** (F1-optimal) and applied
  unchanged to the original test split and to the holdout.
- Before any metric, scoring asserts that both raw files are in time order (never re-sorted),
  that the scored targets' graph labels equal the raw-row labels exactly, and that every
  model's prediction vector has the same length as the labels.
- **Comparisons:** paired bootstrap of PR-AUC differences on identical holdout resamples
  (500, seed 42): each final GNN seed versus the tuned label-free XGBoost trained with the
  **same seed** (42-42, 43-43, 44-44), and each final GNN seed versus the oracle. Signed
  differences with 95% intervals. The pairs are listed in `configs/holdout.yaml`.

## 3. Leakage gate

Every GNN checkpoint that will be scored must pass `scripts/leakage_battery.py` checks a-d
(brute-force reference, future injection at batch sizes 1/2/64/2048, ties and own edges,
alignment) on the validation, test and holdout splits **before the freeze**. Check e
(message-passing ablation) computes a metric, so it is never run on the holdout.

## 4. GNN hyperparameter search

- Model: HeteroGAT, PNA aggregation, device/IP binding, edge attributes, 2 layers, 4 heads,
  batch size 2048 (micro-batches of 512), isolated causal loader (`isolate_targets=True`).
- Search space (random search, **8 sampled configurations**, sampling seed 42, plus the
  anchor below as a 9th):

| Parameter | Values |
|---|---|
| learning_rate | 3e-4, 1e-3, 2e-3 |
| hidden_dim | 64, 128 |
| dropout | 0.1, 0.2, 0.3 |
| weight_decay | 1e-5, 1e-4 |
| num_neighbors | [10, 10], [15, 10] |
| loss | weighted_bce, focal (gamma 2.0, alpha 0.75) |

- Budget per configuration: a **fixed 25% random subset of training targets** (subset seed 42,
  identical for every configuration), **6 epochs**, no early stopping, model seed 42.
  Validation PR-AUC on the **full validation split** (98,818 targets) after every epoch.
- **Anchor (configuration 8):** the known-good settings (learning_rate 1e-3, hidden_dim 64,
  dropout 0.2, weight_decay 1e-5, num_neighbors [10, 10], weighted_bce), run at the same
  budget. A 6-epoch / 25% budget favours configurations that learn fast (5 of the 8 sampled
  use learning_rate 2e-3); the anchor is there so tuning cannot silently select a model worse
  than the one already known.
- Score of a configuration = its best epoch's validation PR-AUC. Each configuration's mean epoch
  time (training plus validation, wall-clock) is measured on Colab during the search itself and
  recorded in `hparam_search.json`.
- **Selection rule (efficiency):** among the 9 configurations whose score is within **0.01** of
  the highest score, the one with the **lowest measured mean epoch time** wins; a tie goes to
  the lower index. Rationale: the 95% bootstrap CI of a validation PR-AUC is about ±0.03 (the
  current HeteroGAT: 0.7973 [0.7650-0.8266]), so differences under 0.01 are within noise,
  while the final runs differ up to about 3x in cost between configurations (estimated
  9.3-28.5 Colab minutes per epoch, `docs/verification/search_timing_cpu.json`). The
  configurations inside the margin and the reason for the choice are printed and saved.
- **Device fallback:** the device name (`torch.cuda.get_device_name()`, or `cpu`) is recorded
  with every epoch's time. The search may span Colab sessions on different GPUs, so if the
  configurations inside the margin did not all run every epoch on the same device, measured
  times are not comparable: they are ranked instead by the hardware-independent relative cost
  of their hidden_dim / num_neighbors combination in
  `docs/verification/search_timing_cpu.json` (CPU full-epoch time relative to hidden 64,
  [10, 10]). The selection report states which basis was used and why.
- The original test split and the holdout are not touched during the search. All 9 results are
  published in `docs/verification/hparam_search.json`.

## 5. Final GNN runs

- The winning configuration, trained on **all** training targets, max **30 epochs**,
  early stopping with **patience 6** on validation PR-AUC, isolated loader, per-epoch
  checkpoint/resume on Google Drive (`configs/hetero_gat_final.yaml`, written from the search
  result by `scripts/write_final_config.py`).
- **Seeds 42, 43 and 44, run in that order**, each a separate resumable run. Each run keeps its
  best-validation epoch.
- **Every seed is run to completion and reported**, whatever its result: per-seed results plus
  their mean and range. No seed is abandoned, skipped, selected or dropped.

## 6. Baselines

- **Label-free XGBoost (tuned, equal effort):** random search of **20 configurations**
  (sampling seed 42) over max_depth {4, 6, 8}, learning_rate {0.03, 0.05, 0.1},
  n_estimators {300, 600, 1000}, min_child_weight {1, 5, 10}, subsample {0.7, 0.8, 1.0},
  colsample_bytree {0.6, 0.8, 1.0}; early stopping on validation (30 rounds), score = validation
  PR-AUC, same selection rule. The winner is retrained with random_state **42, 43 and 44**;
  all three are reported.
- **Untuned, at their reproduced settings (single seed, stated as untuned):** tabular XGBoost,
  oracle XGBoost (uses past fraud labels), MLP on the 27 edge features.
- **Reference:** the isolated-batch HeteroGAT epoch-18 checkpoint (`3d5fa8f3`), selected before
  this protocol; reported as a reference, not as the tuned model.

## 7. Freeze and single scoring

1. After training, the leakage gate (section 3) runs for every GNN checkpoint. The scoring
   path is rehearsed on the ORIGINAL test split (`score_holdout.py --rehearse`), which must
   reproduce each reference model's known test PR-AUC to 1e-6.
2. `docs/verification/freeze.json` lists every model to be scored and the sha256 of each
   checkpoint, and is committed.
3. `scripts/score_holdout.py` refuses to run without `freeze.json`, refuses if any checkpoint
   or holdout file hash differs from the freeze and manifest, and refuses to overwrite an
   existing holdout result.
4. The holdout is scored **exactly once**. All results are published as they are, whatever
   they show.

## Deviations

None yet.
