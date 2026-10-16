# Data Dictionary & Synthetic Dataset Architecture

## 1. Overview

The `fraudgnn` transaction stream simulator generates high-fidelity, heterogeneous financial transactions designed specifically for benchmarking Graph Neural Networks against traditional tabular gradient boosting models (XGBoost).

The dataset captures realistic consumer behavior over 90 simulated days, including per-user lognormal spending profiles, circadian diurnal rhythms, and benign network noise (shared family devices, office VPNs), while injecting 5 distinct financial fraud typologies.

- **Primary Path**: `data/raw/transactions.parquet`
- **Default Scale**: 500,000 transactions across 10,000 users, 1,000 merchants, 15,000 devices, and 25,000 IP addresses.
- **Time Span**: 90 days (starting 2024-01-01 UTC).
- **Target Fraud Prevalence**: ~0.8% (configurable between 0.5% - 1.0%).
- **Compression**: PyArrow Snappy Parquet.

---

## 2. Data Dictionary

| Column Name | Data Type | Nullable | Role | Description | Example |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `transaction_id` | `string` | No | ID | Globally unique transaction identifier, sequentially ordered. | `"tx_00000042"` |
| `timestamp` | `datetime64[ns]` | No | Temporal Index | UTC timestamp with microsecond resolution. Monotonically non-decreasing. | `2024-01-01 14:23:10.450123` |
| `user_id` | `string` | No | Graph Node | Unique customer/account identifier. | `"usr_000123"` |
| `card_id` | `string` | No | Graph Node | Payment card token associated with the transaction. | `"crd_000123_0"` |
| `merchant_id` | `string` | No | Graph Node | Merchant receiving payment. | `"mer_000456"` |
| `device_id` | `string` | No | Graph Node | Hardware device fingerprint / identifier. | `"dev_007890"` |
| `ip_address` | `string` | No | Graph Node | IPv4 address of client making transaction. | `"198.51.10.45"` |
| `amount` | `float64` | No | Continuous Feature | Transaction amount in USD (2 decimal places). | `47.50` |
| `currency` | `string` | No | Categorical Feature | Currency code (fixed to `"USD"` for base simulation). | `"USD"` |
| `country` | `string` | No | Categorical Feature | ISO-2 country code where transaction originated. | `"US"` |
| `merchant_category`| `string` | No | Categorical Feature | MCC category classification. | `"grocery"`, `"electronics"` |
| `channel` | `string` | No | Categorical Feature | Originating channel: `"mobile"`, `"web"`, or `"pos"`. | `"mobile"` |
| `is_fraud` | `int64` | No | **Ground Truth Target** | Binary ground truth label (1 = Fraud, 0 = Legitimate). | `0` or `1` |
| `fraud_type` | `string` | No | **Hidden Meta-Label** | Specific typology injected. **STRICTLY FOR ANALYSIS ONLY. NEVER A MODEL FEATURE.** | `"legitimate"`, `"fraud_ring"` |

> [!WARNING]
> **Label Leakage Protection**: The `fraud_type` column is reserved exclusively for post-hoc error analysis, slicing, and model explainability. It must be explicitly stripped during feature engineering before training any model.

---

## 3. Injected Fraud Typologies

The simulator injects 5 realistic financial fraud mechanisms:

### a) Coordinated Fraud Rings (`fraud_ring`)
- **Mechanism**: Organized criminal syndicates operating pools of 5 to 30 synthetic or stolen identities.
- **Graph Topology**: Dense bipartite cliques. Ring members share a concentrated set of 2-4 devices and 2-5 IPs, transacting in temporal clusters across targeted merchants.
- **Feature Footprint**: Moderate-to-high amounts ($150 - $1,200), tight inter-arrival times, high Jaccard similarity in entity neighborhoods.

### b) Mule Accounts (`mule`)
- **Mechanism**: Accounts used for rapid money-laundering aggregation and dispersal.
- **Graph Topology**: Fast pass-through hubs connecting disparate funding sources to high-risk cash-out merchants (crypto exchanges, remittance, P2P).
- **Feature Footprint**: Bursts of 2-5 transactions within a short temporal window (<1 hour), with closely matching outbound amounts ($500 - $3,500).

### c) Card Testing / Account Probing (`card_testing`)
- **Mechanism**: Automated scripts testing lists of stolen credit card credentials at low-friction online merchants before attempting large purchases.
- **Graph Topology**: Single card/device touching multiple low-dollar digital goods merchants in rapid succession, followed by high-dollar electronics/luxury merchants.
- **Feature Footprint**: 3-6 micro-transactions ($0.50 - $3.00) seconds apart, immediately followed by a high-value purchase ($800 - $3,000+).

### d) Account Takeover (`account_takeover`)
- **Mechanism**: Credential stuffing or session hijacking of established benign user accounts.
- **Graph Topology**: Sudden disconnection from user's historical 1-2 devices and home IP; attachment to novel device and international IP.
- **Feature Footprint**: Sudden country anomaly (foreign IP/country e.g., RU, NG, VN, BR), sharp dollar spike relative to user historical mean (4.5x - 10x typical spend).

### e) Merchant Collusion (`merchant_collusion`)
- **Mechanism**: Fraudulent or compromised merchants processing fabricated sales using stolen payment methods.
- **Graph Topology**: High in-degree star graph with abnormal chargeback / fraud concentration.
- **Feature Footprint**: Diverse cards and users transacting with inflated ticket sizes ($250 - $2,800) concentrated at designated rogue merchants.

---

## 4. Benign Graph Noise

In real-world financial networks, benign users also exhibit shared connectivity:
1. **Shared Devices**: Multiple family members transacting from a shared tablet, or workers using a shared office POS/kiosk (`legit_shared_device_prob: 0.08`).
2. **Shared IPs**: Corporate VPN egress points, university campus subnets, and public Wi-Fi hotspots (`legit_shared_ip_prob: 0.15`).

**ML Significance**: Without this noise, a GNN could achieve near 100% PR-AUC by merely memorizing any shared device edge. Benign noise forces the model to synthesize both graph structural signals and tabular edge features (amount distributions, channel, country transitions).

---

## 5. Temporal Splitting Protocol

Per project design decisions, **random train/test splits are strictly forbidden**. 
All models must be evaluated using temporal splitting:
- **Train Set**: First 60% of timeline (Days 1 to 54)
- **Validation Set**: Next 20% of timeline (Days 55 to 72)
- **Test Set**: Final 20% of timeline (Days 73 to 90)

This prevents lookahead bias and simulates true production deployment against evolving future fraud.
