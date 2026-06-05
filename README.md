# Bot-Detect

Done by Chee Zhao De, Cheng Kei Kei, Chai Jun Yi, Yip Zi Xian as part of Master in Artifical Intelligence at Universiti Malaya (Practical Deep Learning)

A solution for the [Kaggle Facebook Recruiting IV: Human or Robot](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot) competition. The task is bidder-level binary classification — distinguishing bot accounts from human participants in online auctions.

## Problem Overview

Online auction platforms are susceptible to shill bidding by automated bots. Given raw bid-level transaction logs, the goal is to predict whether each bidder is a bot (`outcome = 1`) or human (`outcome = 0`). The dataset is heavily imbalanced (~17% bots, ~83% humans).

## Project Structure

```
Bot-Detect/
├── pdl-preprocessing.py       # [Run 1st] Feature engineering + baseline models (RF, LGBM)
├── pdl-saint.py               # SAINT transformer model
├── pdl-tabm.py                # TabM tabular transformer model
├── pdl-tabr.py                # TabR retrieval-augmented model
├── pdl-sthgnn.py              # Spatio-temporal hypergraph neural network
├── pdl-stmvhgnn.py            # Multi-view ST hypergraph neural network
├── check_cuda.py              # GPU/CUDA availability check
├── utils/
│   ├── shared.py              # Shared utilities (plotting, logging, FocalLoss)
│   └── aggregate_result.py    # Aggregate and compare model results
├── datasets/
│   ├── train.csv              # 1,656 labelled bidders
│   ├── test.csv               # 4,700 unlabelled bidders
│   ├── bids.csv               # ~21.6M bid-level records (~930 MB)
│   └── sampleSubmission.csv
├── plots/                     # EDA visualizations
└── result/                    # Model outputs, logs, and submissions
```

## Pipeline

Run scripts in the following order:

### 1. Preprocessing & Baseline Models

```bash
python pdl-preprocessing.py
```

- Loads and cleans raw bid data
- Engineers ~55 features per bidder (volume, ratios, temporal, burst, entropy, merchandise)
- Generates EDA plots in `plots/`
- Trains and evaluates **Random Forest** and **LightGBM** with isotonic calibration
- Exports `datasets/train_features.csv`, `datasets/test_features.csv`, `datasets/bidder_features.csv`

### 2. Deep Learning Models

Run any or all of the following (they all consume the pre-engineered features):

```bash
python pdl-saint.py
python pdl-tabm.py
python pdl-tabr.py
python pdl-sthgnn.py
python pdl-stmvhgnn.py
```

Each script writes predictions, metrics, and logs to `result/<model>/` and `result/logs/<model>/`.

### 3. Aggregate Results

```bash
python utils/aggregate_result.py
```

Produces `result/aggregate_metric_comparison.csv` and a comparison plot ranking all models.

## Models

| Model | Type | Description |
|---|---|---|
| Random Forest | Ensemble | 300 estimators, class-balanced, isotonic calibration |
| LightGBM | Gradient Boosting | 1000 trees, early stopping, isotonic calibration |
| SAINT | Transformer | Self-attention + intersample attention for tabular data |
| TabM | Transformer | Ensemble of tabular transformers with architecture search |
| TabR | Retrieval-Augmented | Context-retrieval augmented tabular model (FAISS) |
| ST-HGNN | Graph Neural Network | Spatio-temporal hypergraph over auctions, IPs, and devices |
| ST-MV-HGNN | Graph Neural Network | Multi-view hypergraph variant of ST-HGNN |

### Justification

**Random Forest** and **LightGBM** serve as strong tree-based baselines. Prior literature consistently shows that gradient-boosted trees and random forests are competitive on structured tabular data, making them essential reference points for evaluating whether deep learning provides a meaningful uplift on this task.

**TabM** is chosen for its parameter-efficient ensembling over MLP-based predictors. A single neural network can be unstable on small, imbalanced datasets like this one; TabM's implicit ensemble reduces prediction variance without the cost of training multiple independent networks, making it practical for tabular fraud detection.

**TabR** is selected for its retrieval-augmented mechanism, which lets the model compare a target bidder against similar training instances rather than relying solely on global patterns. This is particularly relevant here because suspicious bidders may only be distinguishable when contrasted with behaviourally similar labelled examples — local similarity can surface minority-class signals that a global model would miss.

**SAINT** addresses bot detection through dual attention: column-wise attention captures non-linear feature interactions within a single bidder, while row-wise attention models relationships across different bidders within a batch. Because bot behaviour often emerges as a collective pattern rather than any single feature anomaly, inter-sample context gives SAINT a structural advantage over architectures that treat each record independently.

**ST-HGNN** is motivated by the need to model guilt-by-association. Standard graph edges connect only two nodes, but in auction fraud a single IP address or device may link many bidders simultaneously. Hyperedges represent these N-to-N relationships directly, allowing the model to detect community-level fraud rings. A dual-branch design further combines structural (hypergraph convolution) signals with individual behavioural and temporal features, fused dynamically via a Squeeze-and-Excitation gate.

**ST-MV-HGNN** extends ST-HGNN to counter "fraud camouflage," where sophisticated bots mimic human bidding patterns at the individual level to evade detection. By constructing three separate relational hypergraphs (Auctions, IPs, Devices) and cross-referencing them through a Cross-View Inconsistency Perception module, the model flags discrepancies between a bidder's isolated behaviour and their broader network context — catching well-disguised bots that single-view or tabular models would misclassify as human.

## Features

Features are aggregated from bid-level records to the bidder level:

| Category | Features |
|---|---|
| Volume | `bid_count`, `unique_auctions`, `unique_devices`, `unique_ips`, `unique_urls`, `log_bid_count` |
| Ratios | `ips_per_bid`, `countries_per_bid`, `devices_per_bid`, `bids_per_auction`, `subnets_per_bid` |
| Temporal | `time_span`, `time_mean`, `time_std`, `time_diff_mean`, `time_diff_std`, `zero_interval_ratio` |
| Burst Patterns | `burst_max_strip[1–5]`, `burst_mean_strip[1–5]`, `burst_ratio_strip[1–5]` |
| Entropy | `country_entropy`, `device_entropy`, `url_entropy`, `merchandise_entropy` |
| Geography | `top5_country_ratio`, `unique_subnets` |
| Merchandise | One-hot encoded category ratios |

## Feature Engineering

All features are derived from `bids.csv` and aggregated to the bidder level by `pdl-preprocessing.py`. The pipeline runs in five stages before the feature matrix is exported.

### 1. Data Cleaning

| Step | Detail |
|---|---|
| Duplicate removal | Exact duplicate `bid_id` rows dropped from `bids.csv` |
| Null imputation | Missing `country`, `url`, and `ip` values filled with `"UNKNOWN"`; `address` and `payment_account` in bidder tables filled similarly |
| Sort & time-diff | Bids sorted by `(bidder_id, time)`; `time_diff` computed per bidder via `.diff()` — the foundation for all temporal and zero-interval features |

### 2. Volume & Cardinality Features

Straightforward counts and nuniques aggregated per bidder:

- `bid_count` — total bids placed
- `unique_auctions`, `unique_devices`, `unique_countries`, `unique_ips`, `unique_urls`, `unique_merchandise` — distinct entities seen
- `bids_per_auction` — bid count ÷ unique auctions (activity intensity)
- `ips_per_bid`, `countries_per_bid`, `devices_per_bid` — diversity ratios normalised by bid volume

### 3. Temporal Features

Derived from the sorted `time` column and the `time_diff` series:

- `time_span` — max time − min time (overall activity window)
- `time_mean`, `time_std` — mean and spread of raw timestamps
- `time_diff_mean`, `time_diff_std`, `time_diff_min` — statistics on inter-bid gaps
- `zero_interval_count` — number of consecutive bids with identical timestamps
- `zero_interval_ratio` — fraction of bids that are zero-gap (bots tend to fire rapid bursts with no measurable gap)

### 4. Burst Features

Time is discretised into strips of varying granularity by integer-dividing `time` by `10^n` for `n ∈ {1, 2, 3, 4, 5}`. Within each strip, bids per time-bucket are counted per bidder, yielding three features per strip level:

- `burst_max_stripN` — maximum bids in any single time-bucket at granularity N
- `burst_mean_stripN` — mean bids per time-bucket at granularity N
- `burst_ratio_stripN` — `burst_max_stripN ÷ bid_count` (relative peak intensity)

This produces 15 burst features that capture bot-like spiking behaviour at multiple time resolutions.

### 5. Entropy & Geography Features

- `country_entropy`, `device_entropy`, `url_entropy`, `merchandise_entropy` — Shannon entropy of each categorical dimension per bidder; higher values indicate more uniform spread across categories (a common bot signature for country and device)
- `top5_country_ratio` — fraction of bids placed from the five most common countries globally; bots often operate from less common countries
- `unique_subnets` — number of distinct `/16` IP prefixes (first two octets); `subnets_per_bid` normalises this by bid count

### 6. Merchandise Ratio Features

Each merchandise category is one-hot encoded at the bid level, summed per bidder, then normalised to proportions. This produces one `merch_<category>_ratio` column per category (e.g. `merch_mobile_ratio`, `merch_sporting_goods_ratio`), capturing which auction types each bidder concentrates on.

### 7. Log Transforms

Right-skewed count features are log+1 transformed to reduce the influence of extreme outliers, producing parallel `log_*` columns for:

`bid_count`, `unique_auctions`, `unique_ips`, `unique_urls`, `time_span`, `time_std`, `burst_max_strip[1–5]`

The final feature matrix contains **~55 features** per bidder and is exported to `datasets/train_features.csv` and `datasets/test_features.csv` for consumption by all downstream models.

## EDA Plots

All plots are generated by `pdl-preprocessing.py` and saved to `plots/`.

| File | Description |
|---|---|
| `01_raw_overview.png` | Six-panel dashboard of the raw dataset: class balance (1,910 humans vs 103 bots), bids-per-bidder frequency (log-scale histogram), top-15 bidding countries, top-10 devices, merchandise category distribution (pie chart), and a sampled bid timestamp series. Gives a quick sanity-check of the data before any feature engineering. |
| `02_feature_distributions.png` | Overlapping histograms (bot vs human) for six key engineered features: log bid count, log unique IPs, zero-interval ratio, country entropy, burst ratio (strip-3), and max burst size (strip-4). Bots tend to have higher log bid counts, more unique IPs, and flatter burst distributions compared to humans. |
| `03_correlation_heatmap.png` | Pearson correlation matrix for the top-20 features plus `outcome`. Burst-related features (`burst_max_strip*`, `log_burst_max_strip*`) form a tightly correlated cluster, while `outcome` shows only weak individual correlations — reinforcing why ensemble and deep models outperform simple linear baselines. |
| `04_raw_vs_log_boxplots.png` | Side-by-side boxplots of raw vs log+1 transformed bid count and unique IPs, split by label. Raw distributions are dominated by extreme outliers that compress the interquartile range; the log transform reveals that bots have a noticeably higher and tighter median than humans for both features. |
| `05_zero_interval_scatter.png` | Scatter plot of zero-interval ratio vs log bid count, coloured by label. The near-zero floor for high-volume bidders is a natural mathematical artifact (more bids → lower minimum ratio); bots do not cluster at extreme zero-interval ratios, suggesting this feature is useful primarily in combination with volume features. |
| `06_merchandise_ratio.png` | Grouped bar chart of mean merchandise-category bid ratios for bots vs humans across all 10 categories. Bots over-index on sporting goods, computers, and furniture relative to humans, while humans bid more on mobile and jewelry — a signal captured by the one-hot merchandise ratio features. |
| `07_country_entropy.png` | Density histogram of country entropy by label. Humans spike sharply at entropy ≈ 0 (bidding from a single country), whereas bots spread more uniformly across entropy values up to ~5, indicating they rotate across many geographic origins. |

## Results

Evaluated on out-of-fold (OOF) validation splits:

| Rank | Model | AUC-ROC | Avg Precision | PR-AUC | F1 |
|---|---|---|---|---|---|
| 1 | **SAINT** | **0.9323** | 0.4248 | 0.4195 | 0.4950 |
| 2 | Random Forest | 0.9221 | 0.5106 | 0.5082 | 0.5078 |
| 3 | TabM | 0.9176 | 0.4281 | 0.4217 | 0.5089 |
| 4 | LightGBM | 0.9142 | 0.4873 | 0.4850 | 0.4917 |
| 5 | TabR | 0.9002 | 0.3003 | 0.2936 | 0.4012 |
| 6 | ST-MV-HGNN | 0.8982 | 0.3977 | 0.3926 | 0.4732 |
| 7 | ST-HGNN | 0.8899 | 0.3233 | 0.3182 | 0.4057 |

SAINT achieves the best AUC-ROC at **0.9323**. All models exceed 0.89 AUC, reflecting strong feature engineering.

## Requirements

- Python 3.x
- CUDA-capable GPU (required for SAINT, TabM, TabR, ST-HGNN, ST-MV-HGNN)
- Key packages: `torch`, `torch-geometric`, `lightgbm`, `scikit-learn`, `shap`, `faiss`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify GPU availability before running deep learning models:

```bash
python check_cuda.py
```

## Data

Place the raw Kaggle dataset files in `datasets/` before running the pipeline:

- `train.csv`
- `test.csv`
- `bids.csv`
- `sampleSubmission.csv`

Download from: https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/data
