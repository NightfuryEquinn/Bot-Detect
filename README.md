# Bot-Detect

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
