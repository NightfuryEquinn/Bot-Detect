# CLAUDE.md — Bot-Detect Project

## Project overview

Kaggle competition: **Facebook Recruiting IV: Human or Bot**  
Task: Binary classification — predict whether a bidder in an online auction is a bot (1) or human (0).  
Dataset: Bidder-level features derived from raw bid logs (`bids.csv`).

---

## Repository layout

```
Bot-Detect/
├── datasets/                      # Raw and processed data
│   ├── train.csv                  # Bidder labels (train)
│   ├── test.csv                   # Bidder IDs (test, no labels)
│   ├── bids.csv                   # Raw bid-level logs
│   ├── train_features.csv         # Engineered features for train set  ← produced by pdl-preprocessing.py
│   ├── test_features.csv          # Engineered features for test set   ← produced by pdl-preprocessing.py
│   └── bidder_features.csv        # All bidders combined               ← produced by pdl-preprocessing.py
│
├── plots/                         # EDA visualisations
│
├── result/
│   ├── rf/                        # Random Forest outputs
│   │   ├── rf_*.png
│   │   └── rf_submission.csv
│   ├── lgbm/                      # LightGBM outputs
│   │   ├── lgbm_*.png
│   │   └── lgbm_submission.csv
│   ├── saint/                     # SAINT outputs
│   │   ├── saint_*.png
│   │   └── saint_submission.csv
│   ├── tabm/                      # TabM outputs
│   │   ├── tabm_*.png
│   │   └── tabm_submission.csv
│   ├── tabr/                      # TabR outputs
│   │   ├── tabr_*.png
│   │   └── tabr_submission.csv
│   ├── sthgnn/                    # ST-HGNN outputs
│   │   └── sthgnn_final_oof_results.csv
│   ├── stmvhgnn/                  # ST-MV-HGNN outputs
│   │   └── stmvhgnn_final_oof_results.csv
│   ├── logs/
│   │   ├── rf/rf_summary.csv
│   │   ├── lgbm/lgbm_summary.csv
│   │   ├── saint/saint_summary.csv
│   │   ├── tabm/tabm_summary.csv
│   │   ├── tabr/tabr_summary.csv
│   │   ├── sthgnn/sthgnn_summary.csv
│   │   └── stmvhgnn/stmvhgnn_summary.csv
│   └── aggregate_auc_comparison.csv   ← produced by utils/aggregate_result.py
│
├── utils/
│   ├── shared.py                  # Shared constants and utilities (PALETTE, configure_plots, save_fig, DualLogger, FocalLoss)
│   └── aggregate_result.py        # Cross-model AUC comparison script
│
├── pdl-preprocessing.py           # Feature engineering + RF + LightGBM baselines
├── pdl-saint.py                   # SAINT model
├── pdl-tabm.py                    # TabM model
├── pdl-tabr.py                    # TabR model
├── pdl-sthgnn.py                  # ST-HGNN model
└── pdl-stmvhgnn.py               # ST-MV-HGNN model
```

---

## Run order

Scripts must be run in this order (later scripts depend on CSV outputs from earlier ones):

1. **`pdl-preprocessing.py`**  
   Produces `datasets/train_features.csv`, `datasets/test_features.csv`, `datasets/bidder_features.csv`  
   Also trains and evaluates the RF and LightGBM baselines, saving to `result/rf/` and `result/lgbm/`

2. **`pdl-saint.py`** — can run after step 1 (uses raw `bids.csv` + `train.csv` / `test.csv`, not the feature CSVs)

3. **`pdl-tabm.py`** — requires step 1 output (loads `train_features.csv`, `test_features.csv`)

4. **`pdl-tabr.py`** — requires step 1 output (loads `train_features.csv`, `test_features.csv`)

5. **`pdl-sthgnn.py`** — requires step 1 output (loads `train_features.csv`, `test_features.csv`)

6. **`pdl-stmvhgnn.py`** — requires step 1 output (loads `train_features.csv`, `test_features.csv`)

7. **`utils/aggregate_result.py`** — requires all model summary CSVs (run after all models complete)

---

## Shared utilities (`utils/shared.py`)

All model scripts import from here. Never duplicate these:

| Symbol | Type | Purpose |
|---|---|---|
| `PALETTE` | dict | Hex colour constants (bot, human, neutral, calib, uncalib, accent, tabm, tabr, tune) |
| `PLOT_STYLE` | dict | Matplotlib rcParams for the dark theme |
| `configure_plots()` | function | Applies the project-wide seaborn + matplotlib style |
| `save_fig(fig, save_dir, name)` | function | Saves `fig` to `<save_dir>/<name>.png` and closes it |
| `DualLogger` | class | Tees stdout to terminal + log file simultaneously |
| `setup_logger(output_dir, file_name)` | function | Redirects stdout to a `DualLogger` |
| `FocalLoss` | class | Binary focal loss with label smoothing (used by STHGNN and ST-MV-HGNN) |

---

## Models

| Script | Model | CV strategy | GPU required |
|---|---|---|---|
| `pdl-preprocessing.py` | Random Forest, LightGBM | StratifiedKFold (5-fold) | No |
| `pdl-saint.py` | SAINT | StratifiedKFold (5-fold) | Yes |
| `pdl-tabm.py` | TabM | StratifiedKFold (5-fold) | Yes |
| `pdl-tabr.py` | TabR | Single train/val split (80/20) | Yes |
| `pdl-sthgnn.py` | ST-HGNN | StratifiedKFold (5-fold) | Yes |
| `pdl-stmvhgnn.py` | ST-MV-HGNN | StratifiedKFold (5-fold) | Yes |

---

## Feature engineering

`pdl-preprocessing.py` engineers these feature groups from `bids.csv`:

- **Bid aggregations**: `bid_count`, `auction_count`, `device_count`, `country_count`, `ip_count`, `url_count`, `merch_count`
- **Ratio features**: `bids_per_auction`, `bids_per_device`, `bids_per_ip`, etc.
- **Time statistics**: `time_range`, `time_std`, `gap_mean`, `gap_std`, `gap_cv`, `gap_iqr`
- **Burst features** (5 time-prefix strips): `burst_max_strip{1-5}`, `burst_mean_strip{1-5}`, `burst_ratio_strip{1-5}`
- **Entropy measures**: `merch_entropy`, `country_entropy`
- **IP subnet features**: `/24` and `/16` subnet aggregations
- **Log transforms**: Applied to highly skewed numeric features
- **Merchandise pivots**: Top merchandise categories as binary features

`pdl-saint.py` uses its own feature aggregation scheme (intentionally different — it derives features directly from `bids.csv` rather than loading from the feature CSVs).

---

## Key design decisions

- All model scripts use CUDA; `assert torch.cuda.is_available()` is at the top of each
- SAINT disables row (intersample) attention during inference and SHAP attribution (enables valid i.i.d. attribution)
- TabM uses `tabm` and `rtdl_num_embeddings` libraries; QuantileTransformer or RobustScaler applied per fold
- TabR uses a retrieval mechanism (memory bank of training examples); optional FAISS acceleration
- ST-HGNN and ST-MV-HGNN use `torch_geometric.nn.HypergraphConv`; FocalLoss for class imbalance
- LightGBM uses isotonic calibration to correct bot-rate prediction drift

---

## Adding a new model

1. Create `pdl-<name>.py`
2. Import from `utils.shared`: `from utils.shared import PALETTE, configure_plots, save_fig`
3. Load data from `datasets/train_features.csv` and `datasets/test_features.csv`
4. Save outputs to `result/<name>/` and summaries to `result/logs/<name>/<name>_summary.csv`
5. Register the summary path in `utils/aggregate_result.py`'s `SUMMARY_PATHS` list
