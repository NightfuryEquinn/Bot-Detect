#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# pdl-preprocessing.py
# Facebook Recruiting IV: Human or Robot
# Covers: Data loading · Cleaning · Feature engineering · EDA visualisation
#         Baseline models: Random Forest · LightGBM (with isotonic calibration)
#
# Outputs
#   EDA plots   → plots/
#   RF results  → result/rf/   |  result/logs/rf/
#   LGBM results→ result/lgbm/ |  result/logs/lgbm/
#   Features    → datasets/train_features.csv, test_features.csv, bidder_features.csv
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import lightgbm as lgb
from scipy import stats
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, auc,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder

from utils.shared import PALETTE, configure_plots, save_fig

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)

configure_plots()

DATA_DIR   = "./datasets"
PLOTS_DIR  = "plots"
RF_DIR     = "result/rf"
LGBM_DIR   = "result/lgbm"
RF_LOG     = "result/logs/rf"
LGBM_LOG   = "result/logs/lgbm"
SEED       = 42

for d in [PLOTS_DIR, RF_DIR, LGBM_DIR, RF_LOG, LGBM_LOG]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "="*60)
print("1. LOADING DATA")
print("="*60)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
bids  = pd.read_csv(f"{DATA_DIR}/bids.csv")

for name, df in [("train", train), ("test", test), ("bids", bids)]:
    print(f"\n{name}.csv  shape={df.shape}")
    print(df.dtypes.to_string())


# =============================================================================
# 2. DATA UNDERSTANDING
# =============================================================================
print("\n" + "="*60)
print("2. DATA UNDERSTANDING")
print("="*60)

print("\n── TRAIN ──")
print(train.describe(include="all").T)
print("\n── BIDS (numeric) ──")
print(bids.describe().T)


def null_report(df, label):
    null = df.isnull().sum()
    pct  = null / len(df) * 100
    return pd.DataFrame({"nulls": null, "pct": pct}).query("nulls > 0").assign(source=label)


null_df = pd.concat([null_report(train, "train"),
                     null_report(test,  "test"),
                     null_report(bids,  "bids")])
print("\n── NULL REPORT ──")
print(null_df.to_string() if not null_df.empty else "No nulls found.")

print("\n── CLASS BALANCE (train) ──")
vc = train["outcome"].value_counts()
print(vc.to_string())
print(f"  Bot ratio: {vc[1]/len(train)*100:.1f}%")

train_with_bids = set(train.bidder_id) & set(bids.bidder_id)
test_with_bids  = set(test.bidder_id)  & set(bids.bidder_id)
print(f"\nTrain bidders with bids : {len(train_with_bids)}/{len(train)} "
      f"({len(train_with_bids)/len(train)*100:.1f}%)")
print(f"Test  bidders with bids : {len(test_with_bids)}/{len(test)} "
      f"({len(test_with_bids)/len(test)*100:.1f}%)")

bids_per_bidder_raw = bids.groupby("bidder_id").size()
print(f"\nBids-per-bidder: min={bids_per_bidder_raw.min()} "
      f"max={bids_per_bidder_raw.max()} "
      f"median={bids_per_bidder_raw.median():.0f}")


# =============================================================================
# 3. VISUALISATION — RAW DATA
# =============================================================================
print("\n" + "="*60)
print("3. RAW DATA VISUALISATION")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Raw Data Overview — Facebook Human vs Robot", fontsize=16, y=1.01)

ax = axes[0, 0]
bars = ax.bar(["Human (0)", "Bot (1)"], vc.values,
              color=[PALETTE["human"], PALETTE["bot"]], width=0.5, edgecolor="none")
ax.bar_label(bars, fmt="%d", padding=4, color="#E0E0E0")
ax.set_title("Class Balance (train)")
ax.set_ylabel("Count")

ax = axes[0, 1]
bids_per_bidder_raw.clip(upper=2000).hist(bins=60, ax=ax, color=PALETTE["neutral"],
                                          edgecolor="none", log=True)
ax.set_title("Bids per Bidder (raw, clipped@2k)")
ax.set_xlabel("Bid count"); ax.set_ylabel("Frequency (log)")

ax = axes[0, 2]
bids["country"].value_counts().head(15).plot.barh(ax=ax, color=PALETTE["neutral"],
                                                   edgecolor="none")
ax.set_title("Top 15 Countries (bids)")
ax.invert_yaxis()

ax = axes[1, 0]
bids["device"].value_counts().head(10).plot.bar(ax=ax, color=PALETTE["neutral"],
                                                 edgecolor="none", rot=45)
ax.set_title("Top 10 Devices (bids)")

ax = axes[1, 1]
bids["merchandise"].value_counts().plot.pie(ax=ax, autopct="%1.1f%%",
    colors=sns.color_palette("Set2"), startangle=140, labels=None)
ax.legend(bids["merchandise"].value_counts().index, loc="center left",
          bbox_to_anchor=(1, 0.5), fontsize=8)
ax.set_title("Merchandise Distribution")
ax.set_ylabel("")

ax = axes[1, 2]
sample_time = bids["time"].sample(min(200_000, len(bids)), random_state=42)
ax.hist(sample_time, bins=100, color=PALETTE["human"], edgecolor="none", alpha=0.8)
ax.set_title("Bid Timestamp Distribution (sample)")
ax.set_xlabel("Time"); ax.set_ylabel("Count")

plt.tight_layout()
save_fig(fig, PLOTS_DIR, "01_raw_overview")


# =============================================================================
# 4. DATA CLEANING
# =============================================================================
print("\n" + "="*60)
print("4. DATA CLEANING")
print("="*60)

bids_clean  = bids.copy()
train_clean = train.copy()
test_clean  = test.copy()

dup_bids = bids_clean.duplicated(subset="bid_id").sum()
print(f"Duplicate bid_ids in bids.csv: {dup_bids}")
bids_clean.drop_duplicates(subset="bid_id", inplace=True)

null_country = bids_clean["country"].isnull().sum()
bids_clean["country"] = bids_clean["country"].fillna("UNKNOWN")
print(f"country nulls filled: {null_country}")

null_url = bids_clean["url"].isnull().sum()
bids_clean["url"] = bids_clean["url"].fillna("UNKNOWN")
print(f"url nulls filled: {null_url}")

null_ip = bids_clean["ip"].isnull().sum()
bids_clean["ip"] = bids_clean["ip"].fillna("UNKNOWN")
print(f"ip nulls filled: {null_ip}")

for df in [train_clean, test_clean]:
    df["address"]         = df["address"].fillna("UNKNOWN")
    df["payment_account"] = df["payment_account"].fillna("UNKNOWN")

bpb      = bids_clean.groupby("bidder_id").size()
z_scores = np.abs(stats.zscore(bpb))
outlier_bidders = bpb[z_scores > 4]
print(f"\nBidders with z>4 bid count: {len(outlier_bidders)}")
print(outlier_bidders.sort_values(ascending=False).head(10))

bids_clean = bids_clean.sort_values(["bidder_id", "time"])
bids_clean["time_diff"] = (bids_clean.groupby("bidder_id")["time"]
                            .diff().fillna(0))
zero_interval = (bids_clean["time_diff"] == 0).sum()
print(f"\nBids with zero time-diff (same bidder): {zero_interval} "
      f"({zero_interval/len(bids_clean)*100:.1f}%)")
print(f"Cleaned bids shape: {bids_clean.shape}")


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*60)
print("5. FEATURE ENGINEERING")
print("="*60)


def entropy(series):
    p = series.value_counts(normalize=True)
    return -np.sum(p * np.log2(p + 1e-9))


g = bids_clean.groupby("bidder_id")

feats = pd.DataFrame({
    "bid_count":          g["bid_id"].count(),
    "unique_auctions":    g["auction"].nunique(),
    "unique_devices":     g["device"].nunique(),
    "unique_countries":   g["country"].nunique(),
    "unique_ips":         g["ip"].nunique(),
    "unique_urls":        g["url"].nunique(),
    "unique_merchandise": g["merchandise"].nunique(),
    "bids_per_auction":   g["bid_id"].count() / g["auction"].nunique(),
    "ips_per_bid":        g["ip"].nunique() / g["bid_id"].count(),
    "countries_per_bid":  g["country"].nunique() / g["bid_id"].count(),
    "devices_per_bid":    g["device"].nunique() / g["bid_id"].count(),
    "time_span":          g["time"].max() - g["time"].min(),
    "time_mean":          g["time"].mean(),
    "time_std":           g["time"].std().fillna(0),
    "time_diff_mean":     g["time_diff"].mean(),
    "time_diff_std":      g["time_diff"].std().fillna(0),
    "time_diff_min":      g["time_diff"].min(),
    "zero_interval_count": g["time_diff"].apply(lambda x: (x == 0).sum()),
    "zero_interval_ratio": g["time_diff"].apply(lambda x: (x == 0).mean()),
    "country_entropy":    g["country"].apply(entropy),
    "device_entropy":     g["device"].apply(entropy),
    "url_entropy":        g["url"].apply(entropy),
    "merchandise_entropy": g["merchandise"].apply(entropy),
})

merch_dummies = (bids_clean.groupby(["bidder_id", "merchandise"])
                            .size()
                            .unstack(fill_value=0))
merch_dummies.columns = [f"merch_{c}" for c in merch_dummies.columns]
merch_props   = merch_dummies.div(merch_dummies.sum(axis=1), axis=0)
merch_props.columns = [f"{c}_ratio" for c in merch_dummies.columns]

top_countries = bids_clean["country"].value_counts().head(5).index.tolist()
country_flags = (bids_clean.assign(flag=bids_clean["country"].isin(top_countries))
                             .groupby("bidder_id")["flag"]
                             .mean()
                             .rename("top5_country_ratio"))


def ip_prefix(ip_series, parts=2):
    def extract(ip):
        segs = str(ip).split(".")
        return ".".join(segs[:parts]) if len(segs) >= parts else "OTHER"
    return ip_series.apply(extract)


bids_clean["ip_subnet"]  = ip_prefix(bids_clean["ip"])
feats["unique_subnets"]  = g["ip_subnet"].nunique()
feats["subnets_per_bid"] = feats["unique_subnets"] / feats["bid_count"]

for n_strip in [1, 2, 3, 4, 5]:
    divisor  = 10 ** n_strip
    col_name = f"time_prefix_strip{n_strip}"
    bids_clean[col_name] = (bids_clean["time"] // divisor).astype(np.int64)

    burst_max = (bids_clean
                 .groupby(["bidder_id", col_name]).size()
                 .groupby("bidder_id").max()
                 .rename(f"burst_max_strip{n_strip}"))

    burst_mean = (bids_clean
                  .groupby(["bidder_id", col_name]).size()
                  .groupby("bidder_id").mean()
                  .rename(f"burst_mean_strip{n_strip}"))

    feats = feats.join(burst_max,  how="left")
    feats = feats.join(burst_mean, how="left")

for n_strip in [1, 2, 3, 4, 5]:
    feats[f"burst_ratio_strip{n_strip}"] = (
        feats[f"burst_max_strip{n_strip}"] / feats["bid_count"]
    )

bids_clean.drop(columns=[c for c in bids_clean.columns
                          if c.startswith("time_prefix_strip")], inplace=True)

all_feats = (feats
             .join(merch_props,   how="left")
             .join(country_flags, how="left")
             .fillna(0))

for col in (["bid_count", "unique_auctions", "unique_ips", "unique_urls",
              "time_span", "time_std"] +
            [f"burst_max_strip{n}" for n in [1, 2, 3, 4, 5]]):
    all_feats[f"log_{col}"] = np.log1p(all_feats[col])

print(f"Feature matrix shape: {all_feats.shape}")
print(all_feats.dtypes.value_counts())

train_feat = (train_clean
              .merge(all_feats.reset_index(), on="bidder_id", how="left")
              .fillna(0))
test_feat  = (test_clean
              .merge(all_feats.reset_index(), on="bidder_id", how="left")
              .fillna(0))

print(f"\ntrain_feat shape : {train_feat.shape}")
print(f"test_feat  shape : {test_feat.shape}")


# =============================================================================
# 6. VISUALISATION — ENGINEERED FEATURES
# =============================================================================
print("\n" + "="*60)
print("6. CLEANED / ENGINEERED DATA VISUALISATION")
print("="*60)

label_map = {0: "human", 1: "bot"}
train_feat["label"] = train_feat["outcome"].map(label_map)
colors = [PALETTE["human"], PALETTE["bot"]]

key_feats = [
    ("log_bid_count",       "Log Bid Count"),
    ("log_unique_ips",      "Log Unique IPs"),
    ("zero_interval_ratio", "Zero-interval Ratio"),
    ("country_entropy",     "Country Entropy"),
    ("burst_ratio_strip3",  "Burst Ratio (strip-3)"),
    ("burst_max_strip4",    "Max Burst Size (strip-4)"),
]

fig, axes = plt.subplots(2, 3, figsize=(30, 20))
fig.suptitle("Feature Distributions: Human vs Bot", fontsize=16)
for ax, (col, title) in zip(axes.flat, key_feats):
    for i, (label, grp) in enumerate(train_feat.groupby("label")):
        ax.hist(grp[col].dropna(), bins=50, alpha=0.65, color=colors[i],
                label=label, edgecolor="none", density=True)
    ax.set_title(title); ax.legend()
plt.tight_layout()
save_fig(fig, PLOTS_DIR, "02_feature_distributions")

num_cols    = [c for c in all_feats.columns if all_feats[c].dtype in [np.float64, np.int64]]
corr        = train_feat[num_cols].corr()
outcome_corr = corr["bid_count"].abs().sort_values(ascending=False).head(20).index
sub_corr    = train_feat[list(outcome_corr) + ["outcome"]].corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(sub_corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.7},
            annot=True, fmt=".2f", annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap (top 20 + outcome)")
plt.tight_layout()
save_fig(fig, PLOTS_DIR, "03_correlation_heatmap")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Raw vs Log-Transformed: Bid Count & Unique IPs", fontsize=14)
for i, (raw, log, title) in enumerate([
    ("bid_count",  "log_bid_count",  "Bid Count"),
    ("unique_ips", "log_unique_ips", "Unique IPs"),
]):
    for j, (col, lbl) in enumerate([(raw, "Raw"), (log, "Log+1")]):
        ax = axes[i][j]
        data_h = train_feat.loc[train_feat["label"] == "human", col]
        data_b = train_feat.loc[train_feat["label"] == "bot",   col]
        ax.boxplot([data_h, data_b], labels=["Human", "Bot"],
                   patch_artist=True,
                   boxprops=dict(facecolor="none", color="#A0A0A0"),
                   medianprops=dict(color="#FFD700", linewidth=2),
                   whiskerprops=dict(color="#A0A0A0"),
                   capprops=dict(color="#A0A0A0"),
                   flierprops=dict(marker="o", markersize=4, alpha=0.6,
                                   markerfacecolor="#FF6B6B", markeredgecolor="none"))
        ax.set_title(f"{title} — {lbl}")
plt.tight_layout()
save_fig(fig, PLOTS_DIR, "04_raw_vs_log_boxplots")

fig, ax = plt.subplots(figsize=(10, 7))
for label, grp in train_feat.groupby("label"):
    ax.scatter(grp["log_bid_count"], grp["zero_interval_ratio"],
               c=PALETTE[label], alpha=0.5, s=15, label=label, edgecolors="none")
ax.set_xlabel("Log Bid Count"); ax.set_ylabel("Zero-Interval Ratio")
ax.set_title("Zero-Interval Ratio vs Log Bid Count")
ax.legend()
plt.tight_layout()
save_fig(fig, PLOTS_DIR, "05_zero_interval_scatter")

merch_cols = [c for c in train_feat.columns if c.startswith("merch_") and "ratio" in c]
if merch_cols:
    means = train_feat.groupby("label")[merch_cols].mean().T
    means.index = [c.replace("merch_", "").replace("_ratio", "") for c in means.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    means.plot.bar(ax=ax, color=[PALETTE["human"], PALETTE["bot"]], edgecolor="none", rot=30)
    ax.set_title("Merchandise Category Ratio — Human vs Bot")
    ax.set_ylabel("Mean ratio"); ax.legend()
    plt.tight_layout()
    save_fig(fig, PLOTS_DIR, "06_merchandise_ratio")

fig, ax = plt.subplots(figsize=(9, 5))
for i, (label, grp) in enumerate(train_feat.groupby("label")):
    ax.hist(grp["country_entropy"], bins=40, alpha=0.7, color=colors[i],
            label=label, edgecolor="none", density=True)
ax.set_xlabel("Country Entropy"); ax.set_ylabel("Density")
ax.set_title("Country Entropy Distribution — Human vs Bot")
ax.legend()
plt.tight_layout()
save_fig(fig, PLOTS_DIR, "07_country_entropy")


# =============================================================================
# 7. EXPORT FEATURES
# =============================================================================
print("\n" + "="*60)
print("7. EXPORTING FEATURES")
print("="*60)

train_feat.drop(columns=["label"], inplace=True)
train_feat.to_csv(f"{DATA_DIR}/train_features.csv", index=False)
test_feat.to_csv(f"{DATA_DIR}/test_features.csv",   index=False)
all_feats.to_csv(f"{DATA_DIR}/bidder_features.csv")

print(f"{DATA_DIR}/train_features.csv  → {train_feat.shape}")
print(f"{DATA_DIR}/test_features.csv   → {test_feat.shape}")
print(f"{DATA_DIR}/bidder_features.csv → {all_feats.shape}")

feat_cols = [c for c in train_feat.columns
             if c not in ["bidder_id", "payment_account", "address", "outcome"]]
print(f"\nTotal engineered features : {len(feat_cols)}")
for group, cols in {
    "Volume":      [c for c in feat_cols if any(x in c for x in ["count", "unique", "log"])],
    "Ratio":       [c for c in feat_cols if "ratio" in c or "per_" in c],
    "Time":        [c for c in feat_cols if "time" in c],
    "Burst":       [c for c in feat_cols if "burst" in c],
    "Entropy":     [c for c in feat_cols if "entropy" in c],
    "Merchandise": [c for c in feat_cols if "merch" in c],
    "Geography":   [c for c in feat_cols if "country" in c or "subnet" in c],
}.items():
    print(f"  {group:15s}: {len(cols):3d} features")

print("\n✓ Preprocessing pipeline complete.")


# =============================================================================
# 8. LOAD FEATURES FOR BASELINE MODELS
# =============================================================================
print("\n" + "="*60)
print("8. LOADING FEATURES FOR BASELINE MODELS")
print("="*60)

META_COLS = ["bidder_id", "payment_account", "address", "outcome", "label"]

train = pd.read_csv(f"{DATA_DIR}/train_features.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_features.csv")

FEAT_COLS = [c for c in train.columns if c not in META_COLS]

X      = train[FEAT_COLS].values.astype(np.float32)
y      = train["outcome"].values.astype(np.int32)
X_test = test[FEAT_COLS].values.astype(np.float32)

train_prior = y.mean()
scale_pos   = (1 - train_prior) / train_prior

N_SPLITS = 5
cv       = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds    = list(cv.split(X, y))

print(f"Train  : X={X.shape}  bot_rate={train_prior:.4f}")
print(f"Test   : X_test={X_test.shape}")
print(f"scale_pos_weight = {scale_pos:.1f}")
print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS[:6]} …")


# =============================================================================
# 9. RANDOM FOREST BASELINE
# =============================================================================
print("\n" + "="*60)
print("9. RANDOM FOREST BASELINE")
print("="*60)

rf_base = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", max_features="sqrt",
    n_jobs=-1, random_state=SEED,
)
base_scores = cross_val_score(rf_base, X, y, cv=folds, scoring="roc_auc", n_jobs=-1)
print(f"Baseline AUC  : {base_scores.mean():.4f} ± {base_scores.std():.4f}")

param_dist = {
    "n_estimators":     randint(200, 800),
    "max_depth":        [None, 10, 20, 30, 40],
    "min_samples_leaf": randint(1, 10),
    "min_samples_split": randint(2, 20),
    "max_features":     ["sqrt", "log2", 0.3, 0.5],
    "class_weight":     ["balanced", "balanced_subsample"],
}

search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=SEED),
    param_distributions=param_dist, n_iter=30, scoring="roc_auc",
    cv=folds, refit=True, verbose=1, random_state=SEED, n_jobs=-1,
)
search.fit(X, y)
print(f"\nBest params : {search.best_params_}")
print(f"Best CV AUC : {search.best_score_:.4f}")
best_rf = search.best_estimator_

oof_proba_rf = np.zeros(len(y), dtype=np.float32)
for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
    fold_rf = RandomForestClassifier(**search.best_params_, n_jobs=-1, random_state=SEED)
    fold_rf.fit(X[tr_idx], y[tr_idx])
    oof_proba_rf[val_idx] = fold_rf.predict_proba(X[val_idx])[:, 1]
    print(f"  Fold {fold}  AUC={roc_auc_score(y[val_idx], oof_proba_rf[val_idx]):.4f}")

oof_auc_rf = roc_auc_score(y, oof_proba_rf)
print(f"\nOOF AUC (RF) : {oof_auc_rf:.4f}")

# ── RF plots ──────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y, oof_proba_rf)
prec, rec, _ = precision_recall_curve(y, oof_proba_rf)
ap_rf = average_precision_score(y, oof_proba_rf)
pr_auc_rf = auc(rec, prec)

thresholds_rf = np.arange(0.05, 0.95, 0.01)
f1s_rf = [f1_score(y, (oof_proba_rf >= t).astype(int)) for t in thresholds_rf]
best_thresh_rf = thresholds_rf[int(np.argmax(f1s_rf))]
rf_oof_labels = (oof_proba_rf >= best_thresh_rf).astype(int)
precision_rf = precision_score(y, rf_oof_labels)
recall_rf = recall_score(y, rf_oof_labels)
f1_rf = float(np.max(f1s_rf))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Random Forest — OOF AUC={oof_auc_rf:.4f}", fontsize=14)
ax = axes[0]
ax.plot(fpr, tpr, color=PALETTE["bot"], lw=2, label=f"ROC (AUC={oof_auc_rf:.4f})")
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["bot"])
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
ax = axes[1]
ax.plot(rec, prec, color=PALETTE["human"], lw=2, label=f"PR (AP={ap_rf:.4f})")
ax.fill_between(rec, prec, alpha=0.12, color=PALETTE["human"])
ax.axhline(y.mean(), color="#555", lw=1, linestyle="--", label="Chance")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend()
plt.tight_layout()
save_fig(fig, RF_DIR, "rf_01_roc_pr")

oof_pred_rf = (oof_proba_rf >= 0.5).astype(int)
cm_rf = confusion_matrix(y, oof_pred_rf)
print("\nClassification Report (RF):\n")
print(classification_report(y, oof_pred_rf, target_names=["Human", "Bot"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"],
            linewidths=0.5, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (threshold=0.5)")
plt.tight_layout()
save_fig(fig, RF_DIR, "rf_02_confusion_matrix")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(oof_proba_rf[y == 0], bins=60, alpha=0.7, color=PALETTE["human"],
        label="Human", density=True, edgecolor="none")
ax.hist(oof_proba_rf[y == 1], bins=60, alpha=0.7, color=PALETTE["bot"],
        label="Bot", density=True, edgecolor="none")
ax.axvline(0.5, color="#FFD700", linestyle="--", lw=1.5, label="threshold=0.5")
ax.set_xlabel("Predicted Probability (Bot)"); ax.set_ylabel("Density")
ax.set_title("OOF Predicted Probability Distribution")
ax.legend()
plt.tight_layout()
save_fig(fig, RF_DIR, "rf_03_proba_dist")

gini_imp = pd.Series(best_rf.feature_importances_, index=FEAT_COLS)
top20_gini = gini_imp.nlargest(20)

tr0, va0 = folds[0]
perm_rf = RandomForestClassifier(**search.best_params_, n_jobs=-1, random_state=SEED)
perm_rf.fit(X[tr0], y[tr0])
perm_res = permutation_importance(perm_rf, X[va0], y[va0],
                                   scoring="roc_auc", n_repeats=10,
                                   n_jobs=-1, random_state=SEED)
top20_perm = pd.Series(perm_res.importances_mean, index=FEAT_COLS).nlargest(20)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("RF Feature Importance", fontsize=14)
for ax, imp, title in zip(axes, [top20_gini, top20_perm],
                          ["Gini Importance (top 20)", "Permutation Importance (top 20)"]):
    ax.barh(range(len(imp)), imp.values[::-1], color=PALETTE["bot"], edgecolor="none")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index[::-1], fontsize=9)
    ax.set_title(title); ax.set_xlabel("Importance")
plt.tight_layout()
save_fig(fig, RF_DIR, "rf_04_feature_importance")

cv_results = pd.DataFrame(search.cv_results_)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(cv_results)), cv_results["mean_test_score"],
        color=PALETTE["human"], lw=1.5, label="Mean CV AUC")
ax.fill_between(range(len(cv_results)),
                cv_results["mean_test_score"] - cv_results["std_test_score"],
                cv_results["mean_test_score"] + cv_results["std_test_score"],
                alpha=0.2, color=PALETTE["human"])
ax.axhline(search.best_score_, color=PALETTE["bot"], lw=1.5, linestyle="--",
           label=f"Best={search.best_score_:.4f}")
ax.set_xlabel("Search Iteration"); ax.set_ylabel("CV AUC")
ax.set_title("RandomizedSearch CV AUC Across Iterations"); ax.legend()
plt.tight_layout()
save_fig(fig, RF_DIR, "rf_05_search_scores")

# ── RF submission + summary ────────────────────────────────────────────────────
best_rf.fit(X, y)
test_proba_rf = best_rf.predict_proba(X_test)[:, 1]

pd.DataFrame({"bidder_id": test["bidder_id"], "prediction": test_proba_rf}).to_csv(
    f"{RF_DIR}/rf_submission.csv", index=False)

pd.DataFrame([{
    "model":               "RandomForest",
    "oof_auc":             float(oof_auc_rf),
    "oof_ap":              float(ap_rf),
    "oof_pr_auc":          float(pr_auc_rf),
    "oof_precision":       float(precision_rf),
    "oof_recall":          float(recall_rf),
    "oof_f1":              float(f1_rf),
    "best_threshold":      float(best_thresh_rf),
    "oof_pred_bot_rate":   float(oof_proba_rf.mean()),
    "test_pred_bot_rate":  float(test_proba_rf.mean()),
}]).to_csv(f"{RF_LOG}/rf_summary.csv", index=False)

print(f"\n✓ Random Forest baseline complete.")
print(f"  OOF AUC   : {oof_auc_rf:.4f}  |  OOF AP : {ap_rf:.4f}  |  PR-AUC : {pr_auc_rf:.4f}")
print(f"  Precision : {precision_rf:.4f}  |  Recall : {recall_rf:.4f}  |  F1 : {f1_rf:.4f}  (thresh={best_thresh_rf:.2f})")
print(f"  Plots     : {RF_DIR}/rf_*.png")
print(f"  Submission: {RF_DIR}/rf_submission.csv")
print(f"  Summary   : {RF_LOG}/rf_summary.csv")


# =============================================================================
# 10. LIGHTGBM BASELINE (with isotonic calibration)
# =============================================================================
print("\n" + "="*60)
print("10. LIGHTGBM BASELINE")
print("="*60)

LGBM_PARAMS = dict(
    objective         = "binary",
    metric            = "auc",
    boosting_type     = "gbdt",
    n_estimators      = 1000,
    learning_rate     = 0.03,
    num_leaves        = 63,
    max_depth         = -1,
    min_child_samples = 10,
    feature_fraction  = 0.7,
    bagging_fraction  = 0.8,
    bagging_freq      = 1,
    lambda_l1         = 0.1,
    lambda_l2         = 1.0,
    scale_pos_weight  = scale_pos,
    n_jobs            = -1,
    random_state      = SEED,
    verbose           = -1,
)

oof_proba_lgbm  = np.zeros(len(y), dtype=np.float64)
test_probas_lgbm = np.zeros((len(X_test), N_SPLITS), dtype=np.float64)
best_iters       = []

for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])

    best_iters.append(model.best_iteration_)
    oof_proba_lgbm[val_idx]   = model.predict_proba(X_val)[:, 1]
    test_probas_lgbm[:, fold - 1] = model.predict_proba(X_test)[:, 1]

    print(f"  Fold {fold}  AUC={roc_auc_score(y_val, oof_proba_lgbm[val_idx]):.4f}"
          f"  best_iter={model.best_iteration_}")

oof_auc_lgbm = roc_auc_score(y, oof_proba_lgbm)
avg_iter      = int(np.mean(best_iters))
test_proba_raw_lgbm = test_probas_lgbm.mean(axis=1)

print(f"\nOOF AUC (LGBM raw)  : {oof_auc_lgbm:.4f}")
print(f"Avg best_iter       : {avg_iter}")

# Isotonic calibration
base_lgbm = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": avg_iter})
calibrated = CalibratedClassifierCV(
    base_lgbm, method="isotonic",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
)
calibrated.fit(X, y)

oof_calib_lgbm  = calibrated.predict_proba(X)[:, 1]
test_proba_lgbm = calibrated.predict_proba(X_test)[:, 1]

print(f"Bot rate raw OOF    : {oof_proba_lgbm.mean():.4f}")
print(f"Bot rate calib OOF  : {oof_calib_lgbm.mean():.4f}")
print(f"Bot rate test cal   : {test_proba_lgbm.mean():.4f}  (expected ~{train_prior:.4f})")

# ── LightGBM plots ────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y, oof_proba_lgbm)
prec, rec, _ = precision_recall_curve(y, oof_proba_lgbm)
ap_lgbm = average_precision_score(y, oof_proba_lgbm)
pr_auc_lgbm = auc(rec, prec)

thresholds_lgbm = np.arange(0.05, 0.95, 0.01)
f1s_lgbm = [f1_score(y, (oof_proba_lgbm >= t).astype(int)) for t in thresholds_lgbm]
best_thresh_lgbm = thresholds_lgbm[int(np.argmax(f1s_lgbm))]
lgbm_oof_labels = (oof_proba_lgbm >= best_thresh_lgbm).astype(int)
precision_lgbm = precision_score(y, lgbm_oof_labels)
recall_lgbm = recall_score(y, lgbm_oof_labels)
f1_lgbm = float(np.max(f1s_lgbm))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"LightGBM — OOF AUC={oof_auc_lgbm:.4f}", fontsize=14)
ax = axes[0]
ax.plot(fpr, tpr, color=PALETTE["bot"], lw=2, label=f"ROC (AUC={oof_auc_lgbm:.4f})")
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["bot"])
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
ax = axes[1]
ax.plot(rec, prec, color=PALETTE["human"], lw=2, label=f"PR (AP={ap_lgbm:.4f})")
ax.fill_between(rec, prec, alpha=0.12, color=PALETTE["human"])
ax.axhline(train_prior, color="#555", lw=1, linestyle="--", label="Chance")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend()
plt.tight_layout()
save_fig(fig, LGBM_DIR, "lgbm_01_roc_pr")

oof_pred_lgbm = (oof_calib_lgbm >= 0.5).astype(int)
cm_lgbm = confusion_matrix(y, oof_pred_lgbm)
print("\nClassification Report (LGBM calibrated):")
print(classification_report(y, oof_pred_lgbm, target_names=["Human", "Bot"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_lgbm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"],
            linewidths=0.5, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (calibrated, threshold=0.5)")
plt.tight_layout()
save_fig(fig, LGBM_DIR, "lgbm_02_confusion_matrix")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Probability Calibration — Before vs After", fontsize=14)
for ax, proba, label, color in [
    (axes[0], oof_proba_lgbm,  "Uncalibrated (LightGBM OOF)", PALETTE["uncalib"]),
    (axes[1], oof_calib_lgbm,  "Calibrated (Isotonic)",        PALETTE["calib"]),
]:
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, label=label)
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color=color)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(label); ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
save_fig(fig, LGBM_DIR, "lgbm_03_calibration_curve")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Predicted Probability Distribution — Raw vs Calibrated", fontsize=13)
for ax, proba, title in [
    (axes[0], oof_proba_lgbm,  "Uncalibrated"),
    (axes[1], oof_calib_lgbm,  "Calibrated (Isotonic)"),
]:
    ax.hist(proba[y == 0], bins=60, alpha=0.7, color=PALETTE["human"],
            label="Human", density=True, edgecolor="none")
    ax.hist(proba[y == 1], bins=60, alpha=0.7, color=PALETTE["bot"],
            label="Bot", density=True, edgecolor="none")
    ax.axvline(train_prior, color="#FFD700", lw=1.5, linestyle="--",
               label=f"True prior ({train_prior:.3f})")
    ax.set_xlabel("P(bot)"); ax.set_ylabel("Density")
    ax.set_title(title); ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, LGBM_DIR, "lgbm_04_proba_dist")

final_lgbm = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": avg_iter})
final_lgbm.fit(X, y)
importance_df = pd.DataFrame({
    "feature": FEAT_COLS,
    "gain":    final_lgbm.booster_.feature_importance(importance_type="gain"),
    "split":   final_lgbm.booster_.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle("LightGBM Feature Importance (top 30)", fontsize=14)
for ax, col, title, color in [
    (axes[0], "gain",  "Gain (total reduction in loss)", PALETTE["bot"]),
    (axes[1], "split", "Split count",                    PALETTE["human"]),
]:
    sub = importance_df.nlargest(30, col)
    ax.barh(range(len(sub)), sub[col].values[::-1], color=color, edgecolor="none")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["feature"].values[::-1], fontsize=8)
    ax.set_title(title); ax.set_xlabel(col.capitalize())
plt.tight_layout()
save_fig(fig, LGBM_DIR, "lgbm_05_feature_importance")

burst_feats = importance_df[importance_df["feature"].str.startswith("burst")]
time_feats  = importance_df[importance_df["feature"].str.contains("time")]
spotlight   = pd.concat([burst_feats, time_feats]).drop_duplicates()
if not spotlight.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(spotlight["feature"], spotlight["gain"],
            color=PALETTE["calib"], edgecolor="none")
    ax.set_title("Burst & Time Feature Importance (Gain)")
    ax.set_xlabel("Gain"); ax.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, LGBM_DIR, "lgbm_06_burst_time_importance")

print("\n  Collecting learning curve for fold 0 …")
evals_result = {}
fold0_tr, fold0_val = folds[0]
probe = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": 1000})
probe.fit(
    X[fold0_tr], y[fold0_tr],
    eval_set=[(X[fold0_tr], y[fold0_tr]), (X[fold0_val], y[fold0_val])],
    eval_names=["train", "valid"], eval_metric="auc",
    callbacks=[lgb.record_evaluation(evals_result),
               lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(-1)],
)
train_auc_curve = evals_result["train"]["auc"]
valid_auc_curve = evals_result["valid"]["auc"]
iters = range(1, len(train_auc_curve) + 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(iters, train_auc_curve, color=PALETTE["human"], lw=1.5, label="Train AUC")
ax.plot(iters, valid_auc_curve, color=PALETTE["bot"],   lw=1.5, label="Val AUC")
ax.axvline(probe.best_iteration_, color="#FFD700", lw=1.5, linestyle="--",
           label=f"Best iter={probe.best_iteration_}")
ax.set_xlabel("Iteration"); ax.set_ylabel("AUC")
ax.set_title("Learning Curve — Fold 0"); ax.legend()
plt.tight_layout()
save_fig(fig, LGBM_DIR, "lgbm_07_learning_curve")

# ── LightGBM submission + summary ─────────────────────────────────────────────
pd.DataFrame({"bidder_id": test["bidder_id"], "prediction": test_proba_lgbm}).to_csv(
    f"{LGBM_DIR}/lgbm_submission.csv", index=False)

pd.DataFrame([{
    "model":              "LightGBM",
    "oof_auc":            float(oof_auc_lgbm),
    "oof_ap":             float(ap_lgbm),
    "oof_pr_auc":         float(pr_auc_lgbm),
    "oof_precision":      float(precision_lgbm),
    "oof_recall":         float(recall_lgbm),
    "oof_f1":             float(f1_lgbm),
    "best_threshold":     float(best_thresh_lgbm),
    "oof_pred_bot_rate":  float(oof_proba_lgbm.mean()),
    "test_pred_bot_rate": float(test_proba_lgbm.mean()),
}]).to_csv(f"{LGBM_LOG}/lgbm_summary.csv", index=False)

print(f"\n✓ LightGBM baseline complete.")
print(f"  OOF AUC   : {oof_auc_lgbm:.4f}  |  OOF AP : {ap_lgbm:.4f}  |  PR-AUC : {pr_auc_lgbm:.4f}")
print(f"  Precision : {precision_lgbm:.4f}  |  Recall : {recall_lgbm:.4f}  |  F1 : {f1_lgbm:.4f}  (thresh={best_thresh_lgbm:.2f})")
print(f"  Avg iter  : {avg_iter}")
print(f"  Plots     : {LGBM_DIR}/lgbm_*.png")
print(f"  Submission: {LGBM_DIR}/lgbm_submission.csv")
print(f"  Summary   : {LGBM_LOG}/lgbm_summary.csv")
print(f"\n  Feature importance top-5 (gain):")
print(importance_df[["feature", "gain"]].head(5).to_string(index=False))
