#!/usr/bin/env python
# coding: utf-8

# # Facebook Recruiting IV: Human or Robot — Data Preprocessing Pipeline
# =====================================================================
# ## Covers: Data understanding · Cleaning · Augmentation · Visualization · Feature Engineering

# In[1]:


# ─── 0. Imports ──────────────────────────────────────────────────────────────
import os, warnings, zipfile, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)

# ─── Aesthetic config ─────────────────────────────────────────────────────────
PALETTE = {"bot": "#E84545", "human": "#2B9EB3", "neutral": "#6C757D"}
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 140, "figure.facecolor": "#0F1117",
                     "axes.facecolor": "#1A1D27", "axes.labelcolor": "#E0E0E0",
                     "xtick.color": "#A0A0A0", "ytick.color": "#A0A0A0",
                     "text.color": "#E0E0E0", "grid.color": "#2A2D37",
                     "axes.spines.top": False, "axes.spines.right": False})

SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Helper ───────────────────────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ↳ Saved  {SAVE_DIR}/{name}.png")


# In[2]:


import os
import glob
import zipfile
import pandas as pd

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "="*60)
print("1. LOADING DATA")
print("="*60)

INPUT_DIR = "/kaggle/input/competitions/facebook-recruiting-iv-human-or-bot" 
WORK_DIR = "/kaggle/working" # ← Kaggle's writable directory

# ── Auto-extract any zips in INPUT_DIR to WORK_DIR ───────────────────────────
for zpath in glob.glob(f"{INPUT_DIR}/*.zip"):
    print(f"Extracting {zpath} …")
    with zipfile.ZipFile(zpath, "r") as z:
        members = z.namelist()

        # Only extract CSVs not already present in the WORKING directory
        to_extract = [m for m in members 
                      if m.endswith(".csv") and not os.path.exists(os.path.join(WORK_DIR, m))]

        if to_extract:
            # Extract to WORK_DIR instead of INPUT_DIR
            z.extractall(WORK_DIR, members=to_extract)
            print(f"  ↳ extracted to {WORK_DIR}: {to_extract}")
        else:
            print(f"  ↳ CSVs already present, skipping.")

# Helper function to find the CSV whether it was extracted to working or already unzipped in input
def get_file_path(filename):
    if os.path.exists(f"{WORK_DIR}/{filename}"):
        return f"{WORK_DIR}/{filename}"
    return f"{INPUT_DIR}/{filename}"

# Read the CSVs
train  = pd.read_csv(get_file_path("train.csv"))
test   = pd.read_csv(get_file_path("test.csv"))
bids   = pd.read_csv(get_file_path("bids.csv"))

for name, df in [("train", train), ("test", test), ("bids", bids)]:
    print(f"\n{name}.csv  shape={df.shape}")
    print(df.dtypes.to_string())


# In[3]:


# =============================================================================
# 2. DATA UNDERSTANDING
# =============================================================================
print("\n" + "="*60)
print("2. DATA UNDERSTANDING")
print("="*60)

# ── 2a. Basic stats ──────────────────────────────────────────────────────────
print("\n── TRAIN ──")
print(train.describe(include="all").T)
print("\n── BIDS (numeric) ──")
print(bids.describe().T)

# ── 2b. Null audit ──────────────────────────────────────────────────────────
def null_report(df, label):
    null = df.isnull().sum()
    pct  = null / len(df) * 100
    return pd.DataFrame({"nulls": null, "pct": pct}).query("nulls > 0").assign(source=label)

null_df = pd.concat([null_report(train, "train"),
                     null_report(test,  "test"),
                     null_report(bids,  "bids")])
print("\n── NULL REPORT ──")
print(null_df.to_string() if not null_df.empty else "No nulls found.")

# ── 2c. Class balance ────────────────────────────────────────────────────────
print("\n── CLASS BALANCE (train) ──")
vc = train["outcome"].value_counts()
print(vc.to_string())
print(f"  Bot ratio: {vc[1]/len(train)*100:.1f}%")

# ── 2d. Overlap — how many bidders have bids? ─────────────────────────────────
train_with_bids = set(train.bidder_id) & set(bids.bidder_id)
test_with_bids  = set(test.bidder_id)  & set(bids.bidder_id)
print(f"\nTrain bidders with bids : {len(train_with_bids)}/{len(train)} "
      f"({len(train_with_bids)/len(train)*100:.1f}%)")
print(f"Test  bidders with bids : {len(test_with_bids)}/{len(test)} "
      f"({len(test_with_bids)/len(test)*100:.1f}%)")

# ── 2e. Bids per bidder distribution (raw) ────────────────────────────────────
bids_per_bidder_raw = bids.groupby("bidder_id").size()
print(f"\nBids-per-bidder: min={bids_per_bidder_raw.min()} "
      f"max={bids_per_bidder_raw.max()} "
      f"median={bids_per_bidder_raw.median():.0f}")


# In[4]:


# =============================================================================
# 3. VISUALISATION — RAW DATA
# =============================================================================
print("\n" + "="*60)
print("3. RAW DATA VISUALISATION")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Raw Data Overview — Facebook Human vs Robot", fontsize=16, y=1.01)

# 3a. Class balance
ax = axes[0, 0]
bars = ax.bar(["Human (0)", "Bot (1)"], vc.values,
              color=[PALETTE["human"], PALETTE["bot"]], width=0.5, edgecolor="none")
ax.bar_label(bars, fmt="%d", padding=4, color="#E0E0E0")
ax.set_title("Class Balance (train)")
ax.set_ylabel("Count")

# 3b. Bids-per-bidder (log scale)
ax = axes[0, 1]
bids_per_bidder_raw.clip(upper=2000).hist(bins=60, ax=ax, color=PALETTE["neutral"],
                                          edgecolor="none", log=True)
ax.set_title("Bids per Bidder (raw, clipped@2k)")
ax.set_xlabel("Bid count"); ax.set_ylabel("Frequency (log)")

# 3c. Country distribution (top 15)
ax = axes[0, 2]
bids["country"].value_counts().head(15).plot.barh(ax=ax, color=PALETTE["neutral"],
                                                   edgecolor="none")
ax.set_title("Top 15 Countries (bids)")
ax.invert_yaxis()

# 3d. Device diversity
ax = axes[1, 0]
bids["device"].value_counts().head(10).plot.bar(ax=ax, color=PALETTE["neutral"],
                                                 edgecolor="none", rot=45)
ax.set_title("Top 10 Devices (bids)")

# 3e. Merchandise category
ax = axes[1, 1]
bids["merchandise"].value_counts().plot.pie(ax=ax, autopct="%1.1f%%",
    colors=sns.color_palette("Set2"), startangle=140, labels=None)
ax.legend(bids["merchandise"].value_counts().index, loc="center left",
          bbox_to_anchor=(1, 0.5), fontsize=8)
ax.set_title("Merchandise Distribution")
ax.set_ylabel("")

# 3f. Time distribution (sample 200k rows for speed)
ax = axes[1, 2]
sample_time = bids["time"].sample(min(200_000, len(bids)), random_state=42)
ax.hist(sample_time, bins=100, color=PALETTE["human"], edgecolor="none", alpha=0.8)
ax.set_title("Bid Timestamp Distribution (sample)")
ax.set_xlabel("Time"); ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
save(fig, "01_raw_overview")


# In[5]:


# =============================================================================
# 4. DATA CLEANING
# =============================================================================
print("\n" + "="*60)
print("4. DATA CLEANING")
print("="*60)

bids_clean = bids.copy()
train_clean = train.copy()
test_clean  = test.copy()

# ── 4a. Duplicates ───────────────────────────────────────────────────────────
dup_bids = bids_clean.duplicated(subset="bid_id").sum()
print(f"Duplicate bid_ids in bids.csv: {dup_bids}")
bids_clean.drop_duplicates(subset="bid_id", inplace=True)

# ── 4b. Null handling ────────────────────────────────────────────────────────
# country: fill missing with 'UNKNOWN'
null_country = bids_clean["country"].isnull().sum()
bids_clean["country"] = bids_clean["country"].fillna("UNKNOWN")
print(f"country nulls filled: {null_country}")

# url: fill with 'UNKNOWN', then extract domain prefix (first segment)
null_url = bids_clean["url"].isnull().sum()
bids_clean["url"] = bids_clean["url"].fillna("UNKNOWN")
print(f"url nulls filled: {null_url}")

# ip: fill with UNKNOWN
null_ip = bids_clean["ip"].isnull().sum()
bids_clean["ip"] = bids_clean["ip"].fillna("UNKNOWN")
print(f"ip nulls filled: {null_ip}")

# address (train/test): fill with UNKNOWN
for df in [train_clean, test_clean]:
    df["address"] = df["address"].fillna("UNKNOWN")
    df["payment_account"] = df["payment_account"].fillna("UNKNOWN")

# ── 4c. Outlier audit — bids per bidder ─────────────────────────────────────
bpb = bids_clean.groupby("bidder_id").size()
z_scores = np.abs(stats.zscore(bpb))
outlier_bidders = bpb[z_scores > 4]
print(f"\nBidders with z>4 bid count: {len(outlier_bidders)}")
print(outlier_bidders.sort_values(ascending=False).head(10))

# ── 4d. Time sanity — detect zero-interval bursts ────────────────────────────
bids_clean = bids_clean.sort_values(["bidder_id", "time"])
bids_clean["time_diff"] = (bids_clean.groupby("bidder_id")["time"]
                            .diff().fillna(0))
zero_interval = (bids_clean["time_diff"] == 0).sum()
print(f"\nBids with zero time-diff to prev bid (same bidder): {zero_interval} "
      f"({zero_interval/len(bids_clean)*100:.1f}%)")

print(f"\nCleaned bids shape: {bids_clean.shape}")


# In[6]:


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*60)
print("5. FEATURE ENGINEERING")
print("="*60)

# ── Helper: entropy ───────────────────────────────────────────────────────────
def entropy(series):
    p = series.value_counts(normalize=True)
    return -np.sum(p * np.log2(p + 1e-9))

# ── 5a. Aggregate bids → per-bidder features ─────────────────────────────────
g = bids_clean.groupby("bidder_id")

feats = pd.DataFrame({
    # Volume
    "bid_count"             : g["bid_id"].count(),
    "unique_auctions"       : g["auction"].nunique(),
    "unique_devices"        : g["device"].nunique(),
    "unique_countries"      : g["country"].nunique(),
    "unique_ips"            : g["ip"].nunique(),
    "unique_urls"           : g["url"].nunique(),
    "unique_merchandise"    : g["merchandise"].nunique(),

    # Ratios (diversity / activity)
    "bids_per_auction"      : g["bid_id"].count() / g["auction"].nunique(),
    "ips_per_bid"           : g["ip"].nunique() / g["bid_id"].count(),
    "countries_per_bid"     : g["country"].nunique() / g["bid_id"].count(),
    "devices_per_bid"       : g["device"].nunique() / g["bid_id"].count(),

    # Time-based
    "time_span"             : g["time"].max() - g["time"].min(),
    "time_mean"             : g["time"].mean(),
    "time_std"              : g["time"].std().fillna(0),
    "time_diff_mean"        : g["time_diff"].mean(),
    "time_diff_std"         : g["time_diff"].std().fillna(0),
    "time_diff_min"         : g["time_diff"].min(),
    "zero_interval_count"   : g["time_diff"].apply(lambda x: (x == 0).sum()),
    "zero_interval_ratio"   : g["time_diff"].apply(lambda x: (x == 0).mean()),

    # Entropy (information diversity)
    "country_entropy"       : g["country"].apply(entropy),
    "device_entropy"        : g["device"].apply(entropy),
    "url_entropy"           : g["url"].apply(entropy),
    "merchandise_entropy"   : g["merchandise"].apply(entropy),
})

# ── 5b. Merchandise one-hot pivots ────────────────────────────────────────────
merch_dummies = (bids_clean.groupby(["bidder_id", "merchandise"])
                            .size()
                            .unstack(fill_value=0))
merch_dummies.columns = [f"merch_{c}" for c in merch_dummies.columns]
merch_props = merch_dummies.div(merch_dummies.sum(axis=1), axis=0)
merch_props.columns = [f"{c}_ratio" for c in merch_dummies.columns]

# ── 5c. Top-5 country flags ───────────────────────────────────────────────────
top_countries = bids_clean["country"].value_counts().head(5).index.tolist()
country_flags = (bids_clean.assign(flag=bids_clean["country"].isin(top_countries))
                             .groupby("bidder_id")["flag"]
                             .mean()
                             .rename("top5_country_ratio"))

# ── 5d. IP subnet features ────────────────────────────────────────────────────
def ip_prefix(ip_series, parts=2):
    """Extract /16 subnet (first 2 octets) from IPv4, ignoring non-standard."""
    def extract(ip):
        segs = str(ip).split(".")
        return ".".join(segs[:parts]) if len(segs) >= parts else "OTHER"
    return ip_series.apply(extract)

bids_clean["ip_subnet"] = ip_prefix(bids_clean["ip"])
feats["unique_subnets"] = g["ip_subnet"].nunique()
feats["subnets_per_bid"] = feats["unique_subnets"] / feats["bid_count"]

# ── 5e. Time-prefix burst features ────────────────────────────────────────────
# Bots fire bids in tight clusters that share the same high-order timestamp
# digits. For each prefix length we count the largest single-prefix burst
# per bidder — a strong discriminator between scripted and human behaviour.
#
# time values are large integers (nanoseconds or ticks); stripping the last
# N digits collapses bids into "same-moment" buckets at varying resolutions.
time_max_digits = len(str(int(bids_clean["time"].max())))

for n_strip in [1, 2, 3, 4, 5]:               # coarse → fine granularity
    divisor = 10 ** n_strip
    col_name = f"time_prefix_strip{n_strip}"
    bids_clean[col_name] = (bids_clean["time"] // divisor).astype(np.int64)

    # max burst size = largest number of bids sharing one prefix bucket
    burst_max = (bids_clean
                 .groupby(["bidder_id", col_name])
                 .size()
                 .groupby("bidder_id").max()
                 .rename(f"burst_max_strip{n_strip}"))

    # mean burst size across all buckets (bidder-level burstiness)
    burst_mean = (bids_clean
                  .groupby(["bidder_id", col_name])
                  .size()
                  .groupby("bidder_id").mean()
                  .rename(f"burst_mean_strip{n_strip}"))

    feats = feats.join(burst_max,  how="left")
    feats = feats.join(burst_mean, how="left")

# Ratio: max burst relative to total bids (normalised burstiness)
for n_strip in [1, 2, 3, 4, 5]:
    feats[f"burst_ratio_strip{n_strip}"] = (
        feats[f"burst_max_strip{n_strip}"] / feats["bid_count"]
    )

print(f"  Time-prefix features added — strips 1–5 (max/mean/ratio each)")

# Drop temp prefix columns from bids_clean to save memory
bids_clean.drop(columns=[c for c in bids_clean.columns
                          if c.startswith("time_prefix_strip")], inplace=True)

# ── 5f. Combine all features ──────────────────────────────────────────────────
all_feats = (feats
             .join(merch_props, how="left")
             .join(country_flags, how="left")
             .fillna(0))

# Log-transform heavy-tailed features
for col in ["bid_count", "unique_auctions", "unique_ips", "unique_urls",
            "time_span", "time_std"] + \
           [f"burst_max_strip{n}" for n in [1,2,3,4,5]]:
    all_feats[f"log_{col}"] = np.log1p(all_feats[col])

print(f"Feature matrix shape: {all_feats.shape}")
print(all_feats.dtypes.value_counts())

# ── 5g. Merge with train/test ─────────────────────────────────────────────────
train_feat = (train_clean
              .merge(all_feats.reset_index(), on="bidder_id", how="left")
              .fillna(0))
test_feat  = (test_clean
              .merge(all_feats.reset_index(), on="bidder_id", how="left")
              .fillna(0))

print(f"\ntrain_feat shape : {train_feat.shape}")
print(f"test_feat  shape : {test_feat.shape}")


# In[7]:


# =============================================================================
# 6. VISUALISATION — CLEANED + ENGINEERED DATA
# =============================================================================
print("\n" + "="*60)
print("6. CLEANED / ENGINEERED DATA VISUALISATION")
print("="*60)

label_map = {0: "human", 1: "bot"}
train_feat["label"] = train_feat["outcome"].map(label_map)
colors = [PALETTE["human"], PALETTE["bot"]]

# ── 6a. Key feature distributions — human vs bot ─────────────────────────────
key_feats = [
    ("log_bid_count",           "Log Bid Count"),
    ("log_unique_ips",          "Log Unique IPs"),
    ("zero_interval_ratio",     "Zero-interval Ratio"),
    ("country_entropy",         "Country Entropy"),
    ("burst_ratio_strip3",      "Burst Ratio (strip-3)"),
    ("burst_max_strip4",        "Max Burst Size (strip-4)"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Feature Distributions: Human vs Bot", fontsize=16)

for ax, (col, title) in zip(axes.flat, key_feats):
    for i, (label, grp) in enumerate(train_feat.groupby("label")):
        vals = grp[col].dropna()
        ax.hist(vals, bins=50, alpha=0.65, color=colors[i],
                label=label, edgecolor="none", density=True)
    ax.set_title(title); ax.legend()

plt.tight_layout()
plt.show()
save(fig, "02_feature_distributions")

# ── 6b. Correlation heatmap (top numeric features) ───────────────────────────
num_cols = [c for c in all_feats.columns if all_feats[c].dtype in [np.float64, np.int64]]
corr = train_feat[num_cols].corr()
# keep only top 20 most correlated with outcome
outcome_corr = corr["bid_count"].abs().sort_values(ascending=False).head(20).index
sub_corr = train_feat[list(outcome_corr) + ["outcome"]].corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(sub_corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.7},
            annot=True, fmt=".2f", annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap (top 20 + outcome)")
plt.tight_layout()
plt.show()
save(fig, "03_correlation_heatmap")

# ── 6c. Boxplots — raw vs log-transformed ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Raw vs Log-Transformed: Bid Count & Unique IPs", fontsize=14)

for i, (raw, log, title) in enumerate([
    ("bid_count", "log_bid_count", "Bid Count"),
    ("unique_ips", "log_unique_ips", "Unique IPs"),
]):
    for j, (col, label) in enumerate([(raw, "Raw"), (log, "Log+1")]):
        ax = axes[i][j]
        data_h = train_feat.loc[train_feat["label"] == "human", col]
        data_b = train_feat.loc[train_feat["label"] == "bot",   col]
        ax.boxplot([data_h, data_b], labels=["Human", "Bot"],
                   patch_artist=True,
                   boxprops=dict(facecolor="none", color="#A0A0A0"),
                   medianprops=dict(color="#FFD700", linewidth=2),
                   whiskerprops=dict(color="#A0A0A0"),
                   capprops=dict(color="#A0A0A0"),
                  flierprops=dict(
                       marker="o",                  # Use a full circle instead of a tiny dot
                       markersize=4,                # Doubled the size for better visibility
                       alpha=0.6,                   # Transparency helps if there is dense overlapping
                       markerfacecolor="#FF6B6B",   # Specifically set the fill color
                       markeredgecolor="none"       # Remove the edge so colors don't muddy together
                   ))
        ax.set_title(f"{title} — {label}")

plt.tight_layout()
plt.show()
save(fig, "04_raw_vs_log_boxplots")

# ── 6d. Zero-interval ratio vs bid count (scatter) ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for label, grp in train_feat.groupby("label"):
    ax.scatter(grp["log_bid_count"], grp["zero_interval_ratio"],
               c=PALETTE[label], alpha=0.5, s=15, label=label, edgecolors="none")
ax.set_xlabel("Log Bid Count"); ax.set_ylabel("Zero-Interval Ratio")
ax.set_title("Zero-Interval Ratio vs Log Bid Count")
ax.legend()
plt.tight_layout()
plt.show()
save(fig, "05_zero_interval_scatter")

# ── 6e. Merchandise ratio — bot vs human ──────────────────────────────────────
merch_cols = [c for c in train_feat.columns if c.startswith("merch_") and "ratio" in c]
if merch_cols:
    means = train_feat.groupby("label")[merch_cols].mean().T
    means.index = [c.replace("merch_","").replace("_ratio","") for c in means.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    means.plot.bar(ax=ax, color=[PALETTE["human"], PALETTE["bot"]], edgecolor="none",
                   rot=30)
    ax.set_title("Merchandise Category Ratio — Human vs Bot")
    ax.set_ylabel("Mean ratio"); ax.legend()
    plt.tight_layout()
    plt.show()
    save(fig, "06_merchandise_ratio")

# ── 6f. Country entropy comparison ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for i, (label, grp) in enumerate(train_feat.groupby("label")):
    ax.hist(grp["country_entropy"], bins=40, alpha=0.7, color=colors[i],
            label=label, edgecolor="none", density=True)
ax.set_xlabel("Country Entropy"); ax.set_ylabel("Density")
ax.set_title("Country Entropy Distribution — Human vs Bot")
ax.legend()
plt.tight_layout()
plt.show()
save(fig, "07_country_entropy")


# In[8]:


# =============================================================================
# 7. EXPORT CLEANED FEATURES
# =============================================================================
print("\n" + "="*60)
print("7. EXPORTING")
print("="*60)

train_feat.drop(columns=["label"], inplace=True)
train_feat.to_csv("train_features.csv", index=False)
test_feat.to_csv("test_features.csv",   index=False)
all_feats.to_csv("bidder_features.csv")

print(f"train_features.csv  → {train_feat.shape}")
print(f"test_features.csv   → {test_feat.shape}")
print(f"bidder_features.csv → {all_feats.shape}")

# ── 7a. Feature summary ──────────────────────────────────────────────────────
feat_cols = [c for c in train_feat.columns
             if c not in ["bidder_id", "payment_account", "address", "outcome"]]
print(f"\nTotal engineered features : {len(feat_cols)}")

feature_groups = {
    "Volume"      : [c for c in feat_cols if any(x in c for x in ["count","unique","log"])],
    "Ratio"       : [c for c in feat_cols if "ratio" in c or "per_" in c],
    "Time"        : [c for c in feat_cols if "time" in c],
    "Burst"       : [c for c in feat_cols if "burst" in c],
    "Entropy"     : [c for c in feat_cols if "entropy" in c],
    "Merchandise" : [c for c in feat_cols if "merch" in c],
    "Geography"   : [c for c in feat_cols if "country" in c or "subnet" in c],
}

for group, cols in feature_groups.items():
    print(f"  {group:15s}: {len(cols):3d} features")

print("\n✓ Preprocessing pipeline complete.")
print(f"  Plots saved to   ./{SAVE_DIR}/")
print(f"  Features saved to ./train_features.csv, test_features.csv")


# In[9]:


cleaned_bids = pd.read_csv('/kaggle/working/bidder_features.csv')
cleaned_train = pd.read_csv('/kaggle/working/train_features.csv')
cleaned_test = pd.read_csv('/kaggle/working/test_features.csv')

cleaned_bids.head()


# In[10]:


cleaned_bids.tail()


# In[11]:


cleaned_train.head()


# In[12]:


cleaned_train.tail()


# In[13]:


cleaned_test.head()


# In[14]:


cleaned_test.tail()


# # Facebook Recruiting IV: Human or Robot — Random Forest Classifier
# =================================================================
# ## Requires: train_features.csv, test_features.csv  (output of preprocess.py)
# ## Outputs : submission.csv, plots/rf_*.png

# In[15]:


# ─── Imports ──────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, precision_recall_curve,
                              average_precision_score)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore")

# ─── Aesthetic config ─────────────────────────────────────────────────────────
PALETTE = {"bot": "#E84545", "human": "#2B9EB3", "neutral": "#6C757D"}
plt.rcParams.update({"figure.dpi": 140, "figure.facecolor": "#0F1117",
                     "axes.facecolor": "#1A1D27", "axes.labelcolor": "#E0E0E0",
                     "xtick.color": "#A0A0A0", "ytick.color": "#A0A0A0",
                     "text.color": "#E0E0E0", "grid.color": "#2A2D37",
                     "axes.spines.top": False, "axes.spines.right": False})

SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)
SEED = 42

def save(fig, name):
    fig.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ↳ Saved  {SAVE_DIR}/{name}.png")


# In[16]:


# =============================================================================
# 1. LOAD FEATURES
# =============================================================================
print("\n" + "="*60)
print("1. LOADING FEATURES")
print("="*60)

META_COLS = ["bidder_id", "payment_account", "address", "outcome", "label"]

train = pd.read_csv("train_features.csv")
test  = pd.read_csv("test_features.csv")

FEAT_COLS = [c for c in train.columns if c not in META_COLS]

X      = train[FEAT_COLS].values.astype(np.float32)
y      = train["outcome"].values.astype(np.int32)
X_test = test[FEAT_COLS].values.astype(np.float32)

train_prior = y.mean()
scale_pos   = (1 - train_prior) / train_prior   # for LightGBM pos weighting

print(f"Train  : X={X.shape}  bot_rate={train_prior:.4f}")
print(f"Test   : X_test={X_test.shape}")
print(f"scale_pos_weight = {scale_pos:.1f}  (class imbalance correction)")
print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS[:6]} …")


# In[17]:


# =============================================================================
# 2. BASELINE RF (quick sanity check)
# =============================================================================
print("\n" + "="*60)
print("2. BASELINE RF — 5-FOLD CV")
print("="*60)

rf_base = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",   # handles imbalance without SMOTE
    max_features="sqrt",
    n_jobs=-1,
    random_state=SEED,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
base_scores = cross_val_score(rf_base, X, y, cv=cv,
                               scoring="roc_auc", n_jobs=-1)
print(f"Baseline AUC  : {base_scores.mean():.4f} ± {base_scores.std():.4f}")


# In[18]:


# =============================================================================
# 3. HYPERPARAMETER SEARCH
# =============================================================================
print("\n" + "="*60)
print("3. RANDOMIZED SEARCH (n_iter=30)")
print("="*60)

param_dist = {
    "n_estimators"    : randint(200, 800),
    "max_depth"       : [None, 10, 20, 30, 40],
    "min_samples_leaf": randint(1, 10),
    "min_samples_split": randint(2, 20),
    "max_features"    : ["sqrt", "log2", 0.3, 0.5],
    "class_weight"    : ["balanced", "balanced_subsample"],
}

search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=SEED),
    param_distributions=param_dist,
    n_iter=30,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    refit=True,
    verbose=1,
    random_state=SEED,
    n_jobs=-1,
)
search.fit(X, y)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV AUC : {search.best_score_:.4f}")

best_rf = search.best_estimator_


# In[19]:


# =============================================================================
# 4. FULL TRAIN + OOF PREDICTIONS (for metrics)
# =============================================================================
print("\n" + "="*60)
print("4. OUT-OF-FOLD PREDICTIONS")
print("="*60)

oof_proba = np.zeros(len(y), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    fold_rf = RandomForestClassifier(**search.best_params_,
                                     n_jobs=-1, random_state=SEED)
    fold_rf.fit(X_tr, y_tr)
    oof_proba[val_idx] = fold_rf.predict_proba(X_val)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_proba[val_idx])
    print(f"  Fold {fold+1}  AUC={fold_auc:.4f}")

oof_auc = roc_auc_score(y, oof_proba)
print(f"\nOOF AUC (all folds) : {oof_auc:.4f}")


# In[20]:


# =============================================================================
# 5. EVALUATION PLOTS
# =============================================================================
print("\n" + "="*60)
print("5. EVALUATION PLOTS")
print("="*60)

# ── 5a. ROC + PR curves ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y, oof_proba)
prec, rec, _ = precision_recall_curve(y, oof_proba)
ap = average_precision_score(y, oof_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Random Forest — OOF AUC={oof_auc:.4f}", fontsize=14)

ax = axes[0]
ax.plot(fpr, tpr, color=PALETTE["bot"], lw=2, label=f"ROC (AUC={oof_auc:.4f})")
ax.plot([0,1],[0,1], "--", color="#555", lw=1)
ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["bot"])
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()

ax = axes[1]
ax.plot(rec, prec, color=PALETTE["human"], lw=2, label=f"PR (AP={ap:.4f})")
ax.fill_between(rec, prec, alpha=0.12, color=PALETTE["human"])
ax.axhline(y.mean(), color="#555", lw=1, linestyle="--", label="Chance")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend()

plt.tight_layout()
plt.show()
save(fig, "rf_01_roc_pr")

# ── 5b. Confusion matrix at 0.5 threshold ────────────────────────────────────
oof_pred = (oof_proba >= 0.5).astype(int)
cm = confusion_matrix(y, oof_pred)
print("\nClassification Report:\n")
print(classification_report(y, oof_pred, target_names=["Human", "Bot"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"],
            linewidths=0.5, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (threshold=0.5)")
plt.tight_layout()
plt.show()
save(fig, "rf_02_confusion_matrix")

# ── 5c. Probability distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(oof_proba[y == 0], bins=60, alpha=0.7, color=PALETTE["human"],
        label="Human", density=True, edgecolor="none")
ax.hist(oof_proba[y == 1], bins=60, alpha=0.7, color=PALETTE["bot"],
        label="Bot", density=True, edgecolor="none")
ax.axvline(0.5, color="#FFD700", linestyle="--", lw=1.5, label="threshold=0.5")
ax.set_xlabel("Predicted Probability (Bot)"); ax.set_ylabel("Density")
ax.set_title("OOF Predicted Probability Distribution")
ax.legend()
plt.tight_layout()
plt.show()
save(fig, "rf_03_proba_dist")

# ── 5d. Feature importance (Gini + Permutation) ───────────────────────────────
# Gini importance from best_rf (already fitted on full data by refit=True)
gini_imp = pd.Series(best_rf.feature_importances_, index=FEAT_COLS)
top20_gini = gini_imp.nlargest(20)

# Permutation importance (on OOF fold 0 for speed)
tr0, va0 = next(cv.split(X, y))
perm_rf = RandomForestClassifier(**search.best_params_, n_jobs=-1, random_state=SEED)
perm_rf.fit(X[tr0], y[tr0])
perm_res = permutation_importance(perm_rf, X[va0], y[va0],
                                   scoring="roc_auc", n_repeats=10,
                                   n_jobs=-1, random_state=SEED)
perm_imp = pd.Series(perm_res.importances_mean, index=FEAT_COLS)
top20_perm = perm_imp.nlargest(20)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Feature Importance", fontsize=14)

for ax, imp, title in zip(axes,
                           [top20_gini, top20_perm],
                           ["Gini Importance (top 20)", "Permutation Importance (top 20)"]):
    bars = ax.barh(range(len(imp)), imp.values[::-1],
                   color=PALETTE["bot"], edgecolor="none")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index[::-1], fontsize=9)
    ax.set_title(title); ax.set_xlabel("Importance")

plt.tight_layout()
plt.show()
save(fig, "rf_04_feature_importance")

# ── 5e. CV score distribution ─────────────────────────────────────────────────
cv_results = pd.DataFrame(search.cv_results_)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(cv_results)), cv_results["mean_test_score"],
        color=PALETTE["human"], lw=1.5, label="Mean CV AUC")
ax.fill_between(range(len(cv_results)),
                cv_results["mean_test_score"] - cv_results["std_test_score"],
                cv_results["mean_test_score"] + cv_results["std_test_score"],
                alpha=0.2, color=PALETTE["human"])
ax.axhline(search.best_score_, color=PALETTE["bot"], lw=1.5,
           linestyle="--", label=f"Best={search.best_score_:.4f}")
ax.set_xlabel("Search Iteration"); ax.set_ylabel("CV AUC")
ax.set_title("RandomizedSearch CV AUC Across Iterations"); ax.legend()
plt.tight_layout()
plt.show()
save(fig, "rf_05_search_scores")


# In[21]:


# =============================================================================
# 6. FINAL PREDICTION + SUBMISSION
# =============================================================================
print("\n" + "="*60)
print("6. FINAL PREDICTION")
print("="*60)

# Retrain best RF on full training set
best_rf.fit(X, y)
test_proba = best_rf.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "bidder_id": test["bidder_id"],
    "prediction": test_proba,
})
submission.to_csv("submission.csv", index=False)
print(f"submission.csv  shape={submission.shape}")
print(submission.head(10).to_string())
print(f"\nPredicted bot rate (test): {test_proba.mean():.3f}")

print("\n✓ Random Forest pipeline complete.")
print(f"  OOF AUC   : {oof_auc:.4f}")
print(f"  Best CV   : {search.best_score_:.4f}")
print(f"  Plots     : ./{SAVE_DIR}/rf_*.png")
print(f"  Submission: ./submission.csv")


# # Facebook Recruiting IV: Human or Robot — LightGBM + Calibration
# ================================================================
# ## Requires : train_features.csv, test_features.csv  (output of preprocess.py)
# ## Outputs  : submission.csv, plots/lgbm_*.png
# ## Install  : pip install lightgbm scikit-learn matplotlib seaborn

# In[22]:


# ─── Imports ──────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

# ─── Aesthetic config ─────────────────────────────────────────────────────────
PALETTE = {"bot": "#E84545", "human": "#2B9EB3", "neutral": "#6C757D",
           "calib": "#F5A623", "uncalib": "#9B59B6"}
plt.rcParams.update({"figure.dpi": 140, "figure.facecolor": "#0F1117",
                     "axes.facecolor": "#1A1D27", "axes.labelcolor": "#E0E0E0",
                     "xtick.color": "#A0A0A0", "ytick.color": "#A0A0A0",
                     "text.color": "#E0E0E0", "grid.color": "#2A2D37",
                     "axes.spines.top": False, "axes.spines.right": False})

SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)
SEED = 42

def save(fig, name):
    fig.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ↳ Saved  {SAVE_DIR}/{name}.png")


# In[23]:


# =============================================================================
# 1. LOAD FEATURES
# =============================================================================
print("\n" + "="*60)
print("1. LOADING FEATURES")
print("="*60)

META_COLS = ["bidder_id", "payment_account", "address", "outcome", "label"]

train = pd.read_csv("train_features.csv")
test  = pd.read_csv("test_features.csv")

FEAT_COLS = [c for c in train.columns if c not in META_COLS]

X      = train[FEAT_COLS].values.astype(np.float32)
y      = train["outcome"].values.astype(np.int32)
X_test = test[FEAT_COLS].values.astype(np.float32)

train_prior = y.mean()
scale_pos   = (1 - train_prior) / train_prior   # for LightGBM pos weighting

print(f"Train  : X={X.shape}  bot_rate={train_prior:.4f}")
print(f"Test   : X_test={X_test.shape}")
print(f"scale_pos_weight = {scale_pos:.1f}  (class imbalance correction)")
print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS[:6]} …")


# In[24]:


# =============================================================================
# 2. LIGHTGBM PARAMS
# =============================================================================
# scale_pos_weight replaces class_weight="balanced" — it correctly reflects
# the true 5% bot prior rather than flattening it to 50/50, which was the
# root cause of the inflated predicted bot rate with RandomForest.

LGBM_PARAMS = dict(
    objective          = "binary",
    metric             = "auc",
    boosting_type      = "gbdt",
    n_estimators       = 1000,          # early stopping will trim this
    learning_rate      = 0.03,
    num_leaves         = 63,
    max_depth          = -1,
    min_child_samples  = 10,
    feature_fraction   = 0.7,
    bagging_fraction   = 0.8,
    bagging_freq       = 1,
    lambda_l1          = 0.1,
    lambda_l2          = 1.0,
    scale_pos_weight   = scale_pos,     # ← key fix for calibration
    n_jobs             = -1,
    random_state       = SEED,
    verbose            = -1,
)


# In[25]:


# =============================================================================
# 3. OUT-OF-FOLD PREDICTIONS WITH EARLY STOPPING
# =============================================================================
print("\n" + "="*60)
print("3. OOF TRAINING — 5-FOLD STRATIFIED CV")
print("="*60)

cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_proba    = np.zeros(len(y), dtype=np.float64)
test_probas  = np.zeros((len(X_test), 5), dtype=np.float64)  # per-fold test preds
best_iters   = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set        = [(X_val, y_val)],
        eval_metric     = "auc",
        callbacks       = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),          # suppress per-iter logs
        ],
    )

    best_iters.append(model.best_iteration_)
    oof_proba[val_idx]  = model.predict_proba(X_val)[:, 1]
    test_probas[:, fold] = model.predict_proba(X_test)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_proba[val_idx])
    print(f"  Fold {fold+1}  AUC={fold_auc:.4f}  best_iter={model.best_iteration_}")

oof_auc = roc_auc_score(y, oof_proba)
avg_iter = int(np.mean(best_iters))
print(f"\nOOF AUC          : {oof_auc:.4f}")
print(f"Avg best_iter    : {avg_iter}  (use for final refit)")
print(f"Predicted bot rate (raw OOF) : {oof_proba.mean():.4f}  "
      f"(true={train_prior:.4f})")


# In[26]:


# =============================================================================
# 4. PROBABILITY CALIBRATION
# =============================================================================
print("\n" + "="*60)
print("4. ISOTONIC CALIBRATION")
print("="*60)

# ── Why calibrate?
# LightGBM with scale_pos_weight already partially corrects the prior shift
# that inflated RF's bot rate to 13%. Isotonic calibration post-hoc aligns
# the probability scores with empirical frequencies, making them interpretable.
#
# Method: wrap the best-iter LightGBM in CalibratedClassifierCV (isotonic)
# using the same 5-fold CV so calibration data never saw training data.

base_model = lgb.LGBMClassifier(**{**LGBM_PARAMS,
                                    "n_estimators": avg_iter})   # fixed iters

calibrated = CalibratedClassifierCV(
    base_model,
    method = "isotonic",   # non-parametric; better than sigmoid for large N
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
)
calibrated.fit(X, y)

oof_calib_proba = calibrated.predict_proba(X)[:, 1]   # in-sample approx
test_proba_raw  = test_probas.mean(axis=1)             # ensemble of 5 folds
test_proba_cal  = calibrated.predict_proba(X_test)[:, 1]

print(f"Bot rate  raw OOF  : {oof_proba.mean():.4f}")
print(f"Bot rate  calib    : {oof_calib_proba.mean():.4f}")
print(f"Bot rate  test raw : {test_proba_raw.mean():.4f}")
print(f"Bot rate  test cal : {test_proba_cal.mean():.4f}  ← should be ~{train_prior:.4f}")


# In[27]:


# =============================================================================
# 5. EVALUATION PLOTS
# =============================================================================
print("\n" + "="*60)
print("5. EVALUATION PLOTS")
print("="*60)

# ── 5a. ROC + PR curves ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y, oof_proba)
prec, rec, _ = precision_recall_curve(y, oof_proba)
ap = average_precision_score(y, oof_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"LightGBM — OOF AUC={oof_auc:.4f}", fontsize=14)

ax = axes[0]
ax.plot(fpr, tpr, color=PALETTE["bot"], lw=2, label=f"ROC (AUC={oof_auc:.4f})")
ax.plot([0,1],[0,1], "--", color="#555", lw=1)
ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["bot"])
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curve"); ax.legend()

ax = axes[1]
ax.plot(rec, prec, color=PALETTE["human"], lw=2, label=f"PR (AP={ap:.4f})")
ax.fill_between(rec, prec, alpha=0.12, color=PALETTE["human"])
ax.axhline(train_prior, color="#555", lw=1, linestyle="--", label="Chance")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend()

plt.tight_layout()
plt.show()
save(fig, "lgbm_01_roc_pr")

# ── 5b. Confusion matrix ──────────────────────────────────────────────────────
# Use calibrated proba for threshold to get meaningful confusion matrix
oof_pred = (oof_calib_proba >= 0.5).astype(int)
cm = confusion_matrix(y, oof_pred)
print("\nClassification Report (calibrated, threshold=0.5):")
print(classification_report(y, oof_pred, target_names=["Human", "Bot"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"],
            linewidths=0.5, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (calibrated, threshold=0.5)")
plt.tight_layout()
plt.show()
save(fig, "lgbm_02_confusion_matrix")

# ── 5c. Calibration curve (reliability diagram) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Probability Calibration — Before vs After", fontsize=14)

for ax, proba, label, color in [
    (axes[0], oof_proba,        "Uncalibrated (LightGBM OOF)", PALETTE["uncalib"]),
    (axes[1], oof_calib_proba,  "Calibrated (Isotonic)",        PALETTE["calib"]),
]:
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    ax.plot([0,1],[0,1], "--", color="#555", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, label=label)
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color=color)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(label); ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()
save(fig, "lgbm_03_calibration_curve")

# ── 5d. Probability distribution — raw vs calibrated ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Predicted Probability Distribution — Raw vs Calibrated", fontsize=13)

for ax, proba, title in [
    (axes[0], oof_proba,       "Uncalibrated"),
    (axes[1], oof_calib_proba, "Calibrated (Isotonic)"),
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
plt.show()
save(fig, "lgbm_04_proba_dist")

# ── 5e. Feature importance ────────────────────────────────────────────────────
# Re-fit a single model on full data to extract importances
final_model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": avg_iter})
final_model.fit(X, y)

importance_df = pd.DataFrame({
    "feature"  : FEAT_COLS,
    "gain"     : final_model.booster_.feature_importance(importance_type="gain"),
    "split"    : final_model.booster_.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)

top30 = importance_df.head(30)

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
plt.show()
save(fig, "lgbm_05_feature_importance")

# ── 5f. Burst features importance spotlight ───────────────────────────────────
burst_feats = importance_df[importance_df["feature"].str.startswith("burst")]
time_feats  = importance_df[importance_df["feature"].str.contains("time")]
spotlight   = pd.concat([burst_feats, time_feats]).drop_duplicates()

if not spotlight.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(spotlight["feature"], spotlight["gain"],
                   color=PALETTE["calib"], edgecolor="none")
    ax.set_title("Burst & Time Feature Importance (Gain)")
    ax.set_xlabel("Gain")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    save(fig, "lgbm_06_burst_time_importance")

# ── 5g. Learning curve — AUC per fold vs n_estimators (fold 0) ───────────────
print("\n  Collecting learning curve for fold 0 …")
evals_result = {}
fold0_tr, fold0_val = next(cv.split(X, y))
probe = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": 1000})
probe.fit(
    X[fold0_tr], y[fold0_tr],
    eval_set   = [(X[fold0_tr], y[fold0_tr]), (X[fold0_val], y[fold0_val])],
    eval_names = ["train", "valid"],
    eval_metric= "auc",
    callbacks  = [lgb.record_evaluation(evals_result),
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
plt.show()
save(fig, "lgbm_07_learning_curve")


# In[28]:


# =============================================================================
# 6. FINAL SUBMISSION
# =============================================================================
print("\n" + "="*60)
print("6. FINAL SUBMISSION")
print("="*60)

# Use calibrated probabilities — they reflect the true ~5% bot prior
submission = pd.DataFrame({
    "bidder_id" : test["bidder_id"],
    "prediction": test_proba_cal,
})
submission.to_csv("submission_lgbm.csv", index=False)

print(f"submission_lgbm.csv  shape={submission.shape}")
print(submission.head(10).to_string())
print(f"\nPredicted bot rate (calibrated): {test_proba_cal.mean():.4f}  "
      f"(expected ~{train_prior:.4f})")

print("\n✓ LightGBM pipeline complete.")
print(f"  OOF AUC          : {oof_auc:.4f}")
print(f"  Avg best iter    : {avg_iter}")
print(f"  Plots            : ./{SAVE_DIR}/lgbm_*.png")
print(f"  Submission       : ./submission_lgbm.csv")
print(f"\n  Feature importance top-5 (gain):")
print(importance_df[["feature","gain"]].head(5).to_string(index=False))


# # Facebook Recruiting IV: Human or Robot — TabR (Appended After Preprocessing)
# 
# This section is **appended after the original preprocessing notebook without changing any previous cell**.
# 
# The code below is a **clean reimplementation inspired by the official TabR repository structure**:
# - train-only preprocessing for numerical and categorical inputs
# - TabR-style encoder → retrieval → prediction flow
# - training-time self-neighbor removal
# - training-set memory bank for retrieval
# - validation monitoring with early stopping
# - test-time probability export
# 
# It uses the files already exported by the preprocessing section:
# - `train_features.csv`
# - `test_features.csv`
# - `bidder_features.csv`

# In[29]:


# =============================================================================
# 8. TABR IMPORTS + CONFIG
# =============================================================================
import os
import copy
import math
import random
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # optional; if unavailable we fall back to torch.cdist
    TABR_HAS_FAISS = True
except Exception:
    faiss = None
    TABR_HAS_FAISS = False

SEED = 42

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
print("FAISS AVAILABLE:", TABR_HAS_FAISS)

TABR_CONFIG = {
    "seed": 42,
    "val_size": 0.20,
    "batch_size": 256,
    "eval_batch_size": 512,
    "n_epochs": 200,
    "patience": 30,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "context_size": 96,          # close to the official example config
    "d_main": 192,
    "d_multiplier": 2.0,
    "encoder_n_blocks": 0,       # close to official California example
    "predictor_n_blocks": 1,
    "dropout0": 0.10,
    "dropout1": 0.10,
    "context_dropout": 0.10,
}


# In[30]:


# =============================================================================
# 9. LOAD EXPORTED FEATURES FROM THE PREPROCESSING SECTION
# =============================================================================
WORK_DIR_CANDIDATES = [
    "/kaggle/working",
    ".",
]

def find_existing_file(filename: str):
    for base in WORK_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename} in {WORK_DIR_CANDIDATES}")

TRAIN_FEATURES_PATH = find_existing_file("train_features.csv")
TEST_FEATURES_PATH  = find_existing_file("test_features.csv")

tabr_train_full = pd.read_csv(TRAIN_FEATURES_PATH)
tabr_test_full  = pd.read_csv(TEST_FEATURES_PATH)

print("tabr_train_full:", tabr_train_full.shape)
print("tabr_test_full :", tabr_test_full.shape)

TABR_ID_COL = "bidder_id"
TABR_TARGET_COL = "outcome"

tabr_cat_cols = [
    c for c in tabr_train_full.columns
    if c not in [TABR_ID_COL, TABR_TARGET_COL]
    and (tabr_train_full[c].dtype == "object" or str(tabr_train_full[c].dtype).startswith("category"))
]

tabr_num_cols = [
    c for c in tabr_train_full.columns
    if c not in [TABR_ID_COL, TABR_TARGET_COL] + tabr_cat_cols
]

print("Categorical columns:", tabr_cat_cols)
print("Number of categorical columns:", len(tabr_cat_cols))
print("Number of numerical columns  :", len(tabr_num_cols))

tabr_train_idx, tabr_val_idx = train_test_split(
    np.arange(len(tabr_train_full)),
    test_size=TABR_CONFIG["val_size"],
    stratify=tabr_train_full[TABR_TARGET_COL],
    random_state=TABR_CONFIG["seed"],
)

tabr_train_df = tabr_train_full.iloc[tabr_train_idx].reset_index(drop=True)
tabr_val_df   = tabr_train_full.iloc[tabr_val_idx].reset_index(drop=True)
tabr_test_df  = tabr_test_full.copy().reset_index(drop=True)

print("Train split:", tabr_train_df.shape)
print("Val split  :", tabr_val_df.shape)
print("Test split :", tabr_test_df.shape)


# In[31]:


# =============================================================================
# 10. TRAIN-ONLY PREPROCESSING FOR TABR INPUTS
# =============================================================================
def prepare_tabr_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols,
    cat_cols,
    target_col: str,
):
    # --- Numerical features: fit on train only
    if len(num_cols) > 0:
        num_scaler = StandardScaler()
        X_num_train = num_scaler.fit_transform(train_df[num_cols]).astype(np.float32)
        X_num_val   = num_scaler.transform(val_df[num_cols]).astype(np.float32)
        X_num_test  = num_scaler.transform(test_df[num_cols]).astype(np.float32)
    else:
        num_scaler = None
        X_num_train = X_num_val = X_num_test = None

    # --- Categorical features: fit on train only
    # Similar spirit to the official repo:
    # - categories are encoded using training data
    # - unseen categories at validation/test time are sent to a reserved unknown index
    if len(cat_cols) > 0:
        train_cat = train_df[cat_cols].fillna("__MISSING__").astype(str)
        val_cat   = val_df[cat_cols].fillna("__MISSING__").astype(str)
        test_cat  = test_df[cat_cols].fillna("__MISSING__").astype(str)

        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_cat_train = cat_encoder.fit_transform(train_cat).astype(np.int64) + 1
        X_cat_val   = cat_encoder.transform(val_cat).astype(np.int64) + 1
        X_cat_test  = cat_encoder.transform(test_cat).astype(np.int64) + 1

        X_cat_val[X_cat_val < 0] = 0
        X_cat_test[X_cat_test < 0] = 0

        max_train_codes = X_cat_train.max(axis=0)
        cat_cardinalities = [int(x + 1) for x in max_train_codes]  # +1 for the unknown bucket at index 0
    else:
        cat_encoder = None
        X_cat_train = X_cat_val = X_cat_test = None
        cat_cardinalities = []

    y_train = train_df[target_col].astype(np.int64).to_numpy()
    y_val   = val_df[target_col].astype(np.int64).to_numpy()

    return {
        "X_num_train": X_num_train,
        "X_num_val": X_num_val,
        "X_num_test": X_num_test,
        "X_cat_train": X_cat_train,
        "X_cat_val": X_cat_val,
        "X_cat_test": X_cat_test,
        "y_train": y_train,
        "y_val": y_val,
        "num_scaler": num_scaler,
        "cat_encoder": cat_encoder,
        "cat_cardinalities": cat_cardinalities,
    }

tabr_arrays = prepare_tabr_arrays(
    tabr_train_df,
    tabr_val_df,
    tabr_test_df,
    num_cols=tabr_num_cols,
    cat_cols=tabr_cat_cols,
    target_col=TABR_TARGET_COL,
)

for k, v in tabr_arrays.items():
    if isinstance(v, np.ndarray):
        print(k, v.shape, v.dtype)
    elif isinstance(v, list):
        print(k, v)


# In[32]:


# =============================================================================
# 11. TABR MODEL (CLEAN REIMPLEMENTATION INSPIRED BY THE OFFICIAL REPO)
# =============================================================================
class OneHotCatEncoder(nn.Module):
    def __init__(self, cardinalities):
        super().__init__()
        self.cardinalities = list(cardinalities)

    def forward(self, x_cat):
        if x_cat is None or len(self.cardinalities) == 0:
            return None
        chunks = []
        for i, card in enumerate(self.cardinalities):
            xi = x_cat[:, i].clamp(min=0, max=card - 1)
            chunks.append(F.one_hot(xi, num_classes=card).float())
        return torch.cat(chunks, dim=1)


class TabRResidualBlock(nn.Module):
    def __init__(self, d_main, d_hidden, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_main)
        self.lin1 = nn.Linear(d_main, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_main)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        z = self.lin1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.lin2(z)
        z = self.dropout(z)
        return z


class TabRBinaryClassifier(nn.Module):
    def __init__(
        self,
        n_num_features,
        cat_cardinalities,
        d_main=192,
        d_multiplier=2.0,
        encoder_n_blocks=0,
        predictor_n_blocks=1,
        dropout0=0.10,
        dropout1=0.10,
        context_dropout=0.10,
    ):
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = list(cat_cardinalities)
        self.one_hot_encoder = OneHotCatEncoder(self.cat_cardinalities) if len(self.cat_cardinalities) > 0 else None

        input_dim = self.n_num_features + sum(self.cat_cardinalities)
        if input_dim <= 0:
            raise ValueError("TabR requires at least one numerical or categorical feature.")

        d_hidden = int(round(d_main * d_multiplier))

        self.linear = nn.Linear(input_dim, d_main)

        self.blocks0 = nn.ModuleList([
            TabRResidualBlock(d_main, d_hidden, dropout0)
            for _ in range(encoder_n_blocks)
        ])

        self.norm_for_key = nn.LayerNorm(d_main)
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Linear(d_main, d_main)
        self.label_encoder = nn.Embedding(2, d_main)  # binary task: class 0 / class 1

        self.context_dropout = nn.Dropout(context_dropout)

        self.blocks1 = nn.ModuleList([
            TabRResidualBlock(d_main, d_hidden, dropout1)
            for _ in range(predictor_n_blocks)
        ])

        self.head = nn.Linear(d_main, 1)

    def _encode(self, x_num=None, x_cat=None):
        parts = []
        if x_num is not None:
            parts.append(x_num)
        if x_cat is not None and self.one_hot_encoder is not None:
            parts.append(self.one_hot_encoder(x_cat))
        if not parts:
            raise ValueError("No features were provided to the model.")
        x = torch.cat(parts, dim=1)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(self.norm_for_key(x))
        return x, k

    def _search_neighbors(self, query_k, candidate_k, topk):
        topk = int(min(max(topk, 1), candidate_k.shape[0]))
        if TABR_HAS_FAISS:
            candidate_np = candidate_k.detach().cpu().numpy().astype(np.float32)
            query_np = query_k.detach().cpu().numpy().astype(np.float32)
            index = faiss.IndexFlatL2(candidate_np.shape[1])
            index.add(candidate_np)
            distances, indices = index.search(query_np, topk)
            distances = torch.from_numpy(distances).to(query_k.device)
            indices = torch.from_numpy(indices).to(query_k.device)
        else:
            distances = torch.cdist(query_k, candidate_k, p=2.0) ** 2
            distances, indices = torch.topk(distances, k=topk, dim=1, largest=False)
        return distances, indices

    def forward(
        self,
        x_num,
        x_cat,
        candidate_num,
        candidate_cat,
        candidate_y,
        context_size,
        is_train=False,
        batch_train_idx=None,
        candidate_train_idx=None,
    ):
        # Encode query batch
        x, k = self._encode(x_num=x_num, x_cat=x_cat)

        # Encode retrieval memory (train split memory bank)
        candidate_x, candidate_k = self._encode(x_num=candidate_num, x_cat=candidate_cat)

        # Ask for one extra neighbor during training so the self-match can be removed
        requested_neighbors = context_size + 1 if is_train else context_size
        _, neighbor_idx = self._search_neighbors(k, candidate_k, requested_neighbors)

        context_k = candidate_k[neighbor_idx]
        scores = -((k[:, None, :] - context_k) ** 2).sum(dim=-1)

        if is_train and batch_train_idx is not None and candidate_train_idx is not None:
            retrieved_train_ids = candidate_train_idx[neighbor_idx]
            self_mask = retrieved_train_ids.eq(batch_train_idx[:, None])
            scores = scores.masked_fill(self_mask, float("-inf"))

        # After masking self-neighbors, keep the best context_size neighbors
        keep_k = min(context_size, scores.shape[1])
        rerank_idx = torch.topk(scores, k=keep_k, dim=1, largest=True).indices
        neighbor_idx = torch.gather(neighbor_idx, 1, rerank_idx)
        scores = torch.gather(scores, 1, rerank_idx)

        weights = torch.softmax(scores, dim=1)
        weights = self.context_dropout(weights)

        context_k = candidate_k[neighbor_idx]
        context_y = candidate_y[neighbor_idx].long()
        values = self.label_encoder(context_y) + self.T(k[:, None, :] - context_k)

        context_x = (weights[:, :, None] * values).sum(dim=1)
        x = x + context_x

        for block in self.blocks1:
            x = x + block(x)

        logits = self.head(x).squeeze(1)
        return logits


# In[33]:


# =============================================================================
# 12. TORCH TENSORS FOR TABR
# =============================================================================
def to_tensor_or_none(array, dtype, device):
    if array is None:
        return None
    return torch.as_tensor(array, dtype=dtype, device=device)

X_num_train_t = to_tensor_or_none(tabr_arrays["X_num_train"], torch.float32, DEVICE)
X_num_val_t   = to_tensor_or_none(tabr_arrays["X_num_val"], torch.float32, DEVICE)
X_num_test_t  = to_tensor_or_none(tabr_arrays["X_num_test"], torch.float32, DEVICE)

X_cat_train_t = to_tensor_or_none(tabr_arrays["X_cat_train"], torch.long, DEVICE)
X_cat_val_t   = to_tensor_or_none(tabr_arrays["X_cat_val"], torch.long, DEVICE)
X_cat_test_t  = to_tensor_or_none(tabr_arrays["X_cat_test"], torch.long, DEVICE)

y_train_long_t = torch.as_tensor(tabr_arrays["y_train"], dtype=torch.long, device=DEVICE)
y_train_float_t = y_train_long_t.float()
y_val_np = tabr_arrays["y_val"].astype(np.int64)

n_num_features_tabr = 0 if X_num_train_t is None else X_num_train_t.shape[1]
candidate_train_ids_t = torch.arange(len(y_train_long_t), device=DEVICE)

print("n_num_features_tabr:", n_num_features_tabr)
print("cat_cardinalities  :", tabr_arrays["cat_cardinalities"])
print("n_train:", len(y_train_long_t))
print("n_val  :", len(y_val_np))


# In[34]:


# =============================================================================
# 13. TABR TRAINING + VALIDATION FUNCTIONS
# =============================================================================
@dataclass
class TabRHistoryItem:
    epoch: int
    train_loss: float
    val_auc: float
    val_ap: float
    val_logloss: float

def tabr_predict_proba(
    model,
    X_num,
    X_cat,
    candidate_num,
    candidate_cat,
    candidate_y,
    context_size,
    batch_size=512,
):
    model.eval()
    preds = []
    n = candidate_y.shape[0] if X_num is None and X_cat is None else (X_num.shape[0] if X_num is not None else X_cat.shape[0])

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb_num = None if X_num is None else X_num[start:end]
            xb_cat = None if X_cat is None else X_cat[start:end]

            logits = model(
                x_num=xb_num,
                x_cat=xb_cat,
                candidate_num=candidate_num,
                candidate_cat=candidate_cat,
                candidate_y=candidate_y,
                context_size=context_size,
                is_train=False,
                batch_train_idx=None,
                candidate_train_idx=None,
            )
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    return np.concatenate(preds)

def evaluate_tabr(model):
    val_pred = tabr_predict_proba(
        model=model,
        X_num=X_num_val_t,
        X_cat=X_cat_val_t,
        candidate_num=X_num_train_t,
        candidate_cat=X_cat_train_t,
        candidate_y=y_train_long_t,
        context_size=TABR_CONFIG["context_size"],
        batch_size=TABR_CONFIG["eval_batch_size"],
    )

    val_auc = roc_auc_score(y_val_np, val_pred)
    val_ap = average_precision_score(y_val_np, val_pred)
    val_ll = log_loss(y_val_np, np.clip(val_pred, 1e-7, 1 - 1e-7))
    return val_pred, val_auc, val_ap, val_ll

def train_tabr():
    seed_everything(TABR_CONFIG["seed"])

    model = TabRBinaryClassifier(
        n_num_features=n_num_features_tabr,
        cat_cardinalities=tabr_arrays["cat_cardinalities"],
        d_main=TABR_CONFIG["d_main"],
        d_multiplier=TABR_CONFIG["d_multiplier"],
        encoder_n_blocks=TABR_CONFIG["encoder_n_blocks"],
        predictor_n_blocks=TABR_CONFIG["predictor_n_blocks"],
        dropout0=TABR_CONFIG["dropout0"],
        dropout1=TABR_CONFIG["dropout1"],
        context_dropout=TABR_CONFIG["context_dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TABR_CONFIG["lr"],
        weight_decay=TABR_CONFIG["weight_decay"],
    )

    loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_score = -np.inf
    best_epoch = 0
    wait = 0
    history = []

    n_train = len(y_train_long_t)

    for epoch in range(1, TABR_CONFIG["n_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        batch_losses = []

        for batch_idx in perm.split(TABR_CONFIG["batch_size"]):
            xb_num = None if X_num_train_t is None else X_num_train_t[batch_idx]
            xb_cat = None if X_cat_train_t is None else X_cat_train_t[batch_idx]
            yb = y_train_float_t[batch_idx]

            optimizer.zero_grad()

            logits = model(
                x_num=xb_num,
                x_cat=xb_cat,
                candidate_num=X_num_train_t,
                candidate_cat=X_cat_train_t,
                candidate_y=y_train_long_t,
                context_size=TABR_CONFIG["context_size"],
                is_train=True,
                batch_train_idx=batch_idx,
                candidate_train_idx=candidate_train_ids_t,
            )

            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_pred, val_auc, val_ap, val_ll = evaluate_tabr(model)

        history.append(TabRHistoryItem(
            epoch=epoch,
            train_loss=train_loss,
            val_auc=val_auc,
            val_ap=val_ap,
            val_logloss=val_ll,
        ))

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_auc={val_auc:.5f} | "
            f"val_ap={val_ap:.5f} | "
            f"val_logloss={val_ll:.5f}"
        )

        if val_auc > best_score:
            best_score = val_auc
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= TABR_CONFIG["patience"]:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} | Best val_auc: {best_score:.5f}")

    history_df = pd.DataFrame([vars(x) for x in history])
    return model, history_df

tabr_model, tabr_history = train_tabr()
tabr_history.tail()


# In[35]:


# =============================================================================
# 14. TABR VALIDATION RESULT + TEST PREDICTION
# =============================================================================
tabr_val_pred = tabr_predict_proba(
    model=tabr_model,
    X_num=X_num_val_t,
    X_cat=X_cat_val_t,
    candidate_num=X_num_train_t,
    candidate_cat=X_cat_train_t,
    candidate_y=y_train_long_t,
    context_size=TABR_CONFIG["context_size"],
    batch_size=TABR_CONFIG["eval_batch_size"],
)

tabr_test_pred = tabr_predict_proba(
    model=tabr_model,
    X_num=X_num_test_t,
    X_cat=X_cat_test_t,
    candidate_num=X_num_train_t,
    candidate_cat=X_cat_train_t,
    candidate_y=y_train_long_t,
    context_size=TABR_CONFIG["context_size"],
    batch_size=TABR_CONFIG["eval_batch_size"],
)

tabr_val_auc = roc_auc_score(y_val_np, tabr_val_pred)
tabr_val_ap = average_precision_score(y_val_np, tabr_val_pred)
tabr_val_ll = log_loss(y_val_np, np.clip(tabr_val_pred, 1e-7, 1 - 1e-7))

print("Final Validation AUC     :", round(tabr_val_auc, 6))
print("Final Validation AP      :", round(tabr_val_ap, 6))
print("Final Validation LogLoss :", round(tabr_val_ll, 6))

tabr_val_out = tabr_val_df[[TABR_ID_COL, TABR_TARGET_COL]].copy()
tabr_val_out["tabr_prob"] = tabr_val_pred

tabr_test_out = tabr_test_df[[TABR_ID_COL]].copy()
tabr_test_out["prediction"] = tabr_test_pred

tabr_val_out.head(), tabr_test_out.head()


# In[36]:


# =============================================================================
# 15. TABR EVALUATION PLOTS
# =============================================================================
print("" + "="*60)
print("15. TABR EVALUATION PLOTS")
print("="*60)

# -------------------------------------------------------------------------
# Make sure PALETTE has the colors we need
# -------------------------------------------------------------------------
if "PALETTE" not in globals():
    PALETTE = {}

tabr_color = PALETTE.get("tabr", "#8E44AD")
human_color = PALETTE.get("human", "#2B9EB3")
bot_color = PALETTE.get("bot", "#E84545")
neutral_color = PALETTE.get("neutral", "#6C757D")

# -------------------------------------------------------------------------
# 15a. ROC + PR curves
# -------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_val_np, tabr_val_pred)
prec, rec, _ = precision_recall_curve(y_val_np, tabr_val_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"TabR — Hold-out Validation  AUC={tabr_val_auc:.4f}  AP={tabr_val_ap:.4f}", fontsize=14)

# ROC
ax = axes[0]
ax.plot(fpr, tpr, color=tabr_color, lw=2, label=f"ROC (AUC={tabr_val_auc:.4f})")
ax.plot([0, 1], [0, 1], "--", color=neutral_color, lw=1)
ax.fill_between(fpr, tpr, alpha=0.12, color=tabr_color)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

# PR
ax = axes[1]
ax.plot(rec, prec, color=bot_color, lw=2, label=f"PR (AP={tabr_val_ap:.4f})")
ax.fill_between(rec, prec, alpha=0.12, color=bot_color)
ax.axhline(y_val_np.mean(), color=neutral_color, lw=1, linestyle="--", label="Chance")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve")
ax.legend()

plt.tight_layout()
plt.show()
save(fig, "tabr_01_roc_pr")

# -------------------------------------------------------------------------
# 15b. Confusion matrix
# -------------------------------------------------------------------------
tabr_pred = (tabr_val_pred >= 0.5).astype(int)
cm = confusion_matrix(y_val_np, tabr_pred)

print("Classification Report (TabR, threshold=0.5):")
print(classification_report(y_val_np, tabr_pred, target_names=["Human", "Bot"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    ax=ax,
    xticklabels=["Human", "Bot"],
    yticklabels=["Human", "Bot"],
    linewidths=0.5,
    cbar=False,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("TabR Confusion Matrix (threshold=0.5)")
plt.tight_layout()
plt.show()
save(fig, "tabr_02_confusion_matrix")

# -------------------------------------------------------------------------
# 15c. Score distributions by class
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(tabr_val_pred[y_val_np == 0], bins=30, color=human_color, alpha=0.55, label="Human", ax=ax)
sns.histplot(tabr_val_pred[y_val_np == 1], bins=30, color=bot_color, alpha=0.55, label="Bot", ax=ax)
ax.set_xlabel("Predicted bot probability")
ax.set_ylabel("Count")
ax.set_title("TabR Validation Score Distribution")
ax.legend()
plt.tight_layout()
plt.show()
save(fig, "tabr_03_score_distribution")

# -------------------------------------------------------------------------
# 15d. Training history
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("TabR Training History", fontsize=14)

axes[0].plot(tabr_history["epoch"], tabr_history["train_loss"], lw=2, color=neutral_color)
axes[0].set_title("Train Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[1].plot(tabr_history["epoch"], tabr_history["val_auc"], lw=2, color=tabr_color, label="Val AUC")
axes[1].plot(tabr_history["epoch"], tabr_history["val_ap"], lw=2, color=bot_color, label="Val AP")
axes[1].set_title("Validation AUC / AP")
axes[1].set_xlabel("Epoch")
axes[1].legend()

axes[2].plot(tabr_history["epoch"], tabr_history["val_logloss"], lw=2, color=human_color)
axes[2].set_title("Validation LogLoss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("LogLoss")

plt.tight_layout()
plt.show()
save(fig, "tabr_04_training_history")


# In[37]:


# =============================================================================
# 16. TABR XAI — GLOBAL + LOCAL + RETRIEVAL EXPLANATION
# =============================================================================
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("\n" + "="*60)
print("16. TABR XAI — GLOBAL + LOCAL + RETRIEVAL EXPLANATION")
print("="*60)

if "PALETTE" not in globals():
    PALETTE = {}

tabr_color = PALETTE.get("tabr", "#8E44AD")
human_color = PALETTE.get("human", "#2B9EB3")
bot_color = PALETTE.get("bot", "#E84545")
neutral_color = PALETTE.get("neutral", "#6C757D")


# In[38]:


# =============================================================================
# 16. TABR XAI — GLOBAL + LOCAL + RETRIEVAL EXPLANATION
# =============================================================================
print("\n" + "="*60)
print("16. TABR XAI — GLOBAL + LOCAL + RETRIEVAL EXPLANATION")
print("="*60)

# -------------------------------------------------------------------------
# Make sure PALETTE has the colors we need
# -------------------------------------------------------------------------
if "PALETTE" not in globals():
    PALETTE = {}

tabr_color = PALETTE.get("tabr", "#8E44AD")
human_color = PALETTE.get("human", "#2B9EB3")
bot_color = PALETTE.get("bot", "#E84545")
neutral_color = PALETTE.get("neutral", "#6C757D")

TABR_ALL_FEATURES = list(tabr_num_cols) + list(tabr_cat_cols)

# -------------------------------------------------------------------------
# Helper to clone transformed arrays
# -------------------------------------------------------------------------
def clone_tabr_arrays(X_num, X_cat):
    num = None if X_num is None else X_num.detach().cpu().numpy().copy()
    cat = None if X_cat is None else X_cat.detach().cpu().numpy().copy()
    return num, cat

# -------------------------------------------------------------------------
# 16a. GLOBAL XAI — permutation importance on validation AUC
# -------------------------------------------------------------------------
print("\nComputing permutation importance on validation AUC...")

base_auc = roc_auc_score(y_val_np, tabr_val_pred)
perm_rows = []
rng = np.random.default_rng(TABR_CONFIG["seed"])

X_num_val_np = None if X_num_val_t is None else X_num_val_t.detach().cpu().numpy()
X_cat_val_np = None if X_cat_val_t is None else X_cat_val_t.detach().cpu().numpy()
X_num_train_np = None if X_num_train_t is None else X_num_train_t.detach().cpu().numpy()
X_cat_train_np = None if X_cat_train_t is None else X_cat_train_t.detach().cpu().numpy()

def proba_from_numpy(X_num_np, X_cat_np):
    X_num_t = None if X_num_np is None else torch.as_tensor(X_num_np, dtype=torch.float32, device=DEVICE)
    X_cat_t = None if X_cat_np is None else torch.as_tensor(X_cat_np, dtype=torch.long, device=DEVICE)
    return tabr_predict_proba(
        model=tabr_model,
        X_num=X_num_t,
        X_cat=X_cat_t,
        candidate_num=X_num_train_t,
        candidate_cat=X_cat_train_t,
        candidate_y=y_train_long_t,
        context_size=TABR_CONFIG["context_size"],
        batch_size=TABR_CONFIG["eval_batch_size"],
    )

if X_num_val_np is not None:
    for j, col in enumerate(tabr_num_cols):
        X_num_perm, X_cat_perm = clone_tabr_arrays(X_num_val_t, X_cat_val_t)
        X_num_perm[:, j] = rng.permutation(X_num_perm[:, j])
        perm_pred = proba_from_numpy(X_num_perm, X_cat_perm)
        perm_auc = roc_auc_score(y_val_np, perm_pred)
        perm_rows.append({
            "feature": col,
            "feature_type": "numerical",
            "baseline_auc": base_auc,
            "permuted_auc": perm_auc,
            "auc_drop": base_auc - perm_auc,
        })

if X_cat_val_np is not None:
    for j, col in enumerate(tabr_cat_cols):
        X_num_perm, X_cat_perm = clone_tabr_arrays(X_num_val_t, X_cat_val_t)
        X_cat_perm[:, j] = rng.permutation(X_cat_perm[:, j])
        perm_pred = proba_from_numpy(X_num_perm, X_cat_perm)
        perm_auc = roc_auc_score(y_val_np, perm_pred)
        perm_rows.append({
            "feature": col,
            "feature_type": "categorical",
            "baseline_auc": base_auc,
            "permuted_auc": perm_auc,
            "auc_drop": base_auc - perm_auc,
        })

perm_df = pd.DataFrame(perm_rows).sort_values("auc_drop", ascending=False)
perm_df.to_csv("logs/tabr_xai_global_importance.csv", index=False)
print("Saved logs/tabr_xai_global_importance.csv")
print("\nTop global features:")
print(perm_df.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 7))
top_perm = perm_df.head(20).iloc[::-1]
colors = [tabr_color if t == "numerical" else bot_color for t in top_perm["feature_type"]]
ax.barh(top_perm["feature"], top_perm["auc_drop"], color=colors)
ax.set_title("TabR Global Feature Importance (Permutation AUC drop)")
ax.set_xlabel("AUC drop after shuffling feature")
plt.tight_layout()
plt.show()
save(fig, "tabr_xai_global_permutation")

# -------------------------------------------------------------------------
# 16b. LOCAL XAI — single-feature ablation for one high-risk validation row
# -------------------------------------------------------------------------
print("\nComputing local explanation for one high-risk validation example...")

local_idx = int(np.argmax(tabr_val_pred))
global_row_id = int(tabr_val_idx[local_idx])
base_prob = float(tabr_val_pred[local_idx])
true_label = int(y_val_np[local_idx])

num_baseline = None
cat_baseline = None
if X_num_train_np is not None:
    num_baseline = np.median(X_num_train_np, axis=0)
if X_cat_train_np is not None:
    cat_baseline = np.asarray(
        [np.bincount(X_cat_train_np[:, j]).argmax() for j in range(X_cat_train_np.shape[1])],
        dtype=np.int64,
    )

local_rows = []

if X_num_val_np is not None:
    for j, col in enumerate(tabr_num_cols):
        X_num_local = X_num_val_np[local_idx:local_idx+1].copy()
        X_cat_local = None if X_cat_val_np is None else X_cat_val_np[local_idx:local_idx+1].copy()
        X_num_local[0, j] = num_baseline[j]
        new_prob = float(proba_from_numpy(X_num_local, X_cat_local)[0])
        contrib = base_prob - new_prob
        local_rows.append({
            "feature": col,
            "feature_type": "numerical",
            "original_value": X_num_val_np[local_idx, j],
            "baseline_value": num_baseline[j],
            "prediction_before": base_prob,
            "prediction_after_ablation": new_prob,
            "contribution": contrib,
            "abs_contribution": abs(contrib),
        })

if X_cat_val_np is not None:
    for j, col in enumerate(tabr_cat_cols):
        X_num_local = None if X_num_val_np is None else X_num_val_np[local_idx:local_idx+1].copy()
        X_cat_local = X_cat_val_np[local_idx:local_idx+1].copy()
        X_cat_local[0, j] = cat_baseline[j]
        new_prob = float(proba_from_numpy(X_num_local, X_cat_local)[0])
        contrib = base_prob - new_prob
        local_rows.append({
            "feature": col,
            "feature_type": "categorical",
            "original_value": X_cat_val_np[local_idx, j],
            "baseline_value": cat_baseline[j],
            "prediction_before": base_prob,
            "prediction_after_ablation": new_prob,
            "contribution": contrib,
            "abs_contribution": abs(contrib),
        })

local_df = pd.DataFrame(local_rows).sort_values("abs_contribution", ascending=False)
local_df.to_csv("logs/tabr_xai_local_explanation.csv", index=False)

print(f"Explained validation row: local_idx={local_idx}, global_row_id={global_row_id}")
print(f"True label: {true_label} | Predicted bot probability: {base_prob:.4f}")
print("Saved logs/tabr_xai_local_explanation.csv")
print("\nTop local features:")
print(local_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
top_local = local_df.head(15).iloc[::-1]
colors = [tabr_color if x > 0 else human_color for x in top_local["contribution"]]
ax.barh(top_local["feature"], top_local["contribution"], color=colors)
ax.set_title("TabR Local Explanation (single-feature ablation)")
ax.set_xlabel("Change in predicted bot probability")
plt.tight_layout()
plt.show()
save(fig, "tabr_xai_local_ablation")

# -------------------------------------------------------------------------
# 16c. Retrieval explanation — inspect top retrieved neighbors for the local row
# -------------------------------------------------------------------------
# def tabr_top_neighbors(model, x_num_row, x_cat_row, candidate_num, candidate_cat, candidate_y, context_size=10):
#     model.eval()
#     with torch.no_grad():
#         _, candidate_k = model._encode(candidate_num, candidate_cat)
#         _, query_k = model._encode(x_num_row, x_cat_row)
#         search_k = min(max(context_size, 1), candidate_k.shape[0])
#         neighbor_idx = model._search_neighbors(query_k, candidate_k, search_k)
#         context_k = candidate_k[neighbor_idx]
#         scores = -((query_k[:, None, :] - context_k) ** 2).sum(dim=-1)
#         weights = torch.softmax(scores, dim=1)
#     return neighbor_idx[0].detach().cpu().numpy(), weights[0].detach().cpu().numpy()

def tabr_top_neighbors(model, x_num_row, x_cat_row, candidate_num, candidate_cat, candidate_y, context_size=10):
    model.eval()
    with torch.no_grad():
        _, candidate_k = model._encode(candidate_num, candidate_cat)
        _, query_k = model._encode(x_num_row, x_cat_row)

        search_k = min(max(context_size, 1), candidate_k.shape[0])

        # _search_neighbors returns: distances, indices
        distances, neighbor_idx = model._search_neighbors(query_k, candidate_k, search_k)

        neighbor_idx = neighbor_idx.long()
        distances = distances.float()

        # one query row only
        idx0 = neighbor_idx[0]
        dist0 = distances[0]

        # use negative distance as similarity score
        scores = -dist0
        weights = torch.softmax(scores, dim=0)

    return idx0.detach().cpu().numpy(), weights.detach().cpu().numpy()

    return neighbor_idx[0].detach().cpu().numpy(), weights[0].detach().cpu().numpy()

x_num_local_t = None if X_num_val_t is None else X_num_val_t[local_idx:local_idx+1]
x_cat_local_t = None if X_cat_val_t is None else X_cat_val_t[local_idx:local_idx+1]

nbr_idx, nbr_w = tabr_top_neighbors(
    tabr_model,
    x_num_local_t,
    x_cat_local_t,
    X_num_train_t,
    X_cat_train_t,
    y_train_long_t,
    context_size=min(10, TABR_CONFIG["context_size"]),
)

neighbor_df = tabr_train_df.iloc[nbr_idx][[TABR_ID_COL, TABR_TARGET_COL]].copy().reset_index(drop=True)
neighbor_df["neighbor_weight"] = nbr_w
neighbor_df["rank"] = np.arange(1, len(neighbor_df) + 1)
neighbor_df = neighbor_df[["rank", TABR_ID_COL, TABR_TARGET_COL, "neighbor_weight"]]
neighbor_df.to_csv("logs/tabr_xai_retrieved_neighbors.csv", index=False)

print("\nTop retrieved neighbors for the explained row:")
print(neighbor_df.to_string(index=False))
print("Saved logs/tabr_xai_retrieved_neighbors.csv")

fig, ax = plt.subplots(figsize=(8, 5))
plot_df = neighbor_df.iloc[::-1]
bar_colors = [bot_color if yy == 1 else human_color for yy in plot_df[TABR_TARGET_COL]]
ax.barh(plot_df["rank"].astype(str), plot_df["neighbor_weight"], color=bar_colors)
ax.set_title("Top Retrieved Neighbors for Local Explanation")
ax.set_xlabel("Neighbor weight")
ax.set_ylabel("Neighbor rank (1 = closest)")
plt.tight_layout()
plt.show()
save(fig, "tabr_xai_retrieved_neighbors")

print("\n✓ TabR XAI complete.")
print("Artifacts:")
print("  - logs/tabr_xai_global_importance.csv")
print("  - logs/tabr_xai_local_explanation.csv")
print("  - logs/tabr_xai_retrieved_neighbors.csv")
print("  - plots/tabr_xai_global_permutation.png")
print("  - plots/tabr_xai_local_ablation.png")
print("  - plots/tabr_xai_retrieved_neighbors.png")


# In[39]:


# =============================================================================
# 15. SAVE TABR OUTPUTS
# =============================================================================
TABR_HISTORY_PATH = "tabr_history.csv"
TABR_VAL_PATH = "tabr_val_predictions.csv"
TABR_TEST_PATH = "tabr_test_predictions.csv"
TABR_SUBMISSION_PATH = "tabr_submission.csv"
TABR_MODEL_PATH = "tabr_model.pt"

tabr_history.to_csv(TABR_HISTORY_PATH, index=False)
tabr_val_out.to_csv(TABR_VAL_PATH, index=False)
tabr_test_out.to_csv(TABR_TEST_PATH, index=False)
tabr_test_out.to_csv(TABR_SUBMISSION_PATH, index=False)
torch.save(
    {
        "model_state_dict": tabr_model.state_dict(),
        "config": TABR_CONFIG,
        "num_cols": tabr_num_cols,
        "cat_cols": tabr_cat_cols,
        "cat_cardinalities": tabr_arrays["cat_cardinalities"],
    },
    TABR_MODEL_PATH,
)

print("Saved:", TABR_HISTORY_PATH)
print("Saved:", TABR_VAL_PATH)
print("Saved:", TABR_TEST_PATH)
print("Saved:", TABR_SUBMISSION_PATH)
print("Saved:", TABR_MODEL_PATH)


# In[ ]:




