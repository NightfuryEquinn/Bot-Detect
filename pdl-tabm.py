#!/usr/bin/env python
# coding: utf-8

# # Facebook Recruiting IV: Human or Robot — Data Preprocessing Pipeline
# =====================================================================
# ## Covers: Data understanding · Cleaning· Visualization · Feature Engineering

# In[1]:


# Install once if needed
get_ipython().run_line_magic('pip', 'install -q lightgbm tabm rtdl_num_embeddings')


# In[2]:


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

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Helper ───────────────────────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ↳ Saved  {SAVE_DIR}/{name}.png")


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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
bids_clean["time_diff"] = bids_clean.groupby("bidder_id")["time"].diff()

zero_interval = bids_clean["time_diff"].eq(0).sum()
print(f"\nBids with zero time-diff to prev bid (same bidder): {zero_interval} "
      f"({zero_interval/len(bids_clean)*100:.1f}%)")

print(f"\nCleaned bids shape: {bids_clean.shape}")


# In[7]:


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
    "zero_interval_count"   : g["time_diff"].apply(lambda x: x.dropna().eq(0).sum()),
    "zero_interval_ratio"   : g["time_diff"].apply(
        lambda x: x.dropna().eq(0).mean() if x.dropna().shape[0] > 0 else 0
    ),

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
train_bidder_ids = set(train_clean["bidder_id"])
bids_train_only = bids_clean[bids_clean["bidder_id"].isin(train_bidder_ids)]

top_countries = bids_train_only["country"].value_counts().head(5).index.tolist()
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


# In[8]:


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
num_cols = [c for c in train_feat.columns if pd.api.types.is_numeric_dtype(train_feat[c])]
corr = train_feat[num_cols].corr()
# keep only top 20 most correlated with outcome
outcome_corr = corr["outcome"].abs().sort_values(ascending=False).head(21).index
sub_corr = train_feat[list(outcome_corr)].corr()

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


# In[9]:


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


# In[10]:


cleaned_bids = pd.read_csv('/kaggle/working/bidder_features.csv')
cleaned_train = pd.read_csv('/kaggle/working/train_features.csv')
cleaned_test = pd.read_csv('/kaggle/working/test_features.csv')

cleaned_bids.head()


# In[11]:


cleaned_bids.tail()


# In[12]:


cleaned_train.tail()


# In[13]:


cleaned_train.head()


# In[14]:


cleaned_test.head()


# In[15]:


cleaned_test.tail()


# # Random Forest Classifier
# =================================================================
# ## Requires: train_features.csv, test_features.csv  (output of preprocess.py)
# ## Outputs : submission.csv, plots/rf_*.png

# In[16]:


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


# In[17]:


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

N_SPLITS = 5
folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))

print(f"Train  : X={X.shape}  bot_rate={train_prior:.4f}")
print(f"Test   : X_test={X_test.shape}")
print(f"scale_pos_weight = {scale_pos:.1f}  (class imbalance correction)")
print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS[:6]} …")
print(f"Shared folds prepared: {len(folds)}")
for i, (tr_idx, val_idx) in enumerate(folds, 1):
    print(f"  Fold {i}: train={len(tr_idx)}, val={len(val_idx)}")


# In[18]:


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

base_scores = cross_val_score(rf_base, X, y, cv=folds,
                               scoring="roc_auc", n_jobs=-1)
print(f"Mean AUC: {base_scores.mean():.4f} ± {base_scores.std():.4f}")


# In[19]:


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
    cv=folds,
    refit=True,
    verbose=1,
    random_state=SEED,
    n_jobs=-1,
)
search.fit(X, y)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV AUC : {search.best_score_:.4f}")

best_rf = search.best_estimator_


# In[20]:


# =============================================================================
# 4. FULL TRAIN + OOF PREDICTIONS (for metrics)
# =============================================================================
print("\n" + "="*60)
print("4. OUT-OF-FOLD PREDICTIONS")
print("="*60)

oof_proba = np.zeros(len(y), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    fold_rf = RandomForestClassifier(**search.best_params_,
                                     n_jobs=-1, random_state=SEED)
    fold_rf.fit(X_tr, y_tr)
    oof_proba[val_idx] = fold_rf.predict_proba(X_val)[:, 1]

    fold_auc = roc_auc_score(y_val, oof_proba[val_idx])
    print(f"  Fold {fold}  AUC={fold_auc:.4f}")

oof_auc = roc_auc_score(y, oof_proba)
print(f"\nOOF AUC (all folds) : {oof_auc:.4f}")


# In[21]:


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
tr0, va0 = folds[0]
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


# In[22]:


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


# In[23]:


# =============================================================================
# RF SUMMARY LOG
# =============================================================================
rf_summary_df = pd.DataFrame([{
    "model": "RandomForest",
    "oof_auc": float(oof_auc),
    "oof_ap": float(average_precision_score(y, oof_proba)),
    "oof_pred_bot_rate": float(oof_proba.mean()),
    "test_pred_bot_rate": float(test_proba.mean()),
}])

rf_summary_df.to_csv("logs/rf_summary.csv", index=False)
print("Saved logs/rf_summary.csv")
print(rf_summary_df)


# # LightGBM + Calibration
# ================================================================
# ## Requires : train_features.csv, test_features.csv  (output of preprocess.py)
# ## Outputs  : submission.csv, plots/lgbm_*.png
# ## Install  : pip install lightgbm scikit-learn matplotlib seaborn

# In[24]:


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


# In[25]:


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


# In[26]:


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


# In[27]:


# =============================================================================
# 3. OUT-OF-FOLD PREDICTIONS WITH EARLY STOPPING
# =============================================================================
print("\n" + "="*60)
print("3. OOF TRAINING — 5-FOLD STRATIFIED CV")
print("="*60)

oof_proba    = np.zeros(len(y), dtype=np.float64)
test_probas  = np.zeros((len(X_test), len(folds)), dtype=np.float64)  # per-fold test preds
best_iters   = []

for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set        = [(X_val, y_val)],
        eval_metric     = "auc",
        callbacks       = [
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
    test_probas[:, fold - 1] = model.predict_proba(X_test)[:, 1]
    best_iters.append(model.best_iteration_)

    fold_auc = roc_auc_score(y_val, oof_proba[val_idx])
    print(f"  Fold {fold}  best_iter={model.best_iteration_:4d}  AUC={fold_auc:.4f}")

oof_auc = roc_auc_score(y, oof_proba)
avg_iter = int(np.mean(best_iters))
test_proba = test_probas.mean(axis=1)

print(f"\nOOF AUC (all folds) : {oof_auc:.4f}")
print(f"Mean best_iteration : {avg_iter}")
print(f"Predicted bot rate  : {test_proba.mean():.4f}  (train prior={train_prior:.4f})")


# In[28]:


# =============================================================================
# 4. PROBABILITY CALIBRATION (proper OOF)
# =============================================================================
print("\n" + "="*60)
print("4. ISOTONIC CALIBRATION (proper OOF)")
print("="*60)

oof_calib_proba = np.zeros(len(y), dtype=np.float64)
test_calib_probas = np.zeros((len(X_test), len(folds)), dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    base_model = lgb.LGBMClassifier(
        **{**LGBM_PARAMS, "n_estimators": avg_iter}
    )

    calib_model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=3
    )

    calib_model.fit(X_tr, y_tr)
    oof_calib_proba[val_idx] = calib_model.predict_proba(X_val)[:, 1]
    test_calib_probas[:, fold - 1] = calib_model.predict_proba(X_test)[:, 1]

test_proba_cal = test_calib_probas.mean(axis=1)

print(f"Bot rate raw OOF   : {oof_proba.mean():.4f}")
print(f"Bot rate calib OOF : {oof_calib_proba.mean():.4f}")
print(f"Bot rate test cal  : {test_proba_cal.mean():.4f}")


# In[29]:


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
oof_pred = (oof_calib_proba >= 0.5).astype(int)
cm = confusion_matrix(y, oof_pred)
print("\nClassification Report (proper OOF calibrated, threshold=0.5):")
print(classification_report(y, oof_pred, target_names=["Human", "Bot"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"],
            linewidths=0.5, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (proper OOF calibrated, threshold=0.5)")
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
fold0_tr, fold0_val = folds[0]
probe = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": 1000})

probe.fit(
    X[fold0_tr], y[fold0_tr],
    eval_set=[(X[fold0_tr], y[fold0_tr]), (X[fold0_val], y[fold0_val])],
    eval_names=["train", "valid"],
    eval_metric="auc",
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
plt.show()
save(fig, "lgbm_07_learning_curve")


# In[30]:


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


# In[31]:


# =============================================================================
# LIGHTGBM SUMMARY LOG
# =============================================================================
lgbm_summary_df = pd.DataFrame([{
    "model": "LightGBM",
    "oof_auc": float(oof_auc),
    "oof_ap": float(average_precision_score(y, oof_proba)),
    "oof_pred_bot_rate": float(oof_proba.mean()),
    "test_pred_bot_rate": float(test_proba_cal.mean()),
}])

lgbm_summary_df.to_csv("logs/lgbm_summary.csv", index=False)
print("Saved logs/lgbm_summary.csv")
print(lgbm_summary_df)


# # TabM

# In[32]:


# =============================================================================
# TABM - IMPORTS + SETUP
# =============================================================================
import os, time, copy, random, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tabm
except ImportError:
    get_ipython().system('pip install -q tabm rtdl_num_embeddings')
    import tabm

try:
    import rtdl_num_embeddings
except ImportError:
    get_ipython().system('pip install -q rtdl_num_embeddings')
    import rtdl_num_embeddings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

if "SAVE_DIR" not in globals():
    SAVE_DIR = "plots"
if "LOG_DIR" not in globals():
    LOG_DIR = "logs"

if "SEED" not in globals():
    SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# use existing palette if already defined
if "PALETTE" not in globals():
    PALETTE = {
        "bot": "#E84545",
        "human": "#2B9EB3",
        "neutral": "#6C757D",
        "tabm": "#8E44AD",
        "tune": "#F39C12",
    }

# use existing save() if already defined
if "save" not in globals():
    def save(fig, name):
        fig.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ↳ Saved {SAVE_DIR}/{name}.png")


# ## load data + helper functions

# In[33]:


# =============================================================================
# TABM - IMPORTS + SETUP
# =============================================================================
import os, time, copy, random, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tabm
except ImportError:
    get_ipython().system('pip install -q tabm rtdl_num_embeddings')
    import tabm

try:
    import rtdl_num_embeddings
except ImportError:
    get_ipython().system('pip install -q rtdl_num_embeddings')
    import rtdl_num_embeddings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

if "SAVE_DIR" not in globals():
    SAVE_DIR = "plots"
if "LOG_DIR" not in globals():
    LOG_DIR = "logs"
if "SEED" not in globals():
    SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

if "PALETTE" not in globals():
    PALETTE = {
        "bot": "#E84545",
        "human": "#2B9EB3",
        "neutral": "#6C757D",
        "tabm": "#8E44AD",
        "tune": "#F39C12",
    }

if "save" not in globals():
    def save(fig, name):
        fig.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ↳ Saved {SAVE_DIR}/{name}.png")


# In[34]:


# =============================================================================
# 1. TABM SEARCH SPACE
# =============================================================================
print("\n" + "="*60)
print("1. TABM SEARCH SPACE")
print("="*60)

if "folds" not in globals():
    N_SPLITS = 5
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
    print("Shared folds were not found, so they were recreated.")
else:
    print(f"Reusing shared folds: {len(folds)} folds")

MAX_EPOCHS = 80
PATIENCE = 8
MIN_DELTA = 5e-4
GRAD_CLIP = 0.5
BATCH_SIZE = 256 if DEVICE.type == "cuda" else 128

# Tuning uses one seed for speed; final ensemble uses multiple seeds
TUNING_SEED = 42
FINAL_SEEDS = [42, 52, 62] if DEVICE.type == "cuda" else [42, 52]
TOP_K_CONFIGS = 2
AUC_TIE_GAP = 0.002

# Balanced search space: not too aggressive, not too tiny
if DEVICE.type == "cuda":
    TABM_CONFIGS = [
        {
            "arch_type": "tabm",
            "k": 12,
            "n_blocks": 3,
            "d_block": 192,
            "dropout": 0.15,
            "lr": 0.001,
            "weight_decay": 0.001,
            "use_num_embeddings": True,
            "pos_weight_scale": 0.75,
            "preprocessor": "quantile",
        },
        {
            "arch_type": "tabm",
            "k": 16,
            "n_blocks": 3,
            "d_block": 256,
            "dropout": 0.20,
            "lr": 0.001,
            "weight_decay": 0.001,
            "use_num_embeddings": True,
            "pos_weight_scale": 0.50,
            "preprocessor": "quantile",
        },
        {
            "arch_type": "tabm",
            "k": 8,
            "n_blocks": 2,
            "d_block": 128,
            "dropout": 0.20,
            "lr": 0.0007,
            "weight_decay": 0.001,
            "use_num_embeddings": False,
            "pos_weight_scale": 0.75,
            "preprocessor": "robust",
        },
        {
            "arch_type": "tabm",
            "k": 8,
            "n_blocks": 2,
            "d_block": 128,
            "dropout": 0.25,
            "lr": 0.0007,
            "weight_decay": 0.003,
            "use_num_embeddings": False,
            "pos_weight_scale": 0.50,
            "preprocessor": "quantile",
        },
        {
            "arch_type": "tabm",
            "k": 12,
            "n_blocks": 2,
            "d_block": 192,
            "dropout": 0.15,
            "lr": 0.001,
            "weight_decay": 0.001,
            "use_num_embeddings": True,
            "pos_weight_scale": 0.75,
            "preprocessor": "robust",
        },
    ]
else:
    TABM_CONFIGS = [
        {
            "arch_type": "tabm-mini",
            "k": 8,
            "n_blocks": 2,
            "d_block": 128,
            "dropout": 0.20,
            "lr": 0.001,
            "weight_decay": 0.001,
            "use_num_embeddings": True,
            "pos_weight_scale": 0.75,
            "preprocessor": "quantile",
        },
        {
            "arch_type": "tabm-mini",
            "k": 8,
            "n_blocks": 2,
            "d_block": 128,
            "dropout": 0.25,
            "lr": 0.0007,
            "weight_decay": 0.003,
            "use_num_embeddings": False,
            "pos_weight_scale": 0.50,
            "preprocessor": "robust",
        },
        {
            "arch_type": "tabm-mini",
            "k": 12,
            "n_blocks": 2,
            "d_block": 128,
            "dropout": 0.20,
            "lr": 0.001,
            "weight_decay": 0.001,
            "use_num_embeddings": True,
            "pos_weight_scale": 0.75,
            "preprocessor": "robust",
        },
    ]

print("TabM configs:")
for i, cfg in enumerate(TABM_CONFIGS, 1):
    print(f"  Config {i}: {cfg}")


# In[35]:


# =============================================================================
# 2. TABM HELPERS
# =============================================================================
def make_preprocessor(X_train, seed, method="quantile"):
    if method == "quantile":
        noise = np.random.default_rng(seed).normal(0.0, 1e-5, X_train.shape).astype(np.float32)
        qt = QuantileTransformer(
            n_quantiles=max(min(len(X_train) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
            random_state=seed,
        )
        qt.fit(X_train + noise)
        return qt

    elif method == "robust":
        rs = RobustScaler()
        rs.fit(X_train)
        return rs

    else:
        raise ValueError(f"Unknown preprocessor: {method}")


def build_tabm_model(n_num_features, cfg):
    num_embeddings = None
    if cfg.get("use_num_embeddings", False):
        num_embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_num_features)

    model = tabm.TabM.make(
        n_num_features=n_num_features,
        d_out=1,
        num_embeddings=num_embeddings,
        arch_type=cfg["arch_type"],
        k=int(cfg["k"]),
        n_blocks=int(cfg["n_blocks"]),
        d_block=int(cfg["d_block"]),
        dropout=float(cfg["dropout"]),
    ).to(DEVICE)
    return model


def predict_tabm_proba(model, x_num_tensor, batch_size=1024):
    model.eval()
    probs = []

    with torch.inference_mode():
        for start in range(0, len(x_num_tensor), batch_size):
            xb = x_num_tensor[start:start + batch_size]
            logits = model(xb).squeeze(-1)          # (B, k)
            proba = torch.sigmoid(logits).mean(dim=1)
            probs.append(proba.detach().cpu().numpy())

    return np.concatenate(probs)


def eval_tabm_loss(model, x_num_tensor, y_tensor, pos_weight, batch_size=1024):
    model.eval()
    losses = []

    with torch.inference_mode():
        for start in range(0, len(x_num_tensor), batch_size):
            xb = x_num_tensor[start:start + batch_size]
            yb = y_tensor[start:start + batch_size]

            logits = model(xb).squeeze(-1)
            target = yb[:, None].expand(-1, logits.shape[1]).float()

            loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight
            )
            losses.append(loss.item())

    return float(np.mean(losses))


def train_tabm_one_fold(X_tr_raw, y_tr, X_val_raw, y_val, X_te_raw, cfg, seed,
                        collect_history=False, member_tag=None):
    set_seed(seed)

    pre = make_preprocessor(X_tr_raw, seed=seed, method=cfg["preprocessor"])
    X_tr = pre.transform(X_tr_raw).astype(np.float32)
    X_val = pre.transform(X_val_raw).astype(np.float32)

    X_te = None
    if X_te_raw is not None:
        X_te = pre.transform(X_te_raw).astype(np.float32)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr.astype(np.float32), device=DEVICE)
    X_val_t = torch.tensor(X_val, device=DEVICE)
    y_val_t = torch.tensor(y_val.astype(np.float32), device=DEVICE)
    X_te_t = torch.tensor(X_te, device=DEVICE) if X_te is not None else None

    model = build_tabm_model(X_tr.shape[1], cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    base_pos_weight = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor(
        base_pos_weight * cfg["pos_weight_scale"],
        dtype=torch.float32,
        device=DEVICE
    )

    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    wait = 0
    history_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = np.random.permutation(len(X_tr))
        batch_losses = []

        for start in range(0, len(order), BATCH_SIZE):
            batch_idx = order[start:start + BATCH_SIZE]
            xb = X_tr_t[batch_idx]
            yb = y_tr_t[batch_idx]

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb).squeeze(-1)
            target = yb[:, None].expand(-1, logits.shape[1]).float()

            loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss = eval_tabm_loss(model, X_val_t, y_val_t, pos_weight, batch_size=1024)
        val_proba = predict_tabm_proba(model, X_val_t, batch_size=1024)
        val_auc = roc_auc_score(y_val, val_proba)

        if collect_history:
            history_rows.append({
                "member_tag": member_tag,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
            })

        scheduler.step(val_auc)

        if val_auc > best_auc + MIN_DELTA:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)

    val_proba = predict_tabm_proba(model, X_val_t, batch_size=1024)
    test_proba = predict_tabm_proba(model, X_te_t, batch_size=1024) if X_te_t is not None else None

    return {
        "model": model,
        "val_proba": val_proba,
        "test_proba": test_proba,
        "best_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "history_rows": history_rows,
    }

print("TabM helper functions loaded.")


# In[36]:


# =============================================================================
# 3. TABM HYPERPARAMETER SEARCH
# =============================================================================
print("\n" + "="*60)
print("3. TABM HYPERPARAMETER SEARCH")
print("="*60)

tuning_rows = []

for config_id, cfg in enumerate(TABM_CONFIGS, start=1):
    print(f"\n--- Config {config_id}/{len(TABM_CONFIGS)} ---")
    print(cfg)

    fold_aucs = []
    fold_aps = []
    fold_epochs = []

    for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
        res = train_tabm_one_fold(
            X_tr_raw=X[tr_idx],
            y_tr=y[tr_idx],
            X_val_raw=X[val_idx],
            y_val=y[val_idx],
            X_te_raw=None,
            cfg=cfg,
            seed=TUNING_SEED + config_id * 100 + fold,
            collect_history=False,
        )

        fold_auc = roc_auc_score(y[val_idx], res["val_proba"])
        fold_ap = average_precision_score(y[val_idx], res["val_proba"])

        fold_aucs.append(fold_auc)
        fold_aps.append(fold_ap)
        fold_epochs.append(res["best_epoch"])

        print(f"  Fold {fold}  AUC={fold_auc:.4f}  AP={fold_ap:.4f}  best_epoch={res['best_epoch']}")

    tuning_rows.append({
        "config_id": config_id,
        "arch_type": cfg["arch_type"],
        "k": cfg["k"],
        "n_blocks": cfg["n_blocks"],
        "d_block": cfg["d_block"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "use_num_embeddings": cfg["use_num_embeddings"],
        "pos_weight_scale": cfg["pos_weight_scale"],
        "preprocessor": cfg["preprocessor"],
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "mean_ap": float(np.mean(fold_aps)),
        "mean_best_epoch": float(np.mean(fold_epochs)),
    })

tabm_tuning_df = pd.DataFrame(tuning_rows).sort_values(
    ["mean_auc", "mean_ap"], ascending=False
).reset_index(drop=True)

print("\nTabM tuning leaderboard:")
print(tabm_tuning_df.to_string(index=False))

tabm_tuning_df.to_csv("logs/tabm_tuning_results.csv", index=False)
print("Saved logs/tabm_tuning_results.csv")

max_auc = tabm_tuning_df["mean_auc"].max()
candidates = tabm_tuning_df[tabm_tuning_df["mean_auc"] >= max_auc - AUC_TIE_GAP].copy()

selected_cfg_df = candidates.sort_values(
    ["std_auc", "mean_best_epoch", "mean_ap"],
    ascending=[True, True, False]
).head(TOP_K_CONFIGS).reset_index(drop=True)

print("\nSelected configs for final ensemble:")
print(selected_cfg_df.to_string(index=False))

selected_cfg_df.to_csv("logs/tabm_selected_configs.csv", index=False)
print("Saved logs/tabm_selected_configs.csv")


# In[37]:


# =============================================================================
# 4. FINAL TABM MODEL
# =============================================================================
print("\n" + "="*60)
print("4. FINAL TABM MODEL")
print("="*60)

best_row = selected_cfg_df.iloc[0]

cfg = {
    "config_id": int(best_row["config_id"]),
    "arch_type": best_row["arch_type"],
    "k": int(best_row["k"]),
    "n_blocks": int(best_row["n_blocks"]),
    "d_block": int(best_row["d_block"]),
    "dropout": float(best_row["dropout"]),
    "lr": float(best_row["lr"]),
    "weight_decay": float(best_row["weight_decay"]),
    "use_num_embeddings": bool(best_row["use_num_embeddings"]),
    "pos_weight_scale": float(best_row["pos_weight_scale"]),
    "preprocessor": best_row["preprocessor"],
}

FINAL_SEED = 42

tabm_oof_proba = np.zeros(len(y), dtype=np.float32)
tabm_test_probas = np.zeros((len(X_test), len(folds)), dtype=np.float32)
tabm_history_rows = []
tabm_fold_rows = []

for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
    res = train_tabm_one_fold(
        X_tr_raw=X[tr_idx],
        y_tr=y[tr_idx],
        X_val_raw=X[val_idx],
        y_val=y[val_idx],
        X_te_raw=X_test,
        cfg=cfg,
        seed=FINAL_SEED * 100 + fold,
        collect_history=True,
        member_tag="final_tabm",
    )

    tabm_oof_proba[val_idx] = res["val_proba"]
    tabm_test_probas[:, fold - 1] = res["test_proba"]

    fold_auc = roc_auc_score(y[val_idx], res["val_proba"])
    fold_ap = average_precision_score(y[val_idx], res["val_proba"])

    tabm_fold_rows.append({
        "fold": fold,
        "best_epoch": res["best_epoch"],
        "final_fold_auc": float(fold_auc),
        "final_fold_ap": float(fold_ap),
    })

    for row in res["history_rows"]:
        row["fold"] = fold
    tabm_history_rows.extend(res["history_rows"])

    torch.save(
        res["model"].state_dict(),
        f"logs/tabm_final_fold{fold}_best.pt"
    )

    print(f"Fold {fold}  AUC={fold_auc:.4f}  AP={fold_ap:.4f}  best_epoch={res['best_epoch']}")

tabm_test_proba = tabm_test_probas.mean(axis=1)
tabm_oof_auc = roc_auc_score(y, tabm_oof_proba)
tabm_oof_ap = average_precision_score(y, tabm_oof_proba)

print("\n" + "="*60)
print("TABM SUMMARY")
print("="*60)
print(f"OOF AUC : {tabm_oof_auc:.4f}")
print(f"OOF AP  : {tabm_oof_ap:.4f}")
print(f"OOF bot-rate mean  : {tabm_oof_proba.mean():.4f}")
print(f"Test bot-rate mean : {tabm_test_proba.mean():.4f}")

pd.DataFrame(tabm_history_rows).to_csv("logs/tabm_history_all_folds.csv", index=False)
pd.DataFrame(tabm_fold_rows).to_csv("logs/tabm_fold_metrics.csv", index=False)

pd.DataFrame({
    "row_id": np.arange(len(y)),
    "outcome": y,
    "tabm_oof_proba": tabm_oof_proba,
}).to_csv("logs/tabm_oof_predictions.csv", index=False)

tabm_summary_df = pd.DataFrame([{
    "model": "TabM",
    "oof_auc": float(tabm_oof_auc),
    "oof_ap": float(tabm_oof_ap),
    "oof_pred_bot_rate": float(tabm_oof_proba.mean()),
    "test_pred_bot_rate": float(tabm_test_proba.mean()),
}])
tabm_summary_df.to_csv("logs/tabm_summary.csv", index=False)

with open("logs/tabm_best_config.json", "w") as f:
    json.dump(cfg, f, indent=2)

print("Saved logs/tabm_history_all_folds.csv")
print("Saved logs/tabm_fold_metrics.csv")
print("Saved logs/tabm_oof_predictions.csv")
print("Saved logs/tabm_summary.csv")
print("Saved logs/tabm_best_config.json")


# In[38]:


# =============================================================================
# 5. TABM EVALUATION PLOTS
# =============================================================================
print("\n" + "="*60)
print("5. TABM EVALUATION PLOTS")
print("="*60)

# -------------------------------------------------------------------------
# Make sure PALETTE has the colors we need
# -------------------------------------------------------------------------
if "PALETTE" not in globals():
    PALETTE = {}

tabm_color = PALETTE.get("tabm", "#8E44AD")
human_color = PALETTE.get("human", "#2B9EB3")
neutral_color = PALETTE.get("neutral", "#6C757D")

# -------------------------------------------------------------------------
# 5a. ROC + PR curves
# -------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y, tabm_oof_proba)
prec, rec, _ = precision_recall_curve(y, tabm_oof_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"TabM — OOF AUC={tabm_oof_auc:.4f}", fontsize=14)

# ROC
ax = axes[0]
ax.plot(fpr, tpr, color=tabm_color, lw=2, label=f"ROC (AUC={tabm_oof_auc:.4f})")
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
ax.fill_between(fpr, tpr, alpha=0.12, color=tabm_color)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

# PR
ax = axes[1]
ax.plot(rec, prec, color=human_color, lw=2, label=f"PR (AP={tabm_oof_ap:.4f})")
ax.fill_between(rec, prec, alpha=0.12, color=human_color)
ax.axhline(y.mean(), color="#555", lw=1, linestyle="--", label="Chance")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve")
ax.legend()

plt.tight_layout()
plt.show()
save(fig, "tabm_01_roc_pr")

# -------------------------------------------------------------------------
# 5b. Confusion matrix
# -------------------------------------------------------------------------
tabm_pred = (tabm_oof_proba >= 0.5).astype(int)
cm = confusion_matrix(y, tabm_pred)

print("\nClassification Report (TabM, threshold=0.5):")
print(classification_report(y, tabm_pred, target_names=["Human", "Bot"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=["Human", "Bot"],
    yticklabels=["Human", "Bot"],
    linewidths=0.5,
    cbar=False
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("TabM Confusion Matrix (threshold=0.5)")
plt.tight_layout()
plt.show()
save(fig, "tabm_02_confusion_matrix")

# -------------------------------------------------------------------------
# 5c. Training history (mean over folds, common epochs only)
# -------------------------------------------------------------------------
history_path = "logs/tabm_history_all_folds.csv"

if os.path.exists(history_path):
    hist = pd.read_csv(history_path)

    if not hist.empty and {"fold", "epoch", "train_loss", "val_loss", "val_auc"}.issubset(hist.columns):
        # only keep epochs reached by all folds
        max_common_epoch = hist.groupby("fold")["epoch"].max().min()
        hist_common = hist[hist["epoch"] <= max_common_epoch].copy()

        mean_hist = (
            hist_common.groupby("epoch")[["train_loss", "val_loss", "val_auc"]]
            .mean()
            .reset_index()
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("TabM Training History (mean over folds, common epochs only)", fontsize=13)

        # Loss curve
        ax = axes[0]
        ax.plot(mean_hist["epoch"], mean_hist["train_loss"], lw=2, label="Train Loss")
        ax.plot(mean_hist["epoch"], mean_hist["val_loss"], lw=2, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()

        # Validation AUC
        ax = axes[1]
        ax.plot(mean_hist["epoch"], mean_hist["val_auc"], color=tabm_color, lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("Validation AUC Curve")

        plt.tight_layout()
        plt.show()
        save(fig, "tabm_03_training_history")
    else:
        print(f"\nNo valid training history found in {history_path}.")
else:
    print(f"\nFile not found: {history_path}")


# In[39]:


# =============================================================================
# 6. FINAL PREDICTION + SUBMISSION
# =============================================================================
print("\n" + "="*60)
print("6. FINAL PREDICTION")
print("="*60)

submission_tabm = pd.DataFrame({
    "bidder_id": test["bidder_id"],
    "prediction": tabm_test_proba,
})
submission_tabm.to_csv("submission_tabm.csv", index=False)

print(f"submission_tabm.csv  shape={submission_tabm.shape}")
print(submission_tabm.head(10).to_string())
print(f"\nPredicted bot rate (test): {tabm_test_proba.mean():.4f}")

print("\n✓ TabM ensemble pipeline complete.")
print(f"  OOF AUC   : {tabm_oof_auc:.4f}")
print(f"  OOF AP    : {tabm_oof_ap:.4f}")
print(f"  Plots     : ./plots/tabm_*.png")
print(f"  Submission: ./submission_tabm.csv")


# ## XAI

# In[40]:


# =============================================================================
# 7. TABM XAI — INTEGRATED GRADIENTS (NO CAPTUM)
# =============================================================================
print("\n" + "="*60)
print("7. TABM XAI — INTEGRATED GRADIENTS (NO CAPTUM)")
print("="*60)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -------------------------------------------------------------------------
# Safe colors
# -------------------------------------------------------------------------
if "PALETTE" not in globals():
    PALETTE = {}

tabm_color = PALETTE.get("tabm", "#8E44AD")
human_color = PALETTE.get("human", "#2B9EB3")

# -------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------
required_names = ["folds", "X", "y", "FEAT_COLS", "cfg", "DEVICE",
                  "make_preprocessor", "build_tabm_model"]
missing = [name for name in required_names if name not in globals()]
if missing:
    raise ValueError(f"Missing required variables/functions for XAI: {missing}")

if "FINAL_SEED" not in globals():
    FINAL_SEED = 42

# -------------------------------------------------------------------------
# Helper: rebuild one fold model + transformed data
# -------------------------------------------------------------------------
def load_tabm_fold_for_xai(fold_no, cfg, final_seed=42):
    tr_idx, val_idx = folds[fold_no - 1]

    X_tr_raw, X_val_raw = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    pre = make_preprocessor(
        X_tr_raw,
        seed=final_seed * 100 + fold_no,
        method=cfg["preprocessor"]
    )

    X_tr = pre.transform(X_tr_raw).astype(np.float32)
    X_val = pre.transform(X_val_raw).astype(np.float32)

    model = build_tabm_model(X.shape[1], cfg)
    ckpt_path = f"logs/tabm_final_fold{fold_no}_best.pt"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "preprocessor": pre,
        "tr_idx": tr_idx,
        "val_idx": val_idx,
        "X_tr": X_tr,
        "X_val": X_val,
        "y_tr": y_tr,
        "y_val": y_val,
    }

# -------------------------------------------------------------------------
# Wrapper: TabM -> one probability per row
# -------------------------------------------------------------------------
class TabMProbWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x_num):
        logits = self.base_model(x_num).squeeze(-1)   # (B, k)
        proba = torch.sigmoid(logits).mean(dim=1)     # (B,)
        return proba

# -------------------------------------------------------------------------
# Pure PyTorch Integrated Gradients
# -------------------------------------------------------------------------
def integrated_gradients(model, x, baseline, n_steps=64):
    """
    model: returns shape (B,)
    x: shape (B, D)
    baseline: shape (B, D) or (1, D)
    """
    model.eval()

    if baseline.shape[0] == 1 and x.shape[0] > 1:
        baseline = baseline.repeat(x.shape[0], 1)

    alphas = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=x.device)[1:]
    total_grads = torch.zeros_like(x)

    for alpha in alphas:
        x_step = baseline + alpha * (x - baseline)
        x_step.requires_grad_(True)

        preds = model(x_step)                # (B,)
        grads = torch.autograd.grad(
            outputs=preds.sum(),
            inputs=x_step,
            retain_graph=False,
            create_graph=False
        )[0]

        total_grads += grads.detach()

    avg_grads = total_grads / n_steps
    attributions = (x - baseline) * avg_grads

    with torch.no_grad():
        pred_x = model(x)
        pred_base = model(baseline)
        completeness_delta = pred_x - pred_base - attributions.sum(dim=1)

    return attributions, completeness_delta

# -------------------------------------------------------------------------
# 7a. LOCAL EXPLANATION
# explain one case from the best fold
# -------------------------------------------------------------------------
if os.path.exists("logs/tabm_fold_metrics.csv"):
    fold_metrics = pd.read_csv("logs/tabm_fold_metrics.csv")
    FOLD_TO_EXPLAIN = int(fold_metrics.sort_values("final_fold_auc", ascending=False).iloc[0]["fold"])
else:
    FOLD_TO_EXPLAIN = 1

print(f"Using fold {FOLD_TO_EXPLAIN} for local explanation.")

fold_pack = load_tabm_fold_for_xai(FOLD_TO_EXPLAIN, cfg, final_seed=FINAL_SEED)
xai_model = fold_pack["model"]
X_tr_fold = fold_pack["X_tr"]
X_val_fold = fold_pack["X_val"]
val_idx_fold = fold_pack["val_idx"]

wrapped_model = TabMProbWrapper(xai_model).to(DEVICE)
wrapped_model.eval()

baseline = torch.tensor(
    np.median(X_tr_fold, axis=0, keepdims=True),
    dtype=torch.float32,
    device=DEVICE
)

with torch.no_grad():
    x_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32, device=DEVICE)
    val_proba_fold = wrapped_model(x_val_tensor).cpu().numpy()

local_idx_in_fold = int(np.argmax(val_proba_fold))
global_row_idx = int(val_idx_fold[local_idx_in_fold])

x_local = torch.tensor(
    X_val_fold[local_idx_in_fold:local_idx_in_fold+1],
    dtype=torch.float32,
    device=DEVICE
)

attr_local, delta_local = integrated_gradients(
    wrapped_model,
    x_local,
    baseline,
    n_steps=64
)

attr_local = attr_local.detach().cpu().numpy()[0]

local_df = pd.DataFrame({
    "feature": FEAT_COLS,
    "attribution": attr_local,
    "abs_attribution": np.abs(attr_local),
    "raw_feature_value": X[global_row_idx],
}).sort_values("abs_attribution", ascending=False)

local_df.to_csv("logs/tabm_xai_local_explanation.csv", index=False)
print("Saved logs/tabm_xai_local_explanation.csv")
print(f"Explained global row index: {global_row_idx}")
print(f"Predicted bot probability (fold {FOLD_TO_EXPLAIN} model): {val_proba_fold[local_idx_in_fold]:.4f}")
print(f"Integrated Gradients completeness delta: {float(delta_local.cpu().numpy()[0]):.6f}")

print("\nTop local features:")
print(local_df.head(15).to_string(index=False))

top_local = local_df.head(15).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 6))
colors = [tabm_color if v > 0 else human_color for v in top_local["attribution"]]
ax.barh(top_local["feature"], top_local["attribution"], color=colors)
ax.set_title("TabM Local Explanation (Integrated Gradients)")
ax.set_xlabel("Attribution")
plt.tight_layout()
plt.show()
save(fig, "tabm_xai_local_ig")

# -------------------------------------------------------------------------
# 7b. GLOBAL EXPLANATION
# aggregate mean absolute IG across all folds
# -------------------------------------------------------------------------
MAX_ROWS_PER_FOLD = 128
global_attr_sum = np.zeros(len(FEAT_COLS), dtype=np.float64)
global_count = 0

for fold_no in range(1, len(folds) + 1):
    fold_pack = load_tabm_fold_for_xai(fold_no, cfg, final_seed=FINAL_SEED)
    fold_model = fold_pack["model"]
    X_tr_fold = fold_pack["X_tr"]
    X_val_fold = fold_pack["X_val"]

    wrapped_fold_model = TabMProbWrapper(fold_model).to(DEVICE)
    wrapped_fold_model.eval()

    baseline_fold = torch.tensor(
        np.median(X_tr_fold, axis=0, keepdims=True),
        dtype=torch.float32,
        device=DEVICE
    )

    n_rows = min(MAX_ROWS_PER_FOLD, len(X_val_fold))
    rng = np.random.default_rng(100 + fold_no)
    sample_idx = rng.choice(len(X_val_fold), size=n_rows, replace=False)

    x_sample = torch.tensor(
        X_val_fold[sample_idx],
        dtype=torch.float32,
        device=DEVICE
    )

    attr_global, delta_global = integrated_gradients(
        wrapped_fold_model,
        x_sample,
        baseline_fold,
        n_steps=32
    )

    attr_global = attr_global.detach().cpu().numpy()
    global_attr_sum += np.abs(attr_global).sum(axis=0)
    global_count += attr_global.shape[0]

global_mean_abs = global_attr_sum / max(global_count, 1)

global_df = pd.DataFrame({
    "feature": FEAT_COLS,
    "mean_abs_attribution": global_mean_abs,
}).sort_values("mean_abs_attribution", ascending=False)

global_df.to_csv("logs/tabm_xai_global_importance.csv", index=False)
print("\nSaved logs/tabm_xai_global_importance.csv")
print("\nTop global features:")
print(global_df.head(20).to_string(index=False))

top_global = global_df.head(20).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top_global["feature"], top_global["mean_abs_attribution"], color=tabm_color)
ax.set_title("TabM Global Feature Importance (mean |Integrated Gradients|)")
ax.set_xlabel("Mean absolute attribution")
plt.tight_layout()
plt.show()
save(fig, "tabm_xai_global_ig")

print("\n✓ TabM XAI complete.")
print("Artifacts:")
print("  - logs/tabm_xai_local_explanation.csv")
print("  - logs/tabm_xai_global_importance.csv")
print("  - plots/tabm_xai_local_ig.png")
print("  - plots/tabm_xai_global_ig.png")


# # Model Comparison

# In[41]:


# =============================================================================
# 1. LOAD MODEL SUMMARIES
# =============================================================================
print("\n" + "="*60)
print("1. MODEL COMPARISON")
print("="*60)

summary_frames = []
for path in [
    "logs/rf_summary.csv",
    "logs/lgbm_summary.csv",
    "logs/tabm_summary.csv",
]:
    if os.path.exists(path):
        summary_frames.append(pd.read_csv(path))

comparison_df = pd.concat(summary_frames, ignore_index=True)
comparison_df.to_csv("logs/model_comparison_summary.csv", index=False)

print(comparison_df.to_string(index=False))
print("Saved logs/model_comparison_summary.csv")


# In[42]:


# =============================================================================
# 2. COMPARISON PLOTS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Comparison", fontsize=14)

axes[0].bar(comparison_df["model"], comparison_df["oof_auc"], color=["#2B9EB3", "#E84545", "#8E44AD"][:len(comparison_df)])
axes[0].set_title("OOF AUC")
axes[0].set_ylabel("AUC")

ap_values = comparison_df["oof_ap"] if "oof_ap" in comparison_df.columns else np.nan
axes[1].bar(comparison_df["model"], ap_values, color=["#2B9EB3", "#E84545", "#8E44AD"][:len(comparison_df)])
axes[1].set_title("OOF Average Precision")
axes[1].set_ylabel("AP")

plt.tight_layout()
plt.show()
save(fig, "model_comparison_auc_ap")

best_model_name = comparison_df.sort_values("oof_auc", ascending=False).iloc[0]["model"]
print(f"\nBest model by OOF AUC: {best_model_name}")


# In[ ]:




