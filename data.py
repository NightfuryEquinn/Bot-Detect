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


# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "=" * 60)
print("1. LOADING DATA")
print("=" * 60)

INPUT_DIR = "./datasets"

# Read the CSVs
train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
bids = pd.read_csv(f"{INPUT_DIR}/bids.csv")

for name, df in [("train", train), ("test", test), ("bids", bids)]:
    print(f"\n{name}.csv  shape={df.shape}")
    print(df.dtypes.to_string())


# =============================================================================
# 2. DATA UNDERSTANDING
# =============================================================================
print("\n" + "=" * 60)
print("2. DATA UNDERSTANDING")
print("=" * 60)

# ── 2a. Basic stats ──────────────────────────────────────────────────────────
print("\n── TRAIN ──")
print(train.describe(include="all").T)
print("\n── BIDS (numeric) ──")
print(bids.describe().T)


# ── 2b. Null audit ──────────────────────────────────────────────────────────
def null_report(df, label):
    null = df.isnull().sum()
    pct = null / len(df) * 100
    return pd.DataFrame({"nulls": null, "pct": pct}).query("nulls > 0").assign(source=label)


null_df = pd.concat([null_report(train, "train"),
                     null_report(test, "test"),
                     null_report(bids, "bids")])
print("\n── NULL REPORT ──")
print(null_df.to_string() if not null_df.empty else "No nulls found.")

# ── 2c. Class balance ────────────────────────────────────────────────────────
print("\n── CLASS BALANCE (train) ──")
vc = train["outcome"].value_counts()
print(vc.to_string())
print(f"  Bot ratio: {vc[1] / len(train) * 100:.1f}%")

# ── 2d. Overlap — how many bidders have bids? ─────────────────────────────────
train_with_bids = set(train.bidder_id) & set(bids.bidder_id)
test_with_bids = set(test.bidder_id) & set(bids.bidder_id)
print(f"\nTrain bidders with bids : {len(train_with_bids)}/{len(train)} "
      f"({len(train_with_bids) / len(train) * 100:.1f}%)")
print(f"Test  bidders with bids : {len(test_with_bids)}/{len(test)} "
      f"({len(test_with_bids) / len(test) * 100:.1f}%)")

# ── 2e. Bids per bidder distribution (raw) ────────────────────────────────────
bids_per_bidder_raw = bids.groupby("bidder_id").size()
print(f"\nBids-per-bidder: min={bids_per_bidder_raw.min()} "
      f"max={bids_per_bidder_raw.max()} "
      f"median={bids_per_bidder_raw.median():.0f}")


# =============================================================================
# 3. VISUALISATION — RAW DATA
# =============================================================================
print("\n" + "=" * 60)
print("3. RAW DATA VISUALISATION")
print("=" * 60)

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
ax.set_xlabel("Bid count");
ax.set_ylabel("Frequency (log)")

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
ax.set_xlabel("Time");
ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
save(fig, "01_raw_overview")


# =============================================================================
# 4. DATA CLEANING
# =============================================================================
print("\n" + "=" * 60)
print("4. DATA CLEANING")
print("=" * 60)

bids_clean = bids.copy()
train_clean = train.copy()
test_clean = test.copy()

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
      f"({zero_interval / len(bids_clean) * 100:.1f}%)")

print(f"\nCleaned bids shape: {bids_clean.shape}")


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("5. FEATURE ENGINEERING")
print("=" * 60)


# ── Helper: entropy ───────────────────────────────────────────────────────────
def entropy(series):
    p = series.value_counts(normalize=True)
    return -np.sum(p * np.log2(p + 1e-9))


# ── 5a. Aggregate bids → per-bidder features ─────────────────────────────────
g = bids_clean.groupby("bidder_id")

feats = pd.DataFrame({
    # Volume
    "bid_count": g["bid_id"].count(),
    "unique_auctions": g["auction"].nunique(),
    "unique_devices": g["device"].nunique(),
    "unique_countries": g["country"].nunique(),
    "unique_ips": g["ip"].nunique(),
    "unique_urls": g["url"].nunique(),
    "unique_merchandise": g["merchandise"].nunique(),

    # Ratios (diversity / activity)
    "bids_per_auction": g["bid_id"].count() / g["auction"].nunique(),
    "ips_per_bid": g["ip"].nunique() / g["bid_id"].count(),
    "countries_per_bid": g["country"].nunique() / g["bid_id"].count(),
    "devices_per_bid": g["device"].nunique() / g["bid_id"].count(),

    # Time-based
    "time_span": g["time"].max() - g["time"].min(),
    "time_mean": g["time"].mean(),
    "time_std": g["time"].std().fillna(0),
    "time_diff_mean": g["time_diff"].mean(),
    "time_diff_std": g["time_diff"].std().fillna(0),
    "time_diff_min": g["time_diff"].min(),
    "zero_interval_count": g["time_diff"].apply(lambda x: (x == 0).sum()),
    "zero_interval_ratio": g["time_diff"].apply(lambda x: (x == 0).mean()),

    # Entropy (information diversity)
    "country_entropy": g["country"].apply(entropy),
    "device_entropy": g["device"].apply(entropy),
    "url_entropy": g["url"].apply(entropy),
    "merchandise_entropy": g["merchandise"].apply(entropy),
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

for n_strip in [1, 2, 3, 4, 5]:  # coarse → fine granularity
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

    feats = feats.join(burst_max, how="left")
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
           [f"burst_max_strip{n}" for n in [1, 2, 3, 4, 5]]:
    all_feats[f"log_{col}"] = np.log1p(all_feats[col])

print(f"Feature matrix shape: {all_feats.shape}")
print(all_feats.dtypes.value_counts())

# ── 5g. Merge with train/test ─────────────────────────────────────────────────
train_feat = (train_clean
              .merge(all_feats.reset_index(), on="bidder_id", how="left")
              .fillna(0))
test_feat = (test_clean
             .merge(all_feats.reset_index(), on="bidder_id", how="left")
             .fillna(0))

print(f"\ntrain_feat shape : {train_feat.shape}")
print(f"test_feat  shape : {test_feat.shape}")


# =============================================================================
# 6. VISUALISATION — CLEANED + ENGINEERED DATA
# =============================================================================
print("\n" + "=" * 60)
print("6. CLEANED / ENGINEERED DATA VISUALISATION")
print("=" * 60)

label_map = {0: "human", 1: "bot"}
train_feat["label"] = train_feat["outcome"].map(label_map)
colors = [PALETTE["human"], PALETTE["bot"]]

# ── 6a. Key feature distributions — human vs bot ─────────────────────────────
key_feats = [
    ("log_bid_count", "Log Bid Count"),
    ("log_unique_ips", "Log Unique IPs"),
    ("zero_interval_ratio", "Zero-interval Ratio"),
    ("country_entropy", "Country Entropy"),
    ("burst_ratio_strip3", "Burst Ratio (strip-3)"),
    ("burst_max_strip4", "Max Burst Size (strip-4)"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Feature Distributions: Human vs Bot", fontsize=16)

for ax, (col, title) in zip(axes.flat, key_feats):
    for i, (label, grp) in enumerate(train_feat.groupby("label")):
        vals = grp[col].dropna()
        ax.hist(vals, bins=50, alpha=0.65, color=colors[i],
                label=label, edgecolor="none", density=True)
    ax.set_title(title);
    ax.legend()

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
        data_b = train_feat.loc[train_feat["label"] == "bot", col]
        ax.boxplot([data_h, data_b], labels=["Human", "Bot"],
                   patch_artist=True,
                   boxprops=dict(facecolor="none", color="#A0A0A0"),
                   medianprops=dict(color="#FFD700", linewidth=2),
                   whiskerprops=dict(color="#A0A0A0"),
                   capprops=dict(color="#A0A0A0"),
                   flierprops=dict(
                       marker="o",  # Use a full circle instead of a tiny dot
                       markersize=4,  # Doubled the size for better visibility
                       alpha=0.6,  # Transparency helps if there is dense overlapping
                       markerfacecolor="#FF6B6B",  # Specifically set the fill color
                       markeredgecolor="none"  # Remove the edge so colors don't muddy together
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
ax.set_xlabel("Log Bid Count");
ax.set_ylabel("Zero-Interval Ratio")
ax.set_title("Zero-Interval Ratio vs Log Bid Count")
ax.legend()
plt.tight_layout()
plt.show()
save(fig, "05_zero_interval_scatter")

# ── 6e. Merchandise ratio — bot vs human ──────────────────────────────────────
merch_cols = [c for c in train_feat.columns if c.startswith("merch_") and "ratio" in c]
if merch_cols:
    means = train_feat.groupby("label")[merch_cols].mean().T
    means.index = [c.replace("merch_", "").replace("_ratio", "") for c in means.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    means.plot.bar(ax=ax, color=[PALETTE["human"], PALETTE["bot"]], edgecolor="none",
                   rot=30)
    ax.set_title("Merchandise Category Ratio — Human vs Bot")
    ax.set_ylabel("Mean ratio");
    ax.legend()
    plt.tight_layout()
    plt.show()
    save(fig, "06_merchandise_ratio")

# ── 6f. Country entropy comparison ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for i, (label, grp) in enumerate(train_feat.groupby("label")):
    ax.hist(grp["country_entropy"], bins=40, alpha=0.7, color=colors[i],
            label=label, edgecolor="none", density=True)
ax.set_xlabel("Country Entropy");
ax.set_ylabel("Density")
ax.set_title("Country Entropy Distribution — Human vs Bot")
ax.legend()
plt.tight_layout()
plt.show()
save(fig, "07_country_entropy")


# =============================================================================
# 7. EXPORT CLEANED FEATURES
# =============================================================================
print("\n" + "=" * 60)
print("7. EXPORTING")
print("=" * 60)

train_feat.drop(columns=["label"], inplace=True)
train_feat.to_csv(f"{INPUT_DIR}/train_features.csv", index=False)
test_feat.to_csv(f"{INPUT_DIR}/test_features.csv", index=False)
all_feats.to_csv(f"{INPUT_DIR}/bidder_features.csv")

print(f"train_features.csv  → {train_feat.shape}")
print(f"test_features.csv   → {test_feat.shape}")
print(f"bidder_features.csv → {all_feats.shape}")

# ── 7a. Feature summary ──────────────────────────────────────────────────────
feat_cols = [c for c in train_feat.columns
             if c not in ["bidder_id", "payment_account", "address", "outcome"]]
print(f"\nTotal engineered features : {len(feat_cols)}")

feature_groups = {
    "Volume": [c for c in feat_cols if any(x in c for x in ["count", "unique", "log"])],
    "Ratio": [c for c in feat_cols if "ratio" in c or "per_" in c],
    "Time": [c for c in feat_cols if "time" in c],
    "Burst": [c for c in feat_cols if "burst" in c],
    "Entropy": [c for c in feat_cols if "entropy" in c],
    "Merchandise": [c for c in feat_cols if "merch" in c],
    "Geography": [c for c in feat_cols if "country" in c or "subnet" in c],
}

for group, cols in feature_groups.items():
    print(f"  {group:15s}: {len(cols):3d} features")

print("\n✓ Preprocessing pipeline complete.")
print(f"  Plots saved to   ./{SAVE_DIR}/")
print(f"  Features saved to ./train_features.csv, test_features.csv")
