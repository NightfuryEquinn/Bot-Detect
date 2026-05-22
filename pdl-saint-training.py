#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ─── 0. Install & Imports ────────────────────────────────────────────────────
get_ipython().system('pip install shap -q')

import os, warnings, zipfile, glob, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, classification_report, f1_score)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import shap

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)

PALETTE = {"bot": "#E84545", "human": "#2B9EB3", "neutral": "#6C757D", "accent": "#F5A623"}
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 140, "figure.facecolor": "#0F1117",
                     "axes.facecolor": "#1A1D27", "axes.labelcolor": "#E0E0E0",
                     "xtick.color": "#A0A0A0", "ytick.color": "#A0A0A0",
                     "text.color": "#E0E0E0", "grid.color": "#2A2D37",
                     "axes.spines.top": False, "axes.spines.right": False})

SAVE_DIR = "plots_saint"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

def save(fig, name):
    fig.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ↳ Saved  {SAVE_DIR}/{name}.png")


# In[2]:


# ─── 1. Feature Engineering (mirrors pdl-preprocessing.ipynb) ────────────────
print("\n" + "="*60)
print("1. LOADING & ENGINEERING FEATURES")
print("="*60)

INPUT_DIR = "/kaggle/input/competitions/facebook-recruiting-iv-human-or-bot"
WORK_DIR  = "/kaggle/working"

for zpath in glob.glob(f"{INPUT_DIR}/*.zip"):
    with zipfile.ZipFile(zpath, "r") as z:
        members = z.namelist()
        to_extract = [m for m in members
                      if m.endswith(".csv") and not os.path.exists(os.path.join(WORK_DIR, m))]
        if to_extract:
            z.extractall(WORK_DIR, members=to_extract)

def get_file_path(filename):
    if os.path.exists(f"{WORK_DIR}/{filename}"):
        return f"{WORK_DIR}/{filename}"
    return f"{INPUT_DIR}/{filename}"

train = pd.read_csv(get_file_path("train.csv"))
test  = pd.read_csv(get_file_path("test.csv"))
bids  = pd.read_csv(get_file_path("bids.csv"))

print(f"train shape : {train.shape}")
print(f"test  shape : {test.shape}")
print(f"bids  shape : {bids.shape}")


# In[3]:


# ─── 2. Aggregate Bid-Level Features Per Bidder ──────────────────────────────
print("\n" + "="*60)
print("2. FEATURE AGGREGATION")
print("="*60)

bid_stats = bids.groupby("bidder_id").agg(
    bid_count         = ("bid_id",      "count"),
    auction_count     = ("auction",     "nunique"),
    device_count      = ("device",      "nunique"),
    country_count     = ("country",     "nunique"),
    ip_count          = ("ip",          "nunique"),
    url_count         = ("url",         "nunique"),
    merch_count       = ("merchandise", "nunique"),
    time_min          = ("time",        "min"),
    time_max          = ("time",        "max"),
    time_mean         = ("time",        "mean"),
    time_std          = ("time",        "std"),
).reset_index()

bid_stats["time_range"]          = bid_stats["time_max"] - bid_stats["time_min"]
bid_stats["bids_per_auction"]    = bid_stats["bid_count"] / (bid_stats["auction_count"] + 1)
bid_stats["bids_per_device"]     = bid_stats["bid_count"] / (bid_stats["device_count"]  + 1)
bid_stats["bids_per_country"]    = bid_stats["bid_count"] / (bid_stats["country_count"] + 1)
bid_stats["bids_per_url"]        = bid_stats["bid_count"] / (bid_stats["url_count"]     + 1)
bid_stats["bids_per_ip"]         = bid_stats["bid_count"] / (bid_stats["ip_count"]      + 1)
bid_stats["device_per_auction"]  = bid_stats["device_count"]  / (bid_stats["auction_count"] + 1)
bid_stats["country_per_auction"] = bid_stats["country_count"] / (bid_stats["auction_count"] + 1)
print("  Basic bid stats shape:", bid_stats.shape)

bids_sorted = bids.sort_values(["bidder_id", "time"])
bids_sorted["time_gap"] = bids_sorted.groupby("bidder_id")["time"].diff()
gap_stats = bids_sorted.groupby("bidder_id")["time_gap"].agg(
    gap_mean   = "mean",
    gap_std    = "std",
    gap_min    = "min",
    gap_median = "median",
    gap_q25    = lambda x: x.quantile(0.25),
    gap_q75    = lambda x: x.quantile(0.75),
).reset_index()
gap_stats["gap_iqr"] = gap_stats["gap_q75"] - gap_stats["gap_q25"]
gap_stats["gap_cv"]  = gap_stats["gap_std"] / (gap_stats["gap_mean"] + 1e-9)
print("  Gap stats shape:", gap_stats.shape)

merch_entropy = (bids.groupby("bidder_id")["merchandise"]
                 .apply(lambda x: -(x.value_counts(normalize=True)
                                    .apply(lambda p: p * np.log(p + 1e-9)).sum()))
                 .reset_index()
                 .rename(columns={"merchandise": "merch_entropy"}))

country_entropy = (bids.groupby("bidder_id")["country"]
                   .apply(lambda x: -(x.value_counts(normalize=True)
                                      .apply(lambda p: p * np.log(p + 1e-9)).sum()))
                   .reset_index()
                   .rename(columns={"country": "country_entropy"}))

bids_sorted2 = bids.sort_values(["bidder_id", "auction", "time"])
bids_sorted2["intra_auction_gap"] = bids_sorted2.groupby(["bidder_id","auction"])["time"].diff()
fast_bids = (bids_sorted2.groupby("bidder_id")
             .apply(lambda df: (df["intra_auction_gap"] < 5_000_000).sum() / max(len(df), 1))
             .reset_index()
             .rename(columns={0: "fast_bid_ratio"}))

last_bid  = bids.sort_values("time").groupby(["bidder_id","auction"]).last().reset_index()
win_proxy = last_bid.groupby("bidder_id").size().reset_index().rename(columns={0: "auctions_last_bid"})
bid_stats = bid_stats.merge(win_proxy, on="bidder_id", how="left")
bid_stats["last_bid_ratio"] = bid_stats["auctions_last_bid"] / (bid_stats["auction_count"] + 1)

features = (bid_stats
            .merge(gap_stats,       on="bidder_id", how="left")
            .merge(merch_entropy,   on="bidder_id", how="left")
            .merge(country_entropy, on="bidder_id", how="left")
            .merge(fast_bids,       on="bidder_id", how="left"))

print(f"  Total feature frame shape: {features.shape}")
print("  Columns:", features.columns.tolist())


# In[4]:


# ─── 3. Merge with Train / Test Labels ───────────────────────────────────────
print("\n" + "="*60)
print("3. PREPARING TRAIN / TEST SPLITS")
print("="*60)

FEAT_COLS = [c for c in features.columns if c != "bidder_id"]
print(f"  Feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")

train_feat  = train.merge(features, on="bidder_id", how="left")
X_train_raw = train_feat[FEAT_COLS].copy()
y_train     = train_feat["outcome"].values.astype(int)

test_feat  = test.merge(features, on="bidder_id", how="left")
X_test_raw = test_feat[FEAT_COLS].copy()

print(f"  Train X shape : {X_train_raw.shape}")
print(f"  Train y shape : {y_train.shape}  (bots={y_train.sum()}, humans={len(y_train)-y_train.sum()})")
print(f"  Test  X shape : {X_test_raw.shape}")


# In[5]:


# ─── 4. Pre-processing ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. PRE-PROCESSING")
print("="*60)

X_train_filled = X_train_raw.copy()
X_test_filled  = X_test_raw.copy()

for col in FEAT_COLS:
    median_val = X_train_filled[col].median()
    fill = median_val if not np.isnan(median_val) else 0
    X_train_filled[col] = X_train_filled[col].fillna(fill)
    X_test_filled[col]  = X_test_filled[col].fillna(fill)

print("  NaN fill completed.")
print("  Train NaNs remaining:", X_train_filled.isna().sum().sum())
print("  Test  NaNs remaining:", X_test_filled.isna().sum().sum())

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filled)
X_test_scaled  = scaler.transform(X_test_filled)

print("  Scaling applied.")
print(f"  Train matrix: {X_train_scaled.shape}")
print(f"  Test  matrix: {X_test_scaled.shape}")


# In[6]:


# ─── 5. Class-Imbalance Analysis ─────────────────────────────────────────────
print("\n" + "="*60)
print("5. CLASS IMBALANCE")
print("="*60)

bot_ratio  = y_train.mean()
pos_weight = (1 - bot_ratio) / bot_ratio
print(f"  Bot ratio         : {bot_ratio*100:.2f}%")
print(f"  Positive weight   : {pos_weight:.2f}x  (used for weighted loss)")

fig, ax = plt.subplots(figsize=(5, 4))
vcs  = pd.Series(y_train).map({0: "Human", 1: "Bot"}).value_counts()
bars = ax.bar(vcs.index, vcs.values,
              color=[PALETTE["human"], PALETTE["bot"]], width=0.5, edgecolor="none")
ax.bar_label(bars, fmt="%d", padding=4, color="#E0E0E0")
ax.set_title("Class Balance · Training Set")
ax.set_ylabel("Count")
plt.tight_layout(); plt.show()
save(fig, "00_class_balance")


# In[7]:


# ─── 6. SAINT Architecture ───────────────────────────────────────────────────
# Paper: SAINT: Improved Neural Networks for Tabular Data via Row Attention
#        and Contrastive Pre-Training (Somepalli et al., 2021)
#
# FeatureTokenizer  : scalar_i * W_i + b_i  -> d-dim embedding per feature
# CLS token         : prepended; its final state feeds the MLP head
# Column attention  : standard MHA over the F+1 token sequence (within a row)
# Row attention     : intersample MHA across B samples for each feature
#                     DISABLED at inference / SHAP for i.i.d. attribution
print("\n" + "="*60)
print("6. SAINT ARCHITECTURE")
print("="*60)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, dim))
        self.bias   = nn.Parameter(torch.zeros(n_features, dim))
        self.feature_embed = nn.Parameter(torch.randn(n_features, dim) * 0.02)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias + self.feature_embed


class _PreNormAttn(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        out, _ = self.attn(xn, xn, xn)
        return x + out


class GEGLU(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.proj = nn.Linear(dim, dim * mult * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        return x / keep_prob * random_tensor.floor()


class SAINTLayer(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, dropout, drop_path=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.col_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            GEGLU(dim, ffn_mult),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

        self.norm3 = nn.LayerNorm(dim)
        self.row_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        self.norm4 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path)

    def forward(self, x, use_intersample=True):
        # Column attention
        x = x + self.drop_path(
            self.col_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        )

        # FFN
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        # Row attention (exclude CLS)
        if use_intersample:
            cls, feats = x[:, :1], x[:, 1:]

            feats = feats.permute(1, 0, 2)
            feats = feats + self.drop_path(
                self.row_attn(self.norm3(feats), self.norm3(feats), self.norm3(feats))[0]
            )
            feats = feats + self.drop_path(self.ffn(self.norm4(feats)))
            feats = feats.permute(1, 0, 2)

            x = torch.cat([cls, feats], dim=1)

        return x


class SAINT(nn.Module):
    def __init__(
        self,
        n_features,
        dim=128,
        depth=6,
        n_heads=8,
        ffn_mult=4,
        attn_dropout=0.1,
        mlp_dropout=0.2,
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(n_features, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_features + 1, dim) * 0.02)

        self.input_norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([
            SAINTLayer(dim, n_heads, ffn_mult, attn_dropout)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x, use_intersample=True):
        B = x.shape[0]

        tokens = self.tokenizer(x)
        tokens = self.input_norm(tokens)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        tokens = tokens + self.pos_embed

        for layer in self.layers:
            tokens = layer(tokens, use_intersample)

        # ✅ Mean pooling (better than CLS)
        pooled = tokens[:, 1:].mean(dim=1)

        return self.head(pooled).squeeze(-1)


_probe       = SAINT(n_features=30)
total_params = sum(p.numel() for p in _probe.parameters() if p.requires_grad)
del _probe
print(f"  Architecture ready — {total_params:,} trainable params (probe on 30 features)")
print("  Column attn  : features attend to each other within a row")
print("  Row attn     : samples attend across the batch (train only)")
print("  SHAP mode    : row attn disabled -> i.i.d. forward pass")


# In[52]:


# ─── 7. SAINT Hyperparameters ────────────────────────────────────────────────
print("\n" + "="*60)
print("7. SAINT HYPERPARAMETERS & TRAINING CONFIG")
print("="*60)

SAINT_PARAMS = dict(
    dim          = 128,
    depth        = 4,
    n_heads      = 8,
    ffn_mult     = 4,
    attn_dropout = 0.1,
    mlp_dropout  = 0.1,
)

N_SPLITS     = 5
BATCH_SZ     = 512
MAX_EPOCHS   = 200
PATIENCE     = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for k, v in {**SAINT_PARAMS,
             "N_SPLITS": N_SPLITS, "BATCH_SZ": BATCH_SZ,
             "MAX_EPOCHS": MAX_EPOCHS, "PATIENCE": PATIENCE,
             "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
             "DEVICE": str(DEVICE)}.items():
    print(f"  {k:20s}: {v}")


# In[53]:


from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

# ─── Mixup ────────────────────────────────────────────────────────────────
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix

# ─── Training ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("8. STRATIFIED K-FOLD CROSS VALIDATION (OPTIMIZED)")
print("="*60)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros(len(y_train), dtype=np.float32)
test_preds = np.zeros(len(X_test_scaled), dtype=np.float32)

fold_aucs, fold_aps = [], []
fold_models = []

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
    print(f"\n{'─'*50}\n  FOLD {fold}/{N_SPLITS}\n{'─'*50}")

    # ─── Data ─────────────────────────────────────────────────────────────
    X_tr  = torch.tensor(X_train_scaled[tr_idx], dtype=torch.float32)
    y_tr  = torch.tensor(y_train[tr_idx], dtype=torch.float32)
    X_val = torch.tensor(X_train_scaled[val_idx], dtype=torch.float32)
    y_val_np = y_train[val_idx]

    train_dl = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SZ,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    val_dl = DataLoader(
        TensorDataset(X_val, torch.zeros(len(X_val))),
        batch_size=512,
        shuffle=False,
        pin_memory=True
    )

    # ─── Class imbalance (ONLY pos_weight) ────────────────────────────────
    pos_ratio = y_tr.mean().item()
    pos_weight = torch.tensor([(1 - pos_ratio) / (pos_ratio + 1e-9)], device=DEVICE)

    # ─── Model ───────────────────────────────────────────────────────────
    model = SAINT(n_features=len(FEAT_COLS), **SAINT_PARAMS).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=5e-5
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=8e-4,
        steps_per_epoch=len(train_dl),
        epochs=MAX_EPOCHS,
        pct_start=0.1
    )

    best_auc, best_state = 0.0, None
    patience_ctr = 0

    # ─── Epoch Loop ──────────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            # 🔥 Mixup
            xb, yb = mixup(xb, yb, alpha=0.2)

            optimizer.zero_grad()

            logits = model(xb, use_intersample=False)
            loss = criterion(logits, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # ─── Validation ──────────────────────────────────────────────────
        model.eval()
        preds = []

        with torch.no_grad():
            for xb, _ in val_dl:
                xb = xb.to(DEVICE, non_blocking=True)
                logits = model(xb, use_intersample=False)
                preds.append(torch.sigmoid(logits).cpu())

        val_preds = torch.cat(preds).numpy()
        val_auc = roc_auc_score(y_val_np, val_preds)

        # ─── Early Stopping ──────────────────────────────────────────────
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | "
                  f"loss {train_loss/len(train_dl):.4f} | "
                  f"val AUC {val_auc:.4f} | best {best_auc:.4f} | "
                  f"patience {patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    # ─── Load best model ────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    fold_models.append(model)

    # ─── OOF predictions ────────────────────────────────────────────────
    with torch.no_grad():
        val_logits = model(X_val.to(DEVICE), use_intersample=False)
        val_proba = torch.sigmoid(val_logits).cpu().numpy()

    oof_preds[val_idx] = val_proba

    auc = roc_auc_score(y_val_np, val_proba)
    ap  = average_precision_score(y_val_np, val_proba)

    fold_aucs.append(auc)
    fold_aps.append(ap)

    print(f"  Fold {fold} — AUC: {auc:.4f} | AP: {ap:.4f}")

    # ─── Test predictions ───────────────────────────────────────────────
    with torch.no_grad():
        test_logits = model(X_test_tensor.to(DEVICE), use_intersample=False)
        test_preds += torch.sigmoid(test_logits).cpu().numpy() / N_SPLITS

# ─── Final Metrics ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CV AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  CV AP  : {np.mean(fold_aps):.4f} ± {np.std(fold_aps):.4f}")
print(f"{'='*60}")


# In[54]:


# ─── 9. OOF Evaluation ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("9. OUT-OF-FOLD EVALUATION")
print("="*60)

from sklearn.metrics import roc_curve, precision_recall_curve

thresholds  = np.arange(0.05, 0.95, 0.01)
f1s         = [f1_score(y_train, (oof_preds >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1s)]
best_f1     = np.max(f1s)
print(f"  Best threshold  : {best_thresh:.2f}")
print(f"  Best F1 (OOF)   : {best_f1:.4f}")

oof_labels = (oof_preds >= best_thresh).astype(int)
print(classification_report(y_train, oof_labels, target_names=["Human", "Bot"]))

fpr, tpr, _   = roc_curve(y_train, oof_preds)
prec, rec, _  = precision_recall_curve(y_train, oof_preds)
auc_roc       = roc_auc_score(y_train, oof_preds)
ap_oof        = average_precision_score(y_train, oof_preds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("OOF Performance — SAINT", fontsize=14, y=1.01)

ax = axes[0]
ax.plot(fpr, tpr, color=PALETTE["bot"],   lw=2, label=f"ROC (AUC={auc_roc:.4f})")
ax.plot([0,1],[0,1], color="#4A4D5A", ls="--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend()

ax = axes[1]
ax.plot(rec, prec, color=PALETTE["human"], lw=2, label=f"PR (AP={ap_oof:.4f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend()

plt.tight_layout(); plt.show()
save(fig, "01_oof_roc_pr")


# In[55]:


# ─── 10. Confusion Matrix ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_train, oof_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"],
            ax=ax, linewidths=0.5)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"OOF Confusion Matrix (thresh={best_thresh:.2f})")
plt.tight_layout(); plt.show()
save(fig, "02_confusion_matrix")


# In[56]:


# ─── 11. SHAP Analysis ───────────────────────────────────────────────────────
# GradientExplainer uses backprop-based integrated gradients.
# Row (intersample) attention is disabled so each sample's output depends
# only on its own features -- the prerequisite for valid additive attribution.
print("\n" + "="*60)
print("11. SHAP FEATURE ATTRIBUTION")
print("="*60)

best_fold_idx = int(np.argmax(fold_aucs))
print(f"  Using fold {best_fold_idx+1} model (AUC={fold_aucs[best_fold_idx]:.4f})")

shap_model = fold_models[best_fold_idx]
shap_model.eval()


class _InferWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x, use_intersample=False)
        out = torch.sigmoid(out)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out


wrapped = _InferWrapper(shap_model)

N_BG  = min(200, len(X_train_scaled))
N_EXP = min(300, len(X_train_scaled))
rng   = np.random.default_rng(SEED)
bg_idx  = rng.choice(len(X_train_scaled), N_BG,  replace=False)
exp_idx = rng.choice(len(X_train_scaled), N_EXP, replace=False)

background = torch.tensor(X_train_scaled[bg_idx],  dtype=torch.float32).to(DEVICE)
explain_x  = torch.tensor(X_train_scaled[exp_idx], dtype=torch.float32).to(DEVICE)

print(f"  Background : {N_BG} samples | Explain : {N_EXP} samples")
print("  Computing SHAP values via GradientExplainer ...")

explainer   = shap.GradientExplainer(wrapped, background)
shap_values = explainer.shap_values(explain_x)

if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.array(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

mean_shap = np.abs(shap_values).mean(axis=0)
shap_df   = (pd.DataFrame({"feature": FEAT_COLS, "mean_|shap|": mean_shap})
             .sort_values("mean_|shap|", ascending=False)
             .reset_index(drop=True))
print("\n" + shap_df.to_string())

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=shap_df, x="mean_|shap|", y="feature",
            palette="flare", orient="h", ax=ax)
ax.set_title("SAINT — Mean |SHAP| Feature Importance")
ax.set_xlabel("Mean |SHAP Value|"); ax.set_ylabel("")
plt.tight_layout(); plt.show()
save(fig, "03_shap_importance")

shap.summary_plot(shap_values, X_train_scaled[exp_idx],
                  feature_names=FEAT_COLS, show=False, plot_size=(10, 7))
plt.title("SAINT — SHAP Summary (Beeswarm)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/03b_shap_beeswarm.png",
            bbox_inches="tight", facecolor="#0F1117")
plt.show()
print(f"  ↳ Saved {SAVE_DIR}/03b_shap_beeswarm.png")

top_feat = shap_df.iloc[0]["feature"]
top_idx  = FEAT_COLS.index(top_feat)
shap.dependence_plot(top_idx, shap_values, X_train_scaled[exp_idx],
                     feature_names=FEAT_COLS, show=False)
plt.title(f"SAINT — SHAP Dependence: {top_feat}")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/03c_shap_dependence_{top_feat}.png",
            bbox_inches="tight", facecolor="#0F1117")
plt.show()
print(f"  ↳ Saved dependence plot for '{top_feat}'")


# In[57]:


# ─── 12. Prediction Histogram ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(oof_preds[y_train == 0], bins=50, alpha=0.7, color=PALETTE["human"], label="Human (0)")
ax.hist(oof_preds[y_train == 1], bins=50, alpha=0.7, color=PALETTE["bot"],   label="Bot   (1)")
ax.axvline(best_thresh, color=PALETTE["accent"], ls="--", lw=2, label=f"Threshold={best_thresh:.2f}")
ax.set_xlabel("Predicted Probability (Bot)"); ax.set_ylabel("Count")
ax.set_title("OOF Score Distribution"); ax.legend()
plt.tight_layout(); plt.show()
save(fig, "04_score_distribution")


# In[58]:


# ─── 13. Export Submission ───────────────────────────────────────────────────
print("\n" + "="*60)
print("13. GENERATING SUBMISSION")
print("="*60)

submission = pd.DataFrame({"bidder_id": test["bidder_id"], "prediction": test_preds})
submission_path = os.path.join(WORK_DIR, "submission_saint.csv")
submission.to_csv(submission_path, index=False)
print(f"  Submission saved to: {submission_path}")
print(f"  Shape : {submission.shape}")
print(submission.head(10).to_string())


# In[59]:


# ─── 14. Summary Dashboard ───────────────────────────────────────────────────
print("\n" + "="*60)
print("14. FINAL SUMMARY")
print("="*60)

summary = {
    "Model"             : "SAINT (Self-Attention + Intersample Attention)",
    "Features"          : len(FEAT_COLS),
    "Embedding dim"     : SAINT_PARAMS["dim"],
    "Depth (layers)"    : SAINT_PARAMS["depth"],
    "Attention heads"   : SAINT_PARAMS["n_heads"],
    "CV Folds"          : N_SPLITS,
    "CV AUC (mean)"     : f"{np.mean(fold_aucs):.4f}",
    "CV AUC (std)"      : f"{np.std(fold_aucs):.4f}",
    "CV AP  (mean)"     : f"{np.mean(fold_aps):.4f}",
    "OOF AUC"           : f"{auc_roc:.4f}",
    "OOF AP"            : f"{ap_oof:.4f}",
    "Best F1"           : f"{best_f1:.4f}",
    "Best Threshold"    : f"{best_thresh:.2f}",
    "Explainability"    : "SHAP GradientExplainer",
}
for k, v in summary.items():
    print(f"  {k:25s}: {v}")


# In[ ]:




