#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# pdl-tabm.py — TabM: Ensemble of Tabular Transformers with BatchEnsemble
# =============================================================================

import os, time, copy, random, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import tabm
import rtdl_num_embeddings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, auc,
    f1_score, precision_score, recall_score,
)
from utils.shared import PALETTE, configure_plots, save_fig

warnings.filterwarnings("ignore")
configure_plots()

SEED     = 42
SAVE_DIR = "result/tabm"
LOG_DIR  = "result/logs/tabm"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

assert torch.cuda.is_available(), "CUDA GPU is required but not available."
DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)
print(f"GPU: {torch.cuda.get_device_name(0)}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ─── Load feature data ────────────────────────────────────────────────────────
DATA_DIR  = "./datasets"
META_COLS = ["bidder_id", "payment_account", "address", "outcome", "label"]

train = pd.read_csv(f"{DATA_DIR}/train_features.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_features.csv")

FEAT_COLS = [c for c in train.columns if c not in META_COLS]
X         = train[FEAT_COLS].values.astype(np.float32)
y         = train["outcome"].values.astype(np.float32)
X_test    = test[FEAT_COLS].values.astype(np.float32)

print(f"Train : {X.shape}  (bots={int(y.sum())}, humans={int(len(y) - y.sum())})")
print(f"Test  : {X_test.shape}")
print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS[:6]} …")

N_SPLITS = 5
folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
print(f"Folds : {len(folds)}")


# =============================================================================
# 1. TABM SEARCH SPACE
# =============================================================================
print("\n" + "="*60)
print("1. TABM SEARCH SPACE")
print("="*60)

print(f"Reusing shared folds: {len(folds)} folds")

MAX_EPOCHS = 80
PATIENCE = 8
MIN_DELTA = 5e-4
GRAD_CLIP = 0.5
BATCH_SIZE = 256

# Tuning uses one seed for speed; final ensemble uses multiple seeds
TUNING_SEED = 42
FINAL_SEEDS = [42, 52, 62]
TOP_K_CONFIGS = 2
AUC_TIE_GAP = 0.002

# Balanced search space: not too aggressive, not too tiny
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

print("TabM configs:")
for i, cfg in enumerate(TABM_CONFIGS, 1):
    print(f"  Config {i}: {cfg}")


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

tabm_tuning_df.to_csv("result/logs/tabm/tabm_tuning_results.csv", index=False)
print("Saved result/logs/tabm/tabm_tuning_results.csv")

max_auc = tabm_tuning_df["mean_auc"].max()
candidates = tabm_tuning_df[tabm_tuning_df["mean_auc"] >= max_auc - AUC_TIE_GAP].copy()

selected_cfg_df = candidates.sort_values(
    ["std_auc", "mean_best_epoch", "mean_ap"],
    ascending=[True, True, False]
).head(TOP_K_CONFIGS).reset_index(drop=True)

print("\nSelected configs for final ensemble:")
print(selected_cfg_df.to_string(index=False))

selected_cfg_df.to_csv("result/logs/tabm/tabm_selected_configs.csv", index=False)
print("Saved result/logs/tabm/tabm_selected_configs.csv")


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
        f"result/tabm/tabm_final_fold{fold}_best.pt"
    )

    print(f"Fold {fold}  AUC={fold_auc:.4f}  AP={fold_ap:.4f}  best_epoch={res['best_epoch']}")

tabm_test_proba = tabm_test_probas.mean(axis=1)
tabm_oof_auc = roc_auc_score(y, tabm_oof_proba)
tabm_oof_ap = average_precision_score(y, tabm_oof_proba)
tabm_prec, tabm_rec, _ = precision_recall_curve(y, tabm_oof_proba)
tabm_pr_auc = auc(tabm_rec, tabm_prec)

tabm_thresholds_search = np.arange(0.05, 0.95, 0.01)
tabm_f1s = [f1_score(y, (tabm_oof_proba >= t).astype(int)) for t in tabm_thresholds_search]
tabm_best_thresh = tabm_thresholds_search[int(np.argmax(tabm_f1s))]
tabm_oof_labels_best = (tabm_oof_proba >= tabm_best_thresh).astype(int)
tabm_oof_precision = precision_score(y, tabm_oof_labels_best)
tabm_oof_recall = recall_score(y, tabm_oof_labels_best)
tabm_oof_f1 = float(np.max(tabm_f1s))

print("\n" + "="*60)
print("TABM SUMMARY")
print("="*60)
print(f"OOF AUC : {tabm_oof_auc:.4f}")
print(f"OOF AP  : {tabm_oof_ap:.4f}")
print(f"OOF PR-AUC  : {tabm_pr_auc:.4f}")
print(f"OOF F1      : {tabm_oof_f1:.4f}  (thresh={tabm_best_thresh:.2f})")
print(f"OOF Prec    : {tabm_oof_precision:.4f}  |  OOF Recall : {tabm_oof_recall:.4f}")
print(f"OOF bot-rate mean  : {tabm_oof_proba.mean():.4f}")
print(f"Test bot-rate mean : {tabm_test_proba.mean():.4f}")

pd.DataFrame(tabm_history_rows).to_csv("result/logs/tabm/tabm_history_all_folds.csv", index=False)
pd.DataFrame(tabm_fold_rows).to_csv("result/logs/tabm/tabm_fold_metrics.csv", index=False)

pd.DataFrame({
    "row_id": np.arange(len(y)),
    "outcome": y,
    "tabm_oof_proba": tabm_oof_proba,
}).to_csv("result/logs/tabm/tabm_oof_predictions.csv", index=False)

tabm_summary_df = pd.DataFrame([{
    "model":              "TabM",
    "oof_auc":            float(tabm_oof_auc),
    "oof_ap":             float(tabm_oof_ap),
    "oof_pr_auc":         float(tabm_pr_auc),
    "oof_precision":      float(tabm_oof_precision),
    "oof_recall":         float(tabm_oof_recall),
    "oof_f1":             float(tabm_oof_f1),
    "best_threshold":     float(tabm_best_thresh),
    "oof_pred_bot_rate":  float(tabm_oof_proba.mean()),
    "test_pred_bot_rate": float(tabm_test_proba.mean()),
}])
tabm_summary_df.to_csv("result/logs/tabm/tabm_summary.csv", index=False)

with open("result/logs/tabm/tabm_best_config.json", "w") as f:
    json.dump(cfg, f, indent=2)

print("Saved result/logs/tabm/tabm_history_all_folds.csv")
print("Saved result/logs/tabm/tabm_fold_metrics.csv")
print("Saved result/logs/tabm/tabm_oof_predictions.csv")
print("Saved result/logs/tabm/tabm_summary.csv")
print("Saved result/logs/tabm/tabm_best_config.json")


# =============================================================================
# 5. TABM EVALUATION PLOTS
# =============================================================================
print("\n" + "="*60)
print("5. TABM EVALUATION PLOTS")
print("="*60)

tabm_color = PALETTE.get("tabm", "#8E44AD")
human_color = PALETTE.get("human", "#2B9EB3")
neutral_color = PALETTE.get("neutral", "#6C757D")

# -------------------------------------------------------------------------
# 5a. ROC + PR curves
# -------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y, tabm_oof_proba)
prec, rec = tabm_prec, tabm_rec

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
save_fig(fig, SAVE_DIR, "tabm_01_roc_pr")

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
save_fig(fig, SAVE_DIR, "tabm_02_confusion_matrix")

# -------------------------------------------------------------------------
# 5c. Training history (mean over folds, common epochs only)
# -------------------------------------------------------------------------
history_path = "result/logs/tabm/tabm_history_all_folds.csv"

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
        save_fig(fig, SAVE_DIR, "tabm_03_training_history")
    else:
        print(f"\nNo valid training history found in {history_path}.")
else:
    print(f"\nFile not found: {history_path}")


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
submission_tabm.to_csv("result/tabm/tabm_submission.csv", index=False)

print(f"tabm_submission.csv  shape={submission_tabm.shape}")
print(submission_tabm.head(10).to_string())
print(f"\nPredicted bot rate (test): {tabm_test_proba.mean():.4f}")

print("\n✓ TabM ensemble pipeline complete.")
print(f"  OOF AUC   : {tabm_oof_auc:.4f}  |  OOF AP : {tabm_oof_ap:.4f}  |  PR-AUC : {tabm_pr_auc:.4f}")
print(f"  Precision : {tabm_oof_precision:.4f}  |  Recall : {tabm_oof_recall:.4f}  |  F1 : {tabm_oof_f1:.4f}  (thresh={tabm_best_thresh:.2f})")
print(f"  Plots     : ./result/tabm/tabm_*.png")
print(f"  Submission: ./result/tabm/tabm_submission.csv")


# =============================================================================
# 7. TABM XAI — INTEGRATED GRADIENTS (NO CAPTUM)
# =============================================================================
print("\n" + "="*60)
print("7. TABM XAI — INTEGRATED GRADIENTS (NO CAPTUM)")
print("="*60)

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
    ckpt_path = f"result/tabm/tabm_final_fold{fold_no}_best.pt"

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
if os.path.exists("result/logs/tabm/tabm_fold_metrics.csv"):
    fold_metrics = pd.read_csv("result/logs/tabm/tabm_fold_metrics.csv")
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

local_df.to_csv("result/logs/tabm/tabm_xai_local_explanation.csv", index=False)
print("Saved result/logs/tabm/tabm_xai_local_explanation.csv")
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
save_fig(fig, SAVE_DIR, "tabm_xai_local_ig")

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

global_df.to_csv("result/logs/tabm/tabm_xai_global_importance.csv", index=False)
print("\nSaved result/logs/tabm/tabm_xai_global_importance.csv")
print("\nTop global features:")
print(global_df.head(20).to_string(index=False))

top_global = global_df.head(20).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top_global["feature"], top_global["mean_abs_attribution"], color=tabm_color)
ax.set_title("TabM Global Feature Importance (mean |Integrated Gradients|)")
ax.set_xlabel("Mean absolute attribution")
plt.tight_layout()
save_fig(fig, SAVE_DIR, "tabm_xai_global_ig")

print("\n✓ TabM XAI complete.")
print("Artifacts:")
print("  - result/logs/tabm/tabm_xai_local_explanation.csv")
print("  - result/logs/tabm/tabm_xai_global_importance.csv")
print("  - result/tabm/tabm_xai_local_ig.png")
print("  - result/tabm/tabm_xai_global_ig.png")


# =============================================================================
# 8. MODEL COMPARISON
# =============================================================================

# =============================================================================
# 1. LOAD MODEL SUMMARIES
# =============================================================================
print("\n" + "="*60)
print("1. MODEL COMPARISON")
print("="*60)

summary_frames = []
for path in [
    "result/logs/rf/rf_summary.csv",
    "result/logs/lgbm/lgbm_summary.csv",
    "result/logs/tabm/tabm_summary.csv",
]:
    if os.path.exists(path):
        summary_frames.append(pd.read_csv(path))

comparison_df = pd.concat(summary_frames, ignore_index=True)
comparison_df.to_csv("result/logs/tabm/model_comparison_summary.csv", index=False)

print(comparison_df.to_string(index=False))
print("Saved result/logs/tabm/model_comparison_summary.csv")


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
save_fig(fig, SAVE_DIR, "model_comparison_auc_ap")

best_model_name = comparison_df.sort_values("oof_auc", ascending=False).iloc[0]["model"]
print(f"\nBest model by OOF AUC: {best_model_name}")
