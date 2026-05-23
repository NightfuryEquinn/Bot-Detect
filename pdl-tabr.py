#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# pdl-tabr.py — TabR: Retrieval-Augmented Tabular Model
# =============================================================================

import os
import copy
import math
import random
import numpy as np
import pandas as pd
import warnings

from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report,
    auc, f1_score, precision_score, recall_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import faiss
    TABR_HAS_FAISS = True
except Exception:
    faiss = None
    TABR_HAS_FAISS = False

from utils.shared import PALETTE, configure_plots, save_fig

warnings.filterwarnings("ignore")
configure_plots()

DATA_DIR = "./datasets"
SAVE_DIR = "result/tabr"
LOG_DIR  = "result/logs/tabr"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 42


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

assert torch.cuda.is_available(), "CUDA GPU is required but not available."
DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("FAISS AVAILABLE:", TABR_HAS_FAISS)

TABR_CONFIG = {
    "seed": 42,
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

N_SPLITS = 5


# =============================================================================
# 9. LOAD EXPORTED FEATURES FROM THE PREPROCESSING SECTION
# =============================================================================
TRAIN_FEATURES_PATH = f"{DATA_DIR}/train_features.csv"
TEST_FEATURES_PATH  = f"{DATA_DIR}/test_features.csv"

tabr_train_full = pd.read_csv(TRAIN_FEATURES_PATH)
tabr_test_full  = pd.read_csv(TEST_FEATURES_PATH)

# Drop string metadata columns — not model features; avoids float-conversion errors
_META_DROP = ["payment_account", "address"]
tabr_train_full = tabr_train_full.drop(columns=[c for c in _META_DROP if c in tabr_train_full.columns])
tabr_test_full  = tabr_test_full.drop(columns=[c for c in _META_DROP if c in tabr_test_full.columns])

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

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
labels_full = tabr_train_full[TABR_TARGET_COL].values
tabr_test_df = tabr_test_full.copy().reset_index(drop=True)

oof_preds      = np.zeros(len(tabr_train_full), dtype=np.float32)
test_preds_acc = np.zeros(len(tabr_test_df), dtype=np.float32)
fold_aucs, fold_aps = [], []

print("Full training set:", tabr_train_full.shape)
print("Test set         :", tabr_test_df.shape)


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

# tabr_arrays is prepared inside the K-fold loop below.


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


# =============================================================================
# 12. TORCH TENSORS FOR TABR
# =============================================================================
def to_tensor_or_none(array, dtype, device):
    if array is None:
        return None
    return torch.as_tensor(array, dtype=dtype, device=device)

# Tensors are created inside the K-fold loop below.


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

for fold, (tr_idx, val_idx) in enumerate(skf.split(tabr_train_full, labels_full), 1):
    print(f"\n{'─'*50}\n  FOLD {fold}/{N_SPLITS}\n{'─'*50}")

    tabr_train_df = tabr_train_full.iloc[tr_idx].reset_index(drop=True)
    tabr_val_df   = tabr_train_full.iloc[val_idx].reset_index(drop=True)

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

    X_num_train_t = to_tensor_or_none(tabr_arrays["X_num_train"], torch.float32, DEVICE)
    X_num_val_t   = to_tensor_or_none(tabr_arrays["X_num_val"],   torch.float32, DEVICE)
    X_num_test_t  = to_tensor_or_none(tabr_arrays["X_num_test"],  torch.float32, DEVICE)

    X_cat_train_t = to_tensor_or_none(tabr_arrays["X_cat_train"], torch.long, DEVICE)
    X_cat_val_t   = to_tensor_or_none(tabr_arrays["X_cat_val"],   torch.long, DEVICE)
    X_cat_test_t  = to_tensor_or_none(tabr_arrays["X_cat_test"],  torch.long, DEVICE)

    y_train_long_t  = torch.as_tensor(tabr_arrays["y_train"], dtype=torch.long,  device=DEVICE)
    y_train_float_t = y_train_long_t.float()
    y_val_np        = tabr_arrays["y_val"].astype(np.int64)

    n_num_features_tabr   = 0 if X_num_train_t is None else X_num_train_t.shape[1]
    candidate_train_ids_t = torch.arange(len(y_train_long_t), device=DEVICE)

    print("n_num_features_tabr:", n_num_features_tabr)
    print("cat_cardinalities  :", tabr_arrays["cat_cardinalities"])
    print("n_train:", len(y_train_long_t))
    print("n_val  :", len(y_val_np))

    tabr_model, tabr_history = train_tabr()

    # OOF predictions (memory bank = this fold's training set)
    fold_val_pred = tabr_predict_proba(
        model=tabr_model,
        X_num=X_num_val_t,
        X_cat=X_cat_val_t,
        candidate_num=X_num_train_t,
        candidate_cat=X_cat_train_t,
        candidate_y=y_train_long_t,
        context_size=TABR_CONFIG["context_size"],
        batch_size=TABR_CONFIG["eval_batch_size"],
    )
    oof_preds[val_idx] = fold_val_pred

    # Test predictions averaged across folds
    fold_test_pred = tabr_predict_proba(
        model=tabr_model,
        X_num=X_num_test_t,
        X_cat=X_cat_test_t,
        candidate_num=X_num_train_t,
        candidate_cat=X_cat_train_t,
        candidate_y=y_train_long_t,
        context_size=TABR_CONFIG["context_size"],
        batch_size=TABR_CONFIG["eval_batch_size"],
    )
    test_preds_acc += fold_test_pred / N_SPLITS

    fold_auc = roc_auc_score(tabr_val_df[TABR_TARGET_COL].values, fold_val_pred)
    fold_ap  = average_precision_score(tabr_val_df[TABR_TARGET_COL].values, fold_val_pred)
    fold_aucs.append(fold_auc)
    fold_aps.append(fold_ap)
    print(f"  Fold {fold} — AUC: {fold_auc:.4f} | AP: {fold_ap:.4f}")

    fold_history = tabr_history.copy()
    fold_history["fold"] = fold
    fold_history.to_csv(f"result/logs/tabr/tabr_history_fold{fold}.csv", index=False)

    torch.save(
        {
            "model_state_dict": tabr_model.state_dict(),
            "config": TABR_CONFIG,
            "num_cols": tabr_num_cols,
            "cat_cols": tabr_cat_cols,
            "cat_cardinalities": tabr_arrays["cat_cardinalities"],
            "fold": fold,
        },
        f"result/tabr/tabr_final_fold{fold}_best.pt",
    )

    # Preserve last fold's data/tensors for XAI
    xai_val_df   = tabr_val_df
    xai_val_pred = fold_val_pred
    xai_y_val_np = y_val_np
    xai_val_idx  = val_idx

# ─── Post-loop: OOF metrics ───────────────────────────────────────────────────
tabr_oof_auc = roc_auc_score(labels_full, oof_preds)
tabr_oof_ap  = average_precision_score(labels_full, oof_preds)
tabr_prec, tabr_rec, _ = precision_recall_curve(labels_full, oof_preds)
tabr_pr_auc = auc(tabr_rec, tabr_prec)

tabr_thresholds_search = np.arange(0.05, 0.95, 0.01)
tabr_f1s = [f1_score(labels_full, (oof_preds >= t).astype(int)) for t in tabr_thresholds_search]
tabr_best_thresh = tabr_thresholds_search[int(np.argmax(tabr_f1s))]
tabr_oof_labels = (oof_preds >= tabr_best_thresh).astype(int)
tabr_oof_precision = precision_score(labels_full, tabr_oof_labels)
tabr_oof_recall = recall_score(labels_full, tabr_oof_labels)
tabr_oof_f1 = float(np.max(tabr_f1s))

print(f"\nCV AUC  : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"CV AP   : {np.mean(fold_aps):.4f} ± {np.std(fold_aps):.4f}")
print(f"OOF AUC : {tabr_oof_auc:.4f}")
print(f"OOF AP  : {tabr_oof_ap:.4f}")
print(f"OOF PR-AUC  : {tabr_pr_auc:.4f}")
print(f"OOF F1      : {tabr_oof_f1:.4f}  (thresh={tabr_best_thresh:.2f})")
print(f"OOF Prec    : {tabr_oof_precision:.4f}  |  OOF Recall : {tabr_oof_recall:.4f}")

# Aliases used by downstream sections (plots, XAI)
tabr_val_pred  = oof_preds
tabr_test_pred = test_preds_acc
y_val_np       = labels_full


# =============================================================================
# 14. TABR OOF RESULT + TEST PREDICTION
# =============================================================================
tabr_val_ll = log_loss(y_val_np, np.clip(tabr_val_pred, 1e-7, 1 - 1e-7))

print("OOF AUC     :", round(tabr_oof_auc, 6))
print("OOF AP      :", round(tabr_oof_ap, 6))
print("OOF LogLoss :", round(tabr_val_ll, 6))

tabr_val_out = tabr_train_full[[TABR_ID_COL, TABR_TARGET_COL]].copy()
tabr_val_out["tabr_prob"] = oof_preds

tabr_test_out = tabr_test_df[[TABR_ID_COL]].copy()
tabr_test_out["prediction"] = tabr_test_pred


# =============================================================================
# 15. TABR EVALUATION PLOTS
# =============================================================================
print("" + "="*60)
print("15. TABR EVALUATION PLOTS")
print("="*60)

tabr_color = PALETTE.get("tabr", "#8E44AD")
human_color = PALETTE.get("human", "#2B9EB3")
bot_color = PALETTE.get("bot", "#E84545")
neutral_color = PALETTE.get("neutral", "#6C757D")

# -------------------------------------------------------------------------
# 15a. ROC + PR curves
# -------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_val_np, tabr_val_pred)
prec, rec = tabr_prec, tabr_rec

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"TabR — OOF (5-fold CV)  AUC={tabr_oof_auc:.4f}  AP={tabr_oof_ap:.4f}", fontsize=14)

# ROC
ax = axes[0]
ax.plot(fpr, tpr, color=tabr_color, lw=2, label=f"ROC (AUC={tabr_oof_auc:.4f})")
ax.plot([0, 1], [0, 1], "--", color=neutral_color, lw=1)
ax.fill_between(fpr, tpr, alpha=0.12, color=tabr_color)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

# PR
ax = axes[1]
ax.plot(rec, prec, color=bot_color, lw=2, label=f"PR (AP={tabr_oof_ap:.4f})")
ax.fill_between(rec, prec, alpha=0.12, color=bot_color)
ax.axhline(y_val_np.mean(), color=neutral_color, lw=1, linestyle="--", label="Chance")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve")
ax.legend()

plt.tight_layout()
save_fig(fig, SAVE_DIR, "tabr_01_roc_pr")

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
save_fig(fig, SAVE_DIR, "tabr_02_confusion_matrix")

# -------------------------------------------------------------------------
# 15c. Score distributions by class
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(tabr_val_pred[y_val_np == 0], bins=30, color=human_color, alpha=0.55, label="Human", ax=ax)
sns.histplot(tabr_val_pred[y_val_np == 1], bins=30, color=bot_color, alpha=0.55, label="Bot", ax=ax)
ax.set_xlabel("Predicted bot probability")
ax.set_ylabel("Count")
ax.set_title("TabR OOF Score Distribution")
ax.legend()
plt.tight_layout()
save_fig(fig, SAVE_DIR, "tabr_03_score_distribution")

# -------------------------------------------------------------------------
# 15d. Training history
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"TabR Training History (Fold {N_SPLITS})", fontsize=14)

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
save_fig(fig, SAVE_DIR, "tabr_04_training_history")


# =============================================================================
# 16. TABR XAI — GLOBAL + LOCAL + RETRIEVAL EXPLANATION
# =============================================================================
print("\n" + "="*60)
print("16. TABR XAI — GLOBAL + LOCAL + RETRIEVAL EXPLANATION")
print("="*60)

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

base_auc = roc_auc_score(xai_y_val_np, xai_val_pred)
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
        perm_auc = roc_auc_score(xai_y_val_np, perm_pred)
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
        perm_auc = roc_auc_score(xai_y_val_np, perm_pred)
        perm_rows.append({
            "feature": col,
            "feature_type": "categorical",
            "baseline_auc": base_auc,
            "permuted_auc": perm_auc,
            "auc_drop": base_auc - perm_auc,
        })

perm_df = pd.DataFrame(perm_rows).sort_values("auc_drop", ascending=False)
perm_df.to_csv("result/logs/tabr/tabr_xai_global_importance.csv", index=False)
print("Saved result/logs/tabr/tabr_xai_global_importance.csv")
print("\nTop global features:")
print(perm_df.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 7))
top_perm = perm_df.head(20).iloc[::-1]
colors = [tabr_color if t == "numerical" else bot_color for t in top_perm["feature_type"]]
ax.barh(top_perm["feature"], top_perm["auc_drop"], color=colors)
ax.set_title("TabR Global Feature Importance (Permutation AUC drop)")
ax.set_xlabel("AUC drop after shuffling feature")
plt.tight_layout()
save_fig(fig, SAVE_DIR, "tabr_xai_global_permutation")

# -------------------------------------------------------------------------
# 16b. LOCAL XAI — single-feature ablation for one high-risk validation row
# -------------------------------------------------------------------------
print("\nComputing local explanation for one high-risk validation example...")

local_idx = int(np.argmax(xai_val_pred))
global_row_id = int(xai_val_idx[local_idx])
base_prob = float(xai_val_pred[local_idx])
true_label = int(xai_y_val_np[local_idx])

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
local_df.to_csv("result/logs/tabr/tabr_xai_local_explanation.csv", index=False)

print(f"Explained validation row: local_idx={local_idx}, global_row_id={global_row_id}")
print(f"True label: {true_label} | Predicted bot probability: {base_prob:.4f}")
print("Saved result/logs/tabr/tabr_xai_local_explanation.csv")
print("\nTop local features:")
print(local_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
top_local = local_df.head(15).iloc[::-1]
colors = [tabr_color if x > 0 else human_color for x in top_local["contribution"]]
ax.barh(top_local["feature"], top_local["contribution"], color=colors)
ax.set_title("TabR Local Explanation (single-feature ablation)")
ax.set_xlabel("Change in predicted bot probability")
plt.tight_layout()
save_fig(fig, SAVE_DIR, "tabr_xai_local_ablation")

# -------------------------------------------------------------------------
# 16c. Retrieval explanation — inspect top retrieved neighbors for the local row
# -------------------------------------------------------------------------
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
neighbor_df.to_csv("result/logs/tabr/tabr_xai_retrieved_neighbors.csv", index=False)

print("\nTop retrieved neighbors for the explained row:")
print(neighbor_df.to_string(index=False))
print("Saved result/logs/tabr/tabr_xai_retrieved_neighbors.csv")

fig, ax = plt.subplots(figsize=(8, 5))
plot_df = neighbor_df.iloc[::-1]
bar_colors = [bot_color if yy == 1 else human_color for yy in plot_df[TABR_TARGET_COL]]
ax.barh(plot_df["rank"].astype(str), plot_df["neighbor_weight"], color=bar_colors)
ax.set_title("Top Retrieved Neighbors for Local Explanation")
ax.set_xlabel("Neighbor weight")
ax.set_ylabel("Neighbor rank (1 = closest)")
plt.tight_layout()
save_fig(fig, SAVE_DIR, "tabr_xai_retrieved_neighbors")

print("\n✓ TabR XAI complete.")
print("Artifacts:")
print("  - result/logs/tabr/tabr_xai_global_importance.csv")
print("  - result/logs/tabr/tabr_xai_local_explanation.csv")
print("  - result/logs/tabr/tabr_xai_retrieved_neighbors.csv")
print("  - result/tabr/tabr_xai_global_permutation.png")
print("  - result/tabr/tabr_xai_local_ablation.png")
print("  - result/tabr/tabr_xai_retrieved_neighbors.png")


# =============================================================================
# 17. SAVE TABR OUTPUTS
# =============================================================================
TABR_VAL_PATH        = "result/tabr/tabr_val_predictions.csv"
TABR_TEST_PATH       = "result/tabr/tabr_test_predictions.csv"
TABR_SUBMISSION_PATH = "result/tabr/tabr_submission.csv"
TABR_MODEL_PATH      = "result/tabr/tabr_model.pt"

tabr_val_out.to_csv(TABR_VAL_PATH, index=False)
tabr_test_out.to_csv(TABR_TEST_PATH, index=False)
tabr_test_out.to_csv(TABR_SUBMISSION_PATH, index=False)
# Save the last fold's model as the canonical checkpoint
torch.save(
    {
        "model_state_dict": tabr_model.state_dict(),
        "config": TABR_CONFIG,
        "num_cols": tabr_num_cols,
        "cat_cols": tabr_cat_cols,
        "cat_cardinalities": tabr_arrays["cat_cardinalities"],
        "fold": N_SPLITS,
    },
    TABR_MODEL_PATH,
)

print("Saved:", TABR_VAL_PATH)
print("Saved:", TABR_TEST_PATH)
print("Saved:", TABR_SUBMISSION_PATH)
print("Saved:", TABR_MODEL_PATH)

pd.DataFrame([{
    "model":          "TabR",
    "oof_auc":        float(tabr_oof_auc),
    "oof_ap":         float(tabr_oof_ap),
    "oof_pr_auc":     float(tabr_pr_auc),
    "oof_precision":  float(tabr_oof_precision),
    "oof_recall":     float(tabr_oof_recall),
    "oof_f1":         float(tabr_oof_f1),
    "best_threshold": float(tabr_best_thresh),
    "cv_auc_mean":    float(np.mean(fold_aucs)),
    "cv_auc_std":     float(np.std(fold_aucs)),
    "cv_ap_mean":     float(np.mean(fold_aps)),
}]).to_csv("result/logs/tabr/tabr_summary.csv", index=False)
print("Saved: result/logs/tabr/tabr_summary.csv")
