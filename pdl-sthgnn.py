import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, auc, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import HypergraphConv
import copy

from utils.shared import DualLogger, setup_logger, FocalLoss, PALETTE, configure_plots, save_fig

configure_plots()

# =============================================================================
# 1. SETUP & MULTI-VIEW HYPERGRAPH CONSTRUCTION (OOM-SAFE)
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_logger("./result/logs/sthgnn", f"ST_HGNN_{timestamp}")

DATA_PATH = "./datasets"
assert torch.cuda.is_available(), "CUDA GPU is required but not available."
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load datasets
train_df = pd.read_csv(f"{DATA_PATH}/train_features.csv")
test_df = pd.read_csv(f"{DATA_PATH}/test_features.csv")
raw_bids = pd.read_csv(f"{DATA_PATH}/bids.csv")

# Global Indexing
all_bidders = pd.concat([train_df['bidder_id'], test_df['bidder_id']]).unique()
bidder_to_idx = {bidder: i for i, bidder in enumerate(all_bidders)}
num_nodes = len(all_bidders)

# Feature Preparation
full_features = pd.concat([train_df, test_df]).set_index('bidder_id').reindex(all_bidders)
X_raw = full_features.drop(columns=['payment_account', 'address', 'outcome'], errors='ignore').fillna(0)

# Extract Temporal Indices
temporal_cols = []
for i in range(1, 6):
    temporal_cols.extend([f'burst_max_strip{i}', f'burst_mean_strip{i}', f'burst_ratio_strip{i}'])
temporal_indices = [X_raw.columns.get_loc(col) for col in temporal_cols if col in X_raw.columns]
print(f"Detected {len(temporal_indices)} temporal features for 5 time strips.")

scaler = StandardScaler()
x = torch.from_numpy(scaler.fit_transform(X_raw)).float().to(device)

# Hyperedge Construction (Auction + IP)
unique_auctions = raw_bids['auction'].unique()
auction_to_idx = {auc: i for i, auc in enumerate(unique_auctions)}
unique_ips = raw_bids['ip'].unique()
ip_to_idx = {ip: i + len(unique_auctions) for i, ip in enumerate(unique_ips)}
unique_devices = raw_bids['device'].unique()
dev_to_idx = {dev: i + len(unique_auctions) + len(unique_ips) for i, dev in enumerate(unique_devices)}

valid_bids = raw_bids[raw_bids['bidder_id'].isin(bidder_to_idx)].copy()
nodes = np.concatenate([valid_bids['bidder_id'].map(bidder_to_idx).values] * 3)  # x3 now
hyperedges = np.concatenate([
    valid_bids['auction'].map(auction_to_idx).values,
    valid_bids['ip'].map(ip_to_idx).values,
    valid_bids['device'].map(dev_to_idx).values  # Added device edges
])
edge_index = torch.from_numpy(np.stack([nodes, hyperedges])).long().to(device)

edge_counts = torch.bincount(edge_index[1])
hyperedge_weight = 1.0 / torch.log(edge_counts.float() + 1.5)
hyperedge_weight = hyperedge_weight / hyperedge_weight.mean()
hyperedge_weight = hyperedge_weight.to(device)

print(f"Hyperedge weights calculated for {len(hyperedge_weight)} unique auctions/IPs/Devices.")
print(f"Hyperedge weights normalized. Mean: {hyperedge_weight.mean():.4f}")

# Labels
labels = np.full(num_nodes, -1)
for _, row in train_df.iterrows():
    labels[bidder_to_idx[row['bidder_id']]] = int(row['outcome'])
y = torch.tensor(labels, dtype=torch.long).to(device)


# =============================================================================
# 2. ST-HGNN NETWORK
# =============================================================================
class STHGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_indices=None, seq_len=5):
        super().__init__()
        self.temporal_indices = temporal_indices if temporal_indices is not None else []
        self.static_indices = [i for i in range(in_channels) if i not in self.temporal_indices]
        self.seq_len = seq_len
        self.temp_dim = len(self.temporal_indices) // seq_len if len(self.temporal_indices) > 0 else 0

        # Two-layer spatial branch for deeper community detection
        self.spatial_conv1 = HypergraphConv(in_channels, hidden_channels)
        self.spatial_conv2 = HypergraphConv(hidden_channels, hidden_channels)

        # Stability layers
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)

        # Temporal branch (Transformer)
        if self.temp_dim > 0:
            self.temp_proj = nn.Linear(self.temp_dim, hidden_channels)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=4,
                dim_feedforward=hidden_channels * 2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Behavioral branch
        static_dim = len(self.static_indices) if len(self.static_indices) > 0 else in_channels
        self.behavioral_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        # Squeeze-and-Excitation Gating
        self.gate_fc1 = nn.Linear(hidden_channels * 2, hidden_channels // 2)
        self.gate_fc2 = nn.Linear(hidden_channels // 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_weight):
        # Hyperedge Dropout (10%) during training to improve generalization
        if self.training:
            num_edges = edge_index.size(1)
            mask = torch.bernoulli(torch.full((num_edges,), 0.8)).to(x.device).bool()
            curr_edge_index = edge_index[:, mask]
        else:
            curr_edge_index = edge_index

        # View 1: Structural (Spatial)
        z1 = F.leaky_relu(self.ln1(self.spatial_conv1(x, curr_edge_index, hyperedge_weight=edge_weight)), 0.2)
        z2 = F.leaky_relu(self.ln2(self.spatial_conv2(z1, curr_edge_index, hyperedge_weight=edge_weight)), 0.2)
        z_spatial = z1 + z2

        # View 2: Behavioral (Static Features)
        x_static = x[:, self.static_indices]
        z_behavioral = self.behavioral_mlp(x_static)

        # View 3: Temporal Sequence Modeling
        if self.temp_dim > 0:
            x_temp = x[:, self.temporal_indices].view(-1, self.seq_len, self.temp_dim)
            x_temp_emb = self.temp_proj(x_temp)
            z_temp_seq = self.transformer(x_temp_emb)
            z_temporal = z_temp_seq.mean(dim=1)  # Pool across time

            # Combine individual behavioral traits with temporal evolution
            z_bt = z_behavioral + z_temporal
        else:
            z_bt = z_behavioral

        # Dynamic SE-Gate logic
        combined = torch.cat([z_spatial, z_bt], dim=1)
        se = F.relu(self.gate_fc1(combined))
        g = torch.sigmoid(self.gate_fc2(se))
        z_fused = g * z_spatial + (1 - g) * z_bt

        return self.classifier(z_fused)


# =============================================================================
# 3. K-FOLD TRAINING ENGINE
# =============================================================================
def train_kfold(k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labeled_indices = np.where(labels != -1)[0]
    labeled_y = labels[labeled_indices]

    fold_aucs = []
    fold_aps = []
    oof_probs = np.zeros(len(labeled_indices))
    test_probs_acc = np.zeros(num_nodes)

    fold_metrics = []
    all_fold_history = []

    criterion = FocalLoss(alpha=0.85, gamma=2.5)

    print(f"Starting {k}-Fold Cross-Validation with Checkpointing...")

    EARLY_STOPPING_PATIENCE = 25

    for fold, (t_idx, v_idx) in enumerate(skf.split(labeled_indices, labeled_y)):
        model = STHGNN(x.shape[1], 64, temporal_indices).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

        f_train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        f_val_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        f_train_mask[labeled_indices[t_idx]] = True
        f_val_mask[labeled_indices[v_idx]] = True

        best_auc = 0
        patience_counter = 0
        best_model_state = None

        print(f"\n--- FOLD {fold + 1} ---")

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index, hyperedge_weight).squeeze()
            loss = criterion(out[f_train_mask], y[f_train_mask].float())
            loss.backward()
            optimizer.step()

            # Validation for checkpointing
            model.eval()
            with torch.no_grad():
                val_out = model(x, edge_index, hyperedge_weight).squeeze()
                v_probs_epoch = torch.sigmoid(val_out[f_val_mask]).cpu().numpy()
                v_labels_epoch = y[f_val_mask].cpu().numpy()
                val_auc = roc_auc_score(v_labels_epoch, v_probs_epoch)

                all_fold_history.append({
                    "fold": fold + 1,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_auc": float(val_auc)
                })

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                scheduler.step()

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} (Best: {best_auc:.4f})")

        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            full_logits = model(x, edge_index, hyperedge_weight).squeeze()
            full_probs = torch.sigmoid(full_logits).cpu().numpy()

            v_probs = full_probs[labeled_indices[v_idx]]
            v_labels = labeled_y[v_idx]

            oof_probs[v_idx] = v_probs
            fold_aucs.append(roc_auc_score(v_labels, v_probs))
            fold_aps.append(average_precision_score(v_labels, v_probs))
            test_probs_acc += full_probs / k

        fold_metrics.append({
            "fold": fold + 1,
            "auc": fold_aucs[-1],
            "ap": fold_aps[-1]
        })

        print(f"Fold {fold + 1} Finished. Best Fold AUC: {fold_aucs[-1]:.4f}")

    # =============================================================================
    # 4. FINAL SUMMARY
    # =============================================================================
    oof_auc = roc_auc_score(labeled_y, oof_probs)
    oof_ap = average_precision_score(labeled_y, oof_probs)

    prec, rec, thresholds = precision_recall_curve(labeled_y, oof_probs)
    f1_scores = (2 * prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx]
    pr_auc = auc(rec, prec)
    precision_at_thresh = float(prec[best_idx])
    recall_at_thresh = float(rec[best_idx])

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (ST-HGNN)")
    print("=" * 60)
    print(f"  Model                    : ST-HGNN (Spatio-Temporal Hypergraph)")
    print(f"  CV Folds                 : {k}")
    print(f"  CV AUC (mean)            : {np.mean(fold_aucs):.4f}")
    print(f"  CV AUC (std)             : {np.std(fold_aucs):.4f}")
    print(f"  CV AP  (mean)            : {np.mean(fold_aps):.4f}")
    print(f"  OOF AUC                  : {oof_auc:.4f}")
    print(f"  OOF AP                   : {oof_ap:.4f}")
    print(f"  OOF PR-AUC               : {pr_auc:.4f}")
    print(f"  OOF Precision            : {precision_at_thresh:.4f}")
    print(f"  OOF Recall               : {recall_at_thresh:.4f}")
    print(f"  OOF F1                   : {best_f1:.4f}")
    print(f"  Best Threshold           : {best_thresh:.4f}")
    print("=" * 60 + "\n")

    print(classification_report(labeled_y, (oof_probs > best_thresh).astype(int), target_names=['Human', 'Robot']))

    os.makedirs("result/logs/sthgnn", exist_ok=True)
    pd.DataFrame([{
        "model":          "ST-HGNN",
        "oof_auc":        float(oof_auc),
        "oof_ap":         float(oof_ap),
        "oof_pr_auc":     float(pr_auc),
        "oof_precision":  float(precision_at_thresh),
        "oof_recall":     float(recall_at_thresh),
        "oof_f1":         float(best_f1),
        "best_threshold": float(best_thresh),
        "cv_auc_mean":    float(np.mean(fold_aucs)),
        "cv_auc_std":     float(np.std(fold_aucs)),
        "cv_ap_mean":     float(np.mean(fold_aps)),
    }]).to_csv("result/logs/sthgnn/sthgnn_summary.csv", index=False)

    return test_probs_acc, oof_probs, labeled_y, best_thresh, pd.DataFrame(fold_metrics), pd.DataFrame(all_fold_history), model, v_idx

# =============================================================================
# 4. EXECUTION, PLOTTING & EXPORT PIPELINE
# =============================================================================
MODEL_NAME = "sthgnn"
SAVE_DIR = f"result/{MODEL_NAME}"
LOGS_DIR = f"result/logs/{MODEL_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

final_test_probs, oof_probs, labeled_y_true, best_thresh, df_fold_metrics, df_history, last_model, last_v_idx = train_kfold(
    k=5)

print("\nSaving Experiment Artifacts...")

df_fold_metrics.to_csv(f"{LOGS_DIR}/{MODEL_NAME}_fold_metrics.csv", index=False)
df_history.to_csv(f"{LOGS_DIR}/{MODEL_NAME}_history_all_folds.csv", index=False)

labeled_indices = np.where(labels != -1)[0]
oof_bidders = [all_bidders[idx] for idx in labeled_indices]
pd.DataFrame({
    "bidder_id": oof_bidders,
    "outcome": labeled_y_true,
    f"{MODEL_NAME}_oof_proba": oof_probs,
}).to_csv(f"{LOGS_DIR}/{MODEL_NAME}_oof_predictions.csv", index=False)

test_bidders = test_df['bidder_id'].values
test_preds = [final_test_probs[bidder_to_idx[b_id]] for b_id in test_bidders]
pd.DataFrame({
    'bidder_id': test_bidders,
    'prediction': test_preds
}).to_csv(f"{SAVE_DIR}/{MODEL_NAME}_submission.csv", index=False)

print(f"Saved all CSV artifacts to {LOGS_DIR}/ and {SAVE_DIR}/")

print("\nGenerating Evaluation Plots...")
oof_labels_pred = (oof_probs >= best_thresh).astype(int)
fpr, tpr, _ = roc_curve(labeled_y_true, oof_probs)
prec, rec, _ = precision_recall_curve(labeled_y_true, oof_probs)
oof_auc = roc_auc_score(labeled_y_true, oof_probs)
oof_ap = average_precision_score(labeled_y_true, oof_probs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"{MODEL_NAME.upper()} — OOF Performance", fontsize=14, y=1.01)

axes[0].plot(fpr, tpr, color=PALETTE.get("bot", "#E84545"), lw=2, label=f"ROC (AUC={oof_auc:.4f})")
axes[0].plot([0, 1], [0, 1], color="#4A4D5A", ls="--", lw=1)
axes[0].set_xlabel("False Positive Rate");
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve");
axes[0].legend()

axes[1].plot(rec, prec, color=PALETTE.get("human", "#2B9EB3"), lw=2, label=f"PR (AP={oof_ap:.4f})")
axes[1].set_xlabel("Recall");
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve");
axes[1].legend()

plt.tight_layout()
save_fig(fig, SAVE_DIR, f"{MODEL_NAME}_01_roc_pr")

fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(labeled_y_true, oof_labels_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"],
            ax=ax, linewidths=0.5)
ax.set_xlabel("Predicted");
ax.set_ylabel("Actual")
ax.set_title(f"OOF Confusion Matrix (thresh={best_thresh:.2f})")
plt.tight_layout()
save_fig(fig, SAVE_DIR, f"{MODEL_NAME}_02_confusion_matrix")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(oof_probs[labeled_y_true == 0], bins=50, alpha=0.7, color=PALETTE.get("human", "#2B9EB3"), label="Human (0)")
ax.hist(oof_probs[labeled_y_true == 1], bins=50, alpha=0.7, color=PALETTE.get("bot", "#E84545"), label="Bot (1)")
ax.axvline(best_thresh, color=PALETTE.get("accent", "#F39C12"), ls="--", lw=2, label=f"Threshold={best_thresh:.2f}")
ax.set_xlabel("Predicted Probability (Bot)");
ax.set_ylabel("Count")
ax.set_title("OOF Score Distribution");
ax.legend()
plt.tight_layout()
save_fig(fig, SAVE_DIR, f"{MODEL_NAME}_03_score_distribution")

print(f"Plots successfully saved to {SAVE_DIR}/")

# =============================================================================
# 5. EXPLAINABILITY (XAI) — PERMUTATION IMPORTANCE
# =============================================================================
print("\n" + "=" * 60)
print("5. EXPLAINABILITY (XAI) — PERMUTATION IMPORTANCE")
print("=" * 60)
print("Computing permutation importance on the final fold validation set...")

last_model.eval()
val_mask_indices = labeled_indices[last_v_idx]

# 1. Get Baseline AUC
with torch.no_grad():
    base_logits = last_model(x, edge_index, hyperedge_weight).squeeze()
    base_probs = torch.sigmoid(base_logits)[val_mask_indices].cpu().numpy()
    base_auc_xai = roc_auc_score(labeled_y_true[last_v_idx], base_probs)

perm_rows = []
feature_names = X_raw.columns.tolist()

# 2. Iterate and Shuffle Each Feature
for j, feat_name in enumerate(feature_names):
    x_perm = x.clone()
    # Shuffle the j-th feature only for the validation nodes to prevent graph leakage
    shuffled_vals = x_perm[val_mask_indices, j][torch.randperm(len(val_mask_indices))]
    x_perm[val_mask_indices, j] = shuffled_vals

    with torch.no_grad():
        perm_logits = last_model(x_perm, edge_index, hyperedge_weight).squeeze()
        perm_probs = torch.sigmoid(perm_logits)[val_mask_indices].cpu().numpy()
        perm_auc = roc_auc_score(labeled_y_true[last_v_idx], perm_probs)

    auc_drop = base_auc_xai - perm_auc
    perm_rows.append({
        "feature": feat_name,
        "baseline_auc": base_auc_xai,
        "permuted_auc": perm_auc,
        "auc_drop": auc_drop
    })

# 3. Save XAI CSV
df_perm = pd.DataFrame(perm_rows).sort_values("auc_drop", ascending=False)
df_perm.to_csv(f"{LOGS_DIR}/{MODEL_NAME}_xai_global_importance.csv", index=False)
print(f"\nTop global features:")
print(df_perm.head(15).to_string(index=False))
print(f"\nSaved XAI results to {LOGS_DIR}/{MODEL_NAME}_xai_global_importance.csv")

# 4. Generate XAI Plot
top_perm = df_perm.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))

# Highlight negative drops (noise) vs positive drops (important features)
colors = [PALETTE.get("sthgnn", "#8E44AD") if val > 0 else PALETTE.get("neutral", "#6C757D") for val in
          top_perm["auc_drop"]]
ax.barh(top_perm["feature"], top_perm["auc_drop"], color=colors)

ax.set_title(f"{MODEL_NAME.upper()} Global Feature Importance (Permutation)")
ax.set_xlabel("AUC Drop after shuffling feature")
plt.tight_layout()
save_fig(fig, SAVE_DIR, f"{MODEL_NAME}_04_xai_importance")
print(f"Saved XAI plot to {SAVE_DIR}/{MODEL_NAME}_04_xai_importance.png")
