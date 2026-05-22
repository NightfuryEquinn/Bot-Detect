import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from torch_geometric.nn import HypergraphConv
import copy


# =============================================================================
# 1. SETUP & MULTI-VIEW DATA CONSTRUCTION
# =============================================================================
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logger(output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, f"log_{file_name}.txt")
    sys.stdout = DualLogger(log_file_path)
    return log_file_path


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_logger("./result", f"ST_MV_HGNN_{timestamp}")

DATA_PATH = "./datasets"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# HIMVH MULTI-VIEW HYPEREDGE CONSTRUCTION
valid_bids = raw_bids[raw_bids['bidder_id'].isin(bidder_to_idx)].copy()
bidders_mapped = valid_bids['bidder_id'].map(bidder_to_idx).values

# View 1: Auctions
unique_auctions = raw_bids['auction'].unique()
auction_to_idx = {auc: i for i, auc in enumerate(unique_auctions)}
auction_mapped = valid_bids['auction'].map(auction_to_idx).values
edge_index_auc = torch.from_numpy(np.stack([bidders_mapped, auction_mapped])).long().to(device)

# View 2: IPs
unique_ips = raw_bids['ip'].unique()
ip_to_idx = {ip: i for i, ip in enumerate(unique_ips)}
ip_mapped = valid_bids['ip'].map(ip_to_idx).values
edge_index_ip = torch.from_numpy(np.stack([bidders_mapped, ip_mapped])).long().to(device)

# View 3: Devices
unique_devices = raw_bids['device'].unique()
dev_to_idx = {dev: i for i, dev in enumerate(unique_devices)}
dev_mapped = valid_bids['device'].map(dev_to_idx).values
edge_index_dev = torch.from_numpy(np.stack([bidders_mapped, dev_mapped])).long().to(device)


def get_hyperedge_weights(edge_index):
    edge_counts = torch.bincount(edge_index[1])
    weights = 1.0 / torch.log(edge_counts.float() + 1.5)
    return (weights / weights.mean()).to(device)


weight_auc = get_hyperedge_weights(edge_index_auc)
weight_ip = get_hyperedge_weights(edge_index_ip)
weight_dev = get_hyperedge_weights(edge_index_dev)

print("Multi-View Hypergraphs constructed successfully.")

# Labels
labels = np.full(num_nodes, -1)
for _, row in train_df.iterrows():
    labels[bidder_to_idx[row['bidder_id']]] = int(row['outcome'])
y = torch.tensor(labels, dtype=torch.long).to(device)


# =============================================================================
# 2. ST-MV-HGNN NETWORK ARCHITECTURE
# =============================================================================
class STMVHGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_indices=None, seq_len=5):
        super().__init__()
        self.temporal_indices = temporal_indices if temporal_indices is not None else []
        self.static_indices = [i for i in range(in_channels) if i not in self.temporal_indices]
        self.seq_len = seq_len
        self.temp_dim = len(self.temporal_indices) // seq_len if len(self.temporal_indices) > 0 else 0

        # MODULE 1: Cross-View Inconsistency Perception (Scene Conflict)
        # Separate spatial branches for each view
        self.auction_conv = HypergraphConv(in_channels, hidden_channels)
        self.ip_conv = HypergraphConv(in_channels, hidden_channels)
        self.device_conv = HypergraphConv(in_channels, hidden_channels)

        self.view_attn = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # MODULE 2: Temporal Sequence Modeling
        if self.temp_dim > 0:
            self.temp_proj = nn.Linear(self.temp_dim, hidden_channels)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=4,
                dim_feedforward=hidden_channels * 2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

            self.temp_combine = nn.Linear(hidden_channels * 2, hidden_channels)

        # Behavioral branch (Static features)
        static_dim = len(self.static_indices) if len(self.static_indices) > 0 else in_channels
        self.behavioral_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        # MODULE 3: Novelty-Aware Logic (Match-Mismatch)
        self.mismatch_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid()
        )

        self.mismatch_alpha = nn.Parameter(torch.tensor(0.5))

        # Final Classifier
        self.gate_fc1 = nn.Linear(hidden_channels * 2, hidden_channels // 2)
        self.gate_fc2 = nn.Linear(hidden_channels // 2, hidden_channels)
        static_dim = len(self.static_indices) if len(self.static_indices) > 0 else in_channels
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + static_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def apply_dropout(self, edge_index):
        if self.training:
            mask = torch.bernoulli(torch.full((edge_index.size(1),), 0.8)).to(edge_index.device).bool()
            return edge_index[:, mask]
        return edge_index

    def forward(self, x, e_auc, e_ip, e_dev, w_auc, w_ip, w_dev):
        # 1. Multi-View Spatial Encoding (Detect Scene Conflict)
        e_auc = self.apply_dropout(e_auc)
        e_ip = self.apply_dropout(e_ip)
        e_dev = self.apply_dropout(e_dev)

        z_auc = F.leaky_relu(self.auction_conv(x, e_auc, hyperedge_weight=w_auc), 0.2)
        z_ip = F.leaky_relu(self.ip_conv(x, e_ip, hyperedge_weight=w_ip), 0.2)
        z_dev = F.leaky_relu(self.device_conv(x, e_dev, hyperedge_weight=w_dev), 0.2)

        # Dynamic View-Level Attention instead of flat concatenation
        stacked_views = torch.stack([z_auc, z_ip, z_dev], dim=1)
        # Calculate attention scores for each view
        attn_scores = self.view_attn(stacked_views)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum of the views
        z_spatial = (stacked_views * attn_weights).sum(dim=1)

        # 2. Behavioral & Temporal Features ("Node Identity")
        x_static = x[:, self.static_indices]
        z_behavioral = self.behavioral_mlp(x_static)

        if self.temp_dim > 0:
            x_temp = x[:, self.temporal_indices].view(-1, self.seq_len, self.temp_dim)
            x_temp_emb = self.temp_proj(x_temp)
            z_temp_seq = self.transformer(x_temp_emb)

            # Dual Pooling (Capture both Bursts AND Averages)
            z_temp_max = z_temp_seq.max(dim=1)[0]
            z_temp_mean = z_temp_seq.mean(dim=1)
            z_temporal = self.temp_combine(torch.cat([z_temp_max, z_temp_mean], dim=1))

            # 3. Match-Mismatch Logic (CA1 Anomaly Detection)
            # Find the discrepancy between structural connections (z_spatial) and behavior (z_temporal)
            mismatch_signal = torch.abs(z_temporal - z_spatial)
            mismatch_weight = self.mismatch_gate(torch.cat([z_temporal, z_spatial], dim=1))

            # Amplify temporal signal where behavior fundamentally clashes with network context (Camouflage)
            # z_anom = z_behavioral + (z_temporal * mismatch_weight) + mismatch_signal
            # z_anom = z_behavioral + (z_temporal * mismatch_weight) + (self.mismatch_alpha * mismatch_signal)
            z_anom = z_behavioral + (z_temporal * mismatch_weight) + (torch.clamp(self.mismatch_alpha, 0, 2) * mismatch_signal)
        else:
            z_anom = z_behavioral

        # 4. Final Fusion via SE-Gate
        combined = torch.cat([z_spatial, z_anom], dim=1)
        se = F.relu(self.gate_fc1(combined))
        g = torch.sigmoid(self.gate_fc2(se))
        z_fused = g * z_spatial + (1 - g) * z_anom
        final_repr = torch.cat([z_fused, x_static], dim=1)

        return self.classifier(final_repr)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


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

    criterion = FocalLoss(alpha=0.85, gamma=2.5)

    print(f"Starting {k}-Fold Cross-Validation with Checkpointing...")

    EARLY_STOPPING_PATIENCE = 40

    for fold, (t_idx, v_idx) in enumerate(skf.split(labeled_indices, labeled_y)):
        model = STMVHGNN(x.shape[1], 128, temporal_indices).to(device)
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
            out = model(x, edge_index_auc, edge_index_ip, edge_index_dev, weight_auc, weight_ip, weight_dev).squeeze()
            loss = criterion(out[f_train_mask], y[f_train_mask].float())
            loss.backward()
            optimizer.step()

            # Validation for checkpointing
            model.eval()
            with torch.no_grad():
                val_out = model(x, edge_index_auc, edge_index_ip, edge_index_dev, weight_auc, weight_ip, weight_dev).squeeze()
                v_probs_epoch = torch.sigmoid(val_out[f_val_mask]).cpu().numpy()
                v_labels_epoch = y[f_val_mask].cpu().numpy()
                val_auc = roc_auc_score(v_labels_epoch, v_probs_epoch)

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
            full_logits = model(x, edge_index_auc, edge_index_ip, edge_index_dev, weight_auc, weight_ip, weight_dev).squeeze()
            full_probs = torch.sigmoid(full_logits).cpu().numpy()

            v_probs = full_probs[labeled_indices[v_idx]]
            v_labels = labeled_y[v_idx]

            oof_probs[v_idx] = v_probs
            fold_aucs.append(roc_auc_score(v_labels, v_probs))
            fold_aps.append(average_precision_score(v_labels, v_probs))
            test_probs_acc += full_probs / k

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

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (ST-MV-HGNN)")
    print("=" * 60)
    print(f"  Model                    : ST-MV-HGNN (Spatio-Temporal Multi-View Hypergraph)")
    print(f"  CV Folds                 : {k}")
    print(f"  CV AUC (mean)            : {np.mean(fold_aucs):.4f}")
    print(f"  CV AUC (std)             : {np.std(fold_aucs):.4f}")
    print(f"  CV AP  (mean)            : {np.mean(fold_aps):.4f}")
    print(f"  OOF AUC                  : {oof_auc:.4f}")
    print(f"  OOF AP                   : {oof_ap:.4f}")
    print(f"  Best F1                  : {best_f1:.4f}")
    print(f"  Best Threshold           : {best_thresh:.4f}")
    print("=" * 60 + "\n")

    print(classification_report(labeled_y, (oof_probs > best_thresh).astype(int), target_names=['Human', 'Robot']))

    return test_probs_acc


# Execute and Save
final_test_probs = train_kfold(k=5)

test_bidders = test_df['bidder_id'].values
test_preds = [final_test_probs[bidder_to_idx[b_id]] for b_id in test_bidders]
pd.DataFrame({'bidder_id': test_bidders, 'prediction': test_preds}).to_csv("./result/stmvhgnn_final_oof_results.csv", index=False)
print("\nFinal submission generated: stmvhgnn_final_oof_results.csv")
