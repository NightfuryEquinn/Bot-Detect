"""
utils/aggregate_result.py
Compare AUC ROC across all trained models.

Expected summary CSVs (produced after running each model script):
  result/logs/tabr/tabr_summary.csv         — columns: model, val_auc, val_ap
  result/logs/tabm/tabm_summary.csv         — columns: model, oof_auc, oof_ap, ...
  result/logs/rf/rf_summary.csv             — columns: model, oof_auc, oof_ap, ...
  result/logs/lgbm/lgbm_summary.csv         — columns: model, oof_auc, oof_ap, ...
  result/logs/saint/saint_summary.csv       — columns: model, oof_auc, oof_ap, ...
  result/logs/sthgnn/sthgnn_summary.csv     — columns: model, oof_auc, oof_ap, ...
  result/logs/stmvhgnn/stmvhgnn_summary.csv — columns: model, oof_auc, oof_ap, ...
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared import configure_plots

configure_plots()

# ---------------------------------------------------------------------------
# 1. Load all summaries
# ---------------------------------------------------------------------------
SUMMARY_PATHS = [
    ("TabR",      "result/logs/tabr/tabr_summary.csv"),
    ("RandomForest", "result/logs/rf/rf_summary.csv"),
    ("LightGBM",  "result/logs/lgbm/lgbm_summary.csv"),
    ("TabM",      "result/logs/tabm/tabm_summary.csv"),
    ("SAINT",     "result/logs/saint/saint_summary.csv"),
    ("ST-HGNN",   "result/logs/sthgnn/sthgnn_summary.csv"),
    ("ST-MV-HGNN","result/logs/stmvhgnn/stmvhgnn_summary.csv"),
]

rows = []
for model_name, path in SUMMARY_PATHS:
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found — run the model first.")
        continue
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()

    # Normalise the AUC column name: tabr uses val_auc, others use oof_auc
    if "oof_auc" in row:
        auc = float(row["oof_auc"])
        auc_label = "OOF AUC"
    elif "val_auc" in row:
        auc = float(row["val_auc"])
        auc_label = "Val AUC"
    else:
        print(f"  [WARN] No AUC column found in {path}")
        continue

    ap_col = next((c for c in row if "ap" in c.lower()), None)
    ap = float(row[ap_col]) if ap_col else float("nan")

    rows.append({
        "model":     model_name,
        "auc":       auc,
        "auc_label": auc_label,
        "ap":        ap,
        "source":    path,
    })

if not rows:
    print("\nNo summary files found. Run the model scripts first.")
    raise SystemExit(1)

results_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Print comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL AUC ROC COMPARISON")
print("=" * 70)
print(f"{'Rank':<6}{'Model':<18}{'AUC ROC':<12}{'AP':<12}{'Metric Type'}")
print("-" * 70)
for rank, row in results_df.iterrows():
    print(f"  {rank+1:<4}{row['model']:<18}{row['auc']:.4f}      "
          f"{row['ap']:.4f}      {row['auc_label']}")
print("=" * 70)
print(f"\nBest model: {results_df.iloc[0]['model']}  "
      f"(AUC={results_df.iloc[0]['auc']:.4f})")

# ---------------------------------------------------------------------------
# 3. Save aggregate CSV
# ---------------------------------------------------------------------------
os.makedirs("result", exist_ok=True)
out_path = "result/aggregate_auc_comparison.csv"
results_df[["model", "auc", "ap", "auc_label", "source"]].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# 4. Plot AUC ROC bar chart
# ---------------------------------------------------------------------------
palette = ["#E84545", "#2B9EB3", "#F5A623", "#8E44AD", "#27AE60", "#E67E22", "#1ABC9C"]
colors = [palette[i % len(palette)] for i in range(len(results_df))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison — AUC ROC & Average Precision", fontsize=14, color="#E0E0E0")

ax = axes[0]
bars = ax.bar(results_df["model"], results_df["auc"], color=colors, edgecolor="none")
ax.set_title("AUC ROC", color="#E0E0E0")
ax.set_ylabel("AUC")
ax.set_ylim(max(0, results_df["auc"].min() - 0.05), 1.0)
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, results_df["auc"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#E0E0E0")

ax = axes[1]
ap_vals = results_df["ap"].fillna(0)
bars2 = ax.bar(results_df["model"], ap_vals, color=colors, edgecolor="none")
ax.set_title("Average Precision (AP)", color="#E0E0E0")
ax.set_ylabel("AP")
ax.set_ylim(max(0, ap_vals.min() - 0.05), 1.0)
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars2, ap_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#E0E0E0")

plt.tight_layout()
plot_path = "result/aggregate_auc_comparison.png"
fig.savefig(plot_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {plot_path}")
