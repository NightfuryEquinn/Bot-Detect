"""
utils/aggregate_result.py
Compare evaluation metrics across all trained models.

Expected summary CSVs (produced after running each model script):
  result/logs/tabr/tabr_summary.csv         — columns: model, oof_auc, oof_ap, oof_pr_auc, oof_precision, oof_recall, oof_f1, ...
  result/logs/tabm/tabm_summary.csv         — columns: model, oof_auc, oof_ap, oof_pr_auc, oof_precision, oof_recall, oof_f1, ...
  result/logs/rf/rf_summary.csv             — columns: model, oof_auc, oof_ap, oof_pr_auc, oof_precision, oof_recall, oof_f1, ...
  result/logs/lgbm/lgbm_summary.csv         — columns: model, oof_auc, oof_ap, oof_pr_auc, oof_precision, oof_recall, oof_f1, ...
  result/logs/saint/saint_summary.csv       — columns: model, oof_auc, oof_ap, oof_pr_auc, oof_precision, oof_recall, oof_f1, ...
  result/logs/sthgnn/sthgnn_summary.csv     — columns: model, oof_auc, oof_ap, oof_pr_auc, oof_precision, oof_recall, oof_f1, ...
  result/logs/stmvhgnn/stmvhgnn_summary.csv — columns: model, oof_auc, oof_ap, oof_pr_auc, oof_precision, oof_recall, oof_f1, ...
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
    ("TabR",        "result/logs/tabr/tabr_summary.csv"),
    ("RandomForest","result/logs/rf/rf_summary.csv"),
    ("LightGBM",    "result/logs/lgbm/lgbm_summary.csv"),
    ("TabM",        "result/logs/tabm/tabm_summary.csv"),
    ("SAINT",       "result/logs/saint/saint_summary.csv"),
    ("ST-HGNN",     "result/logs/sthgnn/sthgnn_summary.csv"),
    ("ST-MV-HGNN",  "result/logs/stmvhgnn/stmvhgnn_summary.csv"),
]

METRIC_COLS = [
    ("auc_roc",   "AUC-ROC",   "oof_auc"),
    ("ap",        "Avg Prec",  "oof_ap"),
    ("pr_auc",    "PR-AUC",    "oof_pr_auc"),
    ("precision", "Precision", "oof_precision"),
    ("recall",    "Recall",    "oof_recall"),
    ("f1",        "F1",        "oof_f1"),
]

rows = []
for model_name, path in SUMMARY_PATHS:
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found — run the model first.")
        continue
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()

    entry = {"model": model_name, "source": path}
    for key, label, col in METRIC_COLS:
        entry[key] = float(row[col]) if col in row else float("nan")

    rows.append(entry)

if not rows:
    print("\nNo summary files found. Run the model scripts first.")
    raise SystemExit(1)

results_df = pd.DataFrame(rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Print comparison table
# ---------------------------------------------------------------------------
header_labels = [label for _, label, _ in METRIC_COLS]
col_w = 10

print("\n" + "=" * (8 + 18 + col_w * len(METRIC_COLS)))
print("MODEL METRIC COMPARISON")
print("=" * (8 + 18 + col_w * len(METRIC_COLS)))
header = f"{'Rank':<6}{'Model':<18}" + "".join(f"{h:<{col_w}}" for h in header_labels)
print(header)
print("-" * len(header))
for rank, row in results_df.iterrows():
    vals = "".join(
        f"{row[key]:.4f}    " if not np.isnan(row[key]) else f"{'N/A':<{col_w}}"
        for key, _, _ in METRIC_COLS
    )
    print(f"  {rank+1:<4}{row['model']:<18}{vals}")
print("=" * len(header))
print(f"\nBest model by AUC-ROC: {results_df.iloc[0]['model']}  "
      f"(AUC={results_df.iloc[0]['auc_roc']:.4f})")

# ---------------------------------------------------------------------------
# 3. Save aggregate CSV
# ---------------------------------------------------------------------------
os.makedirs("result", exist_ok=True)
out_cols = ["model"] + [key for key, _, _ in METRIC_COLS] + ["source"]
out_path = "result/aggregate_metric_comparison.csv"
results_df[out_cols].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# 4. Plot — 2 x 3 grid, one subplot per metric
# ---------------------------------------------------------------------------
palette = ["#E84545", "#2B9EB3", "#F5A623", "#8E44AD", "#27AE60", "#E67E22", "#1ABC9C"]
colors = [palette[i % len(palette)] for i in range(len(results_df))]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model Comparison — All Evaluation Metrics", fontsize=14, color="#E0E0E0")

for ax, (key, label, _) in zip(axes.flat, METRIC_COLS):
    vals = results_df[key].fillna(0)
    bars = ax.bar(results_df["model"], vals, color=colors, edgecolor="none")
    ax.set_title(label, color="#E0E0E0")
    ax.set_ylabel(label)
    y_min = max(0.0, vals[vals > 0].min() - 0.05) if (vals > 0).any() else 0.0
    ax.set_ylim(y_min, 1.0)
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7, color="#E0E0E0",
        )

plt.tight_layout()
plot_path = "result/aggregate_metric_comparison.png"
fig.savefig(plot_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {plot_path}")
