#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# utils/shared.py — Shared constants and utilities for all model scripts
# =============================================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must come before pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "bot":     "#E84545",
    "human":   "#2B9EB3",
    "neutral": "#6C757D",
    "calib":   "#F5A623",
    "uncalib": "#9B59B6",
    "accent":  "#F5A623",
    "tabm":    "#8E44AD",
    "tabr":    "#8E44AD",
    "tune":    "#F39C12",
}

# ─── Plot style ───────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.dpi":        140,
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#1A1D27",
    "axes.labelcolor":   "#E0E0E0",
    "xtick.color":       "#A0A0A0",
    "ytick.color":       "#A0A0A0",
    "text.color":        "#E0E0E0",
    "grid.color":        "#2A2D37",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def configure_plots():
    """Apply the project-wide matplotlib + seaborn style."""
    sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
    plt.rcParams.update(PLOT_STYLE)


def save_fig(fig, save_dir, name):
    """Save *fig* to ``<save_dir>/<name>.png`` then close it."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ↳ Saved  {path}")


# ─── Dual logging ─────────────────────────────────────────────────────────────
class DualLogger:
    """Tee stdout to both terminal and a log file simultaneously."""

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logger(output_dir: str, file_name: str) -> str:
    """Redirect stdout to a DualLogger writing to ``<output_dir>/log_<file_name>.txt``."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"log_{file_name}.txt")
    sys.stdout = DualLogger(log_path)
    return log_path


# ─── Focal loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Binary focal loss with optional label smoothing.

    Parameters
    ----------
    alpha:     class-weighting factor for the positive class.
    gamma:     focusing exponent (higher → harder examples get more weight).
    smoothing: label-smoothing coefficient (0 = no smoothing).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smoothing: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(inputs, targets_s, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()
