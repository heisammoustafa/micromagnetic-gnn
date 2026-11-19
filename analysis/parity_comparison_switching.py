"""Parity figure and metrics for the nucleation-field (switching) model.

Produces ``figures/switching_parity.pdf`` (train | test | generalization)
in the same style as ``coercivity_prediction.pdf`` and ``bhmax_prediction.pdf``.
R² and Spearman rho and RMSE are written above each panel. Also prints R²,
RMSE, MAE, and Spearman for all four splits.

Run from the repo root::

    python analysis/parity_comparison_switching.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from micromagnetic_gnn.checkpoints import DEFAULT_CACHE, load_switching_model
from micromagnetic_gnn.plotstyle import STYLE, C_REF, C_TRAIN, figure_dir, width

# Same size and type as coercivity_prediction.pdf: 6.02 in wide, the paper style's 9 pt
# text and 8 pt ticks, included at 0.7\linewidth in a figure*. The height is larger
# than that figure's 2.5 in because the statistics are written above the axes.
FIG_W, FIG_H = width(1.0), 3.0
TICK_FS, LABEL_FS, TITLE_FS, STATS_FS = 8, 9, 9, 9

def _predict(model, loader, scaler, device):
    """Return (y_true, y_pred) in real units for a node-level model."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.view(-1).cpu())
            targets.append(batch.y.view(-1).cpu())
    y_pred_scaled = torch.cat(preds).numpy()
    y_true_scaled = torch.cat(targets).numpy()
    std  = float(scaler.label_scaler.std.reshape(-1)[0])
    mean = float(scaler.label_scaler.mean.reshape(-1)[0])
    return y_true_scaled * std + mean, y_pred_scaled * std + mean


def _metrics(y_true, y_pred):
    resid = y_pred - y_true
    rmse  = float(np.sqrt(np.mean(resid ** 2)))
    mae   = float(np.mean(np.abs(resid)))
    r2    = float(1.0 - np.sum(resid ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    rho   = float(spearmanr(y_true, y_pred).statistic)
    return r2, rmse, mae, rho


def parity(ax, y_true, y_pred, title: str) -> None:
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.04 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.scatter(y_true, y_pred, s=3, color=C_TRAIN, linewidths=0,
               alpha=0.3, rasterized=True, zorder=3)
    ax.plot(lims, lims, color=C_REF, ls="--", lw=0.9, zorder=4)
    r2, rmse, _, rho = _metrics(y_true, y_pred)
    # Statistics above the axes, so they cannot cover any prediction.
    ax.text(0.5, 1.02, f"$R^2$ = {r2:.3f},  $\\rho$ = {rho:.3f}\nRMSE = {rmse:.3g}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=STATS_FS,
            linespacing=1.2)
    ax.set_title(title, fontsize=TITLE_FS, pad=2.4 * STATS_FS + 6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trained = load_switching_model(device=device)
    print(f"Loaded {trained.path}")
    trained.model.eval()

    train_ds, val_ds, test_ds, ext_ds = trained.datasets(DEFAULT_CACHE)
    splits = [
        ("training",       train_ds),
        ("validation",     val_ds),
        ("test",           test_ds),
        ("generalization", ext_ds),
    ]

    results = {}
    print(f"\n{'Set':<14} {'R²':>7} {'RMSE':>8} {'MAE':>8} {'Spearman':>10}")
    print("-" * 52)
    for name, ds in splits:
        loader = DataLoader(ds, batch_size=64)
        y_true, y_pred = _predict(trained.model, loader, trained.scaler, device)
        r2, rmse, mae, rho = _metrics(y_true, y_pred)
        results[name] = (y_true, y_pred)
        print(f"{name:<14} {r2:>7.3f} {rmse:>8.3f} {mae:>8.3f} {rho:>10.3f}")

    # --- parity figure: train | test | generalization ---
    plt.style.use(STYLE)
    plt.rcParams.update({"xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS,
                         "axes.grid": False})
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
    for ax, key, title in zip(
        axes,
        ["training", "test", "generalization"],
        ["train", "test", "generalization"],
    ):
        parity(ax, *results[key], title)

    for ax in axes:
        ax.set_xlabel(r"simulated $H_{\mathrm{nuc}}$  (T)", fontsize=LABEL_FS)
    axes[0].set_ylabel(r"predicted $H_{\mathrm{nuc}}$  (T)", fontsize=LABEL_FS)

    fig.tight_layout()
    out = figure_dir() / "switching_parity.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"\nSaved {out} (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
