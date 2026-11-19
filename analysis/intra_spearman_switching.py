"""Compute intra-structure Spearman correlation for the switching-field model.

For each grain structure, the Spearman correlation between predicted and
simulated nucleation fields is computed across its grains. The per-structure
values are then averaged. This avoids the inter-structure spread that inflates
the pooled Spearman metric.

Run from the repo root::

    python analysis/intra_spearman_switching.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from micromagnetic_gnn.checkpoints import DEFAULT_CACHE, load_switching_model


def _predict(model, loader, scaler, device):
    model.eval()
    preds, targets, graph_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.view(-1).cpu())
            targets.append(batch.y.view(-1).cpu())
            graph_ids.append(batch.batch.cpu())
    y_pred_scaled = torch.cat(preds).numpy()
    y_true_scaled = torch.cat(targets).numpy()
    gids = torch.cat(graph_ids).numpy()
    std  = float(scaler.label_scaler.std.reshape(-1)[0])
    mean = float(scaler.label_scaler.mean.reshape(-1)[0])
    return y_true_scaled * std + mean, y_pred_scaled * std + mean, gids


def intra_spearman(y_true, y_pred, graph_ids):
    rhos = []
    for gid in np.unique(graph_ids):
        mask = graph_ids == gid
        if mask.sum() < 2:
            continue
        rhos.append(spearmanr(y_true[mask], y_pred[mask]).statistic)
    return float(np.mean(rhos)), rhos


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trained = load_switching_model(device=device)
    trained.model.eval()

    train_ds, val_ds, test_ds, ext_ds = trained.datasets(DEFAULT_CACHE)
    splits = [
        ("training",       train_ds),
        ("validation",     val_ds),
        ("test",           test_ds),
        ("generalization", ext_ds),
    ]

    print(f"\n{'Set':<14} {'pooled':>8} {'intra (mean)':>14} {'intra (min)':>12} {'n_structs':>10}")
    print("-" * 55)
    for name, ds in splits:
        loader = DataLoader(ds, batch_size=64)
        y_true, y_pred, gids = _predict(trained.model, loader, trained.scaler, device)
        pooled = spearmanr(y_true, y_pred).statistic
        mean_rho, rhos = intra_spearman(y_true, y_pred, gids)
        print(f"{name:<14} {pooled:>8.3f} {mean_rho:>14.3f} {min(rhos):>12.3f} {len(rhos):>10d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
