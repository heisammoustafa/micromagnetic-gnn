"""Per-structure comparison: analytical BHmax formula vs. GNN prediction.

Both methods are evaluated on every structure in the dataset in the same order,
so results can be matched by index without any value-based lookup.

Produces:
  bhmax_analytic_vs_gnn.pdf   scatter of analytical error vs. GNN error

Run from the repo root::

    python analysis/bhmax_analytic_vs_gnn.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from micromagnetic_gnn.checkpoints import DEFAULT_CACHE, load_bhmax_model
from micromagnetic_gnn.data_pipeline import (
    NODE_FEATURE_MAP, _extract_node_features,
)
from micromagnetic_gnn.dataset import extrapolation_cond
from micromagnetic_gnn.inference import mc_predict
from micromagnetic_gnn.plotstyle import STYLE, C_REF, C_TRAIN, C_TEST, C_GEN, figure_dir, width

CACHE = Path("data_cache.pkl")
JS = 1.61
MU0 = 4e-7 * np.pi
BH_FULL = JS ** 2 / (4 * MU0)
EA = list(NODE_FEATURE_MAP["easy_axis"])
IV = NODE_FEATURE_MAP["volume"][0]
MC_SAMPLES = 20


import re

def analytic_pred(gs) -> tuple[float, float]:
    """Return (bh_analytic, cos_v_weighted)."""
    X = _extract_node_features(gs).numpy()
    ea, V = X[:, EA], X[:, IV]
    ok = np.isfinite(V) & (V > 0) & np.isfinite(ea).all(1)
    ea, V = ea[ok], V[ok]
    if len(V) == 0:
        return float("nan"), float("nan")
    cos = np.abs(ea[:, 2]) / np.linalg.norm(ea, axis=1)
    cos_v = float((V * cos).sum() / V.sum())
    return float(BH_FULL * cos_v ** 2), cos_v


def parse_folder(folder: str) -> dict:
    """Extract gb_t, AR, size_x from folder name."""
    d = {}
    for key in ("gb_t", "AR", "size_x"):
        m = re.search(rf"{key}=([0-9.]+)", folder)
        d[key] = float(m.group(1)) if m else float("nan")
    return d


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- load raw structures --------------------------------------------------
    print(f"Loading {CACHE} ...")
    with open(CACHE, "rb") as f:
        structures = pickle.load(f)
    n = len(structures)
    print(f"  {n} structures")

    folders = [gs.params["folder"] for gs in structures]
    bh_sim = np.array([float(gs.params["bhmax"]) for gs in structures])
    n_grains = np.array([int(gs.params["no_grains"]) for gs in structures])

    analytic_results = [analytic_pred(gs) for gs in structures]
    bh_analytic = np.array([r[0] for r in analytic_results])
    cos_v       = np.array([r[1] for r in analytic_results])

    meta = [parse_folder(f) for f in folders]
    gb_t   = np.array([m["gb_t"]   for m in meta])
    ar     = np.array([m["AR"]     for m in meta])
    size_x = np.array([m["size_x"] for m in meta])

    # split labels: determine which split each structure belongs to
    # generalization is assigned first (deterministic criterion), the rest follow
    # the fixed random seed split used in default_gs_data
    trained = load_bhmax_model(device=device)
    print(f"Loaded {trained.path}")

    # recover split membership by comparing to each split's dataset
    train_ds, val_ds, test_ds, ext_ds = trained.datasets(DEFAULT_CACHE)

    # Replicate the exact split from dataset.py (seed=42, train_frac=0.75,
    # val_frac=0.1) so split labels are assigned by index with no value matching.
    import random as _random
    TRAIN_FRAC, VAL_FRAC, SPLIT_SEED = 0.75, 0.1, 42

    ext_mask  = np.array([extrapolation_cond(gs) for gs in structures])
    ext_idx   = np.where(ext_mask)[0]
    inds_rest = list(np.where(~ext_mask)[0])
    _random.Random(SPLIT_SEED).shuffle(inds_rest)
    n_rest     = len(inds_rest)
    n_train    = int(TRAIN_FRAC * n_rest)
    n_val      = int(VAL_FRAC   * n_rest)
    train_idx  = inds_rest[:n_train]
    val_idx    = inds_rest[n_train:n_train + n_val]
    test_idx   = inds_rest[n_train + n_val:]

    splits_label = np.array(["unknown"] * n, dtype=object)
    splits_label[ext_idx]   = "generalization"
    splits_label[train_idx] = "train"
    splits_label[val_idx]   = "val"
    splits_label[test_idx]  = "test"
    print(f"  Split sizes — train:{len(train_idx)}  val:{len(val_idx)}  "
          f"test:{len(test_idx)}  gen:{len(ext_idx)}")

    # Run GNN inference on each split dataset (correct feature set) and assign
    # predictions back by position in the shuffled index arrays.
    scaler = trained.scaler
    print("Running GNN inference per split ...")
    trained.model.train()  # MC dropout active

    bh_gnn = np.full(n, np.nan)

    def infer_split(ds, idx_array, label):
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        y_mean_sp, _, _ = mc_predict(
            trained.model, loader, scaler, MC_SAMPLES, device
        )
        y_mean_sp = y_mean_sp[:, 0]
        assert len(y_mean_sp) == len(idx_array), \
            f"{label}: {len(y_mean_sp)} predictions but {len(idx_array)} indices"
        bh_gnn[idx_array] = y_mean_sp
        print(f"  {label}: {len(idx_array)} assigned")

    infer_split(train_ds, train_idx, "train")
    infer_split(val_ds,   val_idx,   "val")
    infer_split(test_ds,  test_idx,  "test")
    infer_split(ext_ds,   ext_idx,   "generalization")

    n_pred = (~np.isnan(bh_gnn)).sum()
    print(f"  {n_pred}/{n} GNN predictions")

    # --- errors ---------------------------------------------------------------
    err_analytic = np.abs(bh_sim - bh_analytic) / bh_sim * 100
    err_gnn      = np.abs(bh_sim - bh_gnn)      / bh_sim * 100
    valid = np.isfinite(err_analytic) & np.isfinite(err_gnn)
    print(f"  {valid.sum()} structures with both errors")

    # Pearson correlation
    r = float(np.corrcoef(err_analytic[valid], err_gnn[valid])[0, 1])
    print(f"  Pearson r(analytic_err, gnn_err) = {r:.3f}")

    # --- plot -----------------------------------------------------------------
    plt.style.use(STYLE)
    colors = {"train": C_TRAIN, "val": "#aaaaaa", "test": C_TEST,
              "generalization": C_GEN, "unknown": "#cccccc"}

    fig, ax = plt.subplots(figsize=(width(0.6), 2.8))
    for split in ("train", "val", "test", "generalization"):
        mask = valid & (splits_label == split)
        if mask.sum() == 0:
            continue
        ax.scatter(err_analytic[mask], err_gnn[mask],
                   s=8, alpha=0.6, linewidths=0,
                   color=colors[split],
                   label=f"{split} ({mask.sum()})",
                   zorder=3 if split == "generalization" else 2)

    lim = max(np.nanpercentile(err_analytic[valid], 99.5),
              np.nanpercentile(err_gnn[valid], 99.5)) * 1.1
    ax.plot([0, lim], [0, lim], color=C_REF, ls="--", lw=0.8, zorder=1,
            label="equal error")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"analytical relative error  (%)")
    ax.set_ylabel(r"GNN relative error  (%)")
    ax.text(0.97, 0.05, f"$r$ = {r:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e6e5e1"))
    ax.legend(fontsize=6, frameon=True, facecolor="white", edgecolor="#e6e5e1")
    fig.tight_layout()

    out = figure_dir() / "bhmax_analytic_vs_gnn.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out} (+ .png)")

    # --- detailed top-20 tables -----------------------------------------------
    header = (f"{'folder':<60}  {'N':>5}  {'gb_t':>5}  {'AR':>4}  "
              f"{'size_x':>7}  {'cos_v':>5}  {'ana%':>6}  {'gnn%':>6}  split")
    sep = "-" * len(header)

    def print_table(label, indices_sorted):
        print(f"\n{label}")
        print(header); print(sep)
        for i in indices_sorted[:20]:
            fn = folders[i].replace("ROM_Simulations/", "")
            print(f"  {fn:<58}  {n_grains[i]:>5}  {gb_t[i]:>5.1f}  {ar[i]:>4.1f}  "
                  f"{size_x[i]:>7.0f}  {cos_v[i]:>5.3f}  "
                  f"{err_analytic[i]:>6.1f}  {err_gnn[i]:>6.1f}  {splits_label[i]}")

    valid_idx = np.where(valid)[0]
    print_table("Top-20 GNN failures  (sorted by GNN error):",
                valid_idx[np.argsort(err_gnn[valid])[::-1]])
    print_table("Top-20 analytical failures  (sorted by analytical error):",
                valid_idx[np.argsort(err_analytic[valid])[::-1]])

    # GNN-only failures: high GNN error, low analytical error
    gnn_only = valid & (err_gnn > 5.0) & (err_analytic < 2.0)
    print_table(f"\nGNN-only failures  (gnn>5%, analytic<2%)  — {gnn_only.sum()} structures:",
                np.where(gnn_only)[0][np.argsort(err_gnn[gnn_only])[::-1]])


if __name__ == "__main__":
    main()
