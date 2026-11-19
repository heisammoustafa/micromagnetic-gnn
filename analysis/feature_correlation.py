"""Correlation analysis of node, edge, and global features.

Uses exactly the default feature sets of ``dataset.py::default_datasets``:

* node : easy_axis (reduced to the misalignment angle theta), volume, surface,
  total_edge_length, diameq, sphericity, dihanglemax, dihanglemin, dihangleav
* edge : center_distance / face_distance (reduced to magnitudes),
  face_normal (reduced to |n_z|, since its magnitude is 1 by construction),
  face_area, face_circumference
* global: gb_t

Feature values are taken from the training pipeline's own extractors
(``_extract_node_features`` / ``_extract_edge_features``), so they are exactly
what the model sees (both coefficients are invariant under the affine
StandardScaler).

Spearman rho is the primary coefficient: the size features are related by power
laws (V ~ L^3, area ~ L^2) and are heavy-tailed, so Pearson understates
near-deterministic monotone dependence and is dominated by the largest ~1% of
grains.

Three Spearman heatmaps are written -- node, edge, graph: node
level (+ per-grain switching-field label), edge level, and graph level
(per-structure means + gb_t + the labels H_c and BHmax).

Run from the repo root::

    python analysis/feature_correlation.py            # compute + plot
    python analysis/feature_correlation.py --replot   # plot from saved matrices

The matrices are saved to ``analysis/feature_corr_values/`` (.npz and .csv).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from micromagnetic_gnn.data_pipeline import (
    EDGE_FEATURE_MAP,
    NODE_FEATURE_MAP,
    _extract_edge_features,
    _extract_node_features,
)
from micromagnetic_gnn.plotstyle import figure_dir

HERE = figure_dir()
CACHE = Path("data_cache.pkl")
VALUES = Path(__file__).resolve().parent / "feature_corr_values"  # saved matrices
RNG = np.random.default_rng(0)
MAX_SAMPLES = 300_000  # subsample cap for pooled node/edge tables
MIN_GRAINS = 50        # a within-structure rank correlation needs enough grains

# --- figure layout -----------------------------------------------------------
# Each matrix is drawn at the width it is printed at in the paper (\textwidth is
# 7.0 in for elsarticle 5p; node/edge sit side by side at 0.6 / 0.4, graph at 0.8) and
# included at scale 1, so the font sizes below are the printed sizes.
FIG_W = {"node": 4.2, "edge": 2.8, "graph": 5.6}   # in
TICK_FS = 9.0
ANN_FS_MAX = 9.0
MASK_UPPER = True  # a correlation matrix is symmetric -- show one triangle only

# Scalar node features from dataset.py defaults (easy_axis handled separately).
NODE_SCALARS = ("volume", "surface", "total_edge_length", "diameq",
                "sphericity", "dihanglemax", "dihanglemin", "dihangleav")


def load_structures():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def _theta_deg(ea: np.ndarray) -> np.ndarray:
    """Misalignment angle (deg) of easy axes vs the field (z) axis."""
    return np.degrees(np.arccos(np.clip(np.abs(ea[:, 2]) / np.linalg.norm(ea, axis=1), -1, 1)))


def node_matrix(gs) -> np.ndarray:
    """Default node features (easy_axis -> theta) + switching-field label."""
    X = _extract_node_features(gs).numpy()
    ea = X[:, list(NODE_FEATURE_MAP["easy_axis"])]
    cols = [_theta_deg(ea)]
    cols += [X[:, NODE_FEATURE_MAP[n][0]] for n in NODE_SCALARS]
    cols.append(np.abs(X[:, NODE_FEATURE_MAP["switching_field"][0]]))
    return np.column_stack(cols)


# Tick labels use the paper's symbols only (Tables: features / correlations); the
# caption carries the translation.
_SYM = {
    "volume":            r"$\Omega$",
    "surface":           r"$S$",
    "total_edge_length": r"$L$",
    "diameq":            r"$d_\mathrm{eq}$",
    "sphericity":        r"$\Psi$",
    "dihanglemax":       r"$\alpha^{\max}$",
    "dihanglemin":       r"$\alpha^{\min}$",
    "dihangleav":        r"$\bar{\alpha}$",
}
THETA = r"$\theta_i$"
NODE_COLS = [THETA] + [_SYM[n] for n in NODE_SCALARS] + [r"$H_\mathrm{nuc}$"]


def edge_matrix(gs) -> np.ndarray | None:
    """Default edge features; vectors reduced to |.| (face_normal to |n_z|)."""
    if len(gs.edges) == 0:
        return None
    E = _extract_edge_features(gs).numpy()
    cd = E[:, list(EDGE_FEATURE_MAP["center_distance"])]
    fd = E[:, list(EDGE_FEATURE_MAP["face_distance"])]
    nz = np.abs(E[:, EDGE_FEATURE_MAP["face_normal"][2]])
    return np.column_stack([
        np.linalg.norm(cd, axis=1),
        np.linalg.norm(fd, axis=1),
        nz,
        E[:, EDGE_FEATURE_MAP["face_area"][0]],
        E[:, EDGE_FEATURE_MAP["face_circumference"][0]],
    ])


EDGE_COLS = [r"$|\mathbf{r}_{ij}|$", r"$|\mathbf{d}_{ij}|$", r"$|n_z|$",
             r"$A_{ij}$", r"$P_{ij}$"]

# Graph-level columns are per-structure means of the node/edge features; the
# "mean" is stated in the caption rather than repeated in every tick label.
GRAPH_COLS = ([THETA] + [_SYM[n] for n in NODE_SCALARS] + list(EDGE_COLS)
              + [r"$t_\mathrm{gb}$", r"$H_\mathrm{C}$", r"$(BH)_\mathrm{max}$"])


def graph_matrix(structures) -> np.ndarray:
    """Per-structure means of the default features + gb_t + labels."""
    rows = []
    for gs in structures:
        Xn = node_matrix(gs)[:, :-1]        # drop switching-field label column
        Xe = edge_matrix(gs)
        e_means = Xe.mean(axis=0) if Xe is not None else np.full(len(EDGE_COLS), np.nan)
        rows.append(np.concatenate([
            Xn.mean(axis=0), e_means,
            [gs.params["gb_t"], gs.params["coercivity"], gs.params["bhmax"]],
        ]))
    return np.asarray(rows)


def heatmap(mat, cols, out_name, level, cbar_label=r"Spearman $\rho$"):
    n = len(cols)
    W = FIG_W[level]
    left, right, bottom, top = 0.55, 0.98, 0.55, 0.08   # in
    cell = (W - left - right) / n
    cell_pt = cell * 72
    # "-.45" (the widest entry) must fit inside a cell with a margin.
    ann_fs = float(min(ANN_FS_MAX, 0.40 * cell_pt))

    shown = mat.astype(float).copy()
    if MASK_UPPER:
        shown[np.triu_indices(n, 1)] = np.nan

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#fcfcfb")  # masked triangle blends into the page

    H = cell * n + bottom + top
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([left / W, bottom / H, cell * n / W, cell * n / H])
    im = ax.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), cols, rotation=45, ha="right", rotation_mode="anchor",
                  fontsize=TICK_FS)
    ax.set_yticks(range(n), cols, fontsize=TICK_FS)
    ax.tick_params(length=2, width=0.6, pad=2, colors="#52514e")
    for side in ax.spines.values():
        side.set_visible(False)
    # Hairline separators keep adjacent cells from merging into one blob.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            v = shown[i, j]
            if not np.isfinite(v):
                continue
            txt = "1" if i == j else f"{v:.2f}".replace("0.", ".").replace("-.00", ".00")
            ax.text(j, i, txt,
                    ha="center", va="center", fontsize=ann_fs,
                    color="white" if abs(v) > 0.6 else "#333333")

    cax = fig.add_axes([(W - right + 0.12) / W, bottom / H + 0.08 * cell * n / H,
                        0.12 / W, 0.55 * cell * n / H])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=TICK_FS, color="#52514e")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=TICK_FS - 1, colors="#52514e", length=2, width=0.6)
    for ext in (".pdf", ".png"):
        fig.savefig(HERE / (out_name + ext), facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  saved {HERE / out_name}.pdf (+ .png); {W:.1f} x {H:.1f} in, "
          f"annotations {ann_fs:.1f} pt, ticks {TICK_FS:.1f} pt")


def save_values(name, rho, cols):
    """Write the matrix so figures can be re-styled without recomputing it."""
    VALUES.mkdir(exist_ok=True)
    np.savez(VALUES / f"{name}_spearman.npz", rho=rho, cols=np.array(cols))
    plain = [c.replace("$", "") for c in cols]
    np.savetxt(VALUES / f"{name}_spearman.csv", rho, delimiter=",", fmt="%.4f",
               header=",".join(f'"{c}"' for c in plain), comments="")


def load_values(name):
    d = np.load(VALUES / f"{name}_spearman.npz")
    return d["rho"], [str(c) for c in d["cols"]]


def report(mat, cols, label_cols, kind):
    """Print each label's strongest correlates."""
    print(f"  -- {kind} --")
    for lc in label_cols:
        li = cols.index(lc)
        order = np.argsort(-np.abs(mat[li]))
        top = [(cols[j], mat[li, j]) for j in order if j != li][:6]
        print(f"  {lc}: " + ", ".join(f"{c} ({v:+.2f})" for c, v in top))


def within_structure(structures, matrix_fn, ncols):
    """Mean over structures of the within-structure Spearman matrix.

    The node-level GNN predicts per-grain values inside one microstructure, so
    the relationships it must capture are the ones that hold within a structure.
    Pooling grains across structures would instead be dominated by the spread of
    coercivity levels between structures.
    """
    mats = []
    for gs in structures:
        X = matrix_fn(gs)
        if X is None:
            continue
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) < MIN_GRAINS or (X.max(axis=0) - X.min(axis=0) == 0).any():
            continue
        mats.append(np.atleast_2d(spearmanr(X).statistic))
    return np.nanmean(np.stack(mats), axis=0), len(mats)


def replot():
    """Redraw the figures from the saved matrices (no data, no recomputation)."""
    for name in ("node", "edge", "graph"):
        rho, cols = load_values(name)
        save_values(name, rho, cols)
        heatmap(rho, cols, f"feature_corr_{name}_spearman", name)


def main():
    if "--replot" in sys.argv:
        return replot()
    print(f"Loading {CACHE} ...")
    structures = load_structures()
    print(f"  {len(structures)} structures")

    # --- node level: within structures ---------------------------------------
    rho_node, n_struct = within_structure(structures, node_matrix, len(NODE_COLS))
    print(f"\n=== node level (within structures, {n_struct} structures) ===")
    save_values("node", rho_node, NODE_COLS)
    heatmap(rho_node, NODE_COLS, "feature_corr_node_spearman", "node")
    report(rho_node, NODE_COLS, [NODE_COLS[-1]], "Spearman, within structures")

    # --- edge and graph level: pooled -----------------------------------------
    tables = {
        "edge": (np.concatenate([m for gs in structures
                                 if (m := edge_matrix(gs)) is not None]),
                 EDGE_COLS, []),
        "graph": (graph_matrix(structures), GRAPH_COLS, [r"$H_\mathrm{C}$", r"$(BH)_\mathrm{max}$"]),
    }
    for name, (X, cols, labels) in tables.items():
        if len(X) > MAX_SAMPLES:
            X = X[RNG.choice(len(X), MAX_SAMPLES, replace=False)]
        X = X[~np.isnan(X).any(axis=1)]
        print(f"\n=== {name} level, pooled ({len(X)} samples, {len(cols)} features) ===")
        rho = np.atleast_2d(spearmanr(X).statistic)
        save_values(name, rho, cols)
        heatmap(rho, cols, f"feature_corr_{name}_spearman", name)
        if labels:
            report(rho, cols, labels, "Spearman, pooled")


if __name__ == "__main__":
    main()
