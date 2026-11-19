"""Within-structure node-level analysis of the per-grain switching field.

Answers four questions the pooled correlation matrix cannot:

1. Edge features vs the node target -- edge features are aggregated onto their
   incident node (mean/sum over the grain's shared faces), which puts them on
   the same table as the node descriptors and the switching-field label.
2. Per-graph instead of pooled -- every coefficient is computed *within* a
   structure and then averaged over structures, so the between-structure spread
   of grain scales (>3 orders of magnitude) cannot manufacture correlations.
3. Neighbour count and position -- graph degree, the interior-face fraction
   (1.0 for a fully enclosed grain, <1 for one touching the domain wall) and the
   normalised radial position of the grain centroid.
4. Connectivity of grains with similar switching fields -- the switching-field
   correlation between grains at graph distance 1, 2 and 3, split into
   neighbours stacked along the field axis and neighbours lying to the side.

Run from the repo root::

    python analysis/node_level_analysis.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import warnings

warnings.filterwarnings("ignore", message="An input array is constant")
from scipy.stats import spearmanr

from micromagnetic_gnn.data_pipeline import (
    EDGE_FEATURE_MAP,
    NODE_FEATURE_MAP,
    _extract_edge_features,
    _extract_node_features,
    compute_int_face_frac,
)
from micromagnetic_gnn.plotstyle import figure_dir

HERE = figure_dir()
CACHE = Path("data_cache.pkl")
RNG = np.random.default_rng(0)

MIN_GRAINS = 50        # a within-structure rank correlation needs enough grains
HOP_MAX_GRAINS = 1500  # cap for the multi-hop analysis (A^3 blows up beyond)
HOP_STRUCTURES = 300   # structures subsampled for the hop/direction analysis
AXIAL_COS = 0.866      # |cos| > this  -> neighbour stacked along the field axis
LATERAL_COS = 0.5      # |cos| < this  -> neighbour lying to the side

FIG_W, FIG_H = 7.0, 3.4
FS_TICK, FS_LABEL, FS_NOTE = 8, 9, 8
SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID = "#e6e5e1"

NODE_SCALARS = ("volume", "surface", "total_edge_length", "diameq", "sphericity",
                "dihanglemax", "dihanglemin", "dihangleav")
I_SW = NODE_FEATURE_MAP["switching_field"][0]


def node_table(gs):
    """One row per grain: node descriptors, node-aggregated edge descriptors,
    positional descriptors, and the switching-field label."""
    X = _extract_node_features(gs).numpy()
    gids = sorted(gs.grains)
    pos = {g: k for k, g in enumerate(gids)}
    n = len(gids)

    ea = X[:, list(NODE_FEATURE_MAP["easy_axis"])]
    theta = np.degrees(np.arccos(np.clip(
        np.abs(ea[:, 2]) / np.linalg.norm(ea, axis=1), -1, 1)))
    cols = {"theta (deg)": theta}
    for name in NODE_SCALARS:
        cols[name] = X[:, NODE_FEATURE_MAP[name][0]]

    # --- positional descriptors ---------------------------------------------
    centers = np.array([gs.grains[g].center for g in gids])
    cols["face_frac (interior)"] = np.array(
        [compute_int_face_frac(gs, g) for g in gids])
    span = centers.max(0) - centers.min(0)
    span[span == 0] = 1.0
    rel = (centers - centers.min(0)) / span            # 0..1 within the domain
    cols["dist. to wall"] = np.minimum(rel, 1 - rel).min(axis=1) * 2  # 0 wall, 1 core
    cols["height along field"] = rel[:, 2]

    # --- edge features aggregated onto their incident node -------------------
    deg = np.zeros(n)
    agg = {k: np.zeros(n) for k in
           ("face_area (sum)", "face_area (mean)", "face_circumf. (mean)",
            "|center_dist| (mean)", "face |n_z| (mean)", "dipole_coupl. (mean)")}
    if len(gs.edges):
        E = _extract_edge_features(gs).numpy()
        src = np.array([pos[e] for e in gs.edges[:, 0]])
        cd = np.linalg.norm(E[:, list(EDGE_FEATURE_MAP["center_distance"])], axis=1)
        per_edge = {
            "face_area (sum)": E[:, EDGE_FEATURE_MAP["face_area"][0]],
            "face_area (mean)": E[:, EDGE_FEATURE_MAP["face_area"][0]],
            "face_circumf. (mean)": E[:, EDGE_FEATURE_MAP["face_circumference"][0]],
            "|center_dist| (mean)": cd,
            "face |n_z| (mean)": np.abs(E[:, EDGE_FEATURE_MAP["face_normal"][2]]),
            "dipole_coupl. (mean)": np.abs(E[:, EDGE_FEATURE_MAP["dipole_coupling"][0]]),
        }
        np.add.at(deg, src, 1.0)
        for k, v in per_edge.items():
            np.add.at(agg[k], src, v)
        safe = np.maximum(deg, 1.0)
        for k in agg:
            if k.endswith("(mean)"):
                agg[k] /= safe
    cols["n neighbours"] = deg
    cols.update(agg)
    cols["switching_field"] = np.abs(X[:, I_SW])
    return cols


def adjacency(gs, gids):
    pos = {g: k for k, g in enumerate(gids)}
    n = len(gids)
    if len(gs.edges) == 0:
        return sp.csr_matrix((n, n), dtype=bool)
    i = np.array([pos[a] for a in gs.edges[:, 0]])
    j = np.array([pos[b] for b in gs.edges[:, 1]])
    A = sp.csr_matrix((np.ones(len(i), bool), (i, j)), shape=(n, n))
    return (A + A.T).astype(bool)


def hop_pairs(A, h):
    """Index pairs at graph distance exactly h (h = 1, 2, 3)."""
    n = A.shape[0]
    reach = A.copy()
    prev = sp.identity(n, dtype=bool, format="csr") + A
    for _ in range(h - 1):
        reach = (reach @ A).astype(bool)
        exact = reach - reach.multiply(prev)
        prev = (prev + reach).astype(bool)
    exact = A if h == 1 else exact
    exact = sp.triu(exact, k=1).tocoo()
    return exact.row, exact.col


def main():
    print(f"Loading {CACHE} ...")
    with open(CACHE, "rb") as f:
        structures = pickle.load(f)
    usable = [gs for gs in structures if len(gs.grains) >= MIN_GRAINS]
    print(f"  {len(structures)} structures, {len(usable)} with >= {MIN_GRAINS} grains")

    # ---- 1-3: within-structure correlation with the switching field ---------
    per_struct = {}
    for gs in usable:
        cols = node_table(gs)
        y = cols["switching_field"]
        if np.ptp(y) == 0:
            continue
        for k, v in cols.items():
            if k == "switching_field" or np.ptp(v) == 0 or not np.isfinite(v).all():
                continue
            per_struct.setdefault(k, []).append(spearmanr(v, y).statistic)

    print(f"\n=== within-structure Spearman with |H_sw| "
          f"(mean +/- std over structures) ===")
    summary = sorted(((k, np.nanmean(v), np.nanstd(v), len(v))
                      for k, v in per_struct.items()),
                     key=lambda t: -abs(t[1]))
    for k, m, s, cnt in summary:
        bar = "#" * int(round(abs(m) * 40))
        print(f"  {k:<24s} {m:+.3f} +/- {s:.3f}  (n={cnt:4d})  {bar}")

    # ---- 4: switching-field similarity vs graph distance --------------------
    pool = [gs for gs in usable if len(gs.grains) <= HOP_MAX_GRAINS]
    if len(pool) > HOP_STRUCTURES:
        pool = [pool[i] for i in RNG.choice(len(pool), HOP_STRUCTURES, replace=False)]
    print(f"\n=== switching-field similarity vs graph distance "
          f"({len(pool)} structures, N <= {HOP_MAX_GRAINS}) ===")

    hop_rho = {h: [] for h in (1, 2, 3)}
    hop_dev = {h: [] for h in (1, 2, 3)}
    dir_rho = {"axial": [], "lateral": []}
    for gs in pool:
        gids = sorted(gs.grains)
        A = adjacency(gs, gids)
        X = _extract_node_features(gs).numpy()
        y = np.abs(X[:, I_SW])
        if np.ptp(y) == 0:
            continue
        sd = y.std()
        for h in (1, 2, 3):
            r, c = hop_pairs(A, h)
            if len(r) < 30:
                continue
            rho_h = spearmanr(y[r], y[c]).statistic
            if np.isfinite(rho_h):
                hop_rho[h].append(rho_h)
            hop_dev[h].append(np.mean(np.abs(y[r] - y[c])) / sd)
        # directional split of the 1-hop pairs
        centers = np.array([gs.grains[g].center for g in gids])
        r, c = hop_pairs(A, 1)
        if len(r) < 30:
            continue
        d = centers[c] - centers[r]
        cosz = np.abs(d[:, 2]) / np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        for name, m in (("axial", cosz > AXIAL_COS), ("lateral", cosz < LATERAL_COS)):
            if m.sum() >= 30:
                rho_d = spearmanr(y[r[m]], y[c[m]]).statistic
                if np.isfinite(rho_d):
                    dir_rho[name].append(rho_d)

    for h in (1, 2, 3):
        v, d = np.array(hop_rho[h]), np.array(hop_dev[h])
        print(f"  {h}-hop: rho = {v.mean():+.3f} +/- {v.std():.3f}   "
              f"mean |dH| / sigma = {d.mean():.3f}   (n={len(v)} structures)")
    for name in ("axial", "lateral"):
        v = np.array(dir_rho[name])
        print(f"  1-hop {name:<8s}: rho = {v.mean():+.3f} +/- {v.std():.3f} "
              f"(n={len(v)})")

    plot(summary, hop_rho, dir_rho)


def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=FS_TICK, length=2.5, width=0.7)
    ax.set_axisbelow(True)


def plot(summary, hop_rho, dir_rho):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=300,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor("#fcfcfb")

    ax = axes[0]
    style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    labels = [k for k, *_ in summary]
    means = np.array([m for _, m, *_ in summary])
    errs = np.array([s for _, _, s, _ in summary])
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, means, xerr=errs, height=0.68,
            color=[SERIES_1 if m < 0 else SERIES_2 for m in means],
            error_kw=dict(ecolor="#8a8983", lw=0.8, capsize=1.5))
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(ypos, labels, fontsize=FS_TICK)
    ax.set_xlabel(r"within-structure Spearman $\rho$ with $|H_\mathrm{sw}|$",
                  fontsize=FS_LABEL, color=INK_2)
    ax.set_title("(a)  Correlates of the per-grain switching field",
                 fontsize=FS_LABEL + 1, color=INK, loc="left", pad=8)

    ax = axes[1]
    style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    hops = [1, 2, 3]
    m = [np.mean(hop_rho[h]) for h in hops]
    e = [np.std(hop_rho[h]) for h in hops]
    ax.errorbar(hops, m, yerr=e, marker="o", ms=5, lw=1.6, color=SERIES_1,
                capsize=2.5, label="all neighbours", zorder=3)
    for name, col in (("axial", SERIES_2), ("lateral", SERIES_3)):
        if dir_rho[name]:
            ax.errorbar([1], [np.mean(dir_rho[name])], yerr=[np.std(dir_rho[name])],
                        marker="s", ms=5, lw=0, color=col, capsize=2.5,
                        label=f"1-hop, {name}", zorder=4)
    ax.axhline(0, color=INK_2, linewidth=0.8, linestyle="--")
    ax.set_xticks(hops)
    ax.set_xlabel("graph distance (hops)", fontsize=FS_LABEL, color=INK_2)
    ax.set_ylabel(r"$\rho$ between $|H_\mathrm{sw}|$ of pairs",
                  fontsize=FS_LABEL, color=INK_2)
    ax.set_title("(b)  Spatial correlation of switching",
                 fontsize=FS_LABEL + 1, color=INK, loc="left", pad=8)
    leg = ax.legend(fontsize=FS_NOTE - 0.5, frameon=True, facecolor="#fcfcfb",
                    edgecolor=GRID, loc="upper right")
    leg.get_frame().set_linewidth(0.7)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(HERE / f"node_level_analysis{ext}", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  saved {HERE / 'node_level_analysis'}.pdf (+ .png)")


if __name__ == "__main__":
    main()
