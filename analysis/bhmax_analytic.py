"""Closed-form BHmax from the easy-axis distribution alone.

For a magnet whose demagnetization curve is square, the maximum energy product
is limited by the remanence, BHmax = J_r^2 / (4 mu0). With uniaxial grains whose
easy axes make angles theta_i with the field axis, the remanence is the
volume-weighted projection J_r = J_s <cos theta>_V, giving

    BHmax = ( J_s <cos theta>_V )^2 / (4 mu0)

with no free parameters: J_s = 1.61 T is fixed for Nd2Fe14B and <cos theta>_V is
computed directly from the easy axes and grain volumes the GNN receives as node
features.

Run from the repo root::

    python analysis/bhmax_analytic.py
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from micromagnetic_gnn.plotstyle import C_ACCENT, C_REF, INK_2, STYLE, width, figure_dir

from micromagnetic_gnn.data_pipeline import NODE_FEATURE_MAP, _extract_node_features

HERE = figure_dir()
CACHE = Path("data_cache.pkl")

JS = 1.61                      # T, saturation polarisation of Nd2Fe14B
MU0 = 4e-7 * np.pi
BH_FULL = JS ** 2 / (4 * MU0)  # J/m^3, fully aligned limit

# Discrete bins beat a continuous ramp here: 1612 overlapping markers cannot carry
# a smooth magnitude scale, and the claim the figure supports is categorical
# ("deviations are confined to many-grain structures").
#
# Steps are viridis reversed, truncated to [0.15, 0.92]: two-hue yellow->blue, so
# neighbouring bins differ in hue as well as lightness. Reversed so that large N --
# the structures that carry the argument -- are the dark, salient end. Adjacent
# OKLCH lightness gaps are 0.11-0.13, comfortably past the 0.06 floor. The yellow
# end cannot clear the 2:1 contrast floor against white by colour alone, so the
# markers carry a dark outline instead of a white one.
N_BINS = [1, 50, 150, 500, 1500, 3500]
VIRIDIS_YB = ["#cae11f", "#52c569", "#1f988b", "#306a8e", "#463480"]
CMAP = ListedColormap(VIRIDIS_YB)
NORM = BoundaryNorm(N_BINS, CMAP.N)
plt.style.use(STYLE)

# Figure is built at its final printed size (single-column width), so the
# type below is the type the reader sees -- include it WITHOUT rescaling.
FIG_W, FIG_H = width(0.55), 2.95

EA = list(NODE_FEATURE_MAP["easy_axis"])
IV = NODE_FEATURE_MAP["volume"][0]


def collect():
    with open(CACHE, "rb") as f:
        structures = pickle.load(f)
    rows = []
    for gs in structures:
        X = _extract_node_features(gs).numpy()
        ea, V = X[:, EA], X[:, IV]
        ok = np.isfinite(V) & (V > 0) & np.isfinite(ea).all(1)
        ea, V = ea[ok], V[ok]
        if len(V) == 0:
            continue
        cos = np.abs(ea[:, 2]) / np.linalg.norm(ea, axis=1)
        m = re.search(r"AR=([0-9.]+)", gs.params["folder"])
        rows.append((float(gs.params["bhmax"]), (V * cos).sum() / V.sum(),
                     gs.params["no_grains"], gs.params["size_x"],
                     float(m.group(1)) if m else np.nan))
    return np.array(rows)


def main():
    print(f"Loading {CACHE} ...")
    R = collect()
    bh, cos_v, N, sx, ar = R.T
    pred = BH_FULL * cos_v ** 2
    rel = (bh - pred) / bh * 100
    r2 = 1 - np.var(bh - pred) / np.var(bh)

    print(f"  {len(R)} structures | J_s^2/(4 mu0) = {BH_FULL/1e3:.1f} kJ/m^3")
    print(f"  zero-parameter model: R2 = {r2:.4f}, MAPE = {np.mean(np.abs(rel)):.2f} %, "
          f"median rel. err = {np.median(rel):+.2f} %")
    print(f"  |rel| < 1 %: {(np.abs(rel) < 1).mean()*100:.1f} % of structures")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
    
    lim = [bh.min() / 1e3 * 0.98, BH_FULL / 1e3 * 1.02]
    ax.plot(lim, lim, color=C_REF, linewidth=0.9, linestyle="--", zorder=3)
    order = np.argsort(N)          # many-grain points drawn last, never buried
    sc = ax.scatter(pred[order] / 1e3, bh[order] / 1e3, c=N[order], cmap=CMAP,
                    norm=NORM, s=12, linewidths=0.3, edgecolors=INK_2,
                    alpha=0.92, zorder=2)
    ax.axhline(BH_FULL / 1e3, color=C_ACCENT, linewidth=1.0, linestyle=":", zorder=3)
    ax.text(lim[0] + 6, BH_FULL / 1e3 - 11, "max. theoretical limit",
            color=C_ACCENT, ha="left", va="top")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks([200, 300, 400, 500])
    ax.set_yticks([200, 300, 400, 500])
    ax.set_xlabel(r"analytical calculation   (kJ/m$^3$)")
    ax.set_ylabel(r"simulated $(BH)_\mathrm{max}$   (kJ/m$^3$)")
    ax.text(0.95, 0.26, f"$R^2$ = {r2:.4f}",
            transform=ax.transAxes, va="top", ha="right",
            color=C_REF,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#e6e5e1", linewidth=0.7))
    cb = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.03, spacing="uniform",
                      ticks=N_BINS)
    cb.set_label(r"grains, $N$")
    cb.ax.set_yticklabels([str(b) for b in N_BINS])
    cb.ax.tick_params(length=2, width=0.6)
    cb.outline.set_visible(False)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(HERE / f"bhmax_analytic{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  saved {HERE / 'bhmax_analytic'}.pdf (+ .png)")


if __name__ == "__main__":
    main()
