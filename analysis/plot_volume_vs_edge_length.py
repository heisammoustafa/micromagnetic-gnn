"""Volume vs total_edge_length: why Pearson and Spearman disagree.

Both are default node features of ``dataset.py::default_datasets``. Pooled over
all grains they give r = 0.74 but rho = 0.98 -- the relation is a clean power
law, not a linear one, so Pearson understates a near-deterministic dependence.

Three panels:
  A  linear axes -- what Pearson sees
  B  log-log axes + fitted power law -- what Spearman sees
  C  within-structure coefficients -- the pooled numbers mix structures whose
     grain scales differ by ~6 orders of magnitude

Run from the repo root::

    python analysis/plot_volume_vs_edge_length.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.stats import spearmanr

from micromagnetic_gnn.data_pipeline import NODE_FEATURE_MAP, _extract_node_features

from micromagnetic_gnn.plotstyle import figure_dir

HERE = figure_dir()
CACHE = Path("data_cache.pkl")
RNG = np.random.default_rng(0)
MAX_SAMPLES = 300_000
MIN_GRAINS = 30  # per-structure coefficients need enough grains to be meaningful

# Sequential blue ramp (steps 100 -> 700) for the density hexbins.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DENSITY_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e6e5e1"

I_VOL = NODE_FEATURE_MAP["volume"][0]
I_TEL = NODE_FEATURE_MAP["total_edge_length"][0]


def collect():
    """Pooled (volume, total_edge_length) plus per-structure coefficients."""
    with open(CACHE, "rb") as f:
        structures = pickle.load(f)
    print(f"  {len(structures)} structures")

    pooled, per_structure = [], []
    for gs in structures:
        X = _extract_node_features(gs).numpy()
        v, t = X[:, I_VOL], X[:, I_TEL]
        ok = np.isfinite(v) & np.isfinite(t) & (v > 0) & (t > 0)
        v, t = v[ok], t[ok]
        if len(v) == 0:
            continue
        pooled.append(np.column_stack([v, t]))
        if len(v) >= MIN_GRAINS:
            per_structure.append((np.corrcoef(v, t)[0, 1], spearmanr(v, t).statistic))

    return np.concatenate(pooled), np.asarray(per_structure)


def style(ax):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def annotate(ax, lines, loc="upper left", y=0.96):
    ax.text(0.04 if loc == "upper left" else 0.96, y, "\n".join(lines),
            transform=ax.transAxes, va="top",
            ha="left" if loc == "upper left" else "right",
            fontsize=8, color=INK_2, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fcfcfb",
                      edgecolor=GRID, linewidth=0.8))


def main():
    print(f"Loading {CACHE} ...")
    pooled, per_structure = collect()
    print(f"  {len(pooled)} grains, {len(per_structure)} structures with >= {MIN_GRAINS} grains")

    if len(pooled) > MAX_SAMPLES:
        pooled = pooled[RNG.choice(len(pooled), MAX_SAMPLES, replace=False)]
    v, t = pooled[:, 0], pooled[:, 1]

    r = np.corrcoef(v, t)[0, 1]
    rho = spearmanr(v, t).statistic
    slope, intercept = np.polyfit(np.log10(t), np.log10(v), 1)
    r_log = np.corrcoef(np.log10(t), np.log10(v))[0, 1]
    print(f"  pooled: Pearson r = {r:.3f}, Spearman rho = {rho:.3f}")
    print(f"  power-law fit: V = 10^{intercept:.2f} * L^{slope:.2f}  (r on log-log = {r_log:.3f})")
    print(f"  within-structure: Pearson {per_structure[:, 0].mean():+.2f} +/- {per_structure[:, 0].std():.2f}, "
          f"Spearman {per_structure[:, 1].mean():+.2f} +/- {per_structure[:, 1].std():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    # --- A: linear axes ------------------------------------------------------
    ax = axes[0]
    style(ax)
    hb = ax.hexbin(t, v, gridsize=60, cmap=DENSITY_CMAP, norm=LogNorm(), linewidths=0, zorder=2)
    ax.set_xlabel("total_edge_length  (nm)", fontsize=9, color=INK_2)
    ax.set_ylabel("volume  (nm$^3$)", fontsize=9, color=INK_2)
    ax.set_title("(a)  Linear axes", fontsize=10, color=INK, loc="left", pad=10)
    annotate(ax, [f"Pearson $r$ = {r:.2f}", f"Spearman $\\rho$ = {rho:.2f}"])
    cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("grains per bin", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)

    # --- B: log-log + power-law fit -----------------------------------------
    ax = axes[1]
    style(ax)
    hb = ax.hexbin(t, v, gridsize=60, xscale="log", yscale="log",
                   cmap=DENSITY_CMAP, norm=LogNorm(), linewidths=0, zorder=2)
    xs = np.logspace(np.log10(t.min()), np.log10(t.max()), 100)
    ax.plot(xs, 10 ** intercept * xs ** slope, color=SERIES_2, linewidth=2,
            zorder=3, label=f"fit: $V \\propto L^{{{slope:.2f}}}$")
    ax.set_xlabel("total_edge_length  (nm)", fontsize=9, color=INK_2)
    ax.set_ylabel("volume  (nm$^3$)", fontsize=9, color=INK_2)
    ax.set_title("(b)  Logarithmic axes", fontsize=10, color=INK, loc="left", pad=10)
    leg = ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor="#fcfcfb", edgecolor=GRID)
    leg.get_frame().set_linewidth(0.8)
    for txt in leg.get_texts():
        txt.set_color(INK_2)
    annotate(ax, [f"Pearson $r$ on log–log = {r_log:.2f}",
                  f"Spearman $\\rho$ = {rho:.2f}  (scale-free)"])
    cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("grains per bin", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)

    # --- C: within-structure coefficients ------------------------------------
    ax = axes[2]
    style(ax)
    bins = np.linspace(0.6, 1.0, 40)
    ax.set_xlim(0.6, 1.0)
    ax.hist(per_structure[:, 0], bins=bins, color=SERIES_1, alpha=0.85,
            label="Pearson $r$", zorder=2)
    ax.hist(per_structure[:, 1], bins=bins, color=SERIES_2, alpha=0.75,
            label="Spearman $\\rho$", zorder=3)
    for col, c in ((0, SERIES_1), (1, SERIES_2)):
        ax.axvline(per_structure[:, col].mean(), color=c, linewidth=2,
                   linestyle=":", zorder=4)
    ax.set_xlabel("coefficient within a single structure", fontsize=9, color=INK_2)
    ax.set_ylabel("structures", fontsize=9, color=INK_2)
    ax.set_title("(c)  Within individual structures",
                 fontsize=10, color=INK, loc="left", pad=10)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="#fcfcfb", edgecolor=GRID)
    leg.get_frame().set_linewidth(0.8)
    for txt in leg.get_texts():
        txt.set_color(INK_2)
    annotate(ax, [f"mean $r$ = {per_structure[:, 0].mean():.2f} $\\pm$ {per_structure[:, 0].std():.2f}",
                  f"mean $\\rho$ = {per_structure[:, 1].mean():.2f} $\\pm$ {per_structure[:, 1].std():.2f}",
                  f"$n$ = {len(per_structure)} structures"], y=0.74)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(HERE / f"volume_vs_edge_length{ext}", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {HERE / 'volume_vs_edge_length'}.pdf (+ .png)")


if __name__ == "__main__":
    main()
