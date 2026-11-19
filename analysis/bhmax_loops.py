"""Why Eq. (bhmax-analytic) is an upper bound: demagnetization curves and (BH)max.

The closed form assumes a rectangular second-quadrant curve, so it returns the
remanence-limited product J_r^2/(4 mu0). This figure shows the assumption holding
and failing:

* (a) the parameter-free closed form against the simulated energy product for all
      1612 structures, with the two structures of (b) and (c) marked.
* (b) a structure where simulated and analytical (BH)max agree to <0.01 %.
      Reversal is collective, the curve stays square, and the largest inscribed
      B-H rectangle is the analytical one.
* (c) a many-grain structure over-predicted by ~55 %. Grains reverse
      progressively over a range of applied fields, the curve shears, and no
      rectangle of the analytical size fits underneath it.

Panels (b) and (c) draw the two rectangles whose areas *are* the two (BH)max
values, so the discrepancy is a visible difference in area, not a pair of numbers.

Fields are internal, H_i = H - M/3, matching ``grain_structure.compute_bhmax``.

Needs ``data_cache.pkl`` (for <cos theta>_V) and ``ROM_Simulations/`` (for the
raw curves). Run from the repo root::

    python analysis/bhmax_loops.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import BoundaryNorm, ListedColormap

from micromagnetic_gnn.plotstyle import (C_ACCENT, C_REF, C_TRAIN, INK, INK_2,
                                         STYLE, figure_dir, width)
from micromagnetic_gnn.data_pipeline import NODE_FEATURE_MAP, _extract_node_features

HERE = figure_dir()
CACHE = Path("data_cache.pkl")

JS = 1.61                       # T, saturation polarisation of Nd2Fe14B
MU0 = 4e-7 * np.pi
BH_FULL = JS ** 2 / (4 * MU0)   # J/m^3, fully aligned limit

EA = list(NODE_FEATURE_MAP["easy_axis"])
IV = NODE_FEATURE_MAP["volume"][0]

# (folder, n_grains, panel tag). Picked by |relative deviation| over the whole
# dataset -- see analysis/bhmax_analytic.py for the deviation itself.
PANELS = [
    ("ROM_Simulations/gb_t=2_AR=5.9_size_x=1393", 159, "b"),
    ("ROM_Simulations/ng=652_gb_t=2_AR=2.2_size_x=57287", 652, "c"),
]

# Same discrete bins and steps as analysis/bhmax_analytic.py -- see the comment
# there for why the ramp is truncated and reversed.
N_BINS = [1, 50, 150, 500, 1500, 3500]
VIRIDIS_YB = ["#cae11f", "#52c569", "#1f988b", "#306a8e", "#463480"]
CMAP = ListedColormap(VIRIDIS_YB)
NORM = BoundaryNorm(N_BINS, CMAP.N)

plt.style.use(STYLE)
FIG_W, FIG_H = width(1.0), 2.35


def cos_theta_v(gs) -> float:
    """Volume-weighted <cos theta> over the grains of one structure."""
    X = _extract_node_features(gs).numpy()
    ea, V = X[:, EA], X[:, IV]
    ok = np.isfinite(V) & (V > 0) & np.isfinite(ea).all(1)
    ea, V = ea[ok], V[ok]
    cos = np.abs(ea[:, 2]) / np.linalg.norm(ea, axis=1)
    return float((V * cos).sum() / V.sum())


def read_curve(folder: str, n: int):
    """Internal-field descending branch (mu0*H_i, J, B_i), all in tesla.

    Mirrors ``grain_structure.compute_bhmax`` exactly -- same column choice and
    the same N = 1/3 demagnetizing shear -- so the areas drawn here are the
    ``bhmax`` values carried in the dataset and plotted in Fig. bhmax-analytic.
    """
    path = Path(folder) / f"numGrains_{n}" / "results" / "hys.log"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. ROM_Simulations/ is not in the repository.")
    d = np.loadtxt(path, comments="#")
    h, m = (d[:, 0], d[:, 1]) if d.shape[1] == 2 else (d[:, 1], d[:, 4])
    hi = h - m / 3.0
    return hi, m, hi + m


def energy_product(hi, bi):
    """Peak |B H| on the internal curve, and the (H, B) where it is attained."""
    bh = -bi * hi / MU0
    k = int(np.argmax(bh))
    return float(bh[k]), float(hi[k]), float(bi[k])


def parity(fig, ax, structures, marked):
    """Closed form vs simulation for the whole dataset; marks the two examples."""
    bh, pred, N = [], [], []
    for gs in structures:
        cv = cos_theta_v(gs)
        bh.append(float(gs.params["bhmax"]))
        pred.append(BH_FULL * cv ** 2)
        N.append(int(gs.params["no_grains"]))
    bh, pred, N = np.array(bh), np.array(pred), np.array(N)
    r2 = 1 - np.var(bh - pred) / np.var(bh)

    lim = [bh.min() / 1e3 * 0.98, BH_FULL / 1e3 * 1.02]
    ax.plot(lim, lim, color=C_REF, lw=0.9, ls="--", zorder=3)
    order = np.argsort(N)          # many-grain points drawn last, never buried
    sc = ax.scatter(pred[order] / 1e3, bh[order] / 1e3, c=N[order], cmap=CMAP,
                    norm=NORM, s=10, linewidths=0.25, edgecolors=INK_2,
                    alpha=0.92, zorder=2)
    #ax.axhline(BH_FULL / 1e3, color=C_ACCENT, lw=1.0, ls=":", zorder=3,
    #           label="fully aligned limit")

    # Tie the panels together: show where (b) and (c) sit in the population.
    for tag, (bh_m, pred_m) in marked.items():
        ax.plot(pred_m / 1e3, bh_m / 1e3, "o", ms=6, mfc="none", mec=INK,
                mew=1.1, zorder=6)
        ax.annotate(f"({tag})", (pred_m / 1e3, bh_m / 1e3),
                    textcoords="offset points", xytext=(-8, -11), color=INK,
                    ha="right", fontsize=7, zorder=6)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([200, 300, 400, 500]); ax.set_yticks([200, 300, 400, 500])
    ax.set_xlabel(r"analytical   (kJ/m$^3$)", fontsize=7)
    ax.set_ylabel(r"simulated $(BH)_\mathrm{max}$   (kJ/m$^3$)", fontsize=7)
    ax.text(0.05, 0.96, "(a)", transform=ax.transAxes, va="top", ha="left")
    ax.text(0.95, 0.06, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
            va="bottom", ha="right", color=C_REF, fontsize=6)

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, ticks=N_BINS)
    cb.set_label(r"grains, $N$", fontsize=6)
    cb.ax.set_yticklabels([str(b) for b in N_BINS], fontsize=6)
    cb.ax.tick_params(length=2, width=0.6)
    cb.outline.set_visible(False)
    print(f"  parity: {len(bh)} structures, R2 = {r2:.4f}")


def panel(ax, hi, m, bi, cos_v, n, tag, xlim, ylim):
    bh_sim, h_sim, b_sim = energy_product(hi, bi)
    jr_ana = JS * cos_v                         # remanence the closed form assumes
    bh_ana = jr_ana ** 2 / (4 * MU0)
    h_ana, b_ana = -jr_ana / 2, jr_ana / 2      # optimum rectangle of a square loop

    # The rectangular loop the closed form assumes: J = J_r right up to reversal.
    ax.plot([-jr_ana, 0], [0, jr_ana], color=C_ACCENT, lw=1.0, ls=":", zorder=2,
            label="square-loop assumption")
    ax.add_patch(plt.Rectangle((h_ana, 0), -h_ana, b_ana, facecolor=C_ACCENT,
                               alpha=0.18, edgecolor="none", zorder=1))
    # Outline above the blue patch, so it stays visible where the two coincide.
    ax.add_patch(plt.Rectangle((h_ana, 0), -h_ana, b_ana, facecolor="none",
                               edgecolor=C_ACCENT, lw=1.0, ls="--", zorder=5,
                               label=r"analytical $(BH)_\mathrm{max}$"))
    ax.add_patch(plt.Rectangle((h_sim, 0), -h_sim, b_sim, facecolor=C_TRAIN,
                               alpha=0.25, edgecolor=C_TRAIN, lw=1.0, zorder=3,
                               label=r"simulated $(BH)_\mathrm{max}$"))

    q2 = (hi <= 0) & (bi >= 0)
    ax.plot(hi[q2], bi[q2], color=INK, lw=1.5, zorder=4, label="simulated curve")
    ax.plot([h_sim], [b_sim], "o", ms=4.5, color=C_TRAIN, mec="white", mew=0.8, zorder=6)

    ax.axhline(0, color=C_REF, lw=0.6, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Equal aspect: the analytical optimum spans J_r/2 in mu0*H and J_r/2 in B, so
    # on a tesla-per-tesla scale it is a square. The simulated one is not, and the
    # difference in shape is part of what the panel shows.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\mu_0 H_\mathrm{i}$   (T)", fontsize=7)

    rel = (bh_sim - bh_ana) / bh_sim * 100
    ax.text(0.5, 1.02, f"({tag})   $N = {n}$,   {rel:+.1f} %",
            transform=ax.transAxes, va="bottom", ha="center", fontsize=7)
    # The two energy products, written on the rectangles whose areas they are.
    ax.text(h_ana * 0.92, b_ana * 1.04, f"{bh_ana/1e3:.0f}", color=C_ACCENT,
            ha="left", va="bottom", fontsize=6.5, zorder=8)
    ax.text(h_sim / 2, b_sim * 0.40, f"{bh_sim/1e3:.0f}", color=C_TRAIN,
            ha="center", va="center", fontsize=6.5, zorder=8)
    return bh_sim, bh_ana


def main() -> int:
    print(f"Loading {CACHE} ...")
    with open(CACHE, "rb") as f:
        structures = pickle.load(f)
    by_folder = {str(gs.params["folder"]): gs for gs in structures}

    curves = []
    for folder, n, tag in PANELS:
        gs = by_folder.get(folder)
        if gs is None:
            raise SystemExit(f"{folder} not in {CACHE}")
        hi, m, bi = read_curve(folder, n)
        curves.append((hi, m, bi, cos_theta_v(gs), n, tag))

    # One span for both zoom panels, used for x and y alike so the axes carry the
    # same tesla-per-inch scale and the analytical optimum reads as a square.
    span = 1.06 * max(max(abs(h[(h <= 0) & (b >= 0)].min()),
                          b[(h <= 0) & (b >= 0)].max(), JS * c)
                      for h, _, b, c, _, _ in curves)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), dpi=300,
                             gridspec_kw=dict(width_ratios=[1.30, 1, 1]))
    marked = {}
    for hi, m, bi, cos_v, n, tag in curves:
        marked[tag] = (energy_product(hi, bi)[0], BH_FULL * cos_v ** 2)
    parity(fig, axes[0], structures, marked)

    for ax, (hi, m, bi, cos_v, n, tag) in zip(axes[1:], curves):
        bh_sim, bh_ana = panel(ax, hi, m, bi, cos_v, n, tag,
                               (-span, 0), (0, span))
        print(f"  {tag}: N={n:5d}  simulated {bh_sim/1e3:6.1f}  "
              f"analytical {bh_ana/1e3:6.1f} kJ/m^3")
    axes[1].set_ylabel(r"$B_\mathrm{i}$   (T)", fontsize=7)
    axes[2].tick_params(labelleft=False)

    h_a, l_a = axes[0].get_legend_handles_labels()
    h_b, l_b = axes[1].get_legend_handles_labels()
    h_, l_ = h_a + h_b, l_a + l_b
    fig.legend(h_, l_, loc="lower center", ncol=5, frameon=False,
               handlelength=1.8, fontsize="small", bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(HERE / f"bhmax_loops{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {HERE / 'bhmax_loops'}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
