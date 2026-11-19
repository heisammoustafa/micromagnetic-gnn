"""Demagnetization curves of two large ng= structures vs a typical sweep structure.

Reads the raw simulation output (``results/hys.log`` and ``results/hc.dat``) of
three hand-picked structures, so it needs ``ROM_Simulations/`` -- which is not
part of the repository. Point ``--data-root`` at it if it is not next to the
repo root.

Run from the repo root::

    python analysis/plot_hys.py
    python analysis/plot_hys.py --data-root /path/to/ROM_Simulations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from micromagnetic_gnn.plotstyle import figure_dir

# (folder, label, colour). Okabe-Ito CVD-safe hues, fixed order.
CURVES = [
    ("ng=3389_gb_t=2_AR=1.3_size_x=14787/numGrains_3389",
     "ng structure, N = 3389", "#0072B2"),
    ("ng=2902_gb_t=10_AR=3.0_size_x=54077/numGrains_2902",
     "ng structure, N = 2902", "#E69F00"),
    ("gb_t=10_AR=1.0_size_x=1063/numGrains_52",
     "typical sweep structure, N = 52", "#009E73"),
]

J_S = 1.61          # T, saturation polarization of Nd2Fe14B


def read_curve(folder: Path):
    """Return (H, J, Hc) for one simulation folder."""
    path = folder / "results" / "hys.log"
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}.\nROM_Simulations/ is not part of the repository; "
            "pass --data-root to point at it."
        )
    with open(path) as f:
        has_header = f.readline().lstrip().startswith("#")
    data = np.loadtxt(path, skiprows=1 if has_header else 0)
    if data.shape[1] == 2:          # jaxFMM (ng= batch): Hext, J
        h, j = data[:, 0], data[:, 1]
    else:                           # h2tools sweep: inp, Hext, Mx, My, Mz, M_H, ...
        h, j = data[:, 1], data[:, 5]
    hc = float(np.abs(np.loadtxt(folder / "results" / "hc.dat"))[0])
    return h, j, hc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("ROM_Simulations"),
                    help="Directory holding the simulation folders.")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)

    for folder, label, color in CURVES:
        h, j, hc = read_curve(args.data_root / folder)
        ax.plot(h, j, lw=2, color=color, label=f"{label}  ($\\mu_0H_c$ = {hc:.2f} T)")
        # mark the knee (coercive field) on the curve
        jk = np.interp(-hc, h[::-1], j[::-1])
        ax.plot(-hc, jk, "o", ms=8, color=color, mec="white", mew=1.5, zorder=5)

    # BHmax threshold from Skomski & Coey: the knee must lie left of -Js/2.
    ax.axvline(-J_S / 2, color="#666666", lw=1.2, ls=":")
    ax.annotate(f"$-J_s/2 = {-J_S / 2:.2f}$ T\n(BH)$_{{max}}$ threshold",
                xy=(-J_S / 2, -1.25), xytext=(0.28, -1.15), fontsize=9,
                color="#444444", ha="left",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.9))

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xlim(-6.5, 1.5)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel("applied field  $\\mu_0 H$  (T)")
    ax.set_ylabel("polarization  $J$  (T)")
    ax.set_title("ROM demagnetization curves (descending branch)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.grid(True, color="#eeeeee", lw=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out = figure_dir() / "hysteresis_ng.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
