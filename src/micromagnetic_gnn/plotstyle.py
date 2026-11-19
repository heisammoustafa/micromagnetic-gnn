"""Shared figure style for the GNN permanent-magnet paper.

Send this file together with ``paper.mplstyle`` -- they must sit in the same
directory, since ``STYLE`` is resolved relative to this file. No dependencies
beyond matplotlib, so the pair can be copied anywhere to plot for the paper.

Usage
-----
    from micromagnetic_gnn.plotstyle import STYLE, width, C_TRAIN, C_TEST

    plt.style.use(STYLE)                               # type, grid, spines, saving
    fig, ax = plt.subplots(figsize=(width(0.7), 2.6))  # width from \textwidth
    ax.plot(x, y, color=C_TRAIN, label="training")
    ax.set_xlabel(r"applied field $\mu_0 H$ (T)")
    fig.savefig(figure_dir() / "name.pdf")

Rules the style file cannot enforce
-----------------------------------
* Author at the final printed size. ``width(f)`` gives the width in inches for a
  figure placed with ``\includegraphics[width=f\textwidth]``; that placement is
  then 1:1, so 9 pt here is 9 pt on the page. Choose the height freely by aspect.
  Never let LaTeX rescale a figure -- it rescales the labels with it.
* No title inside the figure; the LaTeX caption is the only description.
* Axis labels as *quantity (unit)*.
* Do not pass ``fontsize``, ``dpi``, ``bbox_inches``, grid or spine settings, or
  a background colour -- the style file owns them, and overriding is what makes
  figures stop matching.
* Use the named colours below rather than ``C0``/``C1``, so the same quantity is
  the same colour in every figure.
"""
import os
from pathlib import Path

STYLE = Path(__file__).with_name("paper.mplstyle")
FULL_W = 6.02          # in, = \textwidth of iopjournal.cls

# <repo>/src/micromagnetic_gnn/plotstyle.py -> <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]


def width(frac: float = 1.0) -> float:
    """Figure width in inches for a float placed at ``frac``\\textwidth."""
    return frac * FULL_W


def figure_dir() -> Path:
    """Directory every script writes its figures to, created on demand.

    ``$FIGURE_DIR`` wins if set, otherwise ``figures/`` at the repo root. One
    place for all output means a figure is never hidden in whichever model
    directory happened to produce it.
    """
    env = os.environ.get("FIGURE_DIR")
    out = Path(env).expanduser() if env else _REPO_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


# One colour per meaning, used identically in every figure.
C_TRAIN, C_VAL, C_TEST, C_GEN = "#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"
C_ALE, C_EPI = "#2a78d6", "#eb6834"
C_ACCENT, C_REF = "#eb6834", "#52514e"
INK, INK_2 = "#0b0b0b", "#52514e"

# Sequential ramp for magnitude (light -> dark, single hue).
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
