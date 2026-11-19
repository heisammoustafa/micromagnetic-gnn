"""Saliency-score bar charts for any trained graph model.

Generates, from the checkpoint in ``results/``:

  <target>_saliency_score.pdf      per-feature saliency on the test set
  <target>_saliency_score_opt.pdf  same, after optimizing the easy axes

Scores are cached as JSON next to the figures so that re-plotting with
different styles does not require rerunning the analysis::

    python analysis/saliency_figures.py --target coercivity --member 15
    python analysis/saliency_figures.py --target coercivity --member 15 --plot-only
    python analysis/saliency_figures.py --target bhmax --no-opt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from micromagnetic_gnn.checkpoints import (
    DEFAULT_CACHE,
    load_bhmax_model,
    load_coercivity_model,
    load_switching_model,
)
from micromagnetic_gnn.explain import saliency_analysis
from micromagnetic_gnn.plotstyle import STYLE, figure_dir, width

LOADERS = {
    "coercivity": load_coercivity_model,
    "bhmax": load_bhmax_model,
    "switching": load_switching_model,
}

NODE_COLOR   = "#4878cf"
EDGE_COLOR   = "#d65f5f"
GLOBAL_COLOR = "#6acc65"


# Feature groups (colour coding); static, so --plot-only needs neither model nor data.
NODE_FEATURES = {"easy_axis", "log_volume", "edge_per_len", "diameq", "sphericity",
                 "dihanglemax", "dihanglemin", "dihangleav"}
EDGE_FEATURES = {"center_distance", "face_distance", "face_normal", "face_area",
                 "face_circumference"}
GLOBAL_FEATURES = {"gb_t"}
STATIC_FEATURE_MAPS = (NODE_FEATURES, EDGE_FEATURES, GLOBAL_FEATURES)

# Bar labels use the quantity names of the paper's feature table (Table: features).
FEATURE_NAMES = {
    "easy_axis":          "easy axis",
    "log_volume":         "logarithmic grain volume",
    "edge_per_len":       "dimensionless edge-length ratio",
    "diameq":             "equivalent diameter",
    "sphericity":         "sphericity",
    "dihanglemax":        "maximum dihedral angle",
    "dihanglemin":        "minimum dihedral angle",
    "dihangleav":         "average dihedral angle",
    "center_distance":    "centroid displacement",
    "face_distance":      "centroid to face-center displacement",
    "face_normal":        "unit normal of shared face",
    "face_area":          "log shared-face area",
    "face_circumference": "log shared-face circumference",
    "gb_t":               "grain boundary thickness",
}

# Drawn at the printed size (one column, \includegraphics[width=\linewidth]) so the
# 9 pt type below is 9 pt on the page. elsarticle 5p: column ~ 246 pt.
FIG_W, FIG_H = 3.4, 3.7   # in
FONT = 9
STYLE_OVERRIDES = {"axes.grid": False, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
                   "legend.fontsize": FONT, "font.size": FONT}


def _label_color(bar_color: str) -> str:
    """White on dark bars, dark grey on light ones."""
    r, g, b = matplotlib.colors.to_rgb(bar_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.5 else "#333333"


def _bar_chart(scores: dict, feature_maps: dict, out: Path) -> None:
    node_map, edge_map, global_map = feature_maps
    def color(f: str) -> str:
        if f in node_map:   return NODE_COLOR
        if f in edge_map:   return EDGE_COLOR
        if f in global_map: return GLOBAL_COLOR
        raise ValueError(f"Unknown feature: {f}")

    names  = list(scores.keys())
    values = list(scores.values())
    colors = [color(f) for f in names]

    seen: set[str] = set()
    label_map = {NODE_COLOR: "node feature", EDGE_COLOR: "edge feature",
                 GLOBAL_COLOR: "global feature"}
    labels = []
    for c in colors:
        lbl = label_map[c]
        labels.append(lbl if lbl not in seen else f"_{lbl}")
        seen.add(lbl)

    with plt.style.context([STYLE, STYLE_OVERRIDES]):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        x = np.arange(len(names))
        ax.bar(x, values, color=colors, label=labels)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_xticks([])                       # names are written on the bars
        # Fixed layout: a figure-level legend above the axes does not shrink them.
        fig.subplots_adjust(left=0.15, right=0.985, bottom=0.02, top=0.92)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3,
                   frameon=False, columnspacing=1.0, handletextpad=0.4,
                   borderaxespad=0.2)
        fig.canvas.draw()

        # Vertical names: inside the bar (in the bar's contrast colour) when it is
        # tall enough, otherwise just above it. y-limit grows until all fit.
        height = ax.get_window_extent().height          # px
        renderer = fig.canvas.get_renderer()
        texts = [ax.text(i, 0, FEATURE_NAMES[n], rotation=90, fontsize=FONT,
                         ha="center", va="bottom") for i, n in enumerate(names)]
        length = [t.get_window_extent(renderer).height for t in texts]   # px
        pad = 0.02 * height
        top = 1.05 * max(values)
        for _ in range(50):
            need = top
            for v, L in zip(values, length):
                if v / top * height < L + 2 * pad:      # label goes above the bar
                    room = max(height - L - pad, 0.1 * height)
                    need = max(need, v * height / room)
            if need <= top * (1 + 1e-9):
                break
            top = need
        ax.set_ylim(0, top)
        for t, v, L, c in zip(texts, values, length, colors):
            inside = v / top * height >= L + 2 * pad
            t.set_y(pad / height * top if inside else v + pad / height * top)
            t.set_color(_label_color(c) if inside else "#333333")

        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"))
        plt.close(fig)
    print(f"  saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=list(LOADERS), default="coercivity")
    ap.add_argument("--member", type=int, default=None,
                    help="Ensemble member index (e.g. 15 loads coercivity_m15.tar).")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--no-opt", action="store_true",
                    help="Skip the optimized-easy-axis saliency figure.")
    ap.add_argument("--opt-steps", type=int, default=1500,
                    help="Easy-axis optimization steps (default 1500).")
    ap.add_argument("--plot-only", action="store_true",
                    help="Reload cached scores from JSON and regenerate figures only.")
    args = ap.parse_args()

    figures = figure_dir()
    scores_path     = figures / f"{args.target}_saliency_scores.json"
    scores_opt_path = figures / f"{args.target}_saliency_scores_opt.json"

    if args.plot_only:
        # --- reload cached scores and replot, no model or data needed ---
        if not scores_path.exists():
            print(f"ERROR: no cached scores at {scores_path}. Run without --plot-only first.")
            return 1
        scores = json.loads(scores_path.read_text())
        feature_maps = STATIC_FEATURE_MAPS
        _bar_chart(scores, feature_maps, figures / f"{args.target}_saliency_score.pdf")
        if not args.no_opt:
            if not scores_opt_path.exists():
                print(f"WARNING: no cached opt scores at {scores_opt_path}, skipping opt figure.")
            else:
                scores_opt = json.loads(scores_opt_path.read_text())
                _bar_chart(scores_opt, feature_maps,
                           figures / f"{args.target}_saliency_score_opt.pdf")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    load_kwargs = {"device": device}
    if args.target == "coercivity" and args.member is not None:
        load_kwargs["member"] = args.member
    trained = LOADERS[args.target](**load_kwargs)
    print(f"Loaded {trained.path}")

    _, _, test_ds, _ = trained.datasets(args.cache)
    trained.model.eval()
    feature_maps = (test_ds.node_feature_map, test_ds.edge_feature_map,
                    test_ds.global_feature_map)

    # --- standard saliency (raw easy axes) ---
    print("  Computing saliency on test set ...")
    sa = saliency_analysis(trained.model, test_ds)
    scores_path.write_text(json.dumps(sa.scores))
    _bar_chart(sa.scores, feature_maps, figures / f"{args.target}_saliency_score.pdf")

    if args.no_opt:
        return 0

    # --- optimized-easy-axis saliency ---
    from micromagnetic_gnn.trainer import opt_easy_axis
    print(f"  Optimizing easy axes ({args.opt_steps} steps) ...")
    data_opt = [
        # lr pinned to the value used for the paper; the trainer default is now 1e-2
        opt_easy_axis(trained.model.to(device), d.clone().to(device), lr=1e-3, steps=args.opt_steps)
        for d in test_ds
    ]
    sa_opt = saliency_analysis(trained.model, data_opt)
    scores_opt_path.write_text(json.dumps(sa_opt.scores))
    _bar_chart(sa_opt.scores, feature_maps,
               figures / f"{args.target}_saliency_score_opt.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
