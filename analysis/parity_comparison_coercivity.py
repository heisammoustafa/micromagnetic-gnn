"""Combined parity figure: MC Dropout vs Deep Ensemble, test vs generalization.

Produces a single one-row figure (``figures/parity_comparison.pdf/.png``) of true vs
predicted coercivity with +-1 sigma_total error bars, four panels left to right:

    MC Dropout (single NLL model, stochastic passes)  test | ext
    Deep Ensemble (5 members, deterministic)          test | ext

Axis limits are shared per split so the two methods are directly comparable.
R2 and RMSE are written between each panel's title and its axes.

Run from the repo root::

    python analysis/parity_comparison_coercivity.py
    python analysis/parity_comparison_coercivity.py --mc-samples 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from torchmetrics import MeanSquaredError, R2Score

from micromagnetic_gnn.checkpoints import (
    DEFAULT_CACHE,
    load_coercivity_ensemble,
    load_coercivity_model,
)
from micromagnetic_gnn.inference import ensemble_predict, mc_predict
from micromagnetic_gnn.plotstyle import STYLE, C_REF, figure_dir, width


# Same type as coercivity_prediction.pdf (the paper style's 9 pt text, 8 pt ticks) and the
# same panel size relative to the text. That figure has three ~1.75 in panels on a 6 in
# canvas; four such panels need a canvas of ~8.4 in. Include at \textwidth (7.0 in, scale
# ~0.83), which prints the type at ~7.5 pt, within a few percent of the reference
# (scale 0.82). The title and statistics above the axes need the extra height.
FIG_W, FIG_H = 8.4, 3.3
TICK_FS, LABEL_FS, TITLE_FS, STATS_FS = 8, 9, 9, 9


def draw_panel(ax, y_true, y_mean, y_std, title: str, lims) -> None:
    ax.errorbar(
        y_true[:, 0], y_mean[:, 0], yerr=y_std[:, 0],
        linestyle="None", ecolor=C_REF, capsize=1.2, fmt="none", alpha=0.5,
        elinewidth=0.7, label=r"$\pm1\sigma$",
    )
    ax.plot(lims, lims, color=C_REF, lw=0.9, ls="--", zorder=3)
    r2 = float(R2Score()(torch.tensor(y_mean), torch.tensor(y_true)))
    rmse = float(MeanSquaredError(squared=False)(torch.tensor(y_mean),
                                                 torch.tensor(y_true)))
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    # Statistics sit between the title and the axes, clear of every data point.
    ax.text(0.5, 1.02, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=STATS_FS,
            linespacing=1.2)
    ax.set_title(title, fontsize=TITLE_FS, pad=3.2 * STATS_FS + 6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mc-samples", type=int, default=30,
                    help="MC-dropout passes for the single model (paper uses 30).")
    ap.add_argument("--member", type=int, default=15,
                    help="Ensemble member to use for MC Dropout (default: 15).")
    ap.add_argument("--ensemble-members", type=int, nargs="+", default=[15,16,17,18,19],
                    help="Member indices for the Deep Ensemble (default: 15-19).")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="Feature cache written by the data pipeline.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    single = load_coercivity_model(member=args.member, device=device)
    print(f"Loaded {single.path}")
    single.model.train() if args.mc_samples > 1 else single.model.eval()

    print("Loading ensemble members ...")
    members = load_coercivity_ensemble(members=args.ensemble_members, device=device)
    models = [m.model for m in members]

    _, _, test_ds, ext_ds = single.datasets(args.cache)
    datasets = {"test": test_ds, "ext": ext_ds}

    preds: dict[tuple[str, str], tuple] = {}
    for name, dataset in datasets.items():
        loader = single.loader(dataset)
        print(f"  predicting {name}: MC Dropout ({args.mc_samples} passes) ...")
        preds[("mcd", name)] = mc_predict(
            single.model, loader, single.scaler, args.mc_samples, device
        )
        print(f"  predicting {name}: Deep Ensemble ({len(models)} members) ...")
        y_mean, _, _, std_tot, y_true = ensemble_predict(
            models, loader, single.scaler, 1, device
        )
        preds[("ens", name)] = (y_mean, std_tot, y_true)

    # Shared limits per split (column) across both methods, incl. error bars.
    lims = {}
    for name in datasets:
        vals = []
        for method in ("mcd", "ens"):
            y_mean, y_std, y_true = preds[(method, name)]
            vals += [y_true.min(), y_true.max(),
                     (y_mean - y_std).min(), (y_mean + y_std).max()]
        lo, hi = float(np.min(vals)), float(np.max(vals))
        pad = 0.03 * (hi - lo)
        lims[name] = (lo - pad, hi + pad)

    method_titles = {"mcd": "MC Dropout", "ens": "Deep Ensemble"}
    split_titles = {"test": "test", "ext": "generalization"}
    plt.style.use(STYLE)
    plt.rcParams.update({"xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS,
                         "axes.grid": False})
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H))
    # One row: MC Dropout (test, generalization), then Deep Ensemble (test, generalization).
    panels = [(m, n) for m in ("mcd", "ens") for n in ("test", "ext")]
    for ax, (method, name) in zip(axes, panels):
        y_mean, y_std, y_true = preds[(method, name)]
        draw_panel(ax, y_true, y_mean, y_std,
                   f"{method_titles[method]}\n{split_titles[name]}", lims[name])
        ax.set_xlabel(r"simulated $H_\mathrm{C}$  (T)", fontsize=LABEL_FS)
    axes[0].set_ylabel(r"predicted $H_\mathrm{C}$  (T)", fontsize=LABEL_FS)

    fig.tight_layout(w_pad=0.6)
    out = figure_dir() / "parity_comparison.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out} (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
