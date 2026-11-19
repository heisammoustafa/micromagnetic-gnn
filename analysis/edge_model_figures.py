"""Results figures for the edge-centric models (Figs. 5, 9 and 10 of the paper).

Regenerates, from the checkpoints in ``results/``:

  bhmax_prediction.pdf      parity on train / test / generalization, MC dropout
  coercivity_prediction.pdf the same for the coercive field
  coercivity_gen.pdf, bhmax_gen.pdf
                            absolute error on the test (x) and generalization (o)
                            sets against grain count and domain edge length,
                            one colour scale, linear axes, no grid

Both models use the split of ``dataset.py`` and the scale-free size descriptors.

Run from the repo root::

    python analysis/edge_model_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch

from micromagnetic_gnn.checkpoints import (
    DEFAULT_CACHE,
    load_bhmax_model,
    load_coercivity_model,
)
from micromagnetic_gnn.dataset import default_gs_data
from micromagnetic_gnn.inference import mc_predict
from micromagnetic_gnn.plotstyle import STYLE, C_REF, C_TRAIN, figure_dir, width

MODELS = {
    "bhmax": dict(load=load_bhmax_model, label=r"$(BH)_\mathrm{max}$",
                  unit=r"kJ/m$^3$", scale=1e-3, out="bhmax_prediction.pdf"),
    "coercivity": dict(load=load_coercivity_model, label=r"$H_\mathrm{C}$",
                       unit="T", scale=1.0, out="coercivity_prediction.pdf"),
}


def parity(ax, y_true, y_mean, y_std, title: str, text_loc: str = "upper left",
           errorbars: bool = False) -> None:
    lo = float(min(y_true.min(), y_mean.min()))
    hi = float(max(y_true.max(), y_mean.max()))
    pad = 0.04 * (hi - lo)
    lims = (lo - pad, hi + pad)
    if errorbars:
        ax.errorbar(y_true, y_mean, yerr=y_std, linestyle="None", fmt="none",
                    ecolor=C_REF, elinewidth=0.7, capsize=1.2, alpha=0.5)
    ax.scatter(y_true, y_mean, s=9, color=C_TRAIN, linewidths=0, zorder=3)
    ax.plot(lims, lims, color=C_REF, ls="--", lw=0.9, zorder=4)
    r2 = 1 - np.sum((y_true - y_mean) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y_true - y_mean) ** 2)))
    x = 0.96 if "right" in text_loc else 0.04
    y = 0.04 if "lower" in text_loc else 0.96
    va = "bottom" if "lower" in text_loc else "top"
    ha = "right" if "right" in text_loc else "left"
    ax.text(x, y, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3g}",
            transform=ax.transAxes, va=va, ha=ha,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e6e5e1"))
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_title(title)


def generalization_vs_size_2d(err_gen, ext_s, err_test, test_s, out: Path,
                              cbar_label: str) -> None:
    """Absolute error against grain count and domain edge length.

    Test structures are drawn as crosses and generalization structures as circles,
    both coloured by the same logarithmic error scale (linear position axes).
    """
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    def axes_of(structs):
        return (np.array([g.params["no_grains"] for g in structs], float),
                np.array([g.params["size_x"] for g in structs], float) / 1000)

    n_gen, size_gen = axes_of(ext_s)
    n_test, size_test = axes_of(test_s)
    both = np.concatenate([err_gen, err_test])
    norm = mcolors.LogNorm(vmin=both[both > 0].min(), vmax=both.max())

    with plt.rc_context({"axes.grid": False}):
        fig, ax = plt.subplots(figsize=(3.4, 2.9))   # one column, printed at scale 1
        ax.scatter(n_test, size_test, c=err_test, s=16, marker="x", linewidths=0.9,
                   norm=norm, cmap="viridis")
        sc = ax.scatter(n_gen, size_gen, c=err_gen, s=18, marker="o", linewidths=0,
                        norm=norm, cmap="viridis")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xlabel("grains per structure, $N$")
        ax.set_ylabel(r"domain edge length  ($\mathrm{\mu}$m)")
        # Legend above the axes: the data fill the plot area.
        ax.legend(handles=[
            Line2D([], [], marker="x", ls="None", color=C_REF, markersize=4,
                   markeredgewidth=0.9, label="test set"),
            Line2D([], [], marker="o", ls="None", color=C_REF, markersize=4,
                   markeredgewidth=0, label="generalization set"),
        ], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
            borderaxespad=0.3, handletextpad=0.2, columnspacing=1.2)
        fig.tight_layout()
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"))
        plt.close(fig)
    print(f"  saved {out}")


def generalization_vs_size(err, train_s, ext_s, out: Path) -> None:
    """Absolute generalization error against the two size axes."""
    n_grains = np.array([g.params["no_grains"] for g in ext_s], float)
    size_x = np.array([g.params["size_x"] for g in ext_s], float) / 1000
    max_n = max(g.params["no_grains"] for g in train_s)
    max_size = max(g.params["size_x"] for g in train_s) / 1000

    fig, axes = plt.subplots(1, 2, figsize=(width(0.8), 2.4))
    for ax, x, limit, xlabel in (
        (axes[0], n_grains, max_n, "grains per structure, $N$"),
        (axes[1], size_x, max_size, r"domain edge length  ($\mathrm{\mu}$m)"),
    ):
        ax.scatter(x, err, s=12, color=C_TRAIN, linewidths=0)
        ax.axvline(limit, color=C_REF, ls="--", lw=0.9, label="training support")
        ax.set_xlabel(xlabel)
        ax.set_xscale("log")
        ax.legend(loc="upper left")
    axes[0].set_ylabel(r"$|\Delta H_\mathrm{C}|$  (T)")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mc-samples", type=int, default=20,
                    help="Monte-Carlo dropout forward passes.")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="Feature cache written by the data pipeline.")
    ap.add_argument("--target", choices=["bhmax", "coercivity"], default=None,
                    help="Generate only this target's figure (default: both).")
    ap.add_argument("--member", type=int, default=None,
                    help="Coercivity ensemble member index (e.g. 15 loads coercivity_m15.tar).")
    ap.add_argument("--errorbars", action="store_true",
                    help="Draw the MC-dropout predictive std as error bars (off by default).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    figures = figure_dir()

    for name, cfg in MODELS.items():
        if args.target is not None and name != args.target:
            continue
        load_kwargs = {"device": device}
        if name == "coercivity" and args.member is not None:
            load_kwargs["member"] = args.member
        trained = cfg["load"](**load_kwargs)
        print(f"Loaded {trained.path}")
        train_ds, _, test_ds, ext_ds = trained.datasets(args.cache)
        splits = {"train": train_ds, "test": test_ds, "generalization": ext_ds}

        trained.model.train()            # dropout active for MC sampling
        plt.style.use(STYLE)
        fig, axes = plt.subplots(1, 3, figsize=(width(1.0), 2.5))
        preds = {}
        for ax, (split_name, dataset) in zip(axes, splits.items()):
            loader = trained.loader(dataset)
            y_mean, y_std, y_true = mc_predict(
                trained.model, loader, trained.scaler, args.mc_samples, device
            )
            scale = cfg["scale"]
            y_mean = y_mean[:, 0] * scale
            y_std = y_std[:, 0] * scale
            y_true = y_true[:, 0] * scale
            preds[split_name] = (y_true, y_mean, y_std)
            loc = "lower right" if (split_name == "generalization" and name == "bhmax") else "upper left"
            parity(ax, y_true, y_mean, y_std, split_name, text_loc=loc,
                   errorbars=args.errorbars)
            r2 = 1 - np.sum((y_true - y_mean) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
            print(f"  {name:<11} {split_name:<15} R2={r2:>7.3f}")
        for ax in axes:
            ax.set_xlabel(f"simulated {cfg['label']}  ({cfg['unit']})")
        axes[0].set_ylabel(f"predicted {cfg['label']}  ({cfg['unit']})")
        fig.tight_layout()
        out = figures / cfg["out"]
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"))
        plt.close(fig)
        print(f"  saved {out}")

        # --- generalization error vs structure size (coercivity_gen / bhmax_gen) ---
        y_true, y_mean, _ = preds["generalization"]
        yt_test, ym_test, _ = preds["test"]
        _, _, test_s, ext_s = default_gs_data(cache_path=args.cache)
        stem = {"coercivity": "coercivity_gen", "bhmax": "bhmax_gen"}[name]
        label = (r"$|\Delta H_\mathrm{C}|$  (T)" if name == "coercivity"
                 else r"$|\Delta (BH)_\mathrm{max}|$  (kJ/m$^3$)")
        generalization_vs_size_2d(np.abs(y_true - y_mean), ext_s,
                                  np.abs(yt_test - ym_test), test_s,
                                  figures / f"{stem}.pdf", label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
