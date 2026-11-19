"""Deep-ensemble inference + uncertainty for the coercivity model.

Loads every ``results/coercivity_m*.tar`` member (trained via
``slurm/train_edge_uq.slurm`` / ``main.py --member K``) and combines them as a
deep ensemble; see :func:`micromagnetic_gnn.inference.ensemble_predict` for the
decomposition. Unlike MC dropout, member disagreement grows on out-of-
distribution inputs, so the epistemic term should actually flag the
generalization split.

With ``--mc-samples > 1`` each member additionally runs MC dropout and the
within-member spread is added to the epistemic term; the default (1) is the
standard deterministic deep ensemble.

Per split it writes ensemble versions of all three diagnostics:
``parity_ens_<split>``, ``calibration_ens_<split>``, ``uncertainty_ens_<split>``.

Run from the repo root (after the ensemble finished training)::

    python analysis/ensemble_coercivity.py
    python analysis/ensemble_coercivity.py --split ext --mc-samples 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from micromagnetic_gnn.checkpoints import DEFAULT_CACHE, load_coercivity_ensemble
from micromagnetic_gnn.inference import (
    ensemble_predict,
    parity_plot,
    plot_decomposition,
    plot_diagnostics,
)
from micromagnetic_gnn.plotstyle import figure_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["test", "ext", "val", "train", "all"], default="all")
    ap.add_argument("--mc-samples", type=int, default=1,
                    help="MC-dropout passes per member (1 = deterministic deep ensemble).")
    ap.add_argument("--n-std", type=float, default=1.0,
                    help="Parity-plot error-bar half-width in standard deviations.")
    ap.add_argument("--n-bins", type=int, default=10,
                    help="Bins for the sigma-vs-residual reliability plot.")
    ap.add_argument("--ensemble-members", type=int, nargs="+", default=[15,16,17,18,19],
                    help="Member indices for the Deep Ensemble (default: 15-19).")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="Feature cache written by the data pipeline.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ensemble members ...")
    members = load_coercivity_ensemble(members=args.ensemble_members, device=device)
    for m in members:
        print(f"  {m.path.name} (member={m.results['config'].get('MEMBER')}, "
              f"seed={m.results['config'].get('SEED')})")

    # All members share the identical split, features and (deterministic) scaler
    # fit, so member 0 defines the datasets for the whole ensemble.
    head = members[0]
    train_ds, val_ds, test_ds, ext_ds = head.datasets(args.cache)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds, "ext": ext_ds}
    models = [m.model for m in members]

    splits = ["test", "ext"] if args.split == "all" else [args.split]
    for name in splits:
        dataset = datasets[name]
        loader = head.loader(dataset)

        y_mean, std_ale, std_epi, std_tot, y_true = ensemble_predict(
            models, loader, head.scaler, args.mc_samples, device
        )

        print(f"\n=== {name} ({len(dataset)} graphs, {len(models)} members, "
              f"{args.mc_samples} MC samples/member) ===")
        yp, yt = torch.tensor(y_mean), torch.tensor(y_true)
        for mname, metric in {
            "R2": R2Score(),
            "RMSE": MeanSquaredError(squared=False),
            "MAE": MeanAbsoluteError(),
        }.items():
            print(f"  {mname}: {float(metric(yp, yt)):.4f}")

        figures = figure_dir()
        parity_plot(y_true, y_mean, std_tot, figures / f"parity_ens_{name}.pdf",
                    n_std=args.n_std)
        cal_stats = plot_diagnostics(
            y_true, y_mean, std_tot, figures / f"calibration_ens_{name}.pdf", args.n_bins,
        )
        dec_stats = plot_decomposition(
            y_true, std_ale, std_epi, figures / f"uncertainty_ens_{name}.pdf",
        )
        for key, value in (cal_stats | dec_stats).items():
            print(f"  {key:24s}: {value:.4f}")

    print("\nGuide: on the generalization split, epistemic_var_fraction should rise "
          "vs the single model (members disagree on unfamiliar inputs) and "
          "sigma_scale should drop toward 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
