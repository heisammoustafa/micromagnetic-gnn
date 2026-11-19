"""Inference + true-vs-predicted plots for the coercivity model.

Loads ``results/coercivity.tar``, runs Monte-Carlo-dropout inference on the
requested split(s), prints R2/RMSE/MAE and writes a parity plot per split.

The checkpoint carries its own feature lists and fitted scaler, so the datasets
are built to match the trained weights rather than from a feature list copied
into this file.

Run from the repo root::

    python analysis/infer_coercivity.py                 # test + generalization
    python analysis/infer_coercivity.py --split test
    python analysis/infer_coercivity.py --mc-samples 50
    python analysis/infer_coercivity.py --member 0      # one ensemble member
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (no display on the cluster)
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from micromagnetic_gnn.checkpoints import DEFAULT_CACHE, load_coercivity_model
from micromagnetic_gnn.inference import mc_predict, parity_plot
from micromagnetic_gnn.plotstyle import figure_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["test", "ext", "val", "train", "all"], default="all")
    ap.add_argument("--member", type=int, default=None,
                    help="Evaluate ensemble member K (results/coercivity_m<K>.tar).")
    ap.add_argument("--mc-samples", type=int, default=20,
                    help="Monte-Carlo dropout forward passes (1 = deterministic).")
    ap.add_argument("--n-std", type=float, default=1.0,
                    help="Error-bar half-width in standard deviations (1 = +-1 sigma).")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="Feature cache written by the data pipeline.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trained = load_coercivity_model(member=args.member, device=device)
    print(f"Loaded {trained.path}")
    train_ds, val_ds, test_ds, ext_ds = trained.datasets(args.cache)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds, "ext": ext_ds}

    # MC dropout keeps dropout active; --mc-samples 1 gives a deterministic pass.
    trained.model.train() if args.mc_samples > 1 else trained.model.eval()

    splits = ["test", "ext"] if args.split == "all" else [args.split]
    for name in splits:
        dataset = datasets[name]
        loader = trained.loader(dataset)
        y_mean, y_std, y_true = mc_predict(
            trained.model, loader, trained.scaler, args.mc_samples, device
        )

        print(f"\n=== {name} ({len(dataset)} graphs, {args.mc_samples} MC samples) ===")
        yp, yt = torch.tensor(y_mean), torch.tensor(y_true)
        for mname, metric in {
            "R2": R2Score(),
            "RMSE": MeanSquaredError(squared=False),
            "MAE": MeanAbsoluteError(),
        }.items():
            print(f"  {mname}: {float(metric(yp, yt)):.4f}")

        parity_plot(y_true, y_mean, y_std, figure_dir() / f"parity_{name}.pdf",
                    n_std=args.n_std)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
