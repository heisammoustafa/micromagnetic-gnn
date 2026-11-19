"""Uncertainty-calibration diagnostics for the coercivity model.

Validates the *quality* of the predicted uncertainty rather than point accuracy,
by comparing the predicted total sigma against the observed residual spread on
held-out data. Per split it writes the three-panel figure documented in
:func:`micromagnetic_gnn.inference.plot_diagnostics` (reliability curve, sigma
vs residual spread, PIT histogram) and prints the scalar metrics.

Uses the same prediction path as ``infer_coercivity.py``, so the assessed
uncertainty is the combined aleatoric + MC-dropout epistemic one.

Run from the repo root::

    python analysis/calibration_coercivity.py                 # test + generalization
    python analysis/calibration_coercivity.py --split test
    python analysis/calibration_coercivity.py --mc-samples 50 --n-bins 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (no display on the cluster)
import torch

from micromagnetic_gnn.checkpoints import DEFAULT_CACHE, load_coercivity_model
from micromagnetic_gnn.inference import mc_predict, plot_diagnostics
from micromagnetic_gnn.plotstyle import figure_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["test", "ext", "val", "train", "all"], default="all")
    ap.add_argument("--member", type=int, default=15,
                    help="Evaluate ensemble member K (results/coercivity_m<K>.tar).")
    ap.add_argument("--mc-samples", type=int, default=20,
                    help="Monte-Carlo dropout forward passes (1 = aleatoric only).")
    ap.add_argument("--n-bins", type=int, default=10,
                    help="Equal-count bins for the sigma-vs-residual reliability plot.")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="Feature cache written by the data pipeline.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trained = load_coercivity_model(member=args.member, device=device)
    print(f"Loaded {trained.path}")
    if not trained.predicts_variance:
        print("  NOTE: checkpoint has no variance head; uncertainty is "
              "MC-dropout epistemic only.")

    train_ds, val_ds, test_ds, ext_ds = trained.datasets(args.cache)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds, "ext": ext_ds}

    # Keep dropout active only when doing MC sampling.
    trained.model.train() if args.mc_samples > 1 else trained.model.eval()

    splits = ["test", "ext"] if args.split == "all" else [args.split]
    for name in splits:
        dataset = datasets[name]
        loader = trained.loader(dataset)
        y_mean, y_std, y_true = mc_predict(
            trained.model, loader, trained.scaler, args.mc_samples, device
        )

        print(f"\n=== {name} ({len(dataset)} graphs, {args.mc_samples} MC samples) ===")
        stats = plot_diagnostics(
            y_true, y_mean, y_std, figure_dir() / f"calibration_{name}.pdf", args.n_bins,
        )
        for key, value in stats.items():
            print(f"  {key:20s}: {value:.4f}")

    print("\nGuide: cov_1sigma -> 0.683, cov_2sigma -> 0.954, rms_z/sigma_scale -> 1.0, "
          "ECE & miscalibration_area -> 0. sigma_scale>1 => overconfident (sigma too small).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
