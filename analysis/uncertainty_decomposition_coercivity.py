"""Aleatoric vs epistemic uncertainty, decoupled, for the coercivity model.

The Gaussian-NLL variance head gives **aleatoric** uncertainty (irreducible data
noise) while the spread of Monte-Carlo-dropout means gives **epistemic**
uncertainty (model / lack-of-data uncertainty). Variances add, so the two can be
shown side by side and summed honestly. Per split this writes the three-panel
figure documented in
:func:`micromagnetic_gnn.inference.plot_decomposition`.

Run from the repo root::

    python analysis/uncertainty_decomposition_coercivity.py
    python analysis/uncertainty_decomposition_coercivity.py --split test --mc-samples 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import torch

from micromagnetic_gnn.checkpoints import DEFAULT_CACHE, load_coercivity_model
from micromagnetic_gnn.inference import mc_predict_decomposed, plot_decomposition
from micromagnetic_gnn.plotstyle import figure_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["test", "ext", "val", "train", "all"], default="all")
    ap.add_argument("--member", type=int, default=15,
                    help="Evaluate ensemble member K (results/coercivity_m<K>.tar).")
    ap.add_argument("--mc-samples", type=int, default=30,
                    help="MC-dropout passes; >1 needed to estimate epistemic uncertainty.")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="Feature cache written by the data pipeline.")
    args = ap.parse_args()

    if args.mc_samples < 2:
        print("WARNING: --mc-samples < 2 gives zero epistemic uncertainty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trained = load_coercivity_model(member=args.member, device=device)
    print(f"Loaded {trained.path}")
    if not trained.predicts_variance:
        print("  NOTE: checkpoint has no variance head; aleatoric uncertainty will "
              "be zero and only the epistemic (MC-dropout) term is shown.")

    train_ds, val_ds, test_ds, ext_ds = trained.datasets(args.cache)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds, "ext": ext_ds}

    trained.model.train() if args.mc_samples > 1 else trained.model.eval()

    splits = ["test", "ext"] if args.split == "all" else [args.split]
    for name in splits:
        dataset = datasets[name]
        loader = trained.loader(dataset)
        _, std_ale, std_epi, _, y_true = mc_predict_decomposed(
            trained.model, loader, trained.scaler, args.mc_samples, device
        )

        print(f"\n=== {name} ({len(dataset)} graphs, {args.mc_samples} MC samples) ===")
        stats = plot_decomposition(
            y_true, std_ale, std_epi, figure_dir() / f"uncertainty_{name}.pdf",
        )
        for key, value in stats.items():
            print(f"  {key:24s}: {value:.4f}")

    print("\nGuide: epistemic_var_fraction ~0 => aleatoric dominates (data noise); "
          "~1 => epistemic dominates (model uncertainty, e.g. extrapolation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
