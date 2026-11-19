"""BHmax prediction with a deep residual GCN.

Usage::

    python bhmax_prediction/main.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader
from torchmetrics import R2Score, SpearmanCorrCoef

from micromagnetic_gnn.checkpoints import results_dir
from micromagnetic_gnn.dataset import default_datasets
from micromagnetic_gnn.models import EdgeAggEncoder, NodeModel
from micromagnetic_gnn.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

DATA_ROOT = Path("ROM_Simulations")
CACHE = Path("data_cache.pkl")
CHECKPOINT = Path("switching_field_prediction/best.pt")
RESULTS = results_dir() / "switching_field.tar"

SEED = 42
BATCH_SIZE = 16
NODE_EMBEDDING_DIM = 150
EDGE_EMBEDDING_DIM = 50
GLOBAL_EMBEDDING_DIM = 50
N_CONV = 6
DROPOUT = 0.1
LR = 1e-3
LR_STOP = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 10000
PATIENCE = 70
EARLY_STOPPING_PATIENCE = 200


def _per_structure_label(inner):
    """Standardize the switching field within each structure.

    About 74 % of the variance of the switching field lies *between* structures,
    so a plain MSE spends most of its gradient on each magnet's overall level --
    which the global features already carry -- and little on ordering grains
    inside a structure, which is what the node-level task is for. Replacing the
    label by its within-structure z-score removes the level from the objective
    entirely. A z-score is invariant under the affine label scaler, so it does
    not matter whether this runs before or after it.
    """
    def transform(d):
        d = inner(d)
        y = d.y.reshape(-1).float()
        if y.numel() > 1:
            std = y.std()
            if torch.isfinite(std) and std > 1e-8:
                d.y = ((y - y.mean()) / std).reshape(d.y.shape)
            else:
                d.y = torch.zeros_like(d.y)
        else:
            d.y = torch.zeros_like(d.y)
        return d
    return transform


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-structure-label", action="store_true",
        help="Train on the within-structure z-score of the switching field "
             "instead of its absolute value, so the loss targets the ordering of "
             "grains inside a magnet rather than the magnet's overall level.",
    )
    args = ap.parse_args()
    suffix = "_norm" if args.per_structure_label else ""
    checkpoint = CHECKPOINT.with_name(f"best{suffix}.pt")
    results_path = RESULTS.with_name(f"switching_field{suffix}.tar")
    torch.manual_seed(SEED)

    print(f"Loading dataset from {DATA_ROOT} ...")

    train_ds, val_ds, test_ds, ext_ds, scaler = default_datasets(
        "switching_field", CACHE, rebuild=False, rotate=False)

    if args.per_structure_label:
        print("  label: within-structure z-score")
        for ds in (train_ds, val_ds, test_ds, ext_ds):
            ds.transform = _per_structure_label(ds.transform)
    
    encoder_config = {
        "node_fea_len": train_ds.num_node_features,
        "edge_fea_len": train_ds.num_edge_features,
        "global_fea_len": train_ds.num_global_features,
        "n_conv": N_CONV,
        "node_embedding_dim": NODE_EMBEDDING_DIM,
        "edge_embedding_dim": EDGE_EMBEDDING_DIM,
        "global_embedding_dim": GLOBAL_EMBEDDING_DIM,
        "dropout": DROPOUT,
        "update_edges": 1,
    }
    encoder = EdgeAggEncoder(**encoder_config)
    model_config: dict[str, Any] = {}
    model = NodeModel(encoder, **model_config)
    
    #model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
        
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    ext_loader = DataLoader(ext_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    # Train
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        other_data_loaders={"ext": ext_loader},
        optimizer=partial(torch.optim.AdamW, lr=LR, weight_decay=WEIGHT_DECAY),
        lr_stop=LR_STOP,
        scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, patience=PATIENCE),
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path=checkpoint,
        verbose=True,
        loss=torch.nn.MSELoss(),
        metrics={"r2": R2Score(), "Spearman": SpearmanCorrCoef()}
    )
    history = trainer.fit()

    # Evaluate
    print("\n--- Evaluation ---")
    train_metrics = trainer.evaluate(train_loader)
    val_metrics = trainer.evaluate(val_loader)
    test_metrics = trainer.evaluate(test_loader)
    ext_metrics = trainer.evaluate(ext_loader)

    def print_result(name: str, metrics: dict) -> None:
        msg = f"{name}: "
        for k, m in metrics.items():
            msg += f"{k}: {m:.4f} | "
        print(msg[:-3])
    
    print_result("Train", train_metrics)
    print_result("Val", val_metrics)
    print_result("Test", test_metrics)
    print_result("Ext", ext_metrics)

    out = {
        "config": {
            "SEED": SEED,
            "BATCH_SIZE": BATCH_SIZE,
            "NODE_EMBEDDING_DIM": NODE_EMBEDDING_DIM,
            "EDGE_EMBEDDING_DIM": EDGE_EMBEDDING_DIM,
            "GLOBAL_EMBEDDING_DIM": GLOBAL_EMBEDDING_DIM,
            "N_CONV": N_CONV,
            "DROPOUT": DROPOUT,
            "LR": LR,
            "LR_STOP": LR_STOP,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "EPOCHS": EPOCHS,
            "PATIENCE": PATIENCE,
            "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        },
        "features": {
            "node_features": train_ds.node_features,
            "edge_features": train_ds.edge_features,
            "global_features": train_ds.global_features,
        },
        "encoder_config": encoder_config,
        "model_config": model_config,
        "scaler_config": scaler.config,
        "encoder_state_dict": encoder.state_dict(),
        "state_dict": model.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "dataset_size": len(train_ds) + len(val_ds) + len(test_ds) + len(ext_ds),
        "model_params": n_params,
        "splits": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "ext": ext_metrics,
        },
        "history": history,
    }
    
    results_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, results_path)
    print(f"\nResults written to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
