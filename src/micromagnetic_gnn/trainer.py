"""Training loop for graph-level regression.

A thin, model-agnostic trainer that works with any ``nn.Module`` accepting a
PyG ``Data`` / ``Batch`` object.  Built on pure PyTorch — no external
framework required.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ConstantLR, LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.regression import R2Score

const_scheduler = partial(ConstantLR, factor=1.0)
mse_loss = nn.MSELoss()
r2_score = R2Score()


class Trainer:
    """Model-agnostic trainer for graph-level regression.

    The trainer never inspects model internals.  It simply passes a
    ``Data`` / ``Batch`` object to ``model(data)`` and expects a tensor
    of shape ``(batch_size, 1)`` (or ``(batch_size,)``) back.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.  Must accept a PyG ``Data`` / ``Batch``.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader or None
        Validation data loader.  If ``None``, no validation or early
        stopping is performed.
    optimizer : str or torch.optim.Optimizer
        Optimizer name (``"AdamW"``, ``"Adam"``, ``"SGD"``) or an
        already-constructed ``Optimizer`` instance.  Default: ``"AdamW"``.
    lr : float
        Learning rate.  Default: ``1e-3``.
    weight_decay : float
        Weight decay for the optimizer.  Default: ``0.0``.
    scheduler : str or torch.optim.lr_scheduler._LRScheduler or None
        LR scheduler name (``"ReduceLROnPlateau"``) or an already-built
        scheduler.  ``None`` disables scheduling.  Default:
        ``"ReduceLROnPlateau"``.
    loss : str or callable
        Loss function name (``"HuberLoss"``, ``"MSELoss"``, ``"L1Loss"``)
        or a callable.  Default: ``"HuberLoss"``.
    epochs : int
        Maximum number of epochs.  Default: ``150``.
    patience : int
        Early-stopping patience (epochs without val-loss improvement).
        Only active when ``val_loader`` is provided.  Default: ``30``.
    device : str
        ``"auto"`` selects CUDA if available, else CPU.  Default:
        ``"auto"``.
    verbose : bool
        Print per-epoch progress.  Default: ``True``.
    checkpoint_path : str or Path or None
        Where to save the best model.  ``None`` disables checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        optimizer: Callable[[Iterator[Parameter]], Optimizer] | Optimizer = AdamW,
        lr_stop: float = 1e-4,
        scheduler: Callable[[Optimizer], LRScheduler] | LRScheduler = const_scheduler,
        loss: nn.Module = mse_loss,
        metrics: dict[str, Metric] = {"r2": r2_score},
        epochs: int = 150,
        patience: int | None = None,
        device: str = "auto",
        other_data_loaders: dict[str, DataLoader] = {},
        verbose: bool = True,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.other_data_loaders = other_data_loaders
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.lr_stop = lr_stop
 
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else optimizer(model.parameters())
        self.criterion = nn.MSELoss() if loss is None else loss
        self.scheduler = scheduler if isinstance(scheduler, LRScheduler) else scheduler(self.optimizer)
        self.metrics = metrics
        
        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _forward_loss(self, data: object) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model and compute the loss, supporting variance heads.

        Returns ``(loss, preds)`` where ``preds`` is the point prediction used
        for metrics.  If the model returns a ``(mean, var)`` tuple (i.e. a
        Gaussian-NLL / uncertainty head), the criterion is called with the
        three-argument ``(mean, target, var)`` signature and ``mean`` is used
        for metrics.  Otherwise the plain ``(pred, target)`` path is taken.
        """
        out = self.model(data)
        target = data.y.view(-1)
        if isinstance(out, tuple):
            mean, var = out
            loss = self.criterion(mean.view(-1), target, var.view(-1))
            return loss, mean
        loss = self.criterion(out.view(-1), target)
        return loss, out

    def fit(self) -> dict:
        """Run the full training loop.

        Returns
        -------
        dict
            History with keys ``epoch``, ``metrics``, ``lr``.
        """
        history: dict = {
            "epoch": [],
            "metrics": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state: dict | None = None

        for epoch in range(1, self.epochs + 1):
            _metrics = {}
            train_metrics = self._train_one_epoch()
            _metrics["train"] = train_metrics
            
            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader)
                _metrics["val"] = val_metrics
                        
            for data_name, dataloader in self.other_data_loaders.items():
                ds_metrics = self.evaluate(dataloader)
                _metrics[data_name] = ds_metrics

            flat_metrics = {
                f"{k1}_{k2}": v
                for k1, inner in _metrics.items()
                for k2, v in inner.items()
            }
            
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                _loss = flat_metrics.get("val_loss", flat_metrics["train_loss"])
                self.scheduler.step(_loss)
            else:
                self.scheduler.step()
            
            next_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            history["epoch"].append(epoch)
            history["metrics"].append(flat_metrics)
            history["lr"].append(current_lr)

            if self.verbose:
                msg = f"Epoch {epoch:03d}/{self.epochs} | LR: {current_lr:.6f}"
                for k, m in flat_metrics.items():
                    msg += f" | {k}: {m:.4f}"

                print(msg)
            
            # Early stopping & checkpointing
            if self.val_loader is not None:
                val_loss = flat_metrics["val_loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
                    epochs_no_improve = 0
                    model_state = copy.deepcopy(self.model.state_dict())
                    opt_state = copy.deepcopy(self.optimizer.state_dict())
                    best_state = {"model_state": model_state, "opt_state": opt_state}
                    if  self.checkpoint_path is not None:
                        torch.save(best_state, self.checkpoint_path)
                else:
                    epochs_no_improve += 1
                    assert self.patience is not None
                    if self.patience is not None and epochs_no_improve >= self.patience:
                        if self.verbose:
                            print(f"Early stopping triggered at epoch {epoch}")
                        
                        break
                
                
            if next_lr < self.lr_stop:
                if self.verbose:
                    print(f"Stop learning rate {self.lr_stop} reached")
                break
        
        # Restore the best-validation weights so post-fit evaluation and any
        # state_dict saved afterwards (e.g. results.tar) reflect the best model.
        if best_state is not None:
            self.model.load_state_dict(best_state["model_state"])
            
        return history

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        for m in self.metrics.values():
            m.reset()
        
        n_batches = 0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss, out = self._forward_loss(data)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            for m in self.metrics.values():
                m.update(out.view(-1), data.y.view(-1))
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"loss": float(avg_loss)}
        metrics = metrics | {k: float(m.compute()) for k, m in self.metrics.items()}
        return metrics

    def evaluate(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        for m in self.metrics.values():
            m.reset()
        n_batches = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                loss, out = self._forward_loss(data)
                total_loss += loss.item()
                for m in self.metrics.values():
                    m.update(out.view(-1), data.y.view(-1))
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"loss": float(avg_loss)}
        metrics = metrics | {k: float(m.compute()) for k, m in self.metrics.items()}
        return metrics


def easy_axis_grad(ea, gs, model):
    ea = ea.clone()
    gs.x[:, gs.node_feature_map["easy_axis"]] = ea
    out = model(gs)
    pred = out[0] if isinstance(out, tuple) else out

    g = torch.autograd.grad(
        outputs=pred,
        inputs=ea,
        grad_outputs=torch.ones_like(pred),
        create_graph=False
    )[0]
    
    g_proj = g - (g * ea).sum(dim=1, keepdim=True) * ea
    return g_proj

def opt_easy_axis(model: torch.nn.Module, gs, lr: float = 1e-2, steps: int = 200):
    ea = gs.x[:, gs.node_feature_map["easy_axis"]]
    x = torch.nn.Parameter(ea)
    optimizer = torch.optim.Adam([x], lr=lr, maximize=True)
    
    for _ in range(steps):
        optimizer.zero_grad()
        grad = easy_axis_grad(x, gs, model)
        x.grad = grad
        optimizer.step()

        with torch.no_grad():
            x /= x.norm(dim=1, keepdim=True)

    gs.x[:, gs.node_feature_map["easy_axis"]] = x
    return gs
