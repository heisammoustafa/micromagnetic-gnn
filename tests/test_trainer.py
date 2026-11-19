"""Tests for the model-agnostic Trainer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from micromagnetic_gnn.trainer import Trainer


class DummyModel(nn.Module):
    """A tiny GNN-like model that operates on PyG Data objects."""

    def __init__(self, in_channels: int = 3, hidden: int = 8):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def forward(self, data):
        # Simple mean pooling over nodes
        x = torch.relu(self.lin1(data.x))
        # Global mean pooling using batch vector
        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        # Manual scatter mean
        out = torch.zeros(batch.max().item() + 1, x.size(1), device=x.device)
        out.index_add_(0, batch, x)
        counts = torch.bincount(batch).float().unsqueeze(1)
        out = out / counts
        return self.lin2(out).squeeze(1)


@pytest.fixture
def dummy_data():
    """Return a list of small PyG Data objects."""
    torch.manual_seed(0)
    data_list = []
    for _ in range(20):
        n_nodes = torch.randint(3, 8, (1,)).item()
        x = torch.randn(n_nodes, 3)
        y = torch.randn(1)
        edge_index = torch.randint(0, n_nodes, (2, 10))
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return data_list


class TestTrainer:
    def test_train_without_val(self, dummy_data):
        model = DummyModel()
        loader = DataLoader(dummy_data, batch_size=4, shuffle=False)
        trainer = Trainer(
            model,
            train_loader=loader,
            val_loader=None,
            epochs=2,
            verbose=False,
        )
        history = trainer.fit()
        assert len(history["epoch"]) == 2
        assert all(not torch.isnan(torch.tensor(v["train_loss"])) for v in history["metrics"])
        assert all(v.get("val_loss") is None for v in history["metrics"])

    def test_train_with_val(self, dummy_data):
        model = DummyModel()
        train_loader = DataLoader(dummy_data[:16], batch_size=4, shuffle=False)
        val_loader = DataLoader(dummy_data[16:], batch_size=4, shuffle=False)
        trainer = Trainer(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            verbose=False,
        )
        history = trainer.fit()
        assert len(history["epoch"]) == 3
        assert all(not torch.isnan(torch.tensor(d["val_r2"])) for d in history["metrics"])

    def test_checkpoint_saved(self, dummy_data, tmp_path):
        ckpt = tmp_path / "best.pt"
        model = DummyModel()
        train_loader = DataLoader(dummy_data[:16], batch_size=4)
        val_loader = DataLoader(dummy_data[16:], batch_size=4)
        trainer = Trainer(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            verbose=False,
            checkpoint_path=ckpt,
        )
        trainer.fit()
        assert ckpt.exists()
        state = torch.load(ckpt, weights_only=False)
        assert "lin1.weight" in state["model_state"]

    def test_early_stopping(self, dummy_data):
        model = DummyModel()
        train_loader = DataLoader(dummy_data[:16], batch_size=4, shuffle=False)
        val_loader = DataLoader(dummy_data[16:], batch_size=4, shuffle=False)
        trainer = Trainer(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            patience=1,
            optimizer=lambda p: torch.optim.Adam(p, lr=0.0),  # frozen weights → identical val loss every epoch
            lr_stop=0.0,  # disable LR stop so early stopping can trigger
            verbose=True,
        )
        history = trainer.fit()
        # With lr=0 val loss never improves after epoch 1, so patience=1
        # triggers at epoch 2.
        assert len(history["epoch"]) == 2

    def test_evaluate_method(self, dummy_data):
        model = DummyModel()
        loader = DataLoader(dummy_data, batch_size=4)
        trainer = Trainer(
            model,
            train_loader=loader,
            val_loader=None,
            epochs=1,
            verbose=False,
        )
        trainer.fit()
        metrics = trainer.evaluate(loader)
        assert isinstance(metrics["loss"], float)
        assert "r2" in metrics

    def test_device_auto(self, dummy_data):
        model = DummyModel()
        loader = DataLoader(dummy_data, batch_size=4)
        trainer = Trainer(
            model,
            train_loader=loader,
            val_loader=None,
            epochs=1,
            verbose=False,
            device="auto",
        )
        assert str(trainer.device) in ("cpu", "cuda")

    def test_custom_optimizer(self, dummy_data):
        model = DummyModel()
        loader = DataLoader(dummy_data, batch_size=4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(
            model,
            train_loader=loader,
            val_loader=None,
            optimizer=opt,
            epochs=1,
            verbose=False,
        )
        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_custom_scheduler(self, dummy_data):
        model = DummyModel()
        loader = DataLoader(dummy_data, batch_size=4)
        opt = torch.optim.Adam(model.parameters())
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        trainer = Trainer(
            model,
            train_loader=loader,
            val_loader=None,
            optimizer=opt,
            scheduler=sched,
            epochs=2,
            verbose=False,
        )
        history = trainer.fit()
        # LR should decay after step
        assert history["lr"][-1] <= history["lr"][0]

    def test_custom_loss(self, dummy_data):
        model = DummyModel()
        loader = DataLoader(dummy_data, batch_size=4)
        trainer = Trainer(
            model,
            train_loader=loader,
            val_loader=None,
            loss=nn.MSELoss(),
            epochs=1,
            verbose=False,
        )
        assert isinstance(trainer.criterion, nn.MSELoss)
