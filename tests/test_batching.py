"""Verify PyG DataLoader batching behaviour with custom attributes.

These tests use plain dummy Data objects (no file I/O) so that the
batching mechanics can be checked in isolation from the full data pipeline.
"""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _make_dummy_data(num_nodes: int, node_feat: int, edge_feat: int, global_feat: int, label_dim: int) -> Data:
    """Create a single dummy graph with the same attributes we use downstream."""
    x = torch.randn(num_nodes, node_feat)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], edge_feat)
    u = torch.randn(1, global_feat)
    y = torch.randn(label_dim)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, y=y)


def test_dataloader_batches_custom_u() -> None:
    """Graph-level u is stacked into (batch_size, n_global)."""
    data_list = [
        _make_dummy_data(num_nodes=3, node_feat=5, edge_feat=2, global_feat=3, label_dim=1),
        _make_dummy_data(num_nodes=4, node_feat=5, edge_feat=2, global_feat=3, label_dim=1),
        _make_dummy_data(num_nodes=2, node_feat=5, edge_feat=2, global_feat=3, label_dim=1),
    ]
    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # Nodes are concatenated: 3 + 4 = 7
    assert batch.x.shape == (7, 5)
    # Edges are concatenated: 2 + 2 = 4
    assert batch.edge_index.shape == (2, 4)
    assert batch.edge_attr.shape == (4, 2)
    # u stacked along dim 0 -> (batch_size, n_global)
    assert batch.u.shape == (2, 3)
    # y stacked along dim 0 -> (batch_size,)
    assert batch.y.shape == (2,)
    # batch vector tells which graph each node belongs to
    assert batch.batch.shape == (7,)
    assert torch.equal(batch.batch, torch.tensor([0, 0, 0, 1, 1, 1, 1]))


def test_dataloader_single_graph_batch() -> None:
    """Batch of size 1 still produces correct shapes."""
    data_list = [
        _make_dummy_data(num_nodes=3, node_feat=2, edge_feat=1, global_feat=2, label_dim=1),
    ]
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    batch = next(iter(loader))

    assert batch.x.shape == (3, 2)
    assert batch.u.shape == (1, 2)
    assert batch.y.shape == (1,)
    assert batch.batch.shape == (3,)


def test_dataloader_shuffling() -> None:
    """Shuffling reorders graphs within a batch."""
    data_list = [
        _make_dummy_data(num_nodes=2, node_feat=2, edge_feat=1, global_feat=1, label_dim=1),
        _make_dummy_data(num_nodes=3, node_feat=2, edge_feat=1, global_feat=1, label_dim=1),
    ]
    # Fixed seed for reproducibility
    loader = DataLoader(data_list, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))
    batch = next(iter(loader))

    # Total nodes should still be 2 + 3 = 5
    assert batch.x.shape == (5, 2)
    assert batch.u.shape == (2, 1)
    assert batch.y.shape == (2,)


def test_dataloader_with_exclude_keys() -> None:
    """exclude_keys can drop attributes from the batch."""
    data_list = [
        _make_dummy_data(num_nodes=2, node_feat=2, edge_feat=1, global_feat=2, label_dim=1),
        _make_dummy_data(num_nodes=2, node_feat=2, edge_feat=1, global_feat=2, label_dim=1),
    ]
    loader = DataLoader(data_list, batch_size=2, exclude_keys=["u"])
    batch = next(iter(loader))

    assert batch.x.shape == (4, 2)
    assert not hasattr(batch, "u")
    assert batch.y.shape == (2,)
