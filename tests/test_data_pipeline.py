"""Tests for the data-pipeline module."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from micromagnetic_gnn.data_pipeline import (
    MicromagneticDataset,
    gs_to_data,
    scale_indices,
    train_val_test_split,
    random_rotation_transform
)
from micromagnetic_gnn.grain_structure import (
    load_grain_structures,
    load_one_grain_structure,
)


def test_gs_to_data_global(test_structure_dir: Path) -> None:
    """Graph-level label is extracted correctly."""
    gs = load_one_grain_structure(test_structure_dir)
    data = gs_to_data(
        gs,
        node_feature_map={"easy_axis": (0, 1, 2), "K": (3,)},
        edge_feature_map={"center_distance": (0, 1, 2)},
        global_feature_map={"coercivity": (2,)},
        label="coercivity",
        task="global",
    )
    assert isinstance(data, Data)
    assert data.x.shape == (5, 4)
    assert data.y.shape == (1, 1)
    assert data.y.item() == pytest.approx(4.54, abs=1e-3)
    assert data.u.shape == (1, 0)
    # Feature maps are remapped to start at 0
    assert data.node_feature_map == {"easy_axis": (0, 1, 2), "K": (3,)}
    assert data.edge_feature_map == {"center_distance": (0, 1, 2)}


def test_gs_to_data_node(test_structure_dir: Path) -> None:
    """Node-level label is extracted correctly."""
    gs = load_one_grain_structure(test_structure_dir)
    data = gs_to_data(
        gs,
        node_feature_map={"easy_axis": (0, 1, 2), "switching_field": (6,)},
        edge_feature_map={},
        global_feature_map={},
        label="switching_field",
        task="node",
    )
    assert data.x.shape == (5, 3)
    assert data.y.shape == (5, 1)
    assert data.y[3, 0].item() == pytest.approx(-4.435, abs=1e-3)
    assert data.edge_attr.shape == (gs.edges.shape[0], 0)


def test_dataset_from_structures_global(test_structure_dir: Path) -> None:
    """``from_structures`` builds a global-label dataset."""
    gs = load_one_grain_structure(test_structure_dir)
    ds = MicromagneticDataset.from_structures(
        [gs],
        node_features=["easy_axis", "K"],
        edge_features=["center_distance"],
        global_features=["coercivity"],
        label_name="coercivity",
        task="global",
    )
    assert len(ds) == 1
    data = ds[0]
    assert data.x.shape == (5, 4)
    assert data.y.shape == (1, 1)
    assert data.u.shape == (1, 0)
    assert ds.num_node_features == 4
    assert ds.num_edge_features == 3
    assert ds.num_global_features == 0
    # Feature maps are remapped to start at 0
    assert ds.node_feature_map["easy_axis"] == (0, 1, 2)
    assert ds.node_feature_map["K"] == (3,)
    assert ds.edge_feature_map["center_distance"] == (0, 1, 2)


def test_dataset_from_structures_node(test_structure_dir: Path) -> None:
    """``from_structures`` builds a node-label dataset."""
    gs = load_one_grain_structure(test_structure_dir)
    ds = MicromagneticDataset.from_structures(
        [gs],
        node_features=["easy_axis", "switching_field"],
        edge_features=["face_area"],
        global_features=["no_grains"],
        label_name="switching_field",
        task="node",
    )
    data = ds[0]
    assert data.x.shape == (5, 3)
    assert data.y.shape == (5, 1)
    assert data.u.shape == (1, 1)
    # Remapped indices
    assert ds.node_feature_map["easy_axis"] == (0, 1, 2)
    assert ds.edge_feature_map["face_area"] == (0,)
    assert ds.global_feature_map["no_grains"] == (0,)


def test_dataset_empty_features(test_structure_dir: Path) -> None:
    """Empty feature selections yield zero-width tensors."""
    gs = load_one_grain_structure(test_structure_dir)
    ds = MicromagneticDataset.from_structures(
        [gs],
        node_features=[],
        edge_features=[],
        global_features=[],
        label_name="coercivity",
        task="global",
    )
    data = ds[0]
    assert data.x.shape == (5, 0)
    assert data.edge_attr.shape == (gs.edges.shape[0], 0)
    assert data.u.shape == (1, 0)
    assert ds.node_feature_map == {}
    assert ds.edge_feature_map == {}
    assert ds.global_feature_map == {}


def test_random_rotation_transform(test_structure_dir: Path) -> None:
    """Vector features rotate; scalar features stay fixed."""
    gs = load_one_grain_structure(test_structure_dir)
    ds = MicromagneticDataset.from_structures(
        [gs],
        node_features=["easy_axis", "K"],
        edge_features=["center_distance", "face_area"],
        label_name="coercivity",
        task="global",
        transform=random_rotation_transform,
    )
    d1 = ds[0]
    d2 = ds[0]

    # easy_axis is a vector → should differ
    assert not torch.allclose(d1.x[:, :3], d2.x[:, :3])
    # K is scalar → should match
    assert torch.allclose(d1.x[:, 3], d2.x[:, 3])

    # center_distance is a vector → should differ
    assert not torch.allclose(d1.edge_attr[:, :3], d2.edge_attr[:, :3])
    # face_area is scalar → should match
    assert torch.allclose(d1.edge_attr[:, 3], d2.edge_attr[:, 3])


def test_train_val_test_split() -> None:
    """Split preserves metadata and yields correct lengths."""
    root = Path(__file__).parent.parent / "ROM_Simulations_test"
    structures = load_grain_structures(root, max_workers=1)
    ds = MicromagneticDataset.from_structures(
        structures,
        node_features=["easy_axis", "K"],
        edge_features=["center_distance"],
        global_features=["coercivity"],
        label_name="coercivity",
        task="global",
    )

    train_ds, val_ds, test_ds = train_val_test_split(
        ds, train_frac=1.0, val_frac=0.0, test_frac=0.0, seed=123
    )

    assert len(train_ds) == 1
    assert len(val_ds) == 0
    assert len(test_ds) == 0

    # Metadata preserved
    assert train_ds.node_feature_map == ds.node_feature_map
    assert train_ds.edge_feature_map == ds.edge_feature_map
    assert train_ds.label == ds.label
    assert train_ds.task == ds.task


def test_train_val_test_split_fractions() -> None:
    """Fractions that don't sum to 1.0 raise."""
    root = Path(__file__).parent.parent / "ROM_Simulations_test"
    structures = load_grain_structures(root, max_workers=1)
    ds = MicromagneticDataset.from_structures(structures, label_name="coercivity", task="global")
    with pytest.raises(ValueError, match="sum to 1.0"):
        train_val_test_split(ds, train_frac=0.5, val_frac=0.5, test_frac=0.1)


def test_scale_indices() -> None:
    """``scale_indices`` returns concatenated column indices."""
    feature_map = {"easy_axis": (0, 1, 2), "K": (3,), "Ms": (4,)}
    assert scale_indices(feature_map, "K") == [3]
    assert scale_indices(feature_map, "K", "Ms") == [3, 4]
    assert scale_indices(feature_map, "easy_axis") == [0, 1, 2]
    assert scale_indices(feature_map, "missing") == []


def test_pyg_dataloader_batching(test_structure_dir: Path) -> None:
    """DataLoader correctly batches global features."""
    gs = load_one_grain_structure(test_structure_dir)
    ds = MicromagneticDataset.from_structures(
        [gs, gs],
        node_features=["easy_axis", "K"],
        edge_features=["center_distance"],
        global_features=["coercivity"],
        label_name="coercivity",
        task="global",
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert batch.x.shape[0] == 10  # 5+5 nodes
    assert batch.edge_index.shape[0] == 2
    assert batch.u.shape == (2, 0)  # coercivity is the label, removed from u
    assert batch.y.shape == (2, 1)

