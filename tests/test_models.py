"""Tests for baseline GNN models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from micromagnetic_gnn.models import StandardScalerModule, GraphDataScaler, GATEncoder
from micromagnetic_gnn.grain_structure import load_one_grain_structure
from micromagnetic_gnn.data_pipeline import MicromagneticDataset


@pytest.fixture
def single_graph_data():
    x = torch.randn(5, 4)
    edge_index = torch.randint(0, 5, (2, 8))
    edge_attr = torch.randn(8, 2)
    u = torch.randn(1, 3)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)


@pytest.fixture(scope="session")
def dummy_grain_structure(test_structure_dir: Path):
    gs = load_one_grain_structure(test_structure_dir)
    return gs


@pytest.fixture(scope="session")
def dummy_dataset_global(dummy_grain_structure):
    data = [dummy_grain_structure, dummy_grain_structure, dummy_grain_structure]
    dataset = MicromagneticDataset.from_structures(data, label_name="bhmax", task="global")
    return dataset


@pytest.fixture(scope="session")
def dummy_dataset_no_label(dummy_grain_structure):
    data = [dummy_grain_structure, dummy_grain_structure, dummy_grain_structure]
    dataset = MicromagneticDataset.from_structures(data, label_name=None, task="global")
    return dataset


@pytest.fixture(scope="session")
def dummy_dataset_node(dummy_grain_structure):
    data = [dummy_grain_structure, dummy_grain_structure, dummy_grain_structure]
    dataset = MicromagneticDataset.from_structures(data, label_name="switching_field", task="node")
    return dataset


@pytest.fixture(scope="session")
def dummy_dataset_link(dummy_grain_structure):
    data = [dummy_grain_structure, dummy_grain_structure, dummy_grain_structure]
    dataset = MicromagneticDataset.from_structures(data, task="link")
    return dataset


class TestStandardScalerModule:
    def setup_method(self, test_method):
        self.rng_state = torch.get_rng_state()
        torch.manual_seed(42)

    def teardown_method(self, test_method):
        torch.set_rng_state(self.rng_state)

    def test_scaler(self):
        scaler = StandardScalerModule(5)
        X = torch.randn((100, 5), ) * 5 + 1
        scaler.fit([X], scale_indices=[0, 1, 2, 3, 4])
        X_scaled = scaler(X)
        assert torch.all(torch.isclose(torch.mean(X_scaled, dim=0), torch.tensor(0.0), atol=1e-6))
        assert torch.all(torch.isclose(torch.std(X_scaled, dim=0), torch.tensor(1.0), atol=1e-6))
        X_inv = scaler.inverse_transform(X_scaled)
        assert torch.all(torch.isclose(X_inv, X))

    def test_scaler_with_scale_indices(self):
        scaler = StandardScalerModule(5)
        X = torch.randn((100, 5)) * 5 + 1
        scaler.fit([X], scale_indices=[0, 1, 2, 3])
        X_scaled = scaler(X)
        assert torch.all(torch.isclose(torch.mean(X_scaled, dim=0)[:4], torch.tensor(0.0), atol=1e-6))
        assert torch.all(torch.isclose(torch.std(X_scaled, dim=0)[:4], torch.tensor(1.0), atol=1e-6))
        assert torch.mean(X_scaled, dim=0)[4] > torch.tensor(0.5)
        assert torch.std(X_scaled, dim=0)[4] > torch.tensor(3.0)

    def test_scaler_with_vector_groups(self):
        scaler = StandardScalerModule(5)
        X = torch.randn((100, 5)) * 5 + 1
        scaler.fit([X], scale_indices=[0, 1], vector_groups=[[2, 3, 4]])
        assert torch.all(scaler.mean[[2, 3, 4]] == scaler.mean[[2]])
        assert torch.all(scaler.std[[2, 3, 4]] == scaler.std[[2]])

    def test_scaler_with_vector(self):
        scaler = StandardScalerModule(5)
        X = torch.randn((100,)) * 5 + 1
        scaler.fit([X], scale_indices=[0])
        X_scaled = scaler(X)
        assert X_scaled.shape == (100,)
        assert torch.all(torch.isclose(torch.mean(X_scaled, dim=0), torch.tensor(0.0), atol=1e-6))
        assert torch.all(torch.isclose(torch.std(X_scaled, dim=0), torch.tensor(1.0), atol=1e-6))

    def test_scaler_with_constant_feature(self):
        scaler = StandardScalerModule(5)
        X = torch.ones((100,))
        scaler.fit([X], scale_indices=[0])
        X_scaled = scaler(X)
        assert torch.all(X_scaled == torch.tensor(0.0))
        assert torch.all(scaler.std == torch.tensor(1.0))


class TestGraphDataScaler:
    def test_graph_data_scaler(self, dummy_dataset_global):
        dataset = dummy_dataset_global
        scaler = GraphDataScaler(
            node_fea_len=dataset.num_node_features,
            edge_fea_len=dataset.num_edge_features,
            global_fea_len=dataset.num_global_features,
        )
        scaler.fit(dataset)
        assert torch.any(scaler.node_scaler.mean != 0)
        assert torch.any(scaler.node_scaler.std != 0)
        assert torch.any(scaler.edge_scaler.mean != 0)
        assert torch.any(scaler.edge_scaler.std != 0) 
        assert torch.any(scaler.global_scaler.mean != 0)
        assert torch.any(scaler.global_scaler.std != 0)  
        assert torch.any(scaler.label_scaler.mean != 0)
        assert torch.any(scaler.label_scaler.std != 0)

        data = dataset.get(0)
        data_scaled = scaler(data.clone())
        data_unscaled = scaler.unscale(data_scaled)
        assert torch.all(torch.isclose(data.x, data_unscaled.x, atol=1e-4))
        assert torch.all(torch.isclose(data.y, data_unscaled.y, atol=1e-4))
        assert torch.all(torch.isclose(data.edge_attr, data_unscaled.edge_attr, atol=1e-4))
        assert torch.all(torch.isclose(data.u, data_unscaled.u, atol=1e-4))

    def test_graph_data_scaler_no_label(self, dummy_dataset_no_label):
        dataset = dummy_dataset_no_label
        scaler = GraphDataScaler(
            node_fea_len=dataset.num_node_features,
            edge_fea_len=dataset.num_edge_features,
            global_fea_len=dataset.num_global_features,
            scale_label=False
        )
        scaler.fit(dataset)
        assert torch.any(scaler.node_scaler.mean != 0)
        assert torch.any(scaler.node_scaler.std != 0)
        assert torch.any(scaler.edge_scaler.mean != 0)
        assert torch.any(scaler.edge_scaler.std != 0) 
        assert torch.any(scaler.global_scaler.mean != 0)
        assert torch.any(scaler.global_scaler.std != 0)  
        assert torch.any(scaler.label_scaler.mean == 0)
        assert torch.any(scaler.label_scaler.std == 0)

        data = dataset.get(0)
        data_scaled = scaler(data.clone())
        data_unscaled = scaler.unscale(data_scaled)
        assert torch.all(torch.isclose(data.x, data_unscaled.x, atol=1e-4))
        assert data.y is None and data_unscaled.y is None
        assert torch.all(torch.isclose(data.edge_attr, data_unscaled.edge_attr, atol=1e-4))
        assert torch.all(torch.isclose(data.u, data_unscaled.u, atol=1e-4))

    def test_graph_data_scaler_node(self, dummy_dataset_node):
        dataset = dummy_dataset_node
        scaler = GraphDataScaler(
            node_fea_len=dataset.num_node_features,
            edge_fea_len=dataset.num_edge_features,
            global_fea_len=dataset.num_global_features,
        )
        scaler.fit(dataset)
        assert torch.any(scaler.node_scaler.mean != 0)
        assert torch.any(scaler.node_scaler.std != 0)
        assert torch.any(scaler.edge_scaler.mean != 0)
        assert torch.any(scaler.edge_scaler.std != 0) 
        assert torch.any(scaler.global_scaler.mean != 0)
        assert torch.any(scaler.global_scaler.std != 0)  
        assert torch.any(scaler.label_scaler.mean != 0)
        assert torch.any(scaler.label_scaler.std != 0)

        data = dataset.get(0)
        data_scaled = scaler(data.clone())
        data_unscaled = scaler.unscale(data_scaled)
        assert torch.all(torch.isclose(data.x, data_unscaled.x, atol=1e-4))
        assert torch.all(torch.isclose(data.y, data_unscaled.y, atol=1e-4))
        assert torch.all(torch.isclose(data.edge_attr, data_unscaled.edge_attr, atol=1e-4))
        assert torch.all(torch.isclose(data.u, data_unscaled.u, atol=1e-4))

    def test_graph_data_scaler_link(self, dummy_dataset_link):
        dataset = dummy_dataset_link
        scaler = GraphDataScaler(
            node_fea_len=dataset.num_node_features,
            edge_fea_len=dataset.num_edge_features,
            global_fea_len=dataset.num_global_features,
            scale_label=False
        )
        scaler.fit(dataset)
        assert torch.any(scaler.node_scaler.mean != 0)
        assert torch.any(scaler.node_scaler.std != 0)
        assert torch.any(scaler.edge_scaler.mean != 0)
        assert torch.any(scaler.edge_scaler.std != 0) 
        assert torch.any(scaler.global_scaler.mean != 0)
        assert torch.any(scaler.global_scaler.std != 0)  
        assert torch.any(scaler.label_scaler.mean == 0)
        assert torch.any(scaler.label_scaler.std == 0)

        data = dataset.get(0)
        data_scaled = scaler(data.clone())
        data_unscaled = scaler.unscale(data_scaled)
        assert torch.all(torch.isclose(data.x, data_unscaled.x, atol=1e-4))
        assert torch.all(torch.isclose(data.y, data_unscaled.y, atol=1e-4))
        assert torch.all(torch.isclose(data_scaled.y, data_unscaled.y, atol=1e-4))
        assert torch.all(torch.isclose(data.edge_attr, data_unscaled.edge_attr, atol=1e-4))
        assert torch.all(torch.isclose(data.u, data_unscaled.u, atol=1e-4))


class TestGraphEncoder:
    def test_forward_single_graph(self, dummy_dataset_global):
        data = dummy_dataset_global[0]
        encoder_config = {
            "node_fea_len": dummy_dataset_global.num_node_features,
            "edge_fea_len": dummy_dataset_global.num_edge_features,
            "global_fea_len": dummy_dataset_global.num_global_features,
            "n_conv": 3,
            "node_embedding_dim": 20,
            "edge_embedding_dim": 10,
            "global_embedding_dim": 10,
            "dropout": 0.1,
            "gat_heads": 2,
            "gat_concat": True,
            "update_edges": 1,
        }
        encoder = GATEncoder(**encoder_config)
        x_emb, edge_attr_emb, u_emb = encoder(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.u,
            torch.zeros((data.x.shape[0],), dtype=torch.long)
        )
        
        assert x_emb.shape[1] == 20
        assert edge_attr_emb.shape[1] == 10
        assert u_emb.shape[1] == 10
