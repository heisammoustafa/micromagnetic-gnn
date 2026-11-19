# """Tests for the explain module."""

# from __future__ import annotations

# import doctest
# import tempfile
# from pathlib import Path

# import pandas as pd
# import pytest
# import torch
# from torch_geometric.data import Data
# from torch_geometric.explain import Explanation

# from micromagnetic_gnn.explain import (
#     ExtendedGNNExplainer,
#     ModelWrapper,
#     feature_ablation,
#     plot_importance,
#     rank_importance,
# )
# # from micromagnetic_gnn.models import ResGCN


# # ---------------------------------------------------------------------------
# # Fixtures
# # ---------------------------------------------------------------------------


# @pytest.fixture
# def tiny_model() -> ResGCN:
#     return ResGCN(
#         node_fea_len=3,
#         edge_fea_len=2,
#         global_fea_len=2,
#         hidden=8,
#         global_hidden=4,
#         n_conv=2,
#         dropout=0.0,
#         task="global",
#     )


# @pytest.fixture
# def tiny_model_node() -> ResGCN:
#     return ResGCN(
#         node_fea_len=3,
#         edge_fea_len=2,
#         global_fea_len=2,
#         hidden=8,
#         global_hidden=4,
#         n_conv=2,
#         dropout=0.0,
#         task="node",
#     )


# @pytest.fixture
# def wrapper(tiny_model: ResGCN) -> ModelWrapper:
#     return ModelWrapper(
#         inner=tiny_model,
#         node_feature_names=["a", "b", "c"],
#         edge_feature_names=["d", "e"],
#         global_feature_names=["f", "g"],
#     )


# @pytest.fixture
# def graph_data() -> Data:
#     x = torch.randn(5, 3)
#     edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 0]], dtype=torch.long)
#     edge_attr = torch.randn(5, 2)
#     u = torch.randn(1, 2)
#     batch = torch.zeros(5, dtype=torch.long)
#     y = torch.randn(1)
#     return Data(
#         x=x,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         u=u,
#         batch=batch,
#         y=y,
#     )


# @pytest.fixture
# def node_graph_data() -> Data:
#     x = torch.randn(5, 3)
#     edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 0]], dtype=torch.long)
#     edge_attr = torch.randn(5, 2)
#     u = torch.randn(1, 2)
#     batch = torch.zeros(5, dtype=torch.long)
#     y = torch.randn(5)
#     return Data(
#         x=x,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         u=u,
#         batch=batch,
#         y=y,
#     )


# # ============================= ModelWrapper =============================


# class TestModelWrapper:
#     def test_forward_shape(self, wrapper: ModelWrapper, graph_data: Data) -> None:
#         out = wrapper(
#             graph_data.x,
#             graph_data.edge_index,
#             edge_attr=graph_data.edge_attr,
#             u=graph_data.u,
#             batch=graph_data.batch,
#         )
#         assert out.shape == (1,), f"Expected (1,), got {out.shape}"

#     def test_forward_node_task(self, tiny_model_node: ResGCN, node_graph_data: Data) -> None:
#         wrapper_node = ModelWrapper(
#             inner=tiny_model_node,
#             node_feature_names=["a", "b", "c"],
#             edge_feature_names=["d", "e"],
#             global_feature_names=["f", "g"],
#         )
#         out = wrapper_node(
#             node_graph_data.x,
#             node_graph_data.edge_index,
#             edge_attr=node_graph_data.edge_attr,
#             u=node_graph_data.u,
#             batch=node_graph_data.batch,
#         )
#         assert out.shape == (5,), f"Expected (5,), got {out.shape}"

#     def test_from_checkpoint_roundtrip(self, wrapper: ModelWrapper, graph_data: Data) -> None:
#         with tempfile.TemporaryDirectory() as tmp:
#             path = Path(tmp) / "model.pt"
#             ckpt = {
#                 "model_config": {
#                     "node_fea_len": 3,
#                     "edge_fea_len": 2,
#                     "global_fea_len": 2,
#                     "hidden": 8,
#                     "global_hidden": 4,
#                     "n_conv": 2,
#                     "dropout": 0.0,
#                     "task": "global",
#                 },
#                 "state_dict": wrapper.inner.state_dict(),
#                 "node_feature_names": ["a", "b", "c"],
#                 "edge_feature_names": ["d", "e"],
#                 "global_feature_names": ["f", "g"],
#             }
#             torch.save(ckpt, path)
#             loaded = ModelWrapper.from_checkpoint(path)
#             assert loaded.node_feature_names == ["a", "b", "c"]
#             assert loaded.edge_feature_names == ["d", "e"]
#             assert loaded.global_feature_names == ["f", "g"]

#             out = loaded(
#                 graph_data.x,
#                 graph_data.edge_index,
#                 edge_attr=graph_data.edge_attr,
#                 u=graph_data.u,
#                 batch=graph_data.batch,
#             )
#             assert out.shape == (1,)

#     def test_scale_data_preserves_shape(self, wrapper: ModelWrapper, graph_data: Data) -> None:
#         # Bypass fit_scalers (which needs a MicromagneticDataset) and
#         # directly set scaler means / stds to identity.
#         wrapper.inner.scaler.node_scaler.mean = torch.zeros(3)
#         wrapper.inner.scaler.node_scaler.std = torch.ones(3)
#         wrapper.inner.scaler.edge_scaler.mean = torch.zeros(2)
#         wrapper.inner.scaler.edge_scaler.std = torch.ones(2)
#         wrapper.inner.scaler.global_scaler.mean = torch.zeros(2)
#         wrapper.inner.scaler.global_scaler.std = torch.ones(2)
#         wrapper.inner.scaler.label_scaler.mean = torch.zeros(1)
#         wrapper.inner.scaler.label_scaler.std = torch.ones(1)

#         result = wrapper.scale_data(graph_data)
#         assert result.x.shape == graph_data.x.shape
#         assert result.edge_attr.shape == graph_data.edge_attr.shape


# # ========================= ExtendedGNNExplainer =========================


# class TestExtendedGNNExplainer:
#     def test_algorithm_supports(self) -> None:
#         alg = ExtendedGNNExplainer()
#         assert alg.supports()

#     def test_default_coeffs(self) -> None:
#         alg = ExtendedGNNExplainer()
#         for key in ("edge_size", "node_feat_size", "edge_attr_size", "u_size"):
#             assert key in alg.coeffs

#     def test_custom_coeffs(self) -> None:
#         alg = ExtendedGNNExplainer(edge_size=0.5, u_size=0.3)
#         assert alg.coeffs["edge_size"] == 0.5
#         assert alg.coeffs["u_size"] == 0.3

#     def test_graph_level_masks(self, wrapper: ModelWrapper, graph_data: Data) -> None:
#         from torch_geometric.explain import Explainer

#         explainer = Explainer(
#             model=wrapper,
#             algorithm=ExtendedGNNExplainer(epochs=5, lr=0.01),
#             explanation_type="model",
#             node_mask_type="attributes",
#             edge_mask_type="object",
#             model_config=dict(mode="regression", task_level="graph", return_type="raw"),
#         )
#         explanation = explainer(
#             graph_data.x,
#             graph_data.edge_index,
#             edge_attr=graph_data.edge_attr,
#             u=graph_data.u,
#             batch=graph_data.batch,
#         )

#         # Node mask: [num_nodes, num_node_features]
#         assert explanation.node_mask.shape == (5, 3), (
#             f"Expected (5, 3), got {explanation.node_mask.shape}"
#         )
#         # Edge mask: [num_edges]
#         assert explanation.edge_mask.shape == (5,), (
#             f"Expected (5,), got {explanation.edge_mask.shape}"
#         )
#         # Edge attr mask: [1, num_edge_features]
#         assert hasattr(explanation, "edge_attr_mask")
#         assert explanation.edge_attr_mask.shape == (1, 2), (
#             f"Expected (1, 2), got {explanation.edge_attr_mask.shape}"
#         )
#         # Global mask: [1, num_global_features]
#         assert hasattr(explanation, "u_mask")
#         assert explanation.u_mask.shape == (1, 2), (
#             f"Expected (1, 2), got {explanation.u_mask.shape}"
#         )

#         # Masks should be in [0, 1] after sigmoid
#         assert explanation.node_mask.min() >= 0.0
#         assert explanation.node_mask.max() <= 1.0
#         assert explanation.edge_mask.min() >= 0.0
#         assert explanation.edge_mask.max() <= 1.0

#     def test_node_level_masks(self, tiny_model_node: ResGCN, node_graph_data: Data) -> None:
#         from torch_geometric.explain import Explainer

#         wrapper_node = ModelWrapper(
#             inner=tiny_model_node,
#             node_feature_names=["a", "b", "c"],
#             edge_feature_names=["d", "e"],
#             global_feature_names=["f", "g"],
#         )
#         explainer = Explainer(
#             model=wrapper_node,
#             algorithm=ExtendedGNNExplainer(epochs=5, lr=0.01),
#             explanation_type="model",
#             node_mask_type="attributes",
#             edge_mask_type="object",
#             model_config=dict(mode="regression", task_level="node", return_type="raw"),
#         )
#         explanation = explainer(
#             node_graph_data.x,
#             node_graph_data.edge_index,
#             edge_attr=node_graph_data.edge_attr,
#             u=node_graph_data.u,
#             batch=node_graph_data.batch,
#             target=node_graph_data.y,
#             index=0,
#         )
#         assert explanation.node_mask.shape == (5, 3)
#         assert hasattr(explanation, "edge_attr_mask")
#         assert hasattr(explanation, "u_mask")

#     def test_masks_have_correct_shapes_with_various_dims(
#         self, wrapper: ModelWrapper, graph_data: Data
#     ) -> None:
#         """Verify mask shapes adapt to different feature dimensionalities."""
#         from torch_geometric.explain import Explainer

#         explainer = Explainer(
#             model=wrapper,
#             algorithm=ExtendedGNNExplainer(epochs=5, lr=0.01),
#             explanation_type="model",
#             node_mask_type="attributes",
#             edge_mask_type="object",
#             model_config=dict(mode="regression", task_level="graph", return_type="raw"),
#         )
#         explanation = explainer(
#             graph_data.x,
#             graph_data.edge_index,
#             edge_attr=graph_data.edge_attr,
#             u=graph_data.u,
#             batch=graph_data.batch,
#         )
#         assert explanation.node_mask.shape == (5, 3), "node_mask should be [N, F]"
#         assert explanation.edge_mask.shape == (5,), "edge_mask should be [E]"
#         assert explanation.edge_attr_mask.shape == (1, 2), "edge_attr_mask should be [1, EF]"
#         assert explanation.u_mask.shape == (1, 2), "u_mask should be [1, UF]"

#     def test_epochs_zero(self, wrapper: ModelWrapper, graph_data: Data) -> None:
#         """0 epochs should still produce valid masks (just initialised)."""
#         from torch_geometric.explain import Explainer

#         explainer = Explainer(
#             model=wrapper,
#             algorithm=ExtendedGNNExplainer(epochs=0, lr=0.01),
#             explanation_type="model",
#             node_mask_type="attributes",
#             edge_mask_type="object",
#             model_config=dict(mode="regression", task_level="graph", return_type="raw"),
#         )
#         explanation = explainer(
#             graph_data.x,
#             graph_data.edge_index,
#             edge_attr=graph_data.edge_attr,
#             u=graph_data.u,
#             batch=graph_data.batch,
#         )
#         assert explanation.node_mask is not None
#         assert explanation.edge_mask is not None

#     def test_clear_masks_restores_model(self, wrapper: ModelWrapper, graph_data: Data) -> None:
#         from torch_geometric.explain import Explainer

#         # Run explainer once
#         explainer = Explainer(
#             model=wrapper,
#             algorithm=ExtendedGNNExplainer(epochs=3, lr=0.01),
#             explanation_type="model",
#             node_mask_type="attributes",
#             edge_mask_type="object",
#             model_config=dict(mode="regression", task_level="graph", return_type="raw"),
#         )
#         _ = explainer(
#             graph_data.x,
#             graph_data.edge_index,
#             edge_attr=graph_data.edge_attr,
#             u=graph_data.u,
#             batch=graph_data.batch,
#         )

#         # After explainer, model should have no explain attributes
#         for module in wrapper.inner.modules():
#             if hasattr(module, "__explain__"):
#                 import torch_geometric.nn

#                 if isinstance(module, torch_geometric.nn.MessagePassing):
#                     pytest.fail("__explain__ attribute was not cleared")


# # ========================= rank_importance =============================


# class TestRankImportance:
#     def test_returns_dataframe(self) -> None:
#         explanation = Explanation(
#             node_mask=torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
#             edge_mask=torch.tensor([0.7, 0.3]),
#         )
#         explanation.edge_attr_mask = torch.tensor([[0.5, 0.5]])
#         explanation.u_mask = torch.tensor([[0.5]])
#         df = rank_importance(
#             explanation,
#             node_feature_names=["nf0", "nf1"],
#             edge_feature_names=["ef0", "ef1"],
#             global_feature_names=["gf0"],
#         )
#         assert isinstance(df, pd.DataFrame)
#         assert list(df.columns) == ["name", "importance", "type"]

#     def test_all_mask_types(self) -> None:
#         explanation = Explanation(
#             node_mask=torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
#             edge_mask=torch.tensor([0.7, 0.3]),
#         )
#         explanation.edge_attr_mask = torch.tensor([[0.8, 0.2, 0.9]])
#         explanation.u_mask = torch.tensor([[0.6, 0.4]])

#         df = rank_importance(
#             explanation,
#             node_feature_names=["nf1", "nf2"],
#             edge_feature_names=["ef1", "ef2", "ef3"],
#             global_feature_names=["gf1", "gf2"],
#         )
#         # 2 node + 1 edge + 3 edge_attr + 2 global = 8
#         assert len(df) == 8
#         # First row should be the highest importance
#         assert df.iloc[0]["importance"] >= df.iloc[1]["importance"]

#     def test_node_feature_names_used(self) -> None:
#         explanation = Explanation(
#             node_mask=torch.tensor([[0.9, 0.1]]),
#             edge_mask=torch.tensor([0.7]),
#         )
#         explanation.edge_attr_mask = torch.tensor([[0.5]])
#         explanation.u_mask = torch.tensor([[0.5]])
#         df = rank_importance(
#             explanation,
#             node_feature_names=["alpha", "beta"],
#             edge_feature_names=["ef0"],
#             global_feature_names=["gf0"],
#         )
#         assert df[df["type"] == "node"]["name"].tolist() == ["alpha", "beta"]

#     def test_edge_connectivity_entry(self) -> None:
#         explanation = Explanation(
#             node_mask=torch.tensor([[0.9, 0.1]]),
#             edge_mask=torch.tensor([0.7, 0.3]),
#         )
#         explanation.edge_attr_mask = torch.tensor([[0.5]])
#         explanation.u_mask = torch.tensor([[0.5]])
#         df = rank_importance(
#             explanation,
#             node_feature_names=["nf0", "nf1"],
#             edge_feature_names=["ef0"],
#             global_feature_names=["gf0"],
#         )
#         edge_rows = df[df["name"] == "edge (connectivity)"]
#         assert len(edge_rows) == 1
#         assert edge_rows.iloc[0]["importance"] == pytest.approx(0.5, abs=0.01)

#     def test_single_node_feature(self) -> None:
#         """node_mask with shape [N, 1] should produce one entry."""
#         explanation = Explanation(
#             node_mask=torch.tensor([[0.9], [0.8]]),
#             edge_mask=torch.tensor([0.7]),
#         )
#         explanation.edge_attr_mask = torch.tensor([[0.5]])
#         explanation.u_mask = torch.tensor([[0.5]])
#         df = rank_importance(
#             explanation,
#             node_feature_names=["only"],
#             edge_feature_names=["ef0"],
#             global_feature_names=["gf0"],
#         )
#         node_df = df[df["type"] == "node"]
#         assert len(node_df) == 1
#         assert node_df.iloc[0]["name"] == "only"


# # ========================= plot_importance =============================


# class TestPlotImportance:
#     def test_returns_figure(self) -> None:
#         df = pd.DataFrame(
#             {
#                 "name": ["a", "b"],
#                 "importance": [0.9, 0.1],
#                 "type": ["node", "edge"],
#             }
#         )
#         fig = plot_importance(df)
#         assert fig is not None
#         import matplotlib

#         assert isinstance(fig, matplotlib.figure.Figure)

#     def test_saves_to_file(self) -> None:
#         df = pd.DataFrame(
#             {
#                 "name": ["a"],
#                 "importance": [1.0],
#                 "type": ["node"],
#             }
#         )
#         with tempfile.TemporaryDirectory() as tmp:
#             path = Path(tmp) / "imp.png"
#             plot_importance(df, save_path=str(path))
#             assert path.exists()
#             assert path.stat().st_size > 0

#     def test_top_k_respected(self) -> None:
#         df = pd.DataFrame(
#             {
#                 "name": [f"f{i}" for i in range(10)],
#                 "importance": list(range(10, 0, -1)),
#                 "type": ["node"] * 10,
#             }
#         )
#         fig = plot_importance(df, top_k=3)
#         ax = fig.axes[0]
#         assert len(ax.get_yticklabels()) == 3


# # ========================= feature_ablation ============================


# class TestFeatureAblation:
#     def test_returns_dataframe(self, wrapper: ModelWrapper, graph_data: Data) -> None:
#         # Fit scalers first (needed for the model to work)
#         wrapper.inner.scaler.node_scaler.mean = torch.zeros(3)
#         wrapper.inner.scaler.node_scaler.std = torch.ones(3)
#         wrapper.inner.scaler.edge_scaler.mean = torch.zeros(2)
#         wrapper.inner.scaler.edge_scaler.std = torch.ones(2)
#         wrapper.inner.scaler.global_scaler.mean = torch.zeros(2)
#         wrapper.inner.scaler.global_scaler.std = torch.ones(2)
#         wrapper.inner.scaler.label_scaler.mean = torch.zeros(1)
#         wrapper.inner.scaler.label_scaler.std = torch.ones(1)

#         df = feature_ablation(
#             wrapper,
#             [graph_data],
#             node_feature_names=["a", "b", "c"],
#             edge_feature_names=["d", "e"],
#             global_feature_names=["f", "g"],
#         )
#         assert isinstance(df, pd.DataFrame)
#         assert list(df.columns) == ["name", "importance", "type"]
#         # 3 node + 2 edge + 2 global = 7 features
#         assert len(df) == 7

#     def test_sorted_by_importance_descending(self, wrapper: ModelWrapper, graph_data: Data) -> None:
#         wrapper.inner.scaler.node_scaler.mean = torch.zeros(3)
#         wrapper.inner.scaler.node_scaler.std = torch.ones(3)
#         wrapper.inner.scaler.edge_scaler.mean = torch.zeros(2)
#         wrapper.inner.scaler.edge_scaler.std = torch.ones(2)
#         wrapper.inner.scaler.global_scaler.mean = torch.zeros(2)
#         wrapper.inner.scaler.global_scaler.std = torch.ones(2)
#         wrapper.inner.scaler.label_scaler.mean = torch.zeros(1)
#         wrapper.inner.scaler.label_scaler.std = torch.ones(1)

#         df = feature_ablation(
#             wrapper,
#             [graph_data],
#             node_feature_names=["a", "b", "c"],
#             edge_feature_names=["d", "e"],
#             global_feature_names=["f", "g"],
#         )
#         for i in range(len(df) - 1):
#             assert df.iloc[i]["importance"] >= df.iloc[i + 1]["importance"], (
#                 "Importance should be sorted descending"
#             )

#     def test_invalid_baseline(self) -> None:
#         with pytest.raises(ValueError, match="Unsupported baseline"):
#             feature_ablation(
#                 model=torch.nn.Linear(1, 1),
#                 data_list=[],
#                 node_feature_names=[],
#                 edge_feature_names=[],
#                 global_feature_names=[],
#                 baseline="median",
#             )


# # ========================= doctests ===================================


# def test_doctests() -> None:
#     """Run doctests embedded in the explain module."""
#     import micromagnetic_gnn.explain as mod

#     results = doctest.testmod(mod, verbose=False, raise_on_error=False)
#     assert results.failed == 0, f"{results.failed} doctest(s) failed in explain module"


import pytest
import torch
from torch_geometric.data import Data

from micromagnetic_gnn.explain import (
    integrated_gradients,
    integrated_gradients_analysis,
    saliency,
)


class _SumModel(torch.nn.Module):
    def forward(self, data: Data) -> torch.Tensor:
        return data.x.sum() + data.edge_attr.sum() + data.u.sum()


def test_saliency_projects_unit_vector_gradients() -> None:
    """Unit-vector saliency excludes physically invalid radial gradients."""
    data = Data(
        x=torch.tensor([[1.0, 0.0, 0.0, 2.0]]),
        edge_attr=torch.tensor([[0.0, 1.0, 0.0, 3.0]]),
        u=torch.tensor([[4.0]]),
        node_feature_map={"easy_axis": (0, 1, 2), "node_scalar": (3,)},
        edge_feature_map={"face_normal": (0, 1, 2), "edge_scalar": (3,)},
        global_feature_map={"global_scalar": (0,)},
    )

    result = saliency(_SumModel(), data)

    torch.testing.assert_close(
        result.node_grad[0, :3], result.node_grad.new_tensor([0.0, 1.0, 1.0])
    )
    torch.testing.assert_close(
        result.edge_grad[0, :3], result.edge_grad.new_tensor([1.0, 0.0, 1.0])
    )
    assert result.node_grad[0, 3] == 1
    assert result.edge_grad[0, 3] == 1


def test_saliency_can_return_unprojected_gradients() -> None:
    """Projection can be disabled without changing its default behavior."""
    data = Data(
        x=torch.tensor([[1.0, 0.0, 0.0, 2.0]]),
        edge_attr=torch.tensor([[0.0, 1.0, 0.0, 3.0]]),
        u=torch.tensor([[4.0]]),
        node_feature_map={"easy_axis": (0, 1, 2), "node_scalar": (3,)},
        edge_feature_map={"face_normal": (0, 1, 2), "edge_scalar": (3,)},
        global_feature_map={"global_scalar": (0,)},
    )

    result = saliency(_SumModel(), data, project_vector_gradients=False)

    torch.testing.assert_close(result.node_grad, torch.ones_like(result.node_grad))
    torch.testing.assert_close(result.edge_grad, torch.ones_like(result.edge_grad))


def test_integrated_gradients_uses_mean_scalar_and_uniform_axis_baseline() -> None:
    """IG uses the configured references and satisfies completeness."""
    data = Data(
        x=torch.tensor([[1.0, 0.0, 0.0, 2.0]]),
        edge_attr=torch.tensor([[0.0, 1.0, 0.0, 3.0]]),
        u=torch.tensor([[4.0]]),
        node_feature_map={"easy_axis": (0, 1, 2), "node_scalar": (3,)},
        edge_feature_map={"face_normal": (0, 1, 2), "edge_scalar": (3,)},
        global_feature_map={"global_scalar": (0,)},
    )

    result = integrated_gradients(_SumModel(), data, steps=8)

    torch.testing.assert_close(
        result.baseline.x[0], result.baseline.x.new_tensor([0.0, 0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(result.baseline.edge_attr[0, :3], result.data.edge_attr[0, :3])
    assert result.baseline.edge_attr[0, 3] == 0
    assert result.baseline.u[0, 0] == 0
    torch.testing.assert_close(
        result.node_ig[0], result.node_ig.new_tensor([1.0, 0.0, -1.0, 2.0])
    )
    torch.testing.assert_close(
        result.edge_ig[0], result.edge_ig.new_tensor([0.0, 0.0, 0.0, 3.0])
    )
    torch.testing.assert_close(result.global_ig[0], result.global_ig.new_tensor([4.0]))
    torch.testing.assert_close(result.completeness_error, result.completeness_error.new_tensor(0.0))


def test_integrated_gradients_rejects_zero_steps() -> None:
    """IG requires at least one integration interval."""
    data = Data(
        x=torch.tensor([[0.0, 0.0, 1.0]]),
        edge_attr=torch.empty((0, 0)),
        u=torch.empty((1, 0)),
    )

    try:
        integrated_gradients(_SumModel(), data, steps=0)
    except ValueError as error:
        assert str(error) == "steps must be at least 1"
    else:
        raise AssertionError("Expected integrated_gradients to reject steps=0")


def test_integrated_gradients_analysis_balances_different_graph_sizes() -> None:
    """Global component moments give every graph equal weight."""
    feature_maps = {
        "node_feature_map": {"easy_axis": (0, 1, 2), "node_scalar": (3,)},
        "edge_feature_map": {"face_normal": (0, 1, 2), "edge_scalar": (3,)},
        "global_feature_map": {"global_scalar": (0,)},
    }
    small = Data(
        x=torch.tensor([[1.0, 0.0, 0.0, 2.0]]),
        edge_attr=torch.tensor([[0.0, 1.0, 0.0, 3.0]]),
        u=torch.tensor([[4.0]]),
        **feature_maps,
    )
    large = Data(
        x=torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
        ),
        edge_attr=torch.tensor(
            [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        ),
        u=torch.tensor([[2.0]]),
        **feature_maps,
    )

    analysis = integrated_gradients_analysis(_SumModel(), [small, large], steps=8)

    assert len(analysis.attribution_maps) == 2
    node_values = analysis.feature_distributions["node_scalar"]
    assert [values.numel() for values in node_values] == [1, 2]
    assert analysis.feature_stats["node_scalar"].mean == pytest.approx(5 / 18)
    assert analysis.feature_stats["node_scalar"].variance == pytest.approx(1 / 324)
    assert sum(stats.mean for stats in analysis.feature_stats.values()) == pytest.approx(1)
