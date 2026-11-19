"""Baseline GNN models and utilities for graph-level regression."""

from __future__ import annotations

from collections.abc import Callable
from itertools import chain
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GeneralConv, global_mean_pool
from torch_geometric.utils import scatter

from micromagnetic_gnn.data_pipeline import MicromagneticDataset


class StandardScalerModule(nn.Module):
    """Non-trainable StandardScaler stored as PyTorch buffers.

    Parameters
    ----------
    scale_indices : list[int]
        Scalar column indices to scale individually.
    vector_groups : list[list[int]]
        Groups of column indices that form vectors.  Each group is scaled
        uniformly (one mean / std shared across all components) so that
        vector direction is preserved.

    Examples
    --------
    >>> scaler = StandardScalerModule(features=11)
    >>> scaler.fit([torch.randn(100, 11), torch.randn(50, 11)], scale_indices=[6], vector_groups=[[0, 1, 2]])
    >>> out = scaler(torch.randn(5, 11))  # cols 6 scaled, cols 0-2 scaled uniformly
    """

    def __init__(self, features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("std", torch.zeros(features))

    def fit(
        self,
        tensors: list[torch.Tensor],
        scale_indices: list[int],
        vector_groups: list[list[int]] = [],
    ) -> StandardScalerModule:
        """Compute mean and std from a list of tensors.

        All tensors must have the same number of columns.
        """
        stacked = torch.cat(tensors, dim=0)
        if stacked.dim() == 1:
            stacked = stacked.unsqueeze(1)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        full_mean = torch.zeros_like(mean)
        full_std = torch.ones_like(std)

        # Scalar columns
        full_mean[scale_indices] = mean[scale_indices]
        full_std[scale_indices] = std[scale_indices]

        # Vector groups – uniform scaling per group
        for group in vector_groups:
            group_mean = stacked[:, group].mean()
            group_std = stacked[:, group].std()
            full_mean[group] = group_mean
            full_std[group] = group_std

        full_std = torch.where(full_std < 1e-5, 1.0, full_std)
        self.mean = full_mean
        self.std = full_std
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class GraphDataScaler(nn.Module):
    """Scaler for PyTorch Geometric ``Data`` objects.

    Holds four :class:`StandardScalerModule` instances (node, edge, global,
    label).  Call :meth:`fit` once on the training dataset, then store the
    scaler (saved in ``state_dict``).
    """

    def __init__(
        self,
        node_fea_len: int,
        edge_fea_len: int,
        global_fea_len: int,
        scale_label: bool = True
    ) -> None:
        super().__init__()
        self.scale_label = scale_label
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len
        self.global_fea_len = global_fea_len
        self.node_scaler = StandardScalerModule(node_fea_len)
        self.edge_scaler = StandardScalerModule(edge_fea_len)
        self.global_scaler = StandardScalerModule(global_fea_len)
        self.label_scaler = StandardScalerModule(1)
    
    @property
    def config(self) -> dict[str, Any]:
        return {
            "node_fea_len": self.node_fea_len,
            "edge_fea_len": self.edge_fea_len,
            "global_fea_len": self.global_fea_len,
            "scale_label": self.scale_label,
        }
        
    def fit(self, dataset: MicromagneticDataset) -> GraphDataScaler:
        """Fit all four scalers on *dataset* (should be training data only)."""
        node_scale_indices = list(
            chain.from_iterable(
                list(v) for k, v in dataset.node_feature_map.items() if k != "easy_axis"
            )
        )
        self.node_scaler.fit(
            [d.x for d in dataset],
            scale_indices=node_scale_indices,
        )

        edge_vector_groups_names = ("center_distance", "face_distance", "face_normal")
        edge_vector_groups = [
            list(dataset.edge_feature_map[g]) for g in edge_vector_groups_names
        ]
        edge_scale_indices = list(
            chain.from_iterable(
                list(v)
                for k, v in dataset.edge_feature_map.items()
                if k not in edge_vector_groups_names
            )
        )
        self.edge_scaler.fit(
            [d.edge_attr for d in dataset],
            scale_indices=edge_scale_indices,
            vector_groups=edge_vector_groups,
        )

        global_scale_indices = list(
            chain.from_iterable(
                list(v) for k, v in dataset.global_feature_map.items() if k != "applied_field"
            )
        )
        self.global_scaler.fit([d.u for d in dataset], scale_indices=global_scale_indices)
        #label_index = [0] if self.scale_label else []
        if self.scale_label:
            self.label_scaler.fit([d.y for d in dataset], scale_indices=[0])
        return self

    def forward(self, data: Data) -> Data:
        """Apply forward scalers to a PyG ``Data`` object **in-place**.

        Use this to pre-process datasets before creating DataLoaders.
        """
        data.x = self.node_scaler(data.x)
        data.edge_attr = self.edge_scaler(data.edge_attr)
        data.u = self.global_scaler(data.u)
        if hasattr(data, "y") and self.scale_label:
            data.y = self.label_scaler(data.y)
        return data

    def unscale(self, data: Data) -> Data:
        """Inverse-transform predicted labels back to original units."""
        data = data.clone()
        data.x = self.node_scaler.inverse_transform(data.x)
        data.edge_attr = self.edge_scaler.inverse_transform(data.edge_attr)
        data.u = self.global_scaler.inverse_transform(data.u)
        if hasattr(data, "y") and self.scale_label:
            data.y = self.label_scaler.inverse_transform(data.y)
        return data

    @classmethod
    def fit_from_dataset(cls, dataset: MicromagneticDataset, scale_label: bool = True) -> GraphDataScaler:
        scaler = cls(dataset.num_node_features, dataset.num_edge_features, dataset.num_global_features, scale_label=scale_label)
        scaler.fit(dataset)
        return scaler

    def scale_dataset(self, dataset: MicromagneticDataset) -> MicromagneticDataset:
        scaled_data = [self(d) for d in dataset]
        scaled_dataset = dataset._clone_with_data(scaled_data)
        return scaled_dataset
    
    def unscale_dataset(self, dataset: MicromagneticDataset) -> MicromagneticDataset:
        scaled_data = [self.unscale(d) for d in dataset]
        scaled_dataset = dataset._clone_with_data(scaled_data)
        return scaled_dataset


class GATBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        global_channels: int,
        edge_dim: int,
        heads: int = 1,
        concat: bool = True,
        update_edges: bool = True,
        activation: type[nn.Module] = nn.SiLU
    ) -> None:
        super().__init__()

        self.conv = GATConv(channels, channels, heads=heads, edge_dim=edge_dim, concat=concat)
        in_channels = channels * heads if concat else channels
        self.x_proj = nn.Linear(in_channels, channels)
        self.u_proj = nn.Linear(channels, global_channels)
        self.u_to_x = nn.Linear(global_channels, channels)
        self.activation = activation()

        if update_edges:
            edge_in_channels = edge_dim + 2 * channels + global_channels
            self.edge_mlp = nn.Sequential(nn.Linear(edge_in_channels, edge_dim), activation())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = x

        # 1. GAT / message passing (uses OLD edges)
        x = self.conv(x, edge_index, edge_attr=edge_attr)

        # 2. project back
        x = self.x_proj(x)

        # 3. EDGE UPDATE (after node interaction, before global + activation)
        if hasattr(self, "edge_mlp"):
            edge_attr = self._edge_update(x, edge_index, edge_attr, u, batch)

        # 4. global update
        pooled = global_mean_pool(x, batch)
        u = u + self.activation(self.u_proj(pooled))

        # 5. global → node injection
        x = x + self.u_to_x(u)[batch]

        # 6. activation
        x = self.activation(x)

        # 7. residual connection
        x = x + res

        return x, edge_attr, u

    def _edge_update(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        assert hasattr(self, "edge_mlp")

        src, dst = edge_index
        edge_batch = batch[src]
        u_edge = u[edge_batch]

        x_i = x[src]
        x_j = x[dst]

        edge_input = torch.cat([x_i, x_j, edge_attr, u_edge], dim=-1)
        edge_attr = self.edge_mlp(edge_input)
        return edge_attr
    

class EdgeAggregation(Module):
    def __init__(self, node_dim: int, edge_dim: int, global_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        #self.x_proj = nn.Linear(edge_dim + global_dim, node_dim)
        #self.edge_proj = nn.Linear(node_dim + edge_dim, edge_dim)
        self.src_msg_lin = nn.Linear(node_dim, edge_dim)
        self.edge_msg_lin = nn.Linear(edge_dim, edge_dim)
        self.dst_msg_lin = nn.Linear(node_dim, edge_dim)
        self.src_lin = nn.Linear(edge_dim, node_dim)
        self.u_lin = nn.Linear(global_dim, node_dim)
        
    def forward(
        self,
        x: Tensor,           # [num_nodes, node_dim]
        edge_index: Tensor,  # [2, num_edges]
        edge_attr: Tensor,   # [num_edges, edge_dim]
        u: Tensor,           # [num_graphs, global_dim]
        batch: Tensor,       # [num_nodes]
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        msg = self.src_msg_lin(x[src]) + self.dst_msg_lin(x[dst]) + self.edge_msg_lin(edge_attr)
        # node aggregation
        z = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce="mean")
        u = self.u_lin(u)
        z = self.src_lin(z) + u[batch]
        return z, msg
    

class EdgeAggConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        global_channels: int,
        edge_dim: int,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.conv = EdgeAggregation(channels, edge_dim, global_channels)
        self.u_proj = nn.Linear(channels, global_channels)
        self.activation = activation()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = x
        # 1. edge aggregation / message passing
        x, e = self.conv(x, edge_index, edge_attr=edge_attr, u=u, batch=batch)
        # 2. edge update
        edge_attr = edge_attr + self.activation(e)
        # 3. global update
        pooled = global_mean_pool(x, batch)
        u = u + self.activation(self.u_proj(pooled))
        # 6. activation + residual connection
        x = self.activation(x) + res
        return x, edge_attr, u


class GATEncoder(nn.Module):
    def __init__(
        self,
        node_fea_len: int,
        edge_fea_len: int,
        global_fea_len: int,
        *,
        n_conv: int,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        global_embedding_dim: int,
        dropout: float = 0.0,
        gat_heads: int = 1,
        gat_concat: bool = True,
        update_edges: int = 1,
        activation: type[nn.Module] = nn.SiLU
    ) -> None:
        super().__init__()
        assert node_fea_len > 0 and edge_fea_len > 0, "This model requires node and edge features"
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len
        self.global_fea_len = global_fea_len
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim

        self.node_embedding = nn.Sequential(
            nn.Linear(node_fea_len, node_embedding_dim),
            activation(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_fea_len, edge_embedding_dim),
            activation(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
        )
        
        self.global_embedding: nn.Module | None = (nn.Linear(global_fea_len, global_embedding_dim)
                                                   if global_fea_len > 0 else None)

        _update_edges = [i % update_edges == 0 for i in range(n_conv)] if update_edges else [False for i in range(n_conv)]
        self.blocks = nn.ModuleList(
            [
                GATBlock(node_embedding_dim, global_channels=global_embedding_dim, edge_dim=edge_embedding_dim,
                         concat=gat_concat, heads=gat_heads, update_edges=ue, activation=activation)
                for _, ue in zip(range(n_conv), _update_edges, strict=True)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor | None,
        batch: torch.Tensor,
        num_graphs: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_emb = self.node_embedding(x)
        edge_attr_emb = self.edge_embedding(edge_attr)
        if self.global_embedding is None:
            assert num_graphs is not None, "Provide argument `num_graphs` to the encoder."
            u_emb = torch.zeros((num_graphs, self.global_embedding_dim), dtype=x.dtype, device=x.device)
        else:
            u_emb = self.global_embedding(u)

        for block in self.blocks[:-1]:
            x_emb, edge_attr_emb, u_emb = block(x_emb, edge_index, edge_attr_emb, u_emb, batch)
            x_emb = self.dropout(x_emb)  # apply dropout only to second last layer
            
        x_emb, edge_attr_emb, u_emb = self.blocks[-1](x_emb, edge_index, edge_attr_emb, u_emb, batch)

        return x_emb, edge_attr_emb, u_emb


class EdgeAggEncoder(nn.Module):
    def __init__(
        self,
        node_fea_len: int,
        edge_fea_len: int,
        global_fea_len: int,
        *,
        n_conv: int,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        global_embedding_dim: int,
        dropout: float = 0.0,
        update_edges: int = 1,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        assert node_fea_len > 0 and edge_fea_len > 0, "This model requires node and edge features"
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len
        self.global_fea_len = global_fea_len
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim

        self.node_embedding = nn.Sequential(
            nn.Linear(node_fea_len, node_embedding_dim),
            activation(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_fea_len, edge_embedding_dim),
            activation(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
        )
        
        self.global_embedding: nn.Module | None = (
            nn.Sequential(
                nn.Linear(global_fea_len, global_embedding_dim),
                activation(),
                nn.Linear(global_embedding_dim, global_embedding_dim),
            )
            if global_fea_len > 0 else None
        )

        self.blocks = nn.ModuleList(
            [
                EdgeAggConvBlock(node_embedding_dim, global_channels=global_embedding_dim, edge_dim=edge_embedding_dim, activation=activation)
                for _ in range(n_conv)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor | None,
        batch: torch.Tensor,
        num_graphs: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_emb = self.node_embedding(x)
        edge_attr_emb = self.edge_embedding(edge_attr)
        if self.global_embedding is None:
            num_graphs = 1 if num_graphs is None else num_graphs  # assert num_graphs is not None, "Provide argument `num_graphs` to the encoder."
            u_emb = torch.zeros((num_graphs, self.global_embedding_dim), dtype=x.dtype, device=x.device)
        else:
            u_emb = self.global_embedding(u)

        for block in self.blocks[:-1]:
            x_emb, edge_attr_emb, u_emb = block(x_emb, edge_index, edge_attr_emb, u_emb, batch)
            x_emb = self.dropout(x_emb)  # apply dropout only to second last layer
            
        x_emb, edge_attr_emb, u_emb = self.blocks[-1](x_emb, edge_index, edge_attr_emb, u_emb, batch)

        return x_emb, edge_attr_emb, u_emb


class GraphModel(nn.Module):
    """Graph-level regression head on top of an encoder.

    Parameters
    ----------
    encoder : GATEncoder
        Graph encoder producing a global embedding.
    head : nn.Module, optional
        Custom output head.  Defaults to a linear layer mapping the global
        embedding to ``2`` outputs when ``predict_variance`` else ``1``.
    predict_variance : bool
        If ``True``, the model predicts a per-graph mean **and** variance for
        Gaussian negative-log-likelihood (aleatoric) uncertainty.  ``forward``
        then returns a ``(mean, var)`` tuple instead of a single mean tensor.
        Default ``False`` (drop-in with the original mean-only behaviour).
    var_eps : float
        Floor added to the softplus-transformed variance for numerical
        stability.  Only used when ``predict_variance``.
    """

    def __init__(
        self,
        encoder: GATEncoder,
        head: nn.Module | None = None,
        predict_variance: bool = False,
        var_eps: float = 1e-6,
    ):
        super().__init__()
        self.encoder = encoder
        self.predict_variance = predict_variance
        self.var_eps = var_eps
        out_dim = 2 if predict_variance else 1
        self.head = (
            nn.Linear(encoder.global_embedding_dim, out_dim)
            if head is None else head
        )

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
        num_graphs = data.num_graphs if isinstance(data, Batch) else None
        batch = data.batch if isinstance(data, Batch) else torch.zeros((x.shape[0],), dtype=torch.int64, device=x.device)
        x_emb, _, u_emb = self.encoder(x, edge_index, edge_attr, u, batch, num_graphs)
        out = cast(torch.Tensor, self.head(u_emb))
        if self.predict_variance:
            mean = out[:, 0:1]
            var = nn.functional.softplus(out[:, 1:2]) + self.var_eps
            return mean, var
        return out
    
    
class NodeModel(nn.Module):
    def __init__(
        self, 
        encoder: GATEncoder,
        head: nn.Module | None = None,
        out_activation: Callable | None = None
    ):
        super().__init__()
        self.encoder = encoder
        self.out_activation = out_activation
        self.head = (
            nn.Linear(encoder.node_embedding_dim, 1)
            if head is None else head
        )
    
    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        x, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
        num_graphs = data.num_graphs if isinstance(data, Batch) else None
        batch = data.batch if isinstance(data, Batch) else torch.zeros((x.shape[0],), dtype=torch.int64, device=x.device)

        x_emb, *_ = self.encoder(x, edge_index, edge_attr, u, batch, num_graphs)
        prediction = cast(torch.Tensor, self.head(x_emb))
        if self.out_activation is not None:
            prediction = self.out_activation(prediction)
            
        if hasattr(data, "label_index"):
            prediction = prediction[data.label_index]
        return prediction
    

class LinkModel(nn.Module):
    def __init__(self, encoder: GATEncoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = BilinearDecoder(encoder.node_embedding_dim)
 
    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        x, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
        num_graphs = data.num_graphs if isinstance(data, Batch) else None
        batch = data.batch if isinstance(data, Batch) else torch.zeros((x.shape[0],), dtype=torch.int64)
        edge_label_index = data.edge_label_index
        
        x_emb, *_ = self.encoder(x, edge_index, edge_attr, u, batch, num_graphs)
        link_logits = cast(torch.Tensor, self.decoder(x_emb, edge_label_index))
        return torch.sigmoid(link_logits)


class BilinearDecoder(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, h: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        h_i, h_j = h[src], h[dst]
        scores = torch.sum((h_i @ self.W) * h_j, dim=-1)
        return scores.unsqueeze(1)
