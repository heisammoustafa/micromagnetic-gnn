
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import torch
from torch_geometric.loader import DataLoader

from micromagnetic_gnn.data_pipeline import Data, MicromagneticDataset


def permute_feature(dataset: MicromagneticDataset, feature_name: str) -> MicromagneticDataset:
    data = list(dataset)
    if feature_name in dataset.node_feature_map:
        attr = "x"
        permute_idx = dataset.node_feature_map[feature_name]
    elif feature_name in dataset.edge_feature_map:
        attr = "edge_attr"
        permute_idx = dataset.edge_feature_map[feature_name]
    elif feature_name in dataset.global_feature_map:
        attr = "u"
        permute_idx = dataset.global_feature_map[feature_name]
    else:
        raise KeyError(f"feature {feature_name} not found.")
    
    x = torch.concat([getattr(d, attr) for d in data], axis=0)
    i = torch.randperm(x.shape[0])
    xi = x[:, permute_idx]
    xi = xi[i, :]
    x[:, permute_idx] = xi

    p1 = 0
    data_perm = []
    for d in data:
        d_perm = d.clone()
        ni = getattr(d_perm, attr).shape[0]
        p2 = p1 + ni
        setattr(d_perm, attr, x[p1:p2])
        data_perm.append(d_perm)
        p1 += ni
        
    ds = dataset._clone_with_data(data_perm)
    ds.transform = None
    ds_orig = dataset._clone_with_data(data)
    ds_orig.transform = None
    return ds, ds_orig
    

def permutation_feature_importance(model, metric, ds: MicromagneticDataset, larger_is_better: bool = False, runs: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = torch.compile(model).to(device)
    features = list(chain.from_iterable([ds.node_feature_map.keys(), ds.edge_feature_map.keys(), ds.global_feature_map.keys()]))
    perm_importance = {}

    for f in features:
        importances = []
        for _ in range(runs):
            ds, ds_orig = permute_feature(ds, f)
            importance = _feature_importance(model, metric, ds, ds_orig)
            importances.append(importance)
        
        _importances = torch.tensor(importances)
        
        importance_mean = torch.mean(_importances)
        importance_std = torch.std(_importances)
        importance = importance if larger_is_better else -importance
        perm_importance[f] = (importance_mean, importance_std)
        #print(f, importance_mean, importance_std)
        
    return perm_importance

def _feature_importance(model, metric, ds: MicromagneticDataset, ds_orig: MicromagneticDataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    loader_orig = DataLoader(ds_orig, batch_size=32, shuffle=False)

    n_batches = 0
    total_m = 0.0
    total_m_orig = 0.0
    
    with torch.no_grad():
        for d, d_orig in zip(loader, loader_orig, strict=True):
            d, d_orig = d.to(device), d_orig.to(device)
            out_orig = model(d_orig)
            out = model(d)
            m_orig = metric(out_orig.view(-1), d_orig.y.view(-1))
            m = metric(out.view(-1), d.y.view(-1))
            total_m += m
            total_m_orig += m_orig
            n_batches += 1

    m = total_m / max(n_batches, 1)
    
    m_orig = total_m_orig / max(n_batches, 1)
    #print(m, m_orig)
    importance = m_orig - m
    return importance


@dataclass
class Saliency:
    data: Data
    node_grad: torch.Tensor
    edge_grad: torch.Tensor
    global_grad: torch.Tensor
    node_saliency: torch.Tensor
    edge_saliency: torch.Tensor
    node_feature_score: dict[str, torch.Tensor]
    edge_feature_score: dict[str, torch.Tensor]
    global_feature_score: dict[str, torch.Tensor]
    
    @property
    def total_node_saliency(self) -> torch.Tensor:
        n = self.data.x.shape[0]
        def _node_edge_score(i):
            edges = self.data.edge_index[0] == i
            return torch.where(edges, self.edge_saliency, 0).sum()
        
        with torch.no_grad():
            idx = torch.arange(n).to(self.data.x.device)
            node_scores = self.node_saliency
            edge_scores = torch.vmap(_node_edge_score)(idx)
        
        return node_scores + edge_scores


@dataclass
class SaliencyAnalysis:
    saliency_maps: list[Saliency]
    node_feature_scores: dict[str, float]
    edge_feature_scores: dict[str, float]
    global_feature_scores: dict[str, float]

    @property
    def scores(self) -> dict[str, float]:
        s = self.node_feature_scores | self.edge_feature_scores | self.global_feature_scores
        return dict(sorted(s.items(), key=lambda d: -d[1]))


@dataclass
class IntegratedGradients:
    """Integrated-gradient attributions for one graph."""

    data: Data
    baseline: Data
    node_ig: torch.Tensor
    edge_ig: torch.Tensor
    global_ig: torch.Tensor
    prediction: torch.Tensor
    baseline_prediction: torch.Tensor

    @property
    def completeness_error(self) -> torch.Tensor:
        """Difference between summed attributions and the output change."""
        attribution = self.node_ig.sum() + self.edge_ig.sum() + self.global_ig.sum()
        return attribution - (self.prediction - self.baseline_prediction)

    @property
    def output_delta(self) -> torch.Tensor:
        """Prediction difference explained by the attributions."""
        return self.prediction - self.baseline_prediction

    def cpu(self) -> "IntegratedGradients":
        """Return a detached CPU copy suitable for dataset-level storage."""
        return IntegratedGradients(
            data=self.data.clone().to("cpu"),
            baseline=self.baseline.clone().to("cpu"),
            node_ig=self.node_ig.detach().cpu(),
            edge_ig=self.edge_ig.detach().cpu(),
            global_ig=self.global_ig.detach().cpu(),
            prediction=self.prediction.detach().cpu(),
            baseline_prediction=self.baseline_prediction.detach().cpu(),
        )


@dataclass(frozen=True)
class IntegratedGradientFeatureStats:
    """Graph-balanced global moments for one input component."""

    feature_type: str
    mean: float
    variance: float

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return self.variance ** 0.5


@dataclass
class IntegratedGradientsAnalysis:
    """Per-graph IG results and global component-level statistics."""

    attribution_maps: list[IntegratedGradients]
    feature_stats: dict[str, IntegratedGradientFeatureStats]
    feature_distributions: dict[str, list[torch.Tensor]]


def saliency_analysis(model, dataset) -> SaliencyAnalysis:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)
    saliencies = []
    
    for d in dataset:
        saliencies.append(saliency(model, d))
        
    return SaliencyAnalysis(
        saliency_maps=saliencies,
        node_feature_scores=_saliency_feature_scores([s.node_feature_score for s in saliencies]),
        edge_feature_scores=_saliency_feature_scores([s.edge_feature_score for s in saliencies]),
        global_feature_scores=_saliency_feature_scores([s.global_feature_score for s in saliencies]),
    )


def _saliency_feature_scores(scores: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    feature_names = scores[0].keys()
    _scores = {f: float(torch.tensor([s[f] for s in scores]).mean().detach().to("cpu"))
               for f in feature_names}
    return _scores


def _project_vector_gradient(values: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    """Project gradients onto the tangent plane of normalized vectors."""
    directions = values / values.norm(dim=1, keepdim=True)
    return gradients - (gradients * directions).sum(dim=1, keepdim=True) * directions


def _scalar_prediction(model: torch.nn.Module, data: Data) -> torch.Tensor:
    prediction = model(data)
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    return prediction.mean()


def _integrated_gradients_baseline(data: Data) -> Data:
    """Create a mean-scalar, uniform-easy-axis baseline in scaled space."""
    baseline = data.clone()
    vector_features = {
        "easy_axis",
        "center_distance",
        "face_distance",
        "face_normal",
        "applied_field",
    }
    for values, feature_map in (
        (baseline.x, baseline.node_feature_map),
        (baseline.edge_attr, baseline.edge_feature_map),
        (baseline.u, baseline.global_feature_map),
    ):
        for name, idx in feature_map.items():
            if name not in vector_features:
                values[:, idx] = 0

    if "easy_axis" in baseline.node_feature_map:
        idx = baseline.node_feature_map["easy_axis"]
        baseline.x[:, idx] = baseline.x.new_tensor((0.0, 0.0, 1.0))
    return baseline


def _spherical_interpolation(
    start: torch.Tensor,
    end: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Interpolate unit vectors along a great circle."""
    start = start / start.norm(dim=1, keepdim=True)
    end = end / end.norm(dim=1, keepdim=True)
    dot = (start * end).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    tangent = end - dot * start
    tangent_norm = tangent.norm(dim=1, keepdim=True)

    basis = torch.zeros_like(start)
    basis.scatter_(1, start.abs().argmin(dim=1, keepdim=True), 1)
    fallback = basis - (basis * start).sum(dim=1, keepdim=True) * start
    fallback = fallback / fallback.norm(dim=1, keepdim=True)
    tangent = torch.where(
        tangent_norm > 1e-7,
        tangent / tangent_norm.clamp_min(1e-7),
        fallback,
    )
    interpolated = torch.cos(alpha * angle) * start + torch.sin(alpha * angle) * tangent
    return torch.where(angle < 1e-7, start, interpolated)


def _integration_point(data: Data, baseline: Data, alpha: float) -> Data:
    point = baseline.clone()
    point.x = baseline.x + alpha * (data.x - baseline.x)
    point.edge_attr = baseline.edge_attr + alpha * (data.edge_attr - baseline.edge_attr)
    point.u = baseline.u + alpha * (data.u - baseline.u)
    if "easy_axis" in data.node_feature_map:
        idx = data.node_feature_map["easy_axis"]
        point.x[:, idx] = _spherical_interpolation(
            baseline.x[:, idx], data.x[:, idx], alpha
        )
    return point


def _point_gradient(
    model: torch.nn.Module,
    data: Data,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    data.x = data.x.detach().requires_grad_(True)
    data.edge_attr = data.edge_attr.detach().requires_grad_(True)
    data.u = data.u.detach().requires_grad_(True)
    prediction = _scalar_prediction(model, data)
    gradients = torch.autograd.grad(
        prediction,
        (data.x, data.edge_attr, data.u),
    )
    return prediction.detach(), tuple(gradient.detach() for gradient in gradients)


def integrated_gradients(
    model: torch.nn.Module,
    data: Data,
    steps: int = 50,
) -> IntegratedGradients:
    """Attribute one graph relative to a generated reference graph.

    Scalar inputs must already be standardized using the training-set scaler,
    so zero represents their training mean. The baseline retains all vector
    features except ``easy_axis``, which is set to ``(0, 0, 1)``. Easy axes
    follow a unit-sphere path; other features follow straight paths.

    Parameters
    ----------
    model : torch.nn.Module
        Model accepting a :class:`~torch_geometric.data.Data` object.
    data : Data
        Scaled graph presented to the model.
    steps : int, optional
        Number of trapezoidal integration intervals, by default 50.

    Returns
    -------
    IntegratedGradients
        Componentwise node, edge and global attributions.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    data = data.clone().to(device)
    data.x = data.x.detach()
    data.edge_attr = data.edge_attr.detach()
    data.u = data.u.detach()
    baseline = _integrated_gradients_baseline(data)

    previous = _integration_point(data, baseline, 0.0)
    baseline_prediction, previous_gradients = _point_gradient(model, previous)
    attributions = [
        torch.zeros_like(data.x),
        torch.zeros_like(data.edge_attr),
        torch.zeros_like(data.u),
    ]
    previous_values = (previous.x.detach(), previous.edge_attr.detach(), previous.u.detach())

    prediction = baseline_prediction
    for step in range(1, steps + 1):
        point = _integration_point(data, baseline, step / steps)
        prediction, gradients = _point_gradient(model, point)
        values = (point.x.detach(), point.edge_attr.detach(), point.u.detach())
        for attribution, previous_gradient, gradient, previous_value, value in zip(
            attributions,
            previous_gradients,
            gradients,
            previous_values,
            values,
            strict=True,
        ):
            attribution += 0.5 * (previous_gradient + gradient) * (value - previous_value)
        previous_gradients = gradients
        previous_values = values

    return IntegratedGradients(
        data=data,
        baseline=baseline,
        node_ig=attributions[0],
        edge_ig=attributions[1],
        global_ig=attributions[2],
        prediction=prediction,
        baseline_prediction=baseline_prediction,
    )


def _component_attributions(
    result: IntegratedGradients,
) -> Iterable[tuple[str, str, torch.Tensor]]:
    for feature_type, attributions, feature_map in (
        ("node", result.node_ig, result.data.node_feature_map),
        ("edge", result.edge_ig, result.data.edge_feature_map),
        ("global", result.global_ig, result.data.global_feature_map),
    ):
        for feature, indices in feature_map.items():
            names = (
                (feature,)
                if len(indices) == 1
                else tuple(f"{feature}[{component}]" for component in range(len(indices)))
            )
            for name, index in zip(names, indices, strict=True):
                yield name, feature_type, attributions[:, index]


def integrated_gradients_analysis(
    model: torch.nn.Module,
    dataset: Iterable[Data],
    steps: int = 50,
    min_output_delta: float = 1e-6,
) -> IntegratedGradientsAnalysis:
    """Compute per-graph IG and graph-balanced global component statistics.

    Each graph's attribution is divided by its input-to-baseline output
    difference. Node and edge values are then multiplied by that graph's
    entity count. Consequently, the mean of a component's values within one
    graph equals its normalized total contribution, while every graph has
    equal weight in the global mean and variance.

    Parameters
    ----------
    model : torch.nn.Module
        Model accepting a :class:`~torch_geometric.data.Data` object.
    dataset : Iterable[Data]
        Scaled graphs presented to the model.
    steps : int, optional
        Number of integration intervals per graph, by default 50.
    min_output_delta : float, optional
        Reject graphs whose absolute input-to-baseline prediction difference
        is smaller than this value, by default ``1e-6``.

    Returns
    -------
    IntegratedGradientsAnalysis
        Raw per-graph attributions, normalized per-graph distributions, and
        global population moments for every scalar or vector component.
    """
    if min_output_delta <= 0:
        raise ValueError("min_output_delta must be positive")

    attribution_maps = []
    feature_distributions: dict[str, list[torch.Tensor]] = {}
    feature_types: dict[str, str] = {}
    for graph_index, data in enumerate(dataset):
        result = integrated_gradients(model, data, steps=steps).cpu()
        output_delta = float(result.output_delta)
        if abs(output_delta) < min_output_delta:
            raise ValueError(
                f"Graph {graph_index} has |F(x) - F(baseline)| = "
                f"{abs(output_delta):.3e}, below min_output_delta={min_output_delta:.3e}"
            )

        attribution_maps.append(result)
        for name, feature_type, values in _component_attributions(result):
            entity_count = values.numel()
            distribution = (
                values * (entity_count / output_delta)
                if entity_count
                else values.new_zeros(1)
            )
            feature_distributions.setdefault(name, []).append(distribution)
            feature_types[name] = feature_type

    if not attribution_maps:
        raise ValueError("dataset must contain at least one graph")

    feature_stats = {}
    for name, distributions in feature_distributions.items():
        graph_means = torch.stack([values.mean() for values in distributions])
        mean = graph_means.mean()
        variance = torch.stack(
            [((values - mean) ** 2).mean() for values in distributions]
        ).mean()
        feature_stats[name] = IntegratedGradientFeatureStats(
            feature_type=feature_types[name],
            mean=float(mean),
            variance=float(variance),
        )

    return IntegratedGradientsAnalysis(
        attribution_maps=attribution_maps,
        feature_stats=feature_stats,
        feature_distributions=feature_distributions,
    )


def saliency(model, data, project_vector_gradients: bool = True):
    """Compute input gradients, optionally projected for unit-vector features."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)
    data.x.requires_grad_(True)
    data.edge_attr.requires_grad_(True)
    data.u.requires_grad_(True)
    
    pred = _scalar_prediction(model, data)
    dx, dedge_attr, du = torch.autograd.grad(
        outputs=pred,
        inputs=(data.x, data.edge_attr, data.u),
    )

    if project_vector_gradients:
        dx = dx.clone()
        dedge_attr = dedge_attr.clone()
        with torch.no_grad():
            for values, gradients, feature_map, name in (
                (data.x, dx, data.node_feature_map, "easy_axis"),
                (data.edge_attr, dedge_attr, data.edge_feature_map, "face_normal"),
            ):
                if name in feature_map:
                    idx = feature_map[name]
                    gradients[:, idx] = _project_vector_gradient(
                        values[:, idx], gradients[:, idx]
                    )
    
    return Saliency(
        data=data.clone(),
        node_grad=dx,
        edge_grad=dedge_attr,
        global_grad=du,
        node_saliency=_saliency_score(data.node_feature_map, dx),
        edge_saliency=_saliency_score(data.edge_feature_map, dedge_attr),
        node_feature_score=_saliency_feature_score(data.node_feature_map, dx),
        edge_feature_score=_saliency_feature_score(data.edge_feature_map, dedge_attr),
        global_feature_score=_saliency_feature_score(data.global_feature_map, du),
    )


def _saliency_score(feature_map, sm):
    _sm = []
    for f in feature_map:
        _sm.append(sm[:, feature_map[f]].abs().sum(axis=1))
        
    scores = torch.stack(_sm)
    return scores.sum(0)


def _saliency_feature_score(feature_map, sm):
    _sm = {}
    for f in feature_map:
        _sm[f] = sm[:, feature_map[f]].abs().sum()
        
    return _sm
