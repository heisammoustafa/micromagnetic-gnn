from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from micromagnetic_gnn.grain_structure import Grain, GrainStructure

logger = logging.getLogger(__name__)

EDGE_FEATURE_MAP = {
    "center_distance": (0, 1, 2),
    "face_distance": (3, 4, 5),
    "face_normal": (6, 7, 8),
    "face_area": (9,),
    "face_circumference": (10,),
    "dipole_coupling": (11,),
    "dist": (12,),
}
NODE_FEATURE_MAP = {
    "easy_axis": (0, 1, 2),
    "K": (3,),
    "Ms": (4,),
    "A": (5,),
    "switching_field": (6,),
    "volume": (7,),
    "surface": (8,),
    "total_edge_length": (9,),
    "n_faces": (10,),
    "vol": (11,),
    "diameq": (12,),
    "sphericity": (13,),
    "convexity": (14,),
    "area": (15,),
    "dihanglemax": (16,),
    "dihanglemin": (17,),
    "dihangleav": (18,),
    "npolynb": (19,),
    "face_frac": (20,),
    "weak_grain": (21,),
    # Scale-aware alternatives to the absolute size descriptors: log volume
    # compresses six decades into a range the network can extrapolate over,
    # and L / V^(1/3) is the dimensionless shape residual (see paper, feature
    # correlations). Both are appended, so existing indices are unchanged.
    "log_volume": (22,),
    "edge_per_len": (23,),
    # Dimensionless GB-to-grain-size ratio. Physically: how large the grain is
    # relative to the GB phase thickness. Non-redundant when gb_t varies across
    # structures. Appended so existing indices are unchanged.
    "log_diameq_gb": (24,),
}
GLOBAL_FEATURE_MAP = {
    "no_grains": (0,),
    "gb_t": (1,),
    "coercivity": (2,),
    "bhmax": (3,),
    "applied_field": (4, 5, 6),
}


def _extract_edge_features(gs: GrainStructure) -> torch.Tensor:
    rows = []
    for e, ef in zip(gs.edges, gs.edge_faces, strict=True):
        g1 = gs.grains[e[0]]
        g2 = gs.grains[e[1]]
        row = []
        r = g2.center - g1.center
        row.extend((r).tolist())
        row.extend((g1.face_centers[ef] - g1.center).tolist())
        row.extend(g1.face_equations[ef, :3].tolist())
        row.append(float(np.log1p(g1.face_areas[ef])))
        row.append(float(np.log1p(g1.face_circumferences[ef])))
        ea1 = g1.params["easy_axis"]
        ea2 = g2.params["easy_axis"]
        dipole_coupling = float(ea1 @ ea2)
        dist = float(np.linalg.norm(r))
        row.extend([dipole_coupling, dist])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float).reshape(-1, _num_features(EDGE_FEATURE_MAP))


def _extract_node_features(gs: GrainStructure) -> torch.Tensor:
    gb_t = float(gs.params.get("gb_t", 4.0))  # nm; varies across structures
    rows = []
    for gid in sorted(gs.grains):
        g = gs.grains[gid]
        row = []
        row.extend(g.params["easy_axis"].tolist())
        row.append(g.params["K"])
        row.append(g.params["Ms"])
        row.append(g.params["A"])
        row.append(g.params["switching_field"])
        row.append(g.volume)
        row.append(g.surface)
        row.append(g.total_edge_length)
        row.append(len(g.faces))
        row.append(g.params["vol"])
        row.append(g.params["diameq"])
        row.append(g.params["sphericity"])
        row.append(g.params["convexity"])
        row.append(g.params["area"])
        row.append(g.params["dihanglemax"])
        row.append(g.params["dihanglemin"])
        row.append(g.params["dihangleav"])
        row.append(g.params["npolynb"])
        row.append(compute_int_face_frac(gs, gid))
        row.append(is_weak_grain(gs, gid, g))
        _vol = max(float(g.volume), 1e-30)
        row.append(math.log(_vol))
        row.append(float(g.total_edge_length) / _vol ** (1.0 / 3.0))
        diam_nm = (6.0 * _vol / math.pi) ** (1.0 / 3.0)
        row.append(math.log(max(diam_nm / max(gb_t, 1e-10), 1e-30)))
        rows.append(np.asarray(row))

    return torch.tensor(np.asarray(rows), dtype=torch.float)


def is_weak_grain(gs: GrainStructure, gid: int, g: Grain) -> bool:
    assert gs.switching_label is not None
    assert gs.switching_links is not None
    out_idx = gs.switching_links[:, 0] == gid
    in_idx = gs.switching_links[:, 1] == gid
    out_degree = np.count_nonzero(gs.switching_label[out_idx])
    in_degree = np.count_nonzero(gs.switching_label[in_idx])
    #return bool((out_degree > 0) & (in_degree == 0))
    return bool(out_degree > in_degree)


def _extract_global_features(gs: GrainStructure) -> torch.Tensor:
    """Dump every global feature into a tensor."""
    global_features = ["no_grains", "gb_t", "coercivity", "bhmax"]
    row = [gs.params[k] for k in global_features]
    row.extend([0.0, 0.0, 1.0])  # applied field direction (unit vector)
    return torch.tensor([row], dtype=torch.float)


def compute_int_face_frac(gs: GrainStructure, grain_id: int) -> float:
    grain = gs.grains[grain_id]
    if len(gs.edges) == 0:
        return 0.0
    else:
        edge_faces = grain.face_areas[gs.edge_faces[gs.edges[:, 0] == grain_id]]
        return float(sum(edge_faces) / grain.surface)


def gs_to_data(
    gs: GrainStructure,
    node_feature_map: dict[str, tuple[int, ...]],
    edge_feature_map: dict[str, tuple[int, ...]],
    global_feature_map: dict[str, tuple[int, ...]],
    label: str | None,
    task: str,
) -> Data:
    """Convert a single ``GrainStructure`` into a PyG ``Data`` object.

    Parameters
    ----------
    gs : GrainStructure
        The grain structure to convert.
    node_feature_map, edge_feature_map, global_feature_map
        Maps feature name → column indices in the raw extracted tensors.
    label : str
        Name of the feature to use as the label.
    task : {"global", "node"}
        Where the label lives.

    Returns
    -------
    Data
        PyG data object with ``x``, ``edge_attr``, ``u``, ``y`` and
        attached feature maps for downstream transforms.
    """
    edge_features_raw = _extract_edge_features(gs)
    node_features_raw = _extract_node_features(gs)
    global_features_raw = _extract_global_features(gs)

    assert gs.switching_links is not None
    edge_index = torch.tensor(gs.edges, dtype=torch.long).t().contiguous()

    edge_label_index = torch.tensor(gs.switching_links, dtype=torch.long).t().contiguous()
    
    tasks = ("global", "node", "link")
    if task not in tasks:
        raise ValueError(f"Unknown task: {task!r}. Allowed tasks: {tasks}")

    if task == "global" and label is not None:
        label_idx = global_feature_map[label][0]
        y = global_features_raw[:, label_idx]
        global_feature_map = {k: v for k, v in global_feature_map.items() if k != label}
        assert y.dim() == 1
        y = y.unsqueeze(1)
    elif task == "node" and label is not None:
        label_idx = node_feature_map[label][0]
        y = node_features_raw[:, label_idx]
        assert y.dim() == 1
        y = y.unsqueeze(1)
        node_feature_map = {k: v for k, v in node_feature_map.items() if k != label}
    elif task == "link":
        y = torch.tensor(gs.switching_label, dtype=torch.float)
        assert y.dim() == 1
        y = y.unsqueeze(1)
    else:
        y = None

    node_features = _filter_features(node_features_raw, node_feature_map)
    edge_features = _filter_features(edge_features_raw, edge_feature_map)
    global_features = _filter_features(global_features_raw, global_feature_map)

    # Remap indices so they match the filtered tensor columns.
    node_feature_map = _remap_feature_map(node_feature_map)
    edge_feature_map = _remap_feature_map(edge_feature_map)
    global_feature_map = _remap_feature_map(global_feature_map)
    
    data = Data(
        x=node_features,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_features,
        edge_label_index=edge_label_index,
        u=global_features,
        node_feature_map=node_feature_map,
        edge_feature_map=edge_feature_map,
        global_feature_map=global_feature_map,
    )
    return data


def _filter_features(
    features: torch.Tensor, feature_map: dict[str, tuple[int, ...]]
) -> torch.Tensor:
    """Select columns from *features* according to *feature_map*."""
    indices = list(chain.from_iterable(feature_map.values()))
    return torch.index_select(features, dim=1, index=torch.tensor(indices, dtype=torch.int32))


def _remap_feature_map(feature_map: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """Renumber indices so the first feature starts at 0."""
    offset = 0
    new_map: dict[str, tuple[int, ...]] = {}
    for name, idxs in feature_map.items():
        n = len(idxs)
        new_map[name] = tuple(range(offset, offset + n))
        offset += n
    return new_map


def random_rotation_transform(data: Data) -> Data:
    """Rotate 3-D vector features by a fresh random SO(3) matrix.

    The rotation is applied to every vector feature that is present in
    the data object's feature maps:
    ``center_distance``, ``face_distance``, ``face_normal`` (edge
    attributes), ``easy_axis`` (node attributes) and ``applied_field``
    (global attributes).  Scalar features are left untouched.

    Because ``MicromagneticDataset.get`` returns a clone, each access
    to the same index receives a *different* random rotation — exactly
    what is needed for data augmentation.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator.
    """
    assert data.x is not None
    device = data.x.device
    dtype = data.x.dtype
    R = random_rotation_matrix(device=device, dtype=dtype)
    return _rotate_vector_features(data, R)


def per_structure_label_transform(inner: Callable[[Data], Data]) -> Callable[[Data], Data]:
    """Wrap a transform to z-score node labels independently per structure."""
    def transform(data: Data) -> Data:
        data = inner(data)
        y = data.y.reshape(-1).float()
        if y.numel() > 1:
            std = y.std()
            if torch.isfinite(std) and std > 1e-8:
                data.y = ((y - y.mean()) / std).reshape(data.y.shape)
            else:
                data.y = torch.zeros_like(data.y)
        else:
            data.y = torch.zeros_like(data.y)
        return data
    return transform


def _rotate_vector_features(data: Data, R: torch.Tensor) -> Data:
    """Apply the rotation matrix ``R`` to all 3-D vector features in-place."""
    # Node vectors
    if "easy_axis" in data.node_feature_map:
        idx = data.node_feature_map["easy_axis"]
        data.x[:, idx] = data.x[:, idx] @ R.T

    # Edge vectors
    for name in ("center_distance", "face_distance", "face_normal"):
        if name in data.edge_feature_map:
            idx = data.edge_feature_map[name]
            assert data.edge_attr is not None
            data.edge_attr[:, idx] = data.edge_attr[:, idx] @ R.T

    # Global vectors
    if "applied_field" in data.global_feature_map:
        idx = data.global_feature_map["applied_field"]
        data.u[:, idx] = data.u[:, idx] @ R.T

    return data


class RandomRotationTransform:
    """Seeded, callable version of :func:`random_rotation_transform`.

    Draws the rotation from a NumPy ``Generator`` supplied at construction so
    that the augmentation sequence is reproducible, e.g.::

        rotate = RandomRotationTransform(np.random.default_rng(SEED))
        train_ds.transform = lambda d: rotate(scaler(d))

    Each call applies a fresh random SO(3) rotation to the same vector
    features as :func:`random_rotation_transform`.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, data: Data) -> Data:
        assert data.x is not None
        device = data.x.device
        dtype = data.x.dtype
        u123 = torch.as_tensor(self.rng.random(3), device=device, dtype=dtype)
        R = random_rotation_matrix(device=device, dtype=dtype, u123=u123)
        return _rotate_vector_features(data, R)


def random_rotation_matrix(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    u123: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a random 3x3 rotation matrix uniformly distributed over SO(3).
    
    Parameters
    ----------
    device : torch.device | None, optional
        _description_, by default None
    dtype : torch.dtype, optional
        _description_, by default torch.float32

    Returns
    -------
    torch.Tensor
        _description_
    """
    if u123 is None:
        u123 = torch.rand(3, device=device, dtype=dtype)
    u1, u2, u3 = u123

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)

    # quaternion (x, y, z, w)
    x, y, z, w = q1, q2, q3, q4

    R = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z),
                         2 * (x * y - z * w),
                         2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w),
                         1 - 2 * (x * x + z * z),
                         2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w),
                         2 * (y * z + x * w),
                         1 - 2 * (x * x + y * y))),
        )
    )

    return R


def train_val_test_split(
    dataset: MicromagneticDataset,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[MicromagneticDataset, MicromagneticDataset, MicromagneticDataset]:
    """Split a ``MicromagneticDataset`` into train / val / test subsets.

    The returned datasets are new ``MicromagneticDataset`` instances that
    preserve the original feature maps, label and task metadata.

    Parameters
    ----------
    dataset : MicromagneticDataset
        The full dataset.
    train_frac, val_frac, test_frac : float
        Fractions for each split.  Must sum to ``1.0``.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    train_dataset, val_dataset, test_dataset
    """
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty")
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0")

    train_size = int(train_frac * total)
    val_size = int(val_frac * total)

    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    perm = torch.randperm(total, generator=generator)

    train_idx = perm[:train_size].tolist()
    val_idx = perm[train_size : train_size + val_size].tolist()
    test_idx = perm[train_size + val_size :].tolist()

    _train_data = [dataset[i] for i in train_idx]
    _val_data = [dataset[i] for i in val_idx]
    _test_data = [dataset[i] for i in test_idx]

    return (
        dataset._clone_with_data(_train_data),
        dataset._clone_with_data(_val_data),
        dataset._clone_with_data(_test_data),
    )


def scale_indices(
    feature_map: dict[str, tuple[int, ...]],
    *names: str,
) -> list[int]:
    """Return column indices of the named features.

    This is an *inclusion* helper: you list the features you want to
    scale, and the function returns their concatenated column indices.

    Examples
    --------
    >>> scale_indices({"easy_axis": (0, 1, 2), "K": (3,)}, "K")
    [3]
    """
    return sorted(i for name, idxs in feature_map.items() for i in idxs if name in names)


def non_vector_scale_indices(feature_map: dict[str, tuple[int, ...]]) -> list[int]:
    """Return column indices of all scalar features (non-vectors) for scaling.

    Automatically excludes 3-D vector features that must not be scaled:
    ``easy_axis``, ``center_distance``, ``face_distance``,
    ``face_normal``, ``applied_field``.

    Examples
    --------
    >>> non_vector_scale_indices({"easy_axis": (0, 1, 2), "K": (3,), "Ms": (4,)})
    [3, 4]
    """
    vectors = {"easy_axis", "center_distance", "face_distance", "face_normal", "applied_field"}
    return sorted(i for name, idxs in feature_map.items() for i in idxs if name not in vectors)


def vector_groups(
    feature_map: dict[str, tuple[int, ...]],
    *names: str,
) -> list[list[int]]:
    """Return column-index groups for named vector features.

    Examples
    --------
    >>> vector_groups({"center_distance": (0, 1, 2), "face_area": (3,)}, "center_distance")
    [[0, 1, 2]]
    """
    return [list(feature_map[name]) for name in names if name in feature_map]


class MicromagneticDataset(Dataset):
    """A list of pre-built PyG ``Data`` objects.

    This dataset is fully materialised at construction time — each
    ``__getitem__`` simply returns a (possibly transformed) copy of the
    stored ``Data``.  No geometry recomputation happens during training.
    """

    @classmethod
    def from_structures(
        cls,
        structures: list[GrainStructure],
        *,
        node_features: Iterable[str] = (
            "easy_axis",
            "K",
            "Ms",
            "A",
            "switching_field",
            "volume",
            "surface",
            "total_edge_length",
            "n_faces",
            "vol",
            "diameq",
            "sphericity",
            "convexity",
            "area",
            "dihanglemax",
            "dihanglemin",
            "dihangleav",
            "npolynb",
            "face_frac",
        ),
        edge_features: Iterable[str] = (
            "center_distance",
            "face_distance",
            "face_normal",
            "face_area",
            "face_circumference",
            "dipole_coupling",
            "dist",
        ),
        global_features: Iterable[str] = (
            "no_grains",
            "gb_t",
            "coercivity",
            "bhmax",
            "applied_field",
        ),
        transform: Callable | None = None,
        label_name: str | None = None,
        task: str = "global",
    ) -> MicromagneticDataset:
        """Convert a list of ``GrainStructure`` objects into a dataset.

        Parameters
        ----------
        structures : list[GrainStructure]
            Pre-loaded grain structures (e.g. from ``load_grain_structures``).
        node_features, edge_features, global_features : Iterable[str]
            Feature names to **include**.  Defaults include every available
            feature of that type.  The label is always kept automatically.
        transform : Callable or None
            PyG ``transform`` applied to each ``Data`` object on access.
        label_name : str
            Name of the column to use as the label.
        task : {"global", "node", "link"}
            ``global`` – label is graph-level.
            ``node``   – label is per-grain.
            ``link``   – label is for each `edge_label_index` either 0 or 1.

        Returns
        -------
        MicromagneticDataset
        """
        node_features = list(node_features) + ([] if label_name is None else [label_name])
        global_features = list(global_features) + ([] if label_name is None else [label_name])
        
        node_feature_map = {k: v for k, v in NODE_FEATURE_MAP.items() if k in node_features}
        edge_feature_map = {k: v for k, v in EDGE_FEATURE_MAP.items() if k in edge_features}
        global_feature_map = {k: v for k, v in GLOBAL_FEATURE_MAP.items() if k in global_features}

        n_structures = len(structures)
        logger.info("Converting %d GrainStructures to Data objects ...", n_structures)

        data_list = [
            gs_to_data(
                gs,
                node_feature_map,
                edge_feature_map,
                global_feature_map,
                label=label_name,
                task=task,
            )
            for gs in structures
        ]

        logger.info("Feature extraction done.")

        node_feature_map = {k: v for k, v in node_feature_map.items() if k != label_name}
        global_feature_map = {k: v for k, v in global_feature_map.items() if k != label_name}

        return cls(
            data_list,
            transform=transform,
            node_feature_map=_remap_feature_map(node_feature_map),
            edge_feature_map=_remap_feature_map(edge_feature_map),
            global_feature_map=_remap_feature_map(global_feature_map),
            label=label_name,
            task=task,
        )

    def __init__(
        self,
        data_list: list[Data],
        *,
        node_feature_map: dict[str, tuple[int, ...]],
        edge_feature_map: dict[str, tuple[int, ...]],
        global_feature_map: dict[str, tuple[int, ...]],
        label: str | None,
        task: str = "global",
        transform: Callable | None = None,
    ):
        self.node_feature_map = node_feature_map
        self.edge_feature_map = edge_feature_map
        self.global_feature_map = global_feature_map
        self.data_list = data_list
        self.label = label
        self.task = task
        super().__init__(transform=transform)

    def len(self) -> int:
        return len(self.data_list)

    @property
    def num_node_features(self) -> int:
        return _num_features(self.node_feature_map)

    @property
    def num_edge_features(self) -> int:
        return _num_features(self.edge_feature_map)

    @property
    def num_global_features(self) -> int:
        return _num_features(self.global_feature_map)
    
    @property
    def node_features(self) -> list[str]:
        return list(self.node_feature_map.keys())

    @property
    def edge_features(self) -> list[str]:
        return list(self.edge_feature_map.keys())
    
    @property
    def global_features(self) -> list[str]:
        return list(self.global_feature_map.keys())

    def get(self, idx: int) -> Data:
        return self.data_list[idx].clone()

    def _clone_with_data(self, data_list: list[Data]) -> MicromagneticDataset:
        """Return a new dataset with the same metadata but different data."""
        return MicromagneticDataset(
            data_list,
            transform=self.transform,
            node_feature_map=self.node_feature_map,
            edge_feature_map=self.edge_feature_map,
            global_feature_map=self.global_feature_map,
            label=self.label,
            task=self.task,
        )



def _num_features(feature_map: dict[str, tuple[int, ...]]) -> int:
    return len(list(chain.from_iterable(feature_map.values())))
