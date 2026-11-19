from __future__ import annotations

from pathlib import Path

from micromagnetic_gnn.grain_structure import GrainStructure, load_datasets
from micromagnetic_gnn.data_pipeline import MicromagneticDataset, random_rotation_transform
from micromagnetic_gnn.models import GraphDataScaler


# Generalization-set selection and shuffle seed. Imported by the training script
# and by every evaluation script, so the split can never drift between them.
SPLIT_SEED = 42


def extrapolation_cond(gs: GrainStructure) -> bool:
    """Structures withheld entirely from training (the generalization set)."""
    return bool(gs.params["no_grains"] > 1500 or gs.params["size_x"] > 30000)


def default_datasets(
        label: str,
        cache_path: Path,
        data_root: str = "ROM_Simulations",
        rebuild: bool = False,
        max_workers: int = 12,
        node_features: tuple[str, ...] = (
            "easy_axis",
            # Scale-aware size descriptors. The absolute volume, surface and
            # total edge length leave their training range on the generalization
            # set and make the network predict unphysical coercivities there;
            # log(volume) and L/V^(1/3) do not. See the feature-correlation
            # subsection of the paper.
            "log_volume",
            "edge_per_len",
            "diameq",
            "sphericity",
            "dihanglemax",
            "dihanglemin",
            "dihangleav",
            #"face_frac",
        ),
        edge_features: tuple[str, ...] = (
            "center_distance",
            "face_distance",
            "face_normal",
            "face_area",
            "face_circumference",
            #"dipole_coupling",
            #"dist",
        ),
        global_features: tuple[str, ...] = (
            "gb_t",
        ),
        scaler: GraphDataScaler | None = None,
        rotate: bool = False
    ) -> tuple[MicromagneticDataset, MicromagneticDataset, MicromagneticDataset, MicromagneticDataset, GraphDataScaler]:
    """Load default train, validation, test and generalization datasets.

    Parameters
    ----------
    label : str
    cache_path : Path
    data_root : str, optional
        by default "ROM_Simulations"
    rebuild : bool, optional
        rebuild cache, by default False
    max_workers : int, optional
    node_features : tuple, optional
        by default ( "easy_axis", "volume", "surface", "total_edge_length", "diameq", "sphericity", "dihanglemax", "dihanglemin", "dihangleav", "face_frac", )
    edge_features : tuple, optional
        by default ( "center_distance", "face_distance", "face_normal", "face_area", "face_circumference", "dipole_coupling", "dist", )
    global_features : tuple, optional
        by default ( "gb_t" )

    Returns
    -------
    tuple[MicromagneticDataset, MicromagneticDataset, MicromagneticDataset, MicromagneticDataset, GraphDataScaler]
    """
    train_ds, val_ds, test_ds, ext_ds = default_gs_data(cache_path=cache_path, data_root=data_root, rebuild=rebuild, max_workers=max_workers)
    
    if label in train_ds[0].params:
        task = "global"
    elif label in train_ds[0].grains[0].params:
        task = "node"
    else:
        raise ValueError(f"label {label} not found")

    train_ds = MicromagneticDataset.from_structures(
        train_ds, label_name=label, task=task,
        node_features=node_features, edge_features=edge_features, global_features=global_features
    )
    val_ds = MicromagneticDataset.from_structures(
        val_ds, label_name=label, task=task,
        node_features=node_features, edge_features=edge_features, global_features=global_features
    )
    test_ds = MicromagneticDataset.from_structures(
        test_ds, label_name=label, task=task,
        node_features=node_features, edge_features=edge_features, global_features=global_features
    )
    ext_ds = MicromagneticDataset.from_structures(
        ext_ds, label_name=label, task=task,
        node_features=node_features, edge_features=edge_features, global_features=global_features
    )
    dataset_size = len(train_ds) + len(val_ds) + len(test_ds) + len(ext_ds)
    print(f"  Graphs: {dataset_size}")
    print(f"  Node features : {train_ds.num_node_features}")
    print(f"  Edge features : {train_ds.num_edge_features}")
    print(f"  Global features: {train_ds.num_global_features}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | Ext: {len(ext_ds)}")
    
    if scaler is None:
        scaler = GraphDataScaler.fit_from_dataset(train_ds).to("cpu")

    def transform(d):
        d = scaler(d)
        d = random_rotation_transform(d) if rotate else d
        return d

    train_ds.transform = transform
    val_ds.transform = transform
    test_ds.transform = transform
    ext_ds.transform = transform

    return train_ds, val_ds, test_ds, ext_ds, scaler


def default_gs_data(
    cache_path: Path,
    data_root: str = "ROM_Simulations",
    rebuild: bool = False,
    max_workers: int = 12,
) -> tuple[list[GrainStructure], list[GrainStructure], list[GrainStructure], list[GrainStructure]]:
    train_ds, val_ds, test_ds, ext_ds = load_datasets(
        data_root,
        train_frac=0.75,
        val_frac=0.1,
        seed=SPLIT_SEED,
        extrapolation_cond=extrapolation_cond,
        skip_errors=True,
        max_workers=max_workers,
        cache_path=cache_path,
        rebuild_cache=rebuild,
    )
    return train_ds, val_ds, test_ds, ext_ds