"""Load the trained models that ship in ``results/``.

Every checkpoint written by a training script is self-describing: it stores the
encoder configuration, the model configuration, the fitted scaler state, and the
exact feature lists the model was trained on.  Loading therefore needs neither a
dataset nor a second copy of the training hyper-parameters, and the feature
lists come from the checkpoint rather than from a hand-maintained constant --
changing the default features can no longer silently break a scaler with a shape
mismatch.

There is one loader per trained model::

    from micromagnetic_gnn.checkpoints import load_coercivity_model

    trained = load_coercivity_model()              # results/coercivity.tar
    train, val, test, ext = trained.datasets()     # its own features + scaler
    loader = trained.loader(test)

All checkpoints live in ``results/`` (see :func:`results_dir`); nothing is
loaded out of a training directory any more.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from micromagnetic_gnn.data_pipeline import MicromagneticDataset
from micromagnetic_gnn.dataset import default_datasets
from micromagnetic_gnn.models import (
    EdgeAggEncoder,
    GraphDataScaler,
    GraphModel,
    LinkModel,
    NodeModel,
)

# <repo>/src/micromagnetic_gnn/checkpoints.py -> <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default feature cache used by :meth:`TrainedModel.datasets`.
DEFAULT_CACHE = Path("data_cache.pkl")


def results_dir() -> Path:
    """Directory holding the trained checkpoints.

    ``$RESULTS_DIR`` wins if set, then ``results/`` below the current working
    directory (scripts are meant to be run from the repo root), then ``results/``
    next to the source tree so an editable install also works from elsewhere.
    """
    env = os.environ.get("RESULTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    local = Path("results")
    if local.is_dir():
        return local.resolve()
    return (_REPO_ROOT / "results").resolve()


def build_encoder(cfg: dict[str, Any]) -> EdgeAggEncoder:
    """Instantiate the encoder a checkpoint was trained with.

    Every model currently in ``results/`` is edge-centric.  The archived GAT runs
    are only used by ``analysis/scalefree_comparison.py``, which builds its
    encoder itself.
    """
    return EdgeAggEncoder(**cfg)


@dataclass(frozen=True)
class TrainedModel:
    """A checkpoint loaded into a ready-to-evaluate model.

    Attributes
    ----------
    model : torch.nn.Module
        On ``device``, in whatever mode ``torch`` defaults to; call ``eval()``
        yourself, or ``train()`` to keep dropout active for MC sampling.
    scaler : GraphDataScaler
        Fitted scaler as saved during training, kept on the CPU.
    label : str
        Name of the target this model predicts (``"coercivity"``, ...).
    path : Path
        The checkpoint the model was read from.
    results : dict
        The raw checkpoint contents (config, metrics, history, ...).
    """

    model: torch.nn.Module
    scaler: GraphDataScaler
    label: str
    path: Path
    results: dict[str, Any]

    @property
    def features(self) -> dict[str, tuple[str, ...]]:
        """Node / edge / global feature names this model was trained on."""
        f = self.results["features"]
        return {k: tuple(v) for k, v in f.items()}

    @property
    def batch_size(self) -> int:
        """Return the evaluation batch size used during training."""
        return int(self.results["config"]["BATCH_SIZE"])

    @property
    def predicts_variance(self) -> bool:
        """True when the model has a Gaussian-NLL variance head."""
        return bool(self.results.get("model_config", {}).get("predict_variance", False))

    @property
    def metrics(self) -> dict[str, Any]:
        """Return metrics stored in the checkpoint."""
        return self.results.get("metrics", {})

    def datasets(
        self,
        cache_path: Path = DEFAULT_CACHE,
        **kwargs: Any,
    ) -> tuple[MicromagneticDataset, MicromagneticDataset,
               MicromagneticDataset, MicromagneticDataset]:
        """Train / val / test / generalization sets for *this* model.

        Wraps :func:`micromagnetic_gnn.dataset.default_datasets` with the
        checkpoint's own feature lists and its fitted scaler, so the columns line
        up with the trained weights by construction.  Extra keyword arguments are
        passed straight through (``data_root``, ``rebuild``, ``max_workers``).
        """
        features = self.features
        train_ds, val_ds, test_ds, ext_ds, _ = default_datasets(
            self.label,
            cache_path,
            node_features=features["node_features"],
            edge_features=features["edge_features"],
            global_features=features["global_features"],
            scaler=self.scaler,
            **kwargs,
        )
        return train_ds, val_ds, test_ds, ext_ds

    def loader(self, dataset: MicromagneticDataset, **kwargs: Any) -> DataLoader:
        """Create an evaluation loader at the batch size used during training."""
        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("shuffle", False)
        return DataLoader(dataset, **kwargs)


def load_checkpoint(name: str | Path, device: Any = None) -> dict[str, Any]:
    """Read one checkpoint. Relative names resolve against :func:`results_dir`."""
    path = Path(name)
    if not path.is_absolute():
        path = results_dir() / path
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Available in {results_dir()}: "
            + ", ".join(sorted(p.name for p in results_dir().glob("*.tar")))
        )
    return torch.load(path, map_location=device or "cpu", weights_only=False)


def _load(
    name: str | Path,
    head_cls: type[torch.nn.Module],
    label: str,
    device: Any = None,
) -> TrainedModel:
    device = device or "cpu"
    path = Path(name)
    if not path.is_absolute():
        path = results_dir() / path
    results = load_checkpoint(path, device)

    model = head_cls(build_encoder(results["encoder_config"]),
                     **results.get("model_config", {}))
    # torch.compile wraps the module and prefixes every parameter name.
    state_dict = {k.replace("_orig_mod.", ""): v
                  for k, v in results["state_dict"].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)

    scaler = GraphDataScaler(**results["scaler_config"])
    scaler.load_state_dict(results["scaler_state_dict"])

    return TrainedModel(model=model, scaler=scaler.to("cpu"), label=label,
                        path=path, results=results)


def load_coercivity_model(member: int | None = None, device: Any = None) -> TrainedModel:
    """Graph-level coercivity model (``results/coercivity.tar``).

    Parameters
    ----------
    member : int, optional
        Deep-ensemble member index; loads ``results/coercivity_m<K>.tar``
        instead of the single model.
    """
    name = "coercivity.tar" if member is None else f"coercivity_m{member}.tar"
    return _load(name, GraphModel, "coercivity", device)


def load_coercivity_huber_model(device: Any = None) -> TrainedModel:
    """Single-head Huber-loss coercivity model (``results/coercivity_huber.tar``)."""
    return _load("coercivity_huber.tar", GraphModel, "coercivity", device)


def load_coercivity_ensemble(
    members: list[int] | None = None,
    device: Any = None,
) -> list[TrainedModel]:
    """Coercivity ensemble members, in member order.

    Parameters
    ----------
    members:
        Explicit list of member indices to load (e.g. ``[15, 16, 17, 18, 19]``).
        When *None*, every ``results/coercivity_m*.tar`` is loaded.
    """
    if members is not None:
        paths = [results_dir() / f"coercivity_m{k}.tar" for k in sorted(members)]
    else:
        paths = sorted(results_dir().glob("coercivity_m*.tar"))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"Found {len(paths)} ensemble member(s) in {results_dir()} (need >= 2). "
            "Train them with:  sbatch slurm/train_edge_uq.slurm"
        )
    return [_load(p, GraphModel, "coercivity", device) for p in paths]


def load_bhmax_model(device: Any = None) -> TrainedModel:
    """Graph-level (BH)max model (``results/bhmax.tar``)."""
    return _load("bhmax.tar", GraphModel, "bhmax", device)


def load_switching_model(per_structure: bool = False, device: Any = None) -> TrainedModel:
    """Node-level switching-field model (``results/switching_field.tar``).

    Parameters
    ----------
    per_structure : bool
        Load the variant trained on the within-structure z-score of the
        switching field (``results/switching_field_norm.tar``).
    """
    name = "switching_field_norm.tar" if per_structure else "switching_field.tar"
    return _load(name, NodeModel, "switching_field", device)


def load_switching_link_model(device: Any = None) -> TrainedModel:
    """Link-level switching model (``results/switching_link.tar``)."""
    return _load("switching_link.tar", LinkModel, "switching_field", device)
