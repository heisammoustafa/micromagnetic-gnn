"""Dataset splitting utilities."""

from __future__ import annotations

import torch
from torch.utils.data import Subset, random_split


def split_dataset(
    dataset,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[Subset, Subset, Subset]:
    """Split a dataset into train / validation / test subsets.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or list
        The full dataset.
    train_frac, val_frac, test_frac : float
        Fractions for each split.  Must sum to ``1.0`` (within tolerance).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    train_subset, val_subset, test_subset

    Raises
    ------
    ValueError
        If the fractions do not sum to ``1.0``.
    """
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty")

    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got "
            f"{train_frac + val_frac + test_frac:.6f}"
        )

    train_size = int(train_frac * total)
    val_size = int(val_frac * total)
    test_size = total - train_size - val_size

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()

    return random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )


def stratified_split_by_size(
    dataset,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified split that sorts graphs by size before cutting.

    Useful when graph sizes (number of grains) vary widely and you want
    each split to have a similar size distribution.

    Returns the index lists directly instead of ``Subset`` objects so
    that you can build :class:`torch.utils.data.Subset` yourself.

    Parameters
    ----------
    dataset : Dataset or list
        Must support ``len(dataset)`` and ``dataset[i].x.shape[0]``
        (number of nodes / grains).
    train_frac, val_frac, test_frac : float
        Fractions summing to ``1.0``.
    seed : int or None
        Random seed.

    Returns
    -------
    train_indices, val_indices, test_indices
    """
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty")

    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0")

    # Sort by graph size (number of grains / nodes)
    sorted_idx = sorted(range(total), key=lambda i: dataset[i].x.shape[0])

    # Shuffle blocks within each split to avoid ordered chunks
    if seed is not None:
        rng = torch.Generator().manual_seed(seed)
    else:
        rng = torch.Generator()

    train_size = int(train_frac * total)
    val_size = int(val_frac * total)

    train_idx = sorted_idx[:train_size]
    val_idx = sorted_idx[train_size : train_size + val_size]
    test_idx = sorted_idx[train_size + val_size :]

    # Shuffle each block in-place for randomness
    perm = torch.randperm(len(train_idx), generator=rng).tolist()
    train_idx = [train_idx[i] for i in perm]

    perm = torch.randperm(len(val_idx), generator=rng).tolist()
    val_idx = [val_idx[i] for i in perm]

    perm = torch.randperm(len(test_idx), generator=rng).tolist()
    test_idx = [test_idx[i] for i in perm]

    return train_idx, val_idx, test_idx
