"""Tests for dataset splitting utilities."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from micromagnetic_gnn.split import split_dataset, stratified_split_by_size


@pytest.fixture
def dummy_dataset():
    """Simple dataset with 100 items."""
    return TensorDataset(torch.arange(100).unsqueeze(1).float())


class TestSplitDataset:
    def test_basic_split(self, dummy_dataset):
        train, val, test = split_dataset(dummy_dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_fraction_sum_error(self, dummy_dataset):
        with pytest.raises(ValueError, match="Fractions must sum"):
            split_dataset(dummy_dataset, train_frac=0.5, val_frac=0.3, test_frac=0.3)

    def test_empty_dataset(self):
        empty = TensorDataset(torch.tensor([]))
        with pytest.raises(ValueError, match="empty"):
            split_dataset(empty)

    def test_reproducibility(self, dummy_dataset):
        t1, v1, te1 = split_dataset(dummy_dataset, seed=42)
        t2, v2, te2 = split_dataset(dummy_dataset, seed=42)
        assert list(t1.indices) == list(t2.indices)
        assert list(v1.indices) == list(v2.indices)
        assert list(te1.indices) == list(te2.indices)

    def test_disjoint(self, dummy_dataset):
        train, val, test = split_dataset(dummy_dataset, seed=123)
        train_set = set(train.indices)
        val_set = set(val.indices)
        test_set = set(test.indices)
        assert not train_set & val_set
        assert not train_set & test_set
        assert not val_set & test_set


class TestStratifiedSplitBySize:
    def test_stratified_split(self):
        # Create dataset where item i has x.shape[0] = i+1
        class SizeDataset:
            def __init__(self, n):
                self.n = n
            def __len__(self):
                return self.n
            def __getitem__(self, i):
                class Item:
                    x = torch.zeros(i + 1, 3)
                return Item()

        ds = SizeDataset(100)
        tr, va, te = stratified_split_by_size(ds)
        assert len(tr) == 70
        assert len(va) == 15
        assert len(te) == 15

        # Check disjointness
        assert set(tr).isdisjoint(va)
        assert set(tr).isdisjoint(te)
        assert set(va).isdisjoint(te)

        # Check union covers all
        assert set(tr) | set(va) | set(te) == set(range(100))

    def test_fraction_error(self):
        class Dummy:
            def __len__(self): return 10
            def __getitem__(self, i):
                class Item:
                    x = torch.zeros(1, 3)
                return Item()
        with pytest.raises(ValueError, match="Fractions must sum"):
            stratified_split_by_size(Dummy(), train_frac=0.5, val_frac=0.3, test_frac=0.3)

    def test_empty_dataset(self):
        class Empty:
            def __len__(self): return 0
            def __getitem__(self, i):
                class Item:
                    x = torch.zeros(1, 3)
                return Item()
        with pytest.raises(ValueError, match="empty"):
            stratified_split_by_size(Empty())
