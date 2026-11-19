"""Tests for the tessellation parser."""

from pathlib import Path

import numpy as np
import pytest

from micromagnetic_gnn.tess_parser import parse_tess_file


def test_parse_tess_file_returns_correct_types(tess_file: Path) -> None:
    vertices, edges, faces, polyhedra = parse_tess_file(str(tess_file))
    assert isinstance(vertices, np.ndarray)
    assert isinstance(edges, dict)
    assert isinstance(faces, dict)
    assert isinstance(polyhedra, dict)


def test_parse_tess_file_zero_based_indices(tess_file: Path) -> None:
    vertices, edges, faces, polyhedra = parse_tess_file(str(tess_file))
    # All IDs should be zero-based (no negative or very large indices)
    for eid in edges:
        assert eid >= 0
    for fid in faces:
        assert fid >= 0
    for pid in polyhedra:
        assert pid >= 0


def test_parse_tess_file_vertices_shape(tess_file: Path) -> None:
    vertices, _, _, polyhedra = parse_tess_file(str(tess_file))
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert len(polyhedra) > 0


def test_parse_tess_file_grain_count_matches_parameters(tess_file: Path, test_structure_dir: Path) -> None:
    _, _, _, polyhedra = parse_tess_file(str(tess_file))
    parameters = test_structure_dir / "parameters.txt"
    with open(parameters) as f:
        no_grains = int(f.readline().split("=")[1].strip())
    assert len(polyhedra) == no_grains
