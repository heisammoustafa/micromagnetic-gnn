"""Consistency test for total edge length."""

from __future__ import annotations

import numpy as np
import pytest

from micromagnetic_gnn.grain_structure import GrainStructure
from micromagnetic_gnn.tess_parser import parse_tess_file


def _tess_total_edge_length(tess_path: str) -> float:
    """Compute total edge length directly from the raw ``.tess`` file.

    Parses the vertex and edge sections and sums Euclidean distances
    for every edge listed in the tessellation.
    """
    vertices, edges, _, _ = parse_tess_file(tess_path)
    total = 0.0
    for v1_idx, v2_idx in edges.values():
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        total += float(np.linalg.norm(v2 - v1))
    return total


def test_total_edge_length_consistency(tess_file, grain_structure: GrainStructure) -> None:
    """Grain-level ``total_edge_length`` should be close to the raw tess sum.

    Internal edges are shared by two grains, so the grain-level sum counts
    them twice.  We therefore expect ``grain_total`` to be somewhat larger than
    ``tess_total`` (roughly 1.5–2.5× for a compact polycrystal).  The important
    thing is that the two numbers are in the same ballpark — i.e. we didn't
    lose edges or compute nonsense lengths.
    """
    tess_total = _tess_total_edge_length(str(tess_file))
    grain_total = sum(g.total_edge_length for g in grain_structure.grains.values())

    assert tess_total > 0
    assert grain_total > 0

    # Internal edges are counted twice; for a compact 3-D polycrystal
    # the grain total is typically between 1.5× and 3× the raw tess total.
    ratio = grain_total / tess_total
    assert 1.0 < ratio < 3.0, (
        f"grain_total ({grain_total:.4f}) / tess_total ({tess_total:.4f}) = "
        f"{ratio:.3f}, expected between 1.0 and 3.0"
    )
