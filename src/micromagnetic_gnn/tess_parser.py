"""Tessellation file parser for NEPER format.

This module reads NEPER ``.tess`` files and extracts the raw geometric
primitives (vertices, edges, faces, polyhedra) as plain Python/NumPy
objects. All IDs are converted from the **1-based** convention used in
NEPER files to **0-based** indices suitable for array indexing.

Examples
--------
Parse a tessellation file:

>>> vertices, edges, faces, polyhedra = parse_tess_file("grain.tess")
>>> vertices.shape
(1024, 3)
>>> len(polyhedra)
148

See Also
--------
:func:`micromagnetic_gnn.grain_structure.GrainStructure.from_tess_file`
    Higher-level wrapper that builds grains and neighbour graphs.
"""

from __future__ import annotations

import numpy as np


def parse_tess_file(path: str) -> tuple[np.ndarray, dict, dict, dict]:
    """Parse a NEPER tessellation file.

    Parameters
    ----------
    path : str
        Path to the ``.tess`` file.

    Returns
    -------
    vertices : np.ndarray
        Array of shape ``(n_vertices, 3)`` with vertex coordinates.
    edges : dict[int, tuple[int, int]]
        Mapping ``edge_id -> (vid1, vid2)`` (0-based vertex indices).
    faces : dict[int, list[int]]
        Mapping ``face_id -> [vid1, vid2, ...]`` (0-based vertex indices).
    polyhedra : dict[int, list[int]]
        Mapping ``pid -> [fid1, fid2, ...]`` (0-based face and grain indices).

    Notes
    -----
    Grain/face/vertex IDs in the file are **1-based**; this function converts
    them to **0-based** indices for internal use.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    vertices: dict[int, np.ndarray] = {}
    edges: dict[int, tuple[int, int]] = {}
    faces: dict[int, list[int]] = {}
    polyhedra: dict[int, list[int]] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("**vertex"):
            n = int(lines[i + 1])
            i += 2
            for _ in range(n):
                parts = lines[i].split()
                vid = int(parts[0]) - 1
                vertices[vid] = np.array(list(map(float, parts[1:4])))
                i += 1
            continue

        if line.startswith("**edge"):
            n = int(lines[i + 1])
            i += 2
            for _ in range(n):
                parts = lines[i].split()
                eid = int(parts[0]) - 1
                edges[eid] = (int(parts[1]) - 1, int(parts[2]) - 1)
                i += 1
            continue

        if line.startswith("**face"):
            n = int(lines[i + 1])
            i += 2
            for _ in range(n):
                header = lines[i].split()
                fid = int(header[0]) - 1
                vids = [int(vid) - 1 for vid in header[2:]]
                faces[fid] = vids
                i += 4  # Matches NEPER format with extra lines per face
            continue

        if line.startswith("**polyhedron"):
            n = int(lines[i + 1])
            i += 2
            for _ in range(n):
                parts = lines[i].split()
                pid = int(parts[0]) - 1
                fids = [abs(int(fid)) - 1 for fid in parts[2:]]
                polyhedra[pid] = fids
                i += 1
            continue

        i += 1

    vertices_arr = np.array(list(vertices.values()))
    return vertices_arr, edges, faces, polyhedra
