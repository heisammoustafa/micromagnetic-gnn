"""Geometry library for grain structures.

This module provides the core geometric data structures and algorithms for
working with polycrystalline grain structures parsed from NEPER tessellation
files. The main entry points are:

- :class:`Grain` — a single convex grain with vertices, faces, and derived metrics.
- :class:`GrainStructure` — a collection of grains with neighbor relationships.
- :func:`GrainStructure.from_tess_file` — build a structure directly from a
  ``.tess`` file, optionally shrinking grains to account for a non-magnetic
  grain-boundary layer.
- :func:`load_one_grain_structure` — parse a simulation folder (``.tess`` +
  sidecar files) into a fully populated ``GrainStructure``.
- :func:`load_grain_structures` — load every simulation folder under a root
  directory, with optional pickle caching.

Notes
-----
All grain / face / vertex IDs are **zero-based** inside this module. The
:mod:`micromagnetic_gnn.tess_parser` module handles the conversion from the
1-based IDs used in NEPER files.

The ``HalfspaceIntersection`` / ``ConvexHull`` chain can raise
``scipy.spatial.QhullError`` for degenerate grains or when the requested
grain-boundary shrink is too large. For the test data set (unit cube) use
``gb=0.0`` to avoid this.
"""

from __future__ import annotations
import errno
import os

from typing import TypeVar, Any, Union, Mapping, Sequence, Callable
import random
import logging
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from micromagnetic_gnn.tess_parser import parse_tess_file

logger = logging.getLogger(__name__)


mu0 = 4e-7*np.pi


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class Grain:
    """A single convex grain.

    The grain is represented by its vertex positions, face connectivity, and
    derived geometric quantities (volume, surface area, face areas, etc.).

    Attributes
    ----------
    centroid : np.ndarray
        Chebyshev centre of the grain, shape ``(3,)``.
    volume : float
        Volume of the convex polyhedron (nm³ after scaling).
    surface : float
        Total surface area.
    vertices : np.ndarray
        Vertex coordinates, shape ``(n_vertices, 3)``.
    edges : np.ndarray
        Unique edges as vertex-index pairs, shape ``(n_edges, 2)``.
    total_edge_length : float
        Sum of all edge lengths.
    faces : dict[int, list[int]]
        Mapping ``face_id -> [vid1, vid2, ...]``.
    face_equations : np.ndarray
        Inward-facing half-space equations, shape ``(n_faces, 4)``.
        Each row is ``[nx, ny, nz, offset]`` describing ``nx*x + ny*y + nz*z + offset <= 0``.
    face_areas : np.ndarray
        Area of each face, shape ``(n_faces,)``.
    face_circumferences : np.ndarray
        Perimeter of each face, shape ``(n_faces,)``.
    face_centers : np.ndarray
        Chebyshev centre of each face, shape ``(n_faces, 3)``.
    global_face_indices : list[int]
        Original face IDs from the tessellation file that belong to this grain.
    params : dict[str, np.ndarray]
        Additional per-grain parameters (e.g. material properties, labels).
        Each entry is a 0-D or 1-D array so that the downstream pipeline
        can concatenate them automatically.

    Examples
    --------
    >>> gs = GrainStructure.from_tess_file("grain.tess", gb=0.0)
    >>> grain = gs.grains[0]
    >>> grain.volume  # doctest: +SKIP
    0.234
    >>> grain.face_equations.shape  # doctest: +SKIP
    (6, 4)
    """

    center: np.ndarray
    volume: float
    surface: float
    vertices: np.ndarray
    edges: np.ndarray
    total_edge_length: float
    faces: dict[int, list[int]]
    face_equations: np.ndarray
    face_areas: np.ndarray
    face_circumferences: np.ndarray
    face_centers: np.ndarray
    global_face_indices: list[int]
    params: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class GrainStructure:
    """Collection of grains with neighbor relationships and derived edges.

    Attributes
    ----------
    grains : dict[int, Grain]
        Mapping ``grain_id -> Grain``.
    neighbors : dict[int, list[int]]
        Adjacency list: ``grain_id -> [neighbor_id, ...]``.
    edges : np.ndarray
        High-level grain adjacency edges, shape ``(n_edges, 2)``.
        Each edge ``(i, j)`` satisfies ``i < j`` and appears once.
    edge_faces : np.ndarray
        For each edge, the face index on the ``edges[:, 0]`` side,
        shape ``(n_edges,)``.
    params : dict[str, np.ndarray | float]
        Global simulation parameters (e.g. AR, size_x) shared by all grains.

    Examples
    --------
    >>> gs = GrainStructure.from_tess_file("grain.tess", gb=0.0)
    >>> len(gs.grains)  # doctest: +SKIP
    148
    >>> gs.neighbors[0][:3]  # doctest: +SKIP
    [1, 2, 5]
    """

    grains: dict[int, Grain]
    neighbors: dict[int, list[int]]
    edges: np.ndarray
    edge_faces: np.ndarray
    switching_links: np.ndarray | None = None
    switching_label: np.ndarray | None = None
    params: dict[str, np.ndarray | float] = field(default_factory=dict)
    """Global simulation parameters (e.g. AR, size_x) shared by all grains."""

    # @property
    # def edge_face_intersections(self) -> np.ndarray:
    #     """Intersection points of centroid-to-centroid lines with shared faces.

    #     For each high-level edge ``(g1, g2)`` the line segment from the
    #     centroid of *g1* to the centroid of *g2* is intersected with the
    #     shared face plane on the *g1* side.

    #     Returns
    #     -------
    #     np.ndarray
    #         Array of shape ``(n_edges, 3)`` with the intersection coordinates.
    #     """
    #     return _edge_face_intersections(self.grains, self.edges, self.edge_faces)

    @classmethod
    def from_tess_file(cls, file_path: str, gb: float, scale: float = 1.0) -> "GrainStructure":
        """Build a :class:`GrainStructure` from a NEPER tessellation file.

        Parameters
        ----------
        file_path : str
            Path to the ``.tess`` file.
        gb : float
            Grain-boundary thickness. Each interior face is shifted inward
            by ``gb / 2`` to account for the non-magnetic boundary layer.
            Set to ``0.0`` to keep the original tessellation geometry.
        scale : float
            Spatial scale factor applied to all vertex coordinates.
            Default ``1.0`` (no rescaling).

        Returns
        -------
        GrainStructure

        Raises
        ------
        scipy.spatial.QhullError
            If a grain becomes degenerate after shrinking (``gb`` too large).
        """
        vertices, _, faces, polyhedra = parse_tess_file(file_path)
        vertices = vertices * scale
        neighbors = find_neighbors(polyhedra)

        grains = {
            gid: make_grain(grain_faces, vertices, faces, shrink_by=gb / 2)
            for gid, grain_faces in polyhedra.items()
        }
        edges, edge_faces = _make_gs_edges(grains, neighbors)
        edge_faces = edge_faces[:, 0]
        
        # some edges can disappears during grain shrinking, so we could update
        # the neighbors. But for link prediction it is actually useful to keep
        # all the original neighbors.
        # update_neighbors(gs)

        gs = cls(grains, neighbors, edges, edge_faces, params={})
        return gs

    def add_grain_params(self, param_name: str, params: Any) -> None:
        """Attach an additional parameter to every grain.

        Parameters
        ----------
        param_name : str
            Key under which the parameter is stored in each grain's
            :attr:`Grain.params` dictionary.
        params : Any
            Mapping ``grain_id -> value``. Every grain present in the
            structure must have an entry.

        """
        for gid in self.grains:
            self.grains[gid].params[param_name] = params[gid]


# ---------------------------------------------------------------------------
# Grain construction helpers
# ---------------------------------------------------------------------------
def update_neighbors(gs):
    edges = gs.edges
    def find_neighbors(grain_id):
        if len(edges) > 0:
            neighbors = list(edges[edges[:, 0] == grain_id][:, 1])
            return neighbors
        else:
            return []

    gs.neighbors = {i: find_neighbors(i) for i in gs.grains.keys()}


def switching_influence(gs: GrainStructure) -> tuple[np.ndarray, np.ndarray]:
    """If a grain switches, this can cause other grains in its neighborhood to switch as well at the
    same applied field strength. This function computes if a grain causes other grains in its
    neighborhood to switch as well. This influence is unidirectional.

    Parameters
    ----------
    gs : GrainStructure

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        neighbor links and switching influence label
        1=causes switching of neighbor, 0=no switching
    """
    gids = gs.grains.keys()
    sf = [g.params["switching_field"] for g in gs.grains.values()]
    so = [g.params["switching_order"] for g in gs.grains.values()]
    neighbors = [nbrs for nbrs in gs.neighbors.values()]

    df = pd.DataFrame(dict(
        switching_field=sf,
        switching_order=so,
        neighbors=neighbors
    ), index=gids)

    switching_field = df["switching_field"].to_numpy()
    switching_order = df["switching_order"].to_numpy()
    neighbors = df["neighbors"]

    links = []
    labels = []

    for gid, nbrs in enumerate(neighbors):
        sf = switching_field[gid]
        so = switching_order[gid]

        for nid in nbrs:
            links.append((gid, nid))

            labels.append(
                np.isclose(sf, switching_field[nid], atol=1e-5) & (switching_order[nid] > so)
            )
    # _links, _labels = np.array(links).reshape(-1, 2), np.array(labels).reshape(-1)
    # positive_links = _links[_labels, 1]
    # bincount = np.bincount(positive_links, minlength=len(gids))
    # labels_prob = _labels.astype(np.float64)
    # labels_prob[_labels] = labels_prob[_labels] / bincount[positive_links]
    # return np.array(_links).reshape(-1, 2), np.array(labels_prob).reshape(-1)
    return np.array(links).reshape(-1, 2), np.array(labels).reshape(-1)


def make_grain(
    faces: list[int],
    all_vertices: np.ndarray,
    all_faces: dict[int, list[int]],
    shrink_by: float,
) -> Grain:
    """Construct a single :class:`Grain` from its face indices.

    The faces are first converted to inward-facing half-space equations.
    The equations are then shifted inward by *shrink_by* to account for the
    grain-boundary layer. A new convex polyhedron is built from the shifted
    half-spaces and all geometric quantities are derived from it.

    Parameters
    ----------
    faces : list[int]
        Face IDs belonging to this grain.
    all_vertices : np.ndarray
        Global vertex array, shape ``(n_vertices, 3)``.
    all_faces : dict[int, list[int]]
        Global mapping ``face_id -> [vid1, ...]``.
    shrink_by : float
        Distance by which each interior face is shifted inward.

    Returns
    -------
    Grain
    """
    grain_vertices = all_vertices[list(set(chain.from_iterable([all_faces[fid] for fid in faces])))]
    equations = _face_equations(
        all_vertices, {fid: all_faces[fid] for fid in faces}, np.mean(grain_vertices, axis=0)
    )
    equations[:, -1] = equations[:, -1] + shrink_by
    return grain_from_equations(equations, faces)


def grain_from_equations(equations: np.ndarray, global_face_indices: list[int]) -> Grain:
    """Build a :class:`Grain` from half-space equations.

    Parameters
    ----------
    equations : np.ndarray
        Stacked half-space inequalities ``[A | b]`` of shape
        ``(n_faces, 4)`` describing ``A x + b <= 0``.
    global_face_indices : list[int]
        Original face IDs from the tessellation file that correspond to
        the rows of *equations*.

    Returns
    -------
    Grain

    Raises
    ------
    AssertionError
        If ``HalfspaceIntersection`` drops any vertices (indicates a
        degeneracy or inconsistency in the input equations).
    scipy.spatial.QhullError
        If the feasible point is not clearly inside all half-spaces.
    """
    chebyshev_center = compute_chebyshev_center(equations)
    hsi = HalfspaceIntersection(equations, chebyshev_center)
    vertices = hsi.intersections
    convex_hull = ConvexHull(vertices)

    _equations = []
    _global_face_index = []
    faces = {}
    for fid, (eq, face_index) in enumerate(zip(equations, global_face_indices)):
        dist = vertices @ eq[:3] + eq[3]
        tol = 1e-8
        mask = np.abs(dist) < tol
        face_vertices = np.where(mask)[0]
        if len(face_vertices) <= 1:
            # no face
            continue
        assert face_vertices.shape[0] >= 3
        faces[fid] = face_vertices
        _equations.append(eq)
        _global_face_index.append(face_index)

    _face_geom = [polygon_geometry_3d(vertices[face]) for face in faces.values()]
    face_areas, face_circumferences, face_centers = zip(*_face_geom)
    face_areas = np.asarray(face_areas)
    face_circumferences = np.asarray(face_circumferences)
    face_centers = np.asarray(face_centers)
    edges = compute_edges(vertices, faces)
    total_edge_length = total_face_edge_lengths(vertices, edges)

    center = np.mean(vertices, axis=0)

    return Grain(
        #center=chebyshev_center,
        center=center,
        volume=float(convex_hull.volume),
        surface=float(convex_hull.area),
        vertices=vertices,
        edges=edges,
        total_edge_length=total_edge_length,
        faces=faces,
        face_equations=np.array(_equations),
        face_areas=face_areas,
        face_circumferences=face_circumferences,
        face_centers=face_centers,
        global_face_indices=_global_face_index,
    )


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def compute_chebyshev_center(equations: np.ndarray) -> np.ndarray:
    """Compute the Chebyshev center of a polyhedron.

    Solves a linear programme to find the largest inscribed ball centre,
    which is guaranteed to be a strictly interior point suitable for
    :class:`scipy.spatial.HalfspaceIntersection`.

    Parameters
    ----------
    equations : np.ndarray
        Stacked inequalities ``Ax + b <= 0`` in format ``[A | b]``,
        shape ``(n_faces, 4)``.

    Returns
    -------
    np.ndarray
        Interior point of shape ``(3,)``.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html
    """
    norm_vector = np.reshape(np.linalg.norm(equations[:, :-1], axis=1), (equations.shape[0], 1))
    c = np.zeros((equations.shape[1],))
    c[-1] = -1
    A = np.hstack((equations[:, :-1], norm_vector))
    b = -equations[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    center = res.x[:-1]
    return center


def compute_chebyshev_center_from_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Compute the Chebyshev center from vertices.
    """

    hull = ConvexHull(vertices)
    return compute_chebyshev_center(hull.equations)


def _face_equations(
    vertices: np.ndarray,
    faces: dict[int, list[int]],
    interior_point: np.ndarray,
) -> np.ndarray:
    """Compute inward-facing half-space equations for each face.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex array, shape ``(n_vertices, 3)``.
    faces : dict[int, list[int]]
        Mapping ``face_id -> [vid1, ...]``.
    interior_point : np.ndarray
        A point known to be inside the polyhedron (used to orient normals
        inward), shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_faces, 4)`` with rows ``[nx, ny, nz, offset]``.
    """

    def equations_for_face(face: list[int]) -> np.ndarray:
        face_vertices = vertices[face]
        return plane_halfspace(face_vertices, interior_point)

    return np.array([equations_for_face(face) for face in faces.values()])


def polygon_geometry_3d(vertices: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Compute area, perimeter and Chebyshev center of a planar polygon in 3-D.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex array of shape ``(n, 3)``. Vertices must be coplanar and
        may be unordered.

    Returns
    -------
    tuple[float, float, np.ndarray]
        ``(area, perimeter, center)``, where ``center`` has shape ``(3,)``.
    """
    # plane basis
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]

    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)

    e1 = v1 / np.linalg.norm(v1)
    e2 = np.cross(n, e1)

    origin = np.mean(vertices, axis=0)

    # project to 2D
    p = vertices - origin

    x = p @ e1
    y = p @ e2

    # cyclic ordering
    angles = np.arctan2(y, x)
    idx = np.argsort(angles)

    x = x[idx]
    y = y[idx]

    pts2d = np.column_stack([x, y])

    # area
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # perimeter
    perimeter = np.sum(np.linalg.norm(pts2d - np.roll(pts2d, -1, axis=0), axis=1))

    #center = compute_chebyshev_center_from_vertices(pts2d)
    #center3d = origin + center[0] * e1 + center[1] * e2
    center3d = np.mean(vertices, axis=0)
    return float(area), float(perimeter), center3d


def plane_halfspace(vertices: np.ndarray, interior_point: np.ndarray) -> np.ndarray:
    """Compute the inward-facing half-space for a face.

    Parameters
    ----------
    vertices : np.ndarray
        At least three coplanar vertices defining the face, shape ``(m, 3)``.
    interior_point : np.ndarray
        A point inside the polyhedron, used to flip the normal if needed.
        Shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Equation ``[nx, ny, nz, offset]`` describing ``n·x + offset <= 0``.
    """
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    n = np.cross(v1, v2)
    n = -n / np.linalg.norm(n)

    b = -np.dot(n, vertices[0])

    is_outside = np.dot(n, interior_point) + b > 0
    n = np.where(is_outside, -n, n)
    b = np.where(is_outside, -b, b)
    return np.r_[n, b]


def is_neighbor(p1: list[int], p2: list[int]) -> bool:
    """Return ``True`` if two polyhedra share at least one face."""
    s1 = set(p1)
    s2 = set(p2)
    return len(s1.difference(s2)) != len(s1)


def find_neighbors(polyhedra: dict[int, list[int]]) -> dict[int, list[int]]:
    """Build a neighbor list for each grain."""
    neighbors: dict[int, list[int]] = {}
    for idx1, p1 in polyhedra.items():
        neighbors_idx = []
        for idx2, p2 in polyhedra.items():
            if idx1 == idx2:
                continue
            if is_neighbor(p1, p2):
                neighbors_idx.append(idx2)
        neighbors[idx1] = neighbors_idx
    return neighbors


def compute_edges(
    vertices: np.ndarray,
    faces: dict[int, list[int]],
) -> np.ndarray:
    """Compute unique edges from a set of faces."""
    edges: set[tuple[int, int]] = set()

    for face in faces.values():
        # if not face:
        #    continue
        pts = vertices[face]

        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)

        e1 = v1 / np.linalg.norm(v1)
        e2 = np.cross(n, e1)

        p = pts - pts.mean(axis=0)
        x = p @ e1
        y = p @ e2

        angles = np.arctan2(y, x)
        idx = np.argsort(angles)
        ordered = np.array(face)[idx]

        for a, b in zip(ordered, np.roll(ordered, -1)):
            edges.add(tuple(sorted((a, b))))

    return np.array(sorted(edges))


def total_face_edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> float:
    """Compute Euclidean lengths of edges."""
    # if edges.shape[0] == 0:
    #     return 0.0
    edge_vertices = vertices[edges]
    edge_vecs = edge_vertices[:, 1] - edge_vertices[:, 0]
    return float(np.sum(np.linalg.norm(edge_vecs, axis=-1)))


# ---------------------------------------------------------------------------
# Grain-structure edge helpers
# ---------------------------------------------------------------------------

def _make_gs_edges(
    grains: dict[int, Grain],
    neighbors: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    edges = []
    faces = []
    for i1, g1 in grains.items():
        for i2 in neighbors[i1]:
            assert i1 != i2, "Grain can't be neighbor of itself"
            g2 = grains[i2]
            face_idx = set(g1.global_face_indices).intersection(g2.global_face_indices)
            if len(face_idx) == 0:
                # drop neighbor if no adjacent face is found
                continue
            
            assert len(face_idx) == 1, "Multiple neighbor faces found"
            face_idx = list(face_idx)[0]
            idx1 = g1.global_face_indices.index(face_idx)
            idx2 = g2.global_face_indices.index(face_idx)
            edges.append((i1, i2))
            faces.append((idx1, idx2))

    return np.array(edges).reshape(-1, 2), np.array(faces).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Random rotation utilities
# ---------------------------------------------------------------------------

def random_rotation_matrix(rng=np.random.default_rng()) -> np.ndarray:
    u1, u2, u3 = rng.random(3)

    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    # quaternion (x, y, z, w)
    x, y, z, w = q1, q2, q3, q4

    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ])

    return R


# ---------------------------------------------------------------------------
# Simulation loading helpers
# ---------------------------------------------------------------------------


def _parse_parameters(path: Path) -> dict[str, float | int]:
    """Parse ``parameters.txt`` into a dictionary."""
    params: dict[str, float | int] = {}
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if '.' in val:
                val = float(val)
            else:
                val = int(val)
            params[key] = val
    return params


def get_tess_file(sim_dir: str | Path) -> Path:
    sim_dir = Path(sim_dir)
    try:
        tess_file = Path(list(sim_dir.glob("*.tess"))[0])
    except IndexError as e:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"{sim_dir}/*.tess") from e
    
    return tess_file


def load_one_grain_structure(
    sim_dir: Path | str,
) -> GrainStructure:
    """Parse .tess and attach sidecar features / labels.

    The grain-boundary thickness is computed on the fly as
    ``gb_t / size_x`` from ``parameters.txt``.

    Parameters
    ----------
    sim_dir : Path
        Simulation folder.

    Returns
    -------
    GrainStructure
        With per-grain params and global params attached.

    Raises
    ------
    FileNotFoundError
        If a requested file does not exist.
    """
    sim_dir = Path(sim_dir)
    tess_file = get_tess_file(sim_dir)
    
    meta = load_simulation_meta(sim_dir)
    n_grains, gb_t, size_x = meta["no_grains"], meta["gb_t"], meta["size_x"]

    krn_file = sim_dir / "simple.krn"
    krn = np.loadtxt(krn_file).reshape(-1, 7)
    
    assert krn.shape[0] >= n_grains
    krn = krn[:n_grains, :]
    
    stpoly_file = sim_dir / f"numGrains_{n_grains}.stpoly"
    stpoly = np.loadtxt(stpoly_file).reshape(-1, 12)
    assert stpoly.shape[0] == n_grains
    
    switched_file = sim_dir / f"numGrains_{n_grains}" / "results" / "switched.txt"
    switched = pd.read_csv(switched_file, names=["switching_field", "grain_index"], 
                        sep=r"\s", engine="python")
    switched["switching_order"] = switched.index
    switched.set_index("grain_index", inplace=True)
    switched.sort_index(inplace=True)
    switched = switched[~switched.index.duplicated()]
    
    # load grain structure
    gs = GrainStructure.from_tess_file(str(tess_file), gb=gb_t, scale=size_x)
    
    # ----------------- Set gs parameters -----------------
    gs_param_names = ["folder", "no_grains", "size_x", "gb_t", "coercivity", "bhmax"]
    gs.params = {k: meta[k] for k in gs_param_names}
    gs.params["coercivity"] = np.abs(gs.params["coercivity"])  # some values are stored negative
    
    # ----------------- Set node parameters -----------------
    # Convert theta / phi (cols 0 / 1) to a 3-D easy-axis unit vector
    theta = krn[:, 0].astype(float)
    phi = krn[:, 1].astype(float)
    easy_axis = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=1,
    )
    param_names = ["easy_axis", "K", "Ms", "A", "coo", "vol", "diameq", "sphericity",
                   "convexity", "area", "dihanglemax",
                   "dihanglemin", "dihangleav", "npolynb", "switching_field", "switching_order"]
    param_values = (
        [easy_axis, krn[:, 2], krn[:, 4], krn[:, 5]]
        + [stpoly[:, :3]]
        + [stpoly[:, i] for i in list(range(3, 12))]
        + [np.asarray(switched.switching_field), np.asarray(switched.switching_order)]
    )
    assert len(param_names) == len(param_values)
    for _name, _params in zip(param_names, param_values):
        gs.add_grain_params(_name, _params)

    # compute the switching graph
    switching_links, switching_label = switching_influence(gs)
    gs.switching_links = switching_links
    gs.switching_label = switching_label
    return gs


def scan_simulation_folders(root_dir: str | Path) -> pd.DataFrame:
    """Scan *root_dir* for simulation folders and return a metadata DataFrame.

    Each row corresponds to one simulation folder.  Grain-level properties
    (easy_axis, K, Ms, A, switching_field) are **not** included — only
    structure-level quantities.

    Folders without a ``.tess`` file or ``parameters.txt`` are skipped.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing simulation sub-folders.

    Returns
    -------
    pd.DataFrame
    """
    rows: list[dict[str, object]] = []
    root = Path(root_dir)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            row = load_simulation_meta(folder)
        except OSError as e:
            print(e)
            continue
        
        rows.append(row)

    return pd.DataFrame(rows)


def load_simulation_meta(folder: Path) -> dict["str", Any]:
    params_file = folder / "parameters.txt"
    params = _parse_parameters(params_file)
    n_grains = params["no_grains"]
    hc_file = folder / f"numGrains_{int(n_grains)}" / "results" / "hc.dat"
    hys_log_file = folder / f"numGrains_{int(n_grains)}" / "results" / "hys.log"
    
    coercivities = hc_file.read_text().splitlines()
    coercivity = abs(float(coercivities[0]))
    hlc = float(coercivities[1])  # field with largest change

    meta: dict[str, object] = dict(params)
    meta["folder"] = str(folder)
    meta["no_grains"] = n_grains
    meta["coercivity"] = coercivity
    meta["hlc"] = hlc
    meta["bhmax"] = compute_bhmax(hys_log_file)
    
    return meta


def compute_bhmax(hys_log_file_path: Path) -> float:
    with open(hys_log_file_path, 'r') as f:
        data = np.loadtxt(f, comments='#')
        
    if data.shape[1] == 2:
        # two col format
        h = data[:, 0]
        m = data[:, 1]    
    else:
        h = data[:, 1]
        m = data[:, 4]
    hi = h - (1 / 3) * m      # Tesla
    bi = hi + m       # Tesla
    bh = -bi * hi / mu0   # J/m³
    bhmax = max(bh)
    return bhmax


def load_datasets(
    root_dir: str | Path,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int | None = 42,
    extrapolation_cond: Callable | None = None,
    max_workers: int | None = None,
    skip_errors: bool = False,
    cache_path: str | Path | None = None,
    rebuild_cache: bool = False,
) -> tuple[list[GrainStructure], list[GrainStructure], list[GrainStructure], list[GrainStructure]]:

    dataset = load_grain_structures(root_dir, max_workers=max_workers, skip_errors=skip_errors, 
                                    cache_path=cache_path, rebuild_cache=rebuild_cache)
    
    extrapolation_set = [] if extrapolation_cond is None else [gs for gs in dataset if extrapolation_cond(gs)]
    dataset = dataset if extrapolation_cond is None else [gs for gs in dataset if not extrapolation_cond(gs)]
    total = len(dataset)
    random.Random(seed).shuffle(dataset)
    
    train_size = int(train_frac * total)
    val_size = int(val_frac * total)

    train_set, val_set, test_set = dataset[:train_size], dataset[train_size:train_size + val_size], dataset[train_size + val_size:]
    return train_set, val_set, test_set, extrapolation_set
    
    
def load_grain_structures(
    root_dir: str | Path,
    *,
    max_workers: int | None = None,
    skip_errors: bool = False,
    cache_path: str | Path | None = None,
    rebuild_cache: bool = False,
) -> list[GrainStructure]:
    """Load every simulation folder under *root_dir*.

    The parsed ``GrainStructure`` objects can be cached as a pickle so
    that subsequent runs skip the file-system parsing.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing simulation folders.
    max_workers : int or None
        Parallel workers. ``None`` → all available CPU cores.
    skip_errors : bool
        If ``False`` (default), the first failed folder raises.
        If ``True``, failed folders are skipped with a warning.
    cache_path : str or Path or None
        Path to a pickle file that stores the loaded structures.
    rebuild_cache : bool
        If ``True``, ignore an existing cache and rebuild it.

    Returns
    -------
    list[GrainStructure]
    """
    root_dir = Path(root_dir)
    cache_path = Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists() and not rebuild_cache:
        logger.info("Loading cached grain structures from %s ...", cache_path)
        with open(cache_path, "rb") as f:
            structures = pickle.load(f)
        logger.info("Cache loaded: %d structures.", len(structures))
        return structures

    folders = sorted(d for d in root_dir.iterdir() if d.is_dir())
    n_total = len(folders)
    logger.info("Loading %d simulation folders from %s ...", n_total, root_dir)

    structures: list[GrainStructure] = []
    n_loaded = 0
    n_failed = 0
    log_every = max(1, n_total // 20)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_one_grain_structure, f): f for f in folders}
        for future in futures:
            folder = futures[future]
            try:
                structures.append(future.result())
                n_loaded += 1
            except Exception as e:
                n_failed += 1
                if skip_errors:
                    logger.error(
                        "Failed to load %s\n%s",
                        folder,
                        traceback.format_exc(),
                    )
                else:
                    raise e
            if (n_loaded + n_failed) % log_every == 0:
                logger.info(
                    "  ... %d/%d done (%d loaded, %d failed)",
                    n_loaded + n_failed,
                    n_total,
                    n_loaded,
                    n_failed,
                )

    logger.info(
        "Finished loading %d/%d folders (%d success, %d failed)",
        n_loaded + n_failed,
        n_total,
        n_loaded,
        n_failed,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing cache to %s ...", cache_path)
        with open(cache_path, "wb") as f:
            pickle.dump(structures, f)

    return structures
