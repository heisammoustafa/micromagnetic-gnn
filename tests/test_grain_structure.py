"""Tests for grain geometry and structure."""
from pathlib import Path

import numpy as np
import pytest

from micromagnetic_gnn.grain_structure import GrainStructure, load_grain_structures, load_one_grain_structure, scan_simulation_folders


def test_grain_structure_loads(grain_structure: GrainStructure) -> None:
    assert len(grain_structure.grains) > 0
    assert len(grain_structure.neighbors) > 0
    assert len(grain_structure.edges) > 0


def test_all_grains_have_positive_volume(grain_structure: GrainStructure) -> None:
    for grain in grain_structure.grains.values():
        assert grain.volume > 0


def test_all_grains_have_faces(grain_structure: GrainStructure) -> None:
    for grain in grain_structure.grains.values():
        assert len(grain.faces) > 0
        assert len(grain.face_equations) == len(grain.faces)


def test_neighbors_are_bidirectional(grain_structure: GrainStructure) -> None:
    for gid, nbrs in grain_structure.neighbors.items():
        for nbr in nbrs:
            assert gid in grain_structure.neighbors[nbr]


def test_edge_faces_within_bounds(grain_structure: GrainStructure) -> None:
    for edge, ef in zip(grain_structure.edges, grain_structure.edge_faces):
        g0 = grain_structure.grains[edge[0]]
        assert 0 <= ef < len(g0.faces)


def test_total_volume_unit_cube_gb_zero(tess_file: pytest.FixtureRequest) -> None:
    """With no grain-boundary shrink the grains fill the unit cube exactly."""
    from pathlib import Path

    gs = GrainStructure.from_tess_file(str(tess_file), gb=0.0)
    total_volume = sum(g.volume for g in gs.grains.values())
    assert total_volume == pytest.approx(1.0, abs=0.01)


def test_total_volume_shrunk_less_than_one(tess_file: pytest.FixtureRequest) -> None:
    """With a realistic gb shrink the total volume drops below the unit cube."""
    from pathlib import Path

    # Original structure: size_x = 148 nm, gb_t = 3 nm
    # Tessellation is scaled to unit cube, so scaled gb = 3 / 148
    gb_scaled = 3.0 / 148.0
    gs = GrainStructure.from_tess_file(str(tess_file), gb=gb_scaled)
    total_volume = sum(g.volume for g in gs.grains.values())
    assert total_volume < 1.0
    assert total_volume > 0.85  # reasonable sanity lower bound


def test_add_grain_params_attaches_values(grain_structure: GrainStructure) -> None:
    """add_grain_params stores per-grain arrays under the given key."""
    params = {gid: np.array([float(gid)]) for gid in grain_structure.grains}
    grain_structure.add_grain_params("switching_field", params)

    for gid, grain in grain_structure.grains.items():
        assert "switching_field" in grain.params
        np.testing.assert_array_equal(grain.params["switching_field"], params[gid])


def test_add_grain_params_missing_grain_raises(grain_structure: GrainStructure) -> None:
    """add_grain_params raises KeyError if any grain is missing from the dict."""
    params = {gid: np.array([1.0]) for gid in list(grain_structure.grains)[:-1]}
    with pytest.raises(KeyError):
        grain_structure.add_grain_params("test", params)


def test_new_grain_has_empty_params(tess_file: pytest.FixtureRequest) -> None:
    """Grains created from tess file start with an empty params dict."""
    gs = GrainStructure.from_tess_file(str(tess_file), gb=0.0)
    for grain in gs.grains.values():
        assert grain.params == {}



def test_load_one_grain_structure_all(test_structure_dir: Path) -> None:
    """Default flags load every sidecar."""
    gs = load_one_grain_structure(test_structure_dir)

    assert len(gs.grains) == 5
    assert set(gs.params.keys()) >= {"bhmax", "coercivity", "no_grains", "size_x"}
    assert gs.params["size_x"] == pytest.approx(148.0, abs=1e-3)
    assert gs.params["coercivity"] == pytest.approx(4.54, abs=1e-3)

    for gid in gs.grains:
        p = gs.grains[gid].params
        assert "easy_axis" in p
        assert "switching_field" in p

    assert gs.grains[3].params["switching_field"] == pytest.approx(-4.435, abs=1e-3)


def test_switching_influence(test_structure_dir: Path) -> None:
    """Switching is a cascade effect and modeled via DAG"""
    gs = load_one_grain_structure(test_structure_dir)
    links, labels = gs.switching_links, gs.switching_label 
    
    a = links[labels==0]
    b = np.array(links[labels==1][:, ::-1])
    a_view = a.view([("", a.dtype)] * a.shape[1])
    b_view = b.view([("", b.dtype)] * b.shape[1])

    is_dag = np.intersect1d(a_view, b_view).size == np.count_nonzero(labels)
    assert is_dag


def test_load_one_grain_structure_missing_raises() -> None:
    """Missing requested file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_one_grain_structure(Path("/nonexistent/path"))


def test_scan_simulation_folders(test_structure_dir: Path) -> None:
    """Metadata DataFrame contains expected columns and values."""
    root = test_structure_dir.parent
    df = scan_simulation_folders(root)
    assert len(df) == 1
    assert df.iloc[0]["folder"] == str(test_structure_dir)
    assert df.iloc[0]["no_grains"] == 5
    assert df.iloc[0]["size_x"] == pytest.approx(148.0, abs=1e-3)
    assert df.iloc[0]["gb_t"] == pytest.approx(3.0, abs=1e-3)
    assert df.iloc[0]["coercivity"] == pytest.approx(4.54, abs=1e-3)
    assert "AR" in df.columns


def test_load_grain_structures_default_no_skip(test_structure_dir: Path) -> None:
    """skip_errors defaults to False."""
    root = Path(__file__).parent.parent / "ROM_Simulations_test"
    structures = load_grain_structures(root, max_workers=1)
    assert len(structures) == 1


def test_load_grain_structures_skip_errors() -> None:
    """With skip_errors=True, broken folders are skipped."""
    root = Path(__file__).parent.parent / "ROM_Simulations_test"
    structures = load_grain_structures(root, max_workers=1, skip_errors=True)
    assert len(structures) == 1


def test_cache_round_trip(tmp_path: Path, test_structure_dir: Path) -> None:
    """Pickle cache round-trip preserves the loaded grain structures."""
    cache = tmp_path / "structures.pkl"

    structures = load_grain_structures(
        test_structure_dir.parent,
        max_workers=1,
        cache_path=cache,
        rebuild_cache=True,
    )
    assert len(structures) == 1
    original_coercivity = structures[0].params["coercivity"]

    structures2 = load_grain_structures(
        test_structure_dir.parent,
        max_workers=1,
        cache_path=cache,
        rebuild_cache=False,
    )
    assert len(structures2) == 1
    assert structures2[0].params["coercivity"] == pytest.approx(original_coercivity)


