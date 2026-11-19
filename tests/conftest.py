"""Shared test fixtures."""

from pathlib import Path

import pytest

from micromagnetic_gnn.grain_structure import GrainStructure


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to the test simulation data directory."""
    return Path(__file__).parent.parent / "ROM_Simulations_test"


@pytest.fixture(scope="session")
def test_structure_dir() -> Path:
    """Path to the test simulation data directory."""
    return Path(__file__).parent.parent / "ROM_Simulations_test" / "random_gb_t=3_AR=4.9_size_x=148"


@pytest.fixture(scope="session")
def tess_file(test_structure_dir: Path) -> Path:
    """Path to the test ``.tess`` file."""
    return test_structure_dir / "numGrains_5.tess"


@pytest.fixture(scope="session")
def grain_structure(tess_file: Path) -> GrainStructure:
    """Parsed ``GrainStructure`` with zero grain-boundary shrink for unit-cube geometry."""
    return GrainStructure.from_tess_file(str(tess_file), gb=0.0)
