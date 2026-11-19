# Usage Guide

## Parsing a Tessellation File

```python
from micromagnetic_gnn import parse_tess_file

vertices, edges, faces, polyhedra = parse_tess_file("path/to/file.tess")
```

## Building a Grain Structure

```python
from micromagnetic_gnn import GrainStructure

# gb = grain boundary thickness (nm)
gs = GrainStructure.from_tess_file("path/to/file.tess", gb=1.0)

# Access individual grains
for gid, grain in gs.grains.items():
    print(f"Grain {gid}: volume={grain.volume:.2f}, surface={grain.surface:.2f}")

# Neighbor relationships
for gid, nbrs in gs.neighbors.items():
    print(f"Grain {gid} neighbors: {nbrs}")
```

## Computing Shrunk Volumes

```python
from micromagnetic_gnn import compute_shrunk_volumes

volumes = compute_shrunk_volumes("path/to/file.tess", scale=1.0, gb_thickness=1.0)
```

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_grain_structure.py

# With coverage
pytest --cov=micromagnetic_gnn --cov-report=term-missing
```

## Type Checking

```bash
mypy src/micromagnetic_gnn
```

## Linting

```bash
ruff check src tests
ruff format src tests
```
