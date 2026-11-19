"""What the grain graphs look like, and how much of one the model can see.

The encoder stacks ``N_CONV`` message-passing layers, so a grain's representation
is built from its ``N_CONV``-hop neighbourhood and nothing beyond. This measures
that neighbourhood against the size of the graph, split by the train /
generalization boundary of ``dataset.py`` (N > 1500 grains or size_x > 30 um).

The question it answers: are the generalization structures *topologically*
different, or only bigger? If the local topology matches and only the reach
changes, the generalization gap is a receptive-field problem rather than a
distribution shift in graph structure.

Run from the repo root::

    python analysis/graph_structure.py
    python analysis/graph_structure.py --n-conv 6 --n-structures 400
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from micromagnetic_gnn.dataset import extrapolation_cond

N_SOURCES = 24        # BFS seeds per structure; full APSP is not needed for a mean


def adjacency(gs) -> sp.csr_matrix:
    """Undirected grain adjacency, grains indexed by sorted id."""
    gids = sorted(gs.grains)
    idx = {g: i for i, g in enumerate(gids)}
    n = len(gids)
    if len(gs.edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.int8)
    rows = [idx[e[0]] for e in gs.edges]
    cols = [idx[e[1]] for e in gs.edges]
    a = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    a = a + a.T
    a.setdiag(0)
    a.eliminate_zeros()
    return (a > 0).astype(np.int8).tocsr()


def hop_reach(a: sp.csr_matrix, n_conv: int, rng) -> float:
    """Mean fraction of the graph within ``n_conv`` hops of a node."""
    n = a.shape[0]
    if n <= 1:
        return 1.0
    sources = rng.choice(n, size=min(N_SOURCES, n), replace=False)
    frac = []
    for s in sources:
        seen = np.zeros(n, bool)
        seen[s] = True
        front = np.array([s])
        for _ in range(n_conv):
            if front.size == 0:
                break
            nxt = np.unique(a[front].indices)
            nxt = nxt[~seen[nxt]]
            seen[nxt] = True
            front = nxt
        frac.append(seen.sum() / n)
    return float(np.mean(frac))


def clustering(a: sp.csr_matrix) -> float:
    """Mean local clustering coefficient (fraction of closed neighbour triples)."""
    deg = np.asarray(a.sum(1)).ravel()
    ok = deg > 1
    if not ok.any():
        return float("nan")
    tri = np.asarray((a @ a).multiply(a).sum(1)).ravel()      # 2x triangles per node
    c = np.zeros(a.shape[0])
    c[ok] = tri[ok] / (deg[ok] * (deg[ok] - 1))
    return float(c[ok].mean())


def describe(gs, n_conv: int, rng) -> dict | None:
    a = adjacency(gs)
    n = a.shape[0]
    if n < 10:
        return None
    deg = np.asarray(a.sum(1)).ravel()
    return dict(
        n_grains=n,
        n_edges=int(a.nnz // 2),
        mean_degree=float(deg.mean()),
        sd_degree=float(deg.std()),
        clustering=clustering(a),
        reach=hop_reach(a, n_conv, rng),
        size_x=float(gs.params.get("size_x", np.nan)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data_cache.pkl"))
    ap.add_argument("--n-conv", type=int, default=6,
                    help="Message-passing layers in the encoder (coercivity model uses 6).")
    ap.add_argument("--n-structures", type=int, default=300, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    with open(args.cache, "rb") as f:
        structures = pickle.load(f)
    if args.n_structures and args.n_structures < len(structures):
        structures = [structures[i] for i in
                      rng.choice(len(structures), args.n_structures, replace=False)]

    groups: dict[str, list] = {"train / test": [], "generalization": []}
    for gs in structures:
        d = describe(gs, args.n_conv, rng)
        if d:
            groups["generalization" if extrapolation_cond(gs) else "train / test"].append(d)

    print(f"\ngrain graphs, {args.n_conv} message-passing layers\n" + "=" * 60)
    rows = [("grains", "n_grains", "{:.0f}"), ("edges", "n_edges", "{:.0f}"),
            ("domain edge length (um)", "size_x", "{:.1f}"),
            ("mean degree", "mean_degree", "{:.2f}"),
            ("degree spread (sd)", "sd_degree", "{:.2f}"),
            ("clustering coefficient", "clustering", "{:.3f}"),
            (f"graph seen in {args.n_conv} hops", "reach", "{:.1%}")]
    names = [k for k, v in groups.items() if v]
    print(f"{'':<26}" + "".join(f"{k:>18}" for k in names))
    for label, key, fmt in rows:
        cells = []
        for k in names:
            v = np.array([d[key] for d in groups[k]], float)
            if key == "size_x":
                v = v / 1000
            cells.append(fmt.format(np.nanmedian(v)))
        print(f"{label:<26}" + "".join(f"{c:>18}" for c in cells))
    print(f"{'structures':<26}" + "".join(f"{len(groups[k]):>18}" for k in names))

    print("\nRead: mean degree and clustering are local quantities -- if they match\n"
          "across the split, the generalization structures are not topologically\n"
          "different, only larger, and the model simply sees a smaller share of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
