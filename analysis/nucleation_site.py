"""Where does reversal start -- at the sample surface, or in the bulk?

Notation: ``H_sim`` is the *simulated* per-grain switching field
(``params["switching_field"]``, from ``results/switched.txt``); ``H_astroid``
is the *analytic* Stoner-Wohlfarth switching field of the same grain treated
as isolated, from its misalignment and anisotropy field. They are different
quantities and are never mixed.

Nucleation is expected at free surfaces, where the demagnetizing field is
strongest and the grain is least constrained by neighbours. This tests that
directly against the simulated per-grain switching fields.

Surface exposure is measured by ``face_frac``: the share of a grain's surface
area shared with a neighbour. A fully interior grain has ``face_frac = 1``; a
grain on the domain boundary has exposed facets and sits below it. It is already
a node feature, so no new geometry is needed.

Every correlation is computed **within** a structure, then averaged over
structures, so differences in overall level between magnets cannot create a
spurious result. Because misalignment is the dominant known driver of the
switching field, the surface effect is reported both raw and partialled on the
Stoner-Wohlfarth astroid -- otherwise ``face_frac`` could simply be standing in
for orientation.

Run from the repo root::

    python analysis/nucleation_site.py
    python analysis/nucleation_site.py --n-structures 400
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from micromagnetic_gnn.data_pipeline import compute_int_face_frac
from micromagnetic_gnn.dataset import extrapolation_cond

MIN_GRAINS = 30
MU0 = 4e-7 * np.pi
INTERIOR = 0.999          # face_frac above this = no exposed facet


def astroid(psi: np.ndarray, h_k: np.ndarray) -> np.ndarray:
    c = np.abs(np.cos(psi)) ** (2.0 / 3.0)
    s = np.abs(np.sin(psi)) ** (2.0 / 3.0)
    return h_k / np.power(c + s, 1.5)


def face_fracs(gs) -> np.ndarray:
    """Vectorised :func:`compute_int_face_frac` for every grain at once.

    ``gs.edges`` stores each pair in both orientations, so summing the shared
    facet areas over column 0 counts each of a grain's internal faces exactly
    once -- the same convention as the per-grain helper.
    """
    gids = sorted(gs.grains)
    idx = {g: i for i, g in enumerate(gids)}
    total = np.array([gs.grains[g].surface for g in gids], float)
    shared = np.zeros(len(gids))
    if len(gs.edges) == 0:
        return shared
    e = np.asarray(gs.edges)
    ef = np.asarray(gs.edge_faces)
    areas = np.array([gs.grains[a].face_areas[f] for a, f in zip(e[:, 0], ef)], float)
    np.add.at(shared, np.array([idx[a] for a in e[:, 0]]), areas)
    return shared / np.maximum(total, 1e-30)


def grain_table(gs, field_axis=np.array([0.0, 0.0, 1.0])):
    """Per-grain (H_sim, H_astroid, face_frac) for one structure."""
    gids = sorted(gs.grains)
    h_sim = np.array([abs(gs.grains[g].params["switching_field"]) for g in gids], float)
    easy = np.array([gs.grains[g].params["easy_axis"] for g in gids], float)
    k1 = np.array([gs.grains[g].params["K"] for g in gids], float)
    j_s = np.array([gs.grains[g].params["Ms"] for g in gids], float)
    psi = np.arccos(np.clip(np.abs(easy @ field_axis), 0.0, 1.0))
    h_k = 2.0 * k1 * MU0 / np.maximum(j_s, 1e-30)
    return h_sim, astroid(psi, h_k), face_fracs(gs)


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman between x and y with the effect of *control* removed.

    Rank-transform all three, then correlate the residuals of x and y after
    linear regression on the control's ranks.
    """
    rx, ry, rc = (rankdata(v).astype(float) for v in (x, y, control))
    if np.std(rc) < 1e-12:
        return spearmanr(x, y).statistic
    rc = (rc - rc.mean()) / np.std(rc)
    res = lambda r: r - rc * float(np.dot(r - r.mean(), rc) / len(rc))
    return float(spearmanr(res(rx), res(ry)).statistic)


def describe(gs, field_axis=np.array([0.0, 0.0, 1.0])) -> dict | None:
    gids = sorted(gs.grains)
    if len(gids) < MIN_GRAINS:
        return None
    h_true = np.array([abs(gs.grains[g].params["switching_field"]) for g in gids], float)
    if not np.all(np.isfinite(h_true)) or np.ptp(h_true) == 0:
        return None

    face_frac = np.array([compute_int_face_frac(gs, g) for g in gids], float)
    easy = np.array([gs.grains[g].params["easy_axis"] for g in gids], float)
    k1 = np.array([gs.grains[g].params["K"] for g in gids], float)
    j_s = np.array([gs.grains[g].params["Ms"] for g in gids], float)
    vol = np.array([gs.grains[g].volume for g in gids], float)

    psi = np.arccos(np.clip(np.abs(easy @ field_axis), 0.0, 1.0))
    h_k = 2.0 * k1 * MU0 / np.maximum(j_s, 1e-30)
    h_astroid = astroid(psi, h_k)

    surface = face_frac < INTERIOR
    out = dict(
        n_grains=len(gids),
        surface_frac=float(surface.mean()),
        # positive rho => more enclosed grains switch at HIGHER field,
        # i.e. exposed (surface) grains reverse first
        rho_facefrac=float(spearmanr(face_frac, h_true).statistic),
        rho_facefrac_partial=partial_spearman(face_frac, h_true, h_astroid),
        rho_astroid=float(spearmanr(h_astroid, h_true).statistic),
        rho_volume=float(spearmanr(vol, h_true).statistic),
    )
    if surface.any() and (~surface).any():
        out["delta_h"] = float(np.median(h_true[~surface]) - np.median(h_true[surface]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data_cache.pkl"))
    ap.add_argument("--n-structures", type=int, default=300, help="0 = all")
    ap.add_argument("--subsample", type=int, default=25,
                    help="Grains sampled per subgroup, so interior and surface\n"
                         "correlations are estimated at equal noise.")
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
        d = describe(gs)
        if d:
            groups["generalization" if extrapolation_cond(gs) else "train / test"].append(d)

    names = [k for k, v in groups.items() if v]
    print("\nwhere reversal starts -- within-structure rank correlations with the")
    print("simulated switching field H_sim, averaged over structures\n" + "=" * 68)
    print(f"{'':<38}" + "".join(f"{k:>15}" for k in names))
    rows = [
        ("grains per structure", "n_grains", "{:.0f}"),
        ("grains with an exposed facet", "surface_frac", "{:.1%}"),
        ("rho(face_frac, H_sim)", "rho_facefrac", "{:+.3f}"),
        ("  ... partialled on astroid", "rho_facefrac_partial", "{:+.3f}"),
        ("rho(H_astroid, H_sim) [reference]", "rho_astroid", "{:+.3f}"),
        ("rho(volume, H_sim)   [reference]", "rho_volume", "{:+.3f}"),
        ("median H_sim: interior - surface (T)", "delta_h", "{:+.3f}"),
    ]
    for label, key, fmt in rows:
        cells = []
        for k in names:
            v = np.array([d[key] for d in groups[k] if key in d], float)
            cells.append(fmt.format(np.nanmedian(v)) if v.size else "-")
        print(f"{label:<38}" + "".join(f"{c:>15}" for c in cells))
    print(f"{'structures':<38}" + "".join(f"{len(groups[k]):>15}" for k in names))

    print("\nRead: rho(face_frac, H_sim) > 0 means enclosed grains need a stronger field,\n"
          "so reversal nucleates at the exposed surface. The partialled row is the one\n"
          "that matters -- it removes orientation, the dominant driver.")

    stratified(structures, args.subsample, rng)
    return 0


def stratified(structures, subsample: int, rng) -> None:
    """Does the astroid lose its grip because of surfaces, or in the bulk too?

    ``rho(H_astroid, H_sim)`` falls from ~0.51 to ~0.06 between small and large
    structures. Two explanations make opposite predictions about periodic
    boundaries: a *composition* effect (small structures are mostly surface
    grains, large ones mostly bulk) would be removed by PBC, whereas genuine
    long-range bulk physics would not.

    Splitting each structure into its interior and surface grains separates
    them. Both subgroups are subsampled to the same number of grains so the
    columns are not compared at different estimator noise.
    """
    rows = []
    for gs in structures:
        try:
            h_sim, h_ast, ff = grain_table(gs)
        except (KeyError, ValueError):
            continue
        if h_sim.size < MIN_GRAINS or not np.all(np.isfinite(h_sim)):
            continue
        interior = ff >= INTERIOR
        rec = {"n_grains": h_sim.size, "n_interior": int(interior.sum())}

        def rho(mask):
            if mask.sum() < subsample:
                return np.nan
            pick = rng.choice(np.flatnonzero(mask), subsample, replace=False)
            a, b = h_ast[pick], h_sim[pick]
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                return np.nan
            return float(spearmanr(a, b).statistic)

        rec["all"] = rho(np.ones_like(interior))
        rec["interior"] = rho(interior)
        rec["surface"] = rho(~interior)
        rows.append(rec)

    if not rows:
        return
    n = np.array([r["n_grains"] for r in rows], float)
    edges = np.array([0, 100, 300, 1000, np.inf])
    labels = ["< 100", "100-300", "300-1000", "> 1000"]

    print("\n\nrho(H_astroid, H_sim) by grain population, "
          f"{subsample} grains sampled per group\n" + "=" * 68)
    print(f"{'grains in structure':<22}{'all':>11}{'interior':>11}{'surface':>11}"
          f"{'structures':>13}")
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = (n >= lo) & (n < hi)
        if not sel.any():
            continue
        cells = []
        for key in ("all", "interior", "surface"):
            v = np.array([r[key] for r, s in zip(rows, sel) if s], float)
            v = v[np.isfinite(v)]
            cells.append(f"{np.median(v):+.3f}" if v.size else "-")
        counts = [np.isfinite([r[k] for r, s in zip(rows, sel) if s]).sum()
                  for k in ("interior",)]
        print(f"{lab:<22}" + "".join(f"{c:>11}" for c in cells)
              + f"{int(sel.sum()):>13}  (interior usable: {counts[0]})")

    print("\nRead: if the interior column stays flat across the size bins, the collapse\n"
          "is a surface-composition effect and periodic boundaries should remove it.\n"
          "If interior grains collapse too, it is bulk physics and PBC will not help.")


if __name__ == "__main__":
    raise SystemExit(main())
