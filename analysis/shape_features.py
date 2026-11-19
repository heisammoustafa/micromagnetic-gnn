"""Direction-aware grain-shape and dipolar-coupling features, and a test of
whether they explain more of the per-grain switching field.

Motivation: every geometric node descriptor currently in use (volume, surface,
total_edge_length, diameq, sphericity, dihedral angles, convexity, npolynb) is
invariant under rotation of the grain. The physics is not -- the field is along
z, and the nucleation field of a grain depends on how its shape is oriented with
respect to it. Two grains identical up to a rotation get identical features and
different switching fields.

New node features
-----------------
From the exact volume-covariance tensor of the (shrunken) grain polyhedron:
  ``a/c``, ``b/c``        principal semi-axis ratios of the equivalent ellipsoid
  ``|cos(long, z)|``      orientation of the long axis w.r.t. the field
  ``N_zz``                demagnetising factor along the field, from the
                          equivalent ellipsoid: N = sum_i N_i u_i u_i^T, so
                          N_zz = sum_i N_i (u_i . z)^2, with N_i the standard
                          ellipsoid demagnetising factors (Osborn integrals,
                          evaluated by vectorised Gauss-Legendre quadrature).
  ``ea . long axis``      easy axis vs the grain's own long axis

From the neighbours, the dipolar interaction that actually drives the cascades
(currently unrepresented -- ``dipole_coupling`` is the easy-axis angle alone,
with no separation, no volume, no geometry):
  ``dip_field_z``         z-component of the stray field from saturated
                          neighbours, sum_j V_j (3 (m_j.r) r - m_j)_z / |r|^3
  ``dip_energy``          the same field projected on the grain's own moment

Run from the repo root::

    python analysis/shape_features.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore", message="An input array is constant")

from micromagnetic_gnn.data_pipeline import NODE_FEATURE_MAP, _extract_node_features

CACHE = Path("data_cache.pkl")
MIN_GRAINS = 50
RNG = np.random.default_rng(0)

I_SW = NODE_FEATURE_MAP["switching_field"][0]
I_VOL = NODE_FEATURE_MAP["volume"][0]
I_SPH = NODE_FEATURE_MAP["sphericity"][0]

# Gauss-Legendre nodes for the Osborn integrals, mapped from [0,1) to [0,inf)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)
_T = 0.5 * (_GL_X + 1.0)
_WT = 0.5 * _GL_W


def h_sw(t):
    return (np.abs(np.cos(t)) ** (2 / 3) + np.abs(np.sin(t)) ** (2 / 3)) ** -1.5


def demag_factors(a, b, c):
    """Ellipsoid demagnetising factors N_a, N_b, N_c (sum to 1), vectorised.

    N_i = (abc/2) int_0^inf ds / ((s + a_i^2) sqrt((s+a^2)(s+b^2)(s+c^2)))
    with s = t/(1-t) mapping the half-line onto [0,1).
    """
    t = _T[None, :]
    s = t / (1.0 - t)
    jac = 1.0 / (1.0 - t) ** 2
    a2, b2, c2 = (x[:, None] ** 2 for x in (a, b, c))
    root = np.sqrt((s + a2) * (s + b2) * (s + c2))
    pref = (a * b * c / 2.0)[:, None]
    out = []
    for ax2 in (a2, b2, c2):
        out.append(np.sum(pref * jac * _WT[None, :] / ((s + ax2) * root), axis=1))
    N = np.stack(out, axis=1)
    return N / np.maximum(N.sum(axis=1, keepdims=True), 1e-30)


def shape_tensor(vertices):
    """Exact volume centroid and covariance of a convex polyhedron.

    The hull is triangulated and each triangle forms a tetrahedron with the
    origin; for a tetrahedron with vertices v_i,
        int x x^T dV = (V/20) (sum_i v_i v_i^T + (sum_i v_i)(sum_i v_i)^T).
    """
    try:
        hull = ConvexHull(vertices)
    except Exception:
        return None
    P = vertices[hull.simplices]                       # (F, 3, 3)
    zero = np.zeros((len(P), 1, 3))
    T = np.concatenate([zero, P], axis=1)              # (F, 4, 3) tetrahedra
    vol = np.abs(np.einsum("fi,fi->f", P[:, 0],
                           np.cross(P[:, 1], P[:, 2]))) / 6.0
    V = vol.sum()
    if V <= 0:
        return None
    m1 = np.einsum("f,fij->j", vol, T) / 4.0           # int x dV
    ssum = T.sum(axis=1)                               # (F, 3)
    m2 = np.einsum("f,fij,fik->jk", vol / 20.0, T, T) \
        + np.einsum("f,fj,fk->jk", vol / 20.0, ssum, ssum)
    cen = m1 / V
    cov = m2 / V - np.outer(cen, cen)
    return V, cen, cov


def node_features(gs):
    X = _extract_node_features(gs).numpy()
    gids = sorted(gs.grains)
    n = len(gids)
    ea = X[:, list(NODE_FEATURE_MAP["easy_axis"])]
    ea = ea / np.linalg.norm(ea, axis=1, keepdims=True)
    th = np.arccos(np.clip(np.abs(ea[:, 2]), -1, 1))

    axes = np.zeros((n, 3))
    long_dir = np.zeros((n, 3))
    nzz = np.full(n, np.nan)
    ok = np.zeros(n, bool)
    for k, g in enumerate(gids):
        st = shape_tensor(gs.grains[g].vertices)
        if st is None:
            continue
        _, _, cov = st
        w, U = np.linalg.eigh(cov)          # ascending
        if np.any(w <= 0):
            continue
        semi = np.sqrt(5.0 * w)             # uniform ellipsoid: lambda = a^2/5
        axes[k] = semi[::-1]                # a >= b >= c
        long_dir[k] = U[:, -1]
        ok[k] = True
    if ok.sum() < 10:
        return None
    a, b, c = axes[:, 0], axes[:, 1], axes[:, 2]
    Nf = np.full((n, 3), np.nan)
    Nf[ok] = demag_factors(a[ok], b[ok], c[ok])
    # N_zz = sum_i N_i (u_i . z)^2 ; recompute eigenvectors' z-components
    for k in np.flatnonzero(ok):
        st = shape_tensor(gs.grains[gids[k]].vertices)
        w, U = np.linalg.eigh(st[2])
        uz = U[2, ::-1]                     # z-components of a, b, c directions
        nzz[k] = float(Nf[k] @ (uz ** 2))

    cols = {
        "h_SW(theta)": h_sw(th),
        "volume": X[:, I_VOL],
        "sphericity": X[:, I_SPH],
        "a/c": np.where(c > 0, a / np.maximum(c, 1e-30), np.nan),
        "b/c": np.where(c > 0, b / np.maximum(c, 1e-30), np.nan),
        "|cos(long, z)|": np.abs(long_dir[:, 2]),
        "N_zz": nzz,
        "|ea . long|": np.abs(np.einsum("ij,ij->i", ea, long_dir)),
    }
    return cols, ea, np.abs(X[:, I_SW]), X[:, I_VOL], ok


def dipolar(gs, ea, vol):
    """Stray field from saturated neighbours, and its projection on m_i."""
    gids = sorted(gs.grains)
    pos = {g: k for k, g in enumerate(gids)}
    n = len(gids)
    Hz = np.zeros(n)
    He = np.zeros(n)
    if len(gs.edges) == 0:
        return Hz, He
    C = np.array([gs.grains[g].center for g in gids], float)
    i = np.array([pos[x] for x in gs.edges[:, 0]])
    j = np.array([pos[x] for x in gs.edges[:, 1]])
    r = C[j] - C[i]
    d = np.linalg.norm(r, axis=1)
    good = d > 0
    i, j, r, d = i[good], j[good], r[good], d[good]
    rh = r / d[:, None]
    mj = ea[j]
    proj = np.einsum("ij,ij->i", mj, rh)
    H = (3.0 * proj[:, None] * rh - mj) * (vol[j] / d ** 3)[:, None]
    np.add.at(Hz, i, H[:, 2])
    np.add.at(He, i, np.einsum("ij,ij->i", H, ea[i]))
    return Hz, He


BASE = ["h_SW(theta)", "volume", "sphericity"]
NEW = ["a/c", "b/c", "|cos(long, z)|", "N_zz", "|ea . long|",
       "dip_field_z", "dip_energy"]


def main():
    print(f"Loading {CACHE} ...")
    with open(CACHE, "rb") as f:
        structures = pickle.load(f)
    usable = [gs for gs in structures if len(gs.grains) >= MIN_GRAINS]
    print(f"  {len(usable)} structures with >= {MIN_GRAINS} grains")

    marg, part = {}, {}
    Xs, ys, gs_id, solo_mask = [], [], [], []
    names = BASE + NEW
    for gi, g in enumerate(usable):
        out = node_features(g)
        if out is None:
            continue
        cols, ea, y, vol, ok = out
        Hz, He = dipolar(g, ea, vol)
        cols["dip_field_z"] = Hz
        cols["dip_energy"] = He
        if np.ptp(y) == 0:
            continue
        z = cols["h_SW(theta)"]
        for k in names:
            v = cols[k]
            m = ok & np.isfinite(v)
            if m.sum() < 20 or np.ptp(v[m]) == 0 or np.ptp(y[m]) == 0:
                continue
            marg.setdefault(k, []).append(spearmanr(v[m], y[m]).statistic)
            if k != "h_SW(theta)" and np.ptp(z[m]) > 0:
                rxy = spearmanr(v[m], y[m]).statistic
                rxz = spearmanr(v[m], z[m]).statistic
                ryz = spearmanr(y[m], z[m]).statistic
                den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
                part.setdefault(k, []).append((rxy - rxz * ryz) / den)
        M = np.column_stack([cols[k] for k in names])
        keep = ok & np.isfinite(M).all(axis=1)
        if keep.sum() < 20:
            continue
        _, cnt = np.unique(np.round(y[keep], 6), return_counts=True)
        u = np.unique(np.round(y[keep], 6))
        tie = cnt[np.searchsorted(u, np.round(y[keep], 6))]
        Xs.append(M[keep])
        ys.append((rankdata(y[keep]) - 0.5) / keep.sum())
        gs_id.append(np.full(keep.sum(), gi))
        solo_mask.append(tie == 1)

    print("\n=== within-structure Spearman with |H_sw| (mean +/- std) ===")
    print(f"  {'feature':<18s} {'marginal':>16s} {'partial | h_SW':>18s}")
    for k in sorted(marg, key=lambda k: -abs(np.nanmean(marg[k]))):
        m, s = np.nanmean(marg[k]), np.nanstd(marg[k])
        p = (f"{np.nanmean(part[k]):+.3f} +/- {np.nanstd(part[k]):.3f}"
             if k in part else "-- (control)")
        tag = "  <- NEW" if k in NEW else ""
        print(f"  {k:<18s} {m:+.3f} +/- {s:.3f} {p:>18s}{tag}")

    from sklearn.ensemble import HistGradientBoostingRegressor

    Xa = np.concatenate(Xs); ya = np.concatenate(ys)
    ga = np.concatenate(gs_id); so = np.concatenate(solo_mask)
    groups = np.unique(ga)
    test_g = set(RNG.choice(groups, max(1, len(groups) // 4), replace=False).tolist())
    te = np.array([g in test_g for g in ga])
    ib = [names.index(k) for k in BASE]
    ia = list(range(len(names)))
    print(f"\n=== gradient boosting on the within-structure rank of |H_sw| ===")
    print(f"  {len(Xa)} grains | {so.mean()*100:.1f} % switch alone")
    for label, rows in (("all grains", np.ones(len(Xa), bool)),
                        ("solitary grains only", so),
                        ("cascade grains only", ~so)):
        tr, ts = (~te) & rows, te & rows
        if ts.sum() < 500:
            continue
        line = f"  {label:<22s}"
        for tag, idx in (("base", ib), ("base+new", ia)):
            m = HistGradientBoostingRegressor(max_iter=300, random_state=0)
            m.fit(Xa[tr][:, idx], ya[tr])
            line += f"   {tag} R2 = {m.score(Xa[ts][:, idx], ya[ts]):.4f}"
        print(line + f"   (n_test={ts.sum()})")


if __name__ == "__main__":
    main()
