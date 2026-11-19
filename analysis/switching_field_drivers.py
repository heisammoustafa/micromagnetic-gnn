"""What drives the per-grain switching field, beyond the marginal correlation.

Three layers the plain rank-correlation matrix cannot reach:

1. A physics-informed candidate instead of raw geometry. Stoner--Wohlfarth
   reversal gives the reduced switching field of an isolated uniaxial grain,
       h_SW(theta) = 1 / (cos^(2/3) theta + sin^(2/3) theta)^(3/2),
   which is non-monotone (minimum 0.5 at 45 deg). Correlating against h_SW
   rather than theta removes that non-monotonicity, which neither Spearman nor
   Pearson can represent.

2. Partial correlations. Almost every descriptor correlates with theta, so its
   marginal correlation with H_sw is largely inherited. The partial Spearman
   coefficient controlling for h_SW isolates what a descriptor adds on top of
   single-grain physics -- i.e. what the *neighbourhood* contributes.

3. A nonlinear multivariate model. The target is rank-normalised within each
   structure (so the between-structure scale spread cannot leak in) and pooled;
   permutation importance on a structure-disjoint test split then ranks the
   descriptors while accounting for interactions.

Neighbour aggregates are weighted by shared face area where a coupling strength
is implied, since exchange through a grain boundary scales with contact area.

Run from the repo root::

    python analysis/switching_field_drivers.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore", message="An input array is constant")

from micromagnetic_gnn.data_pipeline import (
    EDGE_FEATURE_MAP,
    NODE_FEATURE_MAP,
    _extract_edge_features,
    _extract_node_features,
    compute_int_face_frac,
)

CACHE = Path("data_cache.pkl")
MIN_GRAINS = 50
RNG = np.random.default_rng(0)

I_SW = NODE_FEATURE_MAP["switching_field"][0]
I_VOL = NODE_FEATURE_MAP["volume"][0]
I_SPH = NODE_FEATURE_MAP["sphericity"][0]
I_AREA = EDGE_FEATURE_MAP["face_area"][0]
I_DIP = EDGE_FEATURE_MAP["dipole_coupling"][0]


def h_sw(theta_rad):
    """Stoner-Wohlfarth reduced switching field of an isolated uniaxial grain."""
    c = np.abs(np.cos(theta_rad)) ** (2 / 3) + np.abs(np.sin(theta_rad)) ** (2 / 3)
    return c ** (-1.5)


def features(gs):
    """Feature table for one structure; all columns are model-legitimate
    (geometry and easy axes only -- no switching information)."""
    X = _extract_node_features(gs).numpy()
    gids = sorted(gs.grains)
    pos = {g: k for k, g in enumerate(gids)}
    n = len(gids)
    if len(gs.edges) == 0:
        return None

    ea = X[:, list(NODE_FEATURE_MAP["easy_axis"])]
    ea = ea / np.linalg.norm(ea, axis=1, keepdims=True)
    th = np.arccos(np.clip(np.abs(ea[:, 2]), -1, 1))
    hsw = h_sw(th)

    E = _extract_edge_features(gs).numpy()
    src = np.array([pos[a] for a in gs.edges[:, 0]])
    dst = np.array([pos[b] for b in gs.edges[:, 1]])
    area = E[:, I_AREA]
    align = np.abs(E[:, I_DIP])                      # |e_i . e_j|
    dtheta = np.abs(th[dst] - th[src])

    def wsum(v, w):
        out = np.zeros(n)
        np.add.at(out, src, v * w)
        return out

    wtot = np.zeros(n)
    np.add.at(wtot, src, area)
    wtot = np.maximum(wtot, 1e-30)
    deg = np.zeros(n)
    np.add.at(deg, src, 1.0)

    nb_min = np.full(n, np.inf)
    nb_max = np.full(n, -np.inf)
    np.minimum.at(nb_min, src, hsw[dst])
    np.maximum.at(nb_max, src, th[dst])
    nb_min[~np.isfinite(nb_min)] = np.nan
    nb_max[~np.isfinite(nb_max)] = np.nan

    cols = {
        "theta": np.degrees(th),
        "h_SW(theta)": hsw,
        "nb align (area-w.)": wsum(align, area) / wtot,
        "nb |dtheta| (area-w.)": np.degrees(wsum(dtheta, area) / wtot),
        "nb h_SW mean (area-w.)": wsum(hsw[dst], area) / wtot,
        "nb h_SW min": nb_min,
        "nb theta max": np.degrees(nb_max),
        "contact area / V^2/3": wtot / np.maximum(X[:, I_VOL], 1e-30) ** (2 / 3),
        "volume": X[:, I_VOL],
        "sphericity": X[:, I_SPH],
        "n neighbours": deg,
        "face_frac": np.array([compute_int_face_frac(gs, g) for g in gids]),
    }
    y = np.abs(X[:, I_SW])
    return cols, y


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with the influence of z removed."""
    rxy = spearmanr(x, y).statistic
    rxz = spearmanr(x, z).statistic
    ryz = spearmanr(y, z).statistic
    den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    return (rxy - rxz * ryz) / den


def main():
    print(f"Loading {CACHE} ...")
    with open(CACHE, "rb") as f:
        structures = pickle.load(f)
    usable = [gs for gs in structures if len(gs.grains) >= MIN_GRAINS]
    print(f"  {len(usable)} structures with >= {MIN_GRAINS} grains")

    marg, part = {}, {}
    pooled_X, pooled_y, pooled_g = [], [], []
    names = None
    for gi, gs in enumerate(usable):
        out = features(gs)
        if out is None:
            continue
        cols, y = out
        if np.ptp(y) == 0:
            continue
        names = list(cols)
        z = cols["h_SW(theta)"]
        for k, v in cols.items():
            if not np.isfinite(v).all() or np.ptp(v) == 0:
                continue
            marg.setdefault(k, []).append(spearmanr(v, y).statistic)
            if k != "h_SW(theta)" and np.ptp(z) > 0:
                part.setdefault(k, []).append(partial_spearman(v, y, z))
        M = np.column_stack([cols[k] for k in names])
        if np.isfinite(M).all():
            pooled_X.append(M)
            # rank-normalise the target within the structure -> [0, 1]
            pooled_y.append((rankdata(y) - 0.5) / len(y))
            pooled_g.append(np.full(len(y), gi))

    print("\n=== within-structure Spearman with |H_sw| (mean +/- std) ===")
    print(f"  {'descriptor':<24s} {'marginal':>16s} {'partial | h_SW':>18s}")
    order = sorted(marg, key=lambda k: -abs(np.nanmean(marg[k])))
    for k in order:
        m = np.nanmean(marg[k]); s = np.nanstd(marg[k])
        if k in part:
            pm, ps = np.nanmean(part[k]), np.nanstd(part[k])
            ptxt = f"{pm:+.3f} +/- {ps:.3f}"
        else:
            ptxt = "-- (control)"
        print(f"  {k:<24s} {m:+.3f} +/- {s:.3f} {ptxt:>18s}")

    # ---- nonlinear multivariate check ---------------------------------------
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    Xa = np.concatenate(pooled_X)
    ya = np.concatenate(pooled_y)
    ga = np.concatenate(pooled_g)
    groups = np.unique(ga)
    test_g = set(RNG.choice(groups, max(1, len(groups) // 4), replace=False).tolist())
    te = np.array([g in test_g for g in ga])
    print(f"\n=== nonlinear model on the within-structure rank of |H_sw| ===")
    print(f"  {len(Xa)} grains, {len(groups)} structures "
          f"({(~te).sum()} train / {te.sum()} test grains, structure-disjoint)")

    mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                        random_state=0)
    mdl.fit(Xa[~te], ya[~te])
    r2 = mdl.score(Xa[te], ya[te])
    print(f"  test R2 on the rank target = {r2:.4f}")

    sub = RNG.choice(np.flatnonzero(te), min(40000, te.sum()), replace=False)
    imp = permutation_importance(mdl, Xa[sub], ya[sub], n_repeats=5,
                                 random_state=0, n_jobs=4)
    print(f"  {'descriptor':<24s} {'permutation importance':>24s}")
    for i in np.argsort(-imp.importances_mean):
        print(f"  {names[i]:<24s} {imp.importances_mean[i]:>16.4f}"
              f" +/- {imp.importances_std[i]:.4f}")

    # theta / h_SW alone, for reference
    for solo in ("theta", "h_SW(theta)"):
        j = names.index(solo)
        m2 = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        m2.fit(Xa[~te][:, [j]], ya[~te])
        print(f"  [{solo} alone] test R2 = {m2.score(Xa[te][:, [j]], ya[te]):.4f}")


if __name__ == "__main__":
    main()
