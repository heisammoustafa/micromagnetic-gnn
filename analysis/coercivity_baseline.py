"""Does the graph earn its place? Coercivity from pooled grain statistics alone.

Fits ordinary models on per-structure *aggregates* of the grain properties --
no topology, no message passing, no edges -- and compares them against the GNN's
own recorded metrics. If a handful of pooled numbers match the network, the
graph structure is not doing much work; if they do not, that is a concrete
argument for a graph model.

Two separate questions, kept apart because they are not comparable:

**A. Fair baseline.** Uses only what the GNN gets: geometry and material
constants (easy axes, K, Ms, volumes, grain-boundary thickness). The
Stoner-Wohlfarth astroid turns each grain's easy axis and anisotropy into an
isolated-grain switching field; the features are quantiles and moments of that,
plus a few shape aggregates. Directly comparable to the GNN.

**B. Physics probe (oracle, not a baseline).** Uses the *simulated* per-grain
switching fields, which are an output of the simulation and are not available at
prediction time. It cannot be compared with the GNN -- it answers a different
question: is the coercivity of a structure the weakest grain, the median grain,
or something else entirely?

Run from the repo root::

    python analysis/coercivity_baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from micromagnetic_gnn.checkpoints import DEFAULT_CACHE, load_coercivity_model
from micromagnetic_gnn.dataset import default_gs_data

MU0 = 4e-7 * np.pi


def astroid(psi: np.ndarray, h_k: np.ndarray) -> np.ndarray:
    """Stoner-Wohlfarth switching field of an isolated grain, in tesla."""
    c = np.abs(np.cos(psi)) ** (2.0 / 3.0)
    s = np.abs(np.sin(psi)) ** (2.0 / 3.0)
    return h_k / np.power(c + s, 1.5)


def input_features(gs, field_axis=np.array([0.0, 0.0, 1.0])) -> dict[str, float]:
    """Pooled statistics of the grain population -- inputs only, no simulation output."""
    gids = sorted(gs.grains)
    easy = np.array([gs.grains[g].params["easy_axis"] for g in gids], float)
    k1 = np.array([gs.grains[g].params["K"] for g in gids], float)
    j_s = np.array([gs.grains[g].params["Ms"] for g in gids], float)
    vol = np.array([gs.grains[g].volume for g in gids], float)
    sph = np.array([gs.grains[g].params["sphericity"] for g in gids], float)

    cos_psi = np.abs(easy @ field_axis)
    psi = np.arccos(np.clip(cos_psi, 0.0, 1.0))
    h_k = 2.0 * k1 * MU0 / np.maximum(j_s, 1e-30)
    h_sw = astroid(psi, h_k)

    w = vol / vol.sum()
    f = {
        "hsw_min": h_sw.min(),
        "hsw_q05": np.quantile(h_sw, 0.05),
        "hsw_q25": np.quantile(h_sw, 0.25),
        "hsw_median": np.median(h_sw),
        "hsw_mean": h_sw.mean(),
        "hsw_wmean": float(w @ h_sw),
        "hsw_max": h_sw.max(),
        "hsw_sd": h_sw.std(),
        "psi_mean": psi.mean(),
        "psi_sd": psi.std(),
        "cos_wmean": float(w @ cos_psi),          # remanence proxy J_r/J_s
        "logvol_mean": np.log(np.maximum(vol, 1e-30)).mean(),
        "logvol_sd": np.log(np.maximum(vol, 1e-30)).std(),
        "sphericity_mean": sph.mean(),
        "n_grains": float(len(gids)),
        "gb_t": float(gs.params.get("gb_t", np.nan)),
    }
    return f


def simulated_quantiles(gs) -> dict[str, float]:
    """Quantiles of the *simulated* per-grain switching field (oracle)."""
    y = np.array([abs(gs.grains[g].params["switching_field"]) for g in sorted(gs.grains)], float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {}
    return {f"sim_q{int(q*100):02d}": float(np.quantile(y, q))
            for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}


def collect(structures):
    X, y, sim = [], [], []
    for gs in structures:
        c = gs.params.get("coercivity")
        if c is None or not np.isfinite(c):
            continue
        try:
            f = input_features(gs)
            q = simulated_quantiles(gs)
        except (KeyError, ValueError):
            continue
        if not q or not all(np.isfinite(v) for v in f.values()):
            continue
        X.append(f)
        sim.append(q)
        y.append(abs(float(c)))
    names = list(X[0]) if X else []
    return (np.array([[d[k] for k in names] for d in X], float),
            np.asarray(y, float), sim, names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    train_s, val_s, test_s, ext_s = default_gs_data(cache_path=args.cache)
    Xtr, ytr, _, names = collect(train_s + val_s)
    Xte, yte, sim_te, _ = collect(test_s)
    Xex, yex, sim_ex, _ = collect(ext_s)
    print(f"\nstructures: train+val {len(ytr)}, test {len(yte)}, generalization {len(yex)}")
    print(f"features: {len(names)} pooled aggregates, no topology")

    gnn = load_coercivity_model().metrics

    # ---------------------------------------------------------- A. fair baseline
    models = {
        "ridge (linear)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "random forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                               random_state=0, n_jobs=-1),
    }
    print("\n" + "=" * 66)
    print("A. FAIR BASELINE -- pooled grain statistics, no graph")
    print("=" * 66)
    print(f"{'':<34}{'test R2':>12}{'gen R2':>12}")
    for label, mdl in models.items():
        mdl.fit(Xtr, ytr)
        print(f"{label:<34}{r2_score(yte, mdl.predict(Xte)):>12.3f}"
              f"{r2_score(yex, mdl.predict(Xex)):>12.3f}")
    # single-feature reference: the weakest isolated grain
    i_min = names.index("hsw_min")
    for label, idx in (("astroid min, rescaled", i_min),
                       ("astroid vol-weighted mean", names.index("hsw_wmean"))):
        lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        lin.fit(Xtr[:, [idx]], ytr)
        print(f"{label + ' (1 feature)':<34}"
              f"{r2_score(yte, lin.predict(Xte[:, [idx]])):>12.3f}"
              f"{r2_score(yex, lin.predict(Xex[:, [idx]])):>12.3f}")
    print(f"{'GNN (edge-centric, from checkpoint)':<34}"
          f"{gnn['test']['r2']:>12.3f}{gnn['ext']['r2']:>12.3f}")

    # ------------------------------------------------ B. physics probe (oracle)
    print("\n" + "=" * 66)
    print("B. PHYSICS PROBE -- which quantile of the SIMULATED grain switching")
    print("   fields is the coercivity? (oracle: uses simulation output)")
    print("=" * 66)
    print(f"{'':<20}{'test: R2':>12}{'rho':>8}{'   ':>3}{'gen: R2':>12}{'rho':>8}")
    keys = sorted(sim_te[0]) if sim_te else []
    for k in keys:
        row = f"{k:<20}"
        for sims, yy in ((sim_te, yte), (sim_ex, yex)):
            v = np.array([d[k] for d in sims], float)
            row += f"{r2_score(yy, v):>12.3f}{spearmanr(v, yy).statistic:>8.3f}   "
        print(row)

    print("\nRead A: if the pooled baseline matches the GNN on test, topology adds\n"
          "little in-distribution; the gap that matters is the generalization column.")
    print("Read B: the quantile with R2 nearest 1 identifies what the coercive\n"
          "field corresponds to physically -- weakest grain, median grain, or neither.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
