"""Prediction with uncertainty, and the figures that assess it.

The library half of the old ``coercivity_prediction_gat/infer.py``: every
evaluation script imports from here instead of importing another script.

Two sources of uncertainty are combined, and they are additive in *variance*,
so ``sigma_total**2 == sigma_aleatoric**2 + sigma_epistemic**2``:

* **aleatoric** -- the Gaussian-NLL variance head, i.e. data noise the model has
  learned. Zero for a checkpoint trained without ``predict_variance``.
* **epistemic** -- disagreement between stochastic forward passes (MC dropout)
  or between deep-ensemble members, i.e. model uncertainty.

Everything the public functions return is in **real units** (T), obtained by
inverse-transforming with the model's own label scaler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch_geometric.loader import DataLoader

from micromagnetic_gnn.models import GraphDataScaler
from micromagnetic_gnn.plotstyle import (
    C_ALE,
    C_EPI,
    C_REF,
    C_TRAIN,
    STYLE,
    width,
)

__all__ = [
    "mc_forward",
    "true_labels",
    "mc_predict",
    "mc_predict_decomposed",
    "ensemble_predict",
    "calibration_curve",
    "sigma_reliability",
    "summarise",
    "parity_plot",
    "plot_diagnostics",
    "plot_decomposition",
]


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #

def mc_forward(model, loader: DataLoader, n_samples: int, device: Any):
    """Run ``n_samples`` forward passes over *loader*.

    Dropout stays active if the caller left the model in ``train()`` mode, which
    is what makes the passes differ.

    Returns
    -------
    means, aleatoric_vars : np.ndarray
        Both in **scaled** label space with shape ``(n_samples, N, 1)``.
        ``aleatoric_vars`` is all zeros when the model has no variance head.
    """
    predict_var = getattr(model, "predict_variance", False)
    means, aleatoric_vars = [], []
    with torch.no_grad():
        for _ in range(n_samples):
            batch_means, batch_vars = [], []
            for batch in loader:
                out = model(batch.to(device))
                if predict_var:
                    mean, var = out
                else:
                    mean, var = out, torch.zeros_like(out)
                batch_means.append(mean.cpu())
                batch_vars.append(var.cpu())
            means.append(np.asarray(torch.cat(batch_means)))
            aleatoric_vars.append(np.asarray(torch.cat(batch_vars)))
    return np.asarray(means), np.asarray(aleatoric_vars)


def true_labels(loader: DataLoader, scaler: GraphDataScaler) -> np.ndarray:
    """Ground-truth labels of *loader*, inverse-transformed to real units."""
    y_true = torch.cat([b.y for b in loader])
    return np.asarray(scaler.label_scaler.inverse_transform(y_true))


def mc_predict_decomposed(
    model, loader: DataLoader, scaler: GraphDataScaler, n_samples: int, device: Any
):
    """MC-dropout prediction with the uncertainty split into its two sources.

    Returns ``(y_mean, std_aleatoric, std_epistemic, std_total, y_true)``, all in
    real units:

    * **aleatoric** = mean over passes of the predicted variance;
    * **epistemic** = variance over passes of the predicted means.
    """
    label_std = float(scaler.label_scaler.std.reshape(-1)[0])
    means, aleatoric_vars = mc_forward(model, loader, n_samples, device)

    mean_scaled = means.mean(0)                      # (N, 1)
    epistemic_var = means.var(0)                     # (N, 1), scaled
    aleatoric_var = aleatoric_vars.mean(0)           # (N, 1), scaled

    y_mean = np.asarray(
        scaler.label_scaler.inverse_transform(torch.tensor(mean_scaled))
    )
    # A standard deviation scales linearly with the label scale.
    std_aleatoric = np.sqrt(aleatoric_var) * label_std
    std_epistemic = np.sqrt(epistemic_var) * label_std
    std_total = np.sqrt(aleatoric_var + epistemic_var) * label_std

    return y_mean, std_aleatoric, std_epistemic, std_total, true_labels(loader, scaler)


def mc_predict(
    model, loader: DataLoader, scaler: GraphDataScaler, n_samples: int, device: Any
):
    """Predict mean and *total* predictive std plus ``y_true`` (real units).

    Thin wrapper over :func:`mc_predict_decomposed` returning only the combined
    uncertainty.  With no variance head this reduces to MC dropout alone.
    """
    y_mean, _, _, y_std, y_true = mc_predict_decomposed(
        model, loader, scaler, n_samples, device
    )
    return y_mean, y_std, y_true


def ensemble_predict(
    models, loader: DataLoader, scaler: GraphDataScaler, n_samples: int, device: Any
):
    """Deep-ensemble prediction with decomposed uncertainty (real units).

    Follows Lakshminarayanan et al. (2017): the prediction is the mean over
    members, the aleatoric variance is the mean of the members' variance heads,
    and the epistemic variance is the members' disagreement -- plus, when
    ``n_samples > 1``, the within-member MC-dropout spread.  Unlike MC dropout,
    member disagreement grows on out-of-distribution inputs, so this epistemic
    term does flag the generalization split.

    Returns ``(y_mean, std_aleatoric, std_epistemic, std_total, y_true)``.
    """
    label_std = float(scaler.label_scaler.std.reshape(-1)[0])

    member_means, member_avars, member_mcvars = [], [], []
    for model in models:
        model.train() if n_samples > 1 else model.eval()
        means, avars = mc_forward(model, loader, n_samples, device)  # (S, N, 1)
        member_means.append(means.mean(0))
        member_avars.append(avars.mean(0))
        member_mcvars.append(means.var(0))
    stacked = np.asarray(member_means)                              # (M, N, 1)

    mean_scaled = stacked.mean(0)                                   # (N, 1)
    aleatoric_var = np.mean(member_avars, axis=0)
    epistemic_var = stacked.var(0) + np.mean(member_mcvars, axis=0)

    y_mean = np.asarray(
        scaler.label_scaler.inverse_transform(torch.tensor(mean_scaled))
    )
    std_aleatoric = np.sqrt(aleatoric_var) * label_std
    std_epistemic = np.sqrt(epistemic_var) * label_std
    std_total = np.sqrt(aleatoric_var + epistemic_var) * label_std

    return y_mean, std_aleatoric, std_epistemic, std_total, true_labels(loader, scaler)


# --------------------------------------------------------------------------- #
# calibration metrics
# --------------------------------------------------------------------------- #

def calibration_curve(z: np.ndarray, n_levels: int = 20):
    """Observed vs expected coverage of central Gaussian intervals.

    Parameters
    ----------
    z : np.ndarray
        Standardised residuals ``(y_true - mu) / sigma``.
    n_levels : int
        Number of nominal coverage levels in ``(0, 1)``.

    Returns
    -------
    expected, observed : np.ndarray
        Nominal coverage levels and the empirically observed coverage.
    """
    expected = np.linspace(0.0, 1.0, n_levels + 1)[1:]      # (0, 1]
    # A two-sided central interval of nominal coverage q spans +-z_half.
    z_half = norm.ppf(0.5 + expected / 2.0)                 # half-width in sigmas
    observed = np.array([np.mean(np.abs(z) <= zh) for zh in z_half])
    return expected, observed


def sigma_reliability(residuals: np.ndarray, sigma: np.ndarray, n_bins: int):
    """Bin by predicted sigma; return per-bin (mean predicted sigma, RMS residual).

    Uses equal-count (quantile) bins so each point carries similar weight.
    """
    order = np.argsort(sigma)
    sig_sorted = sigma[order]
    res_sorted = residuals[order]
    bins = np.array_split(np.arange(sig_sorted.size), n_bins)
    pred_sigma, obs_rms, counts = [], [], []
    for b in bins:
        if b.size == 0:
            continue
        pred_sigma.append(float(sig_sorted[b].mean()))
        obs_rms.append(float(np.sqrt(np.mean(res_sorted[b] ** 2))))
        counts.append(int(b.size))
    return np.array(pred_sigma), np.array(obs_rms), np.array(counts)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """``np.trapezoid``, falling back to the pre-2.0 ``np.trapz`` spelling."""
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x))


def summarise(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> dict:
    """Scalar UQ metrics."""
    resid = (y_true - y_mean).reshape(-1)
    sigma = np.clip(y_std.reshape(-1), 1e-12, None)
    z = resid / sigma

    expected, observed = calibration_curve(z)
    ece = float(np.mean(np.abs(observed - expected)))
    miscal_area = _trapezoid(np.abs(observed - expected), expected)

    # Gaussian NLL per point (lower is better; rewards sharp *and* calibrated).
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * z**2))

    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mean_sigma": float(sigma.mean()),              # sharpness
        "cov_1sigma": float(np.mean(np.abs(z) <= 1)),   # target 0.683
        "cov_2sigma": float(np.mean(np.abs(z) <= 2)),   # target 0.954
        "rms_z": float(np.sqrt(np.mean(z**2))),         # 1.0 = calibrated scale
        "sigma_scale": float(np.sqrt(np.mean(z**2))),   # multiply sigma by this
        "nll": nll,
        "ece": ece,
        "miscalibration_area": miscal_area,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"  saved {out_path} (+ .png)")


def parity_plot(y_true, y_mean, y_std, out_path: Path, n_std: float = 1.0) -> None:
    """True vs predicted with ``+-n_std`` error bars."""
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(width(0.5), width(0.5)))
    lo, hi = float(y_true.min()), float(y_true.max())
    ax.errorbar(
        y_true[:, 0], y_mean[:, 0], yerr=n_std * y_std[:, 0],
        linestyle="None", ecolor=C_REF, elinewidth=0.7, capsize=1.2,
        fmt="none", alpha=0.5, label=rf"$\pm{n_std:g}\sigma$",
    )
    ax.scatter(y_true[:, 0], y_mean[:, 0], s=9, color=C_TRAIN, linewidths=0, zorder=3)
    ax.plot([lo, hi], [lo, hi], color=C_REF, ls="--", lw=0.9, zorder=4)
    ax.legend(loc="upper left")
    ax.set_xlabel(r"simulated $H_\mathrm{C}$  (T)")
    ax.set_ylabel(r"predicted $H_\mathrm{C}$  (T)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, out_path)


def plot_diagnostics(y_true, y_mean, y_std, out_path: Path, n_bins: int = 10) -> dict:
    """Three-panel calibration figure; returns the scalar metrics.

    1. **Reliability curve** -- observed vs nominal coverage of the predicted
       central intervals. A calibrated model lies on the diagonal.
    2. **sigma vs residual spread** -- points binned by predicted sigma, mean
       predicted sigma against observed RMS residual. Ideal points lie on y = x.
    3. **PIT histogram** -- ``Phi((y-mu)/sigma)`` is Uniform(0,1) iff the
       predictive distribution is calibrated, so flat is good.
    """
    resid = (y_true - y_mean).reshape(-1)
    sigma = np.clip(y_std.reshape(-1), 1e-12, None)
    z = resid / sigma

    stats = summarise(y_true, y_mean, y_std)
    expected, observed = calibration_curve(z)
    pred_sigma, obs_rms, _ = sigma_reliability(np.abs(resid), sigma, n_bins)

    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(width(1.0), 2.35))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=C_REF, ls="--", lw=0.9, label="ideal")
    ax.plot(expected, observed, "o-", ms=3, color=C_TRAIN)
    ax.fill_between(expected, expected, observed, alpha=0.15, color=C_TRAIN)
    ax.set_xlabel("expected coverage")
    ax.set_ylabel("observed coverage")
    ax.set_title(f"reliability (ECE = {stats['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    ax = axes[1]
    hi = float(max(pred_sigma.max(), obs_rms.max())) * 1.05
    ax.plot([0, hi], [0, hi], color=C_REF, ls="--", lw=0.9, label="ideal")
    ax.scatter(pred_sigma, obs_rms, s=18, color=C_TRAIN, linewidths=0, zorder=3)
    ax.set_xlabel(r"predicted $\sigma$ (bin mean, T)")
    ax.set_ylabel("observed RMS residual (T)")
    ax.set_title(r"$\sigma$ vs residual spread")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.hist(norm.cdf(z), bins=15, range=(0, 1), density=True, color=C_TRAIN, alpha=0.8)
    ax.axhline(1.0, color=C_REF, ls="--", lw=0.9, label="uniform (ideal)")
    ax.set_xlabel(r"PIT $= \Phi((y-\mu)/\sigma)$")
    ax.set_ylabel("density")
    ax.set_title(rf"PIT: cov$_{{1\sigma}}$ = {stats['cov_1sigma']:.2f}, "
                 rf"$s$ = {stats['sigma_scale']:.2f}")
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, out_path)
    return stats


def plot_decomposition(y_true, std_ale, std_epi, out_path: Path) -> dict:
    """Three-panel aleatoric-vs-epistemic figure; returns the scalar metrics.

    1. **Stacked variance per sample** (sorted by total) -- which source
       dominates, and how that varies across samples.
    2. **Aleatoric vs epistemic** (log-log, one point per graph) with a ``y = x``
       reference -- the typical magnitude of each and their balance.
    3. **Both vs the true target** -- heteroscedasticity in ``h_c``.
    """
    y_true = y_true.reshape(-1)
    std_ale = std_ale.reshape(-1)
    std_epi = std_epi.reshape(-1)
    var_ale = std_ale**2
    var_epi = std_epi**2
    var_tot = var_ale + var_epi

    order = np.argsort(var_tot)
    idx = np.arange(y_true.size)

    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(width(1.0), 2.35))

    ax = axes[0]
    ax.fill_between(idx, 0, var_ale[order], color=C_ALE, alpha=0.85, label="aleatoric")
    ax.fill_between(idx, var_ale[order], var_tot[order], color=C_EPI, alpha=0.85,
                    label="epistemic")
    ax.set_xlabel("sample (sorted by total variance)")
    ax.set_ylabel(r"variance $\sigma^2$  (T$^2$)")
    ax.set_title("variance decomposition")
    ax.set_xlim(0, y_true.size - 1)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")

    ax = axes[1]
    lo = float(min(std_ale.min(), std_epi.min()))
    hi = float(max(std_ale.max(), std_epi.max()))
    lo = max(lo, hi * 1e-4)  # keep the log axis finite
    ax.plot([lo, hi], [lo, hi], color=C_REF, ls="--", lw=0.9,
            label=r"$\sigma_\mathrm{ale}=\sigma_\mathrm{epi}$")
    ax.scatter(std_ale, std_epi, s=10, alpha=0.6, color=C_ALE, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"aleatoric $\sigma$  (T)")
    ax.set_ylabel(r"epistemic $\sigma$  (T)")
    ax.set_title("per-sample balance")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.scatter(y_true, std_ale, s=10, alpha=0.6, color=C_ALE, linewidths=0,
               label="aleatoric")
    ax.scatter(y_true, std_epi, s=10, alpha=0.6, color=C_EPI, linewidths=0,
               label="epistemic")
    ax.set_xlabel(r"simulated $H_\mathrm{C}$  (T)")
    ax.set_ylabel(r"$\sigma$  (T)")
    ax.set_title("uncertainty vs target")
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save(fig, out_path)

    # Fraction of total predictive variance explained by each source (mean).
    frac_epi = float(np.mean(var_epi / np.clip(var_tot, 1e-12, None)))
    return {
        "mean_std_aleatoric": float(std_ale.mean()),
        "mean_std_epistemic": float(std_epi.mean()),
        "median_std_aleatoric": float(np.median(std_ale)),
        "median_std_epistemic": float(np.median(std_epi)),
        "epistemic_var_fraction": frac_epi,   # 0 => all aleatoric, 1 => all epistemic
    }
