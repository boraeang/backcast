"""Tier 2: Regime-switching multivariate Gaussian returns.

DGP
---
    s_t ~ Markov(P)                  (hidden regime state)
    R_t | s_t = k ~ N(mu_k, Sigma_k)  (regime-conditional MVN)

Regime parameters are constructed by applying per-regime multipliers to the
base Tier 1 parameters:

- vol_multipliers : scale the daily standard deviations
- corr_adjustments : additive shift of the off-diagonal correlations
- mean_adjustments : multiplicative scaling of the base mean

Validation criteria (checked in tests/test_tier2.py)
---------------------------------------------------------
- Empirical transition frequencies within a few standard errors of P.
- Regime-conditional sample mean and covariance match regime parameters.
- Each regime covariance is PSD.
- In the adversarial variant, the non-calm regime never appears in the
  overlap period.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.random as npr
import pandas as pd

from synthgen.calendar import generate_business_days
from synthgen.config import (
    SyntheticConfig,
    Tier2Config,
    build_asset_profiles,
    get_short_start_indices,
)
from synthgen.correlation import build_covariance, is_psd, nearest_psd
from synthgen.masking import apply_masking

logger = logging.getLogger(__name__)

_ANNUAL_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Default regime-parameter generators
# ---------------------------------------------------------------------------

def _default_transition_matrix(n_regimes: int) -> np.ndarray:
    """Return a plausible row-stochastic transition matrix.

    For n_regimes==2, uses the spec's recommended P=[[0.98,0.02],[0.05,0.95]],
    giving mean durations of ~50 (calm) and ~20 (crisis) days.

    For n_regimes!=2, builds a high-persistence matrix with diagonal 0.95 and
    the remaining mass split evenly across off-diagonal entries.
    """
    if n_regimes == 1:
        return np.array([[1.0]])

    if n_regimes == 2:
        return np.array([[0.98, 0.02], [0.05, 0.95]])

    diag = 0.95
    off_mass = (1.0 - diag) / (n_regimes - 1)
    P = np.full((n_regimes, n_regimes), off_mass)
    np.fill_diagonal(P, diag)
    return P


def _default_vol_multipliers(n_regimes: int) -> list[float]:
    if n_regimes == 1:
        return [1.0]
    if n_regimes == 2:
        return [1.0, 2.5]
    return list(np.linspace(1.0, 2.5, n_regimes))


def _default_corr_adjustments(n_regimes: int) -> list[float]:
    if n_regimes == 1:
        return [0.0]
    if n_regimes == 2:
        return [0.0, 0.3]
    return list(np.linspace(0.0, 0.3, n_regimes))


def _default_mean_adjustments(n_regimes: int) -> list[float]:
    if n_regimes == 1:
        return [1.0]
    if n_regimes == 2:
        return [1.0, -0.5]
    return list(np.linspace(1.0, -0.5, n_regimes))


def _resolve_tier2_config(raw: Tier2Config | None) -> Tier2Config:
    """Return a Tier2Config with all ``None`` fields filled with defaults."""
    cfg = raw if raw is not None else Tier2Config()
    K = cfg.n_regimes

    if cfg.transition_matrix is None:
        cfg.transition_matrix = _default_transition_matrix(K).tolist()
    if cfg.regime_vol_multipliers is None:
        cfg.regime_vol_multipliers = _default_vol_multipliers(K)
    if cfg.regime_corr_adjustments is None:
        cfg.regime_corr_adjustments = _default_corr_adjustments(K)
    if cfg.regime_mean_adjustments is None:
        cfg.regime_mean_adjustments = _default_mean_adjustments(K)

    _validate_tier2_config(cfg)
    return cfg


def _validate_tier2_config(cfg: Tier2Config) -> None:
    K = cfg.n_regimes
    P = np.asarray(cfg.transition_matrix, dtype=np.float64)
    if P.shape != (K, K):
        raise ValueError(
            f"transition_matrix shape {P.shape} does not match n_regimes={K}"
        )
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("transition_matrix rows must sum to 1")
    if (P < 0).any():
        raise ValueError("transition_matrix must be non-negative")
    for name, seq in [
        ("regime_vol_multipliers", cfg.regime_vol_multipliers),
        ("regime_corr_adjustments", cfg.regime_corr_adjustments),
        ("regime_mean_adjustments", cfg.regime_mean_adjustments),
    ]:
        if len(seq) != K:
            raise ValueError(f"{name} length {len(seq)} != n_regimes={K}")


# ---------------------------------------------------------------------------
# Per-regime parameter construction
# ---------------------------------------------------------------------------

def _build_regime_params(
    mu_base: np.ndarray,
    vols_daily: np.ndarray,
    corr_base: np.ndarray,
    tier2: Tier2Config,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build per-regime (mu, Sigma, correlation) triples.

    Parameters
    ----------
    mu_base : np.ndarray, shape (N,)
        Base daily mean vector.
    vols_daily : np.ndarray, shape (N,)
        Base daily volatilities.
    corr_base : np.ndarray, shape (N, N)
        Base correlation matrix.
    tier2 : Tier2Config
        Fully-resolved Tier 2 configuration.

    Returns
    -------
    mus : list of np.ndarray
        Per-regime mean vectors.
    sigmas : list of np.ndarray
        Per-regime covariance matrices (each PSD).
    corrs : list of np.ndarray
        Per-regime correlation matrices (post-PSD-projection).
    """
    mus: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    corrs: list[np.ndarray] = []

    off_diag_mask = ~np.eye(corr_base.shape[0], dtype=bool)

    for k in range(tier2.n_regimes):
        # Adjusted mean
        mu_k = mu_base * tier2.regime_mean_adjustments[k]

        # Adjusted correlation: add adjustment to off-diagonal, clip, symmetrise
        adj_corr = corr_base.copy()
        adj_corr[off_diag_mask] += tier2.regime_corr_adjustments[k]
        adj_corr = np.clip(adj_corr, -0.99, 0.99)
        np.fill_diagonal(adj_corr, 1.0)
        adj_corr = (adj_corr + adj_corr.T) / 2.0

        # Scaled volatilities
        vols_k = vols_daily * tier2.regime_vol_multipliers[k]

        # Build covariance and project to PSD if needed
        sigma_k = adj_corr * np.outer(vols_k, vols_k)
        if not is_psd(sigma_k):
            sigma_k = nearest_psd(sigma_k)

        # Re-derive the actual correlation post-projection for ground truth
        d = np.sqrt(np.diag(sigma_k))
        corr_k = sigma_k / np.outer(d, d)
        np.fill_diagonal(corr_k, 1.0)

        mus.append(mu_k.astype(np.float64))
        sigmas.append(sigma_k.astype(np.float64))
        corrs.append(corr_k.astype(np.float64))

    return mus, sigmas, corrs


# ---------------------------------------------------------------------------
# Markov chain simulation
# ---------------------------------------------------------------------------

def _generate_regime_sequence(
    P: np.ndarray, T: int, rng: npr.Generator, initial_state: int = 0
) -> np.ndarray:
    """Sample a discrete-time Markov chain of length T from transition matrix P.

    Parameters
    ----------
    P : np.ndarray, shape (K, K)
        Row-stochastic transition matrix.
    T : int
        Sequence length.
    rng : numpy.random.Generator
        Seeded generator.
    initial_state : int
        Starting regime (defaults to the most persistent one, index 0).

    Returns
    -------
    np.ndarray, shape (T,), dtype int64
        Sampled regime labels.
    """
    K = P.shape[0]
    seq = np.zeros(T, dtype=np.int64)
    seq[0] = initial_state
    # Pre-draw U(0,1) noise vector for vectorised inverse CDF
    u = rng.random(T)
    # Row-wise cumulative sums for inverse-CDF sampling
    P_cum = np.cumsum(P, axis=1)
    for t in range(1, T):
        seq[t] = int(np.searchsorted(P_cum[seq[t - 1]], u[t]))
        if seq[t] >= K:
            seq[t] = K - 1
    return seq


def _compute_regime_durations(seq: np.ndarray, n_regimes: int) -> dict[str, Any]:
    """Summarise consecutive-run lengths per regime.

    Returns
    -------
    dict
        Keys are ``"regime_{k}"``; each value has ``mean``, ``median``,
        ``std``, ``count``, ``total_days``.
    """
    if len(seq) == 0:
        return {}

    change = np.diff(seq) != 0
    run_starts = np.concatenate([[0], np.where(change)[0] + 1])
    run_ends = np.concatenate([np.where(change)[0] + 1, [len(seq)]])
    run_lengths = (run_ends - run_starts).astype(np.int64)
    run_regimes = seq[run_starts]

    out: dict[str, Any] = {}
    for k in range(n_regimes):
        k_lengths = run_lengths[run_regimes == k]
        if len(k_lengths) == 0:
            out[f"regime_{k}"] = {
                "mean": 0.0, "median": 0.0, "std": 0.0,
                "count": 0, "total_days": 0,
            }
        else:
            out[f"regime_{k}"] = {
                "mean": float(k_lengths.mean()),
                "median": float(np.median(k_lengths)),
                "std": float(k_lengths.std(ddof=0)),
                "count": int(len(k_lengths)),
                "total_days": int(k_lengths.sum()),
            }
    return out


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_tier2(
    cfg: SyntheticConfig,
    rng: npr.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate regime-switching multivariate Gaussian returns (Tier 2).

    Parameters
    ----------
    cfg : SyntheticConfig
        Generator configuration.  ``cfg.tier`` should be 2.  ``cfg.tier2_config``
        is used; when None, defaults from :class:`Tier2Config` are applied.
    rng : numpy.random.Generator or None
        Seeded generator.  Created from ``cfg.seed`` when None.

    Returns
    -------
    returns_masked : pd.DataFrame
        Returns matrix with NaN for short-history assets before their start dates.
    returns_complete : pd.DataFrame
        Full returns matrix without masking.
    ground_truth : dict
        True DGP parameters.  Includes ``regime_labels``, ``transition_matrix``,
        ``regime_params``, ``regime_durations``, ``adversarial``,
        ``regime_counts_overlap``, and ``regime_counts_backcast`` on top of
        the Tier 1 fields.
    """
    if rng is None:
        rng = npr.default_rng(cfg.seed)

    tier2 = _resolve_tier2_config(cfg.tier2_config)

    # --- Asset profiles ---------------------------------------------------
    profiles = build_asset_profiles(cfg)
    asset_names = list(profiles.keys())
    long_assets = asset_names[: cfg.n_long_assets]
    short_assets = asset_names[cfg.n_long_assets :]

    logger.info(
        "Tier 2: %d regimes, %d assets, T=%d, adversarial=%s",
        tier2.n_regimes, len(asset_names), cfg.t_total, tier2.adversarial,
    )

    # --- Base daily mean / vol / covariance -------------------------------
    mu_base = np.array(
        [(p[0] / 100.0) / _ANNUAL_TRADING_DAYS for p in profiles.values()],
        dtype=np.float64,
    )
    vols_daily = np.array(
        [(p[1] / 100.0) / np.sqrt(_ANNUAL_TRADING_DAYS) for p in profiles.values()],
        dtype=np.float64,
    )
    cov_result = build_covariance(
        asset_names=asset_names,
        vols_daily=vols_daily,
        method=cfg.correlation_method,
        n_factors=cfg.n_factors,
        rng=rng,
    )
    base_corr = cov_result.corr

    # --- Per-regime parameters -------------------------------------------
    mus, sigmas, corrs = _build_regime_params(mu_base, vols_daily, base_corr, tier2)

    # --- Regime sequence --------------------------------------------------
    P = np.asarray(tier2.transition_matrix, dtype=np.float64)
    seq = _generate_regime_sequence(P, cfg.t_total, rng, initial_state=0)

    # --- Missingness indices ---------------------------------------------
    start_indices = get_short_start_indices(cfg, short_assets) if short_assets else {}
    min_start = min(start_indices.values()) if start_indices else cfg.t_total

    # --- Adversarial override: force overlap period into calm regime -----
    if tier2.adversarial and min_start < cfg.t_total:
        seq[min_start:] = 0
        logger.info(
            "Adversarial: forced overlap period (rows %d:%d) into regime 0",
            min_start, cfg.t_total,
        )

    # --- Draw returns per regime ------------------------------------------
    returns_arr = np.zeros((cfg.t_total, len(asset_names)), dtype=np.float64)
    for k in range(tier2.n_regimes):
        mask = seq == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        returns_arr[mask] = rng.multivariate_normal(mus[k], sigmas[k], size=n_k)

    # --- Dates + DataFrame ------------------------------------------------
    dates = generate_business_days(cfg.start_date, cfg.t_total)
    returns_complete = pd.DataFrame(
        returns_arr, index=dates, columns=asset_names, dtype=np.float64
    )

    # --- Apply masking ----------------------------------------------------
    returns_masked, mask_meta = apply_masking(returns_complete, start_indices)

    # --- Regime counts split by period ------------------------------------
    if short_assets:
        backcast_seq = seq[:min_start]
        overlap_seq = seq[min_start:]
    else:
        backcast_seq = np.array([], dtype=np.int64)
        overlap_seq = seq

    regime_counts_backcast = {
        f"regime_{k}": int((backcast_seq == k).sum()) for k in range(tier2.n_regimes)
    }
    regime_counts_overlap = {
        f"regime_{k}": int((overlap_seq == k).sum()) for k in range(tier2.n_regimes)
    }

    # --- Ground truth ----------------------------------------------------
    fc = cov_result.factor_components

    regime_params: dict[str, Any] = {}
    for k in range(tier2.n_regimes):
        regime_params[f"regime_{k}"] = {
            "mu": mus[k].tolist(),
            "sigma": sigmas[k].tolist(),
            "correlation": corrs[k].tolist(),
            "vol_multiplier": float(tier2.regime_vol_multipliers[k]),
            "corr_adjustment": float(tier2.regime_corr_adjustments[k]),
            "mean_adjustment": float(tier2.regime_mean_adjustments[k]),
        }

    # Marginal long-run distribution over regimes (from P^T stationary distribution)
    # Useful for validating that backcast/overlap counts aren't bizarrely skewed.
    stationary = _stationary_distribution(P)

    ground_truth: dict[str, Any] = {
        "tier": 2,
        "asset_names": asset_names,
        "long_assets": long_assets,
        "short_assets": short_assets,
        "short_asset_start_indices": {k: int(v) for k, v in start_indices.items()},
        "n_observations": int(cfg.t_total),
        # Base (unconditional-if-no-regimes) references
        "mu": mu_base.tolist(),
        "sigma": cov_result.sigma_daily.tolist(),
        "correlation": base_corr.tolist(),
        "mu_daily": mu_base.tolist(),
        "sigma_daily": cov_result.sigma_daily.tolist(),
        # Factor-model components (base covariance)
        "factor_loadings": fc.B.tolist() if fc is not None else None,
        "factor_covariance": fc.Lambda.tolist() if fc is not None else None,
        "idiosyncratic_variance": fc.D.tolist() if fc is not None else None,
        # Tier-2-specific
        "n_regimes": int(tier2.n_regimes),
        "transition_matrix": P.tolist(),
        "stationary_distribution": stationary.tolist(),
        "regime_params": regime_params,
        "regime_labels": seq.tolist(),
        "regime_durations": _compute_regime_durations(seq, tier2.n_regimes),
        "adversarial": bool(tier2.adversarial),
        "regime_counts_overlap": regime_counts_overlap,
        "regime_counts_backcast": regime_counts_backcast,
        # Metadata
        "seed": cfg.seed,
        "correlation_method": cfg.correlation_method,
        "n_factors": cfg.n_factors,
        "start_date": cfg.start_date,
        "missing_fraction": float(mask_meta.missing_fraction),
    }

    logger.info(
        "Tier 2 generation complete: %d regimes, counts=%s, adversarial=%s",
        tier2.n_regimes,
        {k: int((seq == k).sum()) for k in range(tier2.n_regimes)},
        tier2.adversarial,
    )
    return returns_masked, returns_complete, ground_truth


def _stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a row-stochastic matrix P.

    Solves pi P = pi with pi.sum() = 1 via eigendecomposition of P.T.
    Returns the non-negative normalised left eigenvector associated with the
    eigenvalue 1.
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    vec = np.abs(vec)
    s = vec.sum()
    return vec / s if s > 0 else np.full(P.shape[0], 1.0 / P.shape[0])
