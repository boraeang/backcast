"""Tier 1: i.i.d. multivariate Gaussian returns (stationary DGP).

DGP
---
    R_t ~ N(mu, Sigma)  i.i.d. for t = 1, ..., T

Validation criteria (checked in tests/test_tier1.py)
------------------------------------------------------
- Sample mean within 2 * sigma / sqrt(T) of true mean for each asset.
- Sample covariance Frobenius relative error < 5% for T=5000.
- Marginal distributions pass KS test for normality.
- No autocorrelation: Ljung-Box test does not reject at 5% significance.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.random as npr
import pandas as pd

from synthgen.calendar import generate_business_days
from synthgen.config import SyntheticConfig, build_asset_profiles, get_short_start_indices
from synthgen.correlation import build_covariance
from synthgen.masking import apply_masking

logger = logging.getLogger(__name__)

_ANNUAL_TRADING_DAYS = 252


def _annualised_to_daily_mean(mu_annual_pct: float) -> float:
    """Convert annualised mean (%) to daily mean (fractional)."""
    return (mu_annual_pct / 100.0) / _ANNUAL_TRADING_DAYS


def _annualised_to_daily_vol(vol_annual_pct: float) -> float:
    """Convert annualised volatility (%) to daily vol (fractional)."""
    return (vol_annual_pct / 100.0) / np.sqrt(_ANNUAL_TRADING_DAYS)


def generate_tier1(
    cfg: SyntheticConfig,
    rng: npr.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate i.i.d. multivariate Gaussian returns (Tier 1).

    Parameters
    ----------
    cfg : SyntheticConfig
        Generator configuration.  ``cfg.tier`` should be 1.
    rng : numpy.random.Generator or None
        Seeded generator.  Created from ``cfg.seed`` when None.

    Returns
    -------
    returns_masked : pd.DataFrame
        Returns matrix with NaN for short-history assets before their start dates.
    returns_complete : pd.DataFrame
        Full returns matrix without any masking.
    ground_truth : dict
        True DGP parameters including mu, sigma, factor loadings, and metadata.

    Notes
    -----
    The returned ground_truth dict contains the following keys:

    ``mu``
        True daily mean vector as a list of floats.
    ``sigma``
        True daily covariance matrix as a list of lists.
    ``correlation``
        True correlation matrix as a list of lists.
    ``asset_names``
        Ordered list of asset names.
    ``long_assets``
        Names of assets with full history.
    ``short_assets``
        Names of assets with partial history.
    ``short_asset_start_indices``
        Dict mapping short asset name → first valid row index.
    ``n_observations``
        Total number of dates.
    ``tier``
        Always 1 for this generator.
    ``mu_daily``
        Alias for ``mu`` (daily mean vector).
    ``sigma_daily``
        Alias for ``sigma`` (daily covariance matrix).
    ``factor_loadings``
        Factor loading matrix B (None if method != 'factor_model').
    ``factor_covariance``
        Factor covariance Lambda (None if method != 'factor_model').
    ``idiosyncratic_variance``
        Idiosyncratic variance diagonal D (None if method != 'factor_model').
    """
    if rng is None:
        rng = npr.default_rng(cfg.seed)

    # --- Asset profiles --------------------------------------------------
    profiles = build_asset_profiles(cfg)
    asset_names = list(profiles.keys())
    long_assets = asset_names[: cfg.n_long_assets]
    short_assets = asset_names[cfg.n_long_assets :]

    logger.info("Tier 1: %d long assets, %d short assets, T=%d",
                cfg.n_long_assets, cfg.n_short_assets, cfg.t_total)

    # --- Daily mean and vol vectors --------------------------------------
    mu_daily = np.array(
        [_annualised_to_daily_mean(profiles[a][0]) for a in asset_names],
        dtype=np.float64,
    )
    vols_daily = np.array(
        [_annualised_to_daily_vol(profiles[a][1]) for a in asset_names],
        dtype=np.float64,
    )

    # --- Covariance matrix -----------------------------------------------
    cov_result = build_covariance(
        asset_names=asset_names,
        vols_daily=vols_daily,
        method=cfg.correlation_method,
        n_factors=cfg.n_factors,
        rng=rng,
    )
    sigma_daily = cov_result.sigma_daily

    # --- Draw T observations from N(mu, Sigma) ---------------------------
    returns_arr = rng.multivariate_normal(
        mean=mu_daily,
        cov=sigma_daily,
        size=cfg.t_total,
    )  # shape (T, N)

    # --- Date index ------------------------------------------------------
    dates = generate_business_days(cfg.start_date, cfg.t_total)

    # --- Build complete DataFrame ----------------------------------------
    returns_complete = pd.DataFrame(
        returns_arr, index=dates, columns=asset_names, dtype=np.float64
    )

    # --- Apply masking ----------------------------------------------------
    start_indices = get_short_start_indices(cfg, short_assets)
    returns_masked, mask_meta = apply_masking(returns_complete, start_indices)

    # --- Ground truth ----------------------------------------------------
    fc = cov_result.factor_components
    ground_truth: dict[str, Any] = {
        "tier": 1,
        "asset_names": asset_names,
        "long_assets": long_assets,
        "short_assets": short_assets,
        "short_asset_start_indices": {k: int(v) for k, v in start_indices.items()},
        "n_observations": int(cfg.t_total),
        "mu": mu_daily.tolist(),
        "sigma": sigma_daily.tolist(),
        "correlation": cov_result.corr.tolist(),
        # Aliases used by downstream code
        "mu_daily": mu_daily.tolist(),
        "sigma_daily": sigma_daily.tolist(),
        # Factor model components (None if method != factor_model)
        "factor_loadings": fc.B.tolist() if fc is not None else None,
        "factor_covariance": fc.Lambda.tolist() if fc is not None else None,
        "idiosyncratic_variance": fc.D.tolist() if fc is not None else None,
        # Metadata
        "seed": cfg.seed,
        "correlation_method": cfg.correlation_method,
        "n_factors": cfg.n_factors,
        "start_date": cfg.start_date,
        "missing_fraction": float(mask_meta.missing_fraction),
    }

    logger.info(
        "Tier 1 generation complete: returns shape=%s, NaN fraction=%.2f%%",
        returns_masked.shape,
        100.0 * mask_meta.missing_fraction,
    )
    return returns_masked, returns_complete, ground_truth
