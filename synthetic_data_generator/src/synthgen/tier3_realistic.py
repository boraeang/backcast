"""Tier 3: realistic financial DGP with GARCH, fat tails, and TVP betas.

DGP
---
1. **GARCH(1,1) factor returns with Student-t innovations**::

       sigma_{j,t}^2 = omega + alpha * (f_{j,t-1})^2 + beta * sigma_{j,t-1}^2
       f_{j,t}       = sigma_{j,t} * z_{j,t},    z_{j,t} ~ standardised t_nu

   Mean factor returns are zero; asset-level means live in *mu_asset*.

2. **Random-walk factor loadings (TVP betas)**::

       B_t = B_{t-1} + eta_t,   eta_t ~ N(0, sigma_eta^2 * I)

   Clipped elementwise so that ``|B_{ij,t}| <= 3 * |B_{ij,0}|``.

3. **Asset returns**::

       R_t = mu_asset + B_t @ f_t + eps_t,   eps_{i,t} ~ standardised t_nu(0, d_i)

Unconditional covariance
------------------------
Factor unconditional variance :math:`s_f^2 = \\omega/(1 - \\alpha - \\beta)`.
The base factor loadings from ``correlation.build_factor_model_covariance``
assume unit-variance factors (``Lambda = I``), so they are rescaled by
``1 / s_f`` before use.  The analytical unconditional covariance is then::

       Sigma_uncond = B_eff @ diag(s_f^2) @ B_eff.T + D = B_base @ B_base.T + D

which coincides with the Tier 1 covariance — asset vols remain calibrated to
``asset_profiles`` regardless of GARCH parameter choice.

Validation criteria (checked in tests/test_tier3.py)
-----------------------------------------------------------
- Marginal excess kurtosis > 0 (fat tails present).
- Ljung-Box on squared returns rejects the no-autocorrelation null (GARCH).
- Sample unconditional variance close to analytical Sigma diagonal.
- GARCH recursion holds exactly for the recorded conditional vols.
- Beta path respects the 3x clipping bound and starts at the rescaled ``B_eff``.
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
    Tier3Config,
    build_asset_profiles,
    get_short_start_indices,
)
from synthgen.correlation import build_covariance
from synthgen.masking import apply_masking

logger = logging.getLogger(__name__)

_ANNUAL_TRADING_DAYS = 252
_ROLLING_WINDOW = 90


# ---------------------------------------------------------------------------
# Config resolution / validation
# ---------------------------------------------------------------------------

def _resolve_tier3_config(raw: Tier3Config | None) -> Tier3Config:
    """Return a :class:`Tier3Config` — the supplied one or defaults."""
    return raw if raw is not None else Tier3Config()


def _validate_tier3_config(cfg: Tier3Config) -> None:
    if cfg.innovation_distribution not in ("student_t", "gaussian"):
        raise ValueError(
            f"Unknown innovation_distribution {cfg.innovation_distribution!r}; "
            "expected 'student_t' or 'gaussian'"
        )
    if cfg.innovation_distribution == "student_t" and cfg.degrees_of_freedom <= 2.0:
        raise ValueError(
            f"degrees_of_freedom={cfg.degrees_of_freedom} must be > 2 for finite variance"
        )
    if not (0 < cfg.garch_alpha < 1):
        raise ValueError(f"garch_alpha={cfg.garch_alpha} must be in (0, 1)")
    if not (0 <= cfg.garch_beta < 1):
        raise ValueError(f"garch_beta={cfg.garch_beta} must be in [0, 1)")
    if cfg.garch_alpha + cfg.garch_beta >= 1.0:
        raise ValueError(
            f"alpha+beta = {cfg.garch_alpha + cfg.garch_beta} must be < 1 for stationarity"
        )
    if cfg.garch_omega <= 0:
        raise ValueError(f"garch_omega={cfg.garch_omega} must be > 0")
    if cfg.beta_drift_vol < 0:
        raise ValueError(f"beta_drift_vol={cfg.beta_drift_vol} must be >= 0")


# ---------------------------------------------------------------------------
# Innovation sampling
# ---------------------------------------------------------------------------

def _sample_unit_variance_innovations(
    rng: npr.Generator, distribution: str, df: float, size: int | tuple[int, ...]
) -> np.ndarray:
    """Draw unit-variance innovations from the chosen distribution.

    Parameters
    ----------
    rng : numpy.random.Generator
    distribution : str
        'student_t' or 'gaussian'.
    df : float
        Degrees of freedom (used only for student_t).
    size : int or tuple of ints
        Output shape.

    Returns
    -------
    np.ndarray
        Sample with mean 0 and unit variance (exactly for Gaussian,
        analytically for standardised Student-t).
    """
    if distribution == "gaussian":
        return rng.standard_normal(size=size)
    # standard_t has variance df/(df-2); rescale for unit variance
    raw = rng.standard_t(df, size=size)
    return raw * np.sqrt((df - 2.0) / df)


# ---------------------------------------------------------------------------
# GARCH(1,1) simulation
# ---------------------------------------------------------------------------

def _simulate_garch(
    T: int,
    omega: float,
    alpha: float,
    beta: float,
    innovations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a GARCH(1,1) path driven by unit-variance innovations.

    Parameters
    ----------
    T : int
        Sequence length.
    omega, alpha, beta : float
        GARCH(1,1) parameters.  Must satisfy ``alpha + beta < 1``.
    innovations : np.ndarray, shape (T,)
        Unit-variance standardised innovations z_t.

    Returns
    -------
    f : np.ndarray, shape (T,)
        Realised returns ``f_t = sigma_t * z_t``.
    sigma : np.ndarray, shape (T,)
        Conditional standard deviations sigma_t (=sqrt(h_t)).

    Notes
    -----
    sigma_0^2 is initialised at the unconditional variance
    ``omega / (1 - alpha - beta)``.  The recursion is
    ``sigma_t^2 = omega + alpha * f_{t-1}^2 + beta * sigma_{t-1}^2``.
    """
    f = np.zeros(T, dtype=np.float64)
    sigma2 = np.zeros(T, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - beta)
    f[0] = np.sqrt(sigma2[0]) * innovations[0]
    for t in range(1, T):
        sigma2[t] = omega + alpha * f[t - 1] ** 2 + beta * sigma2[t - 1]
        f[t] = np.sqrt(sigma2[t]) * innovations[t]
    return f, np.sqrt(sigma2)


# ---------------------------------------------------------------------------
# Time-varying factor loadings
# ---------------------------------------------------------------------------

def _simulate_beta_path(
    B0: np.ndarray,
    T: int,
    sigma_eta: float,
    rng: npr.Generator,
    clip_multiple: float = 3.0,
) -> np.ndarray:
    """Simulate a clipped random-walk path for the (N, K) factor loading matrix.

    Parameters
    ----------
    B0 : np.ndarray, shape (N, K)
        Initial loading matrix.
    T : int
        Sequence length.
    sigma_eta : float
        Standard deviation of per-period elementwise innovations.
    rng : numpy.random.Generator
    clip_multiple : float
        Elementwise clipping bound: ``|B_{ij,t}| <= clip_multiple * |B_{ij,0}|``.

    Returns
    -------
    np.ndarray, shape (T, N, K)
    """
    N, K = B0.shape
    B_max = clip_multiple * np.abs(B0)
    path = np.empty((T, N, K), dtype=np.float64)
    B_curr = B0.copy()
    path[0] = B_curr
    if sigma_eta == 0.0:
        # Static betas — skip noise generation
        path[:] = B_curr
        return path
    for t in range(1, T):
        B_curr = B_curr + rng.standard_normal(size=(N, K)) * sigma_eta
        np.clip(B_curr, -B_max, B_max, out=B_curr)
        path[t] = B_curr
    return path


# ---------------------------------------------------------------------------
# Rolling realised covariance
# ---------------------------------------------------------------------------

def _rolling_covariance(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling sample covariance of *returns* over *window*-sized windows.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Return matrix.
    window : int
        Rolling window length.

    Returns
    -------
    np.ndarray, shape (T, N, N)
        Rolling covariance at each date.  The first ``window - 1`` matrices
        are NaN because the window is not yet fully populated.
    """
    T, N = returns.shape
    out = np.full((T, N, N), np.nan, dtype=np.float64)
    if T < window:
        return out
    for t in range(window - 1, T):
        block = returns[t - window + 1 : t + 1]
        out[t] = np.cov(block, rowvar=False, ddof=1)
    return out


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_tier3(
    cfg: SyntheticConfig,
    rng: npr.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate Tier 3 returns (GARCH + fat tails + TVP betas).

    Parameters
    ----------
    cfg : SyntheticConfig
        Generator configuration.  Must use ``correlation_method='factor_model'``
        because Tier 3 requires access to the factor-model components.
    rng : numpy.random.Generator or None
        Seeded generator.  Created from ``cfg.seed`` when None.

    Returns
    -------
    returns_masked : pd.DataFrame
    returns_complete : pd.DataFrame
    ground_truth : dict
        Extras on top of Tier 1:
        ``factor_returns``, ``factor_garch_params``, ``factor_conditional_vols``,
        ``beta_path``, ``innovation_df``, ``unconditional_sigma``,
        ``rolling_sigma_90d``.

    Raises
    ------
    ValueError
        If ``cfg.correlation_method`` != 'factor_model' or Tier 3 config is invalid.
    """
    if cfg.correlation_method != "factor_model":
        raise ValueError(
            "Tier 3 requires correlation_method='factor_model' to access "
            "factor components."
        )
    if rng is None:
        rng = npr.default_rng(cfg.seed)

    tier3 = _resolve_tier3_config(cfg.tier3_config)
    _validate_tier3_config(tier3)

    # --- Asset profiles --------------------------------------------------
    profiles = build_asset_profiles(cfg)
    asset_names = list(profiles.keys())
    long_assets = asset_names[: cfg.n_long_assets]
    short_assets = asset_names[cfg.n_long_assets :]
    N = len(asset_names)
    K = cfg.n_factors
    T = cfg.t_total

    logger.info(
        "Tier 3: N=%d, K=%d, T=%d, nu=%s, distribution=%s",
        N, K, T, tier3.degrees_of_freedom, tier3.innovation_distribution,
    )

    # --- Daily mean and base factor-model covariance ---------------------
    mu_daily = np.array(
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
        method="factor_model",
        n_factors=K,
        rng=rng,
    )
    assert cov_result.factor_components is not None
    B_base = cov_result.factor_components.B.copy()      # (N, K), unit-variance factors
    D_diag = cov_result.factor_components.D.copy()      # (N,)

    # --- GARCH unconditional factor variance -----------------------------
    omega = float(tier3.garch_omega)
    alpha = float(tier3.garch_alpha)
    beta = float(tier3.garch_beta)
    s_f2 = omega / (1.0 - alpha - beta)                 # unconditional Var per factor
    s_f = np.sqrt(s_f2)

    # Rescale B so the asset-level unconditional covariance matches target.
    # With Lambda = diag(s_f2) instead of I, Sigma_asset_target = (B/s_f)(s_f^2 I)(B/s_f).T + D
    # = B @ B.T + D, which equals the Tier 1 unconditional covariance.
    B_eff = B_base / s_f                                # (N, K)

    # --- Simulate factor GARCH paths -------------------------------------
    factor_innov = _sample_unit_variance_innovations(
        rng, tier3.innovation_distribution, tier3.degrees_of_freedom, size=(T, K)
    )
    factor_returns = np.empty((T, K), dtype=np.float64)
    factor_cond_vols = np.empty((T, K), dtype=np.float64)
    for j in range(K):
        f_j, sig_j = _simulate_garch(T, omega, alpha, beta, factor_innov[:, j])
        factor_returns[:, j] = f_j
        factor_cond_vols[:, j] = sig_j

    # --- Simulate time-varying factor loadings ---------------------------
    beta_path = _simulate_beta_path(B_eff, T, tier3.beta_drift_vol, rng)

    # --- Idiosyncratic shocks --------------------------------------------
    # eps_{i,t} has variance d_i; use unit-variance innovations scaled by sqrt(d_i).
    idio_innov = _sample_unit_variance_innovations(
        rng, tier3.innovation_distribution, tier3.degrees_of_freedom, size=(T, N)
    )
    eps = idio_innov * np.sqrt(D_diag)                  # broadcast over (T, N)

    # --- Compose asset returns: R_t = mu + B_t @ f_t + eps_t ------------
    # Use einsum to avoid an explicit time loop: beta_path is (T, N, K).
    factor_contrib = np.einsum("tnk,tk->tn", beta_path, factor_returns)
    returns_arr = mu_daily[np.newaxis, :] + factor_contrib + eps

    # --- Dates + DataFrame -----------------------------------------------
    dates = generate_business_days(cfg.start_date, T)
    returns_complete = pd.DataFrame(
        returns_arr, index=dates, columns=asset_names, dtype=np.float64
    )

    # --- Masking ---------------------------------------------------------
    start_indices = get_short_start_indices(cfg, short_assets) if short_assets else {}
    returns_masked, mask_meta = apply_masking(returns_complete, start_indices)

    # --- Analytical unconditional covariance -----------------------------
    Lambda_eff = s_f2 * np.eye(K, dtype=np.float64)
    unconditional_sigma = B_eff @ Lambda_eff @ B_eff.T + np.diag(D_diag)
    # Symmetrise (numerical hygiene)
    unconditional_sigma = 0.5 * (unconditional_sigma + unconditional_sigma.T)
    unconditional_corr = _corr_from_cov(unconditional_sigma)

    # --- Rolling 90-day realised covariance ------------------------------
    rolling_sigma = _rolling_covariance(returns_arr, _ROLLING_WINDOW)

    # --- Ground truth ----------------------------------------------------
    ground_truth: dict[str, Any] = {
        "tier": 3,
        "asset_names": asset_names,
        "long_assets": long_assets,
        "short_assets": short_assets,
        "short_asset_start_indices": {k: int(v) for k, v in start_indices.items()},
        "n_observations": int(T),
        # Unconditional references
        "mu": mu_daily.tolist(),
        "sigma": unconditional_sigma.tolist(),
        "correlation": unconditional_corr.tolist(),
        "mu_daily": mu_daily.tolist(),
        "sigma_daily": unconditional_sigma.tolist(),
        # Factor-model components (with the effective, rescaled B)
        "factor_loadings": B_eff.tolist(),
        "factor_covariance": Lambda_eff.tolist(),
        "idiosyncratic_variance": D_diag.tolist(),
        # Tier-3-specific
        "factor_returns": factor_returns.tolist(),
        "factor_garch_params": {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "unconditional_factor_variance": float(s_f2),
            "unconditional_factor_stdev": float(s_f),
        },
        "factor_conditional_vols": factor_cond_vols.tolist(),
        "beta_path": beta_path.tolist(),
        "beta_initial": B_eff.tolist(),
        "beta_drift_vol": float(tier3.beta_drift_vol),
        "innovation_distribution": tier3.innovation_distribution,
        "innovation_df": (
            float(tier3.degrees_of_freedom)
            if tier3.innovation_distribution == "student_t" else None
        ),
        "unconditional_sigma": unconditional_sigma.tolist(),
        "rolling_sigma_90d": rolling_sigma.tolist(),
        "rolling_window": _ROLLING_WINDOW,
        # Metadata
        "seed": cfg.seed,
        "correlation_method": "factor_model",
        "n_factors": K,
        "start_date": cfg.start_date,
        "missing_fraction": float(mask_meta.missing_fraction),
    }

    logger.info(
        "Tier 3 generation complete: returns shape=%s, factor GARCH uncond var=%.3e",
        returns_masked.shape, s_f2,
    )
    return returns_masked, returns_complete, ground_truth


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _corr_from_cov(sigma: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix with unit diagonal."""
    vols = np.sqrt(np.diag(sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = sigma / np.outer(vols, vols)
    np.fill_diagonal(corr, 1.0)
    return corr
