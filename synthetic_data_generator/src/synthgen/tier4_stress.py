"""Tier 4: stress-test scenarios for numerical robustness.

Four scenarios are supported (selected via :class:`Tier4Config.scenario`):

``short_overlap``
    Only ~1 year (250 days) of overlap between long and short assets.
    Uses the Tier 1 stationary-Gaussian DGP.

``high_dimension``
    25 assets (5 long, 20 short) with the covariance estimated from only the
    long assets' fully-observed rows.  Uses Tier 1 DGP.

``near_singular``
    Two pairs of assets with extreme pairwise correlations (0.98 and -0.95).
    The covariance is still PSD but has a high condition number.

``staggered_heavy``
    10 short-history assets, each starting 100 business days apart
    (rows 2000, 2100, ..., 2900).  Uses Tier 1 DGP.

``all``
    Not a real scenario — handled by :func:`generate_tier4_all`.

Every scenario adds ``scenario``, ``expected_challenges``, and
``condition_number`` to the ground truth and inherits the Tier 1 fields.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import numpy.random as npr
import pandas as pd

from synthgen.calendar import generate_business_days
from synthgen.config import (
    SyntheticConfig,
    Tier4Config,
    build_asset_profiles,
    get_short_start_indices,
)
from synthgen.correlation import build_manual_covariance
from synthgen.masking import apply_masking
from synthgen.tier1_stationary import generate_tier1

logger = logging.getLogger(__name__)

_ANNUAL_TRADING_DAYS = 252
SCENARIO_NAMES = ("short_overlap", "high_dimension", "near_singular", "staggered_heavy")


# ---------------------------------------------------------------------------
# Scenario descriptions
# ---------------------------------------------------------------------------

_EXPECTED_CHALLENGES: dict[str, str] = {
    "short_overlap": (
        "Only ~250 days (~1 year) of joint overlap between long and short "
        "assets. Tests parameter estimation reliability when the conditioning "
        "sample is small."
    ),
    "high_dimension": (
        "25 assets (5 long, 20 short). The 25x25 covariance must be estimated "
        "from only 5 fully-observed assets. Tests EM under a high "
        "short-to-long ratio."
    ),
    "near_singular": (
        "Two pairs of assets with extreme pairwise correlations (0.98 and "
        "-0.95). Covariance is PSD but ill-conditioned. Tests Cholesky "
        "stability and numerical conditioning."
    ),
    "staggered_heavy": (
        "10 short-history assets with starts 100 business days apart "
        "(rows 2000, 2100, ..., 2900). Tests staggered missingness handling "
        "and sequential group processing in EM."
    ),
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_tier4(
    cfg: SyntheticConfig,
    rng: npr.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate a single Tier 4 stress scenario.

    Parameters
    ----------
    cfg : SyntheticConfig
        Generator configuration; ``cfg.tier4_config.scenario`` selects
        the scenario.
    rng : numpy.random.Generator or None

    Returns
    -------
    returns_masked : pd.DataFrame
    returns_complete : pd.DataFrame
    ground_truth : dict

    Raises
    ------
    ValueError
        When ``scenario='all'`` (use :func:`generate_tier4_all`) or an unknown
        scenario name is supplied.
    """
    if rng is None:
        rng = npr.default_rng(cfg.seed)

    t4 = cfg.tier4_config or Tier4Config()
    scenario = t4.scenario
    if scenario == "all":
        raise ValueError(
            "scenario='all' is not handled by generate_tier4; use "
            "generate_tier4_all() to produce all scenarios together."
        )
    if scenario not in SCENARIO_NAMES:
        raise ValueError(
            f"Unknown Tier 4 scenario {scenario!r}; "
            f"valid options: {SCENARIO_NAMES + ('all',)}"
        )

    logger.info("Tier 4 scenario: %s", scenario)

    if scenario == "short_overlap":
        masked, complete, gt = _scenario_short_overlap(cfg, rng)
    elif scenario == "high_dimension":
        masked, complete, gt = _scenario_high_dimension(cfg, rng)
    elif scenario == "near_singular":
        masked, complete, gt = _scenario_near_singular(cfg, rng)
    else:  # staggered_heavy
        masked, complete, gt = _scenario_staggered_heavy(cfg, rng)

    _stamp_tier4_metadata(gt, scenario)
    return masked, complete, gt


def generate_tier4_all(
    cfg: SyntheticConfig,
    rng: npr.Generator | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Generate all four Tier 4 scenarios at once.

    Each scenario uses an independent generator seeded from ``cfg.seed + i``
    so that outputs are reproducible per scenario and differ from each other.

    Parameters
    ----------
    cfg : SyntheticConfig
    rng : numpy.random.Generator or None
        Ignored — each scenario is seeded independently.

    Returns
    -------
    dict[str, (masked, complete, ground_truth)]
        Keyed by scenario name.
    """
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    for i, scen in enumerate(SCENARIO_NAMES):
        scen_cfg = dataclasses.replace(
            cfg,
            seed=cfg.seed + i,
            tier4_config=Tier4Config(scenario=scen),
        )
        results[scen] = generate_tier4(scen_cfg)
    return results


# ---------------------------------------------------------------------------
# Shared metadata stamping
# ---------------------------------------------------------------------------

def _stamp_tier4_metadata(gt: dict[str, Any], scenario: str) -> None:
    """Overwrite tier/scenario fields and derive the condition number."""
    gt["tier"] = 4
    gt["scenario"] = scenario
    gt["expected_challenges"] = _EXPECTED_CHALLENGES[scenario]
    sigma = np.asarray(gt["sigma_daily"], dtype=np.float64)
    gt["condition_number"] = float(np.linalg.cond(sigma))


# ---------------------------------------------------------------------------
# Scenario: short_overlap
# ---------------------------------------------------------------------------

def _scenario_short_overlap(
    cfg: SyntheticConfig, rng: npr.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if cfg.t_total <= 250:
        raise ValueError(
            f"t_total={cfg.t_total} too small for short_overlap "
            "(requires t_total > 250)"
        )
    new_start = cfg.t_total - 250
    new_cfg = dataclasses.replace(cfg, short_start_day=new_start, tier=1)
    return generate_tier1(new_cfg, rng)


# ---------------------------------------------------------------------------
# Scenario: high_dimension
# ---------------------------------------------------------------------------

def _scenario_high_dimension(
    cfg: SyntheticConfig, rng: npr.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # Force 5 long / 20 short regardless of cfg settings
    new_cfg = dataclasses.replace(
        cfg,
        n_long_assets=5,
        n_short_assets=20,
        # Keep n_factors at its current value, but clamp to <= 5 so the
        # covariance remains well-structured.
        n_factors=min(cfg.n_factors, 5),
        # Keep user-provided asset_profiles only if they already match
        asset_profiles=None,
        tier=1,
    )
    return generate_tier1(new_cfg, rng)


# ---------------------------------------------------------------------------
# Scenario: staggered_heavy
# ---------------------------------------------------------------------------

def _scenario_staggered_heavy(
    cfg: SyntheticConfig, rng: npr.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start_days = [2000 + 100 * i for i in range(10)]
    if cfg.t_total <= max(start_days):
        raise ValueError(
            f"t_total={cfg.t_total} too small for staggered_heavy "
            f"(requires t_total > {max(start_days)})"
        )
    new_cfg = dataclasses.replace(
        cfg,
        n_short_assets=10,
        short_start_day=start_days,
        asset_profiles=None,
        tier=1,
    )
    return generate_tier1(new_cfg, rng)


# ---------------------------------------------------------------------------
# Scenario: near_singular (custom DGP — direct MVN draw)
# ---------------------------------------------------------------------------

def _scenario_near_singular(
    cfg: SyntheticConfig, rng: npr.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if cfg.n_long_assets < 2 or cfg.n_short_assets < 2:
        raise ValueError(
            "near_singular requires at least 2 long and 2 short assets "
            f"(got n_long={cfg.n_long_assets}, n_short={cfg.n_short_assets})"
        )

    # Build profiles, means, vols
    profiles = build_asset_profiles(cfg)
    asset_names = list(profiles.keys())
    long_assets = asset_names[: cfg.n_long_assets]
    short_assets = asset_names[cfg.n_long_assets :]

    mu_daily = np.array(
        [(p[0] / 100.0) / _ANNUAL_TRADING_DAYS for p in profiles.values()],
        dtype=np.float64,
    )
    vols_daily = np.array(
        [(p[1] / 100.0) / np.sqrt(_ANNUAL_TRADING_DAYS) for p in profiles.values()],
        dtype=np.float64,
    )

    # Start from identity so the injected extreme correlations are preserved
    # exactly — factor-model bases tend to pull -0.95 back toward their natural
    # value after PSD projection, defeating the stress-test intent.  Identity +
    # two 2x2 blocks with |rho| in {0.95, 0.98} is PSD by construction
    # (eigenvalues 1±0.98 and 1±0.95), and vol-ratio scaling alone still yields
    # a condition number comfortably above 1000.
    N = len(asset_names)
    corr = np.eye(N, dtype=np.float64)

    # Pick pairs that maximise vol ratio (hence condition number).  Condition
    # number of the 2x2 coupling block scales as (max_vol/min_vol)^2 when |rho|
    # is near 1, so pairing the lowest-vol long asset (bond) with the highest-
    # vol short asset (crypto) drives cond(Sigma) well above 1000.
    lc = cfg.n_long_assets
    long_vols = vols_daily[:lc]
    short_vols = vols_daily[lc:]
    long_order = np.argsort(long_vols)            # ascending
    short_order = np.argsort(-short_vols)         # descending

    i_high = int(long_order[0])
    j_high = lc + int(short_order[0])
    i_low = int(long_order[1]) if lc > 1 else i_high
    j_low = lc + int(short_order[1]) if cfg.n_short_assets > 1 else j_high

    high_pair = (i_high, j_high)
    low_pair = (i_low, j_low)
    corr[high_pair[0], high_pair[1]] = corr[high_pair[1], high_pair[0]] = 0.98
    corr[low_pair[0], low_pair[1]] = corr[low_pair[1], low_pair[0]] = -0.95

    cov_result = build_manual_covariance(corr, vols_daily)
    sigma = cov_result.sigma_daily
    realised_corr = cov_result.corr

    # Draw MVN returns
    returns_arr = rng.multivariate_normal(mu_daily, sigma, size=cfg.t_total)

    dates = generate_business_days(cfg.start_date, cfg.t_total)
    returns_complete = pd.DataFrame(
        returns_arr, index=dates, columns=asset_names, dtype=np.float64
    )

    start_indices = get_short_start_indices(cfg, short_assets) if short_assets else {}
    returns_masked, mask_meta = apply_masking(returns_complete, start_indices)

    ground_truth: dict[str, Any] = {
        "tier": 1,  # stamped to 4 later
        "asset_names": asset_names,
        "long_assets": long_assets,
        "short_assets": short_assets,
        "short_asset_start_indices": {k: int(v) for k, v in start_indices.items()},
        "n_observations": int(cfg.t_total),
        "mu": mu_daily.tolist(),
        "sigma": sigma.tolist(),
        "correlation": realised_corr.tolist(),
        "mu_daily": mu_daily.tolist(),
        "sigma_daily": sigma.tolist(),
        "factor_loadings": None,
        "factor_covariance": None,
        "idiosyncratic_variance": None,
        "seed": cfg.seed,
        "correlation_method": "manual_near_singular",
        "n_factors": cfg.n_factors,
        "start_date": cfg.start_date,
        "missing_fraction": float(mask_meta.missing_fraction),
        # Scenario-specific diagnostics
        "injected_correlations": {
            f"{asset_names[high_pair[0]]}__{asset_names[high_pair[1]]}": 0.98,
            f"{asset_names[low_pair[0]]}__{asset_names[low_pair[1]]}": -0.95,
        },
        "realised_correlations": {
            f"{asset_names[high_pair[0]]}__{asset_names[high_pair[1]]}":
                float(realised_corr[high_pair[0], high_pair[1]]),
            f"{asset_names[low_pair[0]]}__{asset_names[low_pair[1]]}":
                float(realised_corr[low_pair[0], low_pair[1]]),
        },
    }
    return returns_masked, returns_complete, ground_truth
