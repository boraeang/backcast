"""Configuration dataclasses for the synthetic data generator."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tier2Config:
    """Configuration for Tier 2 regime-switching DGP.

    Parameters
    ----------
    n_regimes : int
        Number of regimes.
    transition_matrix : list[list[float]] or None
        Row-stochastic Markov transition matrix.  Generated automatically when None.
    regime_vol_multipliers : list[float] or None
        Per-regime volatility multipliers relative to base (e.g. [1.0, 2.5]).
    regime_corr_adjustments : list[float] or None
        Value added to off-diagonal correlations in each regime (e.g. [0.0, 0.3]).
    regime_mean_adjustments : list[float] or None
        Multiplier applied to the base mean vector in each regime (e.g. [1.0, -0.5]).
    adversarial : bool
        If True the crisis regime appears only in the backcast period.
    """

    n_regimes: int = 2
    transition_matrix: list[list[float]] | None = None
    regime_vol_multipliers: list[float] | None = None
    regime_corr_adjustments: list[float] | None = None
    regime_mean_adjustments: list[float] | None = None
    adversarial: bool = False


@dataclass
class Tier3Config:
    """Configuration for Tier 3 GARCH + fat-tail DGP.

    Parameters
    ----------
    innovation_distribution : str
        Distribution for innovations: 'student_t' or 'gaussian'.
    degrees_of_freedom : float
        Degrees of freedom for Student-t innovations.
    garch_omega : float
        GARCH(1,1) constant term omega.
    garch_alpha : float
        ARCH coefficient alpha.
    garch_beta : float
        GARCH coefficient beta.
    beta_drift_vol : float
        Standard deviation of the random walk in factor loadings per period.
    """

    innovation_distribution: str = "student_t"
    degrees_of_freedom: float = 5.0
    garch_omega: float = 0.00001
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    beta_drift_vol: float = 0.001


@dataclass
class Tier4Config:
    """Configuration for Tier 4 stress-test scenarios.

    Parameters
    ----------
    scenario : str
        Which stress scenario to run.  One of:
        'short_overlap', 'high_dimension', 'near_singular', 'staggered_heavy', 'all'.
    """

    scenario: str = "short_overlap"


@dataclass
class SyntheticConfig:
    """Master configuration for the synthetic financial time-series generator.

    Parameters
    ----------
    n_long_assets : int
        Number of assets with full history.
    n_short_assets : int
        Number of assets with partial history (monotone missingness).
    t_total : int
        Total number of trading days (~20 years at 252 days/year).
    short_start_day : int or list[int]
        Row index where short-history assets begin.  A single int means all
        short assets start on the same day; a list gives staggered starts.
    start_date : str
        First trading day in YYYY-MM-DD format.
    calendar : str
        Business day calendar (currently 'NYSE' / generic business days).
    correlation_method : str
        How to build the covariance matrix: 'factor_model', 'random', or 'manual'.
    n_factors : int
        Number of latent factors (used when correlation_method='factor_model').
    asset_profiles : dict or None
        Mapping from asset name to (annual_mean_pct, annual_vol_pct).  Uses
        built-in defaults when None.
    tier : int
        Which DGP tier to use (1–4).
    tier2_config : Tier2Config or None
        Tier 2 settings; created with defaults when tier==2 and this is None.
    tier3_config : Tier3Config or None
        Tier 3 settings; created with defaults when tier==3 and this is None.
    tier4_config : Tier4Config or None
        Tier 4 settings; created with defaults when tier==4 and this is None.
    seed : int
        Master random seed for full reproducibility.
    output_dir : str
        Directory where output files are written.
    save_complete_returns : bool
        If True, also write returns_complete.csv (unmasked returns).
    """

    # Dimensions
    n_long_assets: int = 5
    n_short_assets: int = 3
    t_total: int = 5000

    # Missingness
    short_start_day: int | list[int] = 3000

    # Calendar
    start_date: str = "1990-01-02"
    calendar: str = "NYSE"

    # Correlation structure
    correlation_method: str = "factor_model"
    n_factors: int = 4

    # Asset class calibration — None means use built-in defaults
    asset_profiles: dict | None = None

    # Tier selection
    tier: int = 1
    tier2_config: Tier2Config | None = None
    tier3_config: Tier3Config | None = None
    tier4_config: Tier4Config | None = None

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "./synthetic_output"
    save_complete_returns: bool = True


# ---------------------------------------------------------------------------
# Default asset profiles (annual mean %, annual vol %)
# ---------------------------------------------------------------------------

DEFAULT_LONG_PROFILES: dict[str, tuple[float, float]] = {
    "EQUITY_1": (8.0, 16.0),
    "EQUITY_2": (7.0, 18.0),
    "BOND_1":   (3.0,  5.0),
    "BOND_2":   (4.0,  7.0),
    "GOLD":     (4.0, 15.0),
}

DEFAULT_SHORT_PROFILES: dict[str, tuple[float, float]] = {
    "CRYPTO_1": (30.0, 70.0),
    "CRYPTO_2": (20.0, 65.0),
    "ALT_1":    (10.0, 25.0),
}

# Fallback profiles for extra assets beyond the named defaults
_EXTRA_LONG_TEMPLATE: tuple[float, float] = (8.0, 16.0)   # equity-like
_EXTRA_SHORT_TEMPLATE: tuple[float, float] = (10.0, 25.0)  # alt-like


def build_asset_profiles(cfg: SyntheticConfig) -> dict[str, tuple[float, float]]:
    """Return the full asset-profile mapping for a given configuration.

    If ``cfg.asset_profiles`` is set it is returned as-is (after length
    validation).  Otherwise the built-in defaults are used, extending with
    generic names when the requested counts exceed the named defaults.

    Parameters
    ----------
    cfg : SyntheticConfig
        Generator configuration.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from asset name to (annual_mean_pct, annual_vol_pct).
        Long assets appear first, short assets last.
    """
    if cfg.asset_profiles is not None:
        n_expected = cfg.n_long_assets + cfg.n_short_assets
        if len(cfg.asset_profiles) != n_expected:
            raise ValueError(
                f"asset_profiles has {len(cfg.asset_profiles)} entries "
                f"but n_long_assets + n_short_assets = {n_expected}"
            )
        return cfg.asset_profiles

    long_names = list(DEFAULT_LONG_PROFILES.keys())[: cfg.n_long_assets]
    long_profiles = {k: DEFAULT_LONG_PROFILES[k] for k in long_names}

    # Add generic long assets when count exceeds named defaults
    for i in range(len(long_names), cfg.n_long_assets):
        long_profiles[f"EQUITY_{i + 1}"] = _EXTRA_LONG_TEMPLATE

    short_names = list(DEFAULT_SHORT_PROFILES.keys())[: cfg.n_short_assets]
    short_profiles = {k: DEFAULT_SHORT_PROFILES[k] for k in short_names}

    for i in range(len(short_names), cfg.n_short_assets):
        short_profiles[f"ALT_{i + 1}"] = _EXTRA_SHORT_TEMPLATE

    return {**long_profiles, **short_profiles}


def get_short_start_indices(cfg: SyntheticConfig, short_asset_names: list[str]) -> dict[str, int]:
    """Map each short asset to its start row index.

    Parameters
    ----------
    cfg : SyntheticConfig
        Generator configuration.
    short_asset_names : list[str]
        Names of the short-history assets in order.

    Returns
    -------
    dict[str, int]
        Mapping from asset name to the first valid row index (0-based).
    """
    n = len(short_asset_names)
    if isinstance(cfg.short_start_day, int):
        starts: list[int] = [cfg.short_start_day] * n
    else:
        if len(cfg.short_start_day) != n:
            raise ValueError(
                f"short_start_day list length {len(cfg.short_start_day)} "
                f"!= n_short_assets {n}"
            )
        starts = list(cfg.short_start_day)

    for idx, s in enumerate(starts):
        if not (0 < s < cfg.t_total):
            raise ValueError(
                f"short_start_day[{idx}]={s} must be in (0, {cfg.t_total})"
            )

    return dict(zip(short_asset_names, starts))
