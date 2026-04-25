"""Covariance / correlation matrix builders.

Three methods are provided:

factor_model (default)
    Sigma = B @ Lambda @ B.T + D where B is the (N x K) factor loading matrix,
    Lambda is the (K x K) diagonal factor covariance, and D is the diagonal
    idiosyncratic variance matrix.  Factor exposures are calibrated to produce
    realistic inter-asset correlations.

random
    A random valid correlation matrix generated via a random Cholesky factor,
    then scaled to the desired asset volatilities.

manual
    Accepts a user-supplied correlation matrix and volatility vector; validates
    positive semi-definiteness and converts to a covariance matrix.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import numpy.random as npr
from scipy.linalg import cho_factor, cho_solve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factor-model calibration constants
# ---------------------------------------------------------------------------

# 4-factor model: equity risk, rates/duration, inflation/real, speculative/crypto
# Rows are asset classes; columns are factors.
# Values represent relative factor exposures (will be L2-normalised per row).
_N_BASE_FACTORS = 4

_CLASS_RAW_EXPOSURES: dict[str, np.ndarray] = {
    "equity": np.array([0.90, -0.10,  0.05,  0.10]),
    "bond":   np.array([-0.10,  0.90,  0.10,  0.00]),
    "gold":   np.array([0.10,  0.10,  0.90,  0.00]),
    "crypto": np.array([0.50,  0.00,  0.00,  0.90]),
    "alt":    np.array([0.70,  0.00,  0.30,  0.40]),
}

# Fraction of variance explained by the factor model per class (R²)
_CLASS_R2: dict[str, float] = {
    "equity": 0.70,
    "bond":   0.70,
    "gold":   0.60,
    "crypto": 0.50,
    "alt":    0.60,
}


def _asset_class(name: str) -> str:
    """Infer asset class from a standardised asset name."""
    upper = name.upper()
    if upper.startswith("EQUITY") or upper.startswith("EQ"):
        return "equity"
    if upper.startswith("BOND") or upper.startswith("RATES"):
        return "bond"
    if upper in ("GOLD", "SILVER", "COMMODITY"):
        return "gold"
    if upper.startswith("CRYPTO") or upper.startswith("BTC") or upper.startswith("ETH"):
        return "crypto"
    return "alt"


class FactorModelComponents(NamedTuple):
    """Components of the factor covariance decomposition.

    Attributes
    ----------
    B : np.ndarray, shape (N, K)
        Factor loading matrix (daily units).
    Lambda : np.ndarray, shape (K, K)
        Diagonal factor covariance matrix (daily units).
    D : np.ndarray, shape (N,)
        Diagonal of the idiosyncratic variance matrix (daily units).
    """

    B: np.ndarray
    Lambda: np.ndarray
    D: np.ndarray


class CovarianceResult(NamedTuple):
    """Full output of a covariance build call.

    Attributes
    ----------
    sigma_daily : np.ndarray, shape (N, N)
        Daily covariance matrix (PSD).
    sigma_annual : np.ndarray, shape (N, N)
        Annualised covariance matrix.
    corr : np.ndarray, shape (N, N)
        Correlation matrix derived from sigma_daily.
    vols_daily : np.ndarray, shape (N,)
        Daily standard deviations.
    factor_components : FactorModelComponents or None
        Non-None only for the factor_model method.
    """

    sigma_daily: np.ndarray
    sigma_annual: np.ndarray
    corr: np.ndarray
    vols_daily: np.ndarray
    factor_components: FactorModelComponents | None


# ---------------------------------------------------------------------------
# PSD utilities
# ---------------------------------------------------------------------------

def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the cone of PSD matrices.

    Uses eigenvalue clipping: negative eigenvalues are replaced by *epsilon*.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Symmetric matrix to project.
    epsilon : float
        Minimum eigenvalue in the output.

    Returns
    -------
    np.ndarray
        Symmetric PSD matrix of the same shape.
    """
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.maximum(eigvals, epsilon)
    result = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return (result + result.T) / 2.0


def is_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if *matrix* is positive semi-definite within *tol*."""
    eigvals = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigvals >= -tol))


def _sigma_to_corr(sigma: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    vols = np.sqrt(np.diag(sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = sigma / np.outer(vols, vols)
    np.fill_diagonal(corr, 1.0)
    return corr


# ---------------------------------------------------------------------------
# Method 1: Factor model
# ---------------------------------------------------------------------------

def build_factor_model_covariance(
    asset_names: list[str],
    vols_daily: np.ndarray,
    n_factors: int = 4,
) -> CovarianceResult:
    """Build a covariance matrix via a latent factor model.

    Sigma = B @ Lambda @ B.T + D

    Factor exposures are calibrated so that the resulting asset-level
    correlations match typical financial market values:

    - Equity–equity:  0.50–0.80
    - Equity–bond:   -0.20–0.10
    - Equity–gold:    0.00–0.15
    - Equity–crypto:  0.30–0.50
    - Bond–gold:      0.10–0.30
    - Crypto–alt:     0.30–0.60

    Parameters
    ----------
    asset_names : list[str]
        Names of all assets in order.
    vols_daily : np.ndarray, shape (N,)
        Daily standard deviations for each asset.
    n_factors : int
        Number of latent factors (1–4 currently supported; beyond 4 extra
        factors get near-zero loadings).

    Returns
    -------
    CovarianceResult
        Covariance matrix, correlation matrix, and factor components.

    Notes
    -----
    Factor covariance Lambda is identity (standardised factors), so the
    factor covariance contribution is purely B @ B.T.  Idiosyncratic
    variance D is set so that total per-asset variance equals vols_daily**2.
    """
    n = len(asset_names)
    k_use = min(n_factors, _N_BASE_FACTORS)

    # Build normalised exposure matrix  (N x k_use)
    B_norm = np.zeros((n, k_use), dtype=np.float64)
    r2 = np.zeros(n, dtype=np.float64)

    for i, name in enumerate(asset_names):
        cls = _asset_class(name)
        raw = _CLASS_RAW_EXPOSURES[cls][:k_use].copy()
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw / norm
        B_norm[i] = raw
        r2[i] = _CLASS_R2[cls]

    # Scale exposures so that B @ B.T diagonal = R2 * sigma_i^2
    # b_i = sqrt(R2_i) * sigma_i * normalised_exposure_i
    B = B_norm * (np.sqrt(r2) * vols_daily)[:, np.newaxis]

    # For factors beyond the 4 base factors, append near-zero random columns
    if n_factors > _N_BASE_FACTORS:
        extra_k = n_factors - _N_BASE_FACTORS
        extra_B = np.zeros((n, extra_k), dtype=np.float64)
        B = np.hstack([B, extra_B])

    Lambda = np.eye(n_factors, dtype=np.float64)

    # Idiosyncratic variance: d_i = sigma_i^2 - (B Lambda B.T)[i,i]
    factor_var = np.diag(B @ Lambda @ B.T)
    d = np.maximum(vols_daily**2 - factor_var, 1e-12)
    D = np.diag(d)

    sigma_daily = B @ Lambda @ B.T + D
    sigma_daily = nearest_psd(sigma_daily)

    sigma_annual = sigma_daily * 252.0
    corr = _sigma_to_corr(sigma_daily)

    logger.debug("Factor model: B shape=%s, condition number=%.2f",
                 B.shape, np.linalg.cond(sigma_daily))

    return CovarianceResult(
        sigma_daily=sigma_daily,
        sigma_annual=sigma_annual,
        corr=corr,
        vols_daily=vols_daily,
        factor_components=FactorModelComponents(B=B, Lambda=Lambda, D=d),
    )


# ---------------------------------------------------------------------------
# Method 2: Random correlation matrix
# ---------------------------------------------------------------------------

def build_random_covariance(
    asset_names: list[str],
    vols_daily: np.ndarray,
    rng: npr.Generator,
) -> CovarianceResult:
    """Build a random valid covariance matrix.

    Generates a random lower-triangular Cholesky factor, forms the resulting
    correlation matrix, then scales it by the asset volatilities.

    Parameters
    ----------
    asset_names : list[str]
        Asset names (used only for logging).
    vols_daily : np.ndarray, shape (N,)
        Daily standard deviations.
    rng : numpy.random.Generator
        Seeded random number generator.

    Returns
    -------
    CovarianceResult
        Random covariance matrix with diagonal equal to vols_daily**2.
    """
    n = len(asset_names)

    # Random Cholesky: lower-triangular with positive diagonal
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        L[i, :i] = rng.uniform(-1.0, 1.0, size=i)
        L[i, i] = np.abs(rng.standard_normal()) + 0.5

    raw = L @ L.T
    # Normalise to correlation
    d = np.sqrt(np.diag(raw))
    corr = raw / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    sigma_daily = corr * np.outer(vols_daily, vols_daily)
    sigma_daily = nearest_psd(sigma_daily)
    sigma_annual = sigma_daily * 252.0
    corr = _sigma_to_corr(sigma_daily)

    return CovarianceResult(
        sigma_daily=sigma_daily,
        sigma_annual=sigma_annual,
        corr=corr,
        vols_daily=vols_daily,
        factor_components=None,
    )


# ---------------------------------------------------------------------------
# Method 3: Manual (user-supplied)
# ---------------------------------------------------------------------------

def build_manual_covariance(
    corr_matrix: np.ndarray,
    vols_daily: np.ndarray,
) -> CovarianceResult:
    """Build a covariance matrix from a user-supplied correlation matrix.

    Parameters
    ----------
    corr_matrix : np.ndarray, shape (N, N)
        User-supplied correlation matrix.  Must be symmetric with unit diagonal.
        Will be projected to the nearest PSD matrix if necessary.
    vols_daily : np.ndarray, shape (N,)
        Daily standard deviations.

    Returns
    -------
    CovarianceResult
        Validated covariance matrix and derived quantities.

    Raises
    ------
    ValueError
        If *corr_matrix* is not square, not symmetric (within tolerance), or
        does not have unit diagonal.
    """
    n = len(vols_daily)
    if corr_matrix.shape != (n, n):
        raise ValueError(
            f"corr_matrix shape {corr_matrix.shape} does not match "
            f"n_assets={n}"
        )
    if not np.allclose(corr_matrix, corr_matrix.T, atol=1e-6):
        raise ValueError("corr_matrix is not symmetric")
    if not np.allclose(np.diag(corr_matrix), 1.0, atol=1e-6):
        raise ValueError("corr_matrix diagonal must be all 1.0")

    if not is_psd(corr_matrix):
        logger.warning("Supplied correlation matrix is not PSD; projecting to nearest PSD.")
        corr_matrix = nearest_psd(corr_matrix)

    sigma_daily = corr_matrix * np.outer(vols_daily, vols_daily)
    sigma_annual = sigma_daily * 252.0
    corr = _sigma_to_corr(sigma_daily)

    return CovarianceResult(
        sigma_daily=sigma_daily,
        sigma_annual=sigma_annual,
        corr=corr,
        vols_daily=vols_daily,
        factor_components=None,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def build_covariance(
    asset_names: list[str],
    vols_daily: np.ndarray,
    method: str = "factor_model",
    n_factors: int = 4,
    rng: npr.Generator | None = None,
    manual_corr: np.ndarray | None = None,
) -> CovarianceResult:
    """Build a covariance matrix using the requested method.

    Parameters
    ----------
    asset_names : list[str]
        Asset names in order.
    vols_daily : np.ndarray, shape (N,)
        Daily standard deviations for each asset.
    method : str
        'factor_model', 'random', or 'manual'.
    n_factors : int
        Number of factors for the factor model method.
    rng : numpy.random.Generator or None
        Required when method='random'.
    manual_corr : np.ndarray or None
        Required when method='manual'.

    Returns
    -------
    CovarianceResult
        Covariance matrix and derived quantities.

    Raises
    ------
    ValueError
        On invalid method name or missing required arguments.
    """
    if method == "factor_model":
        return build_factor_model_covariance(asset_names, vols_daily, n_factors)
    elif method == "random":
        if rng is None:
            raise ValueError("rng must be provided for method='random'")
        return build_random_covariance(asset_names, vols_daily, rng)
    elif method == "manual":
        if manual_corr is None:
            raise ValueError("manual_corr must be provided for method='manual'")
        return build_manual_covariance(manual_corr, vols_daily)
    else:
        raise ValueError(f"Unknown correlation_method '{method}'")
