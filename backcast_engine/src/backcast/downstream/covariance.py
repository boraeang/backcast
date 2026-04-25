"""Full-sample covariance estimation from imputed backcast data.

Offers five construction methods, all returning a common
:class:`CovarianceResult`:

- ``from_em_result``          — use Σ̂ directly from EM
- ``sample_covariance``       — plain sample cov on one completed history
- ``combined_covariance``     — Rubin's rules across M imputations
- ``shrink_covariance``       — Ledoit-Wolf shrinkage toward a diagonal target
- ``denoise_covariance``      — Marchenko-Pastur eigenvalue denoising

References
----------
- Ledoit, O. & Wolf, M. (2004).  "A well-conditioned estimator for
  large-dimensional covariance matrices." *J. Multivariate Analysis*.
- Bouchaud, J.-P. & Potters, M. (2009).  *Financial Applications of Random
  Matrix Theory.*
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backcast.models.em_stambaugh import EMResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CovarianceResult:
    """Covariance + diagnostics.

    Attributes
    ----------
    covariance : np.ndarray, shape (N, N)
        PSD, symmetric.
    correlation : np.ndarray, shape (N, N)
    eigenvalues : np.ndarray, shape (N,)
        Descending order.
    eigenvectors : np.ndarray, shape (N, N)
        Column k corresponds to eigenvalues[k].
    condition_number : float
    method : str
    asset_names : list[str]
    within_variance : np.ndarray or None
        Only set by :func:`combined_covariance` — element-wise mean of the
        per-imputation sample covariances.
    between_variance : np.ndarray or None
        Element-wise variance of the per-imputation covariances across M.
    total_variance : np.ndarray or None
        Rubin total: ``W̄ + (1 + 1/M)·B`` on the covariance entries themselves.
    n_imputations : int or None
    """

    covariance: np.ndarray
    correlation: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    method: str
    asset_names: list[str]
    within_variance: Optional[np.ndarray] = None
    between_variance: Optional[np.ndarray] = None
    total_variance: Optional[np.ndarray] = None
    n_imputations: Optional[int] = None


def _build_result(
    cov: np.ndarray, names: list[str], method: str, **extras
) -> CovarianceResult:
    cov = 0.5 * (cov + cov.T)
    vols = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    corr = cov / np.outer(vols, vols)
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    cond = float(eigvals[0] / max(eigvals[-1], 1e-30))
    return CovarianceResult(
        covariance=cov,
        correlation=corr,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        condition_number=cond,
        method=method,
        asset_names=list(names),
        **extras,
    )


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def from_em_result(em_result: EMResult) -> CovarianceResult:
    """Wrap an EM-estimated Σ in a :class:`CovarianceResult`."""
    return _build_result(
        np.asarray(em_result.sigma, dtype=np.float64),
        em_result.asset_order,
        method="em",
    )


def sample_covariance(returns: pd.DataFrame, method_label: str = "sample") -> CovarianceResult:
    """Plain sample covariance on a single (completed) returns DataFrame.

    Raises
    ------
    ValueError
        If *returns* contains NaN.
    """
    if returns.isna().any().any():
        raise ValueError("sample_covariance requires a fully-observed DataFrame")
    cov = np.cov(returns.to_numpy(dtype=np.float64), rowvar=False, bias=False)
    return _build_result(cov, list(returns.columns), method=method_label)


def combined_covariance(imputations: list) -> CovarianceResult:
    """Combine M imputed covariance estimates via Rubin's rules.

    For each imputation ``m``:

        Σ̂_m = sample_cov(imputations[m])

    The combined estimate is ``Σ̄ = mean_m Σ̂_m``.  Within- and between-
    imputation variance refer to the entries of the covariance matrix itself,
    so the caller can assess how much of the reported covariance is driven by
    missing-data uncertainty.

    Parameters
    ----------
    imputations : list of pd.DataFrame
        Complete histories — typically
        :attr:`MultipleImputationResult.imputations`.

    Returns
    -------
    CovarianceResult
    """
    if not imputations:
        raise ValueError("imputations is empty")
    names = list(imputations[0].columns)
    covs = np.stack(
        [np.cov(df.to_numpy(dtype=np.float64), rowvar=False, bias=False) for df in imputations],
        axis=0,
    )  # (M, N, N)
    M = len(covs)
    within = covs.mean(axis=0)
    if M > 1:
        between = covs.var(axis=0, ddof=1)
    else:
        between = np.zeros_like(within)
    total = within + (1.0 + 1.0 / M) * between
    return _build_result(
        within,
        names,
        method="rubin_combined",
        within_variance=within.copy(),
        between_variance=between,
        total_variance=total,
        n_imputations=M,
    )


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage
# ---------------------------------------------------------------------------

def _ledoit_wolf_alpha(returns: np.ndarray, sigma_hat: np.ndarray) -> tuple[float, np.ndarray]:
    """Ledoit-Wolf optimal shrinkage intensity toward the constant-variance target.

    Target = ``(tr Σ̂ / N) · I``.  Returns ``(alpha, target)``.
    """
    T, N = returns.shape
    mu_t = returns.mean(axis=0)
    X = returns - mu_t
    # tr(Σ̂) / N
    target_var = float(np.trace(sigma_hat) / N)
    target = target_var * np.eye(N)

    # π̂ = (1/T) Σ_t ||X_t X_t^T - Σ̂||_F²  (element-wise mean-square-error)
    pi_mat = np.zeros((N, N))
    for t in range(T):
        dev = np.outer(X[t], X[t]) - sigma_hat
        pi_mat += dev ** 2
    pi_hat = pi_mat.sum() / T
    # γ̂ = ||Σ̂ - target||_F²
    gamma_hat = float(np.sum((sigma_hat - target) ** 2))
    # κ̂ = π̂ / γ̂
    kappa_hat = pi_hat / max(gamma_hat, 1e-30)
    alpha = max(0.0, min(1.0, kappa_hat / T))
    return float(alpha), target


def shrink_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> CovarianceResult:
    """Ledoit-Wolf shrunk covariance on a single completed history.

    Parameters
    ----------
    returns : pd.DataFrame
        Fully observed.
    method : str
        Currently only ``'ledoit_wolf'`` is supported.

    Returns
    -------
    CovarianceResult
        ``covariance = (1-α)·Σ̂ + α·target``.  ``α`` is attached to the
        correlation result via the ``method`` string for transparency.
    """
    if method != "ledoit_wolf":
        raise ValueError(f"unknown shrinkage method {method!r}")
    if returns.isna().any().any():
        raise ValueError("shrink_covariance requires a fully-observed DataFrame")
    X = returns.to_numpy(dtype=np.float64)
    sigma_hat = np.cov(X, rowvar=False, bias=False)
    alpha, target = _ledoit_wolf_alpha(X, sigma_hat)
    shrunk = (1.0 - alpha) * sigma_hat + alpha * target
    logger.info("Ledoit-Wolf shrinkage: alpha = %.4f", alpha)
    return _build_result(shrunk, list(returns.columns), method=f"ledoit_wolf(α={alpha:.3f})")


# ---------------------------------------------------------------------------
# Marchenko-Pastur eigenvalue denoising
# ---------------------------------------------------------------------------

def denoise_covariance(
    returns: pd.DataFrame,
    q: Optional[float] = None,
) -> CovarianceResult:
    """Marchenko-Pastur eigenvalue denoising on the correlation matrix.

    Procedure:

    1. Compute the correlation matrix ``C`` (normalised covariance).
    2. Compute eigenvalues ``λ_1 ≥ ... ≥ λ_N``.
    3. Estimate the noise bulk variance ``σ²`` as the mean of eigenvalues
       below the naive MP upper bound.
    4. Replace every eigenvalue below ``λ₊ = σ² (1+√q)²`` with their mean,
       preserving trace.
    5. Reconstruct ``C_clean = V · diag(λ_clean) · V^T`` and rescale back to
       a covariance via the original volatilities.

    Parameters
    ----------
    returns : pd.DataFrame
    q : float, optional
        ``N/T``.  Inferred from the data when None.

    Returns
    -------
    CovarianceResult
    """
    if returns.isna().any().any():
        raise ValueError("denoise_covariance requires a fully-observed DataFrame")
    X = returns.to_numpy(dtype=np.float64)
    T, N = X.shape
    if q is None:
        q = N / T

    cov = np.cov(X, rowvar=False, bias=False)
    vols = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    corr = cov / np.outer(vols, vols)
    np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Iterative noise-bulk estimation: assume top-k are signal, rest noise
    lam_plus_naive = (1.0 + np.sqrt(q)) ** 2
    signal_idx = eigvals > lam_plus_naive
    if signal_idx.all() or not signal_idx.any():
        # Degenerate — treat nothing or everything as signal
        noise_mean = eigvals.mean()
        lam_plus = lam_plus_naive
    else:
        noise_eigs = eigvals[~signal_idx]
        sigma2 = float(noise_eigs.mean() / (1.0 + q))
        lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
        noise_idx = eigvals <= lam_plus
        noise_mean = float(eigvals[noise_idx].mean()) if noise_idx.any() else eigvals.mean()

    clean_eigs = np.where(eigvals > lam_plus, eigvals, noise_mean)
    # Preserve trace
    clean_eigs *= eigvals.sum() / clean_eigs.sum()

    corr_clean = eigvecs @ np.diag(clean_eigs) @ eigvecs.T
    corr_clean = 0.5 * (corr_clean + corr_clean.T)
    np.fill_diagonal(corr_clean, 1.0)
    cov_clean = corr_clean * np.outer(vols, vols)

    return _build_result(
        cov_clean, list(returns.columns),
        method=f"marchenko_pastur(q={q:.3f})",
    )
