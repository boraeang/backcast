"""Uncertainty quantification for robust optimisation.

From ``M`` imputed histories we can derive:

- An **ellipsoidal uncertainty set** for ``μ``::

      { μ : (μ − μ̄)ᵀ S⁻¹ (μ − μ̄) ≤ κ² }

  where ``μ̄`` is the between-imputation mean and ``S`` is the between-
  imputation covariance of the per-imputation mean estimates.  ``κ²`` comes
  from a ``χ²_N`` quantile at the requested confidence level.

- A **box uncertainty set** — component-wise confidence intervals on both
  ``μ`` and every entry of ``Σ``.

- The **distribution of portfolio risk** ``σ_p^{(m)} = √(wᵀ Σ^{(m)} w)``
  across imputations, for any weight vector ``w`` — directly consumable by
  ambiguity-averse allocators.

References
----------
Goldfarb, D. & Iyengar, G. (2003).  "Robust portfolio selection problems."
Mathematics of Operations Research, 28(1), 1-38.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EllipsoidalUncertaintySet:
    """``(μ − μ̄)ᵀ S⁻¹ (μ − μ̄) ≤ κ²``.

    Attributes
    ----------
    mu_center : np.ndarray, shape (N,)
    scaling : np.ndarray, shape (N, N)
        The matrix ``S`` in the ellipsoid definition — between-imputation
        covariance of the per-imputation mean estimates.
    kappa : float
        The radius — ``√(χ²_N(confidence))``.
    confidence : float
    asset_names : list[str]
    n_imputations : int
    """

    mu_center: np.ndarray
    scaling: np.ndarray
    kappa: float
    confidence: float
    asset_names: list[str]
    n_imputations: int


@dataclass
class BoxUncertaintySet:
    """Component-wise confidence intervals on μ and Σ.

    Attributes
    ----------
    mu_lower, mu_upper : np.ndarray, shape (N,)
    cov_lower, cov_upper : np.ndarray, shape (N, N)
        Element-wise lower/upper bounds — NOT necessarily PSD.
    confidence : float
    asset_names : list[str]
    """

    mu_lower: np.ndarray
    mu_upper: np.ndarray
    cov_lower: np.ndarray
    cov_upper: np.ndarray
    confidence: float
    asset_names: list[str]


@dataclass
class PortfolioRiskDistribution:
    """Distribution of ``sqrt(wᵀ Σ^{(m)} w)`` across imputations."""

    weights: np.ndarray
    portfolio_risks: np.ndarray
    median_risk: float
    percentile_5: float
    percentile_95: float
    mean_risk: float
    n_imputations: int


# ---------------------------------------------------------------------------
# Extractors — turn M imputations into (μ_m, Σ_m)
# ---------------------------------------------------------------------------

def _extract_mu_sigma(imputations: list) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-imputation sample mean and covariance.

    Returns
    -------
    mus : np.ndarray, shape (M, N)
    sigmas : np.ndarray, shape (M, N, N)
    names : list[str]
    """
    if not imputations:
        raise ValueError("imputations is empty")
    names = list(imputations[0].columns)
    M = len(imputations)
    mus = np.empty((M, len(names)), dtype=np.float64)
    sigmas = np.empty((M, len(names), len(names)), dtype=np.float64)
    for m, df in enumerate(imputations):
        arr = df.to_numpy(dtype=np.float64)
        mus[m] = arr.mean(axis=0)
        sigmas[m] = np.cov(arr, rowvar=False, bias=False)
    return mus, sigmas, names


# ---------------------------------------------------------------------------
# Ellipsoidal set for μ
# ---------------------------------------------------------------------------

def ellipsoidal_uncertainty(
    imputations: list,
    *,
    confidence: float = 0.95,
    regularization: float = 1e-12,
) -> EllipsoidalUncertaintySet:
    """Construct the ``χ²``-based ellipsoidal set around μ̄.

    Parameters
    ----------
    imputations : list of pd.DataFrame
    confidence : float
        χ² tail probability — bigger → wider ellipsoid.
    regularization : float
        Added to the diagonal of ``S`` before inversion hygiene.  The set's
        scaling matrix is what's returned; the caller computes the inverse
        for their own optimiser.

    Returns
    -------
    EllipsoidalUncertaintySet
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    mus, _, names = _extract_mu_sigma(imputations)
    M, N = mus.shape
    mu_center = mus.mean(axis=0)
    if M > 1:
        S = np.cov(mus, rowvar=False, bias=False)
    else:
        S = np.zeros((N, N))
    S = 0.5 * (S + S.T) + regularization * np.eye(N)
    kappa = float(np.sqrt(chi2.ppf(confidence, df=N)))
    return EllipsoidalUncertaintySet(
        mu_center=mu_center,
        scaling=S,
        kappa=kappa,
        confidence=confidence,
        asset_names=names,
        n_imputations=M,
    )


# ---------------------------------------------------------------------------
# Box set for μ and Σ
# ---------------------------------------------------------------------------

def box_uncertainty(
    imputations: list,
    *,
    confidence: float = 0.95,
) -> BoxUncertaintySet:
    """Component-wise confidence intervals on μ and on every entry of Σ."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    mus, sigmas, names = _extract_mu_sigma(imputations)
    alpha = 1.0 - confidence
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    mu_lower = np.percentile(mus, lo_q, axis=0)
    mu_upper = np.percentile(mus, hi_q, axis=0)
    cov_lower = np.percentile(sigmas, lo_q, axis=0)
    cov_upper = np.percentile(sigmas, hi_q, axis=0)
    return BoxUncertaintySet(
        mu_lower=mu_lower,
        mu_upper=mu_upper,
        cov_lower=cov_lower,
        cov_upper=cov_upper,
        confidence=confidence,
        asset_names=names,
    )


# ---------------------------------------------------------------------------
# Portfolio risk distribution
# ---------------------------------------------------------------------------

def portfolio_risk_distribution(
    weights: np.ndarray,
    imputations: list,
) -> PortfolioRiskDistribution:
    """``sqrt(wᵀ Σ^{(m)} w)`` for each of the M imputations.

    Parameters
    ----------
    weights : np.ndarray, shape (N,)
    imputations : list of pd.DataFrame

    Returns
    -------
    PortfolioRiskDistribution
    """
    _, sigmas, _ = _extract_mu_sigma(imputations)
    M = len(sigmas)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != sigmas.shape[1]:
        raise ValueError(
            f"weights length {w.shape[0]} != n_assets {sigmas.shape[1]}"
        )
    # Vectorised: for each m, compute w^T Σ_m w
    risks = np.sqrt(np.clip(np.einsum("i,mij,j->m", w, sigmas, w), 0, None))
    return PortfolioRiskDistribution(
        weights=w,
        portfolio_risks=risks,
        median_risk=float(np.median(risks)),
        percentile_5=float(np.percentile(risks, 5)),
        percentile_95=float(np.percentile(risks, 95)),
        mean_risk=float(risks.mean()),
        n_imputations=M,
    )
