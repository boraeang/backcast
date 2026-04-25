"""Gaussian Hidden Markov Model for regime detection and regime-conditional
imputation.

Implemented from scratch (no ``hmmlearn`` dependency).  The E-step uses a
log-space forward-backward pass for numerical stability; the M-step is the
standard Baum-Welch update; regime identifiability is fixed by sorting the
estimated regimes by total volatility (``tr Σ_k``) so regime 0 is always
"calm" and regime ``K-1`` is always the highest-vol regime.

Model selection (K ∈ {2, 3, 4}) uses BIC by default.

References
----------
- Hamilton, J.D. (1989).  "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2).
- Rabiner, L.R. (1989).  "A Tutorial on Hidden Markov Models."  *Proc. IEEE*.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.random as npr
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

# NOTE: `_fill_rows_conditional` is imported lazily inside
# `regime_conditional_impute` to avoid a circular import
# (backcast.models.__init__ → regime_hmm → imputation.single_impute →
#  models.em_stambaugh → back into models.__init__).

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HMMResult:
    """Output of :func:`fit_regime_hmm`.

    Attributes
    ----------
    n_regimes : int
    initial_probs : np.ndarray, shape (K,)
    transition_matrix : np.ndarray, shape (K, K)
    means : np.ndarray, shape (K, N)
    covariances : np.ndarray, shape (K, N, N)
    regime_labels : np.ndarray, shape (T,)
        Viterbi-decoded most-likely state at each time.
    posterior : np.ndarray, shape (T, K)
        Smoothed posterior γ_t(k) = P(s_t = k | X, θ̂).
    log_likelihood : float
    n_iter : int
    converged : bool
    ll_trace : list[float]
    asset_order : list[str]
    bic : float
    aic : float
    """

    n_regimes: int
    initial_probs: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    regime_labels: np.ndarray
    posterior: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool
    ll_trace: list[float]
    asset_order: list[str]
    bic: float
    aic: float


@dataclass
class HMMSelectionResult:
    """Output of :func:`fit_and_select_hmm`.

    Attributes
    ----------
    candidates : list[int]
    results : dict[int, HMMResult]
    best_n_regimes : int
    best : HMMResult
    criterion : str
    scores : dict[int, float]
    """

    candidates: list[int]
    results: dict
    best_n_regimes: int
    best: HMMResult
    criterion: str
    scores: dict


# ---------------------------------------------------------------------------
# Emission log-density
# ---------------------------------------------------------------------------

def _log_multivariate_normal(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Row-wise ``log N(X_t | mean, cov)`` via Cholesky."""
    N = len(mean)
    L = np.linalg.cholesky(cov)
    diff = (X - mean).T          # shape (N, T)
    z = solve_triangular(L, diff, lower=True)
    mahal = np.einsum("it,it->t", z, z)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (N * np.log(2.0 * np.pi) + log_det + mahal)


def _log_emissions(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Per-regime log emission probabilities, shape ``(T, K)``."""
    T, _ = X.shape
    K = means.shape[0]
    out = np.empty((T, K), dtype=np.float64)
    for k in range(K):
        out[:, k] = _log_multivariate_normal(X, means[k], covs[k])
    return out


# ---------------------------------------------------------------------------
# Forward-backward + Viterbi (log space)
# ---------------------------------------------------------------------------

def _forward_backward_log(
    log_pi: np.ndarray, log_A: np.ndarray, log_emiss: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Log-space forward-backward.

    Returns
    -------
    log_alpha : np.ndarray, shape (T, K)
    log_beta  : np.ndarray, shape (T, K)
    log_p_x   : float
    """
    T, K = log_emiss.shape
    log_alpha = np.empty((T, K), dtype=np.float64)
    log_beta = np.empty((T, K), dtype=np.float64)

    log_alpha[0] = log_pi + log_emiss[0]
    for t in range(1, T):
        # logsumexp over i: log_alpha[t, k] = lse_i(log_alpha[t-1, i] + log_A[i, k]) + log_emiss[t, k]
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_emiss[t]

    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t, k] = lse_j(log_A[k, j] + log_emiss[t+1, j] + log_beta[t+1, j])
        log_beta[t] = logsumexp(log_A + log_emiss[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

    log_p_x = float(logsumexp(log_alpha[T - 1]))
    return log_alpha, log_beta, log_p_x


def _viterbi_log(
    log_pi: np.ndarray, log_A: np.ndarray, log_emiss: np.ndarray
) -> np.ndarray:
    """Most-likely state sequence via log-space Viterbi."""
    T, K = log_emiss.shape
    delta = np.empty((T, K), dtype=np.float64)
    psi = np.empty((T, K), dtype=np.int64)
    delta[0] = log_pi + log_emiss[0]
    for t in range(1, T):
        # scores[i, j] = delta[t-1, i] + log_A[i, j]
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(K)] + log_emiss[t]
    path = np.empty(T, dtype=np.int64)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# ---------------------------------------------------------------------------
# Initialisation + canonicalisation
# ---------------------------------------------------------------------------

def _initial_params(X: np.ndarray, n_regimes: int, rng: npr.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialise (π, A, means, covs) for Baum-Welch."""
    T, N = X.shape
    pi = np.full(n_regimes, 1.0 / n_regimes, dtype=np.float64)
    A = np.full((n_regimes, n_regimes), 0.1 / max(n_regimes - 1, 1), dtype=np.float64)
    np.fill_diagonal(A, 0.9)
    A /= A.sum(axis=1, keepdims=True)

    idx = rng.choice(T, size=n_regimes, replace=False)
    means = X[idx].astype(np.float64).copy()

    sample_cov = np.cov(X, rowvar=False, bias=False)
    if sample_cov.ndim == 0:
        sample_cov = np.array([[float(sample_cov)]])
    covs = np.tile(sample_cov[np.newaxis, :, :], (n_regimes, 1, 1)).astype(np.float64)
    return pi, A, means, covs


def _canonicalise(
    pi: np.ndarray, A: np.ndarray, means: np.ndarray, covs: np.ndarray,
    posterior: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort regimes by total variance so regime 0 is the lowest-vol regime."""
    K = len(pi)
    traces = np.array([np.trace(covs[k]) for k in range(K)])
    order = np.argsort(traces)                     # old index -> rank
    inv = np.argsort(order)                        # rank -> old index
    pi = pi[order]
    A = A[np.ix_(order, order)]
    means = means[order]
    covs = covs[order]
    posterior = posterior[:, order]
    labels = inv[labels]
    return pi, A, means, covs, posterior, labels


# ---------------------------------------------------------------------------
# Public fit routines
# ---------------------------------------------------------------------------

def fit_regime_hmm(
    X: "np.ndarray | pd.DataFrame",
    n_regimes: int,
    *,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    cov_regularization: float = 1e-10,
    seed: int = 0,
) -> HMMResult:
    """Fit a Gaussian HMM via log-space Baum-Welch.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape (T, N)
        Fully observed data.  NaN is NOT supported.
    n_regimes : int
    max_iter : int
    tolerance : float
        Absolute convergence threshold on the log-likelihood.
    cov_regularization : float
        Value added to the diagonal of each M-step covariance update for
        numerical stability.
    seed : int

    Returns
    -------
    HMMResult

    Notes
    -----
    The returned regimes are canonicalised so that regime 0 has the smallest
    ``tr Σ_k`` (interpretable as the "calm" regime).
    """
    if hasattr(X, "values"):
        asset_order = list(X.columns)           # type: ignore[attr-defined]
        X_arr = np.ascontiguousarray(X.values, dtype=np.float64)
    else:
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        asset_order = [f"col{i}" for i in range(X_arr.shape[1])]
    if np.isnan(X_arr).any():
        raise ValueError("fit_regime_hmm requires a fully-observed X (no NaN)")
    T, N = X_arr.shape
    rng = npr.default_rng(seed)

    pi, A, means, covs = _initial_params(X_arr, n_regimes, rng)
    ll_trace: list[float] = []
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        log_pi = np.log(pi + 1e-300)
        log_A = np.log(A + 1e-300)
        log_emiss = _log_emissions(X_arr, means, covs)

        log_alpha, log_beta, log_p_x = _forward_backward_log(log_pi, log_A, log_emiss)
        ll_trace.append(log_p_x)

        # Posterior γ and pair-posterior ξ
        log_gamma = log_alpha + log_beta - log_p_x
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # log_xi[t, i, j] = log_alpha[t, i] + log_A[i, j] + log_emiss[t+1, j] + log_beta[t+1, j] - log_p_x
        log_xi_all = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + log_emiss[1:, None, :]
            + log_beta[1:, None, :]
            - log_p_x
        )
        xi_sum = np.exp(logsumexp(log_xi_all, axis=0))  # (K, K)

        # M-step
        pi_new = gamma[0].copy()
        denom = gamma[:-1].sum(axis=0)[:, None]
        denom = np.where(denom == 0, 1.0, denom)
        A_new = xi_sum / denom
        A_new /= A_new.sum(axis=1, keepdims=True)

        total_weight = gamma.sum(axis=0)            # (K,)
        total_safe = np.where(total_weight == 0, 1.0, total_weight)
        means_new = (gamma.T @ X_arr) / total_safe[:, None]
        covs_new = np.empty_like(covs)
        for k in range(n_regimes):
            centered = X_arr - means_new[k]
            covs_new[k] = (
                (gamma[:, k][:, None, None] * centered[:, :, None] * centered[:, None, :])
                .sum(axis=0)
                / total_safe[k]
            )
            covs_new[k] += cov_regularization * np.eye(N)

        # Convergence check (increase in log-likelihood)
        if it > 1 and abs(ll_trace[-1] - ll_trace[-2]) < tolerance:
            converged = True
            pi, A, means, covs = pi_new, A_new, means_new, covs_new
            break
        pi, A, means, covs = pi_new, A_new, means_new, covs_new

    # Final forward-backward for posterior/Viterbi on the converged parameters
    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_emiss = _log_emissions(X_arr, means, covs)
    log_alpha, log_beta, log_p_x = _forward_backward_log(log_pi, log_A, log_emiss)
    log_gamma = log_alpha + log_beta - log_p_x
    posterior = np.exp(log_gamma)
    posterior /= posterior.sum(axis=1, keepdims=True)

    labels = _viterbi_log(log_pi, log_A, log_emiss)
    pi, A, means, covs, posterior, labels = _canonicalise(
        pi, A, means, covs, posterior, labels,
    )

    # BIC / AIC
    n_params = (n_regimes - 1) + n_regimes * (n_regimes - 1) + n_regimes * N + n_regimes * N * (N + 1) // 2
    bic = -2.0 * log_p_x + n_params * np.log(T)
    aic = -2.0 * log_p_x + 2.0 * n_params

    return HMMResult(
        n_regimes=n_regimes,
        initial_probs=pi,
        transition_matrix=A,
        means=means,
        covariances=covs,
        regime_labels=labels,
        posterior=posterior,
        log_likelihood=log_p_x,
        n_iter=it,
        converged=converged,
        ll_trace=ll_trace,
        asset_order=asset_order,
        bic=float(bic),
        aic=float(aic),
    )


def fit_and_select_hmm(
    X,
    n_regimes_candidates: tuple[int, ...] = (2, 3, 4),
    criterion: str = "bic",
    **fit_kwargs,
) -> HMMSelectionResult:
    """Fit HMMs for each candidate K, return best by BIC (default) or AIC."""
    if criterion not in ("bic", "aic"):
        raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion!r}")
    results: dict[int, HMMResult] = {}
    scores: dict[int, float] = {}
    for k in n_regimes_candidates:
        res = fit_regime_hmm(X, n_regimes=k, **fit_kwargs)
        results[k] = res
        scores[k] = res.bic if criterion == "bic" else res.aic
        logger.info("HMM K=%d:  log-L=%.1f  BIC=%.1f  AIC=%.1f",
                    k, res.log_likelihood, res.bic, res.aic)
    best_k = min(scores, key=lambda k: scores[k])
    return HMMSelectionResult(
        candidates=list(n_regimes_candidates),
        results=results,
        best_n_regimes=best_k,
        best=results[best_k],
        criterion=criterion,
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Regime-conditional imputation
# ---------------------------------------------------------------------------

def compute_regime_params(
    returns: pd.DataFrame,
    regime_labels: np.ndarray,
    *,
    min_obs_per_regime: int = 30,
) -> dict:
    """Per-regime ``(mu, sigma)`` from fully-observed rows.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        The *overlap* matrix (rows where every column is observed).
    regime_labels : np.ndarray, shape (T,)
        Regime label for each row of ``returns``.
    min_obs_per_regime : int
        Regimes with fewer than this many observed rows are dropped — their
        μ/Σ aren't estimable reliably.

    Returns
    -------
    dict[int, dict]
        Keys are regime indices; values are ``{'mu': np.ndarray (N,),
        'sigma': np.ndarray (N, N), 'n_obs': int}``.
    """
    if len(regime_labels) != len(returns):
        raise ValueError(
            f"regime_labels length {len(regime_labels)} != returns rows {len(returns)}"
        )
    if returns.isna().any().any():
        raise ValueError("returns must be fully observed to estimate regime params")
    R = returns.to_numpy(dtype=np.float64)
    out: dict[int, dict] = {}
    for k in np.unique(regime_labels):
        mask = regime_labels == k
        n_k = int(mask.sum())
        if n_k < min_obs_per_regime:
            logger.warning(
                "Regime %d has only %d observed rows — skipping (need %d).",
                k, n_k, min_obs_per_regime,
            )
            continue
        block = R[mask]
        out[int(k)] = {
            "mu": block.mean(axis=0),
            "sigma": np.cov(block, rowvar=False, bias=False),
            "n_obs": n_k,
        }
    return out


def regime_conditional_impute(
    returns: pd.DataFrame,
    regime_labels: np.ndarray,
    regime_params: dict,
) -> pd.DataFrame:
    """Fill each NaN cell with the **regime-conditional** mean.

    For row ``t`` with regime label ``s_t``, missing columns are filled with
    their conditional expectation given the observed columns under the
    parameters ``(μ^{(s_t)}, Σ^{(s_t)})``.  Rows whose regime lacks estimated
    parameters (see *min_obs_per_regime*) are left unfilled.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix with NaN for missing entries.
    regime_labels : np.ndarray, shape (T,)
    regime_params : dict[int, {'mu': ..., 'sigma': ...}]
        From :func:`compute_regime_params`.

    Returns
    -------
    pd.DataFrame
        Filled returns — same index/columns as input.  Any row whose regime
        has no params will retain its NaNs.

    Raises
    ------
    ValueError
        If *regime_labels* length does not match *returns* rows.
    """
    if len(regime_labels) != len(returns):
        raise ValueError(
            f"regime_labels length {len(regime_labels)} != returns rows {len(returns)}"
        )
    from backcast.imputation.single_impute import _fill_rows_conditional
    R = returns.to_numpy(dtype=np.float64, copy=True)
    for k, params in regime_params.items():
        mask = regime_labels == k
        if not mask.any():
            continue
        R_k = R[mask]
        _fill_rows_conditional(R_k, params["mu"], params["sigma"])
        R[mask] = R_k
    return pd.DataFrame(R, index=returns.index, columns=returns.columns)
