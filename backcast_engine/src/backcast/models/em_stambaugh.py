"""Stambaugh (1997) EM algorithm for an incomplete returns matrix.

References
----------
Stambaugh, R.F. (1997). "Analyzing investments whose histories differ in
length." *Journal of Financial Economics*, 45(3), 285-331.

Algorithm
---------
The data matrix :math:`R` (shape ``T x N``) has monotone missingness.  Partition
the columns of each row into observed (index set :math:`O_t`) and missing
(index set :math:`M_t`).  Given current parameter estimates
:math:`(\\mu, \\Sigma)`:

E-step (per row with missing values)
    Partition :math:`\\mu` into :math:`(\\mu_O, \\mu_M)` and :math:`\\Sigma`
    into ``[[S_OO, S_OM], [S_MO, S_MM]]``.  The conditional distribution of
    :math:`R_{M,t} | R_{O,t}` is Gaussian with

        β = S_MO  S_OO^{-1}              (shape |M| × |O|)
        α = μ_M - β · μ_O                (shape |M|)
        Σ_{M|O} = S_MM - β · S_OM         (conditional covariance)
        E[R_{M,t} | R_{O,t}] = α + β · R_{O,t}

    Rows with the same observed/missing pattern share the same β, α,
    Σ_{M|O}, so the algorithm groups rows by pattern and solves one Cholesky
    system per pattern per iteration.

M-step
    The sufficient statistics of a Gaussian are
    :math:`\\sum_t R_t` and :math:`\\sum_t R_t R_t^\\top`.  After filling in
    the conditional mean for every missing entry:

        μ ← (1/T) Σ_t R_t^filled
        Σ ← (1/T) Σ_t R_t^filled R_t^{filled,\\top} - μ μ^\\top
              + variance-correction term

    The correction adds :math:`\\Sigma_{M|O}` to the ``(M, M)`` block of the
    sufficient statistic for every missing row — without it, the imputed
    entries are treated as if they were noise-free, biasing the covariance
    downward (the classic naive-EM mistake).

After convergence, the returned ``conditional_params`` provides β, α, and
Σ_{M|O} for the canonical long-vs-short partition, ready for consumption by
:mod:`backcast.imputation.single_impute`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ConditionalParams:
    """Conditional distribution of missing columns given observed columns.

    Attributes
    ----------
    observed_cols : np.ndarray, shape (|O|,)
        Column indices observed.
    missing_cols : np.ndarray, shape (|M|,)
        Column indices missing.
    beta : np.ndarray, shape (|M|, |O|)
        Conditional regression coefficient matrix ``Σ_MO · Σ_OO^{-1}``.
    alpha : np.ndarray, shape (|M|,)
        Conditional intercept ``μ_M - β · μ_O``.
    cond_cov : np.ndarray, shape (|M|, |M|)
        Conditional covariance ``Σ_MM - β · Σ_OM`` (PSD).
    """

    observed_cols: np.ndarray
    missing_cols: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray
    cond_cov: np.ndarray


@dataclass
class EMResult:
    """Output of :func:`em_stambaugh`.

    Attributes
    ----------
    mu : np.ndarray, shape (N,)
        Estimated mean vector.
    sigma : np.ndarray, shape (N, N)
        Estimated covariance matrix (PSD, symmetric).
    conditional_params : ConditionalParams
        Conditional distribution for the long/short partition (suitable for
        single or multiple imputation).
    n_iter : int
        Number of EM iterations performed.
    converged : bool
        Whether the Frobenius tolerance was met before ``max_iter``.
    final_delta : float
        Frobenius norm of the final sigma update.
    log_likelihood_trace : list[float]
        Observed-data log-likelihood at each iteration.
    asset_order : list[str]
        Column names corresponding to the mu/sigma ordering.
    """

    mu: np.ndarray
    sigma: np.ndarray
    conditional_params: ConditionalParams
    n_iter: int
    converged: bool
    final_delta: float
    log_likelihood_trace: list[float]
    asset_order: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by eigenvalue clipping."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, epsilon)
    out = eigvecs @ np.diag(clipped) @ eigvecs.T
    return 0.5 * (out + out.T)


def _pairwise_initial_covariance(R: np.ndarray) -> np.ndarray:
    """Pairwise-complete covariance for initialisation.

    For each pair ``(i, j)`` use rows where both columns are observed.  This
    may be non-PSD, so the result is projected to the nearest PSD matrix.
    """
    T, N = R.shape
    mu = np.nanmean(R, axis=0)
    cov = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            mask = ~(np.isnan(R[:, i]) | np.isnan(R[:, j]))
            if mask.sum() < 2:
                cov[i, j] = 0.0 if i != j else 1e-6
            else:
                x = R[mask, i] - mu[i]
                y = R[mask, j] - mu[j]
                cov[i, j] = float((x * y).sum() / (mask.sum() - 1))
            cov[j, i] = cov[i, j]
    return _nearest_psd(cov)


def _partition_by_pattern(R: np.ndarray) -> dict[tuple, np.ndarray]:
    """Group row indices by their missingness pattern (tuple of bool)."""
    T, _N = R.shape
    nan_mask = np.isnan(R)
    # Encode each row's pattern as bytes for fast grouping
    row_bytes = np.ascontiguousarray(nan_mask).view(np.uint8).reshape(T, -1)
    # Use a dict keyed by the bytes representation
    groups: dict[bytes, list[int]] = {}
    for t in range(T):
        key = row_bytes[t].tobytes()
        groups.setdefault(key, []).append(t)
    # Convert back to tuple keys (hashable) and numpy arrays of row indices
    out: dict[tuple, np.ndarray] = {}
    for key, rows in groups.items():
        pattern = tuple(nan_mask[rows[0]].tolist())
        out[pattern] = np.asarray(rows, dtype=np.int64)
    return out


def _solve_conditional(
    mu: np.ndarray,
    sigma: np.ndarray,
    observed_cols: np.ndarray,
    missing_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(beta, alpha, cond_cov)`` for the (obs, mis) partition.

    Uses a Cholesky factor of ``sigma[obs, obs]`` for stability.
    """
    S_OO = sigma[np.ix_(observed_cols, observed_cols)]
    S_OM = sigma[np.ix_(observed_cols, missing_cols)]
    S_MM = sigma[np.ix_(missing_cols, missing_cols)]

    L, low = cho_factor(S_OO, lower=True)
    # β = S_MO · inv(S_OO) — solve S_OO · X = S_OM, then β = X.T
    X = cho_solve((L, low), S_OM)          # shape (|O|, |M|)
    beta = X.T                              # shape (|M|, |O|)
    cond_cov = S_MM - beta @ S_OM           # |M| × |M|
    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    alpha = mu[missing_cols] - beta @ mu[observed_cols]
    return beta, alpha, cond_cov


def _observed_loglikelihood(
    R: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    patterns: dict[tuple, np.ndarray],
) -> float:
    """Compute the marginal log-likelihood of the observed entries."""
    ll = 0.0
    log_2pi = float(np.log(2.0 * np.pi))
    for pattern, rows in patterns.items():
        missing_mask = np.asarray(pattern, dtype=bool)
        obs_cols = np.where(~missing_mask)[0]
        if len(obs_cols) == 0:
            continue
        S_OO = sigma[np.ix_(obs_cols, obs_cols)]
        mu_O = mu[obs_cols]
        sign, logdet = slogdet(S_OO)
        if sign <= 0:
            # S_OO not PSD — numerically degenerate; skip this term
            continue
        L, low = cho_factor(S_OO, lower=True)
        diff = R[np.ix_(rows, obs_cols)] - mu_O
        inv_diff = cho_solve((L, low), diff.T)
        quad = float(np.einsum("ij,ji->i", diff, inv_diff).sum())
        n_rows = len(rows)
        k = len(obs_cols)
        ll += -0.5 * (n_rows * k * log_2pi + n_rows * logdet + quad)
    return ll


# ---------------------------------------------------------------------------
# Public EM entry point
# ---------------------------------------------------------------------------

def em_stambaugh(
    returns: "np.ndarray | pd.DataFrame",
    *,
    long_asset_indices: Optional[np.ndarray] = None,
    short_asset_indices: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tolerance: float = 1e-8,
    psd_epsilon: float = 1e-10,
    track_loglikelihood: bool = True,
) -> EMResult:
    """Estimate ``(μ, Σ)`` by Stambaugh EM on a monotone-missing returns matrix.

    Parameters
    ----------
    returns : np.ndarray or pd.DataFrame, shape (T, N)
        Returns matrix.  Missing entries are NaN.
    long_asset_indices, short_asset_indices : np.ndarray or None
        Column-index arrays identifying the long-vs-short split used to build
        the returned :class:`ConditionalParams`.  When None they are inferred
        from the missingness pattern (columns without any NaN are long).
    max_iter : int
        Maximum number of EM iterations.
    tolerance : float
        Convergence threshold on ``||Σ_{k+1} - Σ_k||_F``.
    psd_epsilon : float
        Eigenvalue floor used by the PSD projection after every M-step.
    track_loglikelihood : bool
        If True, compute and store the observed-data log-likelihood each
        iteration.  Turn off for speed.

    Returns
    -------
    EMResult

    Notes
    -----
    - The E-step groups rows by missingness pattern, so per-iteration cost is
      ``O(|patterns| · (|O|^3 + |M| · T_pattern))``.  For typical monotone
      data there are ``O(N)`` distinct patterns.
    - Σ is re-projected to the nearest PSD matrix after each M-step to guard
      against tiny negative eigenvalues from finite precision.
    """
    if hasattr(returns, "values"):
        asset_order = list(returns.columns)   # type: ignore[attr-defined]
        R = np.ascontiguousarray(returns.values, dtype=np.float64)
    else:
        R = np.ascontiguousarray(returns, dtype=np.float64)
        asset_order = [f"col{i}" for i in range(R.shape[1])]

    T, N = R.shape
    if T < 2:
        raise ValueError(f"Need at least 2 rows; got {T}")

    # Initialisation
    mu = np.nanmean(R, axis=0)
    if np.any(np.isnan(mu)):
        raise ValueError("At least one column is entirely NaN")
    sigma = _pairwise_initial_covariance(R)

    patterns = _partition_by_pattern(R)

    converged = False
    final_delta = float("inf")
    ll_trace: list[float] = []

    if track_loglikelihood:
        ll_trace.append(_observed_loglikelihood(R, mu, sigma, patterns))

    for it in range(1, max_iter + 1):
        sum_R = np.zeros(N, dtype=np.float64)
        sum_RRT = np.zeros((N, N), dtype=np.float64)

        for pattern, rows in patterns.items():
            missing_mask = np.asarray(pattern, dtype=bool)
            obs_cols = np.where(~missing_mask)[0]
            mis_cols = np.where(missing_mask)[0]
            n_rows = len(rows)

            if len(mis_cols) == 0:
                # Fully observed — no imputation for this row group
                block = R[rows]
                sum_R += block.sum(axis=0)
                sum_RRT += block.T @ block
                continue

            if len(obs_cols) == 0:
                # All missing — contribute μ μ^T + Σ per row
                sum_R += n_rows * mu
                sum_RRT += n_rows * (np.outer(mu, mu) + sigma)
                continue

            beta, alpha, cond_cov = _solve_conditional(mu, sigma, obs_cols, mis_cols)
            obs_data = R[np.ix_(rows, obs_cols)]                # (n_rows, |O|)
            cond_mean = alpha + obs_data @ beta.T                # (n_rows, |M|)

            full = np.empty((n_rows, N), dtype=np.float64)
            full[:, obs_cols] = obs_data
            full[:, mis_cols] = cond_mean

            sum_R += full.sum(axis=0)
            sum_RRT += full.T @ full
            # Variance correction: add n_rows * Σ_{M|O} to the (M, M) block
            sum_RRT[np.ix_(mis_cols, mis_cols)] += n_rows * cond_cov

        mu_new = sum_R / T
        sigma_new = sum_RRT / T - np.outer(mu_new, mu_new)
        sigma_new = _nearest_psd(sigma_new, epsilon=psd_epsilon)

        final_delta = float(np.linalg.norm(sigma_new - sigma, "fro"))
        mu, sigma = mu_new, sigma_new

        if track_loglikelihood:
            ll_trace.append(_observed_loglikelihood(R, mu, sigma, patterns))

        if final_delta < tolerance:
            converged = True
            logger.info("EM converged at iteration %d (ΔΣ = %.3e)", it, final_delta)
            break
    else:
        logger.warning(
            "EM reached max_iter=%d without converging (final ΔΣ = %.3e)",
            max_iter, final_delta,
        )

    # Build conditional params for the long/short split (default: all no-NaN cols)
    if long_asset_indices is None or short_asset_indices is None:
        long_asset_indices = np.asarray(
            [i for i in range(N) if not np.isnan(R[:, i]).any()], dtype=np.int64,
        )
        short_asset_indices = np.asarray(
            [i for i in range(N) if np.isnan(R[:, i]).any()], dtype=np.int64,
        )
    if len(short_asset_indices) > 0 and len(long_asset_indices) > 0:
        beta, alpha, cond_cov = _solve_conditional(
            mu, sigma, long_asset_indices, short_asset_indices,
        )
    else:
        beta = np.zeros((len(short_asset_indices), len(long_asset_indices)))
        alpha = np.zeros(len(short_asset_indices))
        cond_cov = np.zeros((len(short_asset_indices), len(short_asset_indices)))
    cond_params = ConditionalParams(
        observed_cols=np.asarray(long_asset_indices, dtype=np.int64),
        missing_cols=np.asarray(short_asset_indices, dtype=np.int64),
        beta=beta,
        alpha=alpha,
        cond_cov=cond_cov,
    )

    return EMResult(
        mu=mu,
        sigma=sigma,
        conditional_params=cond_params,
        n_iter=it,
        converged=converged,
        final_delta=final_delta,
        log_likelihood_trace=ll_trace,
        asset_order=asset_order,
    )
