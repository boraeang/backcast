"""Gaussian Hidden Markov Model for regime detection and regime-conditional
imputation.

Implemented from scratch (no ``hmmlearn`` dependency).  The E-step uses a
log-space forward-backward pass for numerical stability; the M-step is the
standard Baum-Welch update; regime identifiability is fixed by sorting the
estimated regimes by total volatility (``tr Σ_k``) so regime 0 is always
"calm" and regime ``K-1`` is always the highest-vol regime.

Model selection (K ∈ {2, 3, 4}) uses BIC by default.  Degenerate candidate
fits (NaN/inf parameters, non-convergence, numerical failure, or a state with
too few Viterbi observations to estimate the short-asset covariance) are
rejected during selection; if no candidate survives, the selector falls back to
a single-regime (unconditional) model rather than returning a NaN model.

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

# NOTE: imputation helpers are imported lazily inside
# `compute_regime_params` / `regime_conditional_impute` to avoid a circular
# import (backcast.models.__init__ → regime_hmm → imputation.single_impute →
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
        Selection score per candidate; ``+inf`` for rejected candidates.
    surviving_candidates : list[int]
        Candidates whose fit passed every degeneracy check.
    state_occupancy : np.ndarray, shape (K_best,)
        Viterbi state counts of the chosen model over the full sample.
    overlap_occupancy : np.ndarray or None, shape (K_best,)
        Viterbi state counts of the chosen model restricted to the overlap
        rows (``None`` when no overlap mask was supplied).
    fell_back_to_single_regime : bool
        ``True`` if no candidate survived and the K=1 model was used.
    """

    candidates: list[int]
    results: dict
    best_n_regimes: int
    best: HMMResult
    criterion: str
    scores: dict
    surviving_candidates: list[int] = field(default_factory=list)
    state_occupancy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    overlap_occupancy: Optional[np.ndarray] = None
    fell_back_to_single_regime: bool = False


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


def _floor_covariance(cov: np.ndarray, scale: np.ndarray, min_covar: float) -> np.ndarray:
    """Clip eigenvalues of ``cov`` at ``min_covar`` in standardised units.

    Parameters
    ----------
    cov : np.ndarray, shape (N, N)
        State covariance from the M-step.
    scale : np.ndarray, shape (N,)
        Per-asset sample standard deviations.
    min_covar : float
        Eigenvalue floor for the standardised covariance.

    Returns
    -------
    np.ndarray, shape (N, N)
        ``cov`` unchanged if already above the floor, else the floored matrix.
    """
    if min_covar <= 0:
        return cov
    z = cov / np.outer(scale, scale)
    z = 0.5 * (z + z.T)
    vals, vecs = np.linalg.eigh(z)
    if vals.min() >= min_covar:
        return cov
    z = (vecs * np.clip(vals, min_covar, None)) @ vecs.T
    return z * np.outer(scale, scale)


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
    min_covar: float = 1e-3,
    seed: "int | npr.Generator" = 0,
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
        Absolute value added to the diagonal of each M-step covariance update
        for numerical stability.
    min_covar : float
        Relative covariance floor.  After each M-step, every state covariance
        is expressed in variance-standardised coordinates
        ``D^{-1/2} Σ_k D^{-1/2}`` (``D = diag(Var(X))``) and its eigenvalues
        are clipped below at *min_covar*.  Being relative, it is meaningful on
        daily-return scales (variances ~1e-4, where an absolute 1e-3 floor
        would swamp the data); being a clip rather than an additive term, it
        leaves healthy states at their exact MLE and only lifts collapsing
        ones.  Raise it if states keep collapsing.
    seed : int or np.random.Generator
        Seed (or an existing generator) for the initialisation draw.

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
    scale = np.sqrt(np.atleast_1d(np.var(X_arr, axis=0, ddof=1)))
    scale = np.where(scale > 0, scale, 1.0)

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
            covs_new[k] = _floor_covariance(covs_new[k], scale, min_covar)
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


def _degeneracy_reason(
    res: HMMResult, n_short: int, reject_nonconverged: bool = True,
) -> Optional[str]:
    """Return why a fitted HMM is unusable, or ``None`` if it is usable."""
    if not res.converged:
        if reject_nonconverged:
            return f"did not converge in {res.n_iter} iterations"
        logger.warning(
            "HMM K=%d did not converge in %d iterations — kept for selection "
            "(reject_nonconverged=False); its BIC is conservative.",
            res.n_regimes, res.n_iter,
        )
    for name, arr in (("means", res.means), ("covariances", res.covariances),
                      ("transition_matrix", res.transition_matrix)):
        if np.isnan(arr).any():
            return f"NaN in {name}"
        if np.isinf(arr).any():
            return f"inf in {name}"
    if not np.isfinite(res.log_likelihood):
        return "non-finite log-likelihood"
    counts = np.bincount(res.regime_labels, minlength=res.n_regimes)
    if (counts < n_short + 1).any():
        return (
            f"under-filled state (Viterbi counts {counts.tolist()}, "
            f"need >= {n_short + 1} per state)"
        )
    return None


def _fit_hmm_candidate(
    X: "np.ndarray | pd.DataFrame",
    k: int,
    n_short: int,
    min_covar: float,
    n_iter: int,
    rng: npr.Generator,
    *,
    tolerance: float = 1e-4,
    reject_underfilled_states: bool = True,
    reject_nonconverged: bool = True,
) -> tuple[Optional[HMMResult], float]:
    """Fit a K-state Gaussian HMM; return ``(model, bic)`` or ``(None, inf)``.

    A candidate is rejected (``(None, inf)``) if the fit raises, has NaN/inf
    parameters or log-likelihood, does not converge (when
    *reject_nonconverged*), or — when
    *reject_underfilled_states* — any Viterbi state has fewer than
    ``n_short + 1`` observations (its covariance on the short-asset block
    would not be estimable).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (T, N)
        Fully observed long-history returns.
    k : int
        Number of hidden states.
    n_short : int
        Number of short-history assets.
    min_covar : float
        Relative covariance floor (see :func:`fit_regime_hmm`).
    n_iter : int
        Maximum Baum-Welch iterations.
    rng : np.random.Generator
        Generator for the initialisation draw.
    tolerance : float
        Log-likelihood convergence threshold.
    reject_underfilled_states : bool
        Apply the ``n_short + 1`` occupancy check.
    reject_nonconverged : bool
        Reject fits that hit *n_iter* without converging.  When False they
        compete on BIC (with a WARNING); since their likelihood is not yet
        maximised, their BIC is conservative.

    Returns
    -------
    tuple[HMMResult or None, float]
        The fitted model and its BIC, or ``(None, inf)`` if degenerate.
    """
    try:
        res = fit_regime_hmm(
            X, n_regimes=k, max_iter=n_iter, tolerance=tolerance,
            min_covar=min_covar, seed=rng,
        )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        logger.warning("HMM K=%d rejected: fit raised %s: %s",
                       k, type(exc).__name__, exc)
        return None, float("inf")
    reason = _degeneracy_reason(
        res, n_short if reject_underfilled_states else -1, reject_nonconverged,
    )
    if reason is not None:
        logger.warning("HMM K=%d rejected: %s", k, reason)
        return None, float("inf")
    return res, res.bic


def fit_and_select_hmm(
    X: "np.ndarray | pd.DataFrame",
    n_regimes_candidates: tuple[int, ...] = (2, 3, 4),
    criterion: str = "bic",
    *,
    n_short: int = 0,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    min_covar: float = 1e-3,
    reject_underfilled_states: bool = True,
    reject_nonconverged: bool = True,
    fallback_to_single_regime: bool = True,
    overlap_mask: Optional[np.ndarray] = None,
    seed: int = 0,
) -> HMMSelectionResult:
    """Fit HMMs for each candidate K and select the best non-degenerate one.

    Each candidate is fitted by :func:`_fit_hmm_candidate`; degenerate fits
    score ``+inf`` and so lose selection.  If no candidate survives and
    *fallback_to_single_regime* is set, a K=1 (unconditional) model is used
    and a WARNING is logged.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (T, N)
        Fully observed long-history returns.
    n_regimes_candidates : tuple[int, ...]
        Candidate numbers of states.
    criterion : {"bic", "aic"}
        Selection criterion (lower is better).
    n_short : int
        Number of short-history assets; drives the occupancy check.
    max_iter : int
        Maximum Baum-Welch iterations per candidate.
    tolerance : float
        Log-likelihood convergence threshold.
    min_covar : float
        Relative covariance floor (see :func:`fit_regime_hmm`).
    reject_underfilled_states : bool
        Reject candidates with any state holding ``< n_short + 1`` Viterbi
        observations.
    reject_nonconverged : bool
        Reject candidates that hit *max_iter* without converging.
    fallback_to_single_regime : bool
        Use K=1 when no candidate survives; otherwise raise.
    overlap_mask : np.ndarray of bool, shape (T,), optional
        Rows of *X* belonging to the overlap period; used only to report
        per-state overlap occupancy.
    seed : int
        Seed for each candidate's initialisation generator.

    Returns
    -------
    HMMSelectionResult

    Raises
    ------
    BackcastConvergenceError
        If no candidate survives and *fallback_to_single_regime* is False, or
        the K=1 fallback itself is degenerate.
    """
    from backcast.exceptions import BackcastConvergenceError

    if criterion not in ("bic", "aic"):
        raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion!r}")
    results: dict[int, HMMResult] = {}
    scores: dict[int, float] = {}
    for k in n_regimes_candidates:
        res, bic = _fit_hmm_candidate(
            X, k, n_short, min_covar, max_iter, npr.default_rng(seed),
            tolerance=tolerance, reject_underfilled_states=reject_underfilled_states,
            reject_nonconverged=reject_nonconverged,
        )
        if res is None:
            scores[k] = float("inf")
            continue
        results[k] = res
        scores[k] = bic if criterion == "bic" else res.aic
        logger.info("HMM K=%d:  log-L=%.1f  BIC=%.1f  AIC=%.1f",
                    k, res.log_likelihood, res.bic, res.aic)

    surviving = [k for k in n_regimes_candidates if k in results]
    fell_back = False
    if surviving:
        best_k = min(surviving, key=lambda k: scores[k])
    else:
        if not fallback_to_single_regime:
            raise BackcastConvergenceError(
                f"No HMM candidate in {list(n_regimes_candidates)} survived "
                "degeneracy checks and fallback_to_single_regime is disabled."
            )
        logger.warning(
            "No HMM candidate in %s survived degeneracy checks — falling back "
            "to K=1 (single-regime / unconditional model).",
            list(n_regimes_candidates),
        )
        # K=1 is closed-form (sample mean / covariance) after one M-step, so
        # the convergence flag is irrelevant; only NaN/inf can disqualify it.
        res1, bic1 = _fit_hmm_candidate(
            X, 1, n_short, min_covar, max(max_iter, 3), npr.default_rng(seed),
            tolerance=tolerance, reject_underfilled_states=False,
            reject_nonconverged=False,
        )
        if res1 is None:
            raise BackcastConvergenceError("K=1 fallback HMM fit is degenerate.")
        best_k = 1
        results[1] = res1
        scores[1] = bic1 if criterion == "bic" else res1.aic
        fell_back = True

    best = results[best_k]
    occupancy = np.bincount(best.regime_labels, minlength=best_k)
    overlap_occ: Optional[np.ndarray] = None
    if overlap_mask is not None:
        overlap_mask = np.asarray(overlap_mask, dtype=bool)
        if overlap_mask.shape != best.regime_labels.shape:
            raise ValueError(
                f"overlap_mask shape {overlap_mask.shape} != labels shape "
                f"{best.regime_labels.shape}"
            )
        overlap_occ = np.bincount(best.regime_labels[overlap_mask], minlength=best_k)
    logger.info(
        "HMM selected K=%d (surviving %s); state occupancy %s; overlap occupancy %s",
        best_k, surviving, occupancy.tolist(),
        None if overlap_occ is None else overlap_occ.tolist(),
    )
    return HMMSelectionResult(
        candidates=list(n_regimes_candidates),
        results=results,
        best_n_regimes=best_k,
        best=best,
        criterion=criterion,
        scores=scores,
        surviving_candidates=surviving,
        state_occupancy=occupancy,
        overlap_occupancy=overlap_occ,
        fell_back_to_single_regime=fell_back,
    )


# ---------------------------------------------------------------------------
# Regime-conditional imputation
# ---------------------------------------------------------------------------

def compute_regime_params(
    returns: pd.DataFrame,
    regime_labels: np.ndarray,
    *,
    short_assets: Optional[list[str]] = None,
    regimes: Optional[list[int]] = None,
    reliable_threshold: Optional[int] = None,
    shrinkage: "str | float | None" = "auto",
    fallback_to_pooled: bool = True,
    psd_epsilon: float = 1e-10,
    shrink_mean: bool = True,
    min_obs_per_regime: Optional[int] = None,
) -> dict:
    """Per-regime ``(mu, sigma)`` in the legacy dict format.

    Thin wrapper over
    :func:`backcast.imputation.regime_params.build_regime_params`; thin
    regimes are shrunk or pooled rather than dropped, so every regime gets
    usable parameters.  New code should call ``build_regime_params``
    directly.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        The *overlap* matrix (rows where every column is observed).
    regime_labels : np.ndarray, shape (T,)
        Regime label for each row of ``returns``.
    short_assets : list[str], optional
        Short-history columns.  ``None`` treats every column as short, i.e. a
        regime needs ``n_k > N`` rows to avoid pooling.
    regimes : list[int], optional
        All regimes needing parameters (default: those in *regime_labels*).
    reliable_threshold, shrinkage, fallback_to_pooled, psd_epsilon, shrink_mean
        See ``build_regime_params``.
    min_obs_per_regime : int, optional
        DEPRECATED.  Mapped to *reliable_threshold* with a warning.

    Returns
    -------
    dict[int, dict]
        ``{'mu': (N,), 'sigma': (N, N), 'n_obs': int, 'source': str}`` per
        regime.
    """
    from backcast.imputation.regime_params import build_regime_params

    if min_obs_per_regime is not None:
        logger.warning(
            "DEPRECATION: min_obs_per_regime is no longer a hard cutoff; "
            "mapping it to reliable_threshold=%d.", min_obs_per_regime,
        )
        if reliable_threshold is None:
            reliable_threshold = int(min_obs_per_regime)
    params = build_regime_params(
        returns, regime_labels,
        list(returns.columns) if short_assets is None else short_assets,
        regimes=regimes, reliable_threshold=reliable_threshold,
        shrinkage=shrinkage, fallback_to_pooled=fallback_to_pooled,
        psd_epsilon=psd_epsilon, shrink_mean=shrink_mean,
    )
    return {
        k: {"mu": p.mu, "sigma": p.sigma, "n_obs": p.n_obs, "source": p.source}
        for k, p in params.items()
    }


def regime_conditional_impute(
    returns: pd.DataFrame,
    regime_labels: np.ndarray,
    regime_params: dict,
) -> pd.DataFrame:
    """Fill each NaN cell with the **regime-conditional** mean.

    Delegates to
    :func:`backcast.imputation.single_impute.regime_single_impute`.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix with NaN for missing entries.
    regime_labels : np.ndarray, shape (T,)
    regime_params : dict[int, RegimeParams or dict]

    Returns
    -------
    pd.DataFrame
        Filled returns — same index/columns as input, no NaNs.

    Raises
    ------
    ValueError
        If *regime_labels* length does not match *returns* rows.
    BackcastDataError
        If a row needing imputation has a regime without parameters.
    """
    from backcast.imputation.single_impute import regime_single_impute

    return regime_single_impute(returns, regime_labels, regime_params)
