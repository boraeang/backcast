# Claude Code Prompt — Fix HMM NaN & Regime Fallback in the Backcast Engine

## Context

The backcast engine's regime-conditional imputation path is producing two
related bugs:

1. **NaN values in imputed output** that break downstream (covariance,
   optimization, backtest).
2. **Unimputed backcast periods** when `min_obs_per_regime` is set (e.g. 15):
   any regime failing that hard gate gets no parameters, so every backcast date
   assigned to it stays NaN / unfilled.

Root cause: regimes that are active in the backcast period but rare or absent in
the overlap period cannot produce a non-singular conditional covariance. A hard
observation gate turns "thin regime" into "no output" instead of degrading
gracefully. Separately, the HMM fit itself can emit NaN parameters when a
Gaussian state collapses during EM.

Fix both, across `models/regime_hmm.py`, `imputation/multiple_impute.py`,
`imputation/single_impute.py`, and the config. Follow the existing code style in
the repo (type hints, NumPy-style docstrings, `numpy.random.Generator`,
`scipy.linalg.cho_factor`/`cho_solve`, no `np.linalg.inv`, logging module). Read
`docs/backcast_prompt.md` first for the module conventions.

## Part 1 — Harden the HMM fit (`models/regime_hmm.py`)

Replace the current fit-and-select logic so that a NaN or degenerate fit is
*rejected during model selection* rather than propagated.

Requirements:

1. Fit each candidate `K` with `covariance_type="full"` and a `min_covar` floor
   (expose it in config, default `1e-3`; raise it if states keep collapsing).
2. After each fit, reject the candidate (score it `+inf` for BIC / `-inf` log-lik,
   so it loses selection) if ANY of:
   - `np.isnan(model.means_).any()` or `np.isnan(model.covars_).any()`
   - `np.isinf(...)` on either
   - the Viterbi state occupancy has any state with
     `count < n_short + 1` (over-parameterized — covariance not estimable for
     that state on the short-asset block)
   - the fit raised or did not converge
3. The selector then naturally steps down to a `K` the data supports. If NO `K`
   in the candidate list survives, fall back to `K=1` (degenerate single-regime =
   unconditional model) and log a WARNING — never return a NaN model.
4. Return, alongside the chosen model: the per-state overlap occupancy counts and
   the surviving-`K` list, for diagnostics.

Implement a helper:

```python
def _fit_hmm_candidate(
    X: np.ndarray,
    k: int,
    n_short: int,
    min_covar: float,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[object | None, float]:
    """Fit a K-state Gaussian HMM; return (model, bic) or (None, inf) if the
    fit is degenerate (NaN/inf params, or any state with < n_short+1 obs)."""
```

## Part 2 — Replace the hard gate with a fallback cascade (imputation modules)

Remove `min_obs_per_regime` as a hard cutoff. Introduce a per-regime cascade that
**guarantees every regime that appears in the data gets usable parameters**, so
no backcast date is ever left unimputed and no singular covariance reaches the
Cholesky.

For each regime `k`, with `n_k` = overlap observations in regime `k` and
`N_short` = number of short-history assets, and
`reliable_threshold = config value or max(3 * N_short, 60)`:

- `n_k >= reliable_threshold` → **full**: regime's own `(mu_k, Sigma_k, beta_k)`.
- `N_short < n_k < reliable_threshold` → **shrunk**: shrink the regime
  conditional covariance toward the pooled (unconditional) conditional
  covariance,
  `Sigma_k = lam * Sigma_pooled + (1 - lam) * Sigma_k_raw`,
  with `lam = reliable_threshold / (reliable_threshold + n_k)` (or Ledoit-Wolf
  intensity if `regime_shrinkage: "auto"` and the LW estimator is already in the
  codebase — prefer reusing it).
- `n_k <= N_short` → **pooled**: use the unconditional `(mu, Sigma, beta)`
  estimated from ALL overlap observations. Log a WARNING with `k` and `n_k`.

At **every** level, enforce PSD before the Cholesky:

```python
def nearest_psd(sigma: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Clip eigenvalues to >= epsilon to guarantee a PSD matrix."""
    vals, vecs = np.linalg.eigh(sigma)
    return (vecs * np.clip(vals, epsilon, None)) @ vecs.T
```

Put the cascade in one place both imputers call — e.g. a
`build_regime_params(...) -> dict[int, RegimeParams]` function (new module
`imputation/regime_params.py` or alongside the existing regime code), where
`RegimeParams` is a dataclass carrying `mu, sigma_cond, beta, source, n_obs`
(`source ∈ {"full","shrunk","pooled"}`). Both `single_impute.py` and
`multiple_impute.py` consume it so the behavior can't drift between them.

## Part 3 — Source tagging & diagnostics

- Every imputed date carries the `source` of the regime params used to fill it.
- Return a per-asset, per-source breakdown (how many backcast dates were filled
  `full` vs `shrunk` vs `pooled`), surfaced on the imputation result object and
  logged at INFO.
- Add one plot to `visualization/plots.py`: backcast timeline colored by
  `source`, so an over-reliance on `pooled` is visible at a glance.

## Part 4 — Config changes

```yaml
hmm:
  min_covar: 1.0e-3            # Gaussian covariance floor (raise if states collapse)
  reject_underfilled_states: true   # drop K if any state has < n_short+1 obs
  fallback_to_single_regime: true   # if no K survives, use K=1 (unconditional)

imputation:
  method: "regime_conditional"
  # REMOVED: min_obs_per_regime
  regime_reliable_threshold: null   # null -> max(3 * n_short, 60)
  regime_shrinkage: "auto"          # "auto" (Ledoit-Wolf) or fixed float in [0,1]
  regime_fallback_to_pooled: true
  psd_epsilon: 1.0e-10
```

Keep backward-compat: if an old config still sets `min_obs_per_regime`, log a
DEPRECATION warning, map it to `regime_reliable_threshold`, and continue.

## Part 5 — Tests

Add to the test suite:

1. **No NaN guarantee**: on Tier 2 synthetic data, assert
   `all(df.isna().sum().sum() == 0 for df in imputed_histories)`.
2. **Adversarial regime → pooled, not NaN**: build the Tier 2 *adversarial*
   dataset (a regime present only in the backcast period). Assert the backcast is
   fully imputed AND those dates are tagged `source == "pooled"` AND a warning was
   logged.
3. **Thin regime → shrunk**: a regime with `N_short < n_k < reliable_threshold`
   produces `source == "shrunk"` and a PSD covariance (all eigenvalues `> 0`).
4. **HMM degenerate-K rejection**: feed data that supports only 2 regimes but
   request candidates `[2,3,4,5]`; assert the selector returns K≤2 and never a
   NaN model.
5. **PSD enforcement**: feed a deliberately singular regime covariance; assert
   `nearest_psd` output passes `cho_factor` without error.

## Execution order

1. `models/regime_hmm.py` — Part 1, run its tests.
2. `imputation/regime_params.py` (+ dataclass, cascade, `nearest_psd`) — Part 2.
3. Wire `single_impute.py` and `multiple_impute.py` to the shared builder.
4. Config + backward-compat shim — Part 4.
5. `visualization/plots.py` source-timeline plot — Part 3.
6. Tests — Part 5. Run the full suite; show results before finishing.

Stop after Part 1 and show the HMM test results before continuing.
