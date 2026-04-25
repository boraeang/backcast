# Financial Time Series Backcasting Engine — Claude Code Prompt

## Context & Objective

You are building a **production-grade Python library** for backcasting (retropolating) financial time series. The use case is a hedge fund that has long-history assets (S&P 500, global bond indices, gold, etc.) and short-history assets (Bitcoin, alternative assets, newer ETFs — potentially 10+ assets simultaneously). The goal is to extend the short-history assets backward in time using the statistical relationships observed during the overlap period.

**This is a missing-data problem.** The returns matrix has a block of `NaN` values in the upper-left corner (early dates × short-history assets). We are estimating the conditional distribution of the missing returns given the observed long-history returns.

## Key Design Decisions (already made — do not override)

1. **Regime-adaptive volatility**: The model should dictate volatility in the backcast period. If the 1990s were calmer, backcasted BTC-like assets should reflect lower vol in that era. Do NOT preserve the overlap-period vol profile.
2. **Simultaneous multi-asset imputation**: The system must handle 10+ short-history assets at once, preserving cross-sectional correlations — not 10 independent regressions.
3. **Multiple imputation**: Generate N plausible backcast paths, not a single point estimate. Downstream analysis should be runnable on each path.
4. **The downstream uses are**: covariance estimation for portfolio optimization, uncertainty ellipses for robust optimization, copula-based simulation, full backtest simulation, and regime analysis.

## Input Specification

- **Format**: CSV file
- **Structure**: Rows = dates (sorted ascending), Columns = asset names
- **Values**: Daily simple returns (not prices, not log-returns)
- **Missing data pattern**: Monotone missingness — short-history assets have leading `NaN` blocks, then continuous data once they start. There are no gaps in the middle.

Example structure:
```
date,       SP500,  GLOBAL_BONDS, GOLD,    BTC,     ALT_1
1990-01-02, 0.0012, 0.0003,       -0.0005, NaN,     NaN
1990-01-03, -0.0008, 0.0001,      0.0010,  NaN,     NaN
...
2015-01-02, 0.0005, -0.0002,      0.0008,  0.0320,  NaN
...
2020-01-02, 0.0010, 0.0001,       0.0015,  -0.0150, 0.0200
```

## Project Structure

```
backcast_engine/
├── README.md
├── pyproject.toml                  # Use modern Python packaging
├── requirements.txt
├── config/
│   └── default_config.yaml         # All hyperparameters centralized
├── src/
│   └── backcast/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py           # CSV ingestion, validation, alignment
│       │   └── transforms.py       # Return ↔ price conversions, log transforms
│       ├── models/
│       │   ├── __init__.py
│       │   ├── em_stambaugh.py     # EM algorithm (Stambaugh 1997)
│       │   ├── kalman_tvp.py       # Kalman filter with time-varying parameters
│       │   ├── regime_hmm.py       # Hidden Markov Model for regime detection
│       │   └── model_selector.py   # Cross-validation, model comparison
│       ├── imputation/
│       │   ├── __init__.py
│       │   ├── single_impute.py    # Point estimate backcast
│       │   ├── multiple_impute.py  # Multiple imputation wrapper (N paths)
│       │   └── copula_sim.py       # Copula-based simulation engine
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── holdout.py          # Walk-forward holdout validation
│       │   ├── metrics.py          # RMSE, MAE, correlation, distributional tests
│       │   └── diagnostics.py      # Residual analysis, PIT histograms
│       ├── downstream/
│       │   ├── __init__.py
│       │   ├── covariance.py       # Full-sample covariance from imputed data
│       │   ├── uncertainty.py      # Uncertainty ellipses for robust optimization
│       │   └── backtest.py         # Simple backtest harness on imputed histories
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── plots.py            # All diagnostic and result plots
│       └── pipeline.py             # End-to-end orchestration
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_model_comparison.ipynb   # Compare EM vs Kalman vs HMM
│   ├── 03_validation.ipynb         # Holdout validation results
│   └── 04_downstream.ipynb         # Portfolio optimization, copula sim, etc.
├── tests/
│   ├── test_em.py
│   ├── test_kalman.py
│   ├── test_imputation.py
│   └── test_pipeline.py
└── scripts/
    └── run_backcast.py             # CLI entry point
```

## Module Specifications

---

### Module 1: `data/loader.py`

**Purpose**: Load CSV, validate structure, detect missingness pattern, separate long-history and short-history assets.

**Requirements**:
- Load CSV with `date` as index (parse dates automatically).
- Validate that data is daily simple returns (warn if values > 0.5 or < -0.5 on any single day — likely prices not returns).
- Detect monotone missingness pattern. If non-monotone gaps exist, raise a `ValueError` with clear diagnostics.
- Output a `BackcastDataset` dataclass containing:
  - `returns_full`: the complete DataFrame
  - `long_assets`: list of asset names with full history
  - `short_assets`: list of asset names with partial history
  - `overlap_start` / `overlap_end`: date range where all assets have data
  - `backcast_start` / `backcast_end`: date range to impute
  - `overlap_matrix`: returns of ALL assets during the overlap period
  - `long_history_matrix`: returns of long-history assets during the backcast period

**Edge cases to handle**:
- Assets that start on different dates (staggered missingness) — sort by history length.
- Weekends/holidays creating date gaps — this is expected, do not flag as errors.
- Duplicate dates — raise error.

---

### Module 2: `models/em_stambaugh.py`

**Purpose**: Implement the EM algorithm for estimating the unconditional mean and covariance matrix from an incomplete returns matrix (Stambaugh 1997 approach).

**Algorithm (implement from scratch, do not use a library)**:

Let $R$ be the $T \times N$ return matrix with monotone missingness. Partition assets into observed ($R_1$, long history) and partially observed ($R_2$, short history).

**E-step**: Conditional on current parameter estimates $(\hat{\mu}, \hat{\Sigma})$, compute:
- $E[R_{2,t} | R_{1,t}]$ for each missing observation using the conditional normal formula:
  $E[R_2 | R_1 = r_1] = \mu_2 + \Sigma_{21}\Sigma_{11}^{-1}(r_1 - \mu_1)$
- $Var[R_2 | R_1]$ = $\Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}$
- Accumulate sufficient statistics including the conditional variance correction term.

**M-step**: Re-estimate $\mu$ and $\Sigma$ using the completed data plus the variance correction:
- $\hat{\Sigma}_{new}$ = sample covariance of filled data + average conditional variance (this is the key — naive EM without the variance correction underestimates $\Sigma_{22}$).

**Convergence**: Iterate until $||\Sigma^{(k+1)} - \Sigma^{(k)}||_F < \epsilon$ (default $\epsilon = 10^{-8}$), with a max iteration cap (default 500).

**Output**:
- `mu_em`: estimated full mean vector ($N \times 1$)
- `sigma_em`: estimated full covariance matrix ($N \times N$), guaranteed PSD
- `conditional_params`: dict with `beta` ($\Sigma_{21}\Sigma_{11}^{-1}$), `cond_cov` ($\Sigma_{22|1}$), and `alpha` (intercept) for downstream imputation
- `convergence_info`: number of iterations, final tolerance, log-likelihood trace

**Critical implementation notes**:
- Use Cholesky decomposition for inverting $\Sigma_{11}$ — it's faster and numerically stable.
- After final M-step, eigenvalue-clip $\Sigma$ to ensure PSD (set any negative eigenvalues to a small positive $\epsilon$).
- Handle the **staggered missingness** case: if short-history assets start on different dates, process groups of assets with the same start date sequentially, expanding the observed set each time.
- Store the log-likelihood at each iteration for convergence diagnostics.

---

### Module 3: `models/kalman_tvp.py`

**Purpose**: Estimate time-varying factor loadings (betas) of short-history assets on long-history factors using a state-space model.

**Model specification**:
- Observation equation: $r_{2,t} = \alpha_t + B_t \cdot r_{1,t} + \epsilon_t$, where $\epsilon_t \sim N(0, H)$
- State equation: $\beta_t = \beta_{t-1} + \eta_t$, where $\eta_t \sim N(0, Q)$
  (Random walk dynamics for the betas — simple but effective)
- State vector $\beta_t$ includes both the intercept $\alpha_t$ and slope coefficients $B_t$.

**Implementation**:
- Use `statsmodels.tsa.statespace` or implement the Kalman recursions from scratch.
- Estimate $H$ and $Q$ via maximum likelihood (or use a simple prior).
- Run the **Kalman smoother** (not just filter) to get $E[\beta_t | \text{all data}]$ — this uses future information to refine past estimates, which is appropriate here since we're not doing online prediction.
- For the backcast period, use the **earliest smoothed beta** (or a weighted average of the first K smoothed betas) as the fixed loading. Do NOT extrapolate a trend in betas backward — this is unstable.

**Output**:
- `smoothed_betas`: DataFrame of time-varying betas, shape $(T_{overlap} \times N_{short} \times (N_{long}+1))$
- `backcast_betas`: The beta values to use for the backcast period
- `state_covariance`: Time series of state uncertainty
- `residual_variance`: Estimated $H$ matrix

**Design note**: This model is primarily a **robustness check** against the EM model. If the Kalman betas are relatively stable over the overlap period, it validates the EM's stationarity assumption. If they drift significantly, it suggests the EM backcast should be interpreted cautiously.

---

### Module 4: `models/regime_hmm.py`

**Purpose**: Detect market regimes using a Hidden Markov Model on the long-history assets, then use regime labels to condition the backcast.

**Implementation**:
- Fit a Gaussian HMM with K states (default K=2: "calm" and "stressed") on the long-history returns using `hmmlearn`.
- Use BIC/AIC to select K from {2, 3, 4}.
- Decode the most likely state sequence for the full history (Viterbi algorithm).
- Estimate regime-conditional means and covariances: $\mu^{(k)}$, $\Sigma^{(k)}$ for each regime $k$.

**Integration with backcast**:
- During the backcast period, the regime labels are known (from long-history data).
- Impute short-history returns using the **regime-conditional** parameters instead of the unconditional EM parameters.
- This naturally produces **lower vol in calm regimes** and higher vol in stressed regimes — which is exactly the behavior we want.

**Output**:
- `regime_labels`: Series of integer regime labels for the full date range
- `regime_params`: Dict mapping regime → (mu, sigma, conditional_params)
- `transition_matrix`: Estimated regime transition probabilities
- `model_selection`: BIC/AIC for each K tested

---

### Module 5: `imputation/single_impute.py`

**Purpose**: Generate a single point-estimate backcast using conditional expectations.

**For each date $t$ in the backcast period**:
1. Get the observed long-history returns $r_{1,t}$.
2. (Optional) Get the regime label $s_t$ from the HMM.
3. Compute $\hat{r}_{2,t} = \alpha + B \cdot r_{1,t}$ using either:
   - EM conditional parameters (unconditional model)
   - Regime-conditional parameters (if HMM is used)
   - Kalman backcast betas (if TVP model is used)
4. Return the filled returns matrix.

**Output**: Complete returns DataFrame with no `NaN` values.

---

### Module 6: `imputation/multiple_impute.py`

**Purpose**: Generate $M$ plausible backcast paths (default $M=50$) by sampling from the conditional distribution, not just taking the conditional mean.

**For each imputation $m = 1, ..., M$**:
1. For each date $t$ in the backcast period:
   - Compute conditional mean: $\hat{\mu}_{2|1,t} = \alpha + B \cdot r_{1,t}$
   - Compute conditional covariance: $\Sigma_{2|1}$ (from EM or regime-conditional)
   - **Draw**: $r_{2,t}^{(m)} \sim N(\hat{\mu}_{2|1,t}, \Sigma_{2|1})$
2. Optionally incorporate **parameter uncertainty** (Bayesian bootstrap or posterior draws of $\mu, \Sigma$).
3. Return a list of $M$ complete DataFrames.

**Rubin's rules for combining estimates**:
- Implement `combine_estimates(estimates_list)` that takes a statistic computed on each imputed dataset and returns:
  - Combined point estimate (mean across imputations)
  - Within-imputation variance
  - Between-imputation variance
  - Total variance (Rubin's formula)
  - Degrees of freedom adjustment

**Output**: List of $M$ DataFrames + `RubinResult` dataclass for combined inference.

---

### Module 7: `imputation/copula_sim.py`

**Purpose**: Generate simulated return paths using a copula model fitted on the imputed full-history data.

**Implementation**:
- Fit marginal distributions to each asset's full-history returns:
  - Try: Normal, Student-t, skewed-t, empirical CDF
  - Select by KS test or AIC
- Fit a copula to the rank-transformed data:
  - Gaussian copula (baseline)
  - Student-t copula (captures tail dependence — important for crisis scenarios)
  - Use the `pyvinecopulib` library if available, otherwise implement Gaussian/t copula from scratch
- Simulate $S$ scenarios (default $S=10000$) of daily returns for all assets over any desired date range.
- Ensure simulated returns are consistent with the fitted marginals and dependence structure.

**Output**:
- `simulated_returns`: array of shape $(S, T_{sim}, N)$
- `fitted_marginals`: dict of asset → distribution parameters
- `fitted_copula`: copula parameters (correlation matrix, degrees of freedom for t-copula)

---

### Module 8: `validation/holdout.py`

**Purpose**: Validate the backcast models using walk-forward holdout on the overlap period.

**Procedure**:
1. Take the overlap period where all assets have data.
2. Artificially mask the first $H$ days of the short-history assets (e.g., $H$ = 504 trading days ≈ 2 years).
3. Run each model (EM, Kalman, Regime-HMM) to backcast those $H$ days.
4. Compare backcasted returns to the actual observed returns.

**Validation should test**:
- **Marginal accuracy**: Are the backcasted return distributions (mean, vol, skew, kurtosis) close to actuals?
- **Correlation accuracy**: Does the backcasted cross-correlation matrix match the actual?
- **Tail behavior**: Do the backcasted joint tail events (e.g., simultaneous drawdowns) match reality?
- **Serial properties**: Is the autocorrelation structure of backcasted returns realistic?

For multiple imputation: assess **calibration** — do the 95% prediction intervals from the M paths cover ~95% of actual returns?

---

### Module 9: `validation/metrics.py`

Implement these metrics:
- `rmse(actual, predicted)`: Root mean squared error
- `mae(actual, predicted)`: Mean absolute error
- `correlation_error(actual_corr, predicted_corr)`: Frobenius norm of correlation matrix difference
- `vol_ratio(actual_vol, predicted_vol)`: Ratio of annualized vols
- `ks_test(actual_returns, predicted_returns)`: Kolmogorov-Smirnov test per asset
- `pit_histogram(actual, predicted_dist)`: Probability Integral Transform for calibration
- `coverage_rate(actual, lower_bound, upper_bound)`: Empirical coverage of prediction intervals
- `tail_dependence_coeff(returns_1, returns_2, quantile=0.05)`: Lower tail dependence

---

### Module 10: `validation/diagnostics.py`

Implement diagnostic checks:
- Residual normality tests (Jarque-Bera, Shapiro-Wilk)
- Residual autocorrelation (Ljung-Box test)
- Eigenvalue spectrum of imputed vs overlap covariance matrix
- Compare rolling correlations: imputed period vs overlap period
- QQ plots of backcasted returns vs assumed distribution

---

### Module 11: `downstream/covariance.py`

**Purpose**: Estimate the full-sample covariance matrix from imputed data.

**Methods**:
- **Direct EM output**: Use $\hat{\Sigma}$ from the EM algorithm directly.
- **Sample covariance on imputed data**: Compute from the filled matrix (single imputation).
- **Combined covariance (Rubin's rules)**: Average the M covariance matrices from multiple imputation, adding between-imputation variance.
- **Shrinkage estimator**: Apply Ledoit-Wolf shrinkage to the combined covariance for better conditioning.
- **Denoising**: Apply random matrix theory (Marchenko-Pastur) to denoise the covariance eigenvalues.

**Output**: `CovarianceResult` dataclass with the covariance matrix, correlation matrix, eigenvalue decomposition, and condition number.

---

### Module 12: `downstream/uncertainty.py`

**Purpose**: Compute uncertainty ellipses for robust portfolio optimization.

**Implementation**:
- From the $M$ imputed covariance matrices, compute the **distribution of portfolio risk** for any weight vector $w$:
  $\sigma_p^{(m)} = \sqrt{w^T \Sigma^{(m)} w}$
- Compute uncertainty sets:
  - Ellipsoidal uncertainty set: $\{\mu : (\mu - \hat{\mu})^T S^{-1} (\mu - \hat{\mu}) \leq \kappa^2\}$ where $S$ is the between-imputation covariance of $\hat{\mu}$ and $\kappa$ controls the confidence level.
  - Box uncertainty set: Component-wise confidence intervals on $\mu$ and $\Sigma$ entries.
- Output parameters ready for a robust optimization solver (e.g., `cvxpy`).

---

### Module 13: `downstream/backtest.py`

**Purpose**: Simple backtest harness that runs on each imputed history.

**Implementation**:
- Accept a `strategy_fn(returns_df, lookback) -> weights_series` callback.
- Run the strategy on each of the $M$ imputed histories.
- Aggregate results: median path, 5th/95th percentile bands, distribution of Sharpe ratios, max drawdowns, etc.
- Built-in strategies for testing:
  - Equal weight
  - Inverse volatility
  - Minimum variance (using the imputed covariance)
  - Risk parity

---

### Module 14: `visualization/plots.py`

Implement the following plots using `matplotlib` and/or `plotly`:

1. **Missingness heatmap**: Show the NaN pattern in the original data.
2. **EM convergence**: Log-likelihood vs iteration.
3. **Kalman beta evolution**: Time series of smoothed factor loadings with confidence bands.
4. **Regime timeline**: Color-coded bar showing regime labels across the full history.
5. **Backcast fan chart**: Median imputed path ± percentile bands (from multiple imputation) overlaid on actual data where available.
6. **Correlation matrix comparison**: Side-by-side heatmaps of overlap vs backcast period correlations.
7. **Holdout validation scatter**: Actual vs predicted returns, per asset.
8. **PIT histogram**: Should be uniform if model is well-calibrated.
9. **Eigenvalue spectrum**: Imputed covariance vs Marchenko-Pastur bound.
10. **Uncertainty ellipses**: 2D projection of the return uncertainty set for selected asset pairs.
11. **Backtest fan chart**: Cumulative returns across M imputed histories with percentile bands.

---

### Module 15: `pipeline.py`

**Purpose**: End-to-end orchestration with a YAML config.

```python
class BackcastPipeline:
    def __init__(self, config_path: str):
        """Load config, set random seed, initialize logger."""

    def load_data(self, csv_path: str) -> BackcastDataset:
        """Load and validate input data."""

    def fit_models(self, dataset: BackcastDataset) -> dict:
        """Fit EM, Kalman, and HMM models. Return dict of fitted models."""

    def validate(self, dataset: BackcastDataset, models: dict) -> ValidationReport:
        """Run holdout validation on all models. Return comparison report."""

    def select_model(self, report: ValidationReport) -> str:
        """Select best model based on validation metrics."""

    def impute(self, dataset: BackcastDataset, model, n_imputations: int = 50) -> list[pd.DataFrame]:
        """Generate N imputed complete histories."""

    def compute_downstream(self, imputed_histories: list[pd.DataFrame]) -> DownstreamResults:
        """Compute covariance, uncertainty sets, copula sim, backtests."""

    def run(self, csv_path: str) -> FullResults:
        """Execute the full pipeline end-to-end."""

    def export(self, results: FullResults, output_dir: str):
        """Save all results, plots, and reports to output directory."""
```

---

### Config File: `config/default_config.yaml`

```yaml
random_seed: 42

data:
  date_column: "date"
  date_format: null  # auto-detect
  min_overlap_days: 504  # ~2 years minimum overlap required

em:
  max_iterations: 500
  tolerance: 1.0e-8
  psd_epsilon: 1.0e-10  # eigenvalue floor for PSD enforcement

kalman:
  initial_state_cov_scale: 1.0  # scale for initial P0
  state_noise_scale: 0.01       # scale for Q relative to H
  use_smoother: true
  backcast_beta_method: "earliest_smoothed"  # or "mean_first_k"
  backcast_beta_k: 63  # ~3 months of betas to average

hmm:
  n_regimes_candidates: [2, 3, 4]
  selection_criterion: "bic"  # or "aic"
  n_em_iterations: 200
  covariance_type: "full"

imputation:
  n_imputations: 50
  include_parameter_uncertainty: true
  method: "regime_conditional"  # or "unconditional" or "kalman"

copula:
  type: "student_t"  # or "gaussian"
  marginals: "empirical"  # or "student_t" or "skewed_t"
  n_simulations: 10000
  simulation_horizon_days: 252  # 1 year

validation:
  holdout_days: 504  # ~2 years
  n_holdout_windows: 3  # walk-forward windows
  coverage_level: 0.95

downstream:
  covariance_shrinkage: "ledoit_wolf"
  denoise_eigenvalues: true
  uncertainty_confidence: 0.95
  backtest_strategies: ["equal_weight", "inverse_vol", "min_variance", "risk_parity"]

output:
  save_plots: true
  plot_format: "png"
  plot_dpi: 150
  save_imputed_data: true
  export_format: "parquet"  # or "csv"
```

---

## Implementation Guidelines

### Code Quality
- **Type hints everywhere** — use `typing` module. All functions must have full type annotations.
- **Docstrings**: NumPy-style docstrings on all public functions. Include parameter descriptions, return types, mathematical notation where appropriate, and references (e.g., "See Stambaugh (1997), JFQA").
- **Logging**: Use Python's `logging` module. Log at INFO level for pipeline steps, DEBUG for iteration details (EM steps, Kalman updates), WARNING for potential issues (near-singular matrices, slow convergence).
- **Error handling**: Raise informative exceptions. Never silently swallow errors. Use custom exception classes (`BackcastConvergenceError`, `BackcastDataError`, etc.).
- **Reproducibility**: All random operations must use `numpy.random.Generator` with a seed from the config. No global random state.

### Numerical Stability
- Use `scipy.linalg.cho_factor` / `cho_solve` for solving linear systems, not `np.linalg.inv`.
- When computing log-likelihoods, use `scipy.linalg.slogdet` to avoid overflow.
- Regularize covariance matrices before inversion: add `epsilon * I` if condition number exceeds threshold.
- Use `np.float64` throughout — never `float32` for covariance estimation.

### Performance
- Vectorize all date-level operations — no Python loops over $T$ dates.
- The EM algorithm should run in $O(T \cdot N^3)$ per iteration (dominated by the matrix operations, not looping).
- For multiple imputation, parallelize across imputations using `joblib` or `concurrent.futures`.
- Cache expensive computations (e.g., Cholesky decomposition of $\Sigma_{11}$) across EM iterations if $\Sigma_{11}$ hasn't changed.

### Testing
- Unit tests for each module using `pytest`.
- Test the EM algorithm on a fully-observed dataset where you artificially mask data — the estimated $\Sigma$ should converge to the sample covariance.
- Test that the pipeline is deterministic given the same seed.
- Test edge cases: single short-history asset, all assets same length (no imputation needed), very short overlap period.

### Dependencies (ensure Python 3.10+ compatibility)
```
numpy>=1.24
pandas>=2.0
scipy>=1.11
statsmodels>=0.14
scikit-learn>=1.3
hmmlearn>=0.3
matplotlib>=3.7
plotly>=5.15
seaborn>=0.12
pyyaml>=6.0
joblib>=1.3
pytest>=7.4
tqdm>=4.65
```

Optional (for copula module):
```
pyvinecopulib>=0.6  # if available, otherwise implement Gaussian/t copula manually
```

---

## Execution Order

When implementing, build and test modules in this order:

1. `data/loader.py` + `data/transforms.py` → verify with synthetic data
2. `models/em_stambaugh.py` → verify convergence on fully-observed data
3. `imputation/single_impute.py` → verify produces complete matrix
4. `validation/metrics.py` + `validation/holdout.py` → verify on synthetic data
5. `models/kalman_tvp.py` → verify betas are reasonable
6. `models/regime_hmm.py` → verify regime detection on known regime-switching data
7. `imputation/multiple_impute.py` → verify coverage calibration
8. `imputation/copula_sim.py` → verify marginal and copula fits
9. `downstream/covariance.py` + `downstream/uncertainty.py` → verify PSD, Rubin's rules
10. `downstream/backtest.py` → verify with equal-weight strategy
11. `visualization/plots.py` → generate all diagnostic plots
12. `pipeline.py` → wire everything together
13. `tests/` → comprehensive test suite
14. `notebooks/` → demo notebooks

---

## Synthetic Data for Development

The synthetic data generator is a **separate, standalone project** — see `synthetic_data_generator_prompt.md`. Build and run it first to produce test datasets (Tiers 1–4) before developing this backcast engine. The generator outputs CSV files and ground-truth JSON files that this project consumes via `data/loader.py`.

---

## References

- Stambaugh, R.F. (1997). "Analyzing investments whose histories differ in length." Journal of Financial Economics, 45(3), 285-331.
- Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.
- Page, S. (2013). "How to Combine Long and Short Return Histories Efficiently." Financial Analysts Journal, 69(1), 45-52.
- Goldfarb, D. & Iyengar, G. (2003). "Robust portfolio selection problems." Mathematics of Operations Research, 28(1), 1-38.
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series." Econometrica, 57(2), 357-384.
