# Backcast Engine — Jupyter Notebook Tutorial Prompt

## Objective

Create a comprehensive Jupyter notebook tutorial (`tutorial.ipynb`) that walks a quant researcher through the full backcast engine workflow, from raw data to portfolio-ready outputs. The tutorial should be self-contained — a new team member should be able to run it end-to-end and understand every step.

**Place the notebook in `backcast_engine/notebooks/00_tutorial.ipynb`.**

## Prerequisites

The tutorial assumes:
- The `backcast` package is installed (`pip install -e .` from `backcast_engine/`)
- The synthetic data generator has been run and Tier 1 + Tier 2 datasets are available
- If synthetic data is not available, the notebook should generate it inline using a minimal helper function (so it's truly self-contained)

## Notebook Structure

The notebook should follow this exact structure. Each section should have clear markdown explanations before the code cells — not just code dumps. Write it as if you're teaching a junior quant who knows Python and statistics but hasn't used this library before.

---

### Section 1: Introduction & Setup (2-3 cells)

**Markdown:** Explain what backcasting is in 3-4 sentences. State the problem: unequal history lengths → incomplete covariance matrices → biased portfolio construction. Mention the three models (EM, Kalman, HMM) and multiple imputation.

**Code:**
- Import all required modules from `backcast`
- Set plotting defaults (matplotlib inline, figure size, style)
- Set random seed for reproducibility
- Print package version if available

---

### Section 2: Loading & Inspecting Data (4-5 cells)

**Markdown:** Explain the expected data format (CSV, rows=dates, columns=assets, daily simple returns, leading NaN for short-history assets).

**Code cell 1:** Load data using `backcast.data.loader`. If a CSV path is provided, use it. Otherwise, generate synthetic Tier 1 data inline:

```python
# Generate a simple synthetic dataset for the tutorial
# 5 long-history assets (5000 days), 3 short-history assets (starting at day 3000)
# Known covariance structure so we can validate results
```

Provide the full inline generator (multivariate normal, mask short assets) so the notebook runs without external files.

**Code cell 2:** Display the `BackcastDataset` object — show which assets are long vs short, overlap dates, backcast dates, dimensions.

**Code cell 3:** Show the missingness pattern. Use `backcast.visualization.plots` if available, otherwise create a simple heatmap with matplotlib showing NaN locations (assets on y-axis, time on x-axis, color = observed vs missing).

**Code cell 4:** Basic EDA on the overlap period — compute and display:
- Annualized mean returns per asset
- Annualized volatility per asset
- Correlation matrix heatmap
- A few summary statistics (skew, kurtosis)

**Code cell 5:** Plot cumulative returns for all assets during the overlap period to visually confirm the data looks reasonable.

---

### Section 3: Model 1 — EM Algorithm (5-6 cells)

**Markdown:** Explain the EM algorithm in 4-5 sentences. Key points: it estimates the full unconditional mean and covariance from incomplete data. Mention Stambaugh (1997). Explain the E-step (conditional expectations) and M-step (parameter re-estimation with variance correction). Emphasize that the variance correction is critical — without it, covariance of short-history assets is underestimated.

**Code cell 1:** Fit the EM model:
```python
from backcast.models.em_stambaugh import EMStambaugh

em_model = EMStambaugh(max_iterations=500, tolerance=1e-8)
em_result = em_model.fit(dataset)
```

**Code cell 2:** Inspect convergence — plot log-likelihood vs iteration. Print number of iterations and final tolerance. Comment on whether it converged.

**Code cell 3:** Display the estimated full covariance matrix as a heatmap. Compare the EM-estimated correlation matrix side-by-side with the overlap-period sample correlation matrix.

**Code cell 4:** Show the conditional parameters:
- Print the regression coefficients (beta = Σ₂₁ Σ₁₁⁻¹) — these are the factor loadings of short-history assets on long-history assets
- Print the conditional covariance Σ₂₂|₁ — this is the residual uncertainty
- Interpret: "BTC loads X on S&P500, Y on Gold, etc."

**Code cell 5:** If using synthetic data with known ground truth, compare the EM estimate to the true Σ:
- Frobenius norm of the difference
- Element-wise relative error heatmap
- Print: "The EM recovered the true covariance within X% relative error"

---

### Section 4: Model 2 — Kalman Filter (4-5 cells)

**Markdown:** Explain the Kalman filter TVP model in 3-4 sentences. It estimates time-varying betas (factor loadings) of short-history assets on long-history assets. Unlike the EM, it allows the relationship to change over time. Uses a random walk state equation for the betas.

**Code cell 1:** Fit the Kalman model:
```python
from backcast.models.kalman_tvp import KalmanTVP

kalman_model = KalmanTVP(use_smoother=True)
kalman_result = kalman_model.fit(dataset)
```

**Code cell 2:** Plot the smoothed betas over time — one subplot per short-history asset, showing its loading on each long-history factor. Add confidence bands.

**Code cell 3:** Compare the Kalman's earliest betas to the EM's constant betas. If they're similar, the EM's stationarity assumption is validated. If they differ significantly, flag this.

**Code cell 4:** Print the backcast betas that will be used for imputation. Explain which method was used (earliest smoothed, mean of first K).

---

### Section 5: Model 3 — Regime-Switching HMM (4-5 cells)

**Markdown:** Explain the HMM in 3-4 sentences. It detects market regimes (calm vs crisis) using long-history assets, then estimates regime-conditional parameters. The backcast uses regime labels from long-history data to select which parameters to use at each date — producing lower vol in calm periods and higher vol in crisis periods.

**Code cell 1:** Fit the HMM, testing K=2,3,4:
```python
from backcast.models.regime_hmm import RegimeHMM

hmm_model = RegimeHMM(n_regimes_candidates=[2, 3, 4], selection_criterion='bic')
hmm_result = hmm_model.fit(dataset)
```

**Code cell 2:** Print model selection results (BIC for each K). Print the selected K and the transition matrix.

**Code cell 3:** Plot the regime timeline across the full date range — a color-coded bar showing which regime is active at each date. Mark the backcast/overlap boundary.

**Code cell 4:** Print regime-conditional statistics:
- For each regime: mean return vector, volatility vector, and average pairwise correlation
- Highlight the contrast: "In regime 2 (crisis), vol is Xх higher and correlations spike to Y"

**Code cell 5 (if Tier 2 synthetic data with known regimes):** Compare detected regimes to true regime labels. Print accuracy.

---

### Section 6: Model Comparison & Validation (4-5 cells)

**Markdown:** Explain the holdout validation procedure. We mask the earliest H days of the overlap period for short-history assets, run each model to backcast those days, and compare to actuals. This is walk-forward — not random — because temporal ordering matters.

**Code cell 1:** Run holdout validation:
```python
from backcast.validation.holdout import HoldoutValidator

validator = HoldoutValidator(holdout_days=504, n_windows=3)
validation_report = validator.run(dataset, models={'em': em_model, 'kalman': kalman_model, 'hmm': hmm_model})
```

**Code cell 2:** Print the comparison table — RMSE, MAE, correlation error, vol ratio for each model × each short-history asset.

**Code cell 3:** Plot actual vs predicted returns scatter for the best model. One subplot per short-history asset.

**Code cell 4:** Print the recommendation: which model performed best overall and why.

---

### Section 7: Single Imputation (2-3 cells)

**Markdown:** Explain single imputation — filling the NaN block with conditional means. This produces one "best guess" history. Useful for quick analysis but underestimates uncertainty.

**Code cell 1:** Generate the single imputed dataset:
```python
from backcast.imputation.single_impute import SingleImputer

imputer = SingleImputer(model=em_model)  # or best model from validation
imputed_single = imputer.impute(dataset)

print(f"Missing values before: {dataset.returns_full.isna().sum().sum()}")
print(f"Missing values after:  {imputed_single.isna().sum().sum()}")
```

**Code cell 2:** Plot cumulative returns for ALL assets across the FULL date range (backcast + overlap). Use a dashed line for the backcasted portion and a solid line for the observed portion. This is the key visual — the reader sees the extended histories.

**Code cell 3:** Compute and display the full-sample correlation matrix (all assets, all dates). Compare to the overlap-only correlation matrix.

---

### Section 8: Multiple Imputation (4-5 cells)

**Markdown:** Explain why single imputation is not enough. Multiple imputation generates M=50 plausible histories by sampling from the conditional distribution (not just taking the mean). This captures imputation uncertainty. Downstream statistics should be computed on all M datasets and combined using Rubin's rules.

**Code cell 1:** Generate M imputed datasets:
```python
from backcast.imputation.multiple_impute import MultipleImputer

multi_imputer = MultipleImputer(model=em_model, n_imputations=50)
imputed_histories = multi_imputer.impute(dataset)

print(f"Generated {len(imputed_histories)} complete histories")
```

**Code cell 2:** Fan chart — pick one short-history asset (e.g., CRYPTO_1). Plot the median backcasted cumulative return path across the M imputations, plus the 5th/95th percentile bands. Overlay the actual observed data where available.

**Code cell 3:** Demonstrate Rubin's rules:
```python
# Compute mean return for CRYPTO_1 on each imputed dataset
means = [df['CRYPTO_1'].mean() for df in imputed_histories]

# Combine using Rubin's rules
from backcast.imputation.multiple_impute import combine_estimates
result = combine_estimates(means)
print(f"Combined mean: {result.point_estimate:.6f}")
print(f"Within-imputation variance: {result.within_variance:.8f}")
print(f"Between-imputation variance: {result.between_variance:.8f}")
print(f"Total variance: {result.total_variance:.8f}")
print(f"95% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
```

**Code cell 4:** Compare covariance matrices across imputations. Show the spread of a specific covariance entry (e.g., Corr(BTC, S&P500)) as a histogram across the M imputations.

---

### Section 9: Downstream — Covariance Estimation (3-4 cells)

**Markdown:** Explain the four covariance estimation methods: EM direct, sample on imputed data, Rubin combined, and shrinkage (Ledoit-Wolf). For portfolio optimization, you typically want the shrinkage estimator for better conditioning.

**Code cell 1:** Compute all covariance estimates:
```python
from backcast.downstream.covariance import CovarianceEstimator

cov_estimator = CovarianceEstimator(shrinkage='ledoit_wolf', denoise=True)
cov_result = cov_estimator.estimate(imputed_histories, em_result=em_result)

print(f"Condition number (raw): {cov_result.condition_number_raw:.1f}")
print(f"Condition number (shrunk): {cov_result.condition_number_shrunk:.1f}")
```

**Code cell 2:** Display the final correlation matrix heatmap (the one you'd actually use for portfolio construction).

**Code cell 3:** Plot the eigenvalue spectrum. If denoising was applied, show before/after. Mark the Marchenko-Pastur bound.

---

### Section 10: Downstream — Uncertainty Ellipses (2-3 cells)

**Markdown:** Explain uncertainty ellipses for robust optimization. The M imputed covariance matrices define a distribution of portfolio risk for any weight vector. We compute the uncertainty set around the mean estimate and pass it to a robust optimizer.

**Code cell 1:** Compute uncertainty sets:
```python
from backcast.downstream.uncertainty import UncertaintyEstimator

unc = UncertaintyEstimator(confidence=0.95)
unc_result = unc.compute(imputed_histories)
```

**Code cell 2:** Plot the 2D uncertainty ellipse for a pair of assets (e.g., projected mean return of CRYPTO_1 vs S&P 500). Show the M individual estimates as scatter points and the 95% ellipse around them.

---

### Section 11: Downstream — Backtesting (4-5 cells)

**Markdown:** Explain the backtest framework. We run a portfolio strategy on each of the M imputed histories to get a distribution of performance outcomes. This honestly quantifies how much backtest results depend on the imputation.

**Code cell 1:** Run backtests with built-in strategies:
```python
from backcast.downstream.backtest import BacktestEngine

engine = BacktestEngine(strategies=['equal_weight', 'inverse_vol', 'min_variance', 'risk_parity'])
backtest_results = engine.run(imputed_histories)
```

**Code cell 2:** Fan chart of cumulative returns for each strategy — median path ± 5th/95th percentile bands. One subplot per strategy.

**Code cell 3:** Summary table — for each strategy, show:
- Median annualized return (with 90% CI)
- Median annualized vol (with 90% CI)
- Median Sharpe ratio (with 90% CI)
- Median max drawdown (with 90% CI)

**Code cell 4:** Highlight the key insight: "The uncertainty bands show that backtested Sharpe ratios for strategies with heavy crypto allocation have a wide range [X, Y], reflecting imputation uncertainty. Strategies with traditional assets only have much tighter bands."

---

### Section 12: Full Pipeline in One Call (2-3 cells)

**Markdown:** Show that everything above can be run with a single pipeline call using a YAML config.

**Code cell 1:**
```python
from backcast.pipeline import BackcastPipeline

pipeline = BackcastPipeline(config_path='../config/default_config.yaml')
results = pipeline.run(csv_path='path/to/returns.csv')
```

**Code cell 2:** Show how to access results:
```python
# Imputed histories
results.imputed_histories  # list of M DataFrames

# Covariance matrix (ready for optimizer)
results.covariance.shrunk_covariance

# Uncertainty set (ready for robust optimizer)
results.uncertainty.ellipsoidal_set

# Backtest performance
results.backtest.summary_table
```

**Code cell 3:** Show how to export:
```python
pipeline.export(results, output_dir='./results')
# Saves: imputed CSVs, covariance matrices, plots, validation report
```

---

### Section 13: Tips & Gotchas (markdown only)

A final markdown section with practical advice:

1. **Minimum overlap period**: You need at least 504 trading days (~2 years) of overlap for stable estimates. More is better. With less than 252 days, treat all results with extreme caution.

2. **Monotone missingness only**: The library assumes short-history assets have a contiguous block of NaN at the start, then continuous data. If your data has gaps in the middle, you need to handle those separately first.

3. **Returns, not prices**: The input must be daily simple returns. If you have prices, convert first: `returns = prices.pct_change().dropna()`.

4. **Stationarity check**: Always run the Kalman model as a sanity check. If factor loadings drift significantly over the overlap period, the EM's unconditional estimates may be misleading. Consider using regime-conditional imputation instead.

5. **Never use a single backcast path for decisions**: Always use multiple imputation. The fan charts and Rubin's confidence intervals exist for a reason — they tell you how much to trust the backcast.

6. **Backcasted returns are models, not data**: A backtest on imputed BTC returns from 1995 tells you "how an asset with BTC's overlap-period statistical properties would have behaved given 1990s factor returns." It does not tell you what BTC would have actually done.

---

## Implementation Guidelines

### Notebook Style
- **Markdown cells should read like a research note** — concise, technical, no fluff. The audience is quant researchers who know statistics.
- **Code cells should be copy-pasteable** — a reader should be able to take any code cell and use it in their own workflow with minimal modification.
- **Show outputs** — every code cell that produces a result should print or plot something. No silent cells.
- **Use f-strings for numerical output** — format to appropriate decimal places (6 for returns, 2 for percentages, 4 for correlations).

### Plotting Style
- Use `matplotlib` with a clean style (`plt.style.use('seaborn-v0_8-whitegrid')` or similar).
- Consistent figure sizes: `(12, 6)` for time series, `(8, 8)` for heatmaps, `(12, 8)` for multi-panel.
- Label all axes. Include titles.
- Use dashed lines for backcasted portions, solid for observed.
- Use alpha=0.1-0.3 for fan chart bands.
- Color scheme: use a consistent palette throughout (e.g., tab10 or a custom one).

### Error Handling
- Wrap model fitting in try/except with informative messages.
- If a module is not available (e.g., user hasn't built all modules yet), print a clear message and skip that section rather than crashing the entire notebook.
- Include `%load_ext autoreload` and `%autoreload 2` at the top so changes to the library are picked up without kernel restart.

### Inline Data Generation
The notebook MUST be runnable without external files. Include a helper function at the top that generates synthetic data if no CSV path is provided:

```python
def generate_tutorial_data(seed=42):
    """
    Generate synthetic returns for the tutorial.
    5 long-history assets, 3 short-history assets.
    Uses a known factor model covariance structure.
    Returns a DataFrame with NaN for short-history assets before their start date.
    """
    # Implementation here — multivariate normal, factor model, mask short assets
    # Return both the masked DataFrame and the true parameters (for validation)
```

This function should be ~30-40 lines, self-contained, using only numpy and pandas.

### Dependencies
The notebook should only import from:
- `backcast` (the library being tutorialized)
- `numpy`, `pandas`, `matplotlib`, `seaborn` (standard data science stack)
- No other external libraries in the notebook itself

---

## Execution Order

When building this notebook:
1. Create the inline data generator first and verify it produces valid data
2. Build sections 1-3 (data + EM) and verify they run
3. Build sections 4-6 (Kalman + HMM + validation) and verify
4. Build sections 7-8 (imputation) and verify
5. Build sections 9-11 (downstream) and verify
6. Add sections 12-13 (pipeline + tips)
7. Run the full notebook top-to-bottom and fix any issues
8. Clear all outputs and re-run to verify clean execution

---

## Graceful Degradation

Since the backcast engine may not be fully implemented when this notebook is first used, implement graceful fallbacks:

- If a module import fails, print `"⚠️ Module backcast.models.kalman_tvp not yet available — skipping Section 4"` and continue.
- For sections that depend on previous results, check if the variable exists before using it.
- The tutorial should be useful even if only the EM model and single imputation are implemented — those sections should work standalone.

Use this pattern:

```python
try:
    from backcast.models.kalman_tvp import KalmanTVP
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("⚠️ KalmanTVP not available — Section 4 will be skipped")
```

Then guard each section:

```python
if KALMAN_AVAILABLE:
    # ... section 4 code ...
else:
    print("Skipped — KalmanTVP not yet implemented")
```
