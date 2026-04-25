# Backcast

Statistical backcasting for short-history financial assets.

Given a CSV of daily returns with some long-history assets (full data) and
some short-history assets (missing leading rows), the backcast engine extends
the short series backward in time while preserving cross-sectional
dependencies and regime behaviour.

The project has two independent sub-projects:

- **`synthetic_data_generator/`** — produces CSV + ground-truth JSON fixtures
  for development and validation.  Four tiers of increasing realism:
  stationary Gaussian → regime-switching → GARCH + fat tails → stress
  scenarios.
- **`backcast_engine/`** — the production library.  Stambaugh EM, Kalman TVP,
  regime HMM, single and multiple imputation, copula simulation, downstream
  analytics (covariance, uncertainty sets, backtests), pipeline orchestration,
  CLI, and demo notebooks.

Both sub-projects share a common spec style: a prompt file in `docs/`
describes the contract and the execution order; the code implements exactly
what the spec asks for and nothing more.


## Layout

```
.
├── docs/
│   ├── synthetic_data_generator_prompt.md
│   └── backcast_prompt.md
├── synthetic_data_generator/
│   ├── src/synthgen/
│   │   ├── config.py           # SyntheticConfig + tier configs
│   │   ├── calendar.py         # business-day index
│   │   ├── correlation.py      # factor-model / random / manual covariance
│   │   ├── masking.py          # monotone missingness
│   │   ├── io.py               # CSV / JSON export
│   │   ├── tier1_stationary.py
│   │   ├── tier2_regime.py
│   │   ├── tier3_realistic.py  # GARCH + fat tails + TVP betas
│   │   ├── tier4_stress.py
│   │   └── cli.py              # python -m synthgen
│   ├── tests/                  # 145 tests
│   └── output/                 # generated fixtures (tier1 … tier4)
├── backcast_engine/
│   ├── src/backcast/
│   │   ├── data/loader.py      # BackcastDataset + CSV ingestion
│   │   ├── data/transforms.py
│   │   ├── models/
│   │   │   ├── em_stambaugh.py   # Stambaugh (1997) EM
│   │   │   ├── kalman_tvp.py     # Kalman filter + RTS smoother
│   │   │   └── regime_hmm.py     # log-space Baum-Welch + Viterbi
│   │   ├── imputation/
│   │   │   ├── single_impute.py
│   │   │   ├── multiple_impute.py  # Rubin's rules
│   │   │   └── copula_sim.py
│   │   ├── validation/
│   │   │   ├── metrics.py, diagnostics.py, holdout.py
│   │   ├── downstream/
│   │   │   ├── covariance.py     # EM / Rubin / Ledoit-Wolf / MP-denoised
│   │   │   ├── uncertainty.py    # ellipsoidal + box sets
│   │   │   └── backtest.py       # equal weight, inv vol, min var, risk parity
│   │   ├── visualization/plots.py  # 11 diagnostic plots
│   │   └── pipeline.py             # BackcastPipeline
│   ├── scripts/
│   │   ├── run_backcast.py         # CLI entry point
│   │   └── build_notebooks.py      # notebook builder
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_model_comparison.ipynb
│   │   ├── 03_validation.ipynb
│   │   └── 04_downstream.ipynb
│   ├── config/default_config.yaml
│   └── tests/                      # 165 tests
└── README.md                       # this file
```


## Quickstart

### Dependencies

Python 3.9+ with:

    numpy>=1.24  pandas>=2.0  scipy>=1.11  scikit-learn>=1.0
    matplotlib>=3.7  pyyaml>=6.0  pytest>=7.4
    nbformat  nbclient                # optional: building notebooks

No editable install required — each sub-project adds its `src/` directory to
`PYTHONPATH` for local execution.

### 1. Generate synthetic fixtures

    cd synthetic_data_generator
    PYTHONPATH=src python -m synthgen --tier 2 --output ./output/tier2

Produces three files:

    output/tier2/returns.csv           # the masked input
    output/tier2/returns_complete.csv  # unmasked, for validation
    output/tier2/ground_truth.json     # true DGP parameters

Other tiers: `--tier 1` (stationary Gaussian), `--tier 3` (GARCH), or
`--tier 4 --scenario all` (four stress-test subdirectories).

### 2. Run the backcast pipeline

    cd backcast_engine
    PYTHONPATH=src python scripts/run_backcast.py \
        --input  ../synthetic_data_generator/output/tier2/returns.csv \
        --output ./output/tier2_backcast \
        --method regime_conditional \
        --n-imputations 50

Emits a compact summary to stdout and writes ~14 artefacts to the output dir
(13 diagnostic plots + `summary.json`).

The CLI supports:

    -i / --input              required: path to a returns CSV
    -o / --output             required: output directory
    -c / --config             optional: YAML config (default: bundled)
    --method                  unconditional_em | regime_conditional
    --n-imputations           M (default from config: 50)
    --seed                    override random_seed
    --save-imputations        also write every imputed history as parquet
    --log-level / --quiet     logging verbosity

### 3. Build the demo notebooks

    cd backcast_engine
    python scripts/build_notebooks.py          # build + execute
    python scripts/build_notebooks.py --no-run # build, leave empty outputs

The four notebooks walk through EDA, model comparison, holdout validation,
and downstream analytics on the Tier 2 fixture.


## Key concepts

**Monotone missingness** — short-history assets have leading NaN blocks, then
uninterrupted data.  Mid-series gaps are rejected as data errors by the
loader.  Staggered starts (e.g., asset A starts in 2005, asset B in 2010) are
supported.

**Stambaugh (1997) EM** — iterative MLE of `(μ, Σ)` from an incomplete matrix.
The E-step uses Cholesky-based conditional expectations per missingness
pattern; the M-step adds a variance-correction term for each missing row
(naive EM without it underestimates `Σ` of the missing block).  Converges
monotonically in log-likelihood.

**Regime-conditional imputation** — fit a Gaussian HMM on the always-observed
long-history assets; decode regime labels; estimate regime-specific
`(μ^{(k)}, Σ^{(k)})` on the overlap period; impute using regime-dependent
conditional distributions.  On regime-switching data this delivers better
variance calibration than unconditional EM (prediction intervals widen in
crisis regimes, narrow in calm regimes).

**Multiple imputation (M ≥ 50)** — draw
`r_M ~ N(α + β·r_O, Σ_{M|O})` for every missing cell instead of taking the
conditional mean.  Downstream statistics are computed per imputation and
combined via **Rubin's rules** — total = within + (1 + 1/M)·between — so
inference accounts for missing-data uncertainty as well as the usual
sampling variability.

**Covariance estimation** — four interchangeable estimators that share a
`CovarianceResult` container: EM direct, Rubin-combined across imputations,
Ledoit-Wolf shrinkage, and Marchenko-Pastur eigenvalue denoising.  All
guaranteed PSD.

**Uncertainty sets for robust optimisation** — ellipsoidal
`(μ − μ̄)ᵀ S⁻¹ (μ − μ̄) ≤ κ²` and box uncertainty sets derived from the M
imputations, ready to feed into a `cvxpy` robust-portfolio solver.

**Backtest harness** — four built-in strategies (equal weight, inverse vol,
min variance, risk parity) runnable across every imputed history,
producing distributions of Sharpe / max-drawdown / total return instead of a
single point estimate.


## Library API

High-level pipeline:

    from backcast.pipeline import BackcastPipeline

    pipe = BackcastPipeline(config_path="config/default_config.yaml")
    results = pipe.run("returns.csv")
    pipe.export(results, "./output/")

Lower-level primitives:

    from backcast import load_backcast_dataset, em_stambaugh, single_impute
    from backcast.imputation import multiple_impute, prediction_intervals
    from backcast.models import fit_and_select_hmm, fit_kalman_all
    from backcast.downstream import (
        combined_covariance, ellipsoidal_uncertainty, run_backtest,
    )

    ds = load_backcast_dataset("returns.csv")
    em = em_stambaugh(ds.returns_full)
    filled = single_impute(ds, em)                     # point estimate
    mi = multiple_impute(ds, em, n_imputations=50)      # M draws
    _, lower, upper = prediction_intervals(mi, confidence=0.95)

Custom backtest strategy:

    from backcast.downstream.backtest import run_backtest

    def my_strategy(window_df, lookback):
        # Return a weight vector of length n_assets
        ...
    result = run_backtest(
        mi.imputations, strategy=my_strategy,
        lookback=63, rebalance_freq=21,
    )


## Testing

Each sub-project has an independent test suite:

    cd synthetic_data_generator
    PYTHONPATH=src python -m pytest tests/    # 145 tests

    cd backcast_engine
    PYTHONPATH=src python -m pytest tests/    # 165 tests

Both suites are fast (≤ 2 min) and run hermetically with no network or
external fixtures required.  The backcast engine's Tier 2/3 end-to-end tests
depend on the synthetic fixtures having been generated first (i.e., run
the generator's CLI for tiers 1–3 before the engine's tests).


## Calibration results on the Tier 2 fixture

The non-adversarial Tier 2 dataset is a two-regime multivariate Gaussian
DGP with 5 long and 3 short assets (5000 daily rows, overlap starts at row
3000).  Representative pipeline output on this fixture:

    EM:        22 iters, final ΔΣ = 9.6e-9
    HMM:       K=2 selected by BIC, Viterbi label accuracy vs ground truth = 98.6 %
    Holdout:   overall 95 %-PI coverage = 0.952
    Σ recovery: Frobenius relative error = 2.6 % (long-long block: 1.8 %)

Regime-conditional imputation reduces overall backcast RMSE by ~0.5 % and
halves the worst-case per-asset coverage gap vs unconditional EM.


## References

- Stambaugh, R.F. (1997). *Analyzing investments whose histories differ in
  length.*  Journal of Financial Economics, 45(3), 285–331.
- Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys.*  Wiley.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices.*  J. Multivariate Analysis, 88(2),
  365–411.
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series.*  Econometrica, 57(2), 357–384.
- Rabiner, L.R. (1989). *A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition.*  Proceedings of the IEEE, 77(2),
  257–286.
- Bouchaud, J.-P. & Potters, M. (2009). *Financial Applications of Random
  Matrix Theory.*
