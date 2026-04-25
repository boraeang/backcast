# Synthetic Financial Time Series Generator

Standalone Python module that produces synthetic multi-asset return panels
with controlled statistical properties and known ground truth.  Designed as a
development and validation harness for the sibling `backcast_engine`
sub-project, but the generator has zero runtime dependency on it — the only
contract is the CSV + JSON files written to disk.

Four tiers of increasing realism:

| Tier | Module | DGP | Use case |
|---|---|---|---|
| 1 | `tier1_stationary.py` | i.i.d. multivariate Gaussian | Sanity / debug — recover (μ, Σ) exactly |
| 2 | `tier2_regime.py` | Markov regime-switching MVN | Test regime-aware imputation, calibration |
| 3 | `tier3_realistic.py` | GARCH(1,1) factors + Student-t innovations + RW factor loadings | Stress fat tails and TVP betas |
| 4 | `tier4_stress.py` | Edge-case scenarios | Numerical robustness (high-dim, near-singular, staggered) |

Every output bundle includes a `ground_truth.json` with the *exact* DGP
parameters used — μ, Σ, factor loadings, regime labels, GARCH paths,
condition numbers, scenario metadata.  Downstream models can be benchmarked
against truth without ambiguity.


## Layout

```
synthetic_data_generator/
├── README.md                       # this file
├── pyproject.toml
├── requirements.txt
├── src/
│   └── synthgen/
│       ├── __init__.py
│       ├── __main__.py             # enables `python -m synthgen`
│       ├── config.py               # SyntheticConfig + tier configs
│       ├── calendar.py             # business-day index
│       ├── correlation.py          # factor-model / random / manual covariance
│       ├── masking.py              # monotone NaN application
│       ├── io.py                   # CSV + JSON export
│       ├── tier1_stationary.py
│       ├── tier2_regime.py
│       ├── tier3_realistic.py
│       ├── tier4_stress.py
│       └── cli.py                  # argparse CLI driver
├── tests/
│   ├── test_correlation.py
│   ├── test_masking.py
│   ├── test_tier1.py
│   ├── test_tier2.py
│   ├── test_tier3.py
│   ├── test_tier4.py
│   └── test_cli.py
└── examples/
    └── generate_all_tiers.py       # script: one dataset per tier
```


## Quickstart

### Dependencies

Python 3.9+ with:

    numpy>=1.24  pandas>=2.0  scipy>=1.11  pytest>=7.4

No editable install required — set `PYTHONPATH=src` for in-source execution.

### Generate a single tier

    PYTHONPATH=src python -m synthgen --tier 1 --output ./output/tier1

Other examples:

    python -m synthgen --tier 2 --n-regimes 3 --output ./output/tier2
    python -m synthgen --tier 2 --adversarial --output ./output/tier2_adversarial
    python -m synthgen --tier 3 --df 4 --output ./output/tier3
    python -m synthgen --tier 4 --scenario all --output ./output/tier4
    python -m synthgen --tier 1 --n-long 8 --n-short 5 --t-total 7500 --seed 123

Each call writes:

    output/<tier>/
    ├── returns.csv             # masked input (NaN for short-history rows)
    ├── returns_complete.csv    # unmasked truth (validation reference)
    └── ground_truth.json       # full DGP metadata

### Generate all tiers in one go

    python examples/generate_all_tiers.py            # all 4 tiers + Tier 2 adversarial
    python examples/generate_all_tiers.py --tiers 1 3
    python examples/generate_all_tiers.py --output /tmp/synth --seed 99


## Output specification

### `returns.csv`

| col       | type    | meaning |
|-----------|---------|---------|
| `date`    | YYYY-MM-DD | business day index |
| asset 1…N | float   | daily simple returns; NaN until short asset starts |

Long-history assets are observed for every row; short-history assets have a
leading NaN block, then continuous data — the canonical "monotone
missingness" pattern.

### `returns_complete.csv`

Identical layout but no NaN — the values that *would have been* observed if
short assets had existed for the full period.  This is the validation oracle.

### `ground_truth.json`

Always contains:

    tier                       int
    asset_names                list[str]
    long_assets                list[str]
    short_assets               list[str]
    short_asset_start_indices  dict[str, int]
    n_observations             int
    mu, sigma, correlation     true daily parameters
    factor_loadings            B (factor model only)
    factor_covariance          Λ
    idiosyncratic_variance     diag(D)
    seed, correlation_method, n_factors, start_date, missing_fraction

Tier-specific extras:

| Tier | Extra fields |
|---|---|
| 2 | `n_regimes`, `transition_matrix`, `regime_params`, `regime_labels`, `regime_durations`, `regime_counts_overlap`, `regime_counts_backcast`, `adversarial`, `stationary_distribution` |
| 3 | `factor_returns`, `factor_garch_params`, `factor_conditional_vols`, `beta_path`, `beta_initial`, `beta_drift_vol`, `innovation_distribution`, `innovation_df`, `unconditional_sigma`, `rolling_sigma_90d` |
| 4 | `scenario`, `expected_challenges`, `condition_number`, plus scenario-specific entries (e.g., `injected_correlations`, `realised_correlations` for `near_singular`) |


## Calibration notes

Default 5 long + 3 short assets across realistic asset classes, daily vols
calibrated to typical magnitudes:

| Asset       | Class    | Annual mean | Annual vol |
|-------------|----------|-------------|------------|
| `EQUITY_1/2`| Equity   | 7-8 %       | 16-18 %    |
| `BOND_1/2`  | Bond     | 3-4 %       | 5-7 %      |
| `GOLD`      | Commod.  | 4 %         | 15 %       |
| `CRYPTO_1/2`| Crypto   | 20-30 %     | 65-70 %    |
| `ALT_1`     | Alt      | 10 %        | 25 %       |

The factor-model correlation builder enforces realistic inter-class
correlations:

| Pair                      | Target range |
|---------------------------|--------------|
| Equity – Equity           | 0.50 – 0.80  |
| Equity – Bond             | -0.20 – 0.10 |
| Equity – Crypto           | 0.30 – 0.50  |
| Bond – Gold               | 0.10 – 0.30  |
| Crypto – Alt              | 0.30 – 0.60  |

Tier 2 default regimes use the spec-recommended transition matrix:

    P = [[0.98, 0.02],
         [0.05, 0.95]]

→ mean durations ≈ 50 days (calm) and 20 days (crisis), with a 2.5×
volatility multiplier and -0.5× mean shift in the crisis regime.

Tier 3 GARCH defaults: ω = 1×10⁻⁵, α = 0.08, β = 0.90 (α + β = 0.98 → high
persistence, half-life ~34 days), Student-t innovations at ν = 5.

Tier 4 stress scenarios:

| Scenario          | What it tests |
|-------------------|---------------|
| `short_overlap`   | Only ~250 days of joint observation |
| `high_dimension`  | 25 assets (5 long, 20 short) — high short-to-long ratio |
| `near_singular`   | Two pairs at ρ = 0.98 / -0.95; cond(Σ) > 1000 |
| `staggered_heavy` | 10 short assets starting 100 days apart |


## Library API

For programmatic use without the CLI:

    from synthgen import (
        SyntheticConfig, Tier2Config, Tier3Config, Tier4Config,
        generate_tier1, generate_tier2, generate_tier3,
        generate_tier4, generate_tier4_all,
    )

    cfg = SyntheticConfig(n_long_assets=5, n_short_assets=3,
                          t_total=5000, seed=42, tier=1)
    masked, complete, ground_truth = generate_tier1(cfg)

`save_dataset(out_dir, masked, ground_truth, complete)` from `synthgen.io`
writes the three canonical files to disk.


## Testing

    PYTHONPATH=src python -m pytest tests/

Test breakdown (145 total):

| File | Coverage |
|---|---|
| `test_correlation.py`     | factor-model PSD, correlation ranges, manual + random methods |
| `test_masking.py`         | NaN prefix, monotone flag, error cases |
| `test_tier1.py`           | config, calendar, IO, Tier 1 statistics (KS, Ljung-Box, Σ recovery) |
| `test_tier2.py`           | regime sequence, transition matrix, regime-conditional moments, adversarial variant |
| `test_tier3.py`           | GARCH recursion, fat tails, beta-path drift, unconditional variance |
| `test_tier4.py`           | each scenario property, condition number ≥ 1000 for `near_singular` |
| `test_cli.py`             | parser, config build, end-to-end CLI |

The full suite runs in ~5 s.


## References

- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series.*  Econometrica, 57(2), 357–384.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional
  Heteroskedasticity.*  Journal of Econometrics, 31(3), 307–327.
- Higham, N.J. (1988). *Computing a nearest symmetric positive semidefinite
  matrix.*  Linear Algebra and its Applications, 103, 103–118.
