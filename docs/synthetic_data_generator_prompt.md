# Synthetic Financial Time Series Generator — Claude Code Prompt

## Context & Objective

You are building a **standalone Python module** that generates synthetic financial time series with controlled statistical properties and known ground truth. The primary purpose is to serve as a development and validation tool for models that impute missing financial data — but the generator itself must be **fully independent** of any imputation code.

The generator produces CSV files that mimic real-world multi-asset return datasets where some assets have shorter histories than others (monotone missingness pattern). Because the data-generating process (DGP) is fully specified, the user knows the true parameters ($\mu$, $\Sigma$, regime labels, factor loadings, etc.) and can validate any downstream model against ground truth.

## Key Design Principles

1. **Independence**: This module has zero dependencies on any backcast/imputation library. It produces CSV files and ground-truth metadata — nothing more.
2. **Known ground truth**: Every generated dataset comes with a companion file containing the true parameters used to generate it. This is the entire point.
3. **Layered complexity**: Four tiers of increasing realism, each adding one source of complexity. Debug against Tier 1, stress-test against Tier 4.
4. **Real-world calibration**: Parameter magnitudes should match actual financial data — daily equity returns ~3-5 bps mean / ~100 bps vol, bonds ~1 bp mean / ~30 bps vol, BTC-like assets ~15-20 bps mean / ~400 bps vol. Numerical stability issues often only surface at realistic scales.

## Output Specification

Each call to the generator produces:

1. **`returns.csv`**: The synthetic returns matrix.
   - Rows = dates (business days, starting from a configurable start date, e.g., 1990-01-02)
   - Columns = asset names (e.g., `EQUITY_1`, `EQUITY_2`, `BOND_1`, `GOLD`, `CRYPTO_1`, `ALT_1`)
   - Values = daily simple returns
   - Short-history assets have `NaN` before their start date (monotone missingness)

2. **`ground_truth.json`**: The true DGP parameters.
   - `mu`: true mean vector (list of floats)
   - `sigma`: true covariance matrix (list of lists)
   - `correlation`: true correlation matrix
   - `asset_names`: list of asset names
   - `long_assets`: list of assets with full history
   - `short_assets`: list of assets with partial history
   - `short_asset_start_indices`: dict mapping short asset name → row index where it starts
   - `n_observations`: total number of dates
   - `tier`: which tier was used (1, 2, 3, or 4)
   - Tier-specific fields (see below)

3. **`returns_complete.csv`** (optional but recommended): The full returns matrix **without** any masking — i.e., the returns that *would have* been observed if all assets existed for the full period. This is the ultimate validation target.

## Project Structure

```
synthetic_data_generator/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── synthgen/
│       ├── __init__.py
│       ├── config.py            # Dataclass for generator configuration
│       ├── correlation.py       # Correlation/covariance matrix builders
│       ├── tier1_stationary.py  # Tier 1: Stationary Gaussian
│       ├── tier2_regime.py      # Tier 2: Regime-switching Gaussian
│       ├── tier3_realistic.py   # Tier 3: GARCH + fat tails + TVP betas
│       ├── tier4_stress.py      # Tier 4: Edge cases and stress tests
│       ├── masking.py           # Apply missingness patterns
│       ├── calendar.py          # Business day calendar generation
│       ├── io.py                # CSV / JSON export
│       └── cli.py               # Command-line interface
├── tests/
│   ├── test_correlation.py
│   ├── test_tier1.py
│   ├── test_tier2.py
│   ├── test_tier3.py
│   ├── test_tier4.py
│   └── test_masking.py
└── examples/
    └── generate_all_tiers.py    # Script that generates one dataset per tier
```

## Module Specifications

---

### Module: `config.py`

Define a configuration dataclass that controls all aspects of data generation.

```python
@dataclass
class SyntheticConfig:
    # Dimensions
    n_long_assets: int = 5          # Assets with full history
    n_short_assets: int = 3         # Assets with partial history
    t_total: int = 5000             # Total trading days (~20 years)

    # Missingness
    short_start_day: int | list[int] = 3000
    # If int: all short assets start at the same day
    # If list: staggered start dates (length must equal n_short_assets)

    # Calendar
    start_date: str = "1990-01-02"  # First trading day
    calendar: str = "NYSE"          # Business day calendar to use

    # Correlation structure
    correlation_method: str = "factor_model"  # or "random" or "manual"
    n_factors: int = 4              # Number of latent factors (if factor_model)

    # Asset class calibration (annualized, will be converted to daily internally)
    # Format: (mean_annual_pct, vol_annual_pct)
    asset_profiles: dict | None = None
    # If None, use defaults:
    # Equity-like:  mean=8%, vol=16%
    # Bond-like:    mean=3%, vol=5%
    # Gold-like:    mean=4%, vol=15%
    # Crypto-like:  mean=30%, vol=70%
    # Alt-like:     mean=10%, vol=25%

    # Tier-specific
    tier: int = 1
    tier2_config: "Tier2Config | None" = None
    tier3_config: "Tier3Config | None" = None
    tier4_config: "Tier4Config | None" = None

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "./synthetic_output"
    save_complete_returns: bool = True
```

```python
@dataclass
class Tier2Config:
    n_regimes: int = 2
    transition_matrix: list[list[float]] | None = None
    # If None, generate a plausible one (high diagonal, low off-diagonal)
    # Regime parameter multipliers relative to base:
    regime_vol_multipliers: list[float] | None = None  # e.g., [1.0, 2.5] for calm/crisis
    regime_corr_adjustments: list[float] | None = None  # e.g., [0.0, 0.3] added to off-diag corr
    regime_mean_adjustments: list[float] | None = None  # e.g., [0.0, -0.5] multiplier on mean
    adversarial: bool = False
    # If True: one regime appears ONLY in the backcast period (never in overlap)
    # This tests whether models can handle unseen regimes

@dataclass
class Tier3Config:
    innovation_distribution: str = "student_t"  # or "gaussian"
    degrees_of_freedom: float = 5.0             # for student-t
    garch_omega: float = 0.00001                # GARCH(1,1) intercept
    garch_alpha: float = 0.08                   # ARCH coefficient
    garch_beta: float = 0.90                    # GARCH coefficient
    beta_drift_vol: float = 0.001               # Std dev of random walk in factor loadings
    # Per-period, so small = slow drift, large = fast drift

@dataclass
class Tier4Config:
    scenario: str = "short_overlap"
    # Options:
    # "short_overlap"     — only 250 days of overlap (~1 year)
    # "high_dimension"    — 20 short assets, 5 long assets
    # "near_singular"     — two assets with 0.98 correlation
    # "staggered_heavy"   — 10 short assets each starting 100 days apart
    # "all"               — generate all scenarios
```

---

### Module: `correlation.py`

**Purpose**: Build realistic covariance matrices with known structure.

**Method 1 — Factor model (recommended default)**:

Define $K$ latent factors with interpretable labels:
- Factor 1: **Equity risk** — loads heavily on equity-like assets, moderately on alts
- Factor 2: **Rates/duration** — loads on bonds, inversely on equities (mild)
- Factor 3: **Inflation/real assets** — loads on gold, commodities, TIPS-like
- Factor 4: **Speculative/crypto** — loads on crypto and speculative alts

Build the covariance as:
$$\Sigma = B \Lambda B^T + D$$

Where:
- $B$ is the $(N \times K)$ factor loading matrix
- $\Lambda$ is the $(K \times K)$ factor covariance (diagonal for simplicity, or allow mild cross-factor correlation)
- $D$ is the $(N \times N)$ diagonal idiosyncratic variance matrix

The factor loading matrix $B$ should be calibrated so that the resulting asset-level volatilities and correlations match typical real-world values:
- Equity–equity correlation: 0.5–0.8
- Equity–bond correlation: -0.2–0.1
- Equity–gold correlation: 0.0–0.15
- Equity–crypto correlation: 0.3–0.5 (during overlap period)
- Bond–gold correlation: 0.1–0.3
- Crypto–alt correlation: 0.3–0.6

**Store the factor model components** ($B$, $\Lambda$, $D$) in the ground truth file — they're useful for validating factor-based imputation methods.

**Method 2 — Random correlation matrix**:

Generate a random valid correlation matrix via the vine method or by generating a random Cholesky factor. Useful for stress testing but less interpretable.

**Method 3 — Manual**:

Accept a user-provided correlation matrix and volatility vector. Validate that it's PSD.

**All methods must output**:
- A valid PSD covariance matrix $\Sigma$
- The corresponding correlation matrix
- Annualized and daily versions

---

### Module: `tier1_stationary.py`

**Purpose**: Generate i.i.d. multivariate Gaussian returns.

**DGP**:
$$R_t \sim N(\mu, \Sigma) \quad \text{i.i.d. for } t = 1, ..., T$$

**Implementation**:
1. Build $\mu$ from asset profiles (convert annualized mean to daily: $\mu_{daily} = \mu_{annual} / 252$).
2. Build $\Sigma$ using `correlation.py` (convert annualized vol to daily: $\sigma_{daily} = \sigma_{annual} / \sqrt{252}$).
3. Draw $T$ observations from $N(\mu, \Sigma)$ using `numpy.random.Generator.multivariate_normal`.
4. Create date index using business day calendar.
5. Store as DataFrame with asset names as columns.

**Ground truth extras for Tier 1**:
- `mu_daily`: daily mean vector
- `sigma_daily`: daily covariance matrix
- `factor_loadings`: $B$ matrix (if factor model was used)
- `factor_covariance`: $\Lambda$ matrix
- `idiosyncratic_variance`: $D$ diagonal

**Validation criteria** (put these in docstrings so the developer knows what to check):
- Sample mean should be within $2\sigma/\sqrt{T}$ of true mean for each asset.
- Sample covariance should be close to true covariance (Frobenius norm relative error < 5% for T=5000).
- Marginal distributions should pass KS test for normality.
- No autocorrelation (Ljung-Box test should not reject at 5% for reasonable lag).

---

### Module: `tier2_regime.py`

**Purpose**: Generate returns from a regime-switching multivariate Gaussian.

**DGP**:
$$s_t \sim \text{Markov}(P) \quad \text{(hidden state)}$$
$$R_t | s_t = k \sim N(\mu^{(k)}, \Sigma^{(k)})$$

Where $P$ is the $(K \times K)$ transition matrix and $K$ is the number of regimes.

**Implementation**:
1. Build base $(\mu, \Sigma)$ as in Tier 1.
2. Create regime-specific parameters by applying multipliers:
   - **Calm regime** ($k=1$): Use base parameters as-is (or with vol multiplier ~1.0).
   - **Crisis regime** ($k=2$): Inflate volatilities (multiplier ~2.0–2.5), increase correlations (add ~0.2–0.3 to off-diagonal correlations, clip to valid range), reduce means (multiply by ~-0.5 or shift negative).
   - Ensure each $\Sigma^{(k)}$ is PSD after adjustments (re-do nearest PSD projection if needed).
3. Generate regime sequence from the Markov chain:
   - Default transition matrix: high persistence (e.g., $P_{11} = 0.98$, $P_{22} = 0.95$).
   - This produces regimes that last ~50 days (calm) and ~20 days (crisis) on average — realistic.
4. For each date, draw returns from the regime-specific distribution.

**Adversarial variant** (when `adversarial=True`):
- Ensure that one regime (e.g., the crisis regime) appears ONLY in the backcast period (before `short_start_day`), never during the overlap.
- Implement this by overriding the regime sequence: force all overlap-period dates into the calm regime.
- This tests whether models can handle distributional shifts they've never observed.

**Ground truth extras for Tier 2**:
- `regime_labels`: full array of regime labels (length $T$)
- `transition_matrix`: true $P$
- `regime_params`: dict mapping regime index → `{mu, sigma, correlation}`
- `regime_durations`: summary statistics of how long each regime lasted
- `adversarial`: whether the adversarial variant was used
- `regime_counts_overlap`: how many days of each regime fall in the overlap period
- `regime_counts_backcast`: how many days of each regime fall in the backcast period

---

### Module: `tier3_realistic.py`

**Purpose**: Generate returns with realistic financial dynamics — fat tails, volatility clustering, and time-varying betas.

**DGP**:

Step 1 — Generate factor returns with GARCH dynamics:
$$f_{j,t} = \mu_{f,j} + \sigma_{j,t} \cdot z_{j,t}$$
$$\sigma_{j,t}^2 = \omega_j + \alpha_j \cdot (f_{j,t-1} - \mu_{f,j})^2 + \beta_j \cdot \sigma_{j,t-1}^2$$
$$z_{j,t} \sim t_\nu \quad \text{(standardized Student-t with } \nu \text{ degrees of freedom)}$$

For $j = 1, ..., K$ factors. The GARCH parameters $(\omega, \alpha, \beta)$ can be shared across factors or factor-specific.

Step 2 — Generate time-varying factor loadings:
$$B_t = B_{t-1} + \eta_t, \quad \eta_t \sim N(0, \sigma_\eta^2 I)$$

Where $B_0$ is the base factor loading matrix from `correlation.py` and $\sigma_\eta$ is small (e.g., 0.001) so betas drift slowly. Clip betas to prevent sign flips or explosive growth: enforce $|B_{ij,t}| < 3 \cdot |B_{ij,0}|$.

Step 3 — Generate asset returns:
$$R_t = \alpha + B_t \cdot f_t + \epsilon_t$$
$$\epsilon_{i,t} \sim t_\nu(0, d_i) \quad \text{(idiosyncratic shocks, also fat-tailed)}$$

Step 4 — Optionally overlay regime switching on the GARCH dynamics:
- In crisis regimes, scale up $\omega$ (base GARCH variance) by a multiplier.
- This creates periods where GARCH vol is persistently elevated.

**Implementation notes**:
- Generate factor returns first (they're independent of asset returns in the DGP).
- Then generate asset returns conditional on factors.
- The realized covariance matrix is NOT constant — it varies over time. The "true" covariance is the unconditional covariance, which can be computed analytically from the GARCH parameters (unconditional variance = $\omega / (1 - \alpha - \beta)$) and the average factor loadings.
- Store both the unconditional covariance and a time series of realized rolling covariances.

**Ground truth extras for Tier 3**:
- `factor_returns`: the full factor return matrix ($T \times K$)
- `factor_garch_params`: $(\omega, \alpha, \beta)$ per factor
- `factor_conditional_vols`: time series of $\sigma_{j,t}$ for each factor
- `beta_path`: the full time series of $B_t$ (shape $T \times N \times K$)
- `innovation_df`: degrees of freedom of Student-t innovations
- `unconditional_sigma`: the analytically computed unconditional covariance matrix
- `rolling_sigma_90d`: rolling 90-day realized covariance at every date (for comparison)

---

### Module: `tier4_stress.py`

**Purpose**: Generate edge-case datasets that test numerical robustness.

Implement the following scenarios (selectable via `Tier4Config.scenario`):

**Scenario: `short_overlap`**
- Set `short_start_day = t_total - 250` so there's only ~1 year of overlap.
- Use Tier 1 DGP (stationary Gaussian) so the only challenge is the short overlap.
- Tests whether models can estimate reliable parameters from limited data.

**Scenario: `high_dimension`**
- Set `n_long_assets = 5`, `n_short_assets = 20`.
- The covariance matrix is $25 \times 25$ but estimated from only 5 fully-observed assets.
- Tests whether EM handles the high short-to-long ratio.
- Use factor model with fewer factors than short assets to ensure the covariance is well-structured.

**Scenario: `near_singular`**
- Include two assets with a 0.98 pairwise correlation (one long, one short).
- Include another pair with -0.95 correlation.
- Tests Cholesky stability and matrix conditioning.
- The covariance matrix should still be technically PSD but have a very high condition number.

**Scenario: `staggered_heavy`**
- 10 short-history assets, each starting 100 trading days apart.
- Asset 1 starts at day 2000, asset 2 at day 2100, ..., asset 10 at day 2900.
- Tests the staggered missingness handling (sequential group processing in EM).
- Use Tier 1 DGP so the only complexity is the staggered pattern.

**Scenario: `all`**
- Generate all four scenarios above into separate subdirectories.

**Ground truth extras for Tier 4**:
- `scenario`: which stress test was run
- `expected_challenges`: string describing what this scenario is designed to test
- `condition_number`: condition number of the true covariance matrix
- Include all Tier 1 ground truth fields as the base.

---

### Module: `masking.py`

**Purpose**: Apply the monotone missingness pattern to a complete returns DataFrame.

**Implementation**:
- Accept a complete DataFrame and a dict of `{asset_name: start_index}`.
- Set all values for each short asset before its start index to `NaN`.
- Validate that the result has monotone missingness (no mid-series gaps).
- Return both the masked DataFrame and metadata about the missingness pattern.

This module is deliberately simple — it's separated from the tier modules so that any tier can use it, and so you can test masking logic independently.

---

### Module: `calendar.py`

**Purpose**: Generate realistic business day date indices.

**Implementation**:
- Generate business day dates starting from `start_date` for `t_total` trading days.
- Use `pandas.bdate_range` with the specified calendar.
- Handle edge cases: if `start_date` falls on a weekend/holiday, move to next business day.
- Do NOT include weekends or holidays — the output should have consecutive business days only (this matches how real financial data is stored).

---

### Module: `io.py`

**Purpose**: Export generated data to CSV and ground truth to JSON.

**Implementation**:
- Write `returns.csv` with `date` as the first column, formatted as `YYYY-MM-DD`.
- Write `returns_complete.csv` (the unmasked version).
- Write `ground_truth.json` with all parameters. Convert numpy arrays to lists for JSON serialization. Use `float` not `numpy.float64` to avoid JSON encoding issues.
- Create the output directory if it doesn't exist.
- Log file paths and sizes after writing.

---

### Module: `cli.py`

**Purpose**: Command-line interface for generating synthetic datasets.

**Usage**:
```bash
# Generate Tier 1 with defaults
python -m synthgen --tier 1 --output ./data/tier1

# Generate Tier 2 with 3 regimes
python -m synthgen --tier 2 --n-regimes 3 --output ./data/tier2

# Generate Tier 2 adversarial variant
python -m synthgen --tier 2 --adversarial --output ./data/tier2_adversarial

# Generate Tier 3 with student-t innovations (df=4)
python -m synthgen --tier 3 --df 4 --output ./data/tier3

# Generate Tier 4 all stress scenarios
python -m synthgen --tier 4 --scenario all --output ./data/tier4

# Custom dimensions
python -m synthgen --tier 1 --n-long 8 --n-short 5 --t-total 7500 --seed 123
```

Use `argparse`. Map CLI arguments to `SyntheticConfig` fields.

---

## Asset Naming and Profiles

Use descriptive names that reflect asset class, not ticker symbols:

| Asset Name     | Class   | Annual Mean | Annual Vol | History |
|----------------|---------|-------------|------------|---------|
| `EQUITY_1`     | Equity  | 8%          | 16%        | Long    |
| `EQUITY_2`     | Equity  | 7%          | 18%        | Long    |
| `BOND_1`       | Bond    | 3%          | 5%         | Long    |
| `BOND_2`       | Bond    | 4%          | 7%         | Long    |
| `GOLD`         | Commod  | 4%          | 15%        | Long    |
| `CRYPTO_1`     | Crypto  | 30%         | 70%        | Short   |
| `CRYPTO_2`     | Crypto  | 20%         | 65%        | Short   |
| `ALT_1`        | Alt     | 10%         | 25%        | Short   |

These are defaults — configurable via `asset_profiles` in the config.

Convert annualized parameters to daily:
- $\mu_{daily} = \mu_{annual} / 252$
- $\sigma_{daily} = \sigma_{annual} / \sqrt{252}$

---

## Implementation Guidelines

### Code Quality
- **Python 3.10+** required. Use modern syntax: `match` statements, `X | Y` union types, dataclasses with `slots=True`.
- **Type hints everywhere** — all functions fully annotated.
- **NumPy-style docstrings** on all public functions.
- **Logging** via Python `logging` module. INFO for tier-level steps, DEBUG for per-observation details.
- **Reproducibility**: All random operations use `numpy.random.Generator` from a single seeded `default_rng(seed)`. No global random state. Pass the generator explicitly to all functions that need randomness.

### Numerical
- Validate that all generated covariance matrices are PSD (check eigenvalues).
- After applying regime adjustments to correlation matrices, project back to nearest PSD if needed (Higham algorithm or eigenvalue clipping).
- Use `np.float64` throughout.

### Testing
- `pytest` tests for each tier.
- **Tier 1 tests**: verify sample statistics converge to true parameters at $T=5000$. Verify KS test for marginal normality. Verify no autocorrelation.
- **Tier 2 tests**: verify regime sequence follows the transition matrix (empirical transition frequencies within 2 standard errors). Verify regime-conditional sample statistics match true regime parameters.
- **Tier 3 tests**: verify marginal kurtosis > 3 (fat tails present). Verify autocorrelation in squared returns (GARCH signature). Verify unconditional vol is close to analytical unconditional vol.
- **Tier 4 tests**: verify each stress scenario has the intended property (e.g., `near_singular` has condition number > 1000).
- **Masking tests**: verify monotone missingness, verify no data corruption during masking.

### Dependencies (Python 3.10+)
```
numpy>=1.24
pandas>=2.0
scipy>=1.11
pytest>=7.4
```

No other dependencies required. This module should be lightweight.

---

## Execution Order

Build and test in this order:

1. `config.py` → define all dataclasses
2. `calendar.py` → verify business day generation
3. `correlation.py` → verify PSD, verify factor model produces realistic correlations
4. `masking.py` → verify monotone missingness
5. `io.py` → verify CSV/JSON export round-trips correctly
6. `tier1_stationary.py` + tests → verify sample stats match true params
7. `tier2_regime.py` + tests → verify regime detection and conditional params
8. `tier3_realistic.py` + tests → verify GARCH signature and fat tails
9. `tier4_stress.py` + tests → verify each scenario's intended property
10. `cli.py` → verify all CLI arguments work
