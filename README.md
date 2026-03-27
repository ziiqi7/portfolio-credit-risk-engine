# portfolio-credit-risk-engine

`portfolio-credit-risk-engine` is a synthetic-data-only Python project for migration-based portfolio credit risk. It provides a readable Phase 1 baseline across loans, bonds, and off-balance-sheet exposures, with a structure designed to grow into a more advanced quant research engine for correlated migration, factor models, and stress analysis.

## Why migration-based credit portfolio models matter

Portfolio credit risk is not only about default. Over a one-year horizon, rating migration can change mark-to-market valuation, expected loss, spread sensitivity, and portfolio tail outcomes well before an obligor actually defaults. Migration-based models are therefore a practical foundation for:

- scenario-based portfolio loss distributions
- rating-sensitive revaluation across multiple instrument types
- VaR and Expected Shortfall measurement
- future latent-factor and copula-style correlation models

## Synthetic data only

The repository uses synthetic public-safe data only. All obligors, facilities, balances, ratings, sectors, maturities, and transition assumptions are artificial. No proprietary, internal, or real-company data is used.

## Phase 1 baseline

Phase 1 is implemented and runnable today with:

- a unified exposure schema for loans, bonds, and off-balance-sheet exposures
- a synthetic portfolio generator with repeated obligors, multi-facility names, sector concentrations, issuer segmentation, and class-specific instrument mixes
- bundled one-year synthetic transition matrices for corporate, financial institution, and sovereign exposures
- instrument-specific valuation with spread and LGD assumptions that vary by issuer type, instrument subtype, rating, and seniority
- independent Monte Carlo migration simulation
- one-factor latent-variable migration simulation
- multi-factor latent-variable migration simulation with sector clustering
- runnable static and regime-based stress overlays on transition matrices, LGD, CCF, and spread shifts
- a simple scenario-dependent spread-shock valuation layer driven by macro, sector, and regime inputs
- portfolio PnL and loss distributions
- VaR at 95%, 99%, and 99.9%
- Expected Shortfall at 99%
- richer tail diagnostics, mode comparison, CSV export, and loss distribution plotting

## Synthetic portfolio design

The bundled synthetic portfolio is intentionally more structured than a random row generator. It includes:

- repeat obligor IDs with multiple facilities per name
- concentration across sectors and obligors rather than uniformly scattered rows
- distinct credit mixes for corporate, FI, and public-sector books
- issuer segmentation across sovereign, supranational, agency, bank, insurance, and corporate names
- bond segmentation across sovereign, agency, supranational, covered, bank senior, and corporate bond subtypes
- seniority and secured-status fields that feed both valuation and reporting
- instrument-specific maturity ranges
- coupon logic tied to rating, currency, issuer type, instrument subtype, and seniority
- off-balance-sheet facilities with meaningful undrawn amounts for CCF treatment
- simple but defensible guarantee and collateral patterns

## Transition matrix design

The bundled transition matrices are synthetic but designed to be plausible:

- strong diagonal mass
- worsening downgrade and default risk as starting ratings deteriorate
- lower default risk for stronger ratings
- distinct profiles for corporates, financial institutions, and sovereigns
- validation checks for row sums, state ordering, and monotonic migration intuition

These matrices are intended as a transparent public baseline, not as calibrated market estimates.

## Future phases

The architecture is intentionally prepared for:

- Phase 2: richer latent dependence, Gaussian copula style extensions, and deeper threshold mapping
- Phase 3: deeper sector-factor calibration
- Phase 3: contribution to VaR and Expected Shortfall decomposition
- Phase 3: richer attribution and concentration layers

The project remains a synthetic-data research engine and not a production credit platform.

## Repository layout

```text
portfolio-credit-risk-engine/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── synthetic/
│   │   └── sample_portfolio.csv
│   └── transitions/
│       ├── transition_matrix_corporate.csv
│       ├── transition_matrix_fi.csv
│       └── transition_matrix_sovereign.csv
├── docs/
│   └── methodology.md
├── notebooks/
│   ├── 01_baseline_portfolio_demo.ipynb
│   └── 02_correlation_roadmap.ipynb
├── scripts/
│   ├── generate_synthetic_portfolio.py
│   ├── run_demo.py
│   └── run_experiments.py
├── src/
│   ├── config.py
│   ├── schema.py
│   ├── synthetic_data.py
│   ├── transitions.py
│   ├── valuation.py
│   ├── simulation.py
│   ├── metrics.py
│   ├── reporting.py
│   ├── stress.py
│   └── correlation.py
└── tests/
    ├── test_schema.py
    ├── test_transitions.py
    ├── test_valuation.py
    └── test_metrics.py
```

## Installation

```bash
cd portfolio-credit-risk-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the baseline demo

Generate a synthetic portfolio:

```bash
python scripts/generate_synthetic_portfolio.py --num-exposures 150 --seed 42
```

Run the independent migration engine:

```bash
python scripts/run_demo.py --simulation-mode independent --stress none --scenarios 5000 --seed 42
```

Run the one-factor latent migration engine:

```bash
python scripts/run_demo.py --simulation-mode one_factor --stress none --scenarios 5000 --seed 42
```

Run the multi-factor latent migration engine with sector clustering:

```bash
python scripts/run_demo.py --simulation-mode multi_factor --stress none --scenarios 5000 --seed 42
```

Run multi-factor with mild stress overlays:

```bash
python scripts/run_demo.py --simulation-mode multi_factor --stress mild --scenarios 5000 --seed 42
```

Run multi-factor with regime-based stress:

```bash
python scripts/run_demo.py --simulation-mode multi_factor --stress regime --scenarios 5000 --seed 42
```

Compare all simulation modes on the same portfolio, seed, and stress setup:

```bash
python scripts/run_demo.py --simulation-mode multi_factor --stress regime --scenarios 5000 --seed 42 --compare-modes
```

Run the Python batch experiment pipeline:

```bash
python scripts/run_experiments.py
```

The demo will:

- load the sample synthetic portfolio
- load the demo transition matrices
- run the independent, one-factor, or multi-factor migration engine
- optionally apply named stress overlays or a regime mixture to transitions, LGD, CCF, and the spread-shock valuation layer
- benchmark migrated valuations against a same-rating one-year reference under scenario-consistent LGD, CCF, and spread inputs
- print key portfolio metrics
- print default, downgrade, regime, and tail diagnostics
- save a loss distribution plot to `outputs/demo_loss_distribution.png`
- optionally export scenario-level results

The preferred orchestration layer for batch runs is now Python via `scripts/run_experiments.py` rather than a shell wrapper.

## Running Batch Experiments

`python scripts/run_experiments.py` runs a fixed public batch of experiments:

- `independent` with `stress=none`
- `one_factor` with `stress=none`
- `multi_factor` with `stress=none`
- `multi_factor` with `stress=regime`
- `multi_factor` with `stress=regime --compare-modes`

It writes:

- per-run logs to `outputs/logs/`
- per-run scenario CSVs to `outputs/`
- per-run loss plots to `outputs/`
- a comparison chart to `outputs/comparison_distribution.png`

The standard log files are:

- `outputs/logs/independent.log`
- `outputs/logs/one_factor.log`
- `outputs/logs/multi_factor.log`
- `outputs/logs/multi_factor_regime.log`
- `outputs/logs/comparison.log`

## Example outputs

Typical demo outputs include:

- mean portfolio PnL and mean portfolio loss
- loss quantiles and skewness diagnostics
- 95%, 99%, and 99.9% loss VaR
- 99% Expected Shortfall
- default and downgrade summaries by exposure class and issuer type
- probability of clustered defaults and downgrades in the loss tail
- worst-1% tail attribution by issuer type, sector, instrument subtype, and rating bucket
- expected loss breakdowns by instrument type
- expected loss breakdowns by starting rating bucket
- a histogram of portfolio loss outcomes across simulated migration scenarios

Example console summary:

```text
Portfolio Credit Risk Demo
--------------------------
Exposures: 150
Scenarios: 2000
Mean PnL: -18,100,244.19
Mean Loss: 18,100,244.19
VaR 95%: 37,197,155.28
VaR 99%: 46,059,451.38
VaR 99.9%: 55,425,094.48
ES 99%: 50,720,603.82
```

Values will vary with the seed, synthetic portfolio mix, and scenario count.

The current spread-shock layer should be read as a scenario-dependent valuation overlay that amplifies migration and regime tail behavior. It is not presented as a full standalone spread-MTM engine for otherwise unchanged names.

## Engineering principles

- modular, readable Python
- synthetic data only
- minimal dependencies
- simple but real tests
- deliberate extension points rather than overengineered abstractions
- no claims of production readiness

## Roadmap

### Phase 1

Independent migration simulation, one-factor latent migration, multi-factor latent migration with sector clustering, regime-based stress, a simple scenario-dependent spread-shock valuation layer, portfolio loss distribution, VaR and ES, and reporting.

### Phase 2

Gaussian copula style extensions, richer factor structures, and deeper rating-threshold mapping.

### Phase 3

Deeper sector-factor calibration, VaR and ES attribution decomposition, and richer stress overlays across matrices, LGD, CCF, and valuation assumptions.

## License

MIT License. See [LICENSE](LICENSE).
