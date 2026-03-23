# Methodology

## Overview

This project implements a migration-based portfolio credit risk baseline using synthetic exposures across loans, bonds, and off-balance-sheet instruments. The current engine spans a Phase 1 independent baseline plus one-factor and multi-factor latent migration layers, a regime-based stress overlay, and a simple scenario-dependent spread-shock valuation layer, while keeping the code modular for future correlation and attribution work.

## Migration-based portfolio credit modelling

A migration framework recognizes that credit risk is broader than default. Over a one-year horizon, an exposure may:

- remain in the same rating
- upgrade
- downgrade
- default

Each migration outcome changes valuation and portfolio loss. For default states, losses are driven by exposure and LGD assumptions. For non-default states, valuation changes are driven by simplified spread-sensitive discounting and cash-flow roll-down.

## Independent simulation baseline

Phase 1 treats each exposure transition as conditionally independent. For each simulated scenario:

1. A rating transition is drawn from the exposure-class transition matrix.
2. The migrated state is mapped into an instrument-specific valuation.
3. Exposure-level PnL is aggregated into portfolio-level PnL and loss.
4. The scenario distribution is used to compute VaR and Expected Shortfall.

This baseline is intentionally simple and transparent. It provides a stable foundation for future correlated migration work while already supporting portfolio distribution analytics.

The scenario PnL benchmark is a same-rating one-year reference valued under scenario-consistent LGD, CCF, and spread inputs. This means the reported PnL is intended to isolate migration and stress effects relative to a consistent one-year carry reference rather than to represent a standalone unchanged-name spread book.

## One-factor latent migration

The one-factor extension introduces a shared systematic credit driver so that migrations can cluster in stressed scenarios. For each exposure, the latent credit variable is modeled as:

`X_i = sqrt(rho_i) * Z + sqrt(1 - rho_i) * eps_i`

where:

- `Z` is a common systematic factor
- `eps_i` is an idiosyncratic shock
- `rho_i` is a simple exposure-dependent asset correlation

Transition matrices are converted into latent thresholds, and the latent draw is then mapped into a migrated rating state. In the current implementation, public-sector names carry lower asset correlations, financial names carry medium correlations, and corporates carry higher correlations, with modest uplift for weaker ratings.

## Multi-factor latent migration

The multi-factor extension keeps the same threshold mapping but introduces a sector-specific clustering layer. For each exposure, the latent credit variable is modeled as:

`X_i = sqrt(a_i) * Z_macro + sqrt(b_i) * Z_sector(i) + sqrt(1 - a_i - b_i) * eps_i`

where:

- `Z_macro` is a common macro credit factor
- `Z_sector(i)` is a sector factor linked to the exposure sector
- `eps_i` is an idiosyncratic shock
- `a_i` and `b_i` are bounded exposure-dependent loadings

The current implementation varies macro and sector loadings by issuer type, rating, and sector in a simple readable way. Public-sector names retain lower systematic dependence, financial names sit in the middle, and corporates carry higher dependence, with modest uplift for weaker ratings and sectors that are intended to cluster more strongly.

## Regime-based stress and spread shocks

The stress framework now supports both named static overlays and a simple scenario-level regime mixture. Under `regime` mode, each scenario is assigned one of three synthetic labels:

- `normal`
- `stress`
- `crisis`

Each regime maps to its own transition tilt, LGD multiplier, CCF multiplier, and spread-shift level. This creates a transparent mixture distribution without requiring a dense calibrated copula or a full macroeconomic state model.

Scenario valuation also includes a simple spread-shock layer:

`spread_shock_bps_i = beta_macro_i * Z_macro + beta_sector_i * Z_sector(i) + regime_shift_bps`

The exposure-level spread sensitivity is intentionally simple and depends on issuer type, instrument subtype, seniority, rating, and sector. Public-sector and covered instruments carry lower sensitivity, financials carry medium sensitivity, and weaker corporate or subordinated instruments carry higher sensitivity.

This layer is best viewed as a scenario-dependent valuation overlay that amplifies migration and regime tail behavior. It is not intended to imply a fully standalone spread-MTM model for unchanged names.

## Why correlation matters

Independent migration is a useful baseline, but it understates joint tail risk because defaults and downgrades tend to cluster under stressed macro and sector conditions. A more realistic portfolio engine should capture:

- common systematic credit drivers
- co-movement across obligors
- sector-specific stress channels
- heavier portfolio tails and concentration effects

## Latent-factor and copula extensions

The architecture leaves clear extension hooks for latent-variable approaches:

- one-factor models with obligor-level latent credit drivers
- macro-plus-sector multi-factor models with sector clustering
- Gaussian copula style dependence across exposures
- rating-threshold mappings that translate latent draws into migration states

In future phases, the independent sampler can be replaced by a correlated sampler while leaving valuation, metrics, and reporting largely unchanged.

## Stress overlays

Stress analysis can be layered on top of the baseline by modifying core inputs:

- transition matrices can be tilted toward downgrades and default
- LGD assumptions can be increased
- CCF assumptions can be increased for contingent exposures
- spread and discount assumptions can be widened during valuation

This separation keeps the stress framework modular and easy to explain.

The current stress overlays provide named `none`, `mild`, and `severe` regimes plus a simple scenario-level `regime` mixture. These stress modes tilt transition matrices toward downgrades and defaults and simultaneously uplift LGD and CCF assumptions. Scenario-consistent spread inputs then flow through valuation on top of that migration layer.

## Diagnostics and tail interpretation

The engine reports:

- distribution statistics such as mean, standard deviation, skewness, and tail quantiles
- VaR and Expected Shortfall
- default and downgrade summaries by exposure class and issuer type
- scenario-level default count clustering diagnostics
- tail diagnostics such as clustered-default probabilities and worst-1% average default and downgrade counts
- worst-1% loss attribution by issuer type, sector, instrument subtype, and rating bucket

These outputs help illustrate the difference between an independent migration engine and factor-driven migration engines even before a full Gaussian copula framework is introduced.

## Future phases

Important future extensions still left for later phases include:

- Gaussian copula style dependence
- deeper sector-factor calibration
- contribution to VaR and Expected Shortfall
- VaR and Expected Shortfall attribution decomposition
- richer stress overlays on spread and discount assumptions
- more advanced attribution and concentration diagnostics

## Model boundaries

This repository is a research-oriented baseline and not a production credit platform. Inputs, valuation logic, and scenario generation are intentionally simplified to keep the code readable, testable, and easy to extend.
