# Methodology

## Overview

This project implements a migration-based portfolio credit risk baseline using synthetic exposures across loans, bonds, and off-balance-sheet instruments. The Phase 1 engine focuses on one-year credit state migration, scenario-based revaluation, and portfolio loss distribution measurement.

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

## Why correlation matters

Independent migration is a useful baseline, but it understates joint tail risk because defaults and downgrades tend to cluster under stressed macro and sector conditions. A more realistic portfolio engine should capture:

- common systematic credit drivers
- co-movement across obligors
- sector-specific stress channels
- heavier portfolio tails and concentration effects

## Latent-factor and copula extensions

The architecture leaves clear extension hooks for latent-variable approaches:

- one-factor models with obligor-level latent credit drivers
- Gaussian copula style dependence across exposures
- sector factors layered on top of a common market factor
- rating-threshold mappings that translate latent draws into migration states

In future phases, the independent sampler can be replaced by a correlated sampler while leaving valuation, metrics, and reporting largely unchanged.

## Stress overlays

Stress analysis can be layered on top of the baseline by modifying core inputs:

- transition matrices can be tilted toward downgrades and default
- LGD assumptions can be increased
- CCF assumptions can be increased for contingent exposures
- spread and discount assumptions can be widened during valuation

This separation keeps the stress framework modular and easy to explain.

## Model boundaries

This repository is a research-oriented baseline and not a production credit platform. Inputs, valuation logic, and scenario generation are intentionally simplified to keep the code readable, testable, and easy to extend.
