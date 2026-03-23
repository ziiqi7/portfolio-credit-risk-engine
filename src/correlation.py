"""Latent-factor migration utilities and future correlation hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.schema import Exposure
from src.transitions import TransitionMatrix, threshold_mapping_inputs


@dataclass(slots=True)
class LatentFactorSpec:
    """Configuration for latent-factor migration engines."""

    asset_correlation: float = 0.20
    issuer_type_base_correlations: dict[str, float] = field(
        default_factory=lambda: {
            "sovereign": 0.08,
            "supranational": 0.07,
            "agency": 0.09,
            "bank": 0.18,
            "insurance": 0.16,
            "corporate": 0.24,
        }
    )
    rating_adjustments: dict[str, float] = field(
        default_factory=lambda: {
            "AAA": -0.02,
            "AA": -0.01,
            "A": 0.00,
            "BBB": 0.02,
            "BB": 0.04,
            "B": 0.06,
            "CCC": 0.08,
        }
    )
    min_correlation: float = 0.03
    max_correlation: float = 0.45
    random_seed: int | None = None


@dataclass(slots=True)
class MultiFactorSpec:
    """Configuration for a simple macro-plus-sector latent migration engine."""

    issuer_type_macro_loadings: dict[str, float] = field(
        default_factory=lambda: {
            "sovereign": 0.06,
            "supranational": 0.05,
            "agency": 0.07,
            "bank": 0.14,
            "insurance": 0.13,
            "corporate": 0.18,
        }
    )
    issuer_type_sector_loadings: dict[str, float] = field(
        default_factory=lambda: {
            "sovereign": 0.02,
            "supranational": 0.02,
            "agency": 0.03,
            "bank": 0.10,
            "insurance": 0.09,
            "corporate": 0.12,
        }
    )
    rating_macro_adjustments: dict[str, float] = field(
        default_factory=lambda: {
            "AAA": -0.01,
            "AA": -0.005,
            "A": 0.0,
            "BBB": 0.01,
            "BB": 0.02,
            "B": 0.03,
            "CCC": 0.04,
        }
    )
    rating_sector_adjustments: dict[str, float] = field(
        default_factory=lambda: {
            "AAA": -0.005,
            "AA": -0.003,
            "A": 0.0,
            "BBB": 0.008,
            "BB": 0.015,
            "B": 0.022,
            "CCC": 0.03,
        }
    )
    sector_base_loadings: dict[str, float] = field(
        default_factory=lambda: {
            "banking": 0.11,
            "insurance": 0.10,
            "industrials": 0.12,
            "energy": 0.13,
            "consumer": 0.11,
            "real_estate": 0.14,
            "technology": 0.10,
            "healthcare": 0.09,
            "public_sector": 0.04,
            "agency": 0.05,
            "multilateral": 0.04,
            "diversified_financials": 0.11,
            "asset_management": 0.10,
        }
    )
    min_macro_loading: float = 0.01
    max_macro_loading: float = 0.35
    min_sector_loading: float = 0.01
    max_sector_loading: float = 0.25
    max_total_loading: float = 0.80
    random_seed: int | None = None


def build_rating_thresholds(transition_matrix: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Convert transition probabilities into standard-normal thresholds.

    Thresholds are ordered from worst state to best state so they can be used
    with latent credit variables where lower draws correspond to weaker
    credit outcomes.
    """

    normal = NormalDist()
    threshold_inputs = threshold_mapping_inputs(transition_matrix)
    worst_to_best_states = list(threshold_inputs.columns)
    thresholds: dict[str, list[tuple[str, float]]] = {}

    for start_rating, row in threshold_inputs.iterrows():
        cumulative = row.iloc[:-1]
        thresholds[start_rating] = [
            (state, normal.inv_cdf(min(max(float(probability), 1e-6), 1.0 - 1e-6)))
            for state, probability in zip(worst_to_best_states[:-1], cumulative)
        ]
    return thresholds


def build_threshold_lookup(
    transition_matrices: dict[str, TransitionMatrix],
) -> dict[str, dict[str, tuple[list[tuple[str, float]], str]]]:
    """Build threshold lookup tables keyed by exposure class and start rating."""

    lookup: dict[str, dict[str, tuple[list[tuple[str, float]], str]]] = {}
    for exposure_class, matrix in transition_matrices.items():
        thresholds = build_rating_thresholds(matrix.matrix)
        fallback_state = matrix.threshold_cumulative().columns[-1]
        lookup[exposure_class] = {
            start_rating: (threshold_rows, fallback_state)
            for start_rating, threshold_rows in thresholds.items()
        }
    return lookup


def map_latent_to_rating_state(
    latent_value: float,
    thresholds: list[tuple[str, float]],
    fallback_state: str,
) -> str:
    """Map a latent standard-normal draw into a rating state."""

    for state, threshold in thresholds:
        if latent_value <= threshold:
            return state
    return fallback_state


def asset_correlation_for_exposure(exposure: Exposure, spec: LatentFactorSpec | None = None) -> float:
    """Resolve a simple asset correlation for an exposure."""

    factor_spec = spec or LatentFactorSpec()
    base = factor_spec.issuer_type_base_correlations.get(exposure.issuer_type, factor_spec.asset_correlation)
    rating_adjustment = factor_spec.rating_adjustments.get(exposure.rating, 0.0)
    correlation = base + rating_adjustment
    return float(np.clip(correlation, factor_spec.min_correlation, factor_spec.max_correlation))


def supported_sectors(spec: MultiFactorSpec | None = None) -> set[str]:
    """Return the supported sector-factor lookup set."""

    factor_spec = spec or MultiFactorSpec()
    return set(factor_spec.sector_base_loadings)


def multi_factor_loadings_for_exposure(
    exposure: Exposure,
    spec: MultiFactorSpec | None = None,
) -> tuple[float, float]:
    """Resolve bounded macro and sector loadings for an exposure."""

    factor_spec = spec or MultiFactorSpec()
    if exposure.sector not in factor_spec.sector_base_loadings:
        raise ValueError(f"Unsupported sector for multi-factor engine: {exposure.sector}")

    macro_loading = (
        factor_spec.issuer_type_macro_loadings[exposure.issuer_type]
        + factor_spec.rating_macro_adjustments.get(exposure.rating, 0.0)
    )
    sector_loading = (
        factor_spec.issuer_type_sector_loadings[exposure.issuer_type]
        + factor_spec.rating_sector_adjustments.get(exposure.rating, 0.0)
        + (factor_spec.sector_base_loadings[exposure.sector] - 0.08) * 0.25
    )
    macro_loading = float(np.clip(macro_loading, factor_spec.min_macro_loading, factor_spec.max_macro_loading))
    sector_loading = float(np.clip(sector_loading, factor_spec.min_sector_loading, factor_spec.max_sector_loading))

    total_loading = macro_loading + sector_loading
    if total_loading >= factor_spec.max_total_loading:
        scale = factor_spec.max_total_loading / total_loading
        macro_loading *= scale
        sector_loading *= scale

    return macro_loading, sector_loading


def sample_one_factor_latent_variables(
    asset_correlations: np.ndarray,
    num_scenarios: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """Sample latent variables for a one-factor migration model."""

    correlations = np.asarray(asset_correlations, dtype=float)
    rng = np.random.default_rng(random_seed)
    systematic_factor = rng.standard_normal((num_scenarios, 1))
    idiosyncratic_factors = rng.standard_normal((num_scenarios, len(correlations)))
    return np.sqrt(correlations)[None, :] * systematic_factor + np.sqrt(1.0 - correlations)[None, :] * idiosyncratic_factors


def sample_multi_factor_latent_variables(
    macro_loadings: np.ndarray,
    sector_loadings: np.ndarray,
    sector_labels: list[str],
    num_scenarios: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """Sample latent variables for a macro-plus-sector migration model."""

    macro = np.asarray(macro_loadings, dtype=float)
    sector = np.asarray(sector_loadings, dtype=float)
    if (macro < 0.0).any() or (sector < 0.0).any():
        raise ValueError("Macro and sector loadings must be non-negative.")
    if ((macro + sector) >= 1.0).any():
        raise ValueError("Macro and sector loadings must sum to less than one.")

    rng = np.random.default_rng(random_seed)
    systematic_factor = rng.standard_normal((num_scenarios, 1))
    unique_sectors = sorted(set(sector_labels))
    sector_factor_lookup = {sector_name: rng.standard_normal(num_scenarios) for sector_name in unique_sectors}
    idiosyncratic_factors = rng.standard_normal((num_scenarios, len(macro)))

    latent_variables = np.zeros((num_scenarios, len(macro)), dtype=float)
    for column, sector_name in enumerate(sector_labels):
        latent_variables[:, column] = (
            np.sqrt(macro[column]) * systematic_factor[:, 0]
            + np.sqrt(sector[column]) * sector_factor_lookup[sector_name]
            + np.sqrt(1.0 - macro[column] - sector[column]) * idiosyncratic_factors[:, column]
        )
    return latent_variables


def simulate_one_factor_transitions(
    portfolio: list[Exposure],
    transition_matrices: dict[str, TransitionMatrix],
    n_scenarios: int,
    seed: int | None = None,
    factor_spec: LatentFactorSpec | None = None,
) -> np.ndarray:
    """Simulate migrated rating states using a one-factor latent-variable model."""

    spec = factor_spec or LatentFactorSpec()
    threshold_lookup = build_threshold_lookup(transition_matrices)
    asset_correlations = np.array([asset_correlation_for_exposure(exposure, spec) for exposure in portfolio], dtype=float)
    latent_variables = sample_one_factor_latent_variables(
        asset_correlations,
        n_scenarios,
        random_seed=seed if seed is not None else spec.random_seed,
    )

    migrated_ratings = np.empty((n_scenarios, len(portfolio)), dtype=object)
    for column, exposure in enumerate(portfolio):
        thresholds, fallback_state = threshold_lookup[exposure.exposure_class][exposure.rating]
        migrated_ratings[:, column] = [
            map_latent_to_rating_state(float(latent_value), thresholds, fallback_state)
            for latent_value in latent_variables[:, column]
        ]
    return migrated_ratings


def simulate_multi_factor_transitions(
    portfolio: list[Exposure],
    transition_matrices: dict[str, TransitionMatrix],
    n_scenarios: int,
    seed: int | None = None,
    factor_spec: MultiFactorSpec | None = None,
) -> np.ndarray:
    """Simulate migrated rating states using a macro-plus-sector latent model."""

    spec = factor_spec or MultiFactorSpec()
    threshold_lookup = build_threshold_lookup(transition_matrices)
    loading_pairs = [multi_factor_loadings_for_exposure(exposure, spec) for exposure in portfolio]
    macro_loadings = np.array([pair[0] for pair in loading_pairs], dtype=float)
    sector_loadings = np.array([pair[1] for pair in loading_pairs], dtype=float)
    sector_labels = [exposure.sector for exposure in portfolio]
    latent_variables = sample_multi_factor_latent_variables(
        macro_loadings,
        sector_loadings,
        sector_labels,
        n_scenarios,
        random_seed=seed if seed is not None else spec.random_seed,
    )

    migrated_ratings = np.empty((n_scenarios, len(portfolio)), dtype=object)
    for column, exposure in enumerate(portfolio):
        thresholds, fallback_state = threshold_lookup[exposure.exposure_class][exposure.rating]
        migrated_ratings[:, column] = [
            map_latent_to_rating_state(float(latent_value), thresholds, fallback_state)
            for latent_value in latent_variables[:, column]
        ]
    return migrated_ratings


def sample_gaussian_copula_latent_variables(
    correlation_matrix: np.ndarray,
    num_scenarios: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """Placeholder for a Gaussian copula latent sampler."""

    raise NotImplementedError("Future phase: implement Gaussian copula latent migration.")


def sample_sector_latent_variables(
    sectors: list[str],
    num_scenarios: int,
    spec: LatentFactorSpec,
) -> pd.DataFrame:
    """Placeholder for a sector-factor extension."""

    raise NotImplementedError("Future phase: implement sector factor latent migration.")
