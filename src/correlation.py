"""Correlation roadmap and extension hooks for future phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.transitions import threshold_mapping_inputs


@dataclass(slots=True)
class LatentFactorSpec:
    """Configuration for future latent-factor migration engines."""

    asset_correlation: float = 0.20
    sector_correlations: dict[str, float] = field(default_factory=dict)
    random_seed: int | None = None


def build_rating_thresholds(transition_matrix: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Convert transition probabilities into standard-normal thresholds.

    Thresholds are ordered from worst state to best state so they can be used
    later with latent credit variables where lower draws correspond to weaker
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


def map_latent_to_rating_state(
    latent_value: float,
    thresholds: list[tuple[str, float]],
    fallback_state: str,
) -> str:
    """Map a latent standard-normal draw into a rating state.

    This helper is usable today for threshold experiments, even though the
    full correlated migration engine is intentionally deferred to a later phase.
    """

    for state, threshold in thresholds:
        if latent_value <= threshold:
            return state
    return fallback_state


def sample_one_factor_latent_variables(
    num_exposures: int,
    num_scenarios: int,
    spec: LatentFactorSpec,
) -> np.ndarray:
    """Placeholder for a one-factor latent-variable sampler."""

    raise NotImplementedError("Phase 2 extension: implement one-factor latent-variable migration.")


def sample_gaussian_copula_latent_variables(
    correlation_matrix: np.ndarray,
    num_scenarios: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """Placeholder for a Gaussian copula latent sampler."""

    raise NotImplementedError("Phase 2 extension: implement Gaussian copula latent migration.")


def sample_sector_latent_variables(
    sectors: list[str],
    num_scenarios: int,
    spec: LatentFactorSpec,
) -> pd.DataFrame:
    """Placeholder for a sector-factor extension."""

    raise NotImplementedError("Phase 3 extension: implement sector factor latent migration.")
