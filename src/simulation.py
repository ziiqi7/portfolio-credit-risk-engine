"""Monte Carlo portfolio migration simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import DEFAULT_CCF, DEFAULT_HORIZON_YEARS, DEFAULT_LGD, RATINGS
from src.schema import Exposure
from src.transitions import TransitionMatrix, to_cumulative_probabilities
from src.valuation import value_exposure

TransitionSampler = Callable[[list[Exposure], dict[str, TransitionMatrix], int, int | None], np.ndarray]


@dataclass(slots=True)
class SimulationResult:
    """Container for simulation outputs."""

    scenario_results: pd.DataFrame
    exposure_results: pd.DataFrame
    transition_summary: pd.DataFrame


def simulate_independent_transitions(
    portfolio: list[Exposure],
    transition_matrices: dict[str, TransitionMatrix],
    n_scenarios: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample migrated states independently for each exposure."""

    rng = np.random.default_rng(seed)
    states = np.array(RATINGS, dtype=object)
    cumulative_matrices = {name: matrix.cumulative() for name, matrix in transition_matrices.items()}

    migrated_ratings = np.empty((n_scenarios, len(portfolio)), dtype=object)
    for column, exposure in enumerate(portfolio):
        cumulative = cumulative_matrices[exposure.exposure_class].loc[exposure.rating].to_numpy(dtype=float)
        draws = rng.random(n_scenarios)
        state_indices = np.searchsorted(cumulative, draws, side="right")
        migrated_ratings[:, column] = states[state_indices]
    return migrated_ratings


def _build_transition_summary(portfolio: list[Exposure], migrated_ratings: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for idx, exposure in enumerate(portfolio):
        path_series = pd.Series(migrated_ratings[:, idx])
        counts = path_series.value_counts().reindex(RATINGS, fill_value=0)
        for end_rating, count in counts.items():
            records.append(
                {
                    "instrument_type": exposure.instrument_type,
                    "exposure_class": exposure.exposure_class,
                    "start_rating": exposure.rating,
                    "end_rating": end_rating,
                    "count": int(count),
                    "probability": float(count / len(path_series)),
                }
            )
    summary = pd.DataFrame(records)
    group_columns = ["instrument_type", "exposure_class", "start_rating", "end_rating"]
    return summary.groupby(group_columns, as_index=False)[["count", "probability"]].sum()


def simulate_portfolio(
    portfolio: list[Exposure],
    transition_matrices: dict[str, TransitionMatrix],
    n_scenarios: int = 5_000,
    seed: int | None = 42,
    base_lgd: float = DEFAULT_LGD,
    ccf: float = DEFAULT_CCF,
    horizon_years: float = DEFAULT_HORIZON_YEARS,
    allow_positive_pnl: bool = True,
    transition_sampler: TransitionSampler | None = None,
) -> SimulationResult:
    """Simulate portfolio loss and PnL distribution under rating migration."""

    sampler = transition_sampler or simulate_independent_transitions
    migrated_ratings = sampler(portfolio, transition_matrices, n_scenarios, seed)

    current_values = np.array(
        [value_exposure(exposure, exposure.rating, base_lgd=base_lgd, ccf=ccf, horizon_years=0.0) for exposure in portfolio]
    )
    scenario_values = np.zeros((n_scenarios, len(portfolio)), dtype=float)

    for column, exposure in enumerate(portfolio):
        states = migrated_ratings[:, column]
        for state in pd.unique(states):
            mask = states == state
            scenario_values[mask, column] = value_exposure(
                exposure,
                str(state),
                base_lgd=base_lgd,
                ccf=ccf,
                horizon_years=horizon_years,
            )

    pnl_matrix = scenario_values - current_values
    loss_matrix = -pnl_matrix if allow_positive_pnl else np.maximum(-pnl_matrix, 0.0)

    scenario_results = pd.DataFrame(
        {
            "scenario_id": np.arange(1, n_scenarios + 1),
            "total_pnl": pnl_matrix.sum(axis=1),
            "total_loss": loss_matrix.sum(axis=1),
        }
    )

    default_rates = (migrated_ratings == "D").mean(axis=0)
    exposure_results = pd.DataFrame(
        {
            "exposure_id": [exposure.exposure_id for exposure in portfolio],
            "obligor_id": [exposure.obligor_id for exposure in portfolio],
            "instrument_type": [exposure.instrument_type for exposure in portfolio],
            "exposure_class": [exposure.exposure_class for exposure in portfolio],
            "sector": [exposure.sector for exposure in portfolio],
            "rating": [exposure.rating for exposure in portfolio],
            "currency": [exposure.currency for exposure in portfolio],
            "balance": [exposure.balance for exposure in portfolio],
            "current_value": current_values,
            "expected_value": scenario_values.mean(axis=0),
            "expected_pnl": pnl_matrix.mean(axis=0),
            "expected_loss": loss_matrix.mean(axis=0),
            "pnl_std": pnl_matrix.std(axis=0),
            "default_rate": default_rates,
        }
    )

    transition_summary = _build_transition_summary(portfolio, migrated_ratings)
    return SimulationResult(
        scenario_results=scenario_results,
        exposure_results=exposure_results,
        transition_summary=transition_summary,
    )


def build_transition_threshold_template(transition_matrices: dict[str, TransitionMatrix]) -> dict[str, pd.DataFrame]:
    """Expose matrix cumulative probabilities for future correlated engines."""

    return {name: to_cumulative_probabilities(matrix.matrix) for name, matrix in transition_matrices.items()}
