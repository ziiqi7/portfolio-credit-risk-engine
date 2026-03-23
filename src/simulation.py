"""Monte Carlo portfolio migration simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.correlation import (
    LatentFactorSpec,
    MultiFactorSpec,
    simulate_multi_factor_transitions as correlated_multi_factor_sampler,
    simulate_one_factor_transitions as correlated_one_factor_sampler,
)
from src.config import (
    DEFAULT_CCF,
    DEFAULT_HORIZON_YEARS,
    DEFAULT_LGD,
    INSTRUMENT_SUBTYPE_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS,
    ISSUER_SPREAD_SHOCK_MACRO_BETA_BPS,
    RATING_BUCKET_MAP,
    RATING_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS,
    RATING_SPREAD_SHOCK_SECTOR_ADJUSTMENT_BPS,
    RATINGS,
    SECTOR_SPREAD_SHOCK_BETA_BPS,
    SENIORITY_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS,
)
from src.schema import Exposure
from src.stress import RegimeStressConfig, build_regime_stress_overlays, get_regime_stress_config, sample_regime_labels
from src.transitions import TransitionMatrix, to_cumulative_probabilities
from src.valuation import value_exposure

TransitionSampler = Callable[[list[Exposure], dict[str, TransitionMatrix], int, int | None], np.ndarray]


@dataclass(slots=True)
class SimulationResult:
    """Container for simulation outputs."""

    scenario_results: pd.DataFrame
    exposure_results: pd.DataFrame
    transition_summary: pd.DataFrame
    diagnostics: dict[str, pd.DataFrame] = field(default_factory=dict)


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


def _spread_shock_betas_for_exposure(exposure: Exposure) -> tuple[float, float]:
    """Resolve simple macro and sector spread-shock betas in basis points."""

    macro_beta_bps = (
        ISSUER_SPREAD_SHOCK_MACRO_BETA_BPS[exposure.issuer_type]
        + INSTRUMENT_SUBTYPE_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS[exposure.instrument_subtype]
        + SENIORITY_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS[exposure.seniority]
        + RATING_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS.get(exposure.rating, 0.0)
    )
    sector_beta_bps = (
        SECTOR_SPREAD_SHOCK_BETA_BPS[exposure.sector]
        + RATING_SPREAD_SHOCK_SECTOR_ADJUSTMENT_BPS.get(exposure.rating, 0.0)
    )
    return float(max(macro_beta_bps, 4.0)), float(max(sector_beta_bps, 3.0))


def _sample_spread_shock_matrix(
    portfolio: list[Exposure],
    n_scenarios: int,
    seed: int | None,
    regime_shift_bps: np.ndarray,
) -> np.ndarray:
    """Sample scenario-level spread shocks by exposure."""

    rng = np.random.default_rng(None if seed is None else seed + 20_000)
    macro_factor = rng.standard_normal(n_scenarios)
    sector_labels = [exposure.sector for exposure in portfolio]
    unique_sectors = sorted(set(sector_labels))
    sector_factors = {sector: rng.standard_normal(n_scenarios) for sector in unique_sectors}

    shock_matrix = np.zeros((n_scenarios, len(portfolio)), dtype=float)
    macro_betas = np.array([_spread_shock_betas_for_exposure(exposure)[0] for exposure in portfolio], dtype=float)
    shock_matrix += np.outer(macro_factor, macro_betas)
    for column, exposure in enumerate(portfolio):
        _, sector_beta_bps = _spread_shock_betas_for_exposure(exposure)
        shock_matrix[:, column] += sector_beta_bps * sector_factors[exposure.sector]
    shock_matrix += regime_shift_bps[:, None]
    return np.clip(shock_matrix, -125.0, 550.0)


def _simulate_regime_migrations(
    portfolio: list[Exposure],
    transition_matrices: dict[str, TransitionMatrix],
    n_scenarios: int,
    seed: int | None,
    sampler: TransitionSampler,
    base_lgd: float,
    ccf: float,
    regime_config: RegimeStressConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate regime-aware migrations and scenario stress inputs."""

    resolved_regime_config = regime_config or get_regime_stress_config()
    regime_labels = sample_regime_labels(n_scenarios, seed=seed, config=resolved_regime_config)
    regime_overlays = build_regime_stress_overlays(
        transition_matrices=transition_matrices,
        base_lgd=base_lgd,
        base_ccf=ccf,
        config=resolved_regime_config,
    )

    migrated_ratings = np.empty((n_scenarios, len(portfolio)), dtype=object)
    scenario_lgd = np.empty(n_scenarios, dtype=float)
    scenario_ccf = np.empty(n_scenarios, dtype=float)
    regime_shift_bps = np.empty(n_scenarios, dtype=float)

    for offset, regime_label in enumerate(pd.unique(regime_labels)):
        mask = regime_labels == regime_label
        stressed_matrices, stressed_lgd, stressed_ccf, regime_definition = regime_overlays[str(regime_label)]
        regime_seed = None if seed is None else seed + (offset + 1) * 1_000
        migrated_ratings[mask] = sampler(portfolio, stressed_matrices, int(mask.sum()), regime_seed)
        scenario_lgd[mask] = stressed_lgd
        scenario_ccf[mask] = stressed_ccf
        regime_shift_bps[mask] = regime_definition.spread_shift_bps

    return migrated_ratings, regime_labels, scenario_lgd, scenario_ccf, regime_shift_bps


def _build_simulation_diagnostics(
    portfolio: list[Exposure],
    migrated_ratings: np.ndarray,
    regime_labels: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rating_to_index = {rating: index for index, rating in enumerate(RATINGS)}
    start_indices = np.array([rating_to_index[exposure.rating] for exposure in portfolio], dtype=int)
    end_indices = np.vectorize(rating_to_index.get, otypes=[int])(migrated_ratings)

    downgrade_mask = end_indices > start_indices[None, :]
    default_mask = migrated_ratings == "D"

    default_count = default_mask.sum(axis=1)
    downgrade_count = downgrade_mask.sum(axis=1)

    scenario_enrichment = pd.DataFrame(
        {
            "default_count": default_count,
            "default_rate": default_count / len(portfolio),
            "downgrade_count": downgrade_count,
            "downgrade_frequency": downgrade_count / len(portfolio),
        }
    )
    if regime_labels is not None:
        scenario_enrichment["regime_label"] = regime_labels

    records: list[dict[str, object]] = []
    for column, exposure in enumerate(portfolio):
        records.append(
            {
                "exposure_class": exposure.exposure_class,
                "issuer_type": exposure.issuer_type,
                "default_rate": float(default_mask[:, column].mean()),
                "downgrade_frequency": float(downgrade_mask[:, column].mean()),
            }
        )
    diagnostics_frame = pd.DataFrame(records)

    default_rate_by_exposure_class = (
        diagnostics_frame.groupby("exposure_class", as_index=False)[["default_rate", "downgrade_frequency"]].mean()
    )
    default_rate_by_issuer_type = (
        diagnostics_frame.groupby("issuer_type", as_index=False)[["default_rate", "downgrade_frequency"]].mean()
    )
    default_count_distribution = pd.DataFrame(
        [
            {
                "mean_default_count": float(default_count.mean()),
                "median_default_count": float(np.median(default_count)),
                "p95_default_count": float(np.quantile(default_count, 0.95)),
                "p99_default_count": float(np.quantile(default_count, 0.99)),
                "max_default_count": int(default_count.max()),
                "mean_downgrade_count": float(downgrade_count.mean()),
                "p95_downgrade_count": float(np.quantile(downgrade_count, 0.95)),
                "p99_downgrade_count": float(np.quantile(downgrade_count, 0.99)),
            }
        ]
    )
    diagnostics = {
        "default_rate_by_exposure_class": default_rate_by_exposure_class.sort_values("default_rate", ascending=False),
        "default_rate_by_issuer_type": default_rate_by_issuer_type.sort_values("default_rate", ascending=False),
        "default_count_distribution": default_count_distribution,
    }
    if regime_labels is not None:
        regime_distribution = (
            pd.Series(regime_labels, name="regime_label")
            .value_counts(normalize=False)
            .rename_axis("regime_label")
            .reset_index(name="scenario_count")
        )
        regime_distribution["pct_scenarios"] = regime_distribution["scenario_count"] / len(regime_labels) * 100.0
        diagnostics["regime_distribution"] = regime_distribution
    return scenario_enrichment, diagnostics


def _tail_loss_attribution_table(
    loss_matrix: np.ndarray,
    labels: list[str],
    tail_mask: np.ndarray,
) -> pd.DataFrame:
    """Aggregate average loss contributions in the worst loss tail."""

    if not tail_mask.any():
        return pd.DataFrame(columns=["bucket", "avg_tail_loss", "pct_tail_loss"])

    average_tail_loss = loss_matrix[tail_mask].mean(axis=0)
    table = pd.DataFrame({"bucket": labels, "avg_tail_loss": average_tail_loss})
    grouped = table.groupby("bucket", as_index=False)["avg_tail_loss"].sum()
    total_tail_loss = float(grouped["avg_tail_loss"].sum())
    grouped["pct_tail_loss"] = (
        grouped["avg_tail_loss"] / total_tail_loss * 100.0 if total_tail_loss > 0.0 else 0.0
    )
    return grouped.sort_values("avg_tail_loss", ascending=False)


def simulate_one_factor_transitions(
    portfolio: list[Exposure],
    transition_matrices: dict[str, TransitionMatrix],
    n_scenarios: int,
    seed: int | None = None,
    factor_spec: LatentFactorSpec | None = None,
) -> np.ndarray:
    """Public wrapper for the one-factor latent migration engine."""

    return correlated_one_factor_sampler(
        portfolio=portfolio,
        transition_matrices=transition_matrices,
        n_scenarios=n_scenarios,
        seed=seed,
        factor_spec=factor_spec,
    )


def simulate_multi_factor_transitions(
    portfolio: list[Exposure],
    transition_matrices: dict[str, TransitionMatrix],
    n_scenarios: int,
    seed: int | None = None,
    factor_spec: MultiFactorSpec | None = None,
) -> np.ndarray:
    """Public wrapper for the macro-plus-sector latent migration engine."""

    return correlated_multi_factor_sampler(
        portfolio=portfolio,
        transition_matrices=transition_matrices,
        n_scenarios=n_scenarios,
        seed=seed,
        factor_spec=factor_spec,
    )


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
    stress_mode: str = "none",
    spread_shift_bps: float = 0.0,
    regime_config: RegimeStressConfig | None = None,
) -> SimulationResult:
    """Simulate portfolio loss and PnL distribution under rating migration."""

    sampler = transition_sampler or simulate_independent_transitions
    regime_labels: np.ndarray | None = None
    if stress_mode == "regime":
        migrated_ratings, regime_labels, scenario_lgd, scenario_ccf, regime_shift_bps = _simulate_regime_migrations(
            portfolio=portfolio,
            transition_matrices=transition_matrices,
            n_scenarios=n_scenarios,
            seed=seed,
            sampler=sampler,
            base_lgd=base_lgd,
            ccf=ccf,
            regime_config=regime_config,
        )
    else:
        migrated_ratings = sampler(portfolio, transition_matrices, n_scenarios, seed)
        scenario_lgd = np.full(n_scenarios, base_lgd, dtype=float)
        scenario_ccf = np.full(n_scenarios, ccf, dtype=float)
        regime_shift_bps = np.full(n_scenarios, spread_shift_bps, dtype=float)

    spread_shock_matrix = _sample_spread_shock_matrix(
        portfolio=portfolio,
        n_scenarios=n_scenarios,
        seed=seed,
        regime_shift_bps=regime_shift_bps,
    )

    current_values = np.array(
        [
            value_exposure(exposure, exposure.rating, base_lgd=base_lgd, ccf=ccf, horizon_years=0.0, spread_shock_bps=0.0)
            for exposure in portfolio
        ]
    )
    reference_matrix = np.zeros((n_scenarios, len(portfolio)), dtype=float)
    scenario_values = np.zeros((n_scenarios, len(portfolio)), dtype=float)

    for column, exposure in enumerate(portfolio):
        reference_matrix[:, column] = np.array(
            [
                value_exposure(
                    exposure,
                    exposure.rating,
                    base_lgd=float(scenario_lgd[row]),
                    ccf=float(scenario_ccf[row]),
                    horizon_years=horizon_years,
                    spread_shock_bps=float(spread_shock_matrix[row, column]),
                )
                for row in range(n_scenarios)
            ],
            dtype=float,
        )
        scenario_values[:, column] = np.array(
            [
                value_exposure(
                    exposure,
                    str(migrated_ratings[row, column]),
                    base_lgd=float(scenario_lgd[row]),
                    ccf=float(scenario_ccf[row]),
                    horizon_years=horizon_years,
                    spread_shock_bps=float(spread_shock_matrix[row, column]),
                )
                for row in range(n_scenarios)
            ],
            dtype=float,
        )

    pnl_matrix = scenario_values - reference_matrix
    loss_matrix = -pnl_matrix if allow_positive_pnl else np.maximum(-pnl_matrix, 0.0)

    scenario_results = pd.DataFrame(
        {
            "scenario_id": np.arange(1, n_scenarios + 1),
            "total_pnl": pnl_matrix.sum(axis=1),
            "total_loss": loss_matrix.sum(axis=1),
        }
    )
    scenario_enrichment, diagnostics = _build_simulation_diagnostics(portfolio, migrated_ratings, regime_labels=regime_labels)
    scenario_results = pd.concat([scenario_results, scenario_enrichment], axis=1)

    tail_cutoff = float(np.quantile(scenario_results["total_loss"].to_numpy(dtype=float), 0.99))
    tail_mask = scenario_results["total_loss"].to_numpy(dtype=float) >= tail_cutoff
    diagnostics["tail_loss_by_issuer_type"] = _tail_loss_attribution_table(
        loss_matrix,
        [exposure.issuer_type for exposure in portfolio],
        tail_mask,
    )
    diagnostics["tail_loss_by_sector"] = _tail_loss_attribution_table(
        loss_matrix,
        [exposure.sector for exposure in portfolio],
        tail_mask,
    )
    diagnostics["tail_loss_by_instrument_subtype"] = _tail_loss_attribution_table(
        loss_matrix,
        [exposure.instrument_subtype for exposure in portfolio],
        tail_mask,
    )
    diagnostics["tail_loss_by_rating_bucket"] = _tail_loss_attribution_table(
        loss_matrix,
        [RATING_BUCKET_MAP.get(exposure.rating, "Other") for exposure in portfolio],
        tail_mask,
    )

    default_rates = (migrated_ratings == "D").mean(axis=0)
    exposure_results = pd.DataFrame(
        {
            "exposure_id": [exposure.exposure_id for exposure in portfolio],
            "obligor_id": [exposure.obligor_id for exposure in portfolio],
            "instrument_type": [exposure.instrument_type for exposure in portfolio],
            "issuer_type": [exposure.issuer_type for exposure in portfolio],
            "instrument_subtype": [exposure.instrument_subtype for exposure in portfolio],
            "seniority": [exposure.seniority for exposure in portfolio],
            "secured_flag": [exposure.secured_flag for exposure in portfolio],
            "exposure_class": [exposure.exposure_class for exposure in portfolio],
            "sector": [exposure.sector for exposure in portfolio],
            "rating": [exposure.rating for exposure in portfolio],
            "currency": [exposure.currency for exposure in portfolio],
            "balance": [exposure.balance for exposure in portfolio],
            "current_value": current_values,
            "reference_value": reference_matrix.mean(axis=0),
            "expected_value": scenario_values.mean(axis=0),
            "expected_pnl": pnl_matrix.mean(axis=0),
            "expected_loss": loss_matrix.mean(axis=0),
            "pnl_std": pnl_matrix.std(axis=0),
            "default_rate": default_rates,
            "mean_spread_shock_bps": spread_shock_matrix.mean(axis=0),
        }
    )

    transition_summary = _build_transition_summary(portfolio, migrated_ratings)
    return SimulationResult(
        scenario_results=scenario_results,
        exposure_results=exposure_results,
        transition_summary=transition_summary,
        diagnostics=diagnostics,
    )


def build_transition_threshold_template(transition_matrices: dict[str, TransitionMatrix]) -> dict[str, pd.DataFrame]:
    """Expose matrix cumulative probabilities for future correlated engines."""

    return {name: to_cumulative_probabilities(matrix.matrix) for name, matrix in transition_matrices.items()}
