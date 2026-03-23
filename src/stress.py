"""Basic stress overlay utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import pandas as pd

from src.transitions import TransitionMatrix, normalize_transition_matrix, validate_transition_matrix


@dataclass(frozen=True, slots=True)
class StressConfig:
    """Stress configuration applied to transitions and valuation inputs."""

    mode: str
    downgrade_bias: float
    default_bias: float
    lgd_multiplier: float
    ccf_multiplier: float
    spread_shift_bps: float = 0.0


@dataclass(frozen=True, slots=True)
class RegimeDefinition:
    """Scenario regime definition for regime-based stress mixtures."""

    label: str
    probability: float
    downgrade_bias: float
    default_bias: float
    lgd_multiplier: float
    ccf_multiplier: float
    spread_shift_bps: float


@dataclass(frozen=True, slots=True)
class RegimeStressConfig:
    """Mixture specification for scenario-level regime stress."""

    mode: str = "regime"
    regimes: tuple[RegimeDefinition, ...] = field(
        default_factory=lambda: (
            RegimeDefinition(
                label="normal",
                probability=0.72,
                downgrade_bias=0.00,
                default_bias=0.00,
                lgd_multiplier=1.00,
                ccf_multiplier=1.00,
                spread_shift_bps=0.0,
            ),
            RegimeDefinition(
                label="stress",
                probability=0.22,
                downgrade_bias=0.12,
                default_bias=0.22,
                lgd_multiplier=1.15,
                ccf_multiplier=1.10,
                spread_shift_bps=45.0,
            ),
            RegimeDefinition(
                label="crisis",
                probability=0.06,
                downgrade_bias=0.24,
                default_bias=0.38,
                lgd_multiplier=1.30,
                ccf_multiplier=1.20,
                spread_shift_bps=120.0,
            ),
        )
    )


STRESS_CONFIGS = {
    "none": StressConfig(
        "none",
        downgrade_bias=0.0,
        default_bias=0.0,
        lgd_multiplier=1.0,
        ccf_multiplier=1.0,
        spread_shift_bps=0.0,
    ),
    "mild": StressConfig(
        "mild",
        downgrade_bias=0.08,
        default_bias=0.18,
        lgd_multiplier=1.10,
        ccf_multiplier=1.08,
        spread_shift_bps=25.0,
    ),
    "severe": StressConfig(
        "severe",
        downgrade_bias=0.18,
        default_bias=0.30,
        lgd_multiplier=1.25,
        ccf_multiplier=1.18,
        spread_shift_bps=70.0,
    ),
    "regime": StressConfig(
        "regime",
        downgrade_bias=0.0,
        default_bias=0.0,
        lgd_multiplier=1.0,
        ccf_multiplier=1.0,
        spread_shift_bps=0.0,
    ),
}


def get_stress_config(mode: str) -> StressConfig:
    """Resolve a named stress mode."""

    normalized_mode = mode.lower()
    if normalized_mode not in STRESS_CONFIGS:
        raise ValueError(f"Unsupported stress mode: {mode}")
    return STRESS_CONFIGS[normalized_mode]


def get_regime_stress_config() -> RegimeStressConfig:
    """Return the default scenario-level regime mixture."""

    return RegimeStressConfig()


def sample_regime_labels(
    n_scenarios: int,
    seed: int | None = None,
    config: RegimeStressConfig | None = None,
) -> np.ndarray:
    """Sample scenario regime labels for regime-based stress."""

    regime_config = config or get_regime_stress_config()
    labels = [regime.label for regime in regime_config.regimes]
    probabilities = np.array([regime.probability for regime in regime_config.regimes], dtype=float)
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(labels, size=n_scenarios, p=probabilities)


def stress_transition_matrix(
    matrix: pd.DataFrame,
    downgrade_bias: float = 0.15,
    default_bias: float = 0.20,
) -> pd.DataFrame:
    """Tilt a transition matrix toward weaker migration outcomes."""

    stressed = matrix.copy().astype(float)
    states = list(stressed.columns)
    default_index = len(states) - 1
    non_default_count = default_index

    for row_position, row_name in enumerate(stressed.index):
        if row_name == "D":
            continue

        row = stressed.loc[row_name].to_numpy(dtype=float)
        diagonal_shift = min(row[row_position] * downgrade_bias, row[row_position])
        row[row_position] -= diagonal_shift

        downgrade_target = min(row_position + 1, default_index - 1)
        default_weight = 0.35 + 0.65 * (row_position / max(non_default_count - 1, 1))
        row[downgrade_target] += diagonal_shift * (1.0 - default_bias * default_weight)
        row[default_index] += diagonal_shift * default_bias * default_weight
        stressed.loc[row_name] = row

    epsilon = 1e-6
    non_default_rows = [row_name for row_name in stressed.index if row_name != "D"]
    previous_default = float(stressed.loc[non_default_rows[0], "D"])
    for row_name in non_default_rows[1:]:
        current_default = float(stressed.loc[row_name, "D"])
        if current_default < previous_default + epsilon:
            adjustment = previous_default + epsilon - current_default
            stressed.loc[row_name, "D"] += adjustment
            stressed.loc[row_name, row_name] = max(float(stressed.loc[row_name, row_name]) - adjustment, epsilon)
        previous_default = float(stressed.loc[row_name, "D"])

    return validate_transition_matrix(normalize_transition_matrix(stressed))


def stress_lgd(base_lgd: float, multiplier: float = 1.20, cap: float = 0.95) -> float:
    """Apply a multiplicative LGD stress."""

    return min(base_lgd * multiplier, cap)


def stress_ccf(base_ccf: float, multiplier: float = 1.15, cap: float = 1.0) -> float:
    """Apply a multiplicative CCF stress."""

    return min(base_ccf * multiplier, cap)


def build_regime_stress_overlays(
    transition_matrices: dict[str, TransitionMatrix],
    base_lgd: float,
    base_ccf: float,
    config: RegimeStressConfig | None = None,
) -> dict[str, tuple[dict[str, TransitionMatrix], float, float, RegimeDefinition]]:
    """Build stressed inputs for each regime in the mixture."""

    regime_config = config or get_regime_stress_config()
    overlays: dict[str, tuple[dict[str, TransitionMatrix], float, float, RegimeDefinition]] = {}
    for regime in regime_config.regimes:
        stressed_matrices = {
            name: TransitionMatrix(
                exposure_class=matrix.exposure_class,
                matrix=stress_transition_matrix(
                    matrix.matrix,
                    downgrade_bias=regime.downgrade_bias,
                    default_bias=regime.default_bias,
                ),
            )
            for name, matrix in transition_matrices.items()
        }
        overlays[regime.label] = (
            stressed_matrices,
            stress_lgd(base_lgd, multiplier=regime.lgd_multiplier),
            stress_ccf(base_ccf, multiplier=regime.ccf_multiplier),
            regime,
        )
    return overlays


def apply_stress_overlays(
    transition_matrices: dict[str, TransitionMatrix],
    base_lgd: float,
    base_ccf: float,
    mode: str,
) -> tuple[dict[str, TransitionMatrix], float, float, StressConfig]:
    """Apply a named stress regime to matrices, LGD, and CCF."""

    config = get_stress_config(mode)
    if config.mode in {"none", "regime"}:
        return transition_matrices, base_lgd, base_ccf, config

    stressed_matrices = {
        name: TransitionMatrix(
            exposure_class=matrix.exposure_class,
            matrix=stress_transition_matrix(
                matrix.matrix,
                downgrade_bias=config.downgrade_bias,
                default_bias=config.default_bias,
            ),
        )
        for name, matrix in transition_matrices.items()
    }
    stressed_lgd = stress_lgd(base_lgd, multiplier=config.lgd_multiplier)
    stressed_ccf = stress_ccf(base_ccf, multiplier=config.ccf_multiplier)
    return stressed_matrices, stressed_lgd, stressed_ccf, config
