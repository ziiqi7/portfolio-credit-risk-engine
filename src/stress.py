"""Basic stress overlay utilities."""

from __future__ import annotations

import pandas as pd

from src.transitions import normalize_transition_matrix


def stress_transition_matrix(
    matrix: pd.DataFrame,
    downgrade_bias: float = 0.15,
    default_bias: float = 0.20,
) -> pd.DataFrame:
    """Tilt a transition matrix toward weaker migration outcomes."""

    stressed = matrix.copy().astype(float)
    states = list(stressed.columns)
    default_index = len(states) - 1

    for row_position, row_name in enumerate(stressed.index):
        if row_name == "D":
            continue

        row = stressed.loc[row_name].to_numpy(dtype=float)
        diagonal_shift = min(row[row_position] * downgrade_bias, row[row_position])
        row[row_position] -= diagonal_shift

        downgrade_target = min(row_position + 1, default_index - 1)
        row[downgrade_target] += diagonal_shift * (1.0 - default_bias)
        row[default_index] += diagonal_shift * default_bias
        stressed.loc[row_name] = row

    return normalize_transition_matrix(stressed)


def stress_lgd(base_lgd: float, multiplier: float = 1.20, cap: float = 0.95) -> float:
    """Apply a multiplicative LGD stress."""

    return min(base_lgd * multiplier, cap)


def stress_ccf(base_ccf: float, multiplier: float = 1.15, cap: float = 1.0) -> float:
    """Apply a multiplicative CCF stress."""

    return min(base_ccf * multiplier, cap)
