"""Transition matrix loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NON_DEFAULT_RATINGS, RATINGS, TRANSITION_FILES


@dataclass(slots=True)
class TransitionMatrix:
    """Structured transition matrix wrapper."""

    exposure_class: str
    matrix: pd.DataFrame

    @property
    def states(self) -> list[str]:
        return list(self.matrix.columns)

    def cumulative(self) -> pd.DataFrame:
        return to_cumulative_probabilities(self.matrix)

    def threshold_cumulative(self) -> pd.DataFrame:
        return to_threshold_cumulative_probabilities(self.matrix)


def normalize_transition_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Normalize transition matrix rows to unit sum."""

    normalized = matrix.copy().astype(float)
    if (normalized < 0.0).any().any():
        raise ValueError("Transition matrix probabilities must be non-negative.")
    row_sums = normalized.sum(axis=1)
    if (row_sums <= 0.0).any():
        raise ValueError("Transition matrix rows must have positive mass.")
    normalized = normalized.div(row_sums, axis=0)
    return normalized


def _validate_rating_axis(matrix: pd.DataFrame) -> None:
    if list(matrix.index) != RATINGS:
        raise ValueError("Transition matrix index must match the canonical rating scale.")
    if list(matrix.columns) != RATINGS:
        raise ValueError("Transition matrix columns must match the canonical rating scale.")


def _monotonic_non_decreasing(values: pd.Series | np.ndarray, tolerance: float = 1e-10) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(array) >= -tolerance))


def expected_rating_index(matrix: pd.DataFrame) -> pd.Series:
    """Return the expected migrated rating index for each starting rating."""

    normalized = normalize_transition_matrix(matrix).reindex(index=RATINGS, columns=RATINGS)
    state_index = np.arange(len(RATINGS), dtype=float)
    values = {
        rating: float(normalized.loc[rating].to_numpy(dtype=float) @ state_index)
        for rating in NON_DEFAULT_RATINGS
    }
    return pd.Series(values, name="expected_rating_index")


def validate_transition_matrix(matrix: pd.DataFrame, tolerance: float = 1e-10) -> pd.DataFrame:
    """Validate row sums, state ordering, and monotonic migration intuition."""

    normalized = normalize_transition_matrix(matrix).reindex(index=RATINGS, columns=RATINGS)
    _validate_rating_axis(normalized)

    row_sums = normalized.sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        raise ValueError("Transition matrix rows must sum to 1.")

    default_probabilities = normalized.loc[NON_DEFAULT_RATINGS, "D"]
    if not _monotonic_non_decreasing(default_probabilities, tolerance=tolerance):
        raise ValueError("Default probabilities must increase as starting ratings worsen.")

    expected_indices = expected_rating_index(normalized)
    if not _monotonic_non_decreasing(expected_indices, tolerance=tolerance):
        raise ValueError("Expected migrated rating quality must worsen as starting ratings worsen.")

    diagonal = pd.Series(np.diag(normalized.loc[RATINGS, RATINGS]), index=RATINGS)
    if not (diagonal.loc[NON_DEFAULT_RATINGS] > 0.5).all():
        raise ValueError("Non-default transition rows should retain strong diagonal mass.")

    return normalized


def load_transition_matrix(path: Path | str, exposure_class: str | None = None) -> TransitionMatrix:
    """Load a transition matrix CSV and normalize rows."""

    dataframe = pd.read_csv(path, index_col=0)
    dataframe.index = dataframe.index.astype(str).str.upper()
    dataframe.columns = [str(column).upper() for column in dataframe.columns]
    dataframe = dataframe.reindex(index=RATINGS, columns=RATINGS, fill_value=0.0)
    normalized = validate_transition_matrix(dataframe)
    matrix_name = exposure_class or Path(path).stem.replace("transition_matrix_", "")
    return TransitionMatrix(exposure_class=matrix_name, matrix=normalized)


def to_cumulative_probabilities(matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert transition probabilities into row-wise cumulative probabilities."""

    cumulative = validate_transition_matrix(matrix).cumsum(axis=1)
    cumulative.iloc[:, -1] = 1.0
    return cumulative


def to_threshold_cumulative_probabilities(matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert probabilities into worst-state cumulative tails for latent-threshold mapping."""

    normalized = validate_transition_matrix(matrix)
    reversed_columns = list(reversed(RATINGS))
    cumulative = normalized[reversed_columns].cumsum(axis=1)
    cumulative.iloc[:, -1] = 1.0
    return cumulative


def load_demo_transition_matrices(base_dir: Path | None = None) -> dict[str, TransitionMatrix]:
    """Load all bundled demo transition matrices."""

    files = TRANSITION_FILES if base_dir is None else {
        "corporate": base_dir / "transition_matrix_corporate.csv",
        "fi": base_dir / "transition_matrix_fi.csv",
        "sovereign": base_dir / "transition_matrix_sovereign.csv",
    }
    return {name: load_transition_matrix(path, exposure_class=name) for name, path in files.items()}


def threshold_mapping_inputs(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative transition probabilities for later latent-threshold mapping."""

    return to_threshold_cumulative_probabilities(matrix)
