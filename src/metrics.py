"""Risk metrics and attribution helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RATING_BUCKET_MAP


def _quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def calculate_var(losses: pd.Series | np.ndarray, confidence_level: float) -> float:
    """Calculate loss VaR at the requested confidence level."""

    values = np.asarray(losses, dtype=float)
    return _quantile(values, confidence_level)


def calculate_expected_shortfall(losses: pd.Series | np.ndarray, confidence_level: float) -> float:
    """Calculate Expected Shortfall over the VaR tail."""

    values = np.asarray(losses, dtype=float)
    var_value = calculate_var(values, confidence_level)
    tail = values[values >= var_value]
    if tail.size == 0:
        return var_value
    return float(tail.mean())


def summarize_distribution(scenario_results: pd.DataFrame) -> dict[str, float]:
    """Compute baseline portfolio distribution metrics."""

    losses = scenario_results["total_loss"].to_numpy(dtype=float)
    pnls = scenario_results["total_pnl"].to_numpy(dtype=float)
    return {
        "mean_pnl": float(pnls.mean()),
        "mean_loss": float(losses.mean()),
        "var_95": calculate_var(losses, 0.95),
        "var_99": calculate_var(losses, 0.99),
        "var_999": calculate_var(losses, 0.999),
        "es_99": calculate_expected_shortfall(losses, 0.99),
    }


def breakdown_by_instrument_type(exposure_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate expected loss and value by instrument type."""

    columns = ["current_value", "expected_value", "expected_pnl", "expected_loss"]
    summary = exposure_results.groupby("instrument_type", as_index=False)[columns].sum()
    counts = exposure_results.groupby("instrument_type").size().rename("count").reset_index()
    return counts.merge(summary, on="instrument_type", how="left").sort_values("expected_loss", ascending=False)


def breakdown_by_rating_bucket(exposure_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate expected loss and value by starting rating bucket."""

    enriched = exposure_results.copy()
    enriched["rating_bucket"] = enriched["rating"].map(RATING_BUCKET_MAP).fillna("Other")
    columns = ["current_value", "expected_value", "expected_pnl", "expected_loss"]
    summary = enriched.groupby("rating_bucket", as_index=False)[columns].sum()
    counts = enriched.groupby("rating_bucket").size().rename("count").reset_index()
    return counts.merge(summary, on="rating_bucket", how="left").sort_values("expected_loss", ascending=False)
