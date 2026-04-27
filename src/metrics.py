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


def calculate_skewness(values: pd.Series | np.ndarray) -> float:
    """Calculate the sample skewness using the third standardized moment."""

    array = np.asarray(values, dtype=float)
    std_dev = float(array.std())
    if std_dev == 0.0:
        return 0.0
    centered = array - array.mean()
    return float(np.mean((centered / std_dev) ** 3))


def distribution_quantiles(values: pd.Series | np.ndarray, quantiles: list[float]) -> dict[str, float]:
    """Return named quantiles for a numeric distribution."""

    array = np.asarray(values, dtype=float)
    summary: dict[str, float] = {}
    for quantile in quantiles:
        key = f"p{int(round(quantile * 100)):02d}"
        summary[key] = float(np.quantile(array, quantile))
    return summary


def summarize_distribution(scenario_results: pd.DataFrame) -> dict[str, float]:
    """Compute baseline portfolio distribution metrics."""

    losses = scenario_results["total_loss"].to_numpy(dtype=float)
    pnls = scenario_results["total_pnl"].to_numpy(dtype=float)
    default_counts = scenario_results.get("default_count", pd.Series(np.zeros(len(scenario_results), dtype=float))).to_numpy(dtype=float)
    downgrade_counts = scenario_results.get(
        "downgrade_count",
        pd.Series(np.zeros(len(scenario_results), dtype=float)),
    ).to_numpy(dtype=float)
    tail_cutoff = calculate_var(losses, 0.99)
    tail_mask = losses >= tail_cutoff
    quantile_summary = distribution_quantiles(losses, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "mean_pnl": float(pnls.mean()),
        "mean_loss": float(losses.mean()),
        "loss_std": float(losses.std()),
        "loss_min": float(losses.min()),
        "loss_median": float(np.median(losses)),
        "loss_max": float(losses.max()),
        "loss_skewness": calculate_skewness(losses),
        "positive_pnl_pct": float(np.mean(pnls > 0.0) * 100.0),
        "mean_default_count": float(default_counts.mean()),
        "p95_default_count": float(np.quantile(default_counts, 0.95)),
        "p99_default_count": float(np.quantile(default_counts, 0.99)),
        "prob_5plus_defaults": float(np.mean(default_counts >= 5.0) * 100.0),
        "prob_10plus_defaults": float(np.mean(default_counts >= 10.0) * 100.0),
        "prob_20plus_downgrades": float(np.mean(downgrade_counts >= 20.0) * 100.0),
        "tail_avg_default_count": float(default_counts[tail_mask].mean()) if tail_mask.any() else 0.0,
        "tail_avg_downgrade_count": float(downgrade_counts[tail_mask].mean()) if tail_mask.any() else 0.0,
        "var_95": calculate_var(losses, 0.95),
        "var_99": calculate_var(losses, 0.99),
        "var_999": calculate_var(losses, 0.999),
        "es_99": calculate_expected_shortfall(losses, 0.99),
        **quantile_summary,
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
