"""Reporting and visualization utilities."""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".mpl-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import breakdown_by_instrument_type, breakdown_by_rating_bucket, summarize_distribution
from src.config import RATING_BUCKET_MAP


def build_summary_tables(scenario_results: pd.DataFrame, exposure_results: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, float]]:
    """Build high-level summary tables for reporting."""

    return {
        "metrics": summarize_distribution(scenario_results),
        "portfolio_summary": build_portfolio_summary_table(exposure_results),
        "top_obligors": top_obligor_concentration_table(exposure_results),
        "exposure_by_instrument": exposure_breakdown_table(exposure_results, "instrument_type"),
        "exposure_by_rating_bucket": exposure_by_rating_bucket_table(exposure_results),
        "exposure_by_sector": exposure_breakdown_table(exposure_results, "sector"),
        "exposure_by_currency": exposure_breakdown_table(exposure_results, "currency"),
        "exposure_by_issuer_type": exposure_breakdown_table(exposure_results, "issuer_type"),
        "exposure_by_instrument_subtype": exposure_breakdown_table(exposure_results, "instrument_subtype"),
        "exposure_by_seniority": exposure_breakdown_table(exposure_results, "seniority"),
        "by_instrument": breakdown_by_instrument_type(exposure_results),
        "by_rating_bucket": breakdown_by_rating_bucket(exposure_results),
    }


def build_mode_comparison_table(results_by_mode: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize comparable scenario outputs across simulation modes."""

    rows: list[dict[str, float | str]] = []
    for mode_name, scenario_results in results_by_mode.items():
        summary = summarize_distribution(scenario_results)
        rows.append(
            {
                "mode": mode_name,
                "mean_loss": summary["mean_loss"],
                "loss_std": summary["loss_std"],
                "var_95": summary["var_95"],
                "var_99": summary["var_99"],
                "es_99": summary["es_99"],
                "positive_pnl_pct": summary["positive_pnl_pct"],
                "mean_default_count": summary["mean_default_count"],
                "p95_default_count": summary["p95_default_count"],
                "p99_default_count": summary["p99_default_count"],
            }
        )
    return pd.DataFrame(rows)


def build_portfolio_summary_table(exposure_results: pd.DataFrame) -> pd.DataFrame:
    """Build a single-row portfolio summary table using current values."""

    total_exposure = float(exposure_results["current_value"].sum())
    obligor_totals = exposure_results.groupby("obligor_id", as_index=False)["current_value"].sum()
    number_of_obligors = int(obligor_totals["obligor_id"].nunique())
    number_of_facilities = int(len(exposure_results))
    top_10_concentration_pct = float(obligor_totals["current_value"].nlargest(10).sum() / total_exposure * 100.0)

    return pd.DataFrame(
        [
            {
                "total_exposure": total_exposure,
                "number_of_obligors": number_of_obligors,
                "number_of_facilities": number_of_facilities,
                "avg_exposure_per_obligor": total_exposure / number_of_obligors,
                "top_10_obligors_concentration_pct": top_10_concentration_pct,
            }
        ]
    )


def top_obligor_concentration_table(exposure_results: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Build a top-obligor concentration table."""

    total_exposure = float(exposure_results["current_value"].sum())
    table = (
        exposure_results.groupby("obligor_id", as_index=False)["current_value"]
        .sum()
        .rename(columns={"current_value": "total_exposure"})
        .sort_values("total_exposure", ascending=False)
        .head(top_n)
    )
    table["pct_total"] = table["total_exposure"] / total_exposure * 100.0
    return table


def exposure_breakdown_table(exposure_results: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate exposure by a selected portfolio dimension."""

    total_exposure = float(exposure_results["current_value"].sum())
    table = (
        exposure_results.groupby(dimension, as_index=False)["current_value"]
        .sum()
        .rename(columns={dimension: "bucket", "current_value": "total_exposure"})
    )
    table["pct_total"] = table["total_exposure"] / total_exposure * 100.0
    return table.sort_values("total_exposure", ascending=False)


def exposure_by_rating_bucket_table(exposure_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exposure by broad rating bucket in presentation order."""

    table = exposure_results.copy()
    table["rating_bucket"] = table["rating"].map(RATING_BUCKET_MAP).fillna("Other")
    grouped = (
        table.groupby("rating_bucket", as_index=False)["current_value"]
        .sum()
        .rename(columns={"current_value": "total_exposure"})
    )
    total_exposure = float(grouped["total_exposure"].sum())
    grouped["pct_total"] = grouped["total_exposure"] / total_exposure * 100.0
    order = pd.Categorical(grouped["rating_bucket"], categories=["AAA-A", "BBB", "BB-B", "CCC/D", "Other"], ordered=True)
    return grouped.assign(_order=order).sort_values("_order").drop(columns="_order")


def tail_loss_attribution_table(diagnostics: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    """Return a tail attribution table by key if present."""

    return diagnostics.get(key, pd.DataFrame())


def plot_loss_distribution(
    scenario_results: pd.DataFrame,
    output_path: Path | str,
    title: str = "Simulated Portfolio Loss Distribution",
    subtitle: str | None = None,
) -> Path:
    """Save a simple histogram of simulated portfolio losses."""

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_distribution(scenario_results)
    losses = scenario_results["total_loss"]

    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=40, color="#245c7c", edgecolor="white", alpha=0.85)
    if float(losses.min()) <= 0.0 <= float(losses.max()):
        plt.axvline(0.0, color="#3d405b", linestyle=":", linewidth=1.2, label="Break-even")
    plt.axvline(summary["var_95"], color="#e07a24", linestyle="--", linewidth=1.5, label="VaR 95%")
    plt.axvline(summary["var_99"], color="#9b2226", linestyle="--", linewidth=1.5, label="VaR 99%")
    plt.axvline(summary["es_99"], color="#6c757d", linestyle="-.", linewidth=1.3, label="ES 99%")
    plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, fontsize=10, y=0.96)
    plt.xlabel("Portfolio loss")
    plt.ylabel("Scenario count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_path, dpi=180)
    plt.close()
    return target_path


def export_scenario_results(scenario_results: pd.DataFrame, output_path: Path | str) -> Path:
    """Export scenario-level simulation results to CSV."""

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_results.to_csv(target_path, index=False)
    return target_path
