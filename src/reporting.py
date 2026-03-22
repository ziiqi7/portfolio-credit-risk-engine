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


def build_summary_tables(scenario_results: pd.DataFrame, exposure_results: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, float]]:
    """Build high-level summary tables for reporting."""

    return {
        "metrics": summarize_distribution(scenario_results),
        "by_instrument": breakdown_by_instrument_type(exposure_results),
        "by_rating_bucket": breakdown_by_rating_bucket(exposure_results),
    }


def plot_loss_distribution(
    scenario_results: pd.DataFrame,
    output_path: Path | str,
    title: str = "Simulated Portfolio Loss Distribution",
) -> Path:
    """Save a simple histogram of simulated portfolio losses."""

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_distribution(scenario_results)
    losses = scenario_results["total_loss"]

    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=40, color="#245c7c", edgecolor="white", alpha=0.85)
    plt.axvline(summary["var_95"], color="#e07a24", linestyle="--", linewidth=1.5, label="VaR 95%")
    plt.axvline(summary["var_99"], color="#9b2226", linestyle="--", linewidth=1.5, label="VaR 99%")
    plt.title(title)
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
