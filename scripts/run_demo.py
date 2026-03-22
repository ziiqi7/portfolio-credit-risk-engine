"""Run the baseline portfolio credit risk demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR
from src.metrics import breakdown_by_instrument_type, breakdown_by_rating_bucket, summarize_distribution
from src.reporting import (
    build_portfolio_summary_table,
    exposure_breakdown_table,
    exposure_by_rating_bucket_table,
    export_scenario_results,
    plot_loss_distribution,
    top_obligor_concentration_table,
)
from src.simulation import simulate_portfolio
from src.synthetic_data import load_synthetic_portfolio, save_synthetic_portfolio
from src.transitions import load_demo_transition_matrices


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline portfolio credit risk demo.")
    parser.add_argument("--portfolio-path", type=Path, default=Path("data/synthetic/sample_portfolio.csv"))
    parser.add_argument("--scenarios", type=int, default=5_000, help="Number of Monte Carlo scenarios.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--disallow-positive-pnl",
        action="store_true",
        help="Clamp positive exposure PnL when constructing the loss distribution.",
    )
    parser.add_argument("--export-scenarios", action="store_true", help="Export scenario-level CSV results.")
    return parser.parse_args()


def _format_money(value: float) -> str:
    return f"{value:,.2f}"


def main() -> None:
    args = parse_args()
    portfolio_path = _resolve_repo_path(args.portfolio_path)
    if not portfolio_path.exists():
        save_synthetic_portfolio(output_path=portfolio_path, num_exposures=150, seed=args.seed)

    portfolio = load_synthetic_portfolio(portfolio_path)
    transition_matrices = load_demo_transition_matrices()
    result = simulate_portfolio(
        portfolio=portfolio,
        transition_matrices=transition_matrices,
        n_scenarios=args.scenarios,
        seed=args.seed,
        allow_positive_pnl=not args.disallow_positive_pnl,
    )

    metric_summary = summarize_distribution(result.scenario_results)
    plot_path = plot_loss_distribution(result.scenario_results, OUTPUTS_DIR / "demo_loss_distribution.png")
    portfolio_summary = build_portfolio_summary_table(result.exposure_results)
    exposure_by_instrument = exposure_breakdown_table(result.exposure_results, "instrument_type")
    exposure_by_rating_bucket = exposure_by_rating_bucket_table(result.exposure_results)
    exposure_by_sector = exposure_breakdown_table(result.exposure_results, "sector")
    exposure_by_currency = exposure_breakdown_table(result.exposure_results, "currency")
    top_obligors = top_obligor_concentration_table(result.exposure_results)

    print("Portfolio Credit Risk Demo")
    print("-" * 26)
    print(f"Portfolio file: {portfolio_path}")
    print(f"Exposures: {len(portfolio)}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Allow positive PnL offsets in loss distribution: {not args.disallow_positive_pnl}")
    print("\nPortfolio summary")
    print(portfolio_summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nTop 10 obligor concentration")
    print(top_obligors.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nExposure by instrument type")
    print(exposure_by_instrument.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nExposure by rating bucket")
    print(exposure_by_rating_bucket.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nExposure by sector")
    print(exposure_by_sector.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nExposure by currency")
    print(exposure_by_currency.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nDistribution diagnostics")
    print(f"Mean PnL: {_format_money(metric_summary['mean_pnl'])}")
    print(f"Mean Loss: {_format_money(metric_summary['mean_loss'])}")
    print(f"Positive PnL scenarios: {metric_summary['positive_pnl_pct']:.2f}%")
    print(f"Loss std: {_format_money(metric_summary['loss_std'])}")
    print(f"Loss min: {_format_money(metric_summary['loss_min'])}")
    print(f"Loss max: {_format_money(metric_summary['loss_max'])}")
    print(f"Loss skewness: {metric_summary['loss_skewness']:.4f}")
    print(f"VaR 95%: {_format_money(metric_summary['var_95'])}")
    print(f"VaR 99%: {_format_money(metric_summary['var_99'])}")
    print(f"VaR 99.9%: {_format_money(metric_summary['var_999'])}")
    print(f"ES 99%: {_format_money(metric_summary['es_99'])}")
    print(f"Loss distribution plot: {plot_path}")

    by_instrument = breakdown_by_instrument_type(result.exposure_results)
    by_rating_bucket = breakdown_by_rating_bucket(result.exposure_results)

    print("\nExpected loss by instrument type")
    print(by_instrument.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nExpected loss by starting rating bucket")
    print(by_rating_bucket.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    if args.export_scenarios:
        export_path = export_scenario_results(result.scenario_results, OUTPUTS_DIR / "scenario_results.csv")
        print(f"\nScenario results exported to: {export_path}")


if __name__ == "__main__":
    main()
