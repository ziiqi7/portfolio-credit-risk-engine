"""Run the baseline portfolio credit risk demo."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CCF, DEFAULT_LGD, OUTPUTS_DIR
from src.metrics import breakdown_by_instrument_type, breakdown_by_rating_bucket, summarize_distribution
from src.reporting import (
    build_mode_comparison_table,
    build_portfolio_summary_table,
    exposure_breakdown_table,
    exposure_by_rating_bucket_table,
    export_scenario_results,
    plot_loss_distribution,
    tail_loss_attribution_table,
    top_obligor_concentration_table,
)
from src.simulation import simulate_multi_factor_transitions, simulate_one_factor_transitions, simulate_portfolio
from src.stress import apply_stress_overlays, get_regime_stress_config
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
        "--simulation-mode",
        choices=["independent", "one_factor", "multi_factor"],
        default="independent",
        help="Transition simulation engine to use.",
    )
    parser.add_argument(
        "--stress",
        choices=["none", "mild", "severe", "regime"],
        default="none",
        help="Stress overlay regime to apply to transitions, LGD, CCF, and spread shocks.",
    )
    parser.add_argument(
        "--clamp-positive-pnl",
        action="store_true",
        help="Clamp positive exposure PnL when constructing the loss distribution.",
    )
    parser.add_argument(
        "--compare-independent",
        action="store_true",
        help="When running a factor mode, also run simpler benchmark modes on the same inputs for comparison.",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Run and compare independent, one_factor, and multi_factor on the same portfolio, seed, and stress setup.",
    )
    parser.add_argument("--export-scenarios", action="store_true", help="Export scenario-level CSV results.")
    return parser.parse_args()


def _format_money(value: float) -> str:
    return f"{value:,.2f}"


def _format_percentage(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def main() -> None:
    args = parse_args()
    portfolio_path = _resolve_repo_path(args.portfolio_path)
    if not portfolio_path.exists():
        save_synthetic_portfolio(output_path=portfolio_path, num_exposures=150, seed=args.seed)

    portfolio = load_synthetic_portfolio(portfolio_path)
    transition_matrices = load_demo_transition_matrices()
    stressed_transition_matrices, stressed_lgd, stressed_ccf, stress_config = apply_stress_overlays(
        transition_matrices=transition_matrices,
        base_lgd=DEFAULT_LGD,
        base_ccf=DEFAULT_CCF,
        mode=args.stress,
    )

    transition_sampler = None
    if args.simulation_mode == "one_factor":
        transition_sampler = partial(simulate_one_factor_transitions, factor_spec=None)
    elif args.simulation_mode == "multi_factor":
        transition_sampler = partial(simulate_multi_factor_transitions, factor_spec=None)
    regime_config = get_regime_stress_config() if args.stress == "regime" else None

    result = simulate_portfolio(
        portfolio=portfolio,
        transition_matrices=stressed_transition_matrices,
        n_scenarios=args.scenarios,
        seed=args.seed,
        allow_positive_pnl=not args.clamp_positive_pnl,
        base_lgd=stressed_lgd,
        ccf=stressed_ccf,
        transition_sampler=transition_sampler,
        stress_mode=args.stress,
        spread_shift_bps=stress_config.spread_shift_bps,
        regime_config=regime_config,
    )

    metric_summary = summarize_distribution(result.scenario_results)
    subtitle = "Benchmark: migrated one-year PV vs same-rating one-year reference PV under scenario-consistent inputs"
    if args.stress == "regime":
        subtitle = f"{subtitle} | Regime stress: normal / stress / crisis mixture"
    plot_path = plot_loss_distribution(
        result.scenario_results,
        OUTPUTS_DIR / "demo_loss_distribution.png",
        title=f"Portfolio Loss Distribution ({args.simulation_mode}, stress={args.stress})",
        subtitle=subtitle,
    )
    portfolio_summary = build_portfolio_summary_table(result.exposure_results)
    exposure_by_instrument = exposure_breakdown_table(result.exposure_results, "instrument_type")
    exposure_by_rating_bucket = exposure_by_rating_bucket_table(result.exposure_results)
    exposure_by_sector = exposure_breakdown_table(result.exposure_results, "sector")
    exposure_by_currency = exposure_breakdown_table(result.exposure_results, "currency")
    exposure_by_issuer_type = exposure_breakdown_table(result.exposure_results, "issuer_type")
    exposure_by_instrument_subtype = exposure_breakdown_table(result.exposure_results, "instrument_subtype")
    exposure_by_seniority = exposure_breakdown_table(result.exposure_results, "seniority")
    top_obligors = top_obligor_concentration_table(result.exposure_results)
    default_rate_by_exposure_class = result.diagnostics["default_rate_by_exposure_class"]
    default_rate_by_issuer_type = result.diagnostics["default_rate_by_issuer_type"]
    default_count_distribution = result.diagnostics["default_count_distribution"]
    regime_distribution = result.diagnostics.get("regime_distribution")
    tail_loss_by_issuer_type = tail_loss_attribution_table(result.diagnostics, "tail_loss_by_issuer_type")
    tail_loss_by_sector = tail_loss_attribution_table(result.diagnostics, "tail_loss_by_sector")
    tail_loss_by_instrument_subtype = tail_loss_attribution_table(result.diagnostics, "tail_loss_by_instrument_subtype")
    tail_loss_by_rating_bucket = tail_loss_attribution_table(result.diagnostics, "tail_loss_by_rating_bucket")

    print("Portfolio Credit Risk Demo")
    print("-" * 26)
    print(f"Portfolio file: {portfolio_path}")
    print(f"Exposures: {len(portfolio)}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Simulation mode: {args.simulation_mode}")
    print(f"Stress mode: {stress_config.mode}")
    if args.stress == "regime":
        print("Scenario-specific stress inputs are applied inside the simulation.")
        print("Benchmark basis: migrated one-year PV versus same-rating one-year reference PV under scenario-consistent LGD, CCF, and spread inputs")
        regime_definition_table = pd.DataFrame(
            [
                {
                    "regime_label": regime.label,
                    "probability_pct": regime.probability * 100.0,
                    "lgd_multiplier": regime.lgd_multiplier,
                    "ccf_multiplier": regime.ccf_multiplier,
                    "spread_shift_bps": regime.spread_shift_bps,
                }
                for regime in regime_config.regimes
            ]
        )
        print("\nRegime stress definitions")
        print(regime_definition_table.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    else:
        print(f"Stressed LGD: {stressed_lgd:.4f}")
        print(f"Stressed CCF: {stressed_ccf:.4f}")
        print(f"Deterministic spread shift (bps): {stress_config.spread_shift_bps:.2f}")
        print("Benchmark basis: migrated one-year PV versus same-rating one-year reference PV under scenario-consistent inputs")
    print(f"Clamp positive PnL when constructing loss: {args.clamp_positive_pnl}")
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

    print("\nExposure by issuer type")
    print(exposure_by_issuer_type.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nExposure by instrument subtype")
    print(exposure_by_instrument_subtype.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nExposure by seniority")
    print(exposure_by_seniority.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nDistribution diagnostics")
    print(f"Mean PnL: {_format_money(metric_summary['mean_pnl'])}")
    print(f"Mean Loss: {_format_money(metric_summary['mean_loss'])}")
    print(f"Loss median: {_format_money(metric_summary['loss_median'])}")
    print(f"Positive PnL scenarios: {metric_summary['positive_pnl_pct']:.2f}%")
    print(f"Loss std: {_format_money(metric_summary['loss_std'])}")
    print(f"Loss min: {_format_money(metric_summary['loss_min'])}")
    print(f"Loss max: {_format_money(metric_summary['loss_max'])}")
    print(f"Loss skewness: {metric_summary['loss_skewness']:.4f}")
    print(f"Loss p1: {_format_money(metric_summary['p01'])}")
    print(f"Loss p5: {_format_money(metric_summary['p05'])}")
    print(f"Loss p50: {_format_money(metric_summary['p50'])}")
    print(f"Loss p95: {_format_money(metric_summary['p95'])}")
    print(f"Loss p99: {_format_money(metric_summary['p99'])}")
    print(f"VaR 95%: {_format_money(metric_summary['var_95'])}")
    print(f"VaR 99%: {_format_money(metric_summary['var_99'])}")
    print(f"VaR 99.9%: {_format_money(metric_summary['var_999'])}")
    print(f"ES 99%: {_format_money(metric_summary['es_99'])}")
    print(f"Probability of 5+ defaults: {metric_summary['prob_5plus_defaults']:.2f}%")
    print(f"Probability of 10+ defaults: {metric_summary['prob_10plus_defaults']:.2f}%")
    print(f"Probability of 20+ downgrades: {metric_summary['prob_20plus_downgrades']:.2f}%")
    print(f"Average default count in worst 1% loss scenarios: {metric_summary['tail_avg_default_count']:.2f}")
    print(f"Average downgrade count in worst 1% loss scenarios: {metric_summary['tail_avg_downgrade_count']:.2f}")
    print(f"Loss distribution plot: {plot_path}")

    by_instrument = breakdown_by_instrument_type(result.exposure_results)
    by_rating_bucket = breakdown_by_rating_bucket(result.exposure_results)

    print("\nExpected loss by instrument type")
    print(by_instrument.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nExpected loss by starting rating bucket")
    print(by_rating_bucket.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    print("\nDefault and downgrade summary by exposure class")
    print(
        default_rate_by_exposure_class.to_string(
            index=False,
            formatters={
                "default_rate": _format_percentage,
                "downgrade_frequency": _format_percentage,
            },
        )
    )

    print("\nDefault and downgrade summary by issuer type")
    print(
        default_rate_by_issuer_type.to_string(
            index=False,
            formatters={
                "default_rate": _format_percentage,
                "downgrade_frequency": _format_percentage,
            },
        )
    )

    print("\nDefault count distribution across scenarios")
    print(default_count_distribution.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    if regime_distribution is not None:
        print("\nRealized scenario regime distribution")
        print(regime_distribution.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    if not tail_loss_by_issuer_type.empty:
        print("\nWorst 1% tail loss by issuer type")
        print(tail_loss_by_issuer_type.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    if not tail_loss_by_sector.empty:
        print("\nWorst 1% tail loss by sector")
        print(tail_loss_by_sector.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    if not tail_loss_by_instrument_subtype.empty:
        print("\nWorst 1% tail loss by instrument subtype")
        print(tail_loss_by_instrument_subtype.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    if not tail_loss_by_rating_bucket.empty:
        print("\nWorst 1% tail loss by rating bucket")
        print(tail_loss_by_rating_bucket.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    if args.compare_modes or (args.compare_independent and args.simulation_mode in {"one_factor", "multi_factor"}):
        comparison_mode_map: dict[str, object | None] = {
            "independent": None,
            "one_factor": partial(simulate_one_factor_transitions, factor_spec=None),
            "multi_factor": partial(simulate_multi_factor_transitions, factor_spec=None),
        }
        selected_modes = ["independent", "one_factor", "multi_factor"] if args.compare_modes else ["independent", args.simulation_mode]

        comparison_results: dict[str, pd.DataFrame] = {}
        for mode_name in selected_modes:
            sampler = comparison_mode_map[mode_name]
            comparison_result = result if mode_name == args.simulation_mode else simulate_portfolio(
                portfolio=portfolio,
                transition_matrices=stressed_transition_matrices,
                n_scenarios=args.scenarios,
                seed=args.seed,
                allow_positive_pnl=not args.clamp_positive_pnl,
                base_lgd=stressed_lgd,
                ccf=stressed_ccf,
                transition_sampler=sampler,
                stress_mode=args.stress,
                spread_shift_bps=stress_config.spread_shift_bps,
                regime_config=regime_config,
            )
            comparison_results[mode_name] = comparison_result.scenario_results

        comparison_df = build_mode_comparison_table(comparison_results)
        print("\nSimulation mode comparison")
        print(comparison_df.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    if args.export_scenarios:
        export_path = export_scenario_results(result.scenario_results, OUTPUTS_DIR / "scenario_results.csv")
        print(f"\nScenario results exported to: {export_path}")


if __name__ == "__main__":
    main()
