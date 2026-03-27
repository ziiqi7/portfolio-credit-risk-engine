"""Run the baseline portfolio credit risk demo."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
from src.simulation import SimulationResult, simulate_multi_factor_transitions, simulate_one_factor_transitions, simulate_portfolio
from src.stress import RegimeStressConfig, StressConfig, apply_stress_overlays, get_regime_stress_config
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


@dataclass(slots=True)
class DemoRunArtifacts:
    """Structured artifacts returned by a demo simulation run."""

    portfolio_path: Path
    simulation_mode: str
    stress_mode: str
    scenarios: int
    seed: int
    clamp_positive_pnl: bool
    stress_config: StressConfig
    stressed_lgd: float
    stressed_ccf: float
    regime_config: RegimeStressConfig | None
    regime_definition_table: pd.DataFrame | None
    benchmark_basis: str
    plot_path: Path
    scenario_export_path: Path | None
    result: SimulationResult
    metric_summary: dict[str, float]
    portfolio_summary: pd.DataFrame
    top_obligors: pd.DataFrame
    exposure_by_instrument: pd.DataFrame
    exposure_by_rating_bucket: pd.DataFrame
    exposure_by_sector: pd.DataFrame
    exposure_by_currency: pd.DataFrame
    exposure_by_issuer_type: pd.DataFrame
    exposure_by_instrument_subtype: pd.DataFrame
    exposure_by_seniority: pd.DataFrame
    by_instrument: pd.DataFrame
    by_rating_bucket: pd.DataFrame
    default_rate_by_exposure_class: pd.DataFrame
    default_rate_by_issuer_type: pd.DataFrame
    default_count_distribution: pd.DataFrame
    regime_distribution: pd.DataFrame | None
    tail_loss_by_issuer_type: pd.DataFrame
    tail_loss_by_sector: pd.DataFrame
    tail_loss_by_instrument_subtype: pd.DataFrame
    tail_loss_by_rating_bucket: pd.DataFrame
    comparison_df: pd.DataFrame | None


def _resolve_transition_sampler(simulation_mode: str):
    if simulation_mode == "one_factor":
        return partial(simulate_one_factor_transitions, factor_spec=None)
    if simulation_mode == "multi_factor":
        return partial(simulate_multi_factor_transitions, factor_spec=None)
    return None


def _build_regime_definition_table(regime_config: RegimeStressConfig | None) -> pd.DataFrame | None:
    if regime_config is None:
        return None
    return pd.DataFrame(
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


def _build_mode_comparison(
    simulation_mode: str,
    compare_independent: bool,
    compare_modes: bool,
    primary_result: SimulationResult,
    portfolio,
    transition_matrices,
    scenarios: int,
    seed: int,
    clamp_positive_pnl: bool,
    stressed_lgd: float,
    stressed_ccf: float,
    stress_mode: str,
    spread_shift_bps: float,
    regime_config: RegimeStressConfig | None,
) -> pd.DataFrame | None:
    if not compare_modes and not (compare_independent and simulation_mode in {"one_factor", "multi_factor"}):
        return None

    comparison_mode_map = {
        "independent": None,
        "one_factor": partial(simulate_one_factor_transitions, factor_spec=None),
        "multi_factor": partial(simulate_multi_factor_transitions, factor_spec=None),
    }
    selected_modes = ["independent", "one_factor", "multi_factor"] if compare_modes else ["independent", simulation_mode]

    comparison_results: dict[str, pd.DataFrame] = {}
    for mode_name in selected_modes:
        if mode_name == simulation_mode:
            comparison_results[mode_name] = primary_result.scenario_results
            continue

        comparison_result = simulate_portfolio(
            portfolio=portfolio,
            transition_matrices=transition_matrices,
            n_scenarios=scenarios,
            seed=seed,
            allow_positive_pnl=not clamp_positive_pnl,
            base_lgd=stressed_lgd,
            ccf=stressed_ccf,
            transition_sampler=comparison_mode_map[mode_name],
            stress_mode=stress_mode,
            spread_shift_bps=spread_shift_bps,
            regime_config=regime_config,
        )
        comparison_results[mode_name] = comparison_result.scenario_results
    return build_mode_comparison_table(comparison_results)


def run_demo_simulation(
    portfolio_path: Path = Path("data/synthetic/sample_portfolio.csv"),
    scenarios: int = 5_000,
    seed: int = 42,
    simulation_mode: str = "independent",
    stress: str = "none",
    clamp_positive_pnl: bool = False,
    compare_independent: bool = False,
    compare_modes: bool = False,
    export_scenarios: bool = False,
    plot_path: Path | None = None,
    scenario_export_path: Path | None = None,
) -> DemoRunArtifacts:
    """Run a demo simulation and return structured artifacts for reuse."""

    resolved_portfolio_path = _resolve_repo_path(portfolio_path)
    if not resolved_portfolio_path.exists():
        save_synthetic_portfolio(output_path=resolved_portfolio_path, num_exposures=150, seed=seed)

    portfolio = load_synthetic_portfolio(resolved_portfolio_path)
    transition_matrices = load_demo_transition_matrices()
    stressed_transition_matrices, stressed_lgd, stressed_ccf, stress_config = apply_stress_overlays(
        transition_matrices=transition_matrices,
        base_lgd=DEFAULT_LGD,
        base_ccf=DEFAULT_CCF,
        mode=stress,
    )
    transition_sampler = _resolve_transition_sampler(simulation_mode)
    regime_config = get_regime_stress_config() if stress == "regime" else None

    result = simulate_portfolio(
        portfolio=portfolio,
        transition_matrices=stressed_transition_matrices,
        n_scenarios=scenarios,
        seed=seed,
        allow_positive_pnl=not clamp_positive_pnl,
        base_lgd=stressed_lgd,
        ccf=stressed_ccf,
        transition_sampler=transition_sampler,
        stress_mode=stress,
        spread_shift_bps=stress_config.spread_shift_bps,
        regime_config=regime_config,
    )

    benchmark_basis = "migrated one-year PV versus same-rating one-year reference PV under scenario-consistent inputs"
    title = f"Portfolio Loss Distribution ({simulation_mode}, stress={stress})"
    subtitle = f"Benchmark: {benchmark_basis}"
    if stress == "regime":
        subtitle = f"{subtitle} | Regime stress: normal / stress / crisis mixture"

    resolved_plot_path = _resolve_repo_path(plot_path) if plot_path is not None else OUTPUTS_DIR / "demo_loss_distribution.png"
    actual_plot_path = plot_loss_distribution(result.scenario_results, resolved_plot_path, title=title, subtitle=subtitle)

    actual_scenario_export_path: Path | None = None
    if export_scenarios:
        export_target = scenario_export_path or (OUTPUTS_DIR / "scenario_results.csv")
        actual_scenario_export_path = export_scenario_results(result.scenario_results, _resolve_repo_path(export_target))

    comparison_df = _build_mode_comparison(
        simulation_mode=simulation_mode,
        compare_independent=compare_independent,
        compare_modes=compare_modes,
        primary_result=result,
        portfolio=portfolio,
        transition_matrices=stressed_transition_matrices,
        scenarios=scenarios,
        seed=seed,
        clamp_positive_pnl=clamp_positive_pnl,
        stressed_lgd=stressed_lgd,
        stressed_ccf=stressed_ccf,
        stress_mode=stress,
        spread_shift_bps=stress_config.spread_shift_bps,
        regime_config=regime_config,
    )

    return DemoRunArtifacts(
        portfolio_path=resolved_portfolio_path,
        simulation_mode=simulation_mode,
        stress_mode=stress,
        scenarios=scenarios,
        seed=seed,
        clamp_positive_pnl=clamp_positive_pnl,
        stress_config=stress_config,
        stressed_lgd=stressed_lgd,
        stressed_ccf=stressed_ccf,
        regime_config=regime_config,
        regime_definition_table=_build_regime_definition_table(regime_config),
        benchmark_basis=benchmark_basis,
        plot_path=actual_plot_path,
        scenario_export_path=actual_scenario_export_path,
        result=result,
        metric_summary=summarize_distribution(result.scenario_results),
        portfolio_summary=build_portfolio_summary_table(result.exposure_results),
        top_obligors=top_obligor_concentration_table(result.exposure_results),
        exposure_by_instrument=exposure_breakdown_table(result.exposure_results, "instrument_type"),
        exposure_by_rating_bucket=exposure_by_rating_bucket_table(result.exposure_results),
        exposure_by_sector=exposure_breakdown_table(result.exposure_results, "sector"),
        exposure_by_currency=exposure_breakdown_table(result.exposure_results, "currency"),
        exposure_by_issuer_type=exposure_breakdown_table(result.exposure_results, "issuer_type"),
        exposure_by_instrument_subtype=exposure_breakdown_table(result.exposure_results, "instrument_subtype"),
        exposure_by_seniority=exposure_breakdown_table(result.exposure_results, "seniority"),
        by_instrument=breakdown_by_instrument_type(result.exposure_results),
        by_rating_bucket=breakdown_by_rating_bucket(result.exposure_results),
        default_rate_by_exposure_class=result.diagnostics["default_rate_by_exposure_class"],
        default_rate_by_issuer_type=result.diagnostics["default_rate_by_issuer_type"],
        default_count_distribution=result.diagnostics["default_count_distribution"],
        regime_distribution=result.diagnostics.get("regime_distribution"),
        tail_loss_by_issuer_type=tail_loss_attribution_table(result.diagnostics, "tail_loss_by_issuer_type"),
        tail_loss_by_sector=tail_loss_attribution_table(result.diagnostics, "tail_loss_by_sector"),
        tail_loss_by_instrument_subtype=tail_loss_attribution_table(result.diagnostics, "tail_loss_by_instrument_subtype"),
        tail_loss_by_rating_bucket=tail_loss_attribution_table(result.diagnostics, "tail_loss_by_rating_bucket"),
        comparison_df=comparison_df,
    )


def render_demo_report(artifacts: DemoRunArtifacts) -> str:
    """Render a console-style report for a demo run."""

    lines = [
        "Portfolio Credit Risk Demo",
        "-" * 26,
        f"Portfolio file: {artifacts.portfolio_path}",
        f"Exposures: {len(artifacts.result.exposure_results)}",
        f"Scenarios: {artifacts.scenarios}",
        f"Simulation mode: {artifacts.simulation_mode}",
        f"Stress mode: {artifacts.stress_config.mode}",
    ]

    if artifacts.stress_mode == "regime":
        lines.append("Scenario-specific stress inputs are applied inside the simulation.")
        lines.append(f"Benchmark basis: {artifacts.benchmark_basis}")
        if artifacts.regime_definition_table is not None:
            lines.extend(
                [
                    "",
                    "Regime stress definitions",
                    artifacts.regime_definition_table.to_string(index=False, float_format=lambda value: f"{value:,.2f}"),
                ]
            )
    else:
        lines.extend(
            [
                f"Stressed LGD: {artifacts.stressed_lgd:.4f}",
                f"Stressed CCF: {artifacts.stressed_ccf:.4f}",
                f"Deterministic spread shift (bps): {artifacts.stress_config.spread_shift_bps:.2f}",
                f"Benchmark basis: {artifacts.benchmark_basis}",
            ]
        )
    lines.append(f"Clamp positive PnL when constructing loss: {artifacts.clamp_positive_pnl}")

    def append_table(title: str, table: pd.DataFrame, float_format: str = "{:,.2f}") -> None:
        lines.extend(["", title, table.to_string(index=False, float_format=lambda value: float_format.format(value))])

    append_table("Portfolio summary", artifacts.portfolio_summary)
    append_table("Top 10 obligor concentration", artifacts.top_obligors)
    append_table("Exposure by instrument type", artifacts.exposure_by_instrument)
    append_table("Exposure by rating bucket", artifacts.exposure_by_rating_bucket)
    append_table("Exposure by sector", artifacts.exposure_by_sector)
    append_table("Exposure by currency", artifacts.exposure_by_currency)
    append_table("Exposure by issuer type", artifacts.exposure_by_issuer_type)
    append_table("Exposure by instrument subtype", artifacts.exposure_by_instrument_subtype)
    append_table("Exposure by seniority", artifacts.exposure_by_seniority)

    summary = artifacts.metric_summary
    lines.extend(
        [
            "",
            "Distribution diagnostics",
            f"Mean PnL: {_format_money(summary['mean_pnl'])}",
            f"Mean Loss: {_format_money(summary['mean_loss'])}",
            f"Loss median: {_format_money(summary['loss_median'])}",
            f"Positive PnL scenarios: {summary['positive_pnl_pct']:.2f}%",
            f"Loss std: {_format_money(summary['loss_std'])}",
            f"Loss min: {_format_money(summary['loss_min'])}",
            f"Loss max: {_format_money(summary['loss_max'])}",
            f"Loss skewness: {summary['loss_skewness']:.4f}",
            f"Loss p1: {_format_money(summary['p01'])}",
            f"Loss p5: {_format_money(summary['p05'])}",
            f"Loss p50: {_format_money(summary['p50'])}",
            f"Loss p95: {_format_money(summary['p95'])}",
            f"Loss p99: {_format_money(summary['p99'])}",
            f"VaR 95%: {_format_money(summary['var_95'])}",
            f"VaR 99%: {_format_money(summary['var_99'])}",
            f"VaR 99.9%: {_format_money(summary['var_999'])}",
            f"ES 99%: {_format_money(summary['es_99'])}",
            f"Probability of 5+ defaults: {summary['prob_5plus_defaults']:.2f}%",
            f"Probability of 10+ defaults: {summary['prob_10plus_defaults']:.2f}%",
            f"Probability of 20+ downgrades: {summary['prob_20plus_downgrades']:.2f}%",
            f"Average default count in worst 1% loss scenarios: {summary['tail_avg_default_count']:.2f}",
            f"Average downgrade count in worst 1% loss scenarios: {summary['tail_avg_downgrade_count']:.2f}",
            f"Loss distribution plot: {artifacts.plot_path}",
        ]
    )

    append_table("Expected loss by instrument type", artifacts.by_instrument)
    append_table("Expected loss by starting rating bucket", artifacts.by_rating_bucket)
    lines.extend(
        [
            "",
            "Default and downgrade summary by exposure class",
            artifacts.default_rate_by_exposure_class.to_string(
                index=False,
                formatters={"default_rate": _format_percentage, "downgrade_frequency": _format_percentage},
            ),
            "",
            "Default and downgrade summary by issuer type",
            artifacts.default_rate_by_issuer_type.to_string(
                index=False,
                formatters={"default_rate": _format_percentage, "downgrade_frequency": _format_percentage},
            ),
        ]
    )
    append_table("Default count distribution across scenarios", artifacts.default_count_distribution)

    if artifacts.regime_distribution is not None:
        append_table("Realized scenario regime distribution", artifacts.regime_distribution)
    if not artifacts.tail_loss_by_issuer_type.empty:
        append_table("Worst 1% tail loss by issuer type", artifacts.tail_loss_by_issuer_type)
    if not artifacts.tail_loss_by_sector.empty:
        append_table("Worst 1% tail loss by sector", artifacts.tail_loss_by_sector)
    if not artifacts.tail_loss_by_instrument_subtype.empty:
        append_table("Worst 1% tail loss by instrument subtype", artifacts.tail_loss_by_instrument_subtype)
    if not artifacts.tail_loss_by_rating_bucket.empty:
        append_table("Worst 1% tail loss by rating bucket", artifacts.tail_loss_by_rating_bucket)
    if artifacts.comparison_df is not None:
        append_table("Simulation mode comparison", artifacts.comparison_df)
    if artifacts.scenario_export_path is not None:
        lines.extend(["", f"Scenario results exported to: {artifacts.scenario_export_path}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    artifacts = run_demo_simulation(
        portfolio_path=args.portfolio_path,
        scenarios=args.scenarios,
        seed=args.seed,
        simulation_mode=args.simulation_mode,
        stress=args.stress,
        clamp_positive_pnl=args.clamp_positive_pnl,
        compare_independent=args.compare_independent,
        compare_modes=args.compare_modes,
        export_scenarios=args.export_scenarios,
    )
    print(render_demo_report(artifacts))


if __name__ == "__main__":
    main()
