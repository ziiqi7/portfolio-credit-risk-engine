"""Run a batch of portfolio credit risk experiments in Python."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo import DemoRunArtifacts, render_demo_report, run_demo_simulation
from src.config import OUTPUTS_DIR
from src.reporting import plot_mode_comparison_distribution


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Configuration for a single batch experiment."""

    name: str
    simulation_mode: str
    stress: str
    log_filename: str
    plot_filename: str
    scenario_filename: str
    compare_modes: bool = False


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="independent",
        simulation_mode="independent",
        stress="none",
        log_filename="independent.log",
        plot_filename="independent_distribution.png",
        scenario_filename="scenario_results_independent.csv",
    ),
    ExperimentSpec(
        name="one_factor",
        simulation_mode="one_factor",
        stress="none",
        log_filename="one_factor.log",
        plot_filename="one_factor_distribution.png",
        scenario_filename="scenario_results_one_factor.csv",
    ),
    ExperimentSpec(
        name="multi_factor",
        simulation_mode="multi_factor",
        stress="none",
        log_filename="multi_factor.log",
        plot_filename="multi_factor_distribution.png",
        scenario_filename="scenario_results_multi_factor.csv",
    ),
    ExperimentSpec(
        name="multi_factor_regime",
        simulation_mode="multi_factor",
        stress="regime",
        log_filename="multi_factor_regime.log",
        plot_filename="multi_factor_regime_distribution.png",
        scenario_filename="scenario_results_multi_factor_regime.csv",
    ),
    ExperimentSpec(
        name="comparison",
        simulation_mode="multi_factor",
        stress="regime",
        log_filename="comparison.log",
        plot_filename="comparison_primary_distribution.png",
        scenario_filename="scenario_results_comparison.csv",
        compare_modes=True,
    ),
)


def _write_log(log_path: Path, report_text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report_text + "\n", encoding="utf-8")


def _print_experiment_header(spec: ExperimentSpec, log_path: Path) -> None:
    print()
    print("=" * 68)
    print(f"Experiment: {spec.name}")
    print(f"simulation-mode: {spec.simulation_mode}")
    print(f"stress: {spec.stress}")
    print(f"compare-modes: {spec.compare_modes}")
    print(f"log: {log_path}")
    print("=" * 68)


def run_experiment(spec: ExperimentSpec, scenarios: int = 2_000, seed: int = 42) -> DemoRunArtifacts:
    """Run a single experiment and persist its log and core artifacts."""

    log_path = OUTPUTS_DIR / "logs" / spec.log_filename
    plot_path = OUTPUTS_DIR / spec.plot_filename
    scenario_path = OUTPUTS_DIR / spec.scenario_filename

    _print_experiment_header(spec, log_path)
    artifacts = run_demo_simulation(
        scenarios=scenarios,
        seed=seed,
        simulation_mode=spec.simulation_mode,
        stress=spec.stress,
        compare_modes=spec.compare_modes,
        export_scenarios=True,
        plot_path=plot_path,
        scenario_export_path=scenario_path,
    )
    report_text = render_demo_report(artifacts)
    print(report_text)
    _write_log(log_path, report_text)
    return artifacts


def run_all_experiments(scenarios: int = 2_000, seed: int = 42) -> dict[str, DemoRunArtifacts]:
    """Run the full experiment set and generate comparison artifacts."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "logs").mkdir(parents=True, exist_ok=True)

    results: dict[str, DemoRunArtifacts] = {}
    for spec in EXPERIMENTS:
        results[spec.name] = run_experiment(spec, scenarios=scenarios, seed=seed)

    comparison_results = {
        "independent": results["independent"].result.scenario_results,
        "one_factor": results["one_factor"].result.scenario_results,
        "multi_factor": results["multi_factor"].result.scenario_results,
        "multi_factor_regime": results["multi_factor_regime"].result.scenario_results,
    }
    comparison_plot_path = plot_mode_comparison_distribution(
        comparison_results,
        OUTPUTS_DIR / "comparison_distribution.png",
        title="Loss Distribution Comparison Across Simulation Modes",
        subtitle="Independent vs one-factor vs multi-factor vs multi-factor regime stress",
    )
    print()
    print(f"Comparison distribution chart saved to: {comparison_plot_path}")
    return results


def main() -> None:
    run_all_experiments()


if __name__ == "__main__":
    main()
