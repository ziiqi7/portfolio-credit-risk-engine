from pathlib import Path

from scripts.run_demo import render_demo_report, run_demo_simulation
from src.simulation import simulate_portfolio
from src.synthetic_data import generate_synthetic_portfolio
from src.transitions import load_demo_transition_matrices


def test_run_demo_helper_exports_plot_and_scenarios(tmp_path: Path) -> None:
    artifacts = run_demo_simulation(
        scenarios=24,
        seed=9,
        simulation_mode="independent",
        stress="none",
        export_scenarios=True,
        plot_path=tmp_path / "demo_distribution.png",
        scenario_export_path=tmp_path / "scenario_results.csv",
    )

    report = render_demo_report(artifacts)

    assert artifacts.plot_path.exists()
    assert artifacts.scenario_export_path is not None
    assert artifacts.scenario_export_path.exists()
    assert "Simulation mode: independent" in report
    assert "Stress mode: none" in report


def test_same_seed_produces_identical_results() -> None:
    """Reproducibility: identical seeds must yield identical scenario results."""

    portfolio = generate_synthetic_portfolio(num_exposures=30, seed=7)
    matrices = load_demo_transition_matrices()

    result_a = simulate_portfolio(portfolio, matrices, n_scenarios=200, seed=42)
    result_b = simulate_portfolio(portfolio, matrices, n_scenarios=200, seed=42)

    pd_assert = result_a.scenario_results.equals(result_b.scenario_results)
    assert pd_assert
