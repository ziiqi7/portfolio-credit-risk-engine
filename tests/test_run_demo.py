from pathlib import Path

from scripts.run_demo import render_demo_report, run_demo_simulation


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
