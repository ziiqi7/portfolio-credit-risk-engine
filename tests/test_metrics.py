import pandas as pd

from src.metrics import calculate_expected_shortfall, calculate_var, summarize_distribution


def test_var_is_monotonic_in_confidence_level() -> None:
    losses = pd.Series([1, 2, 3, 4, 5, 6, 10, 12, 15, 20])
    assert calculate_var(losses, 0.99) >= calculate_var(losses, 0.95)


def test_expected_shortfall_exceeds_var() -> None:
    losses = pd.Series([1, 2, 3, 4, 5, 6, 10, 12, 15, 20])
    var_99 = calculate_var(losses, 0.99)
    es_99 = calculate_expected_shortfall(losses, 0.99)
    assert es_99 >= var_99


def test_distribution_summary_contains_expected_keys() -> None:
    scenario_results = pd.DataFrame(
        {
            "total_pnl": [-5.0, -1.0, 0.0, 1.0],
            "total_loss": [5.0, 1.0, -0.0, -1.0],
        }
    )
    summary = summarize_distribution(scenario_results)
    assert {"mean_pnl", "mean_loss", "var_95", "var_99", "var_999", "es_99"} <= set(summary)
