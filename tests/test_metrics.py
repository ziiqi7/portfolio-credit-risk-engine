from functools import partial

import pandas as pd

from src.reporting import build_mode_comparison_table
from src.schema import Exposure
from src.metrics import calculate_expected_shortfall, calculate_var, summarize_distribution
from src.simulation import simulate_multi_factor_transitions, simulate_one_factor_transitions, simulate_portfolio
from src.transitions import load_demo_transition_matrices


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
    assert {
        "mean_pnl",
        "mean_loss",
        "loss_std",
        "loss_min",
        "loss_max",
        "loss_skewness",
        "positive_pnl_pct",
        "var_95",
        "var_99",
        "var_999",
        "es_99",
        "prob_5plus_defaults",
        "prob_10plus_defaults",
        "prob_20plus_downgrades",
        "tail_avg_default_count",
        "tail_avg_downgrade_count",
    } <= set(summary)


def test_loss_clamp_disallows_positive_offsets() -> None:
    exposure = Exposure(
        exposure_id="EXP-9001",
        obligor_id="OBL-9001",
        instrument_type="bond",
        exposure_class="corporate",
        issuer_type="corporate",
        instrument_subtype="corporate_bond",
        seniority="senior_unsecured",
        secured_flag=False,
        sector="technology",
        rating="BBB",
        currency="USD",
        balance=5_000_000,
        undrawn=0.0,
        maturity_years=4.0,
        coupon_rate=0.055,
        rate_type="fixed",
        guaranteed=False,
        collateral_type=None,
    )

    def sampler(_portfolio, _transition_matrices, _n_scenarios, _seed):
        return pd.Series(["AA", "BBB"], dtype=object).to_numpy().reshape(2, 1)

    result = simulate_portfolio(
        portfolio=[exposure],
        transition_matrices=load_demo_transition_matrices(),
        n_scenarios=2,
        seed=1,
        allow_positive_pnl=False,
        transition_sampler=sampler,
    )

    assert result.scenario_results["total_pnl"].max() > 0.0
    assert result.scenario_results["total_loss"].min() >= 0.0


def test_same_rating_reference_path_produces_zero_pnl() -> None:
    exposure = Exposure(
        exposure_id="EXP-9002",
        obligor_id="OBL-9002",
        instrument_type="bond",
        exposure_class="sovereign",
        issuer_type="sovereign",
        instrument_subtype="sovereign_bond",
        seniority="senior_unsecured",
        secured_flag=False,
        sector="public_sector",
        rating="AA",
        currency="EUR",
        balance=8_000_000,
        undrawn=0.0,
        maturity_years=6.0,
        coupon_rate=0.028,
        rate_type="fixed",
        guaranteed=False,
        collateral_type=None,
    )

    def sampler(_portfolio, _transition_matrices, _n_scenarios, _seed):
        return pd.Series(["AA", "AA"], dtype=object).to_numpy().reshape(2, 1)

    result = simulate_portfolio(
        portfolio=[exposure],
        transition_matrices=load_demo_transition_matrices(),
        n_scenarios=2,
        seed=1,
        allow_positive_pnl=True,
        transition_sampler=sampler,
    )

    assert (result.scenario_results["total_pnl"].abs() < 1e-6).all()
    assert (result.scenario_results["total_loss"].abs() < 1e-6).all()


def test_regime_mode_adds_regime_label_and_comparison_table_builds() -> None:
    exposure = Exposure(
        exposure_id="EXP-9003",
        obligor_id="OBL-9003",
        instrument_type="bond",
        exposure_class="corporate",
        issuer_type="corporate",
        instrument_subtype="corporate_bond",
        seniority="senior_unsecured",
        secured_flag=False,
        sector="technology",
        rating="BBB",
        currency="USD",
        balance=4_000_000,
        undrawn=0.0,
        maturity_years=5.0,
        coupon_rate=0.05,
        rate_type="fixed",
        guaranteed=False,
        collateral_type=None,
    )
    portfolio = [exposure]
    matrices = load_demo_transition_matrices()
    results_by_mode = {}
    sampler_map = {
        "independent": None,
        "one_factor": partial(simulate_one_factor_transitions, factor_spec=None),
        "multi_factor": partial(simulate_multi_factor_transitions, factor_spec=None),
    }

    for mode_name, sampler in sampler_map.items():
        result = simulate_portfolio(
            portfolio=portfolio,
            transition_matrices=matrices,
            n_scenarios=24,
            seed=7,
            transition_sampler=sampler,
            stress_mode="regime",
        )
        assert "regime_label" in result.scenario_results.columns
        results_by_mode[mode_name] = result.scenario_results

    comparison = build_mode_comparison_table(results_by_mode)
    assert set(comparison["mode"]) == {"independent", "one_factor", "multi_factor"}
    assert {"mean_loss", "loss_std", "var_95", "var_99", "es_99", "positive_pnl_pct"} <= set(comparison.columns)
