import numpy as np

from src.config import RATINGS
from src.correlation import (
    LatentFactorSpec,
    MultiFactorSpec,
    asset_correlation_for_exposure,
    build_threshold_lookup,
    map_latent_to_rating_state,
    multi_factor_loadings_for_exposure,
    sample_multi_factor_latent_variables,
    simulate_multi_factor_transitions,
    simulate_one_factor_transitions,
    supported_sectors,
)
from src.synthetic_data import generate_synthetic_portfolio
from src.transitions import load_demo_transition_matrices


def test_threshold_mapping_extremes_map_to_worst_and_best_states() -> None:
    matrices = load_demo_transition_matrices()
    lookup = build_threshold_lookup(matrices)
    thresholds, fallback_state = lookup["corporate"]["BBB"]

    assert map_latent_to_rating_state(-8.0, thresholds, fallback_state) == "D"
    assert map_latent_to_rating_state(8.0, thresholds, fallback_state) == fallback_state


def test_one_factor_transition_output_shape_and_state_validity() -> None:
    portfolio = generate_synthetic_portfolio(num_exposures=24, seed=11)
    matrices = load_demo_transition_matrices()
    migrated = simulate_one_factor_transitions(portfolio, matrices, n_scenarios=40, seed=5)

    assert migrated.shape == (40, 24)
    assert set(np.unique(migrated)).issubset(set(RATINGS))


def test_asset_correlation_varies_by_issuer_type_and_rating() -> None:
    portfolio = generate_synthetic_portfolio(num_exposures=80, seed=15)
    sovereign_like = next(exposure for exposure in portfolio if exposure.issuer_type in {"sovereign", "agency", "supranational"})
    corporate_like = next(exposure for exposure in portfolio if exposure.issuer_type == "corporate")
    speculative_grade = next(exposure for exposure in portfolio if exposure.rating in {"BB", "B", "CCC"})

    spec = LatentFactorSpec()
    assert asset_correlation_for_exposure(sovereign_like, spec) < asset_correlation_for_exposure(corporate_like, spec)
    assert asset_correlation_for_exposure(speculative_grade, spec) >= spec.issuer_type_base_correlations[speculative_grade.issuer_type]


def test_one_factor_collapses_to_near_independent_at_zero_correlation() -> None:
    """At asset_correlation ≈ 0 the one-factor sampler should produce
    transitions whose marginal default rates closely match the input
    transition matrix, since the systematic factor's contribution is
    bounded by min_correlation."""

    portfolio = generate_synthetic_portfolio(num_exposures=80, seed=21)
    matrices = load_demo_transition_matrices()

    spec = LatentFactorSpec(
        issuer_type_base_correlations={
            "sovereign": 0.03,
            "supranational": 0.03,
            "agency": 0.03,
            "bank": 0.03,
            "insurance": 0.03,
            "corporate": 0.03,
        },
        rating_adjustments={r: 0.0 for r in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]},
        min_correlation=0.03,
        max_correlation=0.05,
    )

    migrated = simulate_one_factor_transitions(
        portfolio, matrices, n_scenarios=4000, seed=11, factor_spec=spec
    )

    for exposure_class in ("corporate", "fi", "sovereign"):
        class_indices = [i for i, exposure in enumerate(portfolio) if exposure.exposure_class == exposure_class]
        if not class_indices:
            continue
        realised = (migrated[:, class_indices] == "D").mean()
        starting_default_probs = []
        for index in class_indices:
            rating = portfolio[index].rating
            starting_default_probs.append(matrices[exposure_class].matrix.loc[rating, "D"])
        expected = float(np.mean(starting_default_probs))
        assert abs(realised - expected) < 0.005


def test_supported_sectors_cover_generated_portfolio() -> None:
    portfolio = generate_synthetic_portfolio(num_exposures=120, seed=8)
    generated_sectors = {exposure.sector for exposure in portfolio}

    assert generated_sectors.issubset(supported_sectors())


def test_multi_factor_loadings_are_bounded_and_valid() -> None:
    portfolio = generate_synthetic_portfolio(num_exposures=60, seed=13)
    spec = MultiFactorSpec()

    for exposure in portfolio:
        macro_loading, sector_loading = multi_factor_loadings_for_exposure(exposure, spec)
        assert macro_loading >= 0.0
        assert sector_loading >= 0.0
        assert macro_loading + sector_loading < 1.0
        assert macro_loading + sector_loading <= spec.max_total_loading + 1e-12


def test_multi_factor_latent_sampler_output_shape() -> None:
    macro_loadings = np.array([0.12, 0.18, 0.09, 0.14], dtype=float)
    sector_loadings = np.array([0.10, 0.07, 0.05, 0.11], dtype=float)
    sector_labels = ["banking", "banking", "technology", "energy"]

    latent = sample_multi_factor_latent_variables(
        macro_loadings=macro_loadings,
        sector_loadings=sector_loadings,
        sector_labels=sector_labels,
        num_scenarios=25,
        random_seed=3,
    )

    assert latent.shape == (25, 4)
    assert np.isfinite(latent).all()


def test_multi_factor_transition_output_shape_and_state_validity() -> None:
    portfolio = generate_synthetic_portfolio(num_exposures=30, seed=17)
    matrices = load_demo_transition_matrices()
    migrated = simulate_multi_factor_transitions(portfolio, matrices, n_scenarios=35, seed=9)

    assert migrated.shape == (35, 30)
    assert set(np.unique(migrated)).issubset(set(RATINGS))
