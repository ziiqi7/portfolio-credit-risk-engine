import numpy as np

from src.config import DEFAULT_CCF, DEFAULT_LGD
from src.stress import apply_stress_overlays, build_regime_stress_overlays, get_stress_config, sample_regime_labels
from src.transitions import load_demo_transition_matrices


def test_stress_config_modes_exist() -> None:
    assert get_stress_config("none").mode == "none"
    assert get_stress_config("mild").mode == "mild"
    assert get_stress_config("severe").mode == "severe"
    assert get_stress_config("regime").mode == "regime"


def test_regime_sampling_returns_supported_labels() -> None:
    labels = sample_regime_labels(n_scenarios=2_000, seed=11)

    assert set(labels).issubset({"normal", "stress", "crisis"})
    assert {"normal", "stress", "crisis"} <= set(labels)


def test_stress_overlay_produces_valid_matrices_and_bounded_inputs() -> None:
    matrices = load_demo_transition_matrices()
    stressed_matrices, stressed_lgd, stressed_ccf, _ = apply_stress_overlays(
        transition_matrices=matrices,
        base_lgd=DEFAULT_LGD,
        base_ccf=DEFAULT_CCF,
        mode="severe",
    )

    for matrix in stressed_matrices.values():
        assert np.allclose(matrix.matrix.sum(axis=1).to_numpy(dtype=float), 1.0)

    assert stressed_lgd >= DEFAULT_LGD
    assert stressed_lgd <= 0.95
    assert stressed_ccf >= DEFAULT_CCF
    assert stressed_ccf <= 1.0


def test_severe_stress_increases_default_probability_for_corporate_bbb() -> None:
    matrices = load_demo_transition_matrices()
    stressed_matrices, _, _, _ = apply_stress_overlays(
        transition_matrices=matrices,
        base_lgd=DEFAULT_LGD,
        base_ccf=DEFAULT_CCF,
        mode="severe",
    )

    base_default_prob = matrices["corporate"].matrix.loc["BBB", "D"]
    stressed_default_prob = stressed_matrices["corporate"].matrix.loc["BBB", "D"]
    assert stressed_default_prob > base_default_prob


def test_regime_overlay_builds_valid_regime_specific_inputs() -> None:
    matrices = load_demo_transition_matrices()
    overlays = build_regime_stress_overlays(
        transition_matrices=matrices,
        base_lgd=DEFAULT_LGD,
        base_ccf=DEFAULT_CCF,
    )

    assert set(overlays) == {"normal", "stress", "crisis"}
    for stressed_matrices, stressed_lgd, stressed_ccf, regime in overlays.values():
        for matrix in stressed_matrices.values():
            assert np.allclose(matrix.matrix.sum(axis=1).to_numpy(dtype=float), 1.0)
        assert DEFAULT_LGD <= stressed_lgd <= 0.95
        assert DEFAULT_CCF <= stressed_ccf <= 1.0
        assert regime.spread_shift_bps >= 0.0
