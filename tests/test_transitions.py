import numpy as np

from src.transitions import expected_rating_index, load_demo_transition_matrices, validate_transition_matrix


def test_transition_matrix_rows_sum_to_one() -> None:
    matrices = load_demo_transition_matrices()
    for matrix in matrices.values():
        row_sums = matrix.matrix.sum(axis=1).to_numpy(dtype=float)
        assert np.allclose(row_sums, 1.0)
        validate_transition_matrix(matrix.matrix)


def test_cumulative_transition_probabilities_end_at_one() -> None:
    matrices = load_demo_transition_matrices()
    for matrix in matrices.values():
        cumulative = matrix.cumulative()
        assert np.allclose(cumulative.iloc[:, -1].to_numpy(dtype=float), 1.0)


def test_transition_monotonicity_and_relative_strength() -> None:
    matrices = load_demo_transition_matrices()
    corporate = matrices["corporate"].matrix
    sovereign = matrices["sovereign"].matrix

    corporate_expected = expected_rating_index(corporate)
    sovereign_expected = expected_rating_index(sovereign)

    assert corporate.loc["AAA", "D"] < corporate.loc["AA", "D"] < corporate.loc["A", "D"]
    assert corporate_expected["AAA"] < corporate_expected["BBB"] < corporate_expected["B"]
    assert sovereign_expected["AAA"] < sovereign_expected["BBB"] < sovereign_expected["B"]
    assert corporate.loc["BBB", "D"] > sovereign.loc["BBB", "D"]
    assert corporate.loc["BB", "D"] > sovereign.loc["BB", "D"]
