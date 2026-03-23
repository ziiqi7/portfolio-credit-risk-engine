from collections import Counter

from src.schema import Exposure
from src.synthetic_data import generate_synthetic_portfolio


def test_exposure_creation() -> None:
    exposure = Exposure(
        exposure_id="EXP-0001",
        obligor_id="OBL-0001",
        instrument_type="loan",
        exposure_class="corporate",
        issuer_type="corporate",
        instrument_subtype="term_loan",
        seniority="senior_secured",
        secured_flag=True,
        sector="industrials",
        rating="BBB",
        currency="EUR",
        balance=1_000_000,
        undrawn=150_000,
        maturity_years=4.5,
        coupon_rate=0.045,
        rate_type="floating",
        guaranteed=True,
        collateral_type="cash",
    )

    assert exposure.exposure_id == "EXP-0001"
    assert exposure.rating == "BBB"
    assert exposure.guaranteed is True


def test_exposure_rejects_invalid_rating() -> None:
    try:
        Exposure(
            exposure_id="EXP-0002",
            obligor_id="OBL-0002",
            instrument_type="bond",
            exposure_class="corporate",
            issuer_type="corporate",
            instrument_subtype="corporate_bond",
            seniority="senior_unsecured",
            secured_flag=False,
            sector="technology",
            rating="WR",
            currency="USD",
            balance=2_000_000,
            undrawn=0,
            maturity_years=3.0,
            coupon_rate=0.05,
            rate_type="fixed",
            guaranteed=False,
        )
    except ValueError as error:
        assert "Unsupported starting rating" in str(error)
    else:
        raise AssertionError("Expected invalid rating to raise ValueError.")


def test_exposure_rejects_bond_with_undrawn() -> None:
    try:
        Exposure(
            exposure_id="EXP-0003",
            obligor_id="OBL-0003",
            instrument_type="bond",
            exposure_class="corporate",
            issuer_type="corporate",
            instrument_subtype="corporate_bond",
            seniority="senior_unsecured",
            secured_flag=False,
            sector="technology",
            rating="A",
            currency="EUR",
            balance=3_000_000,
            undrawn=250_000,
            maturity_years=5.0,
            coupon_rate=0.04,
            rate_type="fixed",
            guaranteed=False,
        )
    except ValueError as error:
        assert "zero undrawn" in str(error)
    else:
        raise AssertionError("Expected invalid bond definition to raise ValueError.")


def test_generated_portfolio_has_repeat_obligors_and_valid_combinations() -> None:
    portfolio = generate_synthetic_portfolio(num_exposures=180, seed=7)
    obligor_counts = Counter(exposure.obligor_id for exposure in portfolio)

    assert len(portfolio) == 180
    assert max(obligor_counts.values()) >= 3
    assert any(count > 1 for count in obligor_counts.values())
    assert all(exposure.balance > 0 for exposure in portfolio)
    assert all(exposure.rating in {"AAA", "AA", "A", "BBB", "BB", "B", "CCC"} for exposure in portfolio)
    assert all(exposure.exposure_class in {"corporate", "fi", "sovereign"} for exposure in portfolio)
    assert all(exposure.issuer_type in {"corporate", "bank", "insurance", "sovereign", "agency", "supranational"} for exposure in portfolio)
    assert all(exposure.instrument_subtype for exposure in portfolio)
    assert all(exposure.maturity_years > 0 for exposure in portfolio)
    assert all(exposure.undrawn == 0.0 for exposure in portfolio if exposure.instrument_type == "bond")
    assert all(exposure.undrawn <= exposure.balance for exposure in portfolio if exposure.instrument_type == "off_balance")
    assert all(exposure.instrument_subtype == "covered_bond" for exposure in portfolio if exposure.seniority == "covered")
