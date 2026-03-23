from src.schema import Exposure
from src.valuation import value_bond, value_loan, value_off_balance


def _loan_exposure() -> Exposure:
    return Exposure(
        exposure_id="EXP-0100",
        obligor_id="OBL-0100",
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
        undrawn=100_000,
        maturity_years=5.0,
        coupon_rate=0.05,
        rate_type="floating",
        guaranteed=False,
        collateral_type=None,
    )


def test_default_value_is_below_performing_value() -> None:
    exposure = _loan_exposure()
    performing_value = value_loan(exposure, target_rating="BBB")
    default_value = value_loan(exposure, target_rating="D", horizon_years=1.0)
    assert default_value < performing_value


def test_downgrade_reduces_value() -> None:
    exposure = _loan_exposure()
    investment_grade_value = value_loan(exposure, target_rating="BBB")
    high_yield_value = value_loan(exposure, target_rating="BB")
    assert high_yield_value < investment_grade_value


def test_off_balance_uses_ccf_adjusted_notional() -> None:
    exposure = Exposure(
        exposure_id="EXP-0200",
        obligor_id="OBL-0200",
        instrument_type="off_balance",
        exposure_class="fi",
        issuer_type="bank",
        instrument_subtype="letter_of_credit",
        seniority="senior_unsecured",
        secured_flag=False,
        sector="banking",
        rating="A",
        currency="USD",
        balance=500_000,
        undrawn=1_000_000,
        maturity_years=3.0,
        coupon_rate=0.02,
        rate_type="floating",
        guaranteed=False,
        collateral_type=None,
    )
    off_balance_value = value_off_balance(exposure, target_rating="A", ccf=0.5)
    bond_like_value = value_bond(
        Exposure(
            exposure_id="EXP-0201",
            obligor_id="OBL-0201",
            instrument_type="bond",
            exposure_class="fi",
            issuer_type="bank",
            instrument_subtype="bank_senior_bond",
            seniority="senior_unsecured",
            secured_flag=False,
            sector="banking",
            rating="A",
            currency="USD",
            balance=500_000,
            undrawn=0,
            maturity_years=3.0,
            coupon_rate=0.02,
            rate_type="fixed",
            guaranteed=False,
            collateral_type=None,
        ),
        target_rating="A",
    )
    assert off_balance_value > bond_like_value


def test_covered_bond_has_higher_recovery_value_than_unsecured_bank_bond() -> None:
    covered = Exposure(
        exposure_id="EXP-0300",
        obligor_id="OBL-0300",
        instrument_type="bond",
        exposure_class="fi",
        issuer_type="bank",
        instrument_subtype="covered_bond",
        seniority="covered",
        secured_flag=True,
        sector="banking",
        rating="A",
        currency="EUR",
        balance=10_000_000,
        undrawn=0.0,
        maturity_years=5.0,
        coupon_rate=0.03,
        rate_type="fixed",
        guaranteed=False,
        collateral_type=None,
    )
    unsecured = Exposure(
        exposure_id="EXP-0301",
        obligor_id="OBL-0301",
        instrument_type="bond",
        exposure_class="fi",
        issuer_type="bank",
        instrument_subtype="bank_senior_bond",
        seniority="senior_unsecured",
        secured_flag=False,
        sector="banking",
        rating="A",
        currency="EUR",
        balance=10_000_000,
        undrawn=0.0,
        maturity_years=5.0,
        coupon_rate=0.03,
        rate_type="fixed",
        guaranteed=False,
        collateral_type=None,
    )
    assert value_bond(covered, target_rating="D", horizon_years=1.0) > value_bond(unsecured, target_rating="D", horizon_years=1.0)


def test_positive_spread_shock_reduces_bond_value() -> None:
    bond = Exposure(
        exposure_id="EXP-0400",
        obligor_id="OBL-0400",
        instrument_type="bond",
        exposure_class="corporate",
        issuer_type="corporate",
        instrument_subtype="corporate_bond",
        seniority="subordinated",
        secured_flag=False,
        sector="energy",
        rating="BB",
        currency="USD",
        balance=7_500_000,
        undrawn=0.0,
        maturity_years=6.0,
        coupon_rate=0.062,
        rate_type="fixed",
        guaranteed=False,
        collateral_type=None,
    )

    base_value = value_bond(bond, target_rating="BB", horizon_years=1.0, spread_shock_bps=0.0)
    stressed_value = value_bond(bond, target_rating="BB", horizon_years=1.0, spread_shock_bps=120.0)

    assert stressed_value < base_value


def test_negative_spread_shock_path_remains_finite() -> None:
    exposure = _loan_exposure()
    value = value_loan(exposure, target_rating="BBB", horizon_years=1.0, spread_shock_bps=-90.0)

    assert value > 0.0
