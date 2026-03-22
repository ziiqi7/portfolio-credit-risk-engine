from src.schema import Exposure
from src.valuation import value_bond, value_loan, value_off_balance


def _loan_exposure() -> Exposure:
    return Exposure(
        exposure_id="EXP-0100",
        obligor_id="OBL-0100",
        instrument_type="loan",
        exposure_class="corporate",
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
