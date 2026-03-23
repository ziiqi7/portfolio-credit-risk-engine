"""Instrument-specific valuation functions."""

from __future__ import annotations

import math

from src.config import (
    DEFAULT_CCF,
    DEFAULT_LGD,
    DEFAULT_RISK_FREE_RATE,
    INSTRUMENT_SUBTYPE_LGD_ADJUSTMENT,
    INSTRUMENT_SUBTYPE_SPREAD_BPS_ADJUSTMENT,
    ISSUER_LGD_ADJUSTMENT,
    ISSUER_SPREAD_BPS_ADJUSTMENT,
    RATING_SPREAD_BPS,
    SENIORITY_LGD_ADJUSTMENT,
    SENIORITY_SPREAD_BPS_ADJUSTMENT,
)
from src.schema import Exposure


def _discount_factor(rate: float, years: float) -> float:
    return 1.0 / ((1.0 + rate) ** max(years, 0.0))


def credit_spread_bps(exposure: Exposure, target_rating: str) -> float:
    """Return the annualized credit spread in basis points for a migrated state."""

    spread_bps = (
        RATING_SPREAD_BPS[target_rating]
        + ISSUER_SPREAD_BPS_ADJUSTMENT[exposure.issuer_type]
        + INSTRUMENT_SUBTYPE_SPREAD_BPS_ADJUSTMENT[exposure.instrument_subtype]
        + SENIORITY_SPREAD_BPS_ADJUSTMENT[exposure.seniority]
    )
    return float(max(spread_bps, 10.0))


def credit_spread_rate(exposure: Exposure, target_rating: str, spread_shock_bps: float = 0.0) -> float:
    """Return the annualized credit spread for a rating, expressed as a decimal."""

    shocked_spread_bps = max(credit_spread_bps(exposure, target_rating) + spread_shock_bps, 5.0)
    return shocked_spread_bps / 10_000.0


def _effective_lgd(exposure: Exposure, base_lgd: float) -> float:
    lgd = (
        base_lgd
        + ISSUER_LGD_ADJUSTMENT[exposure.issuer_type]
        + INSTRUMENT_SUBTYPE_LGD_ADJUSTMENT[exposure.instrument_subtype]
        + SENIORITY_LGD_ADJUSTMENT[exposure.seniority]
    )
    if exposure.guaranteed:
        lgd *= 0.80
    if exposure.collateral_type:
        lgd *= 0.85
    return min(max(lgd, 0.05), 0.95)


def _coupon_amount(exposure: Exposure, notional: float) -> float:
    return notional * exposure.coupon_rate


def _bullet_market_value(notional: float, coupon_rate: float, maturity_years: float, discount_rate: float) -> float:
    if maturity_years <= 0.0:
        return notional

    coupon = notional * coupon_rate
    whole_years = int(math.floor(maturity_years))
    stub = maturity_years - whole_years

    present_value = 0.0
    for year in range(1, whole_years + 1):
        present_value += coupon / ((1.0 + discount_rate) ** year)

    if stub > 1e-9:
        present_value += coupon * stub / ((1.0 + discount_rate) ** maturity_years)

    present_value += notional / ((1.0 + discount_rate) ** maturity_years)
    return present_value


def _value_from_rating(
    exposure: Exposure,
    target_rating: str,
    effective_balance: float,
    base_lgd: float,
    horizon_years: float,
    risk_free_rate: float,
    spread_shock_bps: float = 0.0,
) -> float:
    if target_rating == "D":
        recovery_value = effective_balance * (1.0 - _effective_lgd(exposure, base_lgd))
        return recovery_value * _discount_factor(risk_free_rate, horizon_years)

    discount_rate = risk_free_rate + credit_spread_rate(exposure, target_rating, spread_shock_bps=spread_shock_bps)
    if horizon_years <= 0.0:
        return _bullet_market_value(
            notional=effective_balance,
            coupon_rate=exposure.coupon_rate,
            maturity_years=exposure.maturity_years,
            discount_rate=discount_rate,
        )

    coupon_horizon_years = min(horizon_years, exposure.maturity_years)
    coupon_received = _coupon_amount(exposure, effective_balance) * coupon_horizon_years

    if exposure.maturity_years <= horizon_years:
        horizon_value = effective_balance + coupon_received
        return horizon_value * _discount_factor(discount_rate, horizon_years)

    remaining_maturity = exposure.maturity_years - horizon_years
    terminal_mark = _bullet_market_value(
        notional=effective_balance,
        coupon_rate=exposure.coupon_rate,
        maturity_years=remaining_maturity,
        discount_rate=discount_rate,
    )
    return (coupon_received + terminal_mark) * _discount_factor(discount_rate, horizon_years)


def value_loan(
    exposure: Exposure,
    target_rating: str,
    base_lgd: float = DEFAULT_LGD,
    horizon_years: float = 0.0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    spread_shock_bps: float = 0.0,
) -> float:
    """Value a loan under a target migrated rating."""

    return _value_from_rating(
        exposure=exposure,
        target_rating=target_rating,
        effective_balance=exposure.balance,
        base_lgd=base_lgd,
        horizon_years=horizon_years,
        risk_free_rate=risk_free_rate,
        spread_shock_bps=spread_shock_bps,
    )


def value_bond(
    exposure: Exposure,
    target_rating: str,
    base_lgd: float = DEFAULT_LGD,
    horizon_years: float = 0.0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    spread_shock_bps: float = 0.0,
) -> float:
    """Value a bond under a target migrated rating."""

    return _value_from_rating(
        exposure=exposure,
        target_rating=target_rating,
        effective_balance=exposure.balance,
        base_lgd=base_lgd,
        horizon_years=horizon_years,
        risk_free_rate=risk_free_rate,
        spread_shock_bps=spread_shock_bps,
    )


def value_off_balance(
    exposure: Exposure,
    target_rating: str,
    base_lgd: float = DEFAULT_LGD,
    ccf: float = DEFAULT_CCF,
    horizon_years: float = 0.0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    spread_shock_bps: float = 0.0,
) -> float:
    """Value an off-balance-sheet exposure using CCF-adjusted effective balance."""

    effective_balance = exposure.balance + ccf * exposure.undrawn
    return _value_from_rating(
        exposure=exposure,
        target_rating=target_rating,
        effective_balance=effective_balance,
        base_lgd=base_lgd,
        horizon_years=horizon_years,
        risk_free_rate=risk_free_rate,
        spread_shock_bps=spread_shock_bps,
    )


def value_exposure(
    exposure: Exposure,
    target_rating: str,
    base_lgd: float = DEFAULT_LGD,
    ccf: float = DEFAULT_CCF,
    horizon_years: float = 0.0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    spread_shock_bps: float = 0.0,
) -> float:
    """Dispatch valuation based on exposure instrument type."""

    if exposure.instrument_type == "loan":
        return value_loan(
            exposure=exposure,
            target_rating=target_rating,
            base_lgd=base_lgd,
            horizon_years=horizon_years,
            risk_free_rate=risk_free_rate,
            spread_shock_bps=spread_shock_bps,
        )
    if exposure.instrument_type == "bond":
        return value_bond(
            exposure=exposure,
            target_rating=target_rating,
            base_lgd=base_lgd,
            horizon_years=horizon_years,
            risk_free_rate=risk_free_rate,
            spread_shock_bps=spread_shock_bps,
        )
    if exposure.instrument_type == "off_balance":
        return value_off_balance(
            exposure=exposure,
            target_rating=target_rating,
            base_lgd=base_lgd,
            ccf=ccf,
            horizon_years=horizon_years,
            risk_free_rate=risk_free_rate,
            spread_shock_bps=spread_shock_bps,
        )
    raise ValueError(f"Unsupported instrument_type: {exposure.instrument_type}")
