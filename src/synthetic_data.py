"""Synthetic portfolio generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BASE_RATES_BY_CURRENCY,
    CURRENCIES,
    INSTRUMENT_SUBTYPE_SPREAD_BPS_ADJUSTMENT,
    ISSUER_SPREAD_BPS_ADJUSTMENT,
    NON_DEFAULT_RATINGS,
    RATING_SPREAD_BPS,
    SENIORITY_SPREAD_BPS_ADJUSTMENT,
    SYNTHETIC_DATA_DIR,
)
from src.schema import Exposure

ISSUER_TYPE_WEIGHTS = {
    "corporate": 0.60,
    "bank": 0.14,
    "insurance": 0.08,
    "sovereign": 0.08,
    "agency": 0.06,
    "supranational": 0.04,
}

EXPOSURE_CLASS_BY_ISSUER = {
    "sovereign": "sovereign",
    "supranational": "sovereign",
    "agency": "sovereign",
    "bank": "fi",
    "insurance": "fi",
    "corporate": "corporate",
}

SECTORS_BY_ISSUER = {
    "corporate": ["industrials", "consumer", "energy", "technology", "healthcare", "real_estate"],
    "bank": ["banking", "diversified_financials"],
    "insurance": ["insurance", "asset_management"],
    "sovereign": ["public_sector"],
    "agency": ["agency"],
    "supranational": ["multilateral"],
}

SECTOR_WEIGHTS_BY_ISSUER = {
    "corporate": [0.23, 0.17, 0.15, 0.18, 0.15, 0.12],
    "bank": [0.72, 0.28],
    "insurance": [0.72, 0.28],
    "sovereign": [1.0],
    "agency": [1.0],
    "supranational": [1.0],
}

RATING_WEIGHTS_BY_ISSUER = {
    "corporate": [0.02, 0.06, 0.18, 0.33, 0.21, 0.14, 0.06],
    "bank": [0.06, 0.18, 0.29, 0.24, 0.11, 0.08, 0.04],
    "insurance": [0.08, 0.20, 0.30, 0.22, 0.10, 0.07, 0.03],
    "sovereign": [0.31, 0.29, 0.20, 0.12, 0.05, 0.02, 0.01],
    "agency": [0.22, 0.30, 0.24, 0.14, 0.06, 0.03, 0.01],
    "supranational": [0.39, 0.31, 0.18, 0.08, 0.02, 0.01, 0.01],
}

CURRENCY_WEIGHTS_BY_ISSUER = {
    "corporate": [0.58, 0.42],
    "bank": [0.50, 0.50],
    "insurance": [0.58, 0.42],
    "sovereign": [0.84, 0.16],
    "agency": [0.78, 0.22],
    "supranational": [0.72, 0.28],
}

INSTRUMENT_TYPE_WEIGHTS_BY_ISSUER = {
    "corporate": {"loan": 0.49, "bond": 0.28, "off_balance": 0.23},
    "bank": {"loan": 0.18, "bond": 0.56, "off_balance": 0.26},
    "insurance": {"loan": 0.12, "bond": 0.72, "off_balance": 0.16},
    "sovereign": {"loan": 0.03, "bond": 0.94, "off_balance": 0.03},
    "agency": {"loan": 0.04, "bond": 0.90, "off_balance": 0.06},
    "supranational": {"loan": 0.06, "bond": 0.88, "off_balance": 0.06},
}

SUBTYPE_WEIGHTS = {
    ("corporate", "loan"): {"term_loan": 0.68, "revolving": 0.32},
    ("bank", "loan"): {"term_loan": 0.55, "revolving": 0.45},
    ("insurance", "loan"): {"term_loan": 0.80, "revolving": 0.20},
    ("sovereign", "loan"): {"term_loan": 1.00},
    ("agency", "loan"): {"term_loan": 1.00},
    ("supranational", "loan"): {"term_loan": 1.00},
    ("corporate", "bond"): {"corporate_bond": 1.00},
    ("bank", "bond"): {"bank_senior_bond": 0.73, "covered_bond": 0.27},
    ("insurance", "bond"): {"bank_senior_bond": 1.00},
    ("sovereign", "bond"): {"sovereign_bond": 1.00},
    ("agency", "bond"): {"agency_bond": 1.00},
    ("supranational", "bond"): {"supranational_bond": 1.00},
    ("corporate", "off_balance"): {"guarantee": 0.30, "letter_of_credit": 0.70},
    ("bank", "off_balance"): {"guarantee": 0.38, "letter_of_credit": 0.62},
    ("insurance", "off_balance"): {"guarantee": 0.62, "letter_of_credit": 0.38},
    ("sovereign", "off_balance"): {"guarantee": 1.00},
    ("agency", "off_balance"): {"guarantee": 0.75, "letter_of_credit": 0.25},
    ("supranational", "off_balance"): {"guarantee": 0.80, "letter_of_credit": 0.20},
}

FACILITY_RULES = {
    "term_loan": {"lower": 2_000_000.0, "upper": 45_000_000.0, "scale": 10_000_000.0, "maturity": (1.0, 7.0)},
    "revolving": {"lower": 1_500_000.0, "upper": 35_000_000.0, "scale": 8_000_000.0, "maturity": (0.75, 5.0)},
    "corporate_bond": {"lower": 5_000_000.0, "upper": 85_000_000.0, "scale": 18_000_000.0, "maturity": (2.0, 10.0)},
    "bank_senior_bond": {"lower": 8_000_000.0, "upper": 110_000_000.0, "scale": 26_000_000.0, "maturity": (2.0, 9.0)},
    "covered_bond": {"lower": 12_000_000.0, "upper": 120_000_000.0, "scale": 30_000_000.0, "maturity": (3.0, 8.0)},
    "sovereign_bond": {"lower": 15_000_000.0, "upper": 180_000_000.0, "scale": 48_000_000.0, "maturity": (3.0, 15.0)},
    "agency_bond": {"lower": 12_000_000.0, "upper": 145_000_000.0, "scale": 34_000_000.0, "maturity": (2.0, 12.0)},
    "supranational_bond": {"lower": 12_000_000.0, "upper": 135_000_000.0, "scale": 32_000_000.0, "maturity": (2.0, 12.0)},
    "guarantee": {"lower": 1_000_000.0, "upper": 28_000_000.0, "scale": 6_000_000.0, "maturity": (0.5, 4.0)},
    "letter_of_credit": {"lower": 750_000.0, "upper": 22_000_000.0, "scale": 4_500_000.0, "maturity": (0.5, 3.0)},
}

SENIORITY_WEIGHTS_BY_SUBTYPE = {
    "term_loan": {"senior_secured": 0.72, "senior_unsecured": 0.28},
    "revolving": {"senior_secured": 0.45, "senior_unsecured": 0.55},
    "corporate_bond": {"senior_unsecured": 0.86, "subordinated": 0.14},
    "bank_senior_bond": {"senior_unsecured": 0.90, "subordinated": 0.10},
    "covered_bond": {"covered": 1.00},
    "sovereign_bond": {"senior_unsecured": 1.00},
    "agency_bond": {"senior_unsecured": 1.00},
    "supranational_bond": {"senior_unsecured": 1.00},
    "guarantee": {"senior_unsecured": 1.00},
    "letter_of_credit": {"senior_unsecured": 1.00},
}

COLLATERAL_TYPES_BY_ISSUER = {
    "corporate": ["cash", "real_estate", "inventory", "receivables", "equipment"],
    "bank": ["cash", "securities", "mortgage_pool"],
    "insurance": ["cash", "securities"],
    "sovereign": ["cash"],
    "agency": ["cash"],
    "supranational": ["cash"],
}


@dataclass(slots=True)
class ObligorProfile:
    """Synthetic obligor profile used to create clustered facilities."""

    obligor_id: str
    exposure_class: str
    issuer_type: str
    sector: str
    rating: str
    currency: str
    facility_count: int
    size_factor: float


def _allocate_facilities(num_exposures: int, num_obligors: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.pareto(a=1.55, size=num_obligors) + 1.0
    weights /= weights.sum()
    counts = np.ones(num_obligors, dtype=int)
    counts += rng.multinomial(num_exposures - num_obligors, weights)
    return counts


def _sample_bool(value: object) -> bool:
    return bool(value)


def _sample_obligor_profiles(num_exposures: int, rng: np.random.Generator) -> list[ObligorProfile]:
    num_obligors = min(num_exposures, max(32, int(num_exposures * 0.42)))
    facility_counts = _allocate_facilities(num_exposures=num_exposures, num_obligors=num_obligors, rng=rng)
    issuer_types = list(ISSUER_TYPE_WEIGHTS)

    profiles: list[ObligorProfile] = []
    for idx, facility_count in enumerate(facility_counts):
        issuer_type = str(rng.choice(issuer_types, p=list(ISSUER_TYPE_WEIGHTS.values())))
        sector = str(rng.choice(SECTORS_BY_ISSUER[issuer_type], p=SECTOR_WEIGHTS_BY_ISSUER[issuer_type]))
        rating = str(rng.choice(NON_DEFAULT_RATINGS, p=RATING_WEIGHTS_BY_ISSUER[issuer_type]))
        currency = str(rng.choice(CURRENCIES, p=CURRENCY_WEIGHTS_BY_ISSUER[issuer_type]))
        size_factor = float(np.clip(rng.lognormal(mean=0.0, sigma=0.52), 0.55, 3.40))
        profiles.append(
            ObligorProfile(
                obligor_id=f"OBL-{idx + 1:04d}",
                exposure_class=EXPOSURE_CLASS_BY_ISSUER[issuer_type],
                issuer_type=issuer_type,
                sector=sector,
                rating=rating,
                currency=currency,
                facility_count=int(facility_count),
                size_factor=size_factor,
            )
        )
    return profiles


def _sample_instrument_type(issuer_type: str, rng: np.random.Generator) -> str:
    weights = INSTRUMENT_TYPE_WEIGHTS_BY_ISSUER[issuer_type]
    return str(rng.choice(list(weights), p=list(weights.values())))


def _sample_instrument_subtype(issuer_type: str, instrument_type: str, rng: np.random.Generator) -> str:
    weights = SUBTYPE_WEIGHTS[(issuer_type, instrument_type)]
    return str(rng.choice(list(weights), p=list(weights.values())))


def _sample_seniority(instrument_subtype: str, rng: np.random.Generator) -> tuple[str, bool]:
    weights = SENIORITY_WEIGHTS_BY_SUBTYPE[instrument_subtype]
    seniority = str(rng.choice(list(weights), p=list(weights.values())))
    secured_flag = seniority in {"senior_secured", "covered"}
    return seniority, secured_flag


def _sample_rate_type(instrument_subtype: str, rng: np.random.Generator) -> str:
    if instrument_subtype in {"term_loan", "revolving", "guarantee", "letter_of_credit"}:
        return "floating"
    if instrument_subtype in {"covered_bond", "sovereign_bond", "agency_bond", "supranational_bond"}:
        return str(rng.choice(["fixed", "floating"], p=[0.92, 0.08]))
    return str(rng.choice(["fixed", "floating"], p=[0.86, 0.14]))


def _sample_maturity(instrument_subtype: str, rng: np.random.Generator) -> float:
    lower, upper = FACILITY_RULES[instrument_subtype]["maturity"]
    return round(float(rng.uniform(lower, upper)), 2)


def _spread_bps_for_coupon(rating: str, issuer_type: str, instrument_subtype: str, seniority: str) -> float:
    spread_bps = (
        RATING_SPREAD_BPS[rating]
        + ISSUER_SPREAD_BPS_ADJUSTMENT[issuer_type]
        + INSTRUMENT_SUBTYPE_SPREAD_BPS_ADJUSTMENT[instrument_subtype]
        + SENIORITY_SPREAD_BPS_ADJUSTMENT[seniority]
    )
    return float(max(spread_bps, 15.0))


def _sample_coupon(
    rating: str,
    issuer_type: str,
    instrument_subtype: str,
    seniority: str,
    currency: str,
    rate_type: str,
    rng: np.random.Generator,
) -> float:
    base_rate = BASE_RATES_BY_CURRENCY[currency]
    spread_rate = _spread_bps_for_coupon(rating, issuer_type, instrument_subtype, seniority) / 10_000.0
    floating_adjustment = 0.002 if rate_type == "floating" else 0.0
    coupon = base_rate + spread_rate + floating_adjustment + rng.normal(0.0, 0.0018)
    if instrument_subtype in {"guarantee", "letter_of_credit"}:
        coupon *= 0.72
    return round(float(np.clip(coupon, 0.006, 0.18)), 4)


def _sample_balance(
    instrument_subtype: str,
    issuer_type: str,
    obligor_size_factor: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    rule = FACILITY_RULES[instrument_subtype]
    raw_balance = rule["scale"] * obligor_size_factor * rng.lognormal(mean=0.0, sigma=0.40)
    balance = float(np.clip(raw_balance, rule["lower"], rule["upper"]))

    if instrument_subtype in {
        "sovereign_bond",
        "agency_bond",
        "supranational_bond",
        "bank_senior_bond",
        "covered_bond",
        "corporate_bond",
    }:
        undrawn = 0.0
    elif instrument_subtype == "term_loan":
        undrawn = balance * rng.uniform(0.00, 0.10 if issuer_type in {"sovereign", "agency", "supranational"} else 0.18)
    elif instrument_subtype == "revolving":
        utilization_ratio = {
            "corporate": rng.uniform(0.18, 0.52),
            "bank": rng.uniform(0.15, 0.40),
            "insurance": rng.uniform(0.08, 0.22),
            "sovereign": rng.uniform(0.02, 0.10),
            "agency": rng.uniform(0.03, 0.12),
            "supranational": rng.uniform(0.03, 0.10),
        }[issuer_type]
        undrawn = balance * (1.0 - utilization_ratio)
    else:
        commitment_ratio = {
            "corporate": rng.uniform(0.18, 0.62),
            "bank": rng.uniform(0.15, 0.55),
            "insurance": rng.uniform(0.10, 0.42),
            "sovereign": rng.uniform(0.06, 0.22),
            "agency": rng.uniform(0.08, 0.28),
            "supranational": rng.uniform(0.08, 0.25),
        }[issuer_type]
        undrawn = balance * commitment_ratio

    return round(balance, 2), round(float(undrawn), 2)


def _sample_guaranteed(issuer_type: str, instrument_subtype: str, rating: str, rng: np.random.Generator) -> bool:
    if issuer_type in {"sovereign", "supranational", "agency"}:
        return False
    probability = {
        "term_loan": 0.15,
        "revolving": 0.10,
        "corporate_bond": 0.02,
        "bank_senior_bond": 0.01,
        "covered_bond": 0.0,
        "guarantee": 0.08,
        "letter_of_credit": 0.06,
    }.get(instrument_subtype, 0.0)
    if rating in {"BB", "B", "CCC"}:
        probability *= 1.15
    return _sample_bool(rng.random() < min(probability, 0.30))


def _sample_collateral(
    issuer_type: str,
    instrument_subtype: str,
    seniority: str,
    rating: str,
    rng: np.random.Generator,
) -> str | None:
    if instrument_subtype not in {"term_loan", "revolving"}:
        return None
    probability = 0.62 if seniority == "senior_secured" else 0.12
    if issuer_type in {"bank", "insurance"}:
        probability *= 0.70
    if rating in {"BB", "B", "CCC"}:
        probability *= 1.10
    if rng.random() >= min(probability, 0.80):
        return None
    return str(rng.choice(COLLATERAL_TYPES_BY_ISSUER[issuer_type]))


def validate_portfolio(portfolio: list[Exposure]) -> None:
    """Run portfolio-level sanity checks for generated synthetic exposures."""

    if not portfolio:
        raise ValueError("Synthetic portfolio generation produced no exposures.")

    exposure_ids = {exposure.exposure_id for exposure in portfolio}
    if len(exposure_ids) != len(portfolio):
        raise ValueError("Synthetic portfolio contains duplicate exposure_id values.")

    for exposure in portfolio:
        if exposure.instrument_type == "bond" and exposure.undrawn != 0.0:
            raise ValueError(f"{exposure.exposure_id}: bond exposure has non-zero undrawn amount.")
        if exposure.instrument_type == "off_balance" and exposure.undrawn > exposure.balance:
            raise ValueError(f"{exposure.exposure_id}: off_balance undrawn exceeds balance sanity limit.")
        if exposure.instrument_type == "off_balance" and exposure.rate_type != "floating":
            raise ValueError(f"{exposure.exposure_id}: off_balance exposure should use floating rate_type.")
        if exposure.instrument_type != "loan" and exposure.collateral_type is not None:
            raise ValueError(f"{exposure.exposure_id}: collateral_type is only valid for loans.")
        if exposure.instrument_type == "bond" and exposure.maturity_years < 1.0:
            raise ValueError(f"{exposure.exposure_id}: bond maturity must be at least one year.")
        if exposure.instrument_type == "off_balance" and exposure.maturity_years > 5.0:
            raise ValueError(f"{exposure.exposure_id}: off_balance maturity exceeds sanity limit.")
        if exposure.issuer_type in {"sovereign", "agency", "supranational"} and exposure.instrument_type == "bond":
            if exposure.rating not in {"AAA", "AA", "A", "BBB", "BB", "B", "CCC"}:
                raise ValueError(f"{exposure.exposure_id}: invalid public-sector rating.")


def generate_synthetic_portfolio(num_exposures: int = 150, seed: int = 42) -> list[Exposure]:
    """Generate a synthetic mixed-instrument credit portfolio."""

    if num_exposures <= 0:
        raise ValueError("num_exposures must be positive")

    rng = np.random.default_rng(seed)
    portfolio: list[Exposure] = []
    exposure_index = 1

    for obligor in _sample_obligor_profiles(num_exposures=num_exposures, rng=rng):
        for _ in range(obligor.facility_count):
            instrument_type = _sample_instrument_type(obligor.issuer_type, rng)
            instrument_subtype = _sample_instrument_subtype(obligor.issuer_type, instrument_type, rng)
            seniority, secured_flag = _sample_seniority(instrument_subtype, rng)
            rate_type = _sample_rate_type(instrument_subtype, rng)
            balance, undrawn = _sample_balance(instrument_subtype, obligor.issuer_type, obligor.size_factor, rng)
            maturity_years = _sample_maturity(instrument_subtype, rng)
            guaranteed = _sample_guaranteed(obligor.issuer_type, instrument_subtype, obligor.rating, rng)
            collateral_type = _sample_collateral(obligor.issuer_type, instrument_subtype, seniority, obligor.rating, rng)

            portfolio.append(
                Exposure(
                    exposure_id=f"EXP-{exposure_index:04d}",
                    obligor_id=obligor.obligor_id,
                    instrument_type=instrument_type,
                    exposure_class=obligor.exposure_class,
                    issuer_type=obligor.issuer_type,
                    instrument_subtype=instrument_subtype,
                    seniority=seniority,
                    secured_flag=secured_flag,
                    sector=obligor.sector,
                    rating=obligor.rating,
                    currency=obligor.currency,
                    balance=balance,
                    undrawn=undrawn,
                    maturity_years=maturity_years,
                    coupon_rate=_sample_coupon(
                        obligor.rating,
                        obligor.issuer_type,
                        instrument_subtype,
                        seniority,
                        obligor.currency,
                        rate_type,
                        rng,
                    ),
                    rate_type=rate_type,
                    guaranteed=guaranteed,
                    collateral_type=collateral_type,
                )
            )
            exposure_index += 1

    portfolio = portfolio[:num_exposures]
    validate_portfolio(portfolio)
    return portfolio


def portfolio_to_dataframe(portfolio: list[Exposure]) -> pd.DataFrame:
    """Convert exposure objects into a tabular DataFrame."""

    return pd.DataFrame([exposure.to_dict() for exposure in portfolio])


def dataframe_to_portfolio(dataframe: pd.DataFrame) -> list[Exposure]:
    """Convert a portfolio DataFrame back into exposure objects."""

    records = dataframe.to_dict(orient="records")
    return [Exposure.from_dict(record) for record in records]


def save_synthetic_portfolio(
    output_path: Path | str | None = None,
    num_exposures: int = 150,
    seed: int = 42,
) -> Path:
    """Generate and save a synthetic portfolio CSV."""

    target_path = Path(output_path) if output_path is not None else SYNTHETIC_DATA_DIR / "sample_portfolio.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = portfolio_to_dataframe(generate_synthetic_portfolio(num_exposures=num_exposures, seed=seed))
    dataframe.to_csv(target_path, index=False)
    return target_path


def load_synthetic_portfolio(path: Path | str) -> list[Exposure]:
    """Load a saved synthetic portfolio CSV."""

    dataframe = pd.read_csv(path)
    return dataframe_to_portfolio(dataframe)
