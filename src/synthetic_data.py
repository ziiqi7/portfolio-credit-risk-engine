"""Synthetic portfolio generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NON_DEFAULT_RATINGS, RATING_SPREADS, SYNTHETIC_DATA_DIR
from src.schema import Exposure

SECTORS_BY_CLASS = {
    "corporate": ["industrials", "consumer", "energy", "technology", "healthcare", "real_estate"],
    "fi": ["banking", "insurance", "asset_management", "diversified_financials"],
    "sovereign": ["public_sector", "multilateral", "agency"],
}

SECTOR_WEIGHTS = {
    "corporate": [0.23, 0.17, 0.15, 0.18, 0.15, 0.12],
    "fi": [0.42, 0.23, 0.14, 0.21],
    "sovereign": [0.70, 0.18, 0.12],
}

RATING_WEIGHTS = {
    "corporate": [0.02, 0.06, 0.18, 0.33, 0.21, 0.14, 0.06],
    "fi": [0.06, 0.17, 0.30, 0.25, 0.11, 0.08, 0.03],
    "sovereign": [0.15, 0.29, 0.27, 0.16, 0.08, 0.04, 0.01],
}

CLASS_WEIGHTS = {"corporate": 0.62, "fi": 0.23, "sovereign": 0.15}

INSTRUMENT_WEIGHTS_BY_CLASS = {
    "corporate": {"loan": 0.57, "bond": 0.25, "off_balance": 0.18},
    "fi": {"loan": 0.26, "bond": 0.40, "off_balance": 0.34},
    "sovereign": {"loan": 0.14, "bond": 0.80, "off_balance": 0.06},
}

FACILITY_BALANCE_RULES = {
    ("corporate", "loan"): (1_500_000.0, 40_000_000.0, 7_500_000.0),
    ("corporate", "bond"): (3_000_000.0, 65_000_000.0, 14_000_000.0),
    ("corporate", "off_balance"): (1_000_000.0, 25_000_000.0, 5_000_000.0),
    ("fi", "loan"): (4_000_000.0, 55_000_000.0, 12_000_000.0),
    ("fi", "bond"): (8_000_000.0, 95_000_000.0, 24_000_000.0),
    ("fi", "off_balance"): (2_000_000.0, 45_000_000.0, 10_000_000.0),
    ("sovereign", "loan"): (10_000_000.0, 85_000_000.0, 22_000_000.0),
    ("sovereign", "bond"): (15_000_000.0, 140_000_000.0, 40_000_000.0),
    ("sovereign", "off_balance"): (5_000_000.0, 35_000_000.0, 12_000_000.0),
}

MATURITY_RANGES = {
    ("corporate", "loan"): (1.0, 7.0),
    ("corporate", "bond"): (2.0, 10.0),
    ("corporate", "off_balance"): (0.75, 4.0),
    ("fi", "loan"): (1.0, 5.0),
    ("fi", "bond"): (1.5, 8.0),
    ("fi", "off_balance"): (0.5, 3.0),
    ("sovereign", "loan"): (2.0, 12.0),
    ("sovereign", "bond"): (3.0, 15.0),
    ("sovereign", "off_balance"): (0.5, 2.5),
}

GUARANTEE_PROBABILITY = {
    ("corporate", "loan"): 0.16,
    ("corporate", "bond"): 0.03,
    ("corporate", "off_balance"): 0.12,
    ("fi", "loan"): 0.06,
    ("fi", "bond"): 0.02,
    ("fi", "off_balance"): 0.05,
    ("sovereign", "loan"): 0.03,
    ("sovereign", "bond"): 0.00,
    ("sovereign", "off_balance"): 0.01,
}

COLLATERAL_PROBABILITY = {
    ("corporate", "loan"): 0.54,
    ("fi", "loan"): 0.16,
    ("sovereign", "loan"): 0.03,
}

COLLATERAL_TYPES = {
    "corporate": ["cash", "real_estate", "inventory", "receivables", "equipment"],
    "fi": ["cash", "securities"],
    "sovereign": ["cash"],
}

CURRENCY_WEIGHTS = {
    "corporate": [0.60, 0.40],
    "fi": [0.55, 0.45],
    "sovereign": [0.80, 0.20],
}

BASE_RATES = {"EUR": 0.020, "USD": 0.032}


@dataclass(slots=True)
class ObligorProfile:
    """Synthetic obligor profile used to create clustered facilities."""

    obligor_id: str
    exposure_class: str
    sector: str
    rating: str
    currency: str
    facility_count: int
    size_factor: float


def _allocate_facilities(num_exposures: int, num_obligors: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.pareto(a=1.6, size=num_obligors) + 1.0
    weights /= weights.sum()
    counts = np.ones(num_obligors, dtype=int)
    counts += rng.multinomial(num_exposures - num_obligors, weights)
    return counts


def _sample_obligor_profiles(num_exposures: int, rng: np.random.Generator) -> list[ObligorProfile]:
    num_obligors = min(num_exposures, max(28, int(num_exposures * 0.42)))
    facility_counts = _allocate_facilities(num_exposures=num_exposures, num_obligors=num_obligors, rng=rng)
    exposure_classes = list(CLASS_WEIGHTS)

    profiles: list[ObligorProfile] = []
    for idx, facility_count in enumerate(facility_counts):
        exposure_class = str(rng.choice(exposure_classes, p=list(CLASS_WEIGHTS.values())))
        sector = str(
            rng.choice(
                SECTORS_BY_CLASS[exposure_class],
                p=SECTOR_WEIGHTS[exposure_class],
            )
        )
        rating = str(rng.choice(NON_DEFAULT_RATINGS, p=RATING_WEIGHTS[exposure_class]))
        currency = str(rng.choice(["EUR", "USD"], p=CURRENCY_WEIGHTS[exposure_class]))
        size_factor = float(np.clip(rng.lognormal(mean=0.0, sigma=0.55), 0.55, 3.25))
        profiles.append(
            ObligorProfile(
                obligor_id=f"OBL-{idx + 1:04d}",
                exposure_class=exposure_class,
                sector=sector,
                rating=rating,
                currency=currency,
                facility_count=int(facility_count),
                size_factor=size_factor,
            )
        )
    return profiles


def _sample_instrument_type(exposure_class: str, rng: np.random.Generator) -> str:
    weights = INSTRUMENT_WEIGHTS_BY_CLASS[exposure_class]
    return str(rng.choice(list(weights), p=list(weights.values())))


def _sample_maturity(exposure_class: str, instrument_type: str, rng: np.random.Generator) -> float:
    lower, upper = MATURITY_RANGES[(exposure_class, instrument_type)]
    return round(float(rng.uniform(lower, upper)), 2)


def _sample_coupon(
    rating: str,
    instrument_type: str,
    exposure_class: str,
    currency: str,
    rng: np.random.Generator,
) -> float:
    spread = RATING_SPREADS[rating]
    instrument_margin = {"loan": 0.010, "bond": 0.003, "off_balance": -0.004}[instrument_type]
    class_adjustment = {"corporate": 0.001, "fi": 0.0005, "sovereign": -0.002}[exposure_class]
    coupon = BASE_RATES[currency] + spread + instrument_margin + class_adjustment + rng.normal(0.0, 0.0025)
    if instrument_type == "off_balance":
        coupon = min(coupon, 0.085)
    return round(float(np.clip(coupon, 0.006, 0.18)), 4)


def _sample_balance(
    exposure_class: str,
    instrument_type: str,
    obligor_size_factor: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    lower, upper, base_scale = FACILITY_BALANCE_RULES[(exposure_class, instrument_type)]
    raw_balance = base_scale * obligor_size_factor * rng.lognormal(mean=0.0, sigma=0.40)
    balance = float(np.clip(raw_balance, lower, upper))

    if instrument_type == "bond":
        undrawn = 0.0
    elif instrument_type == "loan":
        utilization_ratio = {
            "corporate": rng.uniform(0.02, 0.20),
            "fi": rng.uniform(0.00, 0.12),
            "sovereign": rng.uniform(0.00, 0.05),
        }[exposure_class]
        undrawn = balance * utilization_ratio
    else:
        commitment_ratio = {
            "corporate": rng.uniform(0.30, 0.85),
            "fi": rng.uniform(0.20, 0.70),
            "sovereign": rng.uniform(0.10, 0.40),
        }[exposure_class]
        undrawn = balance * commitment_ratio

    return round(balance, 2), round(float(undrawn), 2)


def _sample_guarantee(exposure_class: str, instrument_type: str, rating: str, rng: np.random.Generator) -> bool:
    probability = GUARANTEE_PROBABILITY[(exposure_class, instrument_type)]
    if rating in {"BB", "B", "CCC"}:
        probability *= 1.15
    return bool(rng.random() < min(probability, 0.30))


def _sample_collateral(exposure_class: str, instrument_type: str, rating: str, rng: np.random.Generator) -> str | None:
    if instrument_type != "loan":
        return None
    probability = COLLATERAL_PROBABILITY[(exposure_class, instrument_type)]
    if rating in {"BB", "B", "CCC"}:
        probability *= 1.10
    if rng.random() >= min(probability, 0.75):
        return None
    return str(rng.choice(COLLATERAL_TYPES[exposure_class]))


def _sample_rate_type(instrument_type: str, rng: np.random.Generator) -> str:
    if instrument_type == "bond":
        return str(rng.choice(["fixed", "floating"], p=[0.88, 0.12]))
    if instrument_type == "loan":
        return str(rng.choice(["fixed", "floating"], p=[0.22, 0.78]))
    return "floating"


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


def generate_synthetic_portfolio(num_exposures: int = 150, seed: int = 42) -> list[Exposure]:
    """Generate a synthetic mixed-instrument credit portfolio."""

    if num_exposures <= 0:
        raise ValueError("num_exposures must be positive")

    rng = np.random.default_rng(seed)
    portfolio: list[Exposure] = []
    exposure_index = 1

    for obligor in _sample_obligor_profiles(num_exposures=num_exposures, rng=rng):
        for _ in range(obligor.facility_count):
            instrument_type = _sample_instrument_type(obligor.exposure_class, rng)
            balance, undrawn = _sample_balance(obligor.exposure_class, instrument_type, obligor.size_factor, rng)
            maturity_years = _sample_maturity(obligor.exposure_class, instrument_type, rng)
            guaranteed = _sample_guarantee(obligor.exposure_class, instrument_type, obligor.rating, rng)
            collateral_type = _sample_collateral(obligor.exposure_class, instrument_type, obligor.rating, rng)
            portfolio.append(
                Exposure(
                    exposure_id=f"EXP-{exposure_index:04d}",
                    obligor_id=obligor.obligor_id,
                    instrument_type=instrument_type,
                    exposure_class=obligor.exposure_class,
                    sector=obligor.sector,
                    rating=obligor.rating,
                    currency=obligor.currency,
                    balance=balance,
                    undrawn=undrawn,
                    maturity_years=maturity_years,
                    coupon_rate=_sample_coupon(
                        obligor.rating,
                        instrument_type,
                        obligor.exposure_class,
                        obligor.currency,
                        rng,
                    ),
                    rate_type=_sample_rate_type(instrument_type, rng),
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
