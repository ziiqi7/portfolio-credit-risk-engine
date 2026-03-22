"""Unified schema for synthetic credit portfolio exposures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isnan
from typing import Any, Optional

from src.config import CURRENCIES, EXPOSURE_CLASSES, INSTRUMENT_TYPES, NON_DEFAULT_RATINGS, RATE_TYPES


@dataclass(slots=True)
class Exposure:
    """Unified representation of a portfolio exposure."""

    exposure_id: str
    obligor_id: str
    instrument_type: str
    exposure_class: str
    sector: str
    rating: str
    currency: str
    balance: float
    undrawn: float
    maturity_years: float
    coupon_rate: float
    rate_type: str
    guaranteed: bool
    collateral_type: Optional[str] = None

    def __post_init__(self) -> None:
        self.instrument_type = self.instrument_type.lower()
        self.exposure_class = self.exposure_class.lower()
        self.rating = self.rating.upper()
        self.currency = self.currency.upper()
        self.rate_type = self.rate_type.lower()
        self.balance = float(self.balance)
        self.undrawn = float(self.undrawn)
        self.maturity_years = float(self.maturity_years)
        self.coupon_rate = float(self.coupon_rate)
        if isinstance(self.guaranteed, str):
            guaranteed_text = self.guaranteed.strip().lower()
            if guaranteed_text in {"true", "1", "yes"}:
                self.guaranteed = True
            elif guaranteed_text in {"false", "0", "no", ""}:
                self.guaranteed = False
            else:
                raise ValueError(f"Unsupported guaranteed flag: {self.guaranteed}")
        else:
            self.guaranteed = bool(self.guaranteed)

        if self.collateral_type is None:
            self.collateral_type = None
        elif isinstance(self.collateral_type, float) and isnan(self.collateral_type):
            self.collateral_type = None
        else:
            collateral_text = str(self.collateral_type).strip()
            self.collateral_type = collateral_text or None

        if self.instrument_type not in INSTRUMENT_TYPES:
            raise ValueError(f"Unsupported instrument_type: {self.instrument_type}")
        if self.exposure_class not in EXPOSURE_CLASSES:
            raise ValueError(f"Unsupported exposure_class: {self.exposure_class}")
        if self.rating not in NON_DEFAULT_RATINGS:
            raise ValueError(f"Unsupported starting rating: {self.rating}")
        if self.currency not in CURRENCIES:
            raise ValueError(f"Unsupported currency: {self.currency}")
        if self.rate_type not in RATE_TYPES:
            raise ValueError(f"Unsupported rate_type: {self.rate_type}")
        if self.balance <= 0.0:
            raise ValueError("balance must be positive")
        if self.undrawn < 0.0:
            raise ValueError("undrawn must be non-negative")
        if self.maturity_years <= 0.0:
            raise ValueError("maturity_years must be positive")
        if self.coupon_rate < 0.0:
            raise ValueError("coupon_rate must be non-negative")
        if self.coupon_rate > 0.25:
            raise ValueError("coupon_rate must be below 25%")
        if self.instrument_type == "bond" and self.undrawn != 0.0:
            raise ValueError("bond exposures must have zero undrawn amount")
        if self.instrument_type == "bond" and self.collateral_type is not None:
            raise ValueError("bond exposures should not carry loan-style collateral_type")
        if self.instrument_type == "off_balance" and self.undrawn <= 0.0:
            raise ValueError("off_balance exposures must have positive undrawn amount")
        if self.instrument_type != "loan" and self.collateral_type is not None:
            raise ValueError("collateral_type is only supported for loan exposures")

    def to_dict(self) -> dict[str, Any]:
        """Convert exposure to a serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Exposure":
        """Build an exposure from a dictionary."""

        return cls(**data)
