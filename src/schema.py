"""Unified schema for synthetic credit portfolio exposures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isnan
from typing import Any, Optional

from src.config import (
    CURRENCIES,
    EXPOSURE_CLASSES,
    INSTRUMENT_SUBTYPES,
    INSTRUMENT_TYPES,
    ISSUER_TYPES,
    NON_DEFAULT_RATINGS,
    RATE_TYPES,
    SENIORITIES,
)


@dataclass(slots=True)
class Exposure:
    """Unified representation of a portfolio exposure."""

    exposure_id: str
    obligor_id: str
    instrument_type: str
    exposure_class: str
    issuer_type: str
    instrument_subtype: str
    seniority: str
    secured_flag: bool
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
        self.issuer_type = self.issuer_type.lower()
        self.instrument_subtype = self.instrument_subtype.lower()
        self.seniority = self.seniority.lower()
        self.rating = self.rating.upper()
        self.currency = self.currency.upper()
        self.rate_type = self.rate_type.lower()
        self.balance = float(self.balance)
        self.undrawn = float(self.undrawn)
        self.maturity_years = float(self.maturity_years)
        self.coupon_rate = float(self.coupon_rate)
        if isinstance(self.secured_flag, str):
            secured_text = self.secured_flag.strip().lower()
            if secured_text in {"true", "1", "yes"}:
                self.secured_flag = True
            elif secured_text in {"false", "0", "no", ""}:
                self.secured_flag = False
            else:
                raise ValueError(f"Unsupported secured_flag: {self.secured_flag}")
        else:
            self.secured_flag = bool(self.secured_flag)
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
        if self.issuer_type not in ISSUER_TYPES:
            raise ValueError(f"Unsupported issuer_type: {self.issuer_type}")
        if self.instrument_subtype not in INSTRUMENT_SUBTYPES:
            raise ValueError(f"Unsupported instrument_subtype: {self.instrument_subtype}")
        if self.seniority not in SENIORITIES:
            raise ValueError(f"Unsupported seniority: {self.seniority}")
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
        if self.instrument_type == "loan" and self.instrument_subtype not in {"term_loan", "revolving"}:
            raise ValueError("loan exposures must use loan instrument_subtype values")
        if self.instrument_type == "bond" and self.instrument_subtype not in {
            "sovereign_bond",
            "agency_bond",
            "supranational_bond",
            "bank_senior_bond",
            "covered_bond",
            "corporate_bond",
        }:
            raise ValueError("bond exposures must use bond instrument_subtype values")
        if self.instrument_type == "off_balance" and self.instrument_subtype not in {"guarantee", "letter_of_credit"}:
            raise ValueError("off_balance exposures must use contingent instrument_subtype values")
        if self.seniority == "covered" and self.instrument_subtype != "covered_bond":
            raise ValueError("covered seniority is reserved for covered bonds")
        if self.secured_flag and self.seniority not in {"senior_secured", "covered"}:
            raise ValueError("secured_flag requires secured or covered seniority")
        if not self.secured_flag and self.seniority in {"senior_secured", "covered"}:
            raise ValueError("senior_secured or covered exposures must set secured_flag")
        if self.issuer_type == "sovereign" and self.exposure_class != "sovereign":
            raise ValueError("sovereign issuer_type must map to sovereign exposure_class")
        if self.issuer_type in {"supranational", "agency"} and self.exposure_class != "sovereign":
            raise ValueError("public-sector issuer_type must map to sovereign exposure_class")
        if self.issuer_type in {"bank", "insurance"} and self.exposure_class != "fi":
            raise ValueError("financial issuer_type must map to fi exposure_class")
        if self.issuer_type == "corporate" and self.exposure_class != "corporate":
            raise ValueError("corporate issuer_type must map to corporate exposure_class")
        if self.instrument_subtype == "sovereign_bond" and self.issuer_type != "sovereign":
            raise ValueError("sovereign_bond requires sovereign issuer_type")
        if self.instrument_subtype == "agency_bond" and self.issuer_type != "agency":
            raise ValueError("agency_bond requires agency issuer_type")
        if self.instrument_subtype == "supranational_bond" and self.issuer_type != "supranational":
            raise ValueError("supranational_bond requires supranational issuer_type")
        if self.instrument_subtype == "bank_senior_bond" and self.issuer_type not in {"bank", "insurance"}:
            raise ValueError("bank_senior_bond requires bank or insurance issuer_type")
        if self.instrument_subtype == "covered_bond" and self.issuer_type != "bank":
            raise ValueError("covered_bond requires bank issuer_type")
        if self.instrument_subtype == "corporate_bond" and self.issuer_type != "corporate":
            raise ValueError("corporate_bond requires corporate issuer_type")

    def to_dict(self) -> dict[str, Any]:
        """Convert exposure to a serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Exposure":
        """Build an exposure from a dictionary."""

        return cls(**data)
