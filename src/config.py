"""Project-wide configuration and default assumptions."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
TRANSITIONS_DIR = DATA_DIR / "transitions"
OUTPUTS_DIR = BASE_DIR / "outputs"

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
NON_DEFAULT_RATINGS = RATINGS[:-1]
INSTRUMENT_TYPES = ("loan", "bond", "off_balance")
EXPOSURE_CLASSES = ("corporate", "fi", "sovereign")
CURRENCIES = ("EUR", "USD")
RATE_TYPES = ("fixed", "floating")
ISSUER_TYPES = ("sovereign", "supranational", "agency", "bank", "insurance", "corporate")
INSTRUMENT_SUBTYPES = (
    "term_loan",
    "revolving",
    "sovereign_bond",
    "agency_bond",
    "supranational_bond",
    "bank_senior_bond",
    "covered_bond",
    "corporate_bond",
    "guarantee",
    "letter_of_credit",
)
SENIORITIES = ("senior_secured", "senior_unsecured", "subordinated", "covered")

DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_LGD = 0.45
DEFAULT_CCF = 0.5
DEFAULT_HORIZON_YEARS = 1.0

RATING_SPREAD_BPS = {
    "AAA": 45.0,
    "AA": 65.0,
    "A": 95.0,
    "BBB": 150.0,
    "BB": 285.0,
    "B": 470.0,
    "CCC": 760.0,
    "D": 10_000.0,
}

RATING_SPREADS = {rating: spread_bps / 10_000.0 for rating, spread_bps in RATING_SPREAD_BPS.items()}

ISSUER_SPREAD_BPS_ADJUSTMENT = {
    "sovereign": -20.0,
    "supranational": -30.0,
    "agency": -18.0,
    "bank": 18.0,
    "insurance": 8.0,
    "corporate": 32.0,
}

INSTRUMENT_SUBTYPE_SPREAD_BPS_ADJUSTMENT = {
    "term_loan": 15.0,
    "revolving": 28.0,
    "sovereign_bond": -12.0,
    "agency_bond": -10.0,
    "supranational_bond": -15.0,
    "bank_senior_bond": 12.0,
    "covered_bond": -28.0,
    "corporate_bond": 20.0,
    "guarantee": 18.0,
    "letter_of_credit": 10.0,
}

SENIORITY_SPREAD_BPS_ADJUSTMENT = {
    "senior_secured": -18.0,
    "senior_unsecured": 0.0,
    "subordinated": 70.0,
    "covered": -35.0,
}

ISSUER_SPREAD_SHOCK_MACRO_BETA_BPS = {
    "sovereign": 16.0,
    "supranational": 12.0,
    "agency": 18.0,
    "bank": 38.0,
    "insurance": 34.0,
    "corporate": 48.0,
}

INSTRUMENT_SUBTYPE_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS = {
    "term_loan": 4.0,
    "revolving": 6.0,
    "sovereign_bond": -4.0,
    "agency_bond": -3.0,
    "supranational_bond": -5.0,
    "bank_senior_bond": 6.0,
    "covered_bond": -10.0,
    "corporate_bond": 10.0,
    "guarantee": 5.0,
    "letter_of_credit": 4.0,
}

SENIORITY_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS = {
    "senior_secured": -6.0,
    "senior_unsecured": 0.0,
    "subordinated": 18.0,
    "covered": -12.0,
}

RATING_SPREAD_SHOCK_MACRO_ADJUSTMENT_BPS = {
    "AAA": -8.0,
    "AA": -5.0,
    "A": 0.0,
    "BBB": 8.0,
    "BB": 18.0,
    "B": 28.0,
    "CCC": 40.0,
    "D": 0.0,
}

SECTOR_SPREAD_SHOCK_BETA_BPS = {
    "banking": 22.0,
    "insurance": 18.0,
    "industrials": 20.0,
    "energy": 24.0,
    "consumer": 18.0,
    "real_estate": 26.0,
    "technology": 17.0,
    "healthcare": 15.0,
    "public_sector": 8.0,
    "agency": 10.0,
    "multilateral": 7.0,
    "diversified_financials": 21.0,
    "asset_management": 19.0,
}

RATING_SPREAD_SHOCK_SECTOR_ADJUSTMENT_BPS = {
    "AAA": -4.0,
    "AA": -2.0,
    "A": 0.0,
    "BBB": 4.0,
    "BB": 8.0,
    "B": 12.0,
    "CCC": 18.0,
    "D": 0.0,
}

ISSUER_LGD_ADJUSTMENT = {
    "sovereign": -0.20,
    "supranational": -0.24,
    "agency": -0.17,
    "bank": -0.05,
    "insurance": -0.02,
    "corporate": 0.03,
}

INSTRUMENT_SUBTYPE_LGD_ADJUSTMENT = {
    "term_loan": -0.02,
    "revolving": 0.02,
    "sovereign_bond": -0.10,
    "agency_bond": -0.08,
    "supranational_bond": -0.12,
    "bank_senior_bond": 0.02,
    "covered_bond": -0.14,
    "corporate_bond": 0.05,
    "guarantee": -0.04,
    "letter_of_credit": -0.02,
}

SENIORITY_LGD_ADJUSTMENT = {
    "senior_secured": -0.12,
    "senior_unsecured": 0.0,
    "subordinated": 0.14,
    "covered": -0.18,
}

BASE_RATES_BY_CURRENCY = {"EUR": 0.020, "USD": 0.032}

RATING_BUCKET_MAP = {
    "AAA": "AAA-A",
    "AA": "AAA-A",
    "A": "AAA-A",
    "BBB": "BBB",
    "BB": "BB-B",
    "B": "BB-B",
    "CCC": "CCC/D",
    "D": "CCC/D",
}

TRANSITION_FILES = {
    "corporate": TRANSITIONS_DIR / "transition_matrix_corporate.csv",
    "fi": TRANSITIONS_DIR / "transition_matrix_fi.csv",
    "sovereign": TRANSITIONS_DIR / "transition_matrix_sovereign.csv",
}
