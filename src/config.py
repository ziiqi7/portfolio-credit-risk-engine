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

DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_LGD = 0.45
DEFAULT_CCF = 0.5
DEFAULT_HORIZON_YEARS = 1.0

RATING_SPREADS = {
    "AAA": 0.0040,
    "AA": 0.0060,
    "A": 0.0090,
    "BBB": 0.0150,
    "BB": 0.0300,
    "B": 0.0500,
    "CCC": 0.0800,
    "D": 1.0000,
}

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
