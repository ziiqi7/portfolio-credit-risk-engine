"""Generate a synthetic portfolio CSV for the demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_data import save_synthetic_portfolio


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic sample portfolio.")
    parser.add_argument("--num-exposures", type=int, default=150, help="Number of synthetic exposures to create.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/sample_portfolio.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = save_synthetic_portfolio(
        output_path=_resolve_repo_path(args.output),
        num_exposures=args.num_exposures,
        seed=args.seed,
    )
    print(f"Synthetic portfolio saved to: {output_path}")


if __name__ == "__main__":
    main()
