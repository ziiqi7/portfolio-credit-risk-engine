#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

run_case() {
  local title="$1"
  local logfile="$2"
  local mode="$3"
  local stress="$4"
  shift 4

  echo
  echo "============================================================"
  echo "$title"
  echo "simulation-mode: $mode"
  echo "stress: $stress"
  if [[ "$*" == *"--compare-modes"* ]]; then
    echo "compare-modes: enabled"
  fi
  echo "log: $LOG_DIR/$logfile"
  echo "============================================================"

  python scripts/run_demo.py \
    --simulation-mode "$mode" \
    --stress "$stress" \
    --scenarios 2000 \
    --seed 42 \
    "$@" 2>&1 | tee "$LOG_DIR/$logfile"
}

run_case "Independent Baseline" "independent.log" "independent" "none"
run_case "One-Factor Baseline" "one_factor.log" "one_factor" "none"
run_case "Multi-Factor Baseline" "multi_factor.log" "multi_factor" "none"
run_case "Multi-Factor Regime Stress" "multi_factor_regime.log" "multi_factor" "regime"
run_case "Mode Comparison Under Regime Stress" "comparison.log" "multi_factor" "regime" "--compare-modes"

