#!/usr/bin/env bash
set -euo pipefail
read -rsp 'OpenRouter key: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
export OPENROUTER_NODE_TIMEOUT_SECONDS=75
export OPENROUTER_NODE_ATTEMPTS=4
export OPENROUTER_MAX_TOKENS=10000
OUT="runs/openrouter-limited-coverage-18-models-3workers-$(date +%Y%m%d-%H%M%S)"
python3 scripts/run_openrouter_tournament.py \
  --require-api-key \
  --schedule-mode limited-coverage \
  --workers 3 \
  --max-turns 30 \
  --round-size 9 \
  --output-dir "$OUT"
printf 'RUN_OUTPUT_DIR=%s\n' "$OUT"
unset OPENROUTER_API_KEY
