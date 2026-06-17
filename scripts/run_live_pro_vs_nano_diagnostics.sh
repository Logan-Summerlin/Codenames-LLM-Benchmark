#!/usr/bin/env bash
set -euo pipefail

read -rsp "OPENROUTER_API_KEY: " OPENROUTER_API_KEY
printf "\n"
export OPENROUTER_API_KEY

BASE_RUN_DIR="runs/live-matchup-deepseek-v4-pro-vs-gpt-4.1-nano-20260604"
DIAG_DIR="${BASE_RUN_DIR}/diagnostics"
mkdir -p "$DIAG_DIR" "$BASE_RUN_DIR"

run_one_turn() {
  local label="$1"
  local red_model="$2"
  local blue_model="$3"
  local timeout_seconds="$4"
  local attempts="$5"
  local transcript="${DIAG_DIR}/${label}-transcript.json"
  local log="${DIAG_DIR}/${label}.log"

  printf 'DIAG %s start red=%s blue=%s timeout=%s attempts=%s\n' "$label" "$red_model" "$blue_model" "$timeout_seconds" "$attempts" | tee "$log"
  rm -f "$transcript"
  set +e
  OPENROUTER_NODE_TIMEOUT_SECONDS="$timeout_seconds" \
  OPENROUTER_NODE_ATTEMPTS="$attempts" \
  OPENROUTER_MAX_TOKENS=10000 \
  python3 scripts/run_openrouter_matchup_game.py \
    --seed 1 \
    --max-turns 1 \
    --red-model "$red_model" \
    --blue-model "$blue_model" \
    --red-name "diag-red" \
    --blue-name "diag-blue" \
    --transcript "$transcript" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -eq 0 && -f "$transcript" ]]; then
    printf 'DIAG %s success transcript=%s\n' "$label" "$transcript" | tee -a "$log"
    return 0
  fi
  printf 'DIAG %s failed status=%s transcript_exists=%s\n' "$label" "$status" "$(test -f "$transcript" && echo yes || echo no)" | tee -a "$log"
  return "$status"
}

# First isolate whether the key/account/network path works with a known cheap route.
if ! run_one_turn "nano-vs-nano-short" "openai/gpt-4.1-nano" "openai/gpt-4.1-nano" 12 2; then
  printf 'RESULT transport/account/key path failed even for gpt-4.1-nano. Stop before DeepSeek tests.\n'
  exit 20
fi

# Then test the requested DeepSeek Pro route under the short diagnostic timer.
if run_one_turn "pro-vs-pro-short" "deepseek/deepseek-v4-pro" "deepseek/deepseek-v4-pro" 12 2; then
  PRO_ONE_TURN_OK=1
else
  PRO_ONE_TURN_OK=0
fi

# If short timer is too aggressive for Pro, try a bounded longer request once.
if [[ "$PRO_ONE_TURN_OK" -eq 0 ]]; then
  if run_one_turn "pro-vs-pro-medium" "deepseek/deepseek-v4-pro" "deepseek/deepseek-v4-pro" 45 1; then
    PRO_ONE_TURN_OK=1
  fi
fi

# If Pro still fails, verify DeepSeek family responsiveness with Flash. This does not replace Pro.
if [[ "$PRO_ONE_TURN_OK" -eq 0 ]]; then
  run_one_turn "flash-vs-flash-short" "deepseek/deepseek-v4-flash" "deepseek/deepseek-v4-flash" 12 2 || true
  printf 'RESULT deepseek/deepseek-v4-pro did not complete a one-turn diagnostic. Stop before five-game Pro vs Nano run.\n'
  exit 30
fi

printf 'RESULT one-turn DeepSeek Pro route works. Starting five-game Pro vs Nano benchmark.\n'
for seed in 1 2 3 4 5; do
  transcript="${BASE_RUN_DIR}/game-${seed}-transcript.json"
  log="${BASE_RUN_DIR}/game-${seed}.log"
  printf 'RUN seed=%s start red=deepseek/deepseek-v4-pro blue=openai/gpt-4.1-nano\n' "$seed" | tee "$log"
  rm -f "$transcript"
  OPENROUTER_NODE_TIMEOUT_SECONDS=75 \
  OPENROUTER_NODE_ATTEMPTS=4 \
  OPENROUTER_MAX_TOKENS=10000 \
  python3 scripts/run_openrouter_matchup_game.py \
    --seed "$seed" \
    --max-turns 30 \
    --red-model deepseek/deepseek-v4-pro \
    --blue-model openai/gpt-4.1-nano \
    --red-name deepseek-v4-pro \
    --blue-name gpt-4.1-nano \
    --transcript "$transcript" 2>&1 | tee -a "$log"
  printf 'RUN seed=%s done transcript=%s\n' "$seed" "$transcript" | tee -a "$log"
done

python3 - <<'PY'
import json
from pathlib import Path
base = Path('runs/live-matchup-deepseek-v4-pro-vs-gpt-4.1-nano-20260604')
files = sorted(base.glob('game-*-transcript.json'))
print(f'SUMMARY transcripts={len(files)} dir={base}')
for path in files:
    data = json.loads(path.read_text())
    reveals = [e for e in data.get('public_events', []) if e.get('event') == 'reveal']
    mix = {}
    for e in reveals:
        ident = e.get('identity')
        mix[ident] = mix.get(ident, 0) + 1
    print(json.dumps({
        'file': str(path),
        'winner': data.get('winner'),
        'terminal': data.get('terminal'),
        'reason': data.get('reason'),
        'public_events': len(data.get('public_events', [])),
        'private_events': len(data.get('private_events', [])),
        'reveals': mix,
    }, sort_keys=True))
PY
