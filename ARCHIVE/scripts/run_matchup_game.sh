#!/usr/bin/env bash
# Run a single Codenames game between Gemma 4 31B and Kimi K2.5
set -euo pipefail

# Source Hermes environment for API key
source /home/agentbot/.hermes/.env 2>/dev/null || true
cd /home/agentbot/workspace/codenames-llm-benchmark

# Verify API key is loaded
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

# Generate a deterministic seed for reproducibility
SEED="matchup-gemma4-vs-kimi-k2.5-$(date +%s)"

# Set provider order: Kimi K2.5 pinned to ModelRun, Gemma 4 31B uses default routing
PROVIDER_ORDER=$(cat <<'PYEOF'
{"google/gemma-4-31b-it": [], "moonshotai/kimi-k2.5": ["ModelRun"]}
PYEOF
)

# Set timeouts for LLM calls (codenames can be slow - multiple rounds of spymaster + guesser)
export OPENROUTER_NODE_TIMEOUT_SECONDS=75
export OPENROUTER_NODE_ATTEMPTS=4
export OPENROUTER_MAX_TOKENS=10000
export OPENROUTER_PROVIDER_ORDER_JSON="${PROVIDER_ORDER}"

echo "=== Game Configuration ==="
echo "Red team:  Gemma 4 31B (google/gemma-4-31b-it) — default routing"
echo "Blue team: Kimi K2.5 (moonshotai/kimi-k2.5) — pinned to ModelRun"
echo "Seed:      ${SEED}"
echo "Max turns: 30"
echo "=========================="
echo ""

# Run the game using the archived matchup script
cd /home/agentbot/workspace/codenames-llm-benchmark
PYTHONPATH="src:${PYTHONPATH:-}" python3 ARCHIVE/scripts/run_openrouter_matchup_game.py \
    --red-model "google/gemma-4-31b-it" \
    --blue-model "moonshotai/kimi-k2.5" \
    --red-name "Gemma-4-31B" \
    --blue-name "Kimi-K2.5" \
    --seed "${SEED}" \
    --max-turns 30 \
    --transcript "runs/matchup-gemma4-vs-kimi-k2.5/transcript.json"

echo ""
echo "=== Game Complete ==="
