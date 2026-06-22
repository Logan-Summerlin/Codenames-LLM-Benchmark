# Codenames LLM Benchmark

A reproducible benchmark harness for evaluating large language models through repeated Codenames-style team play. Each team uses one model for a spymaster and three guessers. The benchmark measures win/loss outcomes as well as clue quality, deduction, coordination, risk management, legality failures, protocol failures, and strategic use of public opponent clues.

The project includes a deterministic simulator, typed agent protocol, mock agents, an OpenRouter-backed live path, transcript writing, tournament scheduling, ratings, metrics, reports, response caching, run manifests, and privacy-boundary tests.

## Core idea

Codenames is useful as an LLM benchmark because it stresses semantic association under uncertainty. A strong spymaster must compress several hidden target words into one legal clue while avoiding accidental associations with opponent, neutral, and assassin words. Strong guessers must infer partner intent from sparse public information, stop before overreaching, and adapt to the information leaked by both teams' public clues and guesses.

The baseline setup uses homogeneous teams. A model is evaluated as a coordinated group rather than as a single isolated answerer, and the engine keeps private spymaster information separate from public guesser observations.

## Current capabilities

The deterministic game engine models a 25-word Codenames board, hidden identities, revealed state, clue phases, guessing phases, terminal conditions, assassin losses, opponent-word penalties, neutral-word turn endings, voluntary stops, and count-based guess limits.

Board generation is seeded and reproducible. The project includes a base word list, mirrored board support for fair color swaps, and an adversarial board-suite loader for semantically tricky evaluation boards.

Clue legality checking supports strict automated rules, including exact board-word rejection, simple morphological variant rejection, substring-trap rejection, multiword clue rejection, and optional dictionary-mode validation.

The agent protocol defines private spymaster observations, public-only guesser observations, structured spymaster actions, structured guesser actions, and deterministic team aggregation. Privacy tests assert that hidden identities do not leak into guesser prompts.

The runner can execute seeded games between mock teams or LLM-backed teams. It records public events and private actions, and transcripts can be written as JSON for later audit.

The reporting layer includes tournament scheduling, Elo-style ratings, diagnostic metrics, per-run model summaries, Markdown report generation, response caching, and reproducibility manifests.

## Canonical OpenRouter launcher

The active OpenRouter entrypoint is `scripts/run_openrouter_tournament.py`. It replaces the old one-off round-robin, top-four, matchup, smoke, retry, and diagnostic launchers, which now live under `ARCHIVE/scripts/` for historical reference only.

The canonical CLI supports the full field or the top-four preset, plus single round robin, double round robin, and limited-coverage schedules.

A safe schedule-only verification run makes no provider calls:

```bash
python3 scripts/run_openrouter_tournament.py \
  --dry-run \
  --schedule-mode double-round-robin \
  --limit-games 1 \
  --output-dir runs/test-double-round-robin-dry-run
```

A live tournament uses the same launcher after `OPENROUTER_API_KEY` is present in the environment. The launcher pins OpenRouter `provider.order` per model from the live active-provider allowlist. Provider fallback stays disabled when a model resolves to a single route and is only enabled when multiple healthy routes exist, so results do not silently mix backends without that being recorded in the run manifest:

```bash
OPENROUTER_NODE_TIMEOUT_SECONDS=75 \
OPENROUTER_NODE_ATTEMPTS=4 \
OPENROUTER_MAX_TOKENS=10000 \
python3 scripts/run_openrouter_tournament.py \
  --schedule-mode double-round-robin \
  --max-turns 30 \
  --output-dir runs/openrouter-double-round-robin-live
```

For a limited live coverage tournament where every model appears at least once, run:

```bash
OPENROUTER_NODE_TIMEOUT_SECONDS=75 \
OPENROUTER_NODE_ATTEMPTS=4 \
OPENROUTER_MAX_TOKENS=10000 \
python3 scripts/run_openrouter_tournament.py \
  --schedule-mode limited-coverage \
  --max-turns 30 \
  --output-dir runs/openrouter-limited-coverage-live
```

### Adaptive provider failover

When a model lists more than one route in `OPENROUTER_PROVIDER_ORDER_JSON`, the OpenRouter client tracks slow and failing providers and rearranges `provider.order` so a flaky provider stops being tried first. A provider earns a strike each time a request to it times out (`OPENROUTER_NODE_TIMEOUT_SECONDS`) or returns successfully but slower than the slow threshold. Once a provider reaches the strike limit it is rotated to the back of `provider.order` for the rest of the run (fallbacks stay enabled, so it remains a last resort). Strike counts are sticky for the life of the client, so later games in the same run do not keep re-hitting the slow provider first.

Two knobs control this (defaults reproduce "rearrange after 2 timeouts or responses slower than 60 seconds"):

```bash
OPENROUTER_PROVIDER_STRIKE_LIMIT=2 \
OPENROUTER_PROVIDER_SLOW_SECONDS=60 \
OPENROUTER_NODE_TIMEOUT_SECONDS=75 \
OPENROUTER_NODE_ATTEMPTS=4 \
python3 scripts/run_openrouter_tournament.py \
  --schedule-mode limited-coverage \
  --output-dir runs/openrouter-limited-coverage-live
```

Note: the slow-response strike can only fire when a request actually returns, so keep `OPENROUTER_NODE_TIMEOUT_SECONDS` above `OPENROUTER_PROVIDER_SLOW_SECONDS`; otherwise a response slower than the slow threshold is cut off first and counts as a timeout strike instead (which still demotes the provider).

For the four-model preset, add `--model-preset top4`.

A canonical dry run with the top-four preset looks like this:

```bash
python3 scripts/run_openrouter_tournament.py \
  --dry-run \
  --model-preset top4 \
  --schedule-mode single-round-robin \
  --limit-games 1 \
  --output-dir runs/top4-dry-run
```

## Quick validation

Run these commands from the repository root:

```bash
python3 -m compileall -q src tests scripts
python3 -m unittest discover -s tests -v
python3 scripts/run_deterministic_smoke.py
```

Expected current behavior: compilation succeeds, the unittest suite passes (119 tests), and the deterministic smoke script prints a terminal mock-game result similar to:

```json
{"events": 20, "reason": "all_words_found", "terminal": true, "winner": "red"}
```

If `pytest` is installed, `python3 -m pytest -q` should also be usable, but it is not required for the current local validation path.

## Interpreting run artifacts

A transcript records the board, public event log, private spymaster and guesser actions, terminal status, winner, and terminal reason. Use transcripts to distinguish true model skill from harness artifacts.

A terminal game with `reason: all_words_found` or an assassin-loss reason is a completed game outcome. A transcript with `winner: null` and `reason: null` usually means the game ended because the configured turn bound was reached or the live run was stopped before a natural terminal condition. Do not treat bounded non-terminal games as normal wins or losses without an explicit scoring policy.

Diagnostics should track illegal clues, invalid JSON, off-board guesses, neutral hits, opponent hits, assassin hits, stop behavior, clue efficiency, token usage, latency, and provider errors separately. This benchmark is most valuable when it explains why a model wins, not merely whether it wins.

To turn a finished run directory into a per-model summary, use the run-summary tool. It reads the `game-*/transcript.json` artifacts (no provider calls) and reports per-model records, terminal-vs-bounded game counts, illegal clue rates, clue efficiency, and assassin/neutral/opponent hit counts:

```bash
python3 scripts/summarize_run.py runs/<run-directory>
```

Pass `--output summary.json` to also write the summary to a file.

## Project layout

`docs/benchmark_design.md` contains the benchmark architecture, experimental design, scoring philosophy, fairness concerns, and risk mitigations.

`docs/implementation_plan.md` contains the original phased build plan and validation gates. It is now historical; the live launcher architecture is described in this README and the active source files.

`docs/report_schema.md` describes the minimal Markdown report structure.

`src/codenames_benchmark/` contains the benchmark package implementation.

`tests/` contains deterministic unit and simulation tests. The current environment does not have `pytest` installed, but the suite runs through stdlib `unittest`.

`data/wordlists/base_words.txt` contains the default board word list.

`data/board_suites/adversarial.jsonl` contains an adversarial board-suite example.

`scripts/` contains the three active entrypoints: the deterministic smoke script (`run_deterministic_smoke.py`), the canonical OpenRouter tournament launcher (`run_openrouter_tournament.py`), and the run-summary tool (`summarize_run.py`).

`ARCHIVE/` contains retired launchers, one-off recovery scripts, and diagnostics that are preserved for reference only.

## Development notes

Keep deterministic simulator tests passing before expanding live model runs. The benchmark is only useful if board legality, clue legality, turn order, privacy boundaries, and transcript reproducibility are trustworthy.

Prefer small, cached, bounded live diagnostics before expensive tournaments. Use mirrored boards when comparing two models so first-player and color-assignment effects are balanced.

When adding new model providers, keep provider code isolated behind the LLM client interface. Game logic should not depend on provider-specific APIs.

When changing prompts, rule profiles, model IDs, sampling settings, word lists, or seeds, record those details in manifests or run output directories so results remain reproducible.

## Is Codenames a valid LLM benchmark?

Codenames is a defensible probe of two capabilities that are otherwise hard to isolate:

- **Semantic association under constraint.** A spymaster must compress several hidden target words into one legal clue while steering clear of opponent, neutral, and especially assassin associations. This rewards graded, multi-word semantic reasoning rather than single-fact recall, and the strict legality checker (board-word, morphological-variant, substring, and multiword rejection) prevents trivially literal clues.
- **Inference and risk management.** Guessers see only public information and must infer partner intent, weigh confidence, and stop before overreaching. The hidden/public observation split is enforced by the protocol layer and asserted by privacy-boundary tests, so a guesser cannot peek at hidden identities.

The benchmark is therefore a reasonable measure of in-context semantic and strategic skill, but a few design choices bound what its scores mean, and these should be reported alongside any leaderboard:

- **Homogeneous teams.** Each team is one model playing both spymaster and all guessers. This measures a model's internal self-consistency (does it guess the way it cues?) more than open coordination with a different partner. Cross-model pairings would isolate coordination separately.
- **The three guessers are an ensemble of one model.** Three guesser instances of the same model at low temperature vote through a deterministic aggregator (`confidence_threshold=0.7`, `min_consensus_votes=2`). This stabilizes parsing noise but is closer to self-consistency sampling than to three independent reasoners; the prompt's `0.70` confidence gate is intentionally coupled to the aggregator threshold.
- **Small-sample, high-variance outcomes.** Limited-coverage and single-round-robin schedules give each model very few games, so Elo and win-rate are noisy. Prefer double round robin (or many seeds) before ranking, and lean on the diagnostic rates from `summarize_run.py` rather than win/loss alone.
- **Bounded games dilute the win signal.** Live games that hit `--max-turns` end non-terminal with `winner: null`. Treat these as ties only under an explicit policy; `summarize_run.py` reports terminal-vs-bounded counts so this is visible.
- **Legality is heuristic.** Substring and morphological rejection are deliberately strict and can occasionally reject a defensible clue. This is a conservative trade-off (no illegal clue slips through) that should be kept in mind when comparing illegal-clue rates.

In short: it is a legitimate semantic-plus-strategy benchmark, best read through per-model diagnostics rather than raw win counts, and strongest when run with a full double round robin and many seeds.

## Known current limitations

The live-model path is operational but still early. Recent live transcripts in `runs/` are often non-terminal, so full benchmark interpretation needs explicit handling for bounded games and incomplete runs (see `summarize_run.py`).

The README and design documents aim to describe current behavior, but the source tests remain the most precise specification of exact engine and protocol semantics.

Per-model diagnostics are currently aggregated after the fact from transcripts by `summarize_run.py` rather than written inline during the tournament run; wiring those rates directly into each round summary is a natural next step.
