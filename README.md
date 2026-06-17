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

The reporting layer includes tournament scheduling, Elo-style ratings, diagnostic metrics, Markdown report generation, response caching, and reproducibility manifests.

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

A live tournament uses the same launcher after `OPENROUTER_API_KEY` is present in the environment. The launcher pins OpenRouter `provider.order` per model and disables provider fallback so results do not silently mix backends:

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

Expected current behavior: compilation succeeds, the unittest suite passes, and the deterministic smoke script prints a terminal mock-game result similar to:

```json
{"events": 20, "reason": "all_words_found", "terminal": true, "winner": "red"}
```

If `pytest` is installed, `python3 -m pytest -q` should also be usable, but it is not required for the current local validation path.

## Interpreting run artifacts

A transcript records the board, public event log, private spymaster and guesser actions, terminal status, winner, and terminal reason. Use transcripts to distinguish true model skill from harness artifacts.

A terminal game with `reason: all_words_found` or an assassin-loss reason is a completed game outcome. A transcript with `winner: null` and `reason: null` usually means the game ended because the configured turn bound was reached or the live run was stopped before a natural terminal condition. Do not treat bounded non-terminal games as normal wins or losses without an explicit scoring policy.

Diagnostics should track illegal clues, invalid JSON, off-board guesses, neutral hits, opponent hits, assassin hits, stop behavior, clue efficiency, token usage, latency, and provider errors separately. This benchmark is most valuable when it explains why a model wins, not merely whether it wins.

## Project layout

`docs/benchmark_design.md` contains the benchmark architecture, experimental design, scoring philosophy, fairness concerns, and risk mitigations.

`docs/implementation_plan.md` contains the original phased build plan and validation gates. It is now historical; the live launcher architecture is described in this README and the active source files.

`docs/report_schema.md` describes the minimal Markdown report structure.

`src/codenames_benchmark/` contains the benchmark package implementation.

`tests/` contains deterministic unit and simulation tests. The current environment does not have `pytest` installed, but the suite runs through stdlib `unittest`.

`data/wordlists/base_words.txt` contains the default board word list.

`data/board_suites/adversarial.jsonl` contains an adversarial board-suite example.

`scripts/` contains the active deterministic smoke script and the canonical OpenRouter tournament launcher.

`ARCHIVE/` contains retired launchers and diagnostics that are preserved for reference only.

## Development notes

Keep deterministic simulator tests passing before expanding live model runs. The benchmark is only useful if board legality, clue legality, turn order, privacy boundaries, and transcript reproducibility are trustworthy.

Prefer small, cached, bounded live diagnostics before expensive tournaments. Use mirrored boards when comparing two models so first-player and color-assignment effects are balanced.

When adding new model providers, keep provider code isolated behind the LLM client interface. Game logic should not depend on provider-specific APIs.

When changing prompts, rule profiles, model IDs, sampling settings, word lists, or seeds, record those details in manifests or run output directories so results remain reproducible.

## Known current limitations

The live-model path is operational but still early. Recent live transcripts in `runs/` are often non-terminal, so full benchmark interpretation needs explicit handling for bounded games and incomplete runs.

The README and design documents may not capture every implementation detail yet. The source tests are currently the most reliable specification for exact behavior.

There is no local Git metadata in this workspace copy, so branch, commit, and diff provenance cannot be inferred from the checked-out directory alone.

## Recommended next milestone

The best next milestone is a small transcript-analysis command that summarizes a run directory into model outcomes, terminal-vs-bounded counts, illegal clue rates, off-board guess rates, assassin/neutral/opponent hit rates, and parser failures. That will make live benchmarks easier to trust before scaling to larger OpenRouter tournaments.
