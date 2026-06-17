# Codenames LLM Benchmark Implementation Plan

> **Status:** Historical build plan. The active launcher architecture has since been consolidated into `scripts/run_openrouter_tournament.py`, and retired launcher scripts now live under `ARCHIVE/scripts/`.

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a reproducible benchmark harness where homogeneous LLM teams play Codenames-style tournaments and receive Elo plus diagnostic capability scores.

**Architecture:** Implement a deterministic Python game engine first, then add agent protocols, mock agents, LLM adapters, tournament orchestration, and reporting. Keep game state pure and serializable so every game can be replayed from seed and event log.

**Tech Stack:** Python 3.11+, dataclasses or pydantic for typed objects, pytest or unittest for tests, SQLite or JSONL for game logs, pandas for analysis once needed, optional LiteLLM/OpenRouter/provider SDK adapters later.

---

## Phase 1: Deterministic game engine

### Task 1: Create package skeleton

**Objective:** Establish importable source and test directories.

**Files:** Create `src/codenames_benchmark/__init__.py`, `src/codenames_benchmark/game.py`, `tests/test_game.py`.

**Verification:** Run `python3 -m compileall src` and confirm the package compiles.

### Task 2: Define board and state models

**Objective:** Represent board words, hidden identities, revealed state, turn state, and game result without any LLM logic.

**Files:** Modify `src/codenames_benchmark/game.py`; test in `tests/test_game.py`.

**Verification:** Unit tests should construct a board, reveal words, and serialize state to dictionaries.

### Task 3: Implement turn legality and terminal conditions

**Objective:** Enforce correct Codenames-style turn transitions, win conditions, assassin loss, and count-based guess limits.

**Files:** Modify `src/codenames_benchmark/game.py`; add tests in `tests/test_game.py`.

**Verification:** Tests should cover normal win, assassin loss, opponent-word penalty, neutral-word turn end, and voluntary stop.

## Phase 2: Board generation and clue legality

### Task 4: Add deterministic board generator

**Objective:** Generate reproducible 25-word boards from a fixed word list and seed.

**Files:** Create `src/codenames_benchmark/boards.py`, `data/wordlists/base_words.txt`, `tests/test_boards.py`.

**Verification:** Same seed produces same board; mirrored assignment preserves words while swapping team positions.

### Task 5: Add strict clue legality checker

**Objective:** Reject board words, substrings, simple morphological variants, multiword clues when disabled, and non-dictionary clues when dictionary mode is enabled.

**Files:** Create `src/codenames_benchmark/legality.py`, `tests/test_legality.py`.

**Verification:** Tests should include exact board-word clues, plural/suffix variants, substring traps, valid unrelated clues, and configurable permissive exceptions.

## Phase 3: Agent protocol

### Task 6: Define observation and action schemas

**Objective:** Create typed inputs and outputs for spymasters, guessers, and aggregators.

**Files:** Create `src/codenames_benchmark/protocol.py`, `tests/test_protocol.py`.

**Verification:** Tests should assert spymaster observations contain hidden identities and guesser observations do not.

### Task 7: Implement mock agents

**Objective:** Provide deterministic random and embedding-placeholder agents for simulator testing without API cost.

**Files:** Create `src/codenames_benchmark/agents/mock.py`, `tests/test_mock_agents.py`.

**Verification:** Mock agents should complete games without invalid protocol outputs.

### Task 8: Implement team aggregator

**Objective:** Convert three guesser ranked lists into a final legal guess sequence using a deterministic voting and confidence rule.

**Files:** Create `src/codenames_benchmark/agents/aggregate.py`, `tests/test_aggregate.py`.

**Verification:** Tests should cover consensus, disagreement, stop threshold, and confidence tie-breaking.

## Phase 4: Match and tournament runner

### Task 9: Implement single-game runner

**Objective:** Run a complete game between two teams while recording every public and private event.

**Files:** Create `src/codenames_benchmark/runner.py`, `tests/test_runner.py`.

**Verification:** A seeded mock-agent game should produce a deterministic event log and terminal result.

### Task 10: Implement mirrored matchup runner

**Objective:** Run paired games on the same board seed with team colors swapped.

**Files:** Modify `src/codenames_benchmark/runner.py`, add tests in `tests/test_runner.py`.

**Verification:** Tests should assert both games use the same words and opposite team assignments.

### Task 11: Implement round-robin scheduler

**Objective:** Generate all model pairings, color assignments, seeds, and repetitions.

**Files:** Create `src/codenames_benchmark/tournament.py`, `tests/test_tournament.py`.

**Verification:** For N models and R mirrored seeds, expected game count should be `N * (N - 1) / 2 * R * 2`.

## Phase 5: LLM adapters

### Task 12: Create provider-neutral LLM client interface

**Objective:** Isolate model calls behind a small interface that accepts messages, JSON schema hints, model settings, and returns raw plus parsed output.

**Files:** Create `src/codenames_benchmark/llm/base.py`, `tests/test_llm_base.py`.

**Verification:** Fake client should support valid response, invalid JSON response, repair attempt, and provider error cases.

### Task 13: Add one real adapter

**Objective:** Connect the benchmark to one provider without coupling game logic to provider APIs.

**Files:** Create `src/codenames_benchmark/llm/openrouter.py` or another provider-specific adapter.

**Verification:** Use an opt-in smoke script that runs only when credentials are present. Do not print or log secrets.

## Phase 6: Ratings and reporting

### Task 14: Implement Elo ratings

**Objective:** Convert game results into model Elo scores with confidence summaries from bootstrap resampling.

**Files:** Create `src/codenames_benchmark/ratings.py`, `tests/test_ratings.py`.

**Verification:** Known win/loss sequences should produce expected rating direction and stable deterministic output.

### Task 15: Implement diagnostic metrics

**Objective:** Compute clue efficiency, illegal clue rate, assassin rate, opponent-hit rate, neutral-hit rate, stop calibration, and public-opponent-clue exploitation proxies.

**Files:** Create `src/codenames_benchmark/metrics.py`, `tests/test_metrics.py`.

**Verification:** Handcrafted event logs should produce exact metric values.

### Task 16: Add report generator

**Objective:** Produce a benchmark report from tournament logs.

**Files:** Create `src/codenames_benchmark/report.py`, `docs/report_schema.md`.

**Verification:** A mock tournament should produce a readable Markdown report with Elo and diagnostics.

## Phase 7: Cost control and reproducibility

### Task 17: Add response caching

**Objective:** Cache model responses keyed by prompt hash, model ID, sampling settings, and game context.

**Files:** Create `src/codenames_benchmark/cache.py`, `tests/test_cache.py`.

**Verification:** Re-running a cached game should make zero provider calls and reproduce the same event log.

### Task 18: Add run manifest

**Objective:** Save model IDs, prompts, seeds, rules profile, word list hash, code version, and sampling settings for every tournament.

**Files:** Create `src/codenames_benchmark/manifest.py`, `tests/test_manifest.py`.

**Verification:** Every tournament output directory should include a manifest sufficient to reproduce the run.

## Phase 8: Benchmark hardening

### Task 19: Add leakage tests

**Objective:** Prove private spymaster information cannot appear in guesser prompts or public logs except through legal clues and revealed guesses.

**Files:** Add tests in `tests/test_privacy_boundaries.py`.

**Verification:** Tests should fail if hidden color assignments are accidentally included in guesser observations.

### Task 20: Add adversarial board suites

**Objective:** Evaluate models on boards with semantic traps and near-neighbor ambiguity.

**Files:** Create `data/board_suites/adversarial.jsonl`, `src/codenames_benchmark/board_suites.py`, `tests/test_board_suites.py`.

**Verification:** Suite loader should validate 25 unique words and legal identity distributions.

## First build recommendation

Start with Phase 1 and Phase 2 only. Do not call real LLMs until the deterministic simulator, board generator, legality checker, and privacy-boundary tests are passing. This keeps the benchmark honest and prevents expensive model tournaments from producing untrustworthy results.
