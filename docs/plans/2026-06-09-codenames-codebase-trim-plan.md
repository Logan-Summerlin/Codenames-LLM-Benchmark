# Codenames Benchmark Codebase Trim Plan

> **Status:** Implemented. The active launcher has been consolidated into `scripts/run_openrouter_tournament.py`, and the retired launchers now live under `ARCHIVE/scripts/`.

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce the active code surface of the Codenames benchmark by at least 15 percent while preserving benchmark behavior, reproducibility, and the deterministic simulator.

**Architecture:** The repository should keep one canonical OpenRouter tournament pipeline, one shared launcher core, and one small set of simulator modules. Legacy one-off scripts and narrow smoke helpers should move into `ARCHIVE/` with clear mappings back to the canonical CLI. The deterministic engine, legality checks, protocol objects, and tests remain active; only duplicated orchestration, wrapper scripts, and stale experiment entrypoints should be consolidated or retired.

**Tech Stack:** Python 3.12, stdlib `unittest`, JSON/JSONL artifacts, OpenRouter adapter, Markdown documentation.

---

## Baseline and target

The current active Python surface is 5,585 lines across `src/`, `scripts/`, and `tests/`.

- `src/codenames_benchmark`: 1,942 lines across 21 files
- `scripts`: 2,083 lines across 14 files
- `tests`: 1,560 lines across 23 files

The 15 percent reduction target is at least 838 lines removed from the active surface, which means the active Python total must end at 4,747 lines or fewer.

The largest current duplication cluster is in `scripts/`. The current launcher and diagnostic scripts are large enough that archiving only the obviously legacy entrypoints already exceeds the target.

Likely archive candidates, based on current contents, total about 2,002 lines of script code:

- `scripts/run_openrouter_single_round_robin.py`
- `scripts/run_openrouter_double_round_robin.py`
- `scripts/run_top4_single_round_robin.py`
- `scripts/run_top4_round_robin.py`
- `scripts/run_random_limited_tournament.py`
- `scripts/rerun_failed_pairings.py`
- `scripts/run_deepseek_v4_flash_game.py`
- `scripts/run_deepseek_v4_flash_smoke.py`
- `scripts/run_deepseek_pro_vs_nano_reasoning_off.py`
- `scripts/diagnose_openrouter_latency.py`
- `scripts/openrouter_deepseek_response_shapes.py`
- `scripts/openrouter_model_smoke_matrix.py`

The repository does not currently have an `ARCHIVE/` directory, so this pass must create one.

## Canonical path after consolidation

The active launcher should converge on a single canonical OpenRouter entrypoint, most likely `scripts/run_openrouter_tournament.py`, backed by a shared library module in `src/codenames_benchmark/` such as `tournament_runner.py`.

The canonical launcher should support the current live tournament shapes through flags, not through separate near-duplicate scripts. At minimum it should handle:

- single round robin
- double round robin
- limited coverage / smoke coverage
- model presets such as full field and top-four subsets
- dry-run schedule validation
- bounded live runs with transcript output

---

## Task 1: Freeze the active surface and define the canonical API boundary

**Objective:** Confirm which files are canonical, which are generated, and which should be moved to `ARCHIVE/` before any code is rewritten.

**Files:**
- Read only: `README.md`, `docs/benchmark_design.md`, `docs/implementation_plan.md`, `src/codenames_benchmark/tournament.py`, `scripts/run_openrouter_single_round_robin.py`, `scripts/run_openrouter_double_round_robin.py`, `scripts/run_openrouter_matchup_game.py`
- Create later: `ARCHIVE/README.md`

**Work:**
- Reconfirm the active launcher set and the archive candidate list.
- Decide whether `scripts/run_openrouter_matchup_game.py` remains as a thin low-level helper or is absorbed into the canonical launcher.
- Confirm that `runs/` stays generated-only and is not treated as source.
- Write the canonical file map for the rest of the plan.

**Verification:**
- Run `python3 -m compileall -q src tests scripts`.
- Run `python3 -m unittest discover -s tests -v`.
- Record the pre-change active line count as the baseline for the trim gate.

**Expected result:** A stable canonical map with a single active benchmark launcher path and a clearly named archive boundary.

---

## Task 2: Build one shared tournament runner in `src/`

**Objective:** Remove duplicated manifest writing, JSONL appending, provider pinning, environment restoration, result serialization, and run-state bookkeeping from the launcher scripts.

**Files:**
- Create: `src/codenames_benchmark/tournament_runner.py`
- Modify: `scripts/run_openrouter_matchup_game.py`
- Modify: `scripts/run_openrouter_single_round_robin.py`
- Modify: `scripts/run_openrouter_double_round_robin.py`
- Modify: `scripts/run_top4_single_round_robin.py`
- Modify: `scripts/run_top4_round_robin.py`
- Modify: `scripts/run_random_limited_tournament.py`

**Work:**
- Move shared helper logic into `src/codenames_benchmark/tournament_runner.py`.
- Centralize:
  - output directory creation
  - JSON / JSONL write helpers
  - provider order and reasoning-effort environment pinning
  - manifest generation
  - run-state updates
  - transcript writing hooks
  - common game-summary serialization
- Make the scripts call the shared runner instead of reimplementing the same plumbing.

**Suggested tests:**
- Add or extend `tests/test_tournament_runner.py`.
- Verify that one-game, single-round-robin, and double-round-robin runs all write the same shared artifact shape.
- Verify that provider-order and reasoning-effort environment variables are restored after a run.

**Verification:**
- `python3 -m unittest discover -s tests -v`
- `python3 scripts/run_openrouter_single_round_robin.py --dry-run --limit-games 1 --output-dir runs/trim-check-single`
- `python3 scripts/run_openrouter_double_round_robin.py --dry-run --limit-games 1 --output-dir runs/trim-check-double`

**Expected result:** One shared runner replaces the duplicated orchestration in the current launcher set and removes the largest block of repeated code.

---

## Task 3: Replace the launcher sprawl with one canonical CLI and archive the rest

**Objective:** Consolidate the public entrypoints into one canonical command and move retired launchers into `ARCHIVE/scripts/` instead of keeping multiple active one-off runners.

**Files:**
- Create: `scripts/run_openrouter_tournament.py`
- Create: `ARCHIVE/README.md`
- Create: `ARCHIVE/scripts/README.md` or a single archive index under `ARCHIVE/`
- Move to archive: the legacy launcher and diagnostic scripts listed in the baseline section once the canonical CLI covers their behavior
- Modify: `README.md`
- Modify: `docs/benchmark_design.md`

**Work:**
- Implement the canonical CLI with flags for schedule mode, model preset, start game, limit games, workers, max turns, dry run, and transcript output.
- Retire the separate top-four, deepseek-specific, random-limited, retry, and diagnostic launchers after their behavior is covered by flags or a small diagnostic subcommand.
- Preserve compatibility only if a wrapper is truly needed for external automation; otherwise move the old file to `ARCHIVE/`.
- Write the archive README so every retired path points to the new canonical invocation.

**Expected archive mapping examples:**
- `run_openrouter_single_round_robin.py` and `run_openrouter_double_round_robin.py` become modes of the new canonical CLI.
- `run_top4_single_round_robin.py` and `run_top4_round_robin.py` become presets of the new canonical CLI.
- `run_deepseek_v4_flash_*` scripts become archived one-offs unless they are still needed as tiny wrappers.
- `diagnose_openrouter_latency.py`, `openrouter_deepseek_response_shapes.py`, and `openrouter_model_smoke_matrix.py` become a single diagnostics tool or archival references.

**Verification:**
- `python3 scripts/run_openrouter_tournament.py --dry-run --schedule-mode double --limit-games 1 --output-dir runs/trim-check-canonical`
- `python3 scripts/run_openrouter_tournament.py --dry-run --schedule-mode single --limit-games 1 --output-dir runs/trim-check-canonical-single`
- Confirm the old launcher paths are gone from the active tree and present only in `ARCHIVE/` if they are retained at all.

**Expected result:** The repo has one obvious OpenRouter launcher, not a pile of near-duplicate scripts.

---

## Task 4: Trim the remaining internal duplication in `src/`

**Objective:** Remove smaller but still meaningful duplication in the core modules once the launcher consolidation has happened.

**Files:**
- Modify: `src/codenames_benchmark/tournament.py`
- Modify: `src/codenames_benchmark/ratings.py`
- Modify: `src/codenames_benchmark/runner.py`
- Modify: `src/codenames_benchmark/transcript.py`
- Modify only if needed: `src/codenames_benchmark/agents/llm_agents.py`

**Work:**
- In `tournament.py`, collapse repeated seed formatting and schedule helpers into one shared generator path.
- In `ratings.py`, factor the Elo score application into a shared helper so `record_game` and `record_round` do not duplicate the same math.
- In `runner.py` and `transcript.py`, decide whether transcript assembly belongs in one module instead of being split across two thin layers.
- In `agents/llm_agents.py`, trim parser helpers only if the shared runner work leaves a smaller remaining gap to the target.

**Suggested tests:**
- Extend the existing rating and tournament tests instead of adding brand-new duplicate coverage.
- Add a regression test that proves round-based Elo updates still use pre-round ratings.
- Add a regression test that transcript output still contains the same public/private shape after any helper consolidation.

**Verification:**
- `python3 -m unittest discover -s tests -v`
- `python3 -m compileall -q src tests scripts`

**Expected result:** The core package stays behaviorally identical but loses the leftover helper redundancy.

---

## Task 5: Refresh the documentation to match the trimmed repository

**Objective:** Make the docs describe the canonical launcher, the archive boundary, and the current validation path instead of the pre-consolidation script sprawl.

**Files:**
- Modify: `README.md`
- Modify: `docs/benchmark_design.md`
- Modify: `docs/implementation_plan.md`
- Modify: `docs/report_schema.md` if the final report command or artifact names change
- Create/update: `ARCHIVE/README.md`

**Work:**
- Update `README.md` so the quick-start and live-run sections point to the canonical CLI only.
- Document that archived scripts live under `ARCHIVE/` and are no longer the preferred entrypoints.
- Mark `docs/implementation_plan.md` as historical or superseded if its future-phase language no longer matches the trimmed architecture.
- Keep the deterministic validation commands current and minimal.

**Verification:**
- Read the updated docs and confirm they name the new canonical path explicitly.
- Search the docs for retired launcher names and make sure any remaining mentions are clearly marked as archived or historical.

**Expected result:** A newcomer can tell, from the docs alone, which command is real, which code is archived, and how to validate the project.

---

## Task 6: Prove the reduction target and lock the repository state

**Objective:** Verify that the active code surface is below the 4,747-line ceiling and that the consolidated code still passes the project’s normal checks.

**Files:**
- Everything touched above
- Verification only; no new source files expected

**Work:**
- Recount active `.py` lines in `src/`, `scripts/`, and `tests/`.
- Confirm the reduction is at least 838 lines from the 5,585-line baseline.
- Confirm archive files are either absent from the active tree or present only under `ARCHIVE/`.
- Confirm no stale doc points users to retired active launchers.

**Verification commands:**
- `python3 -m compileall -q src tests scripts`
- `python3 -m unittest discover -s tests -v`
- `python3 scripts/run_openrouter_tournament.py --dry-run --schedule-mode double --limit-games 1 --output-dir runs/trim-check-final`
- `python3 - <<'PY'` to recount active `.py` lines and assert the total is `<= 4747`
- `git status --short --untracked-files=all`

**Expected result:** The repository is smaller, cleaner, and still fully validated.

---

## Acceptance criteria

This plan is complete only when all of the following are true:

- active Python lines are reduced by at least 838 lines from the baseline;
- the repository has one canonical OpenRouter tournament CLI;
- retired one-off launchers are moved under `ARCHIVE/` or removed from the active tree;
- the docs point to the canonical path and the archive boundary;
- the deterministic test suite still passes;
- the canonical dry-run command still writes the expected run artifacts.

## Recommended first slice

Start with Task 2. It removes the biggest repeated block of code, gives the archive work a single shared destination, and makes the later documentation pass much simpler.
