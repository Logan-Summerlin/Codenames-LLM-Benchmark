# ARCHIVE

This directory contains retired benchmark launchers and diagnostics that are no longer the canonical path.

The active OpenRouter entrypoint is now `scripts/run_openrouter_tournament.py`.

Retired launcher groups:

- `ARCHIVE/scripts/run_openrouter_single_round_robin.py` and `ARCHIVE/scripts/run_openrouter_double_round_robin.py` were replaced by the canonical tournament CLI.
- `ARCHIVE/scripts/run_top4_single_round_robin.py` and `ARCHIVE/scripts/run_top4_round_robin.py` were replaced by the canonical tournament CLI with the `--model-preset top4` flag.
- `ARCHIVE/scripts/run_openrouter_matchup_game.py` was absorbed into the shared tournament runner.
- `ARCHIVE/scripts/run_deepseek_v4_flash_game.py`, `ARCHIVE/scripts/run_deepseek_v4_flash_smoke.py`, `ARCHIVE/scripts/run_deepseek_pro_vs_nano_reasoning_off.py`, `ARCHIVE/scripts/run_random_limited_tournament.py`, `ARCHIVE/scripts/rerun_failed_pairings.py`, `ARCHIVE/scripts/diagnose_openrouter_latency.py`, `ARCHIVE/scripts/openrouter_deepseek_response_shapes.py`, and `ARCHIVE/scripts/openrouter_model_smoke_matrix.py` are preserved for reference only.
- `ARCHIVE/scripts/rerun_failed_matchups.py`, `ARCHIVE/scripts/run_failed_and_issue_random_matchups.py`, and `ARCHIVE/scripts/run_serial_timeout_rerun.py` are one-off recovery launchers with hard-coded source-run paths and game numbers from past live campaigns. They illustrate how to rerun failed or timed-out games using the shared `tournament_runner` helpers, but they are not parameterized for general use.
- `ARCHIVE/scripts/run_matchup_game.sh` and `ARCHIVE/scripts/run_live_pro_vs_nano_diagnostics.sh` are environment-specific shell wrappers (with absolute machine paths) that drove single-matchup smoke checks during development.

These files remain available for historical inspection, but they are not the supported active interface. To rerun a subset of games today, invoke the canonical CLI with `--start-game`/`--limit-games` against the same `--seed-prefix` and `--schedule-mode`.
