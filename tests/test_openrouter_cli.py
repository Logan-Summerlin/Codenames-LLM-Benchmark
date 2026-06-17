import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> dict:
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "scripts/run_openrouter_tournament.py", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


class OpenRouterTournamentCLITests(unittest.TestCase):
    def test_canonical_tournament_cli_is_safe_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "cli-no-key"
            payload = _run_cli(
                [
                    "--model-preset", "top4",
                    "--schedule-mode", "single-round-robin",
                    "--seed-prefix", "cli-test",
                    "--max-turns", "1",
                    "--output-dir", str(output_dir),
                ]
            )
            self.assertEqual(payload["status"], "configured_no_api_key")
            self.assertEqual(payload["selected_games"], 6)
            self.assertTrue((output_dir / "run_state.json").exists())

    def test_canonical_tournament_cli_dry_run_writes_schedule_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "cli-dry-run"
            payload = _run_cli(
                [
                    "--dry-run",
                    "--limit-games", "1",
                    "--output-dir", str(output_dir),
                ]
            )
            self.assertEqual(payload["status"], "dry_run")
            self.assertEqual(payload["selected_games"], 1)
            self.assertTrue((output_dir / "schedule.json").exists())
            self.assertTrue((output_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
