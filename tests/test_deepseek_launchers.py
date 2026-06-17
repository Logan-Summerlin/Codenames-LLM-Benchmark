import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OpenRouterTournamentCLITests(unittest.TestCase):
    def test_canonical_tournament_cli_is_safe_without_api_key(self):
        env = dict(os.environ)
        env.pop("OPENROUTER_API_KEY", None)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_openrouter_tournament.py",
                "--model-preset",
                "top4",
                "--schedule-mode",
                "single-round-robin",
                "--seed-prefix",
                "cli-test",
                "--max-turns",
                "1",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "configured_no_api_key")
        self.assertEqual(payload["selected_games"], 6)
        self.assertTrue(Path(payload["output_dir"]).name.startswith("codenames-openrouter-single-round-robin-"))

    def test_canonical_tournament_cli_dry_run_writes_schedule_without_api_key(self):
        env = dict(os.environ)
        env.pop("OPENROUTER_API_KEY", None)
        output_dir = ROOT / "tmp" / "canonical-cli-dry-run"
        if output_dir.exists():
            for path in sorted(output_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            if output_dir.exists():
                output_dir.rmdir()
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_openrouter_tournament.py",
                "--dry-run",
                "--limit-games",
                "1",
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["selected_games"], 1)
        self.assertTrue((output_dir / "schedule.json").exists())
        self.assertTrue((output_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
