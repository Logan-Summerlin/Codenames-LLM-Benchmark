import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.report import render_markdown_report

class ReportTests(unittest.TestCase):
    def test_report_contains_ratings_and_metrics(self):
        text = render_markdown_report({"a":1510,"b":1490}, {"total_guesses":5})
        self.assertIn("# Codenames Benchmark Report", text)
        self.assertIn("a: 1510", text)
        self.assertIn("total_guesses: 5", text)

if __name__ == "__main__": unittest.main()
