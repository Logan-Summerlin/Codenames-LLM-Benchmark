import tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.cache import ResponseCache

class CacheTests(unittest.TestCase):
    def test_cache_round_trip_and_key_stability(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ResponseCache(Path(d))
            key = cache.key_for("model", {"a":1}, {"temperature":0})
            cache.set(key, {"raw":"x"})
            self.assertEqual(cache.get(key)["raw"], "x")
            self.assertEqual(key, cache.key_for("model", {"a":1}, {"temperature":0}))

if __name__ == "__main__": unittest.main()
