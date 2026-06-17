import tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codenames_benchmark.manifest import RunManifest

class ManifestTests(unittest.TestCase):
    def test_manifest_writes_reproducibility_fields(self):
        with tempfile.TemporaryDirectory() as d:
            m = RunManifest(models=["a"], seeds=[1], rule_profile="strict", word_list_hash="abc", sampling={"temperature":0})
            path = m.write(Path(d))
            self.assertTrue(path.exists())
            self.assertIn('"rule_profile": "strict"', path.read_text())

if __name__ == "__main__": unittest.main()
