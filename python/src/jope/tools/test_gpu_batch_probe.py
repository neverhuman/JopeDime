import json
from pathlib import Path
import tempfile
import unittest

from gpu_batch_probe import summarize


class BatchProbeTests(unittest.TestCase):
    def fixture(self, root):
        for rank in range(8):
            (root / f"complete-{rank}.json").write_text(json.dumps({"rank": rank, "world": 8,
                "microbatch": 512, "global_batch": 4096, "steps": 7, "eligible_checkpoint": False,
                "parameter_sha256": "a" * 64, "step_seconds": [10.] * 5 + [.1, .2]}))

    def test_warmup_excluded_and_global_example_count_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            value = json.loads((root / "complete-7.json").read_text())
            value["step_seconds"][-1] = .4
            (root / "complete-7.json").write_text(json.dumps(value))
            result = summarize(root, 512, 4096, 7)
            self.assertEqual(result["measured_examples"], 8192)
            self.assertEqual(result["measured_seconds"], .5)
            self.assertEqual(result["examples_per_second"], 16384)

    def test_rejects_every_missing_misbound_and_nonfinite_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, value in [("rank", 0), ("world", 4), ("global_batch", 2048),
                               ("parameter_sha256", "b" * 64), ("eligible_checkpoint", True),
                               ("step_seconds", [float("nan")] * 7),
                               ("step_seconds", [True] * 7), ("step_seconds", [.1] * 6)]:
                with self.subTest(field=key, value=value):
                    self.fixture(root)
                    path = root / "complete-7.json"
                    row = json.loads(path.read_text())
                    row[key] = value
                    path.write_text(json.dumps(row))
                    with self.assertRaises((RuntimeError, ValueError)):
                        summarize(root, 512, 4096, 7)
            self.fixture(root)
            (root / "complete-7.json").unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(root, 512, 4096, 7)


if __name__ == "__main__":
    unittest.main()
