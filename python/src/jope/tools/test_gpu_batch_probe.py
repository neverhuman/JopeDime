import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_batch_probe import sample_gpu, summarize


class BatchProbeTests(unittest.TestCase):
    def test_telemetry_capture_runs_through_owned_process_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "nvidia-smi"
            executable.write_text(
                "#!/usr/bin/python3\nimport sys\nprint('0, 80, 8192, 40960, 250')\nprint('diagnostic', file=sys.stderr)\n"
            )
            executable.chmod(0o700)
            with patch("gpu_batch_probe.runtime.child_env", return_value={"PATH": directory}):
                result = sample_gpu()
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), "0, 80, 8192, 40960, 250")
            self.assertEqual(result.stderr.strip(), "diagnostic")

    def fixture(self, root):
        for rank in range(8):
            (root / f"complete-{rank}.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "world": 8,
                        "microbatch": 512,
                        "global_batch": 4096,
                        "steps": 7,
                        "eligible_checkpoint": False,
                        "vocabulary_sha256": "c" * 64,
                        "vocabulary_size": 401,
                        "parameter_sha256": "a" * 64,
                        "step_seconds": [10.0] * 5 + [0.1, 0.2],
                    }
                )
            )

    def test_warmup_excluded_and_global_example_count_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            value = json.loads((root / "complete-7.json").read_text())
            value["step_seconds"][-1] = 0.4
            (root / "complete-7.json").write_text(json.dumps(value))
            result = summarize(root, 512, 4096, 7, "c" * 64, 401)
            self.assertEqual(result["measured_examples"], 8192)
            self.assertEqual(result["measured_seconds"], 0.5)
            self.assertEqual(result["examples_per_second"], 16384)

    def test_rejects_every_missing_misbound_and_nonfinite_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, value in [
                ("vocabulary_sha256", "d" * 64),
                ("vocabulary_size", 1103),
                ("rank", 0),
                ("world", 4),
                ("global_batch", 2048),
                ("parameter_sha256", "b" * 64),
                ("eligible_checkpoint", True),
                ("step_seconds", [float("nan")] * 7),
                ("step_seconds", [True] * 7),
                ("step_seconds", [0.1] * 6),
            ]:
                with self.subTest(field=key, value=value):
                    self.fixture(root)
                    path = root / "complete-7.json"
                    row = json.loads(path.read_text())
                    row[key] = value
                    path.write_text(json.dumps(row))
                    with self.assertRaises((RuntimeError, ValueError)):
                        summarize(root, 512, 4096, 7, "c" * 64, 401)
            self.fixture(root)
            (root / "complete-7.json").unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(root, 512, 4096, 7, "c" * 64, 401)


if __name__ == "__main__":
    unittest.main()
