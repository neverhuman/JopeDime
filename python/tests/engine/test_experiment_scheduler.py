import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jope.tools import experiment_scheduler as scheduler


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.write("runs/campaign-progress.json", {"deadline": 2000, "current_cycle": "runs/cycle"})
        self.write("runs/cycle-progress.json", {"phase": scheduler.PHASE, "timestamp": 1000, "neural_seeded_gp": {"completed": 10, "total": 100}})
        self.write("runs/gp-service.json", {"timestamp": 1000, "phase": "running"})
        self.write("runs/cycle/auxiliary/complete.json", {"schema": "parallel.auxiliary_training_complete.v1"})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_gpu_gap_is_not_auxiliary_completion(self):
        self.assertIsNone(scheduler.admission_reason(self.root, 1001))
        (self.root / "runs/cycle/auxiliary/complete.json").unlink()
        self.assertIsNotNone(scheduler.admission_reason(self.root, 1001))

    def test_retraining_handoff_and_stale_telemetry_block_admission(self):
        self.assertIn("stale", scheduler.admission_reason(self.root, 1006))
        self.write("runs/cycle-progress.json", {"phase": scheduler.PHASE, "timestamp": 1000, "neural_seeded_gp": {"completed": 90, "total": 100}})
        self.assertIn("90%", scheduler.admission_reason(self.root, 1001))

    def test_comparison_and_encoding_share_one_allocation_and_release_on_error(self):
        with patch.object(scheduler, "admission_reason", return_value=None), patch.object(scheduler.os, "sched_getaffinity", return_value=set(scheduler.SPARE_CPUS)), patch.object(scheduler.os, "sched_setaffinity"):
            with scheduler.spare_lease(self.root, "comparison"):
                with self.assertRaises(BlockingIOError):
                    with scheduler.spare_lease(self.root, "binary", gpu=True):
                        self.fail("double admission")
            with self.assertRaisesRegex(RuntimeError, "cancel"):
                with scheduler.spare_lease(self.root, "binary", gpu=True):
                    raise RuntimeError("cancel")
            with scheduler.spare_lease(self.root, "comparison") as lease:
                self.assertEqual(len(lease["cpus"]), 16)


if __name__ == "__main__":
    unittest.main()
