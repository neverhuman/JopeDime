import json
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from binary_encode import owned_ranks, serial_job_waves, stop
from binary_prepare import freeze_aliases, save
from binary_replay_v2 import audit, commands_for_replay
from binary_verify import sha


class BinaryWorkflowTests(unittest.TestCase):
    def test_replay_report_cannot_inflate_independently_computed_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            save(manifest, {"views": 80, "pairs": 80, "raw_components_per_side": 240})
            save(
                root / "verification.json",
                {
                    "manifest_sha256": sha(manifest),
                    "independently_computed_contexts": 160,
                    "independently_computed_raw_components": 481,
                },
            )
            with self.assertRaisesRegex(ValueError, "reported counts"):
                audit(root, manifest)

    def test_full_replay_dispatch_computes_every_component_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"jobs": [{"prepared": f"dataset-{i // 20}", "cut": i % 20} for i in range(80)]}
            waves = []
            for i, wave in enumerate(serial_job_waves(original)):
                path = root / f"wave-{i}.json"
                save(path, wave)
                waves.append(path)
            specs = {name: root / f"{name}.json" for name in ("target", "foundation", "hyperion")}
            save(specs["hyperion"], {"persistent_binary": "worker"})
            plan = commands_for_replay(
                root, root / "original", root / "fresh", root / "encoder", waves, specs
            )
            self.assertEqual(len(plan), 62)
            for side in ("a", "b"):
                for component in specs:
                    computed = []
                    for name, commands in plan.items():
                        if name.startswith(f"{side}-{component}-"):
                            for command in commands:
                                wave = json.loads(Path(command[command.index("--jobs") + 1]).read_text())
                                rank = int(command[command.index("--rank") + 1])
                                computed.append(wave["jobs"][rank])
                    self.assertEqual(computed, original["jobs"])

    def test_serial_dispatch_preserves_original_gpu_assignment_without_cross_cut_coalescing(self):
        original = {"schema": "jobs", "jobs": [{"cut": i} for i in range(20)]}
        before = json.dumps(original)
        waves = serial_job_waves(original)
        self.assertEqual([len(w["jobs"]) for w in waves], [8, 8, 4])
        self.assertEqual([j for w in waves for j in w["jobs"]], original["jobs"])
        for wave in waves:
            assigned = {}
            for local_index, job in enumerate(wave["jobs"]):
                gpu = local_index % 8
                self.assertEqual(gpu, job["cut"] % 8)
                self.assertNotIn(gpu, assigned)
                assigned[gpu] = job
        self.assertEqual(json.dumps(original), before)

    def test_alias_freeze_preserves_cross_task_partition_and_rejects_conflicts(self):
        audit = {
            "datasets": [
                {
                    "id": "regression-sonar",
                    "task": "regression",
                    "family": "sonar",
                    "source_key": "openml:40",
                    "meta_partition": "meta_test",
                }
            ]
        }
        aliases = freeze_aliases(audit, {"families": {}})
        sonar = next(f for f in aliases["families"] if f["family"] == "sonar")
        self.assertEqual(sonar["partition"], "meta_test")
        self.assertEqual(sonar["assignment"], "preserved_existing")
        registry = {
            "families": {
                "sonar": {"family": "sonar", "partition": "meta_train", "source_keys": ["openml:40"]}
            }
        }
        with self.assertRaisesRegex(ValueError, "Conflicting frozen aliases"):
            freeze_aliases(audit, registry)

    def test_manifest_resume_is_byte_preserving_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.json"
            value = freeze_aliases({"datasets": []}, {"families": {}})
            save(path, value)
            first = path.read_bytes()
            save(path, value)
            self.assertEqual(first, path.read_bytes())
            value["qualification"] = False
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                save(path, value)

    def test_cancellation_terminates_owned_process_group_and_preserves_completed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completed.json"
            path.write_text(json.dumps({"completed": True}))
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
            )
            stop([child])
            self.assertEqual(child.returncode, -signal.SIGKILL)
            self.assertEqual(json.loads(path.read_text()), {"completed": True})

    def test_rank_cleanup_precedes_shared_gpu_lease_release_on_error(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )

        @contextmanager
        def lease():
            try:
                yield
            finally:
                self.assertIsNotNone(child.poll(), "GPU lease released while owned worker still ran")

        with self.assertRaisesRegex(RuntimeError, "injected cancellation"):
            with lease(), owned_ranks([child]):
                raise RuntimeError("injected cancellation")


if __name__ == "__main__":
    unittest.main()
