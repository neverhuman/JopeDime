import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from jope.tools import opportunistic_proposals as library


class ProposalLibraryTests(unittest.TestCase):
    def test_pool_hashes_and_restore_survive_closed_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "library"
            chunk = output / "chunk-0002"
            path = chunk / "neural/dataset/cut-2"
            library.write(chunk / "champion-snapshot.json", {"frozen": True})
            snapshot = library.digest(chunk / "champion-snapshot.json")
            candidates = [{"genome": f"program-{i}"} for i in range(80)]
            seeds = {"checkpoint_sha256": "checkpoint", "context_sha256": "context", "candidates": candidates,
                     "cut": 2, "prepared_sha256": "prepared"}
            ranked = [{"program": c["genome"]} for c in candidates[:64]]
            ranking = {"bindings": {"checkpoint_sha256": "checkpoint", "conditioning_context_sha256": "context"},
                       "training_context_only": True, "heldout_labels_read": False, "gp_scores_read": False,
                       "chosen_count": 64, "ranked": ranked}
            best = {"prediction": ranked[0], "final_tests_opened": False, "training_context_sha256": "context", "dataset": "dataset"}
            for name, value in [("seeds", seeds), ("ranking", ranking), ("best-predicted", best)]:
                library.write(path / f"{name}.json", value)
            generation = {"champion_snapshot_sha256": snapshot, "checkpoint_sha256": "checkpoint",
                          "seeds_sha256": library.digest(path / "seeds.json"),
                          "ranking_sha256": library.digest(path / "ranking.json"),
                          "best_predicted_sha256": library.digest(path / "best-predicted.json")}
            library.write(path / "generation.json", generation)
            pool = library.validate_pool(root, path, "checkpoint", snapshot)
            binding = {"spec_sha256": "spec", "script_sha256": "script"}
            library.write(output / "library-index.json", {"schema": "parallel.proposal_library_index.v1", **binding,
                          "pools": [pool], "chunks": []})
            restored, chunks = library.restore_library(root, output, "checkpoint", binding)
            self.assertEqual(len(restored), 1)
            self.assertEqual(chunks, [])
            self.assertFalse(library.gate(root, root / "absent-cycle", 100)[0])
            self.assertEqual(len(restored), 1)
            (path / "ranking.json").write_text("tampered")
            with self.assertRaises(ValueError):
                library.restore_library(root, output, "checkpoint", binding)

    def test_gate_rejects_stale_wrong_cycle_phase_deadline_and_ninety_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cycle = root / "runs/campaign/cycle-0000"
            campaign = {"current_cycle": str(cycle), "deadline": 200}
            progress = {"phase": library.PHASE, "timestamp": 100,
                        "neural_seeded_gp": {"completed": 89, "total": 100}}
            gp = {"phase": "running", "timestamp": 100}
            library.write(cycle / "auxiliary/complete.json", {"schema": "parallel.auxiliary_training_complete.v1"})
            def check():
                for name, value in [("campaign-progress", campaign), ("cycle-progress", progress), ("gp-service", gp)]:
                    library.write(root / f"runs/{name}.json", value)
                return library.gate(root, cycle, 100)[0]
            self.assertTrue(check())
            for obj, key, value in [(progress, "timestamp", 94), (gp, "timestamp", 94),
                                    (progress, "phase", "training"), (campaign, "deadline", 100),
                                    (campaign, "current_cycle", str(cycle.parent / "cycle-0001"))]:
                old = obj[key]
                obj[key] = value
                self.assertFalse(check())
                obj[key] = old
            progress["neural_seeded_gp"]["completed"] = 90
            self.assertFalse(check())

    def test_cancellation_reaps_owned_separate_group_and_preserves_unrelated_process(self):
        self.assertEqual(ctypes.CDLL(None).prctl(36, 1, 0, 0, 0), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "child.json"
            code = "import subprocess,time,json,pathlib; p=subprocess.Popen(['sleep','30'],start_new_session=True); pathlib.Path(" + repr(str(path)) + ").write_text(json.dumps({'pid':p.pid})); time.sleep(30)"
            unrelated = subprocess.Popen(["sleep", "30"])
            owned = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            try:
                end = time.monotonic() + 5
                while not path.exists() and time.monotonic() < end:
                    time.sleep(.01)
                child = json.loads(path.read_text())["pid"]
                library.stop_owned(owned)
                self.assertIsNotNone(owned.poll())
                with self.assertRaises(ProcessLookupError):
                    os.kill(child, 0)
                self.assertIsNone(unrelated.poll())
            finally:
                library.stop_owned(owned)
                unrelated.kill()
                unrelated.wait()


if __name__ == "__main__":
    unittest.main()
