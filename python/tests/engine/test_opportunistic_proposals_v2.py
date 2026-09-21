import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from jope.tools import opportunistic_proposals_v2 as library
from jope.tools import test_report_cycle as fixture_module


class ProposalLibraryTests(unittest.TestCase):
    def test_pool_hashes_and_restore_survive_closed_admission(self):
        context_bytes = bytes(8192)
        context_sha = fixture_module.digest(context_bytes)
        prepared = {"cuts": [{"index": 0, "training_sha256": "e" * 64}]}
        prepared_bytes = json.dumps(prepared).encode()
        prepared_sha = fixture_module.digest(prepared_bytes)

        def bind_saved_inputs(v):
            v["gen"]["conditioning_context_sha256"] = context_sha
            v["assembly"]["conditioning_context_sha256"] = context_sha
            for name in ("pure", "gp"):
                v[name]["ranking"]["bindings"]["conditioning_context_sha256"] = context_sha
            v["best"].update(
                prepared_sha256=prepared_sha,
                conditioning_context_sha256=context_sha,
                training_context_sha256=context_sha,
            )
            v["seeds"].update(prepared_sha256=prepared_sha, context_sha256=context_sha)

        path, gen, seeds, _, _, _, _ = fixture_module.ProposalProtocolReportTests.fixture(
            self, bind_saved_inputs
        )
        root = path.parents[3]
        (root / "context.f32").write_bytes(context_bytes)
        (root / "prepared").mkdir()
        (root / "prepared/prepared.json").write_bytes(prepared_bytes)
        library.write(path / "generation.json", gen)
        contexts = {
            ("dataset", 0): {
                "dataset": "dataset",
                "cut": 0,
                "partition": "meta_train",
                "sha256": context_sha,
                "path": "context.f32",
                "prepared": "prepared",
                "prepared_sha256": prepared_sha,
            }
        }
        snapshot = library.digest(root / "cycle/champion-snapshot.json")
        pool = library.validate_pool(root, path, "a" * 64, snapshot, contexts)
        self.assertEqual(pool["teacher_overlaps"], 1)
        self.assertEqual(pool["overlap_refills"], 1)
        binding = {"spec_sha256": "spec", "script_sha256": "script", "pool_validator_sha256": "validator"}
        library.write(
            root / "library-index.json",
            {"schema": "parallel.pure_proposal_library_index.v1", **binding, "pools": [pool], "chunks": []},
        )
        restored, chunks = library.restore_library(root, root, "a" * 64, binding, contexts)
        self.assertEqual(len(restored), 1)
        self.assertEqual(chunks, [])
        self.assertFalse(library.gate(root, root / "absent-cycle", 100)[0])
        self.assertEqual(len(restored), 1)
        (path / "gp-ranking.json").write_text("tampered")
        with self.assertRaises(ValueError):
            library.restore_library(root, root, "a" * 64, binding, contexts)

    def test_full_startup_with_closed_gate_publishes_bound_progress_without_gpu(self):
        import signal

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, checkpoint = root / "corpus", root / "checkpoint"
            library.write(
                corpus / "index.json", {"contexts": [{"dataset": "d", "cut": 2, "partition": "meta_train"}]}
            )
            library.write(corpus / "vocabulary.json", {})
            library.write(checkpoint / "model.ot", {})
            library.write(
                checkpoint / "state.json",
                {
                    "corpus_sha256": library.digest(corpus / "index.json"),
                    "model_sha256": library.digest(checkpoint / "model.ot"),
                    "vocabulary_sha256": library.digest(corpus / "vocabulary.json"),
                },
            )
            executable = root / "executable"
            executable.write_bytes(b"never executed")
            spec = {
                "schema": "parallel.opportunistic_proposals_spec.v2",
                "chunk_cuts": 8,
                "block_seconds": 60,
                "service_seconds": 1200,
                "cut_start": 2,
                "cut_end": 10,
                "cycle": "runs/campaign/cycle-0001",
                "corpus": "corpus",
                "checkpoint": "checkpoint",
                "output": "runs/gp-reviewed3/pure-test",
                "executable": "executable",
                "corpus_sha256": library.digest(corpus / "index.json"),
                "checkpoint_sha256": library.digest(checkpoint / "state.json"),
                "executable_sha256": library.digest(executable),
                "pool_validator_sha256": library.digest(
                    Path(library.__file__).parent / "proposal_pool_v2.py"
                ),
                "pure_prediction_protocol": library.PURE_PROTOCOL,
                "gp_assembly_protocol": library.ASSEMBLY_PROTOCOL,
            }
            library.write(root / "spec.json", spec)
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                library.run(root, root / "spec.json")
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
            progress = library.read(root / "runs/pure-proposal-library.json")
            self.assertEqual(progress["phase"], "stopped")
            self.assertEqual(progress["planned_seed_pools"], 1)
            self.assertEqual(progress["saved_seed_pools"], 0)
            self.assertEqual(progress["gp_jobs_admitted"], 0)
            self.assertEqual(progress["encodings_recomputed"], 0)
            self.assertFalse((root / "runs/proposal-library.json").exists())
            self.assertTrue((root / spec["output"] / "wrapper-complete.json").exists())

    def test_gate_rejects_stale_wrong_cycle_phase_deadline_and_ninety_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cycle = root / "runs/campaign/cycle-0001"
            campaign = {"current_cycle": str(cycle), "deadline": 200}
            progress = {
                "phase": library.PHASE,
                "timestamp": 100,
                "neural_seeded_gp": {"completed": 89, "total": 100},
            }
            gp = {"phase": "running", "timestamp": 100}
            library.write(
                cycle / "auxiliary/complete.json", {"schema": "parallel.auxiliary_training_complete.v1"}
            )

            def check():
                for name, value in [
                    ("campaign-progress", campaign),
                    ("cycle-progress", progress),
                    ("gp-service", gp),
                ]:
                    library.write(root / f"runs/{name}.json", value)
                return library.gate(root, cycle, 100)[0]

            self.assertTrue(check())
            for obj, key, value in [
                (progress, "timestamp", 94),
                (gp, "timestamp", 94),
                (progress, "phase", "training"),
                (campaign, "deadline", 100),
                (campaign, "current_cycle", str(cycle.parent / "cycle-0002")),
            ]:
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
            code = (
                "import subprocess,time,json,pathlib; p=subprocess.Popen(['sleep','30'],start_new_session=True); pathlib.Path("
                + repr(str(path))
                + ").write_text(json.dumps({'pid':p.pid})); time.sleep(30)"
            )
            unrelated = subprocess.Popen(["sleep", "30"])
            owned = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            try:
                end = time.monotonic() + 5
                while not path.exists() and time.monotonic() < end:
                    time.sleep(0.01)
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
