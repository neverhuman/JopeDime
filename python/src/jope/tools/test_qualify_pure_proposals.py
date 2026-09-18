import importlib.util
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("qualification", Path(__file__).with_name("qualify_pure_proposals.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QualificationTests(unittest.TestCase):
    def test_frozen_context_count_adapts_without_opening_sealed_or_extra_cuts(self):
        contexts=[dict(dataset=str(i),cut=cut,partition='meta_train') for i in range(70) for cut in range(2)]
        contexts.extend([dict(dataset='sealed',cut=0,partition='meta_test'),
                         dict(dataset='extra',cut=2,partition='meta_train')])
        self.assertEqual(len(module.proposal_contexts(dict(contexts=contexts))),140)
        with self.assertRaises(ValueError):module.proposal_contexts(dict(contexts=contexts+[contexts[0]]))
        with self.assertRaises(ValueError):module.proposal_contexts(dict(contexts=[]))

    def ranking(self):
        rows = [{"program": f"program-{i}", "program_sha256": module.sha(f"program-{i}".encode()),
                 "predicted_quality": 1.0, "predicted_runtime_seconds": 2.0,
                 "reward_heads": [1.0] * 4, "token_count": 5} for i in range(64)]
        rows.sort(key=lambda row: row["program_sha256"])
        return {"schema":"parallel.proposal_quality_ranking.v1", "selection_id":"mean_reward_heads_desc_program_hash_v1",
                "bindings":{"checkpoint_sha256":"c", "conditioning_context_sha256":"x", "vocabulary_sha256":"v"},
                "pool_size":64, "chosen_count":64, "ranked":rows,
                "pool_sha256":module.sha(module.compact([row["program_sha256"] for row in rows])),
                "training_context_only":True, "heldout_labels_read":False, "gp_scores_read":False}

    def validate(self, ranking):
        module.validate_ranking(ranking, "c", "x", "v", 96)

    def test_finite_tie_ordered_ranking_passes(self):
        self.validate(self.ranking())

    def test_duplicate_program_and_non_deterministic_tie_order_are_rejected(self):
        for change in ["duplicate", "order", "program", "binding"]:
            with self.subTest(change=change):
                ranking = self.ranking()
                if change == "duplicate": ranking["ranked"][1] = ranking["ranked"][0]
                if change == "order": ranking["ranked"][0], ranking["ranked"][1] = ranking["ranked"][1], ranking["ranked"][0]
                if change == "program": ranking["ranked"][0]["program"] = "changed"
                if change == "binding": ranking["bindings"]["conditioning_context_sha256"] = "changed"
                with self.assertRaises(ValueError): self.validate(ranking)

    def test_invalid_values_sequence_limits_and_scope_are_rejected(self):
        for field, value in [("predicted_quality", float("nan")), ("predicted_runtime_seconds", -1),
                             ("reward_heads", [float("inf")] * 4), ("token_count", 97)]:
            ranking = self.ranking(); ranking["ranked"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(ranking)
        ranking = self.ranking(); ranking["heldout_labels_read"] = True
        with self.assertRaises(ValueError): self.validate(ranking)

    def test_timeout_kills_owned_separate_process_group_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "timeout.log"
            neighbour = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                code = "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); print(child.pid,flush=True); time.sleep(30)"
                with self.assertRaises(subprocess.TimeoutExpired):
                    module.run_command([sys.executable, "-c", code], log, time.monotonic() + 5, timeout=0.2)
                child = int(log.read_text().strip())
                status = Path(f"/proc/{child}/status")
                try:
                    state = next(line for line in status.read_text().splitlines() if line.startswith("State:"))
                except FileNotFoundError:
                    pass
                else:
                    self.assertIn(state.split()[1], ("Z", "X"))
                self.assertIsNone(neighbour.poll())
            finally:
                neighbour.kill(); neighbour.wait()


if __name__ == "__main__":
    unittest.main()
