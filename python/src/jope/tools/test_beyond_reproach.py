import json
import tempfile
import unittest
from pathlib import Path

import beyond_reproach as b
import closed_loop as loop
import useful_hour
from selfplay_milestones import world_class


class BeyondReproachTests(unittest.TestCase):
    def test_current_rates_are_not_world_class(self):
        rates = dict(neural_seed_rate=0.0, final_gp_rate=2 / 140, final_gp_margin_rate=0.0)
        self.assertTrue(b.refuse_world_class(rates))
        self.assertFalse(world_class(rates))

    def test_recovery_train_stays_closed(self):
        refused, reason = b.refuse_recovery_train(corpus_is_recovery=True)
        self.assertTrue(refused)
        self.assertIn("native funded task", reason)
        ok, _ = loop.launch_train_allowed("held", True, True, 3, corpus_is_recovery=True)
        self.assertFalse(ok)

    def test_useful_hour_without_committed_outputs(self):
        self.assertTrue(b.refuse_useful_hour({"declared": True}))
        self.assertFalse(useful_hour.committed(["gen-054/propose"]))

    def test_flywheel_requires_p1_and_p2(self):
        now = b.flywheel_effect(False, False, False, False, "unavailable")
        self.assertFalse(now["flywheel_effect"])
        self.assertIn("p1_not_accepted", now["blocked_by"])
        self.assertIn("p2_no_successor_discoveries", now["blocked_by"])
        self.assertEqual(now["vs_random"], "unavailable")
        self.assertFalse(now["train_launched"])
        self.assertFalse(now["serving_written"])
        closed = b.flywheel_effect(True, True, True, True, 0.2)
        self.assertTrue(closed["flywheel_effect"])
        self.assertFalse(closed["blocked_by"])

    def test_register_does_not_overwrite_audit_and_keeps_items_open(self):
        with tempfile.TemporaryDirectory() as folder:
            audit = Path(folder) / "gap-register.json"
            original = json.loads(b.AUDIT.read_text())
            audit.write_text(json.dumps(original))
            before = audit.read_bytes()
            dest = Path(folder) / "out"
            summary = b.emit(dest, audit)
            self.assertEqual(audit.read_bytes(), before)
            self.assertFalse(summary["flywheel_effect"])
            self.assertFalse(summary["launch_train_allowed"])
            reg = json.loads((dest / "register.json").read_text())
            self.assertTrue(reg["source_not_overwritten"])
            self.assertGreaterEqual(reg["n_open"], len(original["items"]))
            self.assertEqual(reg["n_closed"], 0)
            ids = {row["id"] for row in reg["items"]}
            self.assertTrue({"S01", "R06", "D01", "L01"} <= ids)
            self.assertIn("stub verify_execution_protocol",
                          next(r["fake_close"] for r in reg["items"] if r["id"] == "R06"))
            required = ("id", "owner", "would_close", "fake_close", "evidence_sha256")
            for row in reg["items"]:
                self.assertIn("closed", row)
                for key in required:
                    self.assertTrue(row.get(key) not in (None, ""), msg=f"{row.get('id')} missing {key}")
            self.assertFalse(next(r["closed"] for r in reg["items"] if r["id"] == "S01"))


if __name__ == "__main__":
    unittest.main()
