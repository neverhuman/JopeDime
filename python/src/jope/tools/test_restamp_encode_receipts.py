import unittest

import restamp_encode_receipts as r


class RestampTests(unittest.TestCase):
    def test_replaces_stale_lock_diagnosis(self):
        out = r.restamp({"apply_contexts_blocked_by": r.STALE, "assembled_contexts": 0})
        self.assertEqual(out["apply_contexts_blocked_by"], r.REAL_BLOCKER)
        self.assertFalse(out["lock_held"])
        self.assertEqual(out["lock_diagnosis"], "stale")
        self.assertFalse(out["writes_live_current_contexts"])

    def test_keeps_real_lock_if_held(self):
        out = r.restamp({"apply_contexts_blocked_by": r.STALE}, lock_held=True)
        self.assertTrue(out["lock_held"])


    def test_restamp_tree_does_not_write_the_lock(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "src"
            src.mkdir()
            (src / "receipt.json").write_text('{"apply_contexts_blocked_by": "encoding-service.lock"}\n')
            out = r.restamp_tree(src, Path(folder) / "dst")
            self.assertEqual(out["n"], 1)
            self.assertFalse(out["writes_live_current_contexts"])


if __name__ == "__main__":
    unittest.main()
