import tempfile
import unittest
from pathlib import Path

from eval_registry import EvalRegistry
from test_eval_registry import full_admission, full_execution


class ShadowRegistryTests(unittest.TestCase):
    def test_shadow_replays_clean_rows_without_authorizing_a_new_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            reg = EvalRegistry(Path(folder) / "shadow.sqlite")
            a, e = full_admission(), full_execution()
            first = reg.shadow_replay([
                dict(admission=a, execution=e, result={"rmse": 0.0}, primary_score=0.0),
            ])
            self.assertEqual(first["replayed"], 1)
            self.assertFalse(first["wired_to_live_gp"])
            self.assertFalse(first["claimed_savings"])
            again = reg.shadow_replay([
                dict(admission=a, execution=e, result={"rmse": 0.0}, primary_score=0.0),
            ])
            self.assertEqual(again["replayed"], 0)
            self.assertGreaterEqual(again["registry_hit"], 1)
            partial = reg.shadow_replay([
                dict(admission={"program_sha256": "abc"}, execution={}, result={"rmse": 1}, primary_score=1),
            ])
            self.assertGreaterEqual(partial["tombstones"], 1)


if __name__ == "__main__":
    unittest.main()
