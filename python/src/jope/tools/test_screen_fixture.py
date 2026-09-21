import tempfile
import unittest
from pathlib import Path

import screen_fixture as s


class ScreenFixtureTests(unittest.TestCase):
    def test_refuses_open_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                s.freeze({"unverified": False, "shares": []}, Path(folder) / "m.json", payloads_opened=True)

    def test_freeze_has_axes_and_no_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            census = dict(
                unverified=False,
                observations=10,
                files=1,
                shares=[
                    dict(head="histgbm2", observations=6, unique_programs=4),
                    dict(head="gpr", observations=4, unique_programs=3),
                ],
            )
            manifest = s.freeze(census, Path(folder) / "manifest.json")
            self.assertEqual(manifest["schema"], "parallel.screen_fixture.v2")
            self.assertFalse(manifest["payloads_opened"])
            self.assertEqual(manifest["new_fits"], 0)
            self.assertFalse(manifest["teaching_admission"])
            self.assertEqual(manifest["axes"][0]["name"], "same_row_f32")
            self.assertEqual(manifest["axes"][1]["name"], "histgbm_proxy")
            self.assertFalse(manifest["qualification_ready"])
            self.assertEqual(manifest["gates"]["retained_by_screen"], 16)
            self.assertEqual(manifest["gates"]["deterministic_rejection_audits"], 5)
            self.assertEqual(manifest["gates"]["missed_champion"], 0)
            self.assertEqual(len(manifest["strata"]), 2)
            again = s.freeze(census, Path(folder) / "manifest.json")
            self.assertEqual(again["manifest_sha256"], manifest["manifest_sha256"])

    def test_old_fixture_or_symlink_requires_a_new_successor(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fixture.json"
            target.write_bytes(b'{"schema":"parallel.screen_fixture.v1"}\n')
            with self.assertRaisesRegex(ValueError, "successor"):
                s.freeze({}, target)
            self.assertEqual(target.read_bytes(), b'{"schema":"parallel.screen_fixture.v1"}\n')
            alias = Path(folder) / "alias.json"
            alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                s.freeze({}, alias)


if __name__ == "__main__":
    unittest.main()
