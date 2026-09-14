import tempfile
import unittest
from pathlib import Path

import expansion_assembly as e


class ExpansionAssemblyTests(unittest.TestCase):
    def test_refuses_live_index(self):
        with self.assertRaisesRegex(ValueError, "live"):
            e.refuse_live_index("x/" + e.LIVE_CONTEXTS)

    def test_refuses_symlink_to_live_index(self):
        with tempfile.TemporaryDirectory() as folder:
            live = Path(folder) / e.LIVE_CONTEXTS
            live.parent.mkdir(parents=True)
            live.write_text("protected")
            alias = Path(folder) / "alias.json"
            alias.symlink_to(live)
            with self.assertRaisesRegex(ValueError, "live"):
                e.refuse_live_index(alias)
            self.assertEqual(live.read_text(), "protected")

    def test_legacy_sorted_tail_cannot_reassign_membership(self):
        with self.assertRaisesRegex(ValueError, "retired"):
            e.assign_partitions([f"f{i}" for i in range(540)])

    def test_legacy_emitters_cannot_create_assembly_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "index.json"
            for emit in (lambda: e.canary_v5(output, ["new"]),
                         lambda: e.v6_plan(["new"], output)):
                with self.assertRaisesRegex(ValueError, "retired"):
                    emit()
                self.assertFalse(output.exists())

    def test_boolean_growth_declaration_cannot_authorize_successor(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "successor.json"
            with self.assertRaisesRegex(ValueError, "assignment file"):
                e.successor_from_growth(dict(sha256="ab" * 32,
                    rows=[dict(family="new", partition="meta_train")], n_test=0), output)
            self.assertFalse(output.exists())

    def test_live_families_are_excluded(self):
        kept, skipped = e.exclude_live_families(["old", "new"], ["old"])
        self.assertEqual((kept, skipped), (["new"], ["old"]))


if __name__ == "__main__":
    unittest.main()
