import json
from pathlib import Path
import shutil
import tempfile
import unittest

import task_status
from task_status import ROOT


class TaskStatusTests(unittest.TestCase):
    def test_committed_status_is_complete_and_rendered(self):
        headings, tasks = task_status.verify()
        self.assertEqual(len(tasks), 92)
        self.assertEqual((ROOT / "STATUS.md").read_text(), task_status.render(headings, tasks))

    def test_mutations_are_rejected(self):
        record = json.loads((ROOT / "task-status.json").read_text())
        first = sorted(record["tasks"])[0]
        mutations = {
            "missing task": (lambda r: r["tasks"].pop(first), "missing"),
            "invented task": (lambda r: r["tasks"].__setitem__("W99-T01", {"states": ["planned"]}), "extra"),
            "unknown state": (lambda r: r["tasks"][first].update(states=["done"]), "states"),
            "planned with progress": (lambda r: r["tasks"][first].update(states=["planned", "implemented"]),
                                      "planned stands alone"),
            "progress without evidence": (lambda r: r["tasks"][first].update(
                states=["implemented"], commits=[], evidence=[]), "needs commits or evidence"),
            "abbreviated commit": (lambda r: r["tasks"][first].update(states=["implemented"], commits=["abc123"]),
                                   "full SHAs"),
            "fixture without tests": (lambda r: r["tasks"][first].update(
                states=["fixture-proven"], commits=["a" * 40], tests=[]), "proving tests"),
            "CI without run": (lambda r: r["tasks"][first].update(states=["CI-verified"], commits=["a" * 40]),
                               "run binding"),
            "live without evidence": (lambda r: r["tasks"][first].update(
                states=["live-qualified"], commits=["a" * 40]), "live evidence"),
            "blocked without blocker": (lambda r: r["tasks"][first].update(states=["blocked"], blocker=""),
                                        "blocker"),
        }
        for label, (mutate, reason) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for path in ROOT.glob("W*.tasks.md"):
                    shutil.copy(path, root / path.name)
                changed = json.loads(json.dumps(record))
                mutate(changed)
                (root / "task-status.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, reason):
                    task_status.verify(root)

    def test_rendering_is_deterministic_and_stale_views_fail(self):
        headings, tasks = task_status.verify()
        self.assertEqual(task_status.render(headings, tasks), task_status.render(headings, dict(tasks)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in list(ROOT.glob("W*.tasks.md")) + [ROOT / "task-status.json"]:
                shutil.copy(path, root / path.name)
            (root / "STATUS.md").write_text("hand-edited view\n")
            headings, tasks = task_status.verify(root)
            self.assertNotEqual((root / "STATUS.md").read_text(), task_status.render(headings, tasks))


if __name__ == "__main__":
    unittest.main()
