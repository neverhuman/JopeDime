import json
from pathlib import Path
import shutil
import tempfile
import unittest

import progress_view

CORE = progress_view.CORE


def body(text):
    """The view without its input-digest table, which intentionally changes with the bytes."""
    head, _, rest = text.partition("## Inputs")
    return head + rest.partition("## Findings and tasks")[2]


class Copy:
    """A throwaway repository holding a copy of the committed core-ip records."""

    def __init__(self, directory):
        self.core = Path(directory) / "docs" / "internal" / "core-ip"
        shutil.copytree(CORE, self.core, ignore=shutil.ignore_patterns("__pycache__"))

    def edit(self, name, change, **dump):
        path = self.core / name
        document = json.loads(path.read_text())
        change(document)
        path.write_text(json.dumps(document, **dump))

    def render(self):
        return progress_view.render(progress_view.collect(self.core))


class ProgressViewTests(unittest.TestCase):
    def test_committed_view_is_current(self):
        text = progress_view.render(progress_view.collect())
        self.assertEqual((CORE / "work-orders" / "PROGRESS.md").read_text(), text)
        self.assertIn("| **Total** |", text)

    def test_regeneration_is_deterministic_and_blind_to_presentation(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Copy(directory)
            first = copy.render()
            self.assertEqual(first, copy.render())

            def restyle(document):
                document["note"] = "renamed presentation note"
            # Key order, indentation and free-text notes are presentation only.
            copy.edit("work-orders/task-status.json", restyle, indent=1, sort_keys=True)
            copy.edit("claims/claims.json", restyle, indent=3, sort_keys=True)
            second = copy.render()
            self.assertNotEqual(first, second)  # the digest table binds the new bytes
            self.assertEqual(body(first), body(second))

    def test_task_states_cannot_close_a_finding(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Copy(directory)

            def prove_all(document):
                for task, item in document["tasks"].items():
                    if task.startswith("W3-"):
                        item.update(states=["reproduced", "implemented", "fixture-proven"],
                                    commits=["a" * 40], tests=["fixture"])
            copy.edit("work-orders/task-status.json", prove_all, indent=2)
            text = copy.render()
            w3 = next(line for line in text.splitlines() if line.startswith("| W3 |"))
            cells = [c.strip() for c in w3.strip("|").split("|")]
            gaps, improvements, open_, closed = map(int, cells[1:5])
            self.assertEqual((open_, closed), (gaps + improvements, 0))
            self.assertIn("- Findings closed: 0 of", text)

    def test_invalid_records_are_refused_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Copy(directory)

            def claim_live(document):
                # A commit alone cannot stand in for the live evidence a live state needs.
                document["tasks"]["W7-T07"].update(states=["live-qualified"], commits=["a" * 40])
            copy.edit("work-orders/task-status.json", claim_live, indent=2)
            with self.assertRaisesRegex(ValueError, "live/measurement states need live evidence"):
                copy.render()

    def test_stale_view_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Copy(directory)
            target = copy.core / "work-orders" / "PROGRESS.md"
            target.write_text(target.read_text().replace("Findings closed: 0", "Findings closed: 1"))
            self.assertNotEqual(target.read_text(), copy.render())


if __name__ == "__main__":
    unittest.main()
