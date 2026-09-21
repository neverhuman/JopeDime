import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import qualify_regression_gain as q


class QualificationCollectionTests(unittest.TestCase):
    def root(self, folder):
        root = Path(folder)
        (root / "runs").mkdir()
        (root / "trial").mkdir()
        return root

    def status(self, root, **changes):
        value = dict(timestamp=time.time(), phase="running", active=104, queued=316)
        value.update(changes)
        q.atomic(root / "runs/gp-service.json", value)

    def test_empty_or_stale_collection_cannot_request_gpu_handoff(self):
        for changes in (dict(active=0, queued=0), dict(timestamp=0), dict(queued=207)):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as folder:
                root = self.root(folder)
                self.status(root, **changes)
                with self.assertRaises(q.CollectionUnavailable):
                    with q.gpu_handoff(root, root / "trial"):
                        self.fail("acquired GPUs with insufficient collection")
                self.assertFalse((root / "runs/qualification-control.json").exists())

    def test_queue_depletion_during_handoff_releases_production(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.root(folder)
            with patch.object(
                q, "require_collection", side_effect=[None, q.CollectionUnavailable("queue depleted")]
            ):
                with self.assertRaises(q.CollectionUnavailable):
                    with q.gpu_handoff(root, root / "trial"):
                        self.fail("acquired depleted handoff")
            self.assertFalse(q.read(root / "runs/qualification-control.json")["active"])
            self.assertEqual(q.read(root / "trial/interrupted.json")["outcome"], "inconclusive")

    def test_low_queue_stops_owned_work_and_preserves_partial_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.root(folder)
            self.status(root, queued=103)
            partial = root / "trial/partial.json"
            q.atomic(partial, dict(step=7))
            children = []
            with self.assertRaises(q.CollectionUnavailable):
                with q.runtime.owned_ranks(children):
                    children.append(q.runtime.popen([sys.executable, "-c", "import time; time.sleep(30)"]))
                    q.wait_children(root, children, 30)
            self.assertIsNotNone(children[0].returncode)
            self.assertEqual(q.read(partial), dict(step=7))

    def test_reserved_queue_allows_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.root(folder)
            self.status(root)
            q.require_collection(root, minimum=312)
            self.status(root, queued=104)
            q.require_collection(root)


if __name__ == "__main__":
    unittest.main()
