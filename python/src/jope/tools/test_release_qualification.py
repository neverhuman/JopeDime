import fcntl
import tempfile
import unittest
from pathlib import Path

from release_qualification import release
from resume_expired_cycle import atomic, read


class ReleaseTests(unittest.TestCase):
    def test_cleanup_preserves_another_owner_and_cannot_unlock_surviving_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runs").mkdir()
            output = root / "runs/trial"
            control = root / "runs/qualification-control.json"
            atomic(control, dict(active=True, output="runs/another"))
            release(root, output)
            self.assertTrue(read(control)["active"])
            atomic(control, dict(active=True, output="runs/trial/gpu-handoff"))
            with (root / "runs/encoding-service.lock").open("a") as child:
                fcntl.flock(child, fcntl.LOCK_EX | fcntl.LOCK_NB)
                release(root, output)
                self.assertFalse(read(control)["active"])
                with (root / "runs/encoding-service.lock").open("a") as newcomer:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(newcomer, fcntl.LOCK_EX | fcntl.LOCK_NB)
