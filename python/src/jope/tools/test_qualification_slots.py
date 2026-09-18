import fcntl
import tempfile
import unittest
from pathlib import Path

import experiment_runtime as runtime
from qualification_slots import acquire, release_stale
from resume_expired_cycle import atomic, read


class SlotTests(unittest.TestCase):
    def test_stale_request_cleanup_preserves_live_owner_and_surviving_child_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.setup_root(folder)
            with acquire(root,root/'trial.json',root/'run',count=1):
                self.assertFalse(release_stale(root,root/'trial.json'))
            request=read(root/'runs/gp-qualification-reserve.json');request['expires_at']=request['created_at']+3600
            atomic(root/'runs/gp-qualification-reserve.json',request)
            with (root/'runs/gp-slot-103.lock').open('a') as survivor:
                fcntl.flock(survivor,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertTrue(release_stale(root,root/'trial.json'))
                with (root/'runs/gp-slot-103.lock').open('a') as replacement:
                    with self.assertRaises(BlockingIOError):fcntl.flock(replacement,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def setup_root(self,folder):
        root=Path(folder);(root/'runs').mkdir()
        atomic(root/'runs/gp-service.json',dict(slot_leases=True,workers=104,threads_each=1))
        atomic(root/'trial.json',dict(frozen=True))
        return root

    def test_qualification_uses_existing_last_sixteen_cores_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.setup_root(folder);before=runtime._lease_fds
            with self.assertRaisesRegex(RuntimeError,'trial interrupted'):
                with acquire(root,root/'trial.json',root/'run',seconds=10) as cpus:
                    self.assertEqual(cpus,tuple(range(106,122)))
                    self.assertEqual(len(runtime._lease_fds),len(before)+16)
                    with (root/'runs/gp-slot-103.lock').open('a') as lease:
                        with self.assertRaises(BlockingIOError):fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    raise RuntimeError('trial interrupted')
            self.assertEqual(runtime._lease_fds,before)
            with (root/'runs/gp-slot-103.lock').open('a') as lease:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_existing_search_is_never_preempted_and_blocks_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.setup_root(folder)
            with (root/'runs/gp-slot-103.lock').open('a') as active:
                fcntl.flock(active,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,'did not release'):
                    with acquire(root,root/'trial.json',root/'run',count=1,wait_seconds=.01):
                        self.fail('active search overlapped qualification')
            self.assertFalse((root/'run/slots-acquired.json').exists())

    def test_old_service_and_overallocation_are_rejected_before_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.setup_root(folder)
            for count in (0,17):
                with self.assertRaises(RuntimeError):
                    with acquire(root,root/'trial.json',root/'run',count=count):pass
            atomic(root/'runs/gp-service.json',dict(workers=104,threads_each=1))
            with self.assertRaisesRegex(RuntimeError,'does not support'):
                with acquire(root,root/'trial.json',root/'run'):pass
            self.assertFalse((root/'runs/gp-qualification-reserve.json').exists())
