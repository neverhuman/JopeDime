import fcntl
import hashlib
import mmap
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from resident_gpu_pool import idle_worker
from shared_teaching_cache import SEALS, Cache, receive_request, report_cache


class SharedCacheTests(unittest.TestCase):
    def test_nonobject_and_fifo_fail_before_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = Cache(root, 32)
            with self.assertRaisesRegex(RuntimeError, "object"):
                cache.get([])
            os.mkfifo(root / "fifo")
            with self.assertRaisesRegex(RuntimeError, "regular file"):
                cache.get(dict(path=str(root / "fifo"), sha256="a" * 64, bytes=4))
            self.assertEqual(cache.bytes, 0)
            client, server = socket.socketpair()
            try:
                server.sendall(b"[]\n")
                with self.assertRaisesRegex(RuntimeError, "object"):
                    receive_request(client)
            finally:
                client.close()
                server.close()

    def test_partial_cache_request_cannot_renew_handshake_deadline(self):
        client, server = socket.socketpair()

        def send_partial():
            try:
                for _ in range(10):
                    server.sendall(b"x")
                    time.sleep(0.01)
            except OSError:
                pass

        thread = threading.Thread(target=send_partial)
        thread.start()
        try:
            started = time.monotonic()
            with self.assertRaises((RuntimeError, TimeoutError)):
                receive_request(client, seconds=0.035)
            self.assertLess(time.monotonic() - started, 1)
        finally:
            client.close()
            thread.join()
            server.close()

    def test_expired_payload_deadline_rejects_before_cache_population(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Cache(Path(directory), 32)
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                cache.get(dict(deadline_unix_ms=1))
            self.assertEqual(cache.bytes, 0)

    def test_request_volume_does_not_fsync_telemetry_on_critical_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = Cache(root, 32)
            with patch("shared_teaching_cache.atomic") as publish:
                deadline = report_cache(cache, root, 0, 100)
                for i in range(10000):
                    cache.hits += 1
                    deadline = report_cache(cache, root, deadline, 100 + i / 1000)
                self.assertEqual(publish.call_count, 1)
                deadline = report_cache(cache, root, deadline, 110)
                self.assertEqual(publish.call_count, 2)
                self.assertEqual(publish.call_args.args[1]["hits"], 10000)

    def test_sealed_identity_is_reused_and_restart_reverifies_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "records"
            path.write_bytes(b"original")
            request = dict(path=str(path), sha256=hashlib.sha256(b"original").hexdigest(), bytes=8)
            cache = Cache(root, 32)
            try:
                fd = cache.get(request)
                self.assertEqual(fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS, SEALS)
                with mmap.mmap(fd, 8, access=mmap.ACCESS_READ) as mapped:
                    with self.assertRaises(OSError):
                        os.pwrite(fd, b"x", 0)
                    with self.assertRaises(OSError):
                        os.ftruncate(fd, 1)
                    path.write_bytes(b"tampered")
                    self.assertEqual(cache.get(request), fd)
                    self.assertEqual(mapped[:], b"original")
                self.assertEqual((cache.verifications, cache.hits), (1, 1))
            finally:
                cache.close()
            restarted = Cache(root, 32)
            try:
                with self.assertRaisesRegex(RuntimeError, "content changed"):
                    restarted.get(request)
                self.assertFalse(restarted.entries)
            finally:
                restarted.close()

    def test_pressure_retains_lent_object_accounting_and_rejects_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = Cache(root, 4)

            def request(name, data):
                path = root / name
                path.write_bytes(data)
                return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

            lent = os.dup(cache.get(request("first", b"abcd")))
            try:
                with mmap.mmap(lent, 4, access=mmap.ACCESS_READ) as mapped:
                    with patch("shared_teaching_cache.os.memfd_create") as allocate:
                        for name in ["second", "third"]:
                            with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
                                cache.get(request(name, b"efgh"))
                        allocate.assert_not_called()
                    self.assertEqual(mapped[:], b"abcd")
                    self.assertEqual(cache.bytes, 4)
                    self.assertEqual(len(cache.entries), 1)
                    self.assertEqual(cache.verifications, 1)
                # Closing one borrowed mapping does not prove other ranks have
                # released theirs. Accounting remains retained until shutdown.
                with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
                    cache.get(request("after-close", b"ijkl"))
                with self.assertRaisesRegex(RuntimeError, "outside workspace"):
                    cache.get(dict(path="/etc/hostname", sha256="a" * 64, bytes=4))
            finally:
                os.close(lent)
                cache.close()

    def test_only_current_idle_process_authorizes_pool_handoff(self):
        self.assertFalse(idle_worker(dict(pid=42, status="idle"), 42))
        self.assertFalse(idle_worker(dict(pid=41, status="idle"), 42))
        self.assertFalse(idle_worker(dict(pid=42, status="busy"), 42))


if __name__ == "__main__":
    unittest.main()
