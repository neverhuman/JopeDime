"""Adversarial mirror progress tests; synthetic bytes only, no live RPCs."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import workspace_store as store
import workspace_sync as sync
from workspace_inventory import Inventory


class MirrorPagingPeerReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.state = store.directory(self.base / "state")
        # These tests exercise custody/progress using tiny fixtures, not a disk
        # reservation decision for a production transfer.
        capacity = patch.object(store, "ensure_capacity", return_value={})
        capacity.start()
        self.addCleanup(capacity.stop)

    def test_changed_before_capture_queues_the_actual_sealed_version_without_clearing_original(self):
        path = self.root / "changing-file"
        path.write_bytes(b"original observed version")
        index = Inventory(self.base / "index.sqlite", {"workspace": str(self.root)},
                          "synthetic-host", {"source_roots": []})
        self.addCleanup(index.close)
        index.mirror_advance(2)
        original_page = index.mirror_page(limit=1)
        original = original_page["page"]["items"][0]
        actual_bytes = b"stable bytes captured between observation and refresh"
        path.write_bytes(actual_bytes)
        with self.assertRaisesRegex(ValueError, "original version remains pending"):
            index.mirror_capture(self.state, original_page["page_id"], original["version"])
        self.assertEqual(index.mirror_summary()["observed_unacknowledged_versions"], 2)

        # Retire this attempt with a failure receipt; the original version must
        # remain pending and the newly captured bytes must receive their own turn.
        receipt = dict(schema="workspace.inventory-page-custody.v1",
                       configuration_sha256=index.identity, page_id=original_page["page_id"],
                       outcomes=[dict(version=original["version"], status="deferred",
                                      attempted=True, reason="source changed before capture")])
        raw = store.encoded(receipt)
        receipt_sha = hashlib.sha256(raw).hexdigest()
        destination = store.object_path(self.state, receipt_sha)
        store.directory(destination.parent)
        destination.write_bytes(raw)
        status = index.mirror_acknowledge(self.state, original_page["page_id"], receipt_sha)
        self.assertEqual(status["acknowledged_versions"], 0)
        self.assertEqual(status["observed_unacknowledged_versions"], 2)
        actual_page = index.mirror_page(limit=1)
        actual = actual_page["page"]["items"][0]
        self.assertNotEqual(actual["version"], original["version"])

        # The live pathname no longer contains those bytes. Retrieval must use
        # the independently queued sealed object, without resealing the new file.
        path.write_bytes(b"another replacement after the already captured version")
        captured = index.mirror_capture(self.state, actual_page["page_id"], actual["version"])
        self.assertEqual(captured["sha256"], hashlib.sha256(actual_bytes).hexdigest())
        self.assertEqual(store.object_path(self.state, captured["sha256"]).read_bytes(), actual_bytes)
        self.assertEqual(index.mirror_summary()["observed_unacknowledged_versions"], 2)

    def test_restarted_controller_rotates_scopes_when_each_window_allows_one_attempt(self):
        attempted = []
        clock = {"now": 0.0, "deadline": 1.0}

        def envelopes(sequence):
            result = {}
            for origin in ("local", "remote"):
                page = dict(schema="workspace.inventory-page.v1", configuration_sha256=origin,
                            sequence=sequence, items=[dict(version=origin + "-pending-version")])
                result[origin] = dict(page_id=store.identity(page), page=page, rpc_scope={},
                                      summary=dict(files_remaining=None, acknowledged_versions=0))
            return result

        def rpc(operation, remote=False, **kwargs):
            if operation == "inventory_capture":
                attempted.append("remote" if remote else "local")
                clock["now"] = clock["deadline"] + 0.01
                raise RuntimeError("synthetic bounded attempt; original stays pending")
            if operation == "inventory_ack":
                return dict(files_remaining=None, acknowledged_versions=0)
            self.fail("unexpected RPC: " + operation)

        for sequence in (1, 2):
            # Timer invocations reconstruct the controller. Fairness cannot
            # depend on an in-memory cursor that disappears between invocations.
            mirror = sync.Mirror.__new__(sync.Mirror)
            mirror.state = self.state
            mirror.config = {"local": {}, "remote": {}}
            clock.update(now=float(sequence * 10), deadline=float(sequence * 10 + 1))
            with patch.object(sync.time, "monotonic", side_effect=lambda: clock["now"]), \
                    patch.object(mirror, "healthy"), \
                    patch.object(mirror, "rpc", side_effect=rpc), \
                    patch.object(mirror, "transfer", return_value=0):
                result = mirror.durable(envelopes(sequence), clock["deadline"])
            self.assertEqual(result["verified_versions_this_pass"], 0)
            self.assertEqual(result["deferred_versions_this_pass"], 2)
        self.assertEqual(len(attempted), 2)
        self.assertEqual(set(attempted), {"local", "remote"})


if __name__ == "__main__":
    unittest.main()
