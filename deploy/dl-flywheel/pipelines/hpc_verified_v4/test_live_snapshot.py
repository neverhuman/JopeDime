"""Live integration regression: a writer advances while custody reads its anchor."""
import unittest
from unittest.mock import patch
from test_native_failure import FailureTests
from neural_outbox import Outbox
from native_failure import VerifiedReadSnapshot


class LiveSnapshotTests(FailureTests):
    def test_concurrent_append_keeps_one_physical_snapshot(self):
        identity = self.emit(0)
        before = self.o.verify()
        verifier = Outbox.verify
        appended = []

        def advance_writer(reader):
            result = verifier(reader)
            self.assertIsInstance(reader.db, VerifiedReadSnapshot)
            self.assertTrue(reader.db.db.in_transaction)
            appended.append(self.emit(1))
            return result

        with patch.object(Outbox, 'verify', advance_writer):
            anchored = self.snap(identity)
        self.assertEqual(len(appended), 1)
        self.assertEqual(anchored['head'], before)
        self.assertEqual(self.o.summary()['observations'], 2)
        self.assertEqual(self.o.verify()['observations'], 2)

    def test_snapshot_adapter_refuses_mutation(self):
        wrapper = VerifiedReadSnapshot(self.o.db)
        for sql in ('UPDATE emissions SET state="recorded"', 'DELETE FROM events',
                    'PRAGMA query_only=OFF', 'ROLLBACK'):
            with self.assertRaises(ValueError):
                wrapper.execute(sql)


if __name__ == '__main__':
    unittest.main()
