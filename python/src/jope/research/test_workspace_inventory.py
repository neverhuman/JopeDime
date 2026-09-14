import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest

from workspace_inventory import Inventory


class IndexedInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'root'
        self.root.mkdir()
        (self.root / 'original').write_bytes(b'abcd')
        self.database = self.base / 'index.sqlite'

    def open(self, roots=None):
        value = Inventory(self.database, roots or {'project': str(self.root)}, 'synthetic-host',
                          {'source_roots': []})
        self.addCleanup(value.close)
        return value

    def test_resume_export_and_actual_hashes(self):
        index = self.open()
        self.assertEqual(index.step(0)['unvisited_paths'], 1)
        self.assertTrue(index.step(2)['traversal_complete'])
        self.assertFalse(index.summary()['content_inventory_complete'])
        self.assertTrue(index.hash_step(2)['content_inventory_complete'])
        out = self.base / 'manifest.jsonl'
        index.export(out)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(rows[-1]['path'], 'original')
        self.assertEqual(rows[-1]['size'], 4)
        self.assertEqual(len(rows[-1]['sha256']), 64)
        self.assertFalse(rows[0]['synchronized'])
        with self.assertRaises(ValueError):
            index.export(out)

    def test_missing_root_and_symlink_escape_remain_unresolved(self):
        (self.root / 'escape').symlink_to(self.base)
        index = self.open({'project': str(self.root), 'missing': str(self.base / 'missing')})
        result = index.step(2)
        self.assertEqual(result['unresolved_paths'], 2)
        self.assertFalse(result['content_inventory_complete'])
        self.assertEqual(index.db.execute("SELECT COUNT(*) FROM entries WHERE path LIKE 'escape/%'").fetchone()[0], 0)

    def test_same_size_mtime_changed_cannot_reuse_hash(self):
        index = self.open()
        index.step(2)
        path = self.root / 'original'
        info = path.stat()
        path.write_bytes(b'efgh')
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        self.assertEqual(index.hash_step(2)['unresolved_paths'], 1)

    def test_new_path_after_traversal_prevents_complete_manifest(self):
        index = self.open()
        index.step(2)
        (self.root / 'new').write_text('not in captured inventory')
        result = index.hash_step(2)
        self.assertGreater(result['changed_since_capture'], 0)
        self.assertFalse(result['content_inventory_complete'])

    def test_directory_cursor_resumes_without_dropping_children(self):
        import time
        index = self.open()
        for n in range(50):
            (self.root / str(n)).touch()
        with index.db:
            self.assertFalse(index._visit('project', '', time.monotonic() - 1))
        self.assertEqual(index.db.execute('SELECT position FROM directory_cursors').fetchone()[0], 1)
        result = index.step(2)
        self.assertTrue(result['traversal_complete'])
        self.assertEqual(result['files'], 51)

    def test_resume_does_not_depend_on_directory_iteration_order(self):
        import time
        from unittest.mock import patch
        import contextlib
        index = self.open()
        (self.root / 'second').touch()
        entries = list(os.scandir(self.root))
        with patch('workspace_inventory.os.scandir', return_value=contextlib.nullcontext(iter(entries))):
            with index.db:
                self.assertFalse(index._visit('project', '', time.monotonic()-1))
        with patch('workspace_inventory.os.scandir', return_value=contextlib.nullcontext(iter(entries[::-1]))):
            with index.db:
                self.assertTrue(index._visit('project', '', time.monotonic()+2))
        self.assertEqual(index.db.execute("SELECT COUNT(*) FROM queue WHERE path!=''").fetchone()[0], 2)

    def test_history_import_is_idempotent_and_does_not_certify_objects(self):
        index = self.open()
        p = self.base / 'old.json'
        p.write_text(json.dumps(dict(schema='workspace.durable-index.v1',
                                    files={'project:path': {'sha256': 'a' * 64}},
                                    history={'project:path': [{'sha256': 'b' * 64}]})))
        index.import_history(p)
        index.import_history(p)
        self.assertEqual(index.db.execute('SELECT COUNT(*) FROM history').fetchone()[0], 2)
        self.assertFalse(index.summary()['content_inventory_complete'])

    def test_scope_drift_and_recursive_output_refused(self):
        self.open()
        with self.assertRaisesRegex(ValueError, 'scope changed'):
            Inventory(self.database, {'other': str(self.root)}, 'synthetic-host', {'source_roots': []})
        with self.assertRaisesRegex(ValueError, 'outside traversed roots'):
            Inventory(self.root / 'recursive.sqlite', {'project': str(self.root)}, 'synthetic-host', {'source_roots': []})

    def test_owned_wal_snapshot_restores_cursor_and_history_then_resumes(self):
        import time
        index = self.open()
        (self.root / 'next').write_text('next')
        old = self.base / 'history.json'
        old.write_text(json.dumps(dict(schema='workspace.durable-index.v1',
                                      files={'old:path': {'sha256': 'a' * 64}}, history={})))
        index.import_history(old)
        with index.db:
            self.assertFalse(index._visit('project', '', time.monotonic() - 1))
        state = self.base / 'custody'
        snapshot = index.snapshot(state)
        self.assertTrue(snapshot['inventory_state_restored'])
        self.assertFalse(snapshot['authority_recovery_verified'])
        self.assertFalse(snapshot['full_project_boundary'])
        restored_path = self.base / 'restored.sqlite'
        shutil.copyfile(state / 'objects' / snapshot['snapshot']['object']['sha256'], restored_path)
        restored = Inventory(restored_path, {'project': str(self.root)}, 'synthetic-host', {'source_roots': []})
        self.addCleanup(restored.close)
        self.assertEqual(restored._state_identity(restored.db), snapshot['state_sha256'])
        self.assertEqual(restored.db.execute('SELECT COUNT(*) FROM history').fetchone()[0], 1)
        self.assertEqual(restored.db.execute('SELECT COUNT(*) FROM directory_cursors').fetchone()[0], 1)
        self.assertTrue(restored.step(2)['traversal_complete'])
        self.assertTrue(restored.hash_step(2)['content_inventory_complete'])
        self.assertEqual(restored.summary()['files'], 2)

    def test_interrupted_hash_batch_retains_committed_progress(self):
        from unittest.mock import patch
        import workspace_store
        index = self.open()
        for n in range(259):
            (self.root / str(n)).write_text(str(n))
        index.step(5)
        original = workspace_store.stable_hash
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 130:
                raise RuntimeError('synthetic hashing interruption')
            return original(*args, **kwargs)
        with patch('workspace_inventory.store.stable_hash', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                index.hash_step(5)
        self.assertEqual(index.db.execute('SELECT COUNT(*) FROM entries WHERE sha256 IS NOT NULL').fetchone()[0], 128)
        self.assertTrue(index.hash_step(5)['content_inventory_complete'])
        self.assertEqual(index.summary()['files'], 260)


if __name__ == '__main__':
    unittest.main()
