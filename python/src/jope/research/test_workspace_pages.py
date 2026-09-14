import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import workspace_store as store
from workspace_inventory import Inventory
from workspace_sync import Mirror


class BoundedMirrorPagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'source'
        self.root.mkdir()
        self.state = store.directory(self.base / 'state')
        self.database = self.base / 'inventory.sqlite'
        self.capacity = patch.object(store, 'ensure_capacity', return_value={})
        self.capacity.start()
        self.addCleanup(self.capacity.stop)

    def index(self):
        return Inventory(self.database, {'workspace': str(self.root)}, 'synthetic-host', {'source_roots': []})

    def receipt(self, index, envelope, outcomes):
        value = dict(schema='workspace.inventory-page-custody.v1',
                     configuration_sha256=index.identity, page_id=envelope['page_id'], outcomes=outcomes)
        raw = store.encoded(value)
        digest = hashlib.sha256(raw).hexdigest()
        path = store.object_path(self.state, digest)
        store.directory(path.parent)
        path.write_bytes(raw)
        return digest

    def test_bounded_page_repeats_until_acknowledged_after_restart(self):
        for i in range(180):
            (self.root / ('file-' + str(i))).write_bytes(b'synthetic')
        index = self.index()
        index.mirror_advance(2)
        page = index.mirror_page(limit=7)
        self.assertEqual(len(page['page']['items']), 7)
        self.assertLessEqual(len(store.encoded(page['page'])), 262144)
        self.assertEqual(page['summary']['observed_unacknowledged_versions'], 180)
        self.assertIsNone(page['summary']['files_remaining'])
        index.close()
        index = self.index()
        self.addCleanup(index.close)
        repeated = index.mirror_page(limit=7)
        self.assertEqual(page['page_id'], repeated['page_id'])
        self.assertEqual(page['page'], repeated['page'])
        outcomes = [dict(version=item['version'], status='deferred', attempted=True, reason='interrupted transport')
                    for item in page['page']['items']]
        digest = self.receipt(index, page, outcomes)
        status = index.mirror_acknowledge(self.state, page['page_id'], digest)
        self.assertEqual(status['observed_unacknowledged_versions'], 180)
        next_page = index.mirror_page(limit=7)
        # A bounded retry share keeps failed originals progressing even while
        # new observations continuously arrive. The remaining six are fresh.
        self.assertEqual(len(set(x['version'] for x in next_page['page']['items'])
                             & set(x['version'] for x in page['page']['items'])), 1)
        self.assertEqual(len(next_page['page']['items']), 7)
        self.assertEqual(index.mirror_acknowledge(self.state, page['page_id'], digest)['observed_unacknowledged_versions'], 180)

    def test_captured_original_survives_source_change_and_lost_ack(self):
        source = self.root / 'file'
        source.write_bytes(b'original')
        index = self.index()
        index.mirror_advance(2)
        envelope = index.mirror_page()
        item = envelope['page']['items'][0]
        captured = index.mirror_capture(self.state, envelope['page_id'], item['version'])
        index.close()
        source.write_bytes(b'changed after original capture')
        index = self.index()
        self.addCleanup(index.close)
        self.assertEqual(index.mirror_capture(self.state, envelope['page_id'], item['version']), captured)
        outcomes = [dict(version=item['version'], status='verified_on_both_hosts', record=captured)]
        digest = self.receipt(index, envelope, outcomes)
        self.assertEqual(index.mirror_acknowledge(self.state, envelope['page_id'], digest)['acknowledged_versions'], 1)
        index.mirror_advance(2)
        self.assertGreater(index.mirror_summary()['observed_unacknowledged_versions'], 0)
        self.assertGreater(index.db.execute('SELECT COUNT(*) FROM history').fetchone()[0], 0)

    def test_changed_before_capture_and_forged_ack_never_clear_original(self):
        path = self.root / 'file'
        path.write_bytes(b'before')
        index = self.index()
        self.addCleanup(index.close)
        index.mirror_advance(2)
        envelope = index.mirror_page()
        item = envelope['page']['items'][0]
        path.write_bytes(b'after')
        with self.assertRaisesRegex(ValueError, 'original version remains pending'):
            index.mirror_capture(self.state, envelope['page_id'], item['version'])
        digest = self.receipt(index, envelope, [dict(version=item['version'], status='verified_on_both_hosts', record={})])
        with self.assertRaisesRegex(ValueError, 'captured original'):
            index.mirror_acknowledge(self.state, envelope['page_id'], digest)
        self.assertEqual(index.mirror_summary()['observed_unacknowledged_versions'], 2)
        self.assertEqual(index.mirror_page()['page_id'], envelope['page_id'])

    def test_directory_cookie_progresses_under_tiny_windows_and_recovers_mutation(self):
        for i in range(40):
            (self.root / ('file-' + str(i))).touch()
        index = self.index()
        self.addCleanup(index.close)
        counts = []
        for _ in range(12):
            with index.db:
                index._mirror_visit('workspace', '', time.monotonic() - 1)
            counts.append(index.db.execute('SELECT COUNT(*) FROM queue').fetchone()[0])
        self.assertGreater(counts[-1], counts[2])
        (self.root / 'added-during-partial-traversal').write_bytes(b'new')
        for _ in range(60):
            with index.db:
                if index._mirror_visit('workspace', '', time.monotonic() - 1):
                    break
        else:
            self.fail('quiescent directory failed to finish after cursor invalidation')
        self.assertEqual(index.db.execute('SELECT COUNT(*) FROM queue').fetchone()[0], 42)
        self.assertGreater(index.db.execute("SELECT COUNT(*) FROM history WHERE origin='invalidated_directory_cookie'").fetchone()[0], 0)

    def test_missing_root_recovers_without_erasing_original_unknown(self):
        self.root.rmdir()
        index = self.index()
        self.addCleanup(index.close)
        self.assertEqual(index.mirror_advance(2)['unresolved_paths'], 1)
        self.root.mkdir()
        (self.root / 'recovered').write_bytes(b'evidence')
        index.mirror_advance(2)
        self.assertEqual(index.mirror_summary()['observed_unacknowledged_versions'], 1)
        self.assertGreater(index.db.execute('SELECT COUNT(*) FROM history').fetchone()[0], 0)
        self.assertFalse(index.mirror_summary()['content_inventory_complete'])

    def test_page_size_limits_and_exact_ack_membership(self):
        (self.root / 'file').write_bytes(b'data')
        index = self.index()
        self.addCleanup(index.close)
        index.mirror_advance(2)
        for kwargs in ({'limit': 129}, {'limit': True}, {'byte_limit': 262145}):
            with self.assertRaises(ValueError):
                index.mirror_page(**kwargs)
        envelope = index.mirror_page()
        digest = self.receipt(index, envelope, [])
        with self.assertRaisesRegex(ValueError, 'exact original page'):
            index.mirror_acknowledge(self.state, envelope['page_id'], digest)
        self.assertEqual(index.mirror_summary()['observed_unacknowledged_versions'], 1)

    def test_controller_uses_bounded_pages_and_never_reads_legacy_catalog(self):
        mirror = Mirror.__new__(Mirror)
        mirror.state = self.state
        mirror.config = {'local': {}, 'remote': {}}
        legacy = self.state / 'inventory-local.json'
        legacy.write_bytes(b'legacy unparsed bytes')
        calls = []
        def rpc(operation, **kwargs):
            calls.append(operation)
            self.assertEqual(operation, 'inventory_page')
            return dict(page_id='a'*64, page={'items': []}, summary={'files_remaining': None})
        with patch.object(mirror, 'rpc', side_effect=rpc):
            mirror.inventory(incremental=True)
            mirror.inventory(incremental=False)
        self.assertEqual(calls, ['inventory_page'] * 4)
        self.assertEqual(legacy.read_bytes(), b'legacy unparsed bytes')

    def test_remote_rpc_executes_bundled_indexed_module_without_installed_imports(self):
        (self.root / 'file').write_bytes(b'bounded metadata fixture')
        config = dict(policy={'source_roots': []}, local=dict(root=str(self.root), state=str(self.state), name='local'),
                      remote=dict(root=str(self.root), state=str(self.base / 'remote-state'),
                                  name='remote', key='/synthetic/no-key', host='synthetic.invalid'))
        mirror = Mirror(config)
        run = subprocess.run
        def remote_python(command, **kwargs):
            self.assertIn('python3 -', command[-1])
            return run([sys.executable, '-B', '-'], **kwargs)
        with patch('workspace_sync.subprocess.run', side_effect=remote_python):
            result = mirror.rpc('inventory_page', remote=True, seconds=2)
        self.assertEqual(len(result['page']['items']), 1)
        self.assertLess(len(store.encoded(result)), 524288)
        self.assertIsNone(result['summary']['files_remaining'])

    def test_replacement_with_sqlite_bytes_cannot_be_sealed_as_ordinary_payload(self):
        path = self.root / 'ordinary.json'
        path.write_bytes(b'ordinary source bytes')
        index = self.index()
        self.addCleanup(index.close)
        index.mirror_advance(2)
        envelope = index.mirror_page()
        item = envelope['page']['items'][0]
        path.write_bytes(b'SQLite format 3\0synthetic header replacing ordinary bytes')
        with self.assertRaisesRegex(ValueError, 'Application snapshot required'):
            index.mirror_capture(self.state, envelope['page_id'], item['version'])
        self.assertEqual(index.mirror_summary()['observed_unacknowledged_versions'], 1)
        self.assertFalse((self.state / 'objects').exists())

    def test_controller_ack_follows_both_object_verification_and_receipt_transfer(self):
        (self.root / 'file').write_bytes(b'original payload')
        mirror = Mirror.__new__(Mirror)
        mirror.state = self.state
        remote_state = store.directory(self.base / 'remote-state')
        index = self.index()
        self.addCleanup(index.close)
        index.mirror_advance(2)
        envelope = dict(index.mirror_page(), rpc_scope={})
        events = []
        def rpc(operation, remote=False, **kwargs):
            state = remote_state if remote else self.state
            if operation == 'inventory_capture':
                return index.mirror_capture(state, kwargs['page_id'], kwargs['version'])
            if operation == 'has':
                for record in kwargs['records']:
                    store.verify_object(state, record)
                events.append(('verified', remote))
                return [r['sha256'] for r in kwargs['records']]
            if operation == 'inventory_ack':
                self.assertIn(('verified', False), events)
                self.assertIn(('verified', True), events)
                self.assertEqual(events[-1], ('transferred', kwargs['receipt_sha256']))
                store.verify_object(remote_state, {'sha256': kwargs['receipt_sha256'],
                    'size': store.object_path(self.state, kwargs['receipt_sha256']).stat().st_size})
                return index.mirror_acknowledge(state, kwargs['page_id'], kwargs['receipt_sha256'])
            self.fail('unexpected operation ' + operation)
        def transfer(records, from_remote=False, deadline=None):
            for record in records:
                source = store.object_path(remote_state if from_remote else self.state, record['sha256'])
                dest = store.object_path(self.state if from_remote else remote_state, record['sha256'])
                store.directory(dest.parent)
                shutil.copyfile(source, dest)
                events.append(('transferred', record['sha256']))
            return sum(r['size'] for r in records)
        with patch.object(mirror, 'rpc', side_effect=rpc), patch.object(mirror, 'transfer', side_effect=transfer), patch.object(mirror, 'healthy'):
            result = mirror.durable({'local': envelope}, time.monotonic() + 10)
        self.assertEqual(result['verified_versions_this_pass'], 1)
        self.assertEqual(index.mirror_summary()['observed_unacknowledged_versions'], 0)
        self.assertIsNone(result['files_remaining'])

    def test_zero_window_does_not_ack_work_or_open_source(self):
        (self.root / 'file').write_bytes(b'original payload')
        index = self.index()
        self.addCleanup(index.close)
        index.mirror_advance(2)
        envelope = dict(index.mirror_page(), rpc_scope={})
        mirror = Mirror.__new__(Mirror)
        mirror.state = self.state
        def rpc(operation, **kwargs):
            self.assertEqual(operation, 'inventory_ack')
            return index.mirror_acknowledge(self.state, kwargs['page_id'], kwargs['receipt_sha256'])
        with patch.object(mirror, 'rpc', side_effect=rpc), patch.object(mirror, 'transfer', return_value=0), patch.object(mirror, 'healthy', side_effect=AssertionError('no capture allowed')):
            result = mirror.durable({'local': envelope}, time.monotonic())
        self.assertEqual(result['verified_versions_this_pass'], 0)
        self.assertEqual(result['deferred_versions_this_pass'], 1)
        self.assertEqual(index.mirror_summary()['observed_unacknowledged_versions'], 1)


if __name__ == '__main__':
    unittest.main()
