"""Bounded mirror batching, deadline, per-version custody and migration checks."""
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import workspace_store as store
import workspace_sync as sync
from workspace_inventory import Inventory


class MirrorBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'root'
        self.root.mkdir()
        self.state = store.directory(self.base / 'state')
        self.mirror = sync.Mirror.__new__(sync.Mirror)
        self.mirror.state = self.state
        self.items = [dict(version=str(i), record=dict(size=1)) for i in range(17)]
        self.records = {item['version']: dict(sha256=hashlib.sha256(item['version'].encode()).hexdigest(), size=1)
                        for item in self.items}

    def test_groups_bound_count_and_bytes_without_starving_large_originals(self):
        groups = list(self.mirror.capture_batches(self.items))
        self.assertEqual([len(group) for group in groups], [8, 8, 1])
        sizes = [5 * 1024**2, 5 * 1024**2, 10 * 1024**2, 1]
        items = [dict(version=str(i), record=dict(size=size)) for i, size in enumerate(sizes)]
        self.assertEqual([len(group) for group in self.mirror.capture_batches(items)], [1, 1, 1, 1])
        self.items[3]['retry_individually'] = True
        self.assertEqual([len(group) for group in self.mirror.capture_batches(self.items[:8])], [3, 1, 4])

    def test_eight_captures_share_transfer_and_both_host_checks(self):
        events = []
        def rpc(op, remote=False, **kwargs):
            if op == 'inventory_capture':
                events.append(('capture', kwargs['version']))
                return self.records[kwargs['version']]
            self.assertEqual(op, 'has')
            self.assertEqual(len(kwargs['records']), 8)
            events.append(('has', remote))
            return [record['sha256'] for record in kwargs['records']]
        def transfer(records, **kwargs):
            events.append(('transfer', len(records)))
            self.assertTrue(kwargs['from_remote'])
            self.assertEqual(kwargs['deadline'], 10)
            return 8
        with patch.object(sync.time, 'monotonic', return_value=0), \
                patch.object(self.mirror, 'healthy', side_effect=lambda: events.append(('health',))), \
                patch.object(self.mirror, 'rpc', side_effect=rpc), \
                patch.object(self.mirror, 'transfer', side_effect=transfer):
            moved, outcomes = self.mirror.capture_batch(self.items[:8], 'original-page', {}, True, 10)
        self.assertEqual(moved, 8)
        self.assertEqual([row['status'] for row in outcomes], ['verified_on_both_hosts'] * 8)
        self.assertEqual(events, [('health',)] + [('capture', str(i)) for i in range(8)]
                         + [('transfer', 8), ('has', False), ('has', True), ('health',)])

    def test_one_capture_failure_and_one_missing_counterpart_preserve_other_outcomes(self):
        def rpc(op, remote=False, **kwargs):
            if op == 'inventory_capture':
                if kwargs['version'] == '1':
                    raise ValueError('original changed; actual separately queued')
                return self.records[kwargs['version']]
            self.assertEqual(op, 'has')
            return [record['sha256'] for record in kwargs['records']
                    if not (remote and record == self.records['2'])]
        with patch.object(sync.time, 'monotonic', return_value=0), \
                patch.object(self.mirror, 'healthy'), patch.object(self.mirror, 'rpc', side_effect=rpc), \
                patch.object(self.mirror, 'transfer', return_value=2) as transfer:
            moved, outcomes = self.mirror.capture_batch(self.items[:3], 'original-page', {}, False, 10)
        self.assertEqual(moved, 2)
        self.assertEqual(transfer.call_args.args[0], [self.records['0'], self.records['2']])
        self.assertEqual([row['status'] for row in outcomes], ['verified_on_both_hosts', 'deferred', 'deferred'])
        self.assertIn('original changed', outcomes[1]['reason'])
        self.assertIn('not verified on both hosts', outcomes[2]['reason'])
        self.assertTrue(all(row['attempted'] for row in outcomes))

    def test_capture_exhausting_window_keeps_cached_original_without_starting_transfer(self):
        clock = {'now': 0}
        def rpc(op, **kwargs):
            self.assertEqual(op, 'inventory_capture')
            clock['now'] = 11
            return self.records[kwargs['version']]
        with patch.object(sync.time, 'monotonic', side_effect=lambda: clock['now']), \
                patch.object(self.mirror, 'healthy'), patch.object(self.mirror, 'rpc', side_effect=rpc) as capture, \
                patch.object(self.mirror, 'transfer', side_effect=AssertionError('window elapsed')):
            moved, outcomes = self.mirror.capture_batch(self.items[:3], 'original-page', {}, False, 10)
        self.assertEqual(moved, 0)
        self.assertEqual(capture.call_count, 1)
        self.assertEqual([row['attempted'] for row in outcomes], [True, False, False])
        self.assertTrue(all(row['status'] == 'deferred' for row in outcomes))

    def test_post_group_health_failure_prevents_page_receipt_acknowledgment(self):
        page = dict(configuration_sha256='synthetic', items=self.items[:2])
        envelope = dict(page=page, page_id=store.identity(page), rpc_scope={}, summary={})
        calls = []
        def rpc(op, **kwargs):
            calls.append(op)
            if op == 'inventory_capture':
                return self.records[kwargs['version']]
            self.assertEqual(op, 'has')
            return [record['sha256'] for record in kwargs['records']]
        with patch.object(sync.time, 'monotonic', return_value=0), \
                patch.object(self.mirror, 'healthy', side_effect=[None, RuntimeError('health changed')]), \
                patch.object(self.mirror, 'rpc', side_effect=rpc), \
                patch.object(self.mirror, 'transfer', return_value=2):
            with self.assertRaisesRegex(RuntimeError, 'health changed'):
                self.mirror.durable({'local': envelope}, 10)
        self.assertNotIn('inventory_ack', calls)
        self.assertFalse((self.state / 'receipts').exists())

    def test_transport_failure_keeps_each_successfully_captured_original_pending(self):
        with patch.object(sync.time, 'monotonic', return_value=0), \
                patch.object(self.mirror, 'healthy'), \
                patch.object(self.mirror, 'rpc', side_effect=lambda op, **kw: self.records[kw['version']]), \
                patch.object(self.mirror, 'transfer', side_effect=RuntimeError('partial retained')):
            _, outcomes = self.mirror.capture_batch(self.items[:2], 'original-page', {}, False, 10)
        self.assertEqual([row['reason'] for row in outcomes], ['partial retained'] * 2)
        self.assertTrue(all(row['status'] == 'deferred' and row['attempted'] for row in outcomes))

    def test_transfer_preparation_cannot_start_rsync_after_original_window(self):
        self.mirror.config = dict(local=dict(state=str(self.state)), bandwidth_kib_s=1024)
        self.mirror.remote = dict(state='/synthetic/remote-state', host='synthetic.invalid')
        self.mirror.ssh = ['ssh', 'synthetic.invalid']
        clock = {'now': 0}
        def rpc(op, **kwargs):
            if op == 'has':
                return []
            if op == 'capacity':
                return dict(required_reserve_bytes=0, free_bytes=10 * 1024**3)
            self.assertEqual(op, 'prepare_transfer')
            clock['now'] = 11
            return '/synthetic/remote-transfer'
        with patch.object(sync.time, 'monotonic', side_effect=lambda: clock['now']), \
                patch.object(self.mirror, 'healthy'), patch.object(self.mirror, 'rpc', side_effect=rpc), \
                patch.object(sync.subprocess, 'Popen', side_effect=AssertionError('must not start rsync')):
            with self.assertRaisesRegex(sync.BulkWindowEnded, 'before starting transfer'):
                self.mirror.transfer([self.records['0']], deadline=10)
        self.assertEqual(list(self.state.glob('.transfer-list-*')), [])

    def receipt(self, index, envelope, attempted):
        value = dict(schema='workspace.inventory-page-custody.v1', configuration_sha256=index.identity,
                     page_id=envelope['page_id'], outcomes=[dict(version=item['version'], status='deferred',
                     attempted=attempted, reason='synthetic deferral') for item in envelope['page']['items']])
        raw = store.encoded(value)
        digest = hashlib.sha256(raw).hexdigest()
        destination = store.object_path(self.state, digest)
        store.directory(destination.parent)
        destination.write_bytes(raw)
        return digest

    def test_latest_unattempted_deferral_is_separate_from_earlier_attempt_failure(self):
        (self.root / 'file').write_bytes(b'original')
        index = Inventory(self.base / 'index.sqlite', {'workspace': str(self.root)}, 'synthetic', {'source_roots': []})
        self.addCleanup(index.close)
        index.mirror_advance(2)
        page = index.mirror_page()
        first_receipt = self.receipt(index, page, True)
        status = index.mirror_acknowledge(self.state, page['page_id'], first_receipt)
        self.assertEqual(status['failed_capture_versions'], 1)
        self.assertEqual(status['attempted_deferred_versions'], 1)
        next_page = index.mirror_page()
        self.assertTrue(next_page['page']['items'][0]['retry_individually'])
        next_receipt = self.receipt(index, next_page, False)
        status = index.mirror_acknowledge(self.state, next_page['page_id'], next_receipt)
        self.assertEqual(status['failed_capture_versions'], 0)
        self.assertEqual(status['attempted_deferred_versions'], 0)
        self.assertEqual(status['unattempted_deferred_versions'], 1)
        self.assertEqual(status['observed_unacknowledged_versions'], 1)
        self.assertGreater(index.db.execute('SELECT attempted_at FROM mirror_versions').fetchone()[0], 0)
        self.assertEqual(index.db.execute('SELECT receipt FROM mirror_pages ORDER BY sequence').fetchall()[0][0], first_receipt)

    def test_legacy_attempt_classification_migrates_as_unknown_without_losing_history(self):
        database = self.base / 'legacy.sqlite'
        with sqlite3.connect(database) as old:
            old.execute('CREATE TABLE mirror_versions(version TEXT PRIMARY KEY,document TEXT NOT NULL,size INTEGER NOT NULL,'
                        'captured TEXT,receipt TEXT,attempted_at REAL NOT NULL DEFAULT 0,last_error TEXT)')
            old.execute('INSERT INTO mirror_versions VALUES (?,?,?,?,?,?,?)',
                        ('original', '{}', 1, None, None, 123, 'old attempted-or-unattempted deferral'))
        index = Inventory(database, {'workspace': str(self.root)}, 'synthetic', {'source_roots': []})
        self.addCleanup(index.close)
        status = index.mirror_summary()
        self.assertEqual(status['failed_capture_versions'], 0)
        self.assertEqual(status['deferred_attempt_status_unknown_versions'], 1)
        self.assertEqual(status['observed_unacknowledged_versions'], 1)
        row = index.db.execute('SELECT attempted_at,last_error FROM mirror_versions').fetchone()
        self.assertEqual(tuple(row), (123, 'old attempted-or-unattempted deferral'))

    def test_failed_originals_get_a_retry_share_despite_continuous_fresh_arrivals(self):
        for number in range(4):
            (self.root / ('old-' + str(number))).write_bytes(b'original pending bytes')
        index = Inventory(self.base / 'index.sqlite', {'workspace': str(self.root)}, 'synthetic', {'source_roots': []})
        self.addCleanup(index.close)
        index.mirror_advance(2)
        page = index.mirror_page(limit=4)
        originals = {item['version'] for item in page['page']['items']}
        index.mirror_acknowledge(self.state, page['page_id'], self.receipt(index, page, True))
        retried = set()
        for generation in range(4):
            for number in range(4):
                (self.root / f'fresh-{generation}-{number}').write_bytes(b'new arrivals every page')
            index.mirror_advance(2)
            page = index.mirror_page(limit=4)
            selected = {item['version'] for item in page['page']['items']}
            self.assertTrue(selected & originals, 'fresh arrivals starved original retry share')
            retried.update(selected & originals)
            self.assertTrue(any(item.get('retry_individually') for item in page['page']['items']))
            index.mirror_acknowledge(self.state, page['page_id'], self.receipt(index, page, True))
        self.assertEqual(retried, originals)
        self.assertEqual(index.mirror_summary()['acknowledged_versions'], 0)


if __name__ == '__main__':
    unittest.main()
