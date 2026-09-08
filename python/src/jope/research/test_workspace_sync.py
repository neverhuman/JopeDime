import errno
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import workspace_store as store
from workspace_sync import Mirror


class WorkspaceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'root'
        self.root.mkdir()
        self.state = self.base / 'state'
        self.policy = {'source_roots': ['src'], 'dependency_roots': []}
        (self.root / 'src').mkdir()
        (self.root / 'src/main.rs').write_text('original')

    def snapshot(self, owner='local'):
        return store.source_snapshot(self.root, self.policy, self.state, owner)

    def test_restore_repeated_sync_and_executable_identity(self):
        (self.root / 'src/run').write_text('original')
        (self.root / 'src/run').chmod(0o755)
        manifest = self.snapshot()
        self.assertEqual(manifest, self.snapshot())
        first = store.materialize(self.state, manifest)
        self.assertEqual(first, store.materialize(self.state, manifest))
        generation = self.state / 'generations' / manifest['generation']
        self.assertEqual((generation / 'src/main.rs').read_text(), 'original')
        self.assertFalse((generation / 'src/main.rs').stat().st_mode & 0o111)
        self.assertTrue((generation / 'src/run').stat().st_mode & 0o111)
        self.assertEqual(len(list((self.state / 'objects').iterdir())), 1)

    def test_conflicting_historical_bytes_both_survive(self):
        before = self.snapshot()
        (self.root / 'src/main.rs').write_text('incoming')
        after = self.snapshot('remote')
        for manifest in (before, after):
            store.materialize(self.state, manifest)
        self.assertNotEqual(before['generation'], after['generation'])
        plan = Mirror.source_plan(None, before, after)
        self.assertEqual(plan['candidates'][0]['status'], 'conflict')
        self.assertFalse(plan['automatic_resolution'])
        self.assertEqual(len(list((self.state / 'objects').iterdir())), 2)

    def test_interrupted_or_corrupt_transfer_never_publishes(self):
        record = store.stable_hash(self.root, 'src/main.rs')
        incoming = store.directory(self.state / 'transfer') / record['sha256']
        incoming.write_bytes(b'ori')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            store.publish_object(self.state, incoming, record)
        self.assertFalse(store.object_path(self.state, record['sha256']).exists())
        incoming.write_bytes(b'original')
        store.publish_object(self.state, incoming, record)
        store.verify_object(self.state, record)

    def test_disk_exhaustion_does_not_publish(self):
        manifest = self.snapshot()
        with patch.object(store.shutil, 'copyfile', side_effect=OSError(errno.ENOSPC, 'disk full')):
            with self.assertRaises(OSError):
                store.materialize(self.state, manifest)
        self.assertFalse((self.state / 'generations' / manifest['generation']).exists())
        for rec in manifest['files'].values():
            store.verify_object(self.state, rec)
        store.materialize(self.state, manifest)

    def test_source_changed_during_hash_is_rejected(self):
        class MutatingWriter(io.BytesIO):
            def write(inner, data):
                (self.root / 'src/main.rs').write_text('concurrent edit')
                return super().write(data)
        with self.assertRaisesRegex(ValueError, 'changed during hashing'):
            store.stable_hash(self.root, 'src/main.rs', MutatingWriter())

    def test_source_changed_across_snapshot_is_rejected(self):
        original = store.seal
        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            (self.root / 'src/main.rs').write_text('another worker')
            return result
        with patch.object(store, 'seal', side_effect=change):
            with self.assertRaisesRegex(ValueError, 'changed while snapshotting'):
                self.snapshot()
        self.assertFalse((self.state / 'manifests').exists())

    def test_symlink_leaf_parent_and_destination_rejected(self):
        outside = self.base / 'outside'
        outside.mkdir()
        (outside / 'secret.txt').write_text('never copy')
        (self.root / 'src/link').symlink_to(outside / 'secret.txt')
        with self.assertRaises((ValueError, OSError)):
            store.stable_hash(self.root, 'src/link')
        (self.root / 'escape').symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            store.stable_hash(self.root, 'escape/secret.txt')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.snapshot()
        (self.base / 'linked-state').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            store.directory(self.base / 'linked-state/objects')
        self.assertEqual((outside / 'secret.txt').read_text(), 'never copy')

    def test_path_and_manifest_tampering_rejected(self):
        manifest = self.snapshot()
        for name in ('../outside', '/absolute', 'src/../../outside', 'src//x', 'src/./x'):
            with self.assertRaises(ValueError):
                store.relative(name)
        manifest['files']['../escape'] = manifest['files'].pop('src/main.rs')
        manifest['generation'] = store.identity({k: v for k, v in manifest.items() if k != 'generation'})
        with self.assertRaises(ValueError):
            store.materialize(self.state, manifest)

    def test_database_content_and_build_evidence_classification(self):
        target = self.root / 'target/custom-recovery'
        target.mkdir(parents=True)
        (target / 'only-source.rs').write_text('irreplaceable')
        (target / 'receipt.json').write_text('{"completed":true}')
        (target / 'cache.rlib').write_bytes(b'generated')
        (self.root / 'src/not-a-db.json').write_bytes(b'SQLite format 3\0data')
        (self.root / 'src/innocent.txt').write_bytes(b'-----BEGIN OPENSSH PRIVATE KEY-----\nsecret')
        inv = store.inventory(self.root, self.policy)
        self.assertIn('target/custom-recovery/only-source.rs', inv['files'])
        self.assertIn('target/custom-recovery/receipt.json', inv['files'])
        self.assertIn('target/custom-recovery/cache.rlib', inv['excluded'])
        self.assertIn('src/innocent.txt', inv['excluded'])
        self.assertEqual(inv['pending']['src/not-a-db.json']['state'], 'application_snapshot_required')

    def test_existing_corrupted_generation_is_never_overwritten(self):
        manifest = self.snapshot()
        store.materialize(self.state, manifest)
        path = self.state / 'generations' / manifest['generation'] / 'src/main.rs'
        path.chmod(0o644)
        path.write_text('other worker')
        with self.assertRaisesRegex(ValueError, 'integrity failure'):
            store.materialize(self.state, manifest)
        self.assertEqual(path.read_text(), 'other worker')

    def test_missing_generation_and_object_symlink_cannot_pass(self):
        manifest = self.snapshot()
        with self.assertRaises(ValueError):
            store.verify_generation(self.state, manifest)
        rec = manifest['files']['src/main.rs']
        obj = store.object_path(self.state, rec['sha256'])
        obj.unlink()
        obj.symlink_to(self.root / 'src/main.rs')
        with self.assertRaises((OSError, ValueError)):
            store.verify_object(self.state, rec)

    def test_interrupted_inventory_resumes_with_explicit_unknown_coverage(self):
        first = store.inventory_step(self.root, self.policy, self.state, seconds=0)
        self.assertFalse(first['complete'])
        self.assertIsNone(first['unvisited_bytes'])
        complete = store.inventory_step(self.root, self.policy, self.state, seconds=2)
        self.assertTrue(complete['complete'])
        self.assertEqual(complete['files'], store.inventory(self.root, self.policy)['files'])
        (self.root / 'src/new.rs').write_text('new source')
        refreshed = store.inventory_step(self.root, self.policy, self.state, seconds=2)
        self.assertIn('src/new.rs', refreshed['files'])

    def test_stale_health_new_failures_and_invocation_drift_block_staging(self):
        good = {'timestamp': 100, 'stable': {'jope-parallel-gp': {'properties': {'InvocationID': 'a'}}},
                'telemetry': {'gp-service.json': {'timestamp': 100, 'phase': 'running',
                    'completed_views': 5, 'completed_evaluations': 10, 'failed_views': 0,
                    'active': 104, 'queued': 1},
                    'cycle-progress.json': {'timestamp': 100},
                    'hardware.json': {'timestamp': 100, 'errors': [], 'available_gb': 100,
                                      'disk_free_gb': 100, 'cpu_total': 95}}}
        mirror = Mirror.__new__(Mirror)
        mirror.state = self.root
        mirror.baseline = good
        with patch.object(mirror, 'health', return_value=good):
            self.assertEqual(mirror.healthy(), good)
        for kind in ('stale', 'failure', 'invocation', 'resource'):
            changed = copy.deepcopy(good)
            if kind == 'stale':
                changed['telemetry']['cycle-progress.json']['timestamp'] = 1
            elif kind == 'failure':
                changed['telemetry']['gp-service.json']['failed_views'] = 1
            elif kind == 'invocation':
                changed['stable']['jope-parallel-gp']['properties']['InvocationID'] = 'b'
            else:
                del changed['telemetry']['hardware.json']['disk_free_gb']
            with patch.object(mirror, 'health', return_value=changed):
                with self.assertRaises(RuntimeError):
                    mirror.healthy()

    def test_zero_bulk_window_persists_the_verifiable_empty_index(self):
        mirror = Mirror.__new__(Mirror)
        mirror.state = self.state
        inventories = {'local': {'files': {}, 'pending': {}, 'root': str(self.root)}}
        with patch.object(mirror, 'rpc', side_effect=lambda op, **kw: store.identity(kw['value'])):
            result = mirror.durable(inventories, time.monotonic())
        index = json.loads((self.state / 'durable-index.json').read_text())
        self.assertEqual(index['files'], {})
        self.assertEqual(result['index_sha256'], store.identity(index))


if __name__ == '__main__':
    unittest.main()
