"""Filesystem/SQLite integration tests with tiny pinned synthetic campaigns."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jope_flywheel import paper_archive as pa


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='jope-paper-test-')
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / 'training'; self.runtime = self.base / 'runtime'
        self.root.mkdir(); self.runtime.mkdir()
        self.out = self.base / 'archive'; self.job = {'run_id': 'dlw4full-v1-s3-test', 'host': 'xbabe3', 'seed': 3, 'arm': 'V1'}
        self.run = self.root / 'runs' / self.job['run_id']; self.run.mkdir(parents=True)
        self.arm = {'id': 'V1', 'host': 'xbabe3', 'configuration': {'regression_loss': 'mse'}}
        self.write(self.runtime / 'arm_V1.json', self.arm)
        self.qual = self.runtime / 'gpu_qualification.json'; self.write(self.qual, {'state': 'passed'})
        self.manifest = {'schema': 'jope.dl-wave4/v1', 'wave': 4, 'model_version': '1.0.0',
            'campaign_id': 'test', 'source_sha': '1' * 40, 'source_archive_sha256': '2' * 64,
            'source_files': {'wave3_train.py': '3' * 64}, 'bundle_sha': '4' * 64,
            'service_deadline': time.time() + 1000,
            'hosts': {'xbabe3': {'root': str(self.root), 'runtime': str(self.runtime)}},
            'jobs': [self.job], 'arms': [self.arm],
            'gpu_qualification': {'path': str(self.qual), 'sha256': pa.sha(self.qual.read_bytes())}}
        self.mpath = self.base / 'campaign.json'; self.pin()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, value):
        path.write_text(json.dumps(value) + '\n')

    def pin(self):
        self.write(self.mpath, self.manifest); self.msha = pa.sha(self.mpath.read_bytes())
        (self.runtime / 'manifest.json').write_bytes(self.mpath.read_bytes())
        self.write(self.runtime / 'status.json', {'host': 'xbabe3', 'campaign_sha': self.msha, 'state': 'running'})
        self.identity = {'schema': 'jope.dl-wave4-run-identity/v1', 'wave': 4,
            'model_version': '1.0.0', 'campaign_id': 'test', 'campaign_manifest_sha256': self.msha,
            'job': self.job, 'source_sha': self.manifest['source_sha'], 'bundle_sha256': self.manifest['bundle_sha'],
            'trainer_engine': 'dlhwm.wave3_train', 'trainer_summary_wave_field': 3,
            'configuration': self.arm['configuration']}
        self.write(self.runtime / (self.job['run_id'] + '.identity.json'), self.identity)
        self.summary = {'run_id': self.job['run_id'], 'seed': 3, 'wave': 3, 'arm': 'V1',
            'git_sha': self.manifest['source_sha'], 'data_manifest_sha256': self.manifest['bundle_sha'],
            'arm_spec_sha256': pa.sha((self.runtime / 'arm_V1.json').read_bytes()),
            'qualification_only': False, 'state': 'running', 'folds': {}}
        self.write(self.run / 'summary.json', self.summary)

    def archive(self):
        return pa.Archive(self.mpath, self.msha, 'xbabe3', self.out)

    def test_live_snapshots_idempotent_and_observation_semantics(self):
        self.write(self.run / 'hist_r0f0.json', [{'ep': 1, 'at': 'source-old-time'}])
        (self.run / 'batch_metrics.jsonl').write_bytes(b'{"loss":1}\n')
        with self.archive() as archive:
            first = archive.cycle(); count = archive.ledger.verify()['event_count']
            second = archive.cycle()
            self.assertEqual(archive.ledger.verify()['event_count'], count)
            self.assertEqual(first['raw_batch_bytes_in_complete_chunks'], 11)
            self.assertEqual(second['runs'][self.job['run_id']]['epochs_observed'], {'r0f0': 1})
            self.assertFalse(second['source_verification']['source_file_bytes_independently_verified'])
            self.assertIn('not established', second['native_order_proof'])
            imported = [json.loads(x) for x in (self.out / 'events.jsonl').read_text().splitlines()][1:]
            self.assertTrue(all(e['evidence_kind'] == 'imported_aggregate' for e in imported))
            self.assertTrue(all(e['time_semantics'] == 'observation_time' for e in imported))
            self.assertTrue((self.out / 'events.csv').exists())

    def test_partial_line_waits_and_restart_recovers_committed_cursor(self):
        path = self.run / 'batch_metrics.jsonl'; path.write_bytes(b'{"n":1}\n{"n":')
        with self.archive() as archive:
            archive.cycle()
            self.assertEqual(archive.cursors[self.job['run_id'] + '/batch_metrics.jsonl']['end'], 8)
        with path.open('ab') as stream:
            stream.write(b'2}\n')
        with self.archive() as archive:
            result = archive.cycle()
            chunks = [x for x in archive.records if x['kind'] == 'jsonl_chunk']
            self.assertEqual([(x['start'], x['end']) for x in chunks], [(0, 8), (8, 16)])
            self.assertEqual(result['raw_batch_bytes_in_complete_chunks'], 16)

    def test_retry_after_blob_before_event_and_after_committed_event(self):
        with self.archive() as archive:
            path = self.run / 'batch_metrics.jsonl'; path.write_bytes(b'{"n":1}\n')
            original = archive.ledger.append_imported
            with mock.patch.object(archive.ledger, 'append_imported', side_effect=RuntimeError('crash-before-event')):
                with self.assertRaises(RuntimeError):
                    archive._chunk('log', path)
            self.assertNotIn('log', archive.cursors)
            def commit_then_crash(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError('crash-after-commit')
            with mock.patch.object(archive.ledger, 'append_imported', side_effect=commit_then_crash):
                with self.assertRaises(RuntimeError):
                    archive._chunk('log', path)
        with self.archive() as archive:
            self.assertEqual(archive.cursors['log']['end'], 8)
            count = archive.ledger.verify()['event_count']
            archive._chunk('log', path)
            self.assertEqual(archive.ledger.verify()['event_count'], count)

    def test_existing_prefix_mutation_and_truncation_refused(self):
        path = self.run / 'batch_metrics.jsonl'; path.write_bytes(b'{"n":1}\n')
        with self.archive() as archive:
            archive.cycle()
            path.write_bytes(b'{"n":2}\n')
            result = archive.cycle()
            self.assertTrue(any('prefix changed' in x.get('reason', '') for x in result['errors']))
            path.write_bytes(b'')
            result = archive.cycle()
            self.assertTrue(any('truncated' in x.get('reason', '') for x in result['errors']))

    def test_inode_replacement_refused(self):
        path = self.run / 'batch_metrics.jsonl'; path.write_bytes(b'{"n":1}\n')
        with self.archive() as archive:
            archive.cycle()
            replacement = self.run / 'replacement'; replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
            result = archive.cycle()
            self.assertTrue(any('inode replaced' in x.get('reason', '') for x in result['errors']))

    def test_cas_tampering_refuses_resume(self):
        with self.archive() as archive:
            archive.cycle(); blob = archive.records[-1]['blob']
        (self.out / blob).write_bytes(b'tampered')
        with self.assertRaises(pa.IntegrityError):
            self.archive()

    def test_confinement_and_symlink_source_never_read(self):
        secret = self.base / 'secret'; secret.write_bytes(b'not JSON and not to be copied')
        (self.run / 'hist_r0f0.json').symlink_to(secret)
        (self.run / 'checkpoints').symlink_to(self.base)
        (self.root / 'data_v1').symlink_to(self.base)
        with self.archive() as archive:
            result = archive.cycle()
            self.assertTrue(any(x['label'].endswith('hist_r0f0.json') for x in result['errors']))
            self.assertNotIn(pa.sha(secret.read_bytes()), [x['sha256'] for x in archive.records])
            self.assertEqual(result['raw_dataset_bytes_read'], 0)
        self.manifest['jobs'][0]['run_id'] = '../escape'; self.pin()
        with self.assertRaises(pa.EvidenceError):
            self.archive()

    def test_foreign_output_and_output_symlink_refused(self):
        self.out.mkdir(); (self.out / 'foreign').write_text('preserve')
        with self.assertRaises(pa.EvidenceError):
            self.archive()
        self.assertEqual((self.out / 'foreign').read_text(), 'preserve')
        other = self.base / 'link'; other.symlink_to(self.out)
        with self.assertRaises(OSError):
            pa.Archive(self.mpath, self.msha, 'xbabe3', other)

    def test_only_same_host_jobs_and_no_oof_before_complete(self):
        self.manifest['jobs'].append({'run_id': 'other-host', 'host': 'xbabe2', 'arm': 'V1', 'seed': 5})
        self.pin(); (self.run / 'oof_r0.npy').write_bytes(b'fake-small-binary')
        with self.archive() as archive:
            result = archive.cycle()
            self.assertEqual(list(result['runs']), [self.job['run_id']])
            self.assertFalse(any(x['label'].endswith('.npy') for x in archive.records))
            self.summary.update(state='all_done', folds={f: {} for f in pa.FOLDS})
            self.write(self.run / 'summary.json', self.summary)
            result = archive.cycle()
            self.assertTrue(result['runs'][self.job['run_id']]['complete_fold_key_closure'])
            self.assertTrue(any(x['label'].endswith('.npy') for x in archive.records))

    def test_mismatched_summary_and_partial_done_fail_closed(self):
        self.summary.update(state='all_done', folds={'r0f0': {}}, seed=4)
        self.write(self.run / 'summary.json', self.summary)
        with self.archive() as archive:
            result = archive.cycle(); view = result['runs'][self.job['run_id']]
            self.assertFalse(view['identity_matches'])
            self.assertEqual(view['source_state'], 'unknown')
            self.summary['seed'] = 3; self.write(self.run / 'summary.json', self.summary)
            self.assertFalse(archive.cycle()['runs'][self.job['run_id']]['complete_fold_key_closure'])

    def test_only_bound_failure_allows_forensic_final_bytes(self):
        (self.run / 'oof_r0.npy').write_bytes(b'forensics')
        with self.archive() as archive:
            self.write(self.runtime / 'status.json', {'host': 'xbabe3', 'campaign_sha': '0' * 64, 'state': 'failed'})
            archive.cycle(); self.assertFalse(any(x['label'].endswith('.npy') for x in archive.records))
            self.write(self.runtime / 'status.json', {'host': 'xbabe3', 'campaign_sha': self.msha, 'state': 'failed'})
            archive.cycle(); self.assertTrue(any(x['label'].endswith('.npy') for x in archive.records))

    def test_caps_refuse_before_publication_and_keep_previous_summary(self):
        with self.archive() as archive:
            archive.cycle(); previous = (self.out / 'evidence_summary.json').read_bytes()
            with mock.patch.object(pa, 'BLOB_CAP', archive._usage()[1] + 2):
                with self.assertRaises(pa.LimitError):
                    archive._capture_bytes('big', self.mpath, b'new bytes too large', 'snapshot')
            self.assertEqual(previous, (self.out / 'evidence_summary.json').read_bytes())
            with mock.patch.object(pa, 'EVENT_CAP', archive.event_count):
                with self.assertRaises(pa.LimitError):
                    archive._capture_bytes('new', self.mpath, b'{}', 'snapshot')
            with mock.patch.object(pa, 'TOTAL_CAP', archive._usage()[0] + 1):
                with self.assertRaises(pa.LimitError):
                    archive._atomic('evidence_summary.json', b'oversized')
            self.assertEqual(previous, (self.out / 'evidence_summary.json').read_bytes())

    def test_partial_large_jsonl_and_invalid_json_are_not_skipped(self):
        path = self.run / 'batch_metrics.jsonl'
        with self.archive() as archive:
            path.write_bytes(b'{' + b'x' * 20)
            with mock.patch.object(pa, 'CHUNK_CAP', 16):
                with self.assertRaises(pa.LimitError):
                    archive._chunk('log', path)
            path.write_bytes(b'{"loss":1e999}\n')
            with self.assertRaises(pa.EvidenceError):
                archive._chunk('log', path)
            self.assertNotIn('log', archive.cursors)

    def test_manifest_pin_deadline_and_live_mutation(self):
        with self.assertRaises(pa.IntegrityError):
            pa.Archive(self.mpath, '0' * 64, 'xbabe3', self.out)
        self.assertFalse(self.out.exists())
        self.manifest['service_deadline'] = time.time() - 901; self.pin()
        with self.assertRaises(pa.LimitError):
            self.archive()
        self.assertFalse(self.out.exists())
        self.manifest['service_deadline'] = time.time() + 1000; self.pin()
        with self.archive() as archive:
            self.mpath.write_text('{}')
            with self.assertRaises(pa.IntegrityError):
                archive.cycle()

    def test_single_writer_and_abandoned_cas_link_recovery(self):
        with self.archive() as archive:
            with self.assertRaises(BlockingIOError):
                self.archive()
            blob = self.out / archive.records[0]['blob']
        temp = self.out / ('.archive-' + 'f' * 32 + '.tmp')
        os.link(blob, temp)
        with self.archive() as archive:
            self.assertFalse(temp.exists())
            self.assertEqual(blob.stat().st_nlink, 1)
            archive.ledger.verify()

    def test_restart_hashes_shared_blob_once_under_read_budget(self):
        shared = b'{"shared":true}\n'
        with self.archive() as archive:
            for index in range(5):
                archive._capture_bytes('shared-' + str(index), self.mpath, shared, 'snapshot')
            unique_bytes = sum(p.stat().st_size for p in (self.out / 'blobs').iterdir())
        with mock.patch.object(pa, 'CYCLE_READ_CAP', unique_bytes):
            with self.archive() as archive:
                self.assertEqual(archive.read_bytes, unique_bytes)

    def test_existing_cas_read_for_new_descriptor_obeys_remaining_budget(self):
        shared = b'{"shared":true}\n'
        with self.archive() as archive:
            archive._capture_bytes('first', self.mpath, shared, 'snapshot')
            count = archive.ledger.verify()['event_count']
            with mock.patch.object(pa, 'CYCLE_READ_CAP', archive.read_bytes):
                with self.assertRaises(pa.LimitError):
                    archive._capture_bytes('second', self.mpath, shared, 'snapshot')
            self.assertEqual(archive.ledger.verify()['event_count'], count)
            archive._capture_bytes('second', self.mpath, shared, 'snapshot')
            self.assertEqual(archive.ledger.verify()['event_count'], count + 1)


if __name__ == '__main__':
    unittest.main()
