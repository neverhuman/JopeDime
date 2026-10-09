"""New cold-continuation fixtures only; no old21 suite or network execution."""
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import trace_recovery_v2 as v


class Fixture:
    def __init__(self, root, complete=1, tail=0):
        self.root = root
        self.old = root / 'old'; self.old.mkdir()
        self.logs = root / 'logs'; self.logs.mkdir()
        (self.logs / 'archive-v1').mkdir()
        (self.old / 'receipts').mkdir()
        self.now = time.time()
        self.created, self.deadline = self.now - 120, self.now - 60
        self.raws, self.receipts = {}, []
        original_source = root / 'original_archive.py'; original_source.write_bytes(b'fixture original archive')
        writer = root / 'host_trace.py'; writer.write_bytes(b'fixture writer')
        head = '0' * 64
        for part in range(154 + complete + int(tail > 0)):
            event = {'host': 'xbabe3', 'seq': part + 1, 'previous': head}
            head = v.a.digest(v.a.canonical(event)); event['sha256'] = head
            body = v.a.canonical(event) + b'\n'
            path = self.logs / f'events-{part:05d}.jsonl'; path.write_bytes(body)
            seal = {'schema': 'jope.hpc-host-log-segment/v1', 'path': str(path), 'bytes': len(body), 'sha256': v.a.digest(body), 'last_seq': part + 1, 'event_head': head}
            seal_raw = v.a.canonical(seal); (self.logs / f'events-{part:05d}.sealed.json').write_bytes(seal_raw)
            self.raws[part] = (body, seal_raw, head)
            if part < 153:
                receipt = self.receipt(part, 'jope.hpc-host-archive/v1', v.a.digest(original_source.read_bytes()))
                target = self.logs / 'archive-v1' / f'{part:05d}.json'; target.write_bytes(v.a.canonical(receipt))
                self.receipts.append({**self.ref(target), 'receipt': receipt})
        index = {'schema': 'jope.trace-prior-archive-index/v1', 'host': 'xbabe3', 'logs_root': str(self.logs), 'receipts': self.receipts, 'original_archive_source': self.ref(original_source), 'trace_writer_source': self.ref(writer)}
        self.write('prior-archive-index.json', index)
        body = self.raws[153][0]
        gap = {'schema': 'jope.trace-unknown-object-reconciliation/v1', 'state': 'absent_preserve_unknown_no_put', 's3_error_code': '404', 'bucket': v.a.BUCKET, 'key': v.a.PREFIX + '00153/' + v.a.digest(body) + '/events.jsonl', 'expected_sha256': v.a.digest(body), 'expected_bytes': len(body), 's3_puts': 0, 'gp_executions': 0, 'local_chain_verified': True}
        self.write('gap-reconciliation.json', gap)
        old = {'schema': 'jope.trace-archive-recovery/v1', 'out': str(self.old), 'logs_root': str(self.logs), 'global_deadline_unix': v.a.GLOBAL, 'created_unix': self.created, 'deadline_unix': self.deadline, 'maximum_segments': 2500, 'maximum_put_attempts': 4692, 'prior_index': self.ref(self.old / 'prior-archive-index.json'), 'gap_reconciliation': self.ref(self.old / 'gap-reconciliation.json')}
        self.write('config.json', old); self.config_sha = self.ref(self.old / 'config.json')['sha256']
        (self.old / 'trace_archive.py').write_bytes(v.base_path.read_bytes())
        self.write('execution-started.json', {'config_sha256': self.config_sha, 'automatic_retry': False, 'utc_ns': int((self.created + 1) * 1e9)})
        for number in range(1, 2 * complete + tail + 1):
            part = 154 + (number - 1) // 2
            ref = self.cloud_ref(part, number % 2 == 1)
            attempt = {'schema': 'jope.trace-upload-attempt/v1', 'attempt': number, 'utc_ns': int((self.created + 5) * 1e9), 'monotonic_ns': number, 'bucket': ref['bucket'], 'key': ref['key'], 'sha256': ref['sha256'], 'bytes': ref['bytes'], 'automatic_retry': False}
            self.write(f'put-attempt-{number:05d}.json', attempt)
            if number <= 2 * complete:
                self.write(f'put-reply-{number:05d}.json', {'key': ref['key'], 'version_id': ref['version_id'], 'utc_ns': int((self.created + 6) * 1e9)})
                self.write(f'verified-object-{number:05d}.json', ref)
            else:
                self.write(f'unknown-put-{number:05d}.json', {**attempt, 'state': 'unknown_no_retry', 'error': 'fixture connection failure'})
        for part in range(154, 154 + complete):
            receipt = self.receipt(part, 'jope.hpc-host-archive-recovery-receipt/v1', v.BASE_SHA)
            receipt.update(config_sha256=self.config_sha, gap_parts=[153])
            self.write(f'receipts/{part:05d}.json', receipt)
        last = 153 + complete
        archived_parts = list(range(153)) + list(range(154, 154 + complete))
        progress = {'schema': 'jope.trace-recovery-progress/v1', 'state': 'stopped_unknown_or_integrity_no_retry' if tail else 'completed_bounded_window_with_preserved_gap', 'deadline_unix': self.deadline, 'unresolved_gap_parts': [153], 'cloud_contiguous_verified_through_part': 152, 'new_verified_segments': complete, 'put_attempts': 2 * complete + tail, 'next_fresh_part': last + 1, 'local_chain_head': self.raws[last][2], 'local_chain_last_seq': last + 1, 'archived_events': 153 + complete, 'verified_bytes': sum(len(self.raws[p][0]) + len(self.raws[p][1]) for p in archived_parts), 'stage_events': 1}
        self.write('progress.json', progress)
        (self.old / 'stage-timing.jsonl').write_bytes(v.a.canonical({'sequence': 1, 'config_sha256': self.config_sha, 'host': 'xbabe3', 'gpu_used': False}) + b'\n')

    def ref(self, path):
        raw = path.read_bytes(); return {'path': str(path), 'bytes': len(raw), 'sha256': v.a.digest(raw)}

    def write(self, name, value):
        (self.old / name).write_bytes(v.a.canonical(value))

    def cloud_ref(self, part, events):
        body, sealed, _ = self.raws[part]; raw = body if events else sealed
        return {'bucket': v.a.BUCKET, 'key': v.a.PREFIX + f'{part:05d}/' + v.a.digest(body) + ('/events.jsonl' if events else '/seal.json'), 'version_id': 'v-' + str(part) + ('-events' if events else '-seal'), 'bytes': len(raw), 'sha256': v.a.digest(raw), 'verified_at_unix_ns': int((self.created + 8) * 1e9)}

    def receipt(self, part, schema, source):
        return {'schema': schema, 'host': 'xbabe3', 'part': part, 'segment': self.cloud_ref(part, True), 'seal': self.cloud_ref(part, False), 'first_seq': part + 1, 'last_seq': part + 1, 'event_head': self.raws[part][2], 'source_sha256': source}

    def capture(self):
        return {'schema': 'jope.trace-recovery1-closed-readonly-capture/v1', 'at': self.now, 'unit': 'MainPID=0\nControlGroup=\nActiveState=inactive\n', 'read_only': True, 'network_S3': 0, 'mutations': 0, 'files': {str(p): self.ref(p) for p in self.old.rglob('*') if p.is_file()}}

    def verify(self, capture=None):
        with patch.object(v, 'OLD_ROOT', self.old), patch.object(v, 'OLD_CONFIG_SHA', self.config_sha), patch.object(v.a, 'LOGS', str(self.logs)):
            return v.cold_prefix(self.capture() if capture is None else capture)


class ColdTests(unittest.TestCase):
    def setUp(self):
        fixtures = Path(__file__).resolve().parent / 'fixtures'; fixtures.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=fixtures); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_full_cold_prefix_preserves_original_gap_and_global_budget(self):
        fixture = Fixture(self.root, complete=2)
        result = fixture.verify()
        self.assertEqual((result['fresh_first_part'], result['verified_segments'], result['prior_global_put_attempts'], result['remaining_put_attempts']), (156, 155, 311, 4689))
        self.assertEqual(result['gap_parts'], [153])
        self.assertEqual(result['sequence'] - result['archived_events'], 1)

    def test_unknown_intent_skips_entire_part_and_charges_global_budget(self):
        result = Fixture(self.root, complete=1, tail=1).verify()
        self.assertEqual(result['fresh_first_part'], 156)
        self.assertEqual(result['gap_parts'], [153, 155])
        self.assertEqual(result['prior_global_put_attempts'], 310)

    def test_two_attempt_unreceipted_tail_is_never_replayed(self):
        result = Fixture(self.root, complete=1, tail=2).verify()
        self.assertEqual(result['fresh_first_part'], 156)
        self.assertEqual(len(result['attempted_keys']), 4)
        self.assertEqual(result['verified_segments'], 154)

    def test_live_prior_pid_refused(self):
        fixture = Fixture(self.root); capture = fixture.capture(); capture['unit'] = 'MainPID=42\nControlGroup=/active\nActiveState=active\n'
        with self.assertRaisesRegex(ValueError, 'prior closure'):
            fixture.verify(capture)

    def test_physical_receipt_change_after_snapshot_refused(self):
        fixture = Fixture(self.root); capture = fixture.capture()
        (fixture.old / 'receipts/00154.json').write_text('{}')
        with self.assertRaises(ValueError): fixture.verify(capture)

    def test_event_chain_corruption_refused(self):
        fixture = Fixture(self.root)
        (fixture.logs / 'events-00010.jsonl').write_text('bad')
        with self.assertRaises(ValueError): fixture.verify()

    def test_missing_attempt_breaks_prior_numbering(self):
        fixture = Fixture(self.root, complete=2)
        (fixture.old / 'put-attempt-00002.json').unlink()
        with self.assertRaisesRegex(ValueError, 'numbering'): fixture.verify()

    def test_receipt_version_must_match_actual_verified_object(self):
        fixture = Fixture(self.root)
        path = fixture.old / 'verified-object-00001.json'; value = json.loads(path.read_bytes()); value['version_id'] = 'foreign'; path.write_bytes(v.a.canonical(value))
        with self.assertRaisesRegex(ValueError, 'exact-version'): fixture.verify()

    def test_completed_progress_cannot_hide_unknown_tail(self):
        fixture = Fixture(self.root, tail=1)
        path = fixture.old / 'progress.json'; value = json.loads(path.read_bytes()); value['state'] = 'completed_bounded_window_with_preserved_gap'; path.write_bytes(v.a.canonical(value))
        with self.assertRaisesRegex(ValueError, 'completed prior'): fixture.verify()

    def test_fresh_constructor_does_not_recreate_old153_attempt_or_reset_count(self):
        fixture = Fixture(self.root); prefix = fixture.verify(); out = self.root / 'new'; out.mkdir(); (out / 'receipts').mkdir()
        config = {'out': str(out), 'logs_root': str(fixture.logs), 'deadline_unix': time.time() + 60, 'maximum_put_attempts': prefix['remaining_put_attempts'], 'maximum_segments': 2500, 'source': {'sha256': 'b' * 64}, 'prior_capture': {'sha256': 'c' * 64}}
        archive = v.Archive(config, None, v.a.Timing(out / 'timing.jsonl', 'd' * 64), prefix)
        self.assertEqual((archive.part, archive.puts), (155, 0))
        archive.report('running'); report = json.loads((out / 'progress.json').read_bytes())
        self.assertEqual(report['global_put_attempts'], 309)
        with self.assertRaisesRegex(ValueError, 'replay'):
            archive.publish(prefix['attempted_keys'][0], b'x')
        self.assertEqual(list(out.glob('put-attempt-*')), [])

    def test_pair_budget_refuses_before_any_single_object_attempt(self):
        fixture = Fixture(self.root); prefix = fixture.verify(); out = self.root / 'new'; out.mkdir()
        config = {'out': str(out), 'logs_root': str(fixture.logs), 'deadline_unix': time.time() + 60, 'maximum_put_attempts': 1}
        archive = v.Archive(config, None, v.a.Timing(out / 'timing.jsonl', 'd' * 64), prefix)
        with self.assertRaisesRegex(ValueError, 'pair attempt'): archive.once()
        self.assertEqual(archive.puts, 0)

    def test_fresh_success_creates_only_new_receipt_with_exact_versions(self):
        fixture = Fixture(self.root); prefix = fixture.verify(); out = self.root / 'new'; out.mkdir(); (out / 'receipts').mkdir()
        before = {str(p): p.read_bytes() for p in fixture.old.rglob('*') if p.is_file()}
        part = prefix['fresh_first_part']
        event = {'host': 'xbabe3', 'seq': prefix['sequence'] + 1, 'previous': prefix['previous']}
        head = v.a.digest(v.a.canonical(event)); event['sha256'] = head; body = v.a.canonical(event) + b'\n'
        path = fixture.logs / f'events-{part:05d}.jsonl'; path.write_bytes(body)
        seal = {'schema': 'jope.hpc-host-log-segment/v1', 'path': str(path), 'bytes': len(body), 'sha256': v.a.digest(body), 'last_seq': event['seq'], 'event_head': head}
        (fixture.logs / f'events-{part:05d}.sealed.json').write_bytes(v.a.canonical(seal))
        class Client:
            def __init__(self): self.objects = {}; self.puts = self.gets = 0
            def put_object(self, **kw):
                self.puts += 1
                intent = json.loads((out / f'put-attempt-{self.puts:05d}.json').read_bytes())
                assert intent['key'] == kw['Key'] and kw['IfNoneMatch'] == '*'
                self.objects[kw['Key']] = kw['Body']; return {'VersionId': 'exact-v' + str(self.puts)}
            def get_object(self, **kw):
                self.gets += 1; raw = self.objects[kw['Key']]
                return {'VersionId': kw['VersionId'], 'ContentLength': len(raw), 'Body': io.BytesIO(raw)}
        client = Client()
        config = {'out': str(out), 'logs_root': str(fixture.logs), 'deadline_unix': time.time() + 60, 'maximum_put_attempts': prefix['remaining_put_attempts'], 'maximum_segments': 2500, 'source': {'sha256': 'b' * 64}, 'prior_capture': {'sha256': 'c' * 64}}
        archive = v.Archive(config, client, v.a.Timing(out / 'timing.jsonl', 'd' * 64), prefix)
        self.assertTrue(archive.once()); archive.report('running')
        receipt = json.loads((out / 'receipts' / f'{part:05d}.json').read_bytes())
        self.assertEqual((receipt['schema'], receipt['global_put_attempts_through'], client.puts, client.gets), ('jope.hpc-host-archive-recovery-receipt/v2', 311, 2, 2))
        self.assertEqual(receipt['gap_parts'], [153])
        self.assertEqual({str(p): p.read_bytes() for p in fixture.old.rglob('*') if p.is_file()}, before)
        self.assertEqual([p.name for p in (out / 'receipts').iterdir()], ['00155.json'])

    def test_expired_monotonic_deadline_blocks_even_backward_wall_clock(self):
        fixture = Fixture(self.root); prefix = fixture.verify(); out = self.root / 'new'; out.mkdir()
        config = {'out': str(out), 'logs_root': str(fixture.logs), 'deadline_unix': time.time() + 60, 'maximum_put_attempts': 2}
        archive = v.Archive(config, None, v.a.Timing(out / 'timing.jsonl', 'd' * 64), prefix)
        archive.monotonic_deadline = time.monotonic() - 1
        with patch.object(v.time, 'time', return_value=1):
            with self.assertRaisesRegex(ValueError, 'monotonic'): archive.tick()


if __name__ == '__main__':
    unittest.main(verbosity=2)
