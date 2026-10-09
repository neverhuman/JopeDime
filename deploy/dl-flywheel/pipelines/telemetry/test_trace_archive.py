"""CPU-only custody boundaries; no network, cloud writes, GPU, or live source."""
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import trace_archive as a


class Failure(Exception):
    def __init__(self, code=None):
        self.response = None if code is None else {'Error': {'Code': code}}
        super().__init__('original connection failure')


class Client:
    def __init__(self, directory):
        self.directory = directory
        self.calls, self.objects = [], {}
        self.put_error = self.get_error = None
        self.wrong_version = self.corrupt = False

    def put_object(self, **kw):
        # This assertion proves custody is physically durable BEFORE network dispatch.
        number = len([x for x in self.calls if x[0] == 'put']) + 1
        intent = json.loads((self.directory/f'put-attempt-{number:05d}.json').read_text())
        assert intent['key'] == kw['Key'] and intent['sha256'] == a.digest(kw['Body'])
        assert kw['IfNoneMatch'] == '*' and kw['Bucket'] == a.BUCKET
        self.calls.append(('put', kw['Key']))
        if self.put_error:
            raise self.put_error
        self.objects[kw['Key']] = kw['Body']
        return {'VersionId': 'immutable1'}

    def head_object(self, **kw):
        self.calls.append(('head', kw['Key']))
        return {'VersionId': 'immutable1', 'ContentLength': len(self.objects[kw['Key']])}

    def get_object(self, **kw):
        self.calls.append(('get', kw['Key']))
        assert kw['VersionId'] == 'immutable1'
        if self.get_error:
            raise self.get_error
        raw = self.objects[kw['Key']]
        return {'VersionId': 'wrong' if self.wrong_version else 'immutable1',
                'ContentLength': len(raw), 'Body': io.BytesIO(b'corruption' if self.corrupt else raw)}


class Tests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get('JOPE_TELEMETRY_FIXTURE_ROOT', str(Path(__file__).resolve().parent/'fixtures')))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root/'logs'; self.logs.mkdir()
        self.out = self.root/'out'; self.out.mkdir(); (self.out/'receipts').mkdir()
        self.config = {'logs_root': str(self.logs), 'out': str(self.out), 'deadline_unix': time.time()+120,
                       'maximum_put_attempts': 4692, 'maximum_segments': 2500,
                       'source': {'sha256': 'a'*64}}
        prefix = {'previous': '0'*64, 'sequence': 0, 'prior_events': 0,
                  'verified_bytes': 0, 'gap': {'state': 'absent'}, 'gap_event_count': 0}
        self.client = Client(self.out)
        self.worker = a.Archive(self.config, self.client, a.Timing(self.out/'timing.jsonl', 'a'*64), prefix)

    def make_segment(self, part=154):
        event = {'host': 'xbabe3', 'seq': 1, 'previous': '0'*64, 'value': 3}
        event['sha256'] = a.digest(a.canonical(event))
        raw = a.canonical(event)+b'\n'; p = self.logs/f'events-{part:05d}.jsonl'; p.write_bytes(raw)
        seal = {'schema': 'jope.hpc-host-log-segment/v1', 'path': str(p), 'bytes': len(raw),
                'sha256': a.digest(raw), 'last_seq': 1, 'event_head': event['sha256']}
        (self.logs/f'events-{part:05d}.sealed.json').write_bytes(a.canonical(seal))
        return raw

    def test_fresh_success_has_actual_version_readbacks_and_gap_counts(self):
        self.make_segment()
        self.assertTrue(self.worker.once())
        self.worker.report('running')
        p = json.loads((self.out/'progress.json').read_text())
        self.assertEqual((p['new_verified_segments'], p['new_verified_objects']), (1, 2))
        self.assertEqual(p['unresolved_gap_parts'], [153])
        self.assertEqual(p['cloud_contiguous_verified_through_part'], 152)
        self.assertEqual([c[0] for c in self.client.calls], ['put', 'get', 'put', 'get'])
        self.assertEqual(json.loads((self.out/'receipts/00154.json').read_text())['seal']['version_id'], 'immutable1')

    def test_receipt_joins_exact_pretty_sealed_config_hash(self):
        raw = json.dumps(self.config, indent=2).encode()+b'\n'
        sealed_sha = a.digest(raw)
        self.assertNotEqual(sealed_sha, a.digest(a.canonical(self.config)))
        self.worker.timing.config_sha = sealed_sha
        self.make_segment(); self.assertTrue(self.worker.once())
        receipt = json.loads((self.out/'receipts/00154.json').read_text())
        self.assertEqual(receipt['config_sha256'], sealed_sha)
        spans = [json.loads(x) for x in (self.out/'timing.jsonl').read_text().splitlines()]
        self.assertTrue(spans)
        self.assertTrue(all(x['config_sha256'] == sealed_sha for x in spans))

    def test_none_response_preserves_original_failure_and_single_attempt(self):
        self.make_segment(); self.client.put_error = Failure()
        with self.assertRaisesRegex(Failure, 'original connection failure'):
            self.worker.once()
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(json.loads((self.out/'unknown-put-00001.json').read_text())['state'], 'unknown_no_retry')
        self.assertFalse((self.out/'receipts/00154.json').exists())

    def test_partial_two_object_closure_preserves_first_object(self):
        self.make_segment(); original = self.client.put_object
        def second(**kw):
            if len(self.client.calls) == 2:
                self.client.put_error = Failure()
            return original(**kw)
        self.client.put_object = second
        with self.assertRaises(Failure): self.worker.once()
        self.assertTrue((self.out/'verified-object-00001.json').exists())
        self.assertTrue((self.out/'unknown-put-00002.json').exists())
        self.assertFalse((self.out/'receipts/00154.json').exists())

    def test_412_can_only_reuse_verified_exact_existing_bytes(self):
        raw = self.make_segment(); key = a.PREFIX+'00154/'+a.digest(raw)+'/events.jsonl'
        self.client.objects[key] = raw; self.client.put_error = Failure('412')
        ref = self.worker.publish(key, raw)
        self.assertEqual(ref['sha256'], a.digest(raw))
        self.assertEqual([x[0] for x in self.client.calls], ['put', 'head', 'get'])

    def test_post_put_get_failure_keeps_intent_and_version_reply_no_retry(self):
        self.make_segment(); self.client.get_error = Failure()
        with self.assertRaises(Failure): self.worker.once()
        self.assertEqual([x[0] for x in self.client.calls], ['put', 'get'])
        self.assertTrue((self.out/'put-attempt-00001.json').exists())
        self.assertTrue((self.out/'put-reply-00001.json').exists())
        self.assertFalse((self.out/'verified-object-00001.json').exists())

    def test_version_mismatch_never_resolves_artifact(self):
        self.make_segment(); self.client.wrong_version = True
        with self.assertRaisesRegex(ValueError, 'exact version'): self.worker.once()
        self.assertFalse((self.out/'verified-object-00001.json').exists())

    def test_readback_corruption_never_resolves_artifact(self):
        self.make_segment(); self.client.corrupt = True
        with self.assertRaisesRegex(ValueError, 'readback'): self.worker.once()
        self.assertFalse((self.out/'receipts/00154.json').exists())

    def test_attempted_gap_key_refused_before_any_put(self):
        with self.assertRaisesRegex(ValueError, 'fresh key'):
            self.worker.publish(a.PREFIX+'00153/'+'a'*64+'/events.jsonl', b'x')
        self.assertEqual(self.client.calls, [])

    def test_deadline_stops_before_intent_and_network(self):
        self.config['deadline_unix'] = time.time()-1
        with self.assertRaisesRegex(ValueError, 'deadline'): self.worker.once()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(list(self.out.glob('put-attempt-*')), [])

    def test_budget_stops_before_intent_and_network(self):
        self.worker.puts = self.config['maximum_put_attempts']
        with self.assertRaisesRegex(ValueError, 'attempt budget'):
            self.worker.publish(a.PREFIX+'00154/'+'a'*64+'/events.jsonl', b'x')
        self.assertEqual(self.client.calls, [])

    def test_wrong_chain_rejected_before_put(self):
        self.make_segment(); self.worker.previous = 'f'*64
        with self.assertRaisesRegex(ValueError, 'event chain'): self.worker.once()
        self.assertEqual(self.client.calls, [])

    def test_changed_sealed_bytes_rejected_before_put(self):
        self.make_segment(); (self.logs/'events-00154.jsonl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'segment bytes'): self.worker.once()
        self.assertEqual(self.client.calls, [])

    def test_symlink_source_rejected(self):
        raw = self.make_segment(); p = self.logs/'events-00154.jsonl'; p.unlink()
        q = self.root/'alias'; q.write_bytes(raw); p.symlink_to(q)
        with self.assertRaisesRegex(ValueError, 'physical path'): self.worker.once()
        self.assertEqual(self.client.calls, [])

    def test_reused_attempt_receipt_is_not_overwritten_or_replayed(self):
        self.make_segment(); p = self.out/'put-attempt-00001.json'; p.write_bytes(b'prior unknown')
        with self.assertRaises(FileExistsError): self.worker.once()
        self.assertEqual(p.read_bytes(), b'prior unknown')
        self.assertEqual(self.client.calls, [])

    def test_prior_reference_wrong_namespace_and_null_version_refused(self):
        raw = self.make_segment(); ref = {'bucket': a.BUCKET, 'key': 'foreign',
                                         'version_id': 'v', 'sha256': a.digest(raw), 'bytes': len(raw)}
        with self.assertRaisesRegex(ValueError, 'authority'):
            a.object_ref(ref, 154, (raw, b'x'), 'events.jsonl')
        ref['key'] = a.PREFIX+'00154/'+a.digest(raw)+'/events.jsonl'; ref['version_id'] = None
        with self.assertRaisesRegex(ValueError, 'authority'):
            a.object_ref(ref, 154, (raw, b'x'), 'events.jsonl')

    def test_local_pin_corruption_is_refused(self):
        p = self.root/'pin'; p.write_bytes(b'stable')
        ref = {'path': str(p), 'bytes': 6, 'sha256': a.digest(b'stable')}
        p.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'pin integrity'): a.pinned(ref)

    def prefix_fixture(self):
        def pin(path, raw):
            path.write_bytes(raw)
            return {'path': str(path), 'bytes': len(raw), 'sha256': a.digest(raw)}
        old = pin(self.root/'archive_trace.py', b'original')
        trace = pin(self.root/'host_trace.py', b'trace')
        source = pin(self.out/'trace_archive.py', b'qualified recovery')
        archive = self.logs/'archive-v1'; archive.mkdir()
        previous, rows = '0'*64, []
        for part in range(154):
            event = {'host': 'xbabe3', 'seq': part+1, 'previous': previous}
            head = a.digest(a.canonical(event)); event['sha256'] = head
            body = a.canonical(event)+b'\n'; path = self.logs/f'events-{part:05d}.jsonl'
            path.write_bytes(body)
            seal = a.canonical({'schema': 'jope.hpc-host-log-segment/v1', 'path': str(path),
                'bytes': len(body), 'sha256': a.digest(body), 'last_seq': part+1, 'event_head': head})
            (self.logs/f'events-{part:05d}.sealed.json').write_bytes(seal)
            def ref(raw, name):
                return {'bucket': a.BUCKET, 'key': a.PREFIX+f'{part:05d}/'+a.digest(body)+'/'+name,
                        'bytes': len(raw), 'sha256': a.digest(raw), 'version_id': 'historical1'}
            if part < 153:
                receipt = {'schema': 'jope.hpc-host-archive/v1', 'host': 'xbabe3', 'part': part,
                           'first_seq': part+1, 'last_seq': part+1, 'event_head': head,
                           'source_sha256': old['sha256'], 'segment': ref(body, 'events.jsonl'),
                           'seal': ref(seal, 'seal.json')}
                rows.append({**pin(archive/f'{part:05d}.json', a.canonical(receipt)), 'receipt': receipt})
            previous = head
        index = pin(self.out/'prior.json', a.canonical({'schema': 'jope.trace-prior-archive-index/v1',
                    'host': 'xbabe3', 'logs_root': str(self.logs), 'receipts': rows,
                    'original_archive_source': old, 'trace_writer_source': trace}))
        gap = pin(self.out/'gap.json', a.canonical({'schema': 'jope.trace-unknown-object-reconciliation/v1',
                  'state': 'absent_preserve_unknown_no_put', 's3_error_code': '404',
                  'bucket': a.BUCKET, 'key': a.PREFIX+'00153/'+a.digest(body)+'/events.jsonl',
                  'expected_bytes': len(body), 'expected_sha256': a.digest(body),
                  's3_puts': 0, 'gp_executions': 0, 'local_chain_verified': True}))
        now = time.time()
        config = {'schema': a.SCHEMA, 'host': 'xbabe3', 'logs_root': str(self.logs), 'out': str(self.out),
                  'created_unix': now, 'deadline_unix': now+120, 'global_deadline_unix': a.GLOBAL,
                  'maximum_segments': 2500, 'maximum_put_attempts': 4692,
                  'prior_index': index, 'gap_reconciliation': gap, 'source': source}
        return config

    def test_full_prefix_preserves_gap_events_without_cloud_coverage(self):
        c = self.prefix_fixture()
        with patch.object(a, 'LOGS', str(self.logs)): result = a.preflight(c)
        self.assertEqual((result['prior_events'], result['sequence'], result['gap_event_count']), (153, 154, 1))
        self.assertEqual(self.client.calls, [])

    def test_changed_old_receipt_blocks_recovery_before_network(self):
        c = self.prefix_fixture(); (self.logs/'archive-v1/00000.json').write_bytes(b'changed')
        with patch.object(a, 'LOGS', str(self.logs)), self.assertRaises(ValueError): a.preflight(c)
        self.assertEqual(self.client.calls, [])

    def test_forged_gap_closure_refused_even_with_new_pin(self):
        c = self.prefix_fixture(); p = Path(c['gap_reconciliation']['path']); gap = json.loads(p.read_text())
        gap['state'] = 'verified'; raw = a.canonical(gap); p.write_bytes(raw)
        c['gap_reconciliation'].update(sha256=a.digest(raw), bytes=len(raw))
        with patch.object(a, 'LOGS', str(self.logs)), self.assertRaisesRegex(ValueError, 'absent unknown gap'): a.preflight(c)
        self.assertEqual(self.client.calls, [])

    def test_extended_clock_and_attempt_ceiling_refused(self):
        c = self.prefix_fixture(); c['deadline_unix'] = c['created_unix']+3601
        with patch.object(a, 'LOGS', str(self.logs)), self.assertRaisesRegex(ValueError, 'clock'): a.preflight(c)
        c['deadline_unix'] = c['created_unix']+120; c['maximum_put_attempts'] = 5000
        with patch.object(a, 'LOGS', str(self.logs)), self.assertRaisesRegex(ValueError, 'original 5000'): a.preflight(c)
