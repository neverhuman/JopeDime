"""Real SQLite durability, concurrency, provenance and owned-path checks."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jope_flywheel import evidence as ev


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='jope-evidence-test-')
        self.root = Path(self.tmp.name)
        self.db = self.root / 'events.sqlite'
        self.ledger = ev.Ledger(self.db)
        self.manifest = {'source_sha256': '1' * 64, 'seed': 7, 'scope': 'test'}
        self.ledger.register_run('run1', self.manifest)

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def append(self, event_id='batch1', **kwargs):
        return self.ledger.append(event_id, 'optimizer.batch', {'loss': .125, 'rows': 7}, 'run1', **kwargs)

    def test_registration_is_immutable_and_source_bound(self):
        first = self.ledger.register_run('run1', self.manifest)
        self.assertEqual(first['seq'], 1)
        self.assertEqual(self.ledger.verify()['event_count'], 1)
        with self.assertRaises(ev.CollisionError):
            self.ledger.register_run('run1', dict(self.manifest, seed=8))
        for m in ({}, {'source_sha256': 'x'}, {'source_sha256': True}):
            with self.assertRaises(ev.EvidenceError):
                self.ledger.register_run('bad', m)
        event = self.append()
        self.assertEqual(event['manifest_sha256'], hashlib.sha256(ev.canonical_json(self.manifest).encode()).hexdigest())
        self.assertEqual(event['source_sha256'], self.manifest['source_sha256'])

    def test_unknown_run_and_reserved_names_refused(self):
        with self.assertRaises(ev.EvidenceError):
            self.ledger.append('x', 'step', {}, 'unregistered')
        for event_id, kind in [('ledger:x', 'step'), ('x', 'ledger.run_registered')]:
            with self.assertRaises(ev.EvidenceError):
                self.ledger.append(event_id, kind, {}, 'run1')

    def test_exact_retry_retains_original_clock_and_hash(self):
        first = self.append(); second = self.append()
        self.assertEqual(first, second)
        self.assertEqual(self.ledger.verify()['event_count'], 2)
        for kind, payload in [('optimizer.batch', {'loss': .126, 'rows': 7}), ('other', {'loss': .125, 'rows': 7})]:
            with self.assertRaises(ev.CollisionError):
                self.ledger.append('batch1', kind, payload, 'run1')
        self.ledger.register_run('run2', self.manifest)
        with self.assertRaises(ev.CollisionError):
            self.ledger.append('batch1', 'optimizer.batch', {'loss': .125, 'rows': 7}, 'run2')

    def test_json_bounds_and_finite_types(self):
        for bad in [float('nan'), float('inf'), -float('inf'), {1: 'not-string'}, (1, 2), object(), 2**257]:
            with self.assertRaises(ev.EvidenceError):
                self.ledger.append('bad', 'step', {'value': bad}, 'run1')
        with self.assertRaises(ev.LimitError):
            self.ledger.append('big', 'step', {'text': 'a' * ev.MAX_JSON_BYTES}, 'run1')
        deep = {}
        for _ in range(ev.MAX_DEPTH + 1):
            deep = {'x': deep}
        with self.assertRaises(ev.LimitError):
            self.ledger.append('deep', 'step', deep, 'run1')
        self.assertEqual(self.ledger.verify()['event_count'], 1)

    def test_observation_time_and_clock_origins_are_explicit(self):
        first = self.append()
        self.assertTrue(first['observed_at_utc'].endswith('Z'))
        self.assertEqual(first['time_semantics'], 'observation_time')
        self.assertGreaterEqual(first['monotonic_ns'], first['clock_origin']['monotonic_ns'])
        with ev.Ledger(self.db) as other:
            second = other.append('batch2', 'optimizer.batch', {}, 'run1')
        self.assertNotEqual(first['clock_origin']['id'], second['clock_origin']['id'])
        self.ledger.verify()

    def test_imported_aggregate_cannot_claim_native_batch_order(self):
        imported = self.ledger.append_imported('old', 'epoch.summary', {'epochs': 3, 'original_utc': '2020-01-01'}, 'run1', '2' * 64)
        self.assertEqual(imported['evidence_kind'], 'imported_aggregate')
        self.assertNotEqual(imported['observed_at_utc'], '2020-01-01')
        with self.assertRaises(ev.EvidenceError):
            self.ledger.append_imported('false-proof', 'epoch.summary', {'nested': [{'batch_order_sha256': '3' * 64}]}, 'run1', '2' * 64)
        with self.assertRaises(ev.EvidenceError):
            self.append('missing-source', evidence_kind='imported_aggregate')

    def test_payload_paths_are_data_and_never_opened(self):
        event = self.ledger.append('path-is-data', 'observation', {'path': '/must/not/be/opened', 'env': '$SECRET'}, 'run1')
        self.assertEqual(event['payload']['path'], '/must/not/be/opened')
        self.ledger.verify()

    def test_body_tampering_detected_and_reopen_refuses(self):
        self.append()
        with sqlite3.connect(self.db) as con:
            con.execute("UPDATE events SET body_json=replace(body_json,'0.125','0.126') WHERE seq=2")
        with self.assertRaises(ev.IntegrityError):
            self.ledger.verify()
        with self.assertRaises(ev.IntegrityError):
            ev.Ledger(self.db)

    def test_rehashed_wrong_previous_link_detected(self):
        self.append()
        with sqlite3.connect(self.db) as con:
            text = con.execute('SELECT body_json FROM events WHERE seq=2').fetchone()[0]
            body = json.loads(text); body['prev_hash'] = 'f' * 64
            text = ev.canonical_json(body)
            con.execute('UPDATE events SET body_json=?, event_hash=? WHERE seq=2', (text, hashlib.sha256(text.encode()).hexdigest()))
        with self.assertRaises(ev.IntegrityError):
            self.ledger.verify()

    def test_manifest_tampering_and_tail_anchor(self):
        self.append(); anchor = self.ledger.verify()
        with sqlite3.connect(self.db) as con:
            con.execute('DELETE FROM events WHERE seq=2')
        # A local valid prefix needs the external anchor to detect removed tail.
        self.assertEqual(self.ledger.verify()['event_count'], 1)
        with self.assertRaises(ev.IntegrityError):
            self.ledger.verify(expected_head=anchor['head_sha256'])
        with self.assertRaises(ev.IntegrityError):
            self.ledger.verify(expected_count=anchor['event_count'])
        with sqlite3.connect(self.db) as con:
            con.execute("UPDATE runs SET source_sha256=?", ('9' * 64,))
        with self.assertRaises(ev.IntegrityError):
            self.ledger.verify()

    def test_seal_hashes_exact_bytes_and_freezes_run(self):
        self.append()
        (self.root / 'model.pt').write_bytes(b'actual artifact\x00bytes')
        seal = self.ledger.seal('run1', {'model': 'model.pt'})
        self.assertEqual(seal['payload']['artifacts'][0]['sha256'], hashlib.sha256(b'actual artifact\x00bytes').hexdigest())
        self.assertEqual(self.ledger.seal('run1', {'model': 'model.pt'}), seal)
        self.assertEqual(self.append()['seq'], 2)  # known retry is still safe after sealing
        with self.assertRaises(ev.EvidenceError):
            self.append('new-after-seal')
        self.assertTrue(self.ledger.verify(expected_head=seal['event_hash'])['runs']['run1']['sealed'])
        self.ledger.verify_artifacts('run1')
        with self.assertRaises(ev.EvidenceError):
            self.ledger.export_jsonl('model.pt')
        (self.root / 'model.pt').write_bytes(b'changed')
        self.ledger.verify()  # does not open paths implicitly
        with self.assertRaises(ev.IntegrityError):
            self.ledger.verify_artifacts('run1')
        with self.assertRaises(ev.CollisionError):
            self.ledger.seal('run1', {'model': 'model.pt'})

    def test_missing_artifact_does_not_seal_and_other_run_stays_open(self):
        with self.assertRaises(FileNotFoundError):
            self.ledger.seal('run1', {'missing': 'missing.bin'})
        self.assertFalse(self.ledger.verify()['runs']['run1']['sealed'])
        (self.root / 'a').write_bytes(b'a')
        self.ledger.seal('run1', {'a': 'a'})
        self.ledger.register_run('run2', self.manifest)
        self.ledger.append('run2-step', 'step', {}, 'run2')
        self.ledger.verify()

    def test_owned_paths_symlinks_fifo_and_database_are_refused(self):
        (self.root / 'safe').write_bytes(b'x')
        (self.root / 'link').symlink_to(self.root / 'safe')
        (self.root / 'dirlink').symlink_to(self.root, target_is_directory=True)
        os.mkfifo(self.root / 'pipe')
        for path in ['../outside', '/etc/passwd', 'link', 'dirlink/safe', 'pipe', 'events.sqlite']:
            with self.subTest(path=path), self.assertRaises((ev.EvidenceError, OSError)):
                self.ledger.seal('run1', {'item': path})
        for path in ['../outside', '/etc/passwd', 'dirlink/report.jsonl', 'events.sqlite', 'events.sqlite-wal']:
            with self.subTest(path=path), self.assertRaises((ev.EvidenceError, OSError)):
                self.ledger.export_jsonl(path)
        self.ledger.verify()

    def test_artifact_and_export_limits_leave_previous_file(self):
        (self.root / 'large').write_bytes(b'x' * 20)
        with mock.patch.object(ev, 'MAX_ARTIFACT_BYTES', 10):
            with self.assertRaises(ev.LimitError):
                self.ledger.seal('run1', {'large': 'large'})
        (self.root / 'report.jsonl').write_text('preserve old export')
        with mock.patch.object(ev, 'MAX_EXPORT_BYTES', 10):
            with self.assertRaises(ev.LimitError):
                self.ledger.export_jsonl('report.jsonl')
        self.assertEqual((self.root / 'report.jsonl').read_text(), 'preserve old export')
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_removed_metadata_and_unrelated_database_are_not_repaired(self):
        with sqlite3.connect(self.db) as con:
            con.execute('DELETE FROM metadata')
        with self.assertRaises(ev.IntegrityError):
            ev.Ledger(self.db)
        unrelated = self.root / 'other.sqlite'
        with sqlite3.connect(unrelated) as con:
            con.execute('CREATE TABLE somebody_elses_data(x)')
        with self.assertRaises(ev.IntegrityError):
            ev.Ledger(unrelated)

    def test_sqlite_itself_enforces_row_byte_limit(self):
        with sqlite3.connect(self.db) as con:
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute('UPDATE events SET body_json=? WHERE seq=1', ('x' * (ev.MAX_JSON_BYTES + 1),))
        self.ledger.verify()

    def test_total_artifact_read_limit_applies_before_next_file_read(self):
        (self.root / 'one').write_bytes(b'123456')
        (self.root / 'two').write_bytes(b'123456')
        with mock.patch.object(ev, 'MAX_ARTIFACT_BYTES', 10):
            with self.assertRaises(ev.LimitError):
                self.ledger.seal('run1', {'one': 'one', 'two': 'two'})
        self.assertFalse(self.ledger.verify()['runs']['run1']['sealed'])

    def test_exports_are_consistent_verifiable_and_lossless(self):
        self.ledger.append('quoted', 'step', {'text': 'quote"\ncomma,unicode α'}, 'run1')
        anchor = self.ledger.export_jsonl('events.jsonl')
        csv_anchor = self.ledger.export_csv('events.csv')
        rows = [json.loads(x) for x in (self.root / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(anchor['head_sha256'], rows[-1]['event_hash'])
        self.assertEqual(csv_anchor['head_sha256'], anchor['head_sha256'])
        for row in rows:
            digest = row.pop('event_hash')
            self.assertEqual(hashlib.sha256(ev.canonical_json(row).encode()).hexdigest(), digest)
            row['event_hash'] = digest
        csv_rows = list(csv.DictReader(io.StringIO((self.root / 'events.csv').read_text())))
        self.assertEqual([json.loads(x['event_json']) for x in csv_rows], rows)

    def test_real_process_crash_before_commit_rolls_back(self):
        code = '''import os,sys
sys.path.insert(0,sys.argv[1])
from jope_flywheel.evidence import Ledger
l=Ledger(sys.argv[2]); original=l._append
def crash(*a,**k):
 original(*a,**k)
 os._exit(23)
l._append=crash
l.append('crash','step',{'written_before_crash':True},'run1')
'''
        result = subprocess.run([sys.executable, '-c', code, str(Path(ev.__file__).parents[1]), str(self.db)], timeout=15, capture_output=True)
        self.assertEqual(result.returncode, 23, result.stderr.decode())
        self.assertEqual(self.ledger.verify()['event_count'], 1)
        event = self.ledger.append('crash', 'step', {'written_before_crash': True}, 'run1')
        self.assertEqual(event['seq'], 2)

    def test_event_limit_is_enforced_before_insert_and_registration_rolls_back(self):
        with mock.patch.object(ev, 'MAX_EVENTS', 2):
            first = self.append()
            self.assertEqual(self.append(), first)
            with self.assertRaises(ev.LimitError):
                self.append('too-many')
            with self.assertRaises(ev.LimitError):
                self.ledger.register_run('run2', self.manifest)
            result = self.ledger.verify()
            self.assertEqual(result['event_count'], 2)
            self.assertNotIn('run2', result['runs'])

    def test_export_and_seal_serialize_hash_through_replace(self):
        (self.root / 'report.jsonl').write_text('old export')
        self.ledger.register_run('run2', self.manifest)
        started = threading.Event(); done = threading.Event(); errors = []
        with ev.Ledger(self.db) as other:
            def seal():
                started.set()
                try:
                    other.seal('run2', {'report': 'report.jsonl'})
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    done.set()
            worker = threading.Thread(target=seal)
            original = os.replace
            def replace(*args, **kwargs):
                worker.start()
                self.assertTrue(started.wait(2))
                self.assertFalse(done.wait(.05), 'seal crossed export write lock')
                return original(*args, **kwargs)
            with mock.patch.object(ev.os, 'replace', side_effect=replace):
                self.ledger.export_jsonl('report.jsonl')
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            other.verify_artifacts('run2')
            with self.assertRaises(ev.EvidenceError):
                self.ledger.export_jsonl('report.jsonl')

    def test_real_multiprocess_writers_and_duplicate_id(self):
        code = '''import sys
sys.path.insert(0,sys.argv[1])
from jope_flywheel.evidence import Ledger
with Ledger(sys.argv[2]) as l:
 l.append('shared','step',{'same':True},'run1')
 for i in range(10):l.append('p'+sys.argv[3]+'-'+str(i),'step',{'i':i},'run1')
'''
        workers = [subprocess.Popen([sys.executable, '-c', code, str(Path(ev.__file__).parents[1]), str(self.db), str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(4)]
        for worker in workers:
            out, err = worker.communicate(timeout=30)
            self.assertEqual(worker.returncode, 0, (out + err).decode())
        verified = self.ledger.verify()
        self.assertEqual(verified['event_count'], 42)
        self.assertEqual(self.ledger.export_jsonl('concurrent.jsonl')['event_count'], 42)

    def test_committed_event_survives_abrupt_process_exit(self):
        code = '''import os,sys
sys.path.insert(0,sys.argv[1])
from jope_flywheel.evidence import Ledger
l=Ledger(sys.argv[2])
l.append('committed','step',{'durable':True},'run1')
os._exit(19)
'''
        result = subprocess.run([sys.executable, '-c', code, str(Path(ev.__file__).parents[1]), str(self.db)], timeout=15, capture_output=True)
        self.assertEqual(result.returncode, 19, result.stderr.decode())
        self.assertEqual(self.ledger.verify()['event_count'], 2)
        with ev.Ledger(self.db) as reopened:
            self.assertEqual(reopened.append('committed', 'step', {'durable': True}, 'run1')['seq'], 2)

    def test_database_symlink_and_hardlink_refused(self):
        (self.root / 'linked.sqlite').symlink_to(self.db)
        os.link(self.db, self.root / 'hardlinked.sqlite')
        for name in ('linked.sqlite', 'hardlinked.sqlite'):
            with self.assertRaises(ev.EvidenceError):
                ev.Ledger(self.root / name)


if __name__ == '__main__':
    unittest.main()
