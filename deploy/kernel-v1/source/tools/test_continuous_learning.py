import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import continuous_learning as c
import experiment_runtime as runtime


class ContinuousLearningTests(unittest.TestCase):
    def test_snapshot_and_inactivity_thresholds(self):
        self.assertFalse(c.snapshot_due(103, True, 100, 699))
        self.assertTrue(c.snapshot_due(104, True, 100, 101))
        self.assertTrue(c.snapshot_due(1, True, 100, 700))
        self.assertFalse(c.snapshot_due(1000, False, 100, 900))
        self.assertIsNone(c.idle_status(119, True, False))
        self.assertEqual(c.idle_status(120, True, False), 'warning')
        self.assertEqual(c.idle_status(300, True, False), 'incident')
        self.assertIsNone(c.idle_status(900, True, True))
        self.assertIsNone(c.idle_status(900, False, False))

    def fixture(self, root):
        contexts = root / 'contexts.json'
        c.atomic(contexts, dict(schema='parallel.context_index.v1', encodings=[]))
        manifest = dict(schema='parallel.continuous_learning.v1', run_until_paused=True,
            world=8, microbatch=256, gp_workers=104, steps=3000,
            contexts='contexts.json', contexts_sha256=c.sha(contexts),
            bindings={'contexts.json': c.sha(contexts)}, withdrawn_families=['withdrawn'],
            bootstrap_cycle='runs/recovery', output='runs/continuous')
        path = root / 'manifest.json'
        c.atomic(path, manifest)
        return path

    def test_check_is_read_only_and_never_launches_fresh_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root)
            before = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch('sys.argv', ['continuous_learning.py', '--root', str(root), '--manifest', str(path), '--check']), contextlib.redirect_stdout(io.StringIO()), patch.object(c.runtime, 'popen', side_effect=AssertionError('launched')):
                c.main()
            self.assertEqual(before, {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(FileNotFoundError):
                c.Supervisor(root, path)

    def test_manifest_hash_and_withdrawal_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = self.fixture(root)
            c.atomic(root / 'contexts.json', dict(schema='parallel.context_index.v1', encodings=[dict(family='withdrawn')]))
            with self.assertRaisesRegex(RuntimeError, 'input changed'):
                c.validate_manifest(root, path)
            manifest = c.read(path)
            manifest['contexts_sha256'] = manifest['bindings']['contexts.json'] = c.sha(root / 'contexts.json')
            c.atomic(path, manifest)
            with self.assertRaisesRegex(RuntimeError, 'withdrawn cohort'):
                c.validate_manifest(root, path)

    def test_pause_resume_are_durable_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = self.fixture(root)
            for flag, expected in [('--pause', True), ('--resume', False)]:
                with patch('sys.argv', ['continuous_learning.py','--root',str(root),'--manifest',str(path),flag]):
                    c.main()
                self.assertEqual(c.read(root / 'runs/continuous-control.json')['paused'], expected)

    def test_committed_stage_cannot_be_applied_twice_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c.atomic(root / 'receipt.json', {'saved':True})
            supervisor = c.Supervisor.__new__(c.Supervisor)
            supervisor.root = supervisor.output = root
            supervisor.manifest_sha = 'test'
            supervisor.state_path = root / 'state.json'
            supervisor.journal = root / 'journal.jsonl'
            supervisor.s = {'generations':[{'stage':2}], 'tasks':{'proposal':{'receipt':'receipt.json','generation':0}}}
            supervisor.finish_task('proposal')
            saved = c.read(supervisor.state_path)
            self.assertEqual(saved['generations'][0]['stage'], 3)
            self.assertEqual(saved['tasks'], {})
            self.assertEqual(runtime.journal(supervisor.journal)[0]['event'], 'stage_committed')

    def test_torn_journal_suffix_is_preserved_before_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'journal.jsonl'
            path.write_bytes(b'{"event":"committed"}\n{"event":')
            self.assertEqual(runtime.journal(path, repair=True), [{'event':'committed'}])
            self.assertEqual(path.read_bytes(), b'{"event":"committed"}\n')
            self.assertEqual(next(path.parent.glob('*.torn-*')).read_bytes(), b'{"event":')

    def test_generation_seeds_are_matched_new_and_cohort_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c.atomic(root / 'contexts.json', {'encodings':[dict(prepared='p',cut=0,partition='meta_train',prepared_sha256='h')]})
            arms = {}
            for arm in ('neural','generic','shuffled'):
                path = root / arm; path.mkdir(); (path / 'seeds.json').write_text('{}')
                arms[arm] = arm
                c.atomic(path / 'jobs.json', dict(vocabulary='v', jobs=[dict(prepared='p',cut=0,prepared_sha256='h',
                    output=arm+'/search',seeds=arm+'/seeds.json',seeds_sha256=c.sha(path/'seeds.json'))]))
            spec = dict(contexts='contexts.json',rng_seed=123,checkpoint='checkpoint',checkpoint_sha256='parent')
            first = c.assemble_jobs(root,0,spec,arms)
            second = c.assemble_jobs(root,1,spec,arms)
            self.assertEqual(len({j['rng_seed'] for j in first['jobs']}),1)
            self.assertNotEqual(first['jobs'][0]['rng_seed'],second['jobs'][0]['rng_seed'])
            c.atomic(root / 'contexts.json', {'encodings':[dict(prepared='p',cut=0,partition='meta_test',prepared_sha256='h')]})
            with self.assertRaisesRegex(RuntimeError,'outside frozen eligible'):
                c.assemble_jobs(root,2,spec,arms)


if __name__ == '__main__':
    unittest.main()
