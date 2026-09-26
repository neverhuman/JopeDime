#!/usr/bin/env python3
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import receipt

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('full_main', ROOT / 'deploy/ci/full-main.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


class FullMainTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {'lanes': {'full-one': {'command':['fixture'], 'requires_nonzero_tests':False, 'result_backend':'none'}}, 'dispositions':{}}
        self.expected = dict(source_sha='a'*40, checkout_sha='a'*40, run_id='123', attempt=1, event='push')

    def fixture(self, root):
        folder = root / 'full-one'
        folder.mkdir()
        (folder/'command.log').write_text('fixture\n')
        return dict(schema='jope.ci-receipt/v1', **self.expected, lane='full-one', command=['fixture'], dirty_source=[],dirty_source_after=[], status='passed',exit_status=0, finished_at='2026-09-22T00:00:01+00:00',
                    inputs={name:receipt.digest(ROOT/name) for name in receipt.INPUTS}, counts=dict.fromkeys(('collected','executed','passed','failed','skipped','excluded')),
                    evidence={'command.log':receipt.digest(folder/'command.log')})

    def report(self, root, data):
        (root/'full-one/receipt.json').write_text(json.dumps(data))
        return full.summarize(self.mapping,root,self.expected)

    def test_failure_does_not_stop_later_families(self):
        mapping = copy.deepcopy(self.mapping)
        mapping['lanes']['full-one']['command'] = [sys.executable,'-c',"print('first failed'); raise SystemExit(23)"]
        mapping['lanes']['full-two'] = dict(command=[sys.executable,'-c',"print('second executed')"],result_backend='none')
        with tempfile.TemporaryDirectory() as scratch, patch.dict(os.environ,dict(GITHUB_RUN_ID='123',GITHUB_RUN_ATTEMPT='1')):
            root = Path(scratch)/'new'
            self.assertEqual(1,full.run(mapping,root))
            self.assertEqual(23,json.loads((root/'full-one/receipt.json').read_text())['exit_status'])
            self.assertEqual('passed',json.loads((root/'full-two/receipt.json').read_text())['status'])

    def test_missing_receipt_and_wrong_identity_fail(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self.assertEqual('failed',full.summarize(self.mapping,root,self.expected)['status'])
            data = self.fixture(root)
            self.assertEqual('passed',self.report(root,data)['status'])
            for field, value in [('source_sha','b'*40),('attempt',2),('run_id','124')]:
                changed = dict(data,**{field:value})
                self.assertEqual('failed',self.report(root,changed)['status'])

    def test_incomplete_unavailable_failed_and_zero_execution_fail(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            data = self.fixture(root)
            for status, code in [('started',None),('failed',124),('failed',78),('unavailable',None)]:
                changed = dict(data,status=status,exit_status=code)
                report = self.report(root,changed)
                self.assertEqual('failed',report['status'])
                if code == 78:
                    self.assertEqual('unavailable',report['families']['full-one']['status'])
            self.mapping['lanes']['full-one']['requires_nonzero_tests'] = True
            self.assertEqual('failed',self.report(root,data)['status'])

    def test_missing_or_tampered_evidence_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            data = self.fixture(root)
            (root/'full-one/command.log').write_text('changed')
            self.assertEqual('failed',self.report(root,data)['status'])
            (root/'full-one/command.log').unlink()
            self.assertEqual('failed',self.report(root,data)['status'])

    def test_loose_fixture_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            data = self.fixture(root)
            link = root / 'full-one' / 'fixture-link'
            link.symlink_to('command.log')
            data['evidence']['fixture-link'] = receipt.digest(link)
            report = self.report(root, data)
            self.assertEqual('failed', report['status'])
            self.assertIn('unsafe or changed evidence', report['errors'][0])

    def test_pytest_archive_retains_links_and_failure_status(self):
        scratch_root = ROOT / 'target' / 'tmp'
        scratch_root.mkdir(parents=True, exist_ok=True)
        # A small command fixture stands in for pytest, exercising the actual shell
        # wrapper and inventory without requiring Python's optional dependencies.
        command = '''import argparse, pathlib, sys
p = argparse.ArgumentParser()
p.add_argument('--basetemp'); p.add_argument('--junitxml'); p.add_argument('--status')
a = p.parse_args()
root = pathlib.Path(a.basetemp); root.mkdir()
(root/'payload.txt').write_text('retained fixture')
(root/'file-link').symlink_to('payload.txt')
(root/'dir-link').symlink_to('..', target_is_directory=True)
(root/'dangling-link').symlink_to('absent')
pathlib.Path(a.junitxml).write_text('<testsuites/>')
sys.exit(int(a.status))
'''
        for status in (0, 7):
            with self.subTest(status=status), tempfile.TemporaryDirectory(prefix='pytest evidence ', dir=scratch_root) as scratch:
                root = Path(scratch)
                data = self.fixture(root)
                folder = root / 'full-one'
                result = subprocess.run(['bash', str(ROOT/'deploy/ci/pytest-evidence.sh'),
                                         sys.executable, '-c', command, '--status', str(status)],
                                        env=dict(os.environ, JOPE_LANE_EVIDENCE=str(folder)),
                                        capture_output=True, text=True)
                self.assertEqual(status, result.returncode, result.stderr)
                archive = folder / 'pytest-fixtures.tar.gz'
                with tarfile.open(archive) as retained:
                    self.assertEqual(b'retained fixture', retained.extractfile('./payload.txt').read())
                    for name, target in [('file-link', 'payload.txt'), ('dir-link', '..'), ('dangling-link', 'absent')]:
                        member = retained.getmember('./' + name)
                        self.assertTrue(member.issym())
                        self.assertEqual(target, member.linkname)
                    self.assertEqual(5, len(retained.getmembers()))
                self.assertFalse(any(p.is_symlink() for p in folder.rglob('*')))
                data.update(status='passed' if status == 0 else 'failed', exit_status=status)
                data['evidence'] = {str(p.relative_to(folder)): receipt.digest(p) for p in folder.rglob('*') if p.is_file()}
                report = self.report(root, data)
                self.assertEqual('passed' if status == 0 else 'failed', report['status'])
                self.assertEqual(status, report['families']['full-one']['exit_status'])
                if status == 0:
                    archive.write_bytes(archive.read_bytes() + b'tampered')
                    self.assertEqual('failed', self.report(root, data)['status'])
                # Reusing a pytest basetemp would delete retained fixtures. Refuse
                # before invoking the command, preserving its existing payload.
                again = subprocess.run(['bash', str(ROOT/'deploy/ci/pytest-evidence.sh'), 'false'],
                                       env=dict(os.environ, JOPE_LANE_EVIDENCE=str(folder)),
                                       capture_output=True, text=True)
                self.assertEqual(2, again.returncode)
                self.assertEqual('retained fixture', Path(str(folder)+'-pytest-fixtures/payload.txt').read_text())

    def test_archive_failure_never_passes_or_hides_pytest_failure(self):
        scratch_root = ROOT / 'target' / 'tmp'
        scratch_root.mkdir(parents=True, exist_ok=True)
        for status, expected in [(0, 1), (7, 7)]:
            with self.subTest(status=status), tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
                root = Path(scratch)
                folder = root / 'evidence'
                folder.mkdir()
                fake_tar = root / 'tar'
                fake_tar.write_text('#!/bin/sh\nexit 23\n')
                fake_tar.chmod(0o755)
                command = "import pathlib,sys; pathlib.Path(sys.argv[2].split('=',1)[1]).mkdir(); sys.exit(int(sys.argv[1]))"
                result = subprocess.run(['bash', str(ROOT/'deploy/ci/pytest-evidence.sh'),
                                         sys.executable, '-c', command, str(status)],
                                        env=dict(os.environ, JOPE_LANE_EVIDENCE=str(folder),
                                                 PATH=str(root)+os.pathsep+os.environ['PATH']),
                                        capture_output=True, text=True)
                self.assertEqual(expected, result.returncode, result.stderr)
                self.assertIn('failed to retain pytest fixtures', result.stderr)

    def test_declared_family_budgets_fit_job_with_upload_reserve(self):
        mapping = json.loads((ROOT/'deploy/ci/coverage.json').read_text())
        lanes = full.families(mapping)
        self.assertEqual({'full-build-reuse','full-default','full-doctests','full-scratch','full-policy','full-s3-offline','full-neural','full-python','full-inputs','full-e2e','full-emitters'},set(lanes))
        total = 0
        for name,lane in lanes.items():
            command = lane['command']
            self.assertEqual(['timeout','--signal=TERM','--kill-after=15s'],command[:3])
            self.assertEqual(['bash','deploy/ci/full-lane.sh',name],command[4:])
            total += int(command[3][:-1]) + 15
        self.assertLessEqual(total,115*60)

    def test_family_budget_covers_its_slowest_recorded_golden_test(self):
        # A family shorter than one of its own tests can never pass: full-scratch was
        # killed at 600 s inside feat-math::golden fast_profile_portfolio_head (~580-619 s).
        mapping = json.loads((ROOT/'deploy/ci/coverage.json').read_text())
        lanes = full.families(mapping)
        timings = json.loads((ROOT/'docs/parity/test-timings.json').read_text())
        slowest = max(t['seconds'] for t in timings['tests'] if t['binary'] == 'feat-math::golden')
        for name in ('full-default','full-scratch'):
            budget = int(lanes[name]['command'][3][:-1])
            self.assertGreaterEqual(budget, 1.25*slowest + 120, name)


if __name__ == '__main__':
    unittest.main()
