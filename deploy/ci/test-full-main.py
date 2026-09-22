#!/usr/bin/env python3
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
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


if __name__ == '__main__':
    unittest.main()
