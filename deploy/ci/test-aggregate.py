#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import tempfile
import unittest

import aggregate
from receipt import digest
from results import counts


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.mapping = json.loads((aggregate.ROOT / 'deploy/ci/coverage.json').read_text())
        self.expected = dict(source_sha='a' * 40, checkout_sha='b' * 40, run_id='42', attempt=2, event='workflow_dispatch')
        self.plan = dict(schema='jope.ci-plan/v1', **self.expected, base_sha='', paths=[], lanes=dict.fromkeys(aggregate.plan_module.LANES, True))
        self.needs = {lane: {'result':'success'} for lane in aggregate.plan_module.LANES}
        self.receipts = {}
        inputs = {name: digest(aggregate.ROOT / name) for name in aggregate.INPUTS}
        for lane in aggregate.plan_module.LANES:
            folder = self.root / lane
            folder.mkdir()
            backend = self.mapping['lanes'][lane]['result_backend']
            logs = {'none':'verified fixture\n', 'rust':'test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n', 'unittest':'Ran 2 tests in 0.1s\n\nOK\n', 'pytest':'2 passed in 0.1s\n', 'nextest':'Summary [1s] 2 tests run: 2 passed, 3 skipped\n'}
            (folder / 'command.log').write_text(logs[backend])
            if lane == 'changes':
                (folder / 'plan.json').write_text(json.dumps(self.plan))
            if lane in ('rust-test', 'python'):
                skip = '' if lane == 'rust-test' else ' skipped="0"'
                (folder / 'junit.xml').write_text(f'<testsuites><testsuite tests="2"{skip}><testcase/><testcase/></testsuite></testsuites>')
            if lane == 'rust-test':
                (folder / 'collection.json').write_text(json.dumps({'test-count':5,'rust-suites':{'fixture':{'testcases':dict.fromkeys(map(str, range(5)), {})}}}))
                for name in ('nextest-started', 'nextest.toml', 'disk-before.txt', 'disk-after.txt', 'target-bytes.txt'):
                    (folder / name).write_text('fixture')
            result_counts = dict.fromkeys(('collected','executed','passed','failed','skipped','excluded')) if backend == 'none' else counts(2, 0, 3 if backend == 'nextest' else 0)
            receipt = dict(schema='jope.ci-receipt/v1', **self.expected, lane=lane, command=['bash','deploy/ci/fast-lane.sh',lane], dirty_source=[], dirty_source_after=[], inputs=inputs, started_at='2026-09-22T00:00:00+00:00', finished_at='2026-09-22T00:01:00+00:00', status='passed', exit_status=0, counts=result_counts,
                           evidence={p.name:digest(p) for p in folder.iterdir()})
            self.receipts[lane] = [(folder / 'receipt.json', receipt)]

    def verify(self):
        return aggregate.verify(self.mapping, self.plan, self.needs, self.receipts, self.expected)

    def test_valid_complete_execution(self):
        self.assertEqual([], self.verify()['errors'])

    def test_product_receipt_is_hashed_data_not_a_ci_lane(self):
        path, receipt = self.receipts['e2e-smoke'][0]
        product = path.parent / 'pipeline/a/receipt.json'
        product.parent.mkdir(parents=True)
        product.write_text('{"schema":"jope.pipeline-receipt/v2"}')
        receipt['evidence'][str(product.relative_to(path.parent))] = digest(product)
        for candidates in self.receipts.values():
            for ci_path, ci_receipt in candidates:
                ci_path.write_text(json.dumps(ci_receipt))
        discovered = aggregate.load_receipts(self.root)
        self.assertEqual(set(aggregate.plan_module.LANES), set(discovered))
        self.assertEqual([], aggregate.verify(self.mapping, self.plan, self.needs, discovered, self.expected)['errors'])

    def test_artifact_without_root_receipt_and_undeclared_files_fail(self):
        with self.assertRaises(ValueError):
            aggregate.load_receipts(self.root)
        (self.receipts['rust-test'][0][0].parent / 'junit-MISSING.txt').write_text('undeclared')
        self.assertTrue(any('manifest differs' in error for error in self.verify()['errors']))

    def test_wrong_source_checkout_attempt_and_run_fail(self):
        for field, wrong in [('source_sha','c'*40),('checkout_sha','c'*40),('attempt',1),('run_id','41')]:
            with self.subTest(field=field):
                receipt = self.receipts['rust-test'][0][1]
                old = receipt[field]
                receipt[field] = wrong
                self.assertTrue(any(field in error for error in self.verify()['errors']))
                receipt[field] = old

    def test_cancelled_failed_missing_and_unavailable_never_pass(self):
        for status in ('cancelled','failure','skipped','missing'):
            self.needs['python']['result'] = status
            self.assertEqual('failed', self.verify()['status'])
        self.needs['python']['result'] = 'success'
        for status in ('started','unavailable','failed'):
            self.receipts['python'][0][1]['status'] = status
            self.assertEqual('failed', self.verify()['status'])
        self.receipts.pop('python')
        self.assertEqual('failed', self.verify()['status'])

    def test_zero_collection_unknown_and_missing_results_fail(self):
        receipt = self.receipts['rust-test'][0][1]
        for value in (counts(0,0,0),dict.fromkeys(receipt['counts'])):
            receipt['counts'] = value
            self.assertEqual('failed', self.verify()['status'])
        (self.receipts['rust-test'][0][0].parent / 'junit.xml').unlink()
        self.assertEqual('failed', self.verify()['status'])

    def test_digest_tampering_and_dirty_sources_fail(self):
        path, receipt = self.receipts['python'][0]
        (path.parent / 'command.log').write_text('tampered')
        self.assertEqual('failed', self.verify()['status'])
        receipt['dirty_source'] = [' M Cargo.toml']
        self.assertTrue(any('dirty' in error for error in self.verify()['errors']))

    def test_docs_only_exclusions_pass_and_forged_plan_fails(self):
        self.expected['event'] = 'pull_request'
        self.plan.update(event='pull_request', paths=['README.md'])
        self.plan['lanes'], _ = aggregate.plan_module.classify(self.plan['paths'], 'pull_request')
        for lane, selected in self.plan['lanes'].items():
            if selected:
                self.receipts[lane][0][1]['event'] = 'pull_request'
            else:
                self.receipts.pop(lane)
                self.needs[lane]['result'] = 'skipped'
        self.assertEqual([], self.verify()['errors'])
        self.plan['paths'] = ['crates/jope-cli/src/main.rs']
        self.assertEqual('failed', self.verify()['status'])

    def test_panel_adapter_changes_require_python_execution(self):
        for name in ('_panel.py', 'test_panel.py', 'xgboost_adapter.py',
                     'lightgbm_adapter.py', 'catboost_adapter.py', 'sklearn_rf_adapter.py'):
            with self.subTest(adapter=name):
                lanes, _ = aggregate.plan_module.classify(
                    [f'crates/jope-bench/adapters/{name}'], 'pull_request')
                self.assertTrue(lanes['python'])
        lanes, _ = aggregate.plan_module.classify(
            ['crates/jope-bench/src/task.rs'], 'pull_request')
        self.assertFalse(lanes['python'])

    def test_campaign_policy_and_renderer_changes_require_inputs(self):
        for path in (
            'crates/jope-campaigns/src/spec_drift.rs',
            'crates/jope-campaigns/src/spec_policy.rs',
            'crates/jope-campaigns/src/bin/new_renderer.rs',
            'crates/jope-campaigns/src/test_adaptive/scq_batch.rs',
            'python/src/jope/research/new_renderer.py',
            'python/src/jope/research/jope-spec-policy.json',
        ):
            with self.subTest(path=path):
                lanes, _ = aggregate.plan_module.classify([path], 'pull_request')
                self.assertTrue(lanes['inputs'])
        for path in ('docs/README.md', 'python/src/jope/policy/model.py'):
            lanes, _ = aggregate.plan_module.classify([path], 'pull_request')
            self.assertFalse(lanes['inputs'])

    def test_duplicate_receipt_and_missing_dependency_fail(self):
        self.receipts['python'].append(copy.deepcopy(self.receipts['python'][0]))
        del self.needs['rust-check']
        self.assertEqual('failed', self.verify()['status'])


if __name__ == '__main__':
    unittest.main()
