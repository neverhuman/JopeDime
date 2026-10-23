#!/usr/bin/env python3
import copy
import json
import re
import shutil
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
        # Each rust-test matrix leg is its own execution unit; slice M of 2 selects fixtures M-1 and M+1
        # of the four the unsharded listing selects (the fifth is ignored in every listing).
        units = [lane for lane in aggregate.plan_module.LANES if lane != 'rust-test'] + ['rust-test-1', 'rust-test-2']
        for unit in units:
            lane = 'rust-test' if unit.startswith('rust-test') else unit
            folder = self.root / unit
            folder.mkdir()
            backend = self.mapping['lanes'][lane]['result_backend']
            logs = {'none':'verified fixture\n', 'rust':'test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n', 'unittest':'Ran 2 tests in 0.1s\n\nOK\n', 'pytest':'2 passed in 0.1s\n', 'nextest':'Summary [1s] 2 tests run: 2 passed, 3 skipped\n'}
            (folder / 'command.log').write_text(logs[backend])
            if lane == 'changes':
                (folder / 'plan.json').write_text(json.dumps(self.plan))
            if lane == 'python':
                (folder / 'junit.xml').write_text('<testsuites><testsuite tests="2" skipped="0"><testcase/><testcase/></testsuite></testsuites>')
            environment = {}
            if lane == 'rust-test':
                shard = int(unit[-1])
                environment['JOPE_RUST_TEST_PARTITION'] = f'slice:{shard}/2'
                mine = {str(shard - 1), str(shard + 1)}
                (folder / 'collection-unsharded.json').write_text(json.dumps(self.listing(set('0123'))))
                (folder / 'collection.json').write_text(json.dumps(self.listing(set('0123'), mine)))
                (folder / 'junit.xml').write_text(self.junit(sorted(mine)))
                for name in ('nextest-started', 'nextest.toml', 'disk-before.txt', 'disk-after.txt', 'target-bytes.txt'):
                    (folder / name).write_text('fixture')
            result_counts = dict.fromkeys(('collected','executed','passed','failed','skipped','excluded')) if backend == 'none' else counts(2, 0, 3 if backend == 'nextest' else 0)
            receipt = dict(schema='jope.ci-receipt/v1', **self.expected, lane=lane, command=['bash','ops/ci/fast-lane.sh',lane], dirty_source=[], dirty_source_after=[], inputs=inputs, started_at='2026-09-22T00:00:00+00:00', finished_at='2026-09-22T00:01:00+00:00', status='passed', exit_status=0, counts=result_counts,
                           environment=environment, evidence={p.name:digest(p) for p in folder.iterdir()})
            self.receipts[unit] = [(folder / 'receipt.json', receipt)]

    @staticmethod
    def listing(selected, partition=None):
        def case(name):
            if name not in selected:
                return {'ignored': True, 'filter-match': {'status': 'mismatch', 'reason': 'ignored'}}
            if partition is not None and name not in partition:
                return {'ignored': False, 'filter-match': {'status': 'mismatch', 'reason': 'partition'}}
            return {'ignored': False, 'filter-match': {'status': 'matches'}}
        return {'test-count': 5, 'rust-suites': {'fixture': {'status': 'listed', 'testcases': {str(n): case(str(n)) for n in range(5)}}}}

    @staticmethod
    def junit(names):
        return '<testsuites><testsuite>' + ''.join(f'<testcase classname="fixture" name="{n}"/>' for n in names) + '</testsuite></testsuites>'

    def rewrite(self, unit, name, text):
        path, receipt = self.receipts[unit][0]
        (path.parent / name).write_text(text)
        receipt['evidence'][name] = digest(path.parent / name)

    def verify(self):
        return aggregate.verify(self.mapping, self.plan, self.needs, self.receipts, self.expected)

    def test_valid_complete_execution(self):
        self.assertEqual([], self.verify()['errors'])

    def test_all_nine_receipt_units_match_actual_workflow_producer_commands(self):
        workflow = (aggregate.ROOT / '.github/workflows/ci.yml').read_text()
        producers = re.findall(r'python3 deploy/ci/receipt.py --lane ([a-z0-9-]+)[^\n]* -- bash ([a-zA-Z0-9_./-]+) ([a-z0-9-]+)$', workflow, re.M)
        self.assertEqual(len(aggregate.plan_module.LANES), len(producers))
        self.assertEqual(set(aggregate.plan_module.LANES), {lane for lane, _, _ in producers})
        commands = {}
        for lane, path, argument in producers:
            self.assertEqual(lane, argument)
            self.assertEqual('ops/ci/fast-lane.sh', path)
            commands[lane] = ['bash', path, argument]
        self.assertEqual(9, len(self.receipts))
        for unit, candidates in self.receipts.items():
            lane = candidates[0][1]['lane']
            candidates[0][1]['command'] = commands[lane]
        self.assertEqual([], self.verify()['errors'])

    def test_every_receipt_unit_rejects_bypass_alias_arguments_and_stale_identity(self):
        for unit, candidates in self.receipts.items():
            value = candidates[0][1]
            original = copy.deepcopy(value)
            lane = value['lane']
            for command in (['bash', 'deploy/ci/fast-lane.sh', lane],
                            ['bash', './ops/ci/fast-lane.sh', lane],
                            ['bash', 'ops/ci/fast-lane.sh', 'other-lane'],
                            ['bash', 'ops/ci/fast-lane.sh', lane, '--extra'],
                            ['bash', '-c', 'true']):
                with self.subTest(unit=unit, command=command):
                    value['command'] = command
                    self.assertIn(unit + ': unexpected command', self.verify()['errors'])
            value.clear()
            value.update(copy.deepcopy(original))
            for field, wrong in (('source_sha', 'c' * 40), ('run_id', '43'), ('attempt', 3)):
                with self.subTest(unit=unit, field=field):
                    value[field] = wrong
                    self.assertIn(unit + ': ' + field + ' mismatch', self.verify()['errors'])
                    value[field] = original[field]

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
        self.assertEqual({unit for lane in aggregate.plan_module.LANES for unit in aggregate.lane_units(lane)}, set(discovered))
        self.assertEqual([], aggregate.verify(self.mapping, self.plan, self.needs, discovered, self.expected)['errors'])

    def test_artifact_without_root_receipt_and_undeclared_files_fail(self):
        with self.assertRaises(ValueError):
            aggregate.load_receipts(self.root)
        (self.receipts['rust-test-1'][0][0].parent / 'junit-MISSING.txt').write_text('undeclared')
        self.assertTrue(any('manifest differs' in error for error in self.verify()['errors']))

    def test_wrong_source_checkout_attempt_and_run_fail(self):
        for field, wrong in [('source_sha','c'*40),('checkout_sha','c'*40),('attempt',1),('run_id','41')]:
            with self.subTest(field=field):
                receipt = self.receipts['rust-test-2'][0][1]
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
        receipt = self.receipts['rust-test-1'][0][1]
        for value in (counts(0,0,0),dict.fromkeys(receipt['counts'])):
            receipt['counts'] = value
            self.assertEqual('failed', self.verify()['status'])
        (self.receipts['rust-test-1'][0][0].parent / 'junit.xml').unlink()
        self.assertEqual('failed', self.verify()['status'])

    def test_rust_test_requires_every_shard_once_with_its_own_slice(self):
        report = self.verify()
        self.assertEqual('passed', report['lanes']['rust-test']['partition']['status'])
        self.assertEqual(4, report['lanes']['rust-test']['partition']['union']['selected'])
        for change in ('missing', 'duplicate', 'wrong-slice', 'unrecorded-slice'):
            with self.subTest(change=change):
                saved = copy.deepcopy(self.receipts)
                if change == 'missing':
                    self.receipts.pop('rust-test-2')
                elif change == 'duplicate':
                    self.receipts['rust-test-2'].append(self.receipts['rust-test-2'][0])
                elif change == 'wrong-slice':
                    self.receipts['rust-test-2'][0][1]['environment']['JOPE_RUST_TEST_PARTITION'] = 'slice:2/3'
                else:
                    del self.receipts['rust-test-2'][0][1]['environment']['JOPE_RUST_TEST_PARTITION']
                errors = self.verify()['errors']
                self.assertTrue(any(error.startswith('rust-test-2:') for error in errors), errors)
                self.receipts = saved

    def test_shards_must_select_and_execute_the_unsharded_set_exactly_once(self):
        for change, expected in [('overlap', 'more than one shard'), ('gap', 'in no shard'),
                                 ('unexecuted', 'executed 1 tests, selected 2'), ('beyond-partition', 'beyond the partition'),
                                 ('unsharded-differs', 'differs between shards')]:
            with self.subTest(change=change):
                if change == 'overlap':
                    self.rewrite('rust-test-2', 'collection.json', json.dumps(self.listing(set('0123'), {'0', '1', '3'})))
                    self.rewrite('rust-test-2', 'junit.xml', self.junit(['0', '1', '3']))
                elif change == 'gap':
                    self.rewrite('rust-test-2', 'collection.json', json.dumps(self.listing(set('0123'), {'1'})))
                    self.rewrite('rust-test-2', 'junit.xml', self.junit(['1']))
                elif change == 'unexecuted':
                    self.rewrite('rust-test-2', 'junit.xml', self.junit(['1']))
                elif change == 'beyond-partition':
                    self.rewrite('rust-test-2', 'collection.json', json.dumps(self.listing(set('013'), {'1', '3'})))
                else:
                    self.rewrite('rust-test-2', 'collection-unsharded.json', json.dumps(self.listing(set('013'))))
                errors = self.verify()['errors']
                self.assertTrue(any(error.startswith('rust-test partition:') and expected in error for error in errors), errors)
                self.setUp()

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
            for unit in aggregate.lane_units(lane):
                if selected:
                    self.receipts[unit][0][1]['event'] = 'pull_request'
                else:
                    self.receipts.pop(unit)
            if not selected:
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
            'crates/jope-campaigns/src/commands/train_only_redteam.rs',
            'crates/jope-arena/src/elo.rs',
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

    def test_pytest_evidence_helper_selects_shared_lanes(self):
        lanes, _ = aggregate.plan_module.classify(
            ['deploy/ci/pytest-evidence.sh'], 'pull_request')
        self.assertTrue(all(lanes.values()), lanes)

    def test_duplicate_receipt_and_missing_dependency_fail(self):
        self.receipts['python'].append(copy.deepcopy(self.receipts['python'][0]))
        del self.needs['rust-check']
        self.assertEqual('failed', self.verify()['status'])


class PublicationTests(unittest.TestCase):
    setUp = AggregateTests.setUp
    listing = staticmethod(AggregateTests.listing)
    junit = staticmethod(AggregateTests.junit)

    def publications(self):
        for candidates in self.receipts.values():
            for path, receipt in candidates:
                path.write_text(json.dumps(receipt))
        original = self.receipts['python'][0][0].parent
        first = original.with_name('ci-42-2-python-pub-1')
        original.rename(first)
        second = first.with_name('ci-42-2-python-pub-2')
        shutil.copytree(first, second)
        return first, second

    def test_acknowledgement_loss_coalesces_only_identical_publications(self):
        first, second = self.publications()
        publications = {}
        received = aggregate.load_receipts(self.root, publications)
        self.assertEqual([first.name, second.name], publications['python'])
        self.assertEqual(1, len(received['python']))
        self.assertEqual([], aggregate.verify(self.mapping, self.plan, self.needs, received, self.expected)['errors'])
        # Recovering transport must not repair a failed test job or stale identity.
        self.needs['python']['result'] = 'failure'
        self.assertEqual('failed', aggregate.verify(self.mapping, self.plan, self.needs, received, self.expected)['status'])
        self.needs['python']['result'] = 'success'
        changed = dict(self.expected, attempt=3)
        self.assertEqual('failed', aggregate.verify(self.mapping, self.plan, self.needs, received, changed)['status'])

    def test_divergent_or_unbounded_publications_refuse(self):
        first, second = self.publications()
        log = second/'command.log'
        original = log.read_bytes()
        log.write_bytes(original+b'changed')
        with self.assertRaisesRegex(ValueError, 'divergent'):
            aggregate.load_receipts(self.root)
        log.write_bytes(original)
        unexpected = second/'extra-file'
        unexpected.write_text('unmanifested')
        with self.assertRaisesRegex(ValueError, 'divergent'):
            aggregate.load_receipts(self.root)
        unexpected.unlink()
        fourth = second.with_name('ci-42-2-python-pub-4')
        second.rename(fourth)
        with self.assertRaisesRegex(ValueError, 'invalid publication'):
            aggregate.load_receipts(self.root)
        fourth.rename(second)
        nested = second/'link'
        nested.symlink_to(first/'command.log')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            aggregate.load_receipts(self.root)

    def test_unlabelled_duplicate_execution_still_refuses(self):
        first, second = self.publications()
        second.rename(second.with_name('unlabelled-copy'))
        with self.assertRaisesRegex(ValueError, 'ambiguous execution'):
            aggregate.load_receipts(self.root)

    def test_missing_first_publication_can_recover_from_second(self):
        first, second = self.publications()
        shutil.rmtree(first)  # synthetic fixture owned by this test
        received = aggregate.load_receipts(self.root)
        self.assertEqual([], aggregate.verify(self.mapping, self.plan, self.needs, received, self.expected)['errors'])
        path = second/'receipt.json'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'lacks a root'):
            aggregate.load_receipts(self.root)



class GoEvidenceTests(unittest.TestCase):
    def test_missing_zero_incomplete_and_uncovered_results_are_refused(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('go_verify', aggregate.ROOT / 'deploy/ci/go/verify.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            profile = root / 'coverage.out'
            with self.assertRaises(FileNotFoundError): module.verify(root,profile)
            for name in ('recipes','fitted','bindings','pipeline','cli','census'):
                (root / (name+'.log')).write_text('test result: ok. 0 passed; 0 failed\n')
            with self.assertRaises(AssertionError): module.verify(root,profile)
            def good(name):
                return ('test result: ok. %d passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;\nGo 0: fixture\nGo 1: fixture\n' %
                        dict(recipes=84,fitted=2,bindings=21,pipeline=18,cli=2,census=5)[name]
                        + ('test go_runtime_is_required ... ok\n' if name == 'recipes' else ''))
            for name in ('recipes','fitted','bindings','pipeline','cli','census'):
                (root / (name+'.log')).write_text(good(name))
            census=[dict(kind=k,total=1,executed=['x'],refused=[],unaccounted=0) for k in ('recipe','bound','legacy-bound')]
            census.append(dict(kind='v2',total=1,executed=1,unaccounted=0))
            text = good('census') + 'Go 0: fixture\nGo 1: fixture\n'*9 + 'Go metadata: strict heading qualified\n'
            text += ''.join('Go census: '+json.dumps(c)+'\n' for c in census)
            (root/'census.log').write_text(text);(root/'contracts.log').write_text('PASS\n')
            profile.write_text('mode: atomic\njope.generated/model/model.go:1.1,2.1 1 1\n')
            self.assertEqual(len(module.verify(root,profile)),4)
            for name in ('recipes','fitted','bindings','pipeline','cli'):
                (root/(name+'.log')).write_text('test result: ok. 1 passed; 0 failed\nGo 0: fixture\nGo 1: fixture\n')
                with self.assertRaises(AssertionError): module.verify(root,profile)
                (root/(name+'.log')).write_text(good(name))
            (root/'recipes.log').write_text(good('recipes').replace('test go_runtime_is_required ... ok\n',''))
            with self.assertRaises(AssertionError): module.verify(root,profile)
            (root/'recipes.log').write_text(good('recipes'))
            (root/'recipes.log').write_text(good('recipes').replace('0 filtered out','1 filtered out'))
            with self.assertRaises(AssertionError): module.verify(root,profile)
            (root/'recipes.log').write_text(good('recipes'))
            for wrong in ('mode: atomic\n','mode: atomic\njope.generated/model/model.go:1.1,2.1 1 0\n'):
                profile.write_text(wrong)
                with self.assertRaises(AssertionError): module.verify(root,profile)
            profile.write_text('mode: atomic\njope.generated/model/model.go:1.1,2.1 1 1\n')
            (root/'census.log').write_text(text.replace('Go 1:','missing:'))
            with self.assertRaises(AssertionError): module.verify(root,profile)

if __name__ == '__main__':
    unittest.main()
