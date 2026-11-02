#!/usr/bin/env python3
import copy
import hashlib
import io
import importlib.util
import json
import os
import re
import tomllib
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
evidence_spec = importlib.util.spec_from_file_location('aggregate_evidence', ROOT / 'deploy/ci/aggregate-evidence.py')
aggregate_evidence = importlib.util.module_from_spec(evidence_spec)
evidence_spec.loader.exec_module(aggregate_evidence)
partition_spec = importlib.util.spec_from_file_location('check_test_partition', ROOT / 'deploy/ci/check_test_partition.py')
partition = importlib.util.module_from_spec(partition_spec)
partition_spec.loader.exec_module(partition)


class FullMainTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {'lanes': {'full-one': {'command':['fixture'], 'requires_nonzero_tests':False, 'result_backend':'none'}}, 'dispositions':{}}
        self.expected = dict(source_sha='a'*40, checkout_sha='a'*40, run_id='123', attempt=1, event='push')

    def test_static_partition_uses_main_for_zero_push_base(self):
        self.assertEqual('origin/main', partition.static_base('0' * 40))
        self.assertEqual('a' * 40, partition.static_base('a' * 40))

    def test_aggregate_needs_filter_preserves_all_nonpublication_results(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source, output = root/'all-needs.json', root/'nested'/'needs.json'
            needs = {'cpu': {'result': 'success'}, 'paper': {'result': 'success'}}
            source.write_text(json.dumps(needs))
            aggregate_evidence.filter_needs(source, output, ['paper'])
            self.assertEqual({'cpu': {'result': 'success'}}, json.loads(output.read_text()))
            with self.assertRaisesRegex(ValueError, 'missing expected workflow dependency'):
                aggregate_evidence.filter_needs(source, root/'bad.json', ['go'])
            self.assertFalse((root/'bad.json').exists())

    def test_frontend_verifier_requires_exact_source_nonzero_and_full_pass(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)/'nested'
            root.mkdir(parents=True)
            receipt_path = root/'receipt.json'
            receipt_path.write_text(json.dumps({
                'schema': 'jope.observatory-ui-ci/v1', 'source_sha': 'a'*40,
                'run_id': '123', 'attempt': '1', 'executed': 4, 'passed': 4, 'failed': 0,
            }))
            aggregate_evidence.verify_frontend(Path(scratch), 'a'*40, '123', '2')
            data = json.loads(receipt_path.read_text())
            data['executed'] = data['passed'] = 0
            receipt_path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                aggregate_evidence.verify_frontend(Path(scratch), 'a'*40, '123', '2')

    def test_publication_verifier_requires_exact_head_and_complete_hash_manifest(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)/'publication'
            (root/'evidence').mkdir(parents=True)
            command = root/'command.log'
            identity_path = root/'evidence'/'source-identity.json'
            command.write_text('publication checks passed\n')
            tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
            identity = {
                'schema': 'jope.publication-source/v1', 'source_sha': 'a'*40,
                'checkout_sha': 'b'*40, 'event_sha': 'b'*40, 'run_id': '123',
                'attempt': 1, 'event': 'workflow_dispatch', 'source_tree': tree,
                'checkout_tree': tree,
            }
            identity_path.write_text(json.dumps(identity))
            evidence = {
                'command.log': hashlib.sha256(command.read_bytes()).hexdigest(),
                'evidence/source-identity.json': hashlib.sha256(identity_path.read_bytes()).hexdigest(),
            }
            receipt_path = root/'receipt.json'
            receipt_path.write_text(json.dumps({
                'schema': 'jope.ci-receipt/v1', 'lane': 'paper', 'source_sha': 'a'*40,
                'checkout_sha': 'b'*40, 'run_id': '123', 'attempt': 1,
                'event': 'workflow_dispatch', 'status': 'passed', 'exit_status': 0,
                'dirty_source': [], 'dirty_source_after': [],
                'command': ['make', '-C', 'docs/paper/innovationzero', 'check', 'self-test'],
                'evidence': evidence,
            }))
            aggregate_evidence.verify_publication(Path(scratch), 'a'*40, 'b'*40, '123', '1', 'workflow_dispatch')
            command.write_text('changed evidence\n')
            with self.assertRaises(AssertionError):
                aggregate_evidence.verify_publication(Path(scratch), 'a'*40, 'b'*40, '123', '1', 'workflow_dispatch')

    def test_rendered_ux_verifier_binds_ui_attempt_and_event_sha(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root/'ui').mkdir()
            (root/'ui'/'receipt.json').write_text(json.dumps({'attempt': '2'}))
            (root/'ui'/'rendered-receipt.json').write_text('{}')
            with patch.object(aggregate_evidence.subprocess, 'run') as run:
                aggregate_evidence.verify_rendered(root, 'a'*40, '123', '3', 'b'*40)
            argv = run.call_args.args[0]
            self.assertEqual(['python3', 'deploy/ci/rendered-ux-receipt.py', 'verify'], argv[:3])
            self.assertEqual(str(root/'ui'), argv[argv.index('--root') + 1])
            self.assertEqual(['--attempt', '2'], argv[argv.index('--attempt'):argv.index('--attempt') + 2])
            self.assertEqual(['--event-sha', 'b'*40], argv[-2:])
            (root/'ui'/'receipt.json').write_text(json.dumps({'attempt': '4'}))
            with self.assertRaises(AssertionError):
                aggregate_evidence.verify_rendered(root, 'a'*40, '123', '3', 'b'*40)


    def test_export_parity_verifier_checks_all_emitted_bytes_and_source_identity(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            payloads = {
                'harness.sh': b'harness', 'generate.rs': b'generator',
                'registry.json': json.dumps({'entries': [
                    {'kind': 'module', 'name': 'sample', 'supported': {'rust': True, 'python': False}},
                ]}).encode(),
                'fixtures.json': b'{}', 'sample/model.json': b'{}',
                'sample/predictor.rs': b'rust', 'sample/predictor.py': b'python',
                'sample/predictor.mjs': b'javascript', 'stdout.log': b'ok', 'stderr.log': b'',
            }
            for name, data in payloads.items():
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            files = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                     for name, data in payloads.items()}
            source_sha = 'a'*40
            report = {
                'schema': 'jope.export-parity/v1', 'status': 'pass', 'passed': True,
                'ci_executed': True, 'source_stable': True, 'source_dirty': [],
                'source_sha': source_sha, 'source_end': source_sha,
                'harness_sha256': hashlib.sha256((ROOT/'deploy/ci/export-parity.sh').read_bytes()).hexdigest(),
                'cargo_lock_sha256': hashlib.sha256((ROOT/'Cargo.lock').read_bytes()).hexdigest(),
                'ci_context': {'GITHUB_REPOSITORY': 'neverhuman/JopeDime',
                    'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1'},
                'fault_injected': False,
                'toolchains': {'rustc': 'rustc 1.97.1 (fixture)', 'python': 'Python 3.12.3', 'node': 'v26.1.0'},
                'registry_sha256': files['registry.json']['sha256'],
                'fixture_sha256': files['fixtures.json']['sha256'],
                'cells': [
                    {'key': 'Module/sample', 'language': 'rust', 'supported': True, 'status': 'pass'},
                    {'key': 'Module/sample', 'language': 'python', 'supported': False, 'status': 'unsupported_registry'},
                ],
                'expected_supported_cells': 1, 'executed_supported_cells': 1,
                'cases': [
                    {'status': 'pass', 'self_check_rows': 64, 'language': 'rust',
                     'decision_measurement': True, 'golden_flip_rate': 0, 'randomized_flip_rate': 0},
                    {'status': 'pass', 'self_check_rows': 64, 'language': 'python'},
                    {'status': 'pass', 'self_check_rows': 64, 'language': 'javascript'},
                ],
                'expected_emitted_modules': 3, 'executed_emitted_modules': 3,
                'raw_numeric_contract': [{'status': 'pass', 'rows': 1} for _ in range(3)],
                'commands': [{'argv': ['fixture'], 'exit': 0, 'stdout': 'stdout.log', 'stderr': 'stderr.log', 'seconds': 0.1}],
                'files': files,
            }
            report_path = root/'report.json'
            report_path.write_text(json.dumps(report))
            aggregate_evidence.verify_export_parity(root, source_sha, '123', '1', 'neverhuman/JopeDime', 'workflow_dispatch')
            (root/'sample/predictor.py').write_bytes(b'tampered')
            with self.assertRaises(AssertionError):
                aggregate_evidence.verify_export_parity(root, source_sha, '123', '1', 'neverhuman/JopeDime', 'workflow_dispatch')

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
        workflow = (ROOT/'.github/workflows/full-main.yml').read_text()
        # The cpu job keeps the single job's deadline, which fits every family, so it fits each group.
        job_minutes = int(re.search(r'^    timeout-minutes: (\d+)$', workflow, re.M)[1])
        self.assertEqual(145, job_minutes)
        self.assertLessEqual(total, (job_minutes - 5) * 60)
        self.assertEqual([], full.partition_errors(lanes, full.GROUPS))
        self.assertEqual(['cpu-1', 'cpu-2'], list(full.GROUPS))
        self.assertEqual(1, workflow.count('        group: [cpu-1, cpu-2]\n'))

    def test_groups_run_only_their_families_in_declared_order(self):
        mapping = copy.deepcopy(self.mapping)
        for name in ('full-two', 'full-three'):
            mapping['lanes'][name] = dict(self.mapping['lanes']['full-one'])
        calls = []
        def execute(name, command, destination, backend, junit):
            calls.append(name)
            return 0
        groups = {'cpu-1': ('full-three', 'full-one'), 'cpu-2': ('full-two',)}
        with tempfile.TemporaryDirectory() as scratch, patch.object(full, 'GROUPS', groups):
            self.assertEqual(0, full.run(mapping, Path(scratch)/'one', execute, 'cpu-1'))
            self.assertEqual(['full-one', 'full-three'], calls)
            self.assertEqual(0, full.run(mapping, Path(scratch)/'all', execute))
            self.assertEqual(['full-one', 'full-three', 'full-one', 'full-two', 'full-three'], calls)
            for broken in ({'cpu-1': ('full-one', 'full-three')},
                           {'cpu-1': ('full-one', 'full-three'), 'cpu-2': ('full-two', 'full-one')},
                           {'cpu-1': ('full-one', 'full-three'), 'cpu-2': ('full-two', 'full-four')}):
                with self.subTest(broken=broken), patch.object(full, 'GROUPS', broken):
                    with self.assertRaisesRegex(ValueError, 'do not partition'):
                        full.run(mapping, Path(scratch)/'never', execute, 'cpu-1')
                    self.assertFalse((Path(scratch)/'never').exists())
            with self.assertRaisesRegex(ValueError, 'unknown full CPU group'):
                full.run(mapping, Path(scratch)/'never', execute, 'cpu-3')


class FullAggregateTests(unittest.TestCase):
    setUp = FullMainTests.setUp

    def downloads(self, root, groups):
        mapping = copy.deepcopy(self.mapping)
        for group, names in groups.items():
            folder = root / f'full-cpu-123-1-{group}'
            folder.mkdir()
            for name in names:
                mapping['lanes'][name] = dict(self.mapping['lanes']['full-one'], command=['fixture', name])
                (folder / name).mkdir()
                (folder / name / 'command.log').write_text('fixture\n')
                data = self.passing_receipt(name)
                data['evidence'] = {'command.log': receipt.digest(folder / name / 'command.log')}
                (folder / name / 'receipt.json').write_text(json.dumps(data))
        return mapping

    def passing_receipt(self, name):
        return dict(schema='jope.ci-receipt/v1', **self.expected, lane=name, command=['fixture', name], dirty_source=[], dirty_source_after=[],
                    status='passed', exit_status=0, finished_at='2026-09-22T00:00:01+00:00',
                    inputs={name: receipt.digest(ROOT/name) for name in receipt.INPUTS},
                    counts=dict.fromkeys(('collected','executed','passed','failed','skipped','excluded')), evidence={})

    def inventories(self, root, mapping, groups):
        for group in groups:
            report = full.summarize(mapping, root / f'full-cpu-123-1-{group}', self.expected, group)
            (root / f'full-cpu-123-1-{group}' / 'inventory.json').write_text(json.dumps(report))

    def test_every_family_once_across_groups_or_fail(self):
        groups = {'cpu-1': ('full-one', 'full-three'), 'cpu-2': ('full-two',)}
        needs = {'cpu': {'result': 'success', 'outputs': {}}}
        with patch.object(full, 'GROUPS', groups):
            with tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                mapping = self.downloads(root, groups)
                self.inventories(root, mapping, groups)
                report = full.aggregate(mapping, root, needs, self.expected)
                self.assertEqual([], report['errors'])
                self.assertEqual(['full-one', 'full-three', 'full-two'], report['partition']['executed'])
                self.assertEqual('failed', full.aggregate(mapping, root, {'cpu': {'result': 'failure'}}, self.expected)['status'])
                self.assertEqual('failed', full.aggregate(mapping, root, needs, dict(self.expected, attempt=2))['status'])
                (root / 'full-cpu-123-1-cpu-1' / 'full-three' / 'command.log').write_text('tampered')
                report = full.aggregate(mapping, root, needs, self.expected)
                self.assertIn('cpu-1: full-three: unsafe or changed evidence: command.log', report['errors'])
            with tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                mapping = self.downloads(root, groups)
                self.inventories(root, mapping, {'cpu-1': groups['cpu-1']})
                report = full.aggregate(mapping, root, needs, self.expected)
                self.assertIn('family full-two has no verified receipt', report['errors'])
                self.inventories(root, mapping, {'cpu-2': groups['cpu-2']})
                self.assertEqual('passed', full.aggregate(mapping, root, needs, self.expected)['status'])
                (root / 'full-cpu-123-1-cpu-3').mkdir()
                self.assertIn('unexpected artifact full-cpu-123-1-cpu-3', full.aggregate(mapping, root, needs, self.expected)['errors'])
                (root / 'full-cpu-123-1-cpu-3').rmdir()
                recorded = json.loads((root / 'full-cpu-123-1-cpu-2' / 'inventory.json').read_text())
                (root / 'full-cpu-123-1-cpu-2' / 'inventory.json').write_text(json.dumps(dict(recorded, group='cpu-1')))
                self.assertEqual('failed', full.aggregate(mapping, root, needs, self.expected)['status'])
            # A family a group does not hold is never reported as verified by another group.
            overlap = {'cpu-1': ('full-one', 'full-three'), 'cpu-2': ('full-two', 'full-three')}
            with patch.object(full, 'GROUPS', overlap), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                mapping = self.downloads(root, overlap)
                report = full.aggregate(mapping, root, needs, self.expected)
                self.assertIn('family full-three is in 2 groups', report['errors'])

    def test_scratch_family_covers_main_test_deadline_and_collection(self):
        mapping = json.loads((ROOT/'deploy/ci/coverage.json').read_text())
        config = tomllib.loads((ROOT/'.config/nextest.toml').read_text())
        timeout = config['profile']['main']['slow-timeout']
        test_seconds = int(timeout['period'].removesuffix('s')) * timeout['terminate-after']
        family_seconds = int(mapping['lanes']['full-scratch']['command'][3].removesuffix('s'))
        # Collection/build consumed 252s before the golden started in run37287624697.
        # The family must leave the existing per-test deadline intact, plus 300s setup.
        self.assertGreaterEqual(family_seconds, test_seconds + 300)

    def test_full_config_preserves_selection_deadlines_and_failure_policy(self):
        script = (ROOT/'deploy/ci/full-lane.sh').read_text()
        snippets = re.findall(r"python3 - <<'PY'\n(.*?)\nPY", script, re.S)
        configure = next(text for text in snippets if 'junit-main.xml' in text)
        original = (ROOT/'.config/nextest.toml').read_text()
        expected = tomllib.loads(original)
        for family in ('full-default', 'full-scratch', 'full-policy', 'full-s3-offline', 'full-neural'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                (root/'nextest.toml').write_text(original)
                subprocess.run([sys.executable, '-c', configure],
                               env=dict(os.environ, JOPE_LANE_EVIDENCE=str(root), JOPE_FULL_FAMILY=family), check=True)
                actual = tomllib.loads((root/'nextest.toml').read_text())
                expected['profile']['main']['junit']['path'] = str(root/'junit.xml')
                # Only scheduling may change. No default filter, timeout, retry,
                # fail-fast, or checked-in profile changes are permitted.
                override = actual['profile']['main'].pop('overrides')
                golden = override.pop(0)
                self.assertEqual(dict(
                    filter='package(feat-math) & binary(=golden) & (test(=fast_profile_portfolio_head) | test(=golden_ridge_gcv_screen_wide) | test(=golden_portfolio_core))',
                    priority=100), golden)
                if family == 'full-neural':
                    self.assertEqual({'full-neural-runtime-proof': {'max-threads': 12}}, actual.pop('test-groups'))
                    self.assertEqual([
                        dict(filter='package(jope-parallel) & test(=teaching_store::read_once_tests::a_search_receipt_is_validated_from_the_bytes_its_ledger_commits)',
                             priority=100, **{'threads-required': 'num-test-threads', 'test-group': 'full-neural-runtime-proof'}),
                        dict(filter='package(jope-parallel) & binary(=view_contracts) & test(=projected_gp_candidates_are_exactly_representable_unique_and_durable)',
                             priority=96, **{'test-group': 'full-neural-runtime-proof'}),
                        dict(filter='package(jope-parallel) & (test(teaching_store::read_once_tests::) | test(=teaching_store::admission_tests::native_strict_admission_quarantines_conflicts_before_dedup_and_keeps_outcomes))',
                             priority=95, **{'test-group': 'full-neural-runtime-proof'}),
                        dict(filter='package(jope-parallel) & (test(teaching_store::read_once_tests::) | test(teaching_store::admission_tests::) | test(eval_registry::authority::) | test(eval_registry::witness::))',
                             priority=90, **{'test-group': 'full-neural-runtime-proof'})], override)
                else:
                    self.assertEqual([], override)
                self.assertEqual(expected, actual)


class InputBuildTests(unittest.TestCase):
    def test_joint_build_keeps_both_checks_and_fails_closed(self):
        scratch_root = ROOT / 'target' / 'tmp'
        scratch_root.mkdir(parents=True, exist_ok=True)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for failure in ('none', 'verify', 'generate', 'build', 'policy', 'elo'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
                root = Path(scratch)
                cargo = root / 'cargo'
                cargo.write_text('''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as out:
    out.write(json.dumps(args) + '\\n')
stage = 'build' if '--no-run' in args else 'policy' if '--lib' in args else 'elo' if args[0] == 'test' else 'verify' if 'verify-inputs' in args else 'generate'
if os.environ['FAILURE'] == stage:
    sys.exit(7)
if stage in ('policy', 'elo'):
    passed = 13 if stage == 'policy' else 2
    print(f'test result: ok. {passed} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out')
''')
                cargo.chmod(0o755)
                evidence = root / 'evidence'
                evidence.mkdir()
                calls = root / 'calls.jsonl'
                result = subprocess.run(['bash', str(ROOT/'deploy/ci/full-lane.sh'), 'full-inputs'],
                                        env=dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                                                 CALLS=str(calls), FAILURE=failure, JOPE_LANE_EVIDENCE=str(evidence),
                                                 JOPE_SOURCE_SHA=head, GITHUB_SHA=head, GITHUB_EVENT_NAME='local-proof',
                                                 GITHUB_RUN_ID='1', GITHUB_RUN_ATTEMPT='1'),
                                        capture_output=True, text=True)
                executed = [json.loads(line) for line in calls.read_text().splitlines()]
                self.assertEqual(['run', '--locked', '--bin', 'jope', '--', 'verify-inputs', '--offline', '--json',
                                  '--root', 'datasets/synthetic'], executed[0])
                self.assertEqual(0 if failure == 'none' else 7, result.returncode, result.stdout+result.stderr)
                self.assertEqual({'verify': 1, 'generate': 2, 'build': 3, 'policy': 4, 'elo': 5, 'none': 5}[failure], len(executed))
                if failure != 'verify':
                    self.assertEqual(['run', '--locked', '--bin', 'jope', '--', 'generate-fixtures', '--check', '--json',
                                      '--output', 'datasets/synthetic'], executed[1])
                if failure not in ('verify', 'generate'):
                    self.assertEqual(['test', '--locked', '-p', 'jope-dime-regressor', '--lib', '--bin',
                                      'jope-dime-regressor', '--no-run'], executed[2])
                if failure in ('none', 'policy', 'elo'):
                    self.assertEqual(['test', '--locked', '-p', 'jope-dime-regressor', '--lib',
                                      'spec_drift::tests', '--', '--test-threads=1'], executed[3])
                if failure in ('none', 'elo'):
                    self.assertEqual(['test', '--locked', '-p', 'jope-dime-regressor', '--bin', 'jope-dime-regressor',
                                      'commands::train_only_redteam::tests::bradley_terry', '--', '--test-threads=1'], executed[4])
                if failure == 'none':
                    for name, passed in [('campaign-spec-policy', 13), ('campaign-elo', 2)]:
                        data = json.loads((evidence/name/'receipt.json').read_text())
                        self.assertEqual(passed, data['counts']['passed'])
                        self.assertEqual(0, data['exit_status'])


class FittedMatrixTests(unittest.TestCase):
    def test_parallel_matrices_both_finish_and_fail_closed(self):
        script = (ROOT/'deploy/ci/fast-lane.sh').read_text()
        block = re.search(r'(if \[ "\$\{JOPE_FULL_EMITTERS_PARALLEL:-0\}".*?\nfi)\n    ;;', script, re.S)[1]
        scratch_root = ROOT/'target/tmp'
        scratch_root.mkdir(parents=True, exist_ok=True)
        for parallel, failure in [(True, 'none'), (True, 'build'),
                                  (True, 'sanitizer'), (True, 'cxx20'), (False, 'none')]:
            with self.subTest(parallel=parallel, failure=failure), tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
                root = Path(scratch)
                cargo = root/'cargo'
                cargo.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ['MATRIX_ROOT'])
args = sys.argv[1:]
if '--no-run' in args:
    (root/'build.json').write_text(json.dumps(args))
    sys.exit(7 if os.environ['FAILURE'] == 'build' else 0)
mode = 'sanitizer' if os.environ.get('JOPE_FITTED_CPP_SANITIZER') else 'cxx20'
(root/(mode+'.json')).write_text(json.dumps(args))
if os.environ['JOPE_FULL_EMITTERS_PARALLEL'] == '1':
    assert (root/'build.json').exists()
    (root/(mode+'.started')).touch()
    other = 'cxx20' if mode == 'sanitizer' else 'sanitizer'
    deadline = time.monotonic() + 5
    while not (root/(other+'.started')).exists():
        assert time.monotonic() < deadline, 'matrices did not overlap'
        time.sleep(0.01)
(root/(mode+'.finished')).touch()
print(mode+' completed', flush=True)
sys.exit(7 if os.environ['FAILURE'] == mode else 0)
''')
                cargo.chmod(0o755)
                env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                           MATRIX_ROOT=str(root), EVIDENCE=str(root), FAILURE=failure,
                           JOPE_FULL_EMITTERS_PARALLEL='1' if parallel else '0')
                env.pop('JOPE_FITTED_CPP_SANITIZER', None)
                env.pop('JOPE_CPP_STD', None)
                result = subprocess.run(['bash', '-euo', 'pipefail', '-c', block],
                                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(0 if failure == 'none' else 7, result.returncode, result.stderr)
                if failure == 'build':
                    self.assertFalse((root/'sanitizer.json').exists())
                    self.assertFalse((root/'cxx20.json').exists())
                    continue
                original = ['test', '-p', 'feat-math', '--locked', '--test', 'advanced_v3_fitted_cpp']
                expected = original+['--', '--test-threads=4'] if parallel else original
                for mode in ('sanitizer', 'cxx20'):
                    self.assertEqual(expected, json.loads((root/(mode+'.json')).read_text()))
                    self.assertTrue((root/(mode+'.finished')).exists())
                    self.assertIn(mode+' completed', result.stdout)
                if parallel:
                    self.assertEqual(original+['--no-run'], json.loads((root/'build.json').read_text()))
                    for name in ('fitted-sanitizer.log', 'fitted-cxx20.log'):
                        self.assertTrue((root/name).is_file())
                else:
                    self.assertFalse((root/'build.json').exists())


class SyntheticExecutorTests(unittest.TestCase):
    def test_collection_binds_helper_and_refuses_missing_or_wrong_executor(self):
        script = (ROOT / 'deploy/ci/full-lane.sh').read_text()
        snippet = re.search(r"python3 - <<'PYBUILD'\n(.*?)\nPYBUILD", script, re.S)[1]
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            executable = root / 'executor'
            executable.write_bytes(b'same-source fixture executable')
            suite = {'kind': 'lib', 'package-name': 'jope-parallel',
                     'binary-path': str(executable),
                     'testcases': {'eval_registry::authority::tests::synthetic_executor': {}}}
            collection = root / 'synthetic-executor-collection.json'
            collection.write_text(json.dumps({'rust-suites': {'jope-parallel': suite}}))
            env = dict(os.environ, JOPE_LANE_EVIDENCE=str(root), JOPE_SOURCE_SHA='a' * 40)
            result = subprocess.run([sys.executable, '-c', snippet], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(str(executable), result.stdout.strip())
            record = root / 'synthetic-executor.json'
            self.assertEqual(dict(executable=str(executable), sha256=receipt.digest(executable),
                                  features=[], feature_selection='no-default-features',
                                  source_sha='a' * 40,
                                  scope='same-source synthetic child; collection only, no test execution'),
                             json.loads(record.read_text()))
            record.unlink()
            for field, value in [('kind', 'bin'), ('package-name', 'other'),
                                 ('testcases', {}), ('binary-path', str(root / 'absent'))]:
                with self.subTest(field=field):
                    invalid = dict(suite, **{field: value})
                    collection.write_text(json.dumps({'rust-suites': {'jope-parallel': invalid}}))
                    result = subprocess.run([sys.executable, '-c', snippet], env=env,
                                            capture_output=True, text=True)
                    self.assertNotEqual(0, result.returncode)
                    self.assertFalse(record.exists())


class CompilerCacheTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'target/tmp').mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / 'target/tmp')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.binary = b'pinned executable'
        self.env = dict(JOPE_RUNNER_LANE='full', GITHUB_REF='refs/heads/main',
                        CARGO_INCREMENTAL='0', CARGO_TARGET_DIR=str(self.root),
                        RUNNER_TEMP=str(self.root), GITHUB_ENV=str(self.root / 'env'),
                        GITHUB_SHA='a' * 40)

    def archive(self):
        tools = self.root / 'compiler-cache-tools'
        tools.mkdir()
        path = tools / full.ARCHIVE
        with tarfile.open(path, 'w:gz') as contents:
            member = tarfile.TarInfo(full.ARCHIVE.removesuffix('.tar.gz') + '/sccache')
            member.size = len(self.binary)
            contents.addfile(member, io.BytesIO(self.binary))
        return path

    def test_pr_or_foreign_lane_refused_before_download_or_server(self):
        for key, value in [('JOPE_RUNNER_LANE', 'ci'), ('GITHUB_REF', 'refs/pull/1/merge'),
                           ('CARGO_INCREMENTAL', '1')]:
            with self.subTest(key=key), patch.dict(os.environ, dict(self.env, **{key: value})), \
                    patch.object(full, 'install_compiler_cache') as install:
                with self.assertRaises(ValueError):
                    full.setup_compiler_cache(self.root / 'evidence')
                install.assert_not_called()

    def test_tampered_archive_refused_before_executable_creation(self):
        self.archive()
        with self.assertRaisesRegex(ValueError, 'archive checksum'):
            full.install_compiler_cache(self.root)
        self.assertFalse((self.root / 'compiler-cache-tools' / f'v{full.VERSION}' / 'sccache').exists())

    def test_linked_tool_directory_refused_before_download(self):
        with patch.object(Path, 'is_symlink', return_value=True), \
                patch.object(full.urllib.request, 'urlopen') as download:
            with self.assertRaisesRegex(ValueError, 'unsafe compiler cache tool directory'):
                full.install_compiler_cache(self.root)
            download.assert_not_called()

    def test_verified_install_and_cached_executable_tampering(self):
        archive = self.archive()
        with patch.object(full, 'ARCHIVE_SHA256', full.cache_digest(archive)), \
                patch.object(full, 'BINARY_SHA256', hashlib.sha256(self.binary).hexdigest()), \
                patch.object(full.urllib.request, 'urlopen') as download:
            binary = full.install_compiler_cache(self.root)
            self.assertEqual('sccache', binary.name)
            self.assertEqual(self.binary, binary.read_bytes())
            self.assertEqual(binary, full.install_compiler_cache(self.root))
            download.assert_not_called()
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'executable checksum'):
                full.install_compiler_cache(self.root)

    def test_setup_uses_full_target_and_own_socket_without_compiler_flags(self):
        binary = self.root / 'sccache'
        with patch.dict(os.environ, self.env), patch.object(full, 'install_compiler_cache', return_value=binary), \
                patch.object(full.subprocess, 'run') as child:
            full.setup_compiler_cache(self.root / 'evidence')
        args, kwargs = child.call_args
        self.assertEqual([str(binary), '--start-server'], args[0])
        self.assertTrue(kwargs['check'])
        generated = dict(line.split('=', 1) for line in (self.root / 'env').read_text().splitlines())
        self.assertEqual(str(self.root / 'compiler-cache'), generated['SCCACHE_DIR'])
        self.assertEqual(str(self.root / 'full-sccache.sock'), generated['SCCACHE_SERVER_UDS'])
        self.assertEqual('0', generated['SCCACHE_IDLE_TIMEOUT'])
        self.assertNotIn('RUSTFLAGS', generated)
        self.assertNotIn('CARGO_TARGET_DIR', generated)
        self.assertNotIn('CARGO_BUILD_JOBS', generated)
        record = json.loads((self.root / 'evidence/setup.json').read_text())
        self.assertEqual(self.env['GITHUB_SHA'], record['source_sha'])
        self.assertEqual(generated, record['environment'])

    def test_failed_server_does_not_publish_wrapper(self):
        with patch.dict(os.environ, self.env), patch.object(full, 'install_compiler_cache', return_value=self.root / 'sccache'), \
                patch.object(full.subprocess, 'run', side_effect=OSError('server failed')):
            with self.assertRaises(OSError):
                full.setup_compiler_cache(self.root / 'evidence')
        self.assertFalse((self.root / 'env').exists())

    def test_stats_preserve_actual_hits_and_misses_and_reject_different_source_setup(self):
        conf = self.root / 'sccache.toml'
        conf.write_text('[cache.disk]\n')
        (self.root / 'setup.json').write_text(json.dumps(dict(source_sha=self.env['GITHUB_SHA'])))
        data = dict(stats=dict(cache_hits={'Rust': 2}, cache_misses={'Rust': 3}))
        env = dict(self.env, RUSTC_WRAPPER=str(self.root / 'sccache'), SCCACHE_CONF=str(conf))
        with patch.dict(os.environ, env), patch.object(full, 'cache_digest', return_value=full.BINARY_SHA256), \
                patch.object(full.subprocess, 'check_output', return_value=json.dumps(data)):
            full.compiler_cache_stats(self.root / 'published')
            self.assertEqual(data['stats'], json.loads((self.root / 'published/stats.json').read_text())['stats'])
            (self.root / 'setup.json').write_text(json.dumps(dict(source_sha='b' * 40)))
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                full.compiler_cache_stats(self.root / 'different-source')
            self.assertFalse((self.root / 'different-source').exists())


if __name__ == '__main__':
    unittest.main()
