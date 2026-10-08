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
        workflow = (ROOT/'.github/workflows/full-main.yml').read_text()
        job_minutes = int(re.search(r'^    timeout-minutes: (\d+)$', workflow, re.M)[1])
        self.assertLessEqual(total, (job_minutes - 5) * 60)

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
                    self.assertEqual({'full-neural-admission': {'max-threads': 4}}, actual.pop('test-groups'))
                    self.assertEqual([
                        dict(filter='package(jope-parallel) & test(=teaching_store::read_once_tests::a_search_receipt_is_validated_from_the_bytes_its_ledger_commits)',
                             priority=100, **{'test-group': 'full-neural-admission'}),
                        dict(filter='package(jope-parallel) & (test(teaching_store::read_once_tests::) | test(teaching_store::admission_tests::))',
                             priority=90, **{'test-group': 'full-neural-admission'})], override)
                else:
                    self.assertEqual([], override)
                self.assertEqual(expected, actual)


class InputBuildTests(unittest.TestCase):
    def test_joint_build_keeps_both_checks_and_fails_closed(self):
        scratch_root = ROOT / 'target' / 'tmp'
        scratch_root.mkdir(parents=True, exist_ok=True)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for failure in ('none', 'build', 'policy', 'elo'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
                root = Path(scratch)
                cargo = root / 'cargo'
                cargo.write_text('''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as out:
    out.write(json.dumps(args) + '\\n')
stage = 'build' if '--no-run' in args else 'policy' if '--lib' in args else 'elo' if args[0] == 'test' else 'fixture'
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
                self.assertEqual(['test', '--locked', '-p', 'jope-dime-regressor', '--lib', '--bin',
                                  'jope-dime-regressor', '--no-run'], executed[0])
                self.assertEqual(0 if failure == 'none' else 7, result.returncode, result.stdout+result.stderr)
                self.assertEqual({'build': 1, 'policy': 4, 'elo': 5, 'none': 5}[failure], len(executed))
                if failure != 'build':
                    self.assertEqual(['run', '--locked', '--bin', 'jope', '--', 'verify-inputs', '--offline', '--json',
                                      '--root', 'datasets/synthetic'], executed[1])
                    self.assertEqual(['run', '--locked', '--bin', 'jope', '--', 'generate-fixtures', '--check', '--json',
                                      '--output', 'datasets/synthetic'], executed[2])
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
        self.root = Path(self.directory.name)
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
