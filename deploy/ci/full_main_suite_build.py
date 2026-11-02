from full_main_test_support import *  # noqa: F403

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
