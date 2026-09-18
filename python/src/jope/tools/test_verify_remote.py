"""Mocked remote runner boundaries; no subprocess, GPU or numerical execution."""
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

RUNNER = Path(__file__).resolve().parents[1] / 'runs/final-run-recovery-20260911/verify_remote.py'
spec = importlib.util.spec_from_file_location('remote_engineering_runner', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def gpu_rows(count=8, model='NVIDIA A100-SXM4-40GB', duplicate=False):
    return '\n'.join(f'{model}, GPU-{0 if duplicate else i}' for i in range(count))


class RemoteRunnerTests(unittest.TestCase):
    def execute(self, plan=None, host='158-101-20-226', rows=None, optimize=0, run=None):
        plan = plan or {'stage': 'fixture', 'tasks': []}
        self.probe = Mock(return_value=subprocess.CompletedProcess([], 0, rows or gpu_rows(), ''))
        self.workload = Mock(side_effect=AssertionError('workload must not start'))
        with patch('socket.gethostname', return_value=host), patch('subprocess.run', self.probe), \
                patch('subprocess.Popen', self.workload):
            exec(compile(self.source(plan, run), '<remote-runner-fixture>', 'exec', optimize=optimize), {})

    @staticmethod
    def source(plan, run=None):
        source = runner.remote_source(plan)
        if run is not None:
            source = source.replace("pathlib.Path('/home/ubuntu/jope-dime/parallel/runs/final-run-recovery-20260911')",
                                    'pathlib.Path(' + repr(str(run)) + ')')
        return source

    def test_wrong_host_rejects_before_even_gpu_probe_when_python_optimized(self):
        with self.assertRaisesRegex(RuntimeError, 'workload host'):
            self.execute(host='xbabe2', optimize=2)
        self.probe.assert_not_called()
        self.workload.assert_not_called()

    def test_wrong_gpu_count_or_model_rejects_before_workload(self):
        for rows in [gpu_rows(count=7), gpu_rows(model='NVIDIA H100'),
                     gpu_rows(model='FAKE NVIDIA A100-SXM4-40GB EXTRA')]:
            with self.subTest(rows=rows), self.assertRaisesRegex(RuntimeError, 'eight-A100'):
                self.execute(rows=rows, optimize=2)
            self.workload.assert_not_called()

    def test_duplicate_or_malformed_gpu_identities_reject_before_workload(self):
        for rows in [gpu_rows(duplicate=True), gpu_rows().replace('GPU-', 'OTHER-')]:
            with self.subTest(rows=rows), self.assertRaisesRegex(RuntimeError, 'identities'):
                self.execute(rows=rows, optimize=2)
            self.workload.assert_not_called()

    def test_plan_cannot_override_cuda_or_thread_limits_before_workload(self):
        original_path = Path
        with tempfile.TemporaryDirectory() as directory:
            run = original_path(directory)
            stage = run / 'fixture'
            stage.mkdir()
            (stage / 'source-manifest.json').write_text(json.dumps({'closure': hashlib.sha256(b'{}').hexdigest(), 'files': {}}))
            for key in ['CUDA_VISIBLE_DEVICES', 'CARGO_BUILD_JOBS', 'RUST_TEST_THREADS', 'OMP_NUM_THREADS',
                        'RAYON_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
                plan = {'stage': 'fixture', 'env': {key: '8'}, 'tasks': []}
                with self.subTest(key=key), \
                        self.assertRaisesRegex(RuntimeError, 'locked CUDA/thread/build limits'):
                    self.execute(plan, optimize=2, run=run)
                self.workload.assert_not_called()

    def test_inherited_environment_is_locked_and_original_receipt_cannot_be_replaced(self):
        original_path = Path
        with tempfile.TemporaryDirectory() as directory:
            run = original_path(directory)
            stage = run / 'fixture'
            (stage / 'source/parallel').mkdir(parents=True)
            (stage / 'source-manifest.json').write_text(json.dumps({'closure': hashlib.sha256(b'{}').hexdigest(), 'files': {}}))
            plan = {'stage': 'fixture', 'tasks': [{'label': 'check', 'command': ['mock-cpu-check'],
                     'cwd': 'parallel', 'timeout': 1}]}
            child = Mock(pid=12345)
            child.wait.return_value = 0
            probe = subprocess.CompletedProcess([], 0, gpu_rows(), '')
            with patch('socket.gethostname', return_value='158-101-20-226'), \
                    patch('subprocess.run', return_value=probe), patch('subprocess.Popen', return_value=child) as launch, \
                    patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'CARGO_BUILD_JOBS': '128',
                        'PYTHONPATH':'/untrusted', 'LD_PRELOAD':'/untrusted/inject.so',
                        'RUSTFLAGS':'--cfg injected', 'CARGO_ENCODED_RUSTFLAGS':'injected'}):
                exec(compile(self.source(plan, run), '<remote-runner-fixture>', 'exec', optimize=2), {})
                self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(launch.call_args.kwargs['env']['CARGO_BUILD_JOBS'], '6')
                self.assertEqual(launch.call_args.kwargs['env']['RUST_TEST_THREADS'], '6')
                effective = launch.call_args.kwargs['env']
                for key in ('PYTHONPATH','LD_PRELOAD','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS'):
                    self.assertNotIn(key, effective)
                self.assertEqual(json.loads((stage / 'check.json').read_text())['environment'], effective)
                self.assertEqual(json.loads((stage / 'check.accepted.json').read_text())['environment'], effective)
                before = (stage / 'check.json').read_bytes()
                with self.assertRaisesRegex(RuntimeError, 'original attempt already exists'):
                    exec(compile(self.source(plan, run), '<remote-runner-fixture>', 'exec', optimize=2), {})
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(before, (stage / 'check.json').read_bytes())


    def test_changed_or_unmanifested_source_rejects_before_workload(self):
        for mutation in ('bytes', 'executable', 'symlink', 'extra', 'build', 'cargo-config', 'header', 'bytecode', 'closure'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                stage = run / 'fixture'
                source = stage / 'source/parallel/src'
                source.mkdir(parents=True)
                path = source / 'lib.rs'
                path.write_bytes(b'original')
                files = {'parallel/src/lib.rs': {'sha256': hashlib.sha256(b'original').hexdigest(),
                         'bytes': 8, 'executable': False}}
                closure = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                manifest = {'closure': closure, 'files': files}
                if mutation == 'bytes': path.write_bytes(b'modified')
                elif mutation == 'executable': path.chmod(0o755)
                elif mutation == 'symlink':
                    target = source / 'target'
                    path.rename(target)
                    path.symlink_to(target)
                elif mutation == 'bytecode': (source / 'injected.pyc').write_bytes(b'cached')
                elif mutation == 'extra': (source / 'extra.rs').write_text('extra')
                elif mutation in ('build', 'cargo-config', 'header'):
                    extra = stage / 'source' / {'build': 'parallel/build.rs',
                            'cargo-config': 'parallel/.cargo/config.toml', 'header': 'vendor/runtime.h'}[mutation]
                    extra.parent.mkdir(parents=True, exist_ok=True)
                    extra.write_text('injected')
                else: manifest['closure'] = 'wrong'
                (stage / 'source-manifest.json').write_text(json.dumps(manifest))
                plan = {'stage': 'fixture', 'tasks': [{'label': 'check', 'command': ['mock-cpu-check'],
                        'cwd': 'parallel', 'timeout': 1}]}
                with self.assertRaisesRegex(RuntimeError, 'source|closure'):
                    self.execute(plan, run=run, optimize=2)
                self.workload.assert_not_called()

    def test_source_mutation_during_check_retains_failed_binding_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            stage = run / 'fixture'
            source = stage / 'source/parallel/src'
            source.mkdir(parents=True)
            (stage / 'source-manifest.json').write_text(json.dumps({
                'closure': hashlib.sha256(b'{}').hexdigest(), 'files': {}}))
            plan = {'stage': 'fixture', 'tasks': [{'label': 'check', 'command': ['mock-cpu-check'],
                    'cwd': 'parallel', 'timeout': 1}]}
            child = Mock(pid=12345)
            def mutate(**kwargs):
                (source / 'injected.rs').write_text('injected')
                return 0
            child.wait.side_effect = mutate
            with patch('socket.gethostname', return_value='158-101-20-226'), \
                    patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, gpu_rows(), '')), \
                    patch('subprocess.Popen', return_value=child):
                with self.assertRaisesRegex(RuntimeError, 'unmanifested source'):
                    exec(compile(self.source(plan, run), '<remote-runner-fixture>', 'exec', optimize=2), {})
            receipt = json.loads((stage / 'check.json').read_text())
            self.assertEqual(receipt['exit_code'], 0)
            self.assertTrue(receipt['source_verified_before'])
            self.assertFalse(receipt['source_verified_after'])
            self.assertIn('unmanifested source', receipt['source_verification_error'])

    def fixture(self, directory):
        run = Path(directory)
        stage = run / 'fixture'
        (stage / 'source/parallel').mkdir(parents=True)
        (stage / 'source-manifest.json').write_text(json.dumps({
            'closure': hashlib.sha256(b'{}').hexdigest(), 'files': {}}))
        plan = {'stage': 'fixture', 'tasks': [{'label': 'check', 'command': ['mock-cpu-check'],
                'cwd': 'parallel', 'timeout': 1}]}
        return run, stage, plan

    def launch_fixture(self, plan, run, launch):
        with patch('socket.gethostname', return_value='158-101-20-226'), \
                patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, gpu_rows(), '')), \
                patch('subprocess.Popen', launch):
            exec(compile(self.source(plan, run), '<remote-runner-fixture>', 'exec', optimize=2), {})

    def test_spawn_failure_has_accepted_and_terminal_receipts_and_no_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            run, stage, plan = self.fixture(directory)
            def fail(*args, **kwargs):
                accepted = json.loads((stage / 'check.accepted.json').read_text())
                self.assertEqual(accepted['outcome'], 'unknown')
                raise FileNotFoundError('fixture missing executable')
            launch = Mock(side_effect=fail)
            with self.assertRaisesRegex(RuntimeError, 'engineering checks failed'):
                self.launch_fixture(plan, run, launch)
            receipt = json.loads((stage / 'check.json').read_text())
            self.assertEqual(receipt['outcome'], 'spawn_failure')
            self.assertIsNone(receipt['exit_code'])
            self.assertIn('FileNotFoundError', receipt['execution_error'])
            self.assertEqual(receipt['accepted_attempt_sha256'],
                hashlib.sha256((stage / 'check.accepted.json').read_bytes()).hexdigest())
            with self.assertRaisesRegex(RuntimeError, 'original attempt already exists'):
                self.launch_fixture(plan, run, launch)
            self.assertEqual(launch.call_count, 1)

    def test_crash_preserved_acceptance_without_terminal_blocks_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            run, stage, plan = self.fixture(directory)
            (stage / 'check.accepted.json').write_text('{"outcome":"unknown"}')
            launch = Mock(side_effect=AssertionError('replay'))
            with self.assertRaisesRegex(RuntimeError, 'original attempt already exists'):
                self.launch_fixture(plan, run, launch)
            launch.assert_not_called()
            plan['tasks'][0]['label'] = 'different'
            with self.assertRaisesRegex(RuntimeError, 'unreconciled prior attempt'):
                self.launch_fixture(plan, run, launch)
            launch.assert_not_called()

    def test_timeout_exit_signal_race_keeps_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            run, stage, plan = self.fixture(directory)
            child = Mock(pid=12345)
            child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 1), 0]
            with patch('os.killpg', side_effect=ProcessLookupError), \
                    self.assertRaisesRegex(RuntimeError, 'engineering checks failed'):
                self.launch_fixture(plan, run, Mock(return_value=child))
            receipt = json.loads((stage / 'check.json').read_text())
            self.assertEqual(receipt['outcome'], 'timeout')
            self.assertTrue(receipt['deadline_expired'])
            self.assertEqual(receipt['exit_code'], 0)

    def test_unreaped_child_remains_unknown_and_prevents_next_task(self):
        with tempfile.TemporaryDirectory() as directory:
            run, stage, plan = self.fixture(directory)
            plan['tasks'].append(dict(plan['tasks'][0], label='next'))
            child = Mock(pid=12345)
            child.wait.side_effect = subprocess.TimeoutExpired('fixture', 1)
            launch = Mock(return_value=child)
            with patch('os.killpg'), self.assertRaisesRegex(RuntimeError, 'unreconciled child'):
                self.launch_fixture(plan, run, launch)
            self.assertEqual(json.loads((stage / 'check.json').read_text())['outcome'], 'unknown')
            self.assertEqual(launch.call_count, 1)
            self.assertFalse((stage / 'next.accepted.json').exists())

    def test_invalid_build_targets_and_unlisted_environment_reject_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            run, stage, plan = self.fixture(directory)
            for target in ('../cargo-target-escape', '/tmp/cargo-target-escape',
                           str(stage / 'source/cargo-target-in-source'), 'source'):
                with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'build target'):
                    self.execute(dict(plan, tasks=[dict(plan['tasks'][0], target=target)]), run=run)
                self.workload.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'explicit allowlist'):
                self.execute(dict(plan, env={'PYTHONPATH':'/injected'}), run=run)
            self.workload.assert_not_called()

    def test_external_target_is_recorded_and_two_distinct_checks_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            run, stage, plan = self.fixture(directory)
            plan['tasks'][0]['target'] = 'cargo-target-fixture'
            plan['tasks'].append(dict(plan['tasks'][0], label='next'))
            child = Mock(pid=12345)
            child.wait.return_value = 0
            launch = Mock(return_value=child)
            self.launch_fixture(plan, run, launch)
            self.assertEqual(launch.call_count, 2)
            for label in ('check','next'):
                receipt = json.loads((stage / (label+'.json')).read_text())
                self.assertEqual(receipt['outcome'], 'success')
                self.assertEqual(receipt['environment']['CARGO_TARGET_DIR'], str(run/'cargo-target-fixture'))


if __name__ == '__main__':
    unittest.main()
