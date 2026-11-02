#!/usr/bin/env python3
import fnmatch
import json
import re
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'deploy/runner' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load('job-policy')
container = load('verify-container')


from test_runner_cases import RunnerTests, MacControllerTests, TrimTests


class SecretScanTests(unittest.TestCase):
    def test_full_head_history_is_independent_of_custody_refs(self):
        scratch = ROOT / 'target' / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='secret-scan-fixture-', dir=scratch) as name:
            repo = Path(name)
            def git(*args):
                return subprocess.run(['git', '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', *args], cwd=repo,
                    capture_output=True, text=True, check=True)
            def scan():
                return subprocess.run(['bash', str(ROOT/'deploy/ci/secret-scan.sh'), str(repo)],
                    capture_output=True, text=True)
            git('init', '-q', '-b', 'main')
            (repo/'README').write_text('synthetic scan fixture\n')
            git('add', 'README'); git('commit', '-qm', 'clean source')
            git('switch', '--orphan', 'custody')
            (repo/'historical.env').write_text('SYNTHETIC_FIXTURE=not-a-credential\n')
            git('add', 'historical.env'); git('commit', '-qm', 'unrelated custody')
            git('switch', 'main')
            # Reproduce the old gate: an unrelated ref changes its result.
            old = subprocess.run(['gitleaks', 'detect', '--redact', '--no-banner',
                '--config', str(ROOT/'.gitleaks.toml'), '--source', str(repo)],
                capture_output=True, text=True)
            self.assertEqual(1, old.returncode, old.stderr)
            self.assertEqual(0, scan().returncode)
            # A removed file in HEAD ancestry must still fail: this is a history scan.
            (repo/'service.env').write_text('SYNTHETIC_FIXTURE=not-a-credential\n')
            git('add', 'service.env'); git('commit', '-qm', 'historical source violation')
            git('rm', 'service.env'); git('commit', '-qm', 'remove current file')
            self.assertEqual(1, scan().returncode)
            # A shallow marker must fail before the scanner can claim complete history.
            (repo/'.git'/'shallow').write_text(git('rev-parse', 'HEAD').stdout)
            refused = scan()
            self.assertEqual(2, refused.returncode)
            self.assertIn('requires full HEAD history', refused.stderr)

    def test_hygiene_requests_complete_history(self):
        workflow = (ROOT/'.github/workflows/ci.yml').read_text()
        hygiene = workflow.split('\n  hygiene:', 1)[1].split('\n  rust-check:', 1)[0]
        self.assertIn('fetch-depth: 0', hygiene)
        # Hygiene delegates to the fail-closed shared security entry point.
        fast_lane = (ROOT/'deploy/ci/fast-lane.sh').read_text()
        check = (ROOT/'deploy/ci/check.sh').read_text()
        self.assertIn('bash deploy/ci/check.sh', fast_lane)
        self.assertNotIn('JOPE_SECURITY_LANE_PRECHECKED', fast_lane)
        self.assertNotIn('JOPE_SECURITY_LANE_PRECHECKED', check)
        self.assertIn('bash tools/security-lane.sh', check)
        self.assertIn('bash deploy/ci/secret-scan.sh', (ROOT/'tools/security-lane.sh').read_text())


class DropInAdmissionTests(unittest.TestCase):
    def test_actual_launcher_refuses_unrecorded_systemd_configuration(self):
        scratch = ROOT / 'target' / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        approved = '/etc/systemd/system/jope-runner@.service.d/runtime-max.conf'
        fragment = '/etc/systemd/system/jope-runner@.service'
        script = r'''hostname() { echo xbabe1; }
systemctl() { case "$*" in *DropInPaths*) printf '%s\n' "$fixture_dropins";; *FragmentPath*) printf '%s\n' "$fixture_fragment";; *) return 91;; esac; }
flock() { exit 77; } # valid admission stops here, before any Docker or credential access
source deploy/runner/run.sh ci
'''
        import os
        with tempfile.TemporaryDirectory(prefix='dropin-fixture-', dir=scratch) as name:
            for dropins, unit, expected in [(approved, fragment, 77),
                    ('', fragment, 2),
                    (approved+' /etc/systemd/system/jope-runner@ci.service.d/resource-budget.conf', fragment, 2),
                    (approved, '/foreign.service', 2)]:
                with self.subTest(dropins=dropins, unit=unit):
                    env = dict(os.environ, JOPE_RUNNER_STATE=name,
                               fixture_dropins=dropins, fixture_fragment=unit)
                    result = subprocess.run(['bash', '-ec', script], cwd=ROOT, env=env,
                                            capture_output=True, text=True)
                    self.assertEqual(expected, result.returncode, result.stderr)


class SwapPinTests(unittest.TestCase):
    """runc leaves docker-<id>.scope at MemorySwapMax=infinity; a daemon-reload then re-applies
    swap=max to live listeners. The launcher must record MemorySwapMax=0 in systemd, or kill."""
    script = r'''docker() {
  echo "docker $*" >> "$fixture_log"
  case $1 in inspect) printf '%s\n' "$fixture_id";; kill) return 0;; *) return 91;; esac
}
systemctl() {
  echo "systemctl $*" >> "$fixture_log"
  case $1 in set-property) return "$fixture_set_rc";; show) printf '%s\n' "$fixture_show";; *) return 91;; esac
}
sleep() { :; }
source deploy/runner/common.sh
jope_runner_pin_swap xbabe2-full-fixture
'''

    def run_pin(self, cid, set_rc, show):
        import os
        scratch = ROOT / 'target' / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='swap-pin-', dir=scratch) as name:
            log = Path(name) / 'calls'
            log.touch()
            env = dict(os.environ, fixture_log=str(log), fixture_id=cid,
                       fixture_set_rc=str(set_rc), fixture_show=show)
            result = subprocess.run(['bash', '-ec', self.script], cwd=ROOT, env=env,
                                    capture_output=True, text=True)
            return result, log.read_text().splitlines()

    def test_records_zero_swap_in_systemd(self):
        result, calls = self.run_pin('c0ffee', 0, '0')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('systemctl set-property --runtime docker-c0ffee.scope MemorySwapMax=0', calls)
        self.assertFalse([c for c in calls if c.startswith('docker kill')])

    def test_fails_closed_when_systemd_does_not_hold_zero(self):
        for set_rc, show in [(1, '0'), (0, 'infinity'), (0, '')]:
            with self.subTest(set_rc=set_rc, show=show):
                result, calls = self.run_pin('c0ffee', set_rc, show)
                self.assertEqual(2, result.returncode)
                self.assertIn('docker kill xbabe2-full-fixture', calls)

    def test_absent_container_never_touches_systemd(self):
        result, calls = self.run_pin('', 0, '0')
        self.assertEqual(2, result.returncode)
        self.assertFalse([c for c in calls if c.startswith('systemctl')])

    def test_launcher_pins_every_listener_without_holding_locks(self):
        run = (ROOT / 'deploy/runner/run.sh').read_text()
        pin = run.index('jope_runner_pin_swap "$name" 8<&- 9>&- &\n')
        self.assertLess(pin, run.index('| exec docker "${docker_args[@]}"'))
        self.assertIn('--memory-swap "$JOPE_RUNNER_MEMORY"', run)

class EmitterEvidenceTests(unittest.TestCase):
    def test_actual_lane_routes_emitter_output_to_run_evidence(self):
        import os
        scratch = ROOT / 'target' / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        script = r'''rustup() { :; }
gcc() { echo fixture; }
clang() { echo fixture; }
g++() { echo fixture; }
clang++() { echo fixture; }
node() { echo fixture; }
python3() { echo fixture; }
rustc() { echo fixture; }
bash() {
  [[ "$1" == deploy/ci/emitters.sh ]] || exit 91
  printf '%s\n' "$EMITTERS_OUT" > "$fixture_capture"
  return 77
}
source deploy/ci/fast-lane.sh emitters
'''
        with tempfile.TemporaryDirectory(prefix='emitter-route-', dir=scratch) as name:
            root = Path(name)
            evidence = root/'workspace-evidence'/'run'/'attempt'/'emitters'
            cache = root/'persistent-cargo'
            capture = root/'captured-path'
            env = dict(os.environ, JOPE_LANE_EVIDENCE=str(evidence),
                       CARGO_TARGET_DIR=str(cache), fixture_capture=str(capture))
            result = subprocess.run(['bash', '-ec', script], cwd=ROOT, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(77, result.returncode, result.stderr)
            self.assertEqual(str(evidence/'emitters'), capture.read_text().strip())
            self.assertFalse(cache.exists())
        workflow = (ROOT/'.github/workflows/ci.yml').read_text()
        lane = workflow.split('\n  emitters:', 1)[1].split('\n  inputs:', 1)[0]
        self.assertIn('$GITHUB_WORKSPACE/target/ci-evidence/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT/$GITHUB_JOB', lane)


class PublicationStatusTests(unittest.TestCase):
    def test_publication_gate_requires_one_bounded_successful_sequence(self):
        import itertools
        admitted = {('success', 'skipped', 'skipped'), ('failure', 'success', 'skipped'),
                    ('failure', 'failure', 'success')}
        for outcomes in itertools.product(['success', 'failure', 'cancelled', 'skipped', ''], repeat=3):
            result = subprocess.run(['bash', str(ROOT/'deploy/ci/publication-status.sh'), *outcomes],
                                    capture_output=True, text=True)
            self.assertEqual(outcomes in admitted, result.returncode == 0, outcomes)

    def test_upload_retries_keep_distinct_artifacts_and_existing_caps(self):
        action = (ROOT/'.github/actions/upload-ci-evidence/action.yml').read_text()
        self.assertNotIn('overwrite:', action)
        self.assertNotIn('timeout-minutes:', action)
        for number in range(1, 4):
            self.assertIn('${{ inputs.name }}-pub-'+str(number), action)
        self.assertNotIn('upload-artifact', action)
        self.assertEqual(3, action.count('actions/cache/save@'))
        self.assertIn('deploy/ci/retain-local-evidence.sh', action)
        self.assertIn('bash deploy/ci/publication-status.sh "$FIRST" "$SECOND" "$THIRD"', action)
        go_workflow = (ROOT/'.github/workflows/go.yml').read_text()
        self.assertEqual(3, go_workflow.count('actions/cache/save@'))
        self.assertEqual(3, go_workflow.count('actions/cache/restore@'))
        self.assertIn('bash deploy/ci/publication-status.sh "$FIRST" "$SECOND" "$THIRD"', go_workflow)
        download_action = (ROOT/'.github/actions/download-ci-evidence/action.yml').read_text()
        self.assertIn('Resolve prior attempts of this run', download_action)

    def test_go_cache_publication_retries_and_download_prior_attempt_fallback(self):
        go_workflow = (ROOT/'.github/workflows/go.yml').read_text()
        base_key = 'go-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.go }}'
        for step_id in ('save-1', 'first', 'save-2', 'second', 'save-3', 'third'):
            self.assertIn(f'id: {step_id}\n', go_workflow)
        self.assertEqual(2, go_workflow.count(f'key: {base_key}\n'))
        self.assertEqual(2, go_workflow.count(f'key: {base_key}-pub-2\n'))
        self.assertEqual(2, go_workflow.count(f'key: {base_key}-pub-3\n'))
        self.assertEqual(2, go_workflow.count("if: ${{ !cancelled() && steps.first.outcome == 'failure' }}\n"))
        self.assertEqual(2, go_workflow.count("if: ${{ !cancelled() && steps.first.outcome == 'failure' && steps.second.outcome == 'failure' }}\n"))
        self.assertEqual(2, go_workflow.count(f'restore-keys: |\n            {base_key}\n'))
        self.assertEqual(3, go_workflow.count('lookup-only: true\n          fail-on-cache-miss: true\n'))
        self.assertIn('FIRST: ${{ steps.first.outcome }}\n          SECOND: ${{ steps.second.outcome }}\n          THIRD: ${{ steps.third.outcome }}\n        run: bash deploy/ci/publication-status.sh "$FIRST" "$SECOND" "$THIRD"', go_workflow)
        download_action = (ROOT/'.github/actions/download-ci-evidence/action.yml').read_text()
        for attempt_idx in (1, 2, 3):
            for pub_idx, pub_name in [(1, 'first'), (2, 'second'), (3, 'third')]:
                self.assertIn(f'id: prior-{attempt_idx}-{pub_name}\n', download_action)
                self.assertIn(f'key: ${{{{ steps.prior.outputs.name_{attempt_idx} }}}}-pub-{pub_idx}\n', download_action)
        resolve_script = textwrap.dedent(
            download_action.split('id: prior\n', 1)[1].split('      run: |\n', 1)[1].split('    - name: Restore prior attempt 1 publication 1\n', 1)[0]
        )
        place_script = textwrap.dedent(
            download_action.split('    - name: Place restored evidence\n', 1)[1].split('      run: |\n', 1)[1]
        )
        scratch = ROOT / 'target' / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='download-fallback-', dir=scratch) as name:
            root = Path(name)
            gh_out = root / 'github-output'
            gh_out.write_text('')
            env = dict(
                os.environ,
                GITHUB_RUN_ID='42',
                GITHUB_RUN_ATTEMPT='2',
                NAME='ci-42-2-python',
                SRC=str(root / 'target/ci-evidence/42/2/python/'),
                GITHUB_OUTPUT=str(gh_out),
            )
            res = subprocess.run(['bash', '-ec', resolve_script], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(0, res.returncode, res.stderr)
            outputs = dict(line.split('=', 1) for line in gh_out.read_text().splitlines())
            self.assertEqual('ci-42-1-python', outputs['name_1'])
            self.assertEqual(str(root / 'target/ci-evidence/42/1/python/'), outputs['path_1'])
            self.assertNotIn('name_2', outputs)
            p1_src = Path(outputs['path_1'])
            p1_src.mkdir(parents=True)
            (p1_src / 'receipt.json').write_text('{"attempt":1}\n')
            into = root / 'downloaded'
            place_env = dict(
                os.environ,
                NAME='ci-42-2-python',
                SRC=str(root / 'target/ci-evidence/42/2/python/'),
                INTO=str(into),
                LAYOUT='key',
                CUR_KEY='',
                P1_KEY='ci-42-1-python-pub-1',
                P2_KEY='',
                P3_KEY='',
                P1_SRC=str(p1_src),
                P2_SRC='',
                P3_SRC='',
            )
            placed = subprocess.run(['bash', '-ec', place_script], cwd=ROOT, env=place_env, capture_output=True, text=True)
            self.assertEqual(0, placed.returncode, placed.stderr)
            self.assertEqual('{"attempt":1}\n', (into / 'ci-42-1-python-pub-1' / 'receipt.json').read_text())

    def test_no_workflow_depends_on_github_artifact_storage(self):
        paths = sorted(p for p in (ROOT/'.github').rglob('*') if p.suffix in ('.yml', '.yaml'))
        self.assertTrue(paths)
        for path in paths:
            text = path.read_text()
            self.assertIsNone(re.search(r'actions/[\w-]*artifact[\w/-]*@', text), path)
            self.assertNotIn('gh run download', text, path)
            for line in text.splitlines():
                if re.match(r'\s*upload-artifact:', line):
                    self.assertEqual('upload-artifact: false', line.strip(), path)

    def test_ci_diff_classifier_is_hosted_for_pr_and_ui_evidence_isolated(self):
        workflow = (ROOT/'.github/workflows/ci.yml').read_text()
        pr_classifier = workflow.split('\n  changes-pr:', 1)[1].split('\n  changes-dispatch:', 1)[0]
        dispatch_classifier = workflow.split('\n  changes-dispatch:', 1)[1].split('\n  changes:', 1)[0]
        self.assertIn('runs-on: ubuntu-24.04', pr_classifier)
        self.assertFalse(any('runs-on:' in line and 'self-hosted' in line for line in pr_classifier.splitlines()))
        self.assertIn("if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'", dispatch_classifier)
        self.assertIn('runs-on: ubuntu-24.04', dispatch_classifier)
        self.assertNotIn('./.github/actions/runner-provenance', dispatch_classifier)
        self.assertIn('runner_pool: jope-ci', dispatch_classifier)
        ui = workflow.split('\n  observatory-ui:', 1)[1].split('\n  typescript:', 1)[0]
        self.assertIn('raw_results="$RUNNER_TEMP/jope-observatory-ui/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT/raw"', ui)
        self.assertIn('node --input-type=module "$raw_results/tests.json"', ui)
        self.assertNotIn('JOPE_UI_EVIDENCE', ui)
        self.assertNotIn('process.env', ui)
        self.assertIn('$GITHUB_WORKSPACE/target/observatory-ui/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT', ui)
        self.assertIn('npm test -- --reporter=json --outputFile="$raw_results/tests.json"', ui)

    def test_cache_evidence_paths_are_workspace_relative_and_match(self):
        for name in ('ci.yml', 'full-main.yml'):
            text = (ROOT/'.github/workflows'/name).read_text()
            saved = re.findall(r'uses: \./\.github/actions/upload-ci-evidence\n\s+with:\n\s+name: (.+)\n\s+path: (.+)', text)
            restored = re.findall(r'uses: \./\.github/actions/download-ci-evidence\n\s+with:\n\s+name: (.+)\n\s+path: (.+)', text)
            self.assertTrue(saved and restored, name)
            for _, path in saved + restored:
                self.assertTrue(path.startswith('target/'), (name, path))
            lanes = {n: p for n, p in saved}
            # Evidence saved by reusable workflows this file calls (e.g. export-parity.yml).
            for called in re.findall(r'uses: \./\.github/workflows/([\w.-]+\.ya?ml)', text):
                called_text = (ROOT/'.github/workflows'/called).read_text()
                lanes.update(re.findall(r'uses: \./\.github/actions/upload-ci-evidence\n\s+with:\n\s+name: (.+)\n\s+path: (.+)', called_text))
            for n, p in restored:
                generic = re.sub(r'(cpu-\d|rust-test-\d|[a-z0-9-]+)/$', '', p)
                self.assertTrue(any(sp.startswith(generic) for sp in lanes.values()), (name, p))


class LocalRetentionTests(unittest.TestCase):
    def retain(self, base, blocked=False):
        import os
        source = base/'source'
        source.mkdir()
        (source/'receipt.json').write_text('{"probe":"required-custody"}\n')
        root = base/'retained'
        if blocked:
            root.write_text('not a directory\n')
        env = dict(os.environ, JOPE_LOCAL_EVIDENCE_ROOT=str(root),
                   GITHUB_REPOSITORY='neverhuman/JopeDime',
                   JOPE_EVIDENCE_S3_URI='', GITHUB_STEP_SUMMARY=str(base/'summary.md'),
                   JOPE_LOCAL_EVIDENCE_DAYS='14', TMPDIR=str(base))
        result = subprocess.run(
            ['bash', str(ROOT/'deploy/ci/retain-local-evidence.sh'), 'fixture', str(source)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        return result, source, root/'neverhuman_JopeDime'/'fixture'

    def test_local_retention_failure_is_fatal(self):
        scratch = ROOT/'target'/'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='retention-blocked-', dir=scratch) as name:
            result, _, destination = self.retain(Path(name), blocked=True)
            self.assertNotEqual(0, result.returncode, result.stdout)
            self.assertNotIn('retained at', result.stdout)
            self.assertFalse(destination.exists())

    def test_default_root_falls_back_when_shared_volume_is_unwritable(self):
        import os
        scratch = ROOT/'target'/'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='retention-fallback-', dir=scratch) as name:
            base = Path(name)
            source = base/'source'
            source.mkdir()
            (source/'receipt.json').write_text('{"probe":"fallback"}\n')
            env = {k: v for k, v in os.environ.items() if k != 'JOPE_LOCAL_EVIDENCE_ROOT'}
            env.update(CARGO_TARGET_DIR=str(base/'lane-target'), HOME=str(base/'home'),
                       GITHUB_REPOSITORY='neverhuman/JopeDime', JOPE_EVIDENCE_S3_URI='',
                       GITHUB_STEP_SUMMARY=str(base/'summary.md'), TMPDIR=str(base))
            result = subprocess.run(
                ['bash', str(ROOT/'deploy/ci/retain-local-evidence.sh'), 'fixture', str(source)],
                cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn('retained at', result.stdout)
            self.assertIn('receipt.json: OK', result.stdout)


    def test_local_retention_success_has_verified_recoverable_bytes(self):
        scratch = ROOT/'target'/'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='retention-ok-', dir=scratch) as name:
            import hashlib
            import os
            import time

            base = Path(name)
            previous = base/'retained'/'neverhuman_JopeDime'/'previous-qualified-run'
            previous.mkdir(parents=True)
            data = b'prior qualified evidence\n'
            manifest = (hashlib.sha256(data).hexdigest() + '  receipt.json\n').encode()
            (previous/'receipt.json').write_bytes(data)
            (previous/'MANIFEST.sha256').write_bytes(manifest)
            old_time = time.time() - 40 * 86400
            os.utime(previous, (old_time, old_time))

            result, source, destination = self.retain(base)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(data, (previous/'receipt.json').read_bytes())
            self.assertEqual(manifest, (previous/'MANIFEST.sha256').read_bytes())
            self.assertIn('retained evidence candidate older than 14d:', result.stdout)
            self.assertIn(str(previous), (base/'summary.md').read_text())
            self.assertEqual((source/'receipt.json').read_bytes(),
                             (destination/'receipt.json').read_bytes())
            verified = subprocess.run(['sha256sum', '--check', 'MANIFEST.sha256'],
                                      cwd=destination, capture_output=True, text=True)
            self.assertEqual(0, verified.returncode, verified.stderr)
            self.assertIn('receipt.json: OK', result.stdout)

class WorkflowAdmissionTests(unittest.TestCase):
    def job(self, text, name):
        match = re.search(rf'^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:|\Z)', text, re.M | re.S)
        self.assertIsNotNone(match, f'missing workflow job {name}')
        return match.group(1)

    def test_typescript_pr_parity_uses_disposable_runner(self):
        workflow = (ROOT/'.github/workflows/typescript.yml').read_text()
        admission = self.job(workflow, 'trust-admission')
        parity = self.job(workflow, 'typescript-parity')
        self.assertIn('PR_BASE_REPOSITORY_ID', admission)
        self.assertIn('PR_HEAD_REPOSITORY_ID', admission)
        self.assertIn("github.workflow_ref", admission)
        self.assertIn('runs-on: ubuntu-24.04', parity)
        self.assertNotIn('self-hosted', parity)
        self.assertNotIn('runner-provenance', parity)
        self.assertNotIn('RUNNER_ENVIRONMENT:', parity)
        self.assertNotIn('runner.environment', parity)
        self.assertIn('CARGO_TARGET_DIR: ${{ github.workspace }}/target/typescript-cargo', parity)
        self.assertIn('repository: ${{ github.event.pull_request.head.repo.full_name || github.repository }}', parity)

    def test_fork_pull_request_hygiene_skips_privileged_runner(self):
        workflow = (ROOT/'.github/workflows/ci.yml').read_text()
        hygiene = self.job(workflow, 'hygiene')
        self.assertIn("github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository", hygiene)
        self.assertNotIn('runs-on: [self-hosted', hygiene)
        self.assertIn("&& 'ubuntu-24.04' || 'xbabe1-ci'", hygiene)
        aggregate = self.job(workflow, 'aggregate')
        collect = aggregate.split('      - name: Collect hygiene evidence', 1)[1].split('      - name:', 1)[0]
        self.assertIn("github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository", collect)
        self.assertIn('HYGIENE_REQUIRED:', aggregate)
        self.assertIn('test "$HYGIENE_RESULT" = "$expected"', aggregate)

    def test_nonmain_manual_source_cannot_reach_privileged_ci_jobs(self):
        workflow = (ROOT/'.github/workflows/ci.yml').read_text()
        dispatch = self.job(workflow, 'changes-dispatch')
        self.assertIn("if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'", dispatch)
        self.assertIn('runs-on: ubuntu-24.04', dispatch)
        self.assertNotIn('uses: ./.github/actions/runner-provenance', dispatch)

        classifier = self.job(workflow, 'changes')
        self.assertIn('needs: [changes-pr, changes-dispatch]', classifier)
        self.assertIn('runs-on: ubuntu-24.04', classifier)
        self.assertIn('non-main manual dispatch was refused before privileged runner admission', classifier)
        self.assertIn('exit 1', classifier)

        for name in ('hygiene', 'jankurai-audit', 'paper', 'aggregate'):
            block = self.job(workflow, name)
            self.assertIn("github.event_name != 'workflow_dispatch' || github.ref == 'refs/heads/main'", block,
                          f'{name} must be skipped on non-main dispatch')
        for name in ('hygiene', 'jankurai-audit', 'rust-check', 'rust-test', 'emitters',
                     'inputs', 'python', 'e2e-smoke', 'observatory-ui', 'aggregate'):
            if name == 'paper':
                continue
            self.assertIn('uses: ./.github/actions/runner-provenance', self.job(workflow, name),
                          f'{name} must retain per-job runner admission')

        # Other self-hosted test jobs require the classifier's successful outputs, so they
        # inherit its fail-closed refusal through the default needs-success condition.
        observatory = self.job(workflow, 'observatory-ui')
        self.assertIn('needs: changes', observatory)
        # The only job-level guard allowed keeps the fail-closed classifier refusal: it runs
        # only when the classifier succeeded (needed so a skipped optional ancestor does not
        # starve it, see test_selected_jobs_survive_the_optional_classifier_ancestor).
        guards = re.findall(r'^    if:(.*)$', observatory, re.M)
        self.assertTrue(all("needs.changes.result == 'success'" in guard for guard in guards), guards)

    def test_workflow_delegates_full_evidence_checks_without_dropping_them(self):
        ci = (ROOT/'.github/workflows/ci.yml').read_text()
        full = (ROOT/'.github/workflows/full-main.yml').read_text()
        hygiene = self.job(ci, 'hygiene')
        self.assertIn('run: bash tools/security-lane.sh', hygiene)
        self.assertIn('JOPE_LANE_EVIDENCE:', hygiene)
        for command in ('publication', 'frontend', 'rendered-ux', 'export-parity', 'filter-needs'):
            self.assertIn(f'aggregate-evidence.py {command}', ci)
        self.assertIn('aggregate-evidence.py filter-needs', full)
        self.assertNotIn('JOPE_UI_EVIDENCE', ci)
        self.assertNotIn('NEEDS_JSON"]); n.pop(', ci + full)


class AgentMapTests(unittest.TestCase):
    # This report is emitted by the full-main Jankurai publisher and is excluded from
    # git status as generated evidence, so it has no tracked file to match in the maps.
    GENERATED_OUTPUTS = {'.jankurai/repo-score.md'}
    ALIASES = {
        'apps/web/': 'web/training-dashboard/',
        'apps/web/**': 'web/training-dashboard/**',
        'db/': 'crates/jope-datasets/',
        'db/**': 'crates/jope-datasets/**',
    }
    TEST_COMMAND_MARKERS = (
        'test-runner.py',
        'test-gitleaks-config.py',
        'test-security-lane.py',
        'python3 -m unittest',
        'cargo test ',
        'just test-pr',
        'just required',
        'just ci-pr',
        'just python-check',
        'pytest',
        'npm ci ',
        'npm test',
        'npm run test:rendered',
    )

    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True
        )
        cls.tracked = set(result.stdout.splitlines())

    def key_matches_tree(self, key):
        if key in self.GENERATED_OUTPUTS:
            return True
        pattern = self.ALIASES.get(key, key)
        if any(char in pattern for char in '*?['):
            return any(fnmatch.fnmatchcase(path, pattern) for path in self.tracked)
        prefix = pattern.rstrip('/')
        return prefix in self.tracked or any(
            path.startswith(prefix + '/') for path in self.tracked
        )

    def test_owner_map_keys_match_tracked_paths_or_explicit_aliases(self):
        owner_map = json.loads((ROOT / 'agent/owner-map.json').read_text())['owners']
        missing = [key for key in owner_map if not self.key_matches_tree(key)]
        self.assertEqual([], missing, f'untracked owner-map keys: {missing}')

    def test_test_map_keys_match_tracked_paths_or_explicit_aliases(self):
        test_map = json.loads((ROOT / 'agent/test-map.json').read_text())['tests']
        missing = [key for key in test_map if not self.key_matches_tree(key)]
        self.assertEqual([], missing, f'untracked test-map keys: {missing}')


    def test_policy_and_generated_zone_routes_are_valid(self):
        import tomllib

        with (ROOT / 'agent/audit-policy.toml').open('rb') as source:
            policy = tomllib.load(source)
        with (ROOT / 'agent/generated-zones.toml').open('rb') as source:
            generated = tomllib.load(source)
        with (ROOT / 'agent/ux-qa.toml').open('rb') as source:
            ux = tomllib.load(source)
        self.assertIn('minimum_score', policy)
        self.assertEqual('pass', ux['visualBaselineMode'])
        self.assertTrue(ux['accessibilityScanRequired'])
        self.assertTrue(ux['routes'])
        self.assertTrue(generated['zone'])
        missing = [
            zone['path'] for zone in generated['zone']
            if (not zone['path'].startswith('target/')
                and not self.key_matches_tree(zone['path']))
            or not zone.get('source') or not zone.get('command')
        ]
        self.assertEqual([], missing, f'incomplete generated-zone routes: {missing}')

    def test_each_mapped_proof_command_invokes_a_test_suite(self):
        test_map = json.loads((ROOT / 'agent/test-map.json').read_text())['tests']
        commands = sorted({route.get('command', '') for route in test_map.values()})
        no_tests = [
            command
            for command in commands
            if not any(marker in command for marker in self.TEST_COMMAND_MARKERS)
        ]
        self.assertEqual([], no_tests, f'proof commands without tests: {no_tests}')


    def test_shared_research_audit_refuses_protected_reads_and_input_writes(self):
        helper = ROOT / 'research/flywheel_selfplay/runtime_audit.py'
        spec = importlib.util.spec_from_file_location('research_runtime_audit', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Audit = module.Audit

        with tempfile.TemporaryDirectory(prefix='research-audit-') as name:
            allowed = Path(name) / 'input.bin'
            audit = Audit([allowed], Path(name) / 'output')
            audit.check('open', (str(allowed), 'rb', 0))
            with self.assertRaisesRegex(RuntimeError, 'input/output boundary refusal'):
                audit.check('open', ('/home/ubuntu/jope-data/sealed/row.bin', 'rb', 0))
            with self.assertRaisesRegex(RuntimeError, 'input/output boundary refusal'):
                audit.check('open', (str(allowed), 'wb', 0))
            self.assertEqual(2, len(audit.refused))




if __name__ == '__main__':
    unittest.main()
