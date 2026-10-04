#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'deploy/runner' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load('job-policy')
container = load('verify-container')


class RunnerTests(unittest.TestCase):
    def env(self, workflow='full-main', event='push', ref='refs/heads/main'):
        return dict(GITHUB_REPOSITORY_ID='1376159324', GITHUB_REPOSITORY='neverhuman/JopeDime', GITHUB_WORKFLOW_REF=f'neverhuman/JopeDime/.github/workflows/{workflow}.yml@{ref}', GITHUB_EVENT_NAME=event, GITHUB_REF=ref, GITHUB_SHA='a'*40)

    def test_repository_transfer_identity_is_pinned(self):
        for domain, workflow, event in [('main', 'full-main', 'push'), ('pr', 'ci', 'pull_request')]:
            for repo, repository_id in [('veox-ai/JopeDime', '1376159324'),
                                        ('foreign/repo', '1376159324'),
                                        ('neverhuman/JopeDime', '1'),
                                        ('neverhuman/JopeDime', None)]:
                with self.subTest(domain=domain, repo=repo, repository_id=repository_id):
                    env = self.env(workflow, event)
                    env['GITHUB_REPOSITORY'] = repo
                    if repository_id is None:
                        env.pop('GITHUB_REPOSITORY_ID')
                    else:
                        env['GITHUB_REPOSITORY_ID'] = repository_id
                    with self.assertRaises(ValueError):
                        policy.authorize({'domain': domain}, env)

    def test_trusted_events(self):
        for workflow, event in [('full-main', 'push'), ('full-main', 'workflow_dispatch'), ('main-guard', 'push')]:
            policy.authorize({'domain':'main'}, self.env(workflow, event))

    def test_untrusted_jobs_cannot_enter_main_storage(self):
        for env in [self.env(event='pull_request'), self.env(ref='refs/heads/attack'), self.env(workflow='ci'), self.env(workflow='release')]:
            with self.assertRaises(ValueError):
                policy.authorize({'domain':'main'}, env)
        env = self.env()
        env['JOPE_RUNNER_DOMAIN'] = 'pr'  # workflow env cannot change the root policy
        env['GITHUB_REPOSITORY'] = 'foreign/repo'
        with self.assertRaises(ValueError):
            policy.authorize({'domain':'main'}, env)

    def test_fast_workflow_is_a_separate_domain(self):
        policy.authorize({'domain':'pr'}, self.env('ci', 'pull_request', 'refs/pull/61/merge'))
        policy.authorize({'domain':'pr'}, self.env('ci', 'workflow_dispatch', 'refs/heads/feature'))
        with self.assertRaises(ValueError):
            policy.authorize({'domain':'pr'}, self.env())

    def test_approved_resources_and_capability_absence(self):
        for host, lane, domain, gib in [('xbabe1','ci','pr',16), ('xbabe1','ci2','pr',16), ('xbabe2','full','main',48)]:
            p = dict(host=host,lane=lane,domain=domain,memory=f'{gib}g',cpus=16,volume_prefix=f'jope-{domain}-{lane}')
            container.verify(p, str(gib*1024**3), '1600000 100000', '0', [])
            for memory, cpu, swap, devices in [('max','1600000 100000','0',[]), (str(gib*1024**3),'2800000 100000','0',[]), (str(gib*1024**3),'1600000 100000','1',[]), (str(gib*1024**3),'1600000 100000','0',['/dev/nvidia0'])]:
                with self.assertRaises(ValueError):
                    container.verify(p,memory,cpu,swap,devices)

    def test_host_roles_and_environment_cannot_raise_budget(self):
        for host, lane, expected in [('xbabe1','ci','16g 16 pr'),('xbabe1','ci2','16g 16 pr'),('xbabe2','full','48g 16 main')]:
            script = 'JOPE_RUNNER_CPUS=28; JOPE_RUNNER_MEMORY=24g; source deploy/runner/common.sh; jope_runner_role "$1" "$2"; echo "$JOPE_RUNNER_MEMORY $JOPE_RUNNER_CPUS $JOPE_RUNNER_DOMAIN"'
            result = subprocess.run(['bash','-ec',script,'test',host,lane],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(0,result.returncode)
            self.assertEqual(expected,result.stdout.strip())
        for host, lane in [('xbabe2','ci'),('xbabe2','gpu'),('xbabe1','full'),('xbabe1','ci3')]:
            result = subprocess.run(['bash','-ec','source deploy/runner/common.sh; jope_runner_role "$1" "$2"','test',host,lane],cwd=ROOT,capture_output=True)
            self.assertEqual(2,result.returncode)

    def test_orphans_and_legacy_roles_refuse_start(self):
        cases = [('xbabe1','ci','abc ci',2), ('xbabe1','ci','abc ci2',0),
                 ('xbabe2','full','abc full',2), ('xbabe2','full','abc ci',2),
                 ('xbabe1','ci','abc ',2), ('xbabe2','full','',0)]
        script = 'source deploy/runner/common.sh; JOPE_RUNNER_HOST=$1; existing_fixture=$3; docker() { printf "%s\\n" "$existing_fixture"; }; jope_runner_check_existing "$2"'
        for host,lane,existing,code in cases:
            with self.subTest(host=host,lane=lane,existing=existing):
                result=subprocess.run(['bash','-c',script,'test',host,lane,existing],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(code,result.returncode,result.stderr)
        result=subprocess.run(['bash','-c','source deploy/runner/common.sh; docker() { return 23; }; jope_runner_check_existing full'],cwd=ROOT,capture_output=True)
        self.assertEqual(23,result.returncode)


    def test_orphan_asm_cgroup_blocks_exclusive_cpu_reuse(self):
        scratch_root = ROOT / 'target' / 'tmp'
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='runner-asm-fixture-', dir=scratch_root) as name:
            root = Path(name)
            group = root / 'jope-asm-fixture.service'
            group.mkdir()
            script = 'source deploy/runner/common.sh; jope_runner_check_asm_groups "$1"'
            cases = [(None, 2), ('populated 1\n', 2), ('populated 0\n', 0),
                     ('frozen 0\n', 2), ('populated unknown\n', 2)]
            for content, expected in cases:
                with self.subTest(content=content):
                    if content is not None:
                        (group / 'cgroup.events').write_text(content)
                    result = subprocess.run(['bash', '-c', script, 'fixture', str(root)],
                                            cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(expected, result.returncode, result.stderr)


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
        self.assertIn('bash deploy/ci/secret-scan.sh', (ROOT/'deploy/ci/check.sh').read_text())


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


class EmitterEvidenceTests(unittest.TestCase):
    def test_actual_lane_routes_emitter_output_to_run_evidence(self):
        import os
        scratch = ROOT / 'target' / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        script = r'''rustup() { :; }
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
        self.assertEqual(3, action.count('retention-days: 7'))
        self.assertEqual(3, action.count('if-no-files-found: error'))
        self.assertIn('bash deploy/ci/publication-status.sh "$FIRST" "$SECOND" "$THIRD"', action)


if __name__ == '__main__':
    unittest.main()
