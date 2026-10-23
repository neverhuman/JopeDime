#!/usr/bin/env python3
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


class RunnerTests(unittest.TestCase):

    def test_cargo_target_selector_covers_the_approved_build_inventory(self):
        inventory = """source deploy/runner/common.sh
for host in xbabe0 xbabe1 xbabe2 xbabe3 xbabe4; do
  for lane in $(jope_runner_lanes "$host"); do
    jope_runner_role "$host" "$lane"
    if [[ $JOPE_RUNNER_DOMAIN != guard ]]; then printf '%s\\n' "$lane"; fi
  done
done
"""
        roles = subprocess.run(['bash', '-ec', inventory], cwd=ROOT,
                               capture_output=True, text=True)
        self.assertEqual(0, roles.returncode, roles.stderr)
        approved = set(roles.stdout.splitlines())
        self.assertTrue(approved)
        action = (ROOT/'.github/actions/cargo-target/action.yml').read_text()
        marker = '      run: |\n'
        self.assertEqual(1, action.count(marker))
        script = textwrap.dedent(action.split(marker, 1)[1])
        scratch = ROOT/'target'/'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cargo-target-contract-', dir=scratch) as name:
            output = Path(name)/'github-env'
            for lane in sorted(approved):
                with self.subTest(lane=lane):
                    output.write_text('')
                    result = subprocess.run(['bash', '-ec', script], cwd=ROOT,
                        env=os.environ | {'JOPE_RUNNER_LANE': lane, 'GITHUB_ENV': str(output)},
                        capture_output=True, text=True)
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertEqual(f'CARGO_TARGET_DIR=/var/cache/jope/target/{lane}\n',
                                     output.read_text())
            for lane in ('guard', 'gpu', 'ci13', '../full', '', 'ci\nINJECTED=1'):
                with self.subTest(rejected_lane=lane):
                    output.write_text('')
                    result = subprocess.run(['bash', '-ec', script], cwd=ROOT,
                        env=os.environ | {'JOPE_RUNNER_LANE': lane, 'GITHUB_ENV': str(output)},
                        capture_output=True, text=True)
                    self.assertNotEqual(0, result.returncode)
                    self.assertEqual('', output.read_text())

    def test_guard_domain_admits_only_the_exact_main_guard_push(self):
        policy.authorize({'domain': 'guard'}, self.env('main-guard', 'push'))
        for workflow, event, ref in [('ci', 'pull_request', 'refs/pull/1/merge'),
                                     ('ci', 'workflow_dispatch', 'refs/heads/main'),
                                     ('full-main', 'push', 'refs/heads/main'),
                                     ('main-guard', 'workflow_dispatch', 'refs/heads/main'),
                                     ('main-guard', 'push', 'refs/heads/feature'),
                                     ('release', 'push', 'refs/heads/main')]:
            with self.subTest(workflow=workflow, event=event, ref=ref):
                with self.assertRaises(ValueError):
                    policy.authorize({'domain': 'guard'}, self.env(workflow, event, ref))
        for key, value in [('GITHUB_REPOSITORY', 'foreign/repo'),
                           ('GITHUB_REPOSITORY_ID', '1'), ('GITHUB_SHA', '')]:
            env = self.env('main-guard', 'push')
            env[key] = value
            with self.assertRaises(ValueError):
                policy.authorize({'domain': 'guard'}, env)

    def test_guard_resources_and_storage_remain_small_and_closed(self):
        script = 'source deploy/runner/common.sh; JOPE_RUNNER_HOST=$1; jope_runner_role "$1" guard; echo "$JOPE_RUNNER_MEMORY $JOPE_RUNNER_CPUS $JOPE_RUNNER_DOMAIN"; jope_runner_labels guard'
        for host in ('xbabe1', 'xbabe2', 'xbabe3'):
            result = subprocess.run(['bash', '-ec', script, 'test', host], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(f'2g 2 guard\n{host}-guard,jope-main-guard', result.stdout.strip())
            launch = dict(host=host, lane='guard', domain='guard', memory='2g', cpus=2,
                          volume_prefix='jope-guard-guard')
            container.verify(launch, str(2*1024**3), '200000 100000', '0', [])
            for memory, cpu, swap, devices in [(str(4*1024**3), '200000 100000', '0', []),
                                                (str(2*1024**3), '1600000 100000', '0', []),
                                                (str(2*1024**3), '200000 100000', '1', []),
                                                (str(2*1024**3), '200000 100000', '0', ['/var/run/docker.sock'])]:
                with self.assertRaises(ValueError):
                    container.verify(launch, memory, cpu, swap, devices)
            for field, value in [('domain', 'main'), ('volume_prefix', 'jope-main-full')]:
                with self.assertRaises(ValueError):
                    container.verify(launch | {field: value}, str(2*1024**3), '200000 100000', '0', [])

    def test_mac_pr_slots_are_explicit_and_never_gain_trusted_labels(self):
        script = 'source deploy/runner/common.sh; JOPE_RUNNER_HOST=xbabe4; jope_runner_labels "$1"'
        for lane in ('ci', 'ci2', 'ci3', 'ci4', 'ci5', 'ci6', 'ci7', 'ci8', 'ci9', 'ci10', 'ci11', 'ci12'):
            result = subprocess.run(['bash', '-ec', script, 'test', lane], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(f'xbabe4-{lane},jope-ci', result.stdout.strip())
            launch = dict(host='xbabe4', lane=lane, domain='pr', memory='16g', cpus=16,
                          volume_prefix=f'jope-pr-{lane}')
            container.verify(launch, str(16*1024**3), '1600000 100000', '0', [])
            with self.assertRaises(ValueError):
                container.verify(launch | {'domain': 'main'}, str(16*1024**3), '1600000 100000', '0', [])
        for lane in ('full', 'guard', 'gpu', 'ci13'):
            result = subprocess.run(['bash', '-ec', script, 'test', lane], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode)

    def test_external_jit_input_is_bounded_host_bound_and_never_echoed(self):
        script = 'source deploy/runner/common.sh; jope_runner_external_jit "$1" "$2" "$3"; printf "%s" "${#JOPE_EXTERNAL_JIT}"'
        token = 'Q'*64  # Synthetic fixture, never a registration credential.
        name = 'xbabe4-ci-20261023T000000Z'
        valid = subprocess.run(['bash', '-ec', script, 'test', 'xbabe4', 'ci', name], cwd=ROOT,
                               input=token, capture_output=True, text=True)
        self.assertEqual(0, valid.returncode, valid.stderr)
        self.assertEqual('64', valid.stdout)
        self.assertNotIn(token, valid.stdout + valid.stderr)
        for host, lane, candidate, payload in [('xbabe2', 'ci', name, token),
                                                ('xbabe4', 'full', name, token),
                                                ('xbabe4', 'ci', 'xbabe4-ci2-20261023T000000Z', token),
                                                ('xbabe4', 'ci', name, ''),
                                                ('xbabe4', 'ci', name, 'x;uname -a'),
                                                ('xbabe4', 'ci', name, 'Q'*262145)]:
            with self.subTest(host=host, lane=lane, length=len(payload)):
                result = subprocess.run(['bash', '-ec', script, 'test', host, lane, candidate], cwd=ROOT,
                                        input=payload, capture_output=True, text=True)
                self.assertNotEqual(0, result.returncode)
                self.assertEqual('', result.stdout)
                self.assertNotIn(token, result.stderr)

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

    def test_main_domain_admits_ci_yml_on_full_jope_ci(self):
        # Owner permanent 2026-10-14: full hosts register jope-ci and may run ci.yml.
        policy.authorize({'domain':'main'}, self.env('ci', 'pull_request', 'refs/pull/61/merge'))
        policy.authorize({'domain':'main'}, self.env('ci', 'workflow_dispatch', 'refs/heads/feature'))

    def test_approved_resources_and_capability_absence(self):
        for host, lane, domain, gib in [('xbabe1','ci','pr',16), ('xbabe1','ci2','pr',16), ('xbabe2','full','main',48), ('xbabe3','full','main',48), ('xbabe2','ci','pr',16), ('xbabe3','ci','pr',16), ('xbabe0','ci','pr',16)]:
            p = dict(host=host,lane=lane,domain=domain,memory=f'{gib}g',cpus=16,volume_prefix=f'jope-{domain}-{lane}')
            container.verify(p, str(gib*1024**3), '1600000 100000', '0', [])
            for memory, cpu, swap, devices in [('max','1600000 100000','0',[]), (str(gib*1024**3),'2800000 100000','0',[]), (str(gib*1024**3),'1600000 100000','1',[]), (str(gib*1024**3),'1600000 100000','0',['/dev/nvidia0'])]:
                with self.assertRaises(ValueError):
                    container.verify(p,memory,cpu,swap,devices)

    def test_full_lane_registers_jope_ci(self):
        script = 'source deploy/runner/common.sh; JOPE_RUNNER_HOST=$1; jope_runner_labels "$2"'
        for host in ('xbabe2', 'xbabe3'):
            result = subprocess.run(['bash','-ec',script,'test',host,'full'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(f'{host}-full,jope-full-cpu,jope-ci', result.stdout.strip())
        for host in ('xbabe1', 'xbabe2', 'xbabe3', 'xbabe0'):
            result = subprocess.run(['bash','-ec',script,'test',host,'ci'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(f'{host}-ci,jope-ci', result.stdout.strip())

    def test_each_host_lists_exactly_its_approved_lanes(self):
        script = 'source deploy/runner/common.sh; jope_runner_lanes "$1"'
        expected = {'xbabe1': 'ci ci2 guard', 'xbabe2': 'full ci guard', 'xbabe3': 'full ci guard', 'xbabe0': 'ci', 'xbabe4': 'ci ci2 ci3 ci4 ci5 ci6 ci7 ci8 ci9 ci10 ci11 ci12'}
        for host, lanes in expected.items():
            result = subprocess.run(['bash','-ec',script,'test',host],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(lanes, result.stdout.strip())
            for lane in lanes.split():
                role = subprocess.run(['bash','-ec','source deploy/runner/common.sh; jope_runner_role "$1" "$2"','test',host,lane],cwd=ROOT,capture_output=True)
                self.assertEqual(0, role.returncode, (host, lane))
        self.assertEqual(22, sum(len(lanes.split()) for lanes in expected.values()))
        for host in ('xbabe5', 'xbabe2-mac', 'Mac-Studio.local', ''):
            result = subprocess.run(['bash','-c',script,'test',host],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(2, result.returncode, host)

    def test_host_roles_and_environment_cannot_raise_budget(self):
        for host, lane, expected in [('xbabe1','ci','16g 16 pr'),('xbabe1','ci2','16g 16 pr'),('xbabe2','full','48g 16 main'),('xbabe3','full','48g 16 main'),('xbabe2','ci','16g 16 pr'),('xbabe3','ci','16g 16 pr'),('xbabe0','ci','16g 16 pr')]:
            script = 'JOPE_RUNNER_CPUS=28; JOPE_RUNNER_MEMORY=24g; source deploy/runner/common.sh; jope_runner_role "$1" "$2"; echo "$JOPE_RUNNER_MEMORY $JOPE_RUNNER_CPUS $JOPE_RUNNER_DOMAIN"'
            result = subprocess.run(['bash','-ec',script,'test',host,lane],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(0,result.returncode)
            self.assertEqual(expected,result.stdout.strip())
        for host, lane in [('xbabe2','ci2'),('xbabe2','gpu'),('xbabe1','full'),('xbabe1','ci3'),('xbabe3','ci2'),('xbabe3','gpu'),('xbabe0','ci2'),('xbabe0','full'),('xbabe0','gpu'),('xbabe4','full'),('xbabe4','guard'),('xbabe4','ci13'),('xbabe5','ci')]:
            result = subprocess.run(['bash','-ec','source deploy/runner/common.sh; jope_runner_role "$1" "$2"','test',host,lane],cwd=ROOT,capture_output=True)
            self.assertEqual(2,result.returncode)

    def test_orphans_and_legacy_roles_refuse_start(self):
        cases = [('xbabe1','ci','abc ci',2), ('xbabe1','ci','abc ci2',0),
                 ('xbabe2','full','abc full',2), ('xbabe2','full','abc ci',0), ('xbabe2','ci','abc full',0),
                 ('xbabe2','ci','abc ci',2), ('xbabe2','full','abc ci2',2),
                 ('xbabe1','ci','abc ',2), ('xbabe2','full','',0),
                 ('xbabe3','full','abc full',2), ('xbabe3','full','abc ci',0), ('xbabe3','ci','abc full',0),
                 ('xbabe3','full','abc ci2',2), ('xbabe3','full','',0),
                 ('xbabe0','ci','abc ci',2), ('xbabe0','ci','abc ci2',2), ('xbabe0','ci','abc full',2), ('xbabe0','ci','',0)]
        script = 'source deploy/runner/common.sh; JOPE_RUNNER_HOST=$1; existing_fixture=$3; docker() { printf "%s\\n" "$existing_fixture"; }; jope_runner_check_existing "$2"'
        for host,lane,existing,code in cases:
            with self.subTest(host=host,lane=lane,existing=existing):
                result=subprocess.run(['bash','-c',script,'test',host,lane,existing],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(code,result.returncode,result.stderr)
        result=subprocess.run(['bash','-c','source deploy/runner/common.sh; docker() { return 23; }; jope_runner_check_existing full'],cwd=ROOT,capture_output=True)
        self.assertEqual(23,result.returncode)

    def test_every_full_lane_and_only_full_lanes_take_the_exclusive_cpu_lock(self):
        # flock is stubbed to refuse, so no live lock inode is ever taken by this test. A host
        # without the provisioned inode (a CI container) refuses before flock: still not idle.
        script = 'source deploy/runner/common.sh; JOPE_RUNNER_HOST=$1; flock() { echo flock-called; return 1; }; jope_runner_exclusive_cpu "$2"'
        for host, lane, code in [('xbabe2','full',2), ('xbabe3','full',2), ('xbabe1','ci',0), ('xbabe1','ci2',0), ('xbabe2','ci',0), ('xbabe3','ci',0), ('xbabe0','ci',0)]:
            with self.subTest(host=host, lane=lane):
                result = subprocess.run(['bash','-c',script,'test',host,lane],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(code, result.returncode, result.stderr)
                if code == 0:
                    self.assertNotIn('flock-called', result.stdout)
                else:
                    self.assertTrue('flock-called' in result.stdout or 'lock not provisioned' in result.stderr, result.stderr)


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


class MacControllerTests(unittest.TestCase):
    def scratch(self):
        base = ROOT / 'target/tmp'; base.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='mac-shell-fixture-', dir=base)

    def fixture(self, parent):
        import json
        cfg = json.loads((ROOT/'deploy/runner/mac-capacity.json').read_text())
        values = dict(policy=cfg, mac=dict(host='Mac-Studio.local', free_bytes=500*1024**3),
            user=dict(login='jepsontaylor'), guest=dict(host='xbabe4', architecture='aarch64',
            cpus=24, total_gib=251, available_gib=80, root_free_gib=80, docker_free_gib=600,
            cgroup_version='2', cgroup_driver='systemd', image_id=cfg['image_id'],
            recorded_image_id=cfg['image_id'], image_architecture='amd64', digest='a'*64,
            recorded_digest='a'*64, sdk_version=cfg['sdk_version'], lane_idle=True,
            active_lanes=[], reserved_gib=0), response=dict(runner=dict(id=7),
            encoded_jit_config='RklYVFVSRV9PTkxZX05PVF9BX1JFQUxfQ1JFREVOVElBTA=='))
        (parent/'state').mkdir(); (parent/'mint-count').write_text('0')
        return values

    def run_fixture(self, parent, values, lane='ci'):
        import json
        for key, value in values.items(): (parent/(key+'.json')).write_text(json.dumps(value))
        driver = r'''
source deploy/runner/mac-controller.sh
fixture=$1
jope_mac_digest() { printf '%064d\n' 0 | tr 0 a; }
jope_mac_api() {
  if [[ $1 == user ]]; then cat "$fixture/user.json"; return; fi
  local n; n=$(cat "$fixture/mint-count"); echo "$((n+1))" > "$fixture/mint-count"
  [[ ! -e $fixture/FAIL_MINT ]] || return 1
  jq -c --argjson request "$2" '
    if .wrong_name then .runner.name="foreign" else .runner.name=$request.name end |
    if .wrong_labels then .runner.labels=[{name:"jope-full-cpu"}]
    else .runner.labels=[$request.labels[]|{name:.}] end' "$fixture/response.json"
}
jope_mac_ssh() {
  case $1 in
    host) cat "$fixture/mac.json" ;;
    preflight) cat "$fixture/guest.json" ;;
    run)
      printf '%s\n' "$*" >> "$fixture/run-arguments"
      local input expected; input=$(cat); expected=$(jq -r .encoded_jit_config "$fixture/response.json")
      [[ $input == "$expected" ]] || return 2
      [[ ! -e $fixture/FAIL_SSH ]] || return 1 ;;
    *) return 2 ;;
  esac
}
jope_mac_launch "$2" "$fixture/state" "$(cat "$fixture/policy.json")" 20261023T120000Z
'''
        return subprocess.run(['bash','-x','-c',driver,'fixture',str(parent),lane],cwd=ROOT,capture_output=True,text=True)

    def test_mac_valid_path_uses_stdin_and_retains_only_public_state(self):
        import json
        for lane in ('ci', 'ci2', 'ci3', 'ci4', 'ci5', 'ci6', 'ci7', 'ci8', 'ci9', 'ci10', 'ci11', 'ci12'):
            with self.subTest(lane=lane), self.scratch() as name:
                root=Path(name); values=self.fixture(root); result=self.run_fixture(root,values,lane)
                self.assertEqual(0,result.returncode,result.stderr)
                self.assertEqual('1',(root/'mint-count').read_text().strip())
                private=values['response']['encoded_jit_config']
                self.assertNotIn(private,result.stdout+result.stderr+(root/'run-arguments').read_text())
                for p in (root/'state').rglob('*.json'):
                    self.assertEqual('complete',json.loads(p.read_text())['phase'])
                    self.assertNotIn(private,p.read_text())
                self.assertEqual(0o600,(root/'state'/f'{lane}.json').stat().st_mode & 0o777)
                again=self.run_fixture(root,values,lane)
                self.assertNotEqual(0,again.returncode)
                self.assertEqual('1',(root/'mint-count').read_text().strip())

    def test_mac_each_guest_precondition_refuses_before_mint(self):
        bad=dict(host='foreign',architecture='x86_64',cpus=16,total_gib=60,available_gib=19,
            root_free_gib=49,docker_free_gib=119,cgroup_version='1',cgroup_driver='cgroupfs',
            image_id='sha256:'+'f'*64,recorded_image_id='sha256:'+'f'*64,image_architecture='arm64',
            digest='b'*64,recorded_digest='b'*64,sdk_version='wrong',lane_idle=False,
            active_lanes=['ci'],reserved_gib=64)
        cases=list(bad.items())+[(key, 'unbounded') for key in ('total_gib','available_gib','root_free_gib','docker_free_gib')]
        for key,value in cases:
            with self.subTest(key=key),self.scratch() as name:
                root=Path(name); values=self.fixture(root);values['guest'][key]=value
                result=self.run_fixture(root,values)
                self.assertNotEqual(0,result.returncode,key)
                self.assertEqual('0',(root/'mint-count').read_text().strip())

    def test_twelfth_slot_admission_and_closed_inventory(self):
        lanes = ['ci'] + ['ci'+str(n) for n in range(2,13)]
        with self.scratch() as name:
            root=Path(name); values=self.fixture(root)
            values['guest'].update(active_lanes=lanes[:-1], reserved_gib=176)
            result=self.run_fixture(root,values,'ci12')
            self.assertEqual(0,result.returncode,result.stderr)
            self.assertEqual('1',(root/'mint-count').read_text().strip())
        for active,reserved in [(lanes,192), (lanes[:-1]+['ci13'],192),
                                (['ci2','ci2'],32), (lanes[:-1],160)]:
            with self.subTest(active=active,reserved=reserved),self.scratch() as name:
                root=Path(name);values=self.fixture(root)
                values['guest'].update(active_lanes=active,reserved_gib=reserved)
                result=self.run_fixture(root,values,'ci12')
                self.assertNotEqual(0,result.returncode)
                self.assertEqual('0',(root/'mint-count').read_text().strip())
        for lane in ['ci0','ci1','ci01','ci13','ci99','guard','full','ci;id']:
            with self.subTest(lane=lane),self.scratch() as name:
                root=Path(name);values=self.fixture(root)
                result=self.run_fixture(root,values,lane)
                self.assertNotEqual(0,result.returncode)
                self.assertEqual('0',(root/'mint-count').read_text().strip())

    def test_mac_unknown_mint_and_ssh_outcomes_block_fresh_process_remint(self):
        import json
        for failure in ('FAIL_MINT','FAIL_SSH'):
            with self.subTest(failure=failure),self.scratch() as name:
                root=Path(name);values=self.fixture(root);(root/failure).touch()
                result=self.run_fixture(root,values)
                self.assertNotEqual(0,result.returncode)
                self.assertEqual('unknown',json.loads((root/'state/ci.json').read_text())['phase'])
                again=self.run_fixture(root,values)
                self.assertNotEqual(0,again.returncode)
                self.assertEqual('1',(root/'mint-count').read_text().strip())
                self.assertNotIn(values['response']['encoded_jit_config'],result.stdout+result.stderr+again.stdout+again.stderr)

    def test_mac_registration_and_private_input_fail_closed(self):
        mutations=[dict(wrong_name=True),dict(wrong_labels=True),dict(encoded_jit_config='A'*48+'\n'),
                   dict(encoded_jit_config='bad\nvalue'),dict(encoded_jit_config='A'*262145)]
        for mutation in mutations:
            with self.scratch() as name:
                root=Path(name);values=self.fixture(root);values['response'].update(mutation)
                result=self.run_fixture(root,values)
                self.assertNotEqual(0,result.returncode)
                self.assertFalse((root/'run-arguments').exists())

    def test_mac_transport_identity_reserve_and_lane_lock_are_closed(self):
        import fcntl
        for failure in ('host','disk','disk_text','identity','alias','profile','lock'):
            with self.subTest(failure=failure),self.scratch() as name:
                root=Path(name);values=self.fixture(root)
                if failure=='host':values['mac']['host']='foreign'
                if failure=='disk':values['mac']['free_bytes']=119*1024**3
                if failure=='disk_text':values['mac']['free_bytes']='unbounded'
                if failure=='identity':values['user']['login']='foreign'
                if failure=='alias':values['policy']['ssh_alias']='foreign'
                if failure=='profile':values['policy']['profile']='jarambit-ci'
                with (root/'state/ci.lock').open('a') as lock:
                    if failure=='lock':fcntl.flock(lock,fcntl.LOCK_EX)
                    result=self.run_fixture(root,values)
                self.assertNotEqual(0,result.returncode)
                self.assertEqual('0',(root/'mint-count').read_text().strip())

    def test_mac_sdk_inventory_and_shell_contracts_match_source(self):
        import json
        cfg=json.loads((ROOT/'deploy/runner/mac-capacity.json').read_text())
        self.assertEqual(json.loads((ROOT/'deploy/receipts/libtorch.json').read_text())['libtorch']['version'],cfg['sdk_version'])
        provenance=load('provenance_bind');paths=provenance.parse_deployed_relpaths((ROOT/'deploy/runner/common.sh').read_text())
        for name in ('mac-controller.sh','mac-guest.sh','mac-capacity.json','jope-mac-runner@.service','jope-mac-runner@.timer'):
            self.assertIn('deploy/runner/'+name,paths)
        for name in ('mac-controller.sh','mac-guest.sh'):
            result=subprocess.run(['bash','-n',str(ROOT/'deploy/runner'/name)],capture_output=True,text=True)
            self.assertEqual(0,result.returncode,result.stderr)
        self.assertIn('jope-main-guard',(ROOT/'.github/workflows/main-guard.yml').read_text())


class TrimTests(unittest.TestCase):
    """deploy/runner/trim-deps.sh and common.sh jope_runner_trim_lane (bounded build-cache trim)."""

    SCRIPT = ROOT / 'deploy/runner/trim-deps.sh'

    def fixture(self, root):
        """A lane target tree. Returns (expected selected paths, every created path)."""
        created, selected = [], set()

        def make(rel, age):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'x' * (len(rel) % 7 + 1))
            stamp = 1_700_000_000 + age
            os.utime(path, (stamp, stamp))
            created.append(path)
            return path

        def h(i):
            return '%016x' % (0xabc000 + i)
        for i in range(9):   # newest 6 rlibs stay; 0, 1, 2 are older duplicates
            p = make(f'ci/debug/deps/libfoo-{h(i)}.rlib', i)
            if i < 3:
                selected.add(p)
        for i in range(3):   # another kind of the same crate counts separately
            make(f'ci/debug/deps/libfoo-{h(i)}.rmeta', i)
        for i in range(8):   # test binaries (no extension)
            p = make(f'ci/debug/deps/foo-{h(i)}', i)
            if i < 2:
                selected.add(p)
        for i in range(7):
            p = make(f'ci/debug/deps/foo-{h(i)}.d', i)
            if i < 1:
                selected.add(p)
        for i in range(7):   # each lane directory counts separately
            p = make(f'ci2/debug/deps/libfoo-{h(i)}.rlib', i)
            if i < 1:
                selected.add(p)
        # Never candidates: other names, other directories, deeper paths, evidence.
        for i in range(9):
            make(f'ci/debug/build/foo-{h(i)}/out/libfoo-{h(i)}.rlib', i)
            make(f'ci/tmp/libfoo-{h(i)}.rlib', i)
            make(f'ci/dist-ci/deps/libfoo-{h(i)}.rlib', i)
            make(f'ci-evidence-retained/debug/deps/libfoo-{h(i)}.rlib', i)
            make(f'ci/debug/deps/README-{i}', i)
            make(f'ci/debug/deps/libbar-{i}23.rlib', i)
            make(f'ci/debug/deps/libbaz-{h(i).upper()}.rlib', i)
        return selected, created

    def run_trim(self, root, *args, awk=None):
        env = dict(os.environ)
        if awk:
            shim = root.parent / 'awk-shim'
            shim.mkdir(exist_ok=True)
            link = shim / 'awk'
            if not link.exists():
                link.symlink_to(awk)
            env['PATH'] = f'{shim}:{env["PATH"]}'
        return subprocess.run(['bash', str(self.SCRIPT), str(root), *args], capture_output=True, text=True, env=env)

    def scratch(self):
        base = ROOT / 'target' / 'tmp'
        base.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='runner-trim-fixture-', dir=base)

    def test_keeps_newest_builds_per_crate_and_kind_and_nothing_else(self):
        awks = [None] + [a for a in (shutil.which('mawk'), shutil.which('gawk')) if a]
        for awk in awks:
            with self.subTest(awk=awk), self.scratch() as name:
                root = Path(name) / 'v'
                selected, created = self.fixture(root)
                dry = self.run_trim(root, '6', awk=awk)
                self.assertEqual(0, dry.returncode, dry.stderr)
                rows = [line.split('\t') for line in dry.stdout.splitlines()]
                self.assertEqual({str(p) for p in selected}, {row[0] for row in rows})
                for row in rows:
                    self.assertEqual(3, len(row))
                    self.assertEqual(Path(row[0]).stat().st_size, int(row[1]))
                self.assertTrue(all(p.exists() for p in created), 'a dry run deleted files')
                applied = self.run_trim(root, '6', '--apply', awk=awk)
                self.assertEqual(0, applied.returncode, applied.stderr)
                self.assertEqual(dry.stdout, applied.stdout)
                for path in created:
                    self.assertEqual(path not in selected, path.exists(), path)
                again = self.run_trim(root, '6', '--apply', awk=awk)
                self.assertEqual((0, ''), (again.returncode, again.stdout))

    def test_refuses_bad_arguments(self):
        with self.scratch() as name:
            root = Path(name) / 'v'
            root.mkdir()
            for args in [('0',), ('-1',), ('six',), ('6', '--delete'), ()]:
                with self.subTest(args=args):
                    self.assertNotEqual(0, self.run_trim(root, *args).returncode)
            missing = self.run_trim(root / 'absent', '6', '--apply')
            self.assertEqual(2, missing.returncode)

    @unittest.skipIf(os.geteuid() == 0, 'root reads unreadable directories')
    def test_listing_error_deletes_nothing(self):
        with self.scratch() as name:
            root = Path(name) / 'v'
            selected, created = self.fixture(root)
            locked = root / 'ci3'
            (locked / 'debug').mkdir(parents=True)
            locked.chmod(0)
            try:
                result = self.run_trim(root, '6', '--apply')
            finally:
                locked.chmod(0o755)
            self.assertNotEqual(0, result.returncode)
            self.assertEqual('', result.stdout)
            self.assertTrue(all(p.exists() for p in created))

    def trim_lane(self, state, docker_body, lane='ci'):
        script = ('source deploy/runner/common.sh; JOPE_RUNNER_STATE=$1; JOPE_RUNNER_DOMAIN=pr; '
                  'calls=$1/calls; docker() { printf "%s\\n" "$*" >> "$calls"; ' + docker_body + ' }; '
                  'jope_runner_trim_lane "$2"')
        return subprocess.run(['bash', '-c', script, 'test', str(state), lane], cwd=ROOT, capture_output=True, text=True)

    def test_lane_trim_is_isolated_rate_limited_and_never_fatal(self):
        row = '/v/ci/debug/deps/libfoo-0123456789abcdef.rlib\\t10\\t1.5\\n'
        ok = f'if [[ $1 == run ]]; then printf "{row}"; fi; return 0;'
        with self.scratch() as name:
            state = Path(name)
            first = self.trim_lane(state, ok)
            self.assertEqual(0, first.returncode, first.stderr)
            calls = (state / 'calls').read_text().splitlines()
            run = [c for c in calls if c.startswith('run ')]
            self.assertEqual(1, len(run), calls)
            args = run[0].split()
            for flag in ('--network none', '--cap-drop ALL', '--read-only', '--user 1001'):
                self.assertIn(flag, run[0])
            mounts = [args[i + 1] for i, a in enumerate(args) if a == '-v']
            self.assertEqual(['jope-pr-ci-target:/v', '/opt/jope-runner/trim-deps.sh:/opt/jope-runner/trim-deps.sh:ro'], mounts)
            self.assertEqual(['/v', '3', '--apply'], args[-3:])
            manifests = sorted((state / 'trim').glob('*-ci.tsv'))
            self.assertEqual(1, len(manifests))
            self.assertIn('libfoo-0123456789abcdef.rlib\t10\t1.5', manifests[0].read_text())
            self.assertIn('removed 1 files, 10 bytes from jope-pr-ci-target', first.stdout)
            self.assertTrue((state / 'trim' / 'ci.last').exists())
            # Within the interval nothing runs again.
            (state / 'calls').unlink()
            second = self.trim_lane(state, ok)
            self.assertEqual(0, second.returncode, second.stderr)
            self.assertFalse((state / 'calls').exists())
            # After the interval it runs; a failing trim is reported, keeps the old stamp, never fails.
            stamp = state / 'trim' / 'ci.last'
            os.utime(stamp, (1, 1))
            failed = self.trim_lane(state, 'if [[ $1 == run ]]; then return 9; fi; return 0;')
            self.assertEqual(0, failed.returncode)
            self.assertIn('trim ci: failed', failed.stderr)
            self.assertEqual(1, int(stamp.stat().st_mtime))
            # A lane without a target volume is skipped.
            missing = self.trim_lane(state, 'if [[ $1 == volume ]]; then return 1; fi; return 0;', lane='ci2')
            self.assertEqual(0, missing.returncode)
            self.assertIn('no volume jope-pr-ci2-target', missing.stdout)

    def test_launcher_trims_only_while_holding_the_idle_lane(self):
        text = (ROOT / 'deploy/runner/run.sh').read_text()
        order = [text.index(marker) for marker in (
            'flock -n 9', 'jope_runner_check_existing "$lane"', 'jope_runner_trim_lane "$lane" || true',
            '/disk-guard.sh"', 'generate-jitconfig"', 'exec docker')]
        self.assertEqual(sorted(order), order)
        common = (ROOT / 'deploy/runner/common.sh').read_text()
        self.assertIn('/opt/jope-runner/trim-deps.sh\n', common.split('JOPE_RUNNER_DEPLOYED=(')[1].split(')')[0])


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
        self.assertIn('bash tools/security-lane.sh', (ROOT/'deploy/ci/check.sh').read_text())
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

    def test_cache_evidence_paths_are_workspace_relative_and_match(self):
        for name in ('ci.yml', 'full-main.yml'):
            text = (ROOT/'.github/workflows'/name).read_text()
            saved = re.findall(r'uses: \./\.github/actions/upload-ci-evidence\n\s+with:\n\s+name: (.+)\n\s+path: (.+)', text)
            restored = re.findall(r'uses: \./\.github/actions/download-ci-evidence\n\s+with:\n\s+name: (.+)\n\s+path: (.+)', text)
            self.assertTrue(saved and restored, name)
            for _, path in saved + restored:
                self.assertTrue(path.startswith('target/'), (name, path))
            lanes = {n: p for n, p in saved}
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


if __name__ == '__main__':
    unittest.main()
