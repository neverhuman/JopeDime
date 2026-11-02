from _runner_case_support import *


class RunnerTests(unittest.TestCase):

    def test_selected_jobs_survive_the_optional_classifier_ancestor(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        blocks = dict(re.findall(r'^  ([a-z][a-z0-9-]*):\n(.*?)(?=^  [a-z][a-z0-9-]*:|\Z)', workflow, re.M | re.S))
        downstream = {name: block for name, block in blocks.items() if re.search(r'^    needs: changes$', block, re.M)}
        self.assertTrue(downstream)
        # Both real event routes skip one ancestor. Actions implicitly adds
        # success() unless a job condition contains an explicit status function.
        for event in ('pull_request', 'workflow_dispatch'):
            for name, block in downstream.items():
                with self.subTest(event=event, job=name):
                    condition = re.search(r'^    if: (.+)$', block, re.M)
                    expression = condition.group(1) if condition else ''
                    has_status = bool(re.search(r'\b(?:always|cancelled|success|failure)\(', expression))
                    self.assertTrue(has_status, 'selected job inherits a skipped classifier ancestor and never runs')
                    self.assertIn("needs.changes.result == 'success'", expression,
                                  'failed classifier must never authorize execution')
                    self.assertIn('!cancelled()', expression, 'canceled cohorts must stop selected work')


    def test_reusable_concurrency_cannot_cancel_its_caller(self):
        workflows = ROOT / '.github/workflows'
        for caller in ('ci.yml', 'full-main.yml'):
            source = (workflows / caller).read_text()
            caller_name = re.search(r'^name: (.+)$', source, re.M).group(1)
            parent = re.search(r'^concurrency:\n  group: (.+)$', source, re.M).group(1)
            children = re.findall(r'uses: \./\.github/workflows/([\w.-]+\.ya?ml)', source)
            for ref, pr in (('refs/heads/feature', ''), ('refs/pull/1716/merge', '1716'), ('refs/heads/main', '')):
                def resolve(value):
                    values = {'github.workflow': caller_name, 'github.ref': ref,
                              'github.sha': 'a' * 40,
                              'github.event.pull_request.number || github.ref': pr or ref}
                    for expression, result in values.items():
                        value = value.replace('${{ ' + expression + ' }}', result)
                    self.assertNotIn('${{', value, 'test must resolve every concurrency expression')
                    return value
                for child in children:
                    text = (workflows / child).read_text()
                    group = re.search(r'^concurrency:\n  group: (.+)$', text, re.M)
                    if group is not None:
                        with self.subTest(caller=caller, child=child, ref=ref):
                            self.assertNotEqual(resolve(parent), resolve(group.group(1)),
                                                'reusable workflow can cancel its own caller')

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
                        env=os.environ | {'JOPE_RUNNER_LANE': lane, 'GITHUB_ENV': str(output),
                                          'HOME': str(Path(name)/'home')},
                        capture_output=True, text=True)
                    self.assertEqual(0, result.returncode, result.stderr)
                    # Persistent lane volume, or the runner-home fallback when it is unwritable.
                    self.assertIn(output.read_text(), {
                        f'CARGO_TARGET_DIR=/var/cache/jope/target/{lane}\n',
                        f'CARGO_TARGET_DIR={Path(name)}/home/.cache/jope/target/{lane}\n'})
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
            self.assertEqual(f'xbabe4-{lane},jope-mac-ci', result.stdout.strip())
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
        for lane in ['ci']+['ci'+str(n) for n in range(2,13)]:
            candidate=f'xbabe4-{lane}-20261023T000000Z'
            accepted=subprocess.run(['bash','-ec',script,'test','xbabe4',lane,candidate],cwd=ROOT,
                                    input=token,capture_output=True,text=True)
            self.assertEqual(0,accepted.returncode,(lane,accepted.stderr))
            self.assertEqual('64',accepted.stdout)
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

    def test_mac_qualification_is_main_dispatch_and_mac_pr_storage_only(self):
        env=self.env('runner-qualification','workflow_dispatch','refs/heads/main')
        for lane in ['ci']+['ci'+str(n) for n in range(2,13)]:
            with self.subTest(lane=lane):
                policy.authorize(dict(host='xbabe4',lane=lane,domain='pr'),env)
        for launch in [dict(domain='pr'),dict(host='xbabe1',lane='ci',domain='pr'),
                       dict(host='xbabe4',lane='ci13',domain='pr'),
                       dict(host='xbabe4',lane='ci',domain='main'),
                       dict(host='xbabe4',lane='ci',domain='guard')]:
            with self.subTest(launch=launch),self.assertRaises(ValueError):
                policy.authorize(launch,env)
        for key,value in [('GITHUB_EVENT_NAME','pull_request'),('GITHUB_EVENT_NAME','push'),
                          ('GITHUB_REF','refs/heads/feature'),('GITHUB_REF','refs/tags/v0.4.1'),
                          ('GITHUB_WORKFLOW_REF','neverhuman/JopeDime/.github/workflows/runner-qualification.yml@refs/heads/feature'),
                          ('GITHUB_REPOSITORY','foreign/repo'),('GITHUB_REPOSITORY_ID','1'),
                          ('GITHUB_SHA','0'*40),('GITHUB_SHA','HEAD'),('GITHUB_SHA','a'*39)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                policy.authorize(dict(host='xbabe4',lane='ci',domain='pr'),env|{key:value})

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



from _runner_case_mac import MacControllerTests
from _runner_case_trim import TrimTests
