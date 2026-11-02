from _runner_case_support import *


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
            active_lanes=[], reserved_gib=0, rosetta_enabled=True, kernel_qemu_enabled=False,
            git_emulation=dict(binary_sha256=cfg['git_emulation']['binary_sha256'],
                binary_bytes=cfg['git_emulation']['binary_bytes'], architecture='aarch64',
                uid=0, gid=0, mode='0555', volume=cfg['git_emulation']['volume'],
                version=cfg['git_emulation']['version'])), response=dict(runner=dict(id=7),
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
            active_lanes=['ci'],reserved_gib=64,rosetta_enabled=False,kernel_qemu_enabled=True,
            git_emulation=dict(binary_sha256='f'*64))
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

    def test_mac_capability_label_mismatch_refuses_before_mint(self):
        for label in [None,'jope-ci','jope-main-guard','foreign','jope-mac-ci\nINJECTED=1']:
            with self.subTest(label=label),self.scratch() as name:
                root=Path(name);values=self.fixture(root);values['policy']['runner_label']=label
                result=self.run_fixture(root,values)
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
        for name in ('mac-controller.sh','mac-guest.sh','mac-git-entrypoint.sh','mac-capacity.json','jope-mac-runner@.service','jope-mac-runner@.timer'):
            self.assertIn('deploy/runner/'+name,paths)
        for name in ('mac-controller.sh','mac-guest.sh','mac-git-entrypoint.sh'):
            result=subprocess.run(['bash','-n',str(ROOT/'deploy/runner'/name)],capture_output=True,text=True)
            self.assertEqual(0,result.returncode,result.stderr)
        self.assertIn('jope-main-guard',(ROOT/'.github/workflows/main-guard.yml').read_text())


    def mac_container(self, lane='ci'):
        import json
        cfg = json.loads((ROOT/'deploy/runner/mac-capacity.json').read_text())
        mounts = [dict(Type='volume', Name='jope-sdk', Destination='/opt/jope-sdk', RW=False),
                  dict(Type='volume', Name=cfg['git_emulation']['volume'],
                       Destination='/opt/jope-git-emulation', RW=False)]
        for kind, destination in [('work','/home/runner/work'), ('cargo','/var/cache/jope/cargo'),
                                  ('target','/var/cache/jope/target'), ('uv','/var/cache/jope/uv')]:
            mounts.append(dict(Type='volume', Name=f'jope-pr-{lane}-{kind}',
                               Destination=destination, RW=True))
        for name in ['job-started.sh','job-policy.py','mac-git-entrypoint.sh','mac-capacity.json']:
            path='/opt/jope-runner/'+name
            mounts.append(dict(Type='bind', Source=path, Destination=path, RW=False))
        mounts.append(dict(Type='bind', Source=f'/var/lib/jope-runner/{lane}-policy.json',
                           Destination='/opt/jope-runner/policy.json', RW=False))
        return cfg, dict(lane=lane, memory=16*1024**3, swap=16*1024**3, cpus=16000000000,
                         image=cfg['image_id'], readonly=True, pids=4096, caps=['ALL'],
                         security=['no-new-privileges'], mounts=mounts)

    def check_mac_container(self, cfg, facts):
        import json
        driver='source deploy/runner/mac-guest.sh; printf "%s" "$1" | jope_mac_guest_container_ok "$2"'
        return subprocess.run(['bash','-c',driver,'fixture',json.dumps(facts),json.dumps(cfg)],
                              cwd=ROOT,capture_output=True,text=True)

    def test_mac_git_mounts_are_closed_and_readonly_for_every_lane(self):
        import copy
        for lane in ['ci']+['ci'+str(n) for n in range(2,13)]:
            cfg, facts=self.mac_container(lane)
            self.assertEqual(0,self.check_mac_container(cfg,facts).returncode,lane)
        cfg, good=self.mac_container()
        for destination in ['/opt/jope-sdk','/opt/jope-git-emulation',
                            '/opt/jope-runner/mac-git-entrypoint.sh','/opt/jope-runner/mac-capacity.json']:
            facts=copy.deepcopy(good)
            next(m for m in facts['mounts'] if m['Destination']==destination)['RW']=True
            self.assertNotEqual(0,self.check_mac_container(cfg,facts).returncode,destination)
        cases=[]
        facts=copy.deepcopy(good);facts['mounts'].pop();cases.append(facts)
        facts=copy.deepcopy(good);facts['mounts'].append(dict(Type='bind',Source='/var/run/docker.sock',
            Destination='/var/run/docker.sock',RW=True));cases.append(facts)
        facts=copy.deepcopy(good);facts['mounts'][-1]=copy.deepcopy(facts['mounts'][-2]);cases.append(facts)
        facts=copy.deepcopy(good);next(m for m in facts['mounts'] if m.get('Name')=='jope-pr-ci-target')['Name']='jope-pr-ci2-target';cases.append(facts)
        facts=copy.deepcopy(good);next(m for m in facts['mounts'] if m['Destination']=='/opt/jope-git-emulation')['Name']='foreign-tool';cases.append(facts)
        for key,value in [('lane','ci13'),('memory',48*1024**3),('swap',0),('readonly',False),
                          ('pids',8192),('cpus',8000000000),('image','sha256:'+'f'*64)]:
            facts=copy.deepcopy(good);facts[key]=value;cases.append(facts)
        for facts in cases:
            self.assertNotEqual(0,self.check_mac_container(cfg,facts).returncode,facts)

    def test_git_prepare_refusal_preserves_opaque_stdin_and_creates_no_shims(self):
        import hashlib
        import json
        with self.scratch() as name:
            root=Path(name);tool=root/'public-tool';data=bytearray(64)
            data[:6]=b'\x7fELF\x02\x01';data[18:20]=(183).to_bytes(2,'little')
            tool.write_bytes(data);tool.chmod(0o700);link=root/'tool-link';link.symlink_to(tool)
            fake=b'FAKE-OPAQUE-STDIN-'+b'A'*2071+b'\n'
            driver=r'''
source deploy/runner/mac-git-entrypoint.sh
if jope_mac_git_prepare "$1" "$2" "$3"; then exit 81; fi
/usr/bin/python3 -I -c 'import sys,json,hashlib; b=sys.stdin.buffer.read();print(json.dumps(dict(size=len(b),sha256=hashlib.sha256(b).hexdigest())))'
'''
            for input_path,digest in [(tool,'0'*64),(link,hashlib.sha256(data).hexdigest()),
                                      (tool,'INVALID-DIGEST')]:
                destination=root/'must-not-exist'
                result=subprocess.run(['bash','-c',driver,'fixture',str(input_path),digest,str(destination)],
                    input=fake,capture_output=True,cwd=ROOT)
                self.assertEqual(0,result.returncode,result.stderr)
                self.assertEqual(dict(size=len(fake),sha256=hashlib.sha256(fake).hexdigest()),json.loads(result.stdout))
                self.assertFalse(destination.exists())
                self.assertNotIn(fake,result.stdout+result.stderr)

    def test_mac_sidecar_and_binary_facts_are_bound_before_acceptance(self):
        import json
        from unittest.mock import patch
        provenance=load('provenance_bind')
        cfg=json.loads((ROOT/'deploy/runner/mac-capacity.json').read_text())['git_emulation']
        expected=dict(binary_sha256=cfg['binary_sha256'],binary_bytes=cfg['binary_bytes'],
                      architecture='aarch64',uid=0,gid=0,mode='0555',volume=cfg['volume'],version=cfg['version'])
        with self.scratch() as name:
            root=Path(name)
            for filename in ['mac-git-entrypoint.sh','mac-capacity.json']:
                (root/filename).write_bytes((ROOT/'deploy/runner'/filename).read_bytes())
            with patch.object(provenance.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(expected),'')) as run:
                provenance.verify_mac_git(ROOT,root,{})
                self.assertEqual('/bin/bash',run.call_args.args[0][0])
                self.assertEqual('--inspect-tool',run.call_args.args[0][2])
            for key,value in [('binary_sha256','f'*64),('binary_bytes',0),('architecture','x86_64'),
                              ('uid',1001),('gid',1001),('mode','0755'),('volume','foreign'),('version','0.0.0')]:
                wrong=dict(expected);wrong[key]=value
                with patch.object(provenance.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(wrong),'')):
                    with self.assertRaises(ValueError):provenance.verify_mac_git(ROOT,root,{})
            (root/'mac-capacity.json').write_text('{}')
            with patch.object(provenance.subprocess,'run') as run:
                with self.assertRaises(ValueError):provenance.verify_mac_git(ROOT,root,{})
                run.assert_not_called()
