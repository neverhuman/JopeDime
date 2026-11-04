#!/usr/bin/env python3
"""Native acceptance probe. Runtime outputs stay outside the source checkout."""
import argparse, datetime, fcntl, hashlib, json, os, pty, select, signal
import struct, subprocess, termios, time, traceback, re, http.client, urllib.parse, resource
from pathlib import Path

a = argparse.ArgumentParser()
a.add_argument('--binary', required=True)
a.add_argument('--source', required=True)
a.add_argument('--source-root', required=True)
a.add_argument('--out', required=True)
a.add_argument('--runtime-root', required=True)
a.add_argument('--s2-only', action='store_true')
a.add_argument('--r4-only', action='store_true')
a.add_argument('--r1-only', action='store_true')
a.add_argument('--assert-model-identity', action='store_true')
a.add_argument('--same-store', action='store_true')
a.add_argument('--assert-s2', action='store_true')
a.add_argument('--all-live', action='store_true')
a.add_argument('--boundary-matrix', action='store_true')
a.add_argument('--capture-sse', action='store_true')
a.add_argument('--train-rows', type=int, default=1000)
a.add_argument('--s2-evaluations', type=int, default=4)
a.add_argument('--wall-seconds', type=int, default=30)
args = a.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
runtime = Path(args.runtime_root); runtime.mkdir(parents=True, exist_ok=True)
assert '/tmp/' not in str(runtime.resolve())
base_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '4'}
base_env.pop('JOPEDIME_ENDPOINT', None)
for key in ['JOPE_STORE_FAILPOINT','JOPE_STORE_FAILPOINT_OCCURRENCE','JOPE_STORE_FAILPOINT_PARK','JOPE_STORE_TRACE_BOUNDARIES']:
    base_env.pop(key,None)
source = Path(args.source_root)
raw = (source/'test-support/datasets/v1/d2-regression/train.csv').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'd9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931'
csv = runtime/'train.csv'
selected_raw=b''.join(raw.splitlines(keepends=True)[:args.train_rows])
csv.write_bytes((','.join([f'x{i}' for i in range(21)]+['target'])+'\n').encode()+selected_raw)
commands = []; children = []; observers = []; recovery_homes = []
report = {'source': args.source, 'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'dataset': '21fbfe8b53dbcd60', 'rows': args.train_rows, 'selected_train_sha256': hashlib.sha256(selected_raw).hexdigest(), 'source_split': 'train',
          'train_sha256': hashlib.sha256(raw).hexdigest(), 'seed': 73,
          'cpu_nice': os.getpriority(os.PRIO_PROCESS, 0), 'load_before': os.getloadavg()[0],
          'open_file_limit': list(resource.getrlimit(resource.RLIMIT_NOFILE)),
          'parent_open_files_start': len(os.listdir('/dev/fd')),
          'binary_sha256': hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()}

def acceptance_passes(result, required_gates):
    return ('error' not in result
            and all(result.get(gate) is True for gate in required_gates)
            and result.get('cleanup_errors', []) == [])

def required_gates(args):
    gates={'projection_oracle_controls'}
    if not args.s2_only and not args.r1_only:gates.add('r4_qualified')
    if not args.r4_only and not args.r1_only:gates.add('s2_qualified')
    if args.r1_only or args.all_live:gates.add('r1_coarse_native_pass')
    if args.boundary_matrix:gates.add('r1_all_internal_background_cuts_qualified')
    return gates

_all_live_gates = {
    'projection_oracle_controls': True,
    'r4_qualified': True,
    's2_qualified': True,
    'r1_coarse_native_pass': True,
}
qualification_controls = {
    'empty_rejected': not acceptance_passes({}, {'projection_oracle_controls'}),
    'missing_gate_rejected': not acceptance_passes(
        {'projection_oracle_controls': True, 'cleanup_errors': []},
        {'projection_oracle_controls', 'r4_qualified'}),
    'false_gate_rejected': not acceptance_passes(
        {**_all_live_gates, 's2_qualified': False, 'cleanup_errors': []}, _all_live_gates),
    'reported_error_rejected': not acceptance_passes(
        {**_all_live_gates, 'error': 'controlled failure', 'cleanup_errors': []}, _all_live_gates),
    'cleanup_failure_rejected': not acceptance_passes(
        {**_all_live_gates, 'cleanup_errors': ['controlled cleanup failure']}, _all_live_gates),
    'complete_pass_accepted': acceptance_passes(
        {**_all_live_gates, 'cleanup_errors': []}, _all_live_gates),
    'all_live_requires_all_gates': required_gates(argparse.Namespace(
        s2_only=False, r4_only=False, r1_only=False, all_live=True,
        boundary_matrix=False)) == set(_all_live_gates),
    'r4_only_requires_r4': required_gates(argparse.Namespace(
        s2_only=False, r4_only=True, r1_only=False, all_live=False,
        boundary_matrix=False)) == {'projection_oracle_controls', 'r4_qualified'},
    's2_only_requires_s2': required_gates(argparse.Namespace(
        s2_only=True, r4_only=False, r1_only=False, all_live=False,
        boundary_matrix=False)) == {'projection_oracle_controls', 's2_qualified'},
    'r1_only_requires_r1': required_gates(argparse.Namespace(
        s2_only=False, r4_only=False, r1_only=True, all_live=False,
        boundary_matrix=False)) == {'projection_oracle_controls', 'r1_coarse_native_pass'},
}
assert all(qualification_controls.values()), 'acceptance predicate negative/positive control failed'
report['qualification_controls'] = qualification_controls

def save(name, value):
    (out/name).write_text(json.dumps(value, indent=2)+'\n')

def check_descriptors():
    current=len(os.listdir('/dev/fd'))
    report['parent_open_files_max']=max(report.get('parent_open_files_max',0),current)
    assert current<=report['parent_open_files_start']+16,'acceptance driver leaked parent descriptors'

def run(home, argv, timeout=45):
    start = time.monotonic()
    r = subprocess.run([args.binary, *argv], env={**base_env, 'JOPEDIME_HOME': str(home)},
                       text=True, capture_output=True, timeout=timeout)
    commands.append({'argv': argv, 'home': str(home), 'exit': r.returncode,
                     'seconds': time.monotonic()-start, 'stdout': r.stdout, 'stderr': r.stderr})
    assert r.returncode == 0, f'{argv[0]} exit {r.returncode}: {r.stderr}'
    return json.loads(r.stdout)

def project(events):
    result=[]
    for original in events:
        event=json.loads(json.dumps(original))
        if event.get('kind')=='warning' and event.get('body',{}).get('host') is True:
            continue
        event.pop('run_id',None);event.pop('time',None)
        body=event.get('body',{})
        if isinstance(body.get('budget_used'),dict):body['budget_used'].pop('wall_seconds',None)
        if isinstance(body.get('warnings'),list):body['warnings']=[w for w in body['warnings'] if w.get('host') is not True]
        result.append(event)
    return result

def start_engine(name):
    home = runtime/name; home.mkdir(exist_ok=False)
    with (out/(name+'-engine.log')).open('w') as log:
        child = subprocess.Popen([args.binary, 'serve', '--idle-exit', '120s'],
            env={**base_env, 'JOPEDIME_HOME': str(home)}, stdout=log, stderr=log)
    children.append(child)
    deadline = time.monotonic()+15
    while not (home/'daemon.json').exists():
        assert child.poll() is None, 'own engine exited during startup'
        assert time.monotonic() < deadline, 'engine startup deadline'
        time.sleep(.02)
    record = json.loads((home/'daemon.json').read_text())
    assert record['pid'] == child.pid
    dataset = run(home, ['dataset','add',str(csv),'--name','int02-real-train',
                        '--idempotency-key','int02-register','--json'])
    return home, dataset['dataset_id']

def spec_argv(dataset, key, evaluations):
    return ['run','--json','--detach','--dataset',dataset,'--target','target',
      '--task','regression','--split',json.dumps({'kind':'random','holdout_fraction':.2,
      'group_column':None,'time_column':None}),'--objective','rmse','--effort','fast',
      '--budget',json.dumps({'evaluations':str(evaluations),'wall_seconds':str(args.wall_seconds)}),
      '--landmarks','none','--guidance','auto','--language','rust','--require-all-languages',
      '--device','cpu','--seed','73','--idempotency-key',key]

def wait_terminal(home, rid):
    deadline=time.monotonic()+65
    while True:
        snapshot=run(home,['status',rid,'--json'])
        if snapshot['state'] in ['completed','failed','cancelled']:
            return snapshot
        assert time.monotonic()<deadline, 'run terminal deadline'
        time.sleep(.1)

class MCP:
    def __init__(self,home):
        with (out/f'mcp-{len(children)}-stderr.log').open('w') as log:
            self.p=subprocess.Popen([args.binary,'mcp'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=log,env={**base_env,'JOPEDIME_HOME':str(home)},text=True,bufsize=1)
        children.append(self.p);self.i=0;self.frames=[]
        self.call('initialize',{'protocolVersion':'2025-03-26','capabilities':{},
            'clientInfo':{'name':'int02-closeout-native','version':'1'}})
        self.p.stdin.write(json.dumps({'jsonrpc':'2.0','method':'notifications/initialized'})+'\n');self.p.stdin.flush()
    def call(self,method,params):
        self.i+=1;request={'jsonrpc':'2.0','id':self.i,'method':method,'params':params}
        self.frames.append({'direction':'request','message':request})
        self.p.stdin.write(json.dumps(request)+'\n');self.p.stdin.flush()
        deadline=time.monotonic()+20
        while True:
            ready,_,_=select.select([self.p.stdout],[],[],max(0,deadline-time.monotonic()))
            assert ready,'MCP response timeout'
            line=self.p.stdout.readline();assert line,'MCP EOF'
            value=json.loads(line);self.frames.append({'direction':'response','message':value})
            if value.get('id')==self.i:
                assert 'error' not in value,value
                return value['result']
    def events(self,rid,after,limit=256):
        value=self.call('tools/call',{'name':'run_events','arguments':{
            'run_id':rid,'after':str(after),'limit':limit}})
        assert not value.get('isError'),value
        if 'structuredContent' in value:return value['structuredContent']
        return json.loads(''.join(x['text'] for x in value['content'] if x['type']=='text'))
    def close(self):
        self.p.stdin.close()
        try:return self.p.wait(timeout=5)
        finally:self.p.stdout.close()

def attach(home,rid,name):
    master,slave=pty.openpty()
    fcntl.ioctl(slave,termios.TIOCSWINSZ,struct.pack('HHHH',40,120,0,0))
    child=subprocess.Popen([args.binary,'attach',rid],stdin=slave,stdout=slave,stderr=slave,
        env={**base_env,'JOPEDIME_HOME':str(home),'TERM':'dumb','NO_COLOR':'1','JAIN_ACCEPT_TERMS':'1','JAIN_ALLOW_LOCAL_FILES':'1'})
    children.append(child);os.close(slave)
    text=bytearray();deadline=time.monotonic()+(.3 if name=='terminal-before-close' and args.capture_sse else 2)
    while time.monotonic()<deadline:
        ready,_,_=select.select([master],[],[],.05)
        if ready:
            try:text.extend(os.read(master,65536))
            except OSError:break
        if child.poll() is not None:break
    os.write(master,b'\\quit\n');deadline=time.monotonic()+5
    while child.poll() is None and time.monotonic()<deadline:
        ready,_,_=select.select([master],[],[],.05)
        if ready:
            try:text.extend(os.read(master,65536))
            except OSError:break
    if child.poll() is None:child.terminate()
    code=child.wait(timeout=5);os.close(master)
    (out/(name+'.pty')).write_bytes(text)
    return {'exit':code,'bytes':len(text),'has_background':b'background' in text,
            'has_run_id':rid.encode() in text,'closed_without_engine_stop':True}

def stop_engine(home,child):
    result=subprocess.run([args.binary,'serve','--stop'],env={**base_env,'JOPEDIME_HOME':str(home)},text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert child.wait(timeout=10)==0,'own seed engine did not stop cleanly'

def traced_engine(home,label,extra):
    logpath=out/(label+'-engine.log')
    with logpath.open('w') as log:
        child=subprocess.Popen([args.binary,'serve','--idle-exit','120s'],env={**base_env,'JOPEDIME_HOME':str(home),**extra},stdout=log,stderr=log)
    children.append(child)
    return child,logpath

def daemon_ready(home,child):
    connection=None
    try:
        record=json.loads((home/'daemon.json').read_text())
        if record['pid']!=child.pid:return False
        address=urllib.parse.urlsplit(record['url']);assert address.hostname=='127.0.0.1'
        connection=http.client.HTTPConnection('127.0.0.1',address.port,timeout=.2)
        connection.request('GET','/v1/system',headers={'Authorization':'Bearer '+record['token']})
        response=connection.getresponse();body=json.loads(response.read())
        return response.status==200 and body['api_version'].startswith('1.')
    except (OSError,ValueError,KeyError,http.client.HTTPException):return False
    finally:
        if connection is not None:connection.close()

def wait_ready(home,child):
    deadline=time.monotonic()+15
    while not daemon_ready(home,child):
        assert child.poll() is None,'traced engine exited before startup'
        assert time.monotonic()<deadline,'traced engine startup deadline'
        time.sleep(.01)

def engine_boundary_matrix():
    hooks=['after_write','after_file_sync','after_hash_verify','after_rename','after_directory_sync','before_commit','after_commit']
    home,dataset=start_engine('matrix-baseline');seed=children[-1];stop_engine(home,seed)
    child,logpath=traced_engine(home,'matrix-baseline',{'JOPE_STORE_TRACE_BOUNDARIES':'1'})
    wait_ready(home,child)
    started=run(home,spec_argv(dataset,'int02-boundary-run',2));snapshot=wait_terminal(home,started['run_id'])
    observer=MCP(home);baseline=observer.events(started['run_id'],0)['events']
    assert snapshot['state']=='completed' and baseline[-1]['body']['nondeterminism']['count']=='0'
    observer.close();stop_engine(home,child)
    seen={hook:[] for hook in hooks}
    for name,ordinal in re.findall(r'JOPE_STORE_BOUNDARY:(\w+):(\d+)',logpath.read_text()):seen[name].append(int(ordinal))
    assert all(seen.values()),'full engine matrix requires --features jope-service/failpoints; missing actual boundary traces'
    for hook,ordinals in seen.items():assert ordinals==list(range(1,max(ordinals)+1)),(hook,ordinals)
    expected=project(baseline);cases=[]
    save('boundary-baseline.json',{'source':args.source,'counts':{k:len(v) for k,v in seen.items()},'events':baseline,'snapshot':snapshot})
    # Schema creation and recovery metadata use their own raw redb commits.
    # Exercise both sides with actual fresh/background engines, beyond the run trace.
    for phase in ['initialization','recovery']:
        for side in ['before','after']:
            hook=side+('_schema_commit' if phase=='initialization' else '_recovery_commit')
            label=f'x-{phase[:1]}-{side[:1]}'
            original_prefix=[];rid=None
            if phase=='initialization':
                h=runtime/label;h.mkdir()
            else:
                h,d=start_engine(label);engine=children[-1]
                rid=run(h,spec_argv(d,'int02-recovery-run',32))['run_id']
                deadline=time.monotonic()+20
                while True:
                    before=run(h,['status',rid,'--json'])
                    assert before['state']=='running','recovery seed finished before cut'
                    if before.get('best') is not None and int(before['event_watermark'])>=5:break
                    assert time.monotonic()<deadline,'no recovery seed model'
                    time.sleep(.02)
                observer=MCP(h);original_prefix=observer.events(rid,0)['events']
