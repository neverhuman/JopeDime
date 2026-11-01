//! Real CLI, terminal PTY, MCP reconnect and engine crash acceptance.
//! The generated Python fixtures use only the standard library and real processes.

use std::{fs, path::PathBuf, process::Command};

pub(super) fn run() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let runtime_base = std::env::var_os("JOPE_INT02_RUNTIME_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".local/state/jope-int02")
        });
    fs::create_dir_all(&runtime_base).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("i-")
        .tempdir_in(&runtime_base)
        .unwrap();
    let out = std::env::var_os("JOPE_INT02_EVIDENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| scratch.path().join("evidence"));
    fs::create_dir_all(&out).unwrap();
    let driver = scratch.path().join("live_probe.py");
    fs::write(&driver, DRIVER).unwrap();
    fs::write(scratch.path().join("observe_http.py"), OBSERVER).unwrap();
    let binary = env!("CARGO_BIN_EXE_jopedime");
    let version = Command::new(binary)
        .args(["--version", "--json"])
        .output()
        .unwrap();
    assert!(version.status.success());
    let version: serde_json::Value = serde_json::from_slice(&version.stdout).unwrap();
    let source = std::env::var("JOPE_INT02_SOURCE_SHA")
        .unwrap_or_else(|_| version["source_sha40"].as_str().unwrap().to_owned());
    let mut command = Command::new("python3");
    command
        .arg(&driver)
        .arg("--binary")
        .arg(binary)
        .arg("--source")
        .arg(&source)
        .arg("--source-root")
        .arg(&root)
        .arg("--out")
        .arg(&out)
        .arg("--runtime-root")
        .arg(scratch.path().join("runtime"))
        .args([
            "--all-live",
            "--capture-sse",
            "--train-rows",
            "96",
            "--s2-evaluations",
            "2",
            "--wall-seconds",
            "300",
            "--assert-model-identity",
            "--assert-s2",
        ]);
    if std::env::var("INT02_ENGINE_BOUNDARY_CUTS").as_deref() == Ok("1") {
        command.arg("--boundary-matrix");
    }
    if std::env::var("INT02_SAME_STORE").as_deref() == Ok("1") {
        command.arg("--same-store");
    }
    let output = command
        .output()
        .expect("native INT-02 acceptance requires python3");
    fs::write(out.join("driver.stdout"), &output.stdout).unwrap();
    fs::write(out.join("driver.stderr"), &output.stderr).unwrap();
    if std::env::var("JOPE_INT02_EVIDENCE_DIR").is_ok() {
        // Retain owned per-run stores with the external qualification receipts.
        let path = scratch.keep();
        fs::write(
            out.join("runtime-location.txt"),
            path.to_string_lossy().as_bytes(),
        )
        .unwrap();
    }
    assert!(
        output.status.success(),
        "INT-02 live acceptance failed: {}\n{}\nreceipts: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        out.display()
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("report.json")).unwrap()).unwrap();
    for gate in [
        "projection_oracle_controls",
        "r4_qualified",
        "s2_qualified",
        "r1_coarse_native_pass",
    ] {
        assert_eq!(report[gate], true, "missing actual acceptance: {gate}");
    }
    eprintln!(
        "INT-02 R4/S2/R1 actual native acceptance PASS; receipts {}",
        out.display()
    );
}

const DRIVER: &str = r####"#!/usr/bin/env python3
"""Native acceptance probe. Runtime outputs stay outside the source checkout."""
import argparse, datetime, fcntl, hashlib, json, os, pty, select, signal
import struct, subprocess, termios, time, traceback, re, http.client, urllib.parse
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
commands = []; children = []; files = []; observers = []; recovery_homes = []
report = {'source': args.source, 'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'dataset': '21fbfe8b53dbcd60', 'rows': args.train_rows, 'selected_train_sha256': hashlib.sha256(selected_raw).hexdigest(), 'source_split': 'train',
          'train_sha256': hashlib.sha256(raw).hexdigest(), 'seed': 73,
          'cpu_nice': os.getpriority(os.PRIO_PROCESS, 0), 'load_before': os.getloadavg()[0],
          'binary_sha256': hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()}

def save(name, value):
    (out/name).write_text(json.dumps(value, indent=2)+'\n')

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
    log = (out/(name+'-engine.log')).open('w'); files.append(log)
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
        log=(out/f'mcp-{len(children)}-stderr.log').open('w');files.append(log)
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

def attach(home,rid,name):
    master,slave=pty.openpty()
    fcntl.ioctl(slave,termios.TIOCSWINSZ,struct.pack('HHHH',40,120,0,0))
    child=subprocess.Popen([args.binary,'attach',rid],stdin=slave,stdout=slave,stderr=slave,
        env={**base_env,'JOPEDIME_HOME':str(home),'TERM':'dumb','NO_COLOR':'1'})
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
    logpath=out/(label+'-engine.log');log=logpath.open('w');files.append(log)
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
    observer.p.stdin.close();observer.p.wait(timeout=5);stop_engine(home,child)
    seen={hook:[] for hook in hooks}
    for name,ordinal in re.findall(r'JOPE_STORE_BOUNDARY:(\w+):(\d+)',logpath.read_text()):seen[name].append(int(ordinal))
    assert all(seen.values()),'full engine matrix requires --features jope-service/failpoints; missing actual boundary traces'
    for hook,ordinals in seen.items():assert ordinals==list(range(1,max(ordinals)+1)),(hook,ordinals)
    expected=project(baseline);cases=[]
    save('boundary-baseline.json',{'source':args.source,'counts':{k:len(v) for k,v in seen.items()},'events':baseline,'snapshot':snapshot})
    for hook in hooks:
        for ordinal in seen[hook]:
            label=f'b-{hooks.index(hook)}-{ordinal}'
            h,d=start_engine(label);seed=children[-1];stop_engine(h,seed)
            child,logpath=traced_engine(h,label,{'JOPE_STORE_TRACE_BOUNDARIES':'1','JOPE_STORE_FAILPOINT':hook,'JOPE_STORE_FAILPOINT_OCCURRENCE':str(ordinal),'JOPE_STORE_FAILPOINT_PARK':'1'})
            request=None;deadline=time.monotonic()+25
            while True:
                text=logpath.read_text()
                if f'JOPE_STORE_CRASH:{hook}' in text.splitlines():break
                assert child.poll() is None,f'{label}: engine exited before selected boundary: {text}'
                if request is None and daemon_ready(h,child):
                    output=(out/(label+'-request.log')).open('w');files.append(output)
                    request=subprocess.Popen([args.binary,*spec_argv(d,'int02-boundary-run',2)],env={**base_env,'JOPEDIME_HOME':str(h)},stdout=output,stderr=output)
                    children.append(request)
                assert time.monotonic()<deadline,f'{label}: real boundary not reached'
                time.sleep(.01)
            assert f'JOPE_STORE_BOUNDARY:{hook}:{ordinal}' in text.splitlines()
            child.kill();signal_exit=child.wait(timeout=5);assert signal_exit==-9
            if request is not None:
                try:request.wait(timeout=5)
                except subprocess.TimeoutExpired:request.terminate();request.wait(timeout=5)
            recovery_homes.append(h)
            client=MCP(h)
            # Invoke an actual tool: stdio initialization alone may defer engine discovery.
            info=client.call('tools/call',{'name':'system_info','arguments':{}})
            assert not info.get('isError'),info
            inventory=run(h,['runs','list'])['runs'];assert len(inventory)<=1
            recovered_events=[];state='not-created';recovered=None
            if inventory:
                rid=inventory[0]['run_id'];recovered=run(h,['status',rid,'--json'])
                state=recovered['state'];assert state in ['paused','completed'],(label,state)
                assert run(h,['status',rid,'--json'])==recovered,'recovery was not idempotent'
                recovered_events=client.events(rid,0)['events']
                assert [int(event['sequence']) for event in recovered_events]==list(range(1,len(recovered_events)+1))
                assert project(recovered_events)==expected[:len(project(recovered_events))],label
            assert not list((h/'tmp').iterdir()),f'{label}: staging survived automatic recovery'
            # Recovery verifies every durable record's CAS references before serving.
            # Execute the authorized continuation as well as proving the recovered prefix.
            if not inventory:
                rid=run(h,spec_argv(d,'int02-boundary-run',2))['run_id']
            elif state=='paused':
                control=['control',rid,'--idempotency-key','int02-boundary-continue','continue']
                resumed=run(h,control)
                assert run(h,control)==resumed,'Continue retry changed its committed outcome'
                changed=out/(label+'-changed-control.json')
                changed.write_text(json.dumps({'action':'continue','expected_revision':'0'}))
                conflict=subprocess.run([args.binary,'control',rid,'--idempotency-key','int02-boundary-continue','--body',str(changed),'continue'],env={**base_env,'JOPEDIME_HOME':str(h)},text=True,capture_output=True,timeout=10)
                commands.append({'argv':['control',rid,'changed-body-negative'],'exit':conflict.returncode,'stderr':conflict.stderr})
                assert conflict.returncode==8,'changed Continue request did not return the conflict exit'
                record=json.loads((h/'daemon.json').read_text());address=urllib.parse.urlsplit(record['url'])
                connection=http.client.HTTPConnection('127.0.0.1',address.port,timeout=10)
                try:
                    connection.request('POST',f'/v1/runs/{rid}/control',body=changed.read_text(),headers={'Authorization':'Bearer '+record['token'],'Content-Type':'application/json','Idempotency-Key':'int02-boundary-continue'})
                    reply=connection.getresponse();problem=json.loads(reply.read())
                    assert reply.status==409 and problem['code']=='IDEMPOTENCY_CONFLICT','changed Continue API request did not return its typed conflict'
                    save(label+'-conflict.json',{'status':reply.status,'problem':problem,'cli_exit':conflict.returncode})
                finally:connection.close()
                # A null is part of command identity even when its semantic default is absent.
                connection=http.client.HTTPConnection('127.0.0.1',address.port,timeout=10)
                try:
                    connection.request('POST',f'/v1/runs/{rid}/control',body=json.dumps({'action':'continue','expected_revision':None}),headers={'Authorization':'Bearer '+record['token'],'Content-Type':'application/json','Idempotency-Key':'int02-boundary-continue'})
                    reply=connection.getresponse();problem=json.loads(reply.read())
                    assert reply.status==409 and problem['code']=='IDEMPOTENCY_CONFLICT','explicit null was lost from Continue identity'
                finally:connection.close()
            completed=wait_terminal(h,rid)
            final_events=client.events(rid,0)['events']
            assert completed['state']=='completed' and completed['best']==snapshot['best'],(label,completed)
            assert final_events[:len(recovered_events)]==recovered_events,'continuation changed durable prefix'
            assert [int(event['sequence']) for event in final_events]==list(range(1,len(final_events)+1))
            assert final_events[-1]['body']['nondeterminism']['count']=='0','continuation hit a resource limit'
            cases.append({'hook':hook,'ordinal':ordinal,'signal':signal_exit,'state':state,'events':len(recovered_events),'staging_clean':True,'checkpoint':recovered,'completed':completed,'continued_sequences':len(final_events),'reference_integrity_checked_on_open':True})
            save('engine-boundary-cases.json',cases)
            client.p.stdin.close();client.p.wait(timeout=5)
            subprocess.run([args.binary,'serve','--stop'],env={**base_env,'JOPEDIME_HOME':str(h)},capture_output=True,text=True,timeout=10,check=True)
    assert len(cases)==sum(map(len,seen.values())) and any(c['state']=='paused' for c in cases) and any(c['state']=='completed' for c in cases)
    normal_count=len(cases)
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
                engine.kill();assert engine.wait(timeout=5)==-9
                observer.p.stdin.close();observer.p.wait(timeout=5)
            engine,logpath=traced_engine(h,label+'-selected',{'JOPE_STORE_TRACE_BOUNDARIES':'1','JOPE_STORE_FAILPOINT':hook,'JOPE_STORE_FAILPOINT_OCCURRENCE':'1','JOPE_STORE_FAILPOINT_PARK':'1'})
            deadline=time.monotonic()+20
            while f'JOPE_STORE_CRASH:{hook}' not in logpath.read_text().splitlines():
                assert engine.poll() is None,'engine did not reach raw commit'
                assert time.monotonic()<deadline,'raw commit hook deadline'
                time.sleep(.01)
            assert f'JOPE_STORE_BOUNDARY:{hook}:1' in logpath.read_text().splitlines()
            engine.kill();signal_exit=engine.wait(timeout=5);assert signal_exit==-9
            recovery_homes.append(h);client=MCP(h)
            info=client.call('tools/call',{'name':'system_info','arguments':{}});assert not info.get('isError'),info
            inventory=run(h,['runs','list'])['runs']
            if phase=='initialization':
                assert not inventory
                d=run(h,['dataset','add',str(csv),'--name','int02-real-train','--idempotency-key','int02-register','--json'])['dataset_id']
                rid=run(h,spec_argv(d,'int02-boundary-run',2))['run_id']
                recovered=None;state='not-created';prefix=[];expected_best=snapshot['best']
            else:
                assert len(inventory)==1 and inventory[0]['run_id']==rid
                recovered=run(h,['status',rid,'--json']);assert recovered['state']=='paused'
                assert recovered==run(h,['status',rid,'--json']),'raw recovery was not idempotent'
                prefix=client.events(rid,0)['events'];assert prefix[:len(original_prefix)]==original_prefix
                assert project(prefix)==project(json.loads((out/'live-mcp-events.json').read_text()))[:len(project(prefix))]
                run(h,['control',rid,'--idempotency-key','int02-raw-continue','continue'])
                state='paused';expected_best=report['r4_final']['best']
            completed=wait_terminal(h,rid);events=client.events(rid,0)['events']
            assert completed['state']=='completed' and completed['best']==expected_best
            assert events[:len(prefix)]==prefix and [int(e['sequence']) for e in events]==list(range(1,len(events)+1))
            assert events[-1]['body']['nondeterminism']['count']=='0'
            assert not list((h/'tmp').iterdir())
            cases.append({'phase':phase,'hook':hook,'ordinal':1,'signal':signal_exit,'state':state,'events':len(prefix),'checkpoint':recovered,'completed':completed,'staging_clean':True,'continued_sequences':len(events),'reference_integrity_checked_on_open':True})
            save('engine-boundary-cases.json',cases)
            client.p.stdin.close();client.p.wait(timeout=5)
            subprocess.run([args.binary,'serve','--stop'],env={**base_env,'JOPEDIME_HOME':str(h)},capture_output=True,text=True,timeout=10,check=True)
    assert len(cases)==normal_count+4
    report['r1_run_boundary_cases']=normal_count;report['r1_initialization_and_recovery_cases']=4
    report['r1_engine_boundary_cases']=len(cases);report['r1_engine_boundary_hooks']=hooks+['before_schema_commit','after_schema_commit','before_recovery_commit','after_recovery_commit']
    assert {case['hook'] for case in cases}==set(report['r1_engine_boundary_hooks'])
    report['r1_all_internal_background_cuts_qualified']=True

try:
    control=[{'run_id':'run_a','time':'a','sequence':'1','kind':'completed','body':{'experiment_id':'e1','model_id':'m1','budget_used':{'wall_seconds':'1','evaluations':'2'},'warnings':[{'host':True,'message':'host'},{'host':False,'message':'semantic'}]}}]
    operational=json.loads(json.dumps(control));operational[0]['run_id']='run_b';operational[0]['time']='b'
    operational[0]['body']['budget_used']['wall_seconds']='99';operational[0]['body']['warnings'][0]['message']='different host'
    assert project(control)==project(operational),'approved operational exclusions changed projection'
    for path in [('sequence',),('body','model_id'),('body','experiment_id'),('body','budget_used','evaluations')]:
        changed=json.loads(json.dumps(control));node=changed[0]
        for key in path[:-1]:node=node[key]
        node[path[-1]]='different'
        assert project(control)!=project(changed),f'projection hid semantic change: {path}'
    report['projection_oracle_controls']=True
    version=run(runtime,['--version','--json']);report['version']=version
    assert version['source_sha40']==args.source, 'binary source pin missing/mismatched'
    if not args.s2_only and not args.r1_only:
        home,dataset=start_engine('live')
        if args.capture_sse:
            from observe_http import Observer
            observers.append(Observer(home))
        mcp=MCP(home)
        started=run(home,spec_argv(dataset,'int02-r4-live',32));rid=started['run_id']
        report['r4_start_state']=started['state']
        report['terminal_first']=attach(home,rid,'terminal-before-close')
        midway=run(home,['status',rid,'--json']);report['state_at_close']=midway['state']
        first=mcp.events(rid,0,2);cursor=int(first['next_after']);events=list(first['events'])
        final=wait_terminal(home,rid);report['r4_final']=final
        report['terminal_reattach']=attach(home,rid,'terminal-after-reattach')
        while cursor<int(final['event_watermark']):
            page=mcp.events(rid,cursor,2);assert int(page['next_after'])>cursor
            cursor=int(page['next_after']);events.extend(page['events'])
        cli=subprocess.run([args.binary,'events',rid,'--after','0'],env={**base_env,'JOPEDIME_HOME':str(home)},
            text=True,capture_output=True,timeout=15)
        commands.append({'argv':['events',rid,'--after','0'],'exit':cli.returncode,'stdout':cli.stdout,'stderr':cli.stderr})
        cli_events=[json.loads(x) for x in cli.stdout.splitlines() if x.strip()] if cli.returncode==0 else []
        save('live-mcp-events.json',events);save('live-cli-events.json',cli_events);save('mcp-frames.json',mcp.frames)
        report['mcp_sequences_contiguous']=[int(x['sequence']) for x in events]==list(range(1,cursor+1))
        report['mcp_equals_cli_events']=events==cli_events
        report['terminal_event_projection_observed']=False
        report['r4_qualified']=False
        if args.capture_sse:
            traces=observers[-1].streams
            save('terminal-sse-streams.json',traces)
            equal=[trace for trace in traces if trace['events']==events]
            screen=(out/'terminal-after-reattach.pty').read_text(errors='replace')
            champion=final.get('best') or {}
            rendered=champion.get('algorithm_id','__missing__') in screen and f"{champion.get('cv_value',float('nan')):.3f}" in screen
            report['terminal_sse_stream_count']=len(traces)
            report['terminal_event_projection_observed']=bool(equal)
            report['terminal_full_stream_events']=len(equal[-1]['events']) if equal else 0
            report['terminal_champion_render_matches_mcp_snapshot']=rendered
            report['r4_qualified']=bool(equal) and rendered and report['mcp_sequences_contiguous'] and report['mcp_equals_cli_events'] and report['state_at_close']=='running' and report['terminal_first']['exit']==report['terminal_reattach']['exit']==0
            assert report['r4_qualified'],'live terminal wire projection/reattach acceptance failed'

    if not args.r4_only and not args.r1_only:
        # Rerun twice in fresh engine homes with identical registered train bytes/budgets/seed.
        pairs=[]
        for index in [1,2]:
            if not args.same_store or index==1:h,d=start_engine('rerun'+str(index))
            key='int02-s2-distinct-'+str(index) if args.same_store else 'int02-s2-same-key'
            argv=spec_argv(d,key,args.s2_evaluations)
            plan_argv=[x for x in argv if x!='--detach']+['--dry-run']
            plan_argv[plan_argv.index('--idempotency-key')+1]+='-plan'
            plan=run(h,plan_argv);save(f's2-{index}-plan.json',plan)
            start=run(h,argv)
            snap=wait_terminal(h,start['run_id']);c=MCP(h)
            page=c.events(start['run_id'],0,2);ev=list(page['events']);cursor=int(page['next_after'])
            c.p.stdin.close();assert c.p.wait(timeout=5)==0,'MCP EOF did not close first observer'
            resumed=MCP(h)
            while cursor<int(snap['event_watermark']):
                page=resumed.events(start['run_id'],cursor,2)
                assert int(page['next_after'])>cursor,'MCP reconnect cursor did not advance'
                cursor=int(page['next_after']);ev.extend(page['events'])
            assert ev==resumed.events(start['run_id'],0)['events'],'MCP reconnect changed the committed journal'
            assert [int(event['sequence']) for event in ev]==list(range(1,len(ev)+1))
            save(f's2-{index}-reconnect-wire.json',{'before_disconnect':c.frames,'after_reconnect':resumed.frames})
            save(f's2-{index}-events.json',ev);save(f's2-{index}-snapshot.json',snap)
            pairs.append((snap,ev))
        assert pairs[0][0]['run_id']!=pairs[1][0]['run_id'],'must compare distinct logical runs, not an idempotent retry'
        report['distinct_run_ids']=True
        report['same_store']=args.same_store
        left,right=map(lambda pair:project(pair[1]),pairs)
        report['s2_event_counts']=[len(x[1]) for x in pairs]
        report['s2_projection_equal_without_run_id_time']=left==right
        report['s2_best_equal']=pairs[0][0].get('best')==pairs[1][0].get('best')
        report['s2_nondeterminism_counts']=[int(pair[1][-1]['body']['nondeterminism']['count']) for pair in pairs]
        report['s2_qualified']=left==right and report['s2_best_equal'] and report['s2_nondeterminism_counts']==[0,0]
        if args.assert_model_identity:
            assert report['s2_best_equal'] and report['s2_nondeterminism_counts']==[0,0], 'canonical model identity regression'
            first_plan=json.loads((out/'s2-1-plan.json').read_text());second_plan=json.loads((out/'s2-2-plan.json').read_text())
            assert first_plan['plan_digest']==second_plan['plan_digest'], 'different plans cannot qualify model determinism'
            report['canonical_model_identity_pass']=True
        save('s2-projections.json',{'first':left,'second':right})
        report['s2_mcp_process_reconnects']=2
        report['projection_ruling']='rulings_t1900:R2'
        if args.assert_s2:assert report['s2_qualified'],'R2 S2 projection/champion mismatch'
    if args.r1_only or args.all_live:
        h,d=start_engine('crash')
        engine=children[-1]
        started=run(h,spec_argv(d,'int02-r1-native',32));rid=started['run_id']
        deadline=time.monotonic()+20
        while True:
            before=run(h,['status',rid,'--json'])
            assert before['state']=='running','run ended before selected live cut'
            if before.get('best') is not None and int(before['event_watermark'])>=5:break
            assert time.monotonic()<deadline,'no committed first model before live cut'
            time.sleep(.02)
        client=MCP(h);prefix=client.events(rid,0)['events']
        engine.kill();signal_exit=engine.wait(timeout=5)
        assert signal_exit==-9,'must be actual own background engine SIGKILL'
        client.p.terminate();client.p.wait(timeout=5)
        recovery_homes.append(h)
        report['r1_signal_exit']=signal_exit
        save('r1-before.json',before);save('r1-prefix.json',prefix);save('report-intermediate.json',report)
        report['r1_next_interactive_attach']=attach(h,rid,'terminal-next-use')
        assert report['r1_next_interactive_attach']['exit']==0,'interactive next-use failed to recover/start engine'
        recovered=run(h,['status',rid,'--json'])
        repeated=run(h,['status',rid,'--json'])
        assert recovered['state']=='paused' and recovered.get('best') is not None
        assert int(recovered['revision'])>=int(before['revision'])+1
        assert recovered==repeated,'repeat next-use changed recovery'
        resumed_client=MCP(h);after=resumed_client.events(rid,0)['events']
        assert after[:len(prefix)]==prefix,'committed public prefix changed after engine crash'
        assert [int(event['sequence']) for event in after]==list(range(1,len(after)+1))
        save('r1-before.json',before);save('r1-recovered.json',recovered)
        save('r1-prefix.json',prefix);save('r1-after-events.json',after)
        report['r1_coarse_native_pass']=True
        report['r1_signal_exit']=signal_exit
        report['r1_prefix_events']=len(prefix)
        report['r1_recovered_events']=len(after)
        report['r1_recovered_state']=recovered['state']
        report['r1_repeat_next_use_idempotent']=True
        report['r1_all_internal_background_cuts_qualified']=False
    if args.boundary_matrix:engine_boundary_matrix()
except Exception as error:
    report['error']=str(error);report['traceback']=traceback.format_exc()
finally:
    # Stop only processes started by this probe. No foreign PID or GPU work is touched.
    cleanup=[]
    for home in recovery_homes:
        try:
            stopped=subprocess.run([args.binary,'serve','--stop'],env={**base_env,'JOPEDIME_HOME':str(home)},text=True,capture_output=True,timeout=10)
            commands.append({'argv':['serve','--stop'],'home':str(home),'exit':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr})
        except Exception as error:cleanup.append(str(error))
    for observer in observers:
        try:observer.close()
        except Exception as error:cleanup.append(str(error))
    for child in reversed(children):
        if child.poll() is None:child.terminate()
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    report['cleanup_errors']=cleanup
    report['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['load_after']=os.getloadavg()[0];save('commands.json',commands);save('report.json',report)
    for f in files:f.close()
print(json.dumps({k:v for k,v in report.items() if k not in ['traceback','r4_final']}))
raise SystemExit(1 if 'error' in report else 0)
"####;

const OBSERVER: &str = r####""""Transparent loopback observer; retain response events, never auth headers."""
import http.client
import http.server
import json
import os
import threading
import urllib.parse
from pathlib import Path

class Observer:
    def __init__(self, home):
        self.home = Path(home)
        record = json.loads((self.home/'daemon.json').read_text())
        address = urllib.parse.urlsplit(record['url'])
        assert address.hostname == '127.0.0.1'
        upstream_port = address.port
        self.streams = []
        owner = self
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def relay(self):
                connection = http.client.HTTPConnection('127.0.0.1', upstream_port, timeout=90)
                headers = {k:v for k,v in self.headers.items() if k.lower() not in ['host','connection']}
                headers['Host'] = f'127.0.0.1:{upstream_port}'
                body = self.rfile.read(int(self.headers.get('Content-Length','0')))
                connection.request(self.command,self.path,body or None,headers)
                response = connection.getresponse()
                sse = 'text/event-stream' in response.getheader('Content-Type','')
                self.send_response(response.status)
                for key,value in response.getheaders():
                    if key.lower() not in ['content-length','transfer-encoding','connection']:
                        self.send_header(key,value)
                self.send_header('Connection','close')
                self.close_connection = True
                if not sse:
                    data=response.read();self.send_header('Content-Length',str(len(data)))
                    self.end_headers();self.wfile.write(data);connection.close();return
                self.send_header('Transfer-Encoding','chunked');self.end_headers()
                trace={'path':self.path,'after':self.headers.get('Last-Event-ID'),
                       'events':[],'complete':False,'client_closed':False}
                owner.streams.append(trace)
                frame=[]
                try:
                    while True:
                        line=response.readline()
                        if not line:break
                        self.wfile.write(f'{len(line):x}\r\n'.encode()+line+b'\r\n');self.wfile.flush()
                        if line.strip():
                            if line.startswith(b'data:'):frame.append(line[5:].strip())
                        elif frame:
                            value=json.loads(b'\n'.join(frame));frame=[]
                            if 'sequence' in value:trace['events'].append(value)
                    self.wfile.write(b'0\r\n\r\n');self.wfile.flush();trace['complete']=True
                except (BrokenPipeError,ConnectionResetError):
                    trace['client_closed']=True
                finally:
                    connection.close()
            do_GET=relay
            do_POST=relay
            do_DELETE=relay
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        # Only this isolated engine's private descriptor changes. Upstream bearer
        # remains in memory and the private daemon file, never in receipts.
        record['url']=f'http://127.0.0.1:{self.server.server_port}'
        temporary=self.home/'observer-descriptor.tmp'
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:json.dump(record,stream)
        os.replace(temporary,self.home/'daemon.json')
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)
"####;
