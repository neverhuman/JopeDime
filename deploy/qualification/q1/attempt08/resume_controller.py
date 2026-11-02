from pathlib import Path
import datetime, fcntl, hashlib, importlib.util, json, os, signal, shutil, stat, subprocess, sys, threading, time
from contextlib import contextmanager

E = Path('/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t0240/attempt-08')
ROOT = Path('/home/ubuntu/JopeDime/.agent/worktrees/qlt-02-t0545')
H = Path('/home/ubuntu/jope_plan/evidence/QLT-04/gov03-t0115')
HANDOFF = Path('/home/ubuntu/jope_plan/evidence/QLT-04/api01-t0250/handoff.ready.json')
ORIGINAL = H / 'run/receipts.jsonl'
PLAN = Path('/home/ubuntu/jope_plan/evidence/QLT-04/api01-t0710/frozen/ccaf14b210b3b55a350218e60f6b49ef76524cf3/metadata/fa07e1336c8fc7c04994faadf8f6efabded0a195/plan.json')
RECOUNT = Path('/home/ubuntu/jope_plan/evidence/QLT-04/gov03-t0115/PARTIAL-INDEPENDENT-RECOUNT.json')
AUTHORITY = Path('/home/ubuntu/nudges/lanes_t0240.md')
FRESH_APPROVAL = Path('/home/ubuntu/nudges/lanes_t0415_lock.md')
T0430_AUTHORITY = Path('/home/ubuntu/nudges/lanes_t0430_rulings.md')
GOV04_REVIEW = Path('/home/ubuntu/jope_plan/master/audit/ready/QLT-04-attempt08-gov04-review-t1548.json')
T0430_SHA = '9f37306a25cafb0d213625d999c1d255bd8139425794ab2e5e82db15583da038'
JRB_SOURCE_SHA = 'bc54a0ba6c42a87a5d1179540fc62d30b6cac90551067da42d2d1c68861155cd'
JRB_SOURCE_COMMIT = '31207ade1cb12da1df2036d1adbbfcc50c3d8dce'
JRB_PLAN_SHA = '96310d6916d79c411a0fc4c53d2824ac678140f95b22520d6fd9fc3b0936abb3'
JRB_WAVE = 'JRB01T0430_96310d69'
JRB_SOURCE_LOADER_SHA = '5a070d46781efc42b5f92fec3e274a5af38363b5772d0085474540810e6ea253'
JRB_RESOURCE_QUALIFICATION_SHA = '796f2a80a537ae06b61f52f41c3770743775c43954af56e18fdec1708571df32'
JRB_BASELINE_SHA = '2f5fd89053b604e9b3edeb3d06ac1c2f2ee0e6de8ec884b5dfffed22444566ce'
JRB_TEL_CONTRACT = Path('/home/ubuntu/jope_plan/evidence/JRB-01/sol-jrb01-t1528/TEL-OWNER-EXTENSION-CONTRACT.json')
JRB_TEL_CONTRACT_SHA = '52ee8d748cbb0535a1e835ee876303ccedd1d00e1ad8ee12369e9bcb514ed62c'
JRB_TEL_ADMISSION = Path('/home/ubuntu/jope_plan/evidence/JRB-01/tel-t1547/ADMISSION-1548.json')
JRB_TEL_ADMISSION_SHA = '224d32eb02c46a0e959b4a51985635bf1769f646124f548079b7588091d83721'
JRB_TEL_CURRENT_CLAIM = Path('/home/ubuntu/jope_plan/evidence/JRB-01/tel-t1547/CURRENT-CLAIM-1548.json')
JRB_TEL_CURRENT_CLAIM_SHA = '22fcecffb4d7344850c3f8c646e249a266f5cbdbdc63dfef2e9205572b441570'
OWNER_EXTENSION = Path('/home/ubuntu/jope_plan/evidence/JRB-01/jepson-deadline-extension-t188/JEPSON-DEADLINE-EXTENSION.json')
OWNER_EXTENSION_SHA = '48d45b3a0188d07fac037a2f30cfd216d9f8693006fc137654cea070ac7acd31'
OWNER_EXTENSION_WINDOW = 22658
OWNER_RESUME_BEFORE = 1793640600  # 2026-11-02T17:30:00Z
OWNER_FINAL_DEADLINE = OWNER_EXTENSION.with_name('FINAL-DEADLINE.json')
OWNER_TIMEOUT = Path('/home/ubuntu/jope_plan/evidence/JRB-01/sol-jrb01-t1528/OUTER-TIMEOUT.json')
RUNTIME_SOURCE_COMMIT = 'fa07e1336c8fc7c04994faadf8f6efabded0a195'
RUNTIME_BINDING_RECEIPT = Path('/home/ubuntu/jope_plan/master/audit/ready/QLT-04-runtime-source-binding-fa07.json')
RUNTIME_BINDING_SHA = '1190a5135e88c461910057075991b396457bb94dd9342985f520ab75813f18e3'
JRB_DIR = Path('/home/ubuntu/jope_plan/evidence/JRB-01/sol-jrb01-t1458')
JRB_CLAIM = JRB_DIR / 'CONTROLLER-CLAIM.json'
JRB_CONTROLLER_CLAIM_SHA = 'd1cf7064aaa6d22a8998581df1337a18330cc822aa89e0e6a16f99a8be835732'
JRB_CONTROLLER_PID = 1968683
JRB_FINAL_DEADLINE = Path('/home/ubuntu/jope_plan/evidence/JRB-01/sol-jrb01-t1528/FINAL-DEADLINE.json')
JRB_FINAL_DEADLINE_SHA = '39ad841c0fe652f0571dbcd9cc2fd3756cc55b3ccdff4435a6de2bfe3f3749c7'
JRB_LEDGER = JRB_DIR / 'resume-ledger.jsonl'
JRB_STUDY = Path('/home/ubuntu/jope-data/jrb-01/sol-jrb01-study1')
JRB_EXECUTION = JRB_STUDY / 'execution.json'
TEL_ADMISSION_OBJ=json.loads(JRB_TEL_ADMISSION.read_text())
JRB_CONTROLLER_CLAIM_OBJ=json.loads(JRB_CLAIM.read_text())
JRB_FINAL_DEADLINE_OBJ=json.loads(JRB_FINAL_DEADLINE.read_text())
TEL_CURRENT_CLAIM_OBJ=json.loads(JRB_TEL_CURRENT_CLAIM.read_text())
assert TEL_ADMISSION_OBJ.get('schema') == 'jope.jrb01.tel-readmission/v1' and TEL_ADMISSION_OBJ.get('decision') == 'ADMISSION_PASS'
assert TEL_ADMISSION_OBJ.get('source_binding',{}).get('controller_sha256') == JRB_SOURCE_SHA
assert TEL_ADMISSION_OBJ.get('source_binding',{}).get('service_head') == JRB_SOURCE_COMMIT
assert TEL_ADMISSION_OBJ.get('launch_contract',{}).get('controller_claim_path') == str(JRB_CLAIM)
assert TEL_ADMISSION_OBJ.get('launch_contract',{}).get('JOPE_JRB_READMISSION') == str(JRB_TEL_ADMISSION)
assert TEL_CURRENT_CLAIM_OBJ.get('schema') == 'jope.jrb01.tel-current-claim/v1' and TEL_CURRENT_CLAIM_OBJ.get('status') == 'FRESH_TEL_EXTENSION_ADMISSION_PASS_LAUNCH_HANDOFF'
assert TEL_CURRENT_CLAIM_OBJ.get('admission_path') == str(JRB_TEL_ADMISSION) and TEL_CURRENT_CLAIM_OBJ.get('admission_sha256') == JRB_TEL_ADMISSION_SHA
assert TEL_CURRENT_CLAIM_OBJ.get('source_head') == JRB_SOURCE_COMMIT and TEL_CURRENT_CLAIM_OBJ.get('source_merge') == 'c2e5cf45c97f081d260b5cf28e30f9e48fa7011d'
assert TEL_CURRENT_CLAIM_OBJ.get('resumer_sha256') == JRB_SOURCE_SHA and TEL_CURRENT_CLAIM_OBJ.get('process_started') is False and TEL_CURRENT_CLAIM_OBJ.get('fifo_ticket_created') is False
assert TEL_CURRENT_CLAIM_OBJ.get('owner_extension_sha256') == OWNER_EXTENSION_SHA and TEL_CURRENT_CLAIM_OBJ.get('owner_extension_window_seconds') == OWNER_EXTENSION_WINDOW
assert JRB_CONTROLLER_CLAIM_OBJ.get('pid') == JRB_CONTROLLER_PID and JRB_CONTROLLER_CLAIM_OBJ.get('source_sha256') == JRB_SOURCE_SHA and JRB_CONTROLLER_CLAIM_OBJ.get('source_head') == JRB_SOURCE_COMMIT
assert JRB_CONTROLLER_CLAIM_OBJ.get('readmission_sha256') == JRB_TEL_ADMISSION_SHA and JRB_CONTROLLER_CLAIM_OBJ.get('owner_extension_sha256') == OWNER_EXTENSION_SHA
assert JRB_FINAL_DEADLINE_OBJ.get('controller_pid') == JRB_CONTROLLER_PID and JRB_FINAL_DEADLINE_OBJ.get('resumer_sha256') == JRB_SOURCE_SHA
assert JRB_FINAL_DEADLINE_OBJ.get('TEL_admission_sha256') == JRB_TEL_ADMISSION_SHA and JRB_FINAL_DEADLINE_OBJ.get('TEL_claim_sha256') == JRB_TEL_CURRENT_CLAIM_SHA
assert JRB_FINAL_DEADLINE_OBJ.get('effective_deadline_unix') == JRB_CONTROLLER_CLAIM_OBJ.get('effective_deadline_unix') and JRB_FINAL_DEADLINE_OBJ.get('window_seconds') == OWNER_EXTENSION_WINDOW
OUT = Path('/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t0240/attempt-03/resume-receipts.jsonl')
PREFLIGHT = E / 'RESUME-PREFLIGHT.json'
LOAD_SAMPLES = E / 'LOAD-SAMPLES.jsonl'
ARMED = E / 'ARMED-ORDER.json'
STOP = threading.Event()
STOP_REASON = {'value': None}
MAIN_PID = os.getpid()

os.umask(0o077)
try: os.setpriority(os.PRIO_PROCESS, 0, 19)
except OSError: pass
os.environ.update(PYTHONDONTWRITEBYTECODE='1', CARGO_INCREMENTAL='0', CARGO_PROFILE_DEV_DEBUG='line-tables-only', CUDA_VISIBLE_DEVICES='', RAYON_NUM_THREADS='8', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', AWS_EC2_METADATA_DISABLED='true')
for k in ('AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN','AWS_PROFILE','JOPE_AWS_PROFILE','GH_TOKEN','GITHUB_TOKEN'):
    os.environ.pop(k, None)

sys.path.insert(0, str(ROOT / 'deploy/qualification/q1/harness'))
import q1
import fifo_launch
from sustained_load_gate import SustainedLoadGate

assert q1.sha_file(JRB_TEL_ADMISSION) == JRB_TEL_ADMISSION_SHA
assert q1.sha_file(JRB_CLAIM) == JRB_CONTROLLER_CLAIM_SHA
assert q1.sha_file(JRB_FINAL_DEADLINE) == JRB_FINAL_DEADLINE_SHA
assert q1.sha_file(OWNER_FINAL_DEADLINE) == JRB_FINAL_DEADLINE_SHA
assert q1.sha_file(JRB_TEL_CURRENT_CLAIM) == JRB_TEL_CURRENT_CLAIM_SHA
HAND=json.loads(HANDOFF.read_text())
PLANOBJ=q1.load(PLAN)
assert q1.sha_file(JRB_TEL_CONTRACT) == JRB_TEL_CONTRACT_SHA
assert q1.sha_file(OWNER_EXTENSION) == OWNER_EXTENSION_SHA
OWNER_RECEIPT=json.loads(OWNER_EXTENSION.read_text())
TEL_OWNER_CONTRACT=json.loads(JRB_TEL_CONTRACT.read_text())
assert OWNER_RECEIPT.get('kind') == 'OWNER_DEADLINE_EXTENSION_APPROVAL'
assert OWNER_RECEIPT.get('approval_ref') == 't188s1838'
assert OWNER_RECEIPT.get('plan_sha256') == JRB_PLAN_SHA
assert OWNER_RECEIPT.get('reservation_sha256') == 'f59cc7e6e8ec2639c723a21db1f0225a0b304c099d63cf332190dcbe7b870242'
assert OWNER_RECEIPT.get('computation',{}).get('window_seconds') == OWNER_EXTENSION_WINDOW
assert OWNER_RECEIPT.get('new_deadline',{}).get('resume_must_occur_before_utc') == '2026-11-02T17:30:00Z (otherwise re-record; approval scope stays all 54 jobs finish)'
assert TEL_OWNER_CONTRACT.get('controller_claim_path') == str(JRB_CLAIM)
assert TEL_OWNER_CONTRACT.get('owner_extension',{}).get('sha256') == OWNER_EXTENSION_SHA
assert TEL_OWNER_CONTRACT.get('owner_extension',{}).get('window_seconds') == OWNER_EXTENSION_WINDOW
assert q1.sha_file(RUNTIME_BINDING_RECEIPT) == RUNTIME_BINDING_SHA
BACKEND=Path(HAND['backend'])
SOURCE='ccaf14b210b3b55a350218e60f6b49ef76524cf3'
assert q1.sha_file(AUTHORITY) == '339b124f5d011104582bb28b17a76725f3bffd2fc03ec27d3011b9870308d1d3'
assert q1.sha_file(FRESH_APPROVAL) == '8b5ed784a04ba1d264bfb9124f82e475de4d575fb0f381950cccbad872d30481'
assert q1.sha_file(T0430_AUTHORITY) == T0430_SHA
assert hashlib.file_digest(ORIGINAL.open('rb'),'sha256').hexdigest() == '8e1250610419bec534e88a118a1450414d112de79dd6b7f4fa0edfa72887966d'
assert q1.sha_file(RECOUNT) == '70f759f3bc7640320ac6d504c72d782e5cfdb219f6375d523e2b979d9ace7ea9'
assert q1.sha_file(BACKEND) == PLANOBJ['backend_sha256']
assert HAND['evaluator_sha256'] == '4e9239a1126589a42c3d80b1742d463d1b2331def59e5a08796169b4f3cec177'
q1.validate_plan(PLANOBJ, ROOT / 'deploy/qualification/q1/PREREGISTRATION.md')
q1.verify_owner(PLANOBJ, HAND['signature'], HAND['allowed_signers'], ROOT / 'deploy/qualification/q1/PREREGISTRATION.md')
assert PLANOBJ['source_commit'] == SOURCE and PLANOBJ['mode'] == 'qualification'
assert PLANOBJ['metadata_source_commit'] == 'fa07e1336c8fc7c04994faadf8f6efabded0a195'
assert len(PLANOBJ['datasets']) * len(PLANOBJ['seeds']) * 2 == 582

# Read only event identity/status metadata from original receipts; never inspect response values.
original_keys=set(); original_finished=set(); original_started={}
for line in ORIGINAL.read_text().splitlines():
    if not line: continue
    row=json.loads(line)
    key=(row['dataset']['id'], row['seed'], row['guidance'])
    if row.get('event')=='started':
        original_started[key]=original_started.get(key,0)+1
    elif row.get('event')=='finished':
        original_finished.add(key)
    else: raise RuntimeError('unexpected original receipt event')
    original_keys.add(key)
assert len(original_keys)==18 and len(original_finished)==17 and len(original_started)==18

# Resume ledger is append-only. Read event identities only; never inspect response values.
resume_started={}; resume_finished=set(); resume_attempts={}; resume_request_hash={}
for line in OUT.read_text().splitlines():
    if not line: continue
    row=json.loads(line); key=(row['dataset']['id'],row['seed'],row['guidance'])
    if row.get('event')=='started':
        resume_started[key]=resume_started.get(key,0)+1
        resume_attempts[key]=max(resume_attempts.get(key,0),row.get('attempt_number',0))
        resume_request_hash[key]=row['request_sha256']
    elif row.get('event')=='finished': resume_finished.add(key)
    else: raise RuntimeError('unexpected resume receipt event')

# Exact plan order. Exclude every original FINISHED and every resume FINISHED unit.
units=[]
for dataset in PLANOBJ['datasets']:
    for seed in PLANOBJ['seeds']:
        policies=['heuristic','learned']
        if seed % 2: policies.reverse()
        for policy in policies:
            key=(dataset['id'],seed,policy)
            if key not in original_finished and key not in resume_finished: units.append((dataset,seed,policy,key))
assert len(units)==565-len(resume_finished) and sum(key in original_started for _,_,_,key in units)==0
assert all(key in resume_started for key in resume_finished)
assert len(resume_started)==15 and len(resume_finished)==14
assert sum(1 for key in resume_started if key not in resume_finished)==1
assert not PREFLIGHT.exists() and not LOAD_SAMPLES.exists() and not ARMED.exists()

load1=os.getloadavg()[0]
local_free=shutil.disk_usage('/').free
remote_probe="python3 -c 'import os,shutil; print(os.getloadavg()[0]); print(shutil.disk_usage(chr(47)).free)'"
while True:
    try: remote_preflight=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','xbabe1',remote_probe],capture_output=True,text=True,timeout=8)
    except subprocess.SubprocessError: remote_preflight=None
    if remote_preflight is not None and remote_preflight.returncode==0 and len(remote_preflight.stdout.splitlines())==2:
        remote_load1=float(remote_preflight.stdout.splitlines()[0]); remote_free=int(remote_preflight.stdout.splitlines()[1])
        if remote_free>=100*1024**3: break
    time.sleep(30)

# New receipt stream plus durable authority/inventory sidecars; no outcome summarization.
refs={
 'schema':'jope.q1-resume-authority/v1',
 'authority':str(AUTHORITY),'authority_sha256':q1.sha_file(AUTHORITY),'fresh_watcher_approval':str(FRESH_APPROVAL),'fresh_watcher_approval_sha256':q1.sha_file(FRESH_APPROVAL),'current_watch_ruling':str(T0430_AUTHORITY),'current_watch_ruling_sha256':T0430_SHA,
 'original_receipts_sha256':'8e1250610419bec534e88a118a1450414d112de79dd6b7f4fa0edfa72887966d',
 'original_ledger_reference_sha256':'7594afd202648dbd6f4663bea4b8a7acb0ab754a9537848714912385e3113190',
 'partial_independent_recount_sha256':'70f759f3bc7640320ac6d504c72d782e5cfdb219f6375d523e2b979d9ace7ea9',
 'frozen16_receipts_sha256':'68b8683cc05708dd43c8a9ebdc9ae7e56ec3add32ad3953d15b98ab999026770',
 'plan_sha256':q1.digest(PLANOBJ),'signature_sha256':q1.sha_file(HAND['signature']),
 'backend_sha256':q1.sha_file(BACKEND),'evaluator_sha256':HAND['evaluator_sha256'],'jrb_resumer_sha256':JRB_SOURCE_SHA,'jrb_claim_path':str(JRB_CLAIM),'jrb_tel_contract_sha256':JRB_TEL_CONTRACT_SHA,'jrb_wave':JRB_WAVE,'runtime_source_commit':RUNTIME_SOURCE_COMMIT,'runtime_binding_receipt_sha256':RUNTIME_BINDING_SHA,
 'remaining_units':len(units),'prior_finished_units_frozen':len(original_finished),
 'prior_started_only_attempts':1,'resume_started_events_already_appended':sum(resume_started.values()),'no_budget_reset':True,
 'analysis_allowed_after_all_resume_units_terminal':True,
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()
}
PREFLIGHT.write_text(json.dumps({'refs':refs,'status':'PREFLIGHT_PASS_WAITING_FOR_REVIEW_JRB_FIFO','xbabe2_load1':load1,'xbabe1_load1':remote_load1,'xbabe2_cpu_count':os.cpu_count(),'xbabe2_root_free_bytes':local_free,'xbabe1_root_free_bytes':remote_free},indent=2)+'\n')

# Validate exact owner-authorized production transport without logging the bearer token.
engine_home=Path(HAND['engine_home'])
backend_home=engine_home
os.environ['JOPEDIME_HOME']=str(backend_home)
import production_backend

# T0445c requires a matching exact-byte Gov04 PASS receipt; API01 marker is not an arm gate.
while True:
    if GOV04_REVIEW.is_file():
        review=json.loads(GOV04_REVIEW.read_text())
        controller_review=review.get('controller',{})
        gate_review=review.get('sustained_load_gate',{})
        bindings=review.get('bindings',{})
        exact=(review.get('schema')=='jope.qlt04.gov04-exact-byte-review/v1' and review.get('reviewer')=='luna_v1_gov04' and controller_review.get('sha256')==q1.sha_file(Path(__file__)) and gate_review.get('sha256')==q1.sha_file(E/'sustained_load_gate.py') and bindings.get('jrb_source_sha256')==JRB_SOURCE_SHA and bindings.get('jrb_source_commit')==JRB_SOURCE_COMMIT and bindings.get('tel_admission_path')==str(JRB_TEL_ADMISSION) and bindings.get('tel_admission_sha256')==JRB_TEL_ADMISSION_SHA and bindings.get('tel_current_claim_path')==str(JRB_TEL_CURRENT_CLAIM) and bindings.get('tel_current_claim_sha256')==JRB_TEL_CURRENT_CLAIM_SHA and bindings.get('jrb_controller_claim_sha256')==JRB_CONTROLLER_CLAIM_SHA and bindings.get('jrb_controller_pid')==JRB_CONTROLLER_PID and bindings.get('jrb_final_deadline_path')==str(JRB_FINAL_DEADLINE) and bindings.get('jrb_final_deadline_sha256')==JRB_FINAL_DEADLINE_SHA and bindings.get('owner_extension_sha256')==OWNER_EXTENSION_SHA and bindings.get('owner_extension_window_seconds')==OWNER_EXTENSION_WINDOW and bindings.get('jrb_claim_path')==str(JRB_CLAIM) and bindings.get('tel_contract_sha256')==JRB_TEL_CONTRACT_SHA and bindings.get('jrb_plan_sha256')==JRB_PLAN_SHA and bindings.get('authority_sha256')==T0430_SHA and bindings.get('wave')==JRB_WAVE and bindings.get('runtime_source_commit')==RUNTIME_SOURCE_COMMIT and bindings.get('runtime_binding_receipt_sha256')==RUNTIME_BINDING_SHA and bindings.get('jrb_completion_gate')=='PASS_COMPLETION_REQUIRED')
        if exact and review.get('disposition')=='PASS_SCOPED_CONTROLLER_REVIEW_NO_LAUNCH':
            break
    time.sleep(15)

# JRB-01 must enter/release its FIFO turn before QLT-04 enqueues. Never jump its order.
paths=fifo_launch.Paths(root=ROOT,driver=ROOT/'deploy/qualification/q1/harness/q1.py',prereg=ROOT/'deploy/qualification/q1/PREREGISTRATION.md',fifo=ROOT/'research/flywheel_selfplay/qualification_lock_fifo.py',cohort=ROOT/'deploy/qualification/q1/cohort.json',input_manifest=ROOT/'deploy/qualification/q1/inputs-manifest.json',lock=Path('/home/ubuntu/jope_plan/qualification.lock'),queue=Path('/home/ubuntu/jope_plan/qualification.queue'))
fifo=fifo_launch.fifo_module(paths)

def oldest_active_ticket_done_aware():
    active=[]
    for entry in paths.queue.iterdir():
        if entry.name.endswith('.done') or not entry.is_file() or entry.is_symlink():
            continue
        parsed=fifo.parse_ticket_name(entry.name)
        if parsed and fifo.pid_is_alive(parsed[1]):
            active.append((*parsed,entry))
    return min(active)[3] if active else None

@contextmanager
def done_aware_lease(wait_seconds, load, poll=10, backoff=45):
    q1.require(paths.queue.is_dir() and not paths.queue.is_symlink(), 'existing FIFO directory required')
    descriptor=os.open(paths.lock,os.O_RDONLY|os.O_NOFOLLOW)
    handle=os.fdopen(descriptor,'rb')
    ticket=None; locked=False; deadline=time.monotonic()+wait_seconds
    try:
        q1.require(stat.S_ISREG(os.fstat(descriptor).st_mode),'existing regular lock required')
        ticket=fifo.enqueue_ticket(paths.queue,wave='Q1')
        time.sleep(min(backoff,max(0,deadline-time.monotonic())))
        while time.monotonic()<deadline:
            if load()<=85 and oldest_active_ticket_done_aware()==ticket:
                try:
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB); locked=True
                except BlockingIOError:
                    pass
                if locked:
                    same=os.fstat(descriptor); current=paths.lock.lstat()
                    q1.require((same.st_dev,same.st_ino)==(current.st_dev,current.st_ino),'qualification lock was replaced')
                    if load()<=85 and oldest_active_ticket_done_aware()==ticket:
                        yield handle,ticket
                        return
                    fcntl.flock(handle,fcntl.LOCK_UN); locked=False
            time.sleep(min(poll,max(0,deadline-time.monotonic())))
        raise TimeoutError('FIFO wait cap exhausted; study was not started')
    finally:
        if locked: fcntl.flock(handle,fcntl.LOCK_UN)
        handle.close()
        fifo.remove_ticket(ticket)
        q1.require(ticket is None or not ticket.exists(),'own FIFO ticket cleanup failed')
def write_armed(status, **fields):
    ARMED.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'authority_sha256':T0430_SHA,'fresh_watcher_approval_sha256':q1.sha_file(FRESH_APPROVAL),'resume_ledger':str(OUT),'resume_ledger_sha256':q1.sha_file(OUT),'JRB_first':True,'handoff_authority':'maintained shared qualification FIFO','jrb_claim_path':str(JRB_CLAIM),'jrb_source_sha256':JRB_SOURCE_SHA,'jrb_source_commit':JRB_SOURCE_COMMIT,'tel_admission_path':str(JRB_TEL_ADMISSION),'tel_admission_sha256':JRB_TEL_ADMISSION_SHA,'tel_current_claim_path':str(JRB_TEL_CURRENT_CLAIM),'tel_current_claim_sha256':JRB_TEL_CURRENT_CLAIM_SHA,'jrb_controller_claim_sha256':JRB_CONTROLLER_CLAIM_SHA,'jrb_controller_pid':JRB_CONTROLLER_PID,'jrb_final_deadline_path':str(JRB_FINAL_DEADLINE),'jrb_final_deadline_sha256':JRB_FINAL_DEADLINE_SHA,'jrb_plan_sha256':JRB_PLAN_SHA,'jrb_wave':JRB_WAVE,'tel_contract_sha256':JRB_TEL_CONTRACT_SHA,**fields},indent=2)+'\n')

def validate_fresh_jrb_claim(claim):
    q1.require(type(claim.get('pid')) is int and claim.get('pid')==JRB_CONTROLLER_PID and claim.get('source_sha256')==JRB_SOURCE_SHA and claim.get('source_head')==JRB_SOURCE_COMMIT and q1.sha_file(JRB_CLAIM)==JRB_CONTROLLER_CLAIM_SHA, 'fresh TEL-1548 JRB claim PID/source head/hash mismatch')
    q1.require(claim.get('authority')==str(T0430_AUTHORITY) and claim.get('authority_sha256')==T0430_SHA and claim.get('plan_sha256')==JRB_PLAN_SHA, 'fresh owner-extension JRB claim authority/plan mismatch')
    q1.require(claim.get('budget_reset') is False, 'fresh owner-extension JRB claim resets reserved campaign budget')
    q1.require(claim.get('source_loader_sha256')==JRB_SOURCE_LOADER_SHA and claim.get('resource_qualification_sha256')==JRB_RESOURCE_QUALIFICATION_SHA and claim.get('recovery_baseline_sha256')==JRB_BASELINE_SHA, 'fresh owner-extension JRB source/readmission binding mismatch')
    readmission_sha=claim.get('readmission_sha256')
    q1.require(readmission_sha==JRB_TEL_ADMISSION_SHA, 'fresh JRB controller claim is not bound to exact TEL-1548 admission')
    q1.require(claim.get('physical_updates_spent')==300 and claim.get('planned_physical_updates')==5600 and claim.get('planned_qualified_updates')==5400, 'fresh owner-extension JRB accounting differs from TEL contract')
    q1.require(claim.get('qualified_results')==str(JRB_STUDY/'qualified-t1458'), 'fresh owner-extension qualified-view path mismatch')
    preflight=claim.get('preflight',{})
    q1.require(preflight.get('status')=='INPUTS_VERIFIED' and preflight.get('plan_sha256')==JRB_PLAN_SHA and preflight.get('contexts')==2620 and preflight.get('jobs')==54, 'fresh owner-extension JRB preflight mismatch')
    q1.require(claim.get('owner_extension_sha256')==OWNER_EXTENSION_SHA and claim.get('owner_extension_receipt')==str(OWNER_EXTENSION) and claim.get('owner_extension_window_seconds')==OWNER_EXTENSION_WINDOW, 'fresh owner-extension receipt binding mismatch')
    resumed_at=claim.get('resumed_started_unix')
    effective=claim.get('effective_deadline_unix')
    original=claim.get('original_deadline_unix')
    prior_seconds=claim.get('prior_execution_seconds')
    q1.require(type(resumed_at) in (int,float) and resumed_at<OWNER_RESUME_BEFORE and type(effective) in (int,float) and abs((effective-resumed_at)-OWNER_EXTENSION_WINDOW)<0.01, 'fresh owner-extension effective deadline mismatch')
    q1.require(type(prior_seconds) in (int,float) and claim.get('maximum_total_execution_seconds')==prior_seconds+OWNER_EXTENSION_WINDOW and claim.get('maximum_deadline_extension_seconds')==effective-original and claim.get('deadline_amendment_seconds')==effective-original, 'fresh owner-extension wall-time accounting mismatch')

def qualified_completion_snapshot(claim):
    if not JRB_EXECUTION.is_file() or JRB_EXECUTION.is_symlink() or not JRB_LEDGER.is_file() or JRB_LEDGER.is_symlink() or not OWNER_FINAL_DEADLINE.is_file() or OWNER_FINAL_DEADLINE.is_symlink() or not OWNER_TIMEOUT.is_file() or OWNER_TIMEOUT.is_symlink():
        return None
    final_deadline=json.loads(OWNER_FINAL_DEADLINE.read_text())
    q1.require(q1.sha_file(JRB_CLAIM)==JRB_CONTROLLER_CLAIM_SHA and q1.sha_file(JRB_FINAL_DEADLINE)==JRB_FINAL_DEADLINE_SHA and q1.sha_file(OWNER_FINAL_DEADLINE)==JRB_FINAL_DEADLINE_SHA, 'JRB live claim/final deadline bytes changed from TEL-1548 handoff')
    q1.require(final_deadline.get('controller_pid')==claim.get('pid') and final_deadline.get('resumer_sha256')==JRB_SOURCE_SHA and final_deadline.get('owner_extension_sha256')==OWNER_EXTENSION_SHA and final_deadline.get('TEL_admission_sha256')==JRB_TEL_ADMISSION_SHA and final_deadline.get('TEL_claim_sha256')==JRB_TEL_CURRENT_CLAIM_SHA and claim.get('readmission_sha256')==JRB_TEL_ADMISSION_SHA and final_deadline.get('resumed_at_unix')==claim.get('resumed_started_unix') and final_deadline.get('effective_deadline_unix')==claim.get('effective_deadline_unix') and final_deadline.get('window_seconds')==OWNER_EXTENSION_WINDOW, 'owner FINAL-DEADLINE receipt does not match fresh claim')
    timeout=json.loads(OWNER_TIMEOUT.read_text())
    q1.require(timeout.get('controller_pid')==claim.get('pid') and timeout.get('effective_deadline_unix')==claim.get('effective_deadline_unix') and timeout.get('status') in ('CONTROLLER_EXITED_BEFORE_DEADLINE','OWNER_DEADLINE_SIGTERM_SENT'), 'owner outer-timeout receipt does not match fresh claim')
    execution=json.loads(JRB_EXECUTION.read_text())
    q1.require(execution.get('jobs')==54 and execution.get('qualified_optimizer_updates')==5400 and execution.get('physical_optimizer_updates')==5600 and execution.get('rejected_optimizer_updates')==200 and execution.get('resume_authority_sha256')==T0430_SHA and execution.get('budget_reset') is False and execution.get('qualified_results')==claim.get('qualified_results'), 'owner-extension execution completion accounting mismatch')
    q1.require(execution.get('owner_extension_sha256')==OWNER_EXTENSION_SHA and execution.get('owner_extension_receipt')==str(OWNER_EXTENSION) and execution.get('owner_extension_window_seconds')==OWNER_EXTENSION_WINDOW, 'owner-extension execution receipt binding mismatch')
    complete_events=[]
    with JRB_LEDGER.open('r',encoding='utf-8') as stream:
        for line in stream:
            if '\"event\": \"ALL_JOBS_COMPLETE\"' in line:
                complete_events.append(json.loads(line))
    if not complete_events:
        return None
    q1.require(len(complete_events)==1, 'TEL-1548 completion event is duplicated')
    event=complete_events[0]
    q1.require(event.get('jobs')==54 and event.get('optimizer_steps')==5400 and event.get('physical_optimizer_steps')==5600 and event.get('rejected_optimizer_steps')==200 and event.get('prior_artifacts_unchanged') is True and event.get('owner_extension_sha256')==OWNER_EXTENSION_SHA and event.get('owner_extension_receipt')==str(OWNER_EXTENSION) and event.get('owner_extension_window_seconds')==OWNER_EXTENSION_WINDOW, 'owner-extension all-jobs completion event mismatch')
    return {'execution_sha256':q1.sha_file(JRB_EXECUTION),'resume_ledger_sha256':q1.sha_file(JRB_LEDGER),'owner_final_deadline_sha256':q1.sha_file(OWNER_FINAL_DEADLINE),'owner_timeout_sha256':q1.sha_file(OWNER_TIMEOUT),'completion_event_utc':event.get('utc')}

write_armed('ARMED_WAITING_FOR_FRESH_TEL1548_BOUND_CONTROLLER_CLAIM')
while not JRB_CLAIM.is_file():
    time.sleep(10)
q1.require(not JRB_CLAIM.is_symlink(), 'fresh T1458 JRB claim must not be a symlink')
jrb_claim=json.loads(JRB_CLAIM.read_text())
validate_fresh_jrb_claim(jrb_claim)
jrb_pid=jrb_claim['pid']
write_armed('ARMED_WAITING_FOR_TEL1548_FIFO_AND_QUALIFIED_COMPLETION',jrb_pid=jrb_pid,readmission_sha256=jrb_claim['readmission_sha256'])

while True:
    current_ticket=None
    for entry in paths.queue.iterdir():
        base=entry.name[:-5] if entry.name.endswith('.done') else entry.name
        parsed=fifo.parse_ticket_name(base)
        if not parsed or parsed[1]!=jrb_pid or parsed[2]!=JRB_WAVE: continue
        q1.require(entry.is_file() and not entry.is_symlink(), 'current T1458 JRB FIFO ticket is not a regular file')
        ticket_fields={line.split('=',1)[0]:line.split('=',1)[1] for line in entry.read_text().splitlines() if '=' in line}
        q1.require(ticket_fields.get('pid')==str(jrb_pid) and ticket_fields.get('wave')==JRB_WAVE and ticket_fields.get('epoch_ns')==str(parsed[0]), 'current T1458 JRB FIFO ticket contents mismatch claim')
        current_ticket={'ticket':base,'path':str(entry),'state':'done' if entry.name.endswith('.done') else 'live','pid':jrb_pid,'epoch_ns':parsed[0]}
        break
    if current_ticket and current_ticket['state']=='done':
        completion=qualified_completion_snapshot(jrb_claim)
        if completion:
            write_armed('CURRENT_TEL1548_JRB_QUALIFIED_COMPLETION_CONFIRMED',readmission_sha256=jrb_claim['readmission_sha256'],**completion,**current_ticket)
            break
        write_armed('CURRENT_TEL1548_TICKET_DONE_WAITING_FOR_QUALIFIED_COMPLETION',readmission_sha256=jrb_claim['readmission_sha256'],**current_ticket)
    elif current_ticket:
        write_armed('CURRENT_TEL1548_JRB_FIFO_LIVE_WAITING_FOR_QUALIFIED_COMPLETION',readmission_sha256=jrb_claim['readmission_sha256'],**current_ticket)
    else:
        write_armed('WAITING_FOR_CURRENT_TEL1548_JRB_FIFO_TICKET',jrb_pid=jrb_pid,readmission_sha256=jrb_claim['readmission_sha256'])
    time.sleep(10)

# Start the pinned engine against its existing journal; separate, exclusive resume names.
remote_home='/mnt/jope-build-native/jope-artifacts/sol_v1_gov03/qlt04-t0115/engine-home'
remote_bin='/mnt/jope-build-native/jope-artifacts/sol_v1_api01/qlt04-rc4-linux/jopedime-q1-linux'
remote_bundle='/home/ubuntu/jope_plan/evidence/QLT-04/api01-linux-rc4-t1955/authenticated-r782'
remote_setup = r'''import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
E=Path('/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t0240/attempt-08'); E.mkdir(mode=0o700,parents=True,exist_ok=True); assert not E.is_symlink() and E.stat().st_mode & 0o077 == 0; H=Path('/mnt/jope-build-native/jope-artifacts/sol_v1_gov03/qlt04-t0115/engine-home'); B=Path('/mnt/jope-build-native/jope-artifacts/sol_v1_api01/qlt04-rc4-linux/jopedime-q1-linux'); bundle=Path('/home/ubuntu/jope_plan/evidence/QLT-04/api01-linux-rc4-t1955/authenticated-r782')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
assert sha(B)=='4e9239a1126589a42c3d80b1742d463d1b2331def59e5a08796169b4f3cec177'
assert not (H/'daemon.json').exists()
assert not (E/'RESUME-ENGINE-STARTED.json').exists()
assert not (E/'resume-engine.stdout').exists()
assert __import__('shutil').disk_usage('/').free > 100*1024**3
for key in ('AWS_','GH_','GITHUB_','JOPE_AWS_'):
    pass
env={k:v for k,v in os.environ.items() if not k.startswith(('AWS_','GH_','GITHUB_','JOPE_AWS_'))}
env.update(JOPEDIME_HOME=str(H),RAYON_NUM_THREADS='8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(H/'scratch'),PYTHONDONTWRITEBYTECODE='1')
args=['nice','-n','19',str(B),'research','serve-q1','--bundle-dir',str(bundle)]
with (E/'resume-engine.stdout').open('xb') as out,(E/'resume-engine.stderr').open('xb') as err:
    p=subprocess.Popen(args,cwd=E,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
rec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'argv':args,'engine_home':str(H),'evaluator_sha256':sha(B),'activity':'same pinned evaluator and existing journal; no request outcome copied or opened'}
(E/'RESUME-ENGINE-STARTED.json').write_text(json.dumps(rec,indent=2)+'\n')
for _ in range(120):
    if p.poll() is not None: raise RuntimeError('engine exited before readiness; inspect private stderr')
    d=H/'daemon.json'
    if d.is_file():
        data=json.loads(d.read_text()); assert data['pid']==p.pid and d.stat().st_mode & 0o077 == 0
        assert sha(Path('/proc')/str(p.pid)/'exe')==rec['evaluator_sha256']
        rec.update(status='ENGINE_READY',endpoint=data['url'],daemon_mode='0600')
        (E/'RESUME-ENGINE-STARTED.json').write_text(json.dumps(rec,indent=2)+'\n')
        print('ENGINE_READY',p.pid,data['url'],flush=True);break
    time.sleep(.25)
else: raise RuntimeError('daemon readiness timeout')
'''
# Fail closed unless all attempt-scoped writes stay in this attempt directory.
attempt_write_targets=(PREFLIGHT,LOAD_SAMPLES,ARMED,E/'ENGINE-READY.txt',E/'RESUME-ENGINE-STARTED.json',E/'RESUME-TUNNEL.json',E/'PROGRESS.json',E/'RESUME-EXECUTION-STATUS.json',E/'resume-engine.stdout',E/'resume-engine.stderr',E/'resume-tunnel.stdout',E/'resume-tunnel.stderr')
q1.require(E.name=='attempt-08' and all(p.parent==E for p in attempt_write_targets), 'attempt-local write target escaped attempt-08')
remote_root_line=[line for line in remote_setup.splitlines() if line.startswith('E=Path(')]
q1.require(len(remote_root_line)==1 and remote_root_line[0].startswith(f'E=Path({str(E)!r});'), 'remote attempt root does not match attempt-08')
q1.require(remote_setup.count('/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t0240/attempt-')==1, 'remote setup has an additional attempt-scoped path')

setup=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','xbabe1','python3 -B -'],input=remote_setup,text=True,capture_output=True,timeout=40)
if setup.returncode: raise RuntimeError('remote pinned engine start failed (see private stderr)')
# Only readiness metadata is printed by the remote launcher.
(E/'ENGINE-READY.txt').write_text(setup.stdout)
# Transfer the mode-0600 daemon record privately; never print or parse its bearer token to output.
local_daemon=engine_home/'daemon.json'
if local_daemon.exists(): raise RuntimeError('local daemon record already exists')
scp=subprocess.run(['scp','-q','xbabe1:'+remote_home+'/daemon.json',str(local_daemon)],capture_output=True,timeout=20)
if scp.returncode: raise RuntimeError('private daemon record transfer failed')
os.chmod(local_daemon,0o600)
actual_endpoint=json.loads(local_daemon.read_text())['url']
from urllib.parse import urlsplit
actual_port=urlsplit(actual_endpoint).port
assert actual_port and urlsplit(actual_endpoint).hostname=='127.0.0.1'
os.environ['JOPE_Q1_ENGINE_URL']=actual_endpoint
production_backend.read_daemon_token(actual_endpoint)

# Start a dedicated SSH loopback tunnel; no forwarding beyond the pinned loopback endpoint.
tun=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-N','-L',f'127.0.0.1:{actual_port}:127.0.0.1:{actual_port}','xbabe1'],stdin=subprocess.DEVNULL,stdout=(E/'resume-tunnel.stdout').open('xb'),stderr=(E/'resume-tunnel.stderr').open('xb'),start_new_session=True)
(E/'RESUME-TUNNEL.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':tun.pid,'argv':['ssh','-N','-L',f'127.0.0.1:{actual_port}:127.0.0.1:{actual_port}','xbabe1'],'endpoint':actual_endpoint},indent=2)+'\n')
time.sleep(.5)
if tun.poll() is not None: raise RuntimeError('resume tunnel failed; inspect private stderr')

# Sustained load monitor. A spike never kills an in-flight unit; it pauses only future dispatch.
PAUSE_LOCK=threading.Lock()
PAUSE_REASONS=set()
if local_free < 200*1000**3: PAUSE_REASONS.add('xbabe2_disk_headroom')
if remote_free < 100*1024**3: PAUSE_REASONS.add('xbabe1_disk_headroom')
LOAD_GATES={'xbabe2':SustainedLoadGate(),'xbabe1':SustainedLoadGate()}
LAST_LOAD={'xbabe2':load1,'xbabe1':remote_load1}
MONITOR_OK=True
INITIAL_HEADROOM_PENDING=os.getloadavg()[0] >= 70

def sample_resources():
    global MONITOR_OK, INITIAL_HEADROOM_PENDING
    local_load=os.getloadavg()[0]
    local_free=shutil.disk_usage('/').free
    cmd="python3 -c 'import os,shutil; print(os.getloadavg()[0]); print(shutil.disk_usage(chr(47)).free)'"
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','xbabe1',cmd],capture_output=True,text=True,timeout=8)
    remote_values=result.stdout.splitlines()
    remote_ok=result.returncode==0 and len(remote_values)==2
    sample={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'xbabe2_load1':local_load,'xbabe2_root_free_bytes':local_free,'sample_interval_seconds':60,'threshold_load':85,'resume_load':70,'sustained_samples_required':5}
    with PAUSE_LOCK:
        if not remote_ok:
            MONITOR_OK=False; PAUSE_REASONS.add('remote_monitor_unavailable')
            for gate in LOAD_GATES.values(): gate.missed()
            sample['monitor_status']='remote_unavailable'
        else:
            remote_load=float(remote_values[0]); remote_free=int(remote_values[1])
            LAST_LOAD['xbabe1']=remote_load
            sample.update(xbabe1_load1=remote_load,xbabe1_root_free_bytes=remote_free,monitor_status='ok')
            MONITOR_OK=True
            if local_load<70 and remote_load<70:
                PAUSE_REASONS.discard('remote_monitor_unavailable'); PAUSE_REASONS.discard('resource_monitor_error')
            for host,value in (('xbabe2',local_load),('xbabe1',remote_load)):
                if LOAD_GATES[host].observe(value,time.monotonic()): PAUSE_REASONS.add(host+'_sustained_load_over_85')
                else: PAUSE_REASONS.discard(host+'_sustained_load_over_85')
            if local_free < 200*1000**3:
                PAUSE_REASONS.add('xbabe2_disk_headroom')
            else: PAUSE_REASONS.discard('xbabe2_disk_headroom')
            if remote_free < 100*1024**3:
                PAUSE_REASONS.add('xbabe1_disk_headroom')
            else: PAUSE_REASONS.discard('xbabe1_disk_headroom')
            if local_load<70: INITIAL_HEADROOM_PENDING=False
            if local_load<70: PAUSE_REASONS.discard('remote_monitor_unavailable')
        sample.update(high_sample_counts={host:gate.high_samples for host,gate in LOAD_GATES.items()},dispatch_paused=bool(PAUSE_REASONS),pause_reasons=sorted(PAUSE_REASONS),initial_headroom_pending=INITIAL_HEADROOM_PENDING)
    with LOAD_SAMPLES.open('a',encoding='utf-8') as stream:
        q1.append_receipt(stream,sample)
    return sample

def monitor():
    while not STOP.wait(60):
        try: sample_resources()
        except BaseException as ex:
            with PAUSE_LOCK:
                PAUSE_REASONS.add('resource_monitor_error')
                globals()['MONITOR_OK']=False
            try:
                with LOAD_SAMPLES.open('a',encoding='utf-8') as stream:
                    q1.append_receipt(stream,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monitor_status':'error','error_type':type(ex).__name__,'dispatch_paused':True})
            except BaseException: pass

threading.Thread(target=monitor,daemon=True).start()

def wait_for_dispatch():
    while not STOP.is_set():
        with PAUSE_LOCK:
            if MONITOR_OK and not PAUSE_REASONS and not INITIAL_HEADROOM_PENDING:
                return
        time.sleep(5)
    raise InterruptedError('controller received an external stop while paused')

def signal_stop(signum, frame):
    STOP.set()
    raise InterruptedError('controller received '+signal.Signals(signum).name)
signal.signal(signal.SIGTERM,signal_stop)

# Shared maintained FIFO and canonical qualification lock; no new ticket survives this run.
# FIFO paths/module were initialized before the JRB-first ordering gate.
wall_cap=6*len(PLANOBJ['datasets'])*600
start_time=time.monotonic()
status='not_started'; completed=0; failures=0
try:
    # Preserve FIFO ordering; transient high load makes the lease wait, not exit.
    with done_aware_lease(25200,lambda: os.getloadavg()[0],poll=10,backoff=45) as (handle,ticket):
        status='locked'
        # First unit waits for <70 headroom; subsequent pauses require five sustained samples.
        for idx,(dataset,seed,policy,key) in enumerate(units,1):
            wait_for_dispatch()
            if time.monotonic()-start_time >= wall_cap: raise TimeoutError('existing signed Q1 serial wall cap exhausted')
            request={'schema':'jope.q1-request/v1','mode':PLANOBJ['mode'],'plan_sha256':q1.digest(PLANOBJ),'source_commit':PLANOBJ['source_commit'],'pins':PLANOBJ['pins'],'metric':'rmse','dataset':dataset,'guidance':policy,'seed':seed,'budget':PLANOBJ['budget']}
            request['request_sha256']=q1.digest(request)
            identity={k:request[k] for k in ('mode','plan_sha256','source_commit','pins','dataset','guidance','seed','budget','request_sha256')}
            prior_resume_starts=resume_started.get(key,0)
            if prior_resume_starts:
                assert key not in resume_finished and request['request_sha256']==resume_request_hash[key]
                attempts=resume_attempts[key]+1
            else:
                attempts=original_started.get(key,0)+prior_resume_starts+1
            identity.update({'resume_authority_sha256':refs['authority_sha256'],'fresh_watcher_approval_sha256':refs['fresh_watcher_approval_sha256'],'original_receipts_sha256':refs['original_receipts_sha256'],'partial_independent_recount_sha256':refs['partial_independent_recount_sha256'],'attempt_number':attempts,'engine_endpoint':actual_endpoint})
            t0=time.monotonic()
            with OUT.open('a',encoding='utf-8') as stream:
                if not prior_resume_starts:
                    q1.append_receipt(stream,{**identity,'event':'started','status':'outcome_unknown','resume_unit_index':idx,'resume_unit_total':len(units)})
                try:
                    raw=q1._invoke_backend([str(BACKEND.resolve())],q1.canonical(request),dict(os.environ),PLANOBJ['timeout_seconds'],5)
                    q1.require(raw.returncode==0,'backend process failed')
                    response=q1.response_valid(q1.decode(raw.stdout),request)
                    row={**identity,'event':'finished','response':response,'wall_seconds':time.monotonic()-t0}
                except (InterruptedError, q1.BackendCleanupError, q1.BackendSpawnError):
                    raise
                except (ValueError,subprocess.TimeoutExpired,OSError) as ex:
                    row={**identity,'event':'finished','response':{'status':'failed','loss':None,'charged':None,'completed':None,'reason':type(ex).__name__+': '+str(ex)},'wall_seconds':time.monotonic()-t0}
                q1.append_receipt(stream,row)
            completed+=1
            failures += row['response']['status']!='complete'
            status='running'
            if idx%10==0:
                progress={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'terminal_resume_units':completed,'remaining_resume_units':len(units)-completed,'execution_state':'NO_ANALYSIS'}
                (E/'PROGRESS.json').write_text(json.dumps(progress,indent=2)+'\n')
        status='all_resume_units_terminal'
except BaseException as ex:
    status='stopped_or_incomplete'
    if not STOP_REASON['value']: STOP_REASON['value']=type(ex).__name__+': '+str(ex)
finally:
    STOP.set()
    ending={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'terminal_resume_units':completed,'resume_units_total':len(units),'resource_stop_reason':STOP_REASON['value'],'elapsed_seconds':time.monotonic()-start_time,'analysis_performed':False,'signature_created':False,'default_changed':False,'report_owner':'sol_v1_gov03','receipt_sha256':q1.sha_file(OUT) if OUT.exists() else None}
    (E/'RESUME-EXECUTION-STATUS.json').write_text(json.dumps(ending,indent=2)+'\n')
    # Stop only this run's tunnel and the pinned engine process. No output includes token bytes.
    try:
        if tun.poll() is None:
            os.killpg(tun.pid,signal.SIGTERM); tun.wait(timeout=5)
    except BaseException: pass
    try:
        stop_code=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','xbabe1',f'env JOPEDIME_HOME={remote_home} CUDA_VISIBLE_DEVICES= {remote_bin} serve --stop'],capture_output=True,timeout=30).returncode
        ending['remote_stop_exit']=stop_code
    except BaseException as ex: ending['remote_stop_exit']=type(ex).__name__
    try:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','xbabe1',f'rm -f {remote_home}/daemon.json'],check=True,capture_output=True,timeout=10)
    except BaseException: ending['remote_daemon_removed']=False
    else: ending['remote_daemon_removed']=True
    try: local_daemon.unlink(missing_ok=True)
    except BaseException: ending['local_daemon_removed']=False
    else: ending['local_daemon_removed']=True
    (E/'RESUME-EXECUTION-STATUS.json').write_text(json.dumps(ending,indent=2)+'\n')
    if status not in ('all_resume_units_terminal',):
        print(json.dumps({'status':status,'terminal_resume_units':completed,'resume_units_total':len(units),'resource_stop_reason':STOP_REASON['value']}),flush=True)
    else:
        print(json.dumps({'status':status,'terminal_resume_units':completed,'resume_units_total':len(units),'analysis_performed':False}),flush=True)
