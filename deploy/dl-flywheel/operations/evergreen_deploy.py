import base64,hashlib,json,math,shlex,subprocess,time
from pathlib import Path
BASE='/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_20261009'
OUT=Path('outputs');SRC=Path('work/flywheel-20261008/patch/deploy/dl-flywheel/jope_flywheel')
sha=lambda b:hashlib.sha256(b).hexdigest()
def remote(cmd,data=None):return subprocess.run(['ssh','-o','BatchMode=yes','xbabe2',shlex.join(cmd)],input=data,text=True,capture_output=True,timeout=150)
mraw=(OUT/'DL_FLYWHEEL_EVERGREEN_CONFIG.json').read_bytes();m=json.loads(mraw);digest=sha(mraw)
review=json.loads((OUT/'DL_FLYWHEEL_EVERGREEN_REVIEW.json').read_text())
files={'config.json':mraw,'review.json':(OUT/'DL_FLYWHEEL_EVERGREEN_REVIEW.json').read_bytes()}
for ref in m['source_pins']:
 b=(SRC/Path(ref['path']).name).read_bytes();assert sha(b)==ref['sha256'];files['jope_flywheel/'+Path(ref['path']).name]=b
claim='Codex root claims new owned evergreen runtime '+BASE+' and service jopedime-flywheel-evergreen-20261009. Exact config '+digest+'; source334f revised review4bc8d168; registry8fe40984 all1068parents currentrole verified. Userauthorized100k goal, first128 unsatisfied variants only, CPU4/2GiB/noGPU, deadline'+str(m['deadline_unix'])+'. New preparedresearch developmental95/5,1014train54validation; no freshholdout or generated teaching claim. Unknown outcomes reserve capacity, failures halt, resourcewaitload>96/memory<32GiB/disk<100GiB. No originalsource/checkout/Grok process control. New source proposal only; no PR lane transfer claimed.'
code='''from pathlib import Path
import base64,hashlib,json,sys,fcntl,subprocess,datetime
x=json.load(sys.stdin);root=Path(x['base']);assert all(not p.is_symlink() for p in(root,*root.parents))
repo=Path('/home/ubuntu/JopeDime')
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
for log in (repo/'AGENT_CHAT.md',Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/DL_TRAINING_DASHBOARD_HANDOFF.md')):
 with Path(str(log)+'.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);prior=log.read_bytes();entry='\\n\\n### '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' — Codex evergreen intake claim\\n'+x['claim']+' Observed canonical '+branch+'@'+head+'. Prior claim tail re-read under lock.\\n'
  with log.open('ab')as out:out.write(entry.encode())
  assert log.read_bytes().startswith(prior)
root.mkdir(exist_ok=False);pins={}
for name,v in x['files'].items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(v)
 with p.open('xb')as f:f.write(b)
 assert p.read_bytes()==b;pins[name]=hashlib.sha256(b).hexdigest()
print(json.dumps({'pins':pins,'head':head,'branch':branch}))
'''
r=remote(['python3','-c',code],json.dumps({'base':BASE,'claim':claim,'files':{n:base64.b64encode(b).decode()for n,b in files.items()}}));r.check_returncode();stage=json.loads(r.stdout)
props=['--property=CPUQuota=400%','--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=32','--property=Nice=15','--property=KillMode=control-group','--property=Restart=no','--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=PYTHONPATH='+BASE]
pre="""import json,numpy as np;from pathlib import Path
from jope_flywheel.evergreen import pinned,source,recipe,byte_sha
conf=json.loads(pinned({'path':CONFIG,'sha256':SHA}))
for ref in conf['source_pins']+conf['authority_pins']:pinned(ref)
parents=[json.loads(s)for s in pinned(conf['parent_registry'],32*1024**2).splitlines()]
constant=[]
for p in parents:
 x,y,f=source(p)
 if recipe(p,0)['target_noise_fraction'] and np.std(y[f==0])==0:constant.append(p['parent_lineage_sha256'])
print(json.dumps({'parents_verified':len(parents),'constant_noop_parents':constant,'manifest_sha256':SHA}))
""".replace('CONFIG',repr(BASE+'/config.json')).replace('SHA',repr(digest))
r=remote(['systemd-run','--user','--wait','--pipe','--unit=jopedime-evergreen-preflight-20261009',*props,'--property=RuntimeMaxSec=120','/home/ubuntu/jopedime_probe_exp/ofm-20261008/venv/bin/python','-B','-c',pre]);pre_rec={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
with (OUT/'DL_FLYWHEEL_EVERGREEN_PREFLIGHT.json').open('x')as f:json.dump(pre_rec,f,indent=2)
r.check_returncode();proof=json.loads(r.stdout);assert proof['parents_verified']==1068 and not proof['constant_noop_parents'],proof
cmd=['systemd-run','--user','--unit=jopedime-flywheel-evergreen-20261009',*props,'--property=RuntimeMaxSec='+str(math.floor(m['deadline_unix']-time.time())),'/home/ubuntu/jopedime_probe_exp/ofm-20261008/venv/bin/python','-B','-m','jope_flywheel.evergreen','--config',BASE+'/config.json','--config-sha',digest]
r=remote(cmd);rec={'at':time.time(),'manifest_sha256':digest,'stage':stage,'preflight':proof,'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
with (OUT/'DL_FLYWHEEL_EVERGREEN_LAUNCH.json').open('x')as f:json.dump(rec,f,indent=2)
r.check_returncode();print(json.dumps({'preflight':proof,'unit':'jopedime-flywheel-evergreen-20261009'}))
