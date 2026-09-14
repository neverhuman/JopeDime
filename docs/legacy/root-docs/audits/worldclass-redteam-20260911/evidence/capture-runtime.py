import json,pathlib,subprocess,time,hashlib,fcntl
root=pathlib.Path('/home/ubuntu/jope-dime');pr=root/'parallel';result={'timestamp':time.time(),'units':{},'processes':[]}
for u in ['jope-parallel-gp','jope-parallel-campaign','jope-parallel-gp-next','jope-parallel-controller-next','jope-parallel-resident-gpu']:
 p=subprocess.run(['systemctl','show',u,'--property=MainPID,Restart,ActiveState,SubState,DropInPaths,ExecCondition'],text=True,capture_output=True)
 result['units'][u]=dict(a.split('=',1) for a in p.stdout.splitlines() if '=' in a)
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cg=(p/'cgroup').read_text()
  if any(x in cg for x in ['jope-parallel-gp','jope-parallel-campaign','jope-parallel-controller-next','jope-parallel-resident-gpu']):
   result['processes'].append({'pid':int(p.name),'args':(p/'cmdline').read_bytes().decode().strip('\0').split('\0'),'cgroup':cg,'stat':(p/'stat').read_text()})
 except OSError:pass
state=pr/'runs/continuous-learning-v2-20260909/state.json';d=json.loads(state.read_text())
result['state']={'sha256':hashlib.sha256(state.read_bytes()).hexdigest(),'scalars':{k:v for k,v in d.items() if not isinstance(v,(list,dict))},'lengths':{k:len(v) for k,v in d.items() if isinstance(v,(list,dict))}}
result['hardware']=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True,capture_output=True).stdout
chat=root/'AGENT_CHAT.md'
with open(str(chat)+'.lock','a') as l:
 fcntl.flock(l,fcntl.LOCK_SH)
 result['remote_chat_tail']=chat.read_text()[-30000:]
result['run_dirs']=[p.name for p in (pr/'runs').iterdir() if p.is_dir() and any(s in p.name for s in ['parity','diagnostic','audit','redteam','flywheel','continuous','sampler','expansion'])]
print(json.dumps(result))
