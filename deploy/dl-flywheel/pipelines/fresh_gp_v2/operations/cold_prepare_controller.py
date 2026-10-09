"""One-use CPU/GET-only cold preparation with owned-unit memory pressure retirement."""
from pathlib import Path
import json,time,os,hashlib,subprocess,sys
ROOT=Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-fresh128-gp-20261009-v3')
PYTHON='/usr/bin/python3.12';PY_SHA='e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f'
UNIT='jopedime-fresh128-cold-prepare-20261009-v3'
SOURCE='4a9f318cd35ac16fea278fdaca411af5e07fd89d5b116de68e0d0f927e90b8d3'
REQUEST='32b8525a6d4e296f9bad517033078ffe95bc7e136e7487a4cc8706cf27d1355b'
GLOBAL=1791592146.380285;GIB=1024**3
OUT=ROOT/'prepared-001';OPS=ROOT/'cold-prepare-001'
def put(path,v):
 with path.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def show():
 r=subprocess.run(['systemctl','--user','show',UNIT,'-p','LoadState','-p','ActiveState','-p','MainPID','-p','ControlGroup','-p','CPUUsageNSec','-p','MemoryPeak','-p','MemorySwapPeak','-p','Result','-p','ExecMainStatus'],capture_output=True,text=True,check=True)
 return dict(x.split('=',1)for x in r.stdout.splitlines()if '='in x)
def resource():
 m={k:int(v.strip().split()[0])*1024 for k,v in [x.split(':',1)for x in Path('/proc/meminfo').read_text().splitlines()]if k in ['MemTotal','MemAvailable']};m['load1']=os.getloadavg()[0];m['minimum_available']=max(36*GIB,int(m['MemTotal']*.15)+2*GIB);return m
def owns(s,expected):
 p=int(s.get('MainPID','0'));assert p and s['ControlGroup'].endswith('/'+UNIT+'.service')
 cmd=(Path('/proc')/str(p)/'cmdline').read_bytes().rstrip(b'\0').split(b'\0');assert cmd==[x.encode()for x in expected]
 return p
def main():
 assert not OUT.exists()and not OPS.exists();OPS.mkdir(mode=0o700)
 assert hashlib.sha256(Path(PYTHON).read_bytes()).hexdigest()==PY_SHA
 assert hashlib.sha256((ROOT/'source-manifest.json').read_bytes()).hexdigest()==SOURCE
 assert hashlib.sha256((ROOT/'prepare-request.json').read_bytes()).hexdigest()==REQUEST
 r=subprocess.run([PYTHON,'-c','import sys,numpy,boto3,json; print(json.dumps([sys.version.split()[0],numpy.__version__,boto3.__version__]))'],capture_output=True,text=True,check=True,timeout=20,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'});assert json.loads(r.stdout)==['3.12.3','1.26.4','1.35.49']
 a=resource();assert a['MemAvailable']>a['minimum_available'] and a['load1']<64
 s=show();assert s.get('LoadState')=='not-found'and s.get('MainPID')=='0'
 started=time.time();mono=time.monotonic();deadline=min(started+600,GLOBAL);assert deadline-started>=300
 put(OPS/'intent.json',{'created_unix':started,'deadline_unix':deadline,'global_deadline_unix':GLOBAL,'resources':a,'cpu_quota_percent':100,'memory_max_bytes':512*1024**2,'memory_swap_max_bytes':0,'scope':'CPU/GET-only preparation; no fits/PUT/claims/broker/epoch','source_sha256':SOURCE,'request_sha256':REQUEST,'python_sha256':PY_SHA})
 argv=['systemd-run','--user','--unit='+UNIT,'--property=CPUQuota=100%','--property=MemoryMax=512M','--property=MemorySwapMax=0','--property=Nice=19','--property=TasksMax=32','--property=Restart=no','--property=KillMode=control-group','--property=RuntimeMaxSec='+str(int(deadline-time.time())),'--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=NUMEXPR_NUM_THREADS=1',PYTHON,'-B',str(ROOT/'source/fresh_gp.py'),'--source-manifest',str(ROOT/'source-manifest.json'),'--source-sha',SOURCE,'prepare','--ref',str(ROOT/'prepare-request.json'),'--sha',REQUEST,'--out',str(OUT)]
 r=subprocess.run(argv,capture_output=True,text=True);put(OPS/'launch.json',{'at':time.time(),'returncode':r.returncode,'stderr':r.stderr,'argv':argv});assert r.returncode==0
 reason='natural_close'
 while True:
  s=show()
  if s.get('ActiveState')in('inactive','failed')and s.get('MainPID')=='0'and not s.get('ControlGroup'):break
  if s.get('ActiveState')=='activating'and s.get('MainPID')=='0'and time.monotonic()-mono<20:
   time.sleep(.1);continue
  owns(s,argv[argv.index(PYTHON):]);a=resource()
  if a['MemAvailable']<a['minimum_available'] or a['load1']>=96 or time.time()>=deadline or time.monotonic()-mono>=600:
   reason='owned_resource_or_clock_stop';put(OPS/'stop-intent.json',{'at':time.time(),'unit':s,'resources':a,'reason':reason});subprocess.run(['systemctl','--user','stop',UNIT],capture_output=True,text=True,check=True);break
  time.sleep(1)
 s=show();assert s.get('ActiveState')in('inactive','failed')and s.get('MainPID')=='0'and not s.get('ControlGroup')
 receipt={'at':time.time(),'wall_ns':int((time.monotonic()-mono)*1e9),'reason':reason,'unit':s,'prepared_complete':(OUT/'prepared.json').is_file(),'source_sha256':SOURCE,'request_sha256':REQUEST,'no_automatic_retry':True,'successful_exit':s.get('Result')=='success'and s.get('ExecMainStatus')=='0'}
 if receipt['prepared_complete']:
  b=(OUT/'prepared.json').read_bytes();v=json.loads(b);assert v['native_fits']==0 and v['S3_PUTs']==0;receipt['prepared']={'path':str(OUT/'prepared.json'),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'source_replica_bytes':v['source_replica_bytes'],'contexts':len(v['selection']),'runtime_versions':v['runtime_versions']}
 put(OPS/'terminal.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
