"""xbabe2-only S3 preparation; back off only this own child when host is busy."""
from pathlib import Path
import json,os,subprocess,sys,time,signal,datetime
import psutil
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h359-hwv');D.mkdir(parents=True,exist_ok=True)
assert os.uname().nodename.split('.')[0]=='xbabe2'
def allowed():return os.getloadavg()[0]<=85 and psutil.cpu_percent(interval=.2)<=85 and psutil.virtual_memory().available>=20*2**30
steps=[]
for args in [['fetch_base_h359.py'],['input_preflight.py'],['../stage_inputs.py']]:
 while not allowed():
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,5,13,15,tzinfo=datetime.timezone.utc):raise SystemExit('Fixed preparation deadline')
  time.sleep(5)
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();env=dict(os.environ,H359_INPUT_PREFIX='/home/ubuntu/jope-data')
 child=subprocess.Popen([sys.executable,str(S/args[0]),*args[1:]],env=env);paused=False
 while child.poll() is None:
  time.sleep(2);ok=allowed()
  if not ok and not paused:child.send_signal(signal.SIGSTOP);paused=True
  elif ok and paused:child.send_signal(signal.SIGCONT);paused=False
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,5,13,15,tzinfo=datetime.timezone.utc):
   child.terminate()
   if paused:child.send_signal(signal.SIGCONT)
   raise SystemExit('Preparation hit fixed compute deadline')
 steps.append(dict(command=args,exit_status=child.returncode,started_utc=start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 if child.returncode:raise SystemExit(child.returncode)
env=dict(os.environ,H359_INPUT_PREFIX=str(D/'inputs'));subprocess.run([sys.executable,str(S/'input_preflight.py')],env=env,check=True)
(P/'fetch-staged.json').write_text(json.dumps(dict(status='PASS',steps=steps),indent=2)+'\n')
