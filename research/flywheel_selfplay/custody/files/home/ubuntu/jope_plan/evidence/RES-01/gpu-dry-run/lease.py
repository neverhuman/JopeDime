import fcntl,os,time,json,datetime,traceback,signal
from pathlib import Path
root=Path('/home/ubuntu/jope_plan/evidence/RES-01/gpu-dry-run');lock=open('/home/ubuntu/jope_plan/qualification.lock','a');signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('qualification lock queue600s')));signal.alarm(600);fcntl.flock(lock,fcntl.LOCK_EX);signal.alarm(0);started=time.monotonic();r={'host':os.uname().nodename,'pid':os.getpid(),'acquired_utc':datetime.datetime.now(datetime.UTC).isoformat(),'lock':'/home/ubuntu/jope_plan/qualification.lock','deadline_seconds':900};(root/'lease.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
try:
 while not (root/'run-ready').exists():
  if time.monotonic()-started>600:raise TimeoutError('dry-run preparation deadline')
  time.sleep(.5)
 exec(compile((root/'run.py').read_text(),str(root/'run.py'),'exec'),{'__name__':'__main__','qualification_lock':lock,'lease_started':started})
except BaseException as e:
 (root/'lease-error.txt').write_text(traceback.format_exc());print('LEASE_ERROR',type(e).__name__,str(e),flush=True)
finally:
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();r['released_utc']=datetime.datetime.now(datetime.UTC).isoformat();r['wall_seconds']=time.monotonic()-started;(root/'lease.json').write_text(json.dumps(r,indent=2)+'\n');print('QUALIFICATION_LOCK_RELEASED',flush=True)
