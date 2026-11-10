"""A1 uses the original resource monitor, unchanged deadline, separate fit/score processes."""
from pathlib import Path
import threading,subprocess,sys,os,json,datetime
import guard_h325 as guard
P=Path(__file__).resolve().parent.parent;D=guard.DATA
if __name__=='__main__':
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,4,17,30,tzinfo=datetime.timezone.utc)
 threading.Thread(target=guard.monitor,daemon=True).start()
 subprocess.run([sys.executable,str(P/'source/resume_h325.py')],check=True)
 # All36 fits are mandatory; reserve5min for complete once-only scoring, never partial-set scoring.
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,4,17,25,tzinfo=datetime.timezone.utc):raise RuntimeError('Insufficient scoring window; INCOMPLETE without outer scoring')
 subprocess.run([sys.executable,str(P/'source/run_h325.py'),'score'],check=True)
