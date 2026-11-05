"""Attach only the corrected own-unit guard; never rerun the percentile."""
from pathlib import Path
import sys,subprocess,time,threading
P=Path(__file__).parent;sys.path.insert(0,str(P/'source'))
import guard_h392 as guard
threading.Thread(target=lambda:guard.monitor('h392-derive.service'),daemon=True).start()
while subprocess.run(['systemctl','--user','is-active','h392-derive.service'],capture_output=True,text=True).stdout.strip() in ('active','activating'):
 time.sleep(2)
