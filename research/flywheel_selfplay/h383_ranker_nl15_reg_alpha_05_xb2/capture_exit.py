from pathlib import Path
import os,json,datetime
p=Path(os.environ.get('H383_EXIT_RECEIPT','/home/ubuntu/jope-data/h383-hwv/control/unit-exit.json'))
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),EXIT_CODE=os.environ.get('EXIT_CODE'),EXIT_STATUS=os.environ.get('EXIT_STATUS'),SERVICE_RESULT=os.environ.get('SERVICE_RESULT'),INVOCATION_ID=os.environ.get('INVOCATION_ID'))
with p.open('w') as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
