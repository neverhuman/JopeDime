"""Host-adapter backstop for the preregistered first-task forecast, no fits/probes."""
from pathlib import Path
import json,subprocess,time,datetime,os
P=Path(__file__).parent;unit='h362-x1-run.service';assert os.uname().nodename.split('.')[0]=='xbabe1'
while True:
 forecast=P/'first-cell-forecast.json'
 if forecast.exists():
  value=json.loads(forecast.read_text())
  if not value['launch_full_36']:
   result=subprocess.run(['systemctl','--user','stop',unit]);record=dict(status='FORECAST_MISS_STOP_ENFORCED',own_unit=unit,stop_exit_status=result.returncode,forecast=value,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'forecast-watch-receipt.json').write_text(json.dumps(record,indent=2)+'\n');raise SystemExit(result.returncode)
  (P/'forecast-watch-receipt.json').write_text(json.dumps(dict(status='FORECAST_PASS_NO_ACTION',own_unit=unit,forecast_end=value['forecast_end'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n');break
 if (P/'unit-exit.json').exists():
  (P/'forecast-watch-receipt.json').write_text(json.dumps(dict(status='UNIT_EXIT_BEFORE_FIRST_COMPLETED_TASK_NO_ACTION',own_unit=unit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n');break
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,5,14,30,tzinfo=datetime.timezone.utc):break
 time.sleep(2)
