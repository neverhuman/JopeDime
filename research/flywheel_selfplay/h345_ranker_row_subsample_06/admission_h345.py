"""One CPU admission probe and fixed conservative resource forecasts; never fit."""
from pathlib import Path
import datetime,time,math
ORDER=[15,19,20,21,22,23,24,25,26,34,35,36,37,55,60,62]
DEADLINE=datetime.datetime(2026,11,5,5,15,tzinfo=datetime.timezone.utc)
def select(idle):return [c for c in ORDER if idle[c]>=85 and idle[c+64]>=85][:8]
def probe():
 def sample():return {int(x.split()[0][3:]):list(map(int,x.split()[1:9])) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('cpu') and x.split()[0][3:].isdigit()}
 start=datetime.datetime.now(datetime.timezone.utc);before=sample();time.sleep(5);after=sample();idle={c:100*(after[c][3]-before[c][3])/(sum(after[c])-sum(before[c])) for c in ORDER+[c+64 for c in ORDER]};chosen=select(idle)
 return dict(probe_started_utc=start.isoformat(),probe_finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=5,probes=1,priority=ORDER,idle_percent=idle,admitted=chosen,status='PASS' if len(chosen)>=4 else 'INCOMPLETE_CPU_ADMISSION')
def forecast(work_seconds,workers,now):
 assert math.isfinite(work_seconds) and work_seconds>=0 and 4<=workers<=8
 seconds=work_seconds/(workers*.9)*1.15+300;end=now+datetime.timedelta(seconds=seconds)
 return dict(now_utc=now.isoformat(),work_seconds=work_seconds,workers=workers,efficiency=.9,safety=1.15,reserve_seconds=300,forecast_seconds=seconds,forecast_finish_utc=end.isoformat(),compute_stop_utc=DEADLINE.isoformat(),fits=end<DEADLINE)
