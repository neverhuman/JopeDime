from pathlib import Path
import subprocess,time,os,signal,datetime,json
import psutil
P=Path(__file__).parent;child=subprocess.Popen(['python3',str(P/'relay.py')]);paused=False
while child.poll() is None:
 time.sleep(2);cpu=psutil.cpu_percent(interval=.2);load=os.getloadavg()[0];mem=psutil.virtual_memory().available;ok=cpu<=70 and load<=int(.85*os.cpu_count()) and mem>=20*2**30
 if not ok and not paused:child.send_signal(signal.SIGSTOP);paused=True
 elif ok and paused:child.send_signal(signal.SIGCONT);paused=False
 with (P/'relay-resources.jsonl').open('a') as f:f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_percent=cpu,load1=load,mem_available_bytes=mem,own_courier_paused=paused))+'\n')
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,5,18,15,tzinfo=datetime.timezone.utc):
  child.terminate()
  if paused:child.send_signal(signal.SIGCONT)
  break
raise SystemExit(child.wait())
