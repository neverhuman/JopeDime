from pathlib import Path
import subprocess,json,hashlib,shlex,datetime
P=Path(__file__).parent;code='''from pathlib import Path
import shutil,json,datetime,os
D=Path('/home/ubuntu/jope-data/h376-x1');assert not D.exists();free=shutil.disk_usage('/').free
mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
r=dict(wave='H376',host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='initial_metadata_resource_preflight',root_free_bytes=free,required_root_free_bytes=80*2**30,mem_available_bytes=mem,host_nproc=sum(x.startswith('cpu') and x.split()[0][3:].isdigit() for x in Path('/proc/stat').read_text().splitlines()),capacity_probes=0,CPU60_windows=0,load60_windows=0,optimizer_fits=0,outer_scores=0,compute_units_launched=0,reason=['root free <80GiB'],initial_observed_root_free_bytes=70442295296,status='INCOMPLETE_RESOURCE_GUARD',resource_gate='Initial observed root-free failure closes this allocation; fresh metadata is custody readback, not a new admission attempt')
D.mkdir(parents=True);(D/'STOP.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
'''
cmd=['ssh','-T','-o','BatchMode=yes','xbabe1','nice -n 19 taskset -c 35 python3 -c '+shlex.quote(code)];raw=subprocess.check_output(cmd);r=json.loads(raw);assert r['initial_observed_root_free_bytes']<r['required_root_free_bytes'];(P/'xbabe1-resource-stop.json').write_text(json.dumps(dict(**r,metadata_capture_exit_status=0,command=cmd,source_sha256=hashlib.sha256(code.encode()).hexdigest()),indent=2)+'\n');print(json.dumps(r))
