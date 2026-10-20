"""Read-only four-node hardware, resource and GPU occupancy snapshot.

This is an observation tool, not a governor, launcher or training qualification.
No process arguments, credentials or GPU/host mutations are requested.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import subprocess


PROBE = r'''
import json,os,platform,subprocess,time,re,shutil
from pathlib import Path
def cmd(args):
 try:
  r=subprocess.run(args,text=True,capture_output=True,timeout=12)
  return {'exit':r.returncode,'out':r.stdout[-5000:],'error':r.stderr[-250:]}
 except Exception as e:return {'error':str(e)}
r={'hostname':platform.node(),'system':platform.system(),'logical_cpus':os.cpu_count(),'observed_unix':time.time()}
if Path('/proc/stat').exists():
 def cpu():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
 a=cpu();time.sleep(1);b=cpu();d=[y-x for x,y in zip(a,b)];r['aggregate_cpu_percent']=100*(sum(d)-d[3]-d[4])/sum(d)
 mem=dict(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines());r['available_ram_bytes']=int(mem['MemAvailable'].split()[0])*1024
 r['gpu']=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory','--format=csv,noheader'])
 r['gpu_compute_processes']=cmd(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'])
 r['guard']=cmd(['systemctl','show','fleetcapd.service','-p','ActiveState','-p','MainPID'])
 values=dict(l.split('=',1)for l in r['guard'].get('out','').splitlines()if'='in l)
 try:
  environment=Path('/proc/'+values['MainPID']+'/environ').read_bytes().split(b'\0')
  allowed=['FLEETCAP_TARGET','FLEETCAP_TICK','FLEETCAP_PEAK_TICKS','FLEETCAP_FALLBACK','FLEETCAP_MIN_FREE_GIB']
  r['guard_environment']={k:v for k,v in (entry.decode().split('=',1)for entry in environment if b'='in entry)if k in allowed}
 except Exception as e:r['guard_environment_unavailable']=type(e).__name__
 cgroup=Path('/sys/fs/cgroup/jarambit.slice/jarambit-evo.slice')
 r['guard_cgroup']={n:(cgroup/n).read_text().strip()for n in ['cpu.max','cgroup.freeze']if(cgroup/n).exists()}
else:
 top=cmd(['top','-l','2','-s','1','-n','0']);idle=re.findall(r'([0-9.]+)% idle',top.get('out',''))
 r['aggregate_cpu_percent']=100-float(idle[-1])if idle else None
 r['gpu']=cmd(['system_profiler','SPDisplaysDataType']);r['memory']=cmd(['sysctl','-n','hw.memsize'])
 r['guard']={'state':'not_qualified_by_this_probe','required':'owner-controlled Metal/CPU job guard'}
r['disk']=dict(zip(['total','used','free'],shutil.disk_usage('/')))
r['below_70_percent_at_sample']=r.get('aggregate_cpu_percent')is not None and r['aggregate_cpu_percent']<70
r['continuous_70_percent_ceiling_qualified']=False
r['gpu_occupancy_is_not_optimizer_progress']=True
print(json.dumps(r))
'''


def probe(host):
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, "python3 -"],
                                input=PROBE, text=True, capture_output=True, timeout=45)
        if result.returncode:
            return {"host": host, "available": False, "error": result.stderr[-500:]}
        return {"host": host, "available": True, **json.loads(result.stdout)}
    except Exception as error:
        return {"host": host, "available": False, "error": str(error)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        observations = list(pool.map(probe, ["xbabe1", "xbabe2", "xbabe3", "xbabe4"]))
    report = {"schema": "jope.research-fleet-observation/v1",
              "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "nodes": observations, "read_only": True,
              "all_nodes_below_70_at_sample": all(n.get("below_70_percent_at_sample", False) for n in observations),
              "continuous_ceiling_qualified": False}
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        with args.out.open("x") as f:
            f.write(text)
    print(text)
