"""Resource monitor owns only h328-run.service; fails closed and stops its cgroup."""
import datetime,json,os,subprocess,threading,time
from pathlib import Path
DATA=Path('/home/ubuntu/jope-data/h328-solhwv')

def cpu():
    a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a[:8]),a[3]+a[4]

def monitor():
    prev=cpu()
    while True:
        time.sleep(5)
        total,idle=cpu();dt=total-prev[0];usage=100*(1-(idle-prev[1])/dt) if dt else 0;prev=(total,idle)
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        size=0
        for p in DATA.rglob('*'):
            try:
                if p.is_file():size+=p.stat().st_size
            except FileNotFoundError:
                pass # Own verified archive reclamation can race a telemetry snapshot.
        utc=datetime.datetime.now(datetime.timezone.utc)
        r=dict(utc=utc.isoformat(),cpu_percent=usage,load1=os.getloadavg()[0],mem_available_bytes=mem,scratch_bytes=size)
        with (DATA/'resource.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        reason=[]
        if usage>85:reason.append('host CPU >85%')
        if os.getloadavg()[0]>85:reason.append('host load >85')
        if mem<20*1024**3:reason.append('MemAvailable <20GiB')
        if size>2_000_000_000:reason.append('scratch >2GB')
        if utc>=datetime.datetime(2026,11,4,20,45,tzinfo=datetime.timezone.utc):reason.append('compute deadline')
        if reason:
            (DATA/'STOP.json').write_text(json.dumps(dict(reason=reason,**r)))
            subprocess.run(['systemctl','--user','stop','h328-run.service'],check=False)
            return

if __name__=='__main__':
    DATA.mkdir(exist_ok=True,parents=True)
    threading.Thread(target=monitor,daemon=True).start()
    import sys
    source=Path(__file__).resolve().parent
    for phase in ['fetch','preflight','regress','fit','score']:
        command=[sys.executable,str(source/'fetch_base_h328.py')] if phase=='fetch' else ([sys.executable,str(source.parent/'input_preflight.py')] if phase=='preflight' else [sys.executable,str(source/'run_h328.py'),phase])
        if phase=='fit':
            # Sample the owner's assigned cores immediately before fits, after IO-heavy regression.
            def cores():
                return {int(x.split()[0][3:]):list(map(int,x.split()[1:9])) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('cpu') and x.split()[0][3:].isdigit()}
            while True:
                before=cores();time.sleep(3);after=cores();eligible=[];readback={}
                for c in [13,15,60,62,77,79,124,126]:
                    a,b=before[c],after[c];total=sum(b)-sum(a);idle=(b[3]-a[3])/total*100
                    readback[c]=idle
                    if idle>=85:eligible.append(c)
                if eligible==[13,15,60,62,77,79,124,126]:
                    eligible=[13,15,60,62];break
                if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,4,20,35,tzinfo=datetime.timezone.utc):raise RuntimeError('No idle assigned set before original deadline; INCOMPLETE')
                print(json.dumps(dict(waiting_for_idle_cores=readback)),flush=True);time.sleep(10)
            env=dict(os.environ,H328_WORKERS=str(len(eligible)))
            (DATA/'pre_fit_resource.json').write_text(json.dumps(dict(eligible_cores=eligible,idle_percent=readback)))
            command=['taskset','-c',','.join(map(str,eligible)),*command]
            subprocess.run(command,check=True,env=env)
        else:
            if phase=='score' and datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,4,20,40,tzinfo=datetime.timezone.utc):raise RuntimeError('Insufficient complete scoring window; INCOMPLETE')
            subprocess.run(command,check=True)
