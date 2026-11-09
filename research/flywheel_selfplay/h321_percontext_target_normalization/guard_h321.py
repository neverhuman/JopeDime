"""Resource monitor owns only h321-run.service; fails closed and stops its cgroup."""
import datetime,json,os,subprocess,threading,time
from pathlib import Path
DATA=Path('/home/ubuntu/jope-data/h321-solhwv')

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
        if utc>=datetime.datetime(2026,11,4,15,15,tzinfo=datetime.timezone.utc):reason.append('compute deadline')
        if reason:
            (DATA/'STOP.json').write_text(json.dumps(dict(reason=reason,**r)))
            subprocess.run(['systemctl','--user','stop','h321-run.service'],check=False)
            return

if __name__=='__main__':
    DATA.mkdir(exist_ok=True,parents=True)
    threading.Thread(target=monitor,daemon=True).start()
    import run_h321
    run_h321.main()
