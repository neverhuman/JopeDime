"""Fresh memory/load gate accounting authorized future cgroup reservations."""
import os
from pathlib import Path
from common import Clock,need,GLOBAL
from history_collect import command

def observe(epoch,runner=command,meminfo=Path('/proc/meminfo')):
    clock=Clock(min(epoch['work_deadline_unix'],__import__('time').time()+30),30)
    lines=dict((p[0].rstrip(':'),int(p[1])*1024)for p in(map(str.split,meminfo.read_text().splitlines()))if len(p)>=2)
    other=0;observed=[];seen=set()
    for r in epoch['resource_reservations']:
        unit=r['unit'];need(unit not in seen and unit.endswith('.service')and '/'not in unit,'unique reserved unit');seen.add(unit)
        v=dict(l.split('=',1)for l in runner(['systemctl','--user','show',unit,'--property=Id,ActiveState,MemoryMax,MemoryCurrent'],clock).splitlines()if'='in l)
        need(v.get('Id')==unit and type(r['memory_max_bytes'])is int and r['memory_max_bytes']>0,'declared reserved cgroup')
        current=int(v['MemoryCurrent'])if v.get('MemoryCurrent','[not set]').isdigit()else 0
        if v.get('ActiveState')in('active','activating'):
            need(v.get('MemoryMax','').isdigit()and int(v['MemoryMax'])<=r['memory_max_bytes'],'live cgroup exceeds declared reservation')
        other+=max(0,r['memory_max_bytes']-current);observed.append({'unit':unit,'current':current,'reservation':r['memory_max_bytes']})
    active=0;used=0
    for s in epoch['shards']:
        v=dict(l.split('=',1)for l in runner(['systemctl','--user','show',s['unit'],'--property=Id,ActiveState,MemoryMax,MemoryCurrent'],clock).splitlines()if'='in l)
        if v.get('ActiveState')in('active','activating'):
            need(v.get('MemoryMax','').isdigit()and int(v['MemoryMax'])<=4*1024**3,'own worker cgroup cap')
            active+=1;used+=int(v['MemoryCurrent'])if v.get('MemoryCurrent','').isdigit()else 0
    need(active<=epoch['authorized_concurrency'],'authorized active worker count exceeded')
    reserved=max(0,epoch['authorized_concurrency']*4*1024**3-used)
    required=24*1024**3+other+reserved
    need(lines['MemAvailable']>=required,'24GiB foreign reserve plus future owned allocations')
    need(os.getloadavg()[0]<=max(1,os.cpu_count())*1.5,'host load envelope')
    return {'available_bytes':lines['MemAvailable'],'required_bytes':required,'other_residual_bytes':other,
            'epoch_residual_bytes':reserved,'epoch_active_workers':active,'declared_other_units':observed,'load1':os.getloadavg()[0]}
