"""Owner's >=85% idle threshold over one complete five-second interval."""
import json
import subprocess

ASSIGNED = (0,2,3,4,5,6,7,10,12,14)


def eligible_from_report(report):
    stats=report['sysstat']['hosts'][0]['statistics']
    assert len(stats)==1, 'Exactly one complete interval required'
    values={int(x['cpu']):float(x['idle']) for x in stats[0]['cpu-load'] if x['cpu']!='all'}
    idle={c:min(values[c],values[c+64]) for c in ASSIGNED}
    assert all(0<=v<=100 for v in values.values())
    return [c for c,v in idle.items() if v>=85][:8],idle


def probe():
    raw=subprocess.check_output(['mpstat','-P',','.join(map(str,(*ASSIGNED,*(c+64 for c in ASSIGNED)))),'5','1','-o','JSON'],text=True)
    eligible,idle=eligible_from_report(json.loads(raw))
    return eligible,idle,raw
