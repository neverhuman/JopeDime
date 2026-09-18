#!/usr/bin/env python3
"""Return to reference packing and standalone workers, retaining newer learning state."""
import argparse
import subprocess
import time
from pathlib import Path

import operational_release
from promotion_gate import immutable
from resume_expired_cycle import atomic, read, require, sha


def run(root,deployment):
    manifest=deployment/'manifest.json';fallback=deployment/'fallback-manifest.json'
    current=read(manifest);state_path=root/current['output']/'state.json'
    require(read(state_path)['manifest_sha256']==sha(manifest),'another deployment now owns the controller')
    require(read(fallback)==operational_release.fallback(current),'fallback manifest changed')
    source=Path(__file__).resolve().parent
    control=root/'runs/continuous-control.json';before=read(control)
    pid=int(subprocess.check_output(['systemctl','show','jope-parallel-campaign.service','-p','MainPID','--value'],text=True))
    group=subprocess.check_output(['systemctl','show','jope-parallel-campaign.service','-p','ControlGroup','--value'],text=True).strip()
    if pid>1:operational_release.stop_controller(root,state_path,pid,group,time.monotonic()+1200)
    else:
        processes=Path('/sys/fs/cgroup')/group.lstrip('/')/'cgroup.procs'
        require(not group or not processes.exists() or not processes.read_text().strip(),
                'stopped controller still has active children')
    operational_release.recover_controller_stops(state_path)
    subprocess.run(['sudo','-n','systemctl','stop','jope-parallel-campaign.service','jope-parallel-reports.service',
                    'jope-parallel-resident-gpu.service'],check=True)
    state=read(state_path)
    immutable(deployment/'rollback-state.before.json',state)
    after=operational_release.migrate(state,sha(manifest),sha(fallback))
    atomic(state_path,after)
    for service,script,extra in [('jope-parallel-campaign.service','continuous_learning.py',''),
                                ('jope-parallel-reports.service','continuous_teacher_progress.py',' --interval 60')]:
        target=Path('/etc/systemd/system')/(service+'.d')/'zz-promotion-safeguards.conf'
        content='[Service]\nExecStart=\nExecStart=/usr/bin/python3 -B '+str(source/script)+' --root '+str(root)+' --manifest '+str(fallback)+extra+'\n'
        path=deployment/(service+'.fallback.conf');path.write_text(content)
        subprocess.run(['sudo','-n','install','-m','644',str(path),str(target)],check=True)
    # Keep the observer attached to the same preserved state with the fallback identity.
    health=deployment/'jope-parallel-health.service.conf'
    if health.is_file():
        target=Path('/etc/systemd/system/jope-parallel-health.service.d/zz-promotion-safeguards.conf')
        replacement=deployment/'jope-parallel-health.service.fallback.conf'
        replacement.write_text(health.read_text().replace(str(manifest),str(fallback)))
        subprocess.run(['sudo','-n','install','-m','644',str(replacement),str(target)],check=True)
    atomic(control,dict(before,manifest_sha256=sha(fallback),timestamp=time.time()))
    subprocess.run(['sudo','-n','systemctl','disable','jope-parallel-resident-gpu.service'],check=True)
    subprocess.run(['sudo','-n','systemctl','daemon-reload'],check=True)
    subprocess.run(['sudo','-n','systemctl','start','jope-parallel-campaign.service','jope-parallel-reports.service'],check=True)
    immutable(deployment/'rollback-complete.json',dict(timestamp=time.time(),manifest_sha256=sha(fallback),
        serving_checkpoint=after['serving_checkpoint'],serving_status=after['serving_status'],
        candidates=len(after['candidates']),consumed_snapshots=len(after['consumed_training_snapshots']),
        promotion_decisions=len(after['promotion_decisions']),final_tests_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--deployment',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.deployment.resolve())
