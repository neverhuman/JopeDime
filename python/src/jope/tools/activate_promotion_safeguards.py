#!/usr/bin/env python3
"""Deploy reviewed supervisor/reporting code after the current durable GPU stage."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import operational_release
from promotion_gate import immutable
from resume_expired_cycle import atomic, read, require, sha

OVERRIDE = 'zz-promotion-safeguards.conf'
SUPERSEDED = {'zzz-flywheel-v2.conf', 'zzzz-flywheel-v3.conf'}


def prepare_overrides(output, services, system=Path('/etc/systemd/system')):
    """Journal exact configuration before removing known superseding launches."""
    journal=output/'service-overrides.before.json'
    if journal.exists():return read(journal)
    entries=[]
    for service in services:
        directory=system/(service+'.d')
        target=directory/OVERRIDE
        entries.append(dict(path=str(target),content=target.read_text() if target.exists() else None,remove=False))
        for path in sorted(directory.glob('*.conf')):
            if path.name<=OVERRIDE:continue
            lines=[line.strip() for line in path.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith('#')]
            if not any(line.startswith(('ExecStart=','ExecStartPre=')) for line in lines):continue
            require(path.name in SUPERSEDED and all(line=='[Service]' or line.startswith('ExecStart=') for line in lines),
                    'unreviewed superseding service configuration: '+str(path))
            entries.append(dict(path=str(path),content=path.read_text(),remove=True))
    immutable(journal,entries)
    return entries


def remove_superseded(entries):
    for entry in entries:
        path=Path(entry['path'])
        if entry['remove'] and path.exists():
            require(path.read_text()==entry['content'],'service override changed during activation: '+str(path))
            subprocess.run(['sudo','-n','rm','--',str(path)],check=True)


def verify_effective_launches(unit_files):
    for service,content in unit_files.items():
        expected=next(line.partition('=')[2] for line in content.splitlines()
                      if line.startswith('ExecStart=') and line!='ExecStart=')
        effective=subprocess.check_output(['systemctl','show',service,'-p','ExecStart','--value'],text=True)
        require('argv[]='+expected+' ;' in effective,'effective service launch differs from reviewed release: '+service)


def health_launch(unit):
    return next(line for line in reversed(unit.splitlines())
                if line.startswith('ExecStart=') and '--baseline' in line)


def wait_controller_identity(root,manifest,timeout=60):
    deadline=time.monotonic()+timeout
    while True:
        path=root/'runs/continuous-progress.json'
        if path.exists() and read(path).get('manifest_sha256')==sha(manifest):return
        require(time.monotonic()<deadline,'controller has not published the activated manifest identity')
        time.sleep(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--operational-release',type=Path)
    p.add_argument('--gp-emission-release',type=Path)
    p.add_argument('--recover-stopped', action='store_true',
                   help='recover a failed/stopped controller after its systemd cgroup is empty')
    args = p.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    require(source.is_relative_to(root) and output.is_relative_to(root), 'activation outside workspace')
    require(not (output/'complete.json').exists(), 'activation already complete')
    old = read(args.manifest)
    state_path = root/old['output']/'state.json'
    control_path = root/'runs/continuous-control.json'
    control_before = read(output/'control.before.json') if (output/'control.before.json').exists() else read(control_path)
    new = dict(old, bindings=dict(old['bindings']))
    if args.gp_emission_release:
        from gp_emission_handoff import verify_release
        cpu=verify_release(root,args.gp_emission_release,old)
        new['cpu_executable']=str(cpu.relative_to(root))
        new['bindings'][new['cpu_executable']]=sha(cpu)
        new['bindings'][str(args.gp_emission_release.relative_to(root))]=sha(args.gp_emission_release)
    release=operational_release.verify(root,args.operational_release) if args.operational_release else None
    if release:
        new.update(release['settings'])
        for key in ('packing_executable','training_executable','monitor_executable'):
            new['bindings'][new[key]]=sha(root/new[key])
        new['bindings'][str(args.operational_release.relative_to(root))]=sha(args.operational_release)
        workers=read(root/release['workers']['path'])
        monitor_baseline=(root/release['workers']['path']).parent/'monitor-0.json'
        require(sha(monitor_baseline)==workers['stages']['monitor-0']['receipt_sha256'],
                'qualified monitor reference changed')
        new['bindings'][str(monitor_baseline.relative_to(root))]=sha(monitor_baseline)
    for path in sorted(source.glob('*.py')):
        new['bindings'][str(path.relative_to(root))] = sha(path)
    manifest = output/'manifest.json'
    immutable(manifest, new)
    if release:immutable(output/'fallback-manifest.json',operational_release.fallback(new))
    subprocess.run(['python3', '-B', str(source/'continuous_learning.py'), '--root', str(root),
                    '--manifest', str(manifest), '--check'], check=True)
    gp_pid = subprocess.check_output(['systemctl','show','jope-parallel-gp.service','-p','MainPID','--value'], text=True).strip()
    pid = int(subprocess.check_output(['systemctl','show','jope-parallel-campaign.service','-p','MainPID','--value'], text=True))
    if args.recover_stopped:
        status = subprocess.check_output(['systemctl', 'show', 'jope-parallel-campaign.service',
            '-p', 'ActiveState', '--value'], text=True).strip()
        cgroup = subprocess.check_output(['systemctl', 'show', 'jope-parallel-campaign.service',
            '-p', 'ControlGroup', '--value'], text=True).strip()
        require(pid == 0 and status in ('failed', 'inactive') and not cgroup,
                'stopped recovery requires an empty controller cgroup and no main PID')
    else:
        require(pid > 1, 'active supervisor required for durable-stage handoff')
    immutable(output/'control.before.json', control_before)
    request_path=output/'request.json'
    if request_path.exists():
        request=read(request_path)
        require(request['old_manifest_sha256']==sha(args.manifest)
                and request['new_manifest_sha256']==sha(manifest),'activation resume identity changed')
    else:
        immutable(request_path, dict(timestamp=time.time(), old_manifest=str(args.manifest),
            old_manifest_sha256=sha(args.manifest), new_manifest_sha256=sha(manifest), supervisor_pid=pid, gp_pid=gp_pid))
    state=read(state_path)
    require(state['manifest_sha256'] in (sha(args.manifest),sha(manifest)),
            'another deployment now owns the controller')
    if not (output/'state.preflight.json').exists():immutable(output/'state.preflight.json',state)
    overrides=prepare_overrides(output,('jope-parallel-campaign.service',
        'jope-parallel-reports.service','jope-parallel-health.service'))
    cgroup=subprocess.check_output(['systemctl','show','jope-parallel-campaign.service',
        '-p','ControlGroup','--value'],text=True).strip()
    if pid>1:operational_release.stop_controller(root,state_path,pid,cgroup,time.monotonic()+1200)
    operational_release.recover_controller_stops(state_path)
    state = read(state_path)
    if not (output/'state.before.json').exists():immutable(output/'state.before.json',state)
    if state['manifest_sha256']!=sha(manifest):
        state=operational_release.migrate(state,sha(args.manifest),sha(manifest),
            release['teaching_sources'] if release else None)
    atomic(state_path, state)
    atomic(control_path, dict(control_before, manifest_sha256=sha(manifest), timestamp=time.time()))
    if release:
        pool_unit=('[Unit]\nDescription=JOPE resident GPU task workers\nAfter=network.target\n'
            '[Service]\nType=simple\nUser=ubuntu\nWorkingDirectory='+str(root)+'\n'
            'Environment=LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib\n'
            'ExecStart=/usr/bin/python3 -B '+str(source/'resident_gpu_pool.py')+' --root '+str(root)+
            ' --binary '+str(root/new['training_executable'])+' --pool '+str(root/new['gpu_pool'])+' --world 8\n'
            'Restart=on-failure\nRestartSec=2\nKillMode=control-group\nTimeoutStopSec=30\nLimitNOFILE=65536\n'
            'CPUAffinity=0 1 2 3 4 5 6 7 62 63 64 65 66 67 68 69\n[Install]\nWantedBy=multi-user.target\n')
        path=output/'jope-parallel-resident-gpu.service';path.write_text(pool_unit)
        subprocess.run(['sudo','-n','install','-m','644',str(path),'/etc/systemd/system/jope-parallel-resident-gpu.service'],check=True)
    unit_files = {
        'jope-parallel-campaign.service': '[Service]\nExecStart=\nExecStart=/usr/bin/python3 -B '+str(source/'continuous_learning.py')+' --root '+str(root)+' --manifest '+str(manifest)+'\n',
        'jope-parallel-reports.service': '[Service]\nExecStart=\nExecStart=/usr/bin/python3 -B '+str(source/'continuous_teacher_progress.py')+' --root '+str(root)+' --manifest '+str(manifest)+' --interval 60\n',
    }
    health_unit = subprocess.check_output(['systemctl','cat','jope-parallel-health.service'], text=True)
    health_start = health_launch(health_unit)
    import shlex
    health_args = shlex.split(health_start.split('=',1)[1])
    baseline = str(monitor_baseline) if release else health_args[health_args.index('--baseline')+1]
    health_output = health_args[health_args.index('--output')+1]
    unit_files['jope-parallel-health.service'] = ('[Service]\nExecStartPre=\nExecStartPre=/usr/bin/sha256sum --check '+str(output/'SHA256SUMS')+
        '\nExecStart=\nExecStart=/usr/bin/python3 -B '+str(source/'continuous_status.py')+' --root '+str(root)+' --manifest '+str(manifest)+
        ' --baseline '+baseline+' --output '+health_output+'\n')
    (output/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f)+'\n' for f in sorted(source.glob('*.py'))))
    rollback = ['#!/bin/bash', 'set -euo pipefail', '# Restores service configuration; preserves all newer state and evidence.',
                'sudo -n systemctl stop jope-parallel-campaign.service jope-parallel-reports.service',
                '# Restore the old manifest hash in state/control only after reviewing newer candidates.']
    for service, content in unit_files.items():
        target = Path('/etc/systemd/system')/(service+'.d')/OVERRIDE
        if target.exists() and not (output/(service+'.before.conf')).exists():
            (output/(service+'.before.conf')).write_bytes(target.read_bytes())
        path = output/(service+'.conf')
        path.write_text(content)
        subprocess.run(['sudo','-n','mkdir','-p',str(target.parent)], check=True)
        subprocess.run(['sudo','-n','install','-m','644',str(path),str(target)], check=True)
        before = output/(service+'.before.conf')
        rollback.append('sudo -n install -m 644 '+str(before)+' '+str(target) if before.exists()
                        else 'sudo -n rm -- '+str(target))
    for number,entry in enumerate(overrides):
        if entry['remove']:
            before=output/f'superseded-{number}.conf'
            before.write_text(entry['content'])
            rollback.append('sudo -n install -m 644 '+str(before)+' '+entry['path'])
    rollback.append('sudo -n systemctl daemon-reload')
    (output/'rollback-config.sh').write_text('\n'.join(rollback)+'\n')
    remove_superseded(overrides)
    subprocess.run(['sudo','-n','systemctl','daemon-reload'], check=True)
    verify_effective_launches(unit_files)
    subprocess.run(['sudo','-n','systemctl','reset-failed','jope-parallel-campaign.service'], check=True)
    if release:subprocess.run(['sudo','-n','systemctl','enable','--now','jope-parallel-resident-gpu.service'],check=True)
    subprocess.run(['sudo','-n','systemctl','start','jope-parallel-campaign.service'], check=True)
    subprocess.run(['sudo','-n','systemctl','restart','jope-parallel-reports.service'], check=True)
    wait_controller_identity(root,manifest)
    subprocess.run(['sudo','-n','systemctl','start','jope-parallel-health.service'], check=True)
    after = read(state_path)
    require(after['checkpoint'] == after['serving_checkpoint'] == state['checkpoint'], 'activation changed serving identity')
    require(after['serving_status'] == state['serving_status']
            and after['serving_checkpoint_sha256'] == state['serving_checkpoint_sha256'],
            'activation changed serving qualification or identity')
    require(subprocess.check_output(['systemctl','show','jope-parallel-gp.service','-p','MainPID','--value'], text=True).strip() == gp_pid,
            'GP worker service restarted during activation')
    immutable(output/'complete.json', dict(timestamp=time.time(), manifest_sha256=sha(manifest),
        serving_checkpoint=after['serving_checkpoint'], serving_checkpoint_sha256=after['serving_checkpoint_sha256'],
        serving_status=after['serving_status'], gp_uninterrupted=True, final_tests_opened=False))
    print(json.dumps(read(output/'complete.json')))


if __name__ == '__main__':
    main()
