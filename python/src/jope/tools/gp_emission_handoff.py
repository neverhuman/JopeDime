"""Drain committed GP searches before upgrading the identity-emitting service."""
import argparse
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import operational_release
from promotion_gate import immutable
from resume_expired_cycle import read, require, sha

SERVICE='jope-parallel-gp.service'


def process(pid):
    try:
        row=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,state=row[0],parent=int(row[1]),birth=int(row[19]),exit_code=int(row[49]))
    except FileNotFoundError:return None


def same_process(identity):
    current=process(identity['pid'])
    return current is not None and current['birth']==identity['birth']


def resume(output):
    request=read(output/'request.json');parent=request['supervisor']
    if (output/'stop-intent.json').exists():return
    if same_process(parent):os.kill(parent['pid'],signal.SIGCONT)


def verify_release(root,path,old_manifest):
    receipt=read(path);request_path=path.parent/'request.json';request=read(request_path)
    require(receipt['status']=='passed' and sha(request_path)==receipt['request_sha256'],
            'GP emission qualification is incomplete or changed')
    binary=root/request['executable'];reference=root/request['reference']
    build_path=binary.parent.parent/'target-build.json';build=read(build_path)
    require(sha(build_path)==request['build_sha256'] and build['status']=='passed'
            and build['bindings']['jope-parallel-native']==request['executable_sha256'],
            'GP emission target build changed')
    require(sha(binary)==receipt['executable_sha256']==request['executable_sha256'],
            'qualified GP executable changed')
    require(sha(reference)==receipt['reference_sha256']==request['reference_sha256']
            ==sha(root/old_manifest['cpu_executable']), 'GP reference evaluator changed')
    return binary


def committed(root,children):
    results=[]
    for child in children:
        current=process(child['pid'])
        require(current is not None and current['birth']==child['birth'], 'GP child identity disappeared during drain')
        if current['state']!='Z':return None
        require(current['exit_code']==0,'a GP search failed during drain; resume ordinary recovery')
        output=root/child['output'];receipt=read(output/'search.json');rows=read(output/'observations.json')
        require(receipt['schema']=='parallel.search.v1' and receipt['completed_evaluations']==len(rows)
                and 0<len(rows)<=1000, 'GP child has no complete evaluation ledger')
        results.append(dict(output=child['output'],pid=child['pid'],birth=child['birth'],
            search_sha256=sha(output/'search.json'),observations_sha256=sha(output/'observations.json'),
            completed_evaluations=len(rows)))
    return results


def drain(root,output,request,seconds=660):
    parent=request['supervisor'];require(same_process(parent),'GP supervisor identity changed before drain')
    os.kill(parent['pid'],signal.SIGSTOP)
    try:
        deadline=time.monotonic()+seconds
        while process(parent['pid'])['state']!='T':
            require(time.monotonic()<deadline,'GP supervisor did not stop dispatch');time.sleep(.05)
        journal=output/'children.json'
        if journal.exists():children=read(journal)
        else:
            children=[]
            pids=Path(f"/proc/{parent['pid']}/task/{parent['pid']}/children").read_text().split()
            for pid in pids:
                identity=process(int(pid));require(identity and identity['parent']==parent['pid'],'unexpected GP child')
                args=Path('/proc',pid,'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                # Already exited children have no cmdline; resolve their last
                # dispatched destination from the supervisor's active rows.
                if identity['state']=='Z':
                    require(False,'a search exited before child capture; resume and retry the drain')
                require('search' in args and '--output' in args and request['executable'] in args,
                        'unrecognized child of GP service')
                destination=Path(args[args.index('--output')+1])
                if not destination.is_absolute():destination=root/destination
                require(destination.is_relative_to(root),'GP child output escapes workspace')
                children.append(dict(identity,output=str(destination.relative_to(root))))
            immutable(journal,children)
        while True:
            require(same_process(parent),'GP supervisor disappeared while draining')
            results=committed(root,children)
            if results is not None:
                immutable(output/'drained.json',dict(timestamp=time.time(),children=results,
                    request_sha256=sha(output/'request.json')))
                return results
            require(time.monotonic()<deadline,'GP drain deadline exceeded; searches will continue')
            time.sleep(1)
    except BaseException:
        resume(output)
        raise


def stop_drained(root,output,request):
    """Keep dispatch stopped while disabling restart and retiring the parent."""
    import stopped_unit
    parent=request['supervisor'];children=read(output/'children.json')
    def boundary():
        require(committed(root,children) is not None,'drained GP children changed')
        expected={row['pid'] for row in children}
        actual={int(pid) for pid in Path(f"/proc/{parent['pid']}/task/{parent['pid']}/children").read_text().split()}
        require(actual==expected,'GP child set changed after drain')
    if not (output/'stop-intent.json').exists():
        require(same_process(parent) and process(parent['pid'])['state']=='T',
                'drained supervisor must remain stopped')
        boundary()
        immutable(output/'stop-intent.json',dict(supervisor=parent,drained_sha256=sha(output/'drained.json'),
            protocol='disable_restart_kill_stopped_parent_then_stop_unit_v1'))
    stopped_unit.stop(SERVICE,parent['pid'],output/'unit-stop',boundary)


def main(root,manifest,release,output):
    output.mkdir(parents=True,exist_ok=True);old=read(manifest)
    binary=verify_release(root,release,old)
    require(not (output/'complete.json').exists(),'GP handoff already complete')
    if (output/'request.json').exists():request=read(output/'request.json')
    else:
        pid=int(subprocess.check_output(['systemctl','show',SERVICE,'-p','MainPID','--value'],text=True))
        identity=process(pid);require(identity is not None,'GP supervisor unavailable')
        executable=str(Path(f'/proc/{pid}/exe').resolve())
        require(sha(Path(executable))==sha(root/old['cpu_executable']),'unexpected live GP evaluator')
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
        require(args[args.index('--workers')+1]=='104' and args[args.index('--threads')+1]=='1'
                and '--wait-for-work' in args,'GP allocation/protocol changed')
        request=dict(schema='parallel.gp_emission_handoff.v1',timestamp=time.time(),supervisor=identity,
            executable=executable,executable_sha256=sha(Path(executable)),command=args,
            candidate=str(binary),candidate_sha256=sha(binary),manifest=str(manifest),
            manifest_sha256=sha(manifest),release=str(release),release_sha256=sha(release))
        immutable(output/'request.json',request)
        immutable(output/'service.before.json',dict(content=subprocess.check_output(['systemctl','cat',SERVICE],text=True)))
    require(request['manifest_sha256']==sha(manifest) and request['release_sha256']==sha(release)
            and request['candidate_sha256']==sha(binary),'GP handoff resume inputs changed')
    try:
        if not (output/'drained.json').exists():drain(root,output,request)
        if same_process(request['supervisor']) or (output/'unit-stop/request.json').exists():
            stop_drained(root,output,request)
        override=Path('/etc/systemd/system')/(SERVICE+'.d')/'zz-flywheel-emission.conf'
        command=[str(binary),*request['command'][1:]]
        content='[Service]\nEnvironment=LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib\nExecStart=\nExecStart='+shlex.join(command)+'\n'
        if not (output/'override.before.json').exists():
            immutable(output/'override.before.json',dict(path=str(override),content=override.read_text() if override.exists() else None))
        path=output/'gp.conf';path.write_text(content)
        subprocess.run(['sudo','-n','install','-m','644',str(path),str(override)],check=True)
        subprocess.run(['sudo','-n','systemctl','daemon-reload'],check=True)
        from activate_promotion_safeguards import verify_effective_launches
        verify_effective_launches({SERVICE:content})
        subprocess.run(['sudo','-n','systemctl','start',SERVICE],check=True)
        deadline=time.monotonic()+600
        while True:
            gp=read(root/'runs/gp-service.json')
            if gp.get('executable_sha256')==sha(binary) and gp.get('slot_leases') is True:break
            require(time.monotonic()<deadline,'new GP service did not publish its identity');time.sleep(1)
        state_path=root/old['output']/'state.json'
        pid=int(subprocess.check_output(['systemctl','show','jope-parallel-campaign.service','-p','MainPID','--value'],text=True))
        if pid>1:
            cgroup=subprocess.check_output(['systemctl','show','jope-parallel-campaign.service','-p','ControlGroup','--value'],text=True).strip()
            operational_release.stop_controller(root,state_path,pid,cgroup,time.monotonic()+1200)
        source=Path(__file__).resolve().parent
        subprocess.run(['python3','-B',str(source/'activate_promotion_safeguards.py'),'--root',str(root),
            '--manifest',str(manifest),'--output',str(output/'controller'),'--gp-emission-release',str(release),
            '--recover-stopped'],check=True)
        immutable(output/'complete.json',dict(timestamp=time.time(),request_sha256=sha(output/'request.json'),
            drained_sha256=sha(output/'drained.json'),controller_sha256=sha(output/'controller/complete.json'),
            executable_sha256=sha(binary),workers=104,threads_each=1,final_tests_opened=False))
    finally:
        # Also called by the transient unit's ExecStopPost after SIGKILL.
        resume(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume-only',action='store_true');p.add_argument('--output',type=Path,required=True)
    for name in ('root','manifest','release'):p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.resume_only:
        if (args.output/'request.json').exists():resume(args.output)
    else:main(args.root.resolve(),args.manifest.resolve(),args.release.resolve(),args.output.resolve())
