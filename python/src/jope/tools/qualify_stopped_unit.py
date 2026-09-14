#!/usr/bin/env python3
"""Exercise committed dispatcher shutdown on disposable systemd units only."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from promotion_gate import immutable
from qualification_budget import Budget
from resume_expired_cycle import read,require,sha
import stopped_unit

UNITS=('jope-drain-lifecycle-qualification-v1.service','jope-drain-recovery-qualification-v1.service')


def command(*args):
    return subprocess.run(list(args),check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10).stdout


def cleanup(output):
    # These names are fixed in source; never accept a production unit argument.
    for unit in UNITS:
        subprocess.run(['sudo','-n','systemctl','stop',unit],check=False,timeout=10)
        path=stopped_unit.RUNTIME_ROOT/(unit+'.d')/'zzzzzzzz-codex-committed-drain.conf'
        if path.exists():
            content=path.read_text()
            require(str(output.resolve()) in content,'refuse cleanup of another fixture override')
            command('sudo','-n','rm','--',str(path))
    command('sudo','-n','systemctl','daemon-reload')


def fixture(output):
    signal.signal(signal.SIGTERM,lambda *_:(output/'dispatched').write_text('resumed by TERM/CONT\n'))
    immutable(output/'ready.json',dict(pid=os.getpid(),birth=stopped_unit.process(os.getpid())['birth']))
    time.sleep(60)


def run(root,output,budget_output):
    request=read(output/'request.json')
    require(request['units']==list(UNITS) and request['ceiling_seconds']==60 and request['cpu_width']==2
            and request['cleanup_core_seconds']==60,
            'disposable qualification contract changed')
    for name,digest in request['source_bindings'].items():
        require(sha(Path(__file__).parent/name)==digest,'qualification source changed: '+name)
    budget=Budget(budget_output)
    reservation=read(budget.path)['reservations'][output.name]
    require(reservation['core_seconds']==180 and reservation['node_seconds']==0 and not reservation['complete'],
            'missing qualification ceiling')
    started=time.monotonic();result=dict(status='inconclusive',cases=[])
    try:
        with (root/'runs/experiment-spare-cpus.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for index,unit in enumerate(UNITS):
                case=output/f'case-{index}';case.mkdir()
                require(int(stopped_unit.property(unit,'MainPID') or 0)==0,'disposable fixture already active')
                override=stopped_unit.RUNTIME_ROOT/(unit+'.d')/'zzzzzzzz-codex-committed-drain.conf'
                require(not override.exists(),'disposable override already exists')
                original=None
                if index:
                    original=f'# Qualification fixture: {output.resolve()}\n[Service]\nRestart=on-failure\n'
                    source=case/'original.conf';stopped_unit.atomic_config(source,original.encode())
                    stopped_unit.install(override,source)
                command('sudo','-n','systemd-run','--quiet','--unit='+unit,'--service-type=exec','--uid=ubuntu',
                    '--property=Restart=on-failure','--property=CPUAffinity=60','--property=RuntimeMaxSec=30',
                    '--property=TimeoutStopSec=5','/usr/bin/python3','-B',str(Path(__file__).resolve()),
                    '--fixture',str(case))
                deadline=time.monotonic()+5
                while not (case/'ready.json').exists():
                    require(time.monotonic()<deadline,'fixture did not initialize');time.sleep(.01)
                identity=read(case/'ready.json');pid=identity['pid']
                require(int(stopped_unit.property(unit,'MainPID'))==pid,'fixture MainPID mismatch')
                require(stopped_unit.property(unit,'Restart')=='on-failure','fixture restart policy mismatch')
                os.kill(pid,signal.SIGSTOP)
                while stopped_unit.process(pid)!=dict(state='T',birth=identity['birth']):
                    require(time.monotonic()<deadline,'fixture did not stop');time.sleep(.01)
                def boundary():
                    require(not (case/'dispatched').exists(),'fixture dispatched before committed kill')
                    require(not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip(),'fixture has children')
                if index:
                    # Interrupt after the original bytes have been restored,
                    # before daemon-reload. Recovery must reestablish effective state.
                    real_restore=stopped_unit.restore
                    def interrupted(journal,request):
                        source=journal/'override-original.conf'
                        stopped_unit.atomic_config(source,original.encode());stopped_unit.install(override,source)
                        raise InterruptedError('injected after restoration before reload')
                    stopped_unit.restore=interrupted
                    try:
                        try:stopped_unit.stop(unit,pid,case/'journal',boundary)
                        except InterruptedError:pass
                        else:raise RuntimeError('restoration interruption was not reached')
                    finally:stopped_unit.restore=real_restore
                    stopped_unit.recover(case/'journal')
                else:stopped_unit.stop(unit,pid,case/'journal',boundary)
                current=stopped_unit.process(pid)
                require(current is None or current['birth']!=identity['birth'] or current['state'] in ('Z','X'),
                        'fixture parent survived')
                require(not (case/'dispatched').exists(),'systemd resumed dispatcher')
                require((override.read_text() if override.exists() else None)==original,'original override bytes lost')
                require(int(stopped_unit.property(unit,'MainPID') or 0)==0,'fixture restarted unexpectedly')
                result['cases'].append(dict(unit=unit,identity=identity,dispatch_marker_absent=True,
                    original_override_restored=True,interrupted_restore=index==1,
                    restart_effective=stopped_unit.property(unit,'Restart'),
                    journal_sha256=sha(case/'journal/complete.json')))
            result['status']='passed'
    except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as error:result['reason']=str(error)
    finally:
        try:cleanup(output)
        except (OSError,RuntimeError,subprocess.SubprocessError) as error:
            result.update(status='inconclusive',cleanup_error=str(error))
        result.update(request_sha256=sha(output/'request.json'),seconds=time.monotonic()-started,
                      production_units_modified=False,final_tests_opened=False)
        immutable(output/'complete.json',result);budget.complete(output.name,output/'complete.json')
    require(result['status']=='passed',result.get('reason',result.get('cleanup_error','incomplete')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path);parser.add_argument('--cleanup',type=Path)
    parser.add_argument('--root',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--budget-output',type=Path)
    args=parser.parse_args()
    if args.fixture:fixture(args.fixture)
    elif args.cleanup:cleanup(args.cleanup)
    else:run(args.root.resolve(),args.output.resolve(),args.budget_output.resolve())
