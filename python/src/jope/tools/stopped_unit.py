"""Stop a committed, SIGSTOPped dispatcher without letting systemd resume it."""
import fcntl
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from promotion_gate import immutable
from resume_expired_cycle import read, require, sha

RUNTIME_ROOT=Path('/run/systemd/system')

def process(pid):
    try:
        row=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(state=row[0],birth=row[19])
    except FileNotFoundError:return None


def property(unit,name):
    return subprocess.check_output(['systemctl','show',unit,'-p',name,'--value'],text=True).strip()


def atomic_config(path,data):
    """Never expose a partial owned override, including during recovery."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fchmod(stream.fileno(),0o644);os.fsync(stream.fileno())
        os.replace(name,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        try:os.unlink(name)
        except FileNotFoundError:pass


def install(path,source):
    subprocess.run(['sudo','-n','/usr/bin/python3','-B',str(Path(__file__).resolve()),
                    '--install',str(source),str(path)],check=True)


def restore(journal,request):
    override=Path(request['override']);owned=journal/'restart-disabled.conf'
    before=read(journal/'override-before.json')
    if override.exists() and override.read_bytes()==owned.read_bytes():
        if before['content'] is None:
            subprocess.run(['sudo','-n','rm','--',str(override)],check=True)
        else:
            saved=journal/'override-original.conf';atomic_config(saved,before['content'].encode())
            install(override,saved)
    else:
        require((not override.exists() and before['content'] is None)
                or (override.exists() and override.read_text()==before['content']),
                'another operation changed the restart override; preserve it')
    # A previous attempt may have restored bytes and crashed before reloading.
    subprocess.run(['sudo','-n','systemctl','daemon-reload'],check=True)
    effective=property(request['unit'],'Restart')
    main=int(property(request['unit'],'MainPID') or 0)
    if main==0 and effective:
        require(effective==request['restart_before'],'effective restart configuration was not restored')


def stop(unit,pid,journal,verify_boundary):
    require(re.fullmatch(r'[A-Za-z0-9_.-]+\.service',unit) is not None,'invalid unit name')
    # Serialize all journals for this unit, including retries after a crash.
    lock=Path('/tmp')/f'jope-committed-drain-{os.getuid()}-{unit}.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _stop(unit,pid,journal,verify_boundary)


def _stop(unit,pid,journal,verify_boundary):
    journal=Path(journal);journal.mkdir(parents=True,exist_ok=True)
    require(re.fullmatch(r'[A-Za-z0-9_.-]+\.service',unit) is not None,'invalid unit name')
    request_path=journal/'request.json'
    if request_path.exists():request=read(request_path)
    else:
        identity=process(pid);require(identity is not None and identity['state']=='T','dispatcher is not stopped')
        override=RUNTIME_ROOT/(unit+'.d')/'zzzzzzzz-codex-committed-drain.conf'
        before=override.read_text() if override.exists() else None
        require(before is None or '# Committed drain owner:' not in before,
                'another drain journal owns this unit; recover that journal first')
        immutable(journal/'override-before.json',dict(content=before))
        request=dict(unit=unit,pid=pid,birth=identity['birth'],override=str(override),
                     restart_before=property(unit,'Restart'),
                     driver_sha256=sha(Path(__file__)),
                     protocol='disable_restart_kill_stopped_parent_then_stop_unit_v1')
        immutable(request_path,request)
    require(request['unit']==unit and request['pid']==pid,'stop journal belongs to another dispatcher')
    require(request['driver_sha256']==sha(Path(__file__)),'stop recovery implementation changed')
    if (journal/'complete.json').exists():return
    current=process(pid)
    same=current is not None and current['birth']==request['birth'] and current['state']!='Z'
    if same:
        require(current['state']=='T','dispatcher resumed before its committed stop')
        owned=journal/'restart-disabled.conf'
        content=f'# Committed drain owner: {journal.resolve()}\n[Service]\nRestart=no\n'
        if owned.exists():require(owned.read_text()==content,'restart override staging changed')
        else:atomic_config(owned,content.encode())
        override=Path(request['override']);before=read(journal/'override-before.json')['content']
        require((not override.exists() and before is None) or (override.exists()
                and override.read_text() in (before,content)),
                'another operation changed the restart override; preserve it')
        install(override,owned)
        subprocess.run(['sudo','-n','systemctl','daemon-reload'],check=True)
        require(property(unit,'Restart')=='no' and int(property(unit,'MainPID'))==pid,
                'automatic restart was not disabled on the expected unit')
        require(process(pid)==dict(state='T',birth=request['birth']),'stopped dispatcher identity changed')
        descriptor=os.pidfd_open(pid)
        try:
            verify_boundary()
            require(process(pid)==dict(state='T',birth=request['birth'])
                    and int(property(unit,'MainPID'))==pid and property(unit,'Restart')=='no',
                    'dispatcher or restart policy changed during boundary verification')
            immutable(journal/'kill-intent.json',dict(request_sha256=sha(request_path)))
            # systemd Stop sends SIGCONT after KillSignal. Kill the stopped
            # parent directly FIRST. The pidfd cannot target a reused PID.
            signal.pidfd_send_signal(descriptor,signal.SIGKILL)
        finally:os.close(descriptor)
    else:
        require((journal/'kill-intent.json').exists(),'dispatcher vanished without a committed kill intent')
    deadline=time.monotonic()+10
    while True:
        current=process(pid)
        if current is None or current['birth']!=request['birth'] or current['state'] in ('Z','X'):break
        require(time.monotonic()<deadline,'stopped dispatcher did not exit');time.sleep(.01)
    main=int(property(unit,'MainPID') or 0)
    main_identity=process(main) if main else None
    same_main=main==pid and main_identity is not None and main_identity['birth']==request['birth']
    if main==0 or same_main:
        subprocess.run(['sudo','-n','systemctl','stop',unit],check=True)
    # A restart of this recovery after another explicit activation must never
    # stop the replacement process. Restore only our own temporary override.
    restore(journal,request)
    immutable(journal/'complete.json',dict(request_sha256=sha(request_path),
        dispatcher_exited=True,restart_override_restored=True,replacement_pid_preserved=main if main and not same_main else None))


def recover(journal):
    request=read(Path(journal)/'request.json')
    def refuse_live():
        raise RuntimeError('live stopped dispatcher requires its original boundary verification')
    stop(request['unit'],request['pid'],Path(journal),refuse_live)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Atomically install a committed drain override')
    parser.add_argument('--install',nargs=2,type=Path,required=True,metavar=('SOURCE','DESTINATION'))
    args=parser.parse_args()
    source,destination=args.install
    require(destination.name=='zzzzzzzz-codex-committed-drain.conf'
            and destination.parent.parent==RUNTIME_ROOT,'unexpected restart override destination')
    atomic_config(destination,source.read_bytes())
