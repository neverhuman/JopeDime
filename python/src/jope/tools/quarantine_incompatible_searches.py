"""Preserve interrupted searches whose evaluator was retired; keep their jobs queued."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from gp_emission_handoff import process,same_process,SERVICE
from promotion_gate import immutable
from resume_expired_cycle import read,require,sha


def tree(path):
    files={}
    for child in sorted(path.rglob('*')):
        require(not child.is_symlink(),'search contains a symbolic link')
        if child.is_file():files[str(child.relative_to(path))]=sha(child)
    return files


def eligible(path,retired):
    require(not (path/'search.json').exists(),'completed search cannot be quarantined')
    identities=set()
    ledger=path/'observations.jsonl'
    if ledger.exists():
        import json
        for line in ledger.read_text().splitlines():
            if line.strip():identities.add(json.loads(line)['evaluator_sha256'])
    for attempt in path.glob('candidate-*.attempt.json'):
        identities.add(read(attempt)['evaluator_sha256'])
    require(identities=={retired},'partial search does not have exactly the retired evaluator identity')


def resume(output):
    path=output/'request.json'
    if path.exists():
        parent=read(path)['supervisor']
        if same_process(parent):os.kill(parent['pid'],signal.SIGCONT)


def move(root,output,request):
    """Called only with the verified supervisor stopped and target jobs inactive."""
    for index,row in enumerate(request['searches']):
        original=root/row['path'];archive=output/'searches'/f'{index:03}'
        if archive.exists():
            require(tree(archive)==row['files'],'quarantine archive changed')
        else:
            require(tree(original)==row['files'],'partial search changed before quarantine')
            eligible(original,request['retired_evaluator_sha256'])
            archive.parent.mkdir(parents=True,exist_ok=True)
            os.rename(original,archive)
            for directory in (original.parent,archive.parent):
                fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
        immutable(output/'receipts'/f'{index:03}.json',dict(original=row['path'],
            archive=str(archive.relative_to(root)),files=row['files'],
            outcome='interrupted_evaluator_retired',new_attempt='original queued job starts under current evaluator',
            historical_observations_preserved=True,observations_relabelled=False))


def run(root,output,retired):
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'request.json').exists():
        pid=int(subprocess.check_output(['systemctl','show',SERVICE,'-p','MainPID','--value'],text=True))
        parent=process(pid);require(parent is not None,'GP supervisor missing')
        status=read(root/'runs/gp-service.json')
        require(0<=time.time()-status['timestamp']<30 and status['phase']=='running','GP status stale')
        current=sha(Path(f'/proc/{pid}/exe'))
        require(current==status['executable_sha256'] and current!=retired,'current evaluator binding changed')
        paths=sorted({row['output'] for row in status['failed_rows']})
        require(paths,'no recorded failed jobs to reconcile')
        searches=[]
        for name in paths:
            path=(root/name).resolve();require(path.is_relative_to(root),'search escapes workspace')
            eligible(path,retired);searches.append(dict(path=name,files=tree(path)))
        immutable(output/'request.json',dict(schema='parallel.retired_evaluator_quarantine.v1',
            supervisor=parent,current_evaluator_sha256=current,retired_evaluator_sha256=retired,
            searches=searches,driver_sha256=sha(Path(__file__)),final_tests_opened=False))
    request=read(output/'request.json');parent=request['supervisor']
    require(request['retired_evaluator_sha256']==retired and request['driver_sha256']==sha(Path(__file__)),
            'recovery inputs changed')
    require(same_process(parent) and sha(Path(f"/proc/{parent['pid']}/exe"))==request['current_evaluator_sha256'],
            'GP supervisor changed; re-audit before recovery')
    os.kill(parent['pid'],signal.SIGSTOP)
    try:
        deadline=time.monotonic()+5
        while process(parent['pid'])['state']!='T':
            require(time.monotonic()<deadline,'dispatcher did not stop');time.sleep(.01)
        targets={str((root/row['path']).resolve()) for row in request['searches']}
        for pid in Path(f"/proc/{parent['pid']}/task/{parent['pid']}/children").read_text().split():
            child=process(int(pid))
            if child and child['state']!='Z':
                args=Path('/proc',pid,'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                require('--output' in args,'unrecognized active GP child')
                require(str(Path(args[args.index('--output')+1]).resolve()) not in targets,
                        'a target search is active; leave it untouched')
        move(root,output,request)
        immutable(output/'complete.json',dict(status='passed',request_sha256=sha(output/'request.json'),
            preserved_searches=len(request['searches']),queued_jobs_unchanged=True,serving_checkpoint_unchanged=True,
            partial_measurements_relabelled=False,timestamp=time.time(),final_tests_opened=False))
    finally:resume(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retired-evaluator');p.add_argument('--resume-only',action='store_true')
    a=p.parse_args()
    if a.resume_only:resume(a.output)
    else:run(a.root.resolve(),a.output.resolve(),a.retired_evaluator)
