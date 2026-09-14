"""Bind operational rollout to measured corpus equality and worker recovery."""
import copy
import os
import signal
import subprocess
import time
from pathlib import Path

from promotion_gate import artifact,immutable
from resume_expired_cycle import read,require,sha

SETTINGS={'packing_executable','training_executable','monitor_executable','gpu_pool','teaching_store',
          'legacy_export_interval_seconds','progress_timeout_seconds','automatic_training'}


def verify(root,path):
    release=read(path)
    require(release['schema']=='parallel.flywheel_operational_release.v1','unknown operational release')
    require(set(release['settings'])<=SETTINGS,'operational release attempts to change scientific policy')
    require(release['settings'].get('automatic_training',False) is False,
            'operational release cannot enable an unqualified training policy')
    build=artifact(root,release['build'])
    require(build['status']=='passed','native build not verified')
    build_dir=(root/release['build']['path']).parent
    require(not (build_dir/'invalidated.json').exists(),'withdrawn native build')
    binary=release['settings']['training_executable']
    expected=sha(root/binary)
    require(build['bindings']['jope-parallel-native']==expected,'native executable differs from tested build')
    require(sha(root/release['settings']['monitor_executable'])==expected,
            'monitor executable differs from qualified resident worker')
    for name in ('packing','workers'):
        result=artifact(root,release[name])
        require(result['status']=='passed','operational qualification incomplete: '+name)
        request_path=(root/release[name]['path']).parent/'request.json'
        require(sha(request_path)==result['request_sha256'],'qualification request changed')
        role='packing_executable' if name=='packing' else 'training_executable'
        require(read(request_path)['executable_sha256']==sha(root/release['settings'][role]),
                'qualification used a different executable for '+name)
        if name=='workers':
            tested={Path(path).name:digest for path,digest in read(request_path).get('python_bindings',{}).items()}
            for filename in ('resident_gpu_pool.py','shared_teaching_cache.py','experiment_runtime.py','resume_expired_cycle.py'):
                require(tested.get(filename)==sha(Path(__file__).resolve().parent/filename),
                        'resident worker Python differs from qualified implementation: '+filename)
    request=read((root/release['packing']['path']).parent/'request.json')
    require(release['teaching_sources']==request['sources'],'teaching source order changed')
    qualified_store=root/request['store'] if 'store' in request else (root/release['packing']['path']).parent/'store'
    require((root/release['settings']['teaching_store']).resolve()==qualified_store.resolve(),
            'release store was not qualified')
    return release


def migrate(state,old_digest,new_digest,teaching_sources=None):
    require(state['manifest_sha256']==old_digest,'active manifest changed during handoff')
    result=copy.deepcopy(state)
    result['manifest_sha256']=new_digest
    if teaching_sources is not None:
        result['teaching_sources']=list(teaching_sources)
    return result


def fallback(manifest):
    result=copy.deepcopy(manifest)
    result.pop('gpu_pool',None)
    result.pop('teaching_store',None)
    result.pop('async_corpus_verification',None)
    result['packing_executable']=result['cpu_executable']
    # Keep the qualified compact reader for already-published shard manifests.
    # Serving, candidates, consumed snapshots and promotion decisions stay put.
    return result


def wait_gpu_boundaries(root,state,cgroup,deadline):
    tasks=[t for k,t in state['tasks'].items() if k in ('train','proposal','monitor')]
    processes=Path('/sys/fs/cgroup')/cgroup.lstrip('/')/'cgroup.procs'
    while True:
        commands=[]
        for pid in processes.read_text().split() if cgroup and processes.is_file() else []:
            try:commands.append(Path('/proc',pid,'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
            except (OSError,UnicodeError):continue
        pending=[t for t in tasks if not (root/t['receipt']).is_file()
                 and any(command[:len(t['command'])]==t['command'] for command in commands)]
        if not pending:return
        require(time.monotonic()<deadline,'durable GPU stage handoff timed out')
        time.sleep(2)


def controller_process(pid):
    from stopped_unit import process
    return process(pid)


def recover_controller_stops(state_path):
    import stopped_unit
    for journal in state_path.parent.glob('controller-stop-*/request.json'):
        if not (journal.parent/'complete.json').exists():stopped_unit.recover(journal.parent)


def stop_controller(root,state_path,pid,cgroup,deadline):
    """Commit the GPU boundary without re-entering a stopped dispatch loop."""
    import stopped_unit
    identity=controller_process(pid);require(identity is not None,'controller disappeared before handoff')
    os.kill(pid,signal.SIGSTOP)
    stop_requested=False
    try:
        while controller_process(pid)!=dict(identity,state='T'):
            require(time.monotonic()<deadline,'controller did not stop dispatch');time.sleep(.01)
        boundary=lambda:wait_gpu_boundaries(root,read(state_path),cgroup,deadline)
        boundary()
        journal=state_path.parent/f'controller-stop-{pid}-{identity["birth"]}'
        stop_requested=True
        stopped_unit.stop('jope-parallel-campaign.service',pid,journal,boundary)
    finally:
        if not stop_requested:
            current=controller_process(pid)
            if current is not None and current['birth']==identity['birth']:
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass
