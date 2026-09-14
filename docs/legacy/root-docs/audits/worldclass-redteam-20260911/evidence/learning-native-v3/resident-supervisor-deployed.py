#!/usr/bin/env python3
"""Keep bounded CUDA ranks alive and release them for explicit GPU handoffs."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import experiment_runtime as runtime
from resume_expired_cycle import atomic, read, require, sha


def suspended(root, now, owner=None):
    pause=root/'runs/continuous-control.json'
    if pause.is_file() and read(pause).get('paused',False):return True
    path=root/'runs/qualification-control.json'
    if path.is_file():
        control=read(path)
        if control.get('active') and control.get('requested_at', now+1) <= now < control.get('deadline', 0):
            return owner is None or control.get('output') != owner
    return owner is not None


def idle_worker(state, pid):
    # A stale file from a previous process must never authorize a handoff.
    return state.get('pid') == pid and state.get('status') in ('idle', 'warming')


def run(root, binary, pool, world, owner=None):
    import fcntl
    pool.mkdir(parents=True,exist_ok=True)
    require(pool.resolve().is_relative_to(root/'runs'), 'worker pool must be private to the learning workspace')
    require(world in (2,4,8), 'unsupported worker pool size')
    expected=sha(binary)
    lock=(pool/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children={}; stopping=False
    cache_log=(pool/f'teaching-cache-{time.time_ns()}.log').open('xb')
    cache=None
    cache_command=[sys.executable,'-B',str(Path(__file__).with_name('shared_teaching_cache.py')),
                   '--root',str(root),'--pool',str(pool)]
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not stopping:
            handoff=suspended(root,time.time(),owner)
            if cache is None or cache.poll() is not None:
                cache=runtime.popen(cache_command,stdout=cache_log,stderr=cache_log)
            ready=True
            for rank in range(world):
                if rank in children and children[rank][0].poll() is not None:
                    child,log=children.pop(rank);log.close()
                    runtime.append(pool/'journal.jsonl',dict(rank=rank,event='worker_exited',exit_code=child.returncode,timestamp=time.time()))
                if handoff and rank in children:
                    path=pool/f'rank-{rank}.json'
                    state=read(path) if path.is_file() else {}
                    if idle_worker(state, children[rank][0].pid):
                        child,log=children.pop(rank);runtime.stop([child]);log.close()
                if not handoff and rank not in children:
                    require(sha(binary)==expected,'worker executable changed')
                    cpu=rank*2 if rank<4 else 62+(rank-4)*2
                    log=(pool/f'rank-{rank}-{time.time_ns()}.log').open('xb')
                    child=runtime.popen(['taskset','-c',f'{cpu},{cpu+1}',str(binary),'--root',str(root),
                        'gpu-worker','--pool',str(pool),'--rank',str(rank)],stdout=log,stderr=log,
                        env=dict(os.environ,JOPE_TEACHING_CACHE=str(pool/'teaching.sock'),OMP_NUM_THREADS='2',RAYON_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                    children[rank]=(child,log)
                    runtime.append(pool/'journal.jsonl',dict(rank=rank,event='worker_started',pid=child.pid,timestamp=time.time()))
                path=pool/f'rank-{rank}.json'
                state=read(path) if path.is_file() else {}
                ready=ready and rank in children and state.get('pid')==children[rank][0].pid and state.get('status') in ('idle','busy')
            atomic(pool/'pool.json',dict(schema='parallel.resident_gpu_pool.v1',timestamp=time.time(),
                phase='suspended' if handoff and not children else 'draining' if handoff else 'running' if ready else 'initializing',
                executable_sha256=expected,pids={str(r):c.pid for r,(c,_) in children.items()},world=world))
            time.sleep(1)
    finally:
        runtime.stop([child for child,_ in children.values()] + ([cache] if cache else []))
        cache_log.close()
        for _,log in children.values():log.close()
        atomic(pool/'pool.json',dict(schema='parallel.resident_gpu_pool.v1',phase='stopped',timestamp=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','binary','pool'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--world',type=int,default=8)
    p.add_argument('--qualification-owner')
    a=p.parse_args();run(a.root.resolve(),a.binary.resolve(),a.pool.resolve(),a.world,a.qualification_owner)
