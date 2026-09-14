#!/usr/bin/env python3
"""Qualify new frozen-module shapes on eight GPUs under the shared GPU lease.

Run extend_component_sampling.py --prepare-only first. This coordinator resumes
complete immutable shards and publishes runtime specs only after complete parity.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--jobs',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--target-spec',type=Path,action='append',required=True)
p.add_argument('--foundation-spec',type=Path,action='append',required=True)
p.add_argument('--context-total',type=int)
p.add_argument('--completed-contexts',type=int,default=0)
a=p.parse_args()
request=a.output/'request.json'
assert request.exists(), 'Run --prepare-only first'
request_sha=hashlib.sha256(request.read_bytes()).hexdigest()
command=[sys.executable,str(a.root/'tools/extend_component_sampling.py'),'--root',str(a.root),'--jobs',str(a.jobs),'--output',str(a.output),'--world-size','8']
for kind,paths in [('target',a.target_spec),('foundation',a.foundation_spec)]:
    for path in paths: command += [f'--{kind}-spec',str(path)]
lease=(a.root/'runs/encoding-service.lock').open('a+')
deadline=time.monotonic()+900
while True:
    try:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if time.monotonic()>deadline: raise RuntimeError('GPU lease wait exceeded 900 seconds')
        time.sleep(.5)
started=time.monotonic()
workers=[]
def heartbeat(phase):
    if a.context_total is None: return
    rows=[]
    for rank in range(8):
        try:
            row=json.loads((a.output/f'progress-rank-{rank}.json').read_text())
            if row['request_sha256']==request_sha: rows.append(row)
        except (OSError,ValueError): pass
    value=dict(schema='parallel.encoding_pipeline.v1',timestamp=int(time.time()),phase=phase,
        completed=a.completed_contexts,total=a.context_total,complete=False,
        reason='New-shape parity: CPU reference loading/sampling and CUDA module comparisons; completed cases below',
        registry='manifests/generated/families/qualification-v4.json',jobs=str(a.jobs.relative_to(a.root)),
        next_task='save target and foundation encodings on all eight GPUs',
        parity_completed=sum(r['target_completed']+r['foundation_completed'] for r in rows),
        parity_total=len(json.loads(request.read_text())['target_new_lengths'])*3+len(json.loads(request.read_text())['foundation_new_shapes']),
        ranks=rows)
    feed=a.root/'runs/encoding-pipeline.json';tmp=feed.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(feed)
try:
    for rank in range(8):
        receipt=a.output/f'qualification-rank-{rank}.json'
        if receipt.exists():
            row=json.loads(receipt.read_text())
            assert row['request_sha256']==request_sha and row['world_size']==8 and row['passed'], 'Existing shard identity/failure requires a new output version'
            continue
        cpu=rank*2 if rank<4 else 62+(rank-4)*2
        log=(a.output/f'qualification-rank-{rank}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        child=subprocess.Popen(['taskset','-c',f'{cpu},{cpu+1}']+command+['--qualify-cuda','--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT,env=env)
        workers.append((rank,child,log))
    while True:
        heartbeat('qualifying new frozen-module shapes')
        for rank,child,_ in workers:
            code=child.poll()
            assert code is None or code==0, f'Parity rank {rank} failed ({code}); see immutable shard/log'
        if all(child.poll() is not None for _,child,_ in workers): break
        if time.monotonic()-started>900: raise TimeoutError('Parity workers exceeded900seconds')
        time.sleep(1)
finally:
    for _,child,log in workers:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        log.close()
    lease.close()
subprocess.run(['taskset','-c','0,1']+command+['--merge-qualification'],check=True)
heartbeat('frozen-module parity passed; runtime specs published')
print(json.dumps(dict(passed=True,world_size=8,elapsed_seconds=time.monotonic()-started,request_sha256=request_sha)),flush=True)
