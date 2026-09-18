#!/usr/bin/env python3
"""Durably assemble one immutable context version with a live phase heartbeat."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--jobs',required=True)
p.add_argument('--output',required=True)
p.add_argument('--registry',required=True)
p.add_argument('--binary',default='bin/jope-parallel-neural-v38')
p.add_argument('--assembly',help='Apply this frozen assembly instead of fitting')
p.add_argument('--hyperion',default='data/encodings/hyperion-batched-cuts')
p.add_argument('--context-total',type=int)
p.add_argument('--completed-contexts',type=int,default=0)
a=p.parse_args()
total=len(json.loads((a.root/a.jobs).read_text())['jobs'])
feed=a.root/'runs/encoding-pipeline.json'
def publish(phase, reason, completed=0, complete=False):
    row=dict(schema='parallel.encoding_pipeline.v1',timestamp=int(time.time()),phase=phase,
        completed=a.completed_contexts+completed,total=a.context_total or total,reason=reason,registry=a.registry,jobs=a.jobs,
        next_task='verify saved vectors and admit encoding-complete GP jobs',complete=complete,
        contexts_index=str(Path(a.output)/'index.json'))
    temp=feed.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(feed)
cmd=[str(a.root/a.binary),'--root',str(a.root),'apply-contexts' if a.assembly else 'assemble-contexts',
    '--jobs',str(a.root/a.jobs),'--hyperion',str(a.root/a.hyperion),
    '--foundation',str(a.root/'data/encodings/foundation-cuts'),'--target',str(a.root/'data/encodings/target-cuts'),
    '--output',str(a.root/a.output),'--qualification']
cmd+=['--assembly',str(a.root/a.assembly)] if a.assembly else ['--registry',str(a.root/a.registry)]
child=subprocess.Popen(cmd,cwd=a.root)
stage=(a.root/a.output).with_name('.'+Path(a.output).name+'.staging')
try:
    while child.poll() is None:
        if (stage/'transforms.ot').exists():
            phase='saving projected context vectors';reason='Frozen transforms saved; persisting vectors and hash-bound metadata before atomic publication'
        elif stage.exists():
            phase='fitting registry-bound context transforms';reason='FP64 covariance/projection on CUDA0; only meta_train families fit transforms, equally weighted by family'
        else:
            phase='checking saved component identities';reason='CPU verifies every component hash, training membership and registry binding before projection'
        if a.assembly:
            phase='applying frozen context transforms';reason='No refit: verify components and apply the existing assembly to saved validation views'
        publish(phase,reason)
        time.sleep(1)
    if child.returncode: raise RuntimeError(f'Context process exited {child.returncode}; failed staging retained')
except BaseException as error:
    if child.poll() is None: child.terminate();child.wait(timeout=10)
    publish('context assembly failed',str(error));raise
publish('contexts saved; verification pending','Immutable context index published; checking persisted bytes before queue admission',total)
