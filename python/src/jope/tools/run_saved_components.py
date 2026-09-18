#!/usr/bin/env python3
"""Run frozen component services sequentially and publish truthful pipeline status."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--jobs', required=True)
p.add_argument('--specs', required=True)
p.add_argument('--binary', default='bin/jope-parallel-neural-v38')
p.add_argument('--registry', default='manifests/generated/families/qualification-v4.json')
p.add_argument('--context-total', type=int, required=True)
p.add_argument('--completed-contexts', type=int, default=0)
a = p.parse_args()
feed = a.root/'runs/encoding-pipeline.json'

def publish(phase, reason, next_task, **extra):
    value = dict(schema='parallel.encoding_pipeline.v1', timestamp=int(time.time()),
        phase=phase, completed=a.completed_contexts, total=a.context_total,
        reason=reason, registry=a.registry, jobs=a.jobs, next_task=next_task,
        complete=False, **extra)
    tmp = feed.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(feed)

for component, batch in [('target',128), ('foundation',2)]:
    publish(f'saving {component} encodings', 'Eight persistent CUDA ranks; vectors and cut identities saved to disk',
            'foundation encodings' if component=='target' else 'registry-bound context assembly')
    command = [str(a.root/a.binary),'--root',str(a.root),f'{component}-service',
        '--jobs',str(a.root/a.jobs),'--spec',str(a.root/a.specs/f'{component}-spec.json'),
        '--output',str(a.root/f'data/encodings/{component}-cuts'),'--batch',str(batch)]
    child = subprocess.Popen(command, cwd=a.root)
    try:
        while child.poll() is None:
            extra = {}
            try:
                live = json.loads((a.root/'runs/encoding-service.json').read_text())
                if live.get('component')==component:
                    extra = dict(component_completed=live.get('completed'), component_total=live.get('total'),
                                 component_timestamp=live.get('timestamp'))
            except (OSError, ValueError):
                pass
            publish(f'saving {component} encodings', 'Eight persistent CUDA ranks; live rank progress is in the encoder panel',
                    'foundation encodings' if component=='target' else 'registry-bound context assembly', **extra)
            time.sleep(1)
        if child.returncode:
            raise RuntimeError(f'{component} service exited {child.returncode}; saved vectors remain resumable')
    except BaseException as error:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        publish(f'{component} encoding failed', str(error), 'Inspect service log before resuming')
        raise
publish('components saved; validating assembly inputs', 'Target and foundation services completed; verifying all three component identities',
        'fit new transforms on registry meta-training families only')
