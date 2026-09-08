#!/usr/bin/env python3
"""Select immutable real-program pointers for the optional GPR qualification."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from audit_gpu_ridge import stages, width

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
groups = defaultdict(list)
sources = {}
seen = set()

def read(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(a.root))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

for arm in ('neural','generic','shuffled'):
    jobs = read(a.root/'runs/regression-cycle-1'/arm/'jobs.json')['jobs']
    for job in jobs:
        prepared = read(a.root/job['prepared']/'prepared.json')
        cut = next(c for c in prepared['cuts'] if c['index'] == job['cut'])
        n, vn = len(cut['train_indices']), len(cut['validation_indices'])
        source_p = len(prepared['storage']['feature_columns'])
        for row in read(a.root/job['output']/'observations.json'):
            if row['failure'] is not None: continue
            parsed = stages(row['genome'])
            if not parsed or parsed[-1][0] != 'gpr': continue
            transformed, unknown = width(parsed[:-1], source_p, n)
            if transformed is None: continue
            params = parsed[-1][1]
            kernel = params.get('kernel','rbf').strip('"')
            m = min(2000, max(50, int(params.get('m',500))))
            steps = min(200, max(1, int(params.get('steps',40))))
            identity = (row['program_sha256'],row['prepared_sha256'],row['cut'])
            if identity in seen: continue
            seen.add(identity)
            if not 0.04 <= row['metrics']['seconds'] <= 8: continue
            key = (kernel, transformed, min(n,m), vn, m, steps)
            groups[key].append(dict(id=hashlib.sha256(str(identity).encode()).hexdigest()[:20],prepared=job['prepared'],prepared_sha256=job['prepared_sha256'],cut=job['cut'],genome=row['genome'],program_sha256=row['program_sha256'],family=row['family'],partition=row['partition'],original_ledger=job['output']+'/observations.json',original_evaluation_index=row['evaluation_index'],original_fit_predict_seconds=row['metrics']['seconds'],inferred_transformed_columns=transformed))

# Eight kernel expressions, four distinct actual programs per kernel. The
# manifest is interleaved so rank modulo eight receives one homogeneous batch.
kernels = ['rbf','rbf*periodic','periodic+linear','rbf+linear','rq','linear','periodic','rq+linear']
selected = []
for kernel in kernels:
    options = [(key, rows) for key,rows in groups.items() if key[0] == kernel and len(rows) >= 4]
    assert options, kernel
    # Prefer representative expensive programs, while keeping the first GPU
    # qualification bounded: target ~0.5 CPU seconds and <=256 basis rows.
    key, rows = min(options, key=lambda item: (item[0][2] > 256, abs(statistics.median(r['original_fit_predict_seconds'] for r in item[1])-.5)))
    rows.sort(key=lambda r:(abs(r['original_fit_predict_seconds']-.5),r['program_sha256']))
    selected.append(rows[:4])
cases = [selected[k][slot] for slot in range(4) for k in range(8)]
out = dict(schema='parallel.experimental_gpr_cases.v1',selection='four_actual_programs_per_each_of_eight_existing_kernel_expressions; modulo-eight ranks receive homogeneous batches',copied_data_bytes=0,evaluation_bias='adaptive_cut_heldout_teacher',final_tests_opened=False,cases=cases,sources=sources)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps([dict(kernel=kernels[i],count=len(rows),columns=rows[0]['inferred_transformed_columns'],family=rows[0]['family'],fit_seconds=[r['original_fit_predict_seconds'] for r in rows]) for i,rows in enumerate(selected)],indent=2))
