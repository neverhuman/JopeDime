#!/usr/bin/env python3
"""Remove obsolete generated cut CSVs after checking their pointer replacements.

Only these implementation-owned diagnostic/preparation directories are eligible.
Original datasets, pointer recipes, encoder outputs and GP ledgers are preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args();root=args.root.resolve()
receipt=root/'manifests/generated/retired-cut-copies.json'
retired=json.loads(receipt.read_text())['files'] if receipt.exists() else []
known={r['path'] for r in retired};pending=[]
for directory in ('qualification-v2','qualification-nox-v2','diagnostic-nox'):
    for p in sorted((root/'data/prepared'/directory).rglob('*.csv')):
        relative=str(p.relative_to(root))
        if p.name not in ('train.csv','validation.csv'):
            raise RuntimeError(f'Unexpected generated CSV {p}')
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        replacement=None
        if directory=='qualification-v2':
            dataset=p.parent.parent.name;cut=int(p.parent.name.removeprefix('cut-'))
            manifest=root/'data/prepared/views-v2'/dataset/'prepared.json'
            prepared=json.loads(manifest.read_text())
            view=next(c for c in prepared['cuts'] if c['index']==cut)
            key='training_sha256' if p.name=='train.csv' else 'validation_sha256'
            if digest!=view[key] or not prepared.get('storage'):
                raise RuntimeError(f'Pointer replacement does not reproduce {p}')
            replacement=str(manifest.relative_to(root))
        row={'path':relative,'sha256':digest,'bytes':p.stat().st_size,
             'disposition':'retired generated cut copy','pointer_replacement':replacement,
             'reason':'Exact pointer representation replaces CSV' if replacement else 'Obsolete diagnostic preprocessing; withheld from training'}
        if relative not in known:retired.append(row);known.add(relative)
        pending.append((p,digest))
receipt.parent.mkdir(parents=True,exist_ok=True)
temporary=receipt.with_suffix('.tmp')
with temporary.open('w') as f:
    json.dump({'schema':'parallel.retired_cut_copies.v1','files':retired,'bytes_released':sum(r['bytes'] for r in retired)},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(temporary,receipt)
for p,digest in pending:
    if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'CSV changed before retirement: {p}')
    p.unlink()
print(json.dumps({'removed_this_run':len(pending),'accounted_files':len(retired),'bytes_released':sum(r['bytes'] for r in retired)}))
