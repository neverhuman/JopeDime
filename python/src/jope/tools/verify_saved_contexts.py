#!/usr/bin/env python3
"""Verify every saved context and its frozen component/prepared identities."""
import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--index',required=True)
p.add_argument('--registry',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
index=read(a.root/a.index)
registry=read(a.root/a.registry)
admitted={d['dataset']:(f,d) for f in registry['families'].values() for d in f['datasets']}
prepared={};seen=set();records=[];partitions={};component_bytes=0
for row in index['encodings']:
    path=a.root/row['path'];payload=path.read_bytes();meta=read(path.with_suffix('.json'))
    assert len(payload)==8192 and sha(path)==row['sha256']==meta['output_sha256']
    assert all(math.isfinite(x) for x in struct.unpack('<2048f',payload))
    assert meta['encoder_spec_sha256']==index['assembly_sha256']
    if 'receipt_sha256' in row: assert sha(path.with_suffix('.json'))==row['receipt_sha256']
    key=(row['dataset'],row['cut']);assert key not in seen;seen.add(key)
    if row['prepared'] not in prepared:
        pp=a.root/row['prepared']/'prepared.json';m=read(pp)
        assert sha(pp)==row['prepared_sha256'];prepared[row['prepared']]=m
    m=prepared[row['prepared']];cut=next(c for c in m['cuts'] if c['index']==row['cut'])
    family,ds=admitted[row['dataset']]
    assert not family['holds'] and ds['preparation_eligible']
    assert row['family']==m['family']==family['family'] and row['partition']==m['meta_partition']==family['partition']
    for k,v in [('dataset',m['dataset']),('family',m['family']),('cut',row['cut']),('prepared_sha256',row['prepared_sha256']),('training_sha256',cut['training_sha256']),('preprocessing_sha256',cut['preprocessing_sha256'])]: assert meta[k]==v,(path,k)
    assert meta['selected_training_rows_sha256']==hashlib.sha256(json.dumps(cut['train_indices'],separators=(',',':')).encode()).hexdigest()
    for name,b in meta['components'].items():
        receipt=a.root/b['receipt'];c=read(receipt);vector=a.root/b['signature']
        assert sha(receipt)==b['receipt_sha256'] and sha(vector)==b['signature_sha256']==c['signature_sha256']
        assert c['spec_sha256']==b['spec_sha256'] and c['final_test_targets_used'] is False
        for k in ['dataset','family','cut','prepared_sha256','training_sha256','preprocessing_sha256']: assert c[k]==meta[k]
        component_bytes+=vector.stat().st_size
    partitions[row['partition']]=partitions.get(row['partition'],0)+1
    records.append(dict(path=row['path'],sha256=row['sha256'],receipt_sha256=sha(path.with_suffix('.json'))))
report=dict(schema='parallel.saved_context_verification.v1',timestamp=int(time.time()),passed=True,index=a.index,index_sha256=sha(a.root/a.index),assembly_sha256=index['assembly_sha256'],registry=a.registry,registry_sha256=sha(a.root/a.registry),views=len(records),datasets=len({r['dataset'] for r in index['encodings']}),families=len({r['family'] for r in index['encodings']}),partitions=partitions,context_bytes=len(records)*8192,verified_component_bytes=component_bytes,copied_csv_bytes=0,final_test_targets_used=False,qualification=True,independent_selection_eligible=False,files=records)
out=a.root/a.output;out.parent.mkdir(parents=True,exist_ok=True)
assert not out.exists(), 'Verification receipts are versioned'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'}))
