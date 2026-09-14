#!/usr/bin/env python3
"""Ask the Rust preparer for new deterministic cuts; persist only new pointers."""
import concurrent.futures,hashlib,json,subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
binary=root/'target/debug/jope-parallel'
base=root/'data/prepared/views-v2'
previous=root/'data/prepared/views-expanded-20'
output=root/'data/prepared/views-extra-20'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def one(source):
    dest=output/source.name
    if not (dest/'extra-cuts.json').exists():
        if dest.exists():raise RuntimeError(f'Partial cut expansion needs inspection: {dest}')
        subprocess.run([str(binary),'expand-cuts','--prepared',str(source),'--output',str(dest),'--rounds','40'],check=True,stdout=subprocess.DEVNULL)
        manifest=dest/'prepared.json';p=json.loads(manifest.read_text())
        old=json.loads((previous/source.name/'prepared.json').read_text())
        lookup={c['index']:c for c in old['cuts']}
        for c in p['cuts']:
            if c['index']<100:
                if c!=lookup[c['index']]:raise RuntimeError(f'Existing cut changed: {source.name}/{c["index"]}')
                f=dest/f'cut-{c["index"]}'/'preprocessing.json'
                if sha(f)!=c['preprocessing_sha256']:raise RuntimeError('preprocessing changed')
                f.unlink();f.parent.rmdir()
        p['cuts']=[c for c in p['cuts'] if c['index']>=100];write(manifest,p)
        expansion=dest/'expansion.json';x=json.loads(expansion.read_text());x['cuts']=[c for c in x['cuts'] if c['cut']>=100];x['stored_round_range']=[20,40];write(expansion,x)
        write(dest/'extra-cuts.json',dict(schema='parallel.additional_cut_views.v1',existing_prepared_sha256=sha(previous/source.name/'prepared.json'),new_cuts=len(p['cuts']),copied_dataset_bytes=0,stored_cut_indices=[c['index'] for c in p['cuts']],independent_families=1))
    p=json.loads((dest/'prepared.json').read_text());return dest,p

output.mkdir(parents=True,exist_ok=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    prepared=list(pool.map(one,sorted(p for p in base.iterdir() if p.is_dir())))
jobs=[];files=[]
for dest,p in prepared:
    for c in p['cuts']:jobs.append(dict(prepared=str(dest.relative_to(root)),prepared_sha256=sha(dest/'prepared.json'),cut=c['index'],output=f'runs/gp-extra20/{p["dataset"]}/cut-{c["index"]}'))
    for f in sorted(dest.rglob('*')):
        if not f.is_file():continue
        if f.suffix=='.csv':raise RuntimeError(f'Unexpected copied dataset: {f}')
        files.append(dict(source=str(f),destination=str(f.relative_to(root)),sha256=sha(f),bytes=f.stat().st_size,dataset=p['dataset'],role='additional_cut_pointer'))
manifest=root/'manifests/generated/gp-extra20-jobs.json';write(manifest,dict(schema='parallel.gp_jobs.v1',qualification=True,jobs=jobs,vocabulary='assets/grammar/featmath.json'))
files.append(dict(source=str(manifest),destination=str(manifest.relative_to(root)),sha256=sha(manifest),bytes=manifest.stat().st_size,dataset=None,role='additional_cut_jobs'))
write(root/'manifests/generated/extra20-transfer.json',dict(schema='parallel.transfer.v1',files=files))
print(json.dumps(dict(new_cuts=len(jobs),datasets=len(prepared),pointer_bytes=sum(f['bytes'] for f in files),copied_dataset_bytes=0)))
