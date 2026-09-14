from pathlib import Path
import json,hashlib,time
root=Path('/home/ubuntu/jope-dime/parallel');base=root/'runs/continuous-learning-v2-20260909';x={'timestamp':time.time(),'lists':{},'documents':{}}
for rel in ['runs/continuous-learning-v2-20260909','runs/continuous-learning-v2-20260909/generation-000066','runs/flywheel-20260910','runs/parity/gpr-cuda-v1']:
 p=root/rel;x['lists'][rel]=[q.name for q in p.iterdir() if q.is_file()]
for p in base.glob('*.json'):
 if p.name=='state.json':continue
 if any(w in p.name for w in ['teacher','progress','quality','budget','summary']) and p.stat().st_size<1000000:
  x['documents'][str(p.relative_to(root))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'document':json.loads(p.read_text())}
for p in (base/'generation-000066').glob('*.json'):
 if 'jobs' not in p.name and p.stat().st_size<1500000:
  x['documents'][str(p.relative_to(root))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'document':json.loads(p.read_text())}
print(json.dumps(x))
