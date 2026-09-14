from pathlib import Path
import json,time,hashlib
root=Path('/home/ubuntu/jope-dime/parallel');base=root/'runs/continuous-learning-v2-20260909';x={'timestamp':time.time(),'files':[],'documents':{}}
for name in ['runs/parity/gpr-cuda-v1/report/report.json','runs/continuous-learning-v2-20260909/teacher-progress.json','runs/continuous-learning-v2-20260909/quality-report.json','runs/flywheel-20260910/budget.json','runs/flywheel-20260910/automatic-training-hold-deploy-v1/manifest.json']:
 p=root/name
 if p.is_file() and p.stat().st_size<2000000:x['documents'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'document':json.loads(p.read_text())}
for gen in sorted(base.glob('generation-*'))[-3:]:
 for p in gen.glob('*'):
  if p.is_dir() and 'report' in p.name:
   for q in p.glob('*.json'):
    x['files'].append({'path':str(q.relative_to(root)),'size':q.stat().st_size})
    if q.name=='report.json' and q.stat().st_size<3000000:x['documents'][str(q.relative_to(root))]={'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'document':json.loads(q.read_text())}
for p in (root/'runs/flywheel-utilization-20260911').glob('*'):
 x['files'].append({'path':str(p.relative_to(root)),'size':p.stat().st_size,'directory':p.is_dir()})
print(json.dumps(x))
