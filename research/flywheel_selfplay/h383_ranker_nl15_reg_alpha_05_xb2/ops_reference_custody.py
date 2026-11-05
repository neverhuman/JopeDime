"""Create-only H383 recovery for validated historical copies before optional reclamation."""
from pathlib import Path
import json,tarfile,sys,datetime
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h383-hwv');sys.path.insert(0,str(P/'source'));import archive_h383 as ar
patterns=['base_native/P*/fold_*/root_*/refit/*.txt','nl15/P*/fold_*/root_*/refit/*.txt','nl15/P*/fold_*/root_*/native_fit_audit.jsonl','nl15/P*/fold_*/root_*/native_model_review.json'];files=sorted(p for pattern in patterns for p in D.glob(pattern));assert len(files)==2232 and len(set(files))==2232
members={str(p.relative_to(D)):dict(sha256=ar.sha(p),bytes=p.stat().st_size) for p in files};(P/'reference-copy-inventory.json').write_text(json.dumps(members,indent=1)+'\n')
archive=D/'reference-copies.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(D)),recursive=False)
r=ar.upload_verify(archive,'reference_copies.tar.gz',{n:v['sha256'] for n,v in members.items()});assert r['all_members_exact_version_verified'] and r['exact_version_get_sha256_verified'];r.update(create_only=True,if_none_match='*')
(P/'reference-archive-verification.json').write_text(json.dumps(dict(status='PASS',record=r,members=members,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Only H383-owned validated BASE/NL15 historical native copies; current treatment fits/models untouched'),indent=1)+'\n')
assert ar.sha(archive)==r['sha256'];archive.unlink();print(json.dumps(dict(verified=True,version_id=r['version_id'],sha256=r['sha256'],members=len(members),historical_copy_bytes=sum(v['bytes'] for v in members.values()))),flush=True)
