"""Reclaim only remote compressed copies already independently S3-verified on xbabe2."""
from pathlib import Path
import json,subprocess,datetime,shlex
P=Path(__file__).parent;records=P/'relay-records';approved={}
for sent in records.glob('*.sent'):
 rec=json.loads(sent.with_suffix('.json').read_text());src=json.loads((records/(sent.stem+'.source.json')).read_text())
 assert rec['sha256']==src['sha256'] and rec['bytes']==src['bytes'] and rec['exact_version_get_sha256_verified'] and rec['all_members_exact_version_verified']
 approved[src['relative_path']]=dict(record=rec,members=src['members'])
code='''from pathlib import Path
import json,sys,hashlib,datetime,re
D=Path('/home/ubuntu/jope-data/h353-x1');approved=json.load(sys.stdin);removed=[]
for rel,proof in approved.items():
 assert re.fullmatch(r'P[012]/fold_[0-3]/root_3000[123]/inner_members.tar.gz',rel)
 p=D/rel;receipt=p.parent/'inner_archive.json'
 if not p.exists() or not receipt.exists():continue
 r=json.loads(receipt.read_text());want=proof['record']
 assert r['version_id']==want['version_id'] and r['sha256']==want['sha256'] and r['key']==want['key'] and r['members']==proof['members'] and r['exact_version_get_sha256_verified'] and r['all_members_exact_version_verified']
 assert not p.is_symlink();h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==want['sha256'] and p.stat().st_size==want['bytes']
 row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),path=rel,bytes=want['bytes'],sha256=want['sha256'],key=want['key'],version_id=want['version_id'],whole_and_all_members_independently_verified_on_xbabe2=True)
 with (D/'control/inner-cache-reclamation.jsonl').open('a') as out:out.write(json.dumps(row)+'\\n');out.flush()
 p.unlink();removed.append(row)
print(json.dumps(dict(files_removed=len(removed),bytes_removed=sum(x['bytes'] for x in removed),files=removed,scope='Verified duplicate compressed inner archives only; source/refits/locks/audits/inputs untouched, all inner model bytes retained in exact-Version S3 objects.')))
'''
r=subprocess.check_output(['ssh','-T','-o','BatchMode=yes','xbabe1','nice -n 19 taskset -c 35 python3 -c '+shlex.quote(code)],input=json.dumps(approved).encode());result=json.loads(r)
with (P/'inner-cache-reclamation.jsonl').open('a') as out:out.write(json.dumps(result)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'}))
