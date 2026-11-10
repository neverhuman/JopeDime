"""xbabe1 sends byte-pinned archive requests; only xbabe2 performs AWS operations."""
from pathlib import Path
import datetime,hashlib,json,os,time
D=Path('/home/ubuntu/jope-data/h383-hwv')

def request_upload(path,key,inventory):
 assert os.uname().nodename.split('.')[0]=='xbabe1'
 path=Path(path);rel=str(path.relative_to(D));assert rel.startswith(('P0/','P1/','P2/')) and path.name=='inner_members.tar.gz'
 assert key.startswith('l1a05_members/') and inventory
 directory=D/'transfer';(directory/'requests').mkdir(parents=True,exist_ok=True);(directory/'responses').mkdir(exist_ok=True)
 token=hashlib.sha256(key.encode()).hexdigest();request=directory/'requests'/(token+'.json');response=directory/'responses'/(token+'.json')
 digest=hashlib.sha256(path.read_bytes()).hexdigest();r=dict(wave='H383',relative_path=rel,key=key,sha256=digest,bytes=path.stat().st_size,members=inventory,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 assert not request.exists() and not response.exists()
 temp=request.with_suffix('.new');temp.write_text(json.dumps(r,sort_keys=True)+'\n');os.replace(temp,request)
 deadline=datetime.datetime(2026,11,5,20,45,tzinfo=datetime.timezone.utc)
 while not response.exists():
  if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Fixed H383 compute deadline awaiting xbabe2 custody')
  time.sleep(1)
 record=json.loads(response.read_text());assert record['sha256']==digest and record['bytes']==r['bytes'] and record['key'].endswith(key)
 assert record['exact_version_get_sha256_verified'] and record['all_members_exact_version_verified'] and record['version_id']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 return record
