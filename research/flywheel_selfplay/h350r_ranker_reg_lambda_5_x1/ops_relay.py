"""xbabe2-only S3 courier; no fits, model scoring or credentials sent to xbabe1."""
from pathlib import Path
import sys,json,hashlib,subprocess,time,datetime,re,os
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h350r-x1');REMOTE='/home/ubuntu/jope-data/h350r-x1';sys.path.insert(0,str(S));import archive_h350r as ar
assert os.uname().nodename.split('.')[0]=='xbabe2'
requests=P/'relay-requests';records=P/'relay-records';requests.mkdir(exist_ok=True);records.mkdir(exist_ok=True);spool=D/'relay-spool';spool.mkdir(parents=True,exist_ok=True)
def sync(src,dst):subprocess.run(['rsync','-a','-e','ssh -T -o BatchMode=yes',src,dst],check=True)
while datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,5,9,15,tzinfo=datetime.timezone.utc):
 sync('xbabe1:'+REMOTE+'/transfer/requests/',str(requests)+'/')
 for request in sorted(requests.glob('*.json')):
  r=json.loads(request.read_text());assert re.fullmatch(r'P[012]/fold_[0-3]/root_3000[123]/inner_members.tar.gz',r['relative_path'])
  assert request.stem==hashlib.sha256(r['key'].encode()).hexdigest() and r['key'].startswith('l2_members/') and r['members']
  record=records/request.name
  if record.exists():
   rec=json.loads(record.read_text());assert rec['sha256']==r['sha256'] and rec['bytes']==r['bytes']
  else:
   # Only the non-scientific courier backs off; no fit process is paused/restarted.
   while os.getloadavg()[0]>85:time.sleep(5)
   local=spool/(request.stem+'.tar.gz');sync('xbabe1:'+REMOTE+'/'+r['relative_path'],str(local));assert ar.sha(local)==r['sha256'] and local.stat().st_size==r['bytes']
   rec=ar.upload_verify(local,r['key'],r['members']);rec.update(all_members_exact_version_verified=True,create_only=True,if_none_match='*')
   record.write_text(json.dumps(rec,indent=2)+'\n')
   (records/(request.stem+'.source.json')).write_text(json.dumps(r,indent=1)+'\n')
   assert rec['exact_version_get_sha256_verified'] and rec['all_members_exact_version_verified'];local.unlink()
   print(json.dumps(dict(archived=r['relative_path'],version_id=rec['version_id'],sha256=rec['sha256'])),flush=True)
  marker=records/(request.stem+'.sent')
  if not marker.exists():
   sync(str(record),'xbabe1:'+REMOTE+'/transfer/responses/'+request.name);marker.write_text('receipt delivered\n')
 result=subprocess.run(['ssh','-T','-o','BatchMode=yes','xbabe1','test -f '+REMOTE+'/control/unit-exit.json'],capture_output=True)
 if result.returncode==0:break
 time.sleep(5)
(P/'relay-finished.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scientific_fits=0,scientific_scores=0,receipt_count=len(list(records.glob('*.sent')))),indent=2)+'\n')
