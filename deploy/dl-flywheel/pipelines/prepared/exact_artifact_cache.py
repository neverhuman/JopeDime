"""Create-only exact-version prefetch. All cached reads still verify SHA/length.

This is an owned acceptance artifact, not a cache for absence or mutable keys.
Only predeclared refs may be read, and no cache is usable before full closure.
"""
import concurrent.futures,hashlib,json,os,threading,time
from pathlib import Path
FIELDS=('bucket','key','version_id','sha256','bytes')
CAP=6*1024**3
def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def identity(ref):
 need(type(ref)is dict,'exact reference')
 r={k:ref[k]for k in FIELDS}
 need(r['bucket']=='veox-jopedime-learning-use2'and type(r['key'])is str and r['key'].startswith('experiments/gp-policy-dl/v2-trainonly/')and len(r['key'])<=4096,'allowed namespace')
 need(type(r['version_id'])is str and r['version_id']not in('','null')and len(r['version_id'])<=1024,'exact VersionId')
 need(type(r['bytes'])is int and 0<r['bytes']<=32*1024**2,'bounded object')
 need(type(r['sha256'])is str and len(r['sha256'])==64 and all(c in'0123456789abcdef'for c in r['sha256']),'object SHA')
 return r,sha(canonical(r))
def safe(path):
 p=Path(path);need(p.is_absolute()and all(not q.is_symlink()for q in(p,*p.parents)),'cache path must not traverse symlinks');return p
def create(path,body):
 safe(path)
 with path.open('xb')as f:f.write(body);f.flush();os.fsync(f.fileno())

class ExactArtifacts:
 def __init__(self,root,refs,getter,*,deadline,cap=CAP):
  self.root=safe(root);need(not self.root.exists(),'fresh cache only');need(type(cap)is int and 0<cap<=CAP,'cache cap');need(type(refs)is list and 0<len(refs)<=10000,'bounded ref roster')
  self.refs={}
  for ref in refs:
   r,k=identity(ref);need(k not in self.refs,'duplicate exact ref');self.refs[k]=r
  self.expected_bytes=sum(r['bytes']for r in self.refs.values());need(self.expected_bytes<=cap,'cache byte cap')
  self.deadline=deadline;self.getter=getter;self.complete=False;self.lock=threading.Lock();self.counters={'network_gets':0,'network_bytes':0,'network_seconds':0.,'verified_local_reads':0,'verified_local_bytes':0,'verified_local_seconds':0.}
  self.tick();self.root.mkdir(mode=0o700);self.manifest=canonical({'schema':'jope.acceptance-exact-artifacts/v1','refs':[self.refs[k]for k in sorted(self.refs)],'declared_bytes':self.expected_bytes,'cap_bytes':cap});need(len(self.manifest)<=8*1024**2,'cache metadata cap');create(self.root/'manifest.json',self.manifest)
 def tick(self):need(time.time()<self.deadline,'exact artifact deadline')
 def prefetch(self,workers=4):
  need(type(workers)is int and 1<=workers<=4 and not self.complete,'bounded one prefetch');self.tick();started=time.monotonic()
  def fetch(item):
   k,r=item;self.tick();start=time.monotonic();body,version=self.getter(r);self.tick()
   need(type(body)is bytes and version==r['version_id']and len(body)==r['bytes']and sha(body)==r['sha256'],'first exact-VersionId GET verification')
   create(self.root/(k+'.bin'),body)
   with self.lock:self.counters['network_gets']+=1;self.counters['network_bytes']+=len(body);self.counters['network_seconds']+=time.monotonic()-start
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers)as pool:list(pool.map(fetch,self.refs.items()))
  self.tick();need(self.counters['network_gets']==len(self.refs)and self.counters['network_bytes']==self.expected_bytes,'all exact objects fetched')
  proof={'schema':'jope.acceptance-exact-artifacts-prefetch/v1','state':'complete','manifest_sha256':sha(self.manifest),'checked_at_unix':time.time(),'objects':len(self.refs),'bytes':self.expected_bytes,'wall_seconds':time.monotonic()-started,'counters':dict(self.counters),'verification':'Every declared exact VersionId GET, length and SHA256; local reads repeat content checks','s3_writes':0}
  create(self.root/'complete.json',canonical(proof));self.complete=True;return proof
 def get_version(self,ref):
  self.tick();need(self.complete,'unclosed cache');r,k=identity(ref);need(k in self.refs and self.refs[k]==r,'ref not in exact declared roster');p=safe(self.root/(k+'.bin'));need(p.is_file()and p.stat().st_size==r['bytes'],'cached file length')
  start=time.monotonic();body=p.read_bytes();need(len(body)==r['bytes']and sha(body)==r['sha256'],'cached bytes corrupted');self.tick()
  with self.lock:self.counters['verified_local_reads']+=1;self.counters['verified_local_bytes']+=len(body);self.counters['verified_local_seconds']+=time.monotonic()-start
  return body,r['version_id']
