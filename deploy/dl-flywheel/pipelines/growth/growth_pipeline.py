"""Independent bounded generation/publication; measured admission stays separate."""
import argparse, concurrent.futures, fcntl, hashlib, importlib.util, json, os, sqlite3, sys, time
from pathlib import Path
SCHEMA='jope.flywheel-growth-pipeline/v2'
STAGE='flywheel_pipeline_stage_v1'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def queue_counts(db):
 states=dict(db.execute('SELECT state,count(*) FROM variants GROUP BY state'))
 encoded=db.execute("SELECT count(*) FROM variants v JOIN "+STAGE+" s ON s.identity=v.identity WHERE s.stage='kumo_small_s3_verified' AND v.state IN ('generated','published','admitted')").fetchone()[0]
 total=sum(states.get(k,0)for k in ('reserved','generated','published','admitted'))
 return dict(states=states,generated=total-states.get('reserved',0),published=states.get('published',0)+states.get('admitted',0),admitted=states.get('admitted',0),required_encoder_verified=encoded,
  raw_pending=states.get('reserved',0)+states.get('generated',0),encoding_pending=total-encoded)

def reserve(db,parents,target,raw_limit,encoding_limit):
 db.execute('BEGIN IMMEDIATE')
 try:
  q=queue_counts(db)
  seq=db.execute('SELECT coalesce(max(sequence)+1,0) FROM variants').fetchone()[0]
  if q['raw_pending']>=raw_limit or q['encoding_pending']>=encoding_limit or q['generated']>=target or seq>=len(parents)*256:
   db.execute('COMMIT');return None
  p=parents[seq%len(parents)];ordinal=seq//len(parents)
  db.execute('INSERT INTO variants(sequence,parent,ordinal,state,created) VALUES(?,?,?,?,?)',(seq,p['parent_lineage_sha256'],ordinal,'reserved',time.time()))
  db.execute('COMMIT');return seq,p,ordinal
 except BaseException:
  if db.in_transaction:db.execute('ROLLBACK')
  raise

def resource_wait(conf):
 mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')}
 disk=os.statvfs(conf['out']);free=disk.f_bavail*disk.f_frsize
 return os.getloadavg()[0]>conf['max_host_load'] or mem['MemAvailable']<conf['min_available_memory_bytes'] or free<conf['min_free_disk_bytes']

def status(out,c,pin,db,state,**extra):
 q=queue_counts(db);x={'schema':SCHEMA,'at':time.time(),'state':state,'config_sha256':pin,'target':c['target'],'parent_lineages':c['parent_lineages'],
  'raw_pending_limit':c['raw_pending_limit'],'encoding_pending_limit':c['encoding_pending_limit'],'required_encoders':['kumo_tabular_s'],
  'deadline_unix':c['deadline_unix'],'publication_not_training':True,**q,**extra}
 tmp=out/'progress.new'
 with tmp.open('xb')as f:f.write(canonical(x));f.flush();os.fsync(f.fileno())
 os.replace(tmp,out/'progress.json');print(json.dumps(x),flush=True)

def check_new_row(row,expected):
 need(row==expected and row['state']=='generated'and type(row['sequence'])is int and 128<=row['sequence']<1068*256,'generated growth row changed')

def verify_database(db,original,parents,g,pub):
 expected=g.digest({'config_sha':pub.GENERATOR_CONFIG_SHA,'parents':parents,'target':original['target'],'backlog':original['backlog_limit']})
 need(db.execute('SELECT sha FROM config WHERE id=1').fetchone()[0]==expected,'original immutable DB config identity')
 need(db.execute("SELECT count(*) FROM variants WHERE state IN ('reserved','generated','failed')").fetchone()[0]==0,'unresolved generation requires reconciliation')
 tables={r[0]for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 if STAGE in tables:need([r[1]for r in db.execute('PRAGMA table_info('+STAGE+')')]==['identity','stage','receipt_json','receipt_sha','created'],'stage table schema')
 return expected

def make_publication(base,pub,g,original,c,pin,row,parent,client,deadline):
 pub.check_authority(base,original)
 root=Path(original['out'])/'variants'/row['identity'];files={name:base.read(root/name,cap=8*base.MIB)for name in ('encoder.jtt1.gz','metadata.json','generation.json')}
 proof=pub.validate_dataset(g,row,parent,files['encoder.jtt1.gz'],files['metadata.json'],json.loads(files['generation.json']))
 refs=[];out=Path(c['out'])/'publications';out.mkdir(exist_ok=True)
 for name,raw in files.items():refs.append(pub.put_verified(base,client,base.PREFIX+'datasets/flywheel/v1/'+row['identity']+'/'+name,raw,deadline))
 receipt={'schema':'jope.flywheel-evergreen-dataset-publication/v1','variant_identity':row['identity'],'content_sha256':row['content_sha'],'metadata_sha256':row['metadata_sha'],
  'sequence':row['sequence'],'parent_lineage_sha256':row['parent'],'generator_config_sha256':pub.GENERATOR_CONFIG_SHA,'pipeline_config_sha256':pin,
  'publication_helper_sha256':c['publisher']['sha256'],'source_inventory_sha256':sha(canonical(row)),'parent_registry':original['parent_registry'],
  'objects':refs,'qualification':proof,'rights':'research','prepared':True,'raw_units':False,'encoders_required_before_dataset_publication':False,
  'fresh_encodings_required':True,'required_encoders':['kumo_tabular_s'],'fresh_gp_evaluations_required':True,'new_gp_labels':0,'teaching_eligible':False,'native_context_admission':'unqualified'}
 raw=canonical(receipt)+b'\n';ref=pub.put_verified(base,client,base.PREFIX+'receipts/flywheel-datasets/v1/'+row['identity']+'/'+sha(raw)+'.json',raw,deadline)
 publication={'qualified':True,'variant_identity':row['identity'],'content_sha256':row['content_sha'],'metadata_sha256':row['metadata_sha'],'s3_receipt':ref,
  'qualification':proof,'publication_helper_sha256':c['publisher']['sha256'],'pipeline_config_sha256':pin}
 p=out/(row['identity']+'.json');body=canonical(publication)+b'\n'
 if p.exists():need(p.read_bytes()==body,'publication retry differs; explicit reconciliation required')
 else:
  with p.open('xb')as f:f.write(body);f.flush();os.fsync(f.fileno())
 return publication

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--config-sha',required=True);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
 raw=Path(a.config).read_bytes();need(sha(raw)==a.config_sha,'config pin');c=json.loads(raw);need(c['schema']==SCHEMA and os.uname().nodename=='xbabe2','host/schema')
 out_path=Path(c['out']);need(out_path==Path(__file__).resolve().parent and out_path.is_absolute()and all(not p.is_symlink()for p in(out_path,*out_path.parents)),'owned output path')
 need(0<c['raw_pending_limit']<=128 and c['raw_pending_limit']<=c['encoding_pending_limit']<=4096 and c['target']==100000,'queue bounds')
 need(time.time()<c['deadline_unix'],'deadline')
 for r in c['source_pins']:
  p=Path(r['path']);need(all(not x.is_symlink()for x in(p,*p.parents)) and sha(p.read_bytes())==r['sha256'],'source pin')
 need(c['publisher']in c['source_pins']and c['base']in c['source_pins'],'publisher descriptor closure')
 pub=load(c['publisher']['path'],'growth_pub');pub.check_row=check_new_row;base,g,original=pub.load_modules(c['base']['path']);need(c['deadline_unix']==original['deadline_unix'],'original deadline')
 need(sha(Path(original['parent_registry']['path']).read_bytes())==original['parent_registry']['sha256'],'parent registry')
 parents=[json.loads(l)for l in base.bound(original['parent_registry'],cap=32*base.MIB).splitlines()];need(len(parents)==c['parent_lineages'],'parent closure')
 qualified=pub.check_authority(base,original);eligible,excluded,views=qualified
 need(all(p['view_id']in eligible and p['view_id']not in views and not(set(p['retained_lineage_context_sha256'])&excluded)for p in parents),'parent authority')
 if a.preflight_only:
  db=sqlite3.connect('file:'+original['out']+'/expansion.sqlite?mode=ro',uri=True);verify_database(db,original,parents,g,pub);print(json.dumps({'state':'preflight_pass','parents':len(parents),'states':dict(db.execute('SELECT state,count(*) FROM variants GROUP BY state')),'generator_config_unchanged':True}));db.close();return
 out=g.safe_path(c['out']);out.mkdir(exist_ok=True)
 with (out/'controller.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  class PipelineExpansion(g.Expansion):
   def reserve(self):return reserve(self.db,self.parents,self.target,c['raw_pending_limit'],c['encoding_pending_limit'])
  controller=PipelineExpansion(original['out'],parents,target=original['target'],backlog=original['backlog_limit'],config_sha=pub.GENERATOR_CONFIG_SHA)
  db=controller.db
  verify_database(db,original,parents,g,pub)
  db.execute('CREATE TABLE IF NOT EXISTS '+STAGE+'(identity TEXT NOT NULL,stage TEXT NOT NULL,receipt_json TEXT NOT NULL,receipt_sha TEXT NOT NULL,created REAL NOT NULL,PRIMARY KEY(identity,stage))')
  dbconfig=db.execute('SELECT sha FROM config WHERE id=1').fetchone()[0];byparent={p['parent_lineage_sha256']:p for p in parents}
  import boto3
  from botocore.config import Config
  client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=4))
  try:
   while time.time()<c['deadline_unix']-120:
    pub.check_authority(base,original)
    if resource_wait(original):status(out,c,a.config_sha,db,'resource_wait');time.sleep(10);continue
    advanced=0
    for _ in range(32):
     if time.time()>=c['deadline_unix']-120:break
     if not controller.step():break
     advanced+=1
    rows=[dict(r)for r in db.execute("SELECT * FROM variants WHERE state='generated' ORDER BY sequence LIMIT 32")]
    status(out,c,a.config_sha,db,'publishing'if rows else 'encoding_backpressure_or_target')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as executor:
     jobs={executor.submit(make_publication,base,pub,g,original,c,a.config_sha,row,byparent[row['parent']],client,c['deadline_unix']):row for row in rows}
     for task in concurrent.futures.as_completed(jobs):
      row=jobs[task];publication=task.result();pub.check_authority(base,original);pub.commit_published(Path(original['out'])/'expansion.sqlite',dbconfig,row,publication,c['deadline_unix'])
      status(out,c,a.config_sha,db,'publishing')
    status(out,c,a.config_sha,db,'growing'if advanced else 'encoding_backpressure_or_target')
    if not advanced:time.sleep(10)
   status(out,c,a.config_sha,db,'deadline_reached')
  except BaseException as e:status(out,c,a.config_sha,db,'failed',error=type(e).__name__+': '+str(e));raise
  finally:controller.close()
if __name__=='__main__':main()
